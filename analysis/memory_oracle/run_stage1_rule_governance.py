"""Run RMG-Track Stage 1 rule-governed template-memory experiments."""

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.memory_oracle.compute_stage0_metrics import (  # noqa: E402
    canonical_hash,
    iou_xywh,
)
from analysis.memory_oracle.compute_stage1_metrics import (  # noqa: E402
    compute_stage1_metrics,
    metrics_from_trace,
)
from analysis.memory_oracle.rule_controller import (  # noqa: E402
    ControllerState,
    RuleThresholds,
    SCHEMA_VERSION,
    decide,
    kinematic_residual,
    observe_prediction,
    select_widest_threshold,
    threshold_quality,
)
from analysis.memory_oracle.run_stage0_memory_oracle import (  # noqa: E402
    bbox_validity,
    checkpoint_path,
    file_sha256,
    load_frame,
    periodic_candidate_frames,
)
from analysis.natural_evidence.dataset_io import (  # noqa: E402
    load_rgbt_sequence,
    load_sequence_list,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "stage1_rule_governance.yaml"
ARMS = ("static", "periodic_pred", "rule_rmg", "pred_good", "gt_good")
IDENTITY_SOURCES = {
    "base_tracker": ROOT / "lib" / "test" / "tracker" / "vipt.py",
    "stage0_runner": Path(__file__).resolve().parent / "run_stage0_memory_oracle.py",
    "dataset_io": ROOT / "analysis" / "natural_evidence" / "dataset_io.py",
    "processing_utils": ROOT / "lib" / "train" / "data" / "processing_utils.py",
    "model": ROOT / "lib" / "models" / "vipt" / "ostrack_prompt.py",
    "box_head": ROOT / "lib" / "models" / "layers" / "head.py",
    "stage0_tracker": ROOT / "lib" / "test" / "tracker" / "vipt_stage0.py",
    "stage1_tracker": ROOT / "lib" / "test" / "tracker" / "vipt_stage1.py",
    "controller": Path(__file__).resolve().parent / "rule_controller.py",
    "runner": Path(__file__).resolve(),
    "metrics": Path(__file__).resolve().parent / "compute_stage1_metrics.py",
}
_SAFE_SEQUENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_WORKER = {}


def load_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required by the ViPT configuration stack.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_config(config):
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Stage 1 config schema must be {SCHEMA_VERSION!r}.")
    if config.get("dataset") != "RGBT234":
        raise ValueError("Stage 1 supports only the RGBT234 dataset.")
    schedule = config.get("schedule", {})
    expected = {"warmup": 30, "interval": 30, "cooldown": 60}
    actual = {key: schedule.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Stage 1 schedule is fixed at {expected}, got {actual}.")
    if tuple(config.get("arms", ())) != ARMS:
        raise ValueError(f"Stage 1 arms must be exactly {list(ARMS)}.")
    tuning = config.get("tuning", {})
    if not tuning.get("entropy_candidates") or not tuning.get("motion_candidates"):
        raise ValueError("Both entropy_candidates and motion_candidates are required.")


def sequence_name_sha256(sequence):
    """Pure SHA-256 key used to assign a sequence without looking at its data."""
    return hashlib.sha256(str(sequence).encode("utf-8")).hexdigest()


def split_sequences_by_sha256(sequences, tune_fraction=0.5, tune_count=None):
    """Pure, order-invariant split of sequence names into tune and confirmation sets."""
    names = list(sequences)
    if len(names) != len(set(names)):
        raise ValueError("Sequence names must be unique before tune/confirm splitting.")
    ordered = sorted(names, key=lambda name: (sequence_name_sha256(name), name))
    if tune_count is None:
        fraction = float(tune_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("tune_fraction must be strictly between zero and one.")
        tune_count = int(math.floor(len(ordered) * fraction + 0.5))
    tune_count = int(tune_count)
    if ordered:
        tune_count = min(max(tune_count, 1), len(ordered) - (len(ordered) > 1))
    elif tune_count:
        raise ValueError("Cannot select tuning sequences from an empty split.")
    tune = ordered[:tune_count]
    confirm = ordered[tune_count:]
    return tune, confirm


def sha256_tune_confirm_split(sequences, tune_fraction=0.5, tune_count=None):
    """Named alias documenting that only SHA-256(sequence name) controls the split."""
    return split_sequences_by_sha256(sequences, tune_fraction, tune_count)


def tune_entropy_threshold(rows, tuning):
    """Pure first tuning step: select the widest entropy-only threshold."""
    return select_widest_threshold(
        rows,
        "entropy",
        tuning["entropy_candidates"],
        min_precision=tuning["min_precision"],
        max_bad_rate=tuning["max_bad_update_rate"],
        good_iou=tuning["good_iou"],
        bad_iou=tuning["bad_iou"],
        min_commits=tuning.get("min_commits", 1),
        min_sequences=tuning.get("min_sequences_with_commits", 1),
        min_coverage=tuning.get("min_commit_coverage", 0.0),
    )


def tune_motion_threshold(rows, tuning, locked_entropy):
    """Pure second tuning step with entropy fixed by the first step."""
    return select_widest_threshold(
        rows,
        "motion",
        tuning["motion_candidates"],
        fixed_entropy=locked_entropy,
        min_precision=tuning["min_precision"],
        max_bad_rate=tuning["max_bad_update_rate"],
        good_iou=tuning["good_iou"],
        bad_iou=tuning["bad_iou"],
        min_commits=tuning.get("min_commits", 1),
        min_sequences=tuning.get("min_sequences_with_commits", 1),
        min_coverage=tuning.get("min_commit_coverage", 0.0),
    )


def safe_sequence_dir(output_dir, sequence):
    """Return a contained, unambiguous output directory for one sequence owner."""
    name = str(sequence)
    if not _SAFE_SEQUENCE.fullmatch(name) or name in (".", ".."):
        raise ValueError(f"Unsafe sequence name for output directory: {name!r}")
    base = (Path(output_dir) / "sequences").resolve()
    root = (base / name).resolve()
    try:
        root.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Sequence output escapes experiment directory: {name!r}") from exc
    root.mkdir(parents=True, exist_ok=True)
    return root


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _atomic_text(path, text)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_tracker(yaml_name):
    from lib.test.tracker.vipt_stage1 import ViPTStage1Track
    import lib.test.parameter.vipt as vipt_params

    params = vipt_params.parameters(yaml_name)
    tracker = ViPTStage1Track(params)
    required = (
        "build_template_snapshot",
        "predict_with_context",
        "commit_template",
        "rollback_to_initial",
        "response_statistics",
    )
    missing = [name for name in required if not hasattr(tracker, name)]
    if missing:
        raise RuntimeError("ViPTStage1Track is missing: " + ", ".join(missing))
    return tracker, params


def _candidate_validity(box, image_shape, config):
    validity = config.get("candidate_validity", {})
    return bbox_validity(
        box,
        image_shape,
        min_size=validity.get("min_size", 8.0),
        min_intersection_ratio=validity.get("min_intersection_ratio", 0.75),
        max_padding_ratio=validity.get("max_padding_ratio", 0.25),
    )


def online_observation(frame_idx, output, image_shape, template_age):
    """Build the controller-facing payload; evaluation fields are intentionally absent."""
    observation = {
        "frame_idx": int(frame_idx),
        "pred_xywh": [float(value) for value in output["target_bbox"][:4]],
        "search_anchor_xywh": [float(value) for value in output["search_anchor"][:4]],
        "image_shape": [int(value) for value in image_shape[:2]],
        "response_peak": float(output["response_peak"]),
        "response_entropy": float(output["response_entropy"]),
        "response_margin": float(output["response_margin"]),
        "response_topk_score_std": float(output["response_topk_score_std"]),
        "response_topk_box_dispersion": float(output["response_topk_box_dispersion"]),
        "template_age": int(template_age),
    }
    return observation


def trace_record(sequence, frame_idx, output, gt_box, arm, action="none", opportunity=False,
                 observation=None, decision_reason=None, motion_residual=None,
                 candidate_valid=None, invalid_reason=None):
    pred = [float(value) for value in output["target_bbox"][:4]]
    row = {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "frame_idx": int(frame_idx),
        "arm": arm,
        "pred_xywh": pred,
        "gt_xywh": [float(value) for value in gt_box[:4]],
        "iou": iou_xywh(pred, gt_box),
        "best_score": float(output.get("best_score", 1.0)),
        "search_anchor_xywh": [float(value) for value in output.get(
            "search_anchor", output.get("state_before", pred))[:4]],
        "template_id_for_prediction": output.get("template_id"),
        "is_update_opportunity": bool(opportunity),
        "action_after_prediction": action,
        "commit_effective_from_frame": (int(frame_idx) + 1 if action.startswith("commit_") else None),
    }
    if observation is not None:
        row["online_observation"] = observation
        row["evaluation_iou"] = row["iou"]
    if decision_reason is not None:
        row["decision_reason"] = decision_reason
    if motion_residual is not None:
        row["motion_residual"] = float(motion_residual)
    if candidate_valid is not None:
        row["candidate_valid"] = bool(candidate_valid)
        row["candidate_invalid_reason"] = invalid_reason
    return row


def _initial_output(gt_box, template_id=None):
    box = [float(value) for value in gt_box[:4]]
    return {
        "target_bbox": box,
        "best_score": 1.0,
        "search_anchor": box,
        "state_before": box,
        "template_id": template_id,
    }


def _load_sequence(dataset_root, sequence):
    rgb, tir, gt, _ = load_rgbt_sequence(dataset_root, sequence, "RGBT234")
    count = min(len(rgb), len(tir), len(gt))
    if count <= 0:
        raise RuntimeError(f"Sequence {sequence!r} has no aligned RGBT frames.")
    return rgb[:count], tir[:count], gt[:count]


def extract_static_features(sequence, rgb_imgs, tir_imgs, gt, tracker_context, config):
    """Run static memory and emit online evidence only at fixed opportunities."""
    tracker, params = tracker_context
    first = load_frame(rgb_imgs[0], tir_imgs[0], params)
    init_box = gt[0].tolist()
    tracker.initialize(first, {"init_bbox": init_box})
    tracker.rollback_to_initial()
    anchor = list(init_box)
    state = ControllerState()
    observe_prediction(state, init_box)
    opportunities = set(periodic_candidate_frames(
        len(gt), config["schedule"]["warmup"], config["schedule"]["interval"],
        config["schedule"]["cooldown"]))
    rows = [trace_record(
        sequence, 0, _initial_output(init_box, tracker.initial_template_snapshot.template_id),
        gt[0], "static")]
    features = []
    for frame_idx in range(1, len(gt)):
        image = load_frame(rgb_imgs[frame_idx], tir_imgs[frame_idx], params)
        output = tracker.predict_with_context(image, anchor, tracker.active_template_snapshot)
        observation = online_observation(
            frame_idx, output, image.shape,
            frame_idx - int(tracker.active_template_snapshot.source_frame))
        valid, invalid_reason = _candidate_validity(observation["pred_xywh"], image.shape, config)
        motion = kinematic_residual(state.predictions, observation["pred_xywh"])
        is_opportunity = frame_idx in opportunities
        if is_opportunity:
            feature = {
                "schema_version": SCHEMA_VERSION,
                "sequence": sequence,
                "frame_idx": int(frame_idx),
                "online_observation": observation,
                "candidate_valid": bool(valid),
                "invalid_reason": invalid_reason,
                "motion_residual": motion,
                "evaluation_iou": iou_xywh(observation["pred_xywh"], gt[frame_idx]),
            }
            # Flat evidence fields retain compatibility with pure threshold functions. GT/IoU
            # remain outside online_observation and therefore outside the controller contract.
            feature.update({key: value for key, value in observation.items()
                            if key not in ("pred_xywh", "search_anchor_xywh", "image_shape")})
            features.append(feature)
        rows.append(trace_record(
            sequence, frame_idx, output, gt[frame_idx], "static",
            opportunity=is_opportunity,
            observation=observation if is_opportunity else None,
            motion_residual=motion if is_opportunity and motion is not None else None,
        ))
        observe_prediction(state, observation["pred_xywh"])
        anchor = observation["pred_xywh"]
    return rows, features


def run_arm(sequence, arm, rgb_imgs, tir_imgs, gt, tracker_context, config, thresholds=None):
    """Run one independent arm; every decision at t is committed only after prediction t."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    if arm == "rule_rmg" and thresholds is None:
        raise RuntimeError("rule_rmg requires locked thresholds.")
    tracker, params = tracker_context
    first = load_frame(rgb_imgs[0], tir_imgs[0], params)
    init_box = gt[0].tolist()
    tracker.initialize(first, {"init_bbox": init_box})
    tracker.rollback_to_initial()
    anchor = list(init_box)
    state = ControllerState()
    observe_prediction(state, init_box)
    opportunities = set(periodic_candidate_frames(
        len(gt), config["schedule"]["warmup"], config["schedule"]["interval"],
        config["schedule"]["cooldown"]))
    rows = [trace_record(
        sequence, 0, _initial_output(init_box, tracker.initial_template_snapshot.template_id),
        gt[0], arm)]
    for frame_idx in range(1, len(gt)):
        image = load_frame(rgb_imgs[frame_idx], tir_imgs[frame_idx], params)
        output = tracker.predict_with_context(image, anchor, tracker.active_template_snapshot)
        pred = [float(value) for value in output["target_bbox"][:4]]
        observation = online_observation(
            frame_idx, output, image.shape,
            frame_idx - int(tracker.active_template_snapshot.source_frame))
        opportunity = frame_idx in opportunities
        action = "none"
        reason = None
        candidate_valid = None
        invalid_reason = None
        motion = kinematic_residual(state.predictions, pred)
        if opportunity:
            candidate_valid, invalid_reason = _candidate_validity(pred, image.shape, config)
        if opportunity and arm != "static":
            candidate_box = pred
            candidate_source = "current_arm_prediction"
            valid = candidate_valid
            should_commit = False
            if arm == "periodic_pred":
                should_commit = valid
                reason = "periodic" if valid else f"invalid:{invalid_reason}"
            elif arm == "rule_rmg":
                # The controller receives only online observation plus geometry validity. Neither
                # GT nor evaluation IoU is passed through this call boundary.
                decision = decide(observation, state, thresholds, valid, invalid_reason)
                should_commit = decision.action == "update"
                reason = decision.reason
                motion = decision.motion_residual
            elif arm == "pred_good":
                should_commit = valid and iou_xywh(pred, gt[frame_idx]) >= float(
                    config["thresholds"]["pred_good_iou"])
                reason = ("oracle_pred_good" if should_commit else
                          (f"invalid:{invalid_reason}" if not valid else "oracle_pred_not_good"))
            elif arm == "gt_good":
                candidate_box = gt[frame_idx].tolist()
                candidate_source = "current_gt"
                valid, invalid_reason = _candidate_validity(candidate_box, image.shape, config)
                candidate_valid = valid
                should_commit = valid
                reason = "oracle_gt_good" if valid else f"invalid:{invalid_reason}"
            if should_commit:
                snapshot = tracker.build_template_snapshot(
                    image, candidate_box, source=candidate_source, source_frame=frame_idx)
                tracker.commit_template(snapshot)
                state.last_commit_frame = int(frame_idx)
                action = f"commit_{candidate_source}"
            else:
                action = f"skip:{reason}"
        rows.append(trace_record(
            sequence, frame_idx, output, gt[frame_idx], arm, action,
            opportunity=opportunity,
            observation=observation if opportunity else None,
            decision_reason=reason,
            motion_residual=motion if opportunity and motion is not None else None,
            candidate_valid=candidate_valid if opportunity else None,
            invalid_reason=invalid_reason,
        ))
        observe_prediction(state, pred)
        anchor = pred
    return rows


def _set_cpu_threads_one():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:
        pass
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass


def _worker_init(operation, payload):
    global _WORKER
    _set_cpu_threads_one()
    _WORKER = {"operation": operation, **payload, "tracker_context": None}


def _worker_tracker():
    if _WORKER["tracker_context"] is None:
        _WORKER["tracker_context"] = make_tracker(_WORKER["yaml_name"])
    return _WORKER["tracker_context"]


def _sequence_worker(sequence):
    output_dir = _WORKER["output_dir"]
    config = _WORKER["config"]
    root = safe_sequence_dir(output_dir, sequence)
    rgb, tir, gt = _load_sequence(_WORKER["dataset_root"], sequence)
    tracker_context = _worker_tracker()
    if _WORKER["operation"] == "features":
        rows, features = extract_static_features(
            sequence, rgb, tir, gt, tracker_context, config)
        write_jsonl(root / "static.trace.jsonl", rows)
        write_jsonl(root / "features.jsonl", features)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "num_frames": len(rows),
            "num_opportunities": len(features),
            "static_trace_hash": canonical_hash(rows),
            "features_hash": canonical_hash(features),
        }
        summary["content_hash"] = canonical_hash(summary)
        write_json(root / "features_summary.json", summary)
        return {"sequence": sequence, "operation": "features", "count": len(features)}
    if _WORKER["operation"] == "arms":
        thresholds = RuleThresholds(**_WORKER["thresholds"])
        arms_dir = root / "arms"
        arms_dir.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for arm in ARMS:
            rows = run_arm(
                sequence, arm, rgb, tir, gt, tracker_context, config,
                thresholds=thresholds if arm == "rule_rmg" else None)
            write_jsonl(arms_dir / f"{arm}.trace.jsonl", rows)
            hashes[arm] = canonical_hash(rows)
        payload = {"schema_version": SCHEMA_VERSION, "sequence": sequence,
                   "trace_hashes": hashes}
        payload["content_hash"] = canonical_hash(payload)
        write_json(root / "arms_summary.json", payload)
        return {"sequence": sequence, "operation": "arms", "count": len(ARMS)}
    raise ValueError(f"Unknown worker operation: {_WORKER['operation']}")


def completed_sequence(operation, output_dir, sequence):
    root = safe_sequence_dir(output_dir, sequence)
    summary_path = root / f"{operation}_summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
        stored = summary.get("content_hash")
        if stored != canonical_hash({key: value for key, value in summary.items()
                                     if key != "content_hash"}):
            return False
        if operation == "features":
            rows = read_jsonl(root / "static.trace.jsonl")
            features = read_jsonl(root / "features.jsonl")
            return (summary.get("static_trace_hash") == canonical_hash(rows)
                    and summary.get("features_hash") == canonical_hash(features))
        if operation == "arms":
            for arm in ARMS:
                rows = read_jsonl(root / "arms" / f"{arm}.trace.jsonl")
                validate_arm_trace(rows, sequence, arm)
                if summary.get("trace_hashes", {}).get(arm) != canonical_hash(rows):
                    return False
            return True
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
        return False
    return False


def map_sequences(operation, sequences, args, config, thresholds=None):
    pending = [sequence for sequence in sequences
               if not completed_sequence(operation, args.output_dir, sequence)]
    if not pending:
        return []
    payload = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "yaml_name": args.yaml_name,
        "config": config,
        "thresholds": thresholds,
    }
    workers = int(args.workers)
    if workers < 1:
        raise ValueError("workers must be at least one.")
    if workers == 1:
        _worker_init(operation, payload)
        return [_sequence_worker(sequence) for sequence in pending]
    context = mp.get_context("spawn")
    with context.Pool(processes=workers, initializer=_worker_init,
                      initargs=(operation, payload)) as pool:
        return list(pool.imap(_sequence_worker, pending, chunksize=1))


def dataset_manifest_hash(dataset_root, sequences):
    root = Path(dataset_root).resolve()
    manifest = []
    for sequence in sequences:
        sequence_root = root / sequence
        entry = {"sequence": sequence}
        for modality in ("visible", "infrared"):
            directory = sequence_root / modality
            files = sorted(path.name for path in directory.iterdir()
                           if path.is_file() and path.suffix.lower() == ".jpg")
            entry[f"{modality}_files_hash"] = canonical_hash(files)
            entry[f"{modality}_count"] = len(files)
            entry[f"{modality}_gt_sha256"] = file_sha256(sequence_root / f"{modality}.txt")
        manifest.append(entry)
    return canonical_hash(manifest)


def build_identity(args, config, sequences):
    split_path = Path(args.split_file).resolve()
    config_path = Path(args.config).resolve()
    external_lock = (Path(args.locked_thresholds).resolve()
                     if args.locked_thresholds else None)
    experiment_yaml = (ROOT / "experiments" / "vipt" / f"{args.yaml_name}.yaml").resolve()
    checkpoint = checkpoint_path(args.yaml_name)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "RGBT234",
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "yaml": args.yaml_name,
        "experiment_config": {"path": str(experiment_yaml), "sha256": file_sha256(experiment_yaml)},
        "runner_config": {"path": str(config_path), "sha256": file_sha256(config_path),
                          "content_hash": canonical_hash(config)},
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
        "split": {"path": str(split_path), "sha256": file_sha256(split_path)},
        "sequences": list(sequences),
        "sequences_sha256": canonical_hash(list(sequences)),
        "dataset_manifest_hash": dataset_manifest_hash(args.dataset_root, sequences),
        "source_sha256": {name: file_sha256(path) for name, path in IDENTITY_SOURCES.items()},
        "external_locked_thresholds": ({
            "path": str(external_lock),
            "sha256": file_sha256(external_lock),
        } if external_lock is not None else None),
        "identity_components": [
            "base_tracker", "stage0_tracker", "stage1_tracker", "controller", "runner",
            "metrics", "runner_config", "checkpoint", "split", "sequences",
            "external_locked_thresholds",
        ],
    }
    return identity, checkpoint


def validate_or_write_metadata(path, identity):
    """Refuse continuation unless every independent identity component is unchanged."""
    identity_hash = canonical_hash(identity)
    path = Path(path)
    if path.exists():
        metadata = read_json(path)
        if metadata.get("identity_hash") != canonical_hash(metadata.get("identity", {})):
            raise RuntimeError("Existing metadata identity hash is internally inconsistent.")
        if metadata.get("identity_hash") != identity_hash or metadata.get("identity") != identity:
            raise RuntimeError("Experiment identity mismatch; refusing to continue in output-dir.")
        return metadata
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "identity_hash": identity_hash,
        "frame_indexing": "zero_based",
        "causality": "predict t, then commit; a commit first affects t+1",
        "completed_phases": [],
        "created_unix": time.time(),
    }
    write_json(path, metadata)
    return metadata


def mark_phase_complete(output_dir, phase):
    path = Path(output_dir) / "metadata.json"
    metadata = read_json(path)
    completed = list(metadata.get("completed_phases", []))
    if phase not in completed:
        completed.append(phase)
    metadata["completed_phases"] = completed
    metadata["updated_unix"] = time.time()
    write_json(path, metadata)


def _feature_rows(output_dir, sequences):
    rows = []
    hashes = {}
    for sequence in sequences:
        root = safe_sequence_dir(output_dir, sequence)
        path = root / "features.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Features phase is incomplete for {sequence}: {path}")
        sequence_rows = read_jsonl(path)
        rows.extend(sequence_rows)
        hashes[sequence] = canonical_hash(sequence_rows)
    return rows, hashes


def run_tune_phase(args, config, sequences, metadata):
    tuning = config["tuning"]
    tune_sequences, confirm_sequences = split_sequences_by_sha256(
        sequences, tuning.get("tune_fraction", 0.5), tuning.get("tune_count"))
    tune_rows, tune_hashes = _feature_rows(args.output_dir, tune_sequences)
    entropy, entropy_audit = tune_entropy_threshold(tune_rows, tuning)
    if entropy is None:
        raise RuntimeError("No entropy threshold passes the locked tuning constraints.")
    motion, motion_audit = tune_motion_threshold(tune_rows, tuning, entropy)
    if motion is None:
        raise RuntimeError("No motion threshold passes after locking entropy.")
    split_payload = {
        "algorithm": "sha256(sequence_name)",
        "tune": tune_sequences,
        "confirm": confirm_sequences,
        "name_sha256": {name: sequence_name_sha256(name) for name in sequences},
    }
    lock = {
        "schema_version": SCHEMA_VERSION,
        "kind": "locked_stage1_rule_thresholds",
        "identity_hash": metadata["identity_hash"],
        "source_identity": metadata["identity"],
        "split": split_payload,
        "split_hash": canonical_hash(split_payload),
        "feature_hashes": {"tune": tune_hashes},
        "feature_set_hash": canonical_hash({"tune": tune_hashes}),
        "tuning_config_hash": canonical_hash(tuning),
        "selection_order": ["entropy", "motion"],
        "thresholds": {"max_entropy": float(entropy),
                       "max_motion_residual": float(motion)},
        "tune_audit": {"entropy": entropy_audit, "motion": motion_audit},
        "confirmation_labels_accessed": False,
    }
    lock["content_hash"] = canonical_hash(lock)
    write_json(Path(args.output_dir) / "locked_thresholds.json", lock)
    return lock


def load_locked_thresholds(output_dir, config, metadata, external_path=None):
    path = (Path(external_path).resolve() if external_path
            else Path(output_dir) / "locked_thresholds.json")
    if not path.is_file():
        raise FileNotFoundError("arms phase requires a valid locked_thresholds.json.")
    lock = read_json(path)
    stored_hash = lock.get("content_hash")
    payload = {key: value for key, value in lock.items() if key != "content_hash"}
    if (lock.get("schema_version") != SCHEMA_VERSION
            or lock.get("kind") != "locked_stage1_rule_thresholds"
            or stored_hash != canonical_hash(payload)):
        raise RuntimeError("Locked thresholds are corrupt or use the wrong schema/kind.")
    if external_path is None and lock.get("identity_hash") != metadata["identity_hash"]:
        raise RuntimeError("Locked thresholds belong to a different experiment identity.")
    if external_path is not None:
        expected = metadata["identity"].get("external_locked_thresholds", {})
        if expected.get("sha256") != file_sha256(path):
            raise RuntimeError("External locked thresholds differ from experiment identity.")
        source = lock.get("source_identity", {})
        current = metadata["identity"]
        compatible = (
            source.get("schema_version") == SCHEMA_VERSION
            and source.get("dataset") == "RGBT234"
            and source.get("checkpoint") == current.get("checkpoint")
            and source.get("experiment_config") == current.get("experiment_config")
            and source.get("runner_config") == current.get("runner_config")
            and source.get("source_sha256") == current.get("source_sha256")
            and lock.get("confirmation_labels_accessed") is False
            and len(lock.get("split", {}).get("tune", [])) == 125
            and len(lock.get("split", {}).get("confirm", [])) == 62
        )
        if not compatible:
            raise RuntimeError("External lock provenance is incompatible with this validation run.")
    if lock.get("tuning_config_hash") != canonical_hash(config["tuning"]):
        raise RuntimeError("Locked thresholds do not match the current tuning config.")
    thresholds = lock.get("thresholds", {})
    required = ("max_entropy", "max_motion_residual")
    if any(key not in thresholds or not math.isfinite(float(thresholds[key])) for key in required):
        raise RuntimeError("Locked thresholds are missing or non-finite.")
    return {key: float(thresholds[key]) for key in required}


def evaluation_sequences(sequences, config, external_lock=False):
    if external_lock:
        return list(sequences)
    _, confirm = split_sequences_by_sha256(
        sequences,
        config["tuning"].get("tune_fraction", 0.5),
        config["tuning"].get("tune_count"),
    )
    if not confirm:
        raise RuntimeError("Development split has no confirmation sequences.")
    return confirm


def validate_arm_trace(rows, sequence, arm):
    if not rows:
        raise RuntimeError(f"Empty trace for {sequence}/{arm}.")
    expected_frames = list(range(len(rows)))
    actual_frames = [int(row.get("frame_idx", -1)) for row in rows]
    if actual_frames != expected_frames:
        raise RuntimeError(f"Non-contiguous frame indices for {sequence}/{arm}.")
    if any(row.get("schema_version") != SCHEMA_VERSION for row in rows):
        raise RuntimeError(f"Schema mismatch in trace for {sequence}/{arm}.")
    if any(row.get("sequence") != sequence or row.get("arm") != arm for row in rows):
        raise RuntimeError(f"Sequence/arm mismatch in trace for {sequence}/{arm}.")


def summarize_experiment(args, config, sequences):
    summaries = []
    for sequence in sequences:
        root = safe_sequence_dir(args.output_dir, sequence)
        arm_metrics = {}
        trace_hashes = {}
        arms_summary_path = root / "arms_summary.json"
        if not arms_summary_path.is_file():
            raise FileNotFoundError(f"Missing completed arms summary for {sequence}.")
        completed = read_json(arms_summary_path)
        completed_hash = completed.get("content_hash")
        if completed_hash != canonical_hash({key: value for key, value in completed.items()
                                             if key != "content_hash"}):
            raise RuntimeError(f"Corrupt arms summary for {sequence}.")
        frame_count = None
        for arm in ARMS:
            path = root / "arms" / f"{arm}.trace.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Arms phase is incomplete for {sequence}/{arm}.")
            rows = read_jsonl(path)
            validate_arm_trace(rows, sequence, arm)
            if frame_count is None:
                frame_count = len(rows)
            elif len(rows) != frame_count:
                raise RuntimeError(f"Arm frame counts differ for {sequence}.")
            arm_metrics[arm] = metrics_from_trace(rows)
            trace_hashes[arm] = canonical_hash(rows)
            if completed.get("trace_hashes", {}).get(arm) != trace_hashes[arm]:
                raise RuntimeError(f"Trace hash mismatch for {sequence}/{arm}.")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "arms": arm_metrics,
            "governance": arm_metrics["rule_rmg"],
            "trace_hashes": trace_hashes,
        }
        summary["content_hash"] = canonical_hash(summary)
        write_json(root / "sequence_summary.json", summary)
        summaries.append(summary)
    aggregate, gate = compute_stage1_metrics(
        summaries,
        bootstrap_seed=config["metrics"]["bootstrap_seed"],
        bootstrap_samples=config["metrics"]["bootstrap_samples"],
        gate_config=config.get("gate"),
    )
    write_json(Path(args.output_dir) / "aggregate_summary.json", aggregate)
    write_json(Path(args.output_dir) / "aggregate.json", aggregate)
    write_json(Path(args.output_dir) / "gate_result.json", gate)
    write_json(Path(args.output_dir) / "gate.json", gate)
    return aggregate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run RMG-Track Stage 1 rule governance on RGBT234.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--yaml", "--yaml-name", dest="yaml_name", required=True,
                        help="ViPT experiment YAML name without extension.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--phase", default="all",
                        choices=("features", "tune", "arms", "metrics", "all"))
    parser.add_argument("--video", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--locked-thresholds", default="",
                        help="External frozen threshold artifact for confirm/validation runs.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    validate_config(config)
    if args.workers < 1:
        raise ValueError("workers must be at least one.")
    sequences = load_sequence_list(args.dataset_root, args.split_file, args.video or None)
    if len(sequences) != len(set(sequences)):
        raise ValueError("Split contains duplicate sequence names.")
    for sequence in sequences:
        if not _SAFE_SEQUENCE.fullmatch(str(sequence)):
            raise ValueError(f"Unsafe sequence name in split: {sequence!r}")
    if args.locked_thresholds and args.phase not in ("arms", "metrics"):
        raise ValueError("External locked thresholds support only explicit arms or metrics phases.")
    identity, checkpoint = build_identity(args, config, sequences)
    if args.phase in ("features", "arms", "all") and not checkpoint.is_file():
        raise FileNotFoundError(f"ViPT checkpoint not found: {checkpoint}")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output)
    metadata = validate_or_write_metadata(output / "metadata.json", identity)

    if args.phase in ("features", "all"):
        map_sequences("features", sequences, args, config)
        mark_phase_complete(output, "features")
    if args.phase in ("tune", "all"):
        run_tune_phase(args, config, sequences, metadata)
        mark_phase_complete(output, "tune")
    evaluated = (evaluation_sequences(
        sequences, config, external_lock=bool(args.locked_thresholds))
        if args.phase in ("arms", "metrics", "all") else [])
    if args.phase in ("arms", "all"):
        thresholds = load_locked_thresholds(
            output, config, metadata, args.locked_thresholds or None)
        map_sequences("arms", evaluated, args, config, thresholds)
        mark_phase_complete(output, "arms")
    if args.phase in ("metrics", "all"):
        summarize_experiment(args, config, evaluated)
        mark_phase_complete(output, "metrics")
    print(f"Stage 1 phase '{args.phase}' complete: {output}")


if __name__ == "__main__":
    main()
