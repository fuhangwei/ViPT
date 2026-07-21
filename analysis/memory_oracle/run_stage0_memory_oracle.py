"""Run the independent RMG-Track Stage 0 template-memory oracle experiment.

The runner deliberately contains no controller, quarantine policy, or token memory.  It
only exercises frozen periodic interventions through ViPTTrack's snapshot API.
"""

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.memory_oracle.compute_stage0_metrics import (  # noqa: E402
    SCHEMA_VERSION,
    aggregate_sequence_summaries,
    canonical_hash,
    deterministic_clustered_bootstrap_ci,
    iou_xywh,
    paired_metric_deltas,
    tracking_metrics,
)
from analysis.natural_evidence.dataset_io import load_rgbt_sequence, load_sequence_list  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "stage0_rgbt234_val47.yaml"


def load_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required by the existing ViPT configuration stack.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def horizon_frame_indices(candidate_frame, horizon, num_frames):
    """Frames affected after a post-prediction commit at ``candidate_frame``."""
    start = int(candidate_frame) + 1
    stop = min(int(num_frames), int(candidate_frame) + int(horizon) + 1)
    return list(range(start, stop))


def periodic_candidate_frames(num_frames, warmup=30, interval=30, cooldown=60):
    """Return all periodic update-arm opportunities, excluding the terminal horizon."""
    if interval <= 0:
        raise ValueError("interval must be positive")
    last_exclusive = max(0, int(num_frames) - int(cooldown))
    return list(range(int(warmup), last_exclusive, int(interval)))


def replay_event_frames(update_frames, cooldown=60):
    """Select non-overlapping replay interventions from denser update opportunities."""
    if cooldown <= 0:
        raise ValueError("cooldown must be positive")
    selected = []
    for frame_idx in sorted(int(frame) for frame in update_frames):
        if not selected or frame_idx - selected[-1] >= int(cooldown):
            selected.append(frame_idx)
    return selected


def bbox_validity(box, image_shape, min_size=8.0, min_intersection_ratio=0.75,
                  max_padding_ratio=0.25):
    """Validate a template crop candidate against finite geometry and image bounds."""
    if box is None or len(box) < 4:
        return False, "bbox_missing"
    values = np.asarray(box[:4], dtype=float)
    if not np.isfinite(values).all():
        return False, "bbox_non_finite"
    x, y, width, height = values.tolist()
    if width < float(min_size) or height < float(min_size):
        return False, "bbox_too_small"
    image_height, image_width = [float(value) for value in image_shape[:2]]
    intersection_width = max(0.0, min(x + width, image_width) - max(x, 0.0))
    intersection_height = max(0.0, min(y + height, image_height) - max(y, 0.0))
    intersection_ratio = intersection_width * intersection_height / (width * height)
    padding_ratio = 1.0 - intersection_ratio
    if intersection_ratio < float(min_intersection_ratio):
        return False, "bbox_low_image_intersection"
    if padding_ratio > float(max_padding_ratio):
        return False, "bbox_excessive_padding"
    return True, None


def deterministic_bad_box(gt_box, sequence, frame_idx, image_shape, bad_iou=0.1,
                          min_intersection_ratio=0.75):
    """Choose a deterministic in-image same-size box with low IoU, or return ``None``."""
    x, y, width, height = [float(value) for value in gt_box[:4]]
    image_height, image_width = [float(value) for value in image_shape[:2]]
    if (not np.isfinite([x, y, width, height]).all() or width < 8.0 or height < 8.0
            or image_width <= 0 or image_height <= 0):
        return None
    max_x = max(0.0, image_width - width)
    max_y = max(0.0, image_height - height)
    candidates = [
        [0.0, min(max(y, 0.0), max_y), width, height],
        [max_x, min(max(y, 0.0), max_y), width, height],
        [min(max(x, 0.0), max_x), 0.0, width, height],
        [min(max(x, 0.0), max_x), max_y, width, height],
        [0.0, 0.0, width, height], [max_x, 0.0, width, height],
        [0.0, max_y, width, height], [max_x, max_y, width, height],
    ]
    digest = hashlib.sha256(f"{sequence}:{int(frame_idx)}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:2], "big") % len(candidates)
    for candidate in candidates[offset:] + candidates[:offset]:
        valid, _ = bbox_validity(candidate, image_shape,
                                 min_intersection_ratio=min_intersection_ratio,
                                 max_padding_ratio=1.0 - min_intersection_ratio)
        if valid and iou_xywh(candidate, gt_box) <= float(bad_iou):
            return candidate
    return None


def build_frozen_manifest(sequence, baseline_rows, warmup=30, interval=30,
                          horizons=(5, 15, 30, 60), cooldown=60,
                          pred_good_iou=0.7, bad_iou=0.1, image_shapes=None,
                          min_candidate_size=8.0, min_intersection_ratio=0.75,
                          max_padding_ratio=0.25):
    """Build update opportunities and non-overlapping replay events from a static trace."""
    rows = sorted(baseline_rows, key=lambda row: int(row["frame_idx"]))
    by_frame = {int(row["frame_idx"]): row for row in rows}
    num_frames = max(by_frame) + 1 if by_frame else 0
    update_frames = periodic_candidate_frames(num_frames, warmup, interval, cooldown)
    event_frames = set(replay_event_frames(update_frames, cooldown))
    events = []
    for frame_idx in update_frames:
        row = by_frame.get(frame_idx)
        if row is None or frame_idx not in event_frames:
            continue
        baseline_iou = iou_xywh(row["pred_xywh"], row["gt_xywh"])
        image_shape = image_shapes[frame_idx] if image_shapes is not None else row.get("image_shape")
        candidate_valid, invalid_reason = (bbox_validity(
            row["pred_xywh"], image_shape, min_candidate_size, min_intersection_ratio,
            max_padding_ratio) if image_shape is not None else (True, None))
        natural_bad = baseline_iou <= bad_iou
        bad_candidate = None
        bad_candidate_valid = False
        bad_invalid_reason = "image_shape_missing"
        if image_shape is not None:
            bad_candidate = ([float(v) for v in row["pred_xywh"]] if natural_bad else
                             deterministic_bad_box(row["gt_xywh"], sequence, frame_idx,
                                                   image_shape, bad_iou,
                                                   min_intersection_ratio))
            if bad_candidate is None:
                bad_invalid_reason = "no_legal_low_iou_bad_box"
            else:
                bad_candidate_valid, bad_invalid_reason = bbox_validity(
                    bad_candidate, image_shape, min_candidate_size, min_intersection_ratio,
                    max_padding_ratio)
                if bad_candidate_valid and iou_xywh(bad_candidate, row["gt_xywh"]) > bad_iou:
                    bad_candidate_valid, bad_invalid_reason = False, "bad_box_iou_above_threshold"
        event = {
            "event_id": f"{sequence}:{frame_idx:06d}",
            "sequence": sequence,
            "candidate_frame": frame_idx,
            "action_timing": "after_current_frame_prediction",
            "candidate_source": "static_trace_prediction",
            "baseline_pred_xywh": [float(v) for v in row["pred_xywh"]],
            "baseline_gt_xywh": [float(v) for v in row["gt_xywh"]],
            "image_shape": list(image_shape[:2]) if image_shape is not None else None,
            "candidate_valid": bool(candidate_valid),
            "invalid_reason": invalid_reason,
            "bad_candidate_xywh": bad_candidate,
            "bad_candidate_valid": bool(bad_candidate_valid),
            "bad_invalid_reason": bad_invalid_reason,
            "baseline_pred_iou": float(baseline_iou),
            "pred_good": bool(baseline_iou >= pred_good_iou),
            "natural_bad": bool(natural_bad),
            "horizons": {
                str(int(horizon)): horizon_frame_indices(frame_idx, horizon, num_frames)
                for horizon in horizons
            },
        }
        event["event_hash"] = canonical_hash(event)
        events.append(event)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "frozen_stage0_manifest",
        "sequence": sequence,
        "num_frames": num_frames,
        "policy": {
            "warmup": int(warmup),
            "update_interval": int(interval),
            "replay_event_min_spacing": int(cooldown),
            "horizons": [int(value) for value in horizons],
            "terminal_cooldown": int(cooldown),
            "pred_good_iou": float(pred_good_iou),
            "bad_iou": float(bad_iou),
            "min_candidate_size": float(min_candidate_size),
            "min_intersection_ratio": float(min_intersection_ratio),
            "max_padding_ratio": float(max_padding_ratio),
        },
        "baseline_trace_hash": canonical_hash(rows),
        "update_frames": update_frames,
        "replay_event_frames": sorted(event_frames),
        "events": events,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    return manifest


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def file_sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_path(yaml_name):
    import lib.test.parameter.vipt as vipt_params

    return Path(vipt_params.parameters(yaml_name).checkpoint).resolve()


def update_run_metadata(output_dir, **updates):
    path = Path(output_dir) / "metadata.json"
    metadata = read_json(path) if path.exists() else {"schema_version": SCHEMA_VERSION}
    metadata.update(updates)
    metadata["run_hash"] = canonical_hash({key: value for key, value in metadata.items()
                                           if key not in ("created_unix", "run_hash")})
    write_json(path, metadata)


def validate_or_write_metadata(path, metadata):
    identity_keys = (
        "dataset", "dataset_root", "split_file", "split_sha256", "yaml_name",
        "checkpoint", "checkpoint_sha256", "experiment_yaml_sha256", "source_sha256",
        "config_hash", "config_sha256", "sequences",
    )
    if Path(path).exists():
        existing = read_json(path)
        mismatches = [key for key in identity_keys if existing.get(key) != metadata.get(key)]
        if mismatches:
            raise RuntimeError(
                "Experiment identity differs from existing metadata: " + ", ".join(mismatches))
        metadata = existing
    else:
        write_json(path, metadata)
    return metadata


def make_tracker(yaml_name):
    from lib.test.tracker.vipt_stage0 import ViPTStage0Track
    import lib.test.parameter.vipt as vipt_params

    params = vipt_params.parameters(yaml_name)
    tracker = ViPTStage0Track(params)
    required = (
        "build_template_snapshot", "predict_with_context", "commit_template",
        "rollback_to_initial",
    )
    missing = [name for name in required if not hasattr(tracker, name)]
    if missing:
        raise RuntimeError(f"ViPTStage0Track is missing the Stage 0 API: {', '.join(missing)}")
    return tracker, params


def load_frame(rgb_path, tir_path, params):
    from lib.train.dataset.depth_utils import get_x_frame

    xtype = getattr(params.cfg.DATA, "XTYPE", "rgbrgb")
    return get_x_frame(rgb_path, tir_path, dtype=xtype)


def prediction_record(sequence, frame_idx, output, gt_box, arm, action="none"):
    pred = [float(value) for value in output["target_bbox"]]
    state_before = output.get("state_before", output.get("search_anchor"))
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "frame_idx": int(frame_idx),
        "arm": arm,
        "pred_xywh": pred,
        "gt_xywh": [float(value) for value in gt_box],
        "iou": iou_xywh(pred, gt_box),
        "best_score": float(output.get("best_score", 1.0)),
        "search_anchor_xywh": ([float(value) for value in state_before]
                               if state_before is not None else None),
        "action_after_prediction": action,
    }


def initial_record(sequence, gt_box, arm):
    return prediction_record(sequence, 0, {"target_bbox": gt_box, "best_score": 1.0,
                                           "state_before": gt_box}, gt_box, arm)


def static_pass(sequence, rgb_imgs, tir_imgs, gt, yaml_name, tracker_context=None):
    tracker, params = make_tracker(yaml_name) if tracker_context is None else tracker_context
    first = load_frame(rgb_imgs[0], tir_imgs[0], params)
    init_box = gt[0].tolist()
    tracker.initialize(first, {"init_bbox": init_box})
    tracker.rollback_to_initial()
    template = tracker.initial_template_snapshot
    anchor = list(init_box)
    initial = initial_record(sequence, init_box, "static")
    initial["image_shape"] = list(first.shape[:2])
    rows = [initial]
    for frame_idx in range(1, len(gt)):
        image = load_frame(rgb_imgs[frame_idx], tir_imgs[frame_idx], params)
        output = tracker.predict_with_context(image, anchor, template)
        row = prediction_record(sequence, frame_idx, output, gt[frame_idx], "static")
        row["image_shape"] = list(image.shape[:2])
        rows.append(row)
        anchor = [float(value) for value in output["target_bbox"]]
    return rows


def candidate_for_arm(arm, frame_idx, image, gt, static_row, event, config):
    if arm == "gt_good":
        candidate_box, source = gt[frame_idx].tolist(), "current_gt"
    else:
        candidate_box, source = static_row["pred_xywh"], "static_prediction"
    validity = config.get("candidate_validity", {})
    valid, reason = bbox_validity(
        candidate_box, image.shape,
        min_size=validity.get("min_size", 8.0),
        min_intersection_ratio=validity.get("min_intersection_ratio", 0.75),
        max_padding_ratio=validity.get("max_padding_ratio", 0.25),
    )
    if not valid:
        return None, source, reason
    if arm == "pred_good" and not event["pred_good"]:
        return None, source, "pred_iou_below_good_threshold"
    return candidate_box, source, None


def run_arm(sequence, arm, rgb_imgs, tir_imgs, gt, baseline_rows, manifest, yaml_name, config,
            tracker_context=None):
    tracker, params = make_tracker(yaml_name) if tracker_context is None else tracker_context
    first = load_frame(rgb_imgs[0], tir_imgs[0], params)
    init_box = gt[0].tolist()
    tracker.initialize(first, {"init_bbox": init_box})
    tracker.rollback_to_initial()
    anchor = list(init_box)
    events = {int(event["candidate_frame"]): event for event in manifest["events"]}
    update_frames = set(int(frame) for frame in manifest["update_frames"])
    baseline = {int(row["frame_idx"]): row for row in baseline_rows}
    rows = [initial_record(sequence, init_box, arm)]
    for frame_idx in range(1, len(gt)):
        image = load_frame(rgb_imgs[frame_idx], tir_imgs[frame_idx], params)
        output = tracker.predict_with_context(image, anchor, tracker.active_template_snapshot)
        action = "none"
        event = events.get(frame_idx)
        if frame_idx in update_frames and arm != "static":
            static_row = baseline[frame_idx]
            event_info = event or {
                "pred_good": iou_xywh(static_row["pred_xywh"], static_row["gt_xywh"])
                >= config["thresholds"]["pred_good_iou"]
            }
            candidate_box, source, invalid_reason = candidate_for_arm(
                arm, frame_idx, image, gt, static_row, event_info, config)
            if candidate_box is not None:
                snapshot = tracker.build_template_snapshot(
                    image, candidate_box, source=source, source_frame=frame_idx
                )
                tracker.commit_template(snapshot)
                action = f"commit_{source}"
            else:
                action = f"skip_invalid:{invalid_reason}"
        rows.append(prediction_record(sequence, frame_idx, output, gt[frame_idx], arm, action))
        anchor = [float(value) for value in output["target_bbox"]]
    return rows


def replay_variants():
    for anchor_mode in ("closed", "open"):
        yield anchor_mode, "skip", None
        yield anchor_mode, "update", "gt_good"
        yield anchor_mode, "update", "bad"


def snapshot_copy(snapshot):
    """Copy a snapshot when possible so replay never relies on mutable aliases."""
    if hasattr(snapshot, "clone"):
        return snapshot.clone()
    try:
        return copy.deepcopy(snapshot)
    except (TypeError, RuntimeError):
        return snapshot


def set_active_snapshot(tracker, snapshot):
    tracker.commit_template(snapshot_copy(snapshot))


def run_event_replay(sequence, event, anchor_mode, action, source, rgb_imgs, tir_imgs,
                     gt, baseline_rows, yaml_name, bad_iou, config, tracker_context=None):
    """Start a counterfactual at frozen baseline state ``t`` and evaluate only ``t+1..``."""
    init_box = gt[0].tolist()
    if tracker_context is None:
        tracker, params = make_tracker(yaml_name)
        first = load_frame(rgb_imgs[0], tir_imgs[0], params)
        tracker.initialize(first, {"init_bbox": init_box})
    else:
        tracker, params = tracker_context
    tracker.rollback_to_initial()
    baseline = {int(row["frame_idx"]): row for row in baseline_rows}
    candidate_frame = int(event["candidate_frame"])
    max_horizon = max(int(value) for value in event["horizons"])
    stop = min(len(gt), candidate_frame + max_horizon + 1)
    variant = f"{anchor_mode}_{action}" + (f"_{source}" if source else "")
    baseline_t = baseline[candidate_frame]
    intervention_output = {
        "target_bbox": baseline_t["pred_xywh"],
        "best_score": baseline_t.get("best_score", 1.0),
        "state_before": baseline_t.get("search_anchor_xywh"),
    }
    action_label = "skip"
    intervention_template = snapshot_copy(tracker.initial_template_snapshot)
    if action == "update":
        image_t = load_frame(rgb_imgs[candidate_frame], tir_imgs[candidate_frame], params)
        if source == "gt_good":
            candidate_box = gt[candidate_frame].tolist()
            snapshot_source = "current_gt"
        else:
            candidate_box = event.get("bad_candidate_xywh")
            snapshot_source = ("natural_bad_static_prediction" if event.get("natural_bad")
                               else "deterministic_displacement")
        validity = config.get("candidate_validity", {})
        valid, invalid_reason = bbox_validity(
            candidate_box, image_t.shape,
            min_size=validity.get("min_size", 8.0),
            min_intersection_ratio=validity.get("min_intersection_ratio", 0.75),
            max_padding_ratio=validity.get("max_padding_ratio", 0.25),
        )
        if valid and source == "bad" and iou_xywh(candidate_box, gt[candidate_frame]) > bad_iou:
            valid, invalid_reason = False, "bad_box_iou_above_threshold"
        if valid:
            intervention_template = tracker.build_template_snapshot(
                image_t, candidate_box, source=snapshot_source, source_frame=candidate_frame
            )
            set_active_snapshot(tracker, intervention_template)
            action_label = f"commit_{snapshot_source}"
        else:
            action_label = f"invalid:{invalid_reason}"
    intervention_row = prediction_record(
        sequence, candidate_frame, intervention_output, gt[candidate_frame], variant, action_label)
    intervention_row["causal_role"] = "common_baseline_intervention_prediction"
    intervention_row["included_in_future_metrics"] = False
    rows = [intervention_row]
    anchor = [float(value) for value in baseline_t["pred_xywh"]]
    for frame_idx in range(candidate_frame + 1, stop):
        image = load_frame(rgb_imgs[frame_idx], tir_imgs[frame_idx], params)
        search_anchor = (baseline[frame_idx - 1]["pred_xywh"] if anchor_mode == "open" else anchor)
        output = tracker.predict_with_context(image, search_anchor, intervention_template)
        row = prediction_record(sequence, frame_idx, output, gt[frame_idx], variant)
        row["causal_role"] = "future_outcome"
        row["included_in_future_metrics"] = True
        rows.append(row)
        anchor = [float(value) for value in output["target_bbox"]]
    return rows


def summarize_trace(rows):
    return tracking_metrics([row["pred_xywh"] for row in rows], [row["gt_xywh"] for row in rows])


def pair_replay_entries(entries):
    """Pair each update with skip for the same event, loop mode, and horizon."""
    skips = {(entry["event_id"], entry["anchor_mode"]): entry for entry in entries
             if entry["action"] == "skip"}
    pairs = []
    for entry in entries:
        if entry["action"] != "update":
            continue
        skip = skips.get((entry["event_id"], entry["anchor_mode"]))
        for horizon, update_metrics in entry["horizons"].items():
            if skip is None or horizon not in skip["horizons"]:
                continue
            valid = bool(entry.get("intervention_valid", False))
            pairs.append({
                "event_id": entry["event_id"],
                "event_hash": entry["event_hash"],
                "sequence": entry["sequence"],
                "anchor_mode": entry["anchor_mode"],
                "source": entry["source"],
                "horizon": int(horizon),
                "intervention_valid": valid,
                "invalid_reason": entry.get("invalid_reason"),
                "skip_trace_hash": skip["trace_hash"],
                "update_trace_hash": entry["trace_hash"],
                "skip_metrics": skip["horizons"][horizon],
                "update_metrics": update_metrics,
                "delta_update_minus_skip": (paired_metric_deltas(
                    skip["horizons"][horizon], update_metrics) if valid else None),
            })
    return pairs


def aggregate_replay_pairs(sequence_replay_summaries, bootstrap_seed=0,
                           bootstrap_samples=2000):
    """Aggregate valid event-level update-minus-skip deltas with sequence clustering."""
    pairs = [pair for summary in sequence_replay_summaries for pair in summary.get("paired_deltas", [])]
    aggregates = {}
    metric_keys = ("success_auc", "mean_iou", "precision20", "normalized_precision",
                   "normalized_precision_at_0_2")
    groups = sorted({(pair["source"], pair["anchor_mode"], pair["horizon"])
                     for pair in pairs if pair["intervention_valid"]})
    for group_index, (source, anchor_mode, horizon) in enumerate(groups):
        matching = [pair for pair in pairs if pair["intervention_valid"] and
                    (pair["source"], pair["anchor_mode"], pair["horizon"]) ==
                    (source, anchor_mode, horizon)]
        key = f"{source}.{anchor_mode}.H{horizon}"
        aggregates[key] = {}
        for metric_index, metric in enumerate(metric_keys):
            by_sequence = {}
            for pair in matching:
                by_sequence.setdefault(pair["sequence"], []).append(
                    pair["delta_update_minus_skip"][metric])
            aggregates[key][metric] = deterministic_clustered_bootstrap_ci(
                by_sequence, seed=bootstrap_seed + group_index * len(metric_keys) + metric_index,
                samples=bootstrap_samples)
    return {
        "aggregation": "paired_event_delta_with_sequence_cluster_bootstrap",
        "num_pairs_total": len(pairs),
        "num_pairs_valid": sum(bool(pair["intervention_valid"]) for pair in pairs),
        "groups": aggregates,
    }


def sequence_paths(output_dir, sequence):
    root = Path(output_dir) / "sequences" / sequence
    (root / "arms").mkdir(parents=True, exist_ok=True)
    (root / "replay").mkdir(parents=True, exist_ok=True)
    return root


def write_manifest_index(output_dir, manifests):
    manifest_index = {
        "schema_version": SCHEMA_VERSION,
        "kind": "frozen_stage0_manifest_index",
        "sequences": [{"sequence": item["sequence"], "manifest_hash": item["manifest_hash"],
                       "num_events": len(item["events"])} for item in manifests],
        "content_hash": canonical_hash(manifests),
    }
    write_json(Path(output_dir) / "manifest.json", manifest_index)
    update_run_metadata(output_dir, manifest_hash=manifest_index["content_hash"])
    return manifest_index


def run_static_phase(args, config, sequences):
    policy = config["schedule"]
    tracker_context = make_tracker(args.yaml_name)
    manifests = []
    for sequence in sequences:
        rgb, tir, gt, _ = load_rgbt_sequence(args.dataset_root, sequence, "RGBT234")
        n = min(len(rgb), len(tir), len(gt))
        rgb, tir, gt = rgb[:n], tir[:n], gt[:n]
        root = sequence_paths(args.output_dir, sequence)
        rows = static_pass(sequence, rgb, tir, gt, args.yaml_name, tracker_context=tracker_context)
        write_jsonl(root / "baseline_trace.jsonl", rows)
        manifest = build_frozen_manifest(
            sequence, rows, warmup=policy["warmup"], interval=policy["interval"],
            horizons=policy["horizons"], cooldown=policy["cooldown"],
            pred_good_iou=config["thresholds"]["pred_good_iou"],
            bad_iou=config["thresholds"]["bad_iou"],
            image_shapes=[row.get("image_shape") for row in rows],
            min_candidate_size=config.get("candidate_validity", {}).get("min_size", 8.0),
            min_intersection_ratio=config.get("candidate_validity", {}).get(
                "min_intersection_ratio", 0.75),
            max_padding_ratio=config.get("candidate_validity", {}).get("max_padding_ratio", 0.25),
        )
        write_json(root / "manifest.json", manifest)
        manifests.append(manifest)
    write_manifest_index(args.output_dir, manifests)


def run_arms_phase(args, config, sequences):
    tracker_context = make_tracker(args.yaml_name)
    for sequence in sequences:
        root = sequence_paths(args.output_dir, sequence)
        baseline_rows = read_jsonl(root / "baseline_trace.jsonl")
        manifest = read_json(root / "manifest.json")
        rgb, tir, gt, _ = load_rgbt_sequence(args.dataset_root, sequence, "RGBT234")
        n = min(len(rgb), len(tir), len(gt))
        rgb, tir, gt = rgb[:n], tir[:n], gt[:n]
        for arm in ("static", "periodic_pred", "pred_good", "gt_good"):
            if arm == "static":
                rows = copy.deepcopy(baseline_rows)
            else:
                rows = run_arm(sequence, arm, rgb, tir, gt, baseline_rows, manifest,
                               args.yaml_name, config, tracker_context=tracker_context)
            write_jsonl(root / "arms" / f"{arm}.trace.jsonl", rows)


def run_replay_phase(args, config, sequences):
    bad_iou = config["thresholds"]["bad_iou"]
    tracker, params = make_tracker(args.yaml_name)
    for sequence in sequences:
        root = sequence_paths(args.output_dir, sequence)
        baseline_rows = read_jsonl(root / "baseline_trace.jsonl")
        manifest = read_json(root / "manifest.json")
        rgb, tir, gt, _ = load_rgbt_sequence(args.dataset_root, sequence, "RGBT234")
        n = min(len(rgb), len(tir), len(gt))
        rgb, tir, gt = rgb[:n], tir[:n], gt[:n]
        first = load_frame(rgb[0], tir[0], params)
        tracker.initialize(first, {"init_bbox": gt[0].tolist()})
        tracker_context = (tracker, params)
        replay_summary = []
        for event in manifest["events"]:
            for anchor_mode, action, source in replay_variants():
                rows = run_event_replay(sequence, event, anchor_mode, action, source,
                                        rgb, tir, gt, baseline_rows, args.yaml_name,
                                        bad_iou, config, tracker_context=tracker_context)
                variant = f"{anchor_mode}_{action}" + (f"_{source}" if source else "")
                path = root / "replay" / f"{event['candidate_frame']:06d}.{variant}.trace.jsonl"
                write_jsonl(path, rows)
                by_frame = {int(row["frame_idx"]): row for row in rows}
                horizon_metrics = {}
                for horizon, frames in event["horizons"].items():
                    selected = [by_frame[frame] for frame in frames if frame in by_frame]
                    horizon_metrics[horizon] = summarize_trace(selected)
                intervention_row = by_frame[int(event["candidate_frame"])]
                intervention_valid = (action == "skip" or
                                      intervention_row["action_after_prediction"].startswith("commit_"))
                invalid_reason = (None if intervention_valid else
                                  intervention_row["action_after_prediction"].partition(":")[2])
                effective_source = source
                if source == "bad":
                    effective_source = ("bad_natural" if event.get("natural_bad")
                                        else "bad_deterministic")
                replay_summary.append({
                    "event_id": event["event_id"], "event_hash": event["event_hash"],
                    "sequence": sequence,
                    "anchor_mode": anchor_mode, "action": action, "source": effective_source,
                    "intervention_valid": intervention_valid,
                    "invalid_reason": invalid_reason,
                    "common_baseline_pred_xywh": intervention_row["pred_xywh"],
                    "intervention_frame_excluded": True,
                    "horizons": horizon_metrics, "trace_hash": canonical_hash(rows),
                })
        paired = pair_replay_entries(replay_summary)
        payload = {
            "schema_version": SCHEMA_VERSION, "sequence": sequence, "events": replay_summary,
            "paired_deltas": paired,
        }
        payload["content_hash"] = canonical_hash(payload)
        write_json(root / "replay_summary.json", payload)


def summarize_experiment(args, config, sequences):
    summaries = []
    manifests = []
    replay_summaries = []
    for sequence in sequences:
        root = sequence_paths(args.output_dir, sequence)
        arms = {}
        for arm in ("static", "periodic_pred", "pred_good", "gt_good"):
            path = root / "arms" / f"{arm}.trace.jsonl"
            if path.exists():
                arms[arm] = summarize_trace(read_jsonl(path))
        replay_path = root / "replay_summary.json"
        replay_summary = read_json(replay_path) if replay_path.exists() else None
        if replay_summary is not None:
            replay_summaries.append(replay_summary)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "arms": arms,
            "manifest_hash": read_json(root / "manifest.json")["manifest_hash"],
            "replay_summary_hash": replay_summary["content_hash"] if replay_summary else None,
        }
        summary["content_hash"] = canonical_hash(summary)
        write_json(root / "sequence_summary.json", summary)
        summaries.append(summary)
        manifests.append(read_json(root / "manifest.json"))
    write_manifest_index(args.output_dir, manifests)
    aggregate = aggregate_sequence_summaries(
        summaries, bootstrap_seed=config["metrics"]["bootstrap_seed"],
        bootstrap_samples=config["metrics"]["bootstrap_samples"]
    )
    aggregate["replay_paired_deltas"] = aggregate_replay_pairs(
        replay_summaries, bootstrap_seed=config["metrics"]["bootstrap_seed"],
        bootstrap_samples=config["metrics"]["bootstrap_samples"])
    aggregate["content_hash"] = canonical_hash({key: value for key, value in aggregate.items()
                                                if key != "content_hash"})
    write_json(Path(args.output_dir) / "aggregate_summary.json", aggregate)
    write_json(Path(args.output_dir) / "aggregate.json", aggregate)
    gate_cfg = config["gate"]
    gt_delta = aggregate.get("frame_weighted_paired_deltas_vs_static", {}).get(
        "gt_good", {}).get("success_auc", {})
    pred_delta = aggregate.get("frame_weighted_paired_deltas_vs_static", {}).get(
        "pred_good", {}).get("success_auc", {})
    gt_delta_macro = aggregate.get("paired_deltas_vs_static", {}).get(
        "gt_good", {}).get("success_auc", {})
    bad_groups = aggregate["replay_paired_deltas"]["groups"]
    bad_gate_source = gate_cfg["bad_gate_source"]
    bad_h60 = bad_groups.get(f"{bad_gate_source}.closed.H60", {}).get("success_auc", {})
    bad_horizons = {
        horizon: bad_groups.get(
            f"{bad_gate_source}.closed.H{horizon}", {}).get("success_auc", {})
        for horizon in gate_cfg["bad_closed_loop_negative_horizons"]
    }
    checks = {
        "gt_good_auc_delta_mean": {
            "value": gt_delta.get("mean", 0.0), "minimum": gate_cfg["gt_good_auc_delta_min"],
            "pass": gt_delta.get("mean", 0.0) >= gate_cfg["gt_good_auc_delta_min"],
        },
        "gt_good_auc_delta_ci_low": {
            "value": gt_delta.get("low", 0.0), "exclusive_minimum": 0.0,
            "pass": gt_delta.get("low", 0.0) > 0.0,
        },
        "bad_closed_h60_auc_delta_mean": {
            "value": bad_h60.get("mean", 0.0), "maximum": gate_cfg["bad_closed_h60_auc_delta_max"],
            "pass": bad_h60.get("mean", 0.0) <= gate_cfg["bad_closed_h60_auc_delta_max"],
        },
        "bad_closed_h60_auc_delta_ci_high": {
            "value": bad_h60.get("high", 0.0), "exclusive_maximum": 0.0,
            "pass": bad_h60.get("high", 0.0) < 0.0,
        },
        "bad_closed_auc_delta_all_horizons_negative": {
            "values": {f"H{horizon}": bad_horizons[horizon].get("mean", 0.0)
                       for horizon in bad_horizons},
            "pass": all(item.get("mean", 0.0) < 0.0 for item in bad_horizons.values()),
        },
        "bad_event_count": {
            "value": bad_h60.get("num_events", 0), "minimum": gate_cfg["bad_event_count_min"],
            "pass": bad_h60.get("num_events", 0) >= gate_cfg["bad_event_count_min"],
        },
        "bad_sequence_coverage": {
            "value": bad_h60.get("num_sequences", 0),
            "minimum": gate_cfg["bad_sequence_coverage_min"],
            "pass": bad_h60.get("num_sequences", 0) >= gate_cfg["bad_sequence_coverage_min"],
        },
    }
    diagnostics = {
        "bad_gate_source": bad_gate_source,
        "gt_good_sequence_macro_auc_delta_vs_static": gt_delta_macro,
        "pred_good_frame_weighted_auc_delta_vs_static": pred_delta,
        "note": "pred_good and the sequence-macro GT delta are diagnostic only",
    }
    gate = {"schema_version": SCHEMA_VERSION, "checks": checks, "diagnostics": diagnostics,
            "pass": all(item["pass"] for item in checks.values())}
    gate["content_hash"] = canonical_hash(gate)
    write_json(Path(args.output_dir) / "gate_result.json", gate)
    write_json(Path(args.output_dir) / "gate.json", gate)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RMG-Track Stage 0 memory-oracle experiments on RGBT234.")
    parser.add_argument("--dataset-root", required=True, help="RGBT234 root containing sequence directories.")
    parser.add_argument("--split-file", required=True, help="Required val47 split, one sequence per line.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--yaml-name", required=True, help="ViPT experiment YAML name without extension.")
    parser.add_argument("--phase", default="all", choices=["static", "arms", "replay", "metrics", "all"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--video", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sequences = load_sequence_list(args.dataset_root, args.split_file or None, args.video or None)
    split_path = Path(args.split_file).resolve()
    config_path = Path(args.config).resolve()
    experiment_yaml = (ROOT / "experiments" / "vipt" / f"{args.yaml_name}.yaml").resolve()
    checkpoint = checkpoint_path(args.yaml_name)
    if args.phase in ("static", "arms", "replay", "all") and not checkpoint.is_file():
        raise FileNotFoundError(f"ViPT checkpoint not found: {checkpoint}")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "RGBT234", "dataset_root": str(Path(args.dataset_root).resolve()),
        "split_file": str(split_path),
        "split_sha256": file_sha256(split_path),
        "yaml_name": args.yaml_name, "phase": args.phase,
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "experiment_yaml": str(experiment_yaml),
        "experiment_yaml_sha256": file_sha256(experiment_yaml),
        "source_sha256": {
            "base_tracker": file_sha256(ROOT / "lib" / "test" / "tracker" / "vipt.py"),
            "stage0_tracker": file_sha256(
                ROOT / "lib" / "test" / "tracker" / "vipt_stage0.py"),
            "runner": file_sha256(Path(__file__).resolve()),
            "metrics": file_sha256(Path(__file__).resolve().parent / "compute_stage0_metrics.py"),
        },
        "sequence_count": len(sequences), "sequences": sequences,
        "config": config, "config_path": str(config_path),
        "config_hash": canonical_hash(config), "config_sha256": file_sha256(config_path),
        "manifest_hash": None,
        "frame_indexing": "zero_based",
        "causality": "predict current frame, then observe current GT for oracle action; future GT evaluation only",
        "created_unix": time.time(),
    }
    metadata["run_hash"] = canonical_hash({key: value for key, value in metadata.items()
                                           if key not in ("created_unix", "phase")})
    validate_or_write_metadata(output / "metadata.json", metadata)

    if args.phase in ("static", "all"):
        run_static_phase(args, config, sequences)
    if args.phase in ("arms", "all"):
        run_arms_phase(args, config, sequences)
    if args.phase in ("replay", "all"):
        run_replay_phase(args, config, sequences)
    if args.phase in ("metrics", "all"):
        summarize_experiment(args, config, sequences)
    print(f"Stage 0 phase '{args.phase}' complete: {output}")


if __name__ == "__main__":
    main()
