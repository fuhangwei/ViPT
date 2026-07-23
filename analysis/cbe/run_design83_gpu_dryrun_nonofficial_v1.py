"""Source-bound non-official design83 GPU dry-run for CBE Stage 0."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
if __name__ == "__main__" and not sys.flags.isolated:
    os.execv(
        sys.executable,
        [sys.executable, "-I", os.path.abspath(__file__), *sys.argv[1:]],
    )

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from analysis.cbe.protocol_v1 import (
    SCHEMA_VERSION,
    atomic_write_json,
    canonical_json_hash,
    deterministic_opportunity_schedule,
    load_json_strict,
    protocol_bundle_hash,
    protocol_bundle_manifest,
    reject_non_finite,
    reject_online_labels,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)
import analysis.cbe.run_vipt_real_smoke_nonofficial_v1 as smoke


DRYRUN_SCHEMA_VERSION = "cbe-stage0-design83-gpu-dryrun-nonofficial-v1"
DRYRUN_SCOPE = "non_official_design83_dry_run"
DESIGN_COUNT = 83
PROTOCOL_FILES = (
    "intervention_registry_v1.json",
    "stage0_protocol_v1.json",
    "stage0_metric_schema_v1.json",
    "stage0_gate_v1.json",
)
SOURCE_PATHS = {
    "runner": ROOT / "analysis" / "cbe" / "run_stage0_diagnostic_v1.py",
    "protocol": ROOT / "analysis" / "cbe" / "protocol_v1.py",
    "dataset": ROOT / "analysis" / "cbe" / "dataset_v1.py",
    "interventions": ROOT / "analysis" / "cbe" / "interventions_v1.py",
    "metrics": ROOT / "analysis" / "cbe" / "metrics_v1.py",
    "adapter": ROOT / "analysis" / "cbe" / "tracker_probe_v1.py",
    "semantic_replay": ROOT / "analysis" / "cbe" / "semantic_replay_v1.py",
    "replay_fixture": ROOT / "analysis" / "cbe" / "fixtures" / "counterfactual_replay_test.json",
    "tracker": ROOT / "lib" / "test" / "tracker" / "vipt_stage0.py",
    "base_tracker": ROOT / "lib" / "test" / "tracker" / "vipt.py",
    "parameters": ROOT / "lib" / "test" / "parameter" / "vipt.py",
}
CURRENT_SOURCE_PATHS = {
    "dryrun_runner": Path(__file__).resolve(),
    "smoke_runner": ROOT / "analysis" / "cbe" / "run_vipt_real_smoke_nonofficial_v1.py",
    "protocol": ROOT / "analysis" / "cbe" / "protocol_v1.py",
    "dataset": ROOT / "analysis" / "cbe" / "dataset_v1.py",
    "interventions": ROOT / "analysis" / "cbe" / "interventions_v1.py",
    "adapter": ROOT / "analysis" / "cbe" / "tracker_probe_v1.py",
    "tracker": ROOT / "lib" / "test" / "tracker" / "vipt_stage0.py",
    "base_tracker": ROOT / "lib" / "test" / "tracker" / "vipt.py",
    "config": ROOT / "lib" / "config" / "vipt" / "config.py",
    "model": ROOT / "lib" / "models" / "vipt" / "ostrack_prompt.py",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")


class DryRunValidationError(RuntimeError):
    pass


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise DryRunValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute_without_symlink(path: str | os.PathLike[str], label: str) -> Path:
    result = Path(path).expanduser().absolute()
    existing = result
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    for component in (existing, *existing.parents):
        if component.is_symlink():
            resolved = component.resolve()
            if component == Path("/var") and resolved == Path("/private/var"):
                continue
            raise DryRunValidationError(f"{label} path contains a symlink: {component}")
    return result


def _regular_file(path: str | os.PathLike[str], label: str) -> Path:
    source = _absolute_without_symlink(path, label)
    if not source.is_file() or source.is_symlink():
        raise DryRunValidationError(f"{label} must be a regular non-symlink file: {source}")
    return source.resolve()


def _directory(path: str | os.PathLike[str], label: str) -> Path:
    source = _absolute_without_symlink(path, label)
    if not source.is_dir() or source.is_symlink():
        raise DryRunValidationError(f"{label} must be a non-symlink directory: {source}")
    return source.resolve()


def _within(path: Path, root: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _new_output_directory(
    path: str | os.PathLike[str], preflight_root: Path | None = None,
) -> Path:
    output = _absolute_without_symlink(path, "dry-run output")
    if output.exists() or output.is_symlink():
        raise DryRunValidationError("dry-run output directory must not already exist")
    if _within(output, ROOT):
        raise DryRunValidationError("dry-run output must be outside the source repository")
    if preflight_root is not None and _within(output, preflight_root):
        raise DryRunValidationError("dry-run output must be outside the formal preflight root")
    _directory(output.parent, "dry-run output parent")
    return output


def _new_worker_output(path: str | os.PathLike[str], preflight_root: Path) -> Path:
    output = _absolute_without_symlink(path, "worker output")
    if output.exists() or output.is_symlink():
        raise DryRunValidationError("worker output must be a new file")
    if _within(output, ROOT) or _within(output, preflight_root):
        raise DryRunValidationError("worker output must be outside source and formal roots")
    _directory(output.parent, "worker output parent")
    return output


def _git_identity() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        repository = subprocess.run(
            ["git", "-C", str(ROOT), "config", "--get", "remote.origin.url"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DryRunValidationError("dry-run requires an auditable Git repository") from exc
    if _GIT_COMMIT.fullmatch(commit) is None or status or not repository:
        raise DryRunValidationError("dry-run requires a clean committed source identity")
    return {"git_commit": commit, "dirty_tree": False, "repository": repository}


def _source_bundle() -> dict[str, Any]:
    files = {
        name: {"path": str(_regular_file(path, f"source {name}")), "sha256": sha256_file(path)}
        for name, path in CURRENT_SOURCE_PATHS.items()
    }
    return {
        "files": files,
        "hash": canonical_json_hash({name: value["sha256"] for name, value in files.items()}),
    }


def _read_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)
    if sha256_file(source) != _sha256(expected_sha256, f"{label} SHA-256"):
        raise DryRunValidationError(f"{label} byte drift")
    value = load_json_strict(source)
    if not isinstance(value, dict):
        raise DryRunValidationError(f"{label} must be a JSON object")
    validate_content_hash(value)
    return value


def _validate_preflight_binding(
    root: Path,
    identity_sha256: str,
    preflight_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _read_bound_json(root / "run_identity.json", identity_sha256, "run identity")
    preflight = _read_bound_json(root / "preflight.json", preflight_sha256, "preflight artifact")
    if (
        identity.get("schema_version") != SCHEMA_VERSION
        or identity.get("artifact_type") != "run_identity"
        or identity.get("scope") != "design83"
        or identity.get("sequence_count") != DESIGN_COUNT
        or not isinstance(identity.get("sequences"), list)
        or len(identity["sequences"]) != DESIGN_COUNT
        or len(set(identity["sequences"])) != DESIGN_COUNT
        or not isinstance(identity.get("dataset_entries"), list)
        or len(identity["dataset_entries"]) != DESIGN_COUNT
    ):
        raise DryRunValidationError("run identity is not the complete official design83 identity")
    if (
        preflight.get("schema_version") != SCHEMA_VERSION
        or preflight.get("scope") != "design83"
        or preflight.get("phase") != "preflight"
        or preflight.get("parent_phase") is not None
        or preflight.get("parent_content_hash") is not None
        or preflight.get("identity_hash") != identity["content_hash"]
        or preflight.get("payload", {}).get("status") != "COMPLETE"
        or preflight.get("payload", {}).get("sequence_count") != DESIGN_COUNT
        or preflight.get("payload", {}).get("dataset_entry_count") != DESIGN_COUNT
    ):
        raise DryRunValidationError("preflight artifact is not the completed design83 preflight")
    return identity, preflight


def _protocol_paths(directory: Path) -> list[Path]:
    return [_regular_file(directory / name, f"protocol file {name}") for name in PROTOCOL_FILES]


def _validate_runtime_bindings(
    identity: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    model_config_path: Path,
    model_config_sha256: str,
    protocol_dir: Path,
) -> tuple[dict[str, Any], Any, smoke._VerifiedCheckpointBytes]:
    checkpoint = _regular_file(checkpoint_path, "checkpoint")
    checkpoint_hash = _sha256(checkpoint_sha256, "checkpoint SHA-256")
    config = _regular_file(model_config_path, "model config")
    config_hash = _sha256(model_config_sha256, "model config SHA-256")
    if sha256_file(checkpoint) != checkpoint_hash:
        raise DryRunValidationError("checkpoint differs from externally frozen SHA-256")
    if sha256_file(config) != config_hash:
        raise DryRunValidationError("model config differs from externally frozen SHA-256")
    inputs = identity.get("inputs", {})
    if inputs.get("checkpoint", {}).get("sha256") != checkpoint_hash:
        raise DryRunValidationError("checkpoint differs from formal preflight identity")
    if inputs.get("model_config", {}).get("sha256") != config_hash:
        raise DryRunValidationError("model config differs from formal preflight identity")
    paths = _protocol_paths(protocol_dir)
    observed_protocol_files = protocol_bundle_manifest(paths)
    observed_protocol_hash = protocol_bundle_hash(paths)
    if (
        observed_protocol_files != inputs.get("protocol_files")
        or observed_protocol_hash != inputs.get("protocol_bundle_hash")
    ):
        raise DryRunValidationError("protocol bundle differs from formal preflight identity")
    formal_sources = inputs.get("sources")
    if not isinstance(formal_sources, Mapping) or set(formal_sources) != set(SOURCE_PATHS):
        raise DryRunValidationError("formal source binding coverage mismatch")
    for name, path in SOURCE_PATHS.items():
        source = _regular_file(path, f"formal source {name}")
        if sha256_file(source) != formal_sources[name].get("sha256"):
            raise DryRunValidationError(f"current source differs from formal preflight source: {name}")
    raw_config = config.read_bytes()
    cfg, resolved_config_hash = smoke.resolved_config_bytes(raw_config)
    checkpoint_raw = checkpoint.read_bytes()
    verified_checkpoint = smoke._VerifiedCheckpointBytes(checkpoint_raw, checkpoint_hash)
    environment = smoke.environment_manifest()
    observed = {
        "checkpoint": {
            "actual_path": str(checkpoint),
            "formal_path": inputs["checkpoint"]["path"],
            "sha256": checkpoint_hash,
        },
        "model_config": {
            "actual_path": str(config),
            "formal_path": inputs["model_config"]["path"],
            "sha256": config_hash,
        },
        "protocol": {
            "actual_directory": str(protocol_dir),
            "formal_directory": inputs["protocol_dir"],
            "files": observed_protocol_files,
            "bundle_hash": observed_protocol_hash,
        },
        "resolved_config_hash": resolved_config_hash,
        "environment": environment,
    }
    return observed, cfg, verified_checkpoint


def _manifest_entry_dict(entry: Any) -> dict[str, Any]:
    return {
        "relative_path": entry.relative_path,
        "size_bytes": int(entry.size_bytes),
        "sha256": entry.sha256,
        "shape": list(entry.shape),
    }


def _load_sequence_manifest(dataset_root: Path, sequence: str, dataset: str) -> Any:
    from analysis.cbe.dataset_v1 import load_sequence_manifest

    return load_sequence_manifest(dataset_root, sequence, dataset)


def _sequence_manifest_dict(manifest: Any) -> dict[str, Any]:
    value = {
        "dataset": manifest.dataset,
        "sequence": manifest.sequence,
        "relative_root": manifest.relative_root,
        "frame_count": manifest.frame_count,
        "visible_images": [_manifest_entry_dict(item) for item in manifest.visible_images],
        "infrared_images": [_manifest_entry_dict(item) for item in manifest.infrared_images],
        "visible_annotation": {
            "relative_path": manifest.visible_annotation.relative_path,
            "size_bytes": manifest.visible_annotation.size_bytes,
            "sha256": manifest.visible_annotation.sha256,
            "row_count": len(manifest.visible_annotation.boxes_xywh),
        },
        "infrared_annotation": {
            "relative_path": manifest.infrared_annotation.relative_path,
            "size_bytes": manifest.infrared_annotation.size_bytes,
            "sha256": manifest.infrared_annotation.sha256,
            "row_count": len(manifest.infrared_annotation.boxes_xywh),
        },
    }
    value["entry_hash"] = canonical_json_hash(value)
    return value


def _select_entries(
    identity: Mapping[str, Any], dataset_root: Path, sequence_count: int,
) -> list[dict[str, Any]]:
    if isinstance(sequence_count, bool) or sequence_count not in (1, 2):
        raise DryRunValidationError("sequence_count must be exactly 1 or 2")
    if str(dataset_root) != identity.get("dataset_root"):
        raise DryRunValidationError("dataset root differs from formal preflight identity")
    names = identity["sequences"][:sequence_count]
    entries = identity["dataset_entries"][:sequence_count]
    if [entry.get("sequence") for entry in entries] != names:
        raise DryRunValidationError("selected entries are not the locked design83 prefix")
    validated = []
    for ordinal, (name, entry) in enumerate(zip(names, entries)):
        manifest = _load_sequence_manifest(dataset_root, name, identity["dataset"])
        observed = _sequence_manifest_dict(manifest)
        if observed != entry:
            raise DryRunValidationError(f"selected dataset entry differs from preflight: {name}")
        validated.append({"ordinal": ordinal, "entry": dict(entry), "manifest": manifest})
    return validated


def build_dryrun_identity(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Any, smoke._VerifiedCheckpointBytes, list[dict[str, Any]]]:
    preflight_root = _directory(args.preflight_root, "formal preflight root")
    identity, preflight = _validate_preflight_binding(
        preflight_root, args.run_identity_sha256, args.preflight_sha256
    )
    dataset_root = _directory(args.dataset_root, "dataset root")
    protocol_dir = _directory(args.protocol_dir, "protocol directory")
    bindings, cfg, checkpoint = _validate_runtime_bindings(
        identity,
        Path(args.checkpoint),
        args.checkpoint_sha256,
        Path(args.model_config),
        args.model_config_sha256,
        protocol_dir,
    )
    selected = _select_entries(identity, dataset_root, args.sequence_count)
    current_source = _source_bundle()
    git = _git_identity()
    dryrun_identity = with_content_hash({
        "schema_version": DRYRUN_SCHEMA_VERSION,
        "artifact_type": "design83_gpu_dryrun_identity",
        "scope": DRYRUN_SCOPE,
        "formal_result": False,
        "official_phase": False,
        "candidate_only": True,
        "selection_rule": "locked_design83_manifest_prefix",
        "sequence_count": args.sequence_count,
        "selected_sequences": [
            {
                "ordinal": item["ordinal"],
                "name": item["entry"]["sequence"],
                "dataset_entry_hash": item["entry"]["entry_hash"],
                "schedule": _schedule(protocol_dir, item["entry"]),
                "schedule_hash": canonical_json_hash(
                    _schedule(protocol_dir, item["entry"])
                ),
            }
            for item in selected
        ],
        "formal_preflight": {
            "root": str(preflight_root),
            "run_identity_sha256": args.run_identity_sha256,
            "run_identity_content_hash": identity["content_hash"],
            "preflight_sha256": args.preflight_sha256,
            "preflight_content_hash": preflight["content_hash"],
        },
        "runtime_bindings": bindings,
        "source_identity": git,
        "source_bundle": current_source,
    })
    return dryrun_identity, identity, cfg, checkpoint, selected


def _decode_frame(dataset_root: Path, entry: Mapping[str, Any], index: int) -> np.ndarray:
    import cv2
    from analysis.cbe.interventions_v1 import merge_six_channel

    relative_root = Path(entry["relative_root"])
    if relative_root.is_absolute() or len(relative_root.parts) != 1:
        raise DryRunValidationError("selected sequence root is unsafe")
    sequence_root = _directory(dataset_root / relative_root, "selected sequence root")
    if sequence_root.parent != dataset_root:
        raise DryRunValidationError("selected sequence root is unsafe")
    modalities = []
    for stream in ("visible_images", "infrared_images"):
        record = entry[stream][index]
        relative_path = Path(record["relative_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise DryRunValidationError("selected frame path is unsafe")
        path = _regular_file(sequence_root / relative_path, "selected frame")
        if sequence_root not in path.parents:
            raise DryRunValidationError("selected frame path is unsafe")
        raw = path.read_bytes()
        if len(raw) != record["size_bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise DryRunValidationError(f"selected frame byte drift: {path}")
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise DryRunValidationError(f"cannot decode selected frame: {path}")
        modalities.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if modalities[0].shape != modalities[1].shape:
        raise DryRunValidationError("selected modality shapes differ")
    return merge_six_channel(*modalities)


def _fingerprint(value: Any) -> dict[str, Any]:
    result = asdict(value)
    result["state"] = repr(result["state"])
    result["frame_id"] = repr(result["frame_id"])
    return result


def _snapshot_hashes(adapter: Any, probe_module: Any) -> dict[str, str]:
    return {
        name: canonical_json_hash({
            "template_id": snapshot.template_id,
            "z_patch_arr": probe_module._array_hash(snapshot.z_patch_arr),
            "z_tensor": probe_module._array_hash(snapshot.z_tensor),
            "box_mask_z": probe_module._array_hash(snapshot.box_mask_z),
        })
        for name, snapshot in adapter.snapshots.items()
    }


def _build_tracker(cfg: Any, checkpoint: smoke._VerifiedCheckpointBytes) -> Any:
    smoke.configure_determinism()
    random.seed(20260720)
    np.random.seed(20260720)
    import torch

    tracker_module = smoke._import_bound_module("lib.test.tracker.vipt_stage0")
    original_load = torch.load
    stream = checkpoint.open()

    def load_verified(source: Any, *load_args: Any, **load_kwargs: Any) -> Any:
        if source is not stream:
            raise DryRunValidationError("tracker attempted to load an unbound checkpoint")
        checkpoint.verify()
        stream.seek(0)
        load_kwargs["weights_only"] = True
        return original_load(stream, *load_args, **load_kwargs)

    torch.load = load_verified
    try:
        tracker = tracker_module.ViPTStage0Track(smoke.build_params(cfg, stream))
    finally:
        torch.load = original_load
        stream.close()
    checkpoint.verify()
    return tracker


def _schedule(protocol_dir: Path, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    protocol = load_json_strict(protocol_dir / "stage0_protocol_v1.json")
    registry = load_json_strict(protocol_dir / "intervention_registry_v1.json")
    config = protocol["schedule"]
    return deterministic_opportunity_schedule(
        entry["sequence"],
        entry["frame_count"],
        [item["direction"] for item in registry["primary_directions"]],
        registry["strengths"],
        warmup=config["warmup_frames"],
        interval=config["interval_frames"],
        max_opportunities=config["max_opportunities_per_sequence"],
        seed=config["assignment_seed"],
    )


def _sequence_transcript(
    identity: Mapping[str, Any],
    selected: Mapping[str, Any],
    dataset_root: Path,
    cfg: Any,
    checkpoint: smoke._VerifiedCheckpointBytes,
    protocol_dir: Path,
) -> dict[str, Any]:
    probe_module = smoke._import_bound_module("analysis.cbe.tracker_probe_v1")
    entry = selected["entry"]
    manifest = selected["manifest"]
    tracker = _build_tracker(cfg, checkpoint)
    frame0 = _decode_frame(dataset_root, entry, 0)
    init_bbox = list(manifest.visible_annotation.boxes_xywh[0])
    tracker.initialize(frame0, {"init_bbox": init_bbox})
    adapter = probe_module.CBEStage0ProbeAdapter(tracker, frame0, init_bbox)
    initialized = probe_module.tracker_state_fingerprint(tracker)
    snapshots = _snapshot_hashes(adapter, probe_module)
    feature_size = int(tracker.params.search_size // tracker.cfg.MODEL.BACKBONE.STRIDE)
    schedule = _schedule(protocol_dir, entry)
    by_frame = {item["frame_index"]: item for item in schedule}
    trajectory = [{
        "schema_version": DRYRUN_SCHEMA_VERSION,
        "scope": DRYRUN_SCOPE,
        "record_type": "trajectory",
        "sequence_name": entry["sequence"],
        "frame_index": 0,
        "pred_xywh": init_bbox,
        "best_score": None,
        "template_id": adapter.factual_snapshot.template_id,
    }]
    opportunities = []
    for frame_index in range(1, entry["frame_count"]):
        image = _decode_frame(dataset_root, entry, frame_index)
        anchor = [float(value) for value in tracker.state]
        if frame_index in by_frame:
            event = by_frame[frame_index]
            before = probe_module.tracker_state_fingerprint(tracker)
            probes = adapter.run_clean_probe_set(image, anchor)
            after = probe_module.tracker_state_fingerprint(tracker)
            if before != after or _snapshot_hashes(adapter, probe_module) != snapshots:
                raise DryRunValidationError("clean probe modified tracker state or snapshots")
            opportunities.append({
                "schema_version": DRYRUN_SCHEMA_VERSION,
                "scope": DRYRUN_SCOPE,
                "record_type": "opportunity",
                "sequence_name": entry["sequence"],
                "frame_index": frame_index,
                "opportunity_index": event["opportunity_index"],
                "assignment_hash": event["assignment_hash"],
                "direction": event["direction"],
                "strengths": event["strengths"],
                "search_anchor_xywh": anchor,
                "factual_template_id": adapter.factual_snapshot.template_id,
                "state_before": _fingerprint(before),
                "state_after": _fingerprint(after),
                "probes": {
                    name: smoke.prediction_summary(prediction, feature_size)
                    for name, prediction in probes.items()
                },
            })
        advanced = adapter.advance_factual(image, anchor)
        trajectory.append({
            "schema_version": DRYRUN_SCHEMA_VERSION,
            "scope": DRYRUN_SCOPE,
            "record_type": "trajectory",
            "sequence_name": entry["sequence"],
            "frame_index": frame_index,
            "pred_xywh": advanced["target_bbox"],
            "best_score": advanced["best_score"],
            "search_anchor_xywh": anchor,
            "template_id": advanced["template_id"],
        })
    transcript = {
        "schema_version": DRYRUN_SCHEMA_VERSION,
        "artifact_type": "design83_gpu_dryrun_sequence_transcript",
        "scope": DRYRUN_SCOPE,
        "formal_result": False,
        "official_phase": False,
        "sequence_name": entry["sequence"],
        "design83_ordinal": selected["ordinal"],
        "dataset_entry_hash": entry["entry_hash"],
        "frame_count": entry["frame_count"],
        "feature_size": feature_size,
        "initialized_fingerprint": _fingerprint(initialized),
        "snapshot_hashes": snapshots,
        "schedule": schedule,
        "schedule_hash": canonical_json_hash(schedule),
        "trajectory": trajectory,
        "opportunities": opportunities,
        "final_fingerprint": _fingerprint(probe_module.tracker_state_fingerprint(tracker)),
    }
    validate_sequence_transcript(transcript)
    return transcript


def _validate_prediction(value: Any, feature_size: int, path: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "anchor_id", "best_score", "maps", "search_patch_hash", "target_bbox", "template_id"
    }:
        raise DryRunValidationError(f"prediction key mismatch at {path}")
    smoke._bbox(value["target_bbox"], f"{path}.target_bbox")
    if not isinstance(value["best_score"], (int, float)):
        raise DryRunValidationError(f"prediction score is invalid at {path}")
    for name, channels in smoke._MAP_CHANNELS.items():
        summary = value["maps"].get(name) if isinstance(value["maps"], Mapping) else None
        if not isinstance(summary, Mapping) or set(summary) != {"dtype", "finite", "sha256", "shape"}:
            raise DryRunValidationError(f"map summary key mismatch at {path}.{name}")
        if (
            summary["dtype"] != "float32"
            or summary["finite"] is not True
            or summary["shape"] != [1, channels, feature_size, feature_size]
        ):
            raise DryRunValidationError(f"map summary is invalid at {path}.{name}")
        _sha256(summary["sha256"], f"{path}.{name}.sha256")


def validate_sequence_transcript(value: Mapping[str, Any]) -> None:
    expected = {
        "artifact_type", "dataset_entry_hash", "design83_ordinal", "feature_size",
        "final_fingerprint", "formal_result", "frame_count", "initialized_fingerprint",
        "official_phase", "opportunities", "schedule", "schedule_hash", "schema_version", "scope",
        "sequence_name", "snapshot_hashes", "trajectory",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DryRunValidationError("sequence transcript key mismatch")
    if (
        value["schema_version"] != DRYRUN_SCHEMA_VERSION
        or value["artifact_type"] != "design83_gpu_dryrun_sequence_transcript"
        or value["scope"] != DRYRUN_SCOPE
        or value["formal_result"] is not False
        or value["official_phase"] is not False
    ):
        raise DryRunValidationError("sequence transcript identity mismatch")
    reject_non_finite(value)
    reject_online_labels(value)
    _sha256(value["dataset_entry_hash"], "dataset_entry_hash")
    _sha256(value["schedule_hash"], "schedule_hash")
    if not isinstance(value["schedule"], list) or value["schedule_hash"] != canonical_json_hash(value["schedule"]):
        raise DryRunValidationError("schedule hash mismatch")
    frame_count = value["frame_count"]
    feature_size = value["feature_size"]
    if (
        not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 1
        or not isinstance(feature_size, int) or isinstance(feature_size, bool) or feature_size <= 0
        or not isinstance(value["trajectory"], list) or len(value["trajectory"]) != frame_count
    ):
        raise DryRunValidationError("sequence transcript counts are invalid")
    for index, row in enumerate(value["trajectory"]):
        expected_keys = {
            "best_score", "frame_index", "pred_xywh", "record_type", "schema_version",
            "scope", "sequence_name", "template_id",
        }
        if index:
            expected_keys.add("search_anchor_xywh")
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise DryRunValidationError("trajectory schema mismatch")
        if (
            row["schema_version"] != DRYRUN_SCHEMA_VERSION
            or row["scope"] != DRYRUN_SCOPE
            or row["record_type"] != "trajectory"
            or row["sequence_name"] != value["sequence_name"]
            or row["frame_index"] != index
        ):
            raise DryRunValidationError("trajectory identity mismatch")
        smoke._bbox(row["pred_xywh"], "trajectory.pred_xywh")
        if not isinstance(row["template_id"], str) or not row["template_id"]:
            raise DryRunValidationError("trajectory template_id is invalid")
        if index == 0:
            if row["best_score"] is not None:
                raise DryRunValidationError("initial trajectory score must be null")
        else:
            smoke._bbox(row["search_anchor_xywh"], "trajectory.search_anchor_xywh")
            if not isinstance(row["best_score"], (int, float)) or isinstance(row["best_score"], bool):
                raise DryRunValidationError("trajectory score is invalid")
    opportunities = value["opportunities"]
    if not isinstance(opportunities, list) or len(opportunities) != len(value["schedule"]):
        raise DryRunValidationError("opportunity count differs from locked schedule")
    for ordinal, (row, scheduled) in enumerate(zip(opportunities, value["schedule"])):
        expected_keys = {
            "assignment_hash", "direction", "factual_template_id", "frame_index",
            "opportunity_index", "probes", "record_type", "schema_version", "scope",
            "search_anchor_xywh", "sequence_name", "state_after", "state_before", "strengths",
        }
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise DryRunValidationError("opportunity schema mismatch")
        if (
            row["schema_version"] != DRYRUN_SCHEMA_VERSION
            or row["scope"] != DRYRUN_SCOPE
            or row["record_type"] != "opportunity"
            or row["sequence_name"] != value["sequence_name"]
            or row["opportunity_index"] != ordinal
            or row["frame_index"] != scheduled.get("frame_index")
            or row["assignment_hash"] != scheduled.get("assignment_hash")
            or row["direction"] != scheduled.get("direction")
            or row["strengths"] != scheduled.get("strengths")
            or row["state_after"] != row["state_before"]
        ):
            raise DryRunValidationError("opportunity differs from locked schedule or invariant")
        smoke._bbox(row["search_anchor_xywh"], "opportunity.search_anchor_xywh")
        trajectory_row = value["trajectory"][row["frame_index"]]
        if (
            row["search_anchor_xywh"] != trajectory_row["search_anchor_xywh"]
            or row["factual_template_id"] != trajectory_row["template_id"]
            or row["probes"].get("factual", {}).get("target_bbox")
            != trajectory_row["pred_xywh"]
            or row["probes"].get("factual", {}).get("best_score")
            != trajectory_row["best_score"]
        ):
            raise DryRunValidationError("opportunity differs from same-frame factual trajectory")
        if not isinstance(row["factual_template_id"], str) or not row["factual_template_id"]:
            raise DryRunValidationError("opportunity factual template is invalid")
        _sha256(row["assignment_hash"], "assignment_hash")
        if not isinstance(row["probes"], Mapping) or set(row["probes"]) != {
            "factual", "rgb_retained", "tir_retained"
        }:
            raise DryRunValidationError("opportunity must contain exactly three clean probes")
        for name, prediction in row["probes"].items():
            _validate_prediction(prediction, feature_size, f"opportunity.{name}")


def worker_run(args: argparse.Namespace) -> dict[str, Any]:
    dryrun_identity, formal_identity, cfg, checkpoint, selected = build_dryrun_identity(args)
    dataset_root = Path(args.dataset_root).resolve()
    protocol_dir = Path(args.protocol_dir).resolve()
    transcripts = [
        _sequence_transcript(
            formal_identity, item, dataset_root, cfg, checkpoint, protocol_dir
        )
        for item in selected
    ]
    if _git_identity() != dryrun_identity["source_identity"] or _source_bundle() != dryrun_identity["source_bundle"]:
        raise DryRunValidationError("source identity changed during dry-run worker execution")
    return with_content_hash({
        "schema_version": DRYRUN_SCHEMA_VERSION,
        "artifact_type": "design83_gpu_dryrun_worker_transcript",
        "scope": DRYRUN_SCOPE,
        "formal_result": False,
        "official_phase": False,
        "dryrun_identity_hash": dryrun_identity["content_hash"],
        "sequence_names": [item["sequence_name"] for item in transcripts],
        "transcript_hash": canonical_json_hash(transcripts),
        "transcripts": transcripts,
    })


def validate_worker(value: Mapping[str, Any], dryrun_identity: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "artifact_type", "content_hash", "dryrun_identity_hash", "formal_result",
        "official_phase", "schema_version", "scope", "sequence_names", "transcript_hash",
        "transcripts",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DryRunValidationError("worker transcript key mismatch")
    validate_content_hash(value)
    if (
        value["schema_version"] != DRYRUN_SCHEMA_VERSION
        or value["artifact_type"] != "design83_gpu_dryrun_worker_transcript"
        or value["scope"] != DRYRUN_SCOPE
        or value["formal_result"] is not False
        or value["official_phase"] is not False
        or value["dryrun_identity_hash"] != dryrun_identity["content_hash"]
    ):
        raise DryRunValidationError("worker transcript identity mismatch")
    expected_names = [item["name"] for item in dryrun_identity["selected_sequences"]]
    if value["sequence_names"] != expected_names:
        raise DryRunValidationError("worker selected sequence mismatch")
    transcripts = value["transcripts"]
    if not isinstance(transcripts, list) or len(transcripts) != len(expected_names):
        raise DryRunValidationError("worker transcript count mismatch")
    selected = dryrun_identity["selected_sequences"]
    for ordinal, transcript in enumerate(transcripts):
        validate_sequence_transcript(transcript)
        if (
            transcript["sequence_name"] != expected_names[ordinal]
            or transcript["design83_ordinal"] != selected[ordinal]["ordinal"]
            or transcript["dataset_entry_hash"] != selected[ordinal]["dataset_entry_hash"]
            or transcript["schedule"] != selected[ordinal]["schedule"]
            or transcript["schedule_hash"] != selected[ordinal]["schedule_hash"]
        ):
            raise DryRunValidationError("worker transcript selection binding mismatch")
    if value["transcript_hash"] != canonical_json_hash(transcripts):
        raise DryRunValidationError("worker transcript hash mismatch")
    return dict(value)


def _worker_command(args: argparse.Namespace, output: Path) -> list[str]:
    return [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--worker",
        "--preflight-root", args.preflight_root,
        "--run-identity-sha256", args.run_identity_sha256,
        "--preflight-sha256", args.preflight_sha256,
        "--dataset-root", args.dataset_root,
        "--checkpoint", args.checkpoint,
        "--checkpoint-sha256", args.checkpoint_sha256,
        "--model-config", args.model_config,
        "--model-config-sha256", args.model_config_sha256,
        "--protocol-dir", args.protocol_dir,
        "--sequence-count", str(args.sequence_count),
        "--worker-output", str(output),
    ]


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    preflight_root = _directory(args.preflight_root, "formal preflight root")
    output = _new_output_directory(args.output_dir, preflight_root)
    dryrun_identity, _, _, _, _ = build_dryrun_identity(args)
    workers = []
    with tempfile.TemporaryDirectory(dir=str(output.parent)) as temporary:
        for repeat in range(2):
            worker_output = Path(temporary) / f"repeat-{repeat}.json"
            completed = subprocess.run(
                _worker_command(args, worker_output), capture_output=True, text=True
            )
            if completed.returncode != 0:
                raise DryRunValidationError(
                    f"design83 dry-run worker {repeat} failed: {completed.stderr.strip()}"
                )
            value = load_json_strict(worker_output)
            workers.append(validate_worker(value, dryrun_identity))
    repeat_hashes = [worker["content_hash"] for worker in workers]
    if repeat_hashes[0] != repeat_hashes[1]:
        raise DryRunValidationError("fresh-process dry-run transcript hashes differ")
    temp_output = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    try:
        atomic_write_json(temp_output / "dryrun_identity.json", dryrun_identity)
        sequence_completion_hashes = {}
        sequences_root = temp_output / "sequences"
        sequences_root.mkdir()
        for transcript in workers[0]["transcripts"]:
            sequence_root = sequences_root / transcript["sequence_name"]
            sequence_root.mkdir()
            transcript_path = sequence_root / "transcript.json"
            atomic_write_json(transcript_path, transcript)
            completion = with_content_hash({
                "schema_version": DRYRUN_SCHEMA_VERSION,
                "artifact_type": "design83_gpu_dryrun_sequence_completion",
                "scope": DRYRUN_SCOPE,
                "formal_result": False,
                "official_phase": False,
                "sequence_name": transcript["sequence_name"],
                "dryrun_identity_hash": dryrun_identity["content_hash"],
                "transcript_sha256": sha256_file(transcript_path),
                "transcript_hash": canonical_json_hash(transcript),
                "status": "COMPLETE",
            })
            atomic_write_json(sequence_root / "completion.json", completion)
            sequence_completion_hashes[transcript["sequence_name"]] = completion["content_hash"]
        report = with_content_hash({
            "schema_version": DRYRUN_SCHEMA_VERSION,
            "artifact_type": "design83_gpu_dryrun_report",
            "scope": DRYRUN_SCOPE,
            "formal_result": False,
            "official_phase": False,
            "candidate_only": True,
            "status": "PASS",
            "dryrun_identity_hash": dryrun_identity["content_hash"],
            "sequence_names": workers[0]["sequence_names"],
            "sequence_completion_hashes": sequence_completion_hashes,
            "repeat_hashes": repeat_hashes,
            "deterministic_repeat_equal": True,
        })
        forbidden = {"phase", "parent_phase", "parent_content_hash", "gate", "stage1_unlock"}
        if forbidden & set(report):
            raise DryRunValidationError("non-official dry-run report contains official fields")
        if report["schema_version"] == SCHEMA_VERSION or report["scope"] == "design83":
            raise DryRunValidationError("non-official dry-run report has official identity")
        atomic_write_json(temp_output / "report.json", report)
        os.replace(temp_output, output)
        return report
    except BaseException:
        shutil.rmtree(temp_output, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a non-official design83 GPU dry-run")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--preflight-root", required=True)
    parser.add_argument("--run-identity-sha256", required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--sequence-count", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output-dir")
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker:
        if args.output_dir or not args.worker_output:
            parser.error("--worker requires only --worker-output")
        preflight_root = _directory(args.preflight_root, "formal preflight root")
        worker_output = _new_worker_output(args.worker_output, preflight_root)
        report = worker_run(args)
        atomic_write_json(worker_output, report)
    else:
        if args.worker_output or not args.output_dir:
            parser.error("parent dry-run requires only --output-dir")
        report = run_parent(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
