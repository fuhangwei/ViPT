"""Fail-closed executable phase runner for CBE Stage 0 diagnostic v1.

The six phases are intentionally separate.  Online artifacts contain no labels;
intervention raw forwards and evaluator labels are separate files; later phases
bind every parent by canonical content hash and the immutable run identity.
There is no combined ``all`` mode and no Stage 1 unlock artifact.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cbe.protocol_v1 import (  # noqa: E402
    OFFICIAL_SCOPE,
    PHASE_ORDER,
    SCHEMA_VERSION,
    ProtocolValidationError,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_hash,
    compute_content_hash,
    deterministic_opportunity_schedule,
    load_json_strict,
    loads_json_strict,
    protocol_bundle_hash,
    protocol_bundle_manifest,
    read_manifest_hash_lock,
    read_semantic_attribute_manifest,
    read_split_role_manifest,
    reject_non_finite,
    reject_online_labels,
    sha256_file,
    validate_content_hash,
    validate_official_gate_input,
    validate_phase_parent,
    verify_manifest_hash_lock,
    with_content_hash,
)
from analysis.cbe.semantic_replay_v1 import (  # noqa: E402
    DEFAULT_FIXTURE,
    deterministic_event_id,
    replay_fixture,
    replay_intervention_record,
)

import numpy as np
from analysis.cbe.metrics_v1 import (  # noqa: E402
    evidence_metrics,
    evaluate_gate_a,
    evaluate_gate_b,
    evaluate_gate_c,
    evaluate_stage0_gates,
    fusion_regret_metrics,
    gt_fractional_cell_weights,
    intervention_metrics,
    iou_xywh,
    sequence_macro_calibration,
    spearman_rho,
    stable_softmax,
)

try:  # Dataset and intervention phases require OpenCV.
    import cv2
    from analysis.cbe.dataset_v1 import load_sequence_manifest
    from analysis.cbe.interventions_v1 import (
        InterventionSpec,
        apply_paired_local_intervention,
        matched_background_mask,
        merge_six_channel,
        target_mask_from_xywh,
    )
    _RUNTIME_IMPORT_ERROR = None
except ImportError as _runtime_error:  # pragma: no cover - environment guard
    cv2 = None
    _RUNTIME_IMPORT_ERROR = _runtime_error

CBE_DIR = Path(__file__).resolve().parent
DEFAULT_PROTOCOL_DIR = CBE_DIR / "configs" / "v1"
PROTOCOL_FILES = (
    "intervention_registry_v1.json",
    "stage0_protocol_v1.json",
    "stage0_metric_schema_v1.json",
    "stage0_gate_v1.json",
)
INNER_NAMESPACE = "rmg-stage1-v2-q5-inner"
OUTER_DEVELOPMENT_COUNT = 125
DESIGN_COUNT = 83
SPLIT_MANIFEST_FILENAMES = {
    "confirm62": "confirm62_sequence_manifest.json",
    "design83": "design83_sequence_manifest.json",
    "development187": "development187_sequence_manifest.json",
    "internal42": "internal42_sequence_manifest.json",
    "val47": "val47_sequence_manifest.json",
}
ATTRIBUTE_MANIFEST_FILENAME = "sequence_attribute_manifest.json"
SPLIT_LOCK_FILENAMES = tuple(sorted(SPLIT_MANIFEST_FILENAMES.values()))
FORMAL_MODULES = {
    "parameter_module": "lib.test.parameter.vipt",
    "tracker_module": "lib.test.tracker.vipt_stage0",
    "adapter_module": "analysis.cbe.tracker_probe_v1",
}
PRIMARY_DIRECTIONS = (
    "rgb_blur", "rgb_low_light", "rgb_desaturation", "rgb_occlusion",
    "tir_contrast_compression", "tir_saturation", "tir_sensor_noise", "tir_blur",
)
SEMANTIC_ATTRIBUTES = (
    "camera_motion", "low_illumination", "occlusion", "fast_motion",
    "thermal_or_modality_challenge",
)
DIRECTION_SPEC = {
    "rgb_blur": ("blur", "rgb"),
    "rgb_low_light": ("low_light", "rgb"),
    "rgb_desaturation": ("desaturation", "rgb"),
    "rgb_occlusion": ("opaque_occlusion", "rgb"),
    "tir_contrast_compression": ("contrast_compression", "tir"),
    "tir_saturation": ("saturation_clipping", "tir"),
    "tir_sensor_noise": ("gaussian_sensor_noise", "tir"),
    "tir_blur": ("blur", "tir"),
}
SOURCE_PATHS = {
    "runner": Path(__file__).resolve(),
    "semantic_replay": CBE_DIR / "semantic_replay_v1.py",
    "protocol": CBE_DIR / "protocol_v1.py",
    "dataset": CBE_DIR / "dataset_v1.py",
    "interventions": CBE_DIR / "interventions_v1.py",
    "metrics": CBE_DIR / "metrics_v1.py",
    "adapter": CBE_DIR / "tracker_probe_v1.py",
    "replay_fixture": DEFAULT_FIXTURE,
    "tracker": ROOT / "lib" / "test" / "tracker" / "vipt_stage0.py",
    "base_tracker": ROOT / "lib" / "test" / "tracker" / "vipt.py",
    "parameters": ROOT / "lib" / "test" / "parameter" / "vipt.py",
}


class Stage0RunError(RuntimeError):
    """A fail-closed phase or identity error."""


def _require_runtime() -> None:
    if _RUNTIME_IMPORT_ERROR is not None:
        raise Stage0RunError(
            "formal CBE phases require the project runtime with OpenCV, NumPy, and tracker dependencies"
        ) from _RUNTIME_IMPORT_ERROR


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _regular_file(path: Path | str, name: str) -> Path:
    result = Path(path).resolve()
    if not result.is_file() or result.is_symlink():
        raise Stage0RunError(f"{name} must be a regular non-symlink file: {result}")
    return result


def _directory(path: Path | str, name: str) -> Path:
    result = Path(path).resolve()
    if not result.is_dir() or result.is_symlink():
        raise Stage0RunError(f"{name} must be a non-symlink directory: {result}")
    return result


def _expected_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise Stage0RunError(f"{name} must be an externally frozen lowercase SHA-256")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=False, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
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


def _read_json(path: Path | str) -> dict[str, Any]:
    value = load_json_strict(path)
    if not isinstance(value, dict):
        raise Stage0RunError(f"artifact must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows = []
    with _regular_file(path, "JSONL artifact").open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise Stage0RunError(f"blank JSONL row at {path}:{line_number}")
            value = loads_json_strict(line)
            if not isinstance(value, dict):
                raise Stage0RunError(f"JSONL row is not an object at {path}:{line_number}")
            reject_non_finite(value)
            rows.append(value)
    return rows


def _phase_path(root: Path, phase: str) -> Path:
    return root / f"{phase}.json"


def _phase_document(phase: str, identity_hash: str, payload: Mapping[str, Any],
                    parent: Mapping[str, Any] | None = None) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scope": OFFICIAL_SCOPE,
        "phase": phase,
        "identity_hash": identity_hash,
        "payload": _jsonable(payload),
    }
    if parent is None:
        document.update({"parent_phase": None, "parent_content_hash": None})
    else:
        document.update({
            "parent_phase": parent["phase"],
            "parent_content_hash": compute_content_hash(parent),
        })
    return with_content_hash(document)


def _load_parent(path: Path | str, expected_phase: str) -> tuple[Path, dict[str, Any]]:
    parent_path = _regular_file(path, "parent phase artifact")
    parent = _read_json(parent_path)
    validate_content_hash(parent)
    if parent.get("schema_version") != SCHEMA_VERSION or parent.get("scope") != OFFICIAL_SCOPE:
        raise Stage0RunError("parent is not an official design83 artifact")
    if parent.get("phase") != expected_phase:
        raise Stage0RunError(f"expected {expected_phase!r} parent, got {parent.get('phase')!r}")
    return parent_path, parent


def _validate_child_parent(child: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
    validate_phase_parent(child, parent)
    if child.get("identity_hash") != parent.get("identity_hash"):
        raise Stage0RunError("run identity drift between phases")


def sequence_name_sha256(sequence: str, namespace: str = "") -> str:
    material = f"{namespace}:{sequence}" if namespace else str(sequence)
    return __import__("hashlib").sha256(material.encode("utf-8")).hexdigest()


def derive_design83(development_names: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    names = [str(name) for name in development_names]
    if len(names) != 187 or len(set(names)) != 187:
        raise Stage0RunError("development manifest must contain exactly 187 unique sequences")
    outer = sorted(names, key=lambda name: (sequence_name_sha256(name), name))
    development125 = outer[:OUTER_DEVELOPMENT_COUNT]
    confirm62 = outer[OUTER_DEVELOPMENT_COUNT:]
    inner = sorted(development125, key=lambda name: (sequence_name_sha256(name, INNER_NAMESPACE), name))
    design83 = inner[:DESIGN_COUNT]
    return design83, inner[DESIGN_COUNT:], confirm62


def _validate_split_role_family(
    manifests: Mapping[str, Any],
    dataset: str,
) -> tuple[list[str], list[str], list[str]]:
    expected_roles = {
        "development187": 187,
        "design83": 83,
        "internal42": 42,
        "confirm62": 62,
        "val47": 47,
    }
    if set(manifests) != set(expected_roles):
        raise Stage0RunError("split manifest role coverage mismatch")
    artifact_ids = []
    for role, count in expected_roles.items():
        manifest = manifests[role]
        if manifest.role != role or len(manifest.names) != count:
            raise Stage0RunError(f"split manifest identity mismatch: {role}")
        if any(bound_dataset != dataset for bound_dataset in manifest.datasets):
            raise Stage0RunError(f"split manifest dataset mismatch: {role}")
        artifact_ids.append(manifest.source.get("artifact_id"))
    if len(set(artifact_ids)) != len(artifact_ids):
        raise Stage0RunError("split manifest provenance artifact IDs must be unique")
    development = manifests["development187"]
    design_expected, internal_expected, confirm_expected = derive_design83(development.names)
    expected_names = {
        "design83": design_expected,
        "internal42": internal_expected,
        "confirm62": confirm_expected,
    }
    development_records = dict(development.records)
    for role, names in expected_names.items():
        manifest = manifests[role]
        if list(manifest.names) != names:
            raise Stage0RunError(f"{role} members/order differ from deterministic split derivation")
        expected_records = tuple((name, development_records[name]) for name in names)
        if manifest.records != expected_records:
            raise Stage0RunError(f"{role} per-sequence identities differ from development187")
    derived_sets = [set(manifests[role].names) for role in ("design83", "internal42", "confirm62")]
    if any(derived_sets[left] & derived_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise Stage0RunError("development split roles overlap")
    if set().union(*derived_sets) != set(development.names):
        raise Stage0RunError("development split roles do not partition development187")
    if set(manifests["val47"].names) & set(development.names):
        raise Stage0RunError("val47 must be disjoint from development187")
    return design_expected, internal_expected, confirm_expected


def _load_manifest_lock_layer(
    *,
    split_paths: Mapping[str, Path],
    split_lock_path: Path,
    expected_split_lock_sha256: str,
    attribute_path: Path,
    attribute_lock_path: Path,
    expected_attribute_lock_sha256: str,
    dataset: str,
) -> tuple[dict[str, Any], Any, Any, Any]:
    split_by_filename = {
        SPLIT_MANIFEST_FILENAMES[role]: path for role, path in split_paths.items()
    }
    try:
        split_lock = read_manifest_hash_lock(
            split_lock_path, required_filenames=SPLIT_LOCK_FILENAMES,
            expected_sha256=expected_split_lock_sha256,
        )
        verify_manifest_hash_lock(split_lock, split_by_filename)
        attribute_lock = read_manifest_hash_lock(
            attribute_lock_path, required_filenames=(ATTRIBUTE_MANIFEST_FILENAME,),
            expected_sha256=expected_attribute_lock_sha256,
        )
        verify_manifest_hash_lock(
            attribute_lock, {ATTRIBUTE_MANIFEST_FILENAME: attribute_path}
        )
        counts = {
            "development187": 187,
            "design83": 83,
            "internal42": 42,
            "confirm62": 62,
            "val47": 47,
        }
        manifests = {
            role: read_split_role_manifest(
                path,
                expected_role=role,
                expected_count=counts[role],
                expected_sha256=split_lock.as_dict()[SPLIT_MANIFEST_FILENAMES[role]],
            )
            for role, path in split_paths.items()
        }
        design_names, _, _ = _validate_split_role_family(manifests, dataset)
        attributes = read_semantic_attribute_manifest(
            attribute_path,
            design_names=design_names,
            expected_attributes=SEMANTIC_ATTRIBUTES,
            minimum_group_size=1,
            expected_sha256=attribute_lock.as_dict()[ATTRIBUTE_MANIFEST_FILENAME],
        )
    except (OSError, ValueError, ProtocolValidationError) as exc:
        raise Stage0RunError(f"manifest lock validation failed: {exc}") from exc
    return manifests, attributes, split_lock, attribute_lock


def _split_manifest_identity(path: Path, manifest: Any) -> dict[str, Any]:
    return {
        "path": str(path),
        "filename": path.name,
        "role": manifest.role,
        "count": len(manifest.names),
        "sha256": manifest.sha256,
        "ordered_records_hash": canonical_json_hash([
            {"name": name, "dataset": dataset}
            for name, dataset in manifest.records
        ]),
        "source": dict(manifest.source),
    }


def _manifest_entry_dict(entry: Any) -> dict[str, Any]:
    return {
        "relative_path": entry.relative_path,
        "size_bytes": int(entry.size_bytes),
        "sha256": entry.sha256,
        "shape": list(entry.shape),
    }


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


def _protocol_paths(protocol_dir: Path) -> list[Path]:
    return [_regular_file(protocol_dir / name, f"protocol file {name}") for name in PROTOCOL_FILES]


def _validate_dataset_identity(identity: Mapping[str, Any]) -> None:
    dataset_root = _directory(identity["dataset_root"], "identity dataset root")
    sequences = identity.get("sequences")
    entries = identity.get("dataset_entries")
    if (not isinstance(sequences, list) or len(sequences) != DESIGN_COUNT
            or len(set(sequences)) != DESIGN_COUNT
            or not isinstance(entries, list) or len(entries) != DESIGN_COUNT):
        raise Stage0RunError("identity must bind exactly 83 unique dataset entries")
    entry_index = {entry.get("sequence"): entry for entry in entries if isinstance(entry, Mapping)}
    if set(entry_index) != set(sequences):
        raise Stage0RunError("identity dataset entries do not exactly match sequences")
    for sequence in sequences:
        entry = entry_index[sequence]
        bound = dict(entry)
        entry_hash = bound.pop("entry_hash", None)
        if entry_hash != canonical_json_hash(bound):
            raise Stage0RunError(f"dataset entry hash mismatch: {sequence}")
        if (entry.get("dataset") != identity.get("dataset")
                or entry.get("sequence") != sequence):
            raise Stage0RunError(f"dataset entry identity mismatch: {sequence}")
        frame_count = entry.get("frame_count")
        if (not isinstance(frame_count, int) or isinstance(frame_count, bool)
                or frame_count <= 1
                or len(entry.get("visible_images", [])) != frame_count
                or len(entry.get("infrared_images", [])) != frame_count
                or entry.get("visible_annotation", {}).get("row_count") != frame_count
                or entry.get("infrared_annotation", {}).get("row_count") != frame_count):
            raise Stage0RunError(f"dataset entry count mismatch: {sequence}")
        relative_root = Path(entry["relative_root"])
        if relative_root.is_absolute() or len(relative_root.parts) != 1:
            raise Stage0RunError(f"unsafe identity relative root: {sequence}")
        sequence_root = (dataset_root / relative_root).resolve()
        if (not sequence_root.is_dir() or sequence_root.is_symlink()
                or sequence_root.parent != dataset_root):
            raise Stage0RunError(f"unsafe identity sequence root: {sequence}")
        file_entries = list(entry["visible_images"]) + list(entry["infrared_images"])
        relative_paths = [item.get("relative_path") for item in file_entries]
        relative_paths.extend((
            entry["visible_annotation"].get("relative_path"),
            entry["infrared_annotation"].get("relative_path"),
        ))
        if (not all(isinstance(path, str) and path for path in relative_paths)
                or len(relative_paths) != len(set(relative_paths))):
            raise Stage0RunError(f"dataset entry paths are invalid or duplicated: {sequence}")
        file_entries.extend((entry["visible_annotation"], entry["infrared_annotation"]))
        for file_entry in file_entries:
            path = (sequence_root / file_entry["relative_path"]).resolve()
            if (not path.is_file() or path.is_symlink()
                    or sequence_root not in path.parents):
                raise Stage0RunError(f"unsafe identity dataset file: {sequence}/{file_entry['relative_path']}")
            if (path.stat().st_size != file_entry["size_bytes"]
                    or sha256_file(path) != file_entry["sha256"]):
                raise Stage0RunError(f"identity dataset file drift: {sequence}/{file_entry['relative_path']}")


def _validate_identity_inputs(identity: Mapping[str, Any]) -> None:
    inputs = identity["inputs"]
    for name in ("checkpoint", "model_config"):
        item = inputs[name]
        path = _regular_file(item["path"], f"identity input {name}")
        if sha256_file(path) != item["sha256"]:
            raise Stage0RunError(f"identity input drift: {name}")
    split_inputs = inputs.get("split_manifests")
    if not isinstance(split_inputs, Mapping) or set(split_inputs) != set(SPLIT_MANIFEST_FILENAMES):
        raise Stage0RunError("identity split manifest coverage mismatch")
    split_paths = {
        role: _regular_file(item["path"], f"identity split manifest {role}")
        for role, item in split_inputs.items()
    }
    split_lock_input = inputs["split_manifest_lock"]
    attribute_input = inputs["attribute_manifest"]
    attribute_lock_input = inputs["attribute_manifest_lock"]
    manifests, attributes, split_lock, attribute_lock = _load_manifest_lock_layer(
        split_paths=split_paths,
        split_lock_path=_regular_file(split_lock_input["path"], "identity split manifest lock"),
        expected_split_lock_sha256=str(split_lock_input["sha256"]),
        attribute_path=_regular_file(attribute_input["path"], "identity attribute manifest"),
        attribute_lock_path=_regular_file(attribute_lock_input["path"], "identity attribute manifest lock"),
        expected_attribute_lock_sha256=str(attribute_lock_input["sha256"]),
        dataset=str(identity["dataset"]),
    )
    for role, manifest in manifests.items():
        if _split_manifest_identity(split_paths[role], manifest) != split_inputs[role]:
            raise Stage0RunError(f"identity split manifest drift: {role}")
    design_names = list(manifests["design83"].names)
    if (identity.get("sequences") != design_names
            or identity.get("sequence_count") != DESIGN_COUNT):
        raise Stage0RunError("identity sequences differ from locked design83 manifest")
    expected_internal = list(manifests["internal42"].names)
    expected_confirm = list(manifests["confirm62"].names)
    split_derivation = identity.get("split_derivation", {})
    if split_derivation != {
        "development_count": 187,
        "outer_rule": "sort_by_sha256(sequence)_then_name_take_125",
        "inner_rule": f"sort_by_sha256('{INNER_NAMESPACE}:'+sequence)_then_name_take_83",
        "development125_count": 125,
        "design83_count": 83,
        "internal42_hash": canonical_json_hash(expected_internal),
        "confirm62_hash": canonical_json_hash(expected_confirm),
        "val47_source_policy": "separate_authoritative_manifest_disjoint_from_development187",
    }:
        raise Stage0RunError("identity split derivation drift")
    if (split_lock.sha256 != split_lock_input.get("sha256")
            or split_lock.as_dict() != split_lock_input.get("entries")):
        raise Stage0RunError("identity split manifest lock drift")
    if (attributes.sha256 != attribute_input.get("sha256")
            or dict(attributes.groups) != {
                name: tuple(members) for name, members in attribute_input.get("groups", {}).items()
            }
            or dict(attributes.sequence_attributes) != {
                name: tuple(members)
                for name, members in attribute_input.get("sequence_attributes", {}).items()
            }
            or dict(attributes.source) != attribute_input.get("source")):
        raise Stage0RunError("identity attribute manifest drift")
    if (attribute_lock.sha256 != attribute_lock_input.get("sha256")
            or attribute_lock.as_dict() != attribute_lock_input.get("entries")):
        raise Stage0RunError("identity attribute manifest lock drift")
    protocol_dir = _directory(inputs["protocol_dir"], "identity protocol directory")
    protocol_paths = _protocol_paths(protocol_dir)
    if protocol_bundle_manifest(protocol_paths) != inputs["protocol_files"]:
        raise Stage0RunError("protocol file drift")
    if protocol_bundle_hash(protocol_paths) != inputs["protocol_bundle_hash"]:
        raise Stage0RunError("protocol bundle drift")
    for name, source in inputs["sources"].items():
        path = _regular_file(source["path"], f"identity source {name}")
        if sha256_file(path) != source["sha256"]:
            raise Stage0RunError(f"source identity drift: {name}")
    if identity.get("execution") != {**FORMAL_MODULES, "workers": 1}:
        raise Stage0RunError("formal runtime module identity drift")
    _validate_dataset_identity(identity)


def _identity_from_root(root: Path, expected_hash: str) -> dict[str, Any]:
    identity = _read_json(root / "run_identity.json")
    validate_content_hash(identity)
    if identity.get("content_hash") != expected_hash:
        raise Stage0RunError("run_identity.json does not match phase identity_hash")
    _validate_identity_inputs(identity)
    return identity


def _validated_protocol_bundle(protocol_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    registry = load_json_strict(protocol_dir / "intervention_registry_v1.json")
    protocol = load_json_strict(protocol_dir / "stage0_protocol_v1.json")
    metric_schema = load_json_strict(protocol_dir / "stage0_metric_schema_v1.json")
    gate = load_json_strict(protocol_dir / "stage0_gate_v1.json")
    artifacts = (
        (registry, "intervention_registry"),
        (protocol, "stage0_protocol"),
        (metric_schema, "stage0_metric_schema"),
        (gate, "stage0_gate"),
    )
    for artifact, artifact_type in artifacts:
        if (not isinstance(artifact, dict)
                or artifact.get("schema_version") != SCHEMA_VERSION
                or artifact.get("artifact_type") != artifact_type):
            raise Stage0RunError(f"invalid frozen protocol artifact: {artifact_type}")
    expected_primary_directions = [
        {
            "direction": "rgb_blur", "modality": "rgb",
            "operator": "gaussian_blur_reflect101", "parameter": "sigma_pixels",
            "values_by_strength": {"0.25": 1.5, "0.5": 2.5, "0.75": 3.5},
        },
        {
            "direction": "rgb_low_light", "modality": "rgb",
            "operator": "linear_intensity_scale_about_zero", "parameter": "gain",
            "values_by_strength": {"0.25": 0.75, "0.5": 0.5, "0.75": 0.25},
        },
        {
            "direction": "rgb_desaturation", "modality": "rgb",
            "operator": "rgb_luminance_interpolation",
            "parameter": "retained_color_fraction",
            "values_by_strength": {"0.25": 0.75, "0.5": 0.5, "0.75": 0.25},
        },
        {
            "direction": "rgb_occlusion", "modality": "rgb",
            "operator": "deterministic_paired_subset_opaque_occlusion",
            "parameter": "mask_area_fraction",
            "values_by_strength": {"0.25": 0.25, "0.5": 0.5, "0.75": 0.75},
        },
        {
            "direction": "tir_contrast_compression", "modality": "tir",
            "operator": "linear_contrast_about_127_5", "parameter": "gain",
            "values_by_strength": {"0.25": 0.75, "0.5": 0.5, "0.75": 0.25},
        },
        {
            "direction": "tir_saturation", "modality": "tir",
            "operator": "linear_contrast_expansion_about_127_5_then_uint8_clip",
            "parameter": "gain",
            "values_by_strength": {"0.25": 1.75, "0.5": 2.5, "0.75": 3.25},
        },
        {
            "direction": "tir_sensor_noise", "modality": "tir",
            "operator": "paired_coordinate_deterministic_gaussian_noise",
            "parameter": "sigma_pixels",
            "values_by_strength": {"0.25": 5.1, "0.5": 10.2, "0.75": 15.3},
        },
        {
            "direction": "tir_blur", "modality": "tir",
            "operator": "gaussian_blur_reflect101", "parameter": "sigma_pixels",
            "values_by_strength": {"0.25": 1.5, "0.5": 2.5, "0.75": 3.5},
        },
    ]
    if registry.get("primary_directions") != expected_primary_directions:
        raise Stage0RunError("registry primary intervention table differs from implementation")
    if registry.get("strengths") != [0.25, 0.5, 0.75]:
        raise Stage0RunError("registry must contain exactly all three strengths")
    expected_root_keys = {
        "intervention_registry": {
            "artifact_type", "diagnostic_controls", "frozen", "numeric_policy",
            "primary_directions", "region_policy", "registry_version",
            "schema_version", "strengths",
        },
        "stage0_protocol": {
            "artifact_type", "bootstrap", "identity", "inputs", "objective",
            "online_semantics", "phase_chain", "protocol_version", "replay",
            "schedule", "scope_policy", "schema_version", "stop_rule",
        },
        "stage0_metric_schema": {
            "aggregate_summary", "artifact_layers", "artifact_type", "calibration",
            "effect_definitions", "evidence_layers", "exact_keys",
            "invalid_opportunity_policy", "metric_schema_version", "schema_version",
            "semantic_attributes",
        },
        "stage0_gate": {
            "artifact_type", "decision_precedence", "failure_action", "gate_version",
            "gates", "schema_version", "scope_requirement",
        },
    }
    for artifact, artifact_type in artifacts:
        if set(artifact) != expected_root_keys[artifact_type]:
            raise Stage0RunError(f"root key mismatch in frozen {artifact_type}")
    expected_region_policy = {
        "annotation_registration_iou_minimum": 0.5,
        "background_minimum_clearance_fraction": 0.0,
        "background_placement": "sha256_ranked_complete_integer_translation_inside_nonpadding_search_support",
        "invalid_reasons": [
            "registration_discordant", "target_mask_invalid",
            "matched_background_unavailable",
        ],
        "minimum_target_pixels": 16,
        "target_expansion": 1.25,
        "target_expansion_units": "bbox_scale_about_center",
        "target_minimum_clip_retention": 0.9,
    }
    if registry.get("region_policy") != expected_region_policy:
        raise Stage0RunError("registry region policy differs from runner semantics")
    expected_numeric = {
        "input_dtype": "uint8", "input_range": [0, 255],
        "intermediate_dtype": "float64", "neutral_rgb": [124, 116, 104],
        "neutral_tir": [124, 116, 104],
        "noise_prng": "numpy_default_rng_seeded_by_sha256_first_64_bits",
        "output_dtype": "uint8",
        "rounding": "numpy_rint_ties_to_even_then_clip",
    }
    if registry.get("numeric_policy") != expected_numeric:
        raise Stage0RunError("registry numeric policy differs from implementation")
    expected_diagnostic_controls = {
        "background_replacement": {
            "fallback": "invalid_no_replacement",
            "operator": "same_sequence_past_background_preserve_current_target",
            "source_offsets": [1, 2],
        },
        "global_suppression": {
            "implementations": [
                "mean_image", "low_frequency_blur", "dataset_statistic_noise",
            ],
            "modalities": ["rgb", "tir"],
            "zero_is_stress_only": True,
        },
        "temporal": {
            "fallback": "invalid_insufficient_history", "offsets": [1, 2],
            "operator": "replace_one_modality_from_same_sequence_past_frame",
            "past_only": True,
        },
        "translation": {
            "directions": [
                "left", "right", "up", "down", "up_left", "up_right",
                "down_left", "down_right",
            ],
            "magnitudes_fraction_of_search_size": [0.02, 0.05, 0.1],
            "neutral_padding": [124, 116, 104],
        },
    }
    if registry.get("diagnostic_controls") != expected_diagnostic_controls:
        raise Stage0RunError("registry diagnostic controls differ from Stage 0 v1")
    expected_schedule = {
        "assignment": "sha256_canonical_json", "assignment_seed": 20260720,
        "frame_index_origin": 0, "interval_frames": 20,
        "invalid_backfill_forbidden": True, "max_opportunities_per_sequence": 8,
        "strengths_per_opportunity": [0.25, 0.5, 0.75], "warmup_frames": 15,
    }
    if protocol.get("schedule") != expected_schedule or protocol.get("phase_chain") != list(PHASE_ORDER):
        raise Stage0RunError("schedule or phase chain differs from Stage 0 v1")
    expected_identity = {
        "bind_checkpoint_bytes": True,
        "bind_dataset_image_and_annotation_bytes": True,
        "bind_exact_model_yaml": True,
        "bind_manifest_lock_raw_bytes": True,
        "bind_protocol_bundle": True,
        "bind_semantic_attribute_manifest_raw_bytes": True,
        "bind_source_files": True,
        "bind_split_manifest_raw_bytes": True,
    }
    expected_inputs = {
        "attribute_manifest_required_before_formal_preflight": True,
        "attribute_manifest_schema": "semantic_attribute_manifest_with_bidirectional_membership_v1",
        "authoritative_confirm62_manifest_required": True,
        "authoritative_design83_manifest_required": True,
        "authoritative_development187_manifest_required": True,
        "authoritative_internal42_manifest_required": True,
        "authoritative_val47_manifest_required": True,
        "intervention_registry": "intervention_registry_v1.json",
        "manifest_hash_lock_format": "lowercase_sha256_two_spaces_filename_sorted_ascii_lf",
        "manifest_lock_expected_sha256_required_out_of_band": True,
        "protocol_bundle_hash_algorithm": "sha256_canonical_json_filename_to_sha256_map",
        "split_manifest_schema": "ordered_per_sequence_dataset_identity_v1",
        "val47_source_policy": "separate_authoritative_manifest_disjoint_from_development187",
    }
    if protocol.get("identity") != expected_identity or protocol.get("inputs") != expected_inputs:
        raise Stage0RunError("manifest identity policy differs from fail-closed Stage 0 v1")
    scope_policy = protocol.get("scope_policy", {})
    if (scope_policy.get("official_scope") != OFFICIAL_SCOPE
            or set(scope_policy.get("sealed_scopes", [])) != {"internal42", "confirm62", "val47"}
            or scope_policy.get("gate_accepts_only_official_scope") is not True
            or scope_policy.get("split_reassignment_forbidden") is not True):
        raise Stage0RunError("scope policy differs from fail-closed Stage 0 v1")
    if metric_schema.get("semantic_attributes") != list(SEMANTIC_ATTRIBUTES):
        raise Stage0RunError("metric schema semantic attributes differ from taxonomy")
    gates = gate.get("gates", {})
    if (gates.get("A", {}).get("pass_value") != 5
            or gates.get("B", {}).get("all_of", [{}])[0].get("value") != 0.2
            or [item.get("value") for item in gates.get("B", {}).get("any_of", [])] != [0.03, 0.05]
            or gates.get("C", {}).get("pass_value") != 4
            or gates.get("C", {}).get("direction_pass_condition", {}).get("value") != 0.3
            or gates.get("D", {}).get("pass_value") != 4
            or gates.get("D", {}).get("semantic_attributes") != list(SEMANTIC_ATTRIBUTES)):
        raise Stage0RunError("gate thresholds differ from Stage 0 v1")
    gate_e = gates.get("E", {}).get("all_of", [])
    if [(item.get("metric"), item.get("value")) for item in gate_e] != [
        ("online_label_leakage_count", 0), ("non_finite_count", 0),
        ("schema_mismatch_count", 0), ("semantic_replay_pass_fraction", 1.0),
    ]:
        raise Stage0RunError("Gate E integrity definition differs from Stage 0 v1")
    bootstrap = protocol.get("bootstrap", {})
    if bootstrap != {
        "aggregation": "sequence_macro", "resampling_unit": "sequence",
        "samples": 10000, "seed": 20260720, "confidence_level": 0.95,
        "interval": "percentile_two_sided",
    }:
        raise Stage0RunError("bootstrap protocol differs from Stage 0 v1")
    return registry, protocol, metric_schema, gate


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _require_runtime()
    if args.workers != 1:
        raise Stage0RunError("preflight and GPU phases support exactly --workers 1; fail closed")
    supplied_modules = {
        name: getattr(args, name) for name in FORMAL_MODULES
    }
    if supplied_modules != FORMAL_MODULES:
        raise Stage0RunError("formal preflight requires the identity-bound tracker modules")
    dataset_root = _directory(args.dataset_root, "dataset root")
    protocol_dir = _directory(args.protocol_dir, "protocol directory")
    split_paths = {
        "development187": _regular_file(args.development_manifest, "development187 manifest"),
        "design83": _regular_file(args.split_manifest, "design83 split manifest"),
        "internal42": _regular_file(args.internal_manifest, "internal42 split manifest"),
        "confirm62": _regular_file(args.confirm_manifest, "confirm62 split manifest"),
        "val47": _regular_file(args.val_manifest, "val47 split manifest"),
    }
    split_lock_path = _regular_file(args.split_manifest_lock, "split manifest hash lock")
    expected_split_lock_sha256 = _expected_sha256(
        args.split_manifest_lock_sha256, "split manifest lock SHA-256"
    )
    attribute_path = _regular_file(args.attribute_manifest, "attribute manifest")
    attribute_lock_path = _regular_file(args.attribute_manifest_lock, "attribute manifest hash lock")
    expected_attribute_lock_sha256 = _expected_sha256(
        args.attribute_manifest_lock_sha256, "attribute manifest lock SHA-256"
    )
    checkpoint = _regular_file(args.checkpoint, "checkpoint")
    expected_checkpoint_sha256 = _expected_sha256(
        args.checkpoint_sha256, "checkpoint SHA-256"
    )
    if sha256_file(checkpoint) != expected_checkpoint_sha256:
        raise Stage0RunError("checkpoint differs from externally frozen SHA-256")
    model_config = _regular_file(args.model_config, "model config")
    expected_model_config_sha256 = _expected_sha256(
        args.model_config_sha256, "model config SHA-256"
    )
    if sha256_file(model_config) != expected_model_config_sha256:
        raise Stage0RunError("model config differs from externally frozen SHA-256")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise Stage0RunError("output directory must not be a symlink")

    manifests, attributes, split_lock, attribute_lock = _load_manifest_lock_layer(
        split_paths=split_paths,
        split_lock_path=split_lock_path,
        expected_split_lock_sha256=expected_split_lock_sha256,
        attribute_path=attribute_path,
        attribute_lock_path=attribute_lock_path,
        expected_attribute_lock_sha256=expected_attribute_lock_sha256,
        dataset=args.dataset,
    )
    development = manifests["development187"]
    design = manifests["design83"]
    internal42 = list(manifests["internal42"].names)
    confirm62 = list(manifests["confirm62"].names)
    attribute_groups = dict(attributes.groups)
    if set(attribute_groups) != set(SEMANTIC_ATTRIBUTES):
        raise Stage0RunError(
            "attribute manifest groups must exactly match the five frozen semantic attributes"
        )
    protocol_paths = _protocol_paths(protocol_dir)
    registry, protocol, metric_schema, gate = _validated_protocol_bundle(protocol_dir)

    sequence_entries = []
    for sequence in design.names:
        sequence_entries.append(_sequence_manifest_dict(
            load_sequence_manifest(dataset_root, sequence, args.dataset)))

    source_hashes = {}
    for name, path in SOURCE_PATHS.items():
        source_hashes[name] = {"path": str(_regular_file(path, f"source {name}")), "sha256": sha256_file(path)}
    identity = with_content_hash({
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "run_identity",
        "scope": OFFICIAL_SCOPE,
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "sequence_count": len(design.names),
        "sequences": list(design.names),
        "split_derivation": {
            "development_count": 187,
            "outer_rule": "sort_by_sha256(sequence)_then_name_take_125",
            "inner_rule": f"sort_by_sha256('{INNER_NAMESPACE}:'+sequence)_then_name_take_83",
            "development125_count": 125,
            "design83_count": 83,
            "internal42_hash": canonical_json_hash(internal42),
            "confirm62_hash": canonical_json_hash(confirm62),
            "val47_source_policy": "separate_authoritative_manifest_disjoint_from_development187",
        },
        "inputs": {
            "split_manifests": {
                role: _split_manifest_identity(split_paths[role], manifest)
                for role, manifest in manifests.items()
            },
            "split_manifest_lock": {
                "path": str(split_lock_path), "sha256": split_lock.sha256,
                "entries": split_lock.as_dict(),
            },
            "attribute_manifest": {
                "path": str(attribute_path), "sha256": attributes.sha256,
                "groups": {name: list(members) for name, members in attribute_groups.items()},
                "sequence_attributes": {
                    name: list(members) for name, members in attributes.sequence_attributes
                },
                "source": dict(attributes.source),
            },
            "attribute_manifest_lock": {
                "path": str(attribute_lock_path), "sha256": attribute_lock.sha256,
                "entries": attribute_lock.as_dict(),
            },
            "checkpoint": {"path": str(checkpoint), "sha256": expected_checkpoint_sha256},
            "model_config": {"path": str(model_config), "sha256": expected_model_config_sha256},
            "protocol_dir": str(protocol_dir),
            "protocol_files": protocol_bundle_manifest(protocol_paths),
            "protocol_bundle_hash": protocol_bundle_hash(protocol_paths),
            "sources": source_hashes,
        },
        "dataset_entries": sequence_entries,
        "frozen_protocol": {
            "schedule": protocol["schedule"],
            "bootstrap": protocol["bootstrap"],
            "gate_hash": canonical_json_hash(gate),
            "metric_schema_hash": canonical_json_hash(metric_schema),
            "registry_hash": canonical_json_hash(registry),
        },
        "execution": {**FORMAL_MODULES, "workers": 1},
    })
    identity_path = output / "run_identity.json"
    if identity_path.exists():
        existing = _read_json(identity_path)
        if existing != identity:
            raise Stage0RunError("existing run identity differs; use a new output directory")
    else:
        atomic_write_json(identity_path, identity)
    _atomic_text(output / "split_manifest_sha256.txt", split_lock_path.read_text(encoding="ascii"))
    _atomic_text(output / "attribute_manifest_sha256.txt", attribute_lock_path.read_text(encoding="ascii"))
    phase = _phase_document("preflight", identity["content_hash"], {
        "status": "COMPLETE",
        "sequence_count": 83,
        "dataset_entry_count": len(sequence_entries),
        "protocol_bundle_hash": identity["inputs"]["protocol_bundle_hash"],
        "run_identity_path": str(identity_path),
    })
    atomic_write_json(_phase_path(output, "preflight"), phase)
    return phase


def _import_identity_bound_module(
    identity: Mapping[str, Any],
    execution_key: str,
    source_key: str,
) -> Any:
    execution = identity["execution"]
    expected_name = FORMAL_MODULES[execution_key]
    if execution.get(execution_key) != expected_name:
        raise Stage0RunError(f"formal module name drift: {execution_key}")
    module = importlib.import_module(expected_name)
    imported_file = getattr(module, "__file__", None)
    if not isinstance(imported_file, str):
        raise Stage0RunError(f"formal module has no source file: {expected_name}")
    imported_path = _regular_file(imported_file, f"imported formal module {expected_name}")
    source = identity["inputs"]["sources"][source_key]
    bound_path = _regular_file(source["path"], f"identity source {source_key}")
    if (imported_path != bound_path
            or sha256_file(imported_path) != source["sha256"]):
        raise Stage0RunError(f"imported module differs from identity-bound source: {expected_name}")
    return module


def _load_runtime(identity: Mapping[str, Any]) -> tuple[Any, Any]:
    parameter_module = _import_identity_bound_module(identity, "parameter_module", "parameters")
    tracker_module = _import_identity_bound_module(identity, "tracker_module", "tracker")
    adapter_module = _import_identity_bound_module(identity, "adapter_module", "adapter")
    model_config = Path(identity["inputs"]["model_config"]["path"])
    parameter_source = Path(identity["inputs"]["sources"]["parameters"]["path"])
    resolved_project_config = parameter_source.parents[3] / "experiments" / "vipt" / model_config.name
    if (resolved_project_config.resolve() != model_config.resolve()
            or sha256_file(resolved_project_config) != identity["inputs"]["model_config"]["sha256"]):
        raise Stage0RunError(
            "parameter module would load a YAML different from the identity-bound model config"
        )
    params = parameter_module.parameters(model_config.stem)
    params.checkpoint = identity["inputs"]["checkpoint"]["path"]
    tracker_class = tracker_module.get_tracker_class()
    tracker = tracker_class(params)
    adapter_class = getattr(adapter_module, "CBEStage0ProbeAdapter")
    return tracker, adapter_class


def _decode_rgb(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise Stage0RunError(f"cannot decode uint8 color frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _frame(identity: Mapping[str, Any], sequence_entry: Mapping[str, Any], index: int) -> np.ndarray:
    root = Path(identity["dataset_root"]) / sequence_entry["relative_root"]
    visible = _decode_rgb(root / sequence_entry["visible_images"][index]["relative_path"])
    infrared = _decode_rgb(root / sequence_entry["infrared_images"][index]["relative_path"])
    return merge_six_channel(visible, infrared)


def _event_schedule(identity: Mapping[str, Any], sequence_entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    protocol_dir = Path(identity["inputs"]["protocol_dir"])
    protocol = load_json_strict(protocol_dir / "stage0_protocol_v1.json")
    registry = load_json_strict(protocol_dir / "intervention_registry_v1.json")
    config = protocol["schedule"]
    schedule = deterministic_opportunity_schedule(
        sequence_entry["sequence"], sequence_entry["frame_count"],
        [item["direction"] for item in registry["primary_directions"]], registry["strengths"],
        warmup=config["warmup_frames"], interval=config["interval_frames"],
        max_opportunities=config["max_opportunities_per_sequence"], seed=config["assignment_seed"],
    )
    for event in schedule:
        event["sequence_name"] = sequence_entry["sequence"]
        event["event_id"] = deterministic_event_id(
            sequence_entry["sequence"], event["frame_index"], event["opportunity_index"],
            event["direction"], event["strengths"], event["assignment_hash"])
    return schedule


def _forward_payload(value: Mapping[str, Any], search_size: int,
                     hann_window: Any) -> dict[str, Any]:
    keys = ("score_map", "size_map", "offset_map", "hann_response", "resize_factor",
            "search_crop_xywh", "target_bbox", "best_score", "search_anchor",
            "anchor_id", "template_id")
    result = {key: _jsonable(value[key]) for key in keys}
    result["hann_window"] = _jsonable(hann_window)
    result["search_size"] = int(search_size)
    return result


_FORWARD_KEYS = {
    "anchor_id", "best_score", "hann_response", "hann_window", "offset_map",
    "resize_factor", "score_map", "search_anchor", "search_crop_xywh",
    "search_size", "size_map", "target_bbox", "template_id",
}
_ONLINE_COMMON_KEYS = {
    "schema_version", "scope", "record_type", "sequence_name", "frame_index",
}
_OPPORTUNITY_KEYS = _ONLINE_COMMON_KEYS | {
    "opportunity_index", "event_id", "assignment_hash", "direction", "strengths",
    "search_anchor_xywh", "factual_template_id", "probes",
}
_RAW_KEYS = {
    "schema_version", "scope", "record_type", "sequence_name", "frame_index",
    "opportunity_index", "event_id", "assignment_hash", "direction", "strengths",
    "source_frame", "search_anchor_xywh", "factual_template_id", "status",
    "strength_arms",
}
_LABEL_KEYS = {
    "schema_version", "scope", "record_type", "sequence_name", "frame_index",
    "opportunity_index", "event_id", "assignment_hash", "direction", "strengths",
    "status", "invalid_reason", "visible_gt_xywh", "tir_gt_xywh",
    "target_mask_gt_xywh", "evaluation_gt_xywh", "registration_iou",
    "target_region_id", "background_region_id",
}


def _require_exact_record_keys(row: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(row, Mapping) or set(row) != expected:
        actual = set(row) if isinstance(row, Mapping) else set()
        raise Stage0RunError(
            f"{name} key mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _validate_official_record(row: Mapping[str, Any], name: str) -> None:
    if (row.get("schema_version") != SCHEMA_VERSION
            or row.get("scope") != OFFICIAL_SCOPE):
        raise Stage0RunError(f"{name} is not an official design83 record")


def _validate_forward_payload(value: Any, name: str) -> None:
    _require_exact_record_keys(value, _FORWARD_KEYS, name)
    reject_non_finite(value)


def _validate_online_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        reject_online_labels(row)
        reject_non_finite(row)
        _validate_official_record(row, "online artifact")
        record_type = row.get("record_type")
        if record_type == "trajectory":
            expected = _ONLINE_COMMON_KEYS | {
                "pred_xywh", "best_score", "template_id",
            }
            if int(row.get("frame_index", -1)) > 0:
                expected = expected | {"search_anchor_xywh"}
            _require_exact_record_keys(row, expected, "trajectory")
        elif record_type == "opportunity":
            _require_exact_record_keys(row, _OPPORTUNITY_KEYS, "opportunity")
            probes = row.get("probes")
            if (not isinstance(probes, Mapping)
                    or set(probes) != {"factual", "rgb_retained", "tir_retained"}):
                raise Stage0RunError("opportunity must contain exactly three clean probes")
            for probe_name, payload in probes.items():
                _validate_forward_payload(payload, f"online probe {probe_name}")
        else:
            raise Stage0RunError(f"unknown online record_type: {record_type!r}")


def _validate_intervention_rows(raw_rows: Sequence[Mapping[str, Any]],
                                label_rows: Sequence[Mapping[str, Any]]) -> None:
    for raw in raw_rows:
        _require_exact_record_keys(raw, _RAW_KEYS, "intervention raw")
        _validate_official_record(raw, "intervention raw")
        reject_online_labels(raw)
        if raw.get("record_type") != "primary_local_intervention_raw":
            raise Stage0RunError("invalid intervention raw record_type")
        arms = raw.get("strength_arms")
        if not isinstance(arms, list):
            raise Stage0RunError("intervention strength_arms must be an array")
        for arm in arms:
            _require_exact_record_keys(arm, {"strength", "regions"}, "strength arm")
            regions = arm.get("regions")
            if not isinstance(regions, Mapping) or len(regions) != 2:
                raise Stage0RunError("strength arm must contain exactly two opaque regions")
            for region_id, payload in regions.items():
                if not isinstance(region_id, str) or not region_id:
                    raise Stage0RunError("opaque region ID must be a non-empty string")
                _validate_forward_payload(payload, f"intervention region {region_id}")
    for label in label_rows:
        expected = set(_LABEL_KEYS)
        if label.get("status") == "VALID":
            expected |= {
                "target_pixel_count", "background_pixel_count", "background_offset_yx",
            }
        _require_exact_record_keys(label, expected, "evaluator label")
        _validate_official_record(label, "evaluator label")
        reject_non_finite(label)
        if label.get("record_type") != "evaluator_label":
            raise Stage0RunError("invalid evaluator label record_type")


def _sequence_completion_path(root: Path, phase: str, sequence: str) -> Path:
    return root / phase / "sequences" / sequence / "completion.json"


def _completed_sequence(root: Path, phase: str, sequence: str,
                        identity_hash: str, files: Sequence[str]) -> bool:
    completion_path = _sequence_completion_path(root, phase, sequence)
    sequence_dir = completion_path.parent
    if not sequence_dir.exists():
        return False
    if not completion_path.is_file():
        raise Stage0RunError(f"incomplete {phase} sequence directory exists: {sequence_dir}")
    completion = _read_json(completion_path)
    validate_content_hash(completion)
    if completion.get("identity_hash") != identity_hash or completion.get("sequence_name") != sequence:
        raise Stage0RunError(f"identity mismatch in resumed {phase} sequence: {sequence}")
    expected = completion.get("file_sha256")
    if not isinstance(expected, dict) or set(expected) != set(files):
        raise Stage0RunError(f"completion file list mismatch for {phase}/{sequence}")
    for name in files:
        if sha256_file(sequence_dir / name) != expected[name]:
            raise Stage0RunError(f"completed artifact hash mismatch: {phase}/{sequence}/{name}")
    return True


def _authenticate_completion(root: Path, phase: str, sequence: str,
                             identity_hash: str, files: Sequence[str],
                             expected_content_hash: str) -> dict[str, Any]:
    if not _completed_sequence(root, phase, sequence, identity_hash, files):
        raise Stage0RunError(f"incomplete authenticated sequence: {phase}/{sequence}")
    completion = _read_json(_sequence_completion_path(root, phase, sequence))
    if completion["content_hash"] != expected_content_hash:
        raise Stage0RunError(
            f"phase-bound completion hash mismatch: {phase}/{sequence}"
        )
    return completion


def _publish_sequence(temp_dir: Path, final_dir: Path, identity_hash: str,
                      sequence: str, phase: str, files: Sequence[str], counts: Mapping[str, int],
                      input_completion_hash: str | None = None) -> None:
    completion = with_content_hash({
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "sequence_completion",
        "phase": phase,
        "scope": OFFICIAL_SCOPE,
        "sequence_name": sequence,
        "identity_hash": identity_hash,
        "file_sha256": {name: sha256_file(temp_dir / name) for name in files},
        "counts": dict(counts),
        "input_completion_hash": input_completion_hash,
        "status": "COMPLETE",
    })
    atomic_write_json(temp_dir / "completion.json", completion)
    if final_dir.exists():
        raise Stage0RunError(f"refusing to replace existing sequence directory: {final_dir}")
    os.replace(temp_dir, final_dir)


def _online_sequence(root: Path, identity: Mapping[str, Any], entry: Mapping[str, Any]) -> None:
    sequence = entry["sequence"]
    identity_hash = identity["content_hash"]
    files = ("trajectory.jsonl", "opportunities.jsonl")
    if _completed_sequence(root, "online", sequence, identity_hash, files):
        return
    parent = root / "online" / "sequences"
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{sequence}.", dir=str(parent)))
    try:
        tracker, adapter_class = _load_runtime(identity)
        frame0 = _frame(identity, entry, 0)
        manifest = load_sequence_manifest(identity["dataset_root"], sequence, identity["dataset"])
        init_box = list(manifest.visible_annotation.boxes_xywh[0])
        tracker.initialize(frame0, {"init_bbox": init_box})
        adapter = adapter_class(tracker, frame0, init_box)
        schedule = _event_schedule(identity, entry)
        by_frame = {event["frame_index"]: event for event in schedule}
        trajectory = [{
            "schema_version": SCHEMA_VERSION, "scope": OFFICIAL_SCOPE,
            "record_type": "trajectory", "sequence_name": sequence, "frame_index": 0,
            "pred_xywh": init_box, "best_score": None, "template_id": adapter.factual_snapshot.template_id,
        }]
        opportunities = []
        for frame_index in range(1, entry["frame_count"]):
            image = _frame(identity, entry, frame_index)
            anchor = [float(value) for value in tracker.state]
            if frame_index in by_frame:
                event = by_frame[frame_index]
                probes = adapter.run_clean_probe_set(image, anchor)
                row = {
                    "schema_version": SCHEMA_VERSION, "scope": OFFICIAL_SCOPE,
                    "record_type": "opportunity", "sequence_name": sequence,
                    "frame_index": frame_index, "opportunity_index": event["opportunity_index"],
                    "event_id": event["event_id"], "assignment_hash": event["assignment_hash"],
                    "direction": event["direction"], "strengths": event["strengths"],
                    "search_anchor_xywh": anchor,
                    "factual_template_id": adapter.factual_snapshot.template_id,
                    "probes": {name: _forward_payload(value, tracker.params.search_size, tracker.output_window)
                               for name, value in probes.items()},
                }
                opportunities.append(row)
            tracked = adapter.advance_factual(image, anchor)
            trajectory.append({
                "schema_version": SCHEMA_VERSION, "scope": OFFICIAL_SCOPE,
                "record_type": "trajectory", "sequence_name": sequence,
                "frame_index": frame_index, "pred_xywh": tracked["target_bbox"],
                "best_score": tracked["best_score"], "search_anchor_xywh": anchor,
                "template_id": tracked["template_id"],
            })
        _validate_online_rows(trajectory)
        _validate_online_rows(opportunities)
        atomic_write_jsonl(temp / files[0], trajectory)
        atomic_write_jsonl(temp / files[1], opportunities)
        _publish_sequence(temp, parent / sequence, identity_hash, sequence, "online", files,
                          {"frames": len(trajectory), "opportunities": len(opportunities)})
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def run_online(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers != 1:
        raise Stage0RunError("--workers > 1 is not safely supported on one GPU; fail closed")
    parent_path, parent = _load_parent(args.parent, "preflight")
    root = parent_path.parent
    if Path(args.output_dir).resolve() != root:
        raise Stage0RunError("output-dir must equal the preflight artifact directory")
    identity = _identity_from_root(root, parent["identity_hash"])
    for entry in identity["dataset_entries"]:
        _online_sequence(root, identity, entry)
    sequence_hashes = {}
    for sequence in identity["sequences"]:
        completion = _read_json(_sequence_completion_path(root, "online", sequence))
        sequence_hashes[sequence] = completion["content_hash"]
    phase = _phase_document("online", identity["content_hash"], {
        "status": "COMPLETE", "sequence_count": len(sequence_hashes),
        "sequence_completion_hashes": sequence_hashes,
        "raw_artifact_root": str(root / "online" / "sequences"),
    }, parent)
    _validate_child_parent(phase, parent)
    atomic_write_json(_phase_path(root, "online"), phase)
    return phase


def _region_id(event_id: str, arm: str) -> str:
    return canonical_json_hash({"event_id": event_id, "opaque_region": arm})


def _search_valid_support_from_crop(image_shape: Sequence[int],
                                    crop_xywh: Sequence[float]) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    if len(crop_xywh) != 4:
        raise ValueError("search crop must be xywh")
    crop = [float(value) for value in crop_xywh]
    if (not all(math.isfinite(value) for value in crop)
            or crop[2] <= 0.0 or crop[3] <= 0.0
            or crop[2] != crop[3]
            or any(value != round(value) for value in crop)):
        raise ValueError("search crop must be a finite integer square")
    x1, y1, crop_size = int(crop[0]), int(crop[1]), int(crop[2])
    x2, y2 = x1 + crop_size, y1 + crop_size
    x1_pad, y1_pad = max(0, -x1), max(0, -y1)
    x2_pad, y2_pad = max(x2 - width + 1, 0), max(y2 - height + 1, 0)
    left, right = x1 + x1_pad, x2 - x2_pad
    top, bottom = y1 + y1_pad, y2 - y2_pad
    support = np.zeros((height, width), dtype=bool)
    if top < bottom and left < right:
        support[top:bottom, left:right] = True
    return support


def _search_valid_support(image_shape: Sequence[int], anchor: Sequence[float],
                          search_factor: float) -> np.ndarray:
    x, y, box_width, box_height = (float(value) for value in anchor[:4])
    crop_size = math.ceil(math.sqrt(box_width * box_height) * float(search_factor))
    crop_x = round(x + 0.5 * box_width - 0.5 * crop_size)
    crop_y = round(y + 0.5 * box_height - 0.5 * crop_size)
    return _search_valid_support_from_crop(
        image_shape, [crop_x, crop_y, crop_size, crop_size]
    )


def _invalid_replay_report(event_id: str, reason: str,
                           registration_iou: float) -> dict[str, Any]:
    check = {
        "check_id": "deterministic_invalid_recorded",
        "passed": bool(
            (reason == "registration_discordant" and registration_iou < 0.5)
            or reason in {"target_mask_invalid", "matched_background_unavailable"}
        ),
        "registration_iou": float(registration_iou),
    }
    return with_content_hash({
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "invalid_intervention_replay",
        "event_id": event_id,
        "invalid_reason": reason,
        "check_count": 1,
        "passed_count": int(check["passed"]),
        "pass_fraction": float(check["passed"]),
        "passed": bool(check["passed"]),
        "checks": [check],
    })


def _intervene_sequence(root: Path, identity: Mapping[str, Any], entry: Mapping[str, Any]) -> None:
    sequence = entry["sequence"]
    identity_hash = identity["content_hash"]
    files = ("raw_forwards.jsonl", "evaluator_labels.jsonl", "replay_reports.jsonl")
    online_dir = root / "online" / "sequences" / sequence
    if not _completed_sequence(root, "online", sequence, identity_hash, ("trajectory.jsonl", "opportunities.jsonl")):
        raise Stage0RunError(f"online sequence is incomplete: {sequence}")
    online_completion = _read_json(
        _sequence_completion_path(root, "online", sequence)
    )
    if _completed_sequence(root, "intervene", sequence, identity_hash, files):
        intervene_completion = _read_json(
            _sequence_completion_path(root, "intervene", sequence)
        )
        if (intervene_completion.get("input_completion_hash")
                != online_completion["content_hash"]):
            raise Stage0RunError(
                f"resumed intervention input binding mismatch: {sequence}"
            )
        return
    opportunities = _read_jsonl(online_dir / "opportunities.jsonl")
    parent = root / "intervene" / "sequences"
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{sequence}.", dir=str(parent)))
    try:
        manifest = load_sequence_manifest(identity["dataset_root"], sequence, identity["dataset"])
        frame0 = _frame(identity, entry, 0)
        tracker, adapter_class = _load_runtime(identity)
        init_box = list(manifest.visible_annotation.boxes_xywh[0])
        tracker.initialize(frame0, {"init_bbox": init_box})
        adapter = adapter_class(tracker, frame0, init_box)
        raw_rows, label_rows, replay_rows = [], [], []
        for online in opportunities:
            frame_index = int(online["frame_index"])
            direction = online["direction"]
            if direction not in DIRECTION_SPEC:
                raise Stage0RunError(f"unimplemented primary intervention family: {direction}")
            operation, modality = DIRECTION_SPEC[direction]
            image = _frame(identity, entry, frame_index)
            visible_gt = list(manifest.visible_annotation.boxes_xywh[frame_index])
            tir_gt = list(manifest.infrared_annotation.boxes_xywh[frame_index])
            target_gt = visible_gt if modality == "rgb" else tir_gt
            evaluation_gt = visible_gt
            target_id = _region_id(online["event_id"], "target")
            background_id = _region_id(online["event_id"], "background")
            invalid_reason = None
            target = background = None
            registration_iou = iou_xywh(visible_gt, tir_gt)
            if registration_iou < 0.5:
                invalid_reason = "registration_discordant"
            else:
                try:
                    target = target_mask_from_xywh(
                        image.shape, target_gt, expansion=1.25,
                        min_clip_retention=0.9, min_pixels=16,
                    )
                except ValueError:
                    invalid_reason = "target_mask_invalid"
                if target is not None:
                    try:
                        background = matched_background_mask(
                            target.mask,
                            seed_key=f"{online['event_id']}:placement",
                            valid_support=_search_valid_support(
                                image.shape, online["search_anchor_xywh"],
                                tracker.params.search_factor,
                            ),
                        )
                    except ValueError:
                        invalid_reason = "matched_background_unavailable"

            raw_record = {
                "schema_version": SCHEMA_VERSION, "scope": OFFICIAL_SCOPE,
                "record_type": "primary_local_intervention_raw", "sequence_name": sequence,
                "frame_index": frame_index, "opportunity_index": online["opportunity_index"],
                "event_id": online["event_id"], "assignment_hash": online["assignment_hash"],
                "direction": direction, "strengths": list(online["strengths"]),
                "source_frame": frame_index, "search_anchor_xywh": online["search_anchor_xywh"],
                "factual_template_id": online["factual_template_id"],
                "status": "INVALID_OPPORTUNITY" if invalid_reason else "VALID",
                "strength_arms": [],
            }
            label_record = {
                "schema_version": SCHEMA_VERSION, "scope": OFFICIAL_SCOPE,
                "record_type": "evaluator_label", "sequence_name": sequence,
                "frame_index": frame_index, "opportunity_index": online["opportunity_index"],
                "event_id": online["event_id"], "assignment_hash": online["assignment_hash"],
                "direction": direction, "strengths": list(online["strengths"]),
                "status": raw_record["status"], "invalid_reason": invalid_reason,
                "visible_gt_xywh": visible_gt, "tir_gt_xywh": tir_gt,
                "target_mask_gt_xywh": target_gt,
                "evaluation_gt_xywh": evaluation_gt,
                "registration_iou": registration_iou,
                "target_region_id": target_id, "background_region_id": background_id,
            }
            if invalid_reason is None:
                label_record.update({
                    "target_pixel_count": target.pixel_count,
                    "background_pixel_count": background.pixel_count,
                    "background_offset_yx": list(background.offset_yx),
                })
                replay_record = dict(raw_record)
                replay_record.update({
                    "target_pixel_count": target.pixel_count,
                    "background_pixel_count": background.pixel_count,
                    "background_offset_yx": list(background.offset_yx),
                })
                replay_arms = []
                for strength in online["strengths"]:
                    spec = InterventionSpec(
                        operation, modality, float(strength), seed_key=online["event_id"]
                    )
                    pair = apply_paired_local_intervention(
                        image, target.mask, background.mask, spec
                    )
                    target_forward = adapter.predict(
                        pair.target, online["search_anchor_xywh"], adapter.factual_snapshot
                    )
                    background_forward = adapter.predict(
                        pair.background, online["search_anchor_xywh"], adapter.factual_snapshot
                    )
                    raw_record["strength_arms"].append({
                        "strength": float(strength),
                        "regions": {
                            target_id: _forward_payload(
                                target_forward, tracker.params.search_size, tracker.output_window
                            ),
                            background_id: _forward_payload(
                                background_forward, tracker.params.search_size, tracker.output_window
                            ),
                        },
                    })
                    replay_arms.append({
                        "strength": float(strength), "seed": int(pair.seed),
                        "parameters": _jsonable(pair.parameters),
                    })
                replay_record["strength_arms"] = replay_arms
                scheduled = dict(online)
                scheduled["template_id"] = online["factual_template_id"]
                replay = replay_intervention_record(
                    replay_record, scheduled_event=scheduled,
                    expected_target_pixel_count=target.pixel_count,
                    expected_background_pixel_count=background.pixel_count,
                    expected_background_offset_yx=background.offset_yx,
                    expected_seed_by_strength={
                        arm["strength"]: arm["seed"] for arm in replay_arms
                    },
                )
                if not replay["passed"]:
                    raise Stage0RunError(
                        f"intervention semantic replay failed: {online['event_id']}"
                    )
            else:
                replay = _invalid_replay_report(
                    online["event_id"], invalid_reason, registration_iou
                )
            raw_rows.append(raw_record)
            replay_rows.append(replay)
            label_rows.append(label_record)
        if len(raw_rows) != len(opportunities):
            raise Stage0RunError("partial intervention output is forbidden")
        _validate_intervention_rows(raw_rows, label_rows)
        atomic_write_jsonl(temp / files[0], raw_rows)
        atomic_write_jsonl(temp / files[1], label_rows)
        atomic_write_jsonl(temp / files[2], replay_rows)
        _publish_sequence(
            temp, parent / sequence, identity_hash, sequence, "intervene", files,
            {"events": len(raw_rows), "labels": len(label_rows), "replays": len(replay_rows)},
            input_completion_hash=online_completion["content_hash"],
        )
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def run_intervene(args: argparse.Namespace) -> dict[str, Any]:
    parent_path, parent = _load_parent(args.parent, "online")
    root = parent_path.parent
    if Path(args.output_dir).resolve() != root:
        raise Stage0RunError("output-dir must equal the online artifact directory")
    identity = _identity_from_root(root, parent["identity_hash"])
    online_hashes = parent.get("payload", {}).get("sequence_completion_hashes")
    if not isinstance(online_hashes, dict) or set(online_hashes) != set(identity["sequences"]):
        raise Stage0RunError("online parent lacks exact sequence completion bindings")
    for sequence in identity["sequences"]:
        _authenticate_completion(
            root, "online", sequence, identity["content_hash"],
            ("trajectory.jsonl", "opportunities.jsonl"), online_hashes[sequence],
        )
    for entry in identity["dataset_entries"]:
        _intervene_sequence(root, identity, entry)
    completions = {sequence: _read_json(_sequence_completion_path(root, "intervene", sequence))["content_hash"]
                   for sequence in identity["sequences"]}
    phase = _phase_document("intervene", identity["content_hash"], {
        "status": "COMPLETE", "sequence_count": len(completions),
        "sequence_completion_hashes": completions,
        "primary_families": list(PRIMARY_DIRECTIONS),
        "secondary_controls": "not_scheduled",
        "raw_labels_separated": True,
    }, parent)
    _validate_child_parent(phase, parent)
    atomic_write_json(_phase_path(root, "intervene"), phase)
    return phase


def _spatial(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise Stage0RunError(f"forward map does not reduce to HxW: {array.shape}")
    return array


def _validate_forward_context(online: Mapping[str, Any], raw: Mapping[str, Any]) -> None:
    probes = online["probes"]
    factual = probes["factual"]
    context_fields = (
        "search_anchor", "anchor_id", "search_crop_xywh", "resize_factor",
        "template_id", "search_size",
    )
    if (factual["search_anchor"] != online["search_anchor_xywh"]
            or factual["template_id"] != online["factual_template_id"]):
        raise Stage0RunError(f"factual forward context mismatch: {online['event_id']}")
    for probe_name, payload in probes.items():
        for field in context_fields:
            if payload[field] != factual[field]:
                raise Stage0RunError(
                    f"clean probe context mismatch for {online['event_id']}: "
                    f"{probe_name}/{field}"
                )
    for arm in raw["strength_arms"]:
        for region_id, payload in arm["regions"].items():
            for field in context_fields:
                if payload[field] != factual[field]:
                    raise Stage0RunError(
                        f"counterfactual context mismatch for {online['event_id']}: "
                        f"{region_id}/{field}"
                    )


def _recompute_intervention_geometry(entry: Mapping[str, Any],
                                     online: Mapping[str, Any],
                                     label: Mapping[str, Any]) -> dict[str, Any]:
    frame_index = int(online["frame_index"])
    visible_gt = list(label["visible_gt_xywh"])
    tir_gt = list(label["tir_gt_xywh"])
    registration_iou = iou_xywh(visible_gt, tir_gt)
    if not math.isclose(
        registration_iou, float(label["registration_iou"]), rel_tol=0.0, abs_tol=1e-12
    ):
        raise Stage0RunError(f"registration IoU mismatch: {online['event_id']}")
    if registration_iou < 0.5:
        return {"invalid_reason": "registration_discordant"}
    image_shape = entry["visible_images"][frame_index]["shape"]
    modality = DIRECTION_SPEC[online["direction"]][1]
    target_gt = visible_gt if modality == "rgb" else tir_gt
    if list(label["target_mask_gt_xywh"]) != target_gt:
        raise Stage0RunError(f"target-mask GT modality mismatch: {online['event_id']}")
    if list(label["evaluation_gt_xywh"]) != visible_gt:
        raise Stage0RunError(f"evaluation GT must be visible GT: {online['event_id']}")
    try:
        target = target_mask_from_xywh(
            image_shape, target_gt, expansion=1.25,
            min_clip_retention=0.9, min_pixels=16,
        )
    except ValueError:
        return {"invalid_reason": "target_mask_invalid"}
    try:
        background = matched_background_mask(
            target.mask,
            seed_key=f"{online['event_id']}:placement",
            valid_support=_search_valid_support_from_crop(
                image_shape, online["probes"]["factual"]["search_crop_xywh"]
            ),
        )
    except ValueError:
        return {"invalid_reason": "matched_background_unavailable"}
    return {
        "invalid_reason": None,
        "target_pixel_count": target.pixel_count,
        "background_pixel_count": background.pixel_count,
        "background_offset_yx": list(background.offset_yx),
    }


def _evidence(forward: Mapping[str, Any], gt: Sequence[float], anchor: Sequence[float]) -> dict[str, Any]:
    score = _spatial(forward["score_map"])
    window = _spatial(forward["hann_window"])
    weights = gt_fractional_cell_weights(
        gt, anchor, forward["search_size"], forward["resize_factor"], score.shape,
        search_crop_xywh=forward["search_crop_xywh"])
    result = evidence_metrics(score, window, weights, forward["size_map"], forward["offset_map"])
    result["belief_map"] = stable_softmax(score, temperature=1.0).tolist()
    result["gt_weights"] = weights.tolist()
    result["forward_best_score"] = float(forward["best_score"])
    result["box_iou"] = iou_xywh(forward["target_bbox"], gt)
    return result


def _joined_events(root: Path, identity: Mapping[str, Any],
                   online_hashes: Mapping[str, str],
                   intervene_hashes: Mapping[str, str]) -> list[dict[str, Any]]:
    joined = []
    shared_fields = (
        "sequence_name", "frame_index", "opportunity_index", "event_id",
        "assignment_hash", "direction", "strengths",
    )
    entries = {entry["sequence"]: entry for entry in identity["dataset_entries"]}
    if set(entries) != set(identity["sequences"]):
        raise Stage0RunError("identity dataset entries do not exactly cover sequences")
    for sequence in identity["sequences"]:
        _authenticate_completion(
            root, "online", sequence, identity["content_hash"],
            ("trajectory.jsonl", "opportunities.jsonl"), online_hashes[sequence],
        )
        intervene_completion = _authenticate_completion(
            root, "intervene", sequence, identity["content_hash"],
            ("raw_forwards.jsonl", "evaluator_labels.jsonl", "replay_reports.jsonl"),
            intervene_hashes[sequence],
        )
        if intervene_completion.get("input_completion_hash") != online_hashes[sequence]:
            raise Stage0RunError(
                f"intervention is not bound to authenticated online input: {sequence}"
            )
        online = _read_jsonl(
            root / "online" / "sequences" / sequence / "opportunities.jsonl"
        )
        expected_schedule = _event_schedule(identity, entries[sequence])
        observed_schedule = [{
            "assignment_hash": row.get("assignment_hash"),
            "direction": row.get("direction"),
            "event_id": row.get("event_id"),
            "frame_index": row.get("frame_index"),
            "opportunity_index": row.get("opportunity_index"),
            "sequence_name": row.get("sequence_name"),
            "strengths": row.get("strengths"),
        } for row in online]
        if observed_schedule != expected_schedule:
            raise Stage0RunError(f"online opportunities differ from frozen schedule: {sequence}")
        raw = _read_jsonl(
            root / "intervene" / "sequences" / sequence / "raw_forwards.jsonl"
        )
        labels = _read_jsonl(
            root / "intervene" / "sequences" / sequence / "evaluator_labels.jsonl"
        )
        _validate_online_rows(online)
        _validate_intervention_rows(raw, labels)
        indices = []
        for rows, name in ((online, "online"), (raw, "raw"), (labels, "labels")):
            index = {}
            for row in rows:
                event_id = row.get("event_id")
                if not isinstance(event_id, str) or event_id in index:
                    raise Stage0RunError(f"invalid or duplicate {name} event_id: {event_id}")
                index[event_id] = row
            indices.append(index)
        if not (set(indices[0]) == set(indices[1]) == set(indices[2])):
            raise Stage0RunError(f"strict online/raw/labels join mismatch for {sequence}")
        for event_id in sorted(
            indices[0], key=lambda key: indices[0][key]["opportunity_index"]
        ):
            online_row, raw_row, label_row = (
                indices[0][event_id], indices[1][event_id], indices[2][event_id]
            )
            for field in shared_fields:
                if not (online_row.get(field) == raw_row.get(field) == label_row.get(field)):
                    raise Stage0RunError(
                        f"strict join field mismatch for {sequence}/{event_id}: {field}"
                    )
            if (set(online_row.get("probes", {}))
                    != {"factual", "rgb_retained", "tir_retained"}):
                raise Stage0RunError(f"clean probe set mismatch for {sequence}/{event_id}")
            if online_row.get("strengths") != [0.25, 0.5, 0.75]:
                raise Stage0RunError(f"strength grid mismatch for {sequence}/{event_id}")
            if (raw_row.get("source_frame") != online_row.get("frame_index")
                    or raw_row.get("search_anchor_xywh") != online_row.get("search_anchor_xywh")
                    or raw_row.get("factual_template_id") != online_row.get("factual_template_id")):
                raise Stage0RunError(
                    f"raw forward context mismatch for {sequence}/{event_id}"
                )
            if raw_row.get("status") != label_row.get("status"):
                raise Stage0RunError(f"status mismatch for {sequence}/{event_id}")
            if raw_row["status"] == "VALID":
                expected_regions = {
                    label_row.get("target_region_id"), label_row.get("background_region_id")
                }
                if None in expected_regions or len(expected_regions) != 2:
                    raise Stage0RunError(f"invalid evaluator region mapping: {event_id}")
                arms = raw_row.get("strength_arms", [])
                if ([arm.get("strength") for arm in arms]
                        != online_row["strengths"]):
                    raise Stage0RunError(f"strength arm coverage mismatch: {event_id}")
                for arm in arms:
                    if set(arm.get("regions", {})) != expected_regions:
                        raise Stage0RunError(f"opaque region join mismatch: {event_id}")
            elif raw_row["status"] != "INVALID_OPPORTUNITY" or raw_row.get("strength_arms"):
                raise Stage0RunError(f"invalid opportunity encoding mismatch: {event_id}")
            joined.append({"online": online_row, "raw": raw_row, "label": label_row})
    return joined


def _aggregate_gate_inputs(event_metrics: Sequence[Mapping[str, Any]], sequences: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    direction_data: dict[str, Any] = {}
    for direction in PRIMARY_DIRECTIONS:
        selected = [
            event for event in event_metrics
            if event["direction"] == direction and event["strength_metrics"]
        ]
        target_by_sequence: dict[str, list[float]] = defaultdict(list)
        faithful_by_sequence: dict[str, list[float]] = defaultdict(list)
        strength_by_sequence: dict[str, list[list[float]]] = defaultdict(list)
        rho_by_sequence: dict[str, list[float]] = defaultdict(list)
        for event in selected:
            sequence = event["sequence_name"]
            pairs = []
            for arm in event["strength_metrics"]:
                target_by_sequence[sequence].append(arm["target_effect"])
                faithful_by_sequence[sequence].append(arm["faithfulness"])
                pair = [arm["strength"], arm["target_effect"]]
                strength_by_sequence[sequence].append(pair)
                pairs.append(pair)
            rho_by_sequence[sequence].append(
                spearman_rho(
                    [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                )
            )
        direction_data[direction] = {
            "target_effects_by_sequence": dict(target_by_sequence),
            "faithfulness_by_sequence": dict(faithful_by_sequence),
            "strength_effects_by_sequence": dict(strength_by_sequence),
            "strength_rho_by_sequence": dict(rho_by_sequence),
            "rho_unit": "same_frame_opportunity_then_sequence_mean",
        }
    clean_flags: dict[str, list[float]] = defaultdict(list)
    covered: dict[str, list[float]] = {}
    events_by_sequence: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in event_metrics:
        events_by_sequence[event["sequence_name"]].append(event)
        clean_flags[event["sequence_name"]].append(float(event["clean_fusion"]["negative_fusion"]))
    for sequence in sequences:
        rows = events_by_sequence.get(sequence, [])
        covered[sequence] = [float(any(row["clean_fusion"]["negative_fusion"] for row in rows))]
        clean_flags.setdefault(sequence, [0.0])
    density = {
        "clean_negative_fusion_by_sequence": dict(clean_flags),
        "coverage_by_sequence": covered,
        "degradation_probe_status": "not_executed",
    }
    return direction_data, density


def _core_gate_signature(gates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "A": {
            "passed": bool(gates["A"]["passed"]),
            "directions": {
                direction: bool(gates["A"]["directions"][direction]["passed"])
                for direction in PRIMARY_DIRECTIONS
            },
        },
        "B": {"passed": bool(gates["B"]["passed"])},
        "C": {
            "passed": bool(gates["C"]["passed"]),
            "directions": {
                direction: bool(gates["C"]["directions"][direction]["passed"])
                for direction in PRIMARY_DIRECTIONS
            },
        },
    }


def _compute_logo(event_metrics: Sequence[Mapping[str, Any]], identity: Mapping[str, Any],
                  full_direction_data: Mapping[str, Any],
                  full_density_data: Mapping[str, Any]) -> dict[str, Any]:
    groups = identity["inputs"]["attribute_manifest"]["groups"]
    if set(groups) != set(SEMANTIC_ATTRIBUTES):
        raise Stage0RunError(
            "attribute LOGO names do not exactly match metrics_v1.SEMANTIC_ATTRIBUTES; INVALID rather than remapping")
    bootstrap = identity["frozen_protocol"]["bootstrap"]
    full_gates = {
        "A": evaluate_gate_a(
            full_direction_data, seed=bootstrap["seed"], samples=bootstrap["samples"],
            confidence=bootstrap["confidence_level"],
        ),
        "B": evaluate_gate_b(full_density_data),
        "C": evaluate_gate_c(full_direction_data),
    }
    baseline = _core_gate_signature(full_gates)

    def callback(retained: Sequence[Mapping[str, Any]],
                 retained_sequences: Sequence[str]) -> dict[str, Any]:
        if not retained_sequences:
            raise Stage0RunError("LOGO omission retained no sequences")
        direction_data, density_data = _aggregate_gate_inputs(
            retained, retained_sequences
        )
        gates = {
            "A": evaluate_gate_a(
                direction_data, seed=bootstrap["seed"], samples=bootstrap["samples"],
                confidence=bootstrap["confidence_level"],
            ),
            "B": evaluate_gate_b(density_data),
            "C": evaluate_gate_c(direction_data),
        }
        signature = _core_gate_signature(gates)
        consistent = signature == baseline
        return {
            "direction_consistent": consistent,
            "baseline_core_gate_signature": baseline,
            "recomputed_core_gate_signature": signature,
            "recomputed_gates": gates,
            "retained_data_hash": canonical_json_hash(retained),
        }

    attributes = {}
    all_sequences = list(identity["sequences"])
    for attribute in SEMANTIC_ATTRIBUTES:
        omitted = set(groups[attribute])
        retained_sequences = [
            sequence for sequence in all_sequences if sequence not in omitted
        ]
        retained = [
            row for row in event_metrics if row["sequence_name"] not in omitted
        ]
        audit = callback(retained, retained_sequences)
        audit.update({
            "attribute": attribute,
            "omitted_sequence_count": len(omitted),
            "retained_sequence_count": len(retained_sequences),
        })
        attributes[attribute] = audit
    return {
        "method": "leave_one_overlapping_group_out",
        "groups_overlap_allowed": True,
        "attributes": attributes,
        "direction_consistent_count": sum(
            int(audit["direction_consistent"]) for audit in attributes.values()
        ),
    }


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    parent_path, parent = _load_parent(args.parent, "intervene")
    root = parent_path.parent
    if Path(args.output_dir).resolve() != root:
        raise Stage0RunError("output-dir must equal the intervene artifact directory")
    identity = _identity_from_root(root, parent["identity_hash"])
    online_phase = _read_json(_phase_path(root, "online"))
    validate_content_hash(online_phase)
    if parent.get("parent_content_hash") != compute_content_hash(online_phase):
        raise Stage0RunError("intervene parent is not bound to the current online phase")
    online_hashes = online_phase.get("payload", {}).get("sequence_completion_hashes")
    intervene_hashes = parent.get("payload", {}).get("sequence_completion_hashes")
    if (not isinstance(online_hashes, dict) or not isinstance(intervene_hashes, dict)
            or set(online_hashes) != set(identity["sequences"])
            or set(intervene_hashes) != set(identity["sequences"])):
        raise Stage0RunError("phase artifacts lack exact sequence completion bindings")
    joined = _joined_events(root, identity, online_hashes, intervene_hashes)
    entry_index = {entry["sequence"]: entry for entry in identity["dataset_entries"]}
    event_metrics = []
    invalid_opportunities = []
    for item in joined:
        online, raw, label = item["online"], item["raw"], item["label"]
        _validate_forward_context(online, raw)
        geometry = _recompute_intervention_geometry(
            entry_index[online["sequence_name"]], online, label
        )
        if geometry["invalid_reason"] != label.get("invalid_reason"):
            raise Stage0RunError(
                f"intervention invalid reason mismatch: {online['event_id']}"
            )
        expected_status = "INVALID_OPPORTUNITY" if geometry["invalid_reason"] else "VALID"
        if raw["status"] != expected_status or label["status"] != expected_status:
            raise Stage0RunError(f"intervention status mismatch: {online['event_id']}")
        if expected_status == "VALID":
            for field in (
                "target_pixel_count", "background_pixel_count", "background_offset_yx",
            ):
                if label.get(field) != geometry[field]:
                    raise Stage0RunError(
                        f"intervention geometry mismatch for {online['event_id']}: {field}"
                    )
        gt = label["evaluation_gt_xywh"]
        anchor = online["search_anchor_xywh"]
        clean_evidence = {
            name: _evidence(value, gt, anchor)
            for name, value in online["probes"].items()
        }
        clean_fusion = fusion_regret_metrics(
            clean_evidence["factual"]["box_iou"],
            [clean_evidence["rgb_retained"]["box_iou"],
             clean_evidence["tir_retained"]["box_iou"]],
        )
        strengths = []
        if raw["status"] == "INVALID_OPPORTUNITY":
            invalid_opportunities.append({
                "sequence_name": online["sequence_name"],
                "frame_index": online["frame_index"],
                "opportunity_index": online["opportunity_index"],
                "event_id": online["event_id"],
                "direction": online["direction"],
                "invalid_reason": label["invalid_reason"],
            })
        target_id = label["target_region_id"]
        background_id = label["background_region_id"]
        for arm in raw["strength_arms"]:
            target = _evidence(arm["regions"][target_id], gt, anchor)
            background = _evidence(arm["regions"][background_id], gt, anchor)
            effects = intervention_metrics(clean_evidence["factual"]["belief_gt_mass"],
                                           target["belief_gt_mass"], background["belief_gt_mass"])
            strengths.append({
                "strength": arm["strength"], **effects,
                "target_box_iou": target["box_iou"], "background_box_iou": background["box_iou"],
                "target_evidence": target, "background_evidence": background,
            })
        event_metrics.append({
            "schema_version": SCHEMA_VERSION, "scope": OFFICIAL_SCOPE,
            "sequence_name": online["sequence_name"], "frame_index": online["frame_index"],
            "opportunity_index": online["opportunity_index"], "event_id": online["event_id"],
            "direction": online["direction"], "clean_evidence": clean_evidence,
            "clean_fusion": clean_fusion, "strength_metrics": strengths,
        })
    direction_data, density_data = _aggregate_gate_inputs(event_metrics, identity["sequences"])
    calibration_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in event_metrics:
        factual = event["clean_evidence"]["factual"]
        calibration_events[event["sequence_name"]].append({
            "belief": factual["belief_map"],
            "gt_weights": factual["gt_weights"],
            "confidence": event["clean_evidence"]["factual"]["forward_best_score"],
            "box_iou": factual["box_iou"],
        })
    calibration = sequence_macro_calibration(calibration_events, bins=10)
    invalid_reason = None
    try:
        logo_results = _compute_logo(
            event_metrics, identity, direction_data, density_data
        )
    except (KeyError, TypeError, ValueError, Stage0RunError) as exc:
        invalid_reason = str(exc)
        logo_results = {
            "method": "leave_one_overlapping_group_out",
            "valid": False,
            "invalid_reason": invalid_reason,
            "attributes": {},
        }
    replay_reports = [
        row for sequence in identity["sequences"]
        for row in _read_jsonl(
            root / "intervene" / "sequences" / sequence / "replay_reports.jsonl"
        )
    ]
    replay_index = {}
    for report in replay_reports:
        validate_content_hash(report)
        event_id = report.get("event_id")
        if (not isinstance(event_id, str) or event_id in replay_index
                or report.get("artifact_type") not in {
                    "intervention_record_replay", "invalid_intervention_replay"
                }
                or report.get("check_count") != report.get("passed_count")
                or report.get("pass_fraction") != 1.0
                or report.get("passed") is not True):
            raise Stage0RunError("invalid or duplicate intervention replay report")
        replay_index[event_id] = report
    expected_event_ids = {item["online"]["event_id"] for item in joined}
    if set(replay_index) != expected_event_ids:
        raise Stage0RunError("intervention replay reports are not one-to-one with events")
    replay_fraction = 1.0 if replay_index else 0.0
    leakage_count = 0
    schema_mismatch_count = 0 if invalid_reason is None else 1
    for sequence in identity["sequences"]:
        for filename in ("trajectory.jsonl", "opportunities.jsonl"):
            for row in _read_jsonl(root / "online" / "sequences" / sequence / filename):
                try:
                    reject_online_labels(row)
                except ProtocolValidationError:
                    leakage_count += 1
    integrity = {
        "online_label_leakage_count": leakage_count,
        "non_finite_count": 0,
        "schema_mismatch_count": schema_mismatch_count,
        "replay_pass_fraction": replay_fraction,
    }
    aggregate = with_content_hash({
        "schema_version": SCHEMA_VERSION, "artifact_type": "stage0_evaluation",
        "scope": OFFICIAL_SCOPE, "aggregation": "sequence_macro",
        "status": "COMPLETE" if invalid_reason is None else "INVALID_RUN",
        "invalid_reason": invalid_reason,
        "identity_hash": identity["content_hash"],
        "intervene_phase_content_hash": parent["content_hash"],
        "event_count": len(event_metrics),
        "invalid_opportunity_count": len(invalid_opportunities),
        "invalid_opportunity_reasons": dict(sorted(
            (reason, sum(row["invalid_reason"] == reason for row in invalid_opportunities))
            for reason in {row["invalid_reason"] for row in invalid_opportunities}
        )),
        "sequence_count": len(identity["sequences"]),
        "direction_data": direction_data, "density_data": density_data,
        "calibration_descriptive_only": calibration,
        "logo_results": logo_results, "integrity": integrity,
        "raw_manifest_hash": canonical_json_hash({
            sequence: _read_json(_sequence_completion_path(root, "intervene", sequence))["file_sha256"]
            for sequence in identity["sequences"]}),
        "protocol_bundle_hash": identity["inputs"]["protocol_bundle_hash"],
    })
    evaluate_dir = root / "evaluate"
    evaluate_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(evaluate_dir / "event_metrics.jsonl", event_metrics)
    atomic_write_json(evaluate_dir / "aggregate.json", aggregate)
    phase = _phase_document("evaluate", identity["content_hash"], {
        "status": aggregate["status"], "aggregate_path": str(evaluate_dir / "aggregate.json"),
        "aggregate_sha256": sha256_file(evaluate_dir / "aggregate.json"),
        "event_metrics_sha256": sha256_file(evaluate_dir / "event_metrics.jsonl"),
        "event_count": len(event_metrics), "invalid_reason": invalid_reason,
    }, parent)
    _validate_child_parent(phase, parent)
    atomic_write_json(_phase_path(root, "evaluate"), phase)
    return phase


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    parent_path, parent = _load_parent(args.parent, "evaluate")
    root = parent_path.parent
    if Path(args.output_dir).resolve() != root:
        raise Stage0RunError("output-dir must equal the evaluate artifact directory")
    identity = _identity_from_root(root, parent["identity_hash"])
    aggregate_path = root / "evaluate" / "aggregate.json"
    event_metrics_path = root / "evaluate" / "event_metrics.jsonl"
    if sha256_file(aggregate_path) != parent.get("payload", {}).get("aggregate_sha256"):
        raise Stage0RunError("evaluate aggregate SHA does not match parent binding")
    if sha256_file(event_metrics_path) != parent.get("payload", {}).get("event_metrics_sha256"):
        raise Stage0RunError("event metrics SHA does not match parent binding")
    aggregate = _read_json(aggregate_path)
    validate_content_hash(aggregate)
    validate_official_gate_input(aggregate)
    if (aggregate.get("identity_hash") != identity["content_hash"]
            or aggregate.get("intervene_phase_content_hash") != parent.get("parent_content_hash")
            or aggregate.get("protocol_bundle_hash") != identity["inputs"]["protocol_bundle_hash"]):
        raise Stage0RunError("evaluate aggregate identity binding mismatch")
    replay_path = _regular_file(args.replay_report, "semantic replay report")
    replay = _read_json(replay_path)
    validate_content_hash(replay)
    if (replay.get("artifact_type") != "semantic_replay_report"
            or replay.get("formal_result") is not False
            or replay.get("fixture_kind") != "synthetic_semantic_only"):
        raise Stage0RunError("gate replay input is not the frozen synthetic semantic report")
    replay_inputs = replay.get("inputs", {})
    expected_protocol_hashes = identity["inputs"]["protocol_files"]
    if (replay_inputs.get("protocol", {}).get("sha256")
            != expected_protocol_hashes["stage0_protocol_v1.json"]
            or replay_inputs.get("registry", {}).get("sha256")
            != expected_protocol_hashes["intervention_registry_v1.json"]
            or replay_inputs.get("fixture", {}).get("sha256")
            != identity["inputs"]["sources"]["replay_fixture"]["sha256"]):
        raise Stage0RunError("semantic replay protocol or fixture binding mismatch")
    integrity = dict(aggregate["integrity"])
    integrity["replay_pass_fraction"] = min(
        float(integrity.get("replay_pass_fraction", 0.0)),
        float(replay.get("pass_fraction", 0.0)),
    )
    if aggregate.get("status") != "COMPLETE":
        integrity["schema_mismatch_count"] = max(1, int(integrity.get("schema_mismatch_count", 0)))
    bootstrap = identity["frozen_protocol"]["bootstrap"]
    decision = evaluate_stage0_gates(
        aggregate["direction_data"], aggregate["density_data"],
        aggregate["logo_results"], integrity, seed=bootstrap["seed"],
        samples=bootstrap["samples"], confidence=bootstrap["confidence_level"])
    decision.update({
        "artifact_type": "stage0_gate_decision", "scope": OFFICIAL_SCOPE,
        "identity_hash": identity["content_hash"],
        "evaluate_content_hash": aggregate["content_hash"],
        "semantic_replay_content_hash": replay["content_hash"],
        "stage1_unlock": None,
    })
    decision = with_content_hash(decision)
    gate_dir = root / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(gate_dir / "decision.json", decision)
    phase = _phase_document("gate", identity["content_hash"], {
        "status": decision["status"], "decision_path": str(gate_dir / "decision.json"),
        "decision_sha256": sha256_file(gate_dir / "decision.json"),
        "stage1_unlock_created": False,
    }, parent)
    _validate_child_parent(phase, parent)
    atomic_write_json(_phase_path(root, "gate"), phase)
    return phase


def _verify_chain(root: Path, identity: Mapping[str, Any], parent: Mapping[str, Any],
                  fixture: Path, protocol_dir: Path) -> tuple[dict[str, Any], bool]:
    checks = []
    try:
        if protocol_dir != Path(identity["inputs"]["protocol_dir"]).resolve():
            raise Stage0RunError("verify protocol directory differs from run identity")
        if fixture != DEFAULT_FIXTURE.resolve():
            raise Stage0RunError("verify fixture differs from the frozen synthetic fixture")
        _validate_identity_inputs(identity)
        identity_ok, identity_detail = True, None
    except (OSError, ValueError, ProtocolValidationError, Stage0RunError) as exc:
        identity_ok, identity_detail = False, str(exc)
    checks.append({"check": "run_identity_inputs", "passed": identity_ok,
                   "detail": identity_detail})
    replay = replay_fixture(fixture, protocol_dir)
    checks.append({"check": "synthetic_semantic_replay", "passed": replay["passed"],
                   "content_hash": replay["content_hash"]})
    previous = None
    for phase in PHASE_ORDER[:-1]:
        artifact = _read_json(_phase_path(root, phase))
        try:
            validate_content_hash(artifact)
            if artifact.get("identity_hash") != identity["content_hash"]:
                raise Stage0RunError("identity mismatch")
            if previous is not None:
                validate_phase_parent(artifact, previous)
            phase_ok, detail = True, None
        except (ProtocolValidationError, ValueError, Stage0RunError) as exc:
            phase_ok, detail = False, str(exc)
        checks.append({"check": f"phase:{phase}", "passed": phase_ok, "detail": detail})
        previous = artifact
    checks.append({"check": "gate_parent_is_current", "passed": previous is not None
                   and previous.get("content_hash") == parent.get("content_hash")})
    try:
        evaluate_phase = _read_json(_phase_path(root, "evaluate"))
        aggregate_path = root / "evaluate" / "aggregate.json"
        event_metrics_path = root / "evaluate" / "event_metrics.jsonl"
        aggregate_ok = (
            sha256_file(aggregate_path) == evaluate_phase["payload"]["aggregate_sha256"]
            and sha256_file(event_metrics_path) == evaluate_phase["payload"]["event_metrics_sha256"]
        )
        aggregate_detail = None
    except (OSError, KeyError, ValueError, ProtocolValidationError, Stage0RunError) as exc:
        aggregate_ok, aggregate_detail = False, str(exc)
    checks.append({"check": "evaluate_files_phase_bound", "passed": aggregate_ok,
                   "detail": aggregate_detail})
    try:
        decision_path = root / "gate" / "decision.json"
        decision = _read_json(decision_path)
        validate_content_hash(decision)
        decision_ok = (
            sha256_file(decision_path) == parent["payload"]["decision_sha256"]
            and decision.get("identity_hash") == identity["content_hash"]
            and decision.get("evaluate_content_hash")
            == _read_json(root / "evaluate" / "aggregate.json").get("content_hash")
            and decision.get("semantic_replay_content_hash") == replay["content_hash"]
            and decision.get("stage1_unlock") is None
        )
        decision_detail = None
    except (OSError, KeyError, ValueError, ProtocolValidationError, Stage0RunError) as exc:
        decision_ok, decision_detail = False, str(exc)
    checks.append({"check": "gate_decision_phase_bound", "passed": decision_ok,
                   "detail": decision_detail})
    phase_bindings = {
        phase: _read_json(_phase_path(root, phase))["payload"]["sequence_completion_hashes"]
        for phase in ("online", "intervene")
    }
    for sequence in identity["sequences"]:
        for phase, files in (
            ("online", ("trajectory.jsonl", "opportunities.jsonl")),
            ("intervene", ("raw_forwards.jsonl", "evaluator_labels.jsonl", "replay_reports.jsonl")),
        ):
            try:
                _authenticate_completion(
                    root, phase, sequence, identity["content_hash"], files,
                    phase_bindings[phase][sequence],
                )
                ok = True
                if phase == "online":
                    for row in _read_jsonl(root / phase / "sequences" / sequence / "trajectory.jsonl"):
                        reject_online_labels(row)
                    for row in _read_jsonl(root / phase / "sequences" / sequence / "opportunities.jsonl"):
                        reject_online_labels(row)
                detail = None
            except (OSError, ValueError, ProtocolValidationError, Stage0RunError) as exc:
                ok, detail = False, str(exc)
            checks.append({"check": f"raw:{phase}:{sequence}", "passed": ok, "detail": detail})
    for name, source in identity["inputs"]["sources"].items():
        path = Path(source["path"])
        observed = sha256_file(path) if path.is_file() else None
        checks.append({"check": f"source:{name}", "passed": observed == source["sha256"],
                       "expected": source["sha256"], "observed": observed})
    passed = bool(checks and all(item["passed"] for item in checks))
    report = with_content_hash({
        "schema_version": SCHEMA_VERSION, "artifact_type": "stage0_verification_report",
        "scope": OFFICIAL_SCOPE, "identity_hash": identity["content_hash"],
        "check_count": len(checks), "passed_count": sum(item["passed"] for item in checks),
        "pass_fraction": sum(item["passed"] for item in checks) / len(checks) if checks else 0.0,
        "passed": passed, "checks": checks,
    })
    return report, passed


def run_verify(args: argparse.Namespace) -> dict[str, Any]:
    parent_path, parent = _load_parent(args.parent, "gate")
    root = parent_path.parent
    if Path(args.output_dir).resolve() != root:
        raise Stage0RunError("output-dir must equal the gate artifact directory")
    identity = _identity_from_root(root, parent["identity_hash"])
    fixture = Path(args.fixture).resolve()
    protocol_dir = Path(args.protocol_dir).resolve()
    report, passed = _verify_chain(root, identity, parent, fixture, protocol_dir)
    verify_dir = root / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(verify_dir / "report.json", report)
    decision_path = root / "gate" / "decision.json"
    decision = _read_json(decision_path)
    decision_bound = False
    try:
        validate_content_hash(decision)
        decision_bound = (
            sha256_file(decision_path) == parent["payload"]["decision_sha256"]
            and decision.get("identity_hash") == identity["content_hash"]
        )
    except (OSError, KeyError, ValueError, ProtocolValidationError):
        decision_bound = False
    final_status = with_content_hash({
        "schema_version": SCHEMA_VERSION, "artifact_type": "stage0_final_status",
        "scope": OFFICIAL_SCOPE, "identity_hash": identity["content_hash"],
        "status": decision["status"] if passed and decision_bound else "INVALID_RUN",
        "verification_passed": bool(passed and decision_bound),
        "decision_immutable": decision_bound,
        "decision_path": str(decision_path),
        "decision_sha256": sha256_file(decision_path),
        "verification_content_hash": report["content_hash"],
    })
    atomic_write_json(verify_dir / "final_status.json", final_status)
    phase = _phase_document("verify", identity["content_hash"], {
        "status": final_status["status"],
        "verification_passed": final_status["verification_passed"],
        "report_path": str(verify_dir / "report.json"),
        "final_status_path": str(verify_dir / "final_status.json"),
        "existing_decision_rewritten": False,
    }, parent)
    _validate_child_parent(phase, parent)
    atomic_write_json(_phase_path(root, "verify"), phase)
    return phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one CBE Stage 0 diagnostic v1 phase")
    parser.add_argument("--phase", required=True, choices=PHASE_ORDER,
                        help="Strict single phase; there is deliberately no all mode")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent", help="Immediately preceding phase JSON artifact")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset", choices=("RGBT234", "LasHeR", "GTOT"))
    parser.add_argument("--split-manifest")
    parser.add_argument("--development-manifest")
    parser.add_argument("--internal-manifest")
    parser.add_argument("--confirm-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--split-manifest-lock")
    parser.add_argument("--split-manifest-lock-sha256")
    parser.add_argument("--attribute-manifest")
    parser.add_argument("--attribute-manifest-lock")
    parser.add_argument("--attribute-manifest-lock-sha256")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--model-config")
    parser.add_argument("--model-config-sha256")
    parser.add_argument("--protocol-dir", default=str(DEFAULT_PROTOCOL_DIR))
    parser.add_argument("--parameter-module", default="lib.test.parameter.vipt")
    parser.add_argument("--tracker-module", default="lib.test.tracker.vipt_stage0")
    parser.add_argument("--adapter-module", default="analysis.cbe.tracker_probe_v1")
    parser.add_argument("--replay-report")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    return parser


def _validate_phase_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    preflight = (
        "dataset_root", "dataset", "split_manifest", "development_manifest",
        "internal_manifest", "confirm_manifest", "val_manifest", "split_manifest_lock",
        "split_manifest_lock_sha256", "attribute_manifest", "attribute_manifest_lock",
        "attribute_manifest_lock_sha256", "checkpoint", "checkpoint_sha256",
        "model_config", "model_config_sha256", "protocol_dir",
    )
    if args.phase == "preflight":
        missing = [name for name in preflight if not getattr(args, name)]
        if missing:
            parser.error("preflight missing required arguments: " + ", ".join("--" + name.replace("_", "-") for name in missing))
        if args.parent:
            parser.error("preflight must not accept --parent")
    else:
        if not args.parent:
            parser.error(f"{args.phase} requires --parent")
        supplied_preflight = [name for name in preflight[:-1] if getattr(args, name)]
        if supplied_preflight:
            parser.error(f"{args.phase} rejects preflight-only arguments: " + ", ".join("--" + name.replace("_", "-") for name in supplied_preflight))
    if args.phase == "gate" and not args.replay_report:
        parser.error("gate requires --replay-report")
    if args.phase != "gate" and args.replay_report:
        parser.error("--replay-report is valid only for gate")
    if args.phase != "verify" and args.fixture != str(DEFAULT_FIXTURE):
        parser.error("--fixture is valid only for verify")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.phase not in ("preflight", "online") and args.workers != 1:
        parser.error("--workers is meaningful only for preflight/online and must remain 1")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_phase_args(parser, args)
    handlers = {
        "preflight": run_preflight,
        "online": run_online,
        "intervene": run_intervene,
        "evaluate": run_evaluate,
        "gate": run_gate,
        "verify": run_verify,
    }
    phase = handlers[args.phase](args)
    print(json.dumps(phase, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    if args.phase == "verify" and not phase["payload"]["verification_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
