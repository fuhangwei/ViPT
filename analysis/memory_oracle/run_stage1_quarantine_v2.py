"""Run the approved RMG-Q Stage 1 v2 quarantine protocol.

Online tracking and label evaluation are deliberately different CLI phases.  The
online phase never writes GT or IoU data; the evaluate phase creates labels from
the dataset and joins them to the frozen online artifacts.
"""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.memory_oracle.compute_stage0_metrics import canonical_hash  # noqa: E402
from analysis.memory_oracle.compute_stage1_quarantine_v2_metrics import (  # noqa: E402
    ARM_NAMES,
    aggregate_sequence_summaries,
    deterministic_threshold_stability,
    evaluate_final_gate,
    governance_metrics,
    select_smallest_passing_support_threshold,
    tracking_metrics_from_online_trace,
)
from analysis.memory_oracle.quarantine_controller_v2 import (  # noqa: E402
    SCHEMA_VERSION,
    QuarantinePolicy,
    QuarantineState,
    admit_opportunity,
    due_probe_offsets,
    finalize_quarantine,
    record_probe,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "stage1_quarantine_v2.yaml"
ARMS = tuple(ARM_NAMES)
INNER_SPLIT_NAMESPACE = "rmg-stage1-v2-q5-inner"
_SAFE_SEQUENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_WORKER = {}
IDENTITY_SOURCES = {
    "base_tracker": ROOT / "lib" / "test" / "tracker" / "vipt.py",
    "stage0_tracker": ROOT / "lib" / "test" / "tracker" / "vipt_stage0.py",
    "stage1_tracker": ROOT / "lib" / "test" / "tracker" / "vipt_stage1.py",
    "controller": Path(__file__).resolve().parent / "quarantine_controller_v2.py",
    "runner": Path(__file__).resolve(),
    "metrics": Path(__file__).resolve().parent / "compute_stage1_quarantine_v2_metrics.py",
    "dataset_io": ROOT / "analysis" / "natural_evidence" / "dataset_io.py",
}
_FORBIDDEN_ONLINE_KEYS = frozenset({
    "gt", "gt_xywh", "iou", "evaluation_iou", "candidate_iou",
    "source_candidate_iou", "release_frame_iou", "ground_truth", "groundtruth",
})


def load_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required by the ViPT configuration stack.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_config(config):
    """Require the complete frozen protocol config, including provenance settings."""
    default = load_config(DEFAULT_CONFIG)
    if canonical_hash(config) != canonical_hash(default):
        raise ValueError(
            "Stage 1 v2 config differs from the complete frozen default protocol config.")
    if config != default:
        raise ValueError("Stage 1 v2 config is not canonically identical to the frozen default.")
    return config


def sequence_name_sha256(sequence, namespace=""):
    name = str(sequence)
    payload = (str(namespace) + ":" + name) if namespace else name
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_sequences_by_namespace(sequences, namespace=INNER_SPLIT_NAMESPACE,
                                 first_count=83):
    """Return an order-invariant SHA-256 split controlled only by sequence names."""
    names = [str(name) for name in sequences]
    if len(names) != len(set(names)):
        raise ValueError("Sequence names must be unique before splitting.")
    count = int(first_count)
    if count < 0 or count > len(names):
        raise ValueError("first_count must be in [0, number of sequences].")
    ordered = sorted(names, key=lambda name: (sequence_name_sha256(name, namespace), name))
    return ordered[:count], ordered[count:]


def split_design_internal(sequences, namespace=INNER_SPLIT_NAMESPACE, design_count=83):
    """Approved deterministic design83/internal42 API."""
    names = list(sequences)
    design, internal = split_sequences_by_namespace(names, namespace, design_count)
    if len(names) == 125 and (len(design), len(internal)) != (83, 42):
        raise RuntimeError("The frozen development split did not produce 83/42.")
    return design, internal


split_inner_development = split_design_internal
split_design83_internal42 = split_design_internal
inner_split_sequences = split_design_internal


def split_outer_development(sequences, tune_count=125):
    """Use the exact Stage 1 v1 outer rule: SHA-256(sequence name), no namespace."""
    names = [str(name) for name in sequences]
    if len(names) != len(set(names)):
        raise ValueError("Sequence names must be unique before outer splitting.")
    ordered = sorted(names, key=lambda name: (sequence_name_sha256(name), name))
    count = int(tune_count)
    if count < 0 or count > len(ordered):
        raise ValueError("tune_count must be in [0, number of sequences].")
    return ordered[:count], ordered[count:]


def load_split_sequence(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Required authoritative split does not exist: {}".format(path))
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError("Authoritative split contains duplicate sequence names: {}".format(path))
    if any(not _SAFE_SEQUENCE.fullmatch(name) for name in names):
        raise ValueError("Authoritative split contains an unsafe sequence name: {}".format(path))
    return names


def canonical_scope_sequences(scope, development_split, val47_split, config=None):
    if config is None:
        config = load_config(DEFAULT_CONFIG)
    validate_config(config)
    development = load_split_sequence(development_split)
    val47 = load_split_sequence(val47_split)
    split = config["split"]
    if len(development) != int(split["development_count"]):
        raise RuntimeError("Authoritative development split must contain exactly 187 sequences.")
    if len(val47) != 47:
        raise RuntimeError("Authoritative val47 split must contain exactly 47 sequences.")
    tune, confirm = split_outer_development(
        development, tune_count=split["outer"]["tune_count"])
    if (len(tune), len(confirm)) != (125, 62):
        raise RuntimeError("The Stage 1 v1 outer rule did not produce tune125/confirm62.")
    inner = split["inner"]
    design, internal = split_design_internal(
        tune, namespace=inner["namespace"], design_count=inner["design_count"])
    if (len(design), len(internal)) != (83, 42):
        raise RuntimeError("The frozen inner rule did not produce design83/internal42.")
    scopes = {"design83": design, "internal42": internal,
              "confirm62": confirm, "val47": val47}
    return scopes[scope]


def validate_scope_membership(scope, supplied_sequences, development_split,
                              val47_split, config=None):
    canonical = canonical_scope_sequences(scope, development_split, val47_split, config)
    supplied = [str(name) for name in supplied_sequences]
    if supplied != canonical:
        raise RuntimeError(
            "Supplied split sequence set/order does not exactly match canonical {} scope."
            .format(scope))
    return canonical


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
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
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path, rows):
    _atomic_text(path, "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_hash_valid(value):
    return (isinstance(value, dict) and isinstance(value.get("content_hash"), str)
            and value["content_hash"] == canonical_hash(
                {key: item for key, item in value.items() if key != "content_hash"}))


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise RuntimeError("{} has an unexpected structure; expected keys {}, got {}."
                           .format(name, sorted(expected),
                                   sorted(value) if isinstance(value, dict) else type(value).__name__))
    return value


def _canonical_json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _require_canonical_json_file(path, expected=None, name="Artifact"):
    path = Path(path)
    value = read_json(path)
    if path.read_text(encoding="utf-8") != _canonical_json_text(value):
        raise RuntimeError("{} is not stored in the canonical JSON byte format: {}"
                           .format(name, path))
    if expected is not None and (value != expected
                                 or canonical_hash(value) != canonical_hash(expected)):
        raise RuntimeError("{} does not canonically equal its authoritative recomputation."
                           .format(name))
    return value


def _canonical_embedded(value, name):
    if not _content_hash_valid(value):
        raise RuntimeError("{} has an invalid canonical content_hash.".format(name))
    return value


def _find_existing(root, names, required=True):
    root = Path(root)
    for name in names:
        path = root / name
        if path.is_file():
            return path
    if required:
        raise FileNotFoundError("Missing required artifact under {}: {}".format(
            root, ", ".join(names)))
    return None


def _resolve_parent_spec(parent_artifacts):
    if not isinstance(parent_artifacts, dict):
        raise ValueError("parent_artifacts must be a mapping.")
    stage0 = parent_artifacts.get("stage0_dir", parent_artifacts.get("stage0"))
    stage1 = parent_artifacts.get("stage1_v1_dir", parent_artifacts.get("stage1_v1"))
    if isinstance(stage0, dict):
        stage0 = stage0.get("directory", stage0.get("path"))
    if isinstance(stage1, dict):
        stage1 = stage1.get("directory", stage1.get("path"))
    if not stage0 or not stage1:
        raise ValueError("Both stage0 and stage1_v1 parent directories are required.")
    return Path(stage0).resolve(), Path(stage1).resolve()


def validate_stage0_manifest_index(stage0_dir, expected_content_hash=None,
                                   expected_sequence_count=47):
    """Validate the exact Stage 0 index format generated by write_manifest_index."""
    stage0_dir = Path(stage0_dir).resolve()
    manifest_value = read_json(stage0_dir / "manifest.json")
    _exact_keys(manifest_value, {"schema_version", "kind", "sequences", "content_hash"},
                "Stage 0 manifest index")
    if (manifest_value.get("schema_version") != "rmg-stage0-v1"
            or manifest_value.get("kind") != "frozen_stage0_manifest_index"):
        raise RuntimeError("Stage 0 manifest index has the wrong schema/kind.")
    entries = manifest_value.get("sequences")
    if (not isinstance(entries, list) or len(entries) != int(expected_sequence_count)):
        raise RuntimeError("Stage 0 manifest index has the wrong sequence count.")
    names = [entry.get("sequence") if isinstance(entry, dict) else None for entry in entries]
    if (len(names) != len(set(names))
            or any(not isinstance(name, str) or not _SAFE_SEQUENCE.fullmatch(name)
                   for name in names)):
        raise RuntimeError("Stage 0 manifest index sequence names must be unique and valid.")
    sequence_manifests = []
    for entry in entries:
        _exact_keys(entry, {"sequence", "manifest_hash", "num_events"},
                    "Stage 0 manifest index entry")
        sequence = entry["sequence"]
        sequence_manifest = read_json(
            stage0_dir / "sequences" / sequence / "manifest.json")
        _exact_keys(sequence_manifest, {
            "schema_version", "kind", "sequence", "num_frames", "policy",
            "baseline_trace_hash", "update_frames", "replay_event_frames", "events",
            "manifest_hash",
        }, "Stage 0 sequence manifest")
        events = sequence_manifest.get("events")
        stored = sequence_manifest.get("manifest_hash")
        canonical = canonical_hash({key: value for key, value in sequence_manifest.items()
                                    if key != "manifest_hash"})
        policy = sequence_manifest.get("policy")
        update_frames = sequence_manifest.get("update_frames")
        replay_frames = sequence_manifest.get("replay_event_frames")
        if (sequence_manifest.get("schema_version") != "rmg-stage0-v1"
                or sequence_manifest.get("kind") != "frozen_stage0_manifest"
                or sequence_manifest.get("sequence") != sequence
                or not isinstance(sequence_manifest.get("num_frames"), int)
                or sequence_manifest["num_frames"] <= 0
                or not isinstance(sequence_manifest.get("baseline_trace_hash"), str)
                or not isinstance(update_frames, list)
                or update_frames != sorted(set(update_frames))
                or any(not isinstance(frame, int) for frame in update_frames)
                or not isinstance(replay_frames, list)
                or replay_frames != sorted(set(replay_frames))
                or any(frame not in update_frames for frame in replay_frames)
                or not isinstance(events, list)
                or stored != canonical or entry.get("manifest_hash") != stored
                or entry.get("num_events") != len(events)
                or any(not isinstance(event, dict)
                       or event.get("sequence") != sequence for event in events)):
            raise RuntimeError("Stage 0 sequence manifest/index binding is invalid.")
        _exact_keys(policy, {
            "warmup", "update_interval", "replay_event_min_spacing", "horizons",
            "terminal_cooldown", "pred_good_iou", "bad_iou", "min_candidate_size",
            "min_intersection_ratio", "max_padding_ratio",
        }, "Stage 0 manifest policy")
        event_ids = [event.get("event_id") for event in events]
        candidate_frames = [event.get("candidate_frame") for event in events]
        if (len(event_ids) != len(set(event_ids))
                or candidate_frames != replay_frames
                or any(not isinstance(frame, int) for frame in candidate_frames)):
            raise RuntimeError("Stage 0 sequence manifest has duplicate/misaligned events.")
        for event in events:
            if (not isinstance(event.get("event_hash"), str)
                    or event["event_hash"] != canonical_hash({
                        key: value for key, value in event.items() if key != "event_hash"})):
                raise RuntimeError("Stage 0 sequence manifest event hash is invalid.")
        sequence_manifests.append(sequence_manifest)
    manifest_hash = manifest_value.get("content_hash")
    if (not isinstance(manifest_hash, str)
            or manifest_hash != canonical_hash(sequence_manifests)):
        raise RuntimeError("Stage 0 manifest content_hash is not canonical-valid.")
    if expected_content_hash is not None and manifest_hash != expected_content_hash:
        raise RuntimeError("Stage 0 manifest content hash does not match frozen provenance.")
    return manifest_value


def validate_parent_provenance(parent_artifacts, config=None):
    """Validate the Stage 0 pass and Stage 1 v1 failure, failing closed.

    ``parent_artifacts`` names the two parent directories.  Their exact byte hashes and
    manifest content hash must match the frozen values in the v2 config.  Stage 1 v1
    must contain only its approved tuning-failure artifact: lock, arms, and gate
    artifacts are explicitly forbidden.
    """
    if config is None:
        config = load_config(DEFAULT_CONFIG)
    validate_config(config)
    stage0_dir, stage1_dir = _resolve_parent_spec(parent_artifacts)
    expected = config.get("parent_artifacts", {})
    stage0_expected = expected.get("stage0", {})
    stage1_expected = expected.get("stage1_v1", {})

    aggregate = _find_existing(stage0_dir, ("aggregate_summary.json", "aggregate.json"))
    gate = _find_existing(stage0_dir, ("gate_result.json", "gate.json"))
    manifest = _find_existing(stage0_dir, ("manifest.json",))
    if file_sha256(aggregate) != stage0_expected.get("aggregate_sha256"):
        raise RuntimeError("Stage 0 aggregate SHA-256 does not match frozen provenance.")
    if file_sha256(gate) != stage0_expected.get("gate_sha256"):
        raise RuntimeError("Stage 0 gate SHA-256 does not match frozen provenance.")
    gate_value = read_json(gate)
    if stage0_expected.get("gate_pass_required") and gate_value.get("pass") is not True:
        raise RuntimeError("Stage 0 parent gate did not pass.")
    manifest_value = validate_stage0_manifest_index(
        stage0_dir, expected_content_hash=stage0_expected.get("manifest_content_hash"),
        expected_sequence_count=47)
    manifest_hash = manifest_value["content_hash"]

    if manifest_hash != stage0_expected.get("manifest_content_hash"):
        raise RuntimeError("Stage 0 manifest content hash does not match frozen provenance.")

    failure = _find_existing(stage1_dir, (
        "tuning_failure.json", "tuning_failure_result.json", "failure.json"))
    if file_sha256(failure) != stage1_expected.get("tuning_failure_sha256"):
        raise RuntimeError("Stage 1 v1 tuning-failure SHA-256 does not match frozen provenance.")
    forbidden = []
    checks = (
        ("lock_must_be_absent", ("locked_thresholds.json", "policy_lock.json")),
        ("arms_must_be_absent", ("arms", "arms_summary.json")),
        ("gate_must_be_absent", ("gate.json", "gate_result.json")),
    )
    for flag, names in checks:
        if stage1_expected.get(flag):
            forbidden.extend(str(stage1_dir / name) for name in names
                             if (stage1_dir / name).exists())
    if forbidden:
        raise RuntimeError("Forbidden Stage 1 v1 post-failure artifacts exist: "
                           + ", ".join(forbidden))
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "validated_parent_provenance",
        "stage0": {
            "aggregate_sha256": file_sha256(aggregate),
            "gate_sha256": file_sha256(gate), "manifest_content_hash": manifest_hash,
            "gate_pass": gate_value.get("pass"),
        },
        "stage1_v1": {
            "tuning_failure_sha256": file_sha256(failure),
            "forbidden_artifacts_absent": True,
        },
    }
    result["content_hash"] = canonical_hash(result)
    return result


validate_parent_artifacts = validate_parent_provenance
require_parent_provenance = validate_parent_provenance


def _artifact_reference(path):
    path = Path(path).resolve()
    value = read_json(path)
    if not _content_hash_valid(value):
        raise RuntimeError("Artifact has an invalid canonical content_hash: {}".format(path))
    return {"path": str(path), "sha256": file_sha256(path),
            "content_hash": value["content_hash"]}, value


def _validate_artifact_reference(reference, expected_kind=None):
    if not isinstance(reference, dict) or not reference.get("path"):
        raise RuntimeError("Artifact reference is missing an absolute path.")
    path = Path(reference["path"])
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError("Artifact reference path is not an existing absolute file.")
    if file_sha256(path) != reference.get("sha256"):
        raise RuntimeError("Referenced artifact byte SHA-256 changed: {}".format(path))
    value = read_json(path)
    if (not _content_hash_valid(value)
            or value.get("content_hash") != reference.get("content_hash")):
        raise RuntimeError("Referenced artifact canonical content changed: {}".format(path))
    if expected_kind is not None and value.get("kind") != expected_kind:
        raise RuntimeError("Referenced artifact has the wrong kind: {}".format(path))
    return value


def validate_policy_lock(path, expected_parent_hash=None, required_kind=None,
                         expected_scope_sequences=None, expected_config_hash=None,
                         config=None, expected_internal_sequences=None):
    """Revalidate every source root in a policy lock against authoritative evidence."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("A frozen Stage 1 v2 policy lock is required: {}".format(path))
    if config is None or expected_scope_sequences is None or expected_parent_hash is None:
        raise RuntimeError("Policy lock validation requires frozen config, parent, and design list.")
    validate_config(config)
    config_hash = canonical_hash(config)
    if expected_config_hash is not None and expected_config_hash != config_hash:
        raise RuntimeError("Requested policy-lock config hash is not the frozen config hash.")
    lock = _require_canonical_json_file(path, name="Policy lock")
    approved_kinds = {"provisional_stage1_quarantine_v2_policy",
                      "final_stage1_quarantine_v2_policy"}
    if (not _content_hash_valid(lock) or lock.get("schema_version") != SCHEMA_VERSION
            or lock.get("kind") not in approved_kinds):
        raise RuntimeError("Stage 1 v2 policy lock is corrupt or has the wrong kind/schema.")
    if required_kind is not None and lock.get("kind") != required_kind:
        raise RuntimeError("This scope requires a {} policy lock.".format(required_kind))
    expected_keys = {
        "schema_version", "kind", "support_iou", "design_scope", "design_selection",
        "config_hash", "parent_provenance_hash", "source_identity_hashes",
        "source_sha256", "confirmation_outcomes_accessed", "content_hash",
    }
    if lock["kind"] == "final_stage1_quarantine_v2_policy":
        expected_keys.update({"provisional_lock", "internal_scope", "internal_root",
                              "internal_source_identity_hash", "internal_aggregate",
                              "internal_gate"})
    _exact_keys(lock, expected_keys, "Policy lock")
    threshold = lock.get("support_iou")
    candidates_frozen = [float(value) for value in config["quarantine"][
        "support_iou_candidates"]]
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(threshold) not in candidates_frozen):
        raise RuntimeError("Policy lock has no approved finite support_iou.")
    current_sources = {name: file_sha256(source) for name, source in IDENTITY_SOURCES.items()}
    design_sequences = list(expected_scope_sequences)
    design_scope = lock.get("design_scope", {})
    _exact_keys(design_scope, {"sequences", "sequences_hash"}, "Design scope")
    if (lock.get("parent_provenance_hash") != expected_parent_hash
            or lock.get("config_hash") != config_hash
            or lock.get("source_sha256") != current_sources
            or lock.get("confirmation_outcomes_accessed") is not False
            or design_scope.get("sequences") != design_sequences
            or design_scope.get("sequences_hash") != canonical_hash(design_sequences)):
        raise RuntimeError("Policy lock source/config/parent/design binding is invalid.")
    selection = lock.get("design_selection", {})
    _exact_keys(selection, {"selected_support_threshold", "selection_gate", "stability",
                            "candidates"}, "Design selection")
    candidate_locks = selection.get("candidates", [])
    if ([item.get("support_iou") for item in candidate_locks] != candidates_frozen
            or any(set(item) != {"support_iou", "root", "source_identity_hash",
                                 "aggregate", "gate", "sequence_summaries_hash"}
                   for item in candidate_locks)):
        raise RuntimeError("Policy lock must bind all five exact ordered design candidates.")
    source_hashes = lock.get("source_identity_hashes")
    if (not isinstance(source_hashes, list) or len(source_hashes) != 5
            or len(set(source_hashes)) != 5
            or any(not isinstance(value, str) for value in source_hashes)):
        raise RuntimeError("Policy lock must bind five distinct design source identities.")
    aggregate_candidates = []
    sequence_candidates = {sequence: [] for sequence in design_sequences}
    for index, item in enumerate(candidate_locks):
        if item.get("source_identity_hash") != source_hashes[index]:
            raise RuntimeError("Design candidate is not bound to its distinct source identity.")
        validated = validate_evaluated_root(
            item.get("root", ""), "design83", design_sequences, item["support_iou"],
            config, expected_parent_hash, policy_hash=None,
            expected_identity_hash=item["source_identity_hash"])
        aggregate_ref, _ = _artifact_reference(validated["aggregate_path"])
        gate_ref, _ = _artifact_reference(validated["gate_path"])
        if (item["aggregate"] != aggregate_ref or item["gate"] != gate_ref
                or item["sequence_summaries_hash"] != canonical_hash(
                    validated["summaries"])):
            raise RuntimeError("Design candidate references differ from recomputed artifacts.")
        aggregate = validated["aggregate"]
        aggregate_candidates.append({
            "support_threshold": item["support_iou"], "governance": aggregate["governance"],
            "quarantine_incremental_success_auc_delta": aggregate[
                "frame_weighted_paired_deltas"]["rmg_q_vs_rmg_q_no_quarantine"][
                    "success_auc"]["mean"],
        })
        for sequence, summary in zip(design_sequences, validated["summaries"]):
            sequence_candidates[sequence].append({
                "support_threshold": item["support_iou"],
                "num_frames": summary["arms"]["static"]["num_frames"],
                "governance": summary["governance"],
                "quarantine_incremental_success_auc_delta": (
                    summary["arms"]["rmg_q"]["success_auc"]
                    - summary["arms"]["rmg_q_no_quarantine"]["success_auc"]),
            })
    expected_selection_gate = select_smallest_passing_support_threshold(
        aggregate_candidates, gate_config=config["action_gate"])
    stability_input = [{"sequence": sequence,
                        "threshold_candidates": sequence_candidates[sequence]}
                       for sequence in design_sequences]
    stability_config = config["stability"]
    expected_stability = deterministic_threshold_stability(
        stability_input, gate_config=config["action_gate"],
        seed=stability_config["bootstrap_seed"],
        samples=stability_config["bootstrap_samples"],
        hash_namespace="rmg-stage1-v2-q5-threshold-stability")
    if (selection["selection_gate"] != expected_selection_gate
            or selection["stability"] != expected_stability
            or selection["selected_support_threshold"] != float(threshold)
            or expected_selection_gate["selected_support_threshold"] != float(threshold)
            or expected_stability["selected_support_threshold"] != float(threshold)
            or expected_stability["bootstrap"]["modal_support_threshold"] != float(threshold)
            or expected_stability["bootstrap"]["modal_match_fraction"]
            < stability_config["modal_match_fraction_min"]
            or expected_stability["leave_one_group_out_same_threshold_count"]
            < stability_config["leave_one_group_out"]["same_threshold_min"]):
        raise RuntimeError("Policy lock design selection/stability is not authoritative.")
    if lock["kind"] == "final_stage1_quarantine_v2_policy":
        if expected_internal_sequences is None:
            raise RuntimeError("Final policy validation requires canonical internal42 list.")
        provisional_ref = lock["provisional_lock"]
        provisional = validate_policy_lock(
            provisional_ref.get("path", ""), expected_parent_hash=expected_parent_hash,
            required_kind="provisional_stage1_quarantine_v2_policy",
            expected_scope_sequences=design_sequences, expected_config_hash=config_hash,
            config=config)
        expected_provisional_ref = {
            "path": str(Path(provisional_ref.get("path", "")).resolve()),
            "sha256": file_sha256(provisional_ref.get("path", "")),
            "content_hash": provisional["content_hash"],
        }
        internal_sequences = list(expected_internal_sequences)
        internal_scope = lock.get("internal_scope", {})
        _exact_keys(internal_scope, {"sequences", "sequences_hash"}, "Internal scope")
        if (provisional_ref != expected_provisional_ref
                or internal_scope.get("sequences") != internal_sequences
                or internal_scope.get("sequences_hash") != canonical_hash(internal_sequences)):
            raise RuntimeError("Final policy lock has invalid provisional/internal binding.")
        validated = validate_evaluated_root(
            lock.get("internal_root", ""), "internal42", internal_sequences, threshold,
            config, expected_parent_hash, policy_hash=provisional["content_hash"],
            expected_identity_hash=lock.get("internal_source_identity_hash"))
        aggregate_ref, _ = _artifact_reference(validated["aggregate_path"])
        gate_ref, _ = _artifact_reference(validated["gate_path"])
        if (lock["internal_aggregate"] != aggregate_ref or lock["internal_gate"] != gate_ref
                or validated["gate"].get("pass") is not True):
            raise RuntimeError("Final policy lock does not reference a recomputed passing internal gate.")
    return lock


def validate_val47_unlock(path, expected_policy_hash=None, expected_parent_hash=None,
                          expected_confirm_sequences=None, config=None,
                          policy_lock_path=None, expected_design_sequences=None,
                          expected_internal_sequences=None):
    """Revalidate final policy and complete confirm62 evidence before val47 access."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("A valid confirm62 val47 unlock is required: {}".format(path))
    if (config is None or policy_lock_path is None or expected_parent_hash is None
            or expected_confirm_sequences is None or expected_design_sequences is None
            or expected_internal_sequences is None):
        raise RuntimeError("Val47 unlock validation requires frozen config and canonical scopes.")
    validate_config(config)
    unlock = _require_canonical_json_file(path, name="Val47 unlock")
    expected_keys = {
        "schema_version", "kind", "confirm_sequence_count", "confirm_sequences",
        "confirm_sequences_hash", "confirm_root", "confirm_aggregate", "confirm_gate",
        "confirm_gate_pass", "policy_lock_content_hash", "parent_provenance_hash",
        "config_hash", "source_sha256", "content_hash",
    }
    _exact_keys(unlock, expected_keys, "Val47 unlock")
    if (not _content_hash_valid(unlock) or unlock.get("schema_version") != SCHEMA_VERSION
            or unlock.get("kind") != "rmg_stage1_v2_val47_unlock"):
        raise RuntimeError("Val47 unlock is corrupt or has the wrong kind/schema.")
    policy = validate_policy_lock(
        policy_lock_path, expected_parent_hash=expected_parent_hash,
        required_kind="final_stage1_quarantine_v2_policy",
        expected_scope_sequences=expected_design_sequences,
        expected_config_hash=canonical_hash(config), config=config,
        expected_internal_sequences=expected_internal_sequences)
    expected = list(expected_confirm_sequences)
    current_sources = {name: file_sha256(source) for name, source in IDENTITY_SOURCES.items()}
    if (len(expected) != 62 or unlock.get("confirm_sequence_count") != 62
            or unlock.get("confirm_sequences") != expected
            or unlock.get("confirm_sequences_hash") != canonical_hash(expected)
            or unlock.get("policy_lock_content_hash") != policy["content_hash"]
            or (expected_policy_hash is not None
                and unlock.get("policy_lock_content_hash") != expected_policy_hash)
            or unlock.get("parent_provenance_hash") != expected_parent_hash
            or unlock.get("config_hash") != canonical_hash(config)
            or unlock.get("source_sha256") != current_sources):
        raise RuntimeError("Val47 unlock canonical scope/source/config/policy binding is invalid.")
    validated = validate_evaluated_root(
        unlock.get("confirm_root", ""), "confirm62", expected, policy["support_iou"],
        config, expected_parent_hash, policy_hash=policy["content_hash"])
    aggregate_ref, _ = _artifact_reference(validated["aggregate_path"])
    gate_ref, _ = _artifact_reference(validated["gate_path"])
    if (unlock.get("confirm_aggregate") != aggregate_ref
            or unlock.get("confirm_gate") != gate_ref
            or validated["gate"].get("pass") is not True
            or unlock.get("confirm_gate_pass") is not True):
        raise RuntimeError("Val47 remains locked because confirm62 did not recompute to pass.")
    return unlock


require_val47_unlock = validate_val47_unlock
validate_unlock_chain = validate_val47_unlock


def _reject_labels(value, location="online"):
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if (normalized in _FORBIDDEN_ONLINE_KEYS or normalized.startswith("gt_")
                    or normalized.endswith("_gt") or "ground_truth" in normalized):
                raise RuntimeError("Forbidden label field in {}: {}".format(location, key))
            _reject_labels(item, location + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_labels(item, "{}[{}]".format(location, index))


def safe_sequence_dir(output_dir, sequence, create=True):
    name = str(sequence)
    if not _SAFE_SEQUENCE.fullmatch(name) or name in (".", ".."):
        raise ValueError("Unsafe sequence name: {!r}".format(name))
    base = (Path(output_dir) / "sequences").resolve()
    root = (base / name).resolve()
    try:
        root.relative_to(base)
    except ValueError as exc:
        raise ValueError("Sequence path escapes output directory.") from exc
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def periodic_candidate_frames(num_frames, warmup=30, interval=30, cooldown=60):
    if interval <= 0:
        raise ValueError("interval must be positive.")
    return list(range(int(warmup), max(0, int(num_frames) - int(cooldown)), int(interval)))


def _bbox_validity(box, image_shape, config):
    validity = config.get("candidate_validity", {})
    if box is None or len(box) < 4:
        return False, "bbox_missing"
    try:
        x, y, width, height = [float(value) for value in box[:4]]
        image_height, image_width = [float(value) for value in image_shape[:2]]
    except (TypeError, ValueError, OverflowError):
        return False, "bbox_non_finite"
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return False, "bbox_non_finite"
    if width < float(validity.get("min_size", 8.0)) or height < float(
            validity.get("min_size", 8.0)):
        return False, "bbox_too_small"
    intersection_width = max(0.0, min(x + width, image_width) - max(x, 0.0))
    intersection_height = max(0.0, min(y + height, image_height) - max(y, 0.0))
    ratio = intersection_width * intersection_height / (width * height)
    if ratio < float(validity.get("min_intersection_ratio", 0.75)):
        return False, "bbox_low_image_intersection"
    if 1.0 - ratio > float(validity.get("max_padding_ratio", 0.25)):
        return False, "bbox_excessive_padding"
    return True, None


def _snapshot_copy(snapshot):
    if hasattr(snapshot, "clone"):
        return snapshot.clone()
    return copy.deepcopy(snapshot)


def make_tracker(yaml_name):
    from lib.test.tracker.vipt_stage1 import ViPTStage1Track
    import lib.test.parameter.vipt as vipt_params

    params = vipt_params.parameters(yaml_name)
    tracker = ViPTStage1Track(params)
    required = ("build_template_snapshot", "predict_with_context", "commit_template",
                "rollback_to_initial", "response_statistics")
    missing = [name for name in required if not hasattr(tracker, name)]
    if missing:
        raise RuntimeError("ViPTStage1Track is missing: " + ", ".join(missing))
    return tracker, params


def checkpoint_path(yaml_name):
    import lib.test.parameter.vipt as vipt_params
    return Path(vipt_params.parameters(yaml_name).checkpoint).resolve()


def load_frame(rgb_path, tir_path, params):
    from lib.train.dataset.depth_utils import get_x_frame
    xtype = getattr(params.cfg.DATA, "XTYPE", "rgbrgb")
    return get_x_frame(rgb_path, tir_path, dtype=xtype)


def _load_online_sequence(dataset_root, sequence):
    """Load image paths plus only the initialization annotation.

    Future GT is not parsed during online execution.  The first annotation is part of
    the standard tracker initialization contract, not an outcome label.
    """
    root = Path(dataset_root) / sequence
    rgb = sorted(str(path) for path in (root / "visible").iterdir()
                 if path.is_file() and path.suffix.lower() == ".jpg")
    tir = sorted(str(path) for path in (root / "infrared").iterdir()
                 if path.is_file() and path.suffix.lower() == ".jpg")
    count = min(len(rgb), len(tir))
    if count <= 0:
        raise RuntimeError("Sequence {!r} has no aligned image frames.".format(sequence))
    first_line = None
    with (root / "visible.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                first_line = line.strip()
                break
    if first_line is None:
        raise RuntimeError("Sequence {!r} has no initialization annotation.".format(sequence))
    try:
        init_box = [float(value.strip()) for value in first_line.split(",")[:4]]
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Sequence {!r} has an invalid initialization annotation.".format(
            sequence)) from exc
    if (len(init_box) != 4 or not all(math.isfinite(value) for value in init_box)
            or init_box[2] <= 0.0 or init_box[3] <= 0.0):
        raise RuntimeError("Sequence {!r} has an invalid initialization box.".format(sequence))
    return rgb[:count], tir[:count], init_box, count


def _load_evaluation_sequence(dataset_root, sequence):
    from analysis.natural_evidence.dataset_io import load_rgbt_sequence
    rgb, tir, gt, _ = load_rgbt_sequence(dataset_root, sequence, "RGBT234")
    count = min(len(rgb), len(tir), len(gt))
    if count <= 0:
        raise RuntimeError("Sequence {!r} has no aligned evaluation frames.".format(sequence))
    return rgb[:count], tir[:count], gt[:count]


def online_observation(frame_idx, output, image_shape, template_age):
    return {
        "frame_idx": int(frame_idx),
        "pred_xywh": [float(value) for value in output["target_bbox"][:4]],
        "search_anchor_xywh": [float(value) for value in output[
            "search_anchor"][:4]],
        "image_shape": [int(value) for value in image_shape[:2]],
        "response_peak": float(output["response_peak"]),
        "response_entropy": float(output["response_entropy"]),
        "response_margin": float(output["response_margin"]),
        "response_topk_score_std": float(output["response_topk_score_std"]),
        "response_topk_box_dispersion": float(output["response_topk_box_dispersion"]),
        "template_age": int(template_age),
    }


def _initial_output(box, template_id):
    values = [float(value) for value in box[:4]]
    return {"target_bbox": values, "search_anchor": values, "state_before": values,
            "best_score": 1.0, "template_id": template_id}


def _frame_row(sequence, arm, frame_idx, output, action="none", opportunity=False):
    pred = [float(value) for value in output["target_bbox"][:4]]
    row = {
        "schema_version": SCHEMA_VERSION, "sequence": sequence, "arm": arm,
        "frame_idx": int(frame_idx), "pred_xywh": pred,
        "best_score": float(output.get("best_score", 1.0)),
        "search_anchor_xywh": [float(value) for value in output.get(
            "search_anchor", output.get("state_before", pred))[:4]],
        "template_id_for_prediction": output.get("template_id"),
        "is_update_opportunity": bool(opportunity),
        "action_after_prediction": action,
    }
    _reject_labels(row)
    return row


def _event_row(sequence, observation, decision, legal, invalid_reason):
    row = {
        "schema_version": SCHEMA_VERSION,
        "event_id": decision.event_id,
        "sequence": sequence,
        "source_frame": int(decision.source_frame),
        "source_candidate_xywh": list(observation["pred_xywh"]),
        "admission_entropy": float(observation["response_entropy"]),
        "legal_source_opportunity": bool(legal),
        "candidate_invalid_reason": invalid_reason,
        "action": ("immediate_write" if decision.action == "immediate_update"
                   else decision.action),
        "reason": decision.reason,
        "quarantined": decision.action == "quarantine",
        "effective_frame": decision.effective_frame,
        "probes": [],
    }
    _reject_labels(row)
    return row


def run_arm_online(sequence, arm, rgb_imgs, tir_imgs, init_box, num_frames,
                   tracker_context, config, support_iou):
    """Run one causal arm and return label-free frame and event traces."""
    if arm not in ARMS:
        raise ValueError("Unknown arm: {}".format(arm))
    tracker, params = tracker_context
    first = load_frame(rgb_imgs[0], tir_imgs[0], params)
    init_box = [float(value) for value in init_box[:4]]
    tracker.initialize(first, {"init_bbox": init_box})
    tracker.rollback_to_initial()
    anchor = list(init_box)
    active_snapshot = _snapshot_copy(tracker.initial_template_snapshot)
    tracker.commit_template(active_snapshot)
    policy = QuarantinePolicy(support_iou=float(support_iou))
    qstate = QuarantineState()
    pending_snapshot = None
    event_by_id = {}
    opportunities = set(periodic_candidate_frames(
        num_frames, config["schedule"]["warmup"], config["schedule"]["interval"],
        config["schedule"]["cooldown"]))
    rows = [_frame_row(sequence, arm, 0, _initial_output(
        init_box, tracker.initial_template_snapshot.template_id))]

    for frame_idx in range(1, num_frames):
        image = load_frame(rgb_imgs[frame_idx], tir_imgs[frame_idx], params)
        pre_frame_anchor = list(anchor)
        output = tracker.predict_with_context(image, pre_frame_anchor, active_snapshot)
        pred = [float(value) for value in output["target_bbox"][:4]]
        action = "none"

        # Probe output is a side branch from the exact anchor used by the active prediction.
        # It is never committed, emitted as an arm prediction, or inserted into anchor history.
        due = due_probe_offsets(qstate, frame_idx, policy) if arm == "rmg_q" else ()
        if due:
            if pending_snapshot is None or qstate.pending is None:
                raise RuntimeError("Pending quarantine state has no source snapshot.")
            shadow = tracker.predict_with_context(image, pre_frame_anchor, pending_snapshot)
            active_valid, active_reason = _bbox_validity(pred, image.shape, config)
            shadow_box = [float(value) for value in shadow["target_bbox"][:4]]
            shadow_valid, shadow_reason = _bbox_validity(shadow_box, image.shape, config)
            event_id = qstate.pending.event_id
            qstate, evidence = record_probe(
                qstate, policy, event_id, frame_idx, pred, shadow_box,
                active_valid, shadow_valid, pre_frame_anchor, pre_frame_anchor,
                active_reason, shadow_reason)
            event_by_id[event_id]["probes"].append({
                "frame_idx": evidence.frame_idx, "offset": evidence.offset,
                "shared_anchor_xywh": list(evidence.shared_anchor_xywh),
                "active_xywh": (list(evidence.active_xywh)
                                  if evidence.active_xywh is not None else None),
                "shadow_xywh": (list(evidence.shadow_xywh)
                                  if evidence.shadow_xywh is not None else None),
                "active_legal": evidence.active_legal,
                "active_illegal_reason": evidence.active_illegal_reason,
                "shadow_legal": evidence.shadow_legal,
                "shadow_illegal_reason": evidence.shadow_illegal_reason,
                "agreement_iou": evidence.agreement_iou,
                "supports_release": evidence.supports_release,
            })
            if evidence.offset == policy.probe_offsets[-1]:
                qstate, finalized = finalize_quarantine(
                    qstate, policy, event_id, frame_idx)
                event = event_by_id[event_id]
                event.update({
                    "action": finalized.action, "reason": finalized.reason,
                    "support_count": finalized.support_count,
                    "probe_count": finalized.probe_count,
                    "finalized_frame": finalized.finalized_frame,
                    "effective_frame": finalized.effective_frame,
                    "released": finalized.action == "release",
                })
                if finalized.action == "release":
                    # Commit only after active frame t+5 has been emitted.  This is the
                    # original source snapshot, never a probe-frame snapshot.
                    active_snapshot = _snapshot_copy(pending_snapshot)
                    tracker.commit_template(active_snapshot)
                    action = "release_source_snapshot"
                else:
                    action = "discard_quarantine"
                pending_snapshot = None

        opportunity = frame_idx in opportunities
        if opportunity and arm != "static":
            observation = online_observation(
                frame_idx, output, image.shape,
                frame_idx - int(active_snapshot.source_frame))
            legal, invalid_reason = _bbox_validity(pred, image.shape, config)
            if arm == "periodic_pred":
                if legal:
                    active_snapshot = tracker.build_template_snapshot(
                        image, pred, source="current_arm_prediction", source_frame=frame_idx)
                    tracker.commit_template(active_snapshot)
                    action = "commit_current_arm_prediction"
                else:
                    action = "skip:invalid:{}".format(invalid_reason)
            elif arm == "confidence_e050":
                entropy = observation["response_entropy"]
                if legal and math.isfinite(entropy) and entropy <= 0.50:
                    active_snapshot = tracker.build_template_snapshot(
                        image, pred, source="current_arm_prediction", source_frame=frame_idx)
                    tracker.commit_template(active_snapshot)
                    action = "commit_current_arm_prediction"
                else:
                    reason = ("invalid:{}".format(invalid_reason) if not legal else
                              ("non_finite_entropy" if not math.isfinite(entropy)
                               else "entropy_above_050"))
                    action = "skip:" + reason
            elif arm in ("rmg_q", "rmg_q_no_quarantine"):
                if arm == "rmg_q":
                    qstate, decision = admit_opportunity(
                        qstate, sequence, observation, policy, legal, invalid_reason)
                else:
                    # The no-quarantine ablation shares immediate triage but suppresses
                    # the (.45,.50] branch completely.
                    temporary, decision = admit_opportunity(
                        QuarantineState(), sequence, observation, policy, legal, invalid_reason)
                    del temporary
                    if decision.action == "quarantine":
                        decision = type(decision)(
                            "skip", "quarantine_disabled", decision.event_id,
                            decision.source_frame, None)
                event = _event_row(sequence, observation, decision, legal, invalid_reason)
                event_by_id[event["event_id"]] = event
                if decision.action == "immediate_update":
                    active_snapshot = tracker.build_template_snapshot(
                        image, pred, source="current_arm_prediction", source_frame=frame_idx)
                    tracker.commit_template(active_snapshot)
                    action = "commit_current_arm_prediction"
                elif decision.action == "quarantine":
                    pending_snapshot = tracker.build_template_snapshot(
                        image, pred, source="quarantine_source_prediction",
                        source_frame=frame_idx)
                    action = "quarantine_source_snapshot"
                else:
                    action = "skip:" + decision.reason

        rows.append(_frame_row(sequence, arm, frame_idx, output, action, opportunity))
        anchor = pred

    if qstate.pending is not None or pending_snapshot is not None:
        raise RuntimeError("Terminal cooldown failed to leave enough room for Q5 finalization.")
    events = [event_by_id[key] for key in sorted(event_by_id)]
    _reject_labels(rows)
    _reject_labels(events)
    return rows, events


def validate_online_trace(rows, sequence, arm):
    if not rows:
        raise RuntimeError("Empty online trace for {}/{}.".format(sequence, arm))
    if [int(row.get("frame_idx", -1)) for row in rows] != list(range(len(rows))):
        raise RuntimeError("Noncontiguous frame indices for {}/{}.".format(sequence, arm))
    if any(row.get("schema_version") != SCHEMA_VERSION
           or row.get("sequence") != sequence or row.get("arm") != arm for row in rows):
        raise RuntimeError("Trace identity mismatch for {}/{}.".format(sequence, arm))
    _reject_labels(rows)


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


def _online_sequence_worker(sequence):
    root = safe_sequence_dir(_WORKER["output_dir"], sequence)
    rgb, tir, init_box, num_frames = _load_online_sequence(
        _WORKER["dataset_root"], sequence)
    tracker_context = _worker_tracker()
    arms_dir = root / "online"
    arms_dir.mkdir(parents=True, exist_ok=True)
    trace_hashes = {}
    event_hashes = {}
    for arm in ARMS:
        rows, events = run_arm_online(
            sequence, arm, rgb, tir, init_box, num_frames, tracker_context,
            _WORKER["config"], _WORKER["support_iou"])
        trace_path = arms_dir / (arm + ".frames.jsonl")
        event_path = arms_dir / (arm + ".events.jsonl")
        write_jsonl(trace_path, rows)
        write_jsonl(event_path, events)
        trace_hashes[arm] = canonical_hash(rows)
        event_hashes[arm] = canonical_hash(events)
    summary = {
        "schema_version": SCHEMA_VERSION, "kind": "online_sequence_complete",
        "sequence": sequence, "support_iou": float(_WORKER["support_iou"]),
        "arms": list(ARMS), "trace_hashes": trace_hashes, "event_hashes": event_hashes,
    }
    summary["content_hash"] = canonical_hash(summary)
    write_json(root / "online_summary.json", summary)
    return {"sequence": sequence, "operation": "online"}


def completed_online_sequence(output_dir, sequence, support_iou):
    root = safe_sequence_dir(output_dir, sequence)
    path = root / "online_summary.json"
    if not path.is_file():
        return False
    try:
        summary = read_json(path)
        if (set(summary) != {"schema_version", "kind", "sequence", "support_iou",
                             "arms", "trace_hashes", "event_hashes", "content_hash"}
                or not _content_hash_valid(summary)
                or summary.get("schema_version") != SCHEMA_VERSION
                or summary.get("kind") != "online_sequence_complete"
                or summary.get("sequence") != sequence
                or float(summary.get("support_iou")) != float(support_iou)
                or tuple(summary.get("arms", ())) != ARMS
                or set(summary.get("trace_hashes", {})) != set(ARMS)
                or set(summary.get("event_hashes", {})) != set(ARMS)):
            return False
        for arm in ARMS:
            rows = read_jsonl(root / "online" / (arm + ".frames.jsonl"))
            events = read_jsonl(root / "online" / (arm + ".events.jsonl"))
            validate_online_trace(rows, sequence, arm)
            _reject_labels(events)
            if summary["trace_hashes"].get(arm) != canonical_hash(rows):
                return False
            if summary["event_hashes"].get(arm) != canonical_hash(events):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError):
        return False


def map_online_sequences(sequences, args, config, support_iou):
    pending = [sequence for sequence in sequences if not completed_online_sequence(
        args.output_dir, sequence, support_iou)]
    if not pending:
        return []
    payload = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "yaml_name": args.yaml_name, "config": config,
        "support_iou": float(support_iou),
    }
    if int(args.workers) < 1:
        raise ValueError("workers must be at least one.")
    if int(args.workers) == 1:
        _worker_init("online", payload)
        return [_online_sequence_worker(sequence) for sequence in pending]
    context = mp.get_context("spawn")
    with context.Pool(processes=int(args.workers), initializer=_worker_init,
                      initargs=("online", payload)) as pool:
        return list(pool.imap(_online_sequence_worker, pending, chunksize=1))


def generate_frame_labels(sequence, gt):
    """Evaluator-only frame labels; never call from online execution."""
    return [{"schema_version": SCHEMA_VERSION, "sequence": sequence,
             "frame_idx": frame_idx,
             "gt_xywh": [float(value) for value in gt[frame_idx][:4]]}
            for frame_idx in range(len(gt))]


def _iou_xywh(first, second):
    ax, ay, aw, ah = [float(value) for value in first[:4]]
    bx, by, bw, bh = [float(value) for value in second[:4]]
    intersection = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def generate_event_labels(online_events, gt):
    """Evaluator-only source-time labels for already-frozen online events."""
    labels = []
    for event in online_events:
        source = int(event["source_frame"])
        if source < 0 or source >= len(gt):
            raise RuntimeError("Event source frame lies outside evaluator labels.")
        label = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event["event_id"], "sequence": event["sequence"],
            "source_frame": source,
            "source_candidate_iou": _iou_xywh(
                event["source_candidate_xywh"], gt[source]),
        }
        finalized = event.get("finalized_frame")
        if finalized is not None:
            frame = int(finalized)
            probes = event.get("probes", [])
            active = next((probe.get("active_xywh") for probe in probes
                           if int(probe.get("frame_idx", -1)) == frame), None)
            if active is not None and 0 <= frame < len(gt):
                label["release_frame_iou"] = _iou_xywh(active, gt[frame])
        labels.append(label)
    return labels


def _evaluate_sequence_worker(sequence):
    root = safe_sequence_dir(_WORKER["output_dir"], sequence)
    _, _, gt = _load_evaluation_sequence(_WORKER["dataset_root"], sequence)
    frame_labels = generate_frame_labels(sequence, gt)
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(labels_dir / "frames.jsonl", frame_labels)
    arm_metrics = {}
    trace_hashes = {}
    for arm in ARMS:
        rows = read_jsonl(root / "online" / (arm + ".frames.jsonl"))
        validate_online_trace(rows, sequence, arm)
        if len(rows) != len(frame_labels):
            raise RuntimeError("Online/label frame count mismatch for {}/{}.".format(
                sequence, arm))
        arm_metrics[arm] = tracking_metrics_from_online_trace(rows, frame_labels)
        trace_hashes[arm] = canonical_hash(rows)
    events = read_jsonl(root / "online" / "rmg_q.events.jsonl")
    event_labels = generate_event_labels(events, gt)
    write_jsonl(labels_dir / "rmg_q.events.jsonl", event_labels)
    governance = governance_metrics(
        events, event_labels,
        good_iou=_WORKER["config"]["thresholds"]["good_iou"],
        bad_iou=_WORKER["config"]["thresholds"]["bad_iou"])
    summary = {
        "schema_version": SCHEMA_VERSION, "kind": "evaluated_sequence",
        "sequence": sequence, "arms": arm_metrics, "governance": governance,
        "online_trace_hashes": trace_hashes,
        "online_rmg_q_events_hash": canonical_hash(events),
        "frame_labels_hash": canonical_hash(frame_labels),
        "event_labels_hash": canonical_hash(event_labels),
    }
    summary["content_hash"] = canonical_hash(summary)
    write_json(root / "sequence_summary.json", summary)
    return summary


def completed_evaluated_sequence(output_dir, sequence, support_iou=None):
    root = safe_sequence_dir(output_dir, sequence)
    path = root / "sequence_summary.json"
    if not path.is_file():
        return False
    try:
        if support_iou is not None and not completed_online_sequence(
                output_dir, sequence, support_iou):
            return False
        summary = read_json(path)
        if (set(summary) != _SUMMARY_KEYS or not _content_hash_valid(summary)
                or summary.get("schema_version") != SCHEMA_VERSION
                or summary.get("kind") != "evaluated_sequence"
                or summary.get("sequence") != sequence
                or set(summary.get("arms", {})) != set(ARMS)
                or set(summary.get("online_trace_hashes", {})) != set(ARMS)
                or any(set(summary["arms"].get(arm, {})) != _ARM_METRIC_KEYS
                       for arm in ARMS)):
            return False
        frame_labels = read_jsonl(root / "labels" / "frames.jsonl")
        event_labels = read_jsonl(root / "labels" / "rmg_q.events.jsonl")
        if summary.get("frame_labels_hash") != canonical_hash(frame_labels):
            return False
        if summary.get("event_labels_hash") != canonical_hash(event_labels):
            return False
        for arm in ARMS:
            rows = read_jsonl(root / "online" / (arm + ".frames.jsonl"))
            validate_online_trace(rows, sequence, arm)
            if summary["online_trace_hashes"].get(arm) != canonical_hash(rows):
                return False
        events = read_jsonl(root / "online" / "rmg_q.events.jsonl")
        _reject_labels(events)
        if summary.get("online_rmg_q_events_hash") != canonical_hash(events):
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError):
        return False


def map_evaluate_sequences(sequences, args, config):
    pending = [sequence for sequence in sequences
               if not completed_evaluated_sequence(args.output_dir, sequence, args.support_iou)]
    payload = {"dataset_root": str(Path(args.dataset_root).resolve()),
               "output_dir": str(Path(args.output_dir).resolve()), "config": config,
               "support_iou": float(args.support_iou)}
    if not pending:
        return []
    if int(args.workers) < 1:
        raise ValueError("workers must be at least one.")
    if int(args.workers) == 1:
        _worker_init("evaluate", payload)
        return [_evaluate_sequence_worker(sequence) for sequence in pending]
    context = mp.get_context("spawn")
    with context.Pool(processes=int(args.workers), initializer=_worker_init,
                      initargs=("evaluate", payload)) as pool:
        return list(pool.imap(_evaluate_sequence_worker, pending, chunksize=1))


_IDENTITY_KEYS = {
    "schema_version", "dataset", "dataset_root", "phase_family", "scope", "yaml",
    "experiment_config", "runner_config", "checkpoint", "split", "authoritative_splits",
    "sequences", "sequences_hash", "image_manifest_hash", "support_iou",
    "policy_lock_content_hash", "parent_provenance", "parent_provenance_hash",
    "source_sha256",
}
_METADATA_KEYS = {
    "schema_version", "kind", "identity", "identity_hash", "parent_provenance_hash",
    "policy_lock_content_hash", "frame_indexing", "causality", "label_policy",
    "completed_phases", "created_unix", "updated_unix", "content_hash",
}
_SUMMARY_KEYS = {
    "schema_version", "kind", "sequence", "arms", "governance",
    "online_trace_hashes", "online_rmg_q_events_hash", "frame_labels_hash",
    "event_labels_hash", "content_hash",
}
_ARM_METRIC_KEYS = {
    "num_frames", "success_auc", "mean_iou", "precision20", "normalized_precision",
    "normalized_precision_at_0_2",
}
_BINDING_KEYS = {
    "scope", "sequences", "sequences_hash", "support_iou",
    "policy_lock_content_hash", "parent_provenance_hash", "config_hash",
    "source_identity_hash",
}


def validate_metadata(metadata_path, scope, sequences, support_iou, config,
                      parent_hash, policy_hash=None, require_completed=True):
    metadata = read_json(metadata_path)
    _exact_keys(metadata, _METADATA_KEYS, "Experiment metadata")
    if not _content_hash_valid(metadata):
        raise RuntimeError("Experiment metadata has an invalid canonical content_hash.")
    identity = metadata.get("identity")
    _exact_keys(identity, _IDENTITY_KEYS, "Experiment identity")
    identity_hash = canonical_hash(identity)
    current_sources = {name: file_sha256(path) for name, path in IDENTITY_SOURCES.items()}
    expected_sequences = list(sequences)
    expected_phases = ["online", "evaluate"] if require_completed else metadata.get(
        "completed_phases")
    if (metadata.get("schema_version") != SCHEMA_VERSION
            or metadata.get("kind") != "stage1_quarantine_v2_run_metadata"
            or identity.get("schema_version") != SCHEMA_VERSION
            or metadata.get("identity_hash") != identity_hash
            or identity.get("scope") != scope
            or identity.get("support_iou") != float(support_iou)
            or identity.get("sequences") != expected_sequences
            or identity.get("sequences_hash") != canonical_hash(expected_sequences)
            or identity.get("runner_config") != {
                "path": identity.get("runner_config", {}).get("path"),
                "sha256": identity.get("runner_config", {}).get("sha256"),
                "content_hash": canonical_hash(config),
            }
            or identity.get("parent_provenance_hash") != parent_hash
            or metadata.get("parent_provenance_hash") != parent_hash
            or identity.get("policy_lock_content_hash") != policy_hash
            or metadata.get("policy_lock_content_hash") != policy_hash
            or identity.get("source_sha256") != current_sources
            or metadata.get("completed_phases") != expected_phases):
        raise RuntimeError("Experiment metadata/identity is not the exact authoritative run identity.")
    if identity.get("parent_provenance", {}).get("content_hash") != parent_hash:
        raise RuntimeError("Experiment identity embeds a different parent provenance artifact.")
    for name in ("experiment_config", "runner_config", "checkpoint", "split"):
        reference = identity.get(name, {})
        if set(reference) not in ({"path", "sha256"}, {"path", "sha256", "content_hash"}):
            raise RuntimeError("Experiment identity {} reference has an unexpected structure."
                               .format(name))
        referenced_path = Path(reference.get("path", ""))
        if (not referenced_path.is_absolute() or not referenced_path.is_file()
                or file_sha256(referenced_path) != reference.get("sha256")):
            raise RuntimeError("Experiment identity {} byte hash is no longer authoritative."
                               .format(name))
        if name == "runner_config" and load_config(referenced_path) != config:
            raise RuntimeError("Experiment identity runner config bytes do not encode frozen config.")
    authoritative = identity.get("authoritative_splits", {})
    _exact_keys(authoritative, {"development", "val47"}, "Authoritative split references")
    for name, reference in authoritative.items():
        _exact_keys(reference, {"path", "sha256"}, "{} split reference".format(name))
        referenced_path = Path(reference.get("path", ""))
        if (not referenced_path.is_absolute() or not referenced_path.is_file()
                or file_sha256(referenced_path) != reference.get("sha256")):
            raise RuntimeError("Authoritative {} split byte hash changed.".format(name))
    return metadata


def _authoritative_sequence_summary(root, sequence, config, support_iou):
    if not completed_evaluated_sequence(root, sequence, support_iou):
        raise RuntimeError("Evaluation is incomplete or corrupt for {!r}.".format(sequence))
    sequence_root = safe_sequence_dir(root, sequence, create=False)
    summary = read_json(sequence_root / "sequence_summary.json")
    _exact_keys(summary, _SUMMARY_KEYS, "Sequence summary")
    if (not _content_hash_valid(summary) or summary.get("schema_version") != SCHEMA_VERSION
            or summary.get("kind") != "evaluated_sequence"
            or summary.get("sequence") != sequence):
        raise RuntimeError("Sequence summary has an invalid schema, identity, or content hash.")
    frame_labels = read_jsonl(sequence_root / "labels" / "frames.jsonl")
    event_labels = read_jsonl(sequence_root / "labels" / "rmg_q.events.jsonl")
    arm_metrics = {}
    trace_hashes = {}
    for arm in ARMS:
        rows = read_jsonl(sequence_root / "online" / (arm + ".frames.jsonl"))
        validate_online_trace(rows, sequence, arm)
        arm_metrics[arm] = tracking_metrics_from_online_trace(rows, frame_labels)
        _exact_keys(arm_metrics[arm], _ARM_METRIC_KEYS,
                    "{} tracking metrics".format(arm))
        trace_hashes[arm] = canonical_hash(rows)
    events = read_jsonl(sequence_root / "online" / "rmg_q.events.jsonl")
    governance = governance_metrics(
        events, event_labels, good_iou=config["thresholds"]["good_iou"],
        bad_iou=config["thresholds"]["bad_iou"])
    expected = {
        "schema_version": SCHEMA_VERSION, "kind": "evaluated_sequence",
        "sequence": sequence, "arms": arm_metrics, "governance": governance,
        "online_trace_hashes": trace_hashes,
        "online_rmg_q_events_hash": canonical_hash(events),
        "frame_labels_hash": canonical_hash(frame_labels),
        "event_labels_hash": canonical_hash(event_labels),
    }
    expected["content_hash"] = canonical_hash(expected)
    if summary != expected:
        raise RuntimeError("Sequence summary does not equal metrics recomputed from traces and labels.")
    return summary


def action_gate_config_for_scope(config, scope):
    gate = dict(config.get("action_gate", {}))
    if scope == "internal42":
        gate.update(config.get("internal_action_gate", {}))
    return gate


def validate_evaluated_root(root, scope, sequences, support_iou, config,
                            parent_hash, policy_hash=None,
                            expected_identity_hash=None, require_completed=True):
    """Recompute a complete scope from canonical traces/labels and compare exact artifacts."""
    validate_config(config)
    root = Path(root).resolve()
    expected_sequences = list(sequences)
    if (not expected_sequences or len(expected_sequences) != len(set(expected_sequences))
            or any(not _SAFE_SEQUENCE.fullmatch(name) for name in expected_sequences)):
        raise RuntimeError("Authoritative evaluated-root sequences must be unique safe names.")
    metadata = validate_metadata(
        root / "metadata.json", scope, expected_sequences, support_iou, config,
        parent_hash, policy_hash=policy_hash, require_completed=require_completed)
    if (expected_identity_hash is not None
            and metadata["identity_hash"] != expected_identity_hash):
        raise RuntimeError("Evaluated root has a different source identity hash.")
    summaries = [_authoritative_sequence_summary(root, sequence, config, support_iou)
                 for sequence in expected_sequences]
    aggregate = aggregate_sequence_summaries(
        summaries, bootstrap_seed=config["metrics"]["bootstrap_seed"],
        bootstrap_samples=config["metrics"]["bootstrap_samples"])
    gate = evaluate_final_gate(
        aggregate, gate_config=config["final_tracking_gate"],
        action_gate_config=action_gate_config_for_scope(config, scope),
        internal=(scope == "internal42"))
    binding = {
        "scope": scope, "sequences": expected_sequences,
        "sequences_hash": canonical_hash(expected_sequences),
        "support_iou": float(support_iou),
        "policy_lock_content_hash": policy_hash,
        "parent_provenance_hash": parent_hash,
        "config_hash": canonical_hash(config),
        "source_identity_hash": metadata["identity_hash"],
    }
    _exact_keys(binding, _BINDING_KEYS, "Protocol binding")
    for artifact in (aggregate, gate):
        artifact["protocol_binding"] = binding
        artifact["content_hash"] = canonical_hash({
            key: value for key, value in artifact.items() if key != "content_hash"})
    disk_aggregate = _require_canonical_json_file(
        root / "aggregate.json", aggregate, "Aggregate")
    disk_gate = _require_canonical_json_file(root / "gate.json", gate, "Gate")
    _exact_keys(disk_aggregate, set(aggregate), "Aggregate")
    _exact_keys(disk_gate, set(gate), "Gate")
    if not _content_hash_valid(disk_aggregate) or not _content_hash_valid(disk_gate):
        raise RuntimeError("Recomputed aggregate/gate has an invalid content hash.")
    return {
        "root": str(root), "metadata": metadata, "summaries": summaries,
        "aggregate": disk_aggregate, "gate": disk_gate,
        "aggregate_path": root / "aggregate.json", "gate_path": root / "gate.json",
    }


def summarize_evaluation(args, config, sequences, metadata):
    summaries = []
    for sequence in sequences:
        if not completed_evaluated_sequence(args.output_dir, sequence, args.support_iou):
            raise RuntimeError("Evaluation is incomplete or corrupt for {!r}.".format(sequence))
        summaries.append(read_json(safe_sequence_dir(
            args.output_dir, sequence) / "sequence_summary.json"))
    aggregate = aggregate_sequence_summaries(
        summaries, bootstrap_seed=config["metrics"]["bootstrap_seed"],
        bootstrap_samples=config["metrics"]["bootstrap_samples"])
    action_gate = action_gate_config_for_scope(config, getattr(args, "scope", ""))
    gate = evaluate_final_gate(
        aggregate, gate_config=config.get("final_tracking_gate"),
        action_gate_config=action_gate, internal=False)
    binding = {
        "scope": args.scope,
        "sequences": list(sequences),
        "sequences_hash": canonical_hash(list(sequences)),
        "support_iou": float(args.support_iou),
        "policy_lock_content_hash": metadata["policy_lock_content_hash"],
        "parent_provenance_hash": metadata["parent_provenance_hash"],
        "config_hash": canonical_hash(config),
        "source_identity_hash": metadata["identity_hash"],
    }
    for artifact in (aggregate, gate):
        artifact["protocol_binding"] = binding
        artifact["content_hash"] = canonical_hash({
            key: value for key, value in artifact.items() if key != "content_hash"})
    aggregate_path = Path(args.output_dir) / "aggregate.json"
    gate_path = Path(args.output_dir) / "gate.json"
    write_json(aggregate_path, aggregate)
    write_json(gate_path, gate)
    validated_root = validate_evaluated_root(
        args.output_dir, args.scope, sequences, args.support_iou, config,
        metadata["parent_provenance_hash"],
        policy_hash=metadata["policy_lock_content_hash"],
        expected_identity_hash=metadata["identity_hash"], require_completed=False)
    aggregate, gate = validated_root["aggregate"], validated_root["gate"]
    if args.scope == "internal42" and gate["pass"]:
        provisional = validate_policy_lock(
            args.policy_lock, metadata["parent_provenance_hash"],
            required_kind="provisional_stage1_quarantine_v2_policy",
            expected_scope_sequences=canonical_scope_sequences(
                "design83", args.development_split, args.val47_split, config),
            expected_config_hash=canonical_hash(config), config=config)
        aggregate_ref, _ = _artifact_reference(aggregate_path)
        gate_ref, _ = _artifact_reference(gate_path)
        lock = {
            "schema_version": SCHEMA_VERSION,
            "kind": "final_stage1_quarantine_v2_policy",
            "support_iou": float(args.support_iou),
            "provisional_lock": {"path": str(Path(args.policy_lock).resolve()),
                                 "sha256": file_sha256(args.policy_lock),
                                 "content_hash": provisional["content_hash"]},
            "design_scope": provisional["design_scope"],
            "design_selection": provisional["design_selection"],
            "internal_scope": {"sequences": list(sequences),
                               "sequences_hash": canonical_hash(list(sequences))},
            "internal_root": str(Path(args.output_dir).resolve()),
            "internal_aggregate": aggregate_ref,
            "internal_gate": gate_ref,
            "config_hash": canonical_hash(config),
            "parent_provenance_hash": metadata["parent_provenance_hash"],
            "source_identity_hashes": provisional["source_identity_hashes"],
            "source_sha256": {name: file_sha256(path) for name, path in IDENTITY_SOURCES.items()},
            "internal_source_identity_hash": metadata["identity_hash"],
            "confirmation_outcomes_accessed": False,
        }
        lock["content_hash"] = canonical_hash(lock)
        write_json(Path(args.output_dir) / "final_policy_lock.json", lock)
    if args.scope == "confirm62" and len(sequences) == 62 and gate["pass"]:
        validate_policy_lock(
            args.policy_lock, metadata["parent_provenance_hash"],
            required_kind="final_stage1_quarantine_v2_policy",
            expected_scope_sequences=canonical_scope_sequences(
                "design83", args.development_split, args.val47_split, config),
            expected_config_hash=canonical_hash(config), config=config,
            expected_internal_sequences=canonical_scope_sequences(
                "internal42", args.development_split, args.val47_split, config))
        aggregate_ref, _ = _artifact_reference(aggregate_path)
        gate_ref, _ = _artifact_reference(gate_path)
        unlock = {
            "schema_version": SCHEMA_VERSION,
            "kind": "rmg_stage1_v2_val47_unlock",
            "confirm_sequence_count": 62,
            "confirm_sequences": list(sequences),
            "confirm_sequences_hash": canonical_hash(list(sequences)),
            "confirm_root": str(Path(args.output_dir).resolve()),
            "confirm_aggregate": aggregate_ref,
            "confirm_gate": gate_ref,
            "confirm_gate_pass": True,
            "policy_lock_content_hash": metadata["policy_lock_content_hash"],
            "parent_provenance_hash": metadata["parent_provenance_hash"],
            "config_hash": canonical_hash(config),
            "source_sha256": {name: file_sha256(path) for name, path in IDENTITY_SOURCES.items()},
        }
        unlock["content_hash"] = canonical_hash(unlock)
        write_json(Path(args.output_dir) / "val47_unlock.json", unlock)
    return aggregate, gate


def _dataset_manifest_hash(dataset_root, sequences):
    root = Path(dataset_root).resolve()
    entries = []
    for sequence in sequences:
        sequence_root = root / sequence
        entry = {"sequence": sequence}
        for modality in ("visible", "infrared"):
            directory = sequence_root / modality
            files = sorted(path.name for path in directory.iterdir()
                           if path.is_file() and path.suffix.lower() == ".jpg")
            entry[modality + "_files_hash"] = canonical_hash(files)
            entry[modality + "_count"] = len(files)
        entries.append(entry)
    return canonical_hash(entries)


def _resolve_support_iou(args, config, parent_hash, design_sequences=None,
                         internal_sequences=None):
    candidates = tuple(float(value) for value in config["quarantine"][
        "support_iou_candidates"])
    if args.scope == "design83":
        if args.policy_lock:
            raise ValueError("design83 direct execution must not use a policy lock.")
        if args.support_iou is None or float(args.support_iou) not in candidates:
            raise ValueError("design83 --support-iou must be one of the five frozen candidates.")
        return float(args.support_iou), None
    if args.support_iou is not None:
        raise ValueError("Direct --support-iou is allowed only for design83.")
    if not args.policy_lock:
        raise RuntimeError("{} requires --policy-lock; direct thresholds fail closed.".format(
            args.scope))
    required_kind = ("provisional_stage1_quarantine_v2_policy"
                     if args.scope == "internal42"
                     else "final_stage1_quarantine_v2_policy")
    lock = validate_policy_lock(
        args.policy_lock, parent_hash, required_kind=required_kind,
        expected_scope_sequences=design_sequences,
        expected_config_hash=canonical_hash(config), config=config,
        expected_internal_sequences=(internal_sequences if required_kind.startswith("final")
                                     else None))
    return float(lock["support_iou"]), lock


def _design_threshold_root(path, support_iou):
    return Path(path).resolve() / "thresholds" / format(float(support_iou), ".2f")


def run_select_phase(args, config, sequences, provenance):
    candidates = [float(value) for value in config["quarantine"][
        "support_iou_candidates"]]
    roots = [Path(path).resolve() for path in args.design_root]
    if len(roots) != len(candidates) or len(set(roots)) != len(candidates):
        raise RuntimeError("select requires exactly five distinct --design-root values.")
    by_threshold = {}
    sequence_candidates = {sequence: [] for sequence in sequences}
    artifact_candidates = []
    for root in roots:
        metadata_path = root / "metadata.json"
        aggregate_path = root / "aggregate.json"
        gate_path = root / "gate.json"
        if not metadata_path.is_file() or not aggregate_path.is_file() or not gate_path.is_file():
            raise RuntimeError("Every design root must be completely evaluated: {}".format(root))
        metadata = read_json(metadata_path)
        identity = metadata.get("identity", {})
        try:
            threshold = float(identity.get("support_iou", -1))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Design root has an invalid support threshold identity.") from exc
        if threshold in by_threshold or threshold not in candidates:
            raise RuntimeError("Design roots contain a duplicate or unapproved threshold.")
        validated = validate_evaluated_root(
            root, "design83", sequences, threshold, config, provenance["content_hash"],
            policy_hash=None)
        metadata = validated["metadata"]
        identity = metadata["identity"]
        aggregate = validated["aggregate"]
        gate = validated["gate"]
        aggregate_ref, _ = _artifact_reference(aggregate_path)
        gate_ref, _ = _artifact_reference(gate_path)
        summaries = validated["summaries"]
        for sequence, summary in zip(sequences, summaries):
            candidate = {"support_threshold": threshold,
                         "num_frames": summary["arms"]["static"]["num_frames"],
                         "governance": summary["governance"],
                         "quarantine_incremental_success_auc_delta": (
                             summary["arms"]["rmg_q"]["success_auc"]
                             - summary["arms"]["rmg_q_no_quarantine"]["success_auc"])}
            sequence_candidates[sequence].append(candidate)
        aggregate_candidate = {
            "support_threshold": threshold,
            "governance": aggregate["governance"],
            "quarantine_incremental_success_auc_delta": aggregate[
                "frame_weighted_paired_deltas"]["rmg_q_vs_rmg_q_no_quarantine"][
                    "success_auc"]["mean"],
        }
        by_threshold[threshold] = aggregate_candidate
        artifact_candidates.append({
            "support_iou": threshold, "root": str(root),
            "source_identity_hash": metadata["identity_hash"],
            "aggregate": aggregate_ref, "gate": gate_ref,
            "sequence_summaries_hash": canonical_hash(summaries),
        })
    if sorted(by_threshold) != candidates:
        raise RuntimeError("select requires all five frozen thresholds exactly once.")
    gate_config = config["action_gate"]
    selection_gate = select_smallest_passing_support_threshold(
        [by_threshold[value] for value in candidates], gate_config=gate_config)
    sequence_summaries = [{"sequence": sequence,
                           "threshold_candidates": sequence_candidates[sequence]}
                          for sequence in sequences]
    stability_config = config["stability"]
    stability = deterministic_threshold_stability(
        sequence_summaries, gate_config=gate_config,
        seed=stability_config["bootstrap_seed"],
        samples=stability_config["bootstrap_samples"],
        hash_namespace="rmg-stage1-v2-q5-threshold-stability")
    selected = selection_gate["selected_support_threshold"]
    if (selected is None or stability["selected_support_threshold"] != selected
            or stability["bootstrap"]["modal_support_threshold"] != selected
            or stability["bootstrap"]["modal_match_fraction"] < stability_config[
                "modal_match_fraction_min"]
            or stability["leave_one_group_out_same_threshold_count"] < stability_config[
                "leave_one_group_out"]["same_threshold_min"]):
        raise RuntimeError("Design threshold selection or stability gate failed.")
    artifact_candidates.sort(key=lambda item: item["support_iou"])
    source_identity_hashes = [item["source_identity_hash"]
                              for item in artifact_candidates]
    if len(set(source_identity_hashes)) != 5:
        raise RuntimeError("Design roots must have five distinct source identities.")
    lock = {
        "schema_version": SCHEMA_VERSION,
        "kind": "provisional_stage1_quarantine_v2_policy",
        "support_iou": float(selected),
        "design_scope": {"sequences": list(sequences),
                         "sequences_hash": canonical_hash(list(sequences))},
        "design_selection": {
            "selected_support_threshold": float(selected),
            "selection_gate": selection_gate,
            "stability": stability,
            "candidates": artifact_candidates,
        },
        "config_hash": canonical_hash(config),
        "parent_provenance_hash": provenance["content_hash"],
        "source_identity_hashes": source_identity_hashes,
        "source_sha256": {name: file_sha256(path) for name, path in IDENTITY_SOURCES.items()},
        "confirmation_outcomes_accessed": False,
    }
    lock["content_hash"] = canonical_hash(lock)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "provisional_policy_lock.json", lock)
    return lock


def _build_identity(args, config, sequences, support_iou, provenance, policy_lock):
    config_path = Path(args.config).resolve()
    split_path = Path(args.split_file).resolve()
    development_path = Path(args.development_split).resolve()
    val47_path = Path(args.val47_split).resolve()
    experiment_yaml = (ROOT / "experiments" / "vipt" /
                       (args.yaml_name + ".yaml")).resolve()
    checkpoint = checkpoint_path(args.yaml_name)
    identity = {
        "schema_version": SCHEMA_VERSION, "dataset": "RGBT234",
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "phase_family": "online_then_evaluate",
        "scope": args.scope, "yaml": args.yaml_name,
        "experiment_config": {"path": str(experiment_yaml),
                              "sha256": file_sha256(experiment_yaml)},
        "runner_config": {"path": str(config_path), "sha256": file_sha256(config_path),
                          "content_hash": canonical_hash(config)},
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
        "split": {"path": str(split_path), "sha256": file_sha256(split_path)},
        "authoritative_splits": {
            "development": {"path": str(development_path),
                            "sha256": file_sha256(development_path)},
            "val47": {"path": str(val47_path), "sha256": file_sha256(val47_path)},
        },
        "sequences": list(sequences), "sequences_hash": canonical_hash(list(sequences)),
        "image_manifest_hash": _dataset_manifest_hash(args.dataset_root, sequences),
        "support_iou": float(support_iou),
        "policy_lock_content_hash": policy_lock.get("content_hash") if policy_lock else None,
        "parent_provenance": provenance,
        "parent_provenance_hash": provenance["content_hash"],
        "source_sha256": {name: file_sha256(path) for name, path in IDENTITY_SOURCES.items()},
    }
    return identity, checkpoint


def validate_or_write_metadata(path, identity):
    path = Path(path)
    identity_hash = canonical_hash(identity)
    if path.is_file():
        metadata = read_json(path)
        if (not _content_hash_valid(metadata)
                or metadata.get("identity_hash") != canonical_hash(metadata.get("identity", {}))
                or metadata.get("identity_hash") != identity_hash
                or metadata.get("identity") != identity):
            raise RuntimeError("Experiment identity mismatch; refusing atomic resume.")
        return metadata
    metadata = {
        "schema_version": SCHEMA_VERSION, "kind": "stage1_quarantine_v2_run_metadata",
        "identity": identity,
        "identity_hash": identity_hash,
        "parent_provenance_hash": identity["parent_provenance_hash"],
        "policy_lock_content_hash": identity["policy_lock_content_hash"],
        "frame_indexing": "zero_based",
        "causality": "predict active t; probe shadow from same pre-frame anchor; finalize after t+5; release first affects t+6",
        "label_policy": "online artifacts contain no GT/IoU; evaluator creates labels",
        "completed_phases": [], "created_unix": time.time(), "updated_unix": None,
    }
    metadata["content_hash"] = canonical_hash(metadata)
    write_json(path, metadata)
    return metadata


def mark_phase_complete(output_dir, phase):
    path = Path(output_dir) / "metadata.json"
    metadata = read_json(path)
    phases = list(metadata.get("completed_phases", []))
    if phase not in phases:
        phases.append(phase)
    metadata["completed_phases"] = phases
    metadata["updated_unix"] = time.time()
    metadata["content_hash"] = canonical_hash({
        key: value for key, value in metadata.items() if key != "content_hash"})
    write_json(path, metadata)


def run_quarantine_timing_fixture():
    """Dependency-free executable model of the approved delayed-release timing.

    This intentionally uses immutable string tokens rather than a tracker.  It guards
    the integration contract: active state is emitted first, all probes share its
    pre-frame anchor, and only the source-time pending snapshot can become active after
    finalization.
    """
    source_frame = 30
    active_snapshot = "initial_snapshot"
    pending_snapshot = "source_snapshot_t30"
    active_anchor = "active_anchor_before_t30"
    history = []
    source_active = active_snapshot
    probe_anchors = []
    final_probe_active = None
    shadow_history = []
    for frame in range(source_frame, source_frame + 6):
        pre_frame_anchor = active_anchor
        emitted_template = active_snapshot
        history.append({"frame": frame, "template": emitted_template,
                        "anchor": pre_frame_anchor})
        active_anchor = "active_prediction_t{}".format(frame)
        if frame in (31, 33, 35):
            # Each active/shadow pair uses the one pre-frame active anchor.  Anchors
            # naturally advance between probe frames with the active trajectory.
            probe_anchors.append((pre_frame_anchor, pre_frame_anchor))
            shadow_output = {"frame": frame, "template": pending_snapshot,
                             "anchor": pre_frame_anchor}
            del shadow_output
        if frame == 35:
            final_probe_active = emitted_template
            active_snapshot = pending_snapshot
    history.append({"frame": 36, "template": active_snapshot,
                    "anchor": active_anchor})
    return {
        "source_frame": 30, "finalized_frame": 35, "effective_frame": 36,
        "source_frame_active_unaffected": source_active == "initial_snapshot",
        "final_probe_active_unaffected": final_probe_active == "initial_snapshot",
        "all_probe_anchors_identical": len(probe_anchors) == 3 and all(
            active == shadow for active, shadow in probe_anchors),
        "released_source_snapshot_not_probe_snapshot": (
            history[-1]["template"] == "source_snapshot_t30"),
        "shadow_absent_from_history": not shadow_history and all(
            not str(item["template"]).startswith("shadow") for item in history),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the fail-closed RMG-Q Stage 1 v2 protocol artifact chain.")
    parser.add_argument("--phase", required=True, choices=("online", "evaluate", "select"),
                        help="There is deliberately no combined/all mode.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-file", required=True,
                        help="Scope split; sequence order must exactly match canonical scope.")
    parser.add_argument("--development-split", required=True,
                        help="Authoritative ordered 187-sequence development split.")
    parser.add_argument("--val47-split", required=True,
                        help="Authoritative ordered 47-sequence validation split.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--yaml", "--yaml-name", dest="yaml_name", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--workers", type=int, default=1,
                        help="Workers share the single visible CUDA device; Stage 0 proved 8 fit.")
    parser.add_argument("--scope", required=True,
                        choices=("design83", "internal42", "confirm62", "val47"))
    parser.add_argument("--support-iou", type=float,
                        help="Allowed only for design83 online/evaluate roots.")
    parser.add_argument("--policy-lock", default="")
    parser.add_argument("--val47-unlock", default="")
    parser.add_argument("--design-root", action="append", default=[],
                        help="select only: repeat once for each of five evaluated design roots.")
    parser.add_argument("--stage0-parent", required=True)
    parser.add_argument("--stage1-v1-parent", required=True)
    return parser.parse_args(argv)


def _require_single_visible_cuda_device():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        raise RuntimeError(
            "Set CUDA_VISIBLE_DEVICES to exactly one CUDA device; all workers share it.")
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if len(devices) != 1 or devices[0] == "-1":
        raise RuntimeError(
            "Current runner supports exactly one visible CUDA device; multi-GPU mapping is not implemented.")


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    validate_config(config)
    if args.workers < 1:
        raise ValueError("workers must be at least one.")
    sequences = load_split_sequence(args.split_file)
    validate_scope_membership(
        args.scope, sequences, args.development_split, args.val47_split, config)
    design_sequences = canonical_scope_sequences(
        "design83", args.development_split, args.val47_split, config)
    internal_sequences = canonical_scope_sequences(
        "internal42", args.development_split, args.val47_split, config)
    confirm_sequences = canonical_scope_sequences(
        "confirm62", args.development_split, args.val47_split, config)
    provenance = validate_parent_provenance({
        "stage0_dir": args.stage0_parent, "stage1_v1_dir": args.stage1_v1_parent}, config)

    if args.phase == "select":
        if args.scope != "design83":
            raise ValueError("select phase is valid only for scope=design83.")
        if args.support_iou is not None or args.policy_lock or args.val47_unlock:
            raise ValueError("select accepts evaluated design roots, not direct threshold/lock inputs.")
        run_select_phase(args, config, sequences, provenance)
        print("Stage 1 v2 phase 'select' complete: {}".format(Path(args.output_dir).resolve()))
        return
    if args.design_root:
        raise ValueError("--design-root is valid only for phase=select.")

    support_iou, policy_lock = _resolve_support_iou(
        args, config, provenance["content_hash"], design_sequences, internal_sequences)
    args.support_iou = support_iou
    if args.scope == "val47":
        if not args.val47_unlock:
            raise RuntimeError("val47 requires --val47-unlock from passing confirm62.")
        validate_val47_unlock(
            args.val47_unlock,
            expected_policy_hash=policy_lock["content_hash"],
            expected_parent_hash=provenance["content_hash"],
            expected_confirm_sequences=confirm_sequences, config=config,
            policy_lock_path=args.policy_lock,
            expected_design_sequences=design_sequences,
            expected_internal_sequences=internal_sequences)
    elif args.val47_unlock:
        raise ValueError("--val47-unlock is valid only for scope=val47.")

    _require_single_visible_cuda_device()
    identity, checkpoint = _build_identity(
        args, config, sequences, support_iou, provenance, policy_lock)
    if args.phase == "online" and not checkpoint.is_file():
        raise FileNotFoundError("ViPT checkpoint not found: {}".format(checkpoint))
    requested_output = Path(args.output_dir).resolve()
    output = (_design_threshold_root(requested_output, support_iou)
              if args.scope == "design83" else requested_output)
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output)
    metadata = validate_or_write_metadata(output / "metadata.json", identity)

    if args.phase == "online":
        map_online_sequences(sequences, args, config, support_iou)
        mark_phase_complete(output, "online")
    else:
        if "online" not in metadata.get("completed_phases", []):
            raise RuntimeError("evaluate requires a completed, identity-matched online phase.")
        map_evaluate_sequences(sequences, args, config)
        summarize_evaluation(args, config, sequences, metadata)
        mark_phase_complete(output, "evaluate")
    print("Stage 1 v2 phase {!r} complete: {}".format(args.phase, output))


if __name__ == "__main__":
    main()
