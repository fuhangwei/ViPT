"""Pure metrics and gates for approved RMG-Q Stage 1 v2."""

import hashlib
import math
from collections import Counter, defaultdict

import numpy as np

try:
    from analysis.memory_oracle.compute_stage0_metrics import (
        canonical_hash,
        deterministic_weighted_bootstrap_ci,
        tracking_metrics,
    )
except ModuleNotFoundError:
    from compute_stage0_metrics import (
        canonical_hash,
        deterministic_weighted_bootstrap_ci,
        tracking_metrics,
    )


SCHEMA_VERSION = "rmg-stage1-v2-q5"
ARM_NAMES = (
    "static",
    "periodic_pred",
    "confidence_e050",
    "rmg_q",
    "rmg_q_no_quarantine",
)
METRIC_KEYS = (
    "success_auc",
    "mean_iou",
    "precision20",
    "normalized_precision",
    "normalized_precision_at_0_2",
)
GOOD_IOU = 0.7
BAD_IOU = 0.1
DEFAULT_ACTION_GATE = {
    "immediate_precision_min": 0.90,
    "immediate_bad_rate_max": 0.02,
    "immediate_coverage_min": 0.03,
    "combined_coverage_min": 0.10,
    "combined_bad_rate_max": 0.05,
    "release_bad_rate_max": 0.05,
    "combined_writes_min": 100,
    "combined_sequences_min": 20,
    "quarantine_incremental_auc_delta_exclusive_min": 0.0,
}
INTERNAL_ACTION_GATE = {
    **DEFAULT_ACTION_GATE,
    "combined_writes_min": 50,
    "combined_sequences_min": 10,
}
DEFAULT_FINAL_GATE = {
    "rmg_q_static_auc_delta_min": 0.005,
    "rmg_q_periodic_auc_delta_min": 0.010,
    "rmg_q_periodic_auc_ci_low_exclusive_min": 0.0,
    "worsened_fraction_max": 0.20,
    "clean_auc_delta_min": -0.002,
    "clean_worsened_fraction_max": 0.10,
    "rmg_q_no_quarantine_auc_delta_exclusive_min": 0.0,
    "rmg_q_confidence_e050_auc_delta_exclusive_min": 0.0,
}
_FORBIDDEN_ONLINE_KEYS = frozenset({
    "iou",
    "evaluation_iou",
    "candidate_iou",
    "source_candidate_iou",
    "release_frame_iou",
})
_FORBIDDEN_ONLINE_KEY_PARTS = ("ground_truth", "groundtruth")


def _hashed(payload):
    result = dict(payload)
    result["content_hash"] = canonical_hash(result)
    return result


def canonical_content_hash(value):
    return canonical_hash({key: item for key, item in value.items()
                           if key != "content_hash"} if isinstance(value, dict) else value)


def content_hash_matches(value):
    return (isinstance(value, dict) and isinstance(value.get("content_hash"), str)
            and value["content_hash"] == canonical_content_hash(value))


def _finite_float(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a finite number.".format(name)) from exc
    if not math.isfinite(number):
        raise ValueError("{} must be a finite number.".format(name))
    return number


def _frame_idx(value):
    if isinstance(value, bool):
        raise ValueError("frame_idx must be a non-negative integer.")
    number = _finite_float(value, "frame_idx")
    if number < 0 or number != int(number):
        raise ValueError("frame_idx must be a non-negative integer.")
    return int(number)


def _identifier(value, name):
    if isinstance(value, (dict, list, tuple, set)) or value is None or value == "":
        raise ValueError("{} must be a non-empty scalar.".format(name))
    return type(value).__name__, str(value)


def _sequence(value):
    if value is None or str(value) == "":
        raise ValueError("sequence must be non-empty.")
    return str(value)


def _is_forbidden_online_key(key):
    normalized = str(key).lower().replace("-", "_")
    return (normalized in _FORBIDDEN_ONLINE_KEYS
            or normalized == "gt" or normalized.startswith("gt_")
            or normalized.endswith("_gt")
            or normalized.endswith("_evaluation_iou")
            or normalized.endswith("_candidate_iou")
            or any(part in normalized for part in _FORBIDDEN_ONLINE_KEY_PARTS))


def _reject_evaluation_data(value, location="row"):
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_forbidden_online_key(key):
                raise ValueError("Online trace contains forbidden GT/IoU key at {}.{}."
                                 .format(location, key))
            _reject_evaluation_data(item, "{}.{}".format(location, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_evaluation_data(item, "{}[{}]".format(location, index))


def _unique_index(rows, key_function, kind):
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("{} rows must be dictionaries.".format(kind))
        key = key_function(row)
        if key in result:
            raise ValueError("Duplicate {} key: {!r}.".format(kind, key))
        result[key] = row
    return result


def _frame_key(row):
    if "sequence" not in row or "frame_idx" not in row:
        raise ValueError("Frame row requires sequence and frame_idx.")
    return _sequence(row["sequence"]), _frame_idx(row["frame_idx"])


def _validate_contiguous_frames(keys, kind):
    by_sequence = defaultdict(list)
    for sequence, frame_idx in keys:
        by_sequence[sequence].append(frame_idx)
    for sequence, frames in by_sequence.items():
        ordered = sorted(frames)
        if ordered != list(range(ordered[0], ordered[-1] + 1)):
            raise ValueError("Noncontiguous {} frames for sequence {!r}."
                             .format(kind, sequence))


def join_frame_traces(online_rows, frame_labels):
    """Strictly join causal predictions to separately stored frame labels."""
    online = list(online_rows)
    labels = list(frame_labels)
    for index, row in enumerate(online):
        _reject_evaluation_data(row, "online_rows[{}]".format(index))
    online_by_key = _unique_index(online, _frame_key, "online frame")
    labels_by_key = _unique_index(labels, _frame_key, "frame label")
    _validate_contiguous_frames(online_by_key, "online")
    _validate_contiguous_frames(labels_by_key, "label")
    online_keys = set(online_by_key)
    label_keys = set(labels_by_key)
    if online_keys != label_keys:
        missing = sorted(online_keys - label_keys)
        extra = sorted(label_keys - online_keys)
        raise ValueError("Frame labels do not match online trace; missing={!r}, extra={!r}."
                         .format(missing, extra))
    joined = []
    for key in sorted(online_by_key):
        online_row = online_by_key[key]
        label = labels_by_key[key]
        if "gt_xywh" not in label:
            raise ValueError("Frame label {!r} is missing gt_xywh.".format(key))
        if "pred_xywh" not in online_row:
            raise ValueError("Online frame {!r} is missing pred_xywh.".format(key))
        joined.append({
            "sequence": key[0],
            "frame_idx": key[1],
            "pred_xywh": online_row["pred_xywh"],
            "gt_xywh": label["gt_xywh"],
        })
    return joined


def tracking_metrics_from_online_trace(online_rows, frame_labels):
    joined = join_frame_traces(online_rows, frame_labels)
    return tracking_metrics(
        [row["pred_xywh"] for row in joined],
        [row["gt_xywh"] for row in joined],
    )


def tracking_metrics_by_sequence(online_rows, frame_labels):
    joined = join_frame_traces(online_rows, frame_labels)
    grouped = defaultdict(list)
    for row in joined:
        grouped[row["sequence"]].append(row)
    return {
        sequence: tracking_metrics(
            [row["pred_xywh"] for row in rows],
            [row["gt_xywh"] for row in rows],
        )
        for sequence, rows in sorted(grouped.items())
    }


def _event_key(row):
    if "event_id" not in row:
        raise ValueError("Event row requires event_id.")
    return _identifier(row["event_id"], "event_id")


def join_event_traces(online_events, event_labels):
    """Strictly join governance decisions to source-candidate labels."""
    online = list(online_events)
    labels = list(event_labels)
    for index, row in enumerate(online):
        _reject_evaluation_data(row, "online_events[{}]".format(index))
    online_by_key = _unique_index(online, _event_key, "online event")
    labels_by_key = _unique_index(labels, _event_key, "event label")
    online_keys = set(online_by_key)
    label_keys = set(labels_by_key)
    if online_keys != label_keys:
        missing = sorted(online_keys - label_keys)
        extra = sorted(label_keys - online_keys)
        raise ValueError("Event labels do not match online trace; missing={!r}, extra={!r}."
                         .format(missing, extra))
    joined = []
    for key in sorted(online_by_key):
        label = labels_by_key[key]
        if "source_candidate_iou" not in label:
            raise ValueError("Event label {!r} is missing source_candidate_iou."
                             .format(key))
        source_iou = _finite_float(label["source_candidate_iou"],
                                   "source_candidate_iou")
        if not 0.0 <= source_iou <= 1.0:
            raise ValueError("source_candidate_iou must be in [0, 1].")
        joined.append({
            "online": online_by_key[key],
            "label": label,
            "source_candidate_iou": source_iou,
        })
    return joined


def _normalized_action(row):
    for key in ("governance_action", "action", "decision", "event_type", "outcome"):
        if row.get(key) is not None:
            return str(row[key]).strip().lower().replace("-", "_").replace(" ", "_")
    return "none"


def _event_categories(row):
    action = _normalized_action(row)
    immediate = bool(row.get("immediate_write", False))
    quarantine = bool(row.get("quarantined", row.get("is_quarantined", False)))
    release = bool(row.get("released", row.get("is_release", False)))
    if "release" in action:
        release = True
    elif "quarantine" in action or action in {"hold", "defer", "deferred"}:
        quarantine = True
    elif action in {"write", "commit", "update", "immediate", "immediate_write",
                    "write_immediate", "commit_immediate"}:
        immediate = True
    return immediate, quarantine, release


def _legal_source_opportunity(row, label):
    for source in (row, label):
        for key in ("legal_source_opportunity", "is_legal_source_opportunity",
                    "source_candidate_valid", "candidate_valid", "legal"):
            if key in source:
                return bool(source[key])
    return True


def _reason(row):
    for key in ("reason", "decision_reason", "action_reason", "governance_reason"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return "unspecified"


def _quality_bucket(items, legal_count, good_opportunity_count, legal_sequences,
                    good_iou, bad_iou):
    count = len(items)
    good = sum(item["source_candidate_iou"] >= good_iou for item in items)
    bad = sum(item["source_candidate_iou"] <= bad_iou for item in items)
    sequences = {item["sequence"] for item in items}
    return {
        "count": int(count),
        "good_count": int(good),
        "bad_count": int(bad),
        "precision": float(good / count) if count else 0.0,
        "bad_rate": float(bad / count) if count else 0.0,
        "coverage": float(count / legal_count) if legal_count else 0.0,
        "good_recall": (float(good / good_opportunity_count)
                        if good_opportunity_count else 0.0),
        "sequence_count": len(sequences),
        "sequence_coverage": (float(len(sequences) / len(legal_sequences))
                              if legal_sequences else 0.0),
    }


def governance_metrics(online_events, event_labels, good_iou=GOOD_IOU,
                       bad_iou=BAD_IOU):
    """Measure immediate and quarantine-release quality from source-time labels."""
    good_iou = float(good_iou)
    bad_iou = float(bad_iou)
    joined = join_event_traces(online_events, event_labels)
    legal = []
    buckets = {name: [] for name in ("immediate", "quarantine", "release")}
    reasons = Counter()
    reasons_by_action = defaultdict(Counter)
    for item in joined:
        row = item["online"]
        label = item["label"]
        sequence = _sequence(row.get("sequence", label.get("sequence", "__unknown__")))
        enriched = {**item, "sequence": sequence}
        is_legal = _legal_source_opportunity(row, label)
        if is_legal:
            legal.append(enriched)
        immediate, quarantine, release = _event_categories(row)
        reason = _reason(row)
        reasons[reason] += 1
        for name, selected in (("immediate", immediate), ("quarantine", quarantine),
                               ("release", release)):
            if selected and is_legal:
                buckets[name].append(enriched)
                reasons_by_action[name][reason] += 1
    legal_count = len(legal)
    good_count = sum(item["source_candidate_iou"] >= good_iou for item in legal)
    legal_sequences = {item["sequence"] for item in legal}
    combined_by_identity = {}
    for item in buckets["immediate"] + buckets["release"]:
        identity = _event_key(item["online"])
        combined_by_identity[identity] = item
    buckets["combined"] = list(combined_by_identity.values())
    quality = {
        name: _quality_bucket(items, legal_count, good_count, legal_sequences,
                              good_iou, bad_iou)
        for name, items in buckets.items()
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "good_iou": good_iou,
        "bad_iou": bad_iou,
        "num_events": len(joined),
        "num_legal_source_opportunities": legal_count,
        "num_good_source_opportunities": int(good_count),
        "num_legal_sequences": len(legal_sequences),
        **quality,
        "reasons": dict(sorted(reasons.items())),
        "reasons_by_action": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(reasons_by_action.items())
        },
    }
    for name, metrics in quality.items():
        for key, value in metrics.items():
            result["{}_{}".format(name, key)] = value
    result["immediate_writes"] = quality["immediate"]["count"]
    result["quarantines"] = quality["quarantine"]["count"]
    result["releases"] = quality["release"]["count"]
    result["combined_writes"] = quality["combined"]["count"]
    return _hashed(result)


def _bucket(metrics, name):
    nested = metrics.get(name, {})
    return {
        key: nested.get(key, metrics.get("{}_{}".format(name, key), 0))
        for key in ("count", "good_count", "bad_count", "precision", "bad_rate",
                    "coverage", "good_recall", "sequence_count", "sequence_coverage")
    }


def _extract_quarantine_delta(candidate):
    for key in ("quarantine_incremental_success_auc_delta",
                "quarantine_incremental_auc_delta",
                "success_auc_delta_vs_no_quarantine"):
        if key in candidate:
            value = candidate[key]
            if isinstance(value, dict):
                value = value.get("mean", 0.0)
            return float(value)
    comparisons = candidate.get("frame_weighted_paired_deltas", {})
    value = comparisons.get("rmg_q_vs_rmg_q_no_quarantine", {}).get("success_auc", 0.0)
    return float(value.get("mean", 0.0) if isinstance(value, dict) else value)


def _check(value, passed, **limits):
    return {"value": value, **limits, "pass": bool(passed)}


def evaluate_action_quality(candidate, gate_config=None, internal=False):
    """Evaluate pure Gate A and Gate B action quality."""
    config = dict(INTERNAL_ACTION_GATE if internal else DEFAULT_ACTION_GATE)
    if gate_config:
        config.update(gate_config)
    governance = candidate.get("governance", candidate)
    immediate = _bucket(governance, "immediate")
    release = _bucket(governance, "release")
    combined = _bucket(governance, "combined")
    incremental = _extract_quarantine_delta(candidate)
    gate_a_checks = {
        "immediate_precision": _check(
            immediate["precision"], immediate["precision"] >= config["immediate_precision_min"],
            minimum=config["immediate_precision_min"]),
        "immediate_bad_rate": _check(
            immediate["bad_rate"], immediate["bad_rate"] <= config["immediate_bad_rate_max"],
            maximum=config["immediate_bad_rate_max"]),
        "immediate_coverage": _check(
            immediate["coverage"], immediate["coverage"] >= config["immediate_coverage_min"],
            minimum=config["immediate_coverage_min"]),
    }
    gate_b_checks = {
        "combined_coverage": _check(
            combined["coverage"], combined["coverage"] >= config["combined_coverage_min"],
            minimum=config["combined_coverage_min"]),
        "combined_bad_rate": _check(
            combined["bad_rate"], combined["bad_rate"] <= config["combined_bad_rate_max"],
            maximum=config["combined_bad_rate_max"]),
        "release_bad_rate": _check(
            release["bad_rate"], release["bad_rate"] <= config["release_bad_rate_max"],
            maximum=config["release_bad_rate_max"]),
        "combined_writes": _check(
            combined["count"], combined["count"] >= config["combined_writes_min"],
            minimum=config["combined_writes_min"]),
        "combined_sequences": _check(
            combined["sequence_count"],
            combined["sequence_count"] >= config["combined_sequences_min"],
            minimum=config["combined_sequences_min"]),
        "quarantine_incremental_success_auc_delta": _check(
            incremental,
            incremental > config["quarantine_incremental_auc_delta_exclusive_min"],
            exclusive_minimum=config["quarantine_incremental_auc_delta_exclusive_min"]),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "gate_a": {"checks": gate_a_checks,
                   "pass": all(item["pass"] for item in gate_a_checks.values())},
        "gate_b": {"checks": gate_b_checks,
                   "pass": all(item["pass"] for item in gate_b_checks.values())},
    }
    result["pass"] = result["gate_a"]["pass"] and result["gate_b"]["pass"]
    return _hashed(result)


def _support_threshold(candidate):
    for key in ("support_threshold", "min_support", "threshold"):
        if key in candidate:
            return _finite_float(candidate[key], key)
    raise ValueError("Threshold candidate is missing support_threshold.")


def select_smallest_passing_support_threshold(candidates, gate_config=None,
                                              internal=False):
    """Select by minimum support only after every action-quality gate passes."""
    audited = []
    seen = set()
    for candidate in candidates:
        threshold = _support_threshold(candidate)
        if threshold in seen:
            raise ValueError("Duplicate support threshold: {!r}.".format(threshold))
        seen.add(threshold)
        gate = evaluate_action_quality(candidate, gate_config, internal)
        audited.append({"support_threshold": threshold, "gate": gate})
    audited.sort(key=lambda item: item["support_threshold"])
    passing = [item for item in audited if item["gate"]["pass"]]
    selected = passing[0]["support_threshold"] if passing else None
    return _hashed({
        "schema_version": SCHEMA_VERSION,
        "selection_rule": "smallest_support_threshold_passing_all_gates",
        "selected_support_threshold": selected,
        "candidates": audited,
    })


def select_support_threshold(candidates, gate_config=None, internal=False):
    return select_smallest_passing_support_threshold(candidates, gate_config, internal)


def _candidate_collection(summary):
    collection = summary.get("threshold_candidates", summary.get("candidates"))
    if collection is None:
        raise ValueError("Sequence summary is missing threshold_candidates.")
    if isinstance(collection, dict):
        return [dict(value, support_threshold=key) if "support_threshold" not in value
                else value for key, value in collection.items()]
    return list(collection)


def _aggregate_candidate_metrics(sequence_summaries, threshold):
    candidates = []
    for summary in sequence_summaries:
        matches = [candidate for candidate in _candidate_collection(summary)
                   if _support_threshold(candidate) == threshold]
        if len(matches) != 1:
            raise ValueError("Each sequence must contain exactly one candidate per threshold.")
        candidates.append((summary, matches[0]))
    legal = sum(int(candidate.get("governance", candidate).get(
        "num_legal_source_opportunities", 0)) for _, candidate in candidates)
    good_opportunities = sum(int(candidate.get("governance", candidate).get(
        "num_good_source_opportunities", 0)) for _, candidate in candidates)
    governance = {
        "num_legal_source_opportunities": legal,
        "num_good_source_opportunities": good_opportunities,
    }
    for name in ("immediate", "quarantine", "release", "combined"):
        source_buckets = [_bucket(candidate.get("governance", candidate), name)
                          for _, candidate in candidates]
        count = sum(int(bucket["count"]) for bucket in source_buckets)
        good = sum(int(bucket["good_count"]) for bucket in source_buckets)
        bad = sum(int(bucket["bad_count"]) for bucket in source_buckets)
        sequence_count = sum(int(bucket["count"]) > 0 for bucket in source_buckets)
        governance[name] = {
            "count": count,
            "good_count": good,
            "bad_count": bad,
            "precision": float(good / count) if count else 0.0,
            "bad_rate": float(bad / count) if count else 0.0,
            "coverage": float(count / legal) if legal else 0.0,
            "good_recall": float(good / good_opportunities) if good_opportunities else 0.0,
            "sequence_count": sequence_count,
            "sequence_coverage": (float(sequence_count / len(candidates))
                                  if candidates else 0.0),
        }
    deltas = [_extract_quarantine_delta(candidate) for _, candidate in candidates]
    weights = [float(candidate.get("num_frames", summary.get("num_frames", 1)))
               for summary, candidate in candidates]
    return {
        "support_threshold": threshold,
        "governance": governance,
        "quarantine_incremental_success_auc_delta": (
            float(np.average(deltas, weights=weights)) if deltas else 0.0),
    }


def aggregate_threshold_candidates(sequence_summaries):
    summaries = list(sequence_summaries)
    if not summaries:
        return []
    threshold_sets = [{_support_threshold(candidate)
                       for candidate in _candidate_collection(summary)}
                      for summary in summaries]
    if any(values != threshold_sets[0] for values in threshold_sets[1:]):
        raise ValueError("Support-threshold candidates differ across sequences.")
    return [_aggregate_candidate_metrics(summaries, threshold)
            for threshold in sorted(threshold_sets[0])]


def _stable_threshold_key(value):
    return "none" if value is None else format(float(value), ".17g")


def _hash_group(sequence, namespace, groups=5):
    payload = "{}:{}".format(namespace, sequence).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16) % int(groups)


def deterministic_threshold_stability(sequence_summaries, gate_config=None,
                                      internal=False, seed=0, samples=2000,
                                      hash_namespace="rmg-stage1-v2-q5-threshold-stability"):
    """Bootstrap sequence-level threshold selection and run five hash-group LOO checks."""
    summaries = sorted(list(sequence_summaries),
                       key=lambda item: str(item.get("sequence", "")))
    if not summaries:
        raise ValueError("Threshold stability requires sequence summaries.")
    names = [_sequence(summary.get("sequence")) for summary in summaries]
    if len(names) != len(set(names)):
        raise ValueError("Threshold stability requires unique sequence names.")

    def select(items):
        result = select_smallest_passing_support_threshold(
            aggregate_threshold_candidates(items), gate_config, internal)
        return result["selected_support_threshold"]

    selected = select(summaries)
    rng = np.random.RandomState(int(seed))
    bootstrap = []
    for _ in range(int(samples)):
        indices = rng.randint(0, len(summaries), size=len(summaries))
        bootstrap.append(select([summaries[index] for index in indices]))
    counts = Counter(_stable_threshold_key(value) for value in bootstrap)
    if counts:
        modal_key = sorted(counts, key=lambda key: (-counts[key], key))[0]
        modal = None if modal_key == "none" else float(modal_key)
        modal_fraction = float(counts[modal_key] / len(bootstrap))
    else:
        modal, modal_fraction = None, 0.0
    assignments = {name: _hash_group(name, hash_namespace) for name in names}
    leave_one_group_out = []
    for group in range(5):
        retained = [summary for summary, name in zip(summaries, names)
                    if assignments[name] != group]
        threshold = select(retained) if retained else None
        leave_one_group_out.append({
            "group": group,
            "num_retained_sequences": len(retained),
            "selected_support_threshold": threshold,
            "same_as_full": threshold == selected,
        })
    return _hashed({
        "schema_version": SCHEMA_VERSION,
        "selected_support_threshold": selected,
        "bootstrap": {
            "seed": int(seed),
            "samples": int(samples),
            "modal_support_threshold": modal,
            "modal_match_fraction": modal_fraction,
            "selection_counts": dict(sorted(counts.items())),
        },
        "hash_groups": {
            "namespace": hash_namespace,
            "count": 5,
            "assignments": dict(sorted(assignments.items())),
        },
        "leave_one_group_out": leave_one_group_out,
        "leave_one_group_out_same_threshold_count": sum(
            item["same_as_full"] for item in leave_one_group_out),
    })


def threshold_stability(sequence_summaries, **kwargs):
    return deterministic_threshold_stability(sequence_summaries, **kwargs)


def _validate_arm_metrics(metrics, sequence, arm):
    frames = int(metrics.get("num_frames", 0))
    if frames <= 0:
        raise ValueError("{} / {} has no frames.".format(sequence, arm))
    for key in METRIC_KEYS:
        _finite_float(metrics.get(key), "{}.{}".format(arm, key))
    return frames


def _paired_delta(summaries, variant, baseline, metric, seed, samples):
    values = [summary["arms"][variant][metric] - summary["arms"][baseline][metric]
              for summary in summaries]
    weights = [summary["arms"][baseline]["num_frames"] for summary in summaries]
    return deterministic_weighted_bootstrap_ci(
        values, weights, seed=seed, samples=samples)


def _is_clean(summary):
    for source in (summary, summary.get("metadata", {})):
        for key in ("clean_subset", "is_clean", "clean"):
            if key in source:
                return bool(source[key])
    return float(summary["arms"]["static"]["success_auc"]) >= 0.60


def _worsened(summaries, baseline="static", threshold=-0.01):
    deltas = [summary["arms"]["rmg_q"]["success_auc"]
              - summary["arms"][baseline]["success_auc"] for summary in summaries]
    count = sum(delta < float(threshold) for delta in deltas)
    return {
        "definition": "rmg_q-minus-{} success AUC < {}".format(baseline, float(threshold)),
        "threshold": float(threshold),
        "num_sequences": len(deltas),
        "num_worsened": int(count),
        "fraction": float(count / len(deltas)) if deltas else 0.0,
    }


def _aggregate_governance_summaries(summaries):
    metrics = [summary.get("governance", {}) for summary in summaries]
    legal = sum(int(item.get("num_legal_source_opportunities", 0)) for item in metrics)
    good_opportunities = sum(int(item.get("num_good_source_opportunities", 0))
                             for item in metrics)
    result = {
        "num_legal_source_opportunities": legal,
        "num_good_source_opportunities": good_opportunities,
        "num_legal_sequences": sum(int(item.get("num_legal_source_opportunities", 0)) > 0
                                   for item in metrics),
    }
    for name in ("immediate", "quarantine", "release", "combined"):
        buckets = [_bucket(item, name) for item in metrics]
        count = sum(int(bucket["count"]) for bucket in buckets)
        good = sum(int(bucket["good_count"]) for bucket in buckets)
        bad = sum(int(bucket["bad_count"]) for bucket in buckets)
        sequence_count = sum(int(bucket["count"]) > 0 for bucket in buckets)
        result[name] = {
            "count": count,
            "good_count": good,
            "bad_count": bad,
            "precision": float(good / count) if count else 0.0,
            "bad_rate": float(bad / count) if count else 0.0,
            "coverage": float(count / legal) if legal else 0.0,
            "good_recall": float(good / good_opportunities) if good_opportunities else 0.0,
            "sequence_count": sequence_count,
            "sequence_coverage": (float(sequence_count / len(metrics))
                                  if metrics else 0.0),
        }
    reasons = Counter()
    reasons_by_action = defaultdict(Counter)
    for item in metrics:
        reasons.update(item.get("reasons", {}))
        for action, counts in item.get("reasons_by_action", {}).items():
            reasons_by_action[action].update(counts)
    result["reasons"] = dict(sorted(reasons.items()))
    result["reasons_by_action"] = {
        action: dict(sorted(counts.items()))
        for action, counts in sorted(reasons_by_action.items())
    }
    return result


def aggregate_sequence_summaries(sequence_summaries, bootstrap_seed=0,
                                 bootstrap_samples=2000):
    """Aggregate all five arms with paired frame-weighted sequence bootstrap."""
    summaries = sorted(list(sequence_summaries),
                       key=lambda item: str(item.get("sequence", "")))
    if not summaries:
        raise ValueError("At least one sequence summary is required.")
    names = [_sequence(summary.get("sequence")) for summary in summaries]
    if len(names) != len(set(names)):
        raise ValueError("Sequence summaries must have unique names.")
    for summary, sequence in zip(summaries, names):
        if set(summary.get("arms", {})) != set(ARM_NAMES):
            raise ValueError("Sequence {!r} must contain exactly the five approved arms."
                             .format(sequence))
        frame_counts = [_validate_arm_metrics(summary["arms"][arm], sequence, arm)
                        for arm in ARM_NAMES]
        if len(set(frame_counts)) != 1:
            raise ValueError("Arm frame counts differ for sequence {!r}.".format(sequence))
    arms = {}
    for arm in ARM_NAMES:
        available = [summary["arms"][arm] for summary in summaries]
        weights = [item["num_frames"] for item in available]
        arms[arm] = {
            "num_sequences": len(available),
            "num_frames": int(sum(weights)),
            **{key: float(np.average([item[key] for item in available], weights=weights))
               for key in METRIC_KEYS},
            "aggregation": "frame_weighted_sequence_metrics",
        }
    baselines = ("static", "periodic_pred", "rmg_q_no_quarantine", "confidence_e050")
    comparisons = {}
    for comparison_index, baseline in enumerate(baselines):
        comparisons["rmg_q_vs_{}".format(baseline)] = {
            metric: _paired_delta(
                summaries, "rmg_q", baseline, metric,
                bootstrap_seed + comparison_index * len(METRIC_KEYS) + metric_index,
                bootstrap_samples,
            )
            for metric_index, metric in enumerate(METRIC_KEYS)
        }
    clean = [summary for summary in summaries if _is_clean(summary)]
    clean_delta = (_paired_delta(
        clean, "rmg_q", "static", "success_auc",
        bootstrap_seed + len(baselines) * len(METRIC_KEYS), bootstrap_samples)
        if clean else deterministic_weighted_bootstrap_ci(
            [], [], seed=bootstrap_seed + len(baselines) * len(METRIC_KEYS),
            samples=bootstrap_samples))
    governance = _aggregate_governance_summaries(summaries)
    governance["quarantine_incremental_success_auc_delta"] = comparisons[
        "rmg_q_vs_rmg_q_no_quarantine"]["success_auc"]["mean"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "num_sequences": len(summaries),
        "expected_arms": list(ARM_NAMES),
        "arms": arms,
        "frame_weighted_paired_deltas": comparisons,
        "governance": governance,
        "worsened_vs_static": _worsened(summaries),
        "clean_subset_preservation": {
            "definition": "explicit clean flag when present, otherwise static success AUC >= 0.60",
            "num_sequences": len(clean),
            "rmg_q_vs_static_success_auc": clean_delta,
            "worsened_vs_static": _worsened(clean) if clean else _worsened([]),
        },
    }
    return _hashed(result)


def evaluate_final_gate(aggregate, gate_config=None, action_gate_config=None,
                        internal=False):
    """Evaluate the locked tracking, preservation, and action-quality gate."""
    config = dict(DEFAULT_FINAL_GATE)
    if gate_config:
        config.update(gate_config)
    comparisons = aggregate.get("frame_weighted_paired_deltas", {})

    def auc(baseline):
        return comparisons.get("rmg_q_vs_{}".format(baseline), {}).get(
            "success_auc", {})

    static = auc("static")
    periodic = auc("periodic_pred")
    no_quarantine = auc("rmg_q_no_quarantine")
    confidence = auc("confidence_e050")
    worsened = aggregate.get("worsened_vs_static", {})
    clean = aggregate.get("clean_subset_preservation", {})
    clean_delta = clean.get("rmg_q_vs_static_success_auc", {})
    clean_worsened = clean.get("worsened_vs_static", {})
    checks = {
        "rmg_q_vs_static_auc_delta_mean": _check(
            static.get("mean", 0.0),
            static.get("mean", 0.0) >= config["rmg_q_static_auc_delta_min"],
            minimum=config["rmg_q_static_auc_delta_min"]),
        "rmg_q_vs_periodic_auc_delta_mean": _check(
            periodic.get("mean", 0.0),
            periodic.get("mean", 0.0) >= config["rmg_q_periodic_auc_delta_min"],
            minimum=config["rmg_q_periodic_auc_delta_min"]),
        "rmg_q_vs_periodic_auc_delta_ci_low": _check(
            periodic.get("low", 0.0),
            periodic.get("low", 0.0) > config[
                "rmg_q_periodic_auc_ci_low_exclusive_min"],
            exclusive_minimum=config["rmg_q_periodic_auc_ci_low_exclusive_min"]),
        "worsened_sequence_fraction": _check(
            worsened.get("fraction", 0.0),
            worsened.get("fraction", 0.0) <= config["worsened_fraction_max"],
            maximum=config["worsened_fraction_max"]),
        "clean_sequence_coverage": _check(
            clean.get("num_sequences", 0), clean.get("num_sequences", 0) > 0,
            exclusive_minimum=0),
        "clean_auc_delta_mean": _check(
            clean_delta.get("mean", 0.0),
            clean_delta.get("mean", 0.0) >= config["clean_auc_delta_min"],
            minimum=config["clean_auc_delta_min"]),
        "clean_worsened_sequence_fraction": _check(
            clean_worsened.get("fraction", 0.0),
            clean_worsened.get("fraction", 0.0) <= config[
                "clean_worsened_fraction_max"],
            maximum=config["clean_worsened_fraction_max"]),
        "rmg_q_vs_no_quarantine_auc_delta_mean": _check(
            no_quarantine.get("mean", 0.0),
            no_quarantine.get("mean", 0.0) > config[
                "rmg_q_no_quarantine_auc_delta_exclusive_min"],
            exclusive_minimum=config[
                "rmg_q_no_quarantine_auc_delta_exclusive_min"]),
        "rmg_q_vs_confidence_e050_auc_delta_mean": _check(
            confidence.get("mean", 0.0),
            confidence.get("mean", 0.0) > config[
                "rmg_q_confidence_e050_auc_delta_exclusive_min"],
            exclusive_minimum=config[
                "rmg_q_confidence_e050_auc_delta_exclusive_min"]),
    }
    action = evaluate_action_quality(
        aggregate, gate_config=action_gate_config, internal=internal)
    result = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "checks": checks,
        "action_quality": action,
        "pass": all(item["pass"] for item in checks.values()) and action["pass"],
    }
    return _hashed(result)


def compute_stage1_quarantine_v2_metrics(sequence_summaries, bootstrap_seed=0,
                                         bootstrap_samples=2000,
                                         gate_config=None,
                                         action_gate_config=None,
                                         internal=False):
    aggregate = aggregate_sequence_summaries(
        sequence_summaries, bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples)
    gate = evaluate_final_gate(
        aggregate, gate_config=gate_config,
        action_gate_config=action_gate_config, internal=internal)
    return aggregate, gate
