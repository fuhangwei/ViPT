"""Pure metrics and gates for the approved RMG-QH Stage 1 v3a protocol."""

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


SCHEMA_VERSION = "rmg-stage1-v3a-qonly"
ARM_NAMES = ("static", "periodic_pred", "rmg_qh_qonly", "rmg_qh_noq")
SUPPORT_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
METRIC_KEYS = (
    "success_auc", "mean_iou", "precision20", "normalized_precision",
    "normalized_precision_at_0_2",
)
GOOD_IOU = 0.7
BAD_IOU = 0.1
DEFAULT_ACTION_GATE = {
    "release_coverage_min": 0.10,
    "combined_coverage_min": 0.10,
    "release_bad_rate_max": 0.05,
    "combined_bad_rate_max": 0.05,
    "release_count_min": 100,
    "release_sequences_min": 20,
    "old_immediate_eligible_min": 1,
    "old_immediate_accounted_fraction_min": 1.0,
}
INTERNAL_ACTION_GATE = {
    **DEFAULT_ACTION_GATE,
    "release_count_min": 50,
    "release_sequences_min": 10,
}
DEFAULT_FINAL_GATE = {
    "qonly_static_auc_delta_exclusive_min": 0.0,
    "qonly_periodic_auc_delta_exclusive_min": 0.0,
    "qonly_noq_auc_delta_exclusive_min": 0.0,
    "release_future_gain_exclusive_min": 0.0,
    "worsened_fraction_max": 0.20,
    "clean_auc_delta_min": -0.002,
    "clean_worsened_fraction_max": 0.10,
    "clean_subset_nonempty": True,
}
_FORBIDDEN_ONLINE_KEYS = frozenset({
    "iou", "evaluation_iou", "candidate_iou", "source_candidate_iou",
    "release_frame_iou", "gt", "gt_xywh", "ground_truth", "groundtruth",
})


def _hashed(payload):
    result = dict(payload)
    result["content_hash"] = canonical_hash(result)
    return result


def canonical_content_hash(value):
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != "content_hash"}
    return canonical_hash(value)


def content_hash_matches(value):
    return (isinstance(value, dict) and isinstance(value.get("content_hash"), str)
            and value["content_hash"] == canonical_content_hash(value))


def _finite_float(value, name):
    if isinstance(value, bool):
        raise ValueError("{} must be a finite number.".format(name))
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} must be a finite number.".format(name)) from exc
    if not math.isfinite(number):
        raise ValueError("{} must be a finite number.".format(name))
    return number


def _frame_idx(value):
    number = _finite_float(value, "frame_idx")
    if number < 0 or number != int(number):
        raise ValueError("frame_idx must be a non-negative integer.")
    return int(number)


def _sequence(value):
    if value is None or str(value) == "":
        raise ValueError("sequence must be non-empty.")
    return str(value)


def _identifier(value, name):
    if isinstance(value, (dict, list, tuple, set)) or value in (None, ""):
        raise ValueError("{} must be a non-empty scalar.".format(name))
    return type(value).__name__, str(value)


def _is_forbidden_online_key(key):
    normalized = str(key).lower().replace("-", "_")
    return (normalized in _FORBIDDEN_ONLINE_KEYS or normalized.startswith("gt_")
            or normalized.endswith("_gt") or "groundtruth" in normalized
            or "ground_truth" in normalized or normalized.endswith("_candidate_iou")
            or normalized.endswith("_evaluation_iou"))


def reject_evaluation_data(value, location="row"):
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_forbidden_online_key(key):
                raise ValueError("Online trace contains forbidden GT/IoU key at {}.{}."
                                 .format(location, key))
            reject_evaluation_data(item, "{}.{}".format(location, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_evaluation_data(item, "{}[{}]".format(location, index))


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
    grouped = defaultdict(list)
    for sequence, frame in keys:
        grouped[sequence].append(frame)
    for sequence, frames in grouped.items():
        ordered = sorted(frames)
        if ordered != list(range(ordered[0], ordered[-1] + 1)):
            raise ValueError("Noncontiguous {} frames for sequence {!r}."
                             .format(kind, sequence))


def join_frame_traces(online_rows, frame_labels):
    online = list(online_rows)
    labels = list(frame_labels)
    for index, row in enumerate(online):
        reject_evaluation_data(row, "online_rows[{}]".format(index))
    online_by_key = _unique_index(online, _frame_key, "online frame")
    labels_by_key = _unique_index(labels, _frame_key, "frame label")
    _validate_contiguous_frames(online_by_key, "online")
    _validate_contiguous_frames(labels_by_key, "label")
    if set(online_by_key) != set(labels_by_key):
        raise ValueError("Frame labels do not match online trace.")
    joined = []
    for key in sorted(online_by_key):
        row, label = online_by_key[key], labels_by_key[key]
        if "pred_xywh" not in row or "gt_xywh" not in label:
            raise ValueError("Joined frame requires pred_xywh and gt_xywh.")
        joined.append({"sequence": key[0], "frame_idx": key[1],
                       "pred_xywh": row["pred_xywh"], "gt_xywh": label["gt_xywh"]})
    return joined


def tracking_metrics_from_online_trace(online_rows, frame_labels):
    joined = join_frame_traces(online_rows, frame_labels)
    return tracking_metrics([row["pred_xywh"] for row in joined],
                            [row["gt_xywh"] for row in joined])


def tracking_metrics_by_sequence(online_rows, frame_labels):
    grouped = defaultdict(list)
    for row in join_frame_traces(online_rows, frame_labels):
        grouped[row["sequence"]].append(row)
    return {sequence: tracking_metrics([row["pred_xywh"] for row in rows],
                                       [row["gt_xywh"] for row in rows])
            for sequence, rows in sorted(grouped.items())}


def _event_key(row):
    if "event_id" not in row:
        raise ValueError("Event row requires event_id.")
    return _identifier(row["event_id"], "event_id")


def join_event_traces(online_events, event_labels):
    online = list(online_events)
    labels = list(event_labels)
    for index, row in enumerate(online):
        reject_evaluation_data(row, "online_events[{}]".format(index))
        if "legal_source_opportunity" not in row:
            raise ValueError("Online event requires explicit legal_source_opportunity.")
        if not isinstance(row["legal_source_opportunity"], bool):
            raise TypeError("legal_source_opportunity must be bool.")
    online_by_key = _unique_index(online, _event_key, "online event")
    labels_by_key = _unique_index(labels, _event_key, "event label")
    if set(online_by_key) != set(labels_by_key):
        raise ValueError("Event labels do not match online trace.")
    joined = []
    for key in sorted(online_by_key):
        label = labels_by_key[key]
        if "source_candidate_iou" not in label:
            raise ValueError("Event label is missing source_candidate_iou.")
        source_iou = _finite_float(label["source_candidate_iou"], "source_candidate_iou")
        if not 0.0 <= source_iou <= 1.0:
            raise ValueError("source_candidate_iou must be in [0, 1].")
        joined.append({"online": online_by_key[key], "label": label,
                       "source_candidate_iou": source_iou})
    return joined


def _quality_bucket(items, legal_count, good_opportunities, legal_sequences,
                    good_iou, bad_iou):
    count = len(items)
    good = sum(item["source_candidate_iou"] >= good_iou for item in items)
    bad = sum(item["source_candidate_iou"] <= bad_iou for item in items)
    sequences = {item["sequence"] for item in items}
    return {
        "count": int(count), "good_count": int(good), "bad_count": int(bad),
        "precision": float(good / count) if count else 0.0,
        "bad_rate": float(bad / count) if count else 0.0,
        "coverage": float(count / legal_count) if legal_count else 0.0,
        "good_recall": float(good / good_opportunities) if good_opportunities else 0.0,
        "sequence_count": len(sequences),
        "sequence_coverage": (float(len(sequences) / len(legal_sequences))
                              if legal_sequences else 0.0),
    }


def governance_metrics(online_events, event_labels, good_iou=GOOD_IOU, bad_iou=BAD_IOU):
    joined = join_event_traces(online_events, event_labels)
    legal = []
    admission, quarantines, releases, discards = [], [], [], []
    old_immediate_eligible = []
    old_immediate_accounted = []
    reasons = Counter()
    reasons_by_action = defaultdict(Counter)
    for item in joined:
        row, label = item["online"], item["label"]
        sequence = _sequence(row.get("sequence", label.get("sequence")))
        enriched = {**item, "sequence": sequence}
        action = str(row.get("action", "")).strip().lower()
        if action not in {"skip", "quarantine", "release", "discard"}:
            raise ValueError("Unsupported v3a governance action: {!r}.".format(action))
        if row.get("immediate_write") not in (None, False):
            raise ValueError("v3a forbids immediate writes.")
        legal_flag = row["legal_source_opportunity"]
        if legal_flag:
            legal.append(enriched)
        admitted = bool(row.get("admitted", row.get("quarantined", False)))
        if action == "quarantine":
            admitted = True
        if admitted and legal_flag:
            admission.append(enriched)
        entropy = (None if row.get("admission_entropy") is None else
                   _finite_float(row["admission_entropy"], "admission_entropy"))
        if legal_flag and entropy is not None and entropy <= 0.45:
            old_immediate_eligible.append(enriched)
            if admitted or (action == "skip" and row.get("reason") == "quarantine_slot_occupied"):
                old_immediate_accounted.append(enriched)
        if bool(row.get("quarantined", False)) and legal_flag:
            quarantines.append(enriched)
        if action == "release" and legal_flag:
            releases.append(enriched)
        if action == "discard" and legal_flag:
            discards.append(enriched)
        reason = str(row.get("reason", "unspecified"))
        reasons[reason] += 1
        reasons_by_action[action][reason] += 1
    legal_count = len(legal)
    good_opportunities = sum(item["source_candidate_iou"] >= good_iou for item in legal)
    legal_sequences = {item["sequence"] for item in legal}
    buckets = {
        "admission": admission, "quarantine": quarantines,
        "release": releases, "discard": discards,
    }
    quality = {name: _quality_bucket(items, legal_count, good_opportunities,
                                     legal_sequences, good_iou, bad_iou)
               for name, items in buckets.items()}
    quality["combined"] = dict(quality["release"])
    result = {
        "schema_version": SCHEMA_VERSION, "good_iou": float(good_iou),
        "bad_iou": float(bad_iou), "num_events": len(joined),
        "num_legal_source_opportunities": legal_count,
        "num_good_source_opportunities": int(good_opportunities),
        "num_legal_sequences": len(legal_sequences), **quality,
        "reasons": dict(sorted(reasons.items())),
        "reasons_by_action": {name: dict(sorted(counts.items()))
                              for name, counts in sorted(reasons_by_action.items())},
        "immediate_writes": 0, "admitted_candidates": quality["admission"]["count"],
        "old_immediate_eligible_count": len(old_immediate_eligible),
        "old_immediate_accounted_count": len(old_immediate_accounted),
        "old_immediate_accounted_fraction": (
            float(len(old_immediate_accounted) / len(old_immediate_eligible))
            if old_immediate_eligible else 0.0),
        "quarantines": quality["quarantine"]["count"],
        "releases": quality["release"]["count"],
        "discards": quality["discard"]["count"],
        "combined_writes": quality["release"]["count"],
    }
    for name, bucket in quality.items():
        for key, value in bucket.items():
            result["{}_{}".format(name, key)] = value
    if result["combined"] != result["release"]:
        raise AssertionError("Combined writes must exactly equal release in v3a.")
    return _hashed(result)


def _bucket(metrics, name):
    nested = metrics.get(name)
    if not isinstance(nested, dict):
        nested = {}
    keys = ("count", "good_count", "bad_count", "precision", "bad_rate", "coverage",
            "good_recall", "sequence_count", "sequence_coverage")
    return {key: nested.get(key, metrics.get("{}_{}".format(name, key), 0)) for key in keys}


def _check(value, passed, **limits):
    return {"value": value, **limits, "pass": bool(passed)}


def evaluate_action_quality(candidate, gate_config=None, internal=False):
    config = dict(INTERNAL_ACTION_GATE if internal else DEFAULT_ACTION_GATE)
    if gate_config:
        config.update(gate_config)
    governance = candidate.get("governance", candidate)
    release = _bucket(governance, "release")
    combined = _bucket(governance, "combined")
    if release != combined:
        raise ValueError("Combined metrics must exactly equal release metrics in v3a.")
    checks = {
        "release_coverage": _check(release["coverage"], release["coverage"] >= config["release_coverage_min"], minimum=config["release_coverage_min"]),
        "combined_coverage": _check(combined["coverage"], combined["coverage"] >= config["combined_coverage_min"], minimum=config["combined_coverage_min"]),
        "release_bad_rate": _check(release["bad_rate"], release["bad_rate"] <= config["release_bad_rate_max"], maximum=config["release_bad_rate_max"]),
        "combined_bad_rate": _check(combined["bad_rate"], combined["bad_rate"] <= config["combined_bad_rate_max"], maximum=config["combined_bad_rate_max"]),
        "release_count": _check(release["count"], release["count"] >= config["release_count_min"], minimum=config["release_count_min"]),
        "release_sequences": _check(release["sequence_count"], release["sequence_count"] >= config["release_sequences_min"], minimum=config["release_sequences_min"]),
        "old_immediate_eligible": _check(
            governance.get("old_immediate_eligible_count", 0),
            governance.get("old_immediate_eligible_count", 0)
            >= config["old_immediate_eligible_min"],
            minimum=config["old_immediate_eligible_min"]),
        "old_immediate_accounted_fraction": _check(
            governance.get("old_immediate_accounted_fraction", 0.0),
            governance.get("old_immediate_accounted_fraction", 0.0)
            >= config["old_immediate_accounted_fraction_min"],
            minimum=config["old_immediate_accounted_fraction_min"]),
    }
    return _hashed({"schema_version": SCHEMA_VERSION, "config": config,
                    "checks": checks, "pass": all(item["pass"] for item in checks.values())})


def _comparison(aggregate, baseline):
    return aggregate.get("frame_weighted_paired_deltas", {}).get(
        "rmg_qh_qonly_vs_{}".format(baseline), {}).get("success_auc", {})


def evaluate_final_gate(aggregate, gate_config=None, action_gate_config=None, internal=False):
    config = dict(DEFAULT_FINAL_GATE)
    if gate_config:
        config.update(gate_config)
    static = _comparison(aggregate, "static")
    periodic = _comparison(aggregate, "periodic_pred")
    noq = _comparison(aggregate, "rmg_qh_noq")
    worsened = aggregate.get("worsened_vs_static", {})
    clean = aggregate.get("clean_subset_preservation", {})
    clean_delta = clean.get("rmg_qh_qonly_vs_static_success_auc", {})
    clean_worsened = clean.get("worsened_vs_static", {})
    future_gain = aggregate.get("protocol_level_release_future_gain", noq.get("mean", 0.0))
    checks = {
        "qonly_vs_static_auc_delta": _check(static.get("mean", 0.0), static.get("mean", 0.0) > config["qonly_static_auc_delta_exclusive_min"], exclusive_minimum=config["qonly_static_auc_delta_exclusive_min"]),
        "qonly_vs_periodic_auc_delta": _check(periodic.get("mean", 0.0), periodic.get("mean", 0.0) > config["qonly_periodic_auc_delta_exclusive_min"], exclusive_minimum=config["qonly_periodic_auc_delta_exclusive_min"]),
        "qonly_vs_noq_auc_delta": _check(noq.get("mean", 0.0), noq.get("mean", 0.0) > config["qonly_noq_auc_delta_exclusive_min"], exclusive_minimum=config["qonly_noq_auc_delta_exclusive_min"]),
        "protocol_level_release_future_gain": _check(future_gain, future_gain > config["release_future_gain_exclusive_min"], exclusive_minimum=config["release_future_gain_exclusive_min"]),
        "worsened_sequence_fraction": _check(worsened.get("fraction", 0.0), worsened.get("fraction", 0.0) <= config["worsened_fraction_max"], maximum=config["worsened_fraction_max"]),
        "clean_sequence_coverage": _check(clean.get("num_sequences", 0), clean.get("num_sequences", 0) > 0, exclusive_minimum=0),
        "clean_auc_delta_mean": _check(clean_delta.get("mean", 0.0), clean_delta.get("mean", 0.0) >= config["clean_auc_delta_min"], minimum=config["clean_auc_delta_min"]),
        "clean_worsened_sequence_fraction": _check(clean_worsened.get("fraction", 0.0), clean_worsened.get("fraction", 0.0) <= config["clean_worsened_fraction_max"], maximum=config["clean_worsened_fraction_max"]),
    }
    action = evaluate_action_quality(aggregate, action_gate_config, internal)
    return _hashed({"schema_version": SCHEMA_VERSION, "config": config,
                    "checks": checks, "action_quality": action,
                    "pass": all(item["pass"] for item in checks.values()) and action["pass"]})


def _support_threshold(candidate):
    for key in ("support_threshold", "support_iou", "threshold"):
        if key in candidate:
            return _finite_float(candidate[key], key)
    raise ValueError("Threshold candidate is missing support_threshold.")


def select_smallest_passing_support_threshold(candidates, gate_config=None,
                                              final_gate_config=None, internal=False):
    candidates = list(candidates)
    thresholds = [_support_threshold(candidate) for candidate in candidates]
    if sorted(thresholds) != list(SUPPORT_THRESHOLDS) or len(set(thresholds)) != 5:
        raise ValueError("Selection requires exactly the five frozen support thresholds.")
    audited = []
    for candidate in candidates:
        threshold = _support_threshold(candidate)
        if "frame_weighted_paired_deltas" in candidate:
            gate = evaluate_final_gate(candidate, final_gate_config, gate_config, internal)
        else:
            gate = evaluate_action_quality(candidate, gate_config, internal)
        audited.append({"support_threshold": threshold, "gate": gate})
    audited.sort(key=lambda item: item["support_threshold"])
    passing = [item for item in audited if item["gate"]["pass"]]
    selected = passing[0]["support_threshold"] if passing else None
    return _hashed({"schema_version": SCHEMA_VERSION,
                    "selection_rule": "smallest_support_threshold_passing_all_gates",
                    "selected_support_threshold": selected, "candidates": audited})


select_support_threshold = select_smallest_passing_support_threshold


def _candidate_collection(summary):
    collection = summary.get("threshold_candidates", summary.get("candidates"))
    if collection is None:
        raise ValueError("Sequence summary is missing threshold_candidates.")
    if isinstance(collection, dict):
        return [dict(value, support_threshold=key) if "support_threshold" not in value else value
                for key, value in collection.items()]
    return list(collection)


def _aggregate_governance_items(items):
    legal = sum(int(item.get("num_legal_source_opportunities", 0)) for item in items)
    good_opportunities = sum(int(item.get("num_good_source_opportunities", 0)) for item in items)
    old_immediate_eligible = sum(
        int(item.get("old_immediate_eligible_count", 0)) for item in items)
    old_immediate_accounted = sum(
        int(item.get("old_immediate_accounted_count", 0)) for item in items)
    result = {"num_legal_source_opportunities": legal,
              "num_good_source_opportunities": good_opportunities,
              "num_legal_sequences": sum(int(item.get("num_legal_source_opportunities", 0)) > 0 for item in items),
              "old_immediate_eligible_count": old_immediate_eligible,
              "old_immediate_accounted_count": old_immediate_accounted,
              "old_immediate_accounted_fraction": (
                  float(old_immediate_accounted / old_immediate_eligible)
                  if old_immediate_eligible else 0.0)}
    for name in ("admission", "quarantine", "release", "discard"):
        buckets = [_bucket(item, name) for item in items]
        count = sum(int(bucket["count"]) for bucket in buckets)
        good = sum(int(bucket["good_count"]) for bucket in buckets)
        bad = sum(int(bucket["bad_count"]) for bucket in buckets)
        sequence_count = sum(int(bucket["count"]) > 0 for bucket in buckets)
        result[name] = {"count": count, "good_count": good, "bad_count": bad,
                        "precision": float(good / count) if count else 0.0,
                        "bad_rate": float(bad / count) if count else 0.0,
                        "coverage": float(count / legal) if legal else 0.0,
                        "good_recall": float(good / good_opportunities) if good_opportunities else 0.0,
                        "sequence_count": sequence_count,
                        "sequence_coverage": float(sequence_count / len(items)) if items else 0.0}
    result["combined"] = dict(result["release"])
    return result


def _aggregate_candidate_metrics(sequence_summaries, threshold):
    matched = []
    for summary in sequence_summaries:
        candidates = [item for item in _candidate_collection(summary)
                      if _support_threshold(item) == threshold]
        if len(candidates) != 1:
            raise ValueError("Each sequence must contain exactly one candidate per threshold.")
        matched.append((summary, candidates[0]))
    governance = _aggregate_governance_items([
        candidate.get("governance", candidate) for _, candidate in matched])
    weights = [float(candidate.get("num_frames", summary.get("num_frames", 1)))
               for summary, candidate in matched]
    result = {"support_threshold": threshold, "governance": governance}
    comparisons = {}
    for baseline in ("static", "periodic_pred", "rmg_qh_noq"):
        values = [float(candidate.get("auc_deltas", {}).get(baseline, 0.0))
                  for _, candidate in matched]
        mean = float(np.average(values, weights=weights)) if values else 0.0
        comparisons["rmg_qh_qonly_vs_{}".format(baseline)] = {
            "success_auc": {"mean": mean, "low": mean, "high": mean,
                            "samples": len(values), "seed": 0}}
    result["frame_weighted_paired_deltas"] = comparisons
    result["protocol_level_release_future_gain"] = comparisons[
        "rmg_qh_qonly_vs_rmg_qh_noq"]["success_auc"]["mean"]
    worsened = [float(candidate.get("auc_deltas", {}).get("static", 0.0)) < -0.01
                for _, candidate in matched]
    result["worsened_vs_static"] = {"fraction": float(sum(worsened) / len(worsened)) if worsened else 0.0}
    clean = [(summary, candidate) for summary, candidate in matched
             if bool(candidate.get(
                 "clean_subset", summary.get("clean_subset", summary.get("is_clean", True))))]
    clean_deltas = [float(candidate.get("auc_deltas", {}).get("static", 0.0)) for _, candidate in clean]
    clean_mean = float(np.average(clean_deltas, weights=[float(candidate.get("num_frames", summary.get("num_frames", 1))) for summary, candidate in clean])) if clean else 0.0
    result["clean_subset_preservation"] = {
        "num_sequences": len(clean),
        "rmg_qh_qonly_vs_static_success_auc": {"mean": clean_mean},
        "worsened_vs_static": {"fraction": float(sum(value < -0.01 for value in clean_deltas) / len(clean_deltas)) if clean_deltas else 0.0},
    }
    return result


def aggregate_threshold_candidates(sequence_summaries):
    summaries = list(sequence_summaries)
    if not summaries:
        raise ValueError("Threshold aggregation requires sequence summaries.")
    threshold_sets = [{_support_threshold(candidate) for candidate in _candidate_collection(summary)}
                      for summary in summaries]
    if any(values != set(SUPPORT_THRESHOLDS) for values in threshold_sets):
        raise ValueError("Every sequence must contain the exact frozen threshold grid.")
    return [_aggregate_candidate_metrics(summaries, threshold)
            for threshold in SUPPORT_THRESHOLDS]


def _stable_threshold_key(value):
    return "none" if value is None else format(float(value), ".17g")


def _hash_group(sequence, namespace, groups=5):
    payload = "{}:{}".format(namespace, sequence).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16) % int(groups)


def deterministic_threshold_stability(sequence_summaries, gate_config=None,
                                      final_gate_config=None, internal=False,
                                      seed=0, samples=2000,
                                      hash_namespace="rmg-stage1-v3a-qonly-threshold-stability"):
    summaries = sorted(list(sequence_summaries), key=lambda item: _sequence(item.get("sequence")))
    if not summaries:
        raise ValueError("Threshold stability requires sequence summaries.")
    names = [_sequence(item.get("sequence")) for item in summaries]
    if len(names) != len(set(names)):
        raise ValueError("Threshold stability requires unique sequence names.")

    def select(items):
        return select_smallest_passing_support_threshold(
            aggregate_threshold_candidates(items), gate_config, final_gate_config,
            internal)["selected_support_threshold"]

    selected = select(summaries)
    rng = np.random.RandomState(int(seed))
    bootstrap = []
    for _ in range(int(samples)):
        indices = rng.randint(0, len(summaries), size=len(summaries))
        bootstrap.append(select([summaries[index] for index in indices]))
    counts = Counter(_stable_threshold_key(value) for value in bootstrap)
    modal_key = sorted(counts, key=lambda key: (-counts[key], key))[0] if counts else "none"
    modal = None if modal_key == "none" else float(modal_key)
    modal_fraction = float(counts[modal_key] / len(bootstrap)) if bootstrap else 0.0
    match_fraction = float(sum(value == selected for value in bootstrap) / len(bootstrap)) if bootstrap else 0.0
    non_none_fraction = float(sum(value is not None for value in bootstrap) / len(bootstrap)) if bootstrap else 0.0
    assignments = {name: _hash_group(name, hash_namespace, 5) for name in names}
    logo = []
    for group in range(5):
        retained = [summary for summary, name in zip(summaries, names)
                    if assignments[name] != group]
        threshold = select(retained) if retained else None
        logo.append({"group": group, "num_retained_sequences": len(retained),
                     "selected_support_threshold": threshold,
                     "same_non_none_as_full": selected is not None and threshold == selected})
    same_count = sum(item["same_non_none_as_full"] for item in logo)
    passed = (selected is not None and modal == selected and match_fraction >= 0.70
              and non_none_fraction >= 0.70 and same_count >= 4)
    return _hashed({"schema_version": SCHEMA_VERSION,
                    "selected_support_threshold": selected,
                    "bootstrap": {"seed": int(seed), "samples": int(samples),
                                  "modal_support_threshold": modal,
                                  "modal_match_fraction": modal_fraction,
                                  "selected_threshold_match_fraction": match_fraction,
                                  "non_none_fraction": non_none_fraction,
                                  "selection_counts": dict(sorted(counts.items()))},
                    "hash_groups": {"namespace": hash_namespace, "count": 5,
                                    "assignments": dict(sorted(assignments.items()))},
                    "leave_one_group_out": logo,
                    "leave_one_group_out_same_non_none_threshold_count": same_count,
                    "pass": passed})


threshold_stability = deterministic_threshold_stability


def _validate_arm_metrics(metrics, sequence, arm):
    frames = int(metrics.get("num_frames", 0))
    if frames <= 0:
        raise ValueError("{} / {} has no frames.".format(sequence, arm))
    for key in METRIC_KEYS:
        _finite_float(metrics.get(key), "{}.{}".format(arm, key))
    return frames


def _paired_delta(summaries, baseline, metric, seed, samples):
    values = [summary["arms"]["rmg_qh_qonly"][metric]
              - summary["arms"][baseline][metric] for summary in summaries]
    weights = [summary["arms"][baseline]["num_frames"] for summary in summaries]
    return deterministic_weighted_bootstrap_ci(values, weights, seed=seed, samples=samples)


def _is_clean(summary):
    for source in (summary, summary.get("metadata", {})):
        for key in ("clean_subset", "is_clean", "clean"):
            if key in source:
                return bool(source[key])
    return float(summary["arms"]["static"]["success_auc"]) >= 0.60


def _worsened(summaries, baseline="static", threshold=-0.01):
    deltas = [summary["arms"]["rmg_qh_qonly"]["success_auc"]
              - summary["arms"][baseline]["success_auc"] for summary in summaries]
    count = sum(delta < float(threshold) for delta in deltas)
    return {"definition": "rmg_qh_qonly-minus-{} success AUC < {}".format(baseline, float(threshold)),
            "threshold": float(threshold), "num_sequences": len(deltas),
            "num_worsened": int(count),
            "fraction": float(count / len(deltas)) if deltas else 0.0}


def aggregate_sequence_summaries(sequence_summaries, bootstrap_seed=0,
                                 bootstrap_samples=2000):
    summaries = sorted(list(sequence_summaries), key=lambda item: _sequence(item.get("sequence")))
    if not summaries:
        raise ValueError("At least one sequence summary is required.")
    names = [_sequence(item.get("sequence")) for item in summaries]
    if len(names) != len(set(names)):
        raise ValueError("Sequence summaries must have unique names.")
    for summary, sequence in zip(summaries, names):
        if set(summary.get("arms", {})) != set(ARM_NAMES):
            raise ValueError("Sequence {!r} must contain exactly the approved arms.".format(sequence))
        counts = [_validate_arm_metrics(summary["arms"][arm], sequence, arm) for arm in ARM_NAMES]
        if len(set(counts)) != 1:
            raise ValueError("Arm frame counts differ for sequence {!r}.".format(sequence))
    arms = {}
    for arm in ARM_NAMES:
        available = [summary["arms"][arm] for summary in summaries]
        weights = [item["num_frames"] for item in available]
        arms[arm] = {"num_sequences": len(available), "num_frames": int(sum(weights)),
                     **{key: float(np.average([item[key] for item in available], weights=weights))
                        for key in METRIC_KEYS},
                     "aggregation": "frame_weighted_sequence_metrics"}
    comparisons = {}
    for comparison_index, baseline in enumerate(("static", "periodic_pred", "rmg_qh_noq")):
        comparisons["rmg_qh_qonly_vs_{}".format(baseline)] = {
            metric: _paired_delta(summaries, baseline, metric,
                                  bootstrap_seed + comparison_index * len(METRIC_KEYS) + metric_index,
                                  bootstrap_samples)
            for metric_index, metric in enumerate(METRIC_KEYS)}
    clean = [summary for summary in summaries if _is_clean(summary)]
    clean_delta = (_paired_delta(clean, "static", "success_auc",
                                  bootstrap_seed + 3 * len(METRIC_KEYS), bootstrap_samples)
                   if clean else deterministic_weighted_bootstrap_ci([], [], seed=bootstrap_seed,
                                                                      samples=bootstrap_samples))
    governance = _aggregate_governance_items([summary.get("governance", {}) for summary in summaries])
    noq_gain = comparisons["rmg_qh_qonly_vs_rmg_qh_noq"]["success_auc"]["mean"]
    result = {"schema_version": SCHEMA_VERSION, "num_sequences": len(summaries),
              "expected_arms": list(ARM_NAMES), "arms": arms,
              "frame_weighted_paired_deltas": comparisons,
              "protocol_level_release_future_gain": noq_gain,
              "combined_write_future_gain": noq_gain,
              "governance": governance, "worsened_vs_static": _worsened(summaries),
              "clean_subset_preservation": {
                  "definition": "explicit clean flag when present, otherwise static success AUC >= 0.60",
                  "num_sequences": len(clean),
                  "rmg_qh_qonly_vs_static_success_auc": clean_delta,
                  "worsened_vs_static": _worsened(clean) if clean else _worsened([]),
              }}
    return _hashed(result)


def compute_stage1_quarantine_v3a_metrics(sequence_summaries, bootstrap_seed=0,
                                          bootstrap_samples=2000,
                                          gate_config=None,
                                          action_gate_config=None,
                                          internal=False):
    aggregate = aggregate_sequence_summaries(sequence_summaries, bootstrap_seed,
                                             bootstrap_samples)
    gate = evaluate_final_gate(aggregate, gate_config, action_gate_config, internal)
    return aggregate, gate
