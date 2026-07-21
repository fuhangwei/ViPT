"""Metrics and locked gate for RMG-Track Stage 1 rule governance."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from analysis.memory_oracle.compute_stage0_metrics import (
        canonical_hash,
        deterministic_weighted_bootstrap_ci,
        load_jsonl,
        tracking_metrics,
    )
except ModuleNotFoundError:  # Allow direct execution from this directory.
    from compute_stage0_metrics import (  # type: ignore
        canonical_hash,
        deterministic_weighted_bootstrap_ci,
        load_jsonl,
        tracking_metrics,
    )


SCHEMA_VERSION = "rmg-stage1-v1"
ARM_NAMES = ("static", "periodic_pred", "rule_rmg", "pred_good", "gt_good")
METRIC_KEYS = (
    "success_auc",
    "mean_iou",
    "precision20",
    "normalized_precision",
    "normalized_precision_at_0_2",
)
DEFAULT_GATE = {
    "rule_static_auc_delta_min": 0.005,
    "rule_periodic_auc_delta_min": 0.010,
    "rule_periodic_auc_delta_ci_low_exclusive_min": 0.0,
    "bad_update_rate_max": 0.05,
    "worsened_sequence_fraction_max": 0.20,
    "clean_auc_delta_mean_min": -0.002,
    "clean_worsened_sequence_fraction_max": 0.10,
    "commit_coverage_min": 0.10,
}


def _finite_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _evaluation_iou(row):
    for key in ("evaluation_iou", "candidate_iou", "iou"):
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    return None


def _candidate_valid(row):
    if "candidate_valid" not in row:
        raise ValueError("Update opportunity is missing frozen candidate_valid.")
    return bool(row["candidate_valid"])


def _is_opportunity(row):
    for key in ("update_opportunity", "is_update_opportunity"):
        if key in row:
            return bool(row[key])
    if "candidate_valid" in row or _evaluation_iou(row) is not None:
        return bool(row.get("candidate_valid", True))
    return False


def _is_commit(row):
    for key in ("controller_action", "rule_action", "action", "decision"):
        action = row.get(key)
        if action is not None:
            return str(action).lower() in {"update", "commit", "write"}
    action = str(row.get("action_after_prediction", "")).lower()
    return action == "update" or action.startswith("commit_")


def update_quality(rows, good_iou=0.7, bad_iou=0.1):
    """Measure update selection quality using GT only after decisions are frozen.

    Recall is over valid good update opportunities.  Coverage is over every valid
    opportunity.  Invalid candidates and ordinary non-opportunity tracking frames
    are excluded from both denominators.
    """
    opportunities = [row for row in rows if _is_opportunity(row) and _candidate_valid(row)]
    committed = [row for row in opportunities if _is_commit(row)]
    good_opportunities = [
        row for row in opportunities
        if _evaluation_iou(row) is not None and _evaluation_iou(row) >= float(good_iou)
    ]
    good_commits = [
        row for row in committed
        if _evaluation_iou(row) is not None and _evaluation_iou(row) >= float(good_iou)
    ]
    bad_commits = [
        row for row in committed
        if _evaluation_iou(row) is not None and _evaluation_iou(row) <= float(bad_iou)
    ]
    return {
        "num_opportunities": len(opportunities),
        "num_commits": len(committed),
        "num_good_opportunities": len(good_opportunities),
        "num_good_commits": len(good_commits),
        "num_bad_commits": len(bad_commits),
        "update_precision": (float(len(good_commits) / len(committed)) if committed else 0.0),
        "update_recall": (float(len(good_commits) / len(good_opportunities))
                          if good_opportunities else 0.0),
        "bad_update_rate": (float(len(bad_commits) / len(committed)) if committed else 0.0),
        "commit_coverage": (float(len(committed) / len(opportunities))
                            if opportunities else 0.0),
    }


def metrics_from_trace(rows):
    rows = sorted(rows, key=lambda row: int(row["frame_idx"]))
    metrics = tracking_metrics(
        [row["pred_xywh"] for row in rows],
        [row["gt_xywh"] for row in rows],
    )
    metrics.update(update_quality(rows))
    return metrics


def _weighted_quality(sequence_summaries, arm="rule_rmg"):
    totals = {
        key: sum(int(summary.get("governance", summary.get("arms", {}).get(arm, {})).get(key, 0))
                 for summary in sequence_summaries)
        for key in ("num_opportunities", "num_commits", "num_good_opportunities",
                    "num_good_commits", "num_bad_commits")
    }
    commits = totals["num_commits"]
    opportunities = totals["num_opportunities"]
    good = totals["num_good_opportunities"]
    return {
        **totals,
        "update_precision": float(totals["num_good_commits"] / commits) if commits else 0.0,
        "update_recall": float(totals["num_good_commits"] / good) if good else 0.0,
        "bad_update_rate": float(totals["num_bad_commits"] / commits) if commits else 0.0,
        "commit_coverage": float(commits / opportunities) if opportunities else 0.0,
        "aggregation": "micro_average_over_update_opportunities",
    }


def _is_clean_summary(summary):
    for key in ("clean_subset", "is_clean", "clean"):
        if key in summary:
            return bool(summary[key])
    metadata = summary.get("metadata", {})
    for key in ("clean_subset", "is_clean", "clean"):
        if key in metadata:
            return bool(metadata[key])
    static = summary.get("arms", {}).get("static", {})
    return float(static.get("success_auc", 0.0)) >= 0.60


def _paired_weighted_delta(sequence_summaries, variant, baseline, metric, seed, samples):
    pairs = [
        (summary["arms"][baseline], summary["arms"][variant])
        for summary in sequence_summaries
        if baseline in summary.get("arms", {}) and variant in summary.get("arms", {})
    ]
    values = [candidate[metric] - reference[metric] for reference, candidate in pairs]
    weights = [reference["num_frames"] for reference, _ in pairs]
    return deterministic_weighted_bootstrap_ci(values, weights, seed=seed, samples=samples)


def _worsened(sequence_summaries, baseline="static", clean_only=False,
              threshold=-0.01):
    eligible = [summary for summary in sequence_summaries
                if baseline in summary.get("arms", {}) and "rule_rmg" in summary.get("arms", {})
                and (not clean_only or _is_clean_summary(summary))]
    deltas = [summary["arms"]["rule_rmg"]["success_auc"]
              - summary["arms"][baseline]["success_auc"] for summary in eligible]
    count = sum(delta < float(threshold) for delta in deltas)
    return {
        "definition": f"rule-minus-{baseline} success AUC < {float(threshold)}",
        "threshold": float(threshold),
        "num_sequences": len(deltas),
        "num_worsened": int(count),
        "fraction": float(count / len(deltas)) if deltas else 0.0,
    }


def aggregate_sequence_summaries(sequence_summaries, bootstrap_seed=0,
                                 bootstrap_samples=2000):
    """Aggregate five-arm tracking and rule-governance metrics.

    The locked comparisons bootstrap paired sequences and retain each sequence's
    frame count as its aggregation weight.
    """
    summaries = sorted(sequence_summaries, key=lambda item: str(item.get("sequence", "")))
    present_arms = [arm for arm in ARM_NAMES
                    if any(arm in summary.get("arms", {}) for summary in summaries)]
    arms = {}
    for arm in present_arms:
        available = [summary["arms"][arm] for summary in summaries
                     if arm in summary.get("arms", {})]
        total_frames = sum(int(item.get("num_frames", 0)) for item in available)
        arms[arm] = {
            "num_sequences": len(available),
            "num_frames": total_frames,
            **{
                key: (float(np.average([item[key] for item in available],
                                       weights=[item["num_frames"] for item in available]))
                      if total_frames else 0.0)
                for key in METRIC_KEYS
            },
            "aggregation": "frame_weighted_sequence_metrics",
        }

    comparisons = {}
    for comparison_index, baseline in enumerate(("static", "periodic_pred")):
        if baseline not in present_arms or "rule_rmg" not in present_arms:
            continue
        key = f"rule_rmg_vs_{baseline}"
        comparisons[key] = {
            metric: _paired_weighted_delta(
                summaries, "rule_rmg", baseline, metric,
                bootstrap_seed + comparison_index * len(METRIC_KEYS) + metric_index,
                bootstrap_samples,
            )
            for metric_index, metric in enumerate(METRIC_KEYS)
        }

    clean = [summary for summary in summaries if _is_clean_summary(summary)]
    clean_delta = _paired_weighted_delta(
        clean, "rule_rmg", "static", "success_auc",
        bootstrap_seed + 2 * len(METRIC_KEYS), bootstrap_samples,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "num_sequences": len(summaries),
        "expected_arms": list(ARM_NAMES),
        "arms": arms,
        "frame_weighted_paired_deltas": comparisons,
        "rule_governance": _weighted_quality(summaries),
        "worsened_vs_static": _worsened(summaries, "static"),
        "worsened_vs_periodic": _worsened(summaries, "periodic_pred"),
        "clean_subset_preservation": {
            "definition": "explicit clean flag when present, otherwise static sequence AUC >= 0.60",
            "num_sequences": len(clean),
            "rule_vs_static_success_auc": clean_delta,
            "worsened_vs_static": _worsened(summaries, "static", clean_only=True),
        },
    }
    # Explicit compatibility aliases make gate artifacts easy to inspect.
    result["frame_weighted_paired_deltas_vs_static"] = {
        "rule_rmg": comparisons.get("rule_rmg_vs_static", {})
    }
    result["frame_weighted_paired_deltas_vs_periodic"] = {
        "rule_rmg": comparisons.get("rule_rmg_vs_periodic_pred", {})
    }
    result["content_hash"] = canonical_hash(result)
    return result


def _check(value, pass_value, **limits):
    return {"value": value, **limits, "pass": bool(pass_value)}


def evaluate_locked_gate(aggregate, gate_config=None):
    """Evaluate the immutable Stage 1 rule-governance continuation gate."""
    config = dict(DEFAULT_GATE)
    if gate_config:
        config.update(gate_config)
    comparisons = aggregate.get("frame_weighted_paired_deltas", {})
    static = comparisons.get("rule_rmg_vs_static", {}).get("success_auc", {})
    periodic = comparisons.get("rule_rmg_vs_periodic_pred", {}).get("success_auc", {})
    governance = aggregate.get("rule_governance", {})
    worsened = aggregate.get("worsened_vs_static", {})
    clean = aggregate.get("clean_subset_preservation", {})
    clean_delta = clean.get("rule_vs_static_success_auc", {})
    clean_worsened = clean.get("worsened_vs_static", {})

    checks = {
        "rule_vs_static_auc_delta_mean": _check(
            static.get("mean", 0.0),
            static.get("mean", 0.0) >= config["rule_static_auc_delta_min"],
            minimum=config["rule_static_auc_delta_min"],
        ),
        "rule_vs_periodic_auc_delta_mean": _check(
            periodic.get("mean", 0.0),
            periodic.get("mean", 0.0) >= config["rule_periodic_auc_delta_min"],
            minimum=config["rule_periodic_auc_delta_min"],
        ),
        "rule_vs_periodic_auc_delta_ci_low": _check(
            periodic.get("low", 0.0),
            periodic.get("low", 0.0)
            > config["rule_periodic_auc_delta_ci_low_exclusive_min"],
            exclusive_minimum=config["rule_periodic_auc_delta_ci_low_exclusive_min"],
        ),
        "bad_update_rate": _check(
            governance.get("bad_update_rate", 0.0),
            governance.get("bad_update_rate", 0.0) <= config["bad_update_rate_max"],
            maximum=config["bad_update_rate_max"],
        ),
        "worsened_sequence_fraction": _check(
            worsened.get("fraction", 0.0),
            worsened.get("fraction", 0.0) <= config["worsened_sequence_fraction_max"],
            maximum=config["worsened_sequence_fraction_max"],
        ),
        "clean_sequence_coverage": _check(
            clean.get("num_sequences", 0),
            clean.get("num_sequences", 0) > 0,
            exclusive_minimum=0,
        ),
        "clean_auc_delta_mean": _check(
            clean_delta.get("mean", 0.0),
            clean_delta.get("mean", 0.0) >= config["clean_auc_delta_mean_min"],
            minimum=config["clean_auc_delta_mean_min"],
        ),
        "clean_worsened_sequence_fraction": _check(
            clean_worsened.get("fraction", 0.0),
            clean_worsened.get("fraction", 0.0)
            <= config["clean_worsened_sequence_fraction_max"],
            maximum=config["clean_worsened_sequence_fraction_max"],
        ),
        "commit_coverage": _check(
            governance.get("commit_coverage", 0.0),
            governance.get("commit_coverage", 0.0) >= config["commit_coverage_min"],
            minimum=config["commit_coverage_min"],
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "locked": True,
        "config": config,
        "checks": checks,
        "pass": all(item["pass"] for item in checks.values()),
    }
    result["content_hash"] = canonical_hash(result)
    return result


def compute_stage1_metrics(sequence_summaries, bootstrap_seed=0, bootstrap_samples=2000,
                           gate_config=None):
    aggregate = aggregate_sequence_summaries(
        sequence_summaries, bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )
    gate = evaluate_locked_gate(aggregate, gate_config)
    return aggregate, gate


def parse_args():
    parser = argparse.ArgumentParser(description="Compute RMG-Track Stage 1 metrics.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--bootstrap-seed", default=0, type=int)
    parser.add_argument("--bootstrap-samples", default=2000, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.experiment_dir)
    summaries = []
    for sequence_dir in sorted((root / "sequences").iterdir()):
        if not sequence_dir.is_dir():
            continue
        arms = {}
        for arm in ARM_NAMES:
            path = sequence_dir / "arms" / f"{arm}.trace.jsonl"
            if path.is_file():
                arms[arm] = metrics_from_trace(load_jsonl(path))
        metadata_path = sequence_dir / "sequence_metadata.json"
        metadata = (json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata_path.is_file() else {})
        summaries.append({"sequence": sequence_dir.name, "arms": arms, "metadata": metadata})
    aggregate, gate = compute_stage1_metrics(
        summaries, bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    (root / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "gate_result.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {root / 'aggregate_summary.json'} and {root / 'gate_result.json'}")


if __name__ == "__main__":
    main()
