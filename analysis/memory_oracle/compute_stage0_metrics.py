"""Metrics and aggregation for RMG-Track Stage 0 memory-oracle experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "rmg-stage0-v1"
SUCCESS_THRESHOLDS = np.linspace(0.0, 1.0, 21)
NORMALIZED_PRECISION_THRESHOLDS = np.linspace(0.0, 0.5, 51)


def iou_xywh(a, b):
    """Intersection over union for two ``[x, y, w, h]`` boxes."""
    if a is None or b is None or len(a) < 4 or len(b) < 4:
        return 0.0
    ax, ay, aw, ah = [float(v) for v in a[:4]]
    bx, by, bw, bh = [float(v) for v in b[:4]]
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def center_error(pred, gt):
    if pred is None or gt is None or len(pred) < 4 or len(gt) < 4:
        return float("inf")
    px, py, pw, ph = [float(v) for v in pred[:4]]
    gx, gy, gw, gh = [float(v) for v in gt[:4]]
    return float(np.hypot(px + 0.5 * pw - gx - 0.5 * gw,
                          py + 0.5 * ph - gy - 0.5 * gh))


def normalized_center_error(pred, gt):
    if pred is None or gt is None or len(pred) < 4 or len(gt) < 4:
        return float("inf")
    px, py, pw, ph = [float(v) for v in pred[:4]]
    gx, gy, gw, gh = [float(v) for v in gt[:4]]
    if gw <= 0 or gh <= 0:
        return float("inf")
    dx = (px + 0.5 * pw - gx - 0.5 * gw) / gw
    dy = (py + 0.5 * ph - gy - 0.5 * gh) / gh
    return float(np.hypot(dx, dy))


def success_auc(ious):
    values = np.asarray(list(ious), dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.mean([(values >= threshold).mean() for threshold in SUCCESS_THRESHOLDS]))


def tracking_metrics(predictions, ground_truth):
    """Compute standard one-pass tracking metrics on aligned boxes."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length.")
    ious = np.asarray([iou_xywh(pred, gt) for pred, gt in zip(predictions, ground_truth)], dtype=float)
    errors = np.asarray([center_error(pred, gt) for pred, gt in zip(predictions, ground_truth)], dtype=float)
    norm_errors = np.asarray(
        [normalized_center_error(pred, gt) for pred, gt in zip(predictions, ground_truth)], dtype=float
    )
    return {
        "num_frames": int(len(predictions)),
        "success_auc": success_auc(ious),
        "mean_iou": float(ious.mean()) if ious.size else 0.0,
        "precision20": float((errors <= 20.0).mean()) if errors.size else 0.0,
        "normalized_precision": float(np.mean([
            (norm_errors <= threshold).mean() for threshold in NORMALIZED_PRECISION_THRESHOLDS
        ])) if norm_errors.size else 0.0,
        "normalized_precision_at_0_2": float((norm_errors <= 0.2).mean()) if norm_errors.size else 0.0,
    }


def paired_metric_deltas(baseline, variant):
    """Return variant-minus-baseline deltas (positive always means improvement)."""
    keys = ("success_auc", "mean_iou", "precision20", "normalized_precision",
            "normalized_precision_at_0_2")
    return {key: float(variant[key] - baseline[key]) for key in keys}


def deterministic_bootstrap_ci(paired_values, seed=0, samples=2000, confidence=0.95):
    """Deterministic percentile CI for the mean of paired scalar deltas."""
    values = np.asarray(list(paired_values), dtype=float)
    if values.size == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "samples": 0, "seed": int(seed)}
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, values.size, size=(int(samples), values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, alpha)),
        "high": float(np.quantile(means, 1.0 - alpha)),
        "samples": int(samples),
        "seed": int(seed),
    }


def deterministic_weighted_bootstrap_ci(values, weights, seed=0, samples=2000,
                                        confidence=0.95):
    """Bootstrap paired sequence deltas while preserving frame-count weights."""
    values = np.asarray(list(values), dtype=float)
    weights = np.asarray(list(weights), dtype=float)
    if values.size == 0 or values.size != weights.size or np.any(weights <= 0):
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "samples": 0, "seed": int(seed),
                "num_sequences": 0, "aggregation": "frame_weighted_sequence_bootstrap"}
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(int(samples)):
        indices = rng.randint(0, values.size, size=values.size)
        means.append(float(np.average(values[indices], weights=weights[indices])))
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.average(values, weights=weights)),
        "low": float(np.quantile(means, alpha)),
        "high": float(np.quantile(means, 1.0 - alpha)),
        "samples": int(samples),
        "seed": int(seed),
        "num_sequences": int(values.size),
        "aggregation": "frame_weighted_sequence_bootstrap",
    }


def deterministic_clustered_bootstrap_ci(values_by_sequence, seed=0, samples=2000,
                                         confidence=0.95):
    """Bootstrap sequence clusters while retaining every paired event in each cluster."""
    clusters = [np.asarray(values, dtype=float) for _, values in sorted(values_by_sequence.items())
                if len(values)]
    event_count = sum(cluster.size for cluster in clusters)
    if not clusters:
        return {
            "mean": 0.0, "low": 0.0, "high": 0.0, "samples": 0, "seed": int(seed),
            "num_events": 0, "num_sequences": 0, "aggregation": "sequence_cluster_bootstrap",
        }
    all_values = np.concatenate(clusters)
    rng = np.random.RandomState(seed)
    bootstrap_means = []
    for _ in range(int(samples)):
        sampled = rng.randint(0, len(clusters), size=len(clusters))
        bootstrap_means.append(float(np.concatenate([clusters[index] for index in sampled]).mean()))
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(all_values.mean()),
        "low": float(np.quantile(bootstrap_means, alpha)),
        "high": float(np.quantile(bootstrap_means, 1.0 - alpha)),
        "samples": int(samples),
        "seed": int(seed),
        "num_events": int(event_count),
        "num_sequences": int(len(clusters)),
        "aggregation": "sequence_cluster_bootstrap",
    }


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metrics_from_trace(rows):
    rows = sorted(rows, key=lambda row: int(row["frame_idx"]))
    return tracking_metrics([row["pred_xywh"] for row in rows], [row["gt_xywh"] for row in rows])


def aggregate_sequence_summaries(sequence_summaries, bootstrap_seed=0, bootstrap_samples=2000):
    """Macro-average arms and compute paired sequence-level CIs against static."""
    metric_keys = ("success_auc", "mean_iou", "precision20", "normalized_precision",
                   "normalized_precision_at_0_2")
    arm_names = sorted({arm for summary in sequence_summaries for arm in summary.get("arms", {})})
    arms = {}
    for arm in arm_names:
        available = [summary["arms"][arm] for summary in sequence_summaries if arm in summary.get("arms", {})]
        arms[arm] = {
            "num_sequences": len(available),
            **{key: float(np.mean([metrics[key] for metrics in available])) for key in metric_keys},
        }
    deltas = {}
    frame_weighted_deltas = {}
    if "static" in arm_names:
        for arm in arm_names:
            if arm == "static":
                continue
            pairs = [(summary["arms"]["static"], summary["arms"][arm])
                     for summary in sequence_summaries
                     if "static" in summary.get("arms", {}) and arm in summary.get("arms", {})]
            deltas[arm] = {}
            frame_weighted_deltas[arm] = {}
            for offset, key in enumerate(metric_keys):
                values = [variant[key] - baseline[key] for baseline, variant in pairs]
                weights = [baseline["num_frames"] for baseline, _ in pairs]
                deltas[arm][key] = deterministic_bootstrap_ci(
                    values, seed=bootstrap_seed + offset, samples=bootstrap_samples
                )
                frame_weighted_deltas[arm][key] = deterministic_weighted_bootstrap_ci(
                    values, weights, seed=bootstrap_seed + offset, samples=bootstrap_samples
                )
    result = {
        "schema_version": SCHEMA_VERSION,
        "num_sequences": len(sequence_summaries),
        "aggregation": "sequence_macro_average",
        "arms": arms,
        "paired_deltas_vs_static": deltas,
        "frame_weighted_paired_deltas_vs_static": frame_weighted_deltas,
    }
    result["content_hash"] = canonical_hash(result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Compute RMG-Track Stage 0 metrics from experiment traces.")
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
        arm_dir = sequence_dir / "arms"
        for trace_path in sorted(arm_dir.glob("*.trace.jsonl")):
            arms[trace_path.name[:-len(".trace.jsonl")]] = metrics_from_trace(load_jsonl(trace_path))
        summaries.append({"sequence": sequence_dir.name, "arms": arms})
    aggregate = aggregate_sequence_summaries(summaries, args.bootstrap_seed, args.bootstrap_samples)
    payload = json.dumps(aggregate, indent=2, ensure_ascii=False)
    (root / "aggregate_summary.json").write_text(payload, encoding="utf-8")
    # Compatibility alias for early Stage 0 runs.
    (root / "aggregate.json").write_text(payload, encoding="utf-8")
    print(f"Wrote {root / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
