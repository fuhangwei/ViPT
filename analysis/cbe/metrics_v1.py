"""Pure-NumPy metrics and fail-closed gates for CBE Stage 0 Diagnostic v1.

All aggregate routines use sequence-macro semantics: events are reduced within
sequence first, then sequences receive equal weight.  No routine silently drops
NaN or infinite values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "cbe-stage0-diagnostic-v1"
PRIMARY_DIRECTIONS = (
    "rgb_blur",
    "rgb_low_light",
    "rgb_desaturation",
    "rgb_occlusion",
    "tir_contrast_compression",
    "tir_saturation",
    "tir_sensor_noise",
    "tir_blur",
)
SEMANTIC_ATTRIBUTES = (
    "day_night",
    "low_illumination",
    "occlusion",
    "fast_motion",
    "thermal_or_modality_challenge",
)


# ---------------------------------------------------------------------------
# Numeric validation and geometry
# ---------------------------------------------------------------------------


def _finite_array(values, name="values", allow_empty=False):
    array = np.asarray(values, dtype=float)
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _finite_scalar(value, name="value"):
    array = _finite_array([value], name)
    return float(array[0])


def _xywh(box, name):
    if box is None:
        raise ValueError(f"{name} must be an xywh box")
    values = _finite_array(box, name).reshape(-1)
    if values.size < 4:
        raise ValueError(f"{name} must contain at least four values")
    x, y, width, height = (float(v) for v in values[:4])
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"{name} width and height must be positive")
    return x, y, width, height


def iou_xywh(a, b):
    """Return IoU for finite positive-area ``[x, y, width, height]`` boxes."""
    ax, ay, aw, ah = _xywh(a, "a")
    bx, by, bw, bh = _xywh(b, "b")
    intersection_width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return float(intersection / union)


def center_error(prediction, ground_truth):
    """Euclidean center error for finite positive-area xywh boxes."""
    px, py, pw, ph = _xywh(prediction, "prediction")
    gx, gy, gw, gh = _xywh(ground_truth, "ground_truth")
    return float(np.hypot((px + 0.5 * pw) - (gx + 0.5 * gw),
                          (py + 0.5 * ph) - (gy + 0.5 * gh)))


def _pair(value, name):
    array = _finite_array(value, name).reshape(-1)
    if array.size == 1:
        pair = (float(array[0]), float(array[0]))
    elif array.size == 2:
        pair = (float(array[0]), float(array[1]))
    else:
        raise ValueError(f"{name} must be a scalar or length-two value")
    if pair[0] <= 0.0 or pair[1] <= 0.0:
        raise ValueError(f"{name} values must be positive")
    return pair


def map_gt_to_feature_grid(gt_xywh, search_anchor_xywh, search_size,
                           resize_factor, feature_shape):
    """Map an original-image GT box to continuous feature-grid coordinates.

    ``search_anchor_xywh`` is the tracker state about whose center the search
    crop was extracted.  The unpadded crop size in original-image pixels is
    ``search_size / resize_factor``.  Coordinates are continuous edge
    coordinates: grid cell ``(row, col)`` occupies
    ``[col, col + 1) x [row, row + 1)``.

    Returns ``[grid_x, grid_y, grid_width, grid_height]``.  The result is not
    clipped; clipping occurs naturally while cell overlaps are calculated.
    """
    gx, gy, gw, gh = _xywh(gt_xywh, "gt_xywh")
    ax, ay, aw, ah = _xywh(search_anchor_xywh, "search_anchor_xywh")
    search_width, search_height = _pair(search_size, "search_size")
    resize_x, resize_y = _pair(resize_factor, "resize_factor")
    feature_height, feature_width = (int(v) for v in feature_shape)
    if feature_height <= 0 or feature_width <= 0:
        raise ValueError("feature_shape must contain positive (height, width)")

    crop_width = search_width / resize_x
    crop_height = search_height / resize_y
    crop_x = ax + 0.5 * aw - 0.5 * crop_width
    crop_y = ay + 0.5 * ah - 0.5 * crop_height
    scale_x = feature_width / crop_width
    scale_y = feature_height / crop_height
    return np.asarray([
        (gx - crop_x) * scale_x,
        (gy - crop_y) * scale_y,
        gw * scale_x,
        gh * scale_y,
    ], dtype=float)


def fractional_cell_overlap_weights(grid_xywh, feature_shape):
    """Return each feature cell's fractional area covered by a grid-space box."""
    x, y, width, height = _xywh(grid_xywh, "grid_xywh")
    feature_height, feature_width = (int(v) for v in feature_shape)
    if feature_height <= 0 or feature_width <= 0:
        raise ValueError("feature_shape must contain positive (height, width)")
    x_edges = np.arange(feature_width + 1, dtype=float)
    y_edges = np.arange(feature_height + 1, dtype=float)
    overlap_x = np.maximum(
        0.0, np.minimum(x_edges[1:], x + width) - np.maximum(x_edges[:-1], x)
    )
    overlap_y = np.maximum(
        0.0, np.minimum(y_edges[1:], y + height) - np.maximum(y_edges[:-1], y)
    )
    weights = overlap_y[:, None] * overlap_x[None, :]
    if not np.any(weights > 0.0):
        raise ValueError("GT box has zero overlap with the feature grid")
    return weights


def gt_fractional_cell_weights(gt_xywh, search_anchor_xywh, search_size,
                               resize_factor, feature_shape,
                               search_crop_xywh=None):
    """Map original-image GT and return fractional feature-cell overlaps."""
    if search_crop_xywh is None:
        grid_box = map_gt_to_feature_grid(
            gt_xywh, search_anchor_xywh, search_size, resize_factor, feature_shape
        )
    else:
        gx, gy, gw, gh = _xywh(gt_xywh, "gt_xywh")
        crop_x, crop_y, crop_width, crop_height = _xywh(
            search_crop_xywh, "search_crop_xywh"
        )
        feature_height, feature_width = (int(value) for value in feature_shape)
        grid_box = np.asarray([
            (gx - crop_x) * feature_width / crop_width,
            (gy - crop_y) * feature_height / crop_height,
            gw * feature_width / crop_width,
            gh * feature_height / crop_height,
        ], dtype=float)
    return fractional_cell_overlap_weights(grid_box, feature_shape)


# Explicit aliases for callers that use noun-first naming.
gt_to_feature_grid_xywh = map_gt_to_feature_grid
fractional_gt_weights = gt_fractional_cell_weights


# ---------------------------------------------------------------------------
# Response-map evidence
# ---------------------------------------------------------------------------


def _spatial_map(value, name):
    array = _finite_array(value, name)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must reduce to a two-dimensional spatial map")
    return array


def _channel_map(value, channels, shape, name):
    array = _finite_array(value, name)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == channels:
        result = array
    elif array.ndim == 3 and array.shape[-1] == channels:
        result = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"{name} must have {channels} channels")
    if result.shape[1:] != shape:
        raise ValueError(f"{name} spatial shape does not match score_map")
    return result


def stable_softmax(score_map, temperature=1.0):
    """Stable spatial softmax; temperature must be finite and positive."""
    scores = _spatial_map(score_map, "score_map")
    temperature = _finite_scalar(temperature, "temperature")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    shifted = scores / temperature
    shifted = shifted - np.max(shifted)
    exponentials = np.exp(shifted)
    denominator = float(np.sum(exponentials))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("softmax normalization is non-finite or zero")
    return exponentials / denominator


def decode_box_at_peak(response_map, size_map, offset_map):
    """Decode a normalized-search ``cx, cy, w, h`` box at response argmax.

    Size and offset maps follow the OSTrack convention: channels ``(w, h)``
    and ``(offset_x, offset_y)``.  This helper deliberately does not compute
    GT IoU; IoU belongs to the evaluator-side join.
    """
    response = _spatial_map(response_map, "response_map")
    shape = response.shape
    sizes = _channel_map(size_map, 2, shape, "size_map")
    offsets = _channel_map(offset_map, 2, shape, "offset_map")
    row, col = np.unravel_index(int(np.argmax(response)), shape)
    height, width = shape
    box = np.asarray([
        (col + offsets[0, row, col]) / width,
        (row + offsets[1, row, col]) / height,
        sizes[0, row, col],
        sizes[1, row, col],
    ], dtype=float)
    if not np.all(np.isfinite(box)):
        raise ValueError("decoded box is non-finite")
    return box


def evidence_metrics(score_map, hann_window, gt_weights, size_map=None,
                     offset_map=None):
    """Compute three explicitly separated forms of spatial evidence.

    * Raw evidence uses the unnormalized score map.
    * Diagnostic belief is ``softmax(score / T)`` with fixed ``T=1``.
    * Windowed response is the elementwise score-by-Hann product.

    ``GT mass`` for raw/windowed maps is the fractional-overlap weighted share
    of the map's total nonnegative response.  The corresponding unnormalized
    weighted sums are retained separately.  Belief GT mass is probabilistic.
    GT peak is the maximum value among cells with nonzero GT overlap.  Entropy
    is full-map Shannon entropy in nats.

    If both size and offset maps are supplied, normalized-search boxes are
    decoded at the raw/belief and windowed peaks.  Supplying only one map is an
    error.  Box IoU is intentionally absent and must be added by
    :func:`evaluator_join_box_iou`.
    """
    scores = _spatial_map(score_map, "score_map")
    window = _spatial_map(hann_window, "hann_window")
    weights = _spatial_map(gt_weights, "gt_weights")
    if window.shape != scores.shape or weights.shape != scores.shape:
        raise ValueError("score_map, hann_window, and gt_weights shapes must match")
    if np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError("gt_weights must lie in [0, 1]")
    gt_area = float(np.sum(weights))
    background_weights = 1.0 - weights
    background_area = float(np.sum(background_weights))
    if gt_area <= 0.0:
        raise ValueError("gt_weights must contain positive overlap")
    if background_area <= 0.0:
        raise ValueError("evidence contrast requires non-GT grid area")

    if np.any(scores < 0.0) or np.any(window < 0.0):
        raise ValueError("raw score and Hann maps must be nonnegative")
    belief = stable_softmax(scores, temperature=1.0)
    windowed = scores * window
    raw_total = float(np.sum(scores))
    windowed_total = float(np.sum(windowed))
    if raw_total <= 0.0 or windowed_total <= 0.0:
        raise ValueError("raw score and windowed response must have positive total mass")
    gt_mask = weights > 0.0
    raw_gt_sum = float(np.sum(scores * weights))
    windowed_gt_sum = float(np.sum(windowed * weights))
    raw_gt_mean = float(raw_gt_sum / gt_area)
    raw_background_mean = float(np.sum(scores * background_weights) / background_area)
    positive_belief = belief[belief > 0.0]
    entropy = float(-np.sum(positive_belief * np.log(positive_belief)))
    normalizer = np.log(float(belief.size)) if belief.size > 1 else 1.0

    result = {
        "schema_version": SCHEMA_VERSION,
        "raw_gt_mean": raw_gt_mean,
        "raw_gt_weighted_sum": raw_gt_sum,
        "raw_gt_mass": float(raw_gt_sum / raw_total),
        "raw_gt_contrast": float(raw_gt_mean - raw_background_mean),
        "belief_gt_mass": float(np.sum(belief * weights)),
        "belief_gt_peak": float(np.max(belief[gt_mask])),
        "belief_entropy": entropy,
        "belief_entropy_normalized": float(entropy / normalizer) if belief.size > 1 else 0.0,
        "windowed_gt_weighted_sum": windowed_gt_sum,
        "windowed_gt_mass": float(windowed_gt_sum / windowed_total),
        "windowed_gt_peak": float(np.max(windowed[gt_mask])),
    }
    if (size_map is None) != (offset_map is None):
        raise ValueError("size_map and offset_map must be supplied together")
    if size_map is not None:
        result["raw_peak_box_cxcywh_normalized_search"] = decode_box_at_peak(
            scores, size_map, offset_map
        ).tolist()
        # Spatial softmax at T=1 preserves the raw-score argmax.
        result["belief_peak_box_cxcywh_normalized_search"] = list(
            result["raw_peak_box_cxcywh_normalized_search"]
        )
        result["windowed_peak_box_cxcywh_normalized_search"] = decode_box_at_peak(
            windowed, size_map, offset_map
        ).tolist()
    return result


def evaluator_join_box_iou(probe_records, evaluator_records,
                           key_fields=("sequence_name", "frame_index", "arm"),
                           prediction_field="pred_xywh", gt_field="gt_xywh",
                           output_field="box_iou"):
    """Join probe output to evaluator GT and compute box IoU evaluator-side.

    Duplicate evaluator keys, duplicate probe keys, missing matches, malformed
    boxes, and non-finite inputs fail closed with ``ValueError``.
    """
    evaluator_index = {}
    for record in evaluator_records:
        key = tuple(record[field] for field in key_fields)
        if key in evaluator_index:
            raise ValueError(f"duplicate evaluator join key: {key!r}")
        evaluator_index[key] = record
    seen = set()
    joined = []
    for probe in probe_records:
        key = tuple(probe[field] for field in key_fields)
        if key in seen:
            raise ValueError(f"duplicate probe join key: {key!r}")
        seen.add(key)
        if key not in evaluator_index:
            raise ValueError(f"missing evaluator record for join key: {key!r}")
        evaluator = evaluator_index[key]
        row = dict(probe)
        row[gt_field] = evaluator[gt_field]
        row[output_field] = iou_xywh(probe[prediction_field], evaluator[gt_field])
        joined.append(row)
    return joined


# ---------------------------------------------------------------------------
# Negative fusion and intervention diagnostics
# ---------------------------------------------------------------------------


def fusion_regret(fusion_iou, probe_ious):
    """Return ``max(probe IoU) - fusion IoU`` (not clipped at zero)."""
    fusion = _finite_scalar(fusion_iou, "fusion_iou")
    probes = _finite_array(probe_ious, "probe_ious").reshape(-1)
    return float(np.max(probes) - fusion)


def fusion_regret_metrics(fusion_iou, probe_ious, negative_threshold=0.10,
                          severe_best_threshold=0.60,
                          severe_fusion_threshold=0.40):
    """Return regret plus preregistered negative/severe fusion indicators."""
    fusion = _finite_scalar(fusion_iou, "fusion_iou")
    probes = _finite_array(probe_ious, "probe_ious").reshape(-1)
    best = float(np.max(probes))
    regret = best - fusion
    return {
        "fusion_iou": fusion,
        "max_probe_iou": best,
        "fusion_regret": float(regret),
        "negative_fusion": bool(regret >= _finite_scalar(negative_threshold)),
        "severe_negative_fusion": bool(
            best >= _finite_scalar(severe_best_threshold)
            and fusion < _finite_scalar(severe_fusion_threshold)
        ),
    }


def intervention_effect(clean_gt_evidence, intervened_gt_evidence):
    """Target evidence drop: clean minus intervened (positive means damage)."""
    return float(_finite_scalar(clean_gt_evidence, "clean_gt_evidence")
                 - _finite_scalar(intervened_gt_evidence, "intervened_gt_evidence"))


def intervention_faithfulness(target_effect, background_effect):
    """Target-minus-background effect (positive means target-local specificity)."""
    return float(_finite_scalar(target_effect, "target_effect")
                 - _finite_scalar(background_effect, "background_effect"))


def intervention_metrics(clean_gt_evidence, target_intervened_gt_evidence,
                         background_intervened_gt_evidence):
    target = intervention_effect(clean_gt_evidence, target_intervened_gt_evidence)
    background = intervention_effect(clean_gt_evidence, background_intervened_gt_evidence)
    return {
        "target_effect": target,
        "background_effect": background,
        "faithfulness": intervention_faithfulness(target, background),
    }


# ---------------------------------------------------------------------------
# Rank correlation and descriptive calibration
# ---------------------------------------------------------------------------


def average_tie_ranks(values):
    """One-based average ranks with stable, deterministic handling of ties."""
    values = _finite_array(values, "values").reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def spearman_rho(x, y):
    """Spearman rho with average tie ranks and fixed constant semantics.

    For fewer than two observations, or when either input is constant, rho is
    defined as ``0.0``.  This makes uninformative strength series deterministic
    and prevents a constant effect from passing a positive monotonicity gate.
    Non-finite values raise ``ValueError``.
    """
    x_values = _finite_array(x, "x").reshape(-1)
    y_values = _finite_array(y, "y").reshape(-1)
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same number of observations")
    if x_values.size < 2:
        return 0.0
    x_ranks = average_tie_ranks(x_values)
    y_ranks = average_tie_ranks(y_values)
    x_centered = x_ranks - np.mean(x_ranks)
    y_centered = y_ranks - np.mean(y_ranks)
    denominator = float(np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2)))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denominator)


def normalized_gt_distribution(gt_weights):
    weights = _finite_array(gt_weights, "gt_weights")
    if np.any(weights < 0.0):
        raise ValueError("gt_weights must be nonnegative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("gt_weights must have positive mass")
    return weights / total


def gt_region_nll(belief, gt_weights, epsilon=1e-12):
    """Region-mass NLL ``-log(sum(p * fractional_GT_overlap))``."""
    probabilities = _finite_array(belief, "belief")
    weights = _finite_array(gt_weights, "gt_weights")
    if probabilities.shape != weights.shape:
        raise ValueError("belief and gt_weights shapes must match")
    if np.any(probabilities < 0.0) or np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError("belief must be nonnegative and gt_weights must lie in [0, 1]")
    total = float(np.sum(probabilities))
    if total <= 0.0 or float(np.sum(weights)) <= 0.0:
        raise ValueError("belief and gt_weights must have positive mass")
    probabilities = probabilities / total
    epsilon = _finite_scalar(epsilon, "epsilon")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    region_mass = float(np.sum(probabilities * weights))
    return float(-np.log(max(region_mass, epsilon)))


def brier_score_gt(belief, gt_weights):
    """Multiclass Brier score ``sum((p - q_gt)^2)``."""
    probabilities = _finite_array(belief, "belief")
    target = normalized_gt_distribution(gt_weights)
    if probabilities.shape != target.shape:
        raise ValueError("belief and gt_weights shapes must match")
    if np.any(probabilities < 0.0):
        raise ValueError("belief must be nonnegative")
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("belief must have positive mass")
    probabilities = probabilities / total
    return float(np.sum((probabilities - target) ** 2))


def quality_ece(confidences, qualities, bins=10):
    """Equal-width quality ECE over confidence in [0, 1] and IoU quality."""
    confidence = _finite_array(confidences, "confidences").reshape(-1)
    quality = _finite_array(qualities, "qualities").reshape(-1)
    if confidence.size != quality.size:
        raise ValueError("confidences and qualities must have equal length")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidences must lie in [0, 1]")
    if np.any((quality < 0.0) | (quality > 1.0)):
        raise ValueError("qualities must lie in [0, 1]")
    bins = int(bins)
    if bins <= 0:
        raise ValueError("bins must be positive")
    # floor maps internal boundaries to the upper bin; confidence 1 stays last.
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = 0.0
    bin_audit = []
    for index in range(bins):
        mask = indices == index
        count = int(np.sum(mask))
        if count:
            mean_confidence = float(np.mean(confidence[mask]))
            mean_quality = float(np.mean(quality[mask]))
            contribution = count / confidence.size * abs(mean_confidence - mean_quality)
        else:
            mean_confidence = None
            mean_quality = None
            contribution = 0.0
        ece += contribution
        bin_audit.append({
            "index": index,
            "low": index / bins,
            "high": (index + 1) / bins,
            "right_closed": index == bins - 1,
            "count": count,
            "mean_confidence": mean_confidence,
            "mean_quality": mean_quality,
            "weighted_gap": float(contribution),
        })
    return {"ece": float(ece), "bins": bin_audit, "num_observations": int(confidence.size)}


def aurc(confidences, qualities):
    """Area under the empirical risk-coverage curve, risk = ``1 - IoU``.

    Stable descending confidence order gives deterministic semantics for ties.
    """
    confidence = _finite_array(confidences, "confidences").reshape(-1)
    quality = _finite_array(qualities, "qualities").reshape(-1)
    if confidence.size != quality.size:
        raise ValueError("confidences and qualities must have equal length")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidences must lie in [0, 1]")
    if np.any((quality < 0.0) | (quality > 1.0)):
        raise ValueError("qualities must lie in [0, 1]")
    order = np.argsort(-confidence, kind="mergesort")
    cumulative_risk = np.cumsum(1.0 - quality[order]) / np.arange(1, quality.size + 1)
    return float(np.mean(cumulative_risk))


def descriptive_calibration(beliefs, gt_weights, confidences, box_ious, bins=10):
    """Return preregistered descriptive calibration metrics for aligned events."""
    beliefs = list(beliefs)
    gt_weights = list(gt_weights)
    if len(beliefs) != len(gt_weights) or not beliefs:
        raise ValueError("beliefs and gt_weights must be nonempty and aligned")
    nll = [gt_region_nll(p, q) for p, q in zip(beliefs, gt_weights)]
    brier = [brier_score_gt(p, q) for p, q in zip(beliefs, gt_weights)]
    confidence = _finite_array(confidences, "confidences").reshape(-1)
    ious = _finite_array(box_ious, "box_ious").reshape(-1)
    if confidence.size != len(beliefs) or ious.size != len(beliefs):
        raise ValueError("all calibration inputs must have equal event counts")
    ece = quality_ece(confidence, ious, bins=bins)
    return {
        "gt_region_nll": float(np.mean(nll)),
        "brier_normalized_gt": float(np.mean(brier)),
        "quality_ece_10bin" if bins == 10 else "quality_ece": ece["ece"],
        "quality_ece_bins": ece["bins"],
        "confidence_iou_spearman": spearman_rho(confidence, ious),
        "aurc": aurc(confidence, ious),
        "num_events": len(beliefs),
    }


def sequence_macro_calibration(events_by_sequence, bins=10):
    """Compute calibration per sequence, then equally average sequences."""
    summaries = {}
    for sequence_name in sorted(events_by_sequence):
        events = list(events_by_sequence[sequence_name])
        if not events:
            raise ValueError(f"sequence {sequence_name!r} has no calibration events")
        summaries[sequence_name] = descriptive_calibration(
            [event["belief"] for event in events],
            [event["gt_weights"] for event in events],
            [event["confidence"] for event in events],
            [event["box_iou"] for event in events],
            bins=bins,
        )
    if not summaries:
        raise ValueError("events_by_sequence must not be empty")
    keys = ("gt_region_nll", "brier_normalized_gt",
            "quality_ece_10bin" if bins == 10 else "quality_ece",
            "confidence_iou_spearman", "aurc")
    return {
        "aggregation": "sequence_macro",
        "num_sequences": len(summaries),
        "metrics": {key: float(np.mean([item[key] for item in summaries.values()]))
                    for key in keys},
        "by_sequence": summaries,
    }


# ---------------------------------------------------------------------------
# Sequence-macro statistics and LOGO
# ---------------------------------------------------------------------------


def _statistic_function(statistic, proportion_threshold=0.0):
    if callable(statistic):
        return statistic, getattr(statistic, "__name__", "callable")
    name = str(statistic).lower()
    if name == "mean":
        return lambda values: float(np.mean(values)), name
    if name == "median":
        return lambda values: float(np.median(values)), name
    if name == "proportion":
        threshold = _finite_scalar(proportion_threshold, "proportion_threshold")
        return lambda values: float(np.mean(np.asarray(values, dtype=float) > threshold)), name
    raise ValueError("statistic must be mean, median, proportion, or a callable")


def sequence_macro_values(values_by_sequence, statistic="mean",
                          proportion_threshold=0.0):
    """Reduce events within each sequence without concatenating sequences."""
    if not isinstance(values_by_sequence, Mapping) or not values_by_sequence:
        raise ValueError("values_by_sequence must be a nonempty mapping")
    function, _ = _statistic_function(statistic, proportion_threshold)
    reduced = {}
    for sequence_name in sorted(values_by_sequence, key=str):
        values = _finite_array(values_by_sequence[sequence_name],
                               f"values_by_sequence[{sequence_name!r}]").reshape(-1)
        result = _finite_scalar(function(values), "sequence statistic")
        reduced[str(sequence_name)] = result
    return reduced


def deterministic_sequence_macro_bootstrap(values_by_sequence, seed=0,
                                           samples=2000, confidence=0.95,
                                           statistic="mean",
                                           proportion_threshold=0.0):
    """Deterministic percentile CI with sequence-macro aggregation.

    First, ``statistic`` reduces events independently inside each sequence.
    Then the same statistic is applied across those per-sequence values; the
    bootstrap resamples only these sequence summaries with replacement.  Thus a
    sequence with many events never receives more weight than a short sequence.

    ``proportion`` means the fraction strictly greater than
    ``proportion_threshold`` within sequence, followed by the equal-weight mean
    of sequence proportions (rather than a second thresholding operation).
    """
    function, name = _statistic_function(statistic, proportion_threshold)
    sequence_values = sequence_macro_values(
        values_by_sequence, statistic=statistic,
        proportion_threshold=proportion_threshold,
    )
    values = _finite_array(list(sequence_values.values()), "sequence_values").reshape(-1)
    samples = int(samples)
    confidence = _finite_scalar(confidence, "confidence")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    macro_function = np.mean if name == "proportion" else function
    estimate = _finite_scalar(macro_function(values), "macro estimate")
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(0, values.size, size=(samples, values.size))
    bootstrap_values = np.asarray(
        [_finite_scalar(macro_function(values[index]), "bootstrap statistic")
         for index in indices], dtype=float
    )
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(bootstrap_values, alpha))
    high = float(np.quantile(bootstrap_values, 1.0 - alpha))
    return {
        "estimate": estimate,
        "value": estimate,
        "mean": estimate,
        "low": low,
        "high": high,
        "confidence": confidence,
        "samples": samples,
        "seed": int(seed),
        "statistic": name,
        "num_sequences": int(values.size),
        "num_events": int(sum(np.asarray(v).size for v in values_by_sequence.values())),
        "aggregation": "within_sequence_then_sequence_macro",
        "resampling_unit": "sequence",
        "sequence_values": sequence_values,
    }


def attribute_logo(values_by_sequence, attribute_groups, recompute_callback,
                   sequence_key=None):
    """Leave each of five potentially overlapping attribute groups out.

    ``attribute_groups`` maps each frozen attribute name to sequence IDs.  Each
    omission starts from the full input, so overlap is handled correctly and
    groups are never forced into a partition.  ``recompute_callback`` receives
    the retained data and may return a bool or a mapping containing
    ``direction_consistent``.  The complete callback result is retained.
    """
    if set(attribute_groups) != set(SEMANTIC_ATTRIBUTES):
        raise ValueError("attribute_groups must contain exactly the five frozen groups")
    if not callable(recompute_callback):
        raise ValueError("recompute_callback must be callable")
    is_mapping = isinstance(values_by_sequence, Mapping)
    if not is_mapping and sequence_key is None:
        sequence_key = "sequence_name"
    results = {}
    for attribute in SEMANTIC_ATTRIBUTES:
        omitted = set(attribute_groups[attribute])
        if is_mapping:
            retained = {key: value for key, value in values_by_sequence.items()
                        if key not in omitted}
        else:
            retained = [record for record in values_by_sequence
                        if record[sequence_key] not in omitted]
        callback_result = recompute_callback(retained)
        if isinstance(callback_result, (bool, np.bool_)):
            consistent = bool(callback_result)
            audit = {"direction_consistent": consistent}
        elif isinstance(callback_result, Mapping):
            if "direction_consistent" not in callback_result:
                raise ValueError("LOGO callback mapping lacks direction_consistent")
            consistent = bool(callback_result["direction_consistent"])
            audit = dict(callback_result)
        else:
            raise ValueError("LOGO callback must return bool or a mapping")
        audit.update({
            "attribute": attribute,
            "omitted_sequence_count": len(omitted),
            "retained_sequence_count": len(retained),
            "direction_consistent": consistent,
        })
        results[attribute] = audit
    return {
        "method": "leave_one_overlapping_group_out",
        "groups_overlap_allowed": True,
        "attributes": results,
        "direction_consistent_count": int(sum(
            item["direction_consistent"] for item in results.values()
        )),
    }


attribute_leave_one_group_out = attribute_logo


# ---------------------------------------------------------------------------
# Gate A-E evaluation
# ---------------------------------------------------------------------------


def _direction_payload(direction_data, direction):
    if direction not in direction_data:
        return {}
    payload = direction_data[direction]
    if not isinstance(payload, Mapping):
        raise ValueError(f"direction payload for {direction} must be a mapping")
    return payload


def _mapping_field(payload, names, required=True):
    for name in names:
        if name in payload:
            value = payload[name]
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be values_by_sequence")
            return value
    if required:
        raise ValueError(f"missing required field; expected one of {names}")
    return None


def evaluate_gate_a(direction_data, seed=0, samples=2000, confidence=0.95):
    """Evaluate Gate A over exactly the frozen eight directions.

    A direction passes iff (1) sequence-macro median target effect is > 0,
    (2) sequence-macro median faithfulness is > 0, and (3) the 95% sequence
    bootstrap CI for median target effect does not cross zero.  A CI is
    non-crossing only when ``low >= 0`` or ``high <= 0`` (zero on a boundary is
    non-crossing).  Because positive median effect is separately required, the
    only acceptable branch is ultimately ``effect_ci.low >= 0``.
    """
    audits = {}
    pass_count = 0
    for index, direction in enumerate(PRIMARY_DIRECTIONS):
        payload = _direction_payload(direction_data, direction)
        try:
            target_values = _mapping_field(
                payload, ("target_effects_by_sequence", "target_effect_by_sequence")
            )
            faithfulness_values = _mapping_field(
                payload, ("faithfulness_by_sequence", "faithfulness_effects_by_sequence")
            )
            target_ci = deterministic_sequence_macro_bootstrap(
                target_values, seed=seed + index * 2, samples=samples,
                confidence=confidence, statistic="median"
            )
            faithfulness_ci = deterministic_sequence_macro_bootstrap(
                faithfulness_values, seed=seed + index * 2 + 1, samples=samples,
                confidence=confidence, statistic="median"
            )
            target_positive = target_ci["estimate"] > 0.0
            faithfulness_positive = faithfulness_ci["estimate"] > 0.0
            ci_non_crossing = target_ci["low"] >= 0.0 or target_ci["high"] <= 0.0
            acceptable_positive_ci = target_ci["low"] >= 0.0
            passed = bool(target_positive and faithfulness_positive
                          and ci_non_crossing and acceptable_positive_ci)
            audit = {
                "valid": True,
                "target_effect_median": target_ci["estimate"],
                "faithfulness_median": faithfulness_ci["estimate"],
                "target_effect_ci": target_ci,
                "faithfulness_ci_diagnostic": faithfulness_ci,
                "target_effect_positive": target_positive,
                "faithfulness_positive": faithfulness_positive,
                "effect_ci_non_crossing": ci_non_crossing,
                "positive_effect_ci_low_ge_zero": acceptable_positive_ci,
                "passed": passed,
            }
        except (KeyError, TypeError, ValueError) as error:
            audit = {"valid": False, "error": str(error), "passed": False}
            passed = False
        audits[direction] = audit
        pass_count += int(passed)
    return {
        "gate": "A",
        "name": "target_evidence_faithfulness",
        "fixed_direction_count": 8,
        "required_pass_count": 5,
        "direction_pass_count": pass_count,
        "ci_boundary_rule": "non_crossing iff low>=0 or high<=0; with positive median, require low>=0",
        "directions": audits,
        "passed": pass_count >= 5,
    }


def _direct_or_macro(data, direct_names, mapping_names, statistic="mean",
                     proportion_threshold=0.0):
    for name in direct_names:
        if name in data:
            return _finite_scalar(data[name], name), {"source": name, "direct": True}
    values = _mapping_field(data, mapping_names)
    result = deterministic_sequence_macro_bootstrap(
        values, seed=0, samples=1, confidence=0.5, statistic=statistic,
        proportion_threshold=proportion_threshold,
    )
    return result["estimate"], {
        "source": next(name for name in mapping_names if name in data),
        "direct": False,
        "sequence_values": result["sequence_values"],
    }


def evaluate_gate_b(density_data):
    """Evaluate density Gate B without substituting unrelated degradation proxies."""
    try:
        clean_rate, clean_source = _direct_or_macro(
            density_data,
            ("clean_negative_fusion_rate", "clean_rate"),
            ("clean_negative_fusion_by_sequence", "clean_flags_by_sequence"),
            statistic="proportion", proportion_threshold=0.0,
        )
        degradation_available = any(
            name in density_data
            for name in ("median_degradation_increase", "degradation_increase",
                         "degradation_increase_by_sequence")
        )
        if degradation_available:
            degradation, degradation_source = _direct_or_macro(
                density_data,
                ("median_degradation_increase", "degradation_increase"),
                ("degradation_increase_by_sequence",),
                statistic="median",
            )
        else:
            degradation, degradation_source = None, {
                "source": "not_executed",
                "direct": False,
            }
        coverage, coverage_source = _direct_or_macro(
            density_data,
            ("sequence_coverage", "coverage"),
            ("coverage_by_sequence",),
            statistic="mean",
        )
        clean_condition = clean_rate >= 0.03
        degradation_condition = bool(
            degradation_available and degradation is not None and degradation >= 0.05
        )
        density_condition = clean_condition or degradation_condition
        coverage_condition = coverage >= 0.20
        return {
            "gate": "B",
            "name": "negative_fusion_density",
            "valid": True,
            "clean_negative_fusion_rate": clean_rate,
            "median_degradation_increase": degradation,
            "degradation_branch_available": degradation_available,
            "sequence_coverage": coverage,
            "sources": {
                "clean_negative_fusion_rate": clean_source,
                "median_degradation_increase": degradation_source,
                "sequence_coverage": coverage_source,
            },
            "clean_condition": clean_condition,
            "degradation_condition": degradation_condition,
            "density_condition": density_condition,
            "coverage_condition": coverage_condition,
            "thresholds": {"clean_rate": 0.03, "degradation_increase": 0.05,
                           "sequence_coverage": 0.20},
            "passed": bool(density_condition and coverage_condition),
        }
    except (KeyError, TypeError, ValueError) as error:
        return {"gate": "B", "name": "negative_fusion_density", "valid": False,
                "error": str(error), "passed": False}


def _rho_values(payload):
    direct = _mapping_field(
        payload, ("strength_rho_by_sequence", "rho_by_sequence"), required=False
    )
    if direct is not None:
        return direct
    effects = _mapping_field(payload, ("strength_effects_by_sequence",))
    result = {}
    for sequence_name, values in effects.items():
        if isinstance(values, Mapping):
            strengths = list(values.keys())
            effect_values = list(values.values())
        else:
            pairs = list(values)
            strengths = [pair[0] for pair in pairs]
            effect_values = [pair[1] for pair in pairs]
        result[sequence_name] = [spearman_rho(strengths, effect_values)]
    return result


def evaluate_gate_c(direction_data):
    """Pass when at least four directions have median sequence rho >= 0.30."""
    audits = {}
    pass_count = 0
    for direction in PRIMARY_DIRECTIONS:
        payload = _direction_payload(direction_data, direction)
        try:
            rho_by_sequence = _rho_values(payload)
            sequence_rho = sequence_macro_values(rho_by_sequence, statistic="mean")
            median_rho = float(np.median(_finite_array(list(sequence_rho.values()))))
            passed = median_rho >= 0.30
            audit = {"valid": True, "median_sequence_rho": median_rho,
                     "rho_by_sequence": sequence_rho, "passed": bool(passed)}
        except (KeyError, TypeError, ValueError) as error:
            audit = {"valid": False, "error": str(error), "passed": False}
            passed = False
        audits[direction] = audit
        pass_count += int(passed)
    return {
        "gate": "C",
        "name": "strength_monotonicity",
        "fixed_direction_count": 8,
        "required_pass_count": 4,
        "rho_threshold": 0.30,
        "direction_pass_count": pass_count,
        "directions": audits,
        "passed": pass_count >= 4,
    }


def evaluate_gate_d(logo_results):
    """Pass when at least four of five LOGO results are direction-consistent."""
    attributes = logo_results.get("attributes", logo_results)
    audits = {}
    pass_count = 0
    valid = True
    for attribute in SEMANTIC_ATTRIBUTES:
        if attribute not in attributes:
            audits[attribute] = {"valid": False, "error": "missing attribute",
                                 "direction_consistent": False}
            valid = False
            continue
        value = attributes[attribute]
        if isinstance(value, (bool, np.bool_)):
            consistent = bool(value)
            audit = {"valid": True, "direction_consistent": consistent}
        elif isinstance(value, Mapping) and "direction_consistent" in value:
            consistent = bool(value["direction_consistent"])
            audit = dict(value)
            audit["valid"] = True
        else:
            consistent = False
            valid = False
            audit = {"valid": False, "error": "invalid LOGO result",
                     "direction_consistent": False}
        audits[attribute] = audit
        pass_count += int(consistent)
    return {
        "gate": "D",
        "name": "semantic_leave_one_group_out_stability",
        "valid": valid,
        "required_pass_count": 4,
        "attribute_pass_count": pass_count,
        "attributes": audits,
        "passed": bool(valid and pass_count >= 4),
    }


def _recursive_nonfinite_count(value):
    if isinstance(value, Mapping):
        return sum(_recursive_nonfinite_count(item) for item in value.values())
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            return int(np.sum(~np.isfinite(value.astype(float))))
        return 0
    if isinstance(value, (list, tuple)):
        return sum(_recursive_nonfinite_count(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return int(not np.isfinite(value))
    return 0


def evaluate_gate_e(integrity, audited_payload=None):
    """Integrity is fail-closed: all counts zero and replay fraction exactly one."""
    leakage = int(integrity.get("online_label_leakage_count",
                                integrity.get("leakage_count", -1)))
    declared_nonfinite = int(integrity.get("non_finite_count",
                                           integrity.get("nonfinite_count", -1)))
    schema_mismatch = int(integrity.get(
        "schema_mismatch_count", integrity.get("schema_violation_count", -1)
    ))
    replay = integrity.get("replay_pass_fraction", integrity.get("replay_match_rate", None))
    observed_nonfinite = _recursive_nonfinite_count(audited_payload) if audited_payload is not None else 0
    replay_valid = replay is not None and np.isfinite(float(replay))
    replay_value = float(replay) if replay_valid else None
    checks = {
        "online_label_leakage_zero": leakage == 0,
        "declared_nonfinite_zero": declared_nonfinite == 0,
        "observed_nonfinite_zero": observed_nonfinite == 0,
        "schema_mismatch_zero": schema_mismatch == 0,
        "replay_pass_fraction_one": replay_valid and replay_value == 1.0,
    }
    return {
        "gate": "E",
        "name": "protocol_integrity",
        "online_label_leakage_count": leakage,
        "non_finite_count": declared_nonfinite,
        "observed_non_finite_count": observed_nonfinite,
        "schema_mismatch_count": schema_mismatch,
        "schema_violation_count": schema_mismatch,
        "replay_pass_fraction": replay_value,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def evaluate_stage0_gates(direction_data, density_data, logo_results, integrity,
                          seed=0, samples=2000, confidence=0.95):
    """Evaluate Gates A-E and return a complete, non-boolean audit.

    E failure dominates and yields ``INVALID_RUN``.  If E passes but any of
    A-D fails, status is ``STOP_CBE``.  Only all-five pass yields
    ``PASS_STAGE0``.
    """
    observed_payload = {
        "direction_data": direction_data,
        "density_data": density_data,
        "logo_results": logo_results,
    }
    gate_a = evaluate_gate_a(direction_data, seed=seed, samples=samples,
                             confidence=confidence)
    gate_b = evaluate_gate_b(density_data)
    gate_c = evaluate_gate_c(direction_data)
    gate_d = evaluate_gate_d(logo_results)
    gate_e = evaluate_gate_e(integrity, audited_payload=observed_payload)
    gates = {"A": gate_a, "B": gate_b, "C": gate_c, "D": gate_d, "E": gate_e}
    if not gate_e["passed"]:
        status = "INVALID_RUN"
    elif not all(gates[name]["passed"] for name in "ABCD"):
        status = "STOP_CBE"
    else:
        status = "PASS_STAGE0"
    return {
        "schema_version": SCHEMA_VERSION,
        "aggregation": "sequence_macro",
        "status": status,
        "passed": status == "PASS_STAGE0",
        "gate_passes": {name: bool(gate["passed"]) for name, gate in gates.items()},
        "gates": gates,
        "bootstrap": {
            "aggregation": "within_sequence_then_sequence_macro",
            "resampling_unit": "sequence",
            "samples": int(samples),
            "seed": int(seed),
            "confidence": float(confidence),
            "interval": "percentile",
        },
        "decision_precedence": [
            "E failure -> INVALID_RUN",
            "E pass and any A-D failure -> STOP_CBE",
            "all A-E pass -> PASS_STAGE0",
        ],
    }


evaluate_gates = evaluate_stage0_gates

# Small compatibility aliases keep the standalone module ergonomic without
# changing any metric semantics.
xywh_iou = iou_xywh
box_iou_xywh = iou_xywh
center_error_xywh = center_error
project_gt_to_feature_grid = map_gt_to_feature_grid
compute_gt_weights = gt_fractional_cell_weights
compute_evidence_metrics = evidence_metrics
compute_fusion_regret = fusion_regret_metrics
compute_intervention_effect = intervention_effect
compute_faithfulness = intervention_faithfulness
spearman = spearman_rho
calibration_metrics = descriptive_calibration
evaluate_stage0 = evaluate_stage0_gates


__all__ = [
    "SCHEMA_VERSION", "PRIMARY_DIRECTIONS", "SEMANTIC_ATTRIBUTES",
    "iou_xywh", "center_error", "map_gt_to_feature_grid",
    "gt_to_feature_grid_xywh", "fractional_cell_overlap_weights",
    "gt_fractional_cell_weights", "fractional_gt_weights", "stable_softmax",
    "decode_box_at_peak", "evidence_metrics", "evaluator_join_box_iou",
    "fusion_regret", "fusion_regret_metrics", "intervention_effect",
    "intervention_faithfulness", "intervention_metrics", "average_tie_ranks",
    "spearman_rho", "normalized_gt_distribution", "gt_region_nll",
    "brier_score_gt", "quality_ece", "aurc", "descriptive_calibration",
    "sequence_macro_calibration", "sequence_macro_values",
    "deterministic_sequence_macro_bootstrap", "attribute_logo",
    "attribute_leave_one_group_out", "evaluate_gate_a", "evaluate_gate_b",
    "evaluate_gate_c", "evaluate_gate_d", "evaluate_gate_e",
    "evaluate_stage0_gates", "evaluate_gates", "xywh_iou", "box_iou_xywh",
    "center_error_xywh", "project_gt_to_feature_grid", "compute_gt_weights",
    "compute_evidence_metrics", "compute_fusion_regret",
    "compute_intervention_effect", "compute_faithfulness", "spearman",
    "calibration_metrics", "evaluate_stage0",
]
