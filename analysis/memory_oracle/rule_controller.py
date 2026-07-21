"""Online-only rule controller for RMG-Track Stage 1."""

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np


SCHEMA_VERSION = "rmg-stage1-v1"
ALLOWED_OBSERVATION_KEYS = frozenset({
    "frame_idx",
    "pred_xywh",
    "search_anchor_xywh",
    "image_shape",
    "response_peak",
    "response_entropy",
    "response_margin",
    "response_topk_score_std",
    "response_topk_box_dispersion",
    "template_age",
})


@dataclass(frozen=True)
class RuleThresholds:
    max_entropy: float
    max_motion_residual: float


@dataclass
class ControllerState:
    predictions: list = field(default_factory=list)
    last_commit_frame: int = 0


@dataclass(frozen=True)
class RuleDecision:
    action: str
    reason: str
    motion_residual: Optional[float]


def validate_observation(observation):
    unknown = sorted(set(observation) - ALLOWED_OBSERVATION_KEYS)
    if unknown:
        raise ValueError("Controller observation contains forbidden fields: " + ", ".join(unknown))
    missing = sorted({"frame_idx", "pred_xywh", "image_shape", "response_entropy"}
                     - set(observation))
    if missing:
        raise ValueError("Controller observation is missing fields: " + ", ".join(missing))


def _center(box):
    x, y, width, height = [float(value) for value in box[:4]]
    return np.asarray([x + 0.5 * width, y + 0.5 * height], dtype=float)


def kinematic_residual(previous_boxes, current_box):
    if len(previous_boxes) < 2:
        return None
    older = np.asarray(previous_boxes[-2][:4], dtype=float)
    previous = np.asarray(previous_boxes[-1][:4], dtype=float)
    current = np.asarray(current_box[:4], dtype=float)
    if (not np.isfinite(np.concatenate((older, previous, current))).all()
            or np.any(older[2:] <= 0) or np.any(previous[2:] <= 0)
            or np.any(current[2:] <= 0)):
        return float("inf")
    expected_center = _center(previous) + (_center(previous) - _center(older))
    diagonal = max(float(np.hypot(previous[2], previous[3])), np.finfo(float).eps)
    center_residual = float(np.linalg.norm(_center(current) - expected_center) / diagonal)
    scale_residual = float(np.max(np.abs(np.log(current[2:] / previous[2:]))))
    return max(center_residual, scale_residual)


def observe_prediction(state, pred_xywh):
    values = [float(value) for value in pred_xywh[:4]]
    state.predictions.append(values)
    if len(state.predictions) > 3:
        del state.predictions[:-3]


def decide(observation, state, thresholds, candidate_valid, invalid_reason=None):
    validate_observation(observation)
    motion = kinematic_residual(state.predictions, observation["pred_xywh"])
    if not candidate_valid:
        return RuleDecision("skip", f"invalid:{invalid_reason}", motion)
    if motion is None:
        return RuleDecision("skip", "insufficient_motion_history", None)
    entropy = float(observation["response_entropy"])
    if not math.isfinite(entropy):
        return RuleDecision("skip", "non_finite_entropy", motion)
    if entropy > float(thresholds.max_entropy):
        return RuleDecision("skip", "entropy_above_threshold", motion)
    if motion > float(thresholds.max_motion_residual):
        return RuleDecision("skip", "motion_above_threshold", motion)
    return RuleDecision("update", "rule_pass", motion)


def threshold_quality(rows, max_entropy, max_motion_residual=None, good_iou=0.7, bad_iou=0.1):
    eligible = []
    for row in rows:
        if not row.get("candidate_valid", False):
            continue
        entropy = float(row["response_entropy"])
        motion = row.get("motion_residual")
        if not math.isfinite(entropy) or entropy > float(max_entropy):
            continue
        if max_motion_residual is not None:
            if motion is None or not math.isfinite(float(motion)):
                continue
            if float(motion) > float(max_motion_residual):
                continue
        eligible.append(row)
    committed = len(eligible)
    good = sum(float(row["evaluation_iou"]) >= float(good_iou) for row in eligible)
    bad = sum(float(row["evaluation_iou"]) <= float(bad_iou) for row in eligible)
    return {
        "committed": committed,
        "precision": float(good / committed) if committed else 0.0,
        "bad_update_rate": float(bad / committed) if committed else 0.0,
    }


def select_widest_threshold(rows, field, candidates, fixed_entropy=None,
                            min_precision=0.7, max_bad_rate=0.05,
                            good_iou=0.7, bad_iou=0.1, min_commits=1,
                            min_sequences=1, min_coverage=0.0):
    if field not in ("entropy", "motion"):
        raise ValueError("field must be entropy or motion")
    audit = []
    for candidate in sorted({float(value) for value in candidates}):
        entropy = candidate if field == "entropy" else fixed_entropy
        if entropy is None:
            raise ValueError("fixed_entropy is required when selecting motion")
        motion = candidate if field == "motion" else None
        quality = threshold_quality(rows, entropy, motion, good_iou, bad_iou)
        committed_sequences = len({
            str(row["sequence"]) for row in rows
            if row.get("candidate_valid", False)
            and float(row["response_entropy"]) <= float(entropy)
            and (motion is None or (row.get("motion_residual") is not None
                 and float(row["motion_residual"]) <= float(motion)))
        })
        valid_opportunities = sum(row.get("candidate_valid", False) for row in rows)
        coverage = (float(quality["committed"] / valid_opportunities)
                    if valid_opportunities else 0.0)
        quality.update({
            "threshold": candidate,
            "committed_sequences": committed_sequences,
            "coverage": coverage,
            "passes": (
                quality["committed"] >= int(min_commits)
                and committed_sequences >= int(min_sequences)
                and coverage >= float(min_coverage)
                and quality["precision"] >= float(min_precision)
                and quality["bad_update_rate"] <= float(max_bad_rate)
            ),
        })
        audit.append(quality)
    passing = [item for item in audit if item["passes"]]
    return (max(item["threshold"] for item in passing) if passing else None), audit
