"""Pure online quarantine controller for approved RMG-QH Stage 1 v3a Q-only."""

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "rmg-stage1-v3a-qonly"
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
REQUIRED_OBSERVATION_KEYS = frozenset({
    "frame_idx", "pred_xywh", "image_shape", "response_entropy",
})


@dataclass(frozen=True)
class QuarantinePolicy:
    support_iou: float
    quarantine_max_entropy: float = 0.50
    probe_offsets: Tuple[int, ...] = (1, 3, 5)
    min_support: int = 2

    def __post_init__(self) -> None:
        if self.quarantine_max_entropy != 0.50:
            raise ValueError("quarantine_max_entropy is fixed at 0.50")
        if self.probe_offsets != (1, 3, 5):
            raise ValueError("probe_offsets are fixed at (1, 3, 5)")
        if self.min_support != 2:
            raise ValueError("min_support is fixed at 2")
        threshold = None if isinstance(self.support_iou, bool) else _finite_float(self.support_iou)
        if threshold is None or not 0.0 <= threshold <= 1.0:
            raise ValueError("support_iou must be finite and in [0, 1]")
        object.__setattr__(self, "support_iou", threshold)


QuarantineConfig = QuarantinePolicy


@dataclass(frozen=True)
class ControllerDecision:
    action: str
    reason: str
    event_id: Optional[str] = None
    source_frame: Optional[int] = None
    effective_frame: Optional[int] = None


@dataclass(frozen=True)
class ProbeEvidence:
    event_id: str
    source_frame: int
    frame_idx: int
    offset: int
    shared_anchor_xywh: Tuple[float, float, float, float]
    active_xywh: Optional[Tuple[float, float, float, float]]
    shadow_xywh: Optional[Tuple[float, float, float, float]]
    active_legal: bool
    active_illegal_reason: Optional[str]
    shadow_legal: bool
    shadow_illegal_reason: Optional[str]
    agreement_iou: Optional[float]
    supports_release: bool


@dataclass(frozen=True)
class PendingCandidate:
    event_id: str
    sequence: str
    source_frame: int
    source_candidate_xywh: Tuple[float, float, float, float]
    admission_entropy: float
    probes: Tuple[ProbeEvidence, ...] = ()


@dataclass(frozen=True)
class QuarantineState:
    pending: Optional[PendingCandidate] = None


ControllerState = QuarantineState


@dataclass(frozen=True)
class FinalizedQuarantine:
    event_id: str
    sequence: str
    source_frame: int
    action: str
    reason: str
    support_count: int
    probe_count: int
    source_candidate_xywh: Tuple[float, float, float, float]
    finalized_frame: int
    effective_frame: Optional[int]


FinalizedDecision = FinalizedQuarantine


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _frame_index(value: Any, name: str = "frame_idx") -> int:
    if isinstance(value, bool):
        raise ValueError("%s must be an integer" % name)
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be an integer" % name) from exc
    try:
        exact = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be an integer" % name) from exc
    if not math.isfinite(exact) or exact != result or result < 0:
        raise ValueError("%s must be a non-negative integer" % name)
    return result


def _box_xywh(box: Any) -> Optional[Tuple[float, float, float, float]]:
    if box is None:
        return None
    try:
        if len(box) != 4:
            return None
        values = tuple(float(value) for value in box)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    if values[2] <= 0.0 or values[3] <= 0.0:
        return None
    return values  # type: ignore[return-value]


def validate_observation(observation: Mapping[str, Any]) -> None:
    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping")
    unknown = sorted(set(observation) - ALLOWED_OBSERVATION_KEYS)
    if unknown:
        raise ValueError("Controller observation contains forbidden fields: " + ", ".join(unknown))
    missing = sorted(REQUIRED_OBSERVATION_KEYS - set(observation))
    if missing:
        raise ValueError("Controller observation is missing fields: " + ", ".join(missing))


def deterministic_event_id(sequence: str, source_frame: int) -> str:
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("sequence must be a non-empty string")
    frame = _frame_index(source_frame, "source_frame")
    return "%s:%06d" % (sequence, frame)


def iou_xywh(first: Sequence[float], second: Sequence[float]) -> float:
    first_box = _box_xywh(first)
    second_box = _box_xywh(second)
    if first_box is None or second_box is None:
        return float("nan")
    ax, ay, aw, ah = first_box
    bx, by, bw, bh = second_box
    intersection_width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    if not math.isfinite(union) or union <= 0.0:
        return float("nan")
    result = intersection / union
    return result if math.isfinite(result) else float("nan")


def triage(observation: Mapping[str, Any], policy: QuarantinePolicy,
           candidate_legal: bool, illegal_reason: Optional[str] = None) -> ControllerDecision:
    validate_observation(observation)
    _frame_index(observation["frame_idx"])
    if not isinstance(candidate_legal, bool):
        raise TypeError("candidate_legal must be bool")
    if not candidate_legal:
        reason = "invalid:%s" % (illegal_reason or "unspecified")
        return ControllerDecision("skip", reason)
    if _box_xywh(observation["pred_xywh"]) is None:
        return ControllerDecision("skip", "invalid:candidate_box")
    entropy = _finite_float(observation["response_entropy"])
    if entropy is None:
        return ControllerDecision("skip", "non_finite_entropy")
    if entropy <= policy.quarantine_max_entropy:
        return ControllerDecision("quarantine", "entropy_qonly_admission")
    return ControllerDecision("skip", "entropy_above_quarantine")


def admit_opportunity(state: QuarantineState, sequence: str,
                      observation: Mapping[str, Any], policy: QuarantinePolicy,
                      candidate_legal: bool,
                      illegal_reason: Optional[str] = None) -> Tuple[QuarantineState, ControllerDecision]:
    validate_observation(observation)
    frame = _frame_index(observation["frame_idx"])
    event_id = deterministic_event_id(sequence, frame)
    if state.pending is not None:
        return state, ControllerDecision(
            "skip", "quarantine_slot_occupied", event_id=event_id, source_frame=frame)
    decision = triage(observation, policy, candidate_legal, illegal_reason)
    decision = replace(decision, event_id=event_id, source_frame=frame,
                       effective_frame=None)
    if decision.action != "quarantine":
        return state, decision
    candidate_box = _box_xywh(observation["pred_xywh"])
    entropy = _finite_float(observation["response_entropy"])
    if candidate_box is None or entropy is None:
        return state, ControllerDecision("skip", "invalid:admission_metadata",
                                         event_id=event_id, source_frame=frame)
    pending = PendingCandidate(
        event_id=event_id,
        sequence=sequence,
        source_frame=frame,
        source_candidate_xywh=candidate_box,
        admission_entropy=entropy,
    )
    return QuarantineState(pending=pending), decision


def due_probe_offsets(state: QuarantineState, frame_idx: int,
                      policy: QuarantinePolicy) -> Tuple[int, ...]:
    frame = _frame_index(frame_idx)
    pending = state.pending
    if pending is None or len(pending.probes) >= len(policy.probe_offsets):
        return ()
    next_offset = policy.probe_offsets[len(pending.probes)]
    return (next_offset,) if pending.source_frame + next_offset == frame else ()


def record_probe(state: QuarantineState, policy: QuarantinePolicy, event_id: str,
                 frame_idx: int, active_xywh: Sequence[float], shadow_xywh: Sequence[float],
                 active_legal: bool, shadow_legal: bool,
                 active_anchor_xywh: Sequence[float], shadow_anchor_xywh: Sequence[float],
                 active_illegal_reason: Optional[str] = None,
                 shadow_illegal_reason: Optional[str] = None) -> Tuple[QuarantineState, ProbeEvidence]:
    pending = state.pending
    if pending is None:
        raise ValueError("no pending quarantine candidate")
    if event_id != pending.event_id:
        raise ValueError("probe event_id does not match pending candidate")
    frame = _frame_index(frame_idx)
    next_index = len(pending.probes)
    if next_index >= len(policy.probe_offsets):
        raise ValueError("all probe offsets are already recorded")
    expected_offset = policy.probe_offsets[next_index]
    if frame != pending.source_frame + expected_offset:
        raise ValueError("probe frame is not the next required offset")
    if any(probe.offset == expected_offset for probe in pending.probes):
        raise ValueError("probe offset is already recorded")
    if not isinstance(active_legal, bool) or not isinstance(shadow_legal, bool):
        raise TypeError("probe legality flags must be bool")

    active_anchor = _box_xywh(active_anchor_xywh)
    shadow_anchor = _box_xywh(shadow_anchor_xywh)
    if active_anchor is None or shadow_anchor is None or active_anchor != shadow_anchor:
        raise ValueError("active and shadow probes must record one identical valid shared anchor")

    active_box = _box_xywh(active_xywh)
    shadow_box = _box_xywh(shadow_xywh)
    agreement = iou_xywh(active_box, shadow_box) if (
        active_box is not None and shadow_box is not None) else float("nan")
    agreement_value = agreement if math.isfinite(agreement) else None
    supports = bool(
        active_legal and shadow_legal
        and active_box is not None and shadow_box is not None
        and agreement_value is not None
        and agreement_value >= policy.support_iou
    )
    evidence = ProbeEvidence(
        event_id=pending.event_id,
        source_frame=pending.source_frame,
        frame_idx=frame,
        offset=expected_offset,
        shared_anchor_xywh=active_anchor,
        active_xywh=active_box,
        shadow_xywh=shadow_box,
        active_legal=active_legal,
        active_illegal_reason=active_illegal_reason,
        shadow_legal=shadow_legal,
        shadow_illegal_reason=shadow_illegal_reason,
        agreement_iou=agreement_value,
        supports_release=supports,
    )
    updated = replace(pending, probes=pending.probes + (evidence,))
    return QuarantineState(pending=updated), evidence


def finalize_quarantine(state: QuarantineState, policy: QuarantinePolicy,
                        event_id: str, frame_idx: int) -> Tuple[QuarantineState, FinalizedQuarantine]:
    pending = state.pending
    if pending is None:
        raise ValueError("no pending quarantine candidate")
    if event_id != pending.event_id:
        raise ValueError("finalization event_id does not match pending candidate")
    frame = _frame_index(frame_idx)
    final_offset = policy.probe_offsets[-1]
    if frame != pending.source_frame + final_offset:
        raise ValueError("quarantine may finalize only at the final probe offset")
    offsets = tuple(probe.offset for probe in pending.probes)
    if offsets != policy.probe_offsets:
        raise ValueError("all probes must be recorded in exact offset order before finalization")
    if pending.probes[-1].frame_idx != frame:
        raise ValueError("final probe frame does not match finalization frame")

    support_count = sum(probe.supports_release for probe in pending.probes)
    release = support_count >= policy.min_support
    result = FinalizedQuarantine(
        event_id=pending.event_id,
        sequence=pending.sequence,
        source_frame=pending.source_frame,
        action="release" if release else "discard",
        reason="probe_support_pass" if release else "insufficient_probe_support",
        support_count=support_count,
        probe_count=len(pending.probes),
        source_candidate_xywh=pending.source_candidate_xywh,
        finalized_frame=frame,
        effective_frame=pending.source_frame + final_offset + 1 if release else None,
    )
    return QuarantineState(), result
