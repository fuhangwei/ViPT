"""Read-only ViPT Stage 0 probes for CBE diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


NEUTRAL_VALUE = (124, 116, 104)
_PROBE_NAMES = ("factual", "rgb_retained", "tir_retained")


@dataclass(frozen=True)
class TrackerStateFingerprint:
    """Exact digest of mutable tracker state relevant to a probe."""

    state: Any
    frame_id: Any
    active_template_id: str | None
    active_template_tensor_hash: str
    active_template_mask_hash: str


def _snapshot_field(snapshot: Any, name: str, default: Any = None) -> Any:
    if isinstance(snapshot, Mapping):
        return snapshot.get(name, default)
    return getattr(snapshot, name, default)


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return ("float", "nan")
        if math.isinf(number):
            return ("float", "inf" if number > 0 else "-inf")
        return ("float", number.hex())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _stable_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_stable_value(item) for item in value)
    if isinstance(value, np.ndarray) or hasattr(value, "detach"):
        return ("array", _array_hash(value))
    return (type(value).__qualname__, repr(value))


def _array_hash(value: Any) -> str:
    digest = hashlib.sha256()
    if value is None:
        digest.update(b"none")
        return digest.hexdigest()
    if hasattr(value, "detach"):
        import torch

        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
        return digest.hexdigest()
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot be fingerprinted")
    contiguous = np.ascontiguousarray(array)
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def tracker_state_fingerprint(tracker: Any) -> TrackerStateFingerprint:
    """Fingerprint state, frame index, and the active template data."""
    active = getattr(tracker, "active_template_snapshot", None)
    template = getattr(tracker, "z_tensor", _snapshot_field(active, "z_tensor"))
    mask = getattr(tracker, "box_mask_z", _snapshot_field(active, "box_mask_z"))
    template_id = _snapshot_field(active, "template_id")
    return TrackerStateFingerprint(
        state=_stable_value(getattr(tracker, "state", None)),
        frame_id=_stable_value(getattr(tracker, "frame_id", None)),
        active_template_id=None if template_id is None else str(template_id),
        active_template_tensor_hash=_array_hash(template),
        active_template_mask_hash=_array_hash(mask),
    )


def _require_unchanged(tracker: Any, before: TrackerStateFingerprint, operation: str) -> None:
    after = tracker_state_fingerprint(tracker)
    if after != before:
        raise RuntimeError(
            f"{operation} modified tracker state: before={before!r}, after={after!r}"
        )


def _six_channel_image(image: Any, name: str = "image") -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 6:
        raise ValueError(f"{name} must be a uint8 HxWx6 array")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{name} must have positive spatial dimensions")
    return array


def _bbox_xywh(value: Sequence[float], name: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 4:
        raise ValueError(f"{name} must contain at least four values")
    box = array[:4]
    if not np.isfinite(box).all() or box[2] <= 0.0 or box[3] <= 0.0:
        raise ValueError(f"{name} must be finite with positive width and height")
    return [float(item) for item in box]


def _cpu_numpy(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    result = np.array(array, copy=True)
    if not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite numeric values")
    return result


def _finite_scalar(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _anchor_id(anchor: Sequence[float]) -> str:
    material = ",".join(float(value).hex() for value in anchor).encode("ascii")
    return "anchor:" + hashlib.sha256(material).hexdigest()


def _neutralize(image: np.ndarray, modality: str) -> np.ndarray:
    if modality not in {"rgb", "tir"}:
        raise ValueError("modality must be 'rgb' or 'tir'")
    output = _six_channel_image(image).copy()
    start = 0 if modality == "rgb" else 3
    output[:, :, start:start + 3] = np.asarray(NEUTRAL_VALUE, dtype=np.uint8)
    return output


def _tracking_helpers():
    from lib.train.data.processing_utils import sample_target
    from lib.utils.box_ops import clip_box

    return sample_target, clip_box


class CBEStage0ProbeAdapter:
    """Wrap an initialized ViPTStage0Track without committing probe results."""

    def __init__(
        self,
        tracker: Any,
        frame0: np.ndarray | None = None,
        init_bbox: Sequence[float] | None = None,
    ) -> None:
        if tracker is None:
            raise TypeError("tracker must not be None")
        if (frame0 is None) != (init_bbox is None):
            raise ValueError("frame0 and init_bbox must be supplied together")
        self.tracker = tracker
        self._snapshots: Mapping[str, Any] = MappingProxyType({})
        self.tracker.network.eval()
        if frame0 is not None:
            self.build_initial_snapshots(frame0, init_bbox)

    @property
    def snapshots(self) -> Mapping[str, Any]:
        return self._snapshots

    @property
    def factual_snapshot(self) -> Any:
        return self._get_snapshot("factual")

    @property
    def rgb_retained_snapshot(self) -> Any:
        return self._get_snapshot("rgb_retained")

    @property
    def tir_retained_snapshot(self) -> Any:
        return self._get_snapshot("tir_retained")

    def _get_snapshot(self, name: str) -> Any:
        if name not in self._snapshots:
            raise RuntimeError("initial snapshots have not been built")
        return self._snapshots[name]

    def build_initial_snapshots(
        self,
        frame0: np.ndarray,
        init_bbox: Sequence[float],
    ) -> Mapping[str, Any]:
        """Build the three fixed templates from one frame-zero image and box."""
        image = _six_channel_image(frame0, "frame0")
        bbox = _bbox_xywh(init_bbox, "init_bbox")
        before = tracker_state_fingerprint(self.tracker)
        try:
            images = {
                "factual": image,
                "rgb_retained": _neutralize(image, "tir"),
                "tir_retained": _neutralize(image, "rgb"),
            }
            snapshots = {
                name: self.tracker.build_template_snapshot(
                    images[name], bbox, source=name, source_frame=0
                )
                for name in _PROBE_NAMES
            }
            for name, snapshot in snapshots.items():
                template_id = _snapshot_field(snapshot, "template_id")
                if not isinstance(template_id, str) or not template_id:
                    raise ValueError(f"{name} snapshot has no template_id")
                if _snapshot_field(snapshot, "z_tensor") is None:
                    raise ValueError(f"{name} snapshot has no template tensor")
        finally:
            _require_unchanged(self.tracker, before, "snapshot construction")
        self._snapshots = MappingProxyType(snapshots)
        return self._snapshots

    def predict(self, image: np.ndarray, anchor: Sequence[float], snapshot: Any) -> dict[str, Any]:
        """Run one contextual prediction and return copied CPU evidence maps."""
        source = _six_channel_image(image)
        search_anchor = _bbox_xywh(anchor, "anchor")
        template_id = _snapshot_field(snapshot, "template_id")
        template = _snapshot_field(snapshot, "z_tensor")
        template_mask = _snapshot_field(snapshot, "box_mask_z")
        if not isinstance(template_id, str) or not template_id:
            raise ValueError("snapshot must have a non-empty template_id")
        if template is None:
            raise ValueError("snapshot must have a template tensor")

        tracker = self.tracker
        tracker.network.eval()
        before = tracker_state_fingerprint(tracker)
        try:
            sample_target, clip_box = _tracking_helpers()
            height, width, _ = source.shape
            crop_size = math.ceil(
                math.sqrt(search_anchor[2] * search_anchor[3])
                * tracker.params.search_factor
            )
            crop_x = round(
                search_anchor[0] + 0.5 * search_anchor[2] - 0.5 * crop_size
            )
            crop_y = round(
                search_anchor[1] + 0.5 * search_anchor[3] - 0.5 * crop_size
            )
            x_patch_arr, resize_factor, _ = sample_target(
                source,
                search_anchor,
                tracker.params.search_factor,
                output_sz=tracker.params.search_size,
            )
            resize_factor = _finite_scalar(resize_factor, "resize_factor")
            if resize_factor <= 0.0:
                raise ValueError("resize_factor must be positive")
            search = tracker.preprocessor.process(x_patch_arr)

            import torch

            with torch.no_grad():
                output = tracker.network.forward(
                    template=template,
                    search=search,
                    ce_template_mask=template_mask,
                )
                score_tensor = output["score_map"]
                size_tensor = output["size_map"]
                offset_tensor = output["offset_map"]
                response_tensor = tracker.output_window * score_tensor
                pred_boxes, best_score = tracker.network.box_head.cal_bbox(
                    response_tensor,
                    size_tensor,
                    offset_tensor,
                    return_score=True,
                )
                pred_boxes = pred_boxes.view(-1, 4)
                pred_box = (
                    pred_boxes.mean(dim=0)
                    * tracker.params.search_size
                    / resize_factor
                ).tolist()

            cx_prev = search_anchor[0] + 0.5 * search_anchor[2]
            cy_prev = search_anchor[1] + 0.5 * search_anchor[3]
            cx, cy, box_width, box_height = pred_box
            half_side = 0.5 * tracker.params.search_size / resize_factor
            cx_real = cx + cx_prev - half_side
            cy_real = cy + cy_prev - half_side
            mapped = [
                cx_real - 0.5 * box_width,
                cy_real - 0.5 * box_height,
                box_width,
                box_height,
            ]
            target_bbox = [
                float(value)
                for value in clip_box(mapped, height, width, margin=10)
            ]
            if not np.isfinite(target_bbox).all():
                raise ValueError("target_bbox contains non-finite values")
            score = _finite_scalar(best_score[0][0].item(), "best_score")

            return {
                "score_map": _cpu_numpy(score_tensor, "score_map"),
                "size_map": _cpu_numpy(size_tensor, "size_map"),
                "offset_map": _cpu_numpy(offset_tensor, "offset_map"),
                "hann_response": _cpu_numpy(response_tensor, "hann_response"),
                "response_map": _cpu_numpy(response_tensor, "response_map"),
                "search_patch_hash": _array_hash(x_patch_arr),
                "resize_factor": resize_factor,
                "search_crop_xywh": [
                    float(crop_x), float(crop_y), float(crop_size), float(crop_size)
                ],
                "target_bbox": target_bbox,
                "best_score": score,
                "search_anchor": list(search_anchor),
                "anchor_id": _anchor_id(search_anchor),
                "template_id": template_id,
            }
        finally:
            _require_unchanged(tracker, before, "contextual prediction")

    def run_clean_probe_set(
        self,
        image: np.ndarray,
        anchor: Sequence[float],
    ) -> dict[str, dict[str, Any]]:
        """Run factual and two retained-modality probes from one clean image."""
        source = _six_channel_image(image)
        search_anchor = _bbox_xywh(anchor, "anchor")
        before = tracker_state_fingerprint(self.tracker)
        try:
            images = {
                "factual": source,
                "rgb_retained": _neutralize(source, "tir"),
                "tir_retained": _neutralize(source, "rgb"),
            }
            results = {
                name: self.predict(images[name], search_anchor, self._get_snapshot(name))
                for name in _PROBE_NAMES
            }
        finally:
            _require_unchanged(self.tracker, before, "clean probe set")
        return results

    def advance_factual(
        self,
        image: np.ndarray,
        anchor: Sequence[float] | None = None,
        *,
        bbox_tolerance: float = 1e-5,
        score_tolerance: float = 1e-6,
    ) -> dict[str, Any]:
        """Advance with track only after checking contextual equivalence."""
        source = _six_channel_image(image)
        tracker_anchor = _bbox_xywh(self.tracker.state, "tracker.state")
        search_anchor = tracker_anchor if anchor is None else _bbox_xywh(anchor, "anchor")
        if search_anchor != tracker_anchor:
            raise ValueError("anchor must exactly match tracker.state")
        bbox_tolerance = _finite_scalar(bbox_tolerance, "bbox_tolerance")
        score_tolerance = _finite_scalar(score_tolerance, "score_tolerance")
        if bbox_tolerance < 0.0 or score_tolerance < 0.0:
            raise ValueError("tolerances must be non-negative")

        factual = self.factual_snapshot
        active = getattr(self.tracker, "active_template_snapshot", None)
        active_tensor = getattr(
            self.tracker, "z_tensor", _snapshot_field(active, "z_tensor")
        )
        active_mask = getattr(
            self.tracker, "box_mask_z", _snapshot_field(active, "box_mask_z")
        )
        if (
            _array_hash(active_tensor) != _array_hash(_snapshot_field(factual, "z_tensor"))
            or _array_hash(active_mask) != _array_hash(_snapshot_field(factual, "box_mask_z"))
        ):
            raise RuntimeError("active tracker template does not match the factual snapshot")

        contextual = self.predict(source, search_anchor, factual)
        tracked = self.tracker.track(source)
        tracked_bbox = _bbox_xywh(tracked.get("target_bbox"), "track target_bbox")
        tracked_score = _finite_scalar(tracked.get("best_score"), "track best_score")
        bbox_match = np.allclose(
            tracked_bbox,
            contextual["target_bbox"],
            rtol=0.0,
            atol=bbox_tolerance,
        )
        score_match = math.isclose(
            tracked_score,
            contextual["best_score"],
            rel_tol=0.0,
            abs_tol=score_tolerance,
        )
        if not bbox_match or not score_match:
            raise RuntimeError(
                "tracker.track disagrees with the same-anchor factual prediction: "
                f"bbox_match={bbox_match}, score_match={score_match}"
            )
        return {
            "target_bbox": tracked_bbox,
            "best_score": tracked_score,
            "search_anchor": list(search_anchor),
            "anchor_id": contextual["anchor_id"],
            "template_id": contextual["template_id"],
            "contextual_target_bbox": contextual["target_bbox"],
            "contextual_best_score": contextual["best_score"],
        }


def run_clean_probe_set(
    adapter: CBEStage0ProbeAdapter,
    image: np.ndarray,
    anchor: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """Run a clean probe set through an existing adapter."""
    if not isinstance(adapter, CBEStage0ProbeAdapter):
        raise TypeError("adapter must be a CBEStage0ProbeAdapter")
    return adapter.run_clean_probe_set(image, anchor)


def load_vipt_stage0_class():
    """Load ViPTStage0Track only when construction is explicitly requested."""
    from lib.test.tracker.vipt_stage0 import ViPTStage0Track

    return ViPTStage0Track


__all__ = [
    "NEUTRAL_VALUE",
    "TrackerStateFingerprint",
    "tracker_state_fingerprint",
    "CBEStage0ProbeAdapter",
    "run_clean_probe_set",
    "load_vipt_stage0_class",
]
