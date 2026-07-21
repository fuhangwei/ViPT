"""Deterministic pure counterfactual interventions for CBE Stage 0 v1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


_ALLOWED_STRENGTHS = (0.0, 0.25, 0.5, 0.75)
_OPERATIONS = (
    "blur",
    "low_light",
    "desaturation",
    "opaque_occlusion",
    "contrast_compression",
    "saturation_clipping",
    "gaussian_sensor_noise",
    "motion_blur",
)
_MODALITIES = ("rgb", "tir")
INTERVENTION_REGISTRY = (
    "rgb_blur",
    "rgb_low_light",
    "rgb_desaturation",
    "rgb_occlusion",
    "tir_contrast_compression",
    "tir_saturation",
    "tir_sensor_noise",
    "tir_blur",
)
_ALLOWED_MODALITY_OPERATIONS = frozenset({
    ("rgb", "blur"),
    ("rgb", "low_light"),
    ("rgb", "desaturation"),
    ("rgb", "opaque_occlusion"),
    ("tir", "contrast_compression"),
    ("tir", "saturation_clipping"),
    ("tir", "gaussian_sensor_noise"),
    ("tir", "blur"),
})
_DIRECTIONS = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
    "up_left": (-1, -1),
    "up_right": (-1, 1),
    "down_left": (1, -1),
    "down_right": (1, 1),
}


@dataclass(frozen=True)
class InterventionSpec:
    """A registered local intervention with an explicit modality and strength."""

    operation: str
    modality: str
    strength: float
    seed_key: str = ""

    def __post_init__(self) -> None:
        modality = _canonical_modality(self.modality)
        if isinstance(self.operation, str) and self.operation.lower().startswith(("rgb_", "tir_")):
            encoded_modality = self.operation.lower().split("_", 1)[0]
            if encoded_modality != modality:
                raise ValueError(
                    f"operation modality {encoded_modality!r} conflicts with modality {modality!r}"
                )
        operation = _canonical_operation(self.operation)
        if (modality, operation) not in _ALLOWED_MODALITY_OPERATIONS:
            raise ValueError(f"unregistered local intervention: {modality}_{operation}")
        strength = float(self.strength)
        if not any(math.isclose(strength, allowed, rel_tol=0.0, abs_tol=1e-12) for allowed in _ALLOWED_STRENGTHS):
            raise ValueError(f"strength must be one of {_ALLOWED_STRENGTHS}; got {self.strength!r}")
        if not isinstance(self.seed_key, str):
            raise ValueError("seed_key must be a string")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "strength", strength)


@dataclass(frozen=True)
class TargetMaskResult:
    mask: np.ndarray
    expanded_xywh: Tuple[float, float, float, float]
    clipped_xywh: Tuple[float, float, float, float]
    clip_retention: float
    pixel_count: int


@dataclass(frozen=True)
class MatchedBackgroundMaskResult:
    mask: np.ndarray
    offset_yx: Tuple[int, int]
    pixel_count: int


@dataclass(frozen=True)
class PairedInterventionResult:
    """Target-only and matched-background-only variants sharing parameters."""

    target: np.ndarray
    background: np.ndarray
    spec: InterventionSpec
    seed: int
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class TemporalReplacementResult:
    image: np.ndarray
    current_index: int
    source_index: int


def _canonical_modality(modality: str) -> str:
    if not isinstance(modality, str):
        raise ValueError("modality must be 'rgb' or 'tir'")
    value = modality.lower()
    if value not in _MODALITIES:
        raise ValueError(f"modality must be one of {_MODALITIES}; got {modality!r}")
    return value


def _canonical_operation(operation: str) -> str:
    if not isinstance(operation, str):
        raise ValueError("operation must be a string")
    aliases = {
        "occlusion": "opaque_occlusion",
        "contrast": "contrast_compression",
        "saturation": "saturation_clipping",
        "clipping": "saturation_clipping",
        "sensor_noise": "gaussian_sensor_noise",
        "noise": "gaussian_sensor_noise",
        "gaussian_noise": "gaussian_sensor_noise",
        "defocus_blur": "blur",
    }
    value = operation.lower()
    if value.startswith("rgb_") or value.startswith("tir_"):
        prefix, value = value.split("_", 1)
        if prefix != _canonical_modality(prefix):
            raise AssertionError(prefix)
    value = aliases.get(value, value)
    if value not in _OPERATIONS:
        raise ValueError(f"operation must be one of {_OPERATIONS}; got {operation!r}")
    return value


def _as_uint8_image(image: np.ndarray, channels: Optional[int] = None, name: str = "image") -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3:
        raise ValueError(f"{name} must be a uint8 HxWxC array")
    if channels is not None and array.shape[2] != channels:
        raise ValueError(f"{name} must have exactly {channels} channels; got shape {array.shape}")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{name} must have positive spatial dimensions")
    return array


def _uint8_rint(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("intervention produced non-finite values")
    return np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8)


def split_six_channel(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split HxWx6 into independent RGB and TIR HxWx3 copies."""
    array = _as_uint8_image(image, 6, "six-channel image")
    return array[:, :, :3].copy(), array[:, :, 3:].copy()


def merge_six_channel(rgb: np.ndarray, tir: np.ndarray) -> np.ndarray:
    """Merge shape-matched uint8 HxWx3 modalities into a new HxWx6 array."""
    rgb_array = _as_uint8_image(rgb, 3, "rgb")
    tir_array = _as_uint8_image(tir, 3, "tir")
    if rgb_array.shape != tir_array.shape:
        raise ValueError(f"rgb/tir shape mismatch: {rgb_array.shape} vs {tir_array.shape}")
    return np.concatenate((rgb_array, tir_array), axis=2).astype(np.uint8, copy=False)


def _neutral_triplet(neutral: int | Sequence[int]) -> np.ndarray:
    values = np.asarray(neutral)
    if values.ndim == 0:
        values = np.repeat(values, 3)
    if values.shape != (3,) or not np.issubdtype(values.dtype, np.number):
        raise ValueError("neutral must be a scalar or three numeric channel values")
    numeric = values.astype(np.float64)
    if not np.isfinite(numeric).all() or np.any(numeric < 0.0) or np.any(numeric > 255.0):
        raise ValueError("neutral values must be finite and in [0, 255]")
    return _uint8_rint(numeric)


def neutralize_modality(
    image: np.ndarray,
    modality: str,
    neutral: int | Sequence[int] = (124, 116, 104),
) -> np.ndarray:
    """Replace one three-channel modality by a constant without mutating input."""
    output = _as_uint8_image(image, 6, "six-channel image").copy()
    start = 0 if _canonical_modality(modality) == "rgb" else 3
    output[:, :, start:start + 3] = _neutral_triplet(neutral)
    return output


def target_mask_from_xywh(
    image_shape: Sequence[int],
    xywh: Sequence[float],
    *,
    expansion: float = 1.25,
    min_clip_retention: float = 0.9,
    min_pixels: int = 16,
) -> TargetMaskResult:
    """Rasterize an expanded half-open box using pixel-center inclusion.

    A pixel ``(row, col)`` is selected iff ``col + .5`` and ``row + .5`` lie
    inside the clipped half-open box.  ``clip_retention`` is continuous clipped
    area divided by expanded area and is validated before rasterization.
    """
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if len(xywh) < 4:
        raise ValueError("xywh must contain at least four values")
    box = np.asarray(xywh[:4], dtype=np.float64)
    if not np.isfinite(box).all() or box[2] <= 0.0 or box[3] <= 0.0:
        raise ValueError("xywh must be finite with positive width and height")
    expansion = float(expansion)
    retention_threshold = float(min_clip_retention)
    if not math.isfinite(expansion) or expansion <= 0.0:
        raise ValueError("expansion must be finite and positive")
    if not math.isfinite(retention_threshold) or not 0.0 <= retention_threshold <= 1.0:
        raise ValueError("min_clip_retention must be in [0, 1]")
    if not isinstance(min_pixels, int) or isinstance(min_pixels, bool) or min_pixels <= 0:
        raise ValueError("min_pixels must be a positive integer")

    expanded_width, expanded_height = float(box[2] * expansion), float(box[3] * expansion)
    left = float(box[0] + 0.5 * (box[2] - expanded_width))
    top = float(box[1] + 0.5 * (box[3] - expanded_height))
    right, bottom = left + expanded_width, top + expanded_height
    clipped_left, clipped_top = max(0.0, left), max(0.0, top)
    clipped_right, clipped_bottom = min(float(width), right), min(float(height), bottom)
    clipped_width = max(0.0, clipped_right - clipped_left)
    clipped_height = max(0.0, clipped_bottom - clipped_top)
    retention = clipped_width * clipped_height / (expanded_width * expanded_height)
    if retention + 1e-15 < retention_threshold:
        raise ValueError(f"expanded target clip retention {retention:.6f} is below {retention_threshold:.6f}")

    x_centers = np.arange(width, dtype=np.float64) + 0.5
    y_centers = np.arange(height, dtype=np.float64) + 0.5
    x_selected = (x_centers >= clipped_left) & (x_centers < clipped_right)
    y_selected = (y_centers >= clipped_top) & (y_centers < clipped_bottom)
    mask = y_selected[:, None] & x_selected[None, :]
    pixel_count = int(mask.sum())
    if pixel_count < min_pixels:
        raise ValueError(f"target mask has {pixel_count} pixels; minimum is {min_pixels}")
    mask.setflags(write=False)
    return TargetMaskResult(
        mask=mask,
        expanded_xywh=(left, top, expanded_width, expanded_height),
        clipped_xywh=(clipped_left, clipped_top, clipped_width, clipped_height),
        clip_retention=float(retention),
        pixel_count=pixel_count,
    )


def _seed_from_key(*parts: object) -> int:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def _validate_mask(mask: np.ndarray, shape: tuple[int, int], name: str, *, nonempty: bool = True) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype != np.bool_ or array.shape != shape:
        raise ValueError(f"{name} must be a bool mask with shape {shape}; got {array.dtype} {array.shape}")
    if nonempty and not array.any():
        raise ValueError(f"{name} must be non-empty")
    return array


def matched_background_mask(
    target_mask: np.ndarray,
    *,
    seed_key: str,
    valid_support: Optional[np.ndarray] = None,
    candidate_radii: Optional[Sequence[int]] = None,
) -> MatchedBackgroundMaskResult:
    """Find a deterministic, complete integer translation of ``target_mask``.

    Candidate direction and radius order is SHA-256-derived.  No clipping is
    permitted: the translated mask has exactly the target pixel count, remains
    inside ``valid_support``, and has zero overlap with the target.
    """
    target = np.asarray(target_mask)
    if target.dtype != np.bool_ or target.ndim != 2 or not target.any():
        raise ValueError("target_mask must be a non-empty 2-D bool array")
    if not isinstance(seed_key, str) or not seed_key:
        raise ValueError("seed_key must be a non-empty string")
    support = np.ones(target.shape, dtype=bool) if valid_support is None else _validate_mask(valid_support, target.shape, "valid_support")
    if np.any(target & ~support):
        raise ValueError("target_mask lies outside valid_support")
    rows, cols = np.nonzero(target)
    target_height = int(rows.max() - rows.min() + 1)
    target_width = int(cols.max() - cols.min() + 1)
    if candidate_radii is None:
        base = max(1, min(target_height, target_width))
        radii = tuple(range(base, max(target.shape) + 1))
    else:
        if not candidate_radii:
            raise ValueError("candidate_radii must be non-empty")
        radii = tuple(int(radius) for radius in candidate_radii)
        if any(radius <= 0 for radius in radii) or len(set(radii)) != len(radii):
            raise ValueError("candidate_radii must contain unique positive integers")
    allowed_radii = set(radii)
    height, width = target.shape
    candidates = []
    for dy in range(-height + 1, height):
        for dx in range(-width + 1, width):
            if not (dy or dx):
                continue
            radius = max(abs(dy), abs(dx))
            if radius not in allowed_radii:
                continue
            rank = _seed_from_key(seed_key, radius, dy, dx)
            candidates.append((rank, dy, dx))
    candidates.sort()
    expected = int(target.sum())
    for _, dy, dx in candidates:
        shifted_rows, shifted_cols = rows + dy, cols + dx
        if shifted_rows.min() < 0 or shifted_rows.max() >= height or shifted_cols.min() < 0 or shifted_cols.max() >= width:
            continue
        if not support[shifted_rows, shifted_cols].all():
            continue
        translated = np.zeros_like(target)
        translated[shifted_rows, shifted_cols] = True
        if int(translated.sum()) != expected or np.any(translated & target):
            continue
        translated.setflags(write=False)
        return MatchedBackgroundMaskResult(mask=translated, offset_yx=(dy, dx), pixel_count=expected)
    raise ValueError("no complete, non-overlapping matched background translation exists in valid_support")


def _blurred(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def _motion_blurred(image: np.ndarray, length: int, horizontal: bool) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float64)
    if horizontal:
        kernel[length // 2, :] = 1.0 / length
    else:
        kernel[:, length // 2] = 1.0 / length
    return cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT_101)


def _parameters(spec: InterventionSpec, seed: int) -> dict[str, Any]:
    strength = spec.strength
    if spec.operation == "blur":
        return {"sigma": 0.5 + 4.0 * strength}
    if spec.operation == "low_light":
        return {"gain": 1.0 - strength}
    if spec.operation == "desaturation":
        return {"color_fraction": 1.0 - strength}
    if spec.operation == "opaque_occlusion":
        shade = int((_seed_from_key(seed, "shade") % 65) + 32)
        return {"area_fraction": strength, "value": shade}
    if spec.operation == "contrast_compression":
        return {"gain": 1.0 - strength, "center": 127.5}
    if spec.operation == "saturation_clipping":
        return {"gain": 1.0 + 3.0 * strength, "center": 127.5}
    if spec.operation == "gaussian_sensor_noise":
        return {"sigma": 20.4 * strength}
    if spec.operation == "motion_blur":
        length = max(1, int(np.rint(1.0 + 12.0 * strength)))
        if length % 2 == 0:
            length += 1
        return {"length": length, "horizontal": bool(_seed_from_key(seed, "axis") & 1)}
    raise AssertionError(spec.operation)


def _degraded_modality(modality_image: np.ndarray, spec: InterventionSpec, seed: int, params: Mapping[str, Any]) -> np.ndarray:
    if spec.strength == 0.0:
        return modality_image.copy()
    source = modality_image.astype(np.float64)
    if spec.operation == "blur":
        return _as_uint8_image(_blurred(modality_image, float(params["sigma"])), 3)
    if spec.operation == "low_light":
        return _uint8_rint(source * float(params["gain"]))
    if spec.operation == "desaturation":
        # Channels are semantically RGB; do not depend on OpenCV's BGR convention.
        gray = source @ np.asarray([0.299, 0.587, 0.114], dtype=np.float64)
        fraction = float(params["color_fraction"])
        return _uint8_rint(fraction * source + (1.0 - fraction) * gray[:, :, None])
    if spec.operation == "opaque_occlusion":
        return np.full_like(modality_image, int(params["value"]), dtype=np.uint8)
    if spec.operation in {"contrast_compression", "saturation_clipping"}:
        center, gain = float(params["center"]), float(params["gain"])
        return _uint8_rint(center + gain * (source - center))
    if spec.operation == "gaussian_sensor_noise":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, float(params["sigma"]), size=source.shape)
        return _uint8_rint(source + noise)
    if spec.operation == "motion_blur":
        return _as_uint8_image(_motion_blurred(modality_image, int(params["length"]), bool(params["horizontal"])), 3)
    raise AssertionError(spec.operation)


def apply_local_intervention(image: np.ndarray, mask: np.ndarray, spec: InterventionSpec) -> np.ndarray:
    """Apply one registered degradation only inside ``mask``."""
    source = _as_uint8_image(image, 6, "six-channel image")
    selected = _validate_mask(mask, source.shape[:2], "mask")
    if not isinstance(spec, InterventionSpec):
        raise ValueError("spec must be an InterventionSpec")
    if spec.strength == 0.0:
        return source.copy()
    seed = _seed_from_key(spec.seed_key, spec.operation, spec.modality, spec.strength)
    params = _parameters(spec, seed)
    rgb, tir = split_six_channel(source)
    modality = rgb if spec.modality == "rgb" else tir
    degraded = _degraded_modality(modality, spec, seed, params)
    output_modality = modality.copy()
    output_modality[selected] = degraded[selected]
    return merge_six_channel(output_modality, tir) if spec.modality == "rgb" else merge_six_channel(rgb, output_modality)


def _paired_mask_coordinates(target: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_coordinates = np.argwhere(target)
    background_coordinates = np.argwhere(background)
    if target_coordinates.shape != background_coordinates.shape:
        raise ValueError("paired masks must contain equal pixel counts")
    target_order = np.lexsort((target_coordinates[:, 1], target_coordinates[:, 0]))
    background_order = np.lexsort((background_coordinates[:, 1], background_coordinates[:, 0]))
    target_coordinates = target_coordinates[target_order]
    background_coordinates = background_coordinates[background_order]
    offsets = background_coordinates - target_coordinates
    if not np.all(offsets == offsets[0]):
        raise ValueError("background mask must be a complete translation of target mask")
    return target_coordinates, background_coordinates


def _deterministic_subset_indices(count: int, fraction: float, seed: int) -> np.ndarray:
    selected_count = max(1, int(np.rint(count * fraction)))
    ranks = np.asarray([_seed_from_key(seed, "subset", index) for index in range(count)], dtype=np.uint64)
    return np.sort(np.argsort(ranks, kind="mergesort")[:selected_count])


def apply_paired_local_intervention(
    image: np.ndarray,
    target_mask: np.ndarray,
    background_mask: np.ndarray,
    spec: InterventionSpec,
) -> PairedInterventionResult:
    """Create target/background arms with identical parameters and deterministic seed."""
    source = _as_uint8_image(image, 6, "six-channel image")
    target = _validate_mask(target_mask, source.shape[:2], "target_mask")
    background = _validate_mask(background_mask, source.shape[:2], "background_mask")
    if np.any(target & background):
        raise ValueError("target_mask and background_mask must have zero overlap")
    if int(target.sum()) != int(background.sum()):
        raise ValueError("target_mask and background_mask must have exactly equal pixel counts")
    if not isinstance(spec, InterventionSpec):
        raise ValueError("spec must be an InterventionSpec")
    seed = _seed_from_key(spec.seed_key, spec.operation, spec.modality, spec.strength)
    params = _parameters(spec, seed)
    if spec.strength == 0.0:
        return PairedInterventionResult(source.copy(), source.copy(), spec, seed, params)

    rgb, tir = split_six_channel(source)
    modality = rgb if spec.modality == "rgb" else tir
    degraded = _degraded_modality(modality, spec, seed, params)
    target_modality, background_modality = modality.copy(), modality.copy()
    if spec.operation == "opaque_occlusion":
        target_coordinates, background_coordinates = _paired_mask_coordinates(
            target, background
        )
        subset = _deterministic_subset_indices(
            len(target_coordinates), float(params["area_fraction"]), seed
        )
        target_selected = target_coordinates[subset]
        background_selected = background_coordinates[subset]
        target_modality[target_selected[:, 0], target_selected[:, 1]] = degraded[
            target_selected[:, 0], target_selected[:, 1]
        ]
        background_modality[background_selected[:, 0], background_selected[:, 1]] = degraded[
            background_selected[:, 0], background_selected[:, 1]
        ]
    elif spec.operation == "gaussian_sensor_noise":
        target_coordinates, background_coordinates = _paired_mask_coordinates(
            target, background
        )
        rng = np.random.default_rng(seed)
        noise = rng.normal(
            0.0, float(params["sigma"]), size=(len(target_coordinates), 3)
        )
        target_values = modality[target_coordinates[:, 0], target_coordinates[:, 1]].astype(np.float64)
        background_values = modality[background_coordinates[:, 0], background_coordinates[:, 1]].astype(np.float64)
        target_modality[target_coordinates[:, 0], target_coordinates[:, 1]] = _uint8_rint(
            target_values + noise
        )
        background_modality[background_coordinates[:, 0], background_coordinates[:, 1]] = _uint8_rint(
            background_values + noise
        )
    else:
        target_modality[target] = degraded[target]
        background_modality[background] = degraded[background]
    if spec.modality == "rgb":
        target_image = merge_six_channel(target_modality, tir)
        background_image = merge_six_channel(background_modality, tir)
    else:
        target_image = merge_six_channel(rgb, target_modality)
        background_image = merge_six_channel(rgb, background_modality)
    return PairedInterventionResult(target_image, background_image, spec, seed, params)


def global_suppression(
    image: np.ndarray,
    modality: str,
    mode: str,
    *,
    seed_key: str = "",
    blur_sigma: float = 5.0,
    noise_sigma: float = 40.0,
) -> np.ndarray:
    """Apply a deterministic whole-modality mean/blur/noise/zero stress test."""
    source = _as_uint8_image(image, 6, "six-channel image")
    modality = _canonical_modality(modality)
    if mode not in {"mean", "blur", "noise", "zero"}:
        raise ValueError("mode must be one of: mean, blur, noise, zero")
    if not math.isfinite(float(blur_sigma)) or float(blur_sigma) <= 0.0:
        raise ValueError("blur_sigma must be finite and positive")
    if not math.isfinite(float(noise_sigma)) or float(noise_sigma) < 0.0:
        raise ValueError("noise_sigma must be finite and non-negative")
    rgb, tir = split_six_channel(source)
    selected = rgb if modality == "rgb" else tir
    if mode == "mean":
        means = _uint8_rint(selected.astype(np.float64).mean(axis=(0, 1)))
        suppressed = np.broadcast_to(means, selected.shape).copy()
    elif mode == "blur":
        suppressed = _as_uint8_image(_blurred(selected, float(blur_sigma)), 3).copy()
    elif mode == "noise":
        rng = np.random.default_rng(_seed_from_key(seed_key, modality, mode))
        suppressed = _uint8_rint(selected.astype(np.float64) + rng.normal(0.0, float(noise_sigma), selected.shape))
    else:
        suppressed = np.zeros_like(selected)
    return merge_six_channel(suppressed, tir) if modality == "rgb" else merge_six_channel(rgb, suppressed)


def translate_modality(
    image: np.ndarray,
    modality: str,
    percent: float,
    direction: str,
    *,
    neutral: int | Sequence[int] = (124, 116, 104),
) -> np.ndarray:
    """Translate one modality by a fixed direction and neutral-pad exposed pixels."""
    source = _as_uint8_image(image, 6, "six-channel image")
    modality = _canonical_modality(modality)
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {tuple(_DIRECTIONS)}")
    percent = float(percent)
    if not math.isfinite(percent) or percent < 0.0 or percent > 1.0:
        raise ValueError("percent must be finite and in [0, 1]")
    height, width = source.shape[:2]
    unit_y, unit_x = _DIRECTIONS[direction]
    dy = unit_y * int(np.rint(percent * height))
    dx = unit_x * int(np.rint(percent * width))
    rgb, tir = split_six_channel(source)
    selected = rgb if modality == "rgb" else tir
    translated = np.empty_like(selected)
    translated[:] = _neutral_triplet(neutral)
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    if source_y0 < source_y1 and source_x0 < source_x1:
        translated[source_y0 + dy:source_y1 + dy, source_x0 + dx:source_x1 + dx] = selected[source_y0:source_y1, source_x0:source_x1]
    return merge_six_channel(translated, tir) if modality == "rgb" else merge_six_channel(rgb, translated)


def _validate_temporal(
    current: np.ndarray,
    past: np.ndarray,
    current_index: int,
    past_index: int,
    current_sequence: str,
    past_sequence: str,
) -> tuple[np.ndarray, np.ndarray]:
    current_array = _as_uint8_image(current, 6, "current image")
    past_array = _as_uint8_image(past, 6, "past image")
    if current_array.shape != past_array.shape:
        raise ValueError(f"current/past shape mismatch: {current_array.shape} vs {past_array.shape}")
    if not isinstance(current_index, int) or isinstance(current_index, bool) or not isinstance(past_index, int) or isinstance(past_index, bool):
        raise ValueError("current_index and past_index must be integers")
    if past_index < 0 or current_index < 0 or past_index >= current_index:
        raise ValueError("temporal replacement source must be past-only: 0 <= past_index < current_index")
    if not isinstance(current_sequence, str) or not current_sequence or current_sequence != past_sequence:
        raise ValueError("temporal replacement requires the same non-empty sequence")
    return current_array, past_array


def replace_modality_from_past(
    current: np.ndarray,
    past: np.ndarray,
    modality: str,
    *,
    current_index: int,
    past_index: int,
    current_sequence: str,
    past_sequence: str,
) -> TemporalReplacementResult:
    """Replace one full modality with a strictly earlier same-sequence frame."""
    current_array, past_array = _validate_temporal(current, past, current_index, past_index, current_sequence, past_sequence)
    modality = _canonical_modality(modality)
    current_rgb, current_tir = split_six_channel(current_array)
    past_rgb, past_tir = split_six_channel(past_array)
    image = merge_six_channel(past_rgb, current_tir) if modality == "rgb" else merge_six_channel(current_rgb, past_tir)
    return TemporalReplacementResult(image=image, current_index=current_index, source_index=past_index)


def replace_background_from_past(
    current: np.ndarray,
    past: np.ndarray,
    target_mask: np.ndarray,
    *,
    current_index: int,
    past_index: int,
    current_sequence: str,
    past_sequence: str,
    modality: Optional[str] = None,
) -> TemporalReplacementResult:
    """Use same-sequence past background while preserving the current target exactly."""
    current_array, past_array = _validate_temporal(current, past, current_index, past_index, current_sequence, past_sequence)
    target = _validate_mask(target_mask, current_array.shape[:2], "target_mask")
    output = current_array.copy()
    background = ~target
    if modality is None:
        output[background] = past_array[background]
    else:
        start = 0 if _canonical_modality(modality) == "rgb" else 3
        channels = output[:, :, start:start + 3]
        past_channels = past_array[:, :, start:start + 3]
        channels[background] = past_channels[background]
    if not np.array_equal(output[target], current_array[target]):
        raise ValueError("internal error: target preservation failed")
    return TemporalReplacementResult(image=output, current_index=current_index, source_index=past_index)


# Compact aliases for protocol call sites.
split_modalities = split_six_channel
merge_modalities = merge_six_channel
build_target_mask = target_mask_from_xywh
build_matched_background_mask = matched_background_mask
apply_local_pair = apply_paired_local_intervention
global_suppress = global_suppression
translate_with_neutral_pad = translate_modality
past_only_temporal_replacement = replace_modality_from_past
same_sequence_past_background_replacement = replace_background_from_past
