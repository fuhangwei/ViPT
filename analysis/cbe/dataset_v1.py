"""Strict, side-effect-free dataset validation for CBE Stage 0 v1.

The loaders intentionally validate every frame instead of relying on ``zip`` or a
tracking dataset wrapper that may silently truncate mismatched modalities.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePath
import re
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SPLIT_RE = re.compile(r"[\s,]+")


@dataclass(frozen=True)
class ImageManifestEntry:
    """One ordered image and the facts checked while constructing a manifest."""

    relative_path: str
    size_bytes: int
    sha256: str
    shape: Tuple[int, int, int]


@dataclass(frozen=True)
class AnnotationManifestEntry:
    """An annotation file plus normalized ``xywh`` rows."""

    relative_path: str
    size_bytes: int
    sha256: str
    boxes_xywh: Tuple[Tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class SequenceManifest:
    """A fully validated, ordered RGB-T sequence manifest."""

    dataset: str
    sequence: str
    relative_root: str
    visible_images: Tuple[ImageManifestEntry, ...]
    infrared_images: Tuple[ImageManifestEntry, ...]
    visible_annotation: AnnotationManifestEntry
    infrared_annotation: AnnotationManifestEntry

    @property
    def frame_count(self) -> int:
        return len(self.visible_images)

    @property
    def image_shapes(self) -> Tuple[Tuple[int, int, int], ...]:
        return tuple(entry.shape for entry in self.visible_images)


@dataclass(frozen=True)
class AuthoritativeNameManifest:
    """An immutable one-name-per-line split manifest."""

    role: str
    names: Tuple[str, ...]
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class AttributeGroupManifest:
    """Five validated (possibly overlapping) sequence attribute groups."""

    groups: Tuple[Tuple[str, Tuple[str, ...]], ...]
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Tuple[str, ...]]:
        return dict(self.groups)


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of a regular file without loading it all at once."""
    path = Path(path)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _safe_name(name: str, what: str = "sequence") -> str:
    if not isinstance(name, str):
        raise ValueError(f"{what} name must be a string")
    if not name or name != name.strip() or "\x00" in name or _SAFE_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"unsafe {what} name: {name!r}")
    pure = PurePath(name)
    if pure.is_absolute() or len(pure.parts) != 1 or name in {".", ".."}:
        raise ValueError(f"unsafe {what} name: {name!r}")
    if "/" in name or "\\" in name:
        raise ValueError(f"unsafe {what} name: {name!r}")
    return name


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or non-regular {label}: {path}")
    return path


def _sequence_root(dataset_root: Path, sequence: str) -> Path:
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")
    if dataset_root.is_symlink():
        raise ValueError(f"dataset root must not be a symlink: {dataset_root}")
    sequence_root = dataset_root / _safe_name(sequence)
    if not sequence_root.is_dir() or sequence_root.is_symlink():
        raise ValueError(f"sequence directory is missing or unsafe: {sequence_root}")
    if sequence_root.resolve().parent != dataset_root.resolve():
        raise ValueError(f"sequence escapes dataset root: {sequence!r}")
    return sequence_root


def _layout(dataset: str) -> tuple[str, str, str, str, frozenset[str]]:
    canonical = dataset.strip() if isinstance(dataset, str) else ""
    if canonical in {"RGBT234", "LasHeR"}:
        return "visible", "infrared", "visible.txt", "infrared.txt", frozenset({".jpg", ".jpeg", ".png"})
    if canonical == "GTOT":
        return "v", "i", "groundTruth_v.txt", "groundTruth_i.txt", frozenset({".png", ".jpg", ".jpeg"})
    raise ValueError(f"unsupported dataset: {dataset!r}")


def _ordered_image_paths(directory: Path, extensions: frozenset[str], label: str) -> tuple[Path, ...]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"missing or unsafe {label} directory: {directory}")
    paths = []
    for child in directory.iterdir():
        if child.suffix.lower() in extensions:
            if not child.is_file() or child.is_symlink():
                raise ValueError(f"unsafe {label} image: {child}")
            paths.append(child)
    paths.sort(key=lambda path: path.name)
    if len(paths) <= 1:
        raise ValueError(f"{label} must contain more than one decodable image; found {len(paths)}")
    return tuple(paths)


def _decode_image_entry(path: Path, sequence_root: Path, label: str) -> ImageManifestEntry:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode {label} image: {path}")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{label} image is not uint8 HxWx3 after decode: {path} ({image.dtype}, {image.shape})")
    height, width, channels = (int(value) for value in image.shape)
    if height <= 0 or width <= 0:
        raise ValueError(f"decoded {label} image is empty: {path}")
    return ImageManifestEntry(
        relative_path=path.relative_to(sequence_root).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        shape=(height, width, channels),
    )


def _parse_annotation_row(text: str, line_number: int, path: Path, polygon: bool) -> tuple[float, float, float, float]:
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"blank annotation row at {path}:{line_number}")
    tokens = [token for token in _SPLIT_RE.split(stripped) if token]
    if len(tokens) < 4:
        raise ValueError(f"annotation row has fewer than four values at {path}:{line_number}")
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"non-numeric annotation at {path}:{line_number}") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite annotation at {path}:{line_number}")
    if polygon and values.size != 4:
        if values.size < 8 or values.size % 2:
            raise ValueError(f"GTOT polygon must contain an even number of coordinates >= 8 at {path}:{line_number}")
        xs, ys = values[0::2], values[1::2]
        x, y = float(xs.min()), float(ys.min())
        width, height = float(xs.max() - x), float(ys.max() - y)
    else:
        x, y, width, height = (float(value) for value in values[:4])
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"annotation width/height must be positive at {path}:{line_number}")
    return x, y, width, height


def _annotation_entry(path: Path, sequence_root: Path, polygon: bool) -> AnnotationManifestEntry:
    _regular_file(path, "annotation file")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"annotation file is not UTF-8: {path}") from exc
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"annotation file is empty: {path}")
    boxes = tuple(_parse_annotation_row(line, index, path, polygon) for index, line in enumerate(lines, 1))
    return AnnotationManifestEntry(
        relative_path=path.relative_to(sequence_root).as_posix(),
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        boxes_xywh=boxes,
    )


def load_sequence_manifest(dataset_root: Path | str, sequence: str, dataset: str) -> SequenceManifest:
    """Validate and describe one RGBT234, LasHeR, or GTOT sequence.

    All four stream lengths must be identical and greater than one.  Every image
    is decoded, and RGB/TIR dimensions are compared frame by frame; filenames do
    not need to match.
    """
    root = Path(dataset_root)
    sequence_root = _sequence_root(root, sequence)
    visible_dir, infrared_dir, visible_gt, infrared_gt, extensions = _layout(dataset)
    visible_paths = _ordered_image_paths(sequence_root / visible_dir, extensions, "visible")
    infrared_paths = _ordered_image_paths(sequence_root / infrared_dir, extensions, "infrared")
    visible_annotation = _annotation_entry(sequence_root / visible_gt, sequence_root, dataset == "GTOT")
    infrared_annotation = _annotation_entry(sequence_root / infrared_gt, sequence_root, dataset == "GTOT")

    counts = (len(visible_paths), len(infrared_paths), len(visible_annotation.boxes_xywh), len(infrared_annotation.boxes_xywh))
    if len(set(counts)) != 1 or counts[0] <= 1:
        raise ValueError(
            "visible images, infrared images, visible GT, and infrared GT must have exactly equal counts > 1; "
            f"got {counts} for {sequence!r}"
        )

    visible_images = tuple(_decode_image_entry(path, sequence_root, "visible") for path in visible_paths)
    infrared_images = tuple(_decode_image_entry(path, sequence_root, "infrared") for path in infrared_paths)
    for frame_index, (visible, infrared) in enumerate(zip(visible_images, infrared_images)):
        if visible.shape != infrared.shape:
            raise ValueError(
                f"modality shape mismatch at frame {frame_index} of {sequence!r}: "
                f"visible={visible.shape}, infrared={infrared.shape}"
            )

    return SequenceManifest(
        dataset=dataset,
        sequence=sequence,
        relative_root=sequence,
        visible_images=visible_images,
        infrared_images=infrared_images,
        visible_annotation=visible_annotation,
        infrared_annotation=infrared_annotation,
    )


def read_authoritative_name_manifest(
    path: Path | str,
    *,
    role: str,
    expected_role: Optional[str] = None,
    expected_count: Optional[int] = None,
    allowed_names: Optional[Iterable[str]] = None,
) -> AuthoritativeNameManifest:
    """Read a UTF-8, authoritative one-safe-name-per-line split manifest."""
    path = _regular_file(Path(path), "name manifest")
    role = _safe_name(role, "role")
    if expected_role is not None and role != _safe_name(expected_role, "expected role"):
        raise ValueError(f"manifest role mismatch: got {role!r}, expected {expected_role!r}")
    if expected_count is not None and (not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count <= 0):
        raise ValueError("expected_count must be a positive integer")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"name manifest is not UTF-8: {path}") from exc
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"name manifest is empty: {path}")
    names = tuple(_safe_name(line, "sequence") for line in lines)
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"duplicate sequence names in manifest: {duplicates}")
    if expected_count is not None and len(names) != expected_count:
        raise ValueError(f"manifest count mismatch for {role!r}: got {len(names)}, expected {expected_count}")
    if allowed_names is not None:
        allowed = {_safe_name(name, "allowed sequence") for name in allowed_names}
        unknown = sorted(set(names) - allowed)
        if unknown:
            raise ValueError(f"manifest role mismatch for {role!r}; sequences are not allowed in this role: {unknown}")
    return AuthoritativeNameManifest(role=role, names=names, size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in attribute manifest: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant in attribute manifest: {value}")


def _normalize_groups(payload: object) -> Mapping[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("attribute manifest root must be a JSON object")
    if set(payload) == {"groups"}:
        groups = payload["groups"]
        if isinstance(groups, dict):
            return groups
        if isinstance(groups, list):
            normalized: dict[str, object] = {}
            for index, item in enumerate(groups):
                if not isinstance(item, dict) or set(item) != {"name", "sequences"}:
                    raise ValueError(f"attribute group entry {index} must contain exactly name and sequences")
                name = item["name"]
                if not isinstance(name, str) or name in normalized:
                    raise ValueError(f"invalid or duplicate attribute group name at entry {index}")
                normalized[name] = item["sequences"]
            return normalized
        raise ValueError("attribute groups must be an object or a list")
    return payload


def read_attribute_group_manifest(
    path: Path | str,
    design83: Sequence[str] | AuthoritativeNameManifest,
    *,
    min_group_size: int = 1,
) -> AttributeGroupManifest:
    """Read exactly five non-empty attribute groups drawn from ``design83``.

    A sequence may intentionally occur in more than one group.  Duplicate members
    inside one group are rejected because they would inflate its apparent size.
    """
    path = _regular_file(Path(path), "attribute manifest")
    if not isinstance(min_group_size, int) or isinstance(min_group_size, bool) or min_group_size <= 0:
        raise ValueError("min_group_size must be a positive integer")
    design_names = design83.names if isinstance(design83, AuthoritativeNameManifest) else tuple(design83)
    design = {_safe_name(name, "design83 sequence") for name in design_names}
    if len(design) != len(design_names) or not design:
        raise ValueError("design83 must be non-empty and contain unique safe names")
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON attribute manifest: {path}") from exc
    groups = _normalize_groups(payload)
    if len(groups) != 5:
        raise ValueError(f"attribute manifest must contain exactly five groups; got {len(groups)}")
    validated = []
    for raw_name, raw_members in groups.items():
        name = _safe_name(raw_name, "attribute group")
        if not isinstance(raw_members, list) or not all(isinstance(member, str) for member in raw_members):
            raise ValueError(f"attribute group {name!r} must be a JSON list of sequence names")
        members = tuple(_safe_name(member, f"member of {name}") for member in raw_members)
        if len(members) < min_group_size:
            raise ValueError(f"attribute group {name!r} has {len(members)} members; minimum is {min_group_size}")
        if len(set(members)) != len(members):
            raise ValueError(f"attribute group {name!r} contains duplicate members")
        unknown = sorted(set(members) - design)
        if unknown:
            raise ValueError(f"attribute group {name!r} contains sequences outside design83: {unknown}")
        validated.append((name, members))
    return AttributeGroupManifest(groups=tuple(validated), size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


# Explicit aliases keep call sites readable without weakening validation.
load_rgbt_sequence_manifest = load_sequence_manifest
load_authoritative_manifest = read_authoritative_name_manifest
load_attribute_manifest = read_attribute_group_manifest
