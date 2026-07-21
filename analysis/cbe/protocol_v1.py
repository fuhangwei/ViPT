"""CBE Stage 0 diagnostic protocol primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "cbe-stage0-diagnostic-v1"
OFFICIAL_SCOPE = "design83"
SEALED_SCOPES = frozenset({"internal42", "confirm62", "val47"})
PHASE_ORDER = ("preflight", "online", "intervene", "evaluate", "gate", "verify")
_SEQUENCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LABEL_KEYS = frozenset({
    "gt",
    "groundtruth",
    "ground_truth",
    "iou",
    "jaccard",
    "annotation",
    "annotations",
    "label",
    "labels",
    "mask",
    "masks",
    "rle",
    "segmentation",
    "polygon",
    "polygons",
    "coordinates",
    "coords",
})
_LABEL_PARTS = frozenset({
    "groundtruth",
    "ground_truth",
    "iou",
    "jaccard",
    "annotation",
    "label",
    "mask",
    "rle",
    "segmentation",
    "polygon",
})


class ProtocolValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SplitRoleManifest:
    role: str
    records: tuple[tuple[str, str], ...]
    source: Mapping[str, str]
    size_bytes: int
    sha256: str

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.records)

    @property
    def datasets(self) -> tuple[str, ...]:
        return tuple(dataset for _, dataset in self.records)


@dataclass(frozen=True)
class SemanticAttributeManifest:
    groups: tuple[tuple[str, tuple[str, ...]], ...]
    sequence_attributes: tuple[tuple[str, tuple[str, ...]], ...]
    source: Mapping[str, str]
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ManifestHashLock:
    entries: tuple[tuple[str, str], ...]
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return dict(self.entries)


def reject_non_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolValidationError(f"non-finite number at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolValidationError(f"non-string object key at {path}")
            reject_non_finite(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_non_finite(item, f"{path}[{index}]")
        return
    raise ProtocolValidationError(f"unsupported JSON value at {path}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    reject_non_finite(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_hash(value: Any) -> str:
    return canonical_json_hash(value)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ProtocolValidationError(f"non-finite JSON constant: {value}")


def loads_json_strict(payload: str | bytes | bytearray) -> Any:
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolValidationError("JSON is not valid UTF-8") from exc
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProtocolValidationError(f"invalid JSON: {exc}") from exc
    reject_non_finite(value)
    return value


def load_json_strict(
    path: str | os.PathLike[str],
    schema: Any | None = None,
    expected_keys: Iterable[str] | None = None,
) -> Any:
    value = loads_json_strict(Path(path).read_bytes())
    if expected_keys is not None:
        require_exact_keys(value, expected_keys)
    if schema is not None:
        validate_exact_schema(value, schema)
    return value


def require_exact_keys(
    value: Any,
    expected_keys: Iterable[str],
    path: str = "$",
) -> None:
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"expected object at {path}")
    expected = set(expected_keys)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ProtocolValidationError(f"key mismatch at {path}; missing={missing}, extra={extra}")


def validate_exact_schema(value: Any, schema: Any, path: str = "$") -> None:
    if isinstance(schema, dict):
        require_exact_keys(value, schema, path)
        for key, child_schema in schema.items():
            validate_exact_schema(value[key], child_schema, f"{path}.{key}")
        return
    if isinstance(schema, list):
        if not isinstance(value, list):
            raise ProtocolValidationError(f"expected array at {path}")
        if len(schema) > 1:
            raise ProtocolValidationError(f"array schema must have zero or one item at {path}")
        if schema:
            for index, item in enumerate(value):
                validate_exact_schema(item, schema[0], f"{path}[{index}]")
        return
    if isinstance(schema, tuple) and schema and all(isinstance(item, type) for item in schema):
        if isinstance(value, bool) and bool not in schema:
            raise ProtocolValidationError(f"unexpected boolean at {path}")
        if not isinstance(value, schema):
            names = ", ".join(item.__name__ for item in schema)
            raise ProtocolValidationError(f"expected {names} at {path}")
        return
    if isinstance(schema, type):
        if schema is int and isinstance(value, bool):
            raise ProtocolValidationError(f"expected int at {path}")
        if not isinstance(value, schema):
            raise ProtocolValidationError(f"expected {schema.__name__} at {path}")
        return
    if callable(schema):
        if not schema(value):
            raise ProtocolValidationError(f"schema predicate failed at {path}")
        return
    if value != schema:
        raise ProtocolValidationError(f"expected {schema!r} at {path}, got {value!r}")


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")


def reject_online_labels(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolValidationError(f"non-string object key at {path}")
            normalized = _normalized_key(key)
            parts = set(normalized.split("_"))
            future_label = "future" in parts and bool(parts & (_LABEL_PARTS | {"gt", "truth", "labels"}))
            gt_field = normalized == "gt" or normalized.startswith("gt_") or normalized.endswith("_gt")
            coordinate_label = bool(parts & {"coord", "coords", "coordinate", "coordinates"}) and bool(
                parts & {"mask", "annotation", "gt", "truth"}
            )
            embedded_label = any(
                normalized == token
                or normalized.startswith(f"{token}_")
                or normalized.endswith(f"_{token}")
                for token in _LABEL_PARTS
            )
            if normalized in _LABEL_KEYS or future_label or gt_field or coordinate_label or embedded_label:
                raise ProtocolValidationError(f"online label field forbidden at {path}.{key}")
            reject_online_labels(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_online_labels(item, f"{path}[{index}]")
        return
    reject_non_finite(value, path)


def canonical_jsonl_bytes(rows: Iterable[Any]) -> bytes:
    encoded = [canonical_json_bytes(row) for row in rows]
    return b"" if not encoded else b"\n".join(encoded) + b"\n"


def _atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=False, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(destination.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    _atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def atomic_write_jsonl(path: str | os.PathLike[str], rows: Iterable[Any]) -> None:
    _atomic_write_bytes(path, canonical_jsonl_bytes(rows))


def content_hash_payload(value: Mapping[str, Any], field: str = "content_hash") -> dict[str, Any]:
    payload = dict(value)
    payload.pop(field, None)
    return payload


def compute_content_hash(value: Mapping[str, Any], field: str = "content_hash") -> str:
    return canonical_json_hash(content_hash_payload(value, field))


def with_content_hash(value: Mapping[str, Any], field: str = "content_hash") -> dict[str, Any]:
    result = content_hash_payload(value, field)
    result[field] = canonical_json_hash(result)
    return result


def validate_content_hash(value: Mapping[str, Any], field: str = "content_hash") -> str:
    if field not in value or not isinstance(value[field], str):
        raise ProtocolValidationError(f"missing {field}")
    expected = compute_content_hash(value, field)
    if value[field] != expected:
        raise ProtocolValidationError(f"{field} mismatch")
    return expected


def raw_manifest_hash(path: str | os.PathLike[str]) -> str:
    return sha256_file(path)


def _regular_manifest_file(path: str | os.PathLike[str], label: str) -> Path:
    result = Path(path).resolve()
    if not result.is_file() or result.is_symlink():
        raise ProtocolValidationError(f"expected regular non-symlink {label}: {result}")
    return result


def _manifest_source(value: Any, path: str) -> dict[str, str]:
    require_exact_keys(value, {"authority", "artifact_id", "issued_at_utc"}, path)
    result = {}
    for key in ("authority", "artifact_id", "issued_at_utc"):
        item = value[key]
        if not isinstance(item, str) or not item or item != item.strip():
            raise ProtocolValidationError(f"invalid manifest provenance at {path}.{key}")
        result[key] = item
    return result


def read_split_role_manifest(
    path: str | os.PathLike[str],
    *,
    expected_role: str,
    expected_count: int,
    expected_sha256: str | None = None,
) -> SplitRoleManifest:
    file_path = _regular_manifest_file(path, "split role manifest")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count <= 0:
        raise ProtocolValidationError("expected_count must be a positive integer")
    expected_role = validate_sequence_name(expected_role)
    raw = file_path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ProtocolValidationError(f"locked manifest byte drift: {file_path.name}")
    value = loads_json_strict(raw)
    require_exact_keys(
        value,
        {"artifact_type", "count", "role", "schema_version", "sequences", "source"},
    )
    if value["schema_version"] != SCHEMA_VERSION or value["artifact_type"] != "split_role_manifest":
        raise ProtocolValidationError("split role manifest identity mismatch")
    if value["role"] != expected_role:
        raise ProtocolValidationError(
            f"split role mismatch: got {value['role']!r}, expected {expected_role!r}"
        )
    if (not isinstance(value["count"], int) or isinstance(value["count"], bool)
            or value["count"] != expected_count):
        raise ProtocolValidationError(
            f"split role count mismatch for {expected_role}: got {value['count']!r}, expected {expected_count}"
        )
    records = value["sequences"]
    if not isinstance(records, list) or len(records) != expected_count:
        raise ProtocolValidationError(f"split role sequence count mismatch for {expected_role}")
    names = []
    datasets = []
    for index, record in enumerate(records):
        require_exact_keys(record, {"dataset", "name", "ordinal"}, f"$.sequences[{index}]")
        if (not isinstance(record["ordinal"], int) or isinstance(record["ordinal"], bool)
                or record["ordinal"] != index):
            raise ProtocolValidationError(f"split role ordinal mismatch at {expected_role}[{index}]")
        if not isinstance(record["dataset"], str) or not record["dataset"] or record["dataset"] != record["dataset"].strip():
            raise ProtocolValidationError(f"invalid dataset identity at {expected_role}[{index}]")
        names.append(validate_sequence_name(record["name"]))
        datasets.append(record["dataset"])
    if len(set(names)) != len(names):
        raise ProtocolValidationError(f"duplicate sequence in split role manifest: {expected_role}")
    source = _manifest_source(value["source"], "$.source")
    return SplitRoleManifest(
        role=expected_role,
        records=tuple(zip(names, datasets)),
        source=source,
        size_bytes=len(raw),
        sha256=observed_sha256,
    )


def read_semantic_attribute_manifest(
    path: str | os.PathLike[str],
    *,
    design_names: Sequence[str],
    expected_attributes: Sequence[str],
    minimum_group_size: int = 1,
    expected_sha256: str | None = None,
) -> SemanticAttributeManifest:
    file_path = _regular_manifest_file(path, "semantic attribute manifest")
    if not isinstance(minimum_group_size, int) or isinstance(minimum_group_size, bool) or minimum_group_size <= 0:
        raise ProtocolValidationError("minimum_group_size must be a positive integer")
    design = tuple(validate_sequence_name(name) for name in design_names)
    attributes = tuple(validate_sequence_name(name) for name in expected_attributes)
    if not design or len(set(design)) != len(design):
        raise ProtocolValidationError("design names must be unique and non-empty")
    if not attributes or len(set(attributes)) != len(attributes):
        raise ProtocolValidationError("semantic attribute names must be unique and non-empty")
    raw = file_path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ProtocolValidationError(f"locked manifest byte drift: {file_path.name}")
    value = loads_json_strict(raw)
    require_exact_keys(
        value,
        {"artifact_type", "attributes", "groups", "schema_version", "sequences", "source"},
    )
    if value["schema_version"] != SCHEMA_VERSION or value["artifact_type"] != "semantic_attribute_manifest":
        raise ProtocolValidationError("semantic attribute manifest identity mismatch")
    if value["attributes"] != list(attributes):
        raise ProtocolValidationError("semantic attribute taxonomy or order mismatch")
    raw_groups = value["groups"]
    if not isinstance(raw_groups, list) or len(raw_groups) != len(attributes):
        raise ProtocolValidationError("semantic attribute group count mismatch")
    groups: list[tuple[str, tuple[str, ...]]] = []
    design_set = set(design)
    for index, record in enumerate(raw_groups):
        require_exact_keys(record, {"name", "sequences"}, f"$.groups[{index}]")
        name = validate_sequence_name(record["name"])
        if name != attributes[index]:
            raise ProtocolValidationError(f"semantic attribute group order mismatch at index {index}")
        members_value = record["sequences"]
        if not isinstance(members_value, list):
            raise ProtocolValidationError(f"semantic attribute group {name!r} must be a list")
        members = tuple(validate_sequence_name(member) for member in members_value)
        if len(members) < minimum_group_size:
            raise ProtocolValidationError(f"semantic attribute group {name!r} is too small")
        if len(set(members)) != len(members):
            raise ProtocolValidationError(f"semantic attribute group {name!r} contains duplicates")
        if not set(members) <= design_set:
            raise ProtocolValidationError(f"semantic attribute group {name!r} contains non-design sequences")
        groups.append((name, members))
    records = value["sequences"]
    if not isinstance(records, list) or len(records) != len(design):
        raise ProtocolValidationError("semantic sequence record count mismatch")
    sequence_attributes: list[tuple[str, tuple[str, ...]]] = []
    allowed_attributes = set(attributes)
    for index, record in enumerate(records):
        require_exact_keys(record, {"attributes", "name", "ordinal"}, f"$.sequences[{index}]")
        if (not isinstance(record["ordinal"], int) or isinstance(record["ordinal"], bool)
                or record["ordinal"] != index or record["name"] != design[index]):
            raise ProtocolValidationError(f"semantic sequence order mismatch at index {index}")
        raw_memberships = record["attributes"]
        if not isinstance(raw_memberships, list):
            raise ProtocolValidationError(f"semantic memberships must be a list at index {index}")
        memberships = tuple(validate_sequence_name(item) for item in raw_memberships)
        if len(set(memberships)) != len(memberships) or not set(memberships) <= allowed_attributes:
            raise ProtocolValidationError(f"invalid semantic memberships at index {index}")
        sequence_attributes.append((design[index], memberships))
    expected_by_sequence = {
        sequence: tuple(attribute for attribute, members in groups if sequence in members)
        for sequence in design
    }
    if dict(sequence_attributes) != expected_by_sequence:
        raise ProtocolValidationError("semantic group and per-sequence memberships disagree")
    source = _manifest_source(value["source"], "$.source")
    return SemanticAttributeManifest(
        groups=tuple(groups),
        sequence_attributes=tuple(sequence_attributes),
        source=source,
        size_bytes=len(raw),
        sha256=observed_sha256,
    )


def read_manifest_hash_lock(
    path: str | os.PathLike[str],
    *,
    required_filenames: Sequence[str],
    expected_sha256: str | None = None,
) -> ManifestHashLock:
    file_path = _regular_manifest_file(path, "manifest hash lock")
    required = tuple(required_filenames)
    if not required or len(set(required)) != len(required) or tuple(sorted(required)) != required:
        raise ProtocolValidationError("required lock filenames must be unique and sorted")
    raw = file_path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ProtocolValidationError(f"manifest hash lock drift: {file_path.name}")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ProtocolValidationError("manifest hash lock must be ASCII") from exc
    if not text.endswith("\n") or "\r" in text or text.count("\n") != len(required):
        raise ProtocolValidationError("manifest hash lock must use one LF-terminated line per entry")
    entries: list[tuple[str, str]] = []
    for index, line in enumerate(text[:-1].split("\n")):
        parts = line.split("  ")
        if len(parts) != 2:
            raise ProtocolValidationError(f"invalid manifest hash lock line {index + 1}")
        digest, filename = parts
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or filename != Path(filename).name:
            raise ProtocolValidationError(f"invalid manifest hash lock entry at line {index + 1}")
        entries.append((filename, digest))
    if tuple(filename for filename, _ in entries) != required:
        raise ProtocolValidationError("manifest hash lock filenames or order mismatch")
    return ManifestHashLock(
        entries=tuple(entries),
        size_bytes=len(raw),
        sha256=observed_sha256,
    )


def verify_manifest_hash_lock(
    lock: ManifestHashLock,
    manifest_paths: Mapping[str, str | os.PathLike[str]],
) -> None:
    expected = lock.as_dict()
    if set(manifest_paths) != set(expected):
        raise ProtocolValidationError("manifest hash lock coverage mismatch")
    for filename, digest in lock.entries:
        path = _regular_manifest_file(manifest_paths[filename], f"locked manifest {filename}")
        if path.name != filename:
            raise ProtocolValidationError(f"locked manifest filename mismatch: {filename}")
        if sha256_file(path) != digest:
            raise ProtocolValidationError(f"locked manifest byte drift: {filename}")


def protocol_bundle_manifest(paths: Iterable[str | os.PathLike[str]]) -> dict[str, str]:
    resolved = [Path(path) for path in paths]
    names = [path.name for path in resolved]
    if len(names) != len(set(names)):
        raise ProtocolValidationError("protocol bundle filenames must be unique")
    return {path.name: sha256_file(path) for path in sorted(resolved, key=lambda item: item.name)}


def protocol_bundle_hash(paths: Iterable[str | os.PathLike[str]]) -> str:
    return canonical_json_hash(protocol_bundle_manifest(paths))


def validate_phase_parent(child: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
    for document, name in ((child, "child"), (parent, "parent")):
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ProtocolValidationError(f"{name} schema_version is not official")
        if document.get("phase") not in PHASE_ORDER:
            raise ProtocolValidationError(f"unknown {name} phase")
    child_index = PHASE_ORDER.index(str(child["phase"]))
    parent_index = PHASE_ORDER.index(str(parent["phase"]))
    if child_index != parent_index + 1:
        raise ProtocolValidationError("phase parent must be immediately preceding phase")
    if child.get("parent_phase") != parent.get("phase"):
        raise ProtocolValidationError("parent_phase mismatch")
    if child.get("parent_content_hash") != compute_content_hash(parent):
        raise ProtocolValidationError("parent_content_hash mismatch")
    validate_content_hash(parent)


def validate_sequence_name(name: str) -> str:
    if not isinstance(name, str) or not _SEQUENCE_NAME.fullmatch(name):
        raise ProtocolValidationError("unsafe sequence name")
    if name in {".", ".."} or ".." in name or name.startswith(".") or name.endswith("."):
        raise ProtocolValidationError("unsafe sequence name")
    return name


def deterministic_hash_assignment(
    sequence_name: str,
    frame_index: int,
    choices: Sequence[Any],
    namespace: str,
    seed: int = 20260720,
) -> dict[str, Any]:
    validate_sequence_name(sequence_name)
    if not choices:
        raise ProtocolValidationError("assignment choices must not be empty")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ProtocolValidationError("frame_index must be a non-negative integer")
    payload = {
        "frame_index": frame_index,
        "namespace": namespace,
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "sequence_name": sequence_name,
    }
    digest = canonical_json_hash(payload)
    index = int(digest, 16) % len(choices)
    return {"choice": choices[index], "choice_index": index, "assignment_hash": digest}


def deterministic_hash_order(
    sequence_name: str,
    choices: Sequence[Any],
    namespace: str,
    seed: int = 20260720,
) -> list[Any]:
    validate_sequence_name(sequence_name)
    if not choices:
        raise ProtocolValidationError("ordering choices must not be empty")
    decorated = []
    for index, choice in enumerate(choices):
        digest = canonical_json_hash({
            "choice": choice,
            "choice_index": index,
            "namespace": namespace,
            "schema_version": SCHEMA_VERSION,
            "seed": int(seed),
            "sequence_name": sequence_name,
        })
        decorated.append((digest, index, choice))
    return [choice for _, _, choice in sorted(decorated)]


def deterministic_opportunity_frames(
    num_frames: int,
    warmup: int = 15,
    interval: int = 20,
    max_opportunities: int = 8,
) -> list[int]:
    values = (num_frames, warmup, interval, max_opportunities)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ProtocolValidationError("schedule values must be integers")
    if num_frames < 0 or warmup < 0 or interval <= 0 or max_opportunities < 0:
        raise ProtocolValidationError("invalid opportunity schedule")
    return list(range(warmup, num_frames, interval))[:max_opportunities]


def deterministic_opportunity_schedule(
    sequence_name: str,
    num_frames: int,
    directions: Sequence[str],
    strengths: Sequence[float],
    warmup: int = 15,
    interval: int = 20,
    max_opportunities: int = 8,
    seed: int = 20260720,
) -> list[dict[str, Any]]:
    validate_sequence_name(sequence_name)
    if not directions or not strengths:
        raise ProtocolValidationError("directions and strengths must not be empty")
    frozen_strengths = [float(value) for value in strengths]
    reject_non_finite(frozen_strengths)
    if len(set(frozen_strengths)) != len(frozen_strengths) or any(
        value <= 0.0 for value in frozen_strengths
    ):
        raise ProtocolValidationError("strengths must be unique positive values")
    frames = deterministic_opportunity_frames(num_frames, warmup, interval, max_opportunities)
    direction_order = deterministic_hash_order(sequence_name, directions, "direction_order", seed)
    schedule = []
    for ordinal, frame_index in enumerate(frames):
        direction = direction_order[ordinal % len(direction_order)]
        assignment_hash = canonical_json_hash({
            "direction": direction,
            "frame_index": frame_index,
            "namespace": "opportunity",
            "opportunity_index": ordinal,
            "schema_version": SCHEMA_VERSION,
            "seed": int(seed),
            "sequence_name": sequence_name,
            "strengths": frozen_strengths,
        })
        schedule.append({
            "assignment_hash": assignment_hash,
            "direction": direction,
            "frame_index": frame_index,
            "opportunity_index": ordinal,
            "strengths": list(frozen_strengths),
        })
    return schedule


def validate_official_gate_input(value: Mapping[str, Any]) -> None:
    schema_version = value.get("schema_version")
    scope = value.get("scope")
    if schema_version != SCHEMA_VERSION:
        raise ProtocolValidationError("non-official schema cannot enter the gate")
    if scope in SEALED_SCOPES:
        raise ProtocolValidationError("sealed scope cannot enter the gate")
    if scope != OFFICIAL_SCOPE:
        raise ProtocolValidationError("official gate scope must be design83")
    reject_non_finite(value)
