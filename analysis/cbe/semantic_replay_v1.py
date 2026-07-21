"""Semantic and integrity replay for the CBE Stage 0 diagnostic protocol.

This module deliberately performs no tracker forward.  It reconstructs protocol
semantics from frozen inputs and audits recorded intervention provenance.  A
report passes only when every individual check passes.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cbe.protocol_v1 import (  # noqa: E402
    OFFICIAL_SCOPE,
    SCHEMA_VERSION,
    ProtocolValidationError,
    atomic_write_json,
    canonical_json_hash,
    deterministic_opportunity_schedule,
    load_json_strict,
    reject_online_labels,
    validate_official_gate_input,
    with_content_hash,
)

CBE_DIR = Path(__file__).resolve().parent
DEFAULT_FIXTURE = CBE_DIR / "fixtures" / "counterfactual_replay_test.json"
DEFAULT_PROTOCOL_DIR = CBE_DIR / "configs" / "v1"
PROTOCOL_FILENAME = "stage0_protocol_v1.json"
REGISTRY_FILENAME = "intervention_registry_v1.json"
REQUIRED_STRENGTHS = (0.25, 0.5, 0.75)
_SHA256_HEX = frozenset("0123456789abcdef")


class ReplayValidationError(ValueError):
    """Raised when an intervention record is not semantically replayable."""


def _check(check_id: str, passed: bool, *, expected: Any = None,
           observed: Any = None, detail: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"check_id": check_id, "passed": bool(passed)}
    if expected is not None:
        result["expected"] = expected
    if observed is not None:
        result["observed"] = observed
    if detail is not None:
        result["detail"] = detail
    return result


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayValidationError(f"{name} must be an object")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReplayValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _SHA256_HEX for character in value)):
        raise ReplayValidationError(f"{name} must be lowercase SHA-256 hex")
    return value


def _float_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReplayValidationError(f"{name} must be an array")
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ReplayValidationError(f"{name} must contain numbers") from exc
    return values


def _registry_directions(registry: Mapping[str, Any]) -> list[str]:
    entries = registry.get("primary_directions")
    if not isinstance(entries, list) or len(entries) != 8:
        raise ReplayValidationError("registry must contain exactly eight primary directions")
    directions = []
    for index, entry in enumerate(entries):
        item = _require_mapping(entry, f"primary_directions[{index}]")
        direction = item.get("direction")
        if not isinstance(direction, str) or not direction:
            raise ReplayValidationError(f"primary_directions[{index}].direction is invalid")
        directions.append(direction)
    if len(set(directions)) != len(directions):
        raise ReplayValidationError("registry primary directions are not unique")
    return directions


def _load_bundle(fixture_path: Path | str, protocol_dir: Path | str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fixture = load_json_strict(fixture_path)
    directory = Path(protocol_dir)
    protocol = load_json_strict(directory / PROTOCOL_FILENAME)
    registry = load_json_strict(directory / REGISTRY_FILENAME)
    for name, value in (("fixture", fixture), ("protocol", protocol), ("registry", registry)):
        if not isinstance(value, dict):
            raise ReplayValidationError(f"{name} root must be an object")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ReplayValidationError(f"{name} schema_version mismatch")
    return fixture, protocol, registry


def deterministic_event_id(sequence_name: str, frame_index: int,
                           opportunity_index: int, direction: str,
                           strengths: Sequence[float],
                           assignment_hash: str) -> str:
    """Return the canonical event ID used by online and intervention artifacts."""
    payload = {
        "assignment_hash": _require_sha256(assignment_hash, "assignment_hash"),
        "direction": str(direction),
        "frame_index": _require_int(frame_index, "frame_index"),
        "opportunity_index": _require_int(opportunity_index, "opportunity_index"),
        "schema_version": SCHEMA_VERSION,
        "sequence_name": str(sequence_name),
        "strengths": _float_list(strengths, "strengths"),
    }
    return canonical_json_hash(payload)


def _schedule_checks(fixture: Mapping[str, Any], protocol: Mapping[str, Any],
                     registry: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    request = _require_mapping(fixture.get("schedule_request"), "schedule_request")
    sequence = _require_mapping(fixture.get("sequence"), "sequence")
    schedule_protocol = _require_mapping(protocol.get("schedule"), "protocol.schedule")
    directions = _registry_directions(registry)
    registry_strengths = _float_list(registry.get("strengths"), "registry.strengths")
    requested_strengths = _float_list(request.get("strengths"), "schedule_request.strengths")
    checks = [
        _check("registry_strengths_complete", registry_strengths == list(REQUIRED_STRENGTHS),
               expected=list(REQUIRED_STRENGTHS), observed=registry_strengths),
        _check("fixture_strengths_complete", requested_strengths == list(REQUIRED_STRENGTHS),
               expected=list(REQUIRED_STRENGTHS), observed=requested_strengths),
        _check("fixture_directions_match_registry", request.get("directions") == directions,
               expected=directions, observed=request.get("directions")),
    ]
    bindings = {
        "warmup_frames": "warmup_frames",
        "interval_frames": "interval_frames",
        "max_opportunities_per_sequence": "max_opportunities_per_sequence",
        "assignment_seed": "assignment_seed",
    }
    for request_key, protocol_key in bindings.items():
        checks.append(_check(
            f"schedule_parameter_{request_key}",
            request.get(request_key) == schedule_protocol.get(protocol_key),
            expected=schedule_protocol.get(protocol_key), observed=request.get(request_key),
        ))
    actual = deterministic_opportunity_schedule(
        str(sequence.get("sequence_name")),
        _require_int(sequence.get("num_frames"), "sequence.num_frames"),
        directions,
        registry_strengths,
        warmup=_require_int(request.get("warmup_frames"), "warmup_frames"),
        interval=_require_int(request.get("interval_frames"), "interval_frames", minimum=1),
        max_opportunities=_require_int(request.get("max_opportunities_per_sequence"), "max_opportunities_per_sequence"),
        seed=_require_int(request.get("assignment_seed"), "assignment_seed"),
    )
    regenerated = deterministic_opportunity_schedule(
        str(sequence.get("sequence_name")), int(sequence.get("num_frames")), directions,
        registry_strengths, warmup=int(request.get("warmup_frames")),
        interval=int(request.get("interval_frames")),
        max_opportunities=int(request.get("max_opportunities_per_sequence")),
        seed=int(request.get("assignment_seed")),
    )
    checks.append(_check("schedule_deterministic_recomputation", actual == regenerated,
                         expected=canonical_json_hash(actual), observed=canonical_json_hash(regenerated)))
    fixture_schedule = fixture.get("expected_schedule")
    checks.append(_check(
        "fixture_schedule_exact_current_format",
        isinstance(fixture_schedule, list) and fixture_schedule == actual,
        expected=fixture_schedule, observed=actual,
    ))
    checks.append(_check(
        "schedule_events_carry_all_strengths",
        all(event.get("strengths") == list(REQUIRED_STRENGTHS) and "strength" not in event for event in actual),
        expected={"strengths": list(REQUIRED_STRENGTHS), "legacy_strength_absent": True},
        observed=[{"strengths": event.get("strengths"), "legacy_strength_present": "strength" in event} for event in actual],
    ))
    return actual, checks


def _canonical_case_checks(fixture: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = []
    cases = fixture.get("canonical_replay_cases")
    if not isinstance(cases, list) or not cases:
        return [_check("canonical_cases_present", False, detail="no canonical replay cases")]
    for index, raw in enumerate(cases):
        case = _require_mapping(raw, f"canonical_replay_cases[{index}]")
        left_hash = canonical_json_hash(case.get("left"))
        right_hash = canonical_json_hash(case.get("right"))
        expected = bool(case.get("expect_same_hash"))
        checks.append(_check(
            f"canonical:{case.get('case_id', index)}", (left_hash == right_hash) == expected,
            expected={"same_hash": expected},
            observed={"left_hash": left_hash, "right_hash": right_hash, "same_hash": left_hash == right_hash},
        ))
    return checks


def _online_firewall_checks(fixture: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = _require_mapping(fixture.get("online_firewall_cases"), "online_firewall_cases")
    checks = []
    for group_name in ("allowed", "forbidden"):
        cases = groups.get(group_name)
        if not isinstance(cases, list):
            raise ReplayValidationError(f"online_firewall_cases.{group_name} must be an array")
        for index, raw in enumerate(cases):
            case = _require_mapping(raw, f"online_firewall_cases.{group_name}[{index}]")
            accepted = True
            error = None
            try:
                reject_online_labels(case.get("payload"))
            except (ProtocolValidationError, ValueError) as exc:
                accepted = False
                error = str(exc)
            expected = bool(case.get("expect_accept")) if "expect_accept" in case else not bool(case.get("expect_reject"))
            checks.append(_check(
                f"online_firewall:{group_name}:{index}", accepted == expected,
                expected={"accepted": expected}, observed={"accepted": accepted, "error": error},
            ))
    return checks


def _scope_firewall_checks(fixture: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = fixture.get("scope_firewall_cases")
    if not isinstance(cases, list):
        raise ReplayValidationError("scope_firewall_cases must be an array")
    checks = []
    for index, raw in enumerate(cases):
        case = _require_mapping(raw, f"scope_firewall_cases[{index}]")
        accepted = True
        error = None
        try:
            validate_official_gate_input(_require_mapping(case.get("payload"), "scope payload"))
        except (ProtocolValidationError, ValueError) as exc:
            accepted = False
            error = str(exc)
        expected = bool(case.get("expect_accept"))
        checks.append(_check(
            f"scope_firewall:{index}", accepted == expected,
            expected={"accepted": expected}, observed={"accepted": accepted, "error": error},
        ))
    return checks


def _first(record: Mapping[str, Any], names: Sequence[str], name: str) -> Any:
    for field in names:
        if field in record:
            return record[field]
    raise ReplayValidationError(f"missing {name}; expected one of {list(names)}")


def replay_intervention_record(record: Mapping[str, Any], *,
                               scheduled_event: Mapping[str, Any] | None = None,
                               expected_source_frame: int | None = None,
                               expected_anchor_xywh: Sequence[float] | None = None,
                               expected_template_id: str | None = None,
                               expected_target_pixel_count: int | None = None,
                               expected_background_pixel_count: int | None = None,
                               expected_background_offset_yx: Sequence[int] | None = None,
                               expected_seed_by_strength: Mapping[float, int] | None = None) -> dict[str, Any]:
    """Audit one recorded intervention against its deterministic event.

    The record must bind event identity, direction, all three strength arms,
    equal nonzero target/background pixel counts, the deterministic background
    translation, per-arm seed, source frame, factual search anchor, and factual
    template identity.  Aliases are accepted only where old raw writers used a
    noun-first spelling; ambiguity is rejected by requiring one resolved value.
    """
    item = _require_mapping(record, "intervention record")
    event = item if scheduled_event is None else _require_mapping(scheduled_event, "scheduled_event")
    strengths = _float_list(_first(event, ("strengths",), "scheduled strengths"), "scheduled strengths")
    assignment_hash = _require_sha256(_first(event, ("assignment_hash",), "assignment_hash"), "assignment_hash")
    expected_event_id = deterministic_event_id(
        str(_first(event, ("sequence_name", "sequence"), "sequence_name")),
        _require_int(_first(event, ("frame_index", "frame_idx"), "frame_index"), "frame_index"),
        _require_int(_first(event, ("opportunity_index",), "opportunity_index"), "opportunity_index"),
        str(_first(event, ("direction",), "direction")), strengths, assignment_hash,
    )
    checks = []
    checks.append(_check("event_id", item.get("event_id") == expected_event_id,
                         expected=expected_event_id, observed=item.get("event_id")))
    checks.append(_check("direction", item.get("direction") == event.get("direction"),
                         expected=event.get("direction"), observed=item.get("direction")))
    observed_strengths = _float_list(_first(item, ("strengths",), "record strengths"), "record strengths")
    checks.append(_check("three_strengths", observed_strengths == list(REQUIRED_STRENGTHS) == strengths,
                         expected=list(REQUIRED_STRENGTHS), observed=observed_strengths))
    arms = _first(item, ("strength_arms", "arms", "interventions"), "strength arms")
    if isinstance(arms, Mapping):
        arm_rows = list(arms.values())
    elif isinstance(arms, list):
        arm_rows = arms
    else:
        raise ReplayValidationError("strength arms must be an object or array")
    arm_strengths = sorted(float(_require_mapping(arm, "strength arm").get("strength")) for arm in arm_rows)
    checks.append(_check("strength_arm_coverage", arm_strengths == list(REQUIRED_STRENGTHS),
                         expected=list(REQUIRED_STRENGTHS), observed=arm_strengths))

    target_count = _require_int(_first(item, ("target_mask_pixel_count", "target_pixel_count"), "target mask pixel count"), "target mask pixel count", minimum=1)
    background_count = _require_int(_first(item, ("background_mask_pixel_count", "background_pixel_count"), "background mask pixel count"), "background mask pixel count", minimum=1)
    expected_target_count = target_count if expected_target_pixel_count is None else _require_int(
        expected_target_pixel_count, "expected target pixel count", minimum=1)
    expected_background_count = background_count if expected_background_pixel_count is None else _require_int(
        expected_background_pixel_count, "expected background pixel count", minimum=1)
    checks.append(_check(
        "mask_pixel_counts",
        target_count == background_count
        and target_count == expected_target_count
        and background_count == expected_background_count,
        expected={"target": expected_target_count, "background": expected_background_count, "equal": True},
        observed={"target": target_count, "background": background_count},
    ))
    offset = _first(item, ("background_offset_yx", "matched_background_offset_yx"), "background offset")
    valid_offset = (isinstance(offset, (list, tuple)) and len(offset) == 2
                    and all(isinstance(value, int) and not isinstance(value, bool) for value in offset)
                    and any(value != 0 for value in offset))
    expected_offset = list(offset) if expected_background_offset_yx is None else list(expected_background_offset_yx)
    checks.append(_check(
        "background_offset", valid_offset and list(offset) == expected_offset,
        expected=expected_offset, observed=offset,
    ))

    source_frame = _require_int(_first(item, ("source_frame", "source_frame_index"), "source frame"), "source frame")
    anchor = _float_list(_first(item, ("search_anchor_xywh", "anchor_xywh"), "search anchor"), "search anchor")
    template_id = _first(item, ("template_id", "factual_template_id"), "template identity")
    if len(anchor) != 4 or anchor[2] <= 0 or anchor[3] <= 0:
        raise ReplayValidationError("search anchor must be four finite xywh values with positive size")
    if not isinstance(template_id, str) or not template_id:
        raise ReplayValidationError("template identity must be a non-empty string")
    if expected_source_frame is None:
        expected_source_frame = _require_int(_first(event, ("frame_index", "frame_idx"), "event frame"), "event frame")
    if expected_anchor_xywh is None:
        expected_anchor_xywh = _float_list(_first(event, ("search_anchor_xywh", "anchor_xywh"), "event anchor"), "event anchor")
    if expected_template_id is None:
        expected_template_id = str(_first(event, ("template_id", "factual_template_id"), "event template identity"))
    checks.extend([
        _check("source_frame", source_frame == expected_source_frame,
               expected=expected_source_frame, observed=source_frame),
        _check("anchor_identity", anchor == [float(value) for value in expected_anchor_xywh],
               expected=list(expected_anchor_xywh), observed=anchor),
        _check("template_identity", template_id == expected_template_id,
               expected=expected_template_id, observed=template_id),
    ])
    direction_to_spec = {
        "rgb_blur": ("blur", "rgb"),
        "rgb_low_light": ("low_light", "rgb"),
        "rgb_desaturation": ("desaturation", "rgb"),
        "rgb_occlusion": ("opaque_occlusion", "rgb"),
        "tir_contrast_compression": ("contrast_compression", "tir"),
        "tir_saturation": ("saturation_clipping", "tir"),
        "tir_sensor_noise": ("gaussian_sensor_noise", "tir"),
        "tir_blur": ("blur", "tir"),
    }
    if event.get("direction") not in direction_to_spec:
        raise ReplayValidationError("direction is not a primary local intervention")
    operation, modality = direction_to_spec[str(event.get("direction"))]
    arm_checks = []
    for raw_arm in arm_rows:
        arm = _require_mapping(raw_arm, "strength arm")
        strength = float(arm.get("strength"))
        material = "\x1f".join(str(part) for part in (
            expected_event_id, operation, modality, strength
        )).encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)
        external_seed = (
            None if expected_seed_by_strength is None
            else expected_seed_by_strength.get(strength)
        )
        if expected_seed_by_strength is not None and external_seed is None:
            raise ReplayValidationError(f"missing expected seed for strength {strength}")
        observed_seed = arm.get("seed")
        arm_check = _check(
            f"seed:{strength}",
            observed_seed == derived_seed
            and (external_seed is None or external_seed == derived_seed),
            expected={"derived": derived_seed, "external": external_seed},
            observed=observed_seed,
        )
        arm_checks.append(arm_check)
    checks.extend(arm_checks)
    passed_count = sum(bool(check["passed"]) for check in checks)
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "intervention_record_replay",
        "event_id": expected_event_id,
        "check_count": len(checks),
        "passed_count": passed_count,
        "pass_fraction": passed_count / len(checks) if checks else 0.0,
        "passed": bool(checks and passed_count == len(checks)),
        "checks": checks,
    }
    return with_content_hash(report)


def replay_fixture(fixture_path: Path | str = DEFAULT_FIXTURE,
                   protocol_dir: Path | str = DEFAULT_PROTOCOL_DIR) -> dict[str, Any]:
    fixture_path = Path(fixture_path).resolve()
    protocol_dir = Path(protocol_dir).resolve()
    fixture, protocol, registry = _load_bundle(fixture_path, protocol_dir)
    schedule, schedule_checks = _schedule_checks(fixture, protocol, registry)
    groups = {
        "bundle": [
            _check("schema_version", fixture.get("schema_version") == SCHEMA_VERSION,
                   expected=SCHEMA_VERSION, observed=fixture.get("schema_version")),
            _check("synthetic_fixture_only", fixture.get("fixture_kind") == "synthetic_semantic_only"
                   and fixture.get("provenance", {}).get("formal_result") is False,
                   expected={"fixture_kind": "synthetic_semantic_only", "formal_result": False},
                   observed={"fixture_kind": fixture.get("fixture_kind"),
                             "formal_result": fixture.get("provenance", {}).get("formal_result")}),
            _check("official_protocol_scope", protocol.get("scope_policy", {}).get("official_scope") == OFFICIAL_SCOPE,
                   expected=OFFICIAL_SCOPE, observed=protocol.get("scope_policy", {}).get("official_scope")),
        ],
        "schedule": schedule_checks,
        "canonical_hash": _canonical_case_checks(fixture),
        "online_firewall": _online_firewall_checks(fixture),
        "scope_firewall": _scope_firewall_checks(fixture),
    }
    checks = [check for values in groups.values() for check in values]
    passed_count = sum(bool(check["passed"]) for check in checks)
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "semantic_replay_report",
        "fixture_kind": fixture.get("fixture_kind"),
        "formal_result": False,
        "inputs": {
            "fixture": {"path": str(fixture_path), "sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest()},
            "protocol": {"path": str(protocol_dir / PROTOCOL_FILENAME), "sha256": hashlib.sha256((protocol_dir / PROTOCOL_FILENAME).read_bytes()).hexdigest()},
            "registry": {"path": str(protocol_dir / REGISTRY_FILENAME), "sha256": hashlib.sha256((protocol_dir / REGISTRY_FILENAME).read_bytes()).hexdigest()},
        },
        "recomputed_schedule": schedule,
        "groups": groups,
        "check_count": len(checks),
        "passed_count": passed_count,
        "failed_count": len(checks) - passed_count,
        "pass_fraction": passed_count / len(checks) if checks else 0.0,
        "passed": bool(checks and passed_count == len(checks)),
    }
    return with_content_hash(report)


verify_synthetic_fixture = replay_fixture
run_semantic_replay = replay_fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify CBE v1 semantic replay fixture")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-synthetic-fixture", action="store_true",
                        help="Explicitly select the only supported CLI replay mode")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = replay_fixture(args.fixture, args.protocol_dir)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0 if report["passed"] and report["pass_fraction"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
