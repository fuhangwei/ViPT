import argparse
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import analysis.cbe.run_stage0_diagnostic_v1 as stage0_runner

try:
    import cv2
    from analysis.cbe.dataset_v1 import load_sequence_manifest
    from analysis.cbe.interventions_v1 import (
        InterventionSpec,
        _parameters,
        apply_paired_local_intervention,
        matched_background_mask,
        merge_six_channel,
        neutralize_modality,
        replace_modality_from_past,
        target_mask_from_xywh,
    )
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False
from analysis.cbe.metrics_v1 import (
    PRIMARY_DIRECTIONS,
    SEMANTIC_ATTRIBUTES,
    deterministic_sequence_macro_bootstrap,
    evaluate_gate_b,
    evaluate_stage0_gates,
    evidence_metrics,
    gt_fractional_cell_weights,
    gt_region_nll,
    stable_softmax,
)
from analysis.cbe.protocol_v1 import (
    ProtocolValidationError,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_hash,
    deterministic_opportunity_schedule,
    loads_json_strict,
    read_manifest_hash_lock,
    read_semantic_attribute_manifest,
    read_split_role_manifest,
    reject_online_labels,
    sha256_file,
    verify_manifest_hash_lock,
    validate_phase_parent,
    with_content_hash,
)
from analysis.cbe.run_stage0_diagnostic_v1 import (
    DEFAULT_PROTOCOL_DIR,
    Stage0RunError,
    _aggregate_gate_inputs,
    _authenticate_completion,
    _compute_logo,
    _load_manifest_lock_layer,
    _phase_document,
    _publish_sequence,
    _validated_protocol_bundle,
    derive_design83,
    run_evaluate,
    run_gate,
    run_intervene,
    run_online,
    run_preflight,
    run_verify,
)
from analysis.cbe.semantic_replay_v1 import DEFAULT_FIXTURE, replay_fixture


class ProtocolReplayTest(unittest.TestCase):
    def test_canonical_json_and_strict_parser(self):
        self.assertEqual(
            canonical_json_hash({"b": 2, "a": 1}),
            canonical_json_hash({"a": 1, "b": 2}),
        )
        with self.assertRaises(ProtocolValidationError):
            loads_json_strict('{"a":1,"a":2}')
        with self.assertRaises(ProtocolValidationError):
            loads_json_strict('{"value":NaN}')

    def test_recursive_online_label_firewall(self):
        reject_online_labels({"status": "GT and IoU are words", "confidence": 0.5})
        for payload in (
            {"ground_truth": [1, 2, 3, 4]},
            {"nested": {"future_labels": [1]}},
            {"candidate_mask_rle": "x"},
            {"annotation_coordinates": [1, 2]},
        ):
            with self.assertRaises(ProtocolValidationError):
                reject_online_labels(payload)

    def test_schedule_has_all_same_frame_strengths(self):
        schedule = deterministic_opportunity_schedule(
            "sequence_01", 180, PRIMARY_DIRECTIONS, [0.25, 0.5, 0.75]
        )
        self.assertEqual([row["frame_index"] for row in schedule], list(range(15, 175, 20)))
        self.assertTrue(all(row["strengths"] == [0.25, 0.5, 0.75] for row in schedule))
        self.assertTrue(all("strength" not in row for row in schedule))

    def test_replay_requires_exact_current_fixture(self):
        report = replay_fixture(DEFAULT_FIXTURE, DEFAULT_PROTOCOL_DIR)
        self.assertTrue(report["passed"])
        self.assertEqual(report["pass_fraction"], 1.0)
        checks = report["groups"]["schedule"]
        self.assertTrue(any(row["check_id"] == "fixture_schedule_exact_current_format" for row in checks))
        self.assertFalse(any("legacy" in row["check_id"] for row in checks))
        fixture = json.loads(Path(DEFAULT_FIXTURE).read_text(encoding="utf-8"))
        fixture["expected_schedule"][0]["strength"] = 0.5
        fixture["expected_schedule"][0].pop("strengths")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            tampered = replay_fixture(path, DEFAULT_PROTOCOL_DIR)
        exact = next(
            row for row in tampered["groups"]["schedule"]
            if row["check_id"] == "fixture_schedule_exact_current_format"
        )
        self.assertFalse(exact["passed"])
        self.assertFalse(tampered["passed"])

    def test_frozen_bundle_cross_file_validation(self):
        registry, protocol, metric_schema, gate = _validated_protocol_bundle(DEFAULT_PROTOCOL_DIR)
        expected_attributes = (
            "camera_motion", "low_illumination", "occlusion", "fast_motion",
            "thermal_or_modality_challenge",
        )
        self.assertEqual(SEMANTIC_ATTRIBUTES, expected_attributes)
        self.assertEqual(stage0_runner.SEMANTIC_ATTRIBUTES, expected_attributes)
        self.assertEqual(registry["strengths"], [0.25, 0.5, 0.75])
        self.assertEqual(protocol["phase_chain"], ["preflight", "online", "intervene", "evaluate", "gate", "verify"])
        self.assertEqual(metric_schema["semantic_attributes"], list(expected_attributes))
        self.assertEqual(gate["gates"]["D"]["semantic_attributes"], list(expected_attributes))
        fixture = json.loads(Path(DEFAULT_FIXTURE).read_text(encoding="utf-8"))
        self.assertEqual(
            tuple(fixture["sequence"]["semantic_attributes"]),
            tuple(sorted(expected_attributes)),
        )
        self.assertNotIn("day_night", json.dumps({
            "metric_schema": metric_schema,
            "gate": gate,
            "fixture": fixture,
        }))

    def test_frozen_bundle_rejects_old_semantic_taxonomy(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol_dir = Path(directory)
            for source in DEFAULT_PROTOCOL_DIR.glob("*.json"):
                (protocol_dir / source.name).write_bytes(source.read_bytes())
            metric_path = protocol_dir / "stage0_metric_schema_v1.json"
            metric_schema = json.loads(metric_path.read_text(encoding="utf-8"))
            metric_schema["semantic_attributes"][0] = "day_night"
            metric_path.write_text(json.dumps(metric_schema), encoding="utf-8")
            with self.assertRaisesRegex(Stage0RunError, "semantic attributes differ from taxonomy"):
                _validated_protocol_bundle(protocol_dir)

    def test_frozen_bundle_rejects_intervention_parameter_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol_dir = Path(directory)
            for source in DEFAULT_PROTOCOL_DIR.glob("*.json"):
                (protocol_dir / source.name).write_bytes(source.read_bytes())
            registry_path = protocol_dir / "intervention_registry_v1.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["primary_directions"][0]["values_by_strength"]["0.5"] = 9.0
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaisesRegex(Stage0RunError, "intervention table"):
                _validated_protocol_bundle(protocol_dir)

    def test_phase_parent_is_immediate_and_content_bound(self):
        preflight = _phase_document("preflight", "identity", {"status": "COMPLETE"})
        online = _phase_document("online", "identity", {"status": "COMPLETE"}, preflight)
        validate_phase_parent(online, preflight)
        changed = dict(preflight)
        changed["payload"] = {"status": "CHANGED"}
        with self.assertRaises(ProtocolValidationError):
            validate_phase_parent(online, changed)


@unittest.skipUnless(HAS_OPENCV, "OpenCV is required for image interventions")
class InterventionTest(unittest.TestCase):
    def setUp(self):
        rows, cols = np.indices((24, 32))
        rgb = np.stack((80 + rows, 90 + cols, 100 + (rows + cols) // 2), axis=2).astype(np.uint8)
        tir = np.stack((110 + rows, 120 + cols, 130 + (rows + cols) // 2), axis=2).astype(np.uint8)
        self.image = merge_six_channel(rgb, tir)
        self.target = np.zeros((24, 32), dtype=bool)
        self.target[4:8, 4:8] = True
        self.background = np.zeros_like(self.target)
        self.background[12:16, 20:24] = True

    def test_neutral_retained_probe_channels_and_dtype(self):
        rgb_removed = neutralize_modality(self.image, "rgb")
        tir_removed = neutralize_modality(self.image, "tir")
        self.assertEqual(rgb_removed.dtype, np.uint8)
        self.assertTrue(np.all(rgb_removed[:, :, :3] == np.asarray([124, 116, 104])))
        np.testing.assert_array_equal(rgb_removed[:, :, 3:], self.image[:, :, 3:])
        np.testing.assert_array_equal(tir_removed[:, :, :3], self.image[:, :, :3])
        self.assertTrue(np.all(tir_removed[:, :, 3:] == np.asarray([124, 116, 104])))

    def test_target_and_background_masks_are_exact_translations(self):
        target = target_mask_from_xywh(self.image.shape, [8, 6, 6, 4])
        background = matched_background_mask(target.mask, seed_key="event")
        self.assertEqual(target.pixel_count, background.pixel_count)
        self.assertFalse(np.any(target.mask & background.mask))
        target_coordinates = np.argwhere(target.mask)
        background_coordinates = np.argwhere(background.mask)
        target_coordinates = target_coordinates[np.lexsort((target_coordinates[:, 1], target_coordinates[:, 0]))]
        background_coordinates = background_coordinates[np.lexsort((background_coordinates[:, 1], background_coordinates[:, 0]))]
        offsets = background_coordinates - target_coordinates
        self.assertTrue(np.all(offsets == offsets[0]))

    def test_all_primary_direction_parameters_match_frozen_registry(self):
        registry = json.loads((DEFAULT_PROTOCOL_DIR / "intervention_registry_v1.json").read_text())
        specs = {
            "rgb_blur": ("blur", "rgb"),
            "rgb_low_light": ("low_light", "rgb"),
            "rgb_desaturation": ("desaturation", "rgb"),
            "rgb_occlusion": ("opaque_occlusion", "rgb"),
            "tir_contrast_compression": ("contrast_compression", "tir"),
            "tir_saturation": ("saturation_clipping", "tir"),
            "tir_sensor_noise": ("gaussian_sensor_noise", "tir"),
            "tir_blur": ("blur", "tir"),
        }
        for entry in registry["primary_directions"]:
            operation, modality = specs[entry["direction"]]
            parameter = entry["parameter"]
            implementation_key = {
                "sigma_pixels": "sigma",
                "gain": "gain",
                "retained_color_fraction": "color_fraction",
                "mask_area_fraction": "area_fraction",
            }[parameter]
            for strength in registry["strengths"]:
                spec = InterventionSpec(operation, modality, strength, seed_key="event")
                seed = __import__("analysis.cbe.interventions_v1", fromlist=["_seed_from_key"])._seed_from_key(
                    spec.seed_key, spec.operation, spec.modality, spec.strength
                )
                observed = _parameters(spec, seed)[implementation_key]
                self.assertAlmostEqual(observed, entry["values_by_strength"][str(strength)])

    def test_strength_zero_identity_and_occlusion_area_is_monotonic(self):
        identity = apply_paired_local_intervention(
            self.image, self.target, self.background,
            InterventionSpec("opaque_occlusion", "rgb", 0.0, "event"),
        )
        np.testing.assert_array_equal(identity.target, self.image)
        np.testing.assert_array_equal(identity.background, self.image)
        changed = []
        for strength in (0.25, 0.5, 0.75):
            result = apply_paired_local_intervention(
                self.image, self.target, self.background,
                InterventionSpec("opaque_occlusion", "rgb", strength, "event"),
            )
            target_changed = np.any(result.target[:, :, :3] != self.image[:, :, :3], axis=2)
            background_changed = np.any(result.background[:, :, :3] != self.image[:, :, :3], axis=2)
            self.assertEqual(int(target_changed.sum()), int(background_changed.sum()))
            changed.append(int(target_changed.sum()))
        self.assertEqual(changed, [4, 8, 12])

    def test_noise_uses_same_field_at_translated_coordinates(self):
        result = apply_paired_local_intervention(
            self.image, self.target, self.background,
            InterventionSpec("gaussian_sensor_noise", "tir", 0.5, "event"),
        )
        target_coordinates = np.argwhere(self.target)
        background_coordinates = np.argwhere(self.background)
        target_delta = (
            result.target[target_coordinates[:, 0], target_coordinates[:, 1], 3:].astype(int)
            - self.image[target_coordinates[:, 0], target_coordinates[:, 1], 3:].astype(int)
        )
        background_delta = (
            result.background[background_coordinates[:, 0], background_coordinates[:, 1], 3:].astype(int)
            - self.image[background_coordinates[:, 0], background_coordinates[:, 1], 3:].astype(int)
        )
        np.testing.assert_array_equal(target_delta, background_delta)

    def test_temporal_replacement_is_strictly_past_only(self):
        past = np.zeros_like(self.image)
        result = replace_modality_from_past(
            self.image, past, "rgb", current_index=5, past_index=3,
            current_sequence="seq", past_sequence="seq",
        )
        self.assertEqual(result.source_index, 3)
        with self.assertRaises(ValueError):
            replace_modality_from_past(
                self.image, past, "rgb", current_index=5, past_index=5,
                current_sequence="seq", past_sequence="seq",
            )


class MetricGateTest(unittest.TestCase):
    def test_mass_shares_and_region_mass_nll(self):
        score = np.ones((2, 2), dtype=float)
        window = np.ones((2, 2), dtype=float)
        weights = np.asarray([[1.0, 0.0], [0.0, 0.0]])
        result = evidence_metrics(score, window, weights)
        self.assertEqual(result["raw_gt_weighted_sum"], 1.0)
        self.assertEqual(result["windowed_gt_weighted_sum"], 1.0)
        self.assertEqual(result["raw_gt_mass"], 0.25)
        self.assertEqual(result["windowed_gt_mass"], 0.25)
        self.assertEqual(result["belief_gt_mass"], 0.25)
        self.assertAlmostEqual(gt_region_nll(stable_softmax(score), weights), -math.log(0.25))

    def test_explicit_crop_geometry_controls_gt_mapping(self):
        weights = gt_fractional_cell_weights(
            [10, 10, 10, 10], [11.2, 11.2, 8, 8], 256, 2.0, (16, 16),
            search_crop_xywh=[0, 0, 128, 128],
        )
        self.assertAlmostEqual(float(weights.sum()), 1.5625)
        self.assertGreater(weights[1:3, 1:3].sum(), 0.0)

    def test_sequence_macro_bootstrap_does_not_event_weight(self):
        values = {"long": [1.0] * 100, "short": [0.0]}
        result = deterministic_sequence_macro_bootstrap(
            values, seed=7, samples=200, statistic="mean"
        )
        self.assertEqual(result["estimate"], 0.5)
        self.assertEqual(result["num_sequences"], 2)
        self.assertEqual(result["num_events"], 101)

    def test_gate_b_never_uses_target_effect_as_degradation_proxy(self):
        density = {
            "clean_negative_fusion_by_sequence": {"a": [1.0], "b": [0.0]},
            "coverage_by_sequence": {"a": [1.0], "b": [0.0]},
            "degradation_probe_status": "not_executed",
        }
        result = evaluate_gate_b(density)
        self.assertTrue(result["passed"])
        self.assertFalse(result["degradation_branch_available"])
        self.assertIsNone(result["median_degradation_increase"])

    def test_gate_status_precedence(self):
        direction_data = {}
        for direction in PRIMARY_DIRECTIONS:
            direction_data[direction] = {
                "target_effects_by_sequence": {"s": [0.2, 0.4, 0.6]},
                "faithfulness_by_sequence": {"s": [0.1, 0.2, 0.3]},
                "strength_rho_by_sequence": {"s": [1.0]},
            }
        density = {
            "clean_negative_fusion_by_sequence": {"s": [1.0]},
            "coverage_by_sequence": {"s": [1.0]},
        }
        logo = {"attributes": {name: {"direction_consistent": True} for name in SEMANTIC_ATTRIBUTES}}
        integrity = {"online_label_leakage_count": 0, "non_finite_count": 0, "schema_mismatch_count": 0, "replay_pass_fraction": 1.0}
        passed = evaluate_stage0_gates(direction_data, density, logo, integrity, samples=20)
        self.assertEqual(passed["status"], "PASS_STAGE0")
        invalid = evaluate_stage0_gates(
            direction_data, density, logo,
            {**integrity, "online_label_leakage_count": 1}, samples=20,
        )
        self.assertEqual(invalid["status"], "INVALID_RUN")
        stopped_data = {key: dict(value) for key, value in direction_data.items()}
        for direction in PRIMARY_DIRECTIONS[:4]:
            stopped_data[direction] = {
                **stopped_data[direction],
                "target_effects_by_sequence": {"s": [-0.2, -0.1, -0.05]},
            }
        stopped = evaluate_stage0_gates(stopped_data, density, logo, integrity, samples=20)
        self.assertEqual(stopped["status"], "STOP_CBE")

    def test_logo_recomputes_gates_a_b_c(self):
        events = []
        sequences = [f"s{index}" for index in range(6)]
        for sequence in sequences:
            for direction in PRIMARY_DIRECTIONS:
                events.append({
                    "sequence_name": sequence,
                    "direction": direction,
                    "clean_fusion": {"negative_fusion": True},
                    "strength_metrics": [
                        {"strength": 0.25, "target_effect": 0.1, "faithfulness": 0.05},
                        {"strength": 0.5, "target_effect": 0.2, "faithfulness": 0.1},
                        {"strength": 0.75, "target_effect": 0.3, "faithfulness": 0.15},
                    ],
                })
        direction_data, density_data = _aggregate_gate_inputs(events, sequences)
        identity = {
            "sequences": sequences,
            "inputs": {"attribute_manifest": {"groups": {
                attribute: [sequences[index]]
                for index, attribute in enumerate(SEMANTIC_ATTRIBUTES)
            }}},
            "frozen_protocol": {"bootstrap": {
                "seed": 9, "samples": 20, "confidence_level": 0.95,
            }},
        }
        result = _compute_logo(events, identity, direction_data, density_data)
        self.assertEqual(result["direction_consistent_count"], 5)
        for audit in result["attributes"].values():
            self.assertIn("recomputed_gates", audit)
            self.assertEqual(set(audit["recomputed_gates"]), {"A", "B", "C"})


class DatasetAndIntegrityTest(unittest.TestCase):
    def _split_payload(self, role, names, dataset="RGBT234"):
        return {
            "schema_version": "cbe-stage0-diagnostic-v1",
            "artifact_type": "split_role_manifest",
            "role": role,
            "count": len(names),
            "sequences": [
                {"ordinal": index, "name": name, "dataset": dataset}
                for index, name in enumerate(names)
            ],
            "source": {
                "authority": "synthetic-test-authority",
                "artifact_id": f"synthetic-{role}-v1",
                "issued_at_utc": "2026-07-20T00:00:00Z",
            },
        }

    def _attribute_payload(self, design_names):
        groups = [
            {"name": attribute, "sequences": [design_names[index]]}
            for index, attribute in enumerate(SEMANTIC_ATTRIBUTES)
        ]
        memberships = {
            sequence: [
                group["name"] for group in groups if sequence in group["sequences"]
            ]
            for sequence in design_names
        }
        return {
            "schema_version": "cbe-stage0-diagnostic-v1",
            "artifact_type": "semantic_attribute_manifest",
            "attributes": list(SEMANTIC_ATTRIBUTES),
            "groups": groups,
            "sequences": [
                {"ordinal": index, "name": name, "attributes": memberships[name]}
                for index, name in enumerate(design_names)
            ],
            "source": {
                "authority": "synthetic-test-authority",
                "artifact_id": "synthetic-attributes-v1",
                "issued_at_utc": "2026-07-20T00:00:00Z",
            },
        }

    def _write_manifest_family(self, root):
        development = [f"dev-{index:03d}" for index in range(187)]
        design, internal, confirm = derive_design83(development)
        val = [f"val-{index:03d}" for index in range(47)]
        names = {
            "development187": development,
            "design83": design,
            "internal42": internal,
            "confirm62": confirm,
            "val47": val,
        }
        filenames = stage0_runner.SPLIT_MANIFEST_FILENAMES
        split_paths = {}
        for role, members in names.items():
            path = root / filenames[role]
            atomic_write_json(path, self._split_payload(role, members))
            split_paths[role] = path
        split_lock_path = root / "split_manifest_sha256.txt"
        split_lock_path.write_text(
            "".join(
                f"{sha256_file(root / filename)}  {filename}\n"
                for filename in sorted(filenames.values())
            ),
            encoding="ascii",
        )
        attribute_path = root / stage0_runner.ATTRIBUTE_MANIFEST_FILENAME
        atomic_write_json(attribute_path, self._attribute_payload(design))
        attribute_lock_path = root / "attribute_manifest_sha256.txt"
        attribute_lock_path.write_text(
            f"{sha256_file(attribute_path)}  {attribute_path.name}\n", encoding="ascii"
        )
        return split_paths, split_lock_path, attribute_path, attribute_lock_path, names

    def _write_image(self, path, value):
        image = np.full((8, 10, 3), value, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))

    @unittest.skipUnless(HAS_OPENCV, "OpenCV is required for dataset decoding")
    def test_dataset_loader_accepts_only_trailing_blank_annotation_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "seq"
            (sequence / "visible").mkdir(parents=True)
            (sequence / "infrared").mkdir()
            for index in (1, 2):
                self._write_image(sequence / "visible" / f"{index:04d}.png", 10 + index)
                self._write_image(sequence / "infrared" / f"{index:04d}.png", 20 + index)
            visible_bytes = b"0,0,4,4\r\n0,0,4,4\r\n\r\n"
            (sequence / "visible.txt").write_bytes(visible_bytes)
            (sequence / "infrared.txt").write_bytes(b"0,0,4,4\r\n0,0,4,4\r\n")
            manifest = load_sequence_manifest(root, "seq", "RGBT234")
            self.assertEqual(manifest.frame_count, 2)
            self.assertEqual(manifest.visible_annotation.size_bytes, len(visible_bytes))
            self.assertEqual(
                manifest.visible_annotation.sha256,
                __import__("hashlib").sha256(visible_bytes).hexdigest(),
            )

            (sequence / "visible.txt").write_bytes(b"0,0,4,4\r\n\r\n0,0,4,4\r\n")
            with self.assertRaisesRegex(ValueError, "blank annotation row"):
                load_sequence_manifest(root, "seq", "RGBT234")

    @unittest.skipUnless(HAS_OPENCV, "OpenCV is required for dataset decoding")
    def test_dataset_loader_rejects_silent_length_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "seq"
            (sequence / "visible").mkdir(parents=True)
            (sequence / "infrared").mkdir()
            self._write_image(sequence / "visible" / "0001.png", 10)
            self._write_image(sequence / "visible" / "0002.png", 20)
            self._write_image(sequence / "infrared" / "0001.png", 30)
            (sequence / "visible.txt").write_text("0,0,4,4\n0,0,4,4\n")
            (sequence / "infrared.txt").write_text("0,0,4,4\n0,0,4,4\n")
            with self.assertRaisesRegex(ValueError, "exactly equal counts"):
                load_sequence_manifest(root, "seq", "RGBT234")

    def test_manifest_lock_layer_accepts_exact_authoritative_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_paths, split_lock_path, attribute_path, attribute_lock_path, names = (
                self._write_manifest_family(root)
            )
            manifests, attributes, split_lock, attribute_lock = _load_manifest_lock_layer(
                split_paths=split_paths,
                split_lock_path=split_lock_path,
                expected_split_lock_sha256=sha256_file(split_lock_path),
                attribute_path=attribute_path,
                attribute_lock_path=attribute_lock_path,
                expected_attribute_lock_sha256=sha256_file(attribute_lock_path),
                dataset="RGBT234",
            )
            self.assertEqual(list(manifests["design83"].names), names["design83"])
            self.assertEqual(set(dict(attributes.groups)), set(SEMANTIC_ATTRIBUTES))
            self.assertEqual(set(split_lock.as_dict()), set(stage0_runner.SPLIT_LOCK_FILENAMES))
            self.assertEqual(attribute_lock.as_dict()[attribute_path.name], sha256_file(attribute_path))

    def test_manifest_lock_rejects_byte_drift_before_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_paths, split_lock_path, attribute_path, attribute_lock_path, _ = (
                self._write_manifest_family(root)
            )
            design_path = split_paths["design83"]
            design_path.write_bytes(b"{malformed unlocked JSON")
            with self.assertRaisesRegex(Stage0RunError, "locked manifest byte drift"):
                _load_manifest_lock_layer(
                    split_paths=split_paths,
                    split_lock_path=split_lock_path,
                    expected_split_lock_sha256=sha256_file(split_lock_path),
                    attribute_path=attribute_path,
                    attribute_lock_path=attribute_lock_path,
                    expected_attribute_lock_sha256=sha256_file(attribute_lock_path),
                    dataset="RGBT234",
                )

    def test_split_family_rejects_derived_role_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_paths, split_lock_path, attribute_path, attribute_lock_path, _ = (
                self._write_manifest_family(root)
            )
            internal_path = split_paths["internal42"]
            internal = json.loads(internal_path.read_text(encoding="utf-8"))
            design = json.loads(split_paths["design83"].read_text(encoding="utf-8"))
            internal["sequences"][0]["name"] = design["sequences"][0]["name"]
            atomic_write_json(internal_path, internal)
            split_lock_path.write_text(
                "".join(
                    f"{sha256_file(root / filename)}  {filename}\n"
                    for filename in sorted(stage0_runner.SPLIT_MANIFEST_FILENAMES.values())
                ),
                encoding="ascii",
            )
            with self.assertRaisesRegex(Stage0RunError, "internal42 members/order"):
                _load_manifest_lock_layer(
                    split_paths=split_paths,
                    split_lock_path=split_lock_path,
                    expected_split_lock_sha256=sha256_file(split_lock_path),
                    attribute_path=attribute_path,
                    attribute_lock_path=attribute_lock_path,
                    expected_attribute_lock_sha256=sha256_file(attribute_lock_path),
                    dataset="RGBT234",
                )

    def test_external_lock_digest_rejects_rehashed_manifest_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_paths, split_lock_path, attribute_path, attribute_lock_path, _ = (
                self._write_manifest_family(root)
            )
            frozen_lock_sha256 = sha256_file(split_lock_path)
            design_path = split_paths["design83"]
            design = json.loads(design_path.read_text(encoding="utf-8"))
            design["source"]["authority"] = "result-dependent-replacement"
            atomic_write_json(design_path, design)
            split_lock_path.write_text(
                "".join(
                    f"{sha256_file(root / filename)}  {filename}\n"
                    for filename in sorted(stage0_runner.SPLIT_MANIFEST_FILENAMES.values())
                ),
                encoding="ascii",
            )
            with self.assertRaisesRegex(Stage0RunError, "manifest hash lock drift"):
                _load_manifest_lock_layer(
                    split_paths=split_paths,
                    split_lock_path=split_lock_path,
                    expected_split_lock_sha256=frozen_lock_sha256,
                    attribute_path=attribute_path,
                    attribute_lock_path=attribute_lock_path,
                    expected_attribute_lock_sha256=sha256_file(attribute_lock_path),
                    dataset="RGBT234",
                )

    def test_split_family_rejects_val47_overlap_and_dataset_mismatch(self):
        for mutation, expected_error in (
            ("val_overlap", "val47 must be disjoint"),
            ("dataset", "split manifest dataset mismatch"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                split_paths, split_lock_path, attribute_path, attribute_lock_path, names = (
                    self._write_manifest_family(root)
                )
                if mutation == "val_overlap":
                    path = split_paths["val47"]
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["sequences"][0]["name"] = names["development187"][0]
                else:
                    path = split_paths["design83"]
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["sequences"][0]["dataset"] = "GTOT"
                atomic_write_json(path, payload)
                split_lock_path.write_text(
                    "".join(
                        f"{sha256_file(root / filename)}  {filename}\n"
                        for filename in sorted(stage0_runner.SPLIT_MANIFEST_FILENAMES.values())
                    ),
                    encoding="ascii",
                )
                with self.assertRaisesRegex(Stage0RunError, expected_error):
                    _load_manifest_lock_layer(
                        split_paths=split_paths,
                        split_lock_path=split_lock_path,
                        expected_split_lock_sha256=sha256_file(split_lock_path),
                        attribute_path=attribute_path,
                        attribute_lock_path=attribute_lock_path,
                        expected_attribute_lock_sha256=sha256_file(attribute_lock_path),
                        dataset="RGBT234",
                    )

    def test_semantic_manifest_rejects_bidirectional_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_paths, split_lock_path, attribute_path, attribute_lock_path, _ = (
                self._write_manifest_family(root)
            )
            attributes = json.loads(attribute_path.read_text(encoding="utf-8"))
            attributes["sequences"][0]["attributes"] = []
            atomic_write_json(attribute_path, attributes)
            attribute_lock_path.write_text(
                f"{sha256_file(attribute_path)}  {attribute_path.name}\n", encoding="ascii"
            )
            with self.assertRaisesRegex(Stage0RunError, "memberships disagree"):
                _load_manifest_lock_layer(
                    split_paths=split_paths,
                    split_lock_path=split_lock_path,
                    expected_split_lock_sha256=sha256_file(split_lock_path),
                    attribute_path=attribute_path,
                    attribute_lock_path=attribute_lock_path,
                    expected_attribute_lock_sha256=sha256_file(attribute_lock_path),
                    dataset="RGBT234",
                )

    def test_manifest_loaders_reject_role_and_lock_order_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            role_path = root / "role.json"
            atomic_write_json(role_path, self._split_payload("design83", ["seq"] * 83))
            with self.assertRaisesRegex(ProtocolValidationError, "split role mismatch"):
                read_split_role_manifest(
                    role_path, expected_role="internal42", expected_count=42
                )
            first = root / "a.json"
            second = root / "b.json"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            lock_path = root / "locks.txt"
            lock_path.write_text(
                f"{sha256_file(second)}  b.json\n{sha256_file(first)}  a.json\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ProtocolValidationError, "filenames or order"):
                read_manifest_hash_lock(
                    lock_path, required_filenames=("a.json", "b.json")
                )

    def test_manifest_schema_rejects_non_integer_count_and_ordinals(self):
        for field, value, expected_error in (
            ("count", 83.0, "count mismatch"),
            ("ordinal", 0.0, "ordinal mismatch"),
            ("ordinal", False, "ordinal mismatch"),
        ):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "design83.json"
                payload = self._split_payload(
                    "design83", [f"sequence-{index:03d}" for index in range(83)]
                )
                if field == "count":
                    payload["count"] = value
                else:
                    payload["sequences"][0]["ordinal"] = value
                atomic_write_json(path, payload)
                with self.assertRaisesRegex(ProtocolValidationError, expected_error):
                    read_split_role_manifest(
                        path, expected_role="design83", expected_count=83
                    )

    def test_preflight_cli_requires_external_lock_digests(self):
        parser = stage0_runner.build_parser()
        args = parser.parse_args([
            "--phase", "preflight", "--output-dir", "out",
            "--dataset-root", "dataset", "--dataset", "RGBT234",
            "--split-manifest", "design.json",
            "--development-manifest", "development.json",
            "--internal-manifest", "internal.json",
            "--confirm-manifest", "confirm.json", "--val-manifest", "val.json",
            "--split-manifest-lock", "split.sha256",
            "--attribute-manifest", "attributes.json",
            "--attribute-manifest-lock", "attributes.sha256",
            "--checkpoint", "checkpoint.pth", "--model-config", "model.yaml",
        ])
        with mock.patch.object(parser, "error", side_effect=ValueError) as error:
            with self.assertRaises(ValueError):
                stage0_runner._validate_phase_args(parser, args)
        message = error.call_args.args[0]
        self.assertIn("--split-manifest-lock-sha256", message)
        self.assertIn("--attribute-manifest-lock-sha256", message)

    def test_runtime_import_rejects_shadow_module(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound = root / "vipt.py"
            shadow = root / "shadow_vipt.py"
            bound.write_text("BOUND = True\n", encoding="utf-8")
            shadow.write_text("SHADOW = True\n", encoding="utf-8")
            identity = {
                "execution": {**stage0_runner.FORMAL_MODULES, "workers": 1},
                "inputs": {"sources": {"parameters": {
                    "path": str(bound), "sha256": sha256_file(bound),
                }}},
            }
            with mock.patch.object(
                stage0_runner.importlib,
                "import_module",
                return_value=SimpleNamespace(__file__=str(shadow)),
            ):
                with self.assertRaisesRegex(Stage0RunError, "identity-bound source"):
                    stage0_runner._import_identity_bound_module(
                        identity, "parameter_module", "parameters"
                    )

    def test_completion_must_match_phase_bound_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "online" / "sequences"
            parent.mkdir(parents=True)
            temp = Path(tempfile.mkdtemp(dir=parent))
            atomic_write_jsonl(temp / "trajectory.jsonl", [{"frame": 0}])
            atomic_write_jsonl(temp / "opportunities.jsonl", [])
            final = parent / "seq"
            _publish_sequence(
                temp, final, "identity", "seq", "online",
                ("trajectory.jsonl", "opportunities.jsonl"), {"frames": 1},
            )
            completion = json.loads((final / "completion.json").read_text())
            _authenticate_completion(
                root, "online", "seq", "identity",
                ("trajectory.jsonl", "opportunities.jsonl"), completion["content_hash"],
            )
            with self.assertRaises(Stage0RunError):
                _authenticate_completion(
                    root, "online", "seq", "identity",
                    ("trajectory.jsonl", "opportunities.jsonl"), "0" * 64,
                )


class SixPhaseIntegrationTest(unittest.TestCase):
    def _sequence_entry(self, dataset_root, sequence, frame_count=15):
        sequence_root = dataset_root / sequence
        visible_root = sequence_root / "visible"
        infrared_root = sequence_root / "infrared"
        visible_root.mkdir(parents=True, exist_ok=True)
        infrared_root.mkdir(exist_ok=True)
        visible_images = []
        infrared_images = []
        for frame_index in range(frame_count):
            for root, entries, value in (
                (visible_root, visible_images, b"v"),
                (infrared_root, infrared_images, b"i"),
            ):
                path = root / f"{frame_index:04d}.png"
                path.write_bytes(value + str(frame_index).encode("ascii"))
                entries.append({
                    "relative_path": path.relative_to(sequence_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "shape": [8, 10, 3],
                })
        annotation = ("0,0,1,1\n" * frame_count).encode("utf-8")
        visible_annotation = sequence_root / "visible.txt"
        infrared_annotation = sequence_root / "infrared.txt"
        visible_annotation.write_bytes(annotation)
        infrared_annotation.write_bytes(annotation)
        value = {
            "dataset": "RGBT234",
            "sequence": sequence,
            "relative_root": sequence,
            "frame_count": frame_count,
            "visible_images": visible_images,
            "infrared_images": infrared_images,
            "visible_annotation": {
                "relative_path": "visible.txt",
                "size_bytes": visible_annotation.stat().st_size,
                "sha256": sha256_file(visible_annotation),
                "row_count": frame_count,
            },
            "infrared_annotation": {
                "relative_path": "infrared.txt",
                "size_bytes": infrared_annotation.stat().st_size,
                "sha256": sha256_file(infrared_annotation),
                "row_count": frame_count,
            },
        }
        value["entry_hash"] = canonical_json_hash(value)
        return value

    def _forward(self):
        return {
            "score_map": [[4.0, 1.0], [1.0, 1.0]],
            "size_map": [
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, 0.5], [0.5, 0.5]],
            ],
            "offset_map": [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            "hann_response": [[4.0, 1.0], [1.0, 1.0]],
            "resize_factor": 1.0,
            "search_crop_xywh": [0.0, 0.0, 2.0, 2.0],
            "target_bbox": [0.0, 0.0, 1.0, 1.0],
            "best_score": 0.9,
            "search_anchor": [0.0, 0.0, 1.0, 1.0],
            "anchor_id": "anchor-0",
            "template_id": "template-0",
            "hann_window": [[1.0, 1.0], [1.0, 1.0]],
            "search_size": 2,
        }

    def test_fake_official_chain_is_deterministic_and_rejects_tampering(self):
        development_names = [f"dev-{index:03d}" for index in range(187)]
        design_names, internal_names, confirm_names = derive_design83(development_names)
        val_names = [f"val-{index:03d}" for index in range(47)]
        first_sequence = design_names[0]
        scheduled = deterministic_opportunity_schedule(
            first_sequence, 16, PRIMARY_DIRECTIONS, [0.25, 0.5, 0.75]
        )[0]
        assignment_hash = scheduled["assignment_hash"]
        direction = scheduled["direction"]
        event_id = stage0_runner.deterministic_event_id(
            first_sequence, scheduled["frame_index"], scheduled["opportunity_index"],
            direction, scheduled["strengths"], assignment_hash,
        )
        forward = self._forward()
        opportunity = {
            "schema_version": "cbe-stage0-diagnostic-v1",
            "scope": "design83",
            "record_type": "opportunity",
            "sequence_name": first_sequence,
            "frame_index": scheduled["frame_index"],
            "opportunity_index": scheduled["opportunity_index"],
            "event_id": event_id,
            "assignment_hash": assignment_hash,
            "direction": direction,
            "strengths": scheduled["strengths"],
            "search_anchor_xywh": [0.0, 0.0, 1.0, 1.0],
            "factual_template_id": "template-0",
            "probes": {
                "factual": forward,
                "rgb_retained": forward,
                "tir_retained": forward,
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "results"
            dataset_root = base / "dataset"
            dataset_root.mkdir()
            split_path = base / "design83_sequence_manifest.json"
            development_path = base / "development187_sequence_manifest.json"
            internal_path = base / "internal42_sequence_manifest.json"
            confirm_path = base / "confirm62_sequence_manifest.json"
            val_path = base / "val47_sequence_manifest.json"
            split_lock_path = base / "split_manifest_sha256.txt"
            attribute_path = base / "sequence_attribute_manifest.json"
            attribute_lock_path = base / "attribute_manifest_sha256.txt"
            checkpoint_path = base / "checkpoint.pth"
            model_config_path = base / "deep_rgbt.yaml"
            for path in (split_path, development_path, internal_path, confirm_path, val_path):
                path.write_text(f"fake {path.name}\n", encoding="utf-8")
            split_lock_path.write_text("fake split lock\n", encoding="ascii")
            attribute_path.write_text("fake attribute manifest\n", encoding="utf-8")
            attribute_lock_path.write_text("fake attribute lock\n", encoding="ascii")
            checkpoint_path.write_bytes(b"not-a-model-checkpoint")
            model_config_path.write_text("MODEL: fake\n", encoding="utf-8")

            def fake_split(role, names, path):
                return SimpleNamespace(
                    role=role,
                    names=tuple(names),
                    datasets=tuple("RGBT234" for _ in names),
                    records=tuple((name, "RGBT234") for name in names),
                    source={
                        "authority": "synthetic-test-authority",
                        "artifact_id": f"synthetic-{role}-v1",
                        "issued_at_utc": "2026-07-20T00:00:00Z",
                    },
                    sha256=sha256_file(path),
                )

            manifest_by_role = {
                "development187": fake_split("development187", development_names, development_path),
                "design83": fake_split("design83", design_names, split_path),
                "internal42": fake_split("internal42", internal_names, internal_path),
                "confirm62": fake_split("confirm62", confirm_names, confirm_path),
                "val47": fake_split("val47", val_names, val_path),
            }
            attribute_groups = tuple(
                (attribute, (design_names[index + 1],))
                for index, attribute in enumerate(SEMANTIC_ATTRIBUTES)
            )
            attribute_manifest = SimpleNamespace(
                sha256=sha256_file(attribute_path),
                groups=attribute_groups,
                sequence_attributes=tuple(
                    (
                        sequence,
                        tuple(attribute for attribute, members in attribute_groups if sequence in members),
                    )
                    for sequence in design_names
                ),
                source={
                    "authority": "synthetic-test-authority",
                    "artifact_id": "synthetic-attributes-v1",
                    "issued_at_utc": "2026-07-20T00:00:00Z",
                },
            )
            split_lock = SimpleNamespace(
                sha256=sha256_file(split_lock_path),
                as_dict=lambda: {path.name: sha256_file(path) for path in (
                    confirm_path, split_path, development_path, internal_path, val_path
                )},
            )
            attribute_lock = SimpleNamespace(
                sha256=sha256_file(attribute_lock_path),
                as_dict=lambda: {attribute_path.name: sha256_file(attribute_path)},
            )

            design_name_set = set(design_names)

            def fake_load_sequence(_root, sequence, _dataset):
                self.assertIn(sequence, design_name_set, "preflight touched a sealed split member")
                return self._sequence_entry(
                    dataset_root, sequence,
                    frame_count=16 if sequence == first_sequence else 15,
                )

            def fake_online_sequence(output_root, identity, entry):
                sequence = entry["sequence"]
                parent = output_root / "online" / "sequences"
                parent.mkdir(parents=True, exist_ok=True)
                final = parent / sequence
                files = ("trajectory.jsonl", "opportunities.jsonl")
                if stage0_runner._completed_sequence(
                    output_root, "online", sequence, identity["content_hash"], files
                ):
                    return
                temp = Path(tempfile.mkdtemp(dir=parent))
                trajectory = [{
                    "schema_version": "cbe-stage0-diagnostic-v1",
                    "scope": "design83",
                    "record_type": "trajectory",
                    "sequence_name": sequence,
                    "frame_index": 0,
                    "pred_xywh": [0.0, 0.0, 1.0, 1.0],
                    "best_score": None,
                    "template_id": "template-0",
                }]
                opportunities = [opportunity] if sequence == first_sequence else []
                stage0_runner._validate_online_rows(trajectory)
                stage0_runner._validate_online_rows(opportunities)
                atomic_write_jsonl(temp / files[0], trajectory)
                atomic_write_jsonl(temp / files[1], opportunities)
                _publish_sequence(
                    temp, final, identity["content_hash"], sequence, "online", files,
                    {"frames": 1, "opportunities": len(opportunities)},
                )

            def fake_intervene_sequence(output_root, identity, entry):
                sequence = entry["sequence"]
                files = (
                    "raw_forwards.jsonl", "evaluator_labels.jsonl",
                    "replay_reports.jsonl",
                )
                parent = output_root / "intervene" / "sequences"
                parent.mkdir(parents=True, exist_ok=True)
                final = parent / sequence
                online_completion = json.loads(
                    (output_root / "online" / "sequences" / sequence / "completion.json")
                    .read_text(encoding="utf-8")
                )
                if stage0_runner._completed_sequence(
                    output_root, "intervene", sequence, identity["content_hash"], files
                ):
                    completion = json.loads((final / "completion.json").read_text(encoding="utf-8"))
                    self.assertEqual(
                        completion["input_completion_hash"], online_completion["content_hash"]
                    )
                    return
                raw_rows = []
                label_rows = []
                replay_rows = []
                if sequence == first_sequence:
                    target_id = stage0_runner._region_id(event_id, "target")
                    background_id = stage0_runner._region_id(event_id, "background")
                    raw_rows.append({
                        "schema_version": "cbe-stage0-diagnostic-v1",
                        "scope": "design83",
                        "record_type": "primary_local_intervention_raw",
                        "sequence_name": sequence,
                        "frame_index": 15,
                        "opportunity_index": 0,
                        "event_id": event_id,
                        "assignment_hash": assignment_hash,
                        "direction": direction,
                        "strengths": [0.25, 0.5, 0.75],
                        "source_frame": 15,
                        "search_anchor_xywh": [0.0, 0.0, 1.0, 1.0],
                        "factual_template_id": "template-0",
                        "status": "INVALID_OPPORTUNITY",
                        "strength_arms": [],
                    })
                    label_rows.append({
                        "schema_version": "cbe-stage0-diagnostic-v1",
                        "scope": "design83",
                        "record_type": "evaluator_label",
                        "sequence_name": sequence,
                        "frame_index": 15,
                        "opportunity_index": 0,
                        "event_id": event_id,
                        "assignment_hash": assignment_hash,
                        "direction": direction,
                        "strengths": [0.25, 0.5, 0.75],
                        "status": "INVALID_OPPORTUNITY",
                        "invalid_reason": "registration_discordant",
                        "visible_gt_xywh": [0.0, 0.0, 1.0, 1.0],
                        "tir_gt_xywh": [2.0, 2.0, 1.0, 1.0],
                        "target_mask_gt_xywh": [0.0, 0.0, 1.0, 1.0],
                        "evaluation_gt_xywh": [0.0, 0.0, 1.0, 1.0],
                        "registration_iou": 0.0,
                        "target_region_id": target_id,
                        "background_region_id": background_id,
                    })
                    replay_rows.append(stage0_runner._invalid_replay_report(
                        event_id, "registration_discordant", 0.0
                    ))
                temp = Path(tempfile.mkdtemp(dir=parent))
                for filename, rows in zip(files, (raw_rows, label_rows, replay_rows)):
                    atomic_write_jsonl(temp / filename, rows)
                _publish_sequence(
                    temp, final, identity["content_hash"], sequence, "intervene", files,
                    {"events": len(raw_rows), "labels": len(label_rows),
                     "replays": len(replay_rows)},
                    input_completion_hash=online_completion["content_hash"],
                )

            preflight_args = argparse.Namespace(
                output_dir=str(root), workers=1, dataset_root=str(dataset_root),
                dataset="RGBT234", split_manifest=str(split_path),
                development_manifest=str(development_path),
                internal_manifest=str(internal_path), confirm_manifest=str(confirm_path),
                val_manifest=str(val_path), split_manifest_lock=str(split_lock_path),
                split_manifest_lock_sha256=sha256_file(split_lock_path),
                attribute_manifest=str(attribute_path),
                attribute_manifest_lock=str(attribute_lock_path),
                attribute_manifest_lock_sha256=sha256_file(attribute_lock_path),
                checkpoint=str(checkpoint_path), checkpoint_sha256=sha256_file(checkpoint_path),
                model_config=str(model_config_path),
                model_config_sha256=sha256_file(model_config_path),
                protocol_dir=str(DEFAULT_PROTOCOL_DIR),
                parameter_module="lib.test.parameter.vipt",
                tracker_module="lib.test.tracker.vipt_stage0",
                adapter_module="analysis.cbe.tracker_probe_v1",
            )
            online_args = argparse.Namespace(
                parent=str(root / "preflight.json"), output_dir=str(root), workers=1
            )
            intervene_args = argparse.Namespace(
                parent=str(root / "online.json"), output_dir=str(root)
            )
            evaluate_args = argparse.Namespace(
                parent=str(root / "intervene.json"), output_dir=str(root)
            )
            replay_report_path = base / "semantic_replay.json"
            atomic_write_json(
                replay_report_path, replay_fixture(DEFAULT_FIXTURE, DEFAULT_PROTOCOL_DIR)
            )
            gate_args = argparse.Namespace(
                parent=str(root / "evaluate.json"), output_dir=str(root),
                replay_report=str(replay_report_path),
            )
            verify_args = argparse.Namespace(
                parent=str(root / "gate.json"), output_dir=str(root),
                fixture=str(DEFAULT_FIXTURE), protocol_dir=str(DEFAULT_PROTOCOL_DIR),
            )

            patches = (
                mock.patch.object(stage0_runner, "_require_runtime", return_value=None),
                mock.patch.object(
                    stage0_runner, "_load_manifest_lock_layer",
                    return_value=(manifest_by_role, attribute_manifest, split_lock, attribute_lock),
                ),
                mock.patch.object(
                    stage0_runner, "load_sequence_manifest", create=True,
                    side_effect=fake_load_sequence,
                ),
                mock.patch.object(
                    stage0_runner, "_sequence_manifest_dict", side_effect=lambda value: value
                ),
                mock.patch.object(
                    stage0_runner, "_online_sequence", side_effect=fake_online_sequence
                ),
                mock.patch.object(
                    stage0_runner, "_intervene_sequence", side_effect=fake_intervene_sequence
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                bad_checkpoint_args = argparse.Namespace(**vars(preflight_args))
                bad_checkpoint_args.checkpoint_sha256 = "0" * 64
                with self.assertRaisesRegex(Stage0RunError, "checkpoint differs"):
                    run_preflight(bad_checkpoint_args)
                self.assertFalse((root / "run_identity.json").exists())

                bad_config_args = argparse.Namespace(**vars(preflight_args))
                bad_config_args.model_config_sha256 = "not-a-sha256"
                with self.assertRaisesRegex(Stage0RunError, "externally frozen lowercase"):
                    run_preflight(bad_config_args)
                self.assertFalse((root / "run_identity.json").exists())

                phases = [
                    run_preflight(preflight_args),
                    run_online(online_args),
                    run_intervene(intervene_args),
                    run_evaluate(evaluate_args),
                    run_gate(gate_args),
                    run_verify(verify_args),
                ]
                self.assertEqual(phases[4]["payload"]["status"], "STOP_CBE")
                self.assertTrue(phases[5]["payload"]["verification_passed"])
                self.assertEqual(phases[5]["payload"]["status"], "STOP_CBE")

                resumed = [
                    run_preflight(preflight_args),
                    run_online(online_args),
                    run_intervene(intervene_args),
                    run_evaluate(evaluate_args),
                    run_gate(gate_args),
                    run_verify(verify_args),
                ]
                self.assertEqual(phases, resumed)

                identity_path = root / "run_identity.json"
                identity_bytes = identity_path.read_bytes()
                identity = json.loads(identity_bytes)
                identity["sequences"][0], identity["sequences"][1] = (
                    identity["sequences"][1], identity["sequences"][0]
                )
                identity["content_hash"] = canonical_json_hash({
                    key: value for key, value in identity.items() if key != "content_hash"
                })
                atomic_write_json(identity_path, identity)
                with self.assertRaisesRegex(Stage0RunError, "locked design83"):
                    stage0_runner._validate_identity_inputs(identity)
                identity_path.write_bytes(identity_bytes)

                bound_image = dataset_root / first_sequence / "visible" / "0000.png"
                bound_image_bytes = bound_image.read_bytes()
                bound_image.write_bytes(bound_image_bytes + b"tamper")
                with self.assertRaisesRegex(Stage0RunError, "dataset file drift"):
                    run_online(online_args)
                bound_image.write_bytes(bound_image_bytes)

                aggregate_path = root / "evaluate" / "aggregate.json"
                aggregate_bytes = aggregate_path.read_bytes()
                aggregate = json.loads(aggregate_bytes)
                aggregate["event_count"] = 99
                aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
                with self.assertRaisesRegex(Stage0RunError, "aggregate SHA"):
                    run_gate(gate_args)
                aggregate_path.write_bytes(aggregate_bytes)

                opportunity_path = root / "online" / "sequences" / first_sequence / "opportunities.jsonl"
                opportunity_bytes = opportunity_path.read_bytes()
                opportunity_path.write_bytes(opportunity_bytes + b"\n")
                online_completion_hash = phases[1]["payload"]["sequence_completion_hashes"][first_sequence]
                with self.assertRaisesRegex(Stage0RunError, "completed artifact hash mismatch"):
                    _authenticate_completion(
                        root, "online", first_sequence, phases[0]["identity_hash"],
                        ("trajectory.jsonl", "opportunities.jsonl"), online_completion_hash,
                    )
                opportunity_path.write_bytes(opportunity_bytes)

                decision_path = root / "gate" / "decision.json"
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
                decision["status"] = "PASS_STAGE0"
                decision_path.write_text(json.dumps(decision), encoding="utf-8")
                tampered_verify = run_verify(verify_args)
                self.assertEqual(tampered_verify["payload"]["status"], "INVALID_RUN")
                self.assertFalse(tampered_verify["payload"]["verification_passed"])


if __name__ == "__main__":
    unittest.main()
