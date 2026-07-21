import importlib.util
import inspect
import math
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = ROOT / "analysis" / "memory_oracle" / "quarantine_controller_v2.py"
METRICS_PATH = ROOT / "analysis" / "memory_oracle" / "compute_stage1_quarantine_v2_metrics.py"
RUNNER_PATH = ROOT / "analysis" / "memory_oracle" / "run_stage1_quarantine_v2.py"
CONFIG_PATH = ROOT / "analysis" / "memory_oracle" / "configs" / "stage1_quarantine_v2.yaml"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = load_module("stage1_quarantine_v2_controller_tests", CONTROLLER_PATH)
metrics = load_module("stage1_quarantine_v2_metrics_tests", METRICS_PATH)


def observation(frame_idx=30, entropy=0.47, box=None):
    return {
        "frame_idx": frame_idx,
        "pred_xywh": [10.0, 12.0, 20.0, 16.0] if box is None else box,
        "search_anchor_xywh": [8.0, 10.0, 24.0, 20.0],
        "image_shape": [100, 120],
        "response_entropy": entropy,
    }


def arm_metrics(success_auc, num_frames=100):
    return {
        "num_frames": num_frames,
        "success_auc": success_auc,
        "mean_iou": success_auc,
        "precision20": success_auc,
        "normalized_precision": success_auc,
        "normalized_precision_at_0_2": success_auc,
    }


def quality_bucket(count, good, bad, coverage, sequence_count):
    return {
        "count": count,
        "good_count": good,
        "bad_count": bad,
        "precision": float(good / count) if count else 0.0,
        "bad_rate": float(bad / count) if count else 0.0,
        "coverage": coverage,
        "good_recall": 0.0,
        "sequence_count": sequence_count,
        "sequence_coverage": 0.0,
    }


def passing_action_candidate(threshold=0.5, incremental=0.01, auc_hint=0.01):
    governance = {
        "num_legal_source_opportunities": 1000,
        "num_good_source_opportunities": 100,
        "immediate": quality_bucket(30, 28, 0, 0.03, 20),
        "quarantine": quality_bucket(100, 0, 0, 0.10, 20),
        "release": quality_bucket(70, 65, 3, 0.07, 20),
        "combined": quality_bucket(100, 93, 3, 0.10, 20),
    }
    return {
        "support_threshold": threshold,
        "governance": governance,
        "quarantine_incremental_success_auc_delta": incremental,
        "auc_hint": auc_hint,
    }


def passing_final_aggregate():
    candidate = passing_action_candidate()
    governance = dict(candidate["governance"])
    governance["quarantine_incremental_success_auc_delta"] = 0.01
    return {
        "governance": governance,
        "frame_weighted_paired_deltas": {
            "rmg_q_vs_static": {"success_auc": {"mean": 0.006, "low": 0.001}},
            "rmg_q_vs_periodic_pred": {
                "success_auc": {"mean": 0.011, "low": 0.001}
            },
            "rmg_q_vs_rmg_q_no_quarantine": {
                "success_auc": {"mean": 0.002, "low": 0.001}
            },
            "rmg_q_vs_confidence_e050": {
                "success_auc": {"mean": 0.001, "low": 0.0001}
            },
        },
        "worsened_vs_static": {"fraction": 0.20},
        "clean_subset_preservation": {
            "num_sequences": 1,
            "rmg_q_vs_static_success_auc": {"mean": -0.002},
            "worsened_vs_static": {"fraction": 0.10},
        },
    }


def write_realistic_stage0_manifest_fixture(runner, root, sequences):
    manifests = []
    for sequence in sequences:
        event = {
            "event_id": "{}:000030".format(sequence), "sequence": sequence,
            "candidate_frame": 30, "horizons": {"5": [31, 32, 33, 34, 35]},
        }
        event["event_hash"] = runner.canonical_hash(event)
        manifest = {
            "schema_version": "rmg-stage0-v1", "kind": "frozen_stage0_manifest",
            "sequence": sequence, "num_frames": 100,
            "policy": {"warmup": 30, "update_interval": 30,
                       "replay_event_min_spacing": 60, "horizons": [5, 15, 30, 60],
                       "terminal_cooldown": 60, "pred_good_iou": 0.7,
                       "bad_iou": 0.1, "min_candidate_size": 8.0,
                       "min_intersection_ratio": 0.75, "max_padding_ratio": 0.25},
            "baseline_trace_hash": runner.canonical_hash([sequence]),
            "update_frames": [30], "replay_event_frames": [30], "events": [event],
        }
        manifest["manifest_hash"] = runner.canonical_hash(manifest)
        sequence_root = root / "sequences" / sequence
        sequence_root.mkdir(parents=True, exist_ok=True)
        runner.write_json(sequence_root / "manifest.json", manifest)
        manifests.append(manifest)
    index = {
        "schema_version": "rmg-stage0-v1", "kind": "frozen_stage0_manifest_index",
        "sequences": [{"sequence": item["sequence"],
                       "manifest_hash": item["manifest_hash"],
                       "num_events": len(item["events"])} for item in manifests],
        "content_hash": runner.canonical_hash(manifests),
    }
    runner.write_json(root / "manifest.json", index)
    return index


def runner_or_skip(testcase):
    if not RUNNER_PATH.is_file():
        testcase.skipTest("Stage 1 quarantine v2 runner is not available yet")
    controller_names = (
        "analysis.memory_oracle.quarantine_controller_v2",
        "quarantine_controller_v2",
    )
    metrics_names = (
        "analysis.memory_oracle.compute_stage1_quarantine_v2_metrics",
        "compute_stage1_quarantine_v2_metrics",
    )
    for name in controller_names:
        sys.modules[name] = controller
    for name in metrics_names:
        sys.modules[name] = metrics
    try:
        return load_module("stage1_quarantine_v2_runner_tests", RUNNER_PATH)
    except (ImportError, ModuleNotFoundError) as error:
        missing = (getattr(error, "name", "") or "").split(".")[0]
        heavy = {"torch", "cv2", "lib", "ltr", "jpeg4py", "yaml"}
        if missing in heavy:
            testcase.skipTest("Runner heavy dependency is unavailable: {}".format(error))
        raise


class QuarantineControllerV2Test(unittest.TestCase):
    def setUp(self):
        self.policy = controller.QuarantinePolicy(support_iou=0.7)

    def test_triage_exact_boundaries_nonfinite_and_invalid(self):
        cases = (
            (0.45, "immediate_update", "entropy_immediate"),
            (math.nextafter(0.45, math.inf), "quarantine", "entropy_quarantine"),
            (0.50, "quarantine", "entropy_quarantine"),
            (math.nextafter(0.50, math.inf), "skip", "entropy_above_quarantine"),
        )
        for entropy, action, reason in cases:
            with self.subTest(entropy=entropy):
                result = controller.triage(
                    observation(entropy=entropy), self.policy, candidate_legal=True)
                self.assertEqual((result.action, result.reason), (action, reason))

        for entropy in (float("nan"), float("inf"), float("-inf"), "not-a-number"):
            with self.subTest(entropy=entropy):
                result = controller.triage(
                    observation(entropy=entropy), self.policy, candidate_legal=True)
                self.assertEqual((result.action, result.reason),
                                 ("skip", "non_finite_entropy"))
        invalid = controller.triage(
            observation(), self.policy, candidate_legal=False,
            illegal_reason="padding_ratio")
        self.assertEqual((invalid.action, invalid.reason),
                         ("skip", "invalid:padding_ratio"))
        invalid_box = controller.triage(
            observation(box=[0.0, 0.0, 0.0, 10.0]), self.policy,
            candidate_legal=True)
        self.assertEqual((invalid_box.action, invalid_box.reason),
                         ("skip", "invalid:candidate_box"))
        with self.assertRaises(TypeError):
            controller.triage(observation(), self.policy, candidate_legal=1)
        with self.assertRaises(ValueError):
            controller.QuarantinePolicy(support_iou=float("nan"))

    def test_controller_rejects_gt_iou_and_unknown_fields(self):
        for field in ("gt", "gt_xywh", "iou", "evaluation_iou",
                      "source_candidate_iou", "agreement_iou", "future_label"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "forbidden fields"):
                    controller.validate_observation(dict(observation(), **{field: 0.5}))
        with self.assertRaisesRegex(ValueError, "missing fields"):
            controller.validate_observation({"frame_idx": 30})

    def test_single_slot_and_deterministic_event_identity(self):
        self.assertEqual(controller.deterministic_event_id("seq", 30), "seq:000030")
        self.assertEqual(controller.deterministic_event_id("seq", 30),
                         controller.deterministic_event_id("seq", 30))
        state, admitted = controller.admit_opportunity(
            controller.QuarantineState(), "seq", observation(), self.policy, True)
        self.assertEqual(admitted.action, "quarantine")
        self.assertEqual(admitted.event_id, "seq:000030")
        self.assertEqual(state.pending.event_id, admitted.event_id)
        occupied, denied = controller.admit_opportunity(
            state, "seq", observation(frame_idx=60), self.policy, True)
        self.assertIs(occupied, state)
        self.assertEqual((denied.action, denied.reason, denied.event_id),
                         ("skip", "quarantine_slot_occupied", "seq:000060"))
        self.assertEqual(occupied.pending.source_frame, 30)

    def _admit(self):
        return controller.admit_opportunity(
            controller.QuarantineState(), "seq", observation(), self.policy, True)[0]

    def _probe(self, state, frame_idx, supports=True, anchor=None):
        active = [10.0, 10.0, 20.0, 20.0]
        shadow = active if supports else [100.0, 100.0, 20.0, 20.0]
        shared = [5.0, 5.0, 30.0, 30.0] if anchor is None else anchor
        return controller.record_probe(
            state, self.policy, state.pending.event_id, frame_idx,
            active, shadow, True, True, shared, shared)

    def test_probes_have_exact_offsets_order_anchor_and_no_early_finalization(self):
        state = self._admit()
        self.assertEqual(controller.due_probe_offsets(state, 30, self.policy), ())
        self.assertEqual(controller.due_probe_offsets(state, 31, self.policy), (1,))
        with self.assertRaisesRegex(ValueError, "next required offset"):
            self._probe(state, 33)
        with self.assertRaisesRegex(ValueError, "identical valid shared anchor"):
            controller.record_probe(
                state, self.policy, state.pending.event_id, 31,
                [10, 10, 20, 20], [10, 10, 20, 20], True, True,
                [5, 5, 30, 30], [6, 5, 30, 30])

        state, first = self._probe(state, 31)
        self.assertEqual((first.offset, first.frame_idx), (1, 31))
        self.assertEqual(first.shared_anchor_xywh, (5.0, 5.0, 30.0, 30.0))
        with self.assertRaisesRegex(ValueError, "next required offset"):
            self._probe(state, 31)
        with self.assertRaisesRegex(ValueError, "all probes"):
            controller.finalize_quarantine(
                state, self.policy, state.pending.event_id, 35)
        self.assertEqual(controller.due_probe_offsets(state, 33, self.policy), (3,))
        state, second = self._probe(state, 33)
        self.assertEqual(second.offset, 3)
        with self.assertRaisesRegex(ValueError, "final probe offset"):
            controller.finalize_quarantine(
                state, self.policy, state.pending.event_id, 34)
        self.assertEqual(controller.due_probe_offsets(state, 35, self.policy), (5,))
        state, third = self._probe(state, 35)
        self.assertEqual(tuple(probe.offset for probe in state.pending.probes), (1, 3, 5))
        self.assertEqual(len({probe.offset for probe in state.pending.probes}), 3)
        self.assertEqual(third.source_frame, 30)

    def test_two_of_three_releases_at_t5_effective_t6(self):
        state = self._admit()
        for frame, supports in ((31, True), (33, False), (35, True)):
            state, _ = self._probe(state, frame, supports)
        empty, result = controller.finalize_quarantine(
            state, self.policy, state.pending.event_id, 35)
        self.assertIsNone(empty.pending)
        self.assertEqual((result.action, result.support_count, result.probe_count),
                         ("release", 2, 3))
        self.assertEqual((result.source_frame, result.finalized_frame,
                          result.effective_frame), (30, 35, 36))
        self.assertEqual(result.source_candidate_xywh, (10.0, 12.0, 20.0, 16.0))

    def test_one_of_three_discards_without_effective_frame(self):
        state = self._admit()
        for frame, supports in ((31, False), (33, True), (35, False)):
            state, _ = self._probe(state, frame, supports)
        _, result = controller.finalize_quarantine(
            state, self.policy, state.pending.event_id, 35)
        self.assertEqual((result.action, result.reason, result.support_count),
                         ("discard", "insufficient_probe_support", 1))
        self.assertIsNone(result.effective_frame)


class QuarantineMetricsV2Test(unittest.TestCase):
    def test_active_shadow_agreement_iou_is_online_but_evaluation_iou_is_not(self):
        online = [{
            "event_id": "s:000030", "sequence": "s", "action": "release",
            "probes": [{"agreement_iou": 0.8, "supports_release": True}],
        }]
        labels = [{"event_id": "s:000030", "source_candidate_iou": 0.8}]
        joined = metrics.join_event_traces(online, labels)
        self.assertEqual(joined[0]["source_candidate_iou"], 0.8)
        for field in ("evaluation_iou", "candidate_iou", "source_candidate_iou",
                      "release_frame_iou", "gt_xywh", "ground_truth"):
            with self.subTest(field=field):
                leaked = [dict(online[0], **{field: 0.9})]
                with self.assertRaisesRegex(ValueError, "forbidden GT/IoU"):
                    metrics.join_event_traces(leaked, labels)

    def test_frame_joins_are_strict_nonleaking_and_metrics_use_exact_labels(self):
        online = [
            {"sequence": "s", "frame_idx": 0, "pred_xywh": [0, 0, 10, 10]},
            {"sequence": "s", "frame_idx": 1, "pred_xywh": [10, 0, 10, 10]},
        ]
        labels = [
            {"sequence": "s", "frame_idx": 0, "gt_xywh": [0, 0, 10, 10]},
            {"sequence": "s", "frame_idx": 1, "gt_xywh": [0, 0, 10, 10]},
        ]
        joined = metrics.join_frame_traces(online, labels)
        self.assertEqual([row["gt_xywh"] for row in joined],
                         [[0, 0, 10, 10], [0, 0, 10, 10]])
        result = metrics.tracking_metrics_from_online_trace(online, labels)
        self.assertAlmostEqual(result["mean_iou"], 0.5)
        with self.assertRaisesRegex(ValueError, "Duplicate online frame"):
            metrics.join_frame_traces(online + [dict(online[0])], labels)
        with self.assertRaisesRegex(ValueError, "do not match"):
            metrics.join_frame_traces(online, labels[:1])
        with self.assertRaisesRegex(ValueError, "Noncontiguous"):
            metrics.join_frame_traces(
                [online[0], dict(online[1], frame_idx=2)],
                [labels[0], dict(labels[1], frame_idx=2)])
        with self.assertRaisesRegex(ValueError, "forbidden GT/IoU"):
            metrics.join_frame_traces([dict(online[0], gt_xywh=[0, 0, 1, 1])],
                                      labels[:1])

    def test_delayed_release_badness_uses_source_candidate_not_release_frame(self):
        online = [{
            "event_id": "s:000030", "sequence": "s", "action": "release",
            "source_frame": 30, "finalized_frame": 35, "effective_frame": 36,
        }]
        labels = [{
            "event_id": "s:000030", "source_candidate_iou": 0.0,
            "release_frame_iou": 1.0,
        }]
        result = metrics.governance_metrics(online, labels)
        self.assertEqual(result["release_count"], 1)
        self.assertEqual(result["release_good_count"], 0)
        self.assertEqual(result["release_bad_count"], 1)
        self.assertEqual(result["release_bad_rate"], 1.0)

    def test_combined_denominator_is_legal_opportunities_with_counts_and_reasons(self):
        online = [
            {"event_id": "e1", "sequence": "s1", "action": "immediate_write",
             "reason": "entropy_immediate", "legal_source_opportunity": True},
            {"event_id": "e2", "sequence": "s1", "action": "quarantine",
             "reason": "entropy_quarantine", "legal_source_opportunity": True},
            {"event_id": "e3", "sequence": "s2", "action": "release",
             "quarantined": True, "reason": "probe_support_pass",
             "legal_source_opportunity": True},
            {"event_id": "e4", "sequence": "s2", "action": "skip",
             "reason": "entropy_above_quarantine", "legal_source_opportunity": True},
            {"event_id": "e5", "sequence": "s3", "action": "immediate_update",
             "reason": "invalid:geometry", "legal_source_opportunity": False},
        ]
        labels = [
            {"event_id": "e1", "source_candidate_iou": 0.8},
            {"event_id": "e2", "source_candidate_iou": 0.5},
            {"event_id": "e3", "source_candidate_iou": 0.0},
            {"event_id": "e4", "source_candidate_iou": 0.9},
            {"event_id": "e5", "source_candidate_iou": 0.9},
        ]
        result = metrics.governance_metrics(online, labels)
        self.assertEqual(result["num_legal_source_opportunities"], 4)
        self.assertEqual((result["immediate_writes"], result["quarantines"],
                          result["releases"], result["combined_writes"]),
                         (1, 2, 1, 2))
        self.assertAlmostEqual(result["immediate_coverage"], 0.25)
        self.assertAlmostEqual(result["release_coverage"], 0.25)
        self.assertAlmostEqual(result["combined_coverage"], 0.50)
        self.assertAlmostEqual(result["combined_bad_rate"], 0.50)
        self.assertEqual(result["reasons"]["invalid:geometry"], 1)
        self.assertEqual(result["reasons_by_action"]["release"]["probe_support_pass"], 1)

    def test_smallest_passing_threshold_is_selected_not_highest_auc(self):
        low = passing_action_candidate(0.5, incremental=0.001, auc_hint=0.001)
        high = passing_action_candidate(0.9, incremental=0.2, auc_hint=0.2)
        selected = metrics.select_smallest_passing_support_threshold([high, low])
        self.assertEqual(selected["selected_support_threshold"], 0.5)
        self.assertEqual(selected["selection_rule"],
                         "smallest_support_threshold_passing_all_gates")
        low["governance"]["combined"]["bad_rate"] = 0.051
        selected = metrics.select_smallest_passing_support_threshold([low, high])
        self.assertEqual(selected["selected_support_threshold"], 0.9)

    def test_gate_a_gate_b_and_final_gate_pass_fail_closed(self):
        passing = passing_action_candidate()
        gate = metrics.evaluate_action_quality(passing)
        self.assertTrue(gate["gate_a"]["pass"])
        self.assertTrue(gate["gate_b"]["pass"])
        self.assertTrue(gate["pass"])

        failing_a = passing_action_candidate()
        failing_a["governance"]["immediate"]["precision"] = 0.899
        self.assertFalse(metrics.evaluate_action_quality(failing_a)["gate_a"]["pass"])
        failing_b = passing_action_candidate(incremental=0.0)
        self.assertFalse(metrics.evaluate_action_quality(failing_b)["gate_b"]["pass"])

        aggregate = passing_final_aggregate()
        self.assertTrue(metrics.evaluate_final_gate(aggregate)["pass"])
        aggregate["frame_weighted_paired_deltas"]["rmg_q_vs_periodic_pred"][
            "success_auc"]["low"] = 0.0
        failed = metrics.evaluate_final_gate(aggregate)
        self.assertFalse(failed["pass"])
        self.assertFalse(failed["checks"][
            "rmg_q_vs_periodic_auc_delta_ci_low"]["pass"])

        empty_clean = passing_final_aggregate()
        empty_clean["clean_subset_preservation"] = {
            "num_sequences": 0,
            "rmg_q_vs_static_success_auc": {"mean": 0.0},
            "worsened_vs_static": {"fraction": 0.0},
        }
        clean_gate = metrics.evaluate_final_gate(empty_clean)
        self.assertFalse(clean_gate["pass"])
        self.assertFalse(clean_gate["checks"]["clean_sequence_coverage"]["pass"])

    def test_threshold_stability_is_deterministic_with_small_bootstrap(self):
        sequence_summaries = []
        for index in range(12):
            candidates = []
            for threshold, incremental in ((0.5, 0.01), (0.6, 0.02)):
                candidate = passing_action_candidate(threshold, incremental)
                candidate["num_frames"] = 10
                candidates.append(candidate)
            sequence_summaries.append({
                "sequence": "seq{:02d}".format(index),
                "num_frames": 10,
                "threshold_candidates": candidates,
            })
        kwargs = {
            "gate_config": {"combined_writes_min": 1, "combined_sequences_min": 1},
            "seed": 17,
            "samples": 25,
            "hash_namespace": "test-rmg-stage1-v2-q5",
        }
        first = metrics.deterministic_threshold_stability(sequence_summaries, **kwargs)
        second = metrics.deterministic_threshold_stability(
            list(reversed(sequence_summaries)), **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["selected_support_threshold"], 0.5)
        self.assertEqual(first["bootstrap"]["samples"], 25)
        self.assertEqual(first["bootstrap"]["modal_support_threshold"], 0.5)
        self.assertEqual(first["bootstrap"]["modal_match_fraction"], 1.0)
        self.assertEqual(first["leave_one_group_out_same_threshold_count"], 5)


class QuarantineRunnerV2OptionalTest(unittest.TestCase):
    def test_namespaced_inner_split_is_deterministic_83_42_when_available(self):
        runner = runner_or_skip(self)
        function = next((getattr(runner, name) for name in (
            "split_design_internal", "split_inner_development",
            "split_design83_internal42", "inner_split_sequences",
            "split_sequences_by_namespace") if hasattr(runner, name)), None)
        if function is None:
            self.skipTest("Runner exposes no inner split API")
        sequences = ["seq{:03d}".format(index) for index in range(125)]
        signature = inspect.signature(function)
        kwargs = {}
        if "namespace" in signature.parameters:
            kwargs["namespace"] = "rmg-stage1-v2-q5-inner"
        if "design_count" in signature.parameters:
            kwargs["design_count"] = 83
        if "first_count" in signature.parameters:
            kwargs["first_count"] = 83
        first = function(sequences, **kwargs)
        second = function(list(reversed(sequences)), **kwargs)
        self.assertEqual(first, second)
        self.assertEqual((len(first[0]), len(first[1])), (83, 42))
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertEqual(set(first[0]) | set(first[1]), set(sequences))
        if hasattr(runner, "INNER_SPLIT_NAMESPACE"):
            self.assertEqual(runner.INNER_SPLIT_NAMESPACE, "rmg-stage1-v2-q5-inner")

    def test_delayed_timing_fake_tracker_when_runner_api_supports_it(self):
        runner = runner_or_skip(self)
        function = next((getattr(runner, name) for name in (
            "run_quarantine_timing_fixture", "exercise_quarantine_timing",
            "run_delayed_timing_fixture") if hasattr(runner, name)), None)
        if function is None:
            self.skipTest("Runner exposes no dependency-free delayed timing fixture API")
        result = function()
        self.assertEqual(result["source_frame"], 30)
        self.assertEqual(result["finalized_frame"], 35)
        self.assertEqual(result["effective_frame"], 36)
        self.assertTrue(result["source_frame_active_unaffected"])
        self.assertTrue(result["final_probe_active_unaffected"])
        self.assertTrue(result["all_probe_anchors_identical"])
        self.assertTrue(result["released_source_snapshot_not_probe_snapshot"])
        self.assertTrue(result["shadow_absent_from_history"])

    def test_complete_config_drift_fails_closed(self):
        runner = runner_or_skip(self)
        try:
            config = runner.load_config(CONFIG_PATH)
        except RuntimeError as error:
            if "PyYAML" in str(error):
                self.skipTest(str(error))
            raise
        runner.validate_config(config)
        drifted = dict(config)
        drifted["metrics"] = dict(config["metrics"], bootstrap_samples=1)
        with self.assertRaisesRegex(ValueError, "complete frozen default"):
            runner.validate_config(drifted)
        weakened_parent = dict(config)
        weakened_parent["parent_artifacts"] = dict(config["parent_artifacts"])
        weakened_parent["parent_artifacts"]["stage0"] = dict(
            config["parent_artifacts"]["stage0"], gate_pass_required=False)
        with self.assertRaises(ValueError):
            runner.validate_config(weakened_parent)

    def test_scope_membership_is_exact_order_and_video_is_removed(self):
        runner = runner_or_skip(self)
        try:
            import yaml  # noqa: F401
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest("PyYAML unavailable: {}".format(error))
        development = ["dev{:03d}".format(index) for index in range(187)]
        val47 = ["val{:03d}".format(index) for index in range(47)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development_path = root / "development.txt"
            val47_path = root / "val47.txt"
            development_path.write_text("\n".join(development) + "\n", encoding="utf-8")
            val47_path.write_text("\n".join(val47) + "\n", encoding="utf-8")
            design = runner.canonical_scope_sequences(
                "design83", development_path, val47_path)
            self.assertEqual(len(design), 83)
            runner.validate_scope_membership(
                "design83", design, development_path, val47_path)
            with self.assertRaisesRegex(RuntimeError, "set/order"):
                runner.validate_scope_membership(
                    "design83", list(reversed(design)), development_path, val47_path)
            with self.assertRaises(SystemExit):
                runner.parse_args([
                    "--phase", "online", "--dataset-root", "/d", "--split-file", "/s",
                    "--development-split", "/dev", "--val47-split", "/v",
                    "--output-dir", "/o", "--yaml", "x", "--scope", "design83",
                    "--support-iou", "0.5", "--video", "one",
                    "--stage0-parent", "/p0", "--stage1-v1-parent", "/p1"])

    def test_direct_threshold_fails_closed_outside_design(self):
        runner = runner_or_skip(self)
        try:
            config = runner.load_config(CONFIG_PATH)
        except RuntimeError as error:
            if "PyYAML" in str(error):
                self.skipTest(str(error))
            raise
        for scope in ("internal42", "confirm62", "val47"):
            with self.subTest(scope=scope):
                args = SimpleNamespace(
                    scope=scope, support_iou=0.5, policy_lock="")
                with self.assertRaisesRegex(ValueError, "only for design83"):
                    runner._resolve_support_iou(args, config, "parent", [])

    def test_completed_evaluation_rejects_stale_rmg_q_events(self):
        runner = runner_or_skip(self)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = "seq"
            sequence_root = runner.safe_sequence_dir(root, sequence)
            (sequence_root / "online").mkdir(parents=True, exist_ok=True)
            (sequence_root / "labels").mkdir(parents=True, exist_ok=True)
            frame = {"schema_version": runner.SCHEMA_VERSION, "sequence": sequence,
                     "arm": "static", "frame_idx": 0, "pred_xywh": [0, 0, 1, 1]}
            trace_hashes = {}
            for arm in runner.ARMS:
                row = dict(frame, arm=arm)
                runner.write_jsonl(sequence_root / "online" / (arm + ".frames.jsonl"), [row])
                trace_hashes[arm] = runner.canonical_hash([row])
                runner.write_jsonl(sequence_root / "online" / (arm + ".events.jsonl"), [])
            runner.write_jsonl(sequence_root / "online" / "rmg_q.events.jsonl", [])
            frame_labels = [{"sequence": sequence, "frame_idx": 0,
                             "gt_xywh": [0, 0, 1, 1]}]
            runner.write_jsonl(sequence_root / "labels" / "frames.jsonl", frame_labels)
            runner.write_jsonl(sequence_root / "labels" / "rmg_q.events.jsonl", [])
            summary = {"schema_version": runner.SCHEMA_VERSION,
                       "kind": "evaluated_sequence", "sequence": sequence,
                       "arms": {arm: arm_metrics(1.0, num_frames=1)
                                for arm in runner.ARMS},
                       "governance": {},
                       "online_trace_hashes": trace_hashes,
                       "online_rmg_q_events_hash": runner.canonical_hash([]),
                       "frame_labels_hash": runner.canonical_hash(frame_labels),
                       "event_labels_hash": runner.canonical_hash([])}
            summary["content_hash"] = runner.canonical_hash(summary)
            runner.write_json(sequence_root / "sequence_summary.json", summary)
            self.assertTrue(runner.completed_evaluated_sequence(root, sequence))
            runner.write_jsonl(sequence_root / "online" / "rmg_q.events.jsonl", [
                {"event_id": "stale", "sequence": sequence}])
            self.assertFalse(runner.completed_evaluated_sequence(root, sequence))

    def test_internal_action_gate_merges_50_10_overrides(self):
        runner = runner_or_skip(self)
        try:
            config = runner.load_config(CONFIG_PATH)
        except RuntimeError as error:
            if "PyYAML" in str(error):
                self.skipTest(str(error))
            raise
        internal = runner.action_gate_config_for_scope(config, "internal42")
        confirm = runner.action_gate_config_for_scope(config, "confirm62")
        self.assertEqual((internal["combined_writes_min"],
                          internal["combined_sequences_min"]), (50, 10))
        self.assertEqual((confirm["combined_writes_min"],
                          confirm["combined_sequences_min"]), (100, 20))

    def test_minimal_synthetic_policy_unlock_and_repeated_identity_are_rejected(self):
        runner = runner_or_skip(self)
        try:
            config = runner.load_config(CONFIG_PATH)
        except RuntimeError as error:
            if "PyYAML" in str(error):
                self.skipTest(str(error))
            raise
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequences = ["s"]
            lock = {"schema_version": runner.SCHEMA_VERSION,
                    "kind": "final_stage1_quarantine_v2_policy",
                    "support_iou": 0.5, "confirmation_outcomes_accessed": False,
                    "parent_provenance_hash": "parent",
                    "config_hash": runner.canonical_hash(config)}
            lock["content_hash"] = runner.canonical_hash(lock)
            lock_path = root / "lock.json"
            runner.write_json(lock_path, lock)
            with self.assertRaises(RuntimeError):
                runner.validate_policy_lock(
                    lock_path, "parent", expected_scope_sequences=sequences,
                    expected_config_hash=runner.canonical_hash(config), config=config,
                    expected_internal_sequences=["i"])

            unlock = {"schema_version": runner.SCHEMA_VERSION,
                      "kind": "rmg_stage1_v2_val47_unlock",
                      "confirm_sequence_count": 62, "confirm_gate_pass": True,
                      "policy_lock_content_hash": "policy",
                      "parent_provenance_hash": "parent",
                      "confirm_sequences": ["s{:02d}".format(i) for i in range(62)],
                      "confirm_sequences_hash": "fabricated"}
            unlock["content_hash"] = runner.canonical_hash(unlock)
            unlock_path = root / "unlock.json"
            runner.write_json(unlock_path, unlock)
            with self.assertRaises(RuntimeError):
                runner.validate_val47_unlock(
                    unlock_path, "policy", "parent", unlock["confirm_sequences"],
                    config=config, policy_lock_path=lock_path,
                    expected_design_sequences=sequences,
                    expected_internal_sequences=["i"])

            candidates = [{"support_iou": threshold, "root": str(root),
                           "source_identity_hash": "identity", "aggregate": {},
                           "gate": {}, "sequence_summaries_hash": "summary"}
                          for threshold in (0.5, 0.6, 0.7, 0.8, 0.9)]
            provisional = {
                "schema_version": runner.SCHEMA_VERSION,
                "kind": "provisional_stage1_quarantine_v2_policy", "support_iou": 0.5,
                "design_scope": {"sequences": sequences,
                                 "sequences_hash": runner.canonical_hash(sequences)},
                "design_selection": {"selected_support_threshold": 0.5,
                                     "selection_gate": {}, "stability": {},
                                     "candidates": candidates},
                "config_hash": runner.canonical_hash(config),
                "parent_provenance_hash": "parent",
                "source_identity_hashes": ["identity"] * 5,
                "source_sha256": {name: runner.file_sha256(path)
                                  for name, path in runner.IDENTITY_SOURCES.items()},
                "confirmation_outcomes_accessed": False,
            }
            provisional["content_hash"] = runner.canonical_hash(provisional)
            provisional_path = root / "provisional.json"
            runner.write_json(provisional_path, provisional)
            with self.assertRaisesRegex(RuntimeError, "five distinct"):
                runner.validate_policy_lock(
                    provisional_path, "parent",
                    required_kind="provisional_stage1_quarantine_v2_policy",
                    expected_scope_sequences=sequences,
                    expected_config_hash=runner.canonical_hash(config), config=config)

    def test_stage0_manifest_index_realistic_and_malformed_rejected(self):
        runner = runner_or_skip(self)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = write_realistic_stage0_manifest_fixture(
                runner, root, ["s{:02d}".format(i) for i in range(3)])
            self.assertEqual(
                runner.validate_stage0_manifest_index(
                    root, index["content_hash"], expected_sequence_count=3), index)
            duplicate = dict(index)
            duplicate["sequences"] = list(index["sequences"])
            duplicate["sequences"][1] = dict(duplicate["sequences"][0])
            runner.write_json(root / "manifest.json", duplicate)
            with self.assertRaisesRegex(RuntimeError, "unique"):
                runner.validate_stage0_manifest_index(
                    root, expected_sequence_count=3)

            index = write_realistic_stage0_manifest_fixture(
                runner, root, ["s{:02d}".format(i) for i in range(3)])
            nested_path = root / "sequences" / "s00" / "manifest.json"
            nested = runner.read_json(nested_path)
            nested["sequence"] = "other"
            nested["manifest_hash"] = runner.canonical_hash({
                key: value for key, value in nested.items() if key != "manifest_hash"})
            runner.write_json(nested_path, nested)
            with self.assertRaisesRegex(RuntimeError, "binding"):
                runner.validate_stage0_manifest_index(
                    root, index["content_hash"], expected_sequence_count=3)

    def test_corrupt_metadata_identity_hash_fails_before_artifact_acceptance(self):
        runner = runner_or_skip(self)
        try:
            config = runner.load_config(CONFIG_PATH)
        except RuntimeError as error:
            if "PyYAML" in str(error):
                self.skipTest(str(error))
            raise
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"schema_version": runner.SCHEMA_VERSION,
                        "kind": "stage1_quarantine_v2_run_metadata",
                        "identity": {"scope": "design83", "support_iou": 0.5},
                        "identity_hash": "fabricated", "parent_provenance_hash": "parent",
                        "policy_lock_content_hash": None, "frame_indexing": "zero_based",
                        "causality": "x", "label_policy": "x",
                        "completed_phases": ["online", "evaluate"],
                        "created_unix": 0, "updated_unix": 1}
            metadata["content_hash"] = runner.canonical_hash(metadata)
            runner.write_json(root / "metadata.json", metadata)
            with self.assertRaises(RuntimeError):
                runner.validate_metadata(
                    root / "metadata.json", "design83", ["s"], 0.5, config,
                    "parent", policy_hash=None)

    def test_lock_provenance_and_confirm_val47_unlock_fail_closed_when_available(self):
        runner = runner_or_skip(self)
        validator = next((getattr(runner, name) for name in (
            "validate_val47_unlock", "require_val47_unlock",
            "validate_unlock_chain") if hasattr(runner, name)), None)
        provenance = next((getattr(runner, name) for name in (
            "validate_parent_provenance", "validate_parent_artifacts",
            "require_parent_provenance") if hasattr(runner, name)), None)
        if validator is None and provenance is None:
            self.skipTest("Runner exposes no provenance or unlock validation API")
        if provenance is not None:
            with self.assertRaises((RuntimeError, ValueError, FileNotFoundError)):
                provenance({})
        if validator is not None:
            signature = inspect.signature(validator)
            with tempfile.TemporaryDirectory() as directory:
                missing = Path(directory) / "missing.json"
                args = []
                kwargs = {}
                for parameter in signature.parameters.values():
                    if parameter.default is not inspect.Parameter.empty:
                        continue
                    value = missing
                    if "hash" in parameter.name:
                        value = "wrong"
                    if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                          inspect.Parameter.POSITIONAL_OR_KEYWORD):
                        args.append(value)
                    else:
                        kwargs[parameter.name] = value
                with self.assertRaises((RuntimeError, ValueError, FileNotFoundError)):
                    validator(*args, **kwargs)


class FrozenConfigurationV2Test(unittest.TestCase):
    def test_yaml_exact_frozen_values(self):
        try:
            import yaml
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest("PyYAML unavailable: {}".format(error))
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["schema_version"], "rmg-stage1-v2-q5")
        self.assertEqual(config["schedule"], {
            "warmup": 30, "interval": 30, "cooldown": 60})
        self.assertEqual(config["candidate_validity"], {
            "min_size": 8.0, "min_intersection_ratio": 0.75,
            "max_padding_ratio": 0.25})
        self.assertEqual(config["triage"], {
            "immediate_max_entropy": 0.45, "quarantine_max_entropy": 0.50})
        self.assertEqual(config["quarantine"], {
            "slots": 1, "probe_offsets": [1, 3, 5], "min_support": 2,
            "support_iou_candidates": [0.50, 0.60, 0.70, 0.80, 0.90]})
        self.assertEqual(config["split"]["outer"], {
            "tune_count": 125, "confirm_count": 62})
        self.assertEqual(config["split"]["inner"], {
            "namespace": "rmg-stage1-v2-q5-inner",
            "design_count": 83, "internal_count": 42})
        self.assertEqual(config["arms"], [
            "static", "periodic_pred", "confidence_e050", "rmg_q",
            "rmg_q_no_quarantine"])
        self.assertEqual(config["thresholds"], {"good_iou": 0.7, "bad_iou": 0.1})
        gate = config["action_gate"]
        self.assertEqual((gate["immediate_precision_min"],
                          gate["immediate_bad_rate_max"],
                          gate["immediate_coverage_min"]), (0.90, 0.02, 0.03))
        self.assertEqual((gate["combined_coverage_min"],
                          gate["combined_bad_rate_max"],
                          gate["release_bad_rate_max"]), (0.10, 0.05, 0.05))
        self.assertEqual(gate["combined_writes_min"], 100)
        self.assertEqual(gate["combined_sequences_min"], 20)
        self.assertEqual(gate["quarantine_incremental_auc_delta_exclusive_min"], 0.0)
        self.assertEqual(config["internal_action_gate"], {
            "combined_writes_min": 50, "combined_sequences_min": 10})
        self.assertEqual(config["stability"], {
            "bootstrap_seed": 20260716, "bootstrap_samples": 2000,
            "modal_match_fraction_min": 0.80,
            "leave_one_group_out": {"groups": 5, "same_threshold_min": 4}})
        self.assertEqual(config["metrics"], {
            "bootstrap_seed": 20260716, "bootstrap_samples": 2000})
        final = config["final_tracking_gate"]
        self.assertEqual(final, {
            "rmg_q_static_auc_delta_min": 0.005,
            "rmg_q_periodic_auc_delta_min": 0.010,
            "rmg_q_periodic_auc_ci_low_exclusive_min": 0.0,
            "worsened_fraction_max": 0.20,
            "clean_auc_delta_min": -0.002,
            "clean_worsened_fraction_max": 0.10,
            "clean_subset_nonempty": True,
            "rmg_q_no_quarantine_auc_delta_exclusive_min": 0.0,
            "rmg_q_confidence_e050_auc_delta_exclusive_min": 0.0,
        })
        self.assertEqual(config["exposure"], {
            "design83": "adaptive_development",
            "internal42": "adaptive_development_one_shot",
            "confirm62": "outcome_blind_one_shot",
            "confirm62_prior_access": "gt_derived_features_integrity_only_no_outcomes",
            "val47": "locked_validation_one_shot",
        })
        parents = config["parent_artifacts"]
        self.assertEqual(parents["stage0"]["aggregate_sha256"],
                         "76a0b7f5e4701ce57086548f405cfdac979ad015bff3c7eda5f28ba187bff9dd")
        self.assertEqual(parents["stage0"]["gate_sha256"],
                         "bc2b339a8956fec47799be4c30c9bdfa52feddf214d0999fe7e2d647466ec0f6")
        self.assertEqual(parents["stage0"]["manifest_content_hash"],
                         "9890d9fd0040024143172db0594684811bfb60a50da861bffdf193b48db194e3")
        self.assertEqual(parents["stage1_v1"]["tuning_failure_sha256"],
                         "cc67615e1eb778cfafe57aa66901f8840ae685fa918a324b92c3489902e647a0")
        self.assertTrue(parents["stage0"]["gate_pass_required"])
        self.assertTrue(parents["stage1_v1"]["lock_must_be_absent"])
        self.assertTrue(parents["stage1_v1"]["arms_must_be_absent"])
        self.assertTrue(parents["stage1_v1"]["gate_must_be_absent"])


if __name__ == "__main__":
    unittest.main()
