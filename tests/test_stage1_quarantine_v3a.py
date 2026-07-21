import copy
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = ROOT / "analysis" / "memory_oracle" / "quarantine_controller_v3a.py"
METRICS_PATH = ROOT / "analysis" / "memory_oracle" / "compute_stage1_quarantine_v3a_metrics.py"
RUNNER_PATH = ROOT / "analysis" / "memory_oracle" / "run_stage1_quarantine_v3a.py"
CONFIG_PATH = ROOT / "analysis" / "memory_oracle" / "configs" / "stage1_quarantine_v3a.yaml"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = load_module("stage1_quarantine_v3a_controller_tests", CONTROLLER_PATH)
metrics = load_module("stage1_quarantine_v3a_metrics_tests", METRICS_PATH)
sys.modules["analysis.memory_oracle.quarantine_controller_v3a"] = controller
sys.modules["analysis.memory_oracle.compute_stage1_quarantine_v3a_metrics"] = metrics
runner = load_module("stage1_quarantine_v3a_runner_tests", RUNNER_PATH)


def observation(frame_idx=30, entropy=0.47, box=None):
    return {"frame_idx": frame_idx,
            "pred_xywh": [10.0, 12.0, 20.0, 16.0] if box is None else box,
            "search_anchor_xywh": [10.0, 12.0, 20.0, 16.0],
            "image_shape": [100, 120], "response_entropy": entropy}


def bucket(count=100, good=98, bad=2, coverage=0.10, sequences=20):
    return {"count": count, "good_count": good, "bad_count": bad,
            "precision": float(good / count) if count else 0.0,
            "bad_rate": float(bad / count) if count else 0.0,
            "coverage": coverage, "good_recall": 0.0,
            "sequence_count": sequences, "sequence_coverage": 0.5}


def replay_config():
    return {"schedule": {"warmup": 30, "interval": 30, "cooldown": 60},
            "candidate_validity": {"min_size": 8.0,
                                   "min_intersection_ratio": 0.75,
                                   "max_padding_ratio": 0.25},
            "thresholds": {"good_iou": 0.7, "bad_iou": 0.1}}


def replay_rows(arm, num_frames=96):
    initial = "initial:0:10.000000,12.000000,20.000000,16.000000"
    rows = []
    for frame in range(num_frames):
        rows.append({"schema_version": runner.SCHEMA_VERSION, "sequence": "s",
                     "arm": arm, "frame_idx": frame,
                     "pred_xywh": [10.0, 12.0, 20.0, 16.0],
                     "best_score": 1.0, "response_entropy": 0.45,
                     "search_anchor_xywh": [10.0, 12.0, 20.0, 16.0],
                     "template_id_for_prediction": initial,
                     "is_update_opportunity": frame == 30,
                     "action_after_prediction": "none"})
    return rows


def replay_event(action="release"):
    source_box = [10.0, 12.0, 20.0, 16.0]
    probes = []
    for offset in (1, 3, 5):
        probes.append({"frame_idx": 30 + offset, "offset": offset,
                       "image_shape": [100, 120],
                       "shared_anchor_xywh": source_box,
                       "active_xywh": source_box, "shadow_xywh": source_box,
                       "active_template_id": (
                           "initial:0:10.000000,12.000000,20.000000,16.000000"),
                       "shadow_template_id": (
                           "quarantine_source_prediction:30:"
                           "10.000000,12.000000,20.000000,16.000000"),
                       "shadow_search_anchor_xywh": source_box,
                       "active_legal": True, "active_illegal_reason": None,
                       "shadow_legal": True, "shadow_illegal_reason": None,
                       "agreement_iou": 1.0, "supports_release": True})
    event = {"schema_version": runner.SCHEMA_VERSION, "event_id": "s:000030",
             "sequence": "s", "source_frame": 30,
             "source_candidate_xywh": source_box,
             "source_search_anchor_xywh": source_box,
             "source_image_shape": [100, 120], "admission_entropy": 0.45,
             "legal_source_opportunity": True, "candidate_invalid_reason": None,
             "action": action, "reason": "probe_support_pass", "admitted": True,
             "quarantined": True, "immediate_write": False, "effective_frame": 36,
             "probes": probes, "support_count": 3, "probe_count": 3,
             "finalized_frame": 35, "released": True}
    return event


def valid_replay_fixture():
    rows = replay_rows("rmg_qh_qonly")
    event = replay_event()
    rows[30]["action_after_prediction"] = "quarantine_source_snapshot"
    rows[35]["action_after_prediction"] = "release_source_snapshot"
    released = runner._canonical_template_id(
        "quarantine_source_prediction", 30, event["source_candidate_xywh"])
    for frame in range(36, len(rows)):
        rows[frame]["template_id_for_prediction"] = released
    shapes = {frame: [100, 120] for frame in (30, 31, 33, 35)}
    return rows, [event], shapes


def candidate(threshold, passes=True):
    release = bucket()
    if not passes:
        release["bad_rate"] = 0.051
    governance = {"num_legal_source_opportunities": 1000,
                  "num_good_source_opportunities": 200,
                  "old_immediate_eligible_count": 50,
                  "old_immediate_accounted_count": 50,
                  "old_immediate_accounted_fraction": 1.0,
                  "admission": bucket(200, 180, 5, 0.20, 20),
                  "quarantine": bucket(200, 180, 5, 0.20, 20),
                  "release": release, "discard": bucket(100, 80, 3, 0.10, 20),
                  "combined": dict(release)}
    comparisons = {}
    for baseline, delta in (("static", 0.01), ("periodic_pred", 0.02),
                            ("rmg_qh_noq", 0.005)):
        comparisons["rmg_qh_qonly_vs_{}".format(baseline)] = {
            "success_auc": {"mean": delta, "low": delta, "high": delta,
                            "samples": 1, "seed": 0}}
    return {"support_threshold": threshold, "governance": governance,
            "frame_weighted_paired_deltas": comparisons,
            "protocol_level_release_future_gain": 0.005,
            "worsened_vs_static": {"fraction": 0.10},
            "clean_subset_preservation": {
                "num_sequences": 10,
                "rmg_qh_qonly_vs_static_success_auc": {"mean": 0.0},
                "worsened_vs_static": {"fraction": 0.05}}}


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.policy = controller.QuarantinePolicy(0.7)

    def test_qonly_boundaries_and_no_immediate_vocabulary(self):
        for entropy in (0.0, 0.45, math.nextafter(0.45, math.inf), 0.50):
            decision = controller.triage(observation(entropy=entropy), self.policy, True)
            self.assertEqual((decision.action, decision.reason),
                             ("quarantine", "entropy_qonly_admission"))
        decision = controller.triage(
            observation(entropy=math.nextafter(0.50, math.inf)), self.policy, True)
        self.assertEqual((decision.action, decision.reason),
                         ("skip", "entropy_above_quarantine"))
        self.assertNotIn("immediate", {controller.triage(
            observation(entropy=value), self.policy, True).action
            for value in (0.0, 0.45, 0.50, 0.9)})

    def test_nonfinite_invalid_and_slot_occupied(self):
        for entropy in (float("nan"), float("inf"), "bad"):
            self.assertEqual(controller.triage(
                observation(entropy=entropy), self.policy, True).reason,
                "non_finite_entropy")
        self.assertEqual(controller.triage(observation(), self.policy, False,
                                           "geometry").reason, "invalid:geometry")
        state, first = controller.admit_opportunity(
            controller.QuarantineState(), "s", observation(entropy=0.45),
            self.policy, True)
        self.assertEqual(first.action, "quarantine")
        same, occupied = controller.admit_opportunity(
            state, "s", observation(60, 0.2), self.policy, True)
        self.assertIs(same, state)
        self.assertEqual((occupied.action, occupied.reason),
                         ("skip", "quarantine_slot_occupied"))

    def _probe(self, state, frame, supports):
        active = [10, 10, 20, 20]
        shadow = active if supports else [100, 100, 20, 20]
        anchor = [5, 5, 30, 30]
        return controller.record_probe(state, self.policy, state.pending.event_id,
                                       frame, active, shadow, True, True,
                                       anchor, anchor)

    def test_q5_shared_anchor_two_of_three_release_t6_source_snapshot(self):
        state, _ = controller.admit_opportunity(
            controller.QuarantineState(), "s", observation(), self.policy, True)
        source_box = state.pending.source_candidate_xywh
        for frame, supports in ((31, True), (33, False), (35, True)):
            self.assertEqual(controller.due_probe_offsets(state, frame, self.policy),
                             (frame - 30,))
            state, evidence = self._probe(state, frame, supports)
            self.assertEqual(evidence.shared_anchor_xywh, (5.0, 5.0, 30.0, 30.0))
        _, finalized = controller.finalize_quarantine(
            state, self.policy, state.pending.event_id, 35)
        self.assertEqual((finalized.action, finalized.support_count,
                          finalized.finalized_frame, finalized.effective_frame),
                         ("release", 2, 35, 36))
        self.assertEqual(finalized.source_candidate_xywh, source_box)

    def test_one_of_three_discards(self):
        state, _ = controller.admit_opportunity(
            controller.QuarantineState(), "s", observation(), self.policy, True)
        for frame, supports in ((31, False), (33, True), (35, False)):
            state, _ = self._probe(state, frame, supports)
        _, finalized = controller.finalize_quarantine(
            state, self.policy, state.pending.event_id, 35)
        self.assertEqual(finalized.action, "discard")
        self.assertIsNone(finalized.effective_frame)


class MetricsTest(unittest.TestCase):
    def test_denominator_legality_is_explicit_and_combined_equals_release(self):
        online = [
            {"event_id": "a", "sequence": "s", "action": "release",
             "reason": "probe_support_pass", "legal_source_opportunity": True,
             "admission_entropy": 0.45, "admitted": True,
             "quarantined": True, "immediate_write": False},
            {"event_id": "b", "sequence": "s", "action": "skip",
             "reason": "entropy_above_quarantine", "legal_source_opportunity": True,
             "admission_entropy": 0.51, "immediate_write": False},
            {"event_id": "c", "sequence": "s", "action": "skip",
             "reason": "invalid:geometry", "legal_source_opportunity": False,
             "admission_entropy": 0.20, "immediate_write": False},
        ]
        labels = [{"event_id": "a", "source_candidate_iou": 0.0,
                   "release_frame_iou": 1.0},
                  {"event_id": "b", "source_candidate_iou": 0.8},
                  {"event_id": "c", "source_candidate_iou": 0.9}]
        result = metrics.governance_metrics(online, labels)
        self.assertEqual(result["num_legal_source_opportunities"], 2)
        self.assertEqual(result["release"], result["combined"])
        self.assertEqual(result["release_bad_count"], 1)
        self.assertEqual(result["immediate_writes"], 0)
        self.assertEqual((result["old_immediate_eligible_count"],
                          result["old_immediate_accounted_count"],
                          result["old_immediate_accounted_fraction"]),
                         (1, 1, 1.0))
        missing = [dict(online[0])]
        del missing[0]["legal_source_opportunity"]
        with self.assertRaisesRegex(ValueError, "explicit legal_source_opportunity"):
            metrics.governance_metrics(missing, labels[:1])
        nonbool = [dict(online[0], legal_source_opportunity=1)]
        with self.assertRaises(TypeError):
            metrics.governance_metrics(nonbool, labels[:1])

    def test_label_firewall_and_strict_joins(self):
        online = [{"sequence": "s", "frame_idx": 0,
                   "pred_xywh": [0, 0, 10, 10]}]
        labels = [{"sequence": "s", "frame_idx": 0,
                   "gt_xywh": [0, 0, 10, 10]}]
        self.assertEqual(metrics.tracking_metrics_from_online_trace(
            online, labels)["mean_iou"], 1.0)
        with self.assertRaisesRegex(ValueError, "forbidden GT/IoU"):
            metrics.join_frame_traces([dict(online[0], gt_xywh=[0, 0, 1, 1])], labels)
        with self.assertRaises(ValueError):
            metrics.join_frame_traces(online + [dict(online[0])], labels)

    def test_fixed_grid_smallest_all_gates_and_combined_mismatch(self):
        candidates = [candidate(value) for value in metrics.SUPPORT_THRESHOLDS]
        selected = metrics.select_smallest_passing_support_threshold(candidates)
        self.assertEqual(selected["selected_support_threshold"], 0.5)
        candidates[0] = candidate(0.5, False)
        self.assertEqual(metrics.select_smallest_passing_support_threshold(
            candidates)["selected_support_threshold"], 0.6)
        with self.assertRaisesRegex(ValueError, "exactly the five"):
            metrics.select_smallest_passing_support_threshold(candidates[:-1])
        broken = candidate(0.5)
        broken["governance"]["combined"]["count"] += 1
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            metrics.evaluate_action_quality(broken)
        unaccounted = candidate(0.5)
        unaccounted["governance"]["old_immediate_accounted_fraction"] = 0.98
        self.assertFalse(metrics.evaluate_action_quality(unaccounted)["pass"])
        missing_pool = candidate(0.5)
        missing_pool["governance"]["old_immediate_eligible_count"] = 0
        self.assertFalse(metrics.evaluate_action_quality(missing_pool)["pass"])

    def test_stability_requires_modal_selected_and_four_logo(self):
        summaries = []
        for index in range(50):
            per_threshold = []
            for threshold in metrics.SUPPORT_THRESHOLDS:
                item = candidate(threshold)
                per_threshold.append({"support_threshold": threshold,
                                      "num_frames": 100,
                                      "governance": item["governance"],
                                      "auc_deltas": {"static": 0.01,
                                                     "periodic_pred": 0.02,
                                                     "rmg_qh_noq": 0.005},
                                      "clean_subset": True})
            summaries.append({"sequence": "s{:02d}".format(index),
                              "threshold_candidates": per_threshold})
        result = metrics.deterministic_threshold_stability(
            summaries, seed=3, samples=25)
        self.assertTrue(result["pass"])
        self.assertEqual(result["selected_support_threshold"], 0.5)
        self.assertEqual(result["bootstrap"]["selected_threshold_match_fraction"], 1.0)
        self.assertGreaterEqual(
            result["leave_one_group_out_same_non_none_threshold_count"], 4)


class RunnerTest(unittest.TestCase):
    def test_exact_v2_inner_namespace_reuse_and_canonical_roles(self):
        self.assertEqual(runner.INNER_SPLIT_NAMESPACE, "rmg-stage1-v2-q5-inner")
        sequences = ["s{:03d}".format(index) for index in range(125)]
        first = runner.split_design_internal(sequences)
        second = runner.split_design_internal(list(reversed(sequences)))
        self.assertEqual(first, second)
        self.assertEqual((len(first[0]), len(first[1])), (83, 42))

    def test_noq_discard_event_and_static_prediction_equivalence(self):
        policy = controller.QuarantinePolicy(0.7)
        _, admitted = controller.admit_opportunity(
            controller.QuarantineState(), "s", observation(30, 0.45), policy, True)
        discarded = type(admitted)("discard", "quarantine_disabled",
                                   admitted.event_id, admitted.source_frame, None)
        row = runner._event_row("s", observation(30, 0.45), discarded, True, None)
        row["admitted"] = True
        self.assertEqual((row["action"], row["reason"], row["quarantined"],
                          row["probes"]),
                         ("discard", "quarantine_disabled", False, []))
        static = [{"frame_idx": 0, "pred_xywh": [1, 2, 3, 4], "arm": "static"}]
        noq = [{"frame_idx": 0, "pred_xywh": [1, 2, 3, 4], "arm": "rmg_qh_noq",
                "action_after_prediction": "discard:quarantine_disabled"}]
        self.assertIsInstance(runner.validate_static_noq_equivalence(static, noq), str)
        noq[0]["pred_xywh"][0] = 2
        with self.assertRaisesRegex(RuntimeError, "not numerically equivalent"):
            runner.validate_static_noq_equivalence(static, noq)

    def test_timing_fixture_and_phase_cli(self):
        timing = runner.run_quarantine_timing_fixture()
        self.assertEqual((timing["source_frame"], timing["finalized_frame"],
                          timing["effective_frame"]), (30, 35, 36))
        self.assertTrue(timing["released_source_snapshot_not_probe_snapshot"])
        with self.assertRaises(SystemExit):
            runner.parse_args(["--phase", "all"])

    def test_strict_replay_rejects_event_omission_and_fabricated_legality(self):
        rows, events, shapes = valid_replay_fixture()
        runner.validate_event_semantic_replay(
            "s", "rmg_qh_qonly", rows, events, replay_config(), 0.7,
            image_shapes=shapes)
        with self.assertRaisesRegex(RuntimeError, "exactly one event"):
            runner.validate_event_semantic_replay(
                "s", "rmg_qh_qonly", rows, [], replay_config(), 0.7,
                image_shapes=shapes)
        forged = copy.deepcopy(events)
        forged[0]["legal_source_opportunity"] = False
        forged[0]["candidate_invalid_reason"] = "bbox_too_small"
        with self.assertRaisesRegex(RuntimeError, "source evidence"):
            runner.validate_event_semantic_replay(
                "s", "rmg_qh_qonly", rows, forged, replay_config(), 0.7,
                image_shapes=shapes)

    def test_strict_replay_rejects_old_immediate_eligible_changed_to_skip(self):
        rows, events, shapes = valid_replay_fixture()
        forged = copy.deepcopy(events)
        event = forged[0]
        event.update({"action": "skip", "reason": "entropy_above_quarantine",
                      "admitted": False, "quarantined": False,
                      "effective_frame": None, "probes": []})
        for key in ("support_count", "probe_count", "finalized_frame", "released"):
            del event[key]
        with self.assertRaisesRegex(RuntimeError, "Admitted qonly event"):
            runner.validate_event_semantic_replay(
                "s", "rmg_qh_qonly", rows, forged, replay_config(), 0.7,
                image_shapes=shapes)

    def test_strict_replay_rejects_forged_support_release_and_template_timing(self):
        rows, events, shapes = valid_replay_fixture()
        attacks = []
        forged_support = copy.deepcopy(events)
        forged_support[0]["probes"][0]["supports_release"] = False
        attacks.append((rows, forged_support, "Probe evidence"))
        forged_agreement = copy.deepcopy(events)
        forged_agreement[0]["probes"][0]["agreement_iou"] = 0.0
        attacks.append((rows, forged_agreement, "Probe evidence"))
        forged_shadow_template = copy.deepcopy(events)
        forged_shadow_template[0]["probes"][0]["shadow_template_id"] = rows[0][
            "template_id_for_prediction"]
        attacks.append((rows, forged_shadow_template, "Probe evidence"))
        forged_shadow_anchor = copy.deepcopy(events)
        forged_shadow_anchor[0]["probes"][0]["shadow_search_anchor_xywh"] = [1, 1, 20, 16]
        attacks.append((rows, forged_shadow_anchor, "Probe evidence"))
        forged_release = copy.deepcopy(events)
        forged_release[0]["effective_frame"] = 35
        attacks.append((rows, forged_release, "finalization"))
        forged_rows = copy.deepcopy(rows)
        forged_rows[36]["template_id_for_prediction"] = forged_rows[0][
            "template_id_for_prediction"]
        attacks.append((forged_rows, events, "template write"))
        forged_anchor_rows = copy.deepcopy(rows)
        forged_anchor_rows[31]["search_anchor_xywh"] = [1, 1, 20, 16]
        attacks.append((forged_anchor_rows, events, "search anchor"))
        for attacked_rows, attacked_events, message in attacks:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    runner.validate_event_semantic_replay(
                        "s", "rmg_qh_qonly", attacked_rows, attacked_events,
                        replay_config(), 0.7, image_shapes=shapes)

    def test_exact_frame_schema_rejects_label_aliases(self):
        rows, events, shapes = valid_replay_fixture()
        rows[0]["truth_xywh"] = [1, 2, 3, 4]
        with self.assertRaisesRegex(RuntimeError, "exact label-free schema"):
            runner.validate_event_semantic_replay(
                "s", "rmg_qh_qonly", rows, events, replay_config(), 0.7,
                image_shapes=shapes)

    def test_static_and_periodic_replay_bind_actions_templates_and_anchor_chain(self):
        shapes = {30: [100, 120]}
        static = replay_rows("static")
        runner.validate_event_semantic_replay(
            "s", "static", static, [], replay_config(), 0.7, image_shapes=shapes)
        static[30]["action_after_prediction"] = "commit_current_arm_prediction"
        with self.assertRaisesRegex(RuntimeError, "Baseline frame action"):
            runner.validate_event_semantic_replay(
                "s", "static", static, [], replay_config(), 0.7, image_shapes=shapes)

        periodic = replay_rows("periodic_pred")
        periodic[30]["action_after_prediction"] = "commit_current_arm_prediction"
        updated = runner._canonical_template_id(
            "current_arm_prediction", 30, periodic[30]["pred_xywh"])
        for frame in range(31, len(periodic)):
            periodic[frame]["template_id_for_prediction"] = updated
        runner.validate_event_semantic_replay(
            "s", "periodic_pred", periodic, [], replay_config(), 0.7,
            image_shapes=shapes)
        periodic[31]["template_id_for_prediction"] = periodic[0][
            "template_id_for_prediction"]
        with self.assertRaisesRegex(RuntimeError, "template transition"):
            runner.validate_event_semantic_replay(
                "s", "periodic_pred", periodic, [], replay_config(), 0.7,
                image_shapes=shapes)

    def test_noq_replay_requires_discard_and_no_probes_or_writes(self):
        rows = replay_rows("rmg_qh_noq")
        event = replay_event()
        for key in ("support_count", "probe_count", "finalized_frame", "released"):
            del event[key]
        event.update({"action": "discard", "reason": "quarantine_disabled",
                      "quarantined": False, "effective_frame": None, "probes": []})
        rows[30]["action_after_prediction"] = "discard:quarantine_disabled"
        shapes = {frame: [100, 120] for frame in (30, 31, 33, 35)}
        runner.validate_event_semantic_replay(
            "s", "rmg_qh_noq", rows, [event], replay_config(), 0.7,
            image_shapes=shapes)
        forged = copy.deepcopy(event)
        forged.update({"action": "skip", "reason": "entropy_above_quarantine",
                       "admitted": False})
        with self.assertRaisesRegex(RuntimeError, "noq event"):
            runner.validate_event_semantic_replay(
                "s", "rmg_qh_noq", rows, [forged], replay_config(), 0.7,
                image_shapes=shapes)

    def test_authoritative_label_regeneration_rejects_fabricated_store(self):
        original_loader = runner._load_evaluation_sequence
        try:
            runner._load_evaluation_sequence = lambda root, sequence: (
                [], [], [[1.0, 2.0, 10.0, 10.0]])
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                labels = root / "sequences" / "s" / "labels"
                labels.mkdir(parents=True)
                runner.write_jsonl(labels / "frames.jsonl", [{
                    "schema_version": runner.SCHEMA_VERSION, "sequence": "s",
                    "frame_idx": 0, "gt_xywh": [9.0, 9.0, 10.0, 10.0]}])
                with self.assertRaisesRegex(RuntimeError, "authoritative GT"):
                    runner._recompute_authoritative_sequence_summary(
                        root, "s", replay_config(), 0.7, root)
        finally:
            runner._load_evaluation_sequence = original_loader

    def test_authoritative_completion_recomputes_from_output_root(self):
        originals = (runner.completed_online_sequence,
                     runner.validate_online_trace,
                     runner._recompute_authoritative_sequence_summary)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sequence_root = root / "sequences" / "s"
                (sequence_root / "labels").mkdir(parents=True)
                (sequence_root / "online").mkdir()
                arm_metrics = {
                    key: 0 if key == "num_frames" else 0.0
                    for key in runner._ARM_METRIC_KEYS
                }
                summary = {
                    "schema_version": runner.SCHEMA_VERSION,
                    "kind": "evaluated_sequence",
                    "sequence": "s",
                    "arms": {arm: dict(arm_metrics) for arm in runner.ARMS},
                    "governance": {},
                    "online_trace_hashes": {
                        arm: runner.canonical_hash([]) for arm in runner.ARMS},
                    "online_rmg_qh_qonly_events_hash": runner.canonical_hash([]),
                    "frame_labels_hash": runner.canonical_hash([]),
                    "event_labels_hash": runner.canonical_hash([]),
                }
                summary["content_hash"] = runner.canonical_hash(summary)
                runner.write_json(sequence_root / "sequence_summary.json", summary)
                runner.write_jsonl(sequence_root / "labels" / "frames.jsonl", [])
                runner.write_jsonl(
                    sequence_root / "labels" / "rmg_qh_qonly.events.jsonl", [])
                for arm in runner.ARMS:
                    runner.write_jsonl(
                        sequence_root / "online" / (arm + ".frames.jsonl"), [])
                runner.write_jsonl(
                    sequence_root / "online" / "rmg_qh_qonly.events.jsonl", [])
                observed = []
                runner.completed_online_sequence = lambda *args, **kwargs: True
                runner.validate_online_trace = lambda *args, **kwargs: None
                runner._recompute_authoritative_sequence_summary = (
                    lambda output_root, *args, **kwargs:
                    observed.append(Path(output_root)) or summary)
                self.assertTrue(runner.completed_evaluated_sequence(
                    root, "s", 0.7, dataset_root=root,
                    config=replay_config(), authoritative=True))
                self.assertEqual(observed, [root])
        finally:
            (runner.completed_online_sequence,
             runner.validate_online_trace,
             runner._recompute_authoritative_sequence_summary) = originals

    def test_dataset_manifest_binds_gt_bytes_and_validation_rejects_change(self):
        original_load_config = runner.load_config
        try:
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory).resolve()
                root = work / "dataset"
                sequence = root / "s"
                for modality in ("visible", "infrared"):
                    image_dir = sequence / modality
                    image_dir.mkdir(parents=True)
                    (image_dir / "0001.jpg").write_bytes(
                        b"jpeg-" + modality.encode())
                (sequence / "visible.txt").write_text(
                    "1,2,10,10\n", encoding="utf-8")
                (sequence / "infrared.txt").write_text(
                    "1,2,10,10\n", encoding="utf-8")
                sealed = runner._dataset_manifest_hash(root, ["s"])
                config = replay_config()
                references = {}
                for name in ("experiment_config", "runner_config", "checkpoint", "split"):
                    path = work / (name + ".bin")
                    path.write_bytes(name.encode())
                    references[name] = {"path": str(path),
                                        "sha256": runner.file_sha256(path)}
                references["runner_config"]["content_hash"] = runner.canonical_hash(config)
                development = work / "development.txt"
                val47 = work / "val47.txt"
                development.write_text("s\n", encoding="utf-8")
                val47.write_text("s\n", encoding="utf-8")
                parent = {"content_hash": "parent"}
                identity = {
                    "schema_version": runner.SCHEMA_VERSION, "dataset": "RGBT234",
                    "dataset_root": str(root), "phase_family": "online_then_evaluate",
                    "scope": "design83", "yaml": "fixture",
                    **references,
                    "authoritative_splits": {
                        "development": {"path": str(development),
                                        "sha256": runner.file_sha256(development)},
                        "val47": {"path": str(val47),
                                  "sha256": runner.file_sha256(val47)}},
                    "sequences": ["s"], "sequences_hash": runner.canonical_hash(["s"]),
                    "image_manifest_hash": sealed, "support_iou": 0.7,
                    "policy_lock_content_hash": None, "parent_provenance": parent,
                    "parent_provenance_hash": "parent",
                    "source_sha256": {name: runner.file_sha256(path)
                                      for name, path in runner.IDENTITY_SOURCES.items()},
                }
                metadata = {
                    "schema_version": runner.SCHEMA_VERSION,
                    "kind": "stage1_quarantine_v3a_run_metadata", "identity": identity,
                    "identity_hash": runner.canonical_hash(identity),
                    "parent_provenance_hash": "parent",
                    "policy_lock_content_hash": None, "frame_indexing": "zero_based",
                    "causality": "fixture", "label_policy": "fixture",
                    "completed_phases": ["online", "evaluate"],
                    "created_unix": 1.0, "updated_unix": 2.0,
                }
                metadata["content_hash"] = runner.canonical_hash(metadata)
                metadata_path = work / "metadata.json"
                runner.write_json(metadata_path, metadata)
                runner.load_config = lambda path: config
                runner.validate_metadata(metadata_path, "design83", ["s"], 0.7,
                                         config, "parent")
                visible_image = sequence / "visible" / "0001.jpg"
                original_image = visible_image.read_bytes()
                visible_image.write_bytes(b"x" * len(original_image))
                self.assertNotEqual(sealed, runner._dataset_manifest_hash(root, ["s"]))
                with self.assertRaisesRegex(RuntimeError, "manifest"):
                    runner.validate_metadata(metadata_path, "design83", ["s"], 0.7,
                                             config, "parent")
                visible_image.write_bytes(original_image)
                (sequence / "visible.txt").write_text(
                    "2,2,10,10\n", encoding="utf-8")
                self.assertNotEqual(sealed, runner._dataset_manifest_hash(root, ["s"]))
                with self.assertRaisesRegex(RuntimeError, "manifest"):
                    runner.validate_metadata(metadata_path, "design83", ["s"], 0.7,
                                             config, "parent")
                with self.assertRaisesRegex(RuntimeError, "absolute"):
                    runner._dataset_manifest_hash(Path("relative"), ["s"])
        finally:
            runner.load_config = original_load_config

    def test_v2_lock_is_rejected_unconditionally(self):
        try:
            config = runner.load_config(CONFIG_PATH)
        except RuntimeError as error:
            if "PyYAML" in str(error):
                self.skipTest(str(error))
            raise
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            value = {"schema_version": "rmg-stage1-v2-q5",
                     "kind": "provisional_stage1_quarantine_v2_policy"}
            value["content_hash"] = runner.canonical_hash(value)
            runner.write_json(path, value)
            with self.assertRaisesRegex(RuntimeError, "wrong kind/schema"):
                runner.validate_policy_lock(
                    path, expected_parent_hash="p", expected_scope_sequences=["s"],
                    expected_config_hash=runner.canonical_hash(config),
                    config=config)

    def test_failure_select_writes_all_audits_and_no_lock(self):
        try:
            config = runner.load_config(CONFIG_PATH)
        except RuntimeError as error:
            if "PyYAML" in str(error):
                self.skipTest(str(error))
            raise
        original_validate = runner.validate_evaluated_root
        original_reference = runner._artifact_reference
        original_file_hash = runner.file_sha256
        try:
            sequences = ["s{:02d}".format(index) for index in range(20)]
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                design_roots = []
                root_map = {}
                for threshold in metrics.SUPPORT_THRESHOLDS:
                    item_root = root / str(threshold)
                    item_root.mkdir()
                    runner.write_json(
                        item_root / "metadata.json",
                        {"identity": {"support_iou": threshold}})
                    for name in ("aggregate.json", "gate.json"):
                        (item_root / name).write_text("{}", encoding="utf-8")
                    design_roots.append(str(item_root))
                    root_map[str(item_root.resolve())] = threshold

                def fake_validate(path, *args, **kwargs):
                    threshold = root_map[str(Path(path).resolve())]
                    failed = candidate(threshold, passes=False)
                    summaries = []
                    for sequence in sequences:
                        arms = {name: {"num_frames": 100, "success_auc": 0.5,
                                       "mean_iou": 0.5, "precision20": 0.5,
                                       "normalized_precision": 0.5,
                                       "normalized_precision_at_0_2": 0.5}
                                for name in metrics.ARM_NAMES}
                        summaries.append({"sequence": sequence, "arms": arms,
                                          "governance": failed["governance"]})
                    return {"metadata": {
                                "identity": {"support_iou": threshold},
                                "identity_hash": "id-{}".format(threshold)},
                            "aggregate": failed, "gate": {"pass": False},
                            "summaries": summaries,
                            "aggregate_path": Path(path) / "aggregate.json",
                            "gate_path": Path(path) / "gate.json"}

                runner.validate_evaluated_root = fake_validate
                runner._artifact_reference = lambda path: (
                    {"path": str(Path(path).resolve()), "sha256": "x",
                     "content_hash": runner.canonical_hash(str(path))}, {})
                runner.file_sha256 = lambda path: "source-hash"
                args = SimpleNamespace(design_root=design_roots,
                                       output_dir=str(root / "selection"))
                result = runner.run_select_phase(
                    args, config, sequences,
                    {"content_hash": "parent"})
                self.assertIsNone(result)
                for name in ("selection_audit_v3a.json",
                             "bootstrap_stability_v3a.json",
                             "logo_stability_v3a.json",
                             "protocol_manifest_v3a.json",
                             "feature_schema_hash_v3a.txt"):
                    self.assertTrue((root / "selection" / name).is_file(), name)
                self.assertFalse((root / "selection" /
                                  "provisional_policy_lock_v3a.json").exists())
        finally:
            runner.validate_evaluated_root = original_validate
            runner._artifact_reference = original_reference
            runner.file_sha256 = original_file_hash


class ConfigTest(unittest.TestCase):
    def test_frozen_config(self):
        try:
            config = runner.load_config(CONFIG_PATH)
        except RuntimeError as error:
            if "PyYAML" in str(error):
                self.skipTest(str(error))
            raise
        runner.validate_config(config)
        self.assertEqual(config["schema_version"], "rmg-stage1-v3a-qonly")
        self.assertEqual(config["arms"], list(metrics.ARM_NAMES))
        self.assertEqual(config["quarantine"]["support_iou_candidates"],
                         list(metrics.SUPPORT_THRESHOLDS))
        self.assertEqual(config["split"]["inner"]["namespace"],
                         "rmg-stage1-v2-q5-inner")
        self.assertNotIn("immediate_max_entropy", config["triage"])
        drifted = dict(config)
        drifted["metrics"] = dict(config["metrics"], bootstrap_samples=1)
        with self.assertRaisesRegex(ValueError, "complete frozen default"):
            runner.validate_config(drifted)


if __name__ == "__main__":
    unittest.main()
