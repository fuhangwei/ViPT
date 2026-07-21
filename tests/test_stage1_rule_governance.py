import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = ROOT / "analysis" / "memory_oracle" / "rule_controller.py"
METRICS_PATH = ROOT / "analysis" / "memory_oracle" / "compute_stage1_metrics.py"
RUNNER_PATH = ROOT / "analysis" / "memory_oracle" / "run_stage1_rule_governance.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


try:
    controller = load_module("stage1_rule_controller", CONTROLLER_PATH)
except TypeError as error:
    if sys.version_info >= (3, 10) or "unsupported operand type" not in str(error):
        raise
    # Exercise controller behavior on the system Python 3.9 while preserving the
    # production file's Python 3.10 union annotation unchanged.
    controller = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("stage1_rule_controller", loader=None))
    controller.__file__ = str(CONTROLLER_PATH)
    sys.modules["stage1_rule_controller"] = controller
    source = "from __future__ import annotations\n" + CONTROLLER_PATH.read_text(encoding="utf-8")
    exec(compile(source, str(CONTROLLER_PATH), "exec"), controller.__dict__)
metrics = load_module("stage1_metrics", METRICS_PATH)


requires_controller = unittest.skipIf(controller is None, "controller unavailable")


def arm_metrics(success_auc, num_frames=100):
    return {
        "num_frames": num_frames,
        "success_auc": success_auc,
        "mean_iou": success_auc,
        "precision20": success_auc,
        "normalized_precision": success_auc,
        "normalized_precision_at_0_2": success_auc,
    }


def governance_counts(opportunities=10, commits=2, good_opportunities=2,
                      good_commits=2, bad_commits=0):
    return {
        "num_opportunities": opportunities,
        "num_commits": commits,
        "num_good_opportunities": good_opportunities,
        "num_good_commits": good_commits,
        "num_bad_commits": bad_commits,
    }


class Stage1RuleGovernanceTest(unittest.TestCase):
    def test_response_entropy_margin_and_dispersion_are_numeric(self):
        try:
            import torch
            from lib.test.tracker.vipt_stage1 import ViPTStage1Track
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(str(error))

        response = torch.tensor([[[[4.0, 1.0], [0.0, 0.0]]]])
        boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0],
                               [3.0, 4.0, 1.0, 1.0],
                               [0.0, 0.0, 1.0, 1.0],
                               [0.0, 0.0, 1.0, 1.0]]])
        result = ViPTStage1Track.response_statistics(response, boxes)
        expected_entropy = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2)) / math.log(4.0)
        self.assertAlmostEqual(result["response_peak"], 4.0)
        self.assertAlmostEqual(result["response_entropy"], expected_entropy, places=6)
        self.assertAlmostEqual(result["response_margin"], 3.0)
        self.assertTrue(math.isfinite(result["response_topk_score_std"]))
        self.assertTrue(math.isfinite(result["response_topk_box_dispersion"]))

    def test_response_entropy_is_one_for_zero_map(self):
        try:
            import torch
            from lib.test.tracker.vipt_stage1 import ViPTStage1Track
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(str(error))
        response = torch.zeros(1, 1, 2, 2)
        boxes = torch.zeros(1, 4, 4)
        result = ViPTStage1Track.response_statistics(response, boxes)
        self.assertAlmostEqual(result["response_entropy"], 1.0)
        self.assertAlmostEqual(result["response_margin"], 0.0)

    @requires_controller
    def test_motion_residual_uses_constant_velocity_and_box_diagonal(self):
        previous = [[0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 10.0, 10.0]]
        current = [22.0, 0.0, 10.0, 10.0]
        self.assertAlmostEqual(
            controller.kinematic_residual(previous, current),
            2.0 / math.sqrt(200.0),
        )
        scaled = [20.0, 0.0, 20.0, 10.0]
        self.assertAlmostEqual(
            controller.kinematic_residual(previous, scaled), math.log(2.0))

    @requires_controller
    def test_rule_skips_when_motion_history_is_insufficient(self):
        state = controller.ControllerState()
        observation = {
            "frame_idx": 30,
            "pred_xywh": [0.0, 0.0, 10.0, 10.0],
            "image_shape": [100, 120],
            "response_entropy": 0.1,
        }
        decision = controller.decide(
            observation, state,
            controller.RuleThresholds(max_entropy=0.5, max_motion_residual=0.5),
            candidate_valid=True,
        )
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "insufficient_motion_history")
        self.assertIsNone(decision.motion_residual)

    @requires_controller
    def test_controller_rejects_gt_and_iou_fields(self):
        base = {
            "frame_idx": 30,
            "pred_xywh": [0.0, 0.0, 10.0, 10.0],
            "image_shape": [100, 120],
            "response_entropy": 0.1,
        }
        for forbidden in ("gt_xywh", "iou", "evaluation_iou", "candidate_iou"):
            with self.subTest(field=forbidden):
                observation = dict(base, **{forbidden: 1.0})
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    controller.validate_observation(observation)

    @requires_controller
    def test_threshold_selection_is_serial_and_chooses_widest_passing(self):
        rows = [
            {"sequence": "s1", "candidate_valid": True, "response_entropy": 0.10,
             "motion_residual": 0.10, "evaluation_iou": 0.9},
            {"sequence": "s1", "candidate_valid": True, "response_entropy": 0.20,
             "motion_residual": 0.20, "evaluation_iou": 0.8},
            {"sequence": "s2", "candidate_valid": True, "response_entropy": 0.30,
             "motion_residual": 0.30, "evaluation_iou": 0.8},
            {"sequence": "s2", "candidate_valid": True, "response_entropy": 0.30,
             "motion_residual": 0.30, "evaluation_iou": 0.0},
        ]
        entropy, entropy_audit = controller.select_widest_threshold(
            rows, "entropy", [0.1, 0.2, 0.3], min_precision=0.7,
            max_bad_rate=0.3,
        )
        self.assertEqual(entropy, 0.3)
        self.assertTrue(entropy_audit[-1]["passes"])
        motion, motion_audit = controller.select_widest_threshold(
            rows, "motion", [0.1, 0.2, 0.3], fixed_entropy=entropy,
            min_precision=0.7, max_bad_rate=0.05,
        )
        self.assertEqual(motion, 0.2)
        self.assertFalse(motion_audit[-1]["passes"])

    def test_update_precision_recall_bad_rate_and_coverage(self):
        rows = [
            {"update_opportunity": True, "candidate_valid": True,
             "action": "update", "evaluation_iou": 0.9},
            {"update_opportunity": True, "candidate_valid": True,
             "action": "update", "evaluation_iou": 0.0},
            {"update_opportunity": True, "candidate_valid": True,
             "action": "skip", "evaluation_iou": 0.8},
            {"update_opportunity": True, "candidate_valid": True,
             "action": "skip", "evaluation_iou": 0.4},
            {"update_opportunity": False, "action": "update", "evaluation_iou": 1.0},
        ]
        result = metrics.update_quality(rows)
        self.assertEqual(result["num_opportunities"], 4)
        self.assertEqual(result["num_commits"], 2)
        self.assertAlmostEqual(result["update_precision"], 0.5)
        self.assertAlmostEqual(result["update_recall"], 0.5)
        self.assertAlmostEqual(result["bad_update_rate"], 0.5)
        self.assertAlmostEqual(result["commit_coverage"], 0.5)

    def test_five_arm_delta_direction_and_locked_gate_pass_fail(self):
        summaries = []
        for index in range(10):
            summaries.append({
                "sequence": f"seq{index:02d}",
                "clean_subset": index < 5,
                "arms": {
                    "static": arm_metrics(0.50),
                    "periodic_pred": arm_metrics(0.49),
                    "pred_good": arm_metrics(0.55),
                    "gt_good": arm_metrics(0.65),
                    "rule_rmg": arm_metrics(0.51),
                },
                "governance": governance_counts(),
            })
        aggregate = metrics.aggregate_sequence_summaries(
            summaries, bootstrap_seed=7, bootstrap_samples=200)
        self.assertEqual(set(aggregate["arms"]), set(metrics.ARM_NAMES))
        static_delta = aggregate["frame_weighted_paired_deltas"][
            "rule_rmg_vs_static"]["success_auc"]
        periodic_delta = aggregate["frame_weighted_paired_deltas"][
            "rule_rmg_vs_periodic_pred"]["success_auc"]
        self.assertAlmostEqual(static_delta["mean"], 0.01)
        self.assertAlmostEqual(periodic_delta["mean"], 0.02)
        self.assertGreater(periodic_delta["low"], 0.0)
        self.assertTrue(metrics.evaluate_locked_gate(aggregate)["pass"])

        aggregate["rule_governance"]["bad_update_rate"] = 0.051
        failed = metrics.evaluate_locked_gate(aggregate)
        self.assertFalse(failed["pass"])
        self.assertFalse(failed["checks"]["bad_update_rate"]["pass"])

    def test_worsened_and_clean_preservation_are_sequence_fractions(self):
        summaries = [
            {"sequence": "clean-good", "clean": True,
             "arms": {"static": arm_metrics(0.5, 10), "periodic_pred": arm_metrics(0.4, 10),
                      "rule_rmg": arm_metrics(0.6, 10)}, "governance": governance_counts()},
            {"sequence": "clean-bad", "clean": True,
             "arms": {"static": arm_metrics(0.5, 90), "periodic_pred": arm_metrics(0.4, 90),
                      "rule_rmg": arm_metrics(0.4, 90)}, "governance": governance_counts()},
            {"sequence": "other", "clean": False,
             "arms": {"static": arm_metrics(0.5, 10), "periodic_pred": arm_metrics(0.4, 10),
                      "rule_rmg": arm_metrics(0.6, 10)}, "governance": governance_counts()},
        ]
        result = metrics.aggregate_sequence_summaries(summaries, bootstrap_samples=100)
        self.assertAlmostEqual(result["worsened_vs_static"]["fraction"], 1.0 / 3.0)
        clean = result["clean_subset_preservation"]
        self.assertEqual(clean["num_sequences"], 2)
        self.assertAlmostEqual(clean["worsened_vs_static"]["fraction"], 0.5)
        self.assertAlmostEqual(clean["rule_vs_static_success_auc"]["mean"], -0.08)

    def test_runner_split_is_deterministic_and_metadata_identity_is_locked_when_available(self):
        if not RUNNER_PATH.is_file():
            self.skipTest("Stage 1 runner is not available")
        try:
            sys.modules["analysis.memory_oracle.rule_controller"] = controller
            runner = load_module("stage1_runner_for_tests", RUNNER_PATH)
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(str(error))

        split_function = next((getattr(runner, name) for name in (
            "split_sequences_by_sha256", "sha256_tune_confirm_split",
            "deterministic_tuning_eval_split", "deterministic_split", "split_sequences")
                               if hasattr(runner, name)), None)
        if split_function is not None:
            sequences = [f"seq{index:02d}" for index in range(10)]
            try:
                first = split_function(sequences, tune_fraction=0.5)
                second = split_function(list(reversed(sequences)), tune_fraction=0.5)
            except TypeError:
                first = split_function(sequences, 0.5)
                second = split_function(list(reversed(sequences)), 0.5)
            self.assertEqual(first, second)
            self.assertFalse(set(first[0]) & set(first[1]))
            self.assertEqual(set(first[0]) | set(first[1]), set(sequences))
            development = [f"dev{index:03d}" for index in range(187)]
            tune, confirm = runner.split_sequences_by_sha256(
                development, tune_fraction=0.6684491978609626)
            self.assertEqual(len(tune), 125)
            self.assertEqual(len(confirm), 62)
            self.assertEqual(
                runner.evaluation_sequences(
                    development,
                    {"tuning": {"tune_fraction": 0.6684491978609626}},
                    external_lock=False),
                confirm,
            )
            self.assertEqual(
                runner.evaluation_sequences(
                    development, {"tuning": {}}, external_lock=True),
                development,
            )

        validator = getattr(runner, "validate_or_write_metadata", None)
        if validator is None:
            self.skipTest("Stage 1 runner exposes no metadata identity validator")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            metadata = {
                "dataset": "RGBT234", "dataset_root": "/a", "split_file": "/s",
                "split_sha256": "1", "yaml_name": "deep_rgbt", "checkpoint": "/c",
                "checkpoint_sha256": "2", "experiment_yaml_sha256": "3",
                "source_sha256": {"tracker": "4"}, "config_hash": "5",
                "config_sha256": "6", "sequences": ["seq"],
            }
            validator(path, metadata)
            changed = dict(metadata, checkpoint_sha256="different")
            with self.assertRaises(RuntimeError):
                validator(path, changed)


if __name__ == "__main__":
    unittest.main()
