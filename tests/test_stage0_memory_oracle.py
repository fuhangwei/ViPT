import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "analysis" / "memory_oracle" / "compute_stage0_metrics.py"
RUNNER_PATH = ROOT / "analysis" / "memory_oracle" / "run_stage0_memory_oracle.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


metrics = load_module("stage0_metrics", METRICS_PATH)
runner = load_module("stage0_runner", RUNNER_PATH)


def make_row(frame_idx, pred=None, gt=None):
    gt = gt or [float(frame_idx), 0.0, 10.0, 10.0]
    pred = pred or list(gt)
    return {"sequence": "seq", "frame_idx": frame_idx, "pred_xywh": pred,
            "gt_xywh": gt, "best_score": 1.0}


class Stage0MemoryOracleTest(unittest.TestCase):
    def test_tracking_metrics_perfect_and_21_threshold_auc(self):
        gt = [[0, 0, 10, 10], [5, 5, 20, 10]]
        result = metrics.tracking_metrics(gt, gt)
        self.assertAlmostEqual(result["success_auc"], 1.0)
        self.assertAlmostEqual(result["mean_iou"], 1.0)
        self.assertAlmostEqual(result["precision20"], 1.0)
        self.assertAlmostEqual(result["normalized_precision"], 1.0)
        self.assertEqual(len(metrics.SUCCESS_THRESHOLDS), 21)
        self.assertAlmostEqual(metrics.success_auc([0.5]), 11.0 / 21.0)

    def test_manifest_is_deterministic_and_only_uses_static_trace(self):
        rows = [make_row(frame) for frame in range(151)]
        manifest_a = runner.build_frozen_manifest("seq", rows)
        manifest_b = runner.build_frozen_manifest("seq", list(reversed(rows)))
        self.assertEqual(manifest_a, manifest_b)
        self.assertEqual(manifest_a["manifest_hash"], manifest_b["manifest_hash"])
        self.assertEqual(manifest_a["update_frames"], [30, 60, 90])
        self.assertEqual(manifest_a["replay_event_frames"], [30, 90])
        self.assertEqual([event["candidate_frame"] for event in manifest_a["events"]],
                         [30, 90])
        self.assertTrue(all(event["candidate_source"] == "static_trace_prediction"
                            for event in manifest_a["events"]))

    def test_horizon_excludes_action_frame_and_has_no_off_by_one(self):
        self.assertEqual(runner.horizon_frame_indices(30, 5, 100), [31, 32, 33, 34, 35])
        self.assertEqual(runner.horizon_frame_indices(97, 5, 100), [98, 99])
        self.assertEqual(runner.horizon_frame_indices(99, 5, 100), [])

    def test_periodic_schedule_respects_warmup_and_cooldown(self):
        self.assertEqual(
            runner.periodic_candidate_frames(151, warmup=30, interval=30, cooldown=60),
            [30, 60, 90],
        )
        self.assertEqual(
            runner.periodic_candidate_frames(90, warmup=30, interval=30, cooldown=60), []
        )

    def test_paired_delta_sign_is_variant_minus_baseline(self):
        baseline = metrics.tracking_metrics([[100, 100, 10, 10]], [[0, 0, 10, 10]])
        variant = metrics.tracking_metrics([[0, 0, 10, 10]], [[0, 0, 10, 10]])
        deltas = metrics.paired_metric_deltas(baseline, variant)
        self.assertGreater(deltas["success_auc"], 0)
        self.assertGreater(deltas["mean_iou"], 0)
        self.assertGreater(deltas["precision20"], 0)

    def test_bootstrap_ci_is_reproducible(self):
        first = metrics.deterministic_bootstrap_ci([-1.0, 0.0, 2.0, 3.0],
                                                   seed=17, samples=200)
        second = metrics.deterministic_bootstrap_ci([-1.0, 0.0, 2.0, 3.0],
                                                    seed=17, samples=200)
        self.assertEqual(first, second)
        self.assertLessEqual(first["low"], first["mean"])
        self.assertLessEqual(first["mean"], first["high"])

    def test_replay_schedule_is_non_overlapping_while_arms_remain_dense(self):
        update_frames = runner.periodic_candidate_frames(151, 30, 30, 60)
        self.assertEqual(update_frames, [30, 60, 90])
        self.assertEqual(runner.replay_event_frames(update_frames, 60), [30, 90])

    def test_bad_box_is_deterministic_legal_and_non_overlapping(self):
        gt = [50.0, 60.0, 20.0, 10.0]
        image_shape = (100, 120, 6)
        first = runner.deterministic_bad_box(gt, "seq", 30, image_shape, bad_iou=0.1)
        second = runner.deterministic_bad_box(gt, "seq", 30, image_shape, bad_iou=0.1)
        self.assertEqual(first, second)
        self.assertIsNotNone(first)
        self.assertEqual(metrics.iou_xywh(first, gt), 0.0)
        self.assertTrue(runner.bbox_validity(first, image_shape)[0])

    def test_bad_box_returns_none_when_no_legal_low_iou_location_exists(self):
        gt = [0.0, 0.0, 20.0, 20.0]
        self.assertIsNone(runner.deterministic_bad_box(gt, "seq", 30, (20, 20, 6)))

    def test_candidate_validity_rejects_small_nonfinite_and_padded_boxes(self):
        shape = (100, 120, 6)
        self.assertEqual(runner.bbox_validity([0, 0, 7, 10], shape)[1], "bbox_too_small")
        self.assertEqual(runner.bbox_validity([float("nan"), 0, 10, 10], shape)[1],
                         "bbox_non_finite")
        self.assertEqual(runner.bbox_validity([-20, 0, 10, 10], shape)[1],
                         "bbox_low_image_intersection")

    def test_frame_weighted_delta_uses_sequence_frame_counts(self):
        summaries = [
            {"sequence": "short", "arms": {
                "static": dict(metrics.tracking_metrics([[0, 0, 10, 10]], [[0, 0, 10, 10]]),
                               success_auc=0.5, num_frames=10),
                "gt_good": dict(metrics.tracking_metrics([[0, 0, 10, 10]], [[0, 0, 10, 10]]),
                                success_auc=0.6, num_frames=10),
            }},
            {"sequence": "long", "arms": {
                "static": dict(metrics.tracking_metrics([[0, 0, 10, 10]], [[0, 0, 10, 10]]),
                               success_auc=0.5, num_frames=90),
                "gt_good": dict(metrics.tracking_metrics([[0, 0, 10, 10]], [[0, 0, 10, 10]]),
                                success_auc=0.4, num_frames=90),
            }},
        ]
        result = metrics.aggregate_sequence_summaries(summaries, bootstrap_samples=100)
        macro = result["paired_deltas_vs_static"]["gt_good"]["success_auc"]["mean"]
        weighted = result["frame_weighted_paired_deltas_vs_static"]["gt_good"]["success_auc"]["mean"]
        self.assertAlmostEqual(macro, 0.0)
        self.assertAlmostEqual(weighted, -0.08)

    def test_clustered_bootstrap_reports_event_and_sequence_coverage(self):
        result = metrics.deterministic_clustered_bootstrap_ci(
            {"s1": [-0.1, -0.2], "s2": [-0.3]}, seed=4, samples=100)
        self.assertEqual(result["num_events"], 3)
        self.assertEqual(result["num_sequences"], 2)
        self.assertLess(result["high"], 0.0)

    def test_stage0_tracker_snapshot_api_has_no_prediction_side_effects(self):
        try:
            import torch
            from types import SimpleNamespace
            from lib.test.tracker.vipt_stage0 import TemplateSnapshot, ViPTStage0Track
        except ModuleNotFoundError as error:
            self.skipTest(str(error))

        class FakePreprocessor:
            def process(self, patch):
                return torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()

        class FakeBoxHead:
            def cal_bbox(self, response, size_map, offset_map, return_score):
                return torch.tensor([[0.5, 0.5, 0.25, 0.25]]), torch.tensor([[0.9]])

        class FakeNetwork:
            box_head = FakeBoxHead()

            def forward(self, template, search, ce_template_mask):
                return {
                    "score_map": torch.ones(1, 1, 1, 1),
                    "size_map": torch.ones(1, 1, 1, 1),
                    "offset_map": torch.zeros(1, 2, 1, 1),
                }

        tracker = ViPTStage0Track.__new__(ViPTStage0Track)
        tracker.params = SimpleNamespace(
            search_factor=2.0, search_size=16, template_factor=2.0, template_size=8)
        tracker.cfg = SimpleNamespace(
            MODEL=SimpleNamespace(BACKBONE=SimpleNamespace(CE_LOC=False)))
        tracker.preprocessor = FakePreprocessor()
        tracker.network = FakeNetwork()
        tracker.output_window = torch.ones(1, 1, 1, 1)
        tracker.state = [10.0, 20.0, 8.0, 8.0]
        tracker.frame_id = 7
        initial = TemplateSnapshot(
            "initial", "initial", 0, (10.0, 20.0, 8.0, 8.0), 1.0,
            np.zeros((4, 4, 6), dtype=np.uint8), torch.zeros(1), None)
        tracker.initial_template_snapshot = initial
        tracker.commit_template(initial)

        image = np.zeros((64, 64, 6), dtype=np.uint8)
        state_before = list(tracker.state)
        active_before = tracker.active_template_snapshot.template_id
        candidate = tracker.build_template_snapshot(
            image, [12.0, 22.0, 8.0, 8.0], "current_gt", 30)
        self.assertEqual(tracker.active_template_snapshot.template_id, active_before)
        output = tracker.predict_with_context(
            image, state_before, tracker.active_template_snapshot)
        self.assertEqual(tracker.state, state_before)
        self.assertEqual(tracker.frame_id, 7)
        self.assertEqual(tracker.active_template_snapshot.template_id, active_before)
        self.assertEqual(output["state_before"], state_before)

        tracker.commit_template(candidate)
        self.assertEqual(tracker.active_template_snapshot.template_id, candidate.template_id)
        tracker.rollback_to_initial()
        first_rollback = tracker.active_template_snapshot
        tracker.rollback_to_initial()
        self.assertEqual(tracker.active_template_snapshot.template_id, "initial")
        self.assertIsNot(tracker.active_template_snapshot, first_rollback)
        self.assertTrue(torch.equal(tracker.z_tensor, initial.z_tensor))

    def test_replay_starts_at_intervention_and_excludes_t_from_future_metrics(self):
        class FakeTracker:
            def __init__(self):
                self.initial_template_snapshot = {"kind": "initial"}
                self.active_template_snapshot = self.initial_template_snapshot

            def initialize(self, image, info):
                self.initial_template_snapshot = {"kind": "initial"}
                self.active_template_snapshot = self.initial_template_snapshot

            def rollback_to_initial(self):
                self.active_template_snapshot = self.initial_template_snapshot

            def build_template_snapshot(self, image, bbox, source, source_frame):
                return {"source": source, "frame": source_frame, "bbox": bbox}

            def commit_template(self, snapshot):
                self.active_template_snapshot = snapshot

            def predict_with_context(self, image, anchor, snapshot):
                return {"target_bbox": list(anchor), "best_score": 1.0, "state_before": anchor}

        baseline = [make_row(frame, pred=[float(frame), 0.0, 10.0, 10.0])
                    for frame in range(100)]
        event = {"candidate_frame": 30, "horizons": {"5": [31, 32, 33, 34, 35]},
                 "bad_candidate_xywh": [0.0, 20.0, 10.0, 10.0], "natural_bad": False}
        original_make_tracker = runner.make_tracker
        original_load_frame = runner.load_frame
        runner.make_tracker = lambda yaml_name: (FakeTracker(), object())
        runner.load_frame = lambda rgb, tir, params: np.zeros((100, 120, 6), dtype=np.uint8)
        try:
            rows = runner.run_event_replay(
                "seq", event, "open", "skip", None, list(range(100)), list(range(100)),
                np.asarray([row["gt_xywh"] for row in baseline]), baseline, "yaml", 0.1,
                {"candidate_validity": {}})
        finally:
            runner.make_tracker = original_make_tracker
            runner.load_frame = original_load_frame
        self.assertEqual(rows[0]["frame_idx"], 30)
        self.assertEqual(rows[0]["pred_xywh"], baseline[30]["pred_xywh"])
        self.assertFalse(rows[0]["included_in_future_metrics"])
        self.assertEqual([row["frame_idx"] for row in rows[1:]], [31, 32, 33, 34, 35])
        self.assertTrue(all(row["included_in_future_metrics"] for row in rows[1:]))

    def test_replay_pairs_keep_bad_sources_separate(self):
        base = {"success_auc": 0.5, "mean_iou": 0.5, "precision20": 0.5,
                "normalized_precision": 0.5, "normalized_precision_at_0_2": 0.5,
                "num_frames": 60}
        common = {"event_hash": "hash", "sequence": "seq", "anchor_mode": "closed",
                  "trace_hash": "trace", "intervention_valid": True, "invalid_reason": None}
        entries = [
            dict(common, event_id="natural", action="skip", source=None, horizons={"60": base}),
            dict(common, event_id="natural", action="update", source="bad_natural",
                 horizons={"60": dict(base, success_auc=0.4)}),
            dict(common, event_id="synthetic", action="skip", source=None, horizons={"60": base}),
            dict(common, event_id="synthetic", action="update", source="bad_deterministic",
                 horizons={"60": dict(base, success_auc=0.3)}),
        ]
        pairs = runner.pair_replay_entries(entries)
        aggregate = runner.aggregate_replay_pairs(
            [{"paired_deltas": pairs}], bootstrap_samples=100)
        self.assertIn("bad_natural.closed.H60", aggregate["groups"])
        self.assertIn("bad_deterministic.closed.H60", aggregate["groups"])

    def test_existing_metadata_rejects_identity_mismatch(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            metadata = {"dataset": "RGBT234", "dataset_root": "/a", "split_file": "/s",
                        "split_sha256": "1", "yaml_name": "deep_rgbt", "checkpoint": "/c",
                        "checkpoint_sha256": "2", "experiment_yaml_sha256": "3",
                        "source_sha256": {"tracker": "4"}, "config_hash": "5",
                        "config_sha256": "6", "sequences": ["seq"]}
            runner.validate_or_write_metadata(path, metadata)
            changed = dict(metadata, checkpoint_sha256="different")
            with self.assertRaises(RuntimeError):
                runner.validate_or_write_metadata(path, changed)

    def test_replay_pairs_match_same_event_loop_and_horizon(self):
        base = {"success_auc": 0.5, "mean_iou": 0.5, "precision20": 0.5,
                "normalized_precision": 0.5, "normalized_precision_at_0_2": 0.5,
                "num_frames": 60}
        improved = dict(base, success_auc=0.6)
        common = {"event_id": "seq:000030", "event_hash": "hash", "sequence": "seq",
                  "anchor_mode": "closed", "trace_hash": "trace", "intervention_valid": True,
                  "invalid_reason": None}
        entries = [
            dict(common, action="skip", source=None, horizons={"60": base}),
            dict(common, action="update", source="gt_good", horizons={"60": improved}),
        ]
        pairs = runner.pair_replay_entries(entries)
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["delta_update_minus_skip"]["success_auc"], 0.1)


if __name__ == "__main__":
    unittest.main()
