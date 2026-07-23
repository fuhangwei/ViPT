from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from analysis.cbe.protocol_v1 import (
    ProtocolValidationError,
    canonical_json_hash,
    validate_official_gate_input,
    with_content_hash,
)
import analysis.cbe.run_design83_gpu_dryrun_nonofficial_v1 as dryrun


class Design83GpuDryRunNonOfficialTest(unittest.TestCase):
    def _formal_identity(self):
        entries = [
            {"sequence": f"seq{index:02d}", "entry_hash": f"{index:064x}"}
            for index in range(83)
        ]
        return {
            "dataset_root": "/dataset",
            "dataset": "RGBT234",
            "sequences": [entry["sequence"] for entry in entries],
            "dataset_entries": entries,
        }

    def _sequence_transcript(self, name="seq00", ordinal=0):
        trajectory = [
            {
                "schema_version": dryrun.DRYRUN_SCHEMA_VERSION,
                "scope": dryrun.DRYRUN_SCOPE,
                "record_type": "trajectory",
                "sequence_name": name,
                "frame_index": 0,
                "pred_xywh": [1.0, 2.0, 3.0, 4.0],
                "best_score": None,
                "template_id": "factual:0",
            },
            {
                "schema_version": dryrun.DRYRUN_SCHEMA_VERSION,
                "scope": dryrun.DRYRUN_SCOPE,
                "record_type": "trajectory",
                "sequence_name": name,
                "frame_index": 1,
                "pred_xywh": [1.0, 2.0, 3.0, 4.0],
                "best_score": 0.5,
                "search_anchor_xywh": [1.0, 2.0, 3.0, 4.0],
                "template_id": "factual:0",
            },
        ]
        return {
            "schema_version": dryrun.DRYRUN_SCHEMA_VERSION,
            "artifact_type": "design83_gpu_dryrun_sequence_transcript",
            "scope": dryrun.DRYRUN_SCOPE,
            "formal_result": False,
            "official_phase": False,
            "sequence_name": name,
            "design83_ordinal": ordinal,
            "dataset_entry_hash": f"{ordinal:064x}",
            "frame_count": 2,
            "feature_size": 16,
            "initialized_fingerprint": {},
            "snapshot_hashes": {},
            "schedule": [],
            "schedule_hash": canonical_json_hash([]),
            "trajectory": trajectory,
            "opportunities": [],
            "final_fingerprint": {},
        }

    def _worker(self, identity):
        transcripts = [
            self._sequence_transcript(item["name"], item["ordinal"])
            for item in identity["selected_sequences"]
        ]
        return with_content_hash({
            "schema_version": dryrun.DRYRUN_SCHEMA_VERSION,
            "artifact_type": "design83_gpu_dryrun_worker_transcript",
            "scope": dryrun.DRYRUN_SCOPE,
            "formal_result": False,
            "official_phase": False,
            "dryrun_identity_hash": identity["content_hash"],
            "sequence_names": [item["name"] for item in identity["selected_sequences"]],
            "transcript_hash": canonical_json_hash(transcripts),
            "transcripts": transcripts,
        })

    def test_selection_is_exact_locked_prefix(self):
        identity = self._formal_identity()
        manifests = {
            name: SimpleNamespace(sequence=name)
            for name in identity["sequences"][:2]
        }

        def load(_root, name, _dataset):
            return manifests[name]

        def serialize(manifest):
            return next(
                entry for entry in identity["dataset_entries"]
                if entry["sequence"] == manifest.sequence
            )

        with mock.patch.object(dryrun, "_load_sequence_manifest", side_effect=load), \
                mock.patch.object(dryrun, "_sequence_manifest_dict", side_effect=serialize):
            selected = dryrun._select_entries(identity, Path("/dataset"), 2)
        self.assertEqual([item["ordinal"] for item in selected], [0, 1])
        self.assertEqual(
            [item["entry"]["sequence"] for item in selected],
            identity["sequences"][:2],
        )

    def test_selection_rejects_invalid_count_and_reordered_entries(self):
        identity = self._formal_identity()
        for count in (0, 3, True):
            with self.subTest(count=count):
                with self.assertRaisesRegex(dryrun.DryRunValidationError, "exactly 1 or 2"):
                    dryrun._select_entries(identity, Path("/dataset"), count)
        identity["dataset_entries"][0], identity["dataset_entries"][1] = (
            identity["dataset_entries"][1], identity["dataset_entries"][0]
        )
        with self.assertRaisesRegex(dryrun.DryRunValidationError, "locked design83 prefix"):
            dryrun._select_entries(identity, Path("/dataset"), 1)

    def test_selection_rejects_dataset_root_drift(self):
        with self.assertRaisesRegex(dryrun.DryRunValidationError, "dataset root differs"):
            dryrun._select_entries(self._formal_identity(), Path("/replacement"), 1)

    def test_preflight_binding_requires_complete_design83_identity(self):
        identity = with_content_hash({
            "schema_version": "cbe-stage0-diagnostic-v1",
            "artifact_type": "run_identity",
            "scope": "design83",
            "sequence_count": 83,
            "sequences": [f"seq{index:02d}" for index in range(83)],
            "dataset_entries": [{"sequence": f"seq{index:02d}"} for index in range(83)],
        })
        preflight = with_content_hash({
            "schema_version": "cbe-stage0-diagnostic-v1",
            "scope": "design83",
            "phase": "preflight",
            "parent_phase": None,
            "parent_content_hash": None,
            "identity_hash": identity["content_hash"],
            "payload": {"status": "COMPLETE", "sequence_count": 83, "dataset_entry_count": 83},
        })
        with mock.patch.object(dryrun, "_read_bound_json", side_effect=(identity, preflight)):
            observed_identity, observed_preflight = dryrun._validate_preflight_binding(
                Path("/formal"), "1" * 64, "2" * 64
            )
        self.assertIs(observed_identity, identity)
        self.assertIs(observed_preflight, preflight)
        preflight["payload"]["sequence_count"] = 2
        with mock.patch.object(dryrun, "_read_bound_json", side_effect=(identity, preflight)):
            with self.assertRaisesRegex(dryrun.DryRunValidationError, "completed design83 preflight"):
                dryrun._validate_preflight_binding(Path("/formal"), "1" * 64, "2" * 64)

    def test_factual_trajectory_uses_tracker_equivalence_tolerances(self):
        trajectory = {"pred_xywh": [1.0, 2.0, 3.0, 4.0], "best_score": 0.5}
        self.assertTrue(dryrun._factual_matches_trajectory(
            {"target_bbox": [1.0 + 9e-6, 2.0, 3.0, 4.0], "best_score": 0.5 + 9e-7},
            trajectory,
        ))
        self.assertFalse(dryrun._factual_matches_trajectory(
            {"target_bbox": [1.0 + 2e-5, 2.0, 3.0, 4.0], "best_score": 0.5},
            trajectory,
        ))
        self.assertFalse(dryrun._factual_matches_trajectory(
            {"target_bbox": [1.0, 2.0, 3.0, 4.0], "best_score": 0.5 + 2e-6},
            trajectory,
        ))

    def test_sequence_transcript_remains_nonofficial_and_label_free(self):
        transcript = self._sequence_transcript()
        dryrun.validate_sequence_transcript(transcript)
        self.assertFalse(transcript["formal_result"])
        self.assertFalse(transcript["official_phase"])
        transcript["trajectory"][1]["ground_truth"] = [1, 2, 3, 4]
        with self.assertRaises(Exception):
            dryrun.validate_sequence_transcript(transcript)

    def test_worker_rejects_content_hash_tamper(self):
        identity = with_content_hash({
            "selected_sequences": [{"ordinal": 0, "name": "seq00", "dataset_entry_hash": "0" * 64, "schedule": [], "schedule_hash": canonical_json_hash([])}],
        })
        worker = self._worker(identity)
        worker["content_hash"] = "0" * 64
        with self.assertRaisesRegex(Exception, "content_hash mismatch"):
            dryrun.validate_worker(worker, identity)

    def test_worker_binds_dataset_entry_and_schedule(self):
        identity = with_content_hash({
            "selected_sequences": [{"ordinal": 0, "name": "seq00", "dataset_entry_hash": "0" * 64, "schedule": [], "schedule_hash": canonical_json_hash([])}],
        })
        worker = self._worker(identity)
        worker["transcripts"][0]["dataset_entry_hash"] = "f" * 64
        worker["transcript_hash"] = canonical_json_hash(worker["transcripts"])
        worker = with_content_hash({
            key: value for key, value in worker.items() if key != "content_hash"
        })
        with self.assertRaisesRegex(dryrun.DryRunValidationError, "selection binding"):
            dryrun.validate_worker(worker, identity)
        transcript = self._sequence_transcript()
        transcript["schedule"] = [{"frame_index": 1}]
        transcript["schedule_hash"] = canonical_json_hash(transcript["schedule"])
        with self.assertRaisesRegex(dryrun.DryRunValidationError, "opportunity count"):
            dryrun.validate_sequence_transcript(transcript)

    def test_parent_accepts_identical_workers_and_writes_nonofficial_report(self):
        identity = with_content_hash({
            "selected_sequences": [{"ordinal": 0, "name": "seq00", "dataset_entry_hash": "0" * 64, "schedule": [], "schedule_hash": canonical_json_hash([])}],
        })
        worker = self._worker(identity)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight = root / "formal-preflight"
            preflight.mkdir()
            output = root / "dryrun"
            args = SimpleNamespace(
                output_dir=str(output), preflight_root=str(preflight)
            )

            def run(command, **_kwargs):
                worker_output = Path(command[-1])
                dryrun.atomic_write_json(worker_output, worker)
                return SimpleNamespace(returncode=0, stderr="")

            with mock.patch.object(dryrun, "build_dryrun_identity", return_value=(identity, None, None, None, None)), \
                    mock.patch.object(dryrun, "_worker_command", side_effect=lambda _args, path: ["worker", str(path)]), \
                    mock.patch.object(dryrun.subprocess, "run", side_effect=run):
                report = dryrun.run_parent(args)
            self.assertEqual(report["status"], "PASS")
            self.assertFalse(report["formal_result"])
            self.assertFalse(report["official_phase"])
            self.assertTrue(report["deterministic_repeat_equal"])
            self.assertTrue((output / "dryrun_identity.json").is_file())
            self.assertTrue((output / "sequences" / "seq00" / "transcript.json").is_file())
            self.assertTrue((output / "report.json").is_file())
            with self.assertRaisesRegex(
                ProtocolValidationError, "non-official schema"
            ):
                validate_official_gate_input(report)

    def test_parent_rejects_fresh_process_repeat_mismatch(self):
        identity = with_content_hash({
            "selected_sequences": [{"ordinal": 0, "name": "seq00", "dataset_entry_hash": "0" * 64, "schedule": [], "schedule_hash": canonical_json_hash([])}],
        })
        workers = [self._worker(identity), self._worker(identity)]
        workers[1]["transcripts"][0]["trajectory"][1]["best_score"] = 0.6
        workers[1] = with_content_hash({
            key: value for key, value in workers[1].items() if key != "content_hash"
        })
        workers[1]["transcript_hash"] = canonical_json_hash(workers[1]["transcripts"])
        workers[1] = with_content_hash({
            key: value for key, value in workers[1].items() if key != "content_hash"
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight = root / "formal-preflight"
            preflight.mkdir()
            args = SimpleNamespace(
                output_dir=str(root / "dryrun"), preflight_root=str(preflight)
            )

            def run(command, **_kwargs):
                value = workers.pop(0)
                dryrun.atomic_write_json(Path(command[-1]), value)
                return SimpleNamespace(returncode=0, stderr="")

            with mock.patch.object(dryrun, "build_dryrun_identity", return_value=(identity, None, None, None, None)), \
                    mock.patch.object(dryrun, "_worker_command", side_effect=lambda _args, path: ["worker", str(path)]), \
                    mock.patch.object(dryrun.subprocess, "run", side_effect=run):
                with self.assertRaisesRegex(dryrun.DryRunValidationError, "transcript hashes differ"):
                    dryrun.run_parent(args)

    def test_output_must_be_new_and_outside_repository(self):
        with self.assertRaisesRegex(dryrun.DryRunValidationError, "outside"):
            dryrun._new_output_directory(dryrun.ROOT / "dryrun")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "formal"
            formal.mkdir()
            with self.assertRaisesRegex(dryrun.DryRunValidationError, "formal preflight root"):
                dryrun._new_output_directory(formal / "dryrun", formal)
            with self.assertRaisesRegex(dryrun.DryRunValidationError, "must not already exist"):
                dryrun._new_output_directory(directory)

    def test_worker_output_cannot_overwrite_formal_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            formal = Path(directory) / "formal"
            formal.mkdir()
            artifact = formal / "preflight.json"
            artifact.write_text("official", encoding="utf-8")
            with self.assertRaisesRegex(dryrun.DryRunValidationError, "new file"):
                dryrun._new_worker_output(artifact, formal)


if __name__ == "__main__":
    unittest.main()
