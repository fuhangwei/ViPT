import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from analysis.cbe.protocol_v1 import atomic_write_json, sha256_file
import analysis.cbe.run_vipt_real_smoke_nonofficial_v1 as smoke_runner
from analysis.cbe.run_vipt_real_smoke_nonofficial_v1 import (
    ROOT,
    SMOKE_SCHEMA_VERSION,
    SmokeValidationError,
    _assert_module_source,
    _authorized_path,
    load_authorization,
    map_summary,
    validate_authorization,
)


class NonOfficialSmokeTest(unittest.TestCase):
    def authorization(self):
        return {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "artifact_type": "vipt_real_smoke_authorization",
            "scope": "non_official_smoke",
            "authority": {
                "name": "synthetic-test-authority",
                "authorization_id": "smoke-auth-v1",
                "issued_at_utc": "2026-07-21T00:00:00Z",
            },
            "dataset": "RGBT234",
            "sequence": {
                "name": "smoke-sequence",
                "relative_root": "smoke-sequence",
                "role": "non_sealed_smoke_only",
            },
            "frames": [
                {
                    "ordinal": 0, "index": 0,
                    "visible_relative_path": "visible/0000.jpg",
                    "visible_sha256": "7" * 64,
                    "infrared_relative_path": "infrared/0000.jpg",
                    "infrared_sha256": "8" * 64,
                },
                {
                    "ordinal": 1, "index": 7,
                    "visible_relative_path": "visible/0007.jpg",
                    "visible_sha256": "9" * 64,
                    "infrared_relative_path": "infrared/0007.jpg",
                    "infrared_sha256": "a" * 64,
                },
            ],
            "init_bbox_xywh": [1.0, 2.0, 10.0, 12.0],
            "source_identity": {
                "git_commit": "a" * 40,
                "dirty_tree": False,
                "repository": "approved-repository",
            },
            "bindings": {
                "checkpoint_sha256": "1" * 64,
                "model_config_sha256": "2" * 64,
                "registry_sha256": "3" * 64,
                "resolved_config_hash": "4" * 64,
                "source_bundle_hash": "5" * 64,
                "environment_hash": "6" * 64,
            },
        }

    def test_authorization_accepts_exact_nonofficial_scope(self):
        value = validate_authorization(self.authorization())
        self.assertEqual(value["scope"], "non_official_smoke")
        self.assertEqual([row["index"] for row in value["frames"]], [0, 7])

    def test_authorization_rejects_official_and_sealed_roles(self):
        for role in ("design83", "internal42", "confirm62", "val47"):
            with self.subTest(role=role):
                value = self.authorization()
                value["sequence"]["role"] = role
                with self.assertRaisesRegex(SmokeValidationError, "non-sealed-only"):
                    validate_authorization(value)

    def test_authorization_rejects_frame_scope_and_unsafe_paths(self):
        mutations = (
            (
                lambda value: value["frames"].append({
                    **value["frames"][1], "ordinal": 2,
                }),
                "strictly increasing",
            ),
            (lambda value: value["frames"][0].update({
                "index": 1,
                "visible_relative_path": "visible/0001.jpg",
                "infrared_relative_path": "infrared/0001.jpg",
            }), "begin at 0"),
            (lambda value: value["frames"][1].update({"visible_relative_path": "../sealed.jpg"}), "unsafe"),
            (lambda value: value["frames"][1].update({"visible_relative_path": "visible/0008.jpg"}), "does not match"),
            (lambda value: value["frames"][1].update({"infrared_relative_path": "infrared/07.jpg"}), "stems must match"),
        )
        for mutate, expected in mutations:
            with self.subTest(expected=expected):
                value = self.authorization()
                mutate(value)
                with self.assertRaisesRegex(SmokeValidationError, expected):
                    validate_authorization(value)

    def test_authorization_bytes_are_externally_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authorization.json"
            atomic_write_json(path, self.authorization())
            expected = sha256_file(path)
            self.assertEqual(load_authorization(path, expected)["scope"], "non_official_smoke")
            value = json.loads(path.read_text(encoding="utf-8"))
            value["authority"]["name"] = "replacement"
            atomic_write_json(path, value)
            with self.assertRaisesRegex(SmokeValidationError, "byte drift"):
                load_authorization(path, expected)

    def test_authorization_requires_clean_commit_and_exact_bindings(self):
        for mutation, expected in (
            (lambda value: value["source_identity"].update({"dirty_tree": True}), "dirty_tree"),
            (lambda value: value["source_identity"].update({"git_commit": "not-a-commit"}), "git_commit"),
            (lambda value: value["bindings"].update({"environment_hash": "ABC"}), "lowercase SHA"),
        ):
            with self.subTest(expected=expected):
                value = self.authorization()
                mutation(value)
                with self.assertRaisesRegex(SmokeValidationError, expected):
                    validate_authorization(value)

    def test_bound_module_rejects_shadow_source(self):
        with tempfile.TemporaryDirectory() as directory:
            shadow = Path(directory) / "protocol_v1.py"
            shadow.write_text("SHADOW = True\n", encoding="utf-8")
            with self.assertRaisesRegex(SmokeValidationError, "bound source"):
                _assert_module_source(SimpleNamespace(__file__=str(shadow)), "protocol")

    def test_authorized_frame_rejects_symlink_component(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "sequence"
            real = sequence / "real"
            real.mkdir(parents=True)
            (real / "frame.jpg").write_bytes(b"frame")
            link = sequence / "visible"
            try:
                link.symlink_to(real, target_is_directory=True)
            except OSError:
                self.skipTest("symlink creation is unavailable")
            with self.assertRaisesRegex(SmokeValidationError, "symlink"):
                _authorized_path(sequence.absolute(), "visible/frame.jpg")

    def test_snapshot_finder_supports_namespace_parents_and_bound_leaves(self):
        source = (smoke_runner.ROOT / "lib" / "config" / "vipt" / "config.py").resolve()
        raw = source.read_bytes()
        with mock.patch.dict(smoke_runner._SOURCE_SNAPSHOT, {source: raw}, clear=True):
            finder = smoke_runner._SnapshotSourceFinder()
            namespace_spec = finder.find_spec("lib.config")
            self.assertIsNone(namespace_spec.loader)
            self.assertEqual(
                list(namespace_spec.submodule_search_locations),
                [str((smoke_runner.ROOT / "lib" / "config").resolve())],
            )
            leaf_spec = finder.find_spec("lib.config.vipt.config")
            self.assertIsInstance(leaf_spec.loader, smoke_runner._SnapshotSourceLoader)
            self.assertEqual(leaf_spec.loader.raw, raw)

    def test_preloaded_foreign_analysis_submodule_is_rejected(self):
        module_name = "analysis.foreign_smoke_test"
        foreign = ModuleType(module_name)
        foreign.__file__ = "/tmp/foreign-analysis.py"
        with mock.patch.dict(sys.modules, {module_name: foreign}):
            with self.assertRaisesRegex(SmokeValidationError, "preloaded module"):
                smoke_runner._reject_preloaded_foreign_modules(
                    "analysis", smoke_runner.ROOT / "analysis"
                )

    def test_missing_pytorch_deterministic_api_is_rejected(self):
        torch = ModuleType("torch")
        with mock.patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(SmokeValidationError, "deterministic APIs"):
                smoke_runner.configure_determinism()

    def _parent_args(self, output):
        return SimpleNamespace(
            authorization="authorization.json",
            authorization_sha256="b" * 64,
            dataset_root="dataset",
            checkpoint="checkpoint.pth",
            model_config="model.yaml",
            registry="registry.json",
            output=str(output),
        )

    def _worker_value(self, marker):
        transcript = {
            "marker": marker,
            "sequence_name": "smoke-sequence",
            "frame_indices": [0, 1],
        }
        return smoke_runner.with_content_hash({
            "schema_version": SMOKE_SCHEMA_VERSION,
            "artifact_type": "vipt_real_smoke_worker_transcript",
            "scope": "non_official_smoke",
            "authorization_sha256": "b" * 64,
            "bindings": {},
            "sequence_name": "smoke-sequence",
            "frame_indices": [0, 1],
            "transcript_hash": smoke_runner.canonical_json_hash(transcript),
            "transcript": transcript,
        })

    def test_worker_rejects_incomplete_transcript_body(self):
        authorization = {
            "sequence": {"name": "smoke-sequence"},
            "frames": [{"index": 0}, {"index": 1}],
        }
        with self.assertRaisesRegex(SmokeValidationError, "key mismatch"):
            smoke_runner.validate_worker_transcript(
                self._worker_value(0), authorization, "b" * 64, {}
            )

    def test_parent_rejects_worker_content_hash_tamper(self):
        authorization = {
            "sequence": {"name": "smoke-sequence"},
            "frames": [{"index": 0}, {"index": 1}],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"

            def worker(command, **_kwargs):
                value = self._worker_value(0)
                value["content_hash"] = "0" * 64
                Path(command[-1]).write_text(json.dumps(value), encoding="utf-8")
                return SimpleNamespace(returncode=0, stderr="")

            with mock.patch.object(smoke_runner, "load_authorization", return_value=authorization), \
                    mock.patch.object(
                        smoke_runner, "verify_bindings",
                        return_value=({"observed": {}}, object(), object()),
                    ), \
                    mock.patch.object(smoke_runner, "validate_transcript_body"), \
                    mock.patch.object(smoke_runner.subprocess, "run", side_effect=worker):
                with self.assertRaisesRegex(ValueError, "content_hash mismatch"):
                    smoke_runner.run_parent(self._parent_args(output))

    def test_verified_checkpoint_rejects_byte_drift(self):
        with self.assertRaisesRegex(SmokeValidationError, "bytes differ"):
            smoke_runner._VerifiedCheckpointBytes(b"replacement", "0" * 64)

    def test_verified_checkpoint_opens_fresh_exact_streams(self):
        expected = smoke_runner.hashlib.sha256(b"approved").hexdigest()
        checkpoint = smoke_runner._VerifiedCheckpointBytes(b"approved", expected)
        first = checkpoint.open()
        self.assertEqual(first.read(), b"approved")
        first.seek(0)
        first.write(b"changed!")
        checkpoint.verify()
        self.assertEqual(checkpoint.open().read(), b"approved")

    def test_parent_rejects_fresh_process_repeat_mismatch(self):
        authorization = {
            "sequence": {"name": "smoke-sequence"},
            "frames": [{"index": 0}, {"index": 1}],
        }
        values = [self._worker_value(0), self._worker_value(1)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"

            def worker(command, **_kwargs):
                value = values.pop(0)
                Path(command[-1]).write_text(
                    json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
                )
                return SimpleNamespace(returncode=0, stderr="")

            with mock.patch.object(smoke_runner, "load_authorization", return_value=authorization), \
                    mock.patch.object(
                        smoke_runner, "verify_bindings",
                        return_value=({"observed": {}}, object(), object()),
                    ), \
                    mock.patch.object(smoke_runner, "validate_transcript_body"), \
                    mock.patch.object(smoke_runner.subprocess, "run", side_effect=worker):
                with self.assertRaisesRegex(SmokeValidationError, "transcript hashes differ"):
                    smoke_runner.run_parent(self._parent_args(output))

    def test_parent_accepts_identical_workers_but_remains_nonofficial(self):
        authorization = {
            "sequence": {"name": "smoke-sequence"},
            "frames": [{"index": 0}, {"index": 1}],
        }
        value = self._worker_value(0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"

            def worker(command, **_kwargs):
                Path(command[-1]).write_text(
                    json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
                )
                return SimpleNamespace(returncode=0, stderr="")

            with mock.patch.object(smoke_runner, "load_authorization", return_value=authorization), \
                    mock.patch.object(
                        smoke_runner, "verify_bindings",
                        return_value=({"observed": {}}, object(), object()),
                    ), \
                    mock.patch.object(smoke_runner, "validate_transcript_body"), \
                    mock.patch.object(smoke_runner.subprocess, "run", side_effect=worker):
                report = smoke_runner.run_parent(self._parent_args(output))
            self.assertEqual(report["status"], "PASS")
            self.assertFalse(report["formal_result"])
            with self.assertRaises(ValueError):
                smoke_runner.validate_official_gate_input(report)

    def test_output_inside_repository_is_rejected(self):
        with self.assertRaisesRegex(SmokeValidationError, "outside the source repository"):
            smoke_runner._resolved_outside_repository(
                smoke_runner.ROOT / "smoke-report.json", "smoke output"
            )

    def test_map_summary_requires_exact_shape_dtype_and_finite(self):
        valid = np.ones((1, 1, 16, 16), dtype=np.float32)
        result = map_summary(valid, "score_map", 16)
        self.assertEqual(result["shape"], [1, 1, 16, 16])
        for value, expected in (
            (np.ones((1, 1, 8, 8), dtype=np.float32), "shape mismatch"),
            (np.ones((1, 1, 16, 16), dtype=np.float64), "dtype mismatch"),
            (np.full((1, 1, 16, 16), np.nan, dtype=np.float32), "non-finite"),
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(SmokeValidationError, expected):
                    map_summary(value, "score_map", 16)


if __name__ == "__main__":
    unittest.main()
