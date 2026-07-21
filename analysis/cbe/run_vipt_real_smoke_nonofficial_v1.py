"""Audited non-official real-model smoke for the CBE ViPT probe path."""

from __future__ import annotations

import os
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
if __name__ == "__main__" and (not sys.flags.isolated or not sys.flags.no_site):
    os.execv(
        sys.executable,
        [sys.executable, "-I", "-S", os.path.abspath(__file__), *sys.argv[1:]],
    )

import argparse
from dataclasses import asdict
import hashlib
import importlib
import importlib.abc
import importlib.util
from importlib.machinery import ModuleSpec
import io
import inspect
import json
from pathlib import Path, PurePath
import platform
import random
from types import ModuleType, SimpleNamespace
import re
import subprocess
import sysconfig
import tempfile
from typing import Any, Mapping, Optional, Sequence, Union


class SmokeValidationError(RuntimeError):
    pass


if sys.version_info < (3, 9):
    raise SmokeValidationError("real smoke requires Python 3.9 or newer")

ROOT = Path(__file__).resolve().parents[2]
while str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
_FORMAL_CLI = __name__ == "__main__"
_SOURCE_SNAPSHOT: dict[Path, bytes] = {}
_SOURCE_SNAPSHOT_COMMIT: Optional[str] = None


class _SnapshotSourceLoader(importlib.abc.Loader):
    def __init__(self, path: Path, raw: bytes):
        self.path = path
        self.raw = raw

    def create_module(self, spec: ModuleSpec) -> Optional[ModuleType]:
        return None

    def exec_module(self, module: ModuleType) -> None:
        module.__file__ = str(self.path)
        module.__cbe_source_sha256__ = hashlib.sha256(self.raw).hexdigest()
        code = compile(self.raw, str(self.path), "exec", dont_inherit=True)
        exec(code, module.__dict__)


class _SnapshotSourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None,
    ) -> Optional[ModuleSpec]:
        if fullname == "analysis" or fullname == "analysis.cbe" or fullname == "lib":
            return None
        parts = fullname.split(".")
        if parts[0] == "analysis" and parts[1:2] == ["cbe"]:
            relative = Path(*parts).with_suffix(".py")
            package_relative = Path(*parts) / "__init__.py"
        elif parts[0] == "lib":
            relative = Path(*parts).with_suffix(".py")
            package_relative = Path(*parts) / "__init__.py"
        else:
            return None
        for candidate, is_package in ((package_relative, True), (relative, False)):
            source_path = (ROOT / candidate).resolve()
            raw = _SOURCE_SNAPSHOT.get(source_path)
            if raw is None:
                continue
            loader = _SnapshotSourceLoader(source_path, raw)
            spec = importlib.util.spec_from_loader(
                fullname, loader, origin=str(source_path), is_package=is_package
            )
            if spec is not None and is_package:
                spec.submodule_search_locations = [str(source_path.parent)]
            return spec
        namespace_directory = (ROOT / Path(*parts)).resolve()
        if any(namespace_directory in source.parents for source in _SOURCE_SNAPSHOT):
            spec = ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [str(namespace_directory)]
            return spec
        raise ModuleNotFoundError(
            f"project module is absent from startup source snapshot: {fullname}"
        )


def _capture_project_source_snapshot() -> None:
    global _SOURCE_SNAPSHOT_COMMIT
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SmokeValidationError("project source snapshot requires a Git HEAD") from exc
    if re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
        raise SmokeValidationError("project source snapshot Git HEAD is invalid")
    roots = (ROOT / "analysis" / "cbe", ROOT / "lib")
    paths = sorted(path for root in roots for path in root.rglob("*.py"))
    if not paths:
        raise SmokeValidationError("project source snapshot is empty")
    try:
        tracked_output = subprocess.run(
            [
                "git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", commit,
                "--", "analysis/cbe", "lib",
            ],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SmokeValidationError("cannot enumerate Git-bound project sources") from exc
    tracked = {
        line for line in tracked_output.splitlines()
        if line.endswith(".py")
    }
    observed = {path.relative_to(ROOT).as_posix() for path in paths}
    if observed != tracked:
        raise SmokeValidationError(
            "project Python source set differs from Git HEAD; "
            f"missing={sorted(tracked - observed)}, extra={sorted(observed - tracked)}"
        )
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise SmokeValidationError(f"project source is not a regular file: {path}")
        relative = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        try:
            indexed = subprocess.run(
                ["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
                check=True, capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SmokeValidationError(
                f"project source is not bound to the Git index: {relative}"
            ) from exc
        if raw != indexed:
            raise SmokeValidationError(f"project source differs from Git index: {relative}")
        _SOURCE_SNAPSHOT[path.resolve()] = raw
    _SOURCE_SNAPSHOT_COMMIT = commit
    sys.meta_path.insert(0, _SnapshotSourceFinder())


if _FORMAL_CLI:
    _capture_project_source_snapshot()


def _path_within(path: Path, directory: Path) -> bool:
    resolved = path.resolve()
    root = directory.resolve()
    return resolved == root or root in resolved.parents


def _reject_preloaded_foreign_modules(
    prefix: str, directory: Path, reject_all: bool = False,
) -> None:
    for module_name, module in tuple(sys.modules.items()):
        if module_name != prefix and not module_name.startswith(prefix + "."):
            continue
        if reject_all:
            raise SmokeValidationError(
                f"CLI process preloaded project module before source binding: {module_name}"
            )
        locations = tuple(getattr(module, "__path__", ()))
        loaded_file = getattr(module, "__file__", None)
        paths = [Path(item) for item in locations]
        if isinstance(loaded_file, str):
            paths.append(Path(loaded_file))
        if not paths or any(not _path_within(path, directory) for path in paths):
            raise SmokeValidationError(
                f"preloaded module differs from project source: {module_name}"
            )


def _bind_repository_package(
    name: str, directory: Path, init_file: Optional[Path],
) -> ModuleType:
    expected_directory = directory.resolve(strict=True)
    expected_init = None if init_file is None else init_file.resolve(strict=True)
    existing = sys.modules.get(name)
    if existing is not None:
        paths = tuple(Path(path).resolve() for path in getattr(existing, "__path__", ()))
        loaded_file = getattr(existing, "__file__", None)
        if paths != (expected_directory,):
            raise SmokeValidationError(f"preloaded package differs from project source: {name}")
        if expected_init is not None and (
            not isinstance(loaded_file, str) or Path(loaded_file).resolve() != expected_init
        ):
            raise SmokeValidationError(f"preloaded package initializer differs from project source: {name}")
        return existing
    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(expected_directory)]
    package.__file__ = None if expected_init is None else str(expected_init)
    spec = ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(expected_directory)]
    package.__spec__ = spec
    sys.modules[name] = package
    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        setattr(sys.modules[parent_name], child_name, package)
    return package


_reject_preloaded_foreign_modules(
    "analysis", ROOT / "analysis", reject_all=__name__ == "__main__"
)
_reject_preloaded_foreign_modules(
    "lib", ROOT / "lib", reject_all=__name__ == "__main__"
)
_bind_repository_package("analysis", ROOT / "analysis", None)
_bind_repository_package(
    "analysis.cbe", ROOT / "analysis" / "cbe", ROOT / "analysis" / "cbe" / "__init__.py"
)
_bind_repository_package("lib", ROOT / "lib", ROOT / "lib" / "__init__.py")

approved_site_paths = [sysconfig.get_paths().get(key) for key in ("purelib", "platlib")]
for site_path in approved_site_paths:
    if isinstance(site_path, str) and site_path and site_path not in sys.path:
        sys.path.append(site_path)

import numpy as np

if _FORMAL_CLI:
    allowed_packages = {"analysis", "analysis.cbe", "lib"}
    unexpected = sorted(
        name for name in sys.modules
        if (name.startswith("analysis.") or name.startswith("lib."))
        and name not in allowed_packages
    )
    if unexpected:
        raise SmokeValidationError(
            "third-party dependency preloaded project modules: " + ", ".join(unexpected)
        )

import analysis.cbe.protocol_v1 as protocol_v1  # noqa: E402
from analysis.cbe.protocol_v1 import (  # noqa: E402
    atomic_write_json,
    canonical_json_hash,
    load_json_strict,
    loads_json_strict,
    require_exact_keys,
    sha256_file,
    validate_content_hash,
    validate_official_gate_input,
    validate_sequence_name,
    with_content_hash,
)

SMOKE_SCHEMA_VERSION = "cbe-vipt-real-smoke-nonofficial-v1"
SMOKE_SCOPE = "non_official_smoke"
FORBIDDEN_ROLES = frozenset({"design83", "internal42", "confirm62", "val47"})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
SOURCE_PATHS = {
    "smoke_runner": Path(__file__).resolve(),
    "protocol": ROOT / "analysis" / "cbe" / "protocol_v1.py",
    "probe": ROOT / "analysis" / "cbe" / "tracker_probe_v1.py",
    "interventions": ROOT / "analysis" / "cbe" / "interventions_v1.py",
    "tracker_stage0": ROOT / "lib" / "test" / "tracker" / "vipt_stage0.py",
    "tracker": ROOT / "lib" / "test" / "tracker" / "vipt.py",
    "config": ROOT / "lib" / "config" / "vipt" / "config.py",
    "model": ROOT / "lib" / "models" / "vipt" / "ostrack_prompt.py",
}
MODULE_SOURCE_KEYS = {
    "analysis.cbe.protocol_v1": "protocol",
    "analysis.cbe.tracker_probe_v1": "probe",
    "analysis.cbe.interventions_v1": "interventions",
    "lib.test.tracker.vipt_stage0": "tracker_stage0",
    "lib.test.tracker.vipt": "tracker",
    "lib.config.vipt.config": "config",
    "lib.models.vipt.ostrack_prompt": "model",
}
_MAP_CHANNELS = {
    "score_map": 1,
    "size_map": 2,
    "offset_map": 2,
    "hann_response": 1,
    "response_map": 1,
}


def _absolute_without_symlink(path: Union[str, os.PathLike[str]], label: str) -> Path:
    result = Path(path).expanduser().absolute()
    existing = result
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    for component in (existing, *existing.parents):
        if component.is_symlink():
            resolved_component = component.resolve()
            if component == Path("/var") and resolved_component == Path("/private/var"):
                continue
            raise SmokeValidationError(f"{label} path contains a symlink: {component}")
    return result


def _resolved_outside_repository(path: Union[str, os.PathLike[str]], label: str) -> Path:
    result = _absolute_without_symlink(path, label)
    resolved = result.resolve(strict=False)
    if resolved == ROOT or ROOT in resolved.parents:
        raise SmokeValidationError(f"{label} must be outside the source repository")
    return result


def _regular_file(path: Union[str, os.PathLike[str]], label: str) -> Path:
    result = _absolute_without_symlink(path, label)
    if not result.is_file() or result.is_symlink():
        raise SmokeValidationError(f"{label} must be a regular non-symlink file: {result}")
    return result


def _directory(path: Union[str, os.PathLike[str]], label: str) -> Path:
    result = _absolute_without_symlink(path, label)
    if not result.is_dir() or result.is_symlink():
        raise SmokeValidationError(f"{label} must be a non-symlink directory: {result}")
    return result


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise SmokeValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_keys(value: Any, keys: set[str], path: str) -> Mapping[str, Any]:
    try:
        require_exact_keys(value, keys, path)
    except ValueError as exc:
        raise SmokeValidationError(str(exc)) from exc
    return value


def _bbox(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise SmokeValidationError(f"{label} must contain exactly four values")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all() or array[2] <= 0.0 or array[3] <= 0.0:
        raise SmokeValidationError(f"{label} must be finite with positive width and height")
    return [float(item) for item in array]


def validate_authorization(value: Any) -> dict[str, Any]:
    root = _strict_keys(value, {
        "artifact_type", "authority", "bindings", "dataset", "frames",
        "init_bbox_xywh", "schema_version", "scope", "sequence", "source_identity",
    }, "$")
    if root["schema_version"] != SMOKE_SCHEMA_VERSION:
        raise SmokeValidationError("authorization schema_version mismatch")
    if root["artifact_type"] != "vipt_real_smoke_authorization":
        raise SmokeValidationError("authorization artifact_type mismatch")
    if root["scope"] != SMOKE_SCOPE:
        raise SmokeValidationError("authorization scope must be non_official_smoke")
    authority = _strict_keys(
        root["authority"], {"authorization_id", "issued_at_utc", "name"}, "$.authority"
    )
    for key, item in authority.items():
        if not isinstance(item, str) or not item or item != item.strip():
            raise SmokeValidationError(f"invalid authorization authority field: {key}")
    if not isinstance(root["dataset"], str) or not root["dataset"].strip():
        raise SmokeValidationError("dataset must be a non-empty string")
    sequence = _strict_keys(root["sequence"], {"name", "relative_root", "role"}, "$.sequence")
    name = validate_sequence_name(sequence["name"])
    if sequence["role"] in FORBIDDEN_ROLES or sequence["role"] != "non_sealed_smoke_only":
        raise SmokeValidationError("smoke authorization must bind a non-sealed-only role")
    relative_root = PurePath(sequence["relative_root"])
    if (relative_root.is_absolute() or len(relative_root.parts) != 1
            or sequence["relative_root"] != name):
        raise SmokeValidationError("sequence relative_root must exactly equal its safe name")
    frames = root["frames"]
    if not isinstance(frames, list) or len(frames) not in (2, 3):
        raise SmokeValidationError("authorization must bind exactly 2 or 3 frames")
    indices = []
    for ordinal, frame in enumerate(frames):
        _strict_keys(
            frame,
            {
                "index", "infrared_relative_path", "infrared_sha256", "ordinal",
                "visible_relative_path", "visible_sha256",
            },
            f"$.frames[{ordinal}]",
        )
        if (not isinstance(frame["ordinal"], int) or isinstance(frame["ordinal"], bool)
                or frame["ordinal"] != ordinal):
            raise SmokeValidationError(f"frame ordinal mismatch at index {ordinal}")
        index = frame["index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise SmokeValidationError(f"invalid frame index at ordinal {ordinal}")
        indices.append(index)
        frame_stems = []
        for key in ("visible_relative_path", "infrared_relative_path"):
            relative = PurePath(frame[key])
            if (not isinstance(frame[key], str) or relative.is_absolute()
                    or not relative.parts or ".." in relative.parts):
                raise SmokeValidationError(f"unsafe authorized frame path: {key}")
            stem = relative.stem
            if not stem.isascii() or not stem.isdigit() or int(stem) != index:
                raise SmokeValidationError(
                    f"authorized frame path does not match frame index: {key}"
                )
            frame_stems.append(stem)
            _sha256(frame[key.replace("relative_path", "sha256")], f"authorized {key} bytes")
        if frame_stems[0] != frame_stems[1]:
            raise SmokeValidationError("visible and infrared frame stems must match")
    if indices[0] != 0 or indices != sorted(set(indices)):
        raise SmokeValidationError("frame indices must be strictly increasing and begin at 0")
    root["init_bbox_xywh"] = _bbox(root["init_bbox_xywh"], "init_bbox_xywh")
    source_identity = _strict_keys(
        root["source_identity"], {"dirty_tree", "git_commit", "repository"},
        "$.source_identity",
    )
    if source_identity["dirty_tree"] is not False:
        raise SmokeValidationError("smoke requires dirty_tree=false")
    if not isinstance(source_identity["git_commit"], str) or _GIT_COMMIT.fullmatch(source_identity["git_commit"]) is None:
        raise SmokeValidationError("authorization git_commit is invalid")
    if not isinstance(source_identity["repository"], str) or not source_identity["repository"].strip():
        raise SmokeValidationError("authorization repository is invalid")
    bindings = _strict_keys(root["bindings"], {
        "checkpoint_sha256", "environment_hash", "model_config_sha256",
        "registry_sha256", "resolved_config_hash", "source_bundle_hash",
    }, "$.bindings")
    for key, item in bindings.items():
        _sha256(item, f"binding {key}")
    return dict(root)


def load_authorization(path: Path, expected_sha256: str) -> dict[str, Any]:
    authorization = _regular_file(path, "authorization")
    expected = _sha256(expected_sha256, "authorization SHA-256")
    raw = authorization.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SmokeValidationError("authorization byte drift")
    return validate_authorization(loads_json_strict(raw))


def _assert_module_source(module: Any, source_key: str) -> None:
    imported_file = getattr(module, "__file__", None)
    if not isinstance(imported_file, str):
        raise SmokeValidationError(f"imported module has no source file: {module!r}")
    imported = _regular_file(imported_file, f"imported module {source_key}")
    bound = _regular_file(SOURCE_PATHS[source_key], f"bound source {source_key}")
    if imported != bound:
        raise SmokeValidationError(f"imported module differs from bound source: {source_key}")
    if _FORMAL_CLI:
        snapshot = _SOURCE_SNAPSHOT.get(bound.resolve())
        spec = getattr(module, "__spec__", None)
        loader = getattr(spec, "loader", None)
        if (
            snapshot is None
            or not isinstance(loader, _SnapshotSourceLoader)
            or loader.path.resolve() != bound.resolve()
            or loader.raw != snapshot
        ):
            raise SmokeValidationError(
                f"imported module was not executed from bound snapshot: {source_key}"
            )
    elif sha256_file(imported) != sha256_file(bound):
        raise SmokeValidationError(f"imported module differs from bound source: {source_key}")


def verify_loaded_module_sources() -> None:
    expected_packages = {
        "analysis": (ROOT / "analysis", None),
        "analysis.cbe": (ROOT / "analysis" / "cbe", ROOT / "analysis" / "cbe" / "__init__.py"),
        "lib": (ROOT / "lib", ROOT / "lib" / "__init__.py"),
    }
    for name, (directory, init_file) in expected_packages.items():
        package = sys.modules.get(name)
        paths = tuple(Path(path).resolve() for path in getattr(package, "__path__", ()))
        if paths != (directory.resolve(),):
            raise SmokeValidationError(f"{name} package path differs from project source")
        if init_file is not None:
            loaded_file = getattr(package, "__file__", None)
            if not isinstance(loaded_file, str) or Path(loaded_file).resolve() != init_file.resolve():
                raise SmokeValidationError(
                    f"{name} package initializer differs from project source"
                )
    _assert_module_source(protocol_v1, "protocol")
    for module_name, source_key in MODULE_SOURCE_KEYS.items():
        module = sys.modules.get(module_name)
        if module is not None:
            _assert_module_source(module, source_key)


def _import_bound_module(module_name: str) -> Any:
    source_key = MODULE_SOURCE_KEYS[module_name]
    source_path = _regular_file(SOURCE_PATHS[source_key], f"bound source {source_key}")
    existing = sys.modules.get(module_name)
    if existing is not None:
        _assert_module_source(existing, source_key)
        return existing
    if _FORMAL_CLI:
        module = importlib.import_module(module_name)
    else:
        parent_name = module_name.rpartition(".")[0]
        if parent_name:
            importlib.import_module(parent_name)
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise SmokeValidationError(f"cannot load identity-bound module: {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    _assert_module_source(module, source_key)
    return module


def source_bundle() -> dict[str, Any]:
    verify_loaded_module_sources()
    files = {}
    for name, path in SOURCE_PATHS.items():
        source = _regular_file(path, f"source {name}")
        if _FORMAL_CLI:
            raw = _SOURCE_SNAPSHOT.get(source.resolve())
            if raw is None:
                raise SmokeValidationError(f"source is absent from startup snapshot: {name}")
            digest = hashlib.sha256(raw).hexdigest()
        else:
            digest = sha256_file(source)
        files[name] = {"path": str(source), "sha256": digest}
    return {"files": files, "hash": canonical_json_hash({name: item["sha256"] for name, item in files.items()})}


def git_identity() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout
        repository = subprocess.run(
            ["git", "-C", str(ROOT), "config", "--get", "remote.origin.url"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SmokeValidationError("smoke requires an auditable Git repository") from exc
    if _FORMAL_CLI and commit != _SOURCE_SNAPSHOT_COMMIT:
        raise SmokeValidationError("Git HEAD changed after source snapshot")
    return {"git_commit": commit, "dirty_tree": bool(status), "repository": repository}


def configure_determinism() -> None:
    import torch

    required = ("use_deterministic_algorithms", "are_deterministic_algorithms_enabled")
    missing = [name for name in required if not callable(getattr(torch, name, None))]
    if missing:
        raise SmokeValidationError(
            "real smoke requires PyTorch deterministic APIs: " + ", ".join(missing)
        )
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise SmokeValidationError("real smoke requires torch.load(weights_only=...)")
    random.seed(20260720)
    np.random.seed(20260720)
    torch.manual_seed(20260720)
    torch.cuda.manual_seed_all(20260720)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def environment_manifest() -> dict[str, Any]:
    try:
        import cv2
        import torch
    except ImportError as exc:
        raise SmokeValidationError("real smoke requires OpenCV and PyTorch") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SmokeValidationError("real smoke requires exactly one visible CUDA device")
    configure_determinism()
    device = torch.cuda.get_device_properties(0)
    value = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": device.name,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_driver": torch._C._cuda_getDriverVersion() if hasattr(torch._C, "_cuda_getDriverVersion") else None,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_isolated": sys.flags.isolated,
    }
    for package_name in ("easydict", "pyyaml", "timm"):
        try:
            import importlib.metadata
            value[f"package_{package_name}"] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            value[f"package_{package_name}"] = None
    return {"values": value, "hash": canonical_json_hash(value)}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolved_config_bytes(raw: bytes) -> tuple[Any, str]:
    try:
        import yaml
    except ImportError as exc:
        raise SmokeValidationError("real smoke requires PyYAML") from exc
    config_module = _import_bound_module("lib.config.vipt.config")
    try:
        parsed = yaml.safe_load(raw.decode("utf-8"))
        exp_config = type(config_module.cfg)(parsed)
        config_module._update_config(config_module.cfg, exp_config)
    except (UnicodeDecodeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise SmokeValidationError("model config bytes are invalid") from exc
    value = _plain(config_module.cfg)
    return config_module.cfg, canonical_json_hash(value)


def _read_regular_bytes(path: Path, label: str) -> tuple[Path, bytes, str]:
    source = _regular_file(path, label)
    raw = source.read_bytes()
    return source, raw, hashlib.sha256(raw).hexdigest()


def inspect_bindings(checkpoint: Path, model_config: Path, registry: Path) -> dict[str, Any]:
    _, checkpoint_raw, checkpoint_hash = _read_regular_bytes(checkpoint, "checkpoint")
    _, config_raw, config_file_hash = _read_regular_bytes(model_config, "model config")
    _, registry_raw, registry_hash = _read_regular_bytes(registry, "intervention registry")
    loads_json_strict(registry_raw)
    _, config_hash = resolved_config_bytes(config_raw)
    return with_content_hash({
        "schema_version": SMOKE_SCHEMA_VERSION,
        "artifact_type": "vipt_real_smoke_candidate_bindings",
        "scope": SMOKE_SCOPE,
        "candidate_only": True,
        "formal_result": False,
        "source_identity": git_identity(),
        "bindings": {
            "checkpoint_sha256": checkpoint_hash,
            "model_config_sha256": config_file_hash,
            "registry_sha256": registry_hash,
            "resolved_config_hash": config_hash,
            "source_bundle_hash": source_bundle()["hash"],
            "environment_hash": environment_manifest()["hash"],
        },
    })


def verify_bindings(
    authorization: Mapping[str, Any], checkpoint: Path, model_config: Path, registry: Path,
) -> tuple[dict[str, Any], Any, "_VerifiedCheckpointBytes"]:
    _, checkpoint_raw, checkpoint_hash = _read_regular_bytes(checkpoint, "checkpoint")
    _, config_raw, config_file_hash = _read_regular_bytes(model_config, "model config")
    _, registry_raw, registry_hash = _read_regular_bytes(registry, "intervention registry")
    loads_json_strict(registry_raw)
    cfg, config_hash = resolved_config_bytes(config_raw)
    source = source_bundle()
    environment = environment_manifest()
    observed = {
        "checkpoint_sha256": checkpoint_hash,
        "model_config_sha256": config_file_hash,
        "registry_sha256": registry_hash,
        "resolved_config_hash": config_hash,
        "source_bundle_hash": source["hash"],
        "environment_hash": environment["hash"],
    }
    if observed != authorization["bindings"]:
        raise SmokeValidationError("runtime bindings differ from authorization")
    if git_identity() != authorization["source_identity"]:
        raise SmokeValidationError("Git identity differs from authorization")
    verified_checkpoint = _VerifiedCheckpointBytes(
        checkpoint_raw, authorization["bindings"]["checkpoint_sha256"]
    )
    return (
        {"observed": observed, "source_bundle": source, "environment": environment},
        cfg,
        verified_checkpoint,
    )


def _authorized_path(sequence_root: Path, relative_path: str) -> Path:
    path = _absolute_without_symlink(sequence_root / relative_path, "authorized frame")
    current = path
    while current != sequence_root:
        if current.is_symlink():
            raise SmokeValidationError(f"authorized frame path contains a symlink: {current}")
        current = current.parent
    if not path.is_file() or sequence_root not in path.parents:
        raise SmokeValidationError(f"unsafe or missing authorized frame: {relative_path}")
    return path


def decode_authorized_frames(dataset_root: Path, authorization: Mapping[str, Any]) -> list[np.ndarray]:
    import cv2
    interventions = _import_bound_module("analysis.cbe.interventions_v1")

    root = _directory(dataset_root, "dataset root")
    sequence_root = (root / authorization["sequence"]["relative_root"]).resolve()
    if not sequence_root.is_dir() or sequence_root.is_symlink() or sequence_root.parent != root:
        raise SmokeValidationError("authorized sequence root is missing or unsafe")
    frames = []
    for frame in authorization["frames"]:
        modalities = []
        for key in ("visible_relative_path", "infrared_relative_path"):
            path = _authorized_path(sequence_root, frame[key])
            expected_sha256 = frame[key.replace("relative_path", "sha256")]
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected_sha256:
                raise SmokeValidationError(f"authorized frame byte drift: {path}")
            encoded = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise SmokeValidationError(f"cannot decode authorized uint8 frame: {path}")
            modalities.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(interventions.merge_six_channel(*modalities))
    return frames


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def map_summary(value: Any, name: str, feature_size: int) -> dict[str, Any]:
    array = np.asarray(value)
    expected = (1, _MAP_CHANNELS[name], feature_size, feature_size)
    if array.shape != expected:
        raise SmokeValidationError(f"{name} shape mismatch: got {array.shape}, expected {expected}")
    if array.dtype != np.float32:
        raise SmokeValidationError(f"{name} dtype mismatch: got {array.dtype}, expected float32")
    if not np.isfinite(array).all():
        raise SmokeValidationError(f"{name} contains non-finite values")
    return {"shape": list(array.shape), "dtype": str(array.dtype), "finite": True, "sha256": _array_hash(array)}


def prediction_summary(value: Mapping[str, Any], feature_size: int) -> dict[str, Any]:
    maps = {name: map_summary(value[name], name, feature_size) for name in _MAP_CHANNELS}
    return {
        "maps": maps,
        "target_bbox": _bbox(value["target_bbox"], "prediction target_bbox"),
        "best_score": float(value["best_score"]),
        "search_patch_hash": value["search_patch_hash"],
        "anchor_id": value["anchor_id"],
        "template_id": value["template_id"],
    }


def _fingerprint(value: Any) -> dict[str, Any]:
    result = asdict(value)
    result["state"] = repr(result["state"])
    result["frame_id"] = repr(result["frame_id"])
    return result


def _snapshot_hashes(adapter: Any) -> dict[str, str]:
    probe_module = _import_bound_module("analysis.cbe.tracker_probe_v1")

    return {
        name: canonical_json_hash({
            "template_id": snapshot.template_id,
            "z_patch_arr": probe_module._array_hash(snapshot.z_patch_arr),
            "z_tensor": probe_module._array_hash(snapshot.z_tensor),
            "box_mask_z": probe_module._array_hash(snapshot.box_mask_z),
        })
        for name, snapshot in adapter.snapshots.items()
    }


class _VerifiedCheckpointBytes:
    def __init__(self, raw: bytes, expected_sha256: str):
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise SmokeValidationError("checkpoint bytes differ from authorization")
        self._raw = bytes(raw)
        self._expected_sha256 = expected_sha256

    def verify(self) -> None:
        if hashlib.sha256(self._raw).hexdigest() != self._expected_sha256:
            raise SmokeValidationError("verified checkpoint bytes changed in memory")

    def open(self) -> io.BytesIO:
        self.verify()
        return io.BytesIO(self._raw)


def build_params(cfg: Any, checkpoint_stream: io.BytesIO) -> Any:
    params = SimpleNamespace()
    params.cfg = cfg
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = checkpoint_stream
    params.save_all_boxes = False
    params.debug = 0
    return params


def execute_transcript(
    authorization: Mapping[str, Any], frames: Sequence[np.ndarray], cfg: Any,
    checkpoint: _VerifiedCheckpointBytes,
) -> dict[str, Any]:
    interventions = _import_bound_module("analysis.cbe.interventions_v1")
    probe_module = _import_bound_module("analysis.cbe.tracker_probe_v1")
    tracker_module = _import_bound_module("lib.test.tracker.vipt_stage0")

    configure_determinism()
    verify_loaded_module_sources()
    import torch

    original_torch_load = torch.load
    checkpoint_stream = checkpoint.open()

    def load_verified_checkpoint(source: Any, *load_args: Any, **load_kwargs: Any) -> Any:
        if source is not checkpoint_stream:
            raise SmokeValidationError("tracker attempted to load an unbound checkpoint")
        checkpoint.verify()
        checkpoint_stream.seek(0)
        load_kwargs["weights_only"] = True
        return original_torch_load(checkpoint_stream, *load_args, **load_kwargs)

    torch.load = load_verified_checkpoint
    try:
        tracker = tracker_module.ViPTStage0Track(build_params(cfg, checkpoint_stream))
    finally:
        torch.load = original_torch_load
        checkpoint_stream.close()
    checkpoint.verify()
    init_bbox = authorization["init_bbox_xywh"]
    tracker.initialize(frames[0], {"init_bbox": init_bbox})
    initialized = probe_module.tracker_state_fingerprint(tracker)
    adapter = probe_module.CBEStage0ProbeAdapter(tracker, frames[0], init_bbox)
    tracker.commit_template(adapter.factual_snapshot)
    committed = probe_module.tracker_state_fingerprint(tracker)
    snapshot_hashes = _snapshot_hashes(adapter)
    feature_size = int(tracker.params.search_size // tracker.cfg.MODEL.BACKBONE.STRIDE)
    records = []
    for authorization_frame, image in zip(authorization["frames"][1:], frames[1:]):
        before = probe_module.tracker_state_fingerprint(tracker)
        anchor = list(tracker.state)
        probes = adapter.run_clean_probe_set(image, anchor)
        after_clean = probe_module.tracker_state_fingerprint(tracker)
        mask = np.zeros(image.shape[:2], dtype=bool)
        height, width = mask.shape
        mask[height // 4:max(height // 4 + 1, 3 * height // 4),
             width // 4:max(width // 4 + 1, 3 * width // 4)] = True
        zero = interventions.apply_local_intervention(
            image, mask, interventions.InterventionSpec("blur", "rgb", 0.0, "nonofficial-smoke")
        )
        if not np.array_equal(zero, image):
            raise SmokeValidationError("strength-zero intervention is not byte-identical")
        zero_prediction = adapter.predict(zero, anchor, adapter.factual_snapshot)
        factual_summary = prediction_summary(probes["factual"], feature_size)
        zero_summary = prediction_summary(zero_prediction, feature_size)
        if zero_summary != factual_summary:
            raise SmokeValidationError("strength-zero prediction differs from factual prediction")
        counterfactual = interventions.apply_local_intervention(
            image, mask, interventions.InterventionSpec("blur", "rgb", 0.25, "nonofficial-smoke")
        )
        counterfactual_prediction = adapter.predict(
            counterfactual, anchor, adapter.factual_snapshot
        )
        after_counterfactual = probe_module.tracker_state_fingerprint(tracker)
        advanced = adapter.advance_factual(image, anchor)
        after_advance = probe_module.tracker_state_fingerprint(tracker)
        if before != after_clean or before != after_counterfactual:
            raise SmokeValidationError("read-only smoke probe changed tracker state")
        if _snapshot_hashes(adapter) != snapshot_hashes:
            raise SmokeValidationError("read-only smoke probe changed an adapter snapshot")
        records.append({
            "frame_index": authorization_frame["index"],
            "anchor": anchor,
            "state_before": _fingerprint(before),
            "state_after_clean": _fingerprint(after_clean),
            "state_after_counterfactual": _fingerprint(after_counterfactual),
            "state_after_advance": _fingerprint(after_advance),
            "clean_probes": {
                name: prediction_summary(prediction, feature_size)
                for name, prediction in probes.items()
            },
            "strength_zero": {
                "input_sha256": _array_hash(image),
                "output_sha256": _array_hash(zero),
                "byte_identity": True,
                "prediction": zero_summary,
            },
            "counterfactual": {
                "operation": "rgb_blur", "strength": 0.25,
                "image_sha256": _array_hash(counterfactual),
                "prediction": prediction_summary(counterfactual_prediction, feature_size),
            },
            "factual_advance": advanced,
        })
    return {
        "sequence_name": authorization["sequence"]["name"],
        "frame_indices": [item["index"] for item in authorization["frames"]],
        "initialized_fingerprint": _fingerprint(initialized),
        "committed_factual_fingerprint": _fingerprint(committed),
        "snapshot_hashes": snapshot_hashes,
        "feature_size": feature_size,
        "records": records,
    }


def worker_run(args: argparse.Namespace) -> dict[str, Any]:
    authorization = load_authorization(Path(args.authorization), args.authorization_sha256)
    bindings, cfg, checkpoint = verify_bindings(
        authorization, Path(args.checkpoint), Path(args.model_config), Path(args.registry)
    )
    frames = decode_authorized_frames(Path(args.dataset_root), authorization)
    transcript = execute_transcript(authorization, frames, cfg, checkpoint)
    return with_content_hash({
        "schema_version": SMOKE_SCHEMA_VERSION,
        "artifact_type": "vipt_real_smoke_worker_transcript",
        "scope": SMOKE_SCOPE,
        "authorization_sha256": args.authorization_sha256,
        "bindings": bindings["observed"],
        "sequence_name": authorization["sequence"]["name"],
        "frame_indices": [item["index"] for item in authorization["frames"]],
        "transcript_hash": canonical_json_hash(transcript),
        "transcript": transcript,
    })


def _validate_prediction_transcript(
    value: Any, path: str, feature_size: int,
) -> None:
    prediction = _strict_keys(value, {
        "anchor_id", "best_score", "maps", "search_patch_hash", "target_bbox",
        "template_id",
    }, path)
    _bbox(prediction["target_bbox"], f"{path}.target_bbox")
    if not isinstance(prediction["best_score"], (int, float)):
        raise SmokeValidationError(f"{path}.best_score must be numeric")
    maps = _strict_keys(prediction["maps"], set(_MAP_CHANNELS), f"{path}.maps")
    for name, channels in _MAP_CHANNELS.items():
        summary = _strict_keys(
            maps[name], {"dtype", "finite", "sha256", "shape"},
            f"{path}.maps.{name}",
        )
        if summary["dtype"] != "float32" or summary["finite"] is not True:
            raise SmokeValidationError(f"{path}.maps.{name} is not finite float32")
        shape = summary["shape"]
        if (not isinstance(shape, list) or len(shape) != 4 or shape[:2] != [1, channels]
                or shape[2] != shape[3] or shape[2] != feature_size
                or not isinstance(shape[2], int) or shape[2] <= 0):
            raise SmokeValidationError(f"{path}.maps.{name} shape is invalid")
        _sha256(summary["sha256"], f"{path}.maps.{name}.sha256")
    for key in ("anchor_id", "search_patch_hash", "template_id"):
        if not isinstance(prediction[key], str) or not prediction[key]:
            raise SmokeValidationError(f"{path}.{key} must be a non-empty string")


def _validate_fingerprint_transcript(value: Any, path: str) -> None:
    fingerprint = _strict_keys(value, {
        "active_template_id", "active_template_mask_hash",
        "active_template_tensor_hash", "frame_id", "state",
    }, path)
    for key in ("active_template_mask_hash", "active_template_tensor_hash"):
        _sha256(fingerprint[key], f"{path}.{key}")


def validate_transcript_body(
    transcript: Any, authorization: Mapping[str, Any],
) -> None:
    body = _strict_keys(transcript, {
        "committed_factual_fingerprint", "feature_size", "frame_indices",
        "initialized_fingerprint", "records", "sequence_name", "snapshot_hashes",
    }, "$worker.transcript")
    expected_name = authorization["sequence"]["name"]
    expected_indices = [item["index"] for item in authorization["frames"]]
    if body["sequence_name"] != expected_name or body["frame_indices"] != expected_indices:
        raise SmokeValidationError("worker transcript identity mismatch")
    if not isinstance(body["feature_size"], int) or isinstance(body["feature_size"], bool) or body["feature_size"] <= 0:
        raise SmokeValidationError("worker transcript feature_size is invalid")
    _validate_fingerprint_transcript(
        body["initialized_fingerprint"], "$worker.transcript.initialized_fingerprint"
    )
    _validate_fingerprint_transcript(
        body["committed_factual_fingerprint"],
        "$worker.transcript.committed_factual_fingerprint",
    )
    snapshots = _strict_keys(
        body["snapshot_hashes"], {"factual", "rgb_retained", "tir_retained"},
        "$worker.transcript.snapshot_hashes",
    )
    for name, digest in snapshots.items():
        _sha256(digest, f"$worker.transcript.snapshot_hashes.{name}")
    records = body["records"]
    if not isinstance(records, list) or len(records) != len(expected_indices) - 1:
        raise SmokeValidationError("worker transcript record count mismatch")
    for ordinal, (record, frame_index) in enumerate(zip(records, expected_indices[1:])):
        path = f"$worker.transcript.records[{ordinal}]"
        row = _strict_keys(record, {
            "anchor", "clean_probes", "counterfactual", "factual_advance", "frame_index",
            "state_after_advance", "state_after_clean", "state_after_counterfactual",
            "state_before", "strength_zero",
        }, path)
        if row["frame_index"] != frame_index:
            raise SmokeValidationError(f"{path}.frame_index mismatch")
        _bbox(row["anchor"], f"{path}.anchor")
        for key in (
            "state_before", "state_after_clean", "state_after_counterfactual",
            "state_after_advance",
        ):
            _validate_fingerprint_transcript(row[key], f"{path}.{key}")
        if row["state_after_clean"] != row["state_before"]:
            raise SmokeValidationError(f"{path} clean probes changed tracker state")
        if row["state_after_counterfactual"] != row["state_before"]:
            raise SmokeValidationError(f"{path} counterfactual changed tracker state")
        probes = _strict_keys(
            row["clean_probes"], {"factual", "rgb_retained", "tir_retained"},
            f"{path}.clean_probes",
        )
        for name, prediction in probes.items():
            _validate_prediction_transcript(
                prediction, f"{path}.clean_probes.{name}", body["feature_size"]
            )
        zero = _strict_keys(
            row["strength_zero"], {"byte_identity", "input_sha256", "output_sha256", "prediction"},
            f"{path}.strength_zero",
        )
        if zero["byte_identity"] is not True or zero["input_sha256"] != zero["output_sha256"]:
            raise SmokeValidationError(f"{path}.strength_zero identity mismatch")
        _sha256(zero["input_sha256"], f"{path}.strength_zero.input_sha256")
        _validate_prediction_transcript(
            zero["prediction"], f"{path}.strength_zero.prediction", body["feature_size"]
        )
        if zero["prediction"] != probes["factual"]:
            raise SmokeValidationError(f"{path}.strength_zero prediction mismatch")
        counterfactual = _strict_keys(
            row["counterfactual"], {"image_sha256", "operation", "prediction", "strength"},
            f"{path}.counterfactual",
        )
        if counterfactual["operation"] != "rgb_blur" or counterfactual["strength"] != 0.25:
            raise SmokeValidationError(f"{path}.counterfactual protocol mismatch")
        _sha256(counterfactual["image_sha256"], f"{path}.counterfactual.image_sha256")
        _validate_prediction_transcript(
            counterfactual["prediction"], f"{path}.counterfactual.prediction",
            body["feature_size"],
        )
        advance = _strict_keys(row["factual_advance"], {
            "anchor_id", "best_score", "contextual_best_score", "contextual_target_bbox",
            "search_anchor", "target_bbox", "template_id",
        }, f"{path}.factual_advance")
        tracked_bbox = _bbox(advance["target_bbox"], f"{path}.factual_advance.target_bbox")
        contextual_bbox = _bbox(
            advance["contextual_target_bbox"],
            f"{path}.factual_advance.contextual_target_bbox",
        )
        if not np.allclose(tracked_bbox, contextual_bbox, rtol=0.0, atol=1e-5):
            raise SmokeValidationError(f"{path}.factual_advance bbox mismatch")
        if not isinstance(advance["best_score"], (int, float)) or not isinstance(
            advance["contextual_best_score"], (int, float)
        ):
            raise SmokeValidationError(f"{path}.factual_advance score is invalid")
        if not np.isclose(
            float(advance["best_score"]), float(advance["contextual_best_score"]),
            rtol=0.0, atol=1e-6,
        ):
            raise SmokeValidationError(f"{path}.factual_advance score mismatch")


def validate_worker_transcript(
    value: Any, authorization: Mapping[str, Any], authorization_sha256: str,
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    worker = _strict_keys(value, {
        "artifact_type", "authorization_sha256", "bindings", "content_hash",
        "frame_indices", "schema_version", "scope", "sequence_name", "transcript",
        "transcript_hash",
    }, "$worker")
    validate_content_hash(worker)
    if worker["schema_version"] != SMOKE_SCHEMA_VERSION:
        raise SmokeValidationError("worker schema_version mismatch")
    if worker["artifact_type"] != "vipt_real_smoke_worker_transcript":
        raise SmokeValidationError("worker emitted an invalid transcript")
    if worker["scope"] != SMOKE_SCOPE:
        raise SmokeValidationError("worker scope mismatch")
    if worker["authorization_sha256"] != authorization_sha256:
        raise SmokeValidationError("worker authorization binding mismatch")
    if worker["bindings"] != bindings:
        raise SmokeValidationError("worker runtime bindings mismatch")
    expected_name = authorization["sequence"]["name"]
    expected_indices = [item["index"] for item in authorization["frames"]]
    if worker["sequence_name"] != expected_name:
        raise SmokeValidationError("worker sequence mismatch")
    if worker["frame_indices"] != expected_indices:
        raise SmokeValidationError("worker frame indices mismatch")
    transcript = worker["transcript"]
    if not isinstance(transcript, Mapping):
        raise SmokeValidationError("worker transcript body must be an object")
    if worker["transcript_hash"] != canonical_json_hash(transcript):
        raise SmokeValidationError("worker transcript hash mismatch")
    validate_transcript_body(transcript, authorization)
    return dict(worker)


def _worker_command(args: argparse.Namespace, output: Path) -> list[str]:
    return [
        sys.executable, "-I", "-S", str(Path(__file__).resolve()), "--worker",
        "--authorization", args.authorization,
        "--authorization-sha256", args.authorization_sha256,
        "--dataset-root", args.dataset_root,
        "--checkpoint", args.checkpoint,
        "--model-config", args.model_config,
        "--registry", args.registry,
        "--worker-output", str(output),
    ]


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    authorization = load_authorization(Path(args.authorization), args.authorization_sha256)
    parent_bindings, _, _ = verify_bindings(
        authorization, Path(args.checkpoint), Path(args.model_config), Path(args.registry)
    )
    output = _resolved_outside_repository(args.output, "smoke output")
    output.parent.mkdir(parents=False, exist_ok=True)
    transcripts = []
    with tempfile.TemporaryDirectory(dir=str(output.parent)) as directory:
        for repeat in range(2):
            worker_output = Path(directory) / f"repeat-{repeat}.json"
            completed = subprocess.run(
                _worker_command(args, worker_output), capture_output=True, text=True
            )
            if completed.returncode != 0:
                raise SmokeValidationError(
                    f"real smoke worker {repeat} failed: {completed.stderr.strip()}"
                )
            value = load_json_strict(worker_output)
            transcripts.append(validate_worker_transcript(
                value,
                authorization,
                args.authorization_sha256,
                parent_bindings["observed"],
            ))
    hashes = [item["content_hash"] for item in transcripts]
    if hashes[0] != hashes[1]:
        raise SmokeValidationError("fresh-process smoke transcript hashes differ")
    artifact = with_content_hash({
        "schema_version": SMOKE_SCHEMA_VERSION,
        "artifact_type": "vipt_real_smoke_report",
        "scope": SMOKE_SCOPE,
        "formal_result": False,
        "official_phase": False,
        "authorization_sha256": args.authorization_sha256,
        "sequence_name": authorization["sequence"]["name"],
        "frame_indices": [item["index"] for item in authorization["frames"]],
        "repeat_hashes": hashes,
        "deterministic_repeat_equal": True,
        "status": "PASS",
        "transcript": transcripts[0]["transcript"],
    })
    if any(key in artifact for key in ("phase", "parent_phase", "parent_content_hash", "gate")):
        raise SmokeValidationError("non-official smoke artifact contains official phase fields")
    try:
        validate_official_gate_input(artifact)
    except ValueError:
        pass
    else:
        raise SmokeValidationError("non-official smoke artifact entered the official gate")
    atomic_write_json(output, artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an audited non-official ViPT real-model smoke")
    parser.add_argument("--inspect-bindings", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--authorization")
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--dataset-root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output")
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.inspect_bindings:
            if args.worker or args.authorization or args.dataset_root or args.worker_output:
                parser.error("--inspect-bindings rejects authorization, dataset, and worker arguments")
            report = inspect_bindings(
                Path(args.checkpoint), Path(args.model_config), Path(args.registry)
            )
            if args.output:
                atomic_write_json(
                    _resolved_outside_repository(args.output, "inspect output"), report
                )
        else:
            required = ("authorization", "authorization_sha256", "dataset_root")
            missing = [name for name in required if not getattr(args, name)]
            if missing:
                parser.error("missing required arguments: " + ", ".join("--" + name.replace("_", "-") for name in missing))
            if args.worker:
                if not args.worker_output:
                    parser.error("--worker requires --worker-output")
                report = worker_run(args)
                atomic_write_json(
                    _resolved_outside_repository(args.worker_output, "worker output"), report
                )
            else:
                if args.worker_output:
                    parser.error("--worker-output is internal-only")
                if not args.output:
                    parser.error("parent smoke requires --output")
                report = run_parent(args)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, SmokeValidationError) as exc:
        print(f"smoke validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
