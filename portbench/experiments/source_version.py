"""Stable source signatures for experiment cache provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE_PATHS = (
    "portbench",
    "requirements.txt",
    "setup.py",
)
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".json"}

RUNTIME_BEHAVIOR_PATHS = (
    "portbench/agent_eval",
    "portbench/sandbox",
    "portbench/metrics/ceps.py",
    "portbench/experiments/config.py",
    "portbench/experiments/freeze.py",
    "portbench/experiments/providers.py",
    "portbench/experiments/runner.py",
    "portbench/experiments/source_version.py",
)


def _hash_paths(root: str | Path, relative_paths: tuple[str, ...]) -> str:
    """Hash selected source files in stable relative-path order."""
    root_path = Path(root).resolve()
    files: list[Path] = []

    for relative in relative_paths:
        base = root_path / relative
        if not base.exists():
            continue
        if base.is_file():
            files.append(base)
            continue
        files.extend(
            path
            for path in base.rglob("*")
            if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
        )

    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root_path).as_posix()):
        relative = path.relative_to(root_path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def source_tree_hash(root: str | Path = ".") -> str:
    """Hash behavior-bearing source files in stable relative-path order."""
    return _hash_paths(root, SOURCE_PATHS)


def runtime_behavior_hash(root: str | Path = ".") -> str:
    """Hash runtime paths that can alter an SA sandbox episode."""
    return _hash_paths(root, RUNTIME_BEHAVIOR_PATHS)


__all__ = ["runtime_behavior_hash", "source_tree_hash"]
