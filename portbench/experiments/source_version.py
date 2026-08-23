"""Stable source signatures for experiment cache provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE_PATHS = (
    "portbench",
    "tests",
    "requirements.txt",
    "setup.py",
)
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".json"}


def source_tree_hash(root: str | Path = ".") -> str:
    """Hash behavior-bearing source files in stable relative-path order."""
    root_path = Path(root).resolve()
    files: list[Path] = []

    # Exclude reports and generated artifacts from cache provenance.
    for relative in SOURCE_PATHS:
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
    # Include paths so moving a module invalidates artifacts from the old layout.
    for path in sorted(files, key=lambda item: item.relative_to(root_path).as_posix()):
        relative = path.relative_to(root_path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = ["source_tree_hash"]
