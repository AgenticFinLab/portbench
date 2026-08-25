"""Freeze and validate behavior-bearing experiment inputs for resumable runs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .config import ExperimentConfig, ModelSpec
from .source_version import runtime_behavior_hash


FREEZE_MANIFEST_VERSION = "sa-upgrade-v2"


def _canonical(value: Any) -> str:
    """Serialize an object deterministically for a stable manifest digest."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_text(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def data_snapshot_hash(data_dir: str | Path, data_version: str) -> str:
    """Hash the declared dataset files without including source-code revisions."""
    root = Path(data_dir)
    if not root.exists():
        return _sha256_text(_canonical({"data_version": data_version, "missing": str(root)}))
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "size": path.stat().st_size,
            }
        )
    return _sha256_text(_canonical({"data_version": data_version, "files": entries}))


def build_freeze_manifest(cfg: ExperimentConfig, spec: ModelSpec) -> dict[str, Any]:
    """Build the resume contract from fields that can change observed behavior."""
    normal_periods = [
        {
            "start": str(period.start),
            "end": str(period.end),
            "label": period.label,
        }
        for period in cfg.normal_periods
    ]
    return {
        "manifest_version": FREEZE_MANIFEST_VERSION,
        "model": {
            "provider": spec.provider,
            "model": spec.model,
            "architecture_id": spec.architecture_id,
            "temperature": spec.temperature,
            "max_tokens": spec.max_tokens,
        },
        "pipeline": {
            "schema_version": cfg.pipeline_schema_version,
            "sa_only": cfg.sa_only,
            "memory_mode": "none" if cfg.sa_only else "architecture-defined",
            "tool_mode": "none" if not cfg.use_tools else "enabled",
            "oracle_mode": cfg.oracle_mode,
            "propagation_weight": cfg.propagation_weight,
            "generation": asdict(cfg.generation),
            "resource_budget": asdict(cfg.resource_budget),
            "call_max_attempts": cfg.call_max_attempts,
            "call_artifact_root": cfg.call_artifact_root,
            "runtime_behavior_hash": runtime_behavior_hash(),
        },
        "data": {
            "provider": cfg.data_provider,
            "data_version": cfg.data_version,
            "snapshot_hash": data_snapshot_hash(cfg.data_dir, cfg.data_version),
        },
        "episodes": {
            "profiles": list(cfg.profiles),
            "stress_scenarios": cfg.resolved_stress_scenarios(),
            "normal_periods": normal_periods,
            "rebalance": cfg.rebalance,
            "max_rebalances_per_window": cfg.max_rebalances_per_window,
            "initial_nav": cfg.initial_nav,
            "seed": cfg.seed,
        },
        "interventions": asdict(cfg.interventions),
        "qa": (
            {
                **asdict(cfg.qa),
                "dataset_snapshot_hash": data_snapshot_hash(
                    cfg.qa.dataset_path,
                    cfg.qa.template_version,
                ),
            }
            if cfg.run_qa
            else None
        ),
    }


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Return a stable digest for a freeze manifest."""
    return _sha256_text(_canonical(dict(manifest)))


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Atomically persist one manifest with its stable digest."""
    destination = Path(path)
    payload = dict(manifest)
    payload["manifest_digest"] = manifest_digest(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, destination)


def manifest_matches(path: str | Path, expected: Mapping[str, Any]) -> bool:
    """Return true only when a persisted manifest exactly matches the contract."""
    source = Path(path)
    if not source.exists():
        return False
    try:
        actual = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    actual_digest = actual.pop("manifest_digest", "")
    if actual_digest != manifest_digest(actual):
        return False
    return actual == dict(expected)


__all__ = [
    "FREEZE_MANIFEST_VERSION",
    "build_freeze_manifest",
    "data_snapshot_hash",
    "manifest_digest",
    "manifest_matches",
    "write_manifest",
]
