"""Tests for strict historical S1-S3 stage-reuse admission."""

from __future__ import annotations

import json
import hashlib

from portbench.experiments.config import ExperimentConfig
from portbench.experiments.freeze import data_snapshot_hash
from portbench.experiments.stage_reuse_audit import audit_stage_reuse


def _config(data_dir: str) -> ExperimentConfig:
    """Build the smallest SA-only target contract for an audit test."""
    return ExperimentConfig.from_dict(
        {
            "data_dir": data_dir,
            "data_version": "processed-v1",
            "pipeline_schema_version": "pipeline-v4-sa-causal",
            "sa_only": True,
            "use_tools": False,
            "workers_per_experiment": 1,
            "models": [{"provider": "demo", "model": "model", "architecture_id": "SA"}],
            "generation": {"temperature": 0.0, "max_tokens": 8192},
        }
    )


def _episode(contract: dict | None) -> dict:
    """Build one archived S1-S3 record with all replay payloads present."""
    episode = {
        "decision_date": "2024-02-01",
        "stages": [
            {
                "stage_id": stage,
                "prompt": f"{stage} prompt",
                "raw_response": f"{stage} response",
                "parsed_output": {"value": stage},
            }
            for stage in ("S1", "S2", "S3")
        ],
    }
    if contract is not None:
        episode["provenance"] = {"stage_reuse_contract": contract}
    return episode


def _sha256_text(value: str) -> str:
    """Return the source prompt digest used by the archive contract fixture."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_episode(root, episode: dict) -> None:
    """Write one legacy-shaped episode at the documented source location."""
    path = root / "monthly" / "demo" / "model" / "run" / "balanced" / "normal_2024"
    path = path / "pipeline_logs" / "log" / "episodes"
    path.mkdir(parents=True)
    (path / "2024-02-01_0001.json").write_text(json.dumps(episode), encoding="utf-8")


def test_audit_rejects_legacy_episode_without_source_contract(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_root = tmp_path / "source"
    _write_episode(source_root, _episode(contract=None))

    report = audit_stage_reuse(source_root, _config(str(data_dir)))

    assert report["provider_calls"] == 0
    assert not report["passed"]
    assert report["eligible_episodes"] == 0
    assert report["reason_counts"] == {"missing provenance.stage_reuse_contract": 1}


def test_audit_accepts_matching_source_contract(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = _config(str(data_dir))
    contract = {
        "architecture_id": "SA",
        "memory_mode": "none",
        "tool_mode": "none",
        "pipeline_schema_version": config.pipeline_schema_version,
        "data_version": config.data_version,
        "data_snapshot_hash": data_snapshot_hash(config.data_dir, config.data_version),
        "generation": {"temperature": 0.0, "max_tokens": 8192},
        "provider": "demo",
        "model": "model",
        "model_revision": "2026-08-25",
        "stages": {
            stage: {
                "call_key": f"{stage}-key",
                "prompt_hash": _sha256_text(f"{stage} prompt"),
                "raw_response_hash": _sha256_text(f"{stage} response"),
                "visible_input_hash": f"{stage}-input",
                "response_schema_hash": f"{stage}-schema",
            }
            for stage in ("S1", "S2", "S3")
        },
    }
    source_root = tmp_path / "source"
    _write_episode(source_root, _episode(contract=contract))

    report = audit_stage_reuse(source_root, config)

    assert report["passed"]
    assert report["eligible_episodes"] == 1
