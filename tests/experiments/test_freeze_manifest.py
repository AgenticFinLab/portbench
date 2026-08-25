"""Freeze manifest tests for SA-only restart safety."""

from __future__ import annotations

import portbench.experiments.freeze as freeze_module

from portbench.experiments.config import ExperimentConfig
from portbench.experiments.freeze import (
    build_freeze_manifest,
    manifest_matches,
    write_manifest,
)


def _config(
    data_dir: str, *, temperature: float = 0.0, max_rebalances_per_window: int = 0
) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "models": [
                {
                    "provider": "tencent",
                    "model": "hy3-preview",
                    "architecture_id": "SA",
                }
            ],
            "data_provider": "mock",
            "data_version": "fixture-v1",
            "data_dir": data_dir,
            "pipeline_schema_version": "pipeline-v4-sa-causal",
            "sa_only": True,
            "workers_per_experiment": 1,
            "use_tools": False,
            "max_rebalances_per_window": max_rebalances_per_window,
            "generation": {"temperature": temperature, "max_tokens": 8192},
        }
    )


def test_freeze_manifest_rejects_behavior_changes_without_code_commit(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "prices.csv").write_text("date,SPY\n2024-01-01,100\n", encoding="utf-8")
    config = _config(str(data_dir))
    manifest = build_freeze_manifest(config, config.models[0])
    manifest_path = tmp_path / "freeze_manifest.json"
    write_manifest(manifest_path, manifest)

    assert manifest_matches(manifest_path, manifest)
    changed_temperature = _config(str(data_dir), temperature=0.2)
    assert not manifest_matches(
        manifest_path,
        build_freeze_manifest(changed_temperature, changed_temperature.models[0]),
    )
    changed_limit = _config(str(data_dir), max_rebalances_per_window=1)
    assert not manifest_matches(
        manifest_path,
        build_freeze_manifest(changed_limit, changed_limit.models[0]),
    )


def test_freeze_manifest_rejects_data_snapshot_changes(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_path = data_dir / "prices.csv"
    data_path.write_text("date,SPY\n2024-01-01,100\n", encoding="utf-8")
    config = _config(str(data_dir))
    manifest_path = tmp_path / "freeze_manifest.json"
    write_manifest(manifest_path, build_freeze_manifest(config, config.models[0]))

    data_path.write_text("date,SPY\n2024-01-01,101\n", encoding="utf-8")
    assert not manifest_matches(
        manifest_path,
        build_freeze_manifest(config, config.models[0]),
    )


def test_freeze_manifest_rejects_runtime_behavior_changes(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "prices.csv").write_text("date,SPY\n2024-01-01,100\n", encoding="utf-8")
    config = _config(str(data_dir))
    manifest_path = tmp_path / "freeze_manifest.json"

    monkeypatch.setattr(freeze_module, "runtime_behavior_hash", lambda: "runtime-a")
    write_manifest(manifest_path, build_freeze_manifest(config, config.models[0]))

    monkeypatch.setattr(freeze_module, "runtime_behavior_hash", lambda: "runtime-b")
    assert not manifest_matches(
        manifest_path,
        build_freeze_manifest(config, config.models[0]),
    )
