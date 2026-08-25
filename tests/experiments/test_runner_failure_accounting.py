"""Batch status must report a model with failed profiles as failed."""

from __future__ import annotations

import threading

import pytest

import portbench.experiments.runner as runner
from portbench.experiments.config import ExperimentConfig


def test_model_run_returns_failed_profile_names(tmp_path, monkeypatch):
    config = ExperimentConfig.from_dict(
        {
            "models": [
                {
                    "provider": "tencent",
                    "model": "hy3-preview",
                    "architecture_id": "SA",
                }
            ],
            "profiles": ["balanced"],
            "data_provider": "mock",
            "pipeline_schema_version": "pipeline-v4-sa-causal",
            "sa_only": True,
            "workers_per_experiment": 1,
            "use_tools": False,
            "output_root": str(tmp_path / "experiments"),
        }
    )
    spec = config.models[0]

    monkeypatch.setattr(runner, "_build_strategy", lambda *args, **kwargs: object())

    def _fail_profile(*args, **kwargs):
        raise RuntimeError("profile failed")

    monkeypatch.setattr(runner, "_run_profile", _fail_profile)

    _, failed_profiles = runner._run_one_model(
        config,
        spec,
        provider=object(),
        asset_class_map={},
        prov_name="tencent",
        model_name="hy3-preview__SA",
        timestamp="20260825_000000",
        errors_lock=threading.Lock(),
    )

    assert failed_profiles == ["balanced"]
