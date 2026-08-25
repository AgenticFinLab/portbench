"""Tests for provider-scoped formal experiment invocations."""

import pytest

from portbench.experiments.__main__ import _filter_models_by_provider
from portbench.experiments.config import ExperimentConfig


def _config() -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "models": [
                {"provider": "tencent", "model": "hy3-preview", "architecture_id": "SA"},
                {"provider": "ark", "model": "doubao", "architecture_id": "SA"},
                {"provider": "dashscope", "model": "qwen", "architecture_id": "SA"},
            ],
            "pipeline_schema_version": "pipeline-v4-sa-causal",
            "sa_only": True,
            "use_tools": False,
            "workers_per_experiment": 1,
        }
    )


def test_provider_filter_preserves_only_requested_configured_models():
    config = _config()

    selected = _filter_models_by_provider(config, "tencent,dashscope")

    assert selected == ("tencent", "dashscope")
    assert [(spec.provider, spec.model) for spec in config.models] == [
        ("tencent", "hy3-preview"),
        ("dashscope", "qwen"),
    ]


def test_provider_filter_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unconfigured providers"):
        _filter_models_by_provider(_config(), "deepseek")
