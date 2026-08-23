"""Architecture experiment configuration invariants."""

from __future__ import annotations

import pytest

from portbench.experiments.config import ExperimentConfig
from portbench.experiments.runner import _cache_contract_matches, _current_code_commit


def test_large_scale_configs_cover_eight_cells():
    for path in (
        "configs/experiments/agentic_large_scale_hy3_balanced.yaml",
        "configs/experiments/agentic_large_scale_hy3_conservative.yaml",
    ):
        config = ExperimentConfig.from_yaml(path)
        assert len(config.models) == 8
        assert len({model.architecture_id for model in config.models}) == 8
        assert config.workers_per_experiment == 1
        assert config.oracle_mode == "lookback"


def test_perturb_operator_loads_from_yaml():
    config = ExperimentConfig.from_dict(
        {
            "models": [{"provider": "tencent", "model": "hy3-preview"}],
            "interventions": {"enabled": True, "operator": "perturb", "mode": "offline"},
        }
    )
    assert config.interventions.operator == "perturb"
    assert config.interventions.closed_loop is True


def test_parallel_scenarios_rejected_for_architecture_usage_accounting():
    with pytest.raises(ValueError, match="workers_per_experiment=1"):
        ExperimentConfig.from_dict(
            {
                "models": [
                    {"provider": "tencent", "model": "hy3-preview", "architecture_id": "SA"}
                ],
                "workers_per_experiment": 2,
            }
        )


def test_closed_loop_cache_requires_current_contract():
    config = ExperimentConfig.from_yaml(
        "configs/experiments/agentic_large_scale_hy3_balanced.yaml"
    )
    spec = config.models[0]
    artifact = {
        "architecture_id": "SA",
        "result_protocol": "closed-loop",
        "schema_version": "pipeline-v3-collab",
        "data_version": config.data_version,
        "code_commit": _current_code_commit(),
    }
    assert _cache_contract_matches(artifact, config, spec)
    for field, changed in (
        ("architecture_id", "MA"),
        ("result_protocol", "step-replay"),
        ("schema_version", "pipeline-v1"),
        ("data_version", "old-data"),
        ("code_commit", "old-code"),
    ):
        mutant = dict(artifact)
        mutant[field] = changed
        assert not _cache_contract_matches(mutant, config, spec)
