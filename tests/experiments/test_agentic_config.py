"""Architecture experiment configuration invariants."""

from __future__ import annotations

import pytest

from portbench.experiments.config import ExperimentConfig
from portbench.experiments.runner import _cache_contract_matches, _current_code_commit


ARCHITECTURE_IDS = {"SA", "SA-T", "SA-M", "SA-MT", "MA", "MA-T", "MA-M", "MA-MT"}
PRODUCTION_LLM = {
    ("tencent", "hy3-preview"),
    ("deepseek", "deepseek-v4-pro"),
    ("deepseek", "deepseek-v4-flash"),
    ("ark", "doubao-seed-2-0-pro-260215"),
    ("ark", "doubao-seed-2-0-lite-260215"),
    ("dashscope", "qwen3.7-max"),
    ("dashscope", "qwen3.6-plus"),
    ("dashscope", "kimi-k2.6"),
    ("dashscope", "glm-5.1"),
    ("dashscope", "qwen3.6-35b-a3b"),
}


def test_full_hy3_config_covers_eight_cells_and_baselines():
    config = ExperimentConfig.from_yaml("configs/experiments/full_hy3.yaml")
    llm = [m for m in config.models if m.provider]
    baselines = [m.baseline for m in config.models if m.baseline]
    assert {m.architecture_id for m in llm} == ARCHITECTURE_IDS
    assert {(m.provider, m.model) for m in llm} == {("tencent", "hy3-preview")}
    assert len(llm) == 8
    assert baselines == [
        "equal_weight",
        "risk_parity",
        "sixty_forty",
        "cov_risk_parity",
        "min_variance",
        "black_litterman",
    ]
    assert config.workers_per_experiment == 1
    assert config.oracle_mode == "lookback"
    assert config.profiles == ["conservative", "balanced", "aggressive"]
    assert config.resolved_stress_scenarios() == [
        "2015_china_shock",
        "2020_covid_flash_crash",
        "2022_crypto_collapse",
    ]
    assert [p.label for p in config.normal_periods] == [
        "balanced_bull_2024",
        "conservative_pressure_2022h2",
    ]
    assert config.experiment_tag == "agentic_full_hy3_v1"
    assert config.interventions.operator == "repair"
    assert config.interventions.mode == "offline"
    assert config.interventions.closed_loop is True
    assert config.interventions.enabled is True


def test_full_config_covers_architecture_matrix():
    config = ExperimentConfig.from_yaml("configs/experiments/full.yaml")
    llm = [m for m in config.models if m.provider]
    baselines = [m.baseline for m in config.models if m.baseline]
    assert {m.architecture_id for m in llm} == ARCHITECTURE_IDS
    assert {(m.provider, m.model) for m in llm} == PRODUCTION_LLM
    assert len(llm) == len(PRODUCTION_LLM) * len(ARCHITECTURE_IDS)
    assert baselines == [
        "equal_weight",
        "risk_parity",
        "sixty_forty",
        "cov_risk_parity",
        "min_variance",
        "black_litterman",
    ]
    assert config.workers_per_experiment == 1
    assert config.oracle_mode == "lookback"
    assert config.profiles == ["conservative", "balanced", "aggressive"]
    assert config.resolved_stress_scenarios() == [
        "2015_china_shock",
        "2020_covid_flash_crash",
        "2022_crypto_collapse",
    ]
    assert [p.label for p in config.normal_periods] == [
        "balanced_bull_2024",
        "conservative_pressure_2022h2",
    ]
    assert config.experiment_tag == "agentic_full_v1"
    assert config.output_root.endswith("agentic_full_v1")
    assert config.interventions.operator == "repair"
    assert config.interventions.mode == "offline"
    assert config.interventions.closed_loop is True


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
    config = ExperimentConfig.from_yaml("configs/experiments/full_hy3.yaml")
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


def test_stress_scenario_uses_yaml_rebalance():
    import inspect

    from portbench.experiments.runner import _run_one_scenario

    source = inspect.getsource(_run_one_scenario)
    assert 'rebalance_freq="weekly"' not in source
    assert "rebalance_freq=cfg.rebalance" in source
