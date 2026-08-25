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
    assert config.profiles == ["balanced"]
    assert config.resolved_stress_scenarios() == ["2022_crypto_collapse"]
    assert [p.label for p in config.normal_periods] == ["balanced_bull_2024"]
    assert config.experiment_tag == "agentic_full_hy3_v3"
    assert config.interventions.operator == "repair"
    assert config.interventions.mode == "online"
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


def test_pit_prefix_requires_an_s1_s3_prefix():
    with pytest.raises(ValueError, match="factual_pit_prefix_stages"):
        ExperimentConfig.from_dict(
            {
                "models": [
                    {"provider": "tencent", "model": "hy3-preview", "architecture_id": "SA"}
                ],
                "factual_pit_prefix_stages": ["S2"],
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


def test_sa_upgrade_configs_pin_single_agent_without_tools_or_memory():
    configs = {
        "pilot": ExperimentConfig.from_yaml("configs/experiments/sa_upgrade_pilot.yaml"),
        "full": ExperimentConfig.from_yaml("configs/experiments/sa_upgrade_full.yaml"),
        "causal": ExperimentConfig.from_yaml("configs/experiments/sa_upgrade_causal.yaml"),
        "qa": ExperimentConfig.from_yaml("configs/experiments/sa_upgrade_qa_v2.yaml"),
        "qa_validation": ExperimentConfig.from_yaml(
            "configs/experiments/sa_upgrade_qa_v2_val.yaml"
        ),
    }
    for config in configs.values():
        assert config.sa_only is True
        assert config.pipeline_schema_version == "pipeline-v4-sa-causal"
        assert config.use_tools is False
        assert config.workers_per_experiment == 1
        assert {model.architecture_id for model in config.models} == {"SA"}
        assert config.call_max_attempts == 3
        assert config.generation.temperature == 0.0
        assert config.generation.max_tokens == 8192
    assert len(configs["full"].models) == 10
    assert [period.label for period in configs["full"].normal_periods] == [
        "balanced_bull_2024"
    ]
    assert configs["full"].max_rebalances_per_window == 0
    assert configs["full"].factual_pit_prefix_stages == []
    assert configs["causal"].profiles == ["balanced"]
    assert [period.label for period in configs["causal"].normal_periods] == [
        "balanced_bull_2024"
    ]
    assert configs["causal"].max_rebalances_per_window == 0
    assert configs["causal"].factual_pit_prefix_stages == []
    assert configs["causal"].interventions.mode == "online"
    assert configs["causal"].interventions.closed_loop is False
    assert configs["pilot"].max_rebalances_per_window == 1
    assert configs["pilot"].factual_pit_prefix_stages == ["S1", "S2", "S3"]
    assert configs["pilot"].interventions.stages == ["S4", "S5"]
    assert configs["qa"].run_sandbox is False
    assert configs["qa"].qa.template_version == "constraint-v2"
    assert configs["qa"].qa.templates == ["T3", "T4"]
    assert configs["qa"].qa.freeze_manifest.endswith("constraint_v2_test_manifest.json")
    assert configs["qa_validation"].qa.split == "val"
    assert configs["qa_validation"].qa.max_pairs_per_template == 20
