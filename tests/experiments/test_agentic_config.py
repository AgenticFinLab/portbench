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
        "qa_decision": ExperimentConfig.from_yaml(
            "configs/experiments/sa_upgrade_qa_decision.yaml"
        ),
        "full_c_stress": ExperimentConfig.from_yaml(
            "configs/experiments/sa_upgrade_full_c_stress.yaml"
        ),
        "full_c_normal": ExperimentConfig.from_yaml(
            "configs/experiments/sa_upgrade_full_c_normal.yaml"
        ),
        "causal_c_stress": ExperimentConfig.from_yaml(
            "configs/experiments/sa_upgrade_causal_c_stress.yaml"
        ),
        "causal_c_normal": ExperimentConfig.from_yaml(
            "configs/experiments/sa_upgrade_causal_c_normal.yaml"
        ),
    }
    for name, config in configs.items():
        assert config.sa_only is True
        assert config.pipeline_schema_version == "pipeline-v4-sa-causal"
        assert config.use_tools is False
        assert config.workers_per_experiment == 1
        assert {model.architecture_id for model in config.models} == {"SA"}
        assert config.call_max_attempts == 3
        assert config.generation.temperature == 0.0
        expected_max_tokens = 8192 if name == "pilot" else 4096
        assert config.generation.max_tokens == expected_max_tokens
    assert configs["pilot"].max_rebalances_per_window == 1
    assert configs["pilot"].factual_pit_prefix_stages == ["S1", "S2", "S3"]
    assert configs["pilot"].interventions.stages == ["S4", "S5"]
    assert configs["qa_decision"].qa.template_version == "constraint-decision-v2"
    assert configs["qa_decision"].qa.split == "test"
    assert configs["qa_decision"].qa.max_pairs_per_template == 50
    assert configs["qa_decision"].qa.freeze_manifest.endswith(
        "constraint_decision_v2_test_manifest.json"
    )
    assert len(configs["full_c_stress"].models) == 10
    assert configs["full_c_stress"].rebalance == "monthly"
    assert not configs["full_c_stress"].run_normal
    assert configs["full_c_stress"].legacy_stage_reuse_root.endswith(
        "sa_upgrade_full_c_stress_v6"
    )
    assert configs["full_c_normal"].rebalance == "monthly"
    assert configs["full_c_normal"].stress_scenarios == []
    assert configs["causal_c_stress"].profiles == ["balanced"]
    assert configs["causal_c_stress"].rebalance == "monthly"
    deepseek_stress = [
        model
        for model in configs["causal_c_stress"].models
        if model.model in {"deepseek-v4-pro", "deepseek-v4-flash"}
    ]
    assert all(model.max_tokens is None for model in deepseek_stress)
    assert all(model.intervention_max_tokens == 16384 for model in deepseek_stress)
    assert configs["causal_c_normal"].interventions.mode == "online"
