"""Offline release-gate tests."""

from __future__ import annotations

import json

from portbench.experiments.config import ExperimentConfig
from portbench.experiments.gates import _check_episode_logs, evaluate_qa_validation
from portbench.qa_eval.paths import qa_template_dir


def test_qa_validation_gate_requires_complete_and_variable_scores(tmp_path):
    config = ExperimentConfig.from_dict(
        {
            "models": [
                {"provider": "tencent", "model": "hy3-preview", "architecture_id": "SA"},
                {"provider": "dashscope", "model": "qwen3.7-max", "architecture_id": "SA"},
            ],
            "pipeline_schema_version": "pipeline-v4-sa-causal",
            "sa_only": True,
            "workers_per_experiment": 1,
            "use_tools": False,
            "output_root": str(tmp_path),
            "run_qa": True,
            "qa": {
                "split": "val",
                "templates": ["T3", "T4"],
                "max_pairs_per_template": 20,
                "template_version": "constraint-v2",
            },
        }
    )
    for spec in config.models:
        for template in config.qa.templates:
            path = qa_template_dir(config.output_root, spec.provider, spec.model, template)
            path.mkdir(parents=True)
            records = [
                {
                    "qa_id": f"{template}-{index}",
                    "score": 0.6 if index % 2 else 0.8,
                    "template_version": "constraint-v2",
                }
                for index in range(20)
            ]
            (path / "results.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
    verdict = evaluate_qa_validation(config)
    assert verdict["passed"] is True


def test_qa_validation_gate_rejects_near_perfect_locked_test_pilot(tmp_path):
    config = ExperimentConfig.from_dict(
        {
            "models": [{"provider": "tencent", "model": "hy3-preview", "architecture_id": "SA"}],
            "pipeline_schema_version": "pipeline-v4-sa-causal",
            "sa_only": True,
            "workers_per_experiment": 1,
            "use_tools": False,
            "output_root": str(tmp_path),
            "run_qa": True,
            "qa": {
                "split": "val",
                "templates": ["T3"],
                "max_pairs_per_template": 20,
                "template_version": "constraint-v2",
            },
        }
    )
    path = qa_template_dir(config.output_root, "tencent", "hy3-preview", "T3")
    path.mkdir(parents=True)
    (path / "results.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "qa_id": f"T3-{index}",
                    "score": 1.0,
                    "template_version": "constraint-v2",
                }
            )
            + "\n"
            for index in range(20)
        ),
        encoding="utf-8",
    )
    verdict = evaluate_qa_validation(config)
    assert verdict["passed"] is False
    assert "discriminative-score gate failed" in verdict["findings"][0]


def test_pilot_gate_requires_the_declared_pit_prefix_provenance(tmp_path):
    scenario_dir = tmp_path / "stress_2022_crypto_collapse"
    episode_dir = scenario_dir / "pipeline_logs" / "run" / "episodes"
    episode_dir.mkdir(parents=True)
    (scenario_dir / "backtest_result.json").write_text(
        json.dumps({"n_rebalances": 1}),
        encoding="utf-8",
    )
    episode = {
        "architecture_id": "SA",
        "schema_version": "pipeline-v4-sa-causal",
        "provenance": {
            "reused_stage_sources": {
                "S1": "pit-repair-v2",
                "S2": "pit-repair-v2",
                "S3": "pit-repair-v2",
            }
        },
        "interventions": [
            {"stage_id": "S4", "operator": "repair", "mode": "online"},
            {"stage_id": "S5", "operator": "repair", "mode": "online"},
        ],
        "stages": [
            {"stage_id": "S4", "prompt": "", "parsed_output": {}, "error": ""},
            {"stage_id": "S5", "prompt": "", "parsed_output": {}, "error": ""},
        ],
    }
    episode_path = episode_dir / "2022-05-02_0001.json"
    episode_path.write_text(json.dumps(episode), encoding="utf-8")

    assert _check_episode_logs(
        scenario_dir,
        {"S4", "S5"},
        {"S1", "S2", "S3"},
        "2022_crypto_collapse",
    ) == []

    episode["provenance"]["reused_stage_sources"]["S1"] = "ground-truth"
    episode_path.write_text(json.dumps(episode), encoding="utf-8")
    findings = _check_episode_logs(
        scenario_dir,
        {"S4", "S5"},
        {"S1", "S2", "S3"},
        "2022_crypto_collapse",
    )
    assert any("invalid reused-stage provenance" in finding for finding in findings)
