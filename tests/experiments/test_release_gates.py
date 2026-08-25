"""Offline release-gate tests."""

from __future__ import annotations

import json

from portbench.experiments.config import ExperimentConfig
from portbench.experiments.gates import evaluate_qa_validation
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


def test_qa_validation_gate_rejects_near_perfect_scores(tmp_path):
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
