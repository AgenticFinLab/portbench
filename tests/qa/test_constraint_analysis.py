"""Tests for frozen constraint-v2 QA statistics and paper exports."""

from __future__ import annotations

import json

from portbench.experiments.paper_export import export_paper_artifacts
from portbench.qa_builder.freeze_constraint_dataset import freeze_constraint_v2_test
from portbench.qa_eval.constraint_analysis import (
    load_constraint_v2_records,
    summarize_constraint_v2,
    write_constraint_v2_artifacts,
)


def test_constraint_analysis_reads_valid_results_and_exports_bootstrap(tmp_path):
    result_path = tmp_path / "qa_eval" / "provider" / "model" / "T3" / "results.jsonl"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "qa_id": f"T3-{index}",
                    "score": score,
                    "template_version": "constraint-v2",
                }
            )
            for index, score in enumerate((0.2, 0.8, 1.0))
        )
        + "\n",
        encoding="utf-8",
    )
    ignored_path = tmp_path / "qa_eval" / "provider" / "model" / "T4" / "results.jsonl"
    ignored_path.parent.mkdir(parents=True)
    ignored_path.write_text(
        json.dumps({"qa_id": "bad", "score": 1.0, "template_version": "legacy"}) + "\n",
        encoding="utf-8",
    )

    summary = summarize_constraint_v2(load_constraint_v2_records(tmp_path), n_bootstrap=100, seed=7)
    assert summary["model_template"]["provider/model"]["T3"]["n"] == 3
    assert summary["model_template"]["provider/model"]["T4"]["n"] == 0
    summary_path = write_constraint_v2_artifacts(summary, tmp_path / "stats")
    assert summary_path.exists()

    causal_path = tmp_path / "causal.json"
    causal_path.write_text(
        json.dumps({"delta_ceps": {"S1": {"effect": 0.1, "ci95": [0.0, 0.2], "n_raw": 3}}}),
        encoding="utf-8",
    )
    manifest = export_paper_artifacts(
        causal_summary_path=causal_path,
        qa_summary_path=summary_path,
        output_dir=tmp_path / "paper",
    )
    assert manifest.exists()
    assert "S1 & 0.1000" in (tmp_path / "paper" / "causal_delta_ceps_rows.tex").read_text(encoding="utf-8")
    assert (tmp_path / "paper" / "qa_constraint_v2_rows.tex").exists()


def test_constraint_test_manifest_freezes_and_detects_prompt_changes(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pairs = []
    for template in ("T3", "T4"):
        for index in range(2):
            pairs.append(
                {
                    "id": f"{template}-{index}",
                    "template": template,
                    "question": f"original {index}",
                    "metadata": {
                        "template_version": "constraint-v2",
                        "generator_version": "fixture-v1",
                    },
                }
            )
    test_path = dataset / "test.jsonl"
    test_path.write_text(
        "".join(json.dumps(pair) + "\n" for pair in pairs), encoding="utf-8"
    )
    manifest = freeze_constraint_v2_test(dataset, max_pairs_per_template=2)
    from portbench.qa_eval.evaluator import _apply_freeze_manifest

    selected = _apply_freeze_manifest(pairs, str(manifest))
    assert [pair["id"] for pair in selected] == ["T3-0", "T3-1", "T4-0", "T4-1"]
    pairs[0]["question"] = "changed"
    try:
        _apply_freeze_manifest(pairs, str(manifest))
    except ValueError as exc:
        assert "changed after manifest creation" in str(exc)
    else:
        raise AssertionError("changed frozen prompt must be rejected")
