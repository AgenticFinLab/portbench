"""Tests for paired simulator intervention aggregation."""

from __future__ import annotations

import json

import pytest

from portbench.experiments.causal_analysis import (
    load_online_repair_records,
    summarize_causal_attribution,
)


def _record(model: str, window: str, decision_date: str, stage_id: str) -> dict:
    stage_order = ["S1", "S2", "S3", "S4", "S5"]
    start = stage_order.index(stage_id)
    return {
        "model": model,
        "window": window,
        "decision_date": decision_date,
        "stage_id": stage_id,
        "score_delta": {
            stage: 0.1 * (index - start + 1)
            for index, stage in enumerate(stage_order[start:], start=start)
        },
        "ceps_delta": 0.05 * (start + 1),
    }


def test_causal_summary_reports_stage_effects_and_fdr():
    records = []
    for model in ("a/m1", "b/m2"):
        for window in ("stress_a", "normal_b"):
            for decision_date in ("2024-01-01", "2024-02-01", "2024-03-01"):
                records.extend(
                    _record(model, window, decision_date, stage_id)
                    for stage_id in ("S1", "S2", "S3", "S4", "S5")
                )
    summary = summarize_causal_attribution(records, n_bootstrap=100, seed=7, block_size=2)
    assert summary["analysis_protocol"].startswith("paired simulator intervention")
    assert summary["influence_matrix"]["S1"]["S1"]["effect"] == pytest.approx(0.1)
    assert summary["influence_matrix"]["S1"]["S5"]["effect"] == pytest.approx(0.5)
    assert summary["influence_matrix"]["S2"].get("S1") is None
    assert summary["delta_ceps"]["S3"]["effect"] == pytest.approx(0.15)
    assert summary["influence_matrix"]["S1"]["S1"]["fdr_q_value"] is not None
    assert summary["decomposition"]["S1"]["downstream_propagation_effect"] == pytest.approx(0.35)
    assert summary["mechanism_gate"]["passed"] is True


def test_causal_loader_accepts_only_online_v4_sa_repairs(tmp_path):
    episode_dir = (
        tmp_path
        / "tencent"
        / "hy3-preview__SA"
        / "run"
        / "balanced"
        / "stress_2022"
        / "pipeline_logs"
        / "stress_2022"
        / "episodes"
    )
    episode_dir.mkdir(parents=True)
    payload = {
        "architecture_id": "SA",
        "schema_version": "pipeline-v4-sa-causal",
        "decision_date": "2024-02-01",
        "interventions": [
            {
                "stage_id": "S2",
                "operator": "repair",
                "repair_definition": "oracle-stage-replacement-v1",
                "mode": "online",
                "score_delta": {"S2": 0.1, "S3": 0.2},
                "ceps_delta": 0.1,
            },
            {
                "stage_id": "S3",
                "operator": "repair",
                "mode": "offline",
                "score_delta": {"S3": 0.2},
                "ceps_delta": 0.2,
            },
        ],
    }
    (episode_dir / "episode.json").write_text(json.dumps(payload), encoding="utf-8")
    records = load_online_repair_records(tmp_path)
    assert len(records) == 1
    assert records[0]["model"] == "tencent/hy3-preview__SA"
    assert records[0]["window"] == "stress_2022"
