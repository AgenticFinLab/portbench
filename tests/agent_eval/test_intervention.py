"""Intervention operators on toy episode; future data blocked."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from portbench.agent_eval.base import EpisodeResult, MarketSnapshot, StageID
from portbench.agent_eval.contracts import ProvenanceSource, ResultProvenance
from portbench.agent_eval.intervention import (
    apply_perturb,
    apply_repair,
    compute_descriptive_delta,
    default_perturb,
    run_episode_interventions,
    run_offline_stage_intervention,
    store_intervention_result,
)
from portbench.agent_eval.replay_adapter import INTERVENTION_NAMESPACE, ReplayAdapter
from portbench.agent_eval.stages import (
    S1MarketInterpretation,
    S2SignalGeneration,
    S3WeightOptimization,
    S4ExecutionSimulation,
    S5RiskMonitoring,
)


LOOKBACK = {"SPY": [0.01, -0.02, 0.005], "TLT": [0.0, 0.001, -0.001]}


def test_repair_blocks_future():
    with pytest.raises(PermissionError):
        apply_repair("S1", {"lookback_returns": LOOKBACK, "future_return_data": {"SPY": [1.0]}})


def test_perturb_and_delta_on_toy():
    factual = apply_repair("S1", lookback_returns=LOOKBACK)

    def bump(output):
        out = dict(output)
        views = dict(out["asset_views"])
        views["SPY"] *= 1.5
        out["asset_views"] = views
        return out

    intervened = apply_perturb("S1", factual, perturb_fn=bump)
    delta = compute_descriptive_delta(
        {"view": factual["asset_views"]["SPY"]},
        {"view": intervened["asset_views"]["SPY"]},
        keys=["view"],
    )
    assert "view" in delta
    assert isinstance(delta["view"], float)


def test_default_perturb_clips_and_renormalizes():
    s1 = default_perturb("S1", {"asset_views": {"SPY": 1.0, "TLT": -1.0}})
    assert s1["asset_views"]["SPY"] == 1.0
    assert s1["asset_views"]["TLT"] == -1.0
    s2 = default_perturb("S2", {"strengths": {"SPY": 0.95}})
    assert 0.0 <= s2["strengths"]["SPY"] <= 1.0
    s3 = default_perturb("S3", {"weights": {"SPY": 0.6, "TLT": 0.4}})
    assert abs(sum(s3["weights"].values()) - 1.0) < 1e-9
    s5 = default_perturb("S5", {"decision": {"action": "hold"}})
    assert s5["decision"]["action"] == "scale_down"
    assert s5["decision"]["scale_factor"] == 0.9


def test_store_uses_intervention_namespace():
    adapter = ReplayAdapter()
    prov = ResultProvenance(source=ProvenanceSource.CURRENT_CACHE.value, cache_key="k1")
    store_intervention_result(adapter, "k1", {"ok": True}, prov)
    assert adapter.lookup("k1", namespace=f"{INTERVENTION_NAMESPACE}__default") is not None
    assert adapter.lookup("k1") is None


def _snapshot() -> MarketSnapshot:
    returns = {
        "SPY": pd.Series([0.01, -0.02, 0.005] * 12),
        "BIL": pd.Series([0.0001] * 36),
    }
    return MarketSnapshot(
        decision_date=date(2024, 2, 1),
        price_data={"SPY": pd.Series([100.0]), "BIL": pd.Series([100.0])},
        return_data=returns,
        current_weights={"SPY": 0.5, "BIL": 0.5},
        portfolio_value=1_000_000.0,
        market_regime="sideways",
    )


def _factual_from_gt(snapshot: MarketSnapshot) -> EpisodeResult:
    stages = [
        S1MarketInterpretation(),
        S2SignalGeneration(),
        S3WeightOptimization(oracle_mode="lookback"),
        S4ExecutionSimulation(),
        S5RiskMonitoring(),
    ]
    outputs = {stage.stage_id: stage.compute_ground_truth(snapshot) for stage in stages}
    scores = {sid: 1.0 for sid in outputs}
    return EpisodeResult(
        decision_date=snapshot.decision_date,
        stage_outputs=outputs,
        gt_outputs=dict(outputs),
        stage_scores=scores,
        schema_version="pipeline-v1",
    )


def test_offline_repair_and_perturb_write_ceps_delta():
    snapshot = _snapshot()
    factual = _factual_from_gt(snapshot)
    repaired = run_offline_stage_intervention(snapshot, factual, "S3", operator="repair")
    perturbed = run_offline_stage_intervention(snapshot, factual, "S3", operator="perturb")
    assert repaired["mode"] == "offline"
    assert "score_delta" in repaired
    assert "ceps_delta" in repaired
    assert perturbed["operator"] == "perturb"
    assert "S3" in perturbed["score_delta"]


def test_run_episode_interventions_offline_does_not_need_pipeline():
    snapshot = _snapshot()
    factual = _factual_from_gt(snapshot)
    records = run_episode_interventions(
        pipeline=None,
        snapshot=snapshot,
        factual=factual,
        stages=["S1", "S2"],
        operator="repair",
        mode="offline",
    )
    assert len(records) == 2
    assert all(item["mode"] == "offline" for item in records)
    assert StageID.S1_MARKET_INTERPRETATION in factual.stage_outputs
