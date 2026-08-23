"""Schema-aware CEPS rescore for pipeline-v3-collab episodes."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from portbench.agent_eval.base import MarketSnapshot
from portbench.agent_eval.stages import S4ExecutionSimulation
from portbench.experiments.rescore import _episode_schema_version, _rescore_episode


class _Builder:
    def __init__(self, snapshot: MarketSnapshot) -> None:
        self.snapshot = snapshot

    def build(self, dec_date, current_weights, nav, forward_days):
        return self.snapshot


def _snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        decision_date=date(2024, 2, 1),
        price_data={"SPY": pd.Series([100.0]), "BIL": pd.Series([100.0])},
        return_data={
            "SPY": pd.Series([0.01, -0.02, 0.005] * 12),
            "BIL": pd.Series([0.0001] * 36),
        },
        current_weights={"SPY": 0.5, "BIL": 0.5},
        portfolio_value=1_000_000.0,
        market_regime="sideways",
        asset_class_map={"SPY": "equities", "BIL": "cash"},
    )


def _v3_episode() -> dict:
    return {
        "decision_date": "2024-02-01",
        "schema_version": "pipeline-v3-collab",
        "stages": [
            {"stage_id": "S1", "score": 0.7, "parsed_output": {"asset_views": {"SPY": 0.2}}},
            {"stage_id": "S2", "score": 0.8, "parsed_output": {"signals": {"SPY": "buy"}}},
            {
                "stage_id": "S3",
                "score": 0.5,
                "parsed_output": {"weights": {"SPY": 0.6, "BIL": 0.4}},
            },
            {
                "stage_id": "S4",
                "score": 0.1,
                "parsed_output": {
                    "schema_version": "pipeline-v3-collab",
                    "executed_weights": {"SPY": 0.6, "BIL": 0.4},
                    "plan": {
                        "order_type": "market",
                        "orders": [
                            {
                                "asset": "SPY",
                                "direction": "buy",
                                "target_weight": 0.6,
                                "order_type": "market",
                            },
                            {
                                "asset": "BIL",
                                "direction": "sell",
                                "target_weight": 0.4,
                                "order_type": "market",
                            },
                        ],
                    },
                },
            },
            {
                "stage_id": "S5",
                "score": 0.1,
                "parsed_output": {
                    "schema_version": "pipeline-v3-collab",
                    "decision": {"action": "hold", "alerts": []},
                    "final_weights": {"SPY": 0.6, "BIL": 0.4},
                },
            },
        ],
    }


def test_episode_schema_detects_v3_from_plan_fields():
    episode = _v3_episode()
    episode["schema_version"] = "pipeline-v1"
    assert _episode_schema_version(episode) == "pipeline-v3-collab"


def test_v3_rescore_does_not_use_legacy_s4(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("legacy S4 score must not run for v3 episodes")

    monkeypatch.setattr(S4ExecutionSimulation, "score", boom)
    episode = _v3_episode()
    result = _rescore_episode(
        episode,
        _Builder(_snapshot()),
        forward_days=21,
        propagation_weight=0.1,
        current_weights={"SPY": 0.5, "BIL": 0.5},
    )
    assert result is not None
    ceps, per_stage = result
    assert ceps >= 0.0
    assert "S4" in per_stage
    assert episode["stages"][3]["score_plan"] == pytest.approx(per_stage["S4"])
    assert "score_outcome" in episode["stages"][3]
