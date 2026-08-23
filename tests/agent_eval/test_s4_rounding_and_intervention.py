
"""S4 four-decimal rounding must stay on the simplex; interventions must not abort."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from portbench.agent_eval.base import EpisodeResult, MarketSnapshot, S3Output, StageID
from portbench.agent_eval.intervention import run_episode_interventions
from portbench.agent_eval.s4_s5_stages import AgenticS5Stage, AgenticStageError
from portbench.agent_eval.stages import S4ExecutionSimulation, _round_executed_weights


def test_four_decimal_rounding_does_not_overflow_simplex():
    weights = {f"A{i:02d}": 0.0101 for i in range(96)}
    weights["NVDA"] = 0.0151
    weights["MSFT"] = 0.0121
    weights["QQQ"] = 0.0121
    assert sum(weights.values()) > 1.0 + 1e-3
    filled = _round_executed_weights(weights)
    assert sum(filled.values()) <= 1.0 + 1e-12


def test_legacy_s4_execute_keeps_filled_book_on_simplex():
    assets = [f"A{i:02d}" for i in range(80)]
    weights = {asset: 0.0126 for asset in assets}
    prices = {asset: pd.Series([100.0, 100.0]) for asset in assets}
    returns = {asset: pd.Series([0.001] * 20) for asset in assets}
    snapshot = MarketSnapshot(
        decision_date=date(2024, 2, 1),
        price_data=prices,
        return_data=returns,
        current_weights=dict(weights),
        portfolio_value=100_000.0,
    )
    result = S4ExecutionSimulation()._execute(
        S3Output(weights=dict(weights)), snapshot, slippage_rate=0.0
    )
    total = float(sum(result.executed_weights.values()))
    assert total <= 1.0 + 1e-3


def test_agentic_s5_still_rejects_overweight_book():
    stage = AgenticS5Stage(SimpleNamespace(complete=lambda prompt: "{}"))
    with pytest.raises(AgenticStageError, match="sum above one"):
        stage.run(
            weights={"SPY": 0.7, "BIL": 0.5},
            return_data={"SPY": pd.Series([0.01] * 20), "BIL": pd.Series([0.0] * 20)},
        )


def test_online_intervention_error_is_recorded_not_raised():
    snapshot = MarketSnapshot(
        decision_date=date(2024, 2, 1),
        price_data={"SPY": pd.Series([100.0])},
        return_data={"SPY": pd.Series([0.01] * 12)},
        current_weights={"SPY": 1.0},
        portfolio_value=100_000.0,
    )
    factual = EpisodeResult(decision_date=snapshot.decision_date)
    factual.stage_outputs = {
        StageID.S3_WEIGHT_OPTIMIZATION: S3Output(weights={"SPY": 1.0}),
    }

    class BoomPipeline:
        def run_episode(self, *args, **kwargs):
            raise RuntimeError("S4 executed weights must not sum above one (sum=1.003800)")

    records = run_episode_interventions(
        BoomPipeline(),
        snapshot,
        factual,
        stages=["S4"],
        operator="repair",
        mode="online",
    )
    assert len(records) == 1
    assert records[0]["stage_id"] == StageID.S4_EXECUTION_SIMULATION.value
    assert "1.003800" in records[0]["error"]
