"""Legacy S4/S5 bridge round-trip tests."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from portbench.agent_eval.contracts import S4S5_SCHEMA_DETERMINISTIC
from portbench.agent_eval.s4_s5_bridge import (
    legacy_s4_to_plan_and_result,
    legacy_s5_to_decision_and_eval,
    run_s4_deterministic_from_weights,
)
from portbench.sandbox.execution import simulate_execution


def test_legacy_s4_bridge_schema_and_roundtrip():
    current = {"SPY": 0.4, "TLT": 0.4, "BIL": 0.2}
    target = {"SPY": 0.55, "TLT": 0.25, "BIL": 0.2}
    snap = SimpleNamespace(
        current_weights=current,
        portfolio_value=100_000.0,
        price_data={
            "SPY": pd.Series([100.0, 101.0]),
            "TLT": pd.Series([50.0, 49.0]),
            "BIL": pd.Series([100.0, 100.0]),
        },
    )
    plan0, result0 = run_s4_deterministic_from_weights(
        target, snap, current, nav=100_000.0
    )
    # Fake legacy S4Output-like object from env result
    s4_like = SimpleNamespace(
        executed_weights=result0.filled_weights,
        total_cost=result0.cost,
        turnover=result0.turnover,
        orders=[],
    )
    plan, result = legacy_s4_to_plan_and_result(s4_like, snap)
    assert plan.metadata["schema_version"] == S4S5_SCHEMA_DETERMINISTIC
    assert result.metadata["schema_version"] == S4S5_SCHEMA_DETERMINISTIC

    # Re-simulating reconstructed plan should reproduce turnover/cost structure
    again = simulate_execution(plan, snap, current, nav=100_000.0)
    assert abs(again.turnover - result0.turnover) < 0.05
    # Costs should be in the same ballpark (allowing cash-drag rounding)
    assert again.cost >= 0.0
    if result0.cost > 0:
        assert abs(again.cost - result0.cost) / result0.cost < 0.25


def test_legacy_s5_bridge_schema():
    alert = SimpleNamespace(
        metric="drawdown", value=-0.12, threshold=-0.10, severity="critical", action="rebalance"
    )
    s5 = SimpleNamespace(
        portfolio_var=-0.03,
        portfolio_drawdown=-0.12,
        weight_drift=0.08,
        alerts=[alert],
        rebalance_needed=True,
    )
    decision, ev = legacy_s5_to_decision_and_eval(s5)
    assert decision.metadata["schema_version"] == S4S5_SCHEMA_DETERMINISTIC
    assert ev.metadata["schema_version"] == S4S5_SCHEMA_DETERMINISTIC
    assert decision.action == "rebalance"
    assert "drawdown" in decision.alerts
