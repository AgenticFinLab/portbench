"""Deterministic S4/S5 environment tests."""

from __future__ import annotations

import pandas as pd

from portbench.agent_eval.contracts import ExecutionPlan, OrderIntent, RiskControlDecision
from portbench.sandbox.execution import simulate_execution
from portbench.sandbox.risk_control import evaluate_risk


def _snap(prices=None):
    class S:
        price_data = prices or {
            "SPY": pd.Series([100.0, 101.0]),
            "TLT": pd.Series([50.0, 49.5]),
            "BIL": pd.Series([100.0, 100.0]),
        }
        return_data = {
            "SPY": pd.Series([0.01, -0.02, 0.015, -0.01] * 5),
            "TLT": pd.Series([0.002, 0.001, -0.003, 0.0] * 5),
            "BIL": pd.Series([0.0001] * 20),
        }

    return S()


def test_simulate_execution_deterministic():
    plan = ExecutionPlan(
        orders=[
            OrderIntent(asset="SPY", direction="buy", target_weight=0.5),
            OrderIntent(asset="TLT", direction="sell", target_weight=0.3),
            OrderIntent(asset="BIL", direction="hold", target_weight=0.2),
        ]
    )
    current = {"SPY": 0.4, "TLT": 0.4, "BIL": 0.2}
    snap = _snap()
    a = simulate_execution(plan, snap, current, nav=100_000.0)
    b = simulate_execution(plan, snap, current, nav=100_000.0)
    assert a.filled_weights == b.filled_weights
    assert a.turnover == b.turnover
    assert a.cost == b.cost
    assert a.implementation_shortfall == b.implementation_shortfall


def test_costs_increase_with_turnover():
    current = {"SPY": 0.5, "TLT": 0.3, "BIL": 0.2}
    snap = _snap()
    small = ExecutionPlan(
        orders=[OrderIntent(asset="SPY", direction="buy", target_weight=0.55),
                OrderIntent(asset="TLT", direction="sell", target_weight=0.25),
                OrderIntent(asset="BIL", direction="hold", target_weight=0.2)]
    )
    large = ExecutionPlan(
        orders=[OrderIntent(asset="SPY", direction="buy", target_weight=0.8),
                OrderIntent(asset="TLT", direction="sell", target_weight=0.1),
                OrderIntent(asset="BIL", direction="hold", target_weight=0.1)]
    )
    r_small = simulate_execution(small, snap, current, nav=100_000.0)
    r_large = simulate_execution(large, snap, current, nav=100_000.0)
    assert r_large.turnover > r_small.turnover
    assert r_large.cost > r_small.cost


def test_zero_slippage_rate():
    plan = ExecutionPlan(
        orders=[
            OrderIntent(asset="SPY", direction="buy", target_weight=0.6),
            OrderIntent(asset="TLT", direction="sell", target_weight=0.2),
            OrderIntent(asset="BIL", direction="hold", target_weight=0.2),
        ]
    )
    current = {"SPY": 0.4, "TLT": 0.4, "BIL": 0.2}
    snap = _snap()
    with_slip = simulate_execution(plan, snap, current, nav=100_000.0, slippage_rate=0.001)
    no_slip = simulate_execution(plan, snap, current, nav=100_000.0, slippage_rate=0.0)
    assert no_slip.cost < with_slip.cost
    # With zero slippage, cost is commission-only and still non-negative
    assert no_slip.cost >= 0.0
    for o in no_slip.metadata["orders"]:
        assert o["slippage"] == 0.0


def test_evaluate_risk_records_violations_and_corrective():
    weights = {"SPY": 0.9, "BIL": 0.1}
    snap = _snap()
    decision = RiskControlDecision(
        action="scale_down",
        alerts=["var_breach"],
        corrective_weights={"SPY": 0.5, "BIL": 0.5},
    )
    # Force drift/var by using tight limits
    result = evaluate_risk(
        decision,
        weights,
        snap.return_data,
        var_limit=0.0,  # any negative var breaches
        drawdown_limit=0.0,
        drift_limit=0.01,
    )
    assert "final_weights" in result.metadata
    assert result.metadata["final_weights"]["SPY"] == 0.5
    assert isinstance(result.constraint_violations, list)


def test_scale_down_changes_post_action_risk():
    weights = {"SPY": 1.0}
    returns = {
        "SPY": pd.Series([0.04, -0.08, 0.03, -0.06] * 8),
        "CASH": pd.Series([0.0] * 32),
    }
    decision = RiskControlDecision(action="scale_down", scale_factor=0.5)
    result = evaluate_risk(decision, weights, returns)
    pre = result.metadata["pre_action"]
    post = result.metadata["post_action"]
    assert result.metadata["final_weights"] == {"SPY": 0.5, "CASH": 0.5}
    assert abs(post["var"]) < abs(pre["var"])
    assert abs(post["drawdown"]) < abs(pre["drawdown"])
