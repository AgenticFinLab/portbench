"""Deterministic S4/S5 environment tests."""

from __future__ import annotations

import pandas as pd
import pytest

from portbench.agent_eval.contracts import ExecutionPlan, OrderIntent, RiskControlDecision
from portbench.sandbox.execution import resolve_target_weights, simulate_execution
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


def test_target_weight_book_exits_unlisted_holdings():
    plan = ExecutionPlan(
        orders=[
            OrderIntent(asset="BIL", direction="buy", target_weight=0.5),
            OrderIntent(asset="SHV", direction="buy", target_weight=0.5),
        ]
    )
    current = {"SPY": 0.33, "BIL": 0.33, "CSHI": 0.34}
    snap = _snap()
    snap.price_data["SHV"] = pd.Series([100.0, 100.0])
    snap.price_data["CSHI"] = pd.Series([100.0, 100.0])
    result = simulate_execution(plan, snap, current, nav=100_000.0)
    targets = result.metadata["target_weights"]
    assert abs(sum(targets.values()) - 1.0) < 1e-6
    assert targets.get("CSHI", 0.0) == 0.0
    assert targets.get("SPY", 0.0) == 0.0
    assert abs(targets["BIL"] - 0.5) < 1e-6
    assert abs(targets["SHV"] - 0.5) < 1e-6


def test_delta_orders_keep_omitted_holdings():
    plan = ExecutionPlan(
        orders=[OrderIntent(asset="SPY", direction="buy", delta_weight=0.1)]
    )
    current = {"SPY": 0.4, "TLT": 0.4, "BIL": 0.2}
    targets = resolve_target_weights(plan, current)
    assert abs(targets["SPY"] - 0.5) < 1e-9
    assert abs(targets["TLT"] - 0.4) < 1e-9
    assert abs(targets["BIL"] - 0.2) < 1e-9


def test_off_simplex_targets_are_projected():
    plan = ExecutionPlan(
        orders=[
            OrderIntent(asset="SPY", direction="buy", target_weight=0.6),
            OrderIntent(asset="BIL", direction="hold", target_weight=0.6),
        ]
    )
    current = {"SPY": 0.5, "BIL": 0.5}
    result = simulate_execution(plan, _snap(), current, nav=100_000.0)
    targets = result.metadata["target_weights"]
    assert abs(sum(targets.values()) - 1.0) < 1e-6
    assert result.metadata.get("renormalized") is True
    assert abs(targets["SPY"] - 0.5) < 1e-6
    assert abs(targets["BIL"] - 0.5) < 1e-6
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


def test_unfilled_limit_sell_does_not_over_allocate():
    """A high-urgency limit sell that cannot fill must not leave leftover mass on filled buys."""
    current = {"SPY": 0.45, "TLT": 0.45, "^VIX": 0.10}
    plan = ExecutionPlan(
        order_type="limit",
        urgency="normal",
        slip_limit=0.001,
        orders=[
            OrderIntent(
                asset="SPY",
                direction="buy",
                target_weight=0.5,
                order_type="limit",
                urgency="normal",
                slip_limit=0.001,
            ),
            OrderIntent(
                asset="TLT",
                direction="buy",
                target_weight=0.5,
                order_type="limit",
                urgency="normal",
                slip_limit=0.001,
            ),
            OrderIntent(
                asset="^VIX",
                direction="sell",
                target_weight=0.0,
                order_type="limit",
                urgency="high",
                slip_limit=0.001,
            ),
        ],
    )
    snap = _snap()
    snap.price_data["^VIX"] = pd.Series([20.0, 21.0])
    result = simulate_execution(plan, snap, current, nav=100_000.0)
    filled = result.filled_weights
    assert sum(filled.values()) <= 1.0 + 1e-3
    assert abs(filled["^VIX"] - 0.10) < 1e-6
    assert filled["SPY"] <= 0.45 + 1e-6
    assert filled["TLT"] <= 0.45 + 1e-6
    statuses = {row["asset"]: row["status"] for row in result.metadata["orders"]}
    assert statuses["^VIX"] == "unfilled_limit"
    evaluate_risk(
        RiskControlDecision(action="hold"),
        filled,
        snap.return_data,
    )


def test_unfilled_limit_buy_parks_residual_in_cash():
    """An unfilled buy must not keep the matching sell's proceeds in a >1 book."""
    current = {"SPY": 0.4, "BIL": 0.6}
    plan = ExecutionPlan(
        orders=[
            OrderIntent(
                asset="SPY",
                direction="buy",
                target_weight=0.7,
                order_type="limit",
                urgency="high",
                slip_limit=0.001,
            ),
            OrderIntent(
                asset="BIL",
                direction="sell",
                target_weight=0.3,
                order_type="market",
                urgency="normal",
            ),
        ]
    )
    result = simulate_execution(plan, _snap(), current, nav=100_000.0)
    filled = result.filled_weights
    assert sum(filled.values()) <= 1.0 + 1e-3
    assert abs(sum(filled.values()) - 1.0) < 1e-3
    assert abs(filled["SPY"] - 0.4) < 1e-6
    assert abs(filled["BIL"] - 0.6) < 1e-3


def test_full_fill_reaches_target_before_cost_drag():
    current = {"SPY": 0.4, "TLT": 0.4, "BIL": 0.2}
    plan = ExecutionPlan(
        orders=[
            OrderIntent(asset="SPY", direction="buy", target_weight=0.5),
            OrderIntent(asset="TLT", direction="sell", target_weight=0.3),
            OrderIntent(asset="BIL", direction="hold", target_weight=0.2),
        ]
    )
    result = simulate_execution(plan, _snap(), current, nav=100_000.0)
    filled = result.filled_weights
    assert sum(filled.values()) <= 1.0 + 1e-3
    assert abs(filled["SPY"] - 0.5) < 1e-3
    assert abs(filled["TLT"] - 0.3) < 1e-3
    assert filled["BIL"] <= 0.2 + 1e-6


def test_overweight_current_is_projected_before_fill():
    current = {"SPY": 0.7, "BIL": 0.5}
    plan = ExecutionPlan(
        orders=[
            OrderIntent(asset="SPY", direction="hold", target_weight=0.7),
            OrderIntent(asset="BIL", direction="hold", target_weight=0.5),
        ]
    )
    result = simulate_execution(plan, _snap(), current, nav=100_000.0)
    assert result.metadata.get("current_projected") is True
    assert sum(result.filled_weights.values()) <= 1.0 + 1e-3


def test_evaluate_risk_rejects_overweight_book():
    with pytest.raises(ValueError, match="sum above one"):
        evaluate_risk(
            RiskControlDecision(action="hold"),
            {"SPY": 0.7, "BIL": 0.5},
            _snap().return_data,
        )


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


def test_off_simplex_corrective_weights_are_projected():
    weights = {"SPY": 0.9, "BIL": 0.1}
    decision = RiskControlDecision(
        action="rebalance",
        corrective_weights={"SPY": 0.6, "BIL": 0.6},
    )
    result = evaluate_risk(decision, weights, _snap().return_data)
    final = result.metadata["final_weights"]
    assert abs(sum(final.values()) - 1.0) < 1e-6
    assert abs(final["SPY"] - 0.5) < 1e-6
    assert abs(final["BIL"] - 0.5) < 1e-6
