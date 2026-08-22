"""Tests for deterministic Point-in-Time portfolio tools."""

from datetime import date

import pandas as pd

from portbench.agent_eval.base import MarketSnapshot
from portbench.agent_eval.tools import dispatch_tool, get_tools


def _snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        decision_date=date(2024, 1, 1),
        price_data={},
        return_data={
            "SPY": pd.Series([0.01, -0.02, 0.015, -0.005]),
            "BIL": pd.Series([0.0001, 0.0001, 0.0001, 0.0001]),
        },
        current_weights={"SPY": 0.5, "BIL": 0.5},
        portfolio_value=100_000.0,
        future_return_data={"SPY": pd.Series([0.9])},
    )


def test_snapshot_tools_are_bound_and_numerically_consistent():
    tools = get_tools(snapshot=_snapshot())
    names = {tool.name for tool in tools}
    assert {"portfolio_risk", "risk_contribution", "execution_cost"} <= names
    risk = dispatch_tool("portfolio_risk", {"weights": {"SPY": 0.6, "BIL": 0.4}}, tools)
    contribution = dispatch_tool(
        "risk_contribution", {"weights": {"SPY": 0.6, "BIL": 0.4}}, tools
    )
    cost = dispatch_tool(
        "execution_cost", {"target_weights": {"SPY": 0.6, "BIL": 0.4}}, tools
    )
    assert risk["annualized_volatility"] > 0.0
    assert abs(sum(contribution.values()) - 1.0) < 1e-8
    assert abs(cost["turnover"] - 0.2) < 1e-8
    assert abs(cost["estimated_cost"] - 30.0) < 1e-8


def test_snapshot_tools_do_not_depend_on_future_returns():
    first = _snapshot()
    second = _snapshot()
    second.future_return_data = {"SPY": pd.Series([-0.9])}
    first_tools = get_tools(snapshot=first)
    second_tools = get_tools(snapshot=second)
    args = {"weights": {"SPY": 0.6, "BIL": 0.4}}
    assert dispatch_tool("portfolio_risk", args, first_tools) == dispatch_tool(
        "portfolio_risk", args, second_tools
    )
