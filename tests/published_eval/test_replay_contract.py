"""Contract checks for the published replay, using synthetic prices only."""

import pandas as pd
import pytest

from portbench.published_eval import CASH, correlation_score, project, replay, risk_state, turnover


def test_excluded_weight_stays_in_cash_and_is_not_rescaled() -> None:
    assert project({"SPY": 0.7, "^VIX": 0.3}, ["SPY"]) == {"SPY": 0.7, CASH: 0.3}
    assert project({"SPY": 0.7, "TLT": 0.3}, ["SPY"]) == {"SPY": 0.7, CASH: 0.3}


def test_class_score_uses_neutral_cross_term_without_cross_weight() -> None:
    correlation = pd.DataFrame(
        [[1.0, 0.2], [0.2, 1.0]],
        index=["SPY", "TLT"],
        columns=["SPY", "TLT"],
    )
    score = correlation_score(
        {"SPY": 1.0, "TLT": 0.0},
        {"correlation": correlation, "assets": ["SPY", "TLT"]},
        {"SPY": "equities", "TLT": "bonds"},
    )
    # No cross-class weight product, so the inter-class term is 0.5.
    assert score == pytest.approx(0.75)


def test_turnover_excludes_cash() -> None:
    current = {"SPY": 0.5, CASH: 0.5}
    target = {"SPY": 0.6, CASH: 0.4}
    assert turnover(current, target) == pytest.approx(0.1)
    assert turnover(current, target, include_cash=True) == pytest.approx(0.2)


def test_rebalance_flag_ignores_var() -> None:
    calm = pd.DataFrame({"SPY": [0.0] * 19 + [-0.02]})
    var, drawdown, flag = risk_state(
        {"SPY": 1.0},
        {"returns": calm, "assets": ["SPY"]},
        "conservative",
    )
    assert var < 0
    assert drawdown > -0.10
    assert flag is False

    flat = pd.DataFrame({"SPY": [0.0] * 5, "TLT": [0.0] * 5})
    var, _, flag = risk_state(
        {"SPY": 0.8, "TLT": 0.2},
        {"returns": flat, "assets": ["SPY", "TLT"]},
        "conservative",
    )
    assert var == pytest.approx(0.0)
    assert flag is True


def test_replay_executes_at_the_next_close_and_charges_fifteen_bps() -> None:
    calendar = pd.to_datetime(
        ["2024-02-01", "2024-02-02", "2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08", "2024-02-09"]
    )
    returns = pd.DataFrame(
        {"SPY": [0.0, 0.01, 0.02, -0.01, 0.0, 0.03, -0.02], CASH: 0.0},
        index=calendar,
    )

    class _Market:
        pass

    market = _Market()
    market.calendar = calendar
    market.return_frame = returns
    metrics, rows = replay(market, {"2024-02-01": {"SPY": 1.0, CASH: 0.0}}, "2024-02-01", "2024-02-09")
    execution = calendar[calendar > "2024-02-01"][0]
    after = returns.loc[(calendar > execution) & (calendar <= "2024-02-09"), "SPY"]
    expected = 0.9985 * (1 + after).prod()
    assert rows[0]["nav"] == pytest.approx(1.0)
    assert metrics["executed_decisions"] == 1
    assert rows[-1]["nav"] == pytest.approx(expected)
