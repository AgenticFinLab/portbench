"""S1 JSON-only vs tools-then-JSON contracts."""

from __future__ import annotations

from datetime import date

import pandas as pd

from portbench.agent_eval.base import MarketSnapshot
from portbench.agent_eval.prompts import build_s1_prompt


def _snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        decision_date=date(2024, 1, 2),
        price_data={"SPY": pd.Series([100.0])},
        return_data={"SPY": pd.Series([0.01, -0.02])},
        current_weights={"SPY": 1.0},
        portfolio_value=100_000.0,
    )


def test_s1_json_only_without_tools():
    prompt = build_s1_prompt(
        _snapshot(),
        ["SPY"],
        "px",
        "macro",
        "corr",
        20,
        use_tools=False,
    )
    assert "Reply with EXACTLY ONE JSON object and nothing else" in prompt
    assert "You MAY call the provided tools" not in prompt


def test_s1_tools_then_json():
    prompt = build_s1_prompt(
        _snapshot(),
        ["SPY"],
        "px",
        "macro",
        "corr",
        20,
        use_tools=True,
    )
    assert "You MAY call the provided tools" in prompt
    assert "FINAL assistant message must be EXACTLY ONE" in prompt
