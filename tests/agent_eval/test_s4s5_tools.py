"""SA-T / tools_enabled S4 and S5 call complete_with_tools."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from portbench.agent_eval import build_default_pipeline
from portbench.agent_eval.architectures import IsoTokenBudget
from portbench.agent_eval.base import (
    AgentAdapter,
    MarketSnapshot,
    S1Output,
    S2Output,
    S3Output,
    StageID,
)


class ToolSpyAdapter(AgentAdapter):
    def __init__(self) -> None:
        self.complete_calls = 0
        self.tool_calls = 0

    @property
    def model_name(self) -> str:
        return "tool-spy"

    def complete(self, prompt: str) -> str:
        self.complete_calls += 1
        return self._payload(prompt)

    def complete_with_tools(self, prompt: str, tools: list) -> str:
        self.tool_calls += 1
        assert tools
        return self._payload(prompt)

    def _payload(self, prompt: str) -> str:
        if "execution-planning" in prompt:
            return json.dumps(
                {
                    "orders": [
                        {
                            "asset": "SPY",
                            "direction": "buy",
                            "target_weight": 0.6,
                            "order_type": "market",
                            "urgency": "normal",
                            "slip_limit": 0.001,
                        },
                        {
                            "asset": "BIL",
                            "direction": "sell",
                            "target_weight": 0.4,
                            "order_type": "market",
                            "urgency": "normal",
                            "slip_limit": 0.001,
                        },
                    ],
                    "order_type": "market",
                    "urgency": "normal",
                    "slip_limit": 0.001,
                    "rationale": "test",
                }
            )
        if "risk-control" in prompt:
            return json.dumps(
                {
                    "action": "hold",
                    "alerts": [],
                    "corrective_weights": None,
                    "scale_factor": None,
                    "rationale": "within limits",
                }
            )
        raise AssertionError("S1-S3 should be reused")


def test_sa_t_s4_s5_use_complete_with_tools(tmp_path):
    adapter = ToolSpyAdapter()
    pipeline = build_default_pipeline(
        adapter,
        architecture_id="SA-T",
        cache_dir=str(tmp_path / "cache"),
        budget=IsoTokenBudget(max_tokens_per_episode=8000, max_requests_per_episode=4),
        provider="test",
        profile_name="balanced",
        data_version="test-v1",
        code_commit="test-tree",
        oracle_mode="lookback",
    )
    snapshot = MarketSnapshot(
        decision_date=date(2024, 2, 1),
        price_data={"SPY": pd.Series([100.0]), "BIL": pd.Series([100.0])},
        return_data={
            "SPY": pd.Series([0.01, -0.02, 0.005] * 10),
            "BIL": pd.Series([0.0001] * 30),
        },
        current_weights={"SPY": 0.5, "BIL": 0.5},
        portfolio_value=1_000_000.0,
        market_regime="sideways",
    )
    reused = {
        StageID.S1_MARKET_INTERPRETATION: S1Output(
            asset_views={"SPY": 0.2, "BIL": 0.0},
            macro_summary="cached",
            detected_regime="sideways",
            confidence=0.5,
        ),
        StageID.S2_SIGNAL_GENERATION: S2Output(
            signals={"SPY": "buy", "BIL": "hold"},
            strengths={"SPY": 0.5, "BIL": 0.0},
            reasoning="cached",
        ),
        StageID.S3_WEIGHT_OPTIMIZATION: S3Output(
            weights={"SPY": 0.6, "BIL": 0.4},
            expected_return=0.0,
            expected_vol=0.0,
            sharpe_estimate=0.0,
        ),
    }
    pipeline.run_episode(snapshot, reuse_outputs=reused)
    assert adapter.tool_calls == 2
    assert adapter.complete_calls == 0
