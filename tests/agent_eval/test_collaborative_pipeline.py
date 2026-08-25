"""Acceptance tests for the eight-call collaborative pipeline."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from portbench.agent_eval import build_default_pipeline
from portbench.agent_eval.architectures import AgentMessage, MessageBus
from portbench.agent_eval.base import AgentAdapter, MarketSnapshot


class CollaborationAdapter(AgentAdapter):
    """Return deterministic valid JSON for every logical collaboration call."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "collaboration-test"

    def complete(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if "RISK CRITIQUE" in prompt:
            return json.dumps(
                {
                    "risk_assessment": "acceptable",
                    "constraint_violations": [],
                    "recommended_changes": {"SPY": 0.55, "BIL": 0.45},
                }
            )
        if "EXECUTION CRITIQUE" in prompt:
            return json.dumps(
                {
                    "execution_assessment": "low cost",
                    "turnover_concerns": [],
                    "recommended_changes": {"SPY": 0.55, "BIL": 0.45},
                }
            )
        if "OPTIMIZER REVISION" in prompt:
            assert "risk_assessment" in prompt
            assert "execution_assessment" in prompt
            return json.dumps(
                {
                    "weights": {"SPY": 0.55, "BIL": 0.45},
                    "expected_return": 0.05,
                    "expected_vol": 0.10,
                    "sharpe_estimate": 0.5,
                    "revision_rationale": "incorporated both critiques",
                }
            )
        if "execution-planning" in prompt:
            return json.dumps(
                {
                    "orders": [
                        {
                            "asset": "SPY",
                            "direction": "buy",
                            "target_weight": 0.55,
                            "order_type": "market",
                            "urgency": "normal",
                            "slip_limit": 0.001,
                        },
                        {
                            "asset": "BIL",
                            "direction": "sell",
                            "target_weight": 0.45,
                            "order_type": "market",
                            "urgency": "normal",
                            "slip_limit": 0.001,
                        },
                    ],
                    "order_type": "market",
                    "urgency": "normal",
                    "slip_limit": 0.001,
                    "rationale": "implement revised weights",
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
        if '"asset_views"' in prompt:
            return json.dumps(
                {
                    "asset_views": {"SPY": 0.4, "BIL": 0.0},
                    "detected_regime": "sideways",
                    "confidence": 0.7,
                    "macro_summary": "stable",
                }
            )
        if '"allocation_scores"' in prompt:
            return json.dumps(
                {
                    "allocation_scores": {"SPY": 0.6, "BIL": 0.4},
                    "expected_return": 0.06,
                    "expected_vol": 0.11,
                    "sharpe_estimate": 0.55,
                }
            )
        if '"signals"' in prompt:
            return json.dumps(
                {
                    "signals": {"SPY": "buy", "BIL": "hold"},
                    "strengths": {"SPY": 0.7, "BIL": 0.2},
                    "reasoning": "positive equity signal",
                }
            )
        if '"weights"' in prompt:
            return json.dumps(
                {
                    "weights": {"SPY": 0.6, "BIL": 0.4},
                    "expected_return": 0.06,
                    "expected_vol": 0.11,
                    "sharpe_estimate": 0.55,
                }
            )
        raise AssertionError(f"unexpected prompt: {prompt[:200]}")


def _snapshot() -> MarketSnapshot:
    """Return a small Point-in-Time snapshot with two assets."""
    return MarketSnapshot(
        decision_date=date(2024, 2, 1),
        price_data={"SPY": pd.Series([99.0, 100.0]), "BIL": pd.Series([100.0, 100.1])},
        return_data={
            "SPY": pd.Series([0.01, -0.005, 0.007] * 12),
            "BIL": pd.Series([0.0001] * 36),
        },
        current_weights={"SPY": 0.5, "BIL": 0.5},
        portfolio_value=1_000_000.0,
        market_regime="sideways",
        asset_class_map={"SPY": "equities", "BIL": "cash"},
    )


@pytest.mark.parametrize(("architecture_id", "expected_calls"), [("SA", 5), ("MA", 8)])
def test_sa_and_ma_have_declared_logical_call_counts(architecture_id, expected_calls):
    adapter = CollaborationAdapter()
    pipeline = build_default_pipeline(
        adapter,
        architecture_id=architecture_id,
        provider="test",
        profile_name="balanced",
        data_version="test-v1",
        code_commit="test-tree",
        oracle_mode="lookback",
    )
    result = pipeline.run_episode(_snapshot())
    assert adapter.calls == expected_calls
    assert result.resource_usage["logical_call_count"] == expected_calls
    assert result.resource_usage["request_count"] == expected_calls
    assert result.schema_version == "pipeline-v3-collab"
    if architecture_id == "MA":
        assert len(result.collaboration_trace) == 8
        assert result.resource_usage["by_agent"]["optimizer"]["logical_call_count"] == 2
        assert result.resource_usage["by_agent"]["risk"]["logical_call_count"] == 2
        assert result.resource_usage["by_agent"]["executor"]["logical_call_count"] == 2
        assert "risk_critique" in result.stage_outputs[next(
            stage_id for stage_id in result.stage_outputs if stage_id.value == "S3"
        )].collaboration


def test_message_bus_rejects_forward_data():
    bus = MessageBus()
    message = AgentMessage(
        message_id="m1",
        sender="optimizer",
        recipient="risk",
        episode_id="ep1",
        decision_date="2024-01-01",
        stage_id="S3",
        round_id="proposal",
        message_type="candidate",
        payload={"future_return_data": {"SPY": [0.1]}},
    )
    with pytest.raises(PermissionError, match="forbids future data"):
        bus.publish(message)
