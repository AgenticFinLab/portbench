"""Legacy S4/S5 bridge round-trip tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

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


def test_agentic_s5_accepts_off_simplex_corrective():
    import json

    from portbench.agent_eval.s4_s5_stages import AgenticS5Stage

    class Adapter:
        @property
        def model_name(self) -> str:
            return "s5-bad-sum"

        def complete(self, prompt: str) -> str:
            return json.dumps(
                {
                    "action": "rebalance",
                    "alerts": ["drawdown"],
                    "corrective_weights": {"SPY": 0.7, "BIL": 0.7},
                    "scale_factor": None,
                    "rationale": "does not sum to one",
                }
            )

    returns = {
        "SPY": pd.Series([0.04, -0.08, 0.03, -0.06] * 8),
        "BIL": pd.Series([0.0001] * 32),
    }
    bundle = AgenticS5Stage(Adapter()).run(
        weights={"SPY": 0.9, "BIL": 0.1},
        return_data=returns,
    )
    final = bundle.eval_result.metadata["final_weights"]
    assert abs(sum(final.values()) - 1.0) < 1e-6
    assert bundle.plan_scores["corrective_compliance"] == 0.0


def test_agentic_s5_prompt_lists_contract_without_answer_policy():
    import json

    from portbench.agent_eval.s4_s5_stages import AgenticS5Stage

    class Adapter:
        @property
        def model_name(self) -> str:
            return "s5-policy"

        def complete(self, prompt: str) -> str:
            return json.dumps(
                {
                    "action": "hold",
                    "alerts": [],
                    "corrective_weights": None,
                    "scale_factor": None,
                    "rationale": "ok",
                }
            )

    returns = {"SPY": pd.Series([0.001] * 32), "BIL": pd.Series([0.0001] * 32)}
    stage = AgenticS5Stage(Adapter())
    stage.run(weights={"SPY": 0.5, "BIL": 0.5}, return_data=returns)
    prompt = stage.last_prompt
    assert "weight_drift" in prompt
    assert "var_breach" in prompt
    assert "drawdown" in prompt
    assert "only weight_drift exceeds its limit: action=rebalance" not in prompt
    assert "Explain the risk evidence" in prompt


def test_agentic_s5_rejects_non_tradable_corrective_asset():
    import json

    from portbench.agent_eval.s4_s5_stages import AgenticS5Stage

    class Adapter:
        def complete(self, prompt: str) -> str:
            return json.dumps(
                {
                    "action": "rebalance",
                    "alerts": ["drawdown"],
                    "corrective_weights": {"CASH": 1.0},
                    "scale_factor": None,
                    "rationale": "move to cash",
                }
            )

    returns = {"SPY": pd.Series([0.001] * 32), "BIL": pd.Series([0.0001] * 32)}
    bundle = AgenticS5Stage(Adapter()).run(
        weights={"SPY": 0.6, "BIL": 0.4},
        return_data=returns,
    )

    assert bundle.decision.action == "hold"
    assert bundle.decision.metadata.get("parse_error")
    assert set(bundle.eval_result.metadata["final_weights"]) == {"SPY", "BIL"}
    assert all(value == 0.0 for value in bundle.plan_scores.values())


def test_agentic_s4_retries_then_marks_parse_error_and_zeros_plan():
    from portbench.agent_eval.agentic_pipeline_stages import AgenticS4PipelineStage
    from portbench.agent_eval.base import MarketSnapshot, S3Output
    from portbench.agent_eval.s4_s5_stages import AgenticS4Stage, _JSON_RETRY_LIMIT
    from datetime import date

    class BadAdapter:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def model_name(self) -> str:
            return "s4-bad"

        def complete(self, prompt: str) -> str:
            self.calls += 1
            return "not-json"

    current = {"SPY": 0.6, "BIL": 0.4}
    snap = SimpleNamespace(
        current_weights=current,
        portfolio_value=100_000.0,
        price_data={
            "SPY": pd.Series([100.0, 101.0]),
            "BIL": pd.Series([100.0, 100.0]),
        },
    )
    adapter = BadAdapter()
    bundle = AgenticS4Stage(adapter).run(
        snapshot_like=snap,
        current_weights=current,
        target_weights=current,
        nav=100_000.0,
    )
    assert adapter.calls == _JSON_RETRY_LIMIT + 1
    assert bundle.plan.metadata.get("parse_error")
    assert bundle.plan_scores["order_legality"] == 0.0
    assert bundle.plan_scores["target_tracking"] == 0.0
    assert bundle.plan_scores["plan_quality"] == 0.0

    pipeline_stage = AgenticS4PipelineStage(adapter)
    snapshot = MarketSnapshot(
        decision_date=date(2024, 2, 1),
        price_data=snap.price_data,
        return_data={"SPY": pd.Series([0.01] * 10), "BIL": pd.Series([0.0] * 10)},
        current_weights=current,
        portfolio_value=100_000.0,
    )
    output = pipeline_stage.run(snapshot, S3Output(weights=current))
    assert output.refused is True


def test_agentic_s4_normalizes_order_enum_casing():
    from portbench.agent_eval.s4_s5_stages import _plan_from_payload

    plan = _plan_from_payload(
        {
            "orders": [
                {
                    "asset": "SPY",
                    "direction": "SELL",
                    "delta_weight": -0.1,
                    "order_type": "MARKET",
                    "urgency": "HIGH",
                    "slip_limit": 0.001,
                }
            ]
        }
    )

    order = plan.normalized_orders()[0]
    assert order.direction == "sell"
    assert order.order_type == "market"
    assert order.urgency == "high"


def test_agentic_s4_compact_target_book_expands_liquidations():
    from portbench.agent_eval.s4_s5_stages import _plan_from_payload

    plan = _plan_from_payload(
        {
            "target_weights": {"BIL": 0.7, "SPY": 0.3},
            "order_type": "LIMIT",
            "urgency": "HIGH",
            "slip_limit": 0.002,
            "rationale": "Use one bounded policy for the rebalance.",
        },
        allowed_assets={"BIL", "SPY", "TLT"},
        current_weights={"BIL": 0.2, "SPY": 0.5, "TLT": 0.3},
    )

    orders = {order.asset: order for order in plan.normalized_orders()}
    assert orders["BIL"].direction == "buy"
    assert orders["SPY"].direction == "sell"
    assert orders["TLT"].direction == "sell"
    assert orders["TLT"].target_weight == 0.0
    assert all(order.order_type == "limit" for order in orders.values())
    assert plan.metadata["response_representation"] == "compact-target-book-v1"


def test_agentic_s4_rejects_truncated_saved_response_shape():
    from portbench.agent_eval.s4_s5_stages import AgenticStageError, _extract_json

    # This prefix mirrors a saved formal response truncated inside its orders array.
    raw = (
        '{"orders": ['
        '{"asset": "VLUE", "direction": "SELL", "delta_weight": -0.0099, '
        '"order_type": "MARKET", "urgency": "MEDIUM", "slip_limit": 0.0015},'
        '{"asset": "JNK", "direction": "SELL"'
    )

    with pytest.raises(AgenticStageError, match="JSON"):
        _extract_json(raw)


def test_agentic_s4_prompt_requests_bound_target_policy():
    from portbench.agent_eval.s4_s5_stages import AgenticS4Stage

    class Adapter:
        def complete(self, prompt: str) -> str:
            return json.dumps(
                {
                    "order_type": "market",
                    "urgency": "normal",
                    "slip_limit": 0.001,
                    "rationale": "Use immediate execution for a liquid book.",
                }
            )

    current = {"SPY": 0.5, "BIL": 0.4, "TLT": 0.1}
    snapshot = SimpleNamespace(
        current_weights=current,
        portfolio_value=100_000.0,
        price_data={
            "SPY": pd.Series([100.0, 101.0]),
            "BIL": pd.Series([100.0, 100.0]),
            "TLT": pd.Series([100.0, 99.0]),
        },
        return_data={
            "SPY": pd.Series([0.01, -0.01]),
            "BIL": pd.Series([0.0, 0.0]),
            "TLT": pd.Series([0.01, -0.01]),
        },
        macro_data={},
    )
    stage = AgenticS4Stage(Adapter())

    bundle = stage.run(
        snapshot_like=snapshot,
        current_weights=current,
        target_weights={"SPY": 0.6, "BIL": 0.4},
        nav=100_000.0,
    )

    assert "bound by the runtime" in stage.last_prompt
    assert "Do not return target_weights" in stage.last_prompt
    assert bundle.plan.metadata["response_representation"] == "bound-target-policy-v1"
    assert bundle.plan_scores["order_legality"] == 1.0
    assert bundle.plan_scores["target_tracking"] == 1.0


def test_agentic_s4_bound_target_ignores_incomplete_echo_and_normalizes_medium():
    from portbench.agent_eval.s4_s5_stages import _plan_from_payload

    plan = _plan_from_payload(
        {
            "target_weights": {"SPY": 0.6},
            "order_type": "limit",
            "urgency": "medium",
            "slip_limit": 0.002,
            "rationale": "Use a bounded global execution policy.",
        },
        allowed_assets={"BIL", "SPY", "TLT"},
        current_weights={"BIL": 0.2, "SPY": 0.5, "TLT": 0.3},
        bound_target_weights={"BIL": 0.4, "SPY": 0.6},
    )

    orders = {order.asset: order for order in plan.normalized_orders()}
    assert orders["BIL"].target_weight == 0.4
    assert orders["SPY"].target_weight == 0.6
    assert orders["TLT"].target_weight == 0.0
    assert all(order.urgency == "normal" for order in orders.values())
    assert plan.metadata["response_representation"] == "bound-target-policy-v1"


def test_agentic_s4_rejects_non_tradable_order_asset():
    from portbench.agent_eval.s4_s5_stages import AgenticStageError, _plan_from_payload

    with pytest.raises(AgenticStageError, match="visible assets"):
        _plan_from_payload(
            {
                "orders": [
                    {
                        "asset": "CASH",
                        "direction": "buy",
                        "target_weight": 1.0,
                        "order_type": "market",
                        "urgency": "normal",
                    }
                ]
            },
            allowed_assets={"SPY", "BIL"},
        )


def test_agentic_s5_rejects_overweight_incoming_book():
    import json

    from portbench.agent_eval.s4_s5_stages import AgenticS5Stage, AgenticStageError

    class Adapter:
        def complete(self, prompt: str) -> str:
            raise AssertionError("S5 must not call the model on an illegal S4 book")

    returns = {
        "SPY": pd.Series([0.04, -0.08, 0.03, -0.06] * 8),
        "BIL": pd.Series([0.0001] * 32),
    }
    with pytest.raises(AgenticStageError, match="sum above one"):
        AgenticS5Stage(Adapter()).run(
            weights={"SPY": 0.7, "BIL": 0.5},
            return_data=returns,
        )


def test_agentic_s5_equal_weight_rule_expands_in_parser():
    from portbench.agent_eval.s4_s5_stages import _decision_from_payload

    decision = _decision_from_payload(
        {
            "action": "rebalance",
            "alerts": ["weight_drift"],
            "corrective_rule": "equal_weight",
            "corrective_weights": None,
            "scale_factor": None,
            "rationale": "Restore the declared active-book reference.",
        },
        allowed_assets={"SPY", "BIL", "TLT"},
    )

    assert decision.corrective_weights == pytest.approx(
        {"SPY": 1.0 / 3.0, "BIL": 1.0 / 3.0, "TLT": 1.0 / 3.0}
    )
    assert decision.metadata["response_representation"] == "corrective-rule-v1"


def test_agentic_s5_rejects_two_corrective_representations():
    from portbench.agent_eval.s4_s5_stages import AgenticStageError, _decision_from_payload

    with pytest.raises(AgenticStageError, match="not both"):
        _decision_from_payload(
            {
                "action": "rebalance",
                "corrective_rule": "equal_weight",
                "corrective_weights": {"BIL": 1.0},
            },
            allowed_assets={"BIL"},
        )


def test_deterministic_s4_off_simplex_does_not_abort():
    current = {"AMAT": 0.096, "SPY": 0.904}
    target = {"AMAT": 0.10, "SPY": 0.95}
    snap = SimpleNamespace(
        current_weights=current,
        portfolio_value=100_000.0,
        price_data={
            "AMAT": pd.Series([100.0, 100.0]),
            "SPY": pd.Series([100.0, 101.0]),
        },
    )
    plan, result = run_s4_deterministic_from_weights(
        target, snap, current, nav=100_000.0
    )
    assert abs(sum(result.filled_weights.values()) - 1.0) < 1e-3
    amat = next(o for o in plan.normalized_orders() if o.asset == "AMAT")
    assert amat.direction == "sell"
    assert "AMAT" not in result.metadata.get("direction_conflicts", [])
