"""Agent-planned S4/S5 stages with deterministic environment evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from portbench.agent_eval.contracts import (
    S4S5_SCHEMA_AGENTIC,
    ExecutionPlan,
    ExecutionResult,
    OrderIntent,
    RiskControlDecision,
    RiskEvaluationResult,
)
from portbench.agent_eval.s4_s5_bridge import run_s4_deterministic_from_weights
from portbench.metrics.plan_outcome_scores import (
    score_s4_environment_outcome,
    score_s4_plan_quality,
    score_s5_environment_outcome,
    score_s5_plan_quality,
)
from portbench.sandbox.execution import simulate_execution
from portbench.sandbox.risk_control import evaluate_risk, summarize_pre_action_risk


class AgenticStageError(RuntimeError):
    """Raised when an agentic stage cannot produce a valid decision."""


@dataclass
class S4AgenticBundle:
    """Store the S4 plan separately from deterministic fill outcomes."""

    plan: ExecutionPlan
    result: ExecutionResult
    raw_response: str = ""
    plan_scores: Dict[str, float] = field(default_factory=dict)
    outcome_scores: Dict[str, float] = field(default_factory=dict)
    schema_version: str = S4S5_SCHEMA_AGENTIC


@dataclass
class S5AgenticBundle:
    """Store the S5 decision separately from post-action outcomes."""

    decision: RiskControlDecision
    eval_result: RiskEvaluationResult
    raw_response: str = ""
    plan_scores: Dict[str, float] = field(default_factory=dict)
    outcome_scores: Dict[str, float] = field(default_factory=dict)
    schema_version: str = S4S5_SCHEMA_AGENTIC


def _extract_json(text: str) -> Mapping[str, Any]:
    """Extract one JSON object and reject empty or malformed responses."""
    if not text or not text.strip():
        raise AgenticStageError("empty model response")
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", stripped)
        if match is None:
            raise AgenticStageError("response does not contain a JSON object")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise AgenticStageError(f"invalid JSON response: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise AgenticStageError("stage response must be a JSON object")
    return payload


def _plan_from_payload(payload: Mapping[str, Any]) -> ExecutionPlan:
    """Parse and validate a multi-asset execution plan."""
    orders_raw = payload.get("orders")
    if not isinstance(orders_raw, list) or not orders_raw:
        raise AgenticStageError("S4 response requires a non-empty orders list")
    orders = []
    assets = set()
    for item in orders_raw:
        if not isinstance(item, Mapping):
            raise AgenticStageError("each S4 order must be an object")
        order = OrderIntent(
            asset=str(item.get("asset", "")),
            direction=str(item.get("direction", "hold")),
            target_weight=item.get("target_weight"),
            delta_weight=item.get("delta_weight"),
            order_type=str(item.get("order_type", payload.get("order_type", "market"))),
            urgency=str(item.get("urgency", payload.get("urgency", "normal"))),
            slip_limit=item.get("slip_limit", payload.get("slip_limit")),
        )
        if not order.asset or order.asset in assets:
            raise AgenticStageError("S4 orders require unique non-empty assets")
        assets.add(order.asset)
        orders.append(order)
    return ExecutionPlan(
        order_direction="rebalance_to_target",
        order_type=str(payload.get("order_type", "market")),
        urgency=str(payload.get("urgency", "normal")),
        slip_limit=payload.get("slip_limit"),
        orders=orders,
        metadata={"rationale": str(payload.get("rationale", ""))},
    )


def _decision_from_payload(payload: Mapping[str, Any]) -> RiskControlDecision:
    """Parse and validate an S5 control decision."""
    corrective = payload.get("corrective_weights")
    if corrective is not None and not isinstance(corrective, Mapping):
        raise AgenticStageError("corrective_weights must be an object or null")
    try:
        return RiskControlDecision(
            action=str(payload.get("action", "")),
            alerts=list(payload.get("alerts") or []),
            corrective_weights=(
                {str(asset): float(weight) for asset, weight in corrective.items()}
                if isinstance(corrective, Mapping)
                else None
            ),
            scale_factor=payload.get("scale_factor"),
            metadata={"rationale": str(payload.get("rationale", ""))},
        )
    except (TypeError, ValueError) as exc:
        raise AgenticStageError(f"invalid S5 decision: {exc}") from exc


def _snapshot_context(snapshot_like: Any) -> Dict[str, Any]:
    """Extract only Point-in-Time execution information for the S4 prompt."""
    prices: Dict[str, float] = {}
    price_data = getattr(snapshot_like, "price_data", None)
    if price_data is None and isinstance(snapshot_like, Mapping):
        price_data = snapshot_like.get("price_data", {})
    for asset, values in (price_data or {}).items():
        if hasattr(values, "empty") and not values.empty:
            prices[str(asset)] = float(values.iloc[-1])
    return {"decision_date": str(getattr(snapshot_like, "decision_date", "")), "prices": prices}


def _reference_risk_decision(
    weights: Mapping[str, float],
    pre_risk: Mapping[str, Any],
    var_limit: float,
    drawdown_limit: float,
    drift_limit: float,
) -> RiskControlDecision:
    """Construct a deterministic Point-in-Time reference decision."""
    alerts = []
    if float(pre_risk["var"]) < var_limit:
        alerts.append("var_breach")
    if float(pre_risk["drawdown"]) < drawdown_limit:
        alerts.append("drawdown")
    if alerts:
        return RiskControlDecision(action="scale_down", alerts=alerts, scale_factor=0.8)
    if float(pre_risk["weight_drift"]) > drift_limit:
        assets = list(weights)
        equal = {asset: 1.0 / len(assets) for asset in assets} if assets else None
        return RiskControlDecision(action="rebalance", alerts=["weight_drift"], corrective_weights=equal)
    return RiskControlDecision(action="hold")


class AgenticS4Stage:
    """Ask the model for an execution plan and simulate its fills."""

    def __init__(self, adapter: Any):
        if adapter is None or not hasattr(adapter, "complete"):
            raise ValueError("AgenticS4Stage requires an adapter")
        self.adapter = adapter
        self.last_prompt = ""

    def run(
        self,
        *,
        snapshot_like: Any,
        current_weights: Mapping[str, float],
        target_weights: Mapping[str, float],
        nav: float,
        reference_plan: Optional[ExecutionPlan] = None,
        reference_result: Optional[ExecutionResult] = None,
        slippage_rate: float = 0.001,
        commission_rate: float = 0.0005,
    ) -> S4AgenticBundle:
        expected_directions = {}
        for asset, target_weight in target_weights.items():
            delta = float(target_weight) - float(current_weights.get(asset, 0.0))
            expected_directions[asset] = "hold" if abs(delta) < 1e-6 else "buy" if delta > 0 else "sell"
        context = {
            "market": _snapshot_context(snapshot_like),
            "current_weights": dict(current_weights),
            "target_weights": dict(target_weights),
            "expected_directions": expected_directions,
            "nav": float(nav),
            "slippage_rate": slippage_rate,
            "commission_rate": commission_rate,
        }
        self.last_prompt = (
            "You are the execution-planning stage. Use only the supplied Point-in-Time context. "
            "Return exactly one JSON object with keys orders, order_type, urgency, slip_limit, rationale. "
            "orders must contain every target asset with asset, direction, target_weight, order_type, urgency, slip_limit. "
            "Copy each supplied target weight and expected direction exactly. "
            f"Context: {json.dumps(context, ensure_ascii=False, default=str)}"
        )
        try:
            raw = self.adapter.complete(self.last_prompt)
        except Exception as exc:
            raise AgenticStageError(f"S4 model call failed: {exc}") from exc
        plan = _plan_from_payload(_extract_json(str(raw)))
        result = simulate_execution(
            plan,
            snapshot_like,
            current_weights,
            nav,
            slippage_rate=slippage_rate,
            commission_rate=commission_rate,
        )
        result.metadata["schema_version"] = S4S5_SCHEMA_AGENTIC
        plan.metadata["schema_version"] = S4S5_SCHEMA_AGENTIC
        ref_plan = reference_plan
        if ref_plan is None:
            ref_plan, automatic_result = run_s4_deterministic_from_weights(
                target_weights,
                snapshot_like,
                current_weights,
                nav,
                slippage_rate=slippage_rate,
                commission_rate=commission_rate,
            )
            if reference_result is None:
                reference_result = automatic_result
        if reference_result is None:
            raise ValueError("reference_result is required when reference_plan is supplied")
        return S4AgenticBundle(
            plan=plan,
            result=result,
            raw_response=str(raw),
            plan_scores=score_s4_plan_quality(plan, ref_plan, target_weights=target_weights),
            outcome_scores=score_s4_environment_outcome(result, reference_result),
        )


class AgenticS5Stage:
    """Ask the model for a risk action and evaluate its post-action outcome."""

    def __init__(self, adapter: Any):
        if adapter is None or not hasattr(adapter, "complete"):
            raise ValueError("AgenticS5Stage requires an adapter")
        self.adapter = adapter
        self.last_prompt = ""

    def run(
        self,
        *,
        weights: Mapping[str, float],
        return_data: Mapping[str, Any],
        reference_decision: Optional[RiskControlDecision] = None,
        reference_eval: Optional[RiskEvaluationResult] = None,
        var_limit: float = -0.02,
        drawdown_limit: float = -0.10,
        drift_limit: float = 0.05,
    ) -> S5AgenticBundle:
        pre_risk = summarize_pre_action_risk(weights, return_data)
        context = {
            "weights": dict(weights),
            "pre_action_risk": pre_risk,
            "limits": {"var": var_limit, "drawdown": drawdown_limit, "weight_drift": drift_limit},
        }
        self.last_prompt = (
            "You are the portfolio risk-control stage. Use only the supplied Point-in-Time risk summary. "
            "Return exactly one JSON object with action, alerts, corrective_weights, scale_factor, rationale. "
            "action must be hold, scale_down, or rebalance. Rebalance requires non-negative weights summing to one; "
            "scale_down requires a scale_factor in [0,1]. "
            f"Context: {json.dumps(context, ensure_ascii=False, default=str)}"
        )
        try:
            raw = self.adapter.complete(self.last_prompt)
        except Exception as exc:
            raise AgenticStageError(f"S5 model call failed: {exc}") from exc
        decision = _decision_from_payload(_extract_json(str(raw)))
        decision.metadata["schema_version"] = S4S5_SCHEMA_AGENTIC
        eval_result = evaluate_risk(
            decision,
            weights,
            return_data,
            var_limit=var_limit,
            drawdown_limit=drawdown_limit,
            drift_limit=drift_limit,
        )
        eval_result.metadata["schema_version"] = S4S5_SCHEMA_AGENTIC
        ref_decision = reference_decision or _reference_risk_decision(
            weights, pre_risk, var_limit, drawdown_limit, drift_limit
        )
        ref_result = reference_eval or evaluate_risk(
            ref_decision,
            weights,
            return_data,
            var_limit=var_limit,
            drawdown_limit=drawdown_limit,
            drift_limit=drift_limit,
        )
        return S5AgenticBundle(
            decision=decision,
            eval_result=eval_result,
            raw_response=str(raw),
            plan_scores=score_s5_plan_quality(decision, ref_decision),
            outcome_scores=score_s5_environment_outcome(eval_result, ref_result),
        )


__all__ = [
    "AgenticStageError",
    "S4AgenticBundle",
    "S5AgenticBundle",
    "AgenticS4Stage",
    "AgenticS5Stage",
]
