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
from portbench.agent_eval.prompts import build_format_correction_suffix, format_contract
from portbench.agent_eval.s4_s5_bridge import run_s4_deterministic_from_weights
from portbench.agent_eval.stages import _JSON_RETRY_LIMIT
from portbench.agent_eval.tools import get_tools
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
        try:
            order = OrderIntent(
                asset=str(item.get("asset", "")),
                direction=str(item.get("direction", "hold")),
                target_weight=item.get("target_weight"),
                delta_weight=item.get("delta_weight"),
                order_type=str(item.get("order_type", payload.get("order_type", "market"))),
                urgency=str(item.get("urgency", payload.get("urgency", "normal"))),
                slip_limit=item.get("slip_limit", payload.get("slip_limit")),
            )
        except (TypeError, ValueError) as exc:
            raise AgenticStageError(f"invalid S4 order: {exc}") from exc
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


def _complete(adapter: Any, prompt: str, *, use_tools: bool, snapshot: Any = None) -> str:
    """Call complete or complete_with_tools for one agentic stage."""
    if use_tools:
        return adapter.complete_with_tools(prompt, get_tools(snapshot=snapshot))
    return adapter.complete(prompt)


def _complete_json(
    adapter: Any,
    prompt: str,
    *,
    use_tools: bool,
    snapshot: Any,
    parse,
) -> tuple[Any, str, Optional[Exception]]:
    """Retry JSON parse like S1–S3. Return (parsed, raw, error); parsed is None on failure."""
    last_err: Optional[Exception] = None
    last_raw = ""
    for attempt in range(_JSON_RETRY_LIMIT + 1):
        full_prompt = (
            prompt
            if attempt == 0
            else prompt + build_format_correction_suffix(str(last_err))
        )
        last_raw = _complete(adapter, full_prompt, use_tools=use_tools, snapshot=snapshot)
        try:
            return parse(_extract_json(str(last_raw))), last_raw, None
        except (AgenticStageError, ValueError, TypeError) as exc:
            last_err = exc
    return None, last_raw, last_err


S5_DECISION_POLICY = (
    "alerts must be a list of these tokens only: var_breach, drawdown, weight_drift "
    "(omit a token if that limit is not breached). "
    "If VaR or drawdown breaches its limit: action=scale_down with scale_factor in [0,1] "
    "and include the matching alert tokens. "
    "If only weight_drift exceeds its limit: action=rebalance with non-negative "
    "corrective_weights that sum to exactly 1.0 and alerts=[\"weight_drift\"]. "
    "If no limit is breached: action=hold, alerts=[], corrective_weights=null. "
    "corrective_weights are required only for rebalance and must be non-negative "
    "and sum to 1.0; otherwise null."
)


class AgenticS4Stage:
    """Ask the model for an execution plan and simulate its fills."""

    def __init__(self, adapter: Any, use_tools: bool = False):
        if adapter is None or not hasattr(adapter, "complete"):
            raise ValueError("AgenticS4Stage requires an adapter")
        self.adapter = adapter
        self.use_tools = bool(use_tools)
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
            "orders must contain every target asset with asset, direction, target_weight, order_type, urgency, slip_limit. "
            "Copy each supplied target weight and expected direction exactly. "
            f"Context: {json.dumps(context, ensure_ascii=False, default=str)}\n"
            f"{format_contract(self.use_tools)}"
        )
        try:
            plan, raw, parse_err = _complete_json(
                self.adapter,
                self.last_prompt,
                use_tools=self.use_tools,
                snapshot=snapshot_like,
                parse=_plan_from_payload,
            )
        except Exception as exc:
            raise AgenticStageError(f"S4 model call failed: {exc}") from exc
        if plan is None:
            plan = ExecutionPlan(
                orders=[],
                metadata={"parse_error": str(parse_err), "rationale": ""},
            )
        try:
            result = simulate_execution(
                plan,
                snapshot_like,
                current_weights,
                nav,
                slippage_rate=slippage_rate,
                commission_rate=commission_rate,
            )
        except ValueError as exc:
            held = {str(k): round(float(v), 4) for k, v in dict(current_weights).items()}
            result = ExecutionResult(
                filled_weights=held,
                metadata={"execution_error": str(exc), "held_current": True},
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
            plan_scores=score_s4_plan_quality(
                plan,
                ref_plan,
                target_weights=target_weights,
                current_weights=current_weights,
            ),
            outcome_scores=score_s4_environment_outcome(result, reference_result),
        )


class AgenticS5Stage:
    """Ask the model for a risk action and evaluate its post-action outcome."""

    def __init__(self, adapter: Any, use_tools: bool = False):
        if adapter is None or not hasattr(adapter, "complete"):
            raise ValueError("AgenticS5Stage requires an adapter")
        self.adapter = adapter
        self.use_tools = bool(use_tools)
        self.last_prompt = ""

    def run(
        self,
        *,
        weights: Mapping[str, float],
        return_data: Mapping[str, Any],
        snapshot_like: Any = None,
        reference_decision: Optional[RiskControlDecision] = None,
        reference_eval: Optional[RiskEvaluationResult] = None,
        var_limit: float = -0.02,
        drawdown_limit: float = -0.10,
        drift_limit: float = 0.05,
    ) -> S5AgenticBundle:
        book = {str(asset): float(weight) for asset, weight in dict(weights).items()}
        if any(weight < 0.0 for weight in book.values()):
            raise AgenticStageError("S4 executed weights must be non-negative")
        total_weight = float(sum(book.values()))
        if total_weight > 1.0 + 1e-3:
            raise AgenticStageError(
                f"S4 executed weights must not sum above one (sum={total_weight:.6f})"
            )
        weights = book
        pre_risk = summarize_pre_action_risk(weights, return_data)
        context = {
            "weights": dict(weights),
            "pre_action_risk": pre_risk,
            "limits": {"var": var_limit, "drawdown": drawdown_limit, "weight_drift": drift_limit},
        }
        self.last_prompt = (
            "You are the portfolio risk-control stage. Use only the supplied Point-in-Time risk summary. "
            "Return a JSON object with action, alerts, corrective_weights, scale_factor, rationale. "
            "action must be hold, scale_down, or rebalance. "
            f"{S5_DECISION_POLICY} "
            f"Context: {json.dumps(context, ensure_ascii=False, default=str)}\n"
            f"{format_contract(self.use_tools)}"
        )
        bind_snapshot = snapshot_like
        if bind_snapshot is None:
            bind_snapshot = type(
                "LookbackSnapshot",
                (),
                {
                    "return_data": return_data,
                    "current_weights": dict(weights),
                    "price_data": {},
                    "decision_date": "",
                    "portfolio_value": 0.0,
                },
            )()
        try:
            decision, raw, parse_err = _complete_json(
                self.adapter,
                self.last_prompt,
                use_tools=self.use_tools,
                snapshot=bind_snapshot,
                parse=_decision_from_payload,
            )
        except Exception as exc:
            raise AgenticStageError(f"S5 model call failed: {exc}") from exc
        if decision is None:
            decision = RiskControlDecision(
                action="hold",
                metadata={"parse_error": str(parse_err), "rationale": ""},
            )
        decision.metadata.setdefault("schema_version", S4S5_SCHEMA_AGENTIC)
        try:
            eval_result = evaluate_risk(
                decision,
                weights,
                return_data,
                var_limit=var_limit,
                drawdown_limit=drawdown_limit,
                drift_limit=drift_limit,
            )
        except ValueError as exc:
            eval_result = evaluate_risk(
                RiskControlDecision(action="hold"),
                weights,
                return_data,
                var_limit=var_limit,
                drawdown_limit=drawdown_limit,
                drift_limit=drift_limit,
            )
            eval_result.metadata["execution_error"] = str(exc)
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
