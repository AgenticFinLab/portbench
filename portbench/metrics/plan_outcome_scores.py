"""Dual-layer S4/S5 scoring: plan quality vs environment outcome.

Plan-quality keys and environment-outcome keys must never be mixed into a
single unlabeled ranking column. Use ``portbench.agent_eval.result_gates.assert_homogeneous_schema_versions``
(or ``build_ranking_rows``) when assembling comparison tables.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from portbench.agent_eval.contracts import (
    S4S5_SCHEMA_AGENTIC,
    S4S5_SCHEMA_DETERMINISTIC,
    ExecutionPlan,
    ExecutionResult,
    RiskControlDecision,
    RiskEvaluationResult,
)

S4_PLAN_QUALITY_KEYS: frozenset[str] = frozenset(
    {"order_legality", "target_tracking", "plan_quality"}
)
S4_ENV_OUTCOME_KEYS: frozenset[str] = frozenset(
    {"implementation_shortfall", "cost", "turnover", "filled_weight_error"}
)
S5_PLAN_QUALITY_KEYS: frozenset[str] = frozenset(
    {"alert_identification", "action_choice", "corrective_compliance"}
)
S5_ENV_OUTCOME_KEYS: frozenset[str] = frozenset(
    {"cvar", "drawdown", "violation", "period_return"}
)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _mean(vals: Sequence[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _weights_from_plan(plan: ExecutionPlan) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for intent in plan.normalized_orders():
        if intent.target_weight is not None:
            out[intent.asset] = float(intent.target_weight)
        elif intent.delta_weight is not None:
            out[intent.asset] = float(intent.delta_weight)
    meta = plan.metadata or {}
    if isinstance(meta.get("target_weights"), Mapping):
        for a, w in meta["target_weights"].items():
            out[str(a)] = float(w)
    return out


def _l1_tracking(actual: Mapping[str, float], target: Mapping[str, float]) -> float:
    keys = set(actual) | set(target)
    if not keys:
        return 1.0
    err = sum(abs(float(actual.get(k, 0.0)) - float(target.get(k, 0.0))) for k in keys)
    # Two-sided L1 on weights is in [0, 2]; map to [0, 1] score
    return _clip01(1.0 - err / 2.0)


def score_s4_plan_quality(
    plan: ExecutionPlan,
    reference_plan: Optional[ExecutionPlan] = None,
    target_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Subscores for the agent execution *plan* (not the fill)."""
    orders = plan.normalized_orders()
    legal = 0.0
    if not orders and not (plan.metadata or {}).get("target_weights"):
        # Empty plan is legal only if reference is also empty / scale-only
        legal = 1.0 if reference_plan is None or not reference_plan.normalized_orders() else 0.5
    else:
        ok = 0
        n = 0
        for o in orders:
            n += 1
            direction_ok = o.direction in ("buy", "sell", "hold")
            has_size = (
                o.target_weight is not None
                or o.delta_weight is not None
                or o.direction == "hold"
            )
            asset_ok = bool(o.asset)
            weight_ok = o.target_weight is None or 0.0 <= float(o.target_weight) <= 1.0
            slip_ok = o.slip_limit is None or float(o.slip_limit) >= 0.0
            order_type_ok = o.order_type in {"market", "limit"}
            if direction_ok and has_size and asset_ok and weight_ok and slip_ok and order_type_ok:
                ok += 1
        if n == 0 and (plan.metadata or {}).get("target_weights"):
            legal = 1.0
        else:
            legal = ok / n if n else 0.0

    planned_for_validation = _weights_from_plan(plan)
    if planned_for_validation and abs(sum(planned_for_validation.values()) - 1.0) > 1e-4:
        legal = 0.0

    planned = _weights_from_plan(plan)
    ref_w: Dict[str, float] = {}
    if target_weights is not None:
        ref_w = {str(k): float(v) for k, v in dict(target_weights).items()}
    elif reference_plan is not None:
        ref_w = _weights_from_plan(reference_plan)
    tracking = _l1_tracking(planned, ref_w) if ref_w or planned else 1.0

    plan_quality = _mean([legal, tracking])
    return {
        "order_legality": round(legal, 6),
        "target_tracking": round(tracking, 6),
        "plan_quality": round(plan_quality, 6),
    }


def score_s4_environment_outcome(
    result: ExecutionResult,
    reference: ExecutionResult,
) -> Dict[str, float]:
    """Subscores for deterministic execution *outcome* vs reference fill."""
    # Higher is better for all returned keys.
    ref_cost = abs(float(reference.cost)) + 1e-9
    cost_score = _clip01(1.0 - abs(float(result.cost) - float(reference.cost)) / max(ref_cost, abs(float(result.cost)), 1e-9))

    ref_to = abs(float(reference.turnover)) + 1e-9
    to_score = _clip01(
        1.0
        - abs(float(result.turnover) - float(reference.turnover))
        / max(ref_to, abs(float(result.turnover)), 1e-4)
    )

    ref_sf = abs(float(reference.implementation_shortfall)) + 1e-12
    sf_score = _clip01(
        1.0
        - abs(float(result.implementation_shortfall) - float(reference.implementation_shortfall))
        / max(ref_sf, abs(float(result.implementation_shortfall)), 1e-12)
    )

    w_err = 1.0 - _l1_tracking(result.filled_weights, reference.filled_weights)
    # Convert L1 weight error into a similarity score where 1 means perfect match.
    filled_match = _clip01(1.0 - w_err)

    return {
        "implementation_shortfall": round(sf_score, 6),
        "cost": round(cost_score, 6),
        "turnover": round(to_score, 6),
        "filled_weight_error": round(filled_match, 6),
    }


def _alert_tokens(decision: RiskControlDecision) -> Set[str]:
    tokens: Set[str] = set()
    for a in decision.alerts or []:
        if isinstance(a, str):
            tokens.add(a.lower())
        elif isinstance(a, Mapping):
            for key in ("metric", "type", "name", "alert"):
                if key in a:
                    tokens.add(str(a[key]).lower())
                    break
            else:
                tokens.add(str(a).lower())
        else:
            metric = getattr(a, "metric", None)
            tokens.add(str(metric).lower() if metric is not None else str(a).lower())
    return tokens


def score_s5_plan_quality(
    decision: RiskControlDecision,
    reference_decision: RiskControlDecision,
) -> Dict[str, float]:
    """Subscores for the agent risk-control *decision*."""
    act = (decision.action or "hold").lower()
    ref_act = (reference_decision.action or "hold").lower()
    action_choice = 1.0 if act == ref_act else 0.0

    pred = _alert_tokens(decision)
    ref = _alert_tokens(reference_decision)
    if not ref and not pred:
        alert_id = 1.0
    elif not ref:
        alert_id = 1.0 if not pred else 0.5
    else:
        alert_id = len(pred & ref) / len(ref)

    # Corrective compliance: if reference requires weights, measure L1 match
    ref_cw = reference_decision.corrective_weights
    cw = decision.corrective_weights
    if ref_cw:
        if not cw:
            corrective = 0.0
        else:
            corrective = _l1_tracking(cw, ref_cw)
    else:
        corrective = 1.0 if not cw or act == "hold" else 0.8

    return {
        "alert_identification": round(_clip01(alert_id), 6),
        "action_choice": round(action_choice, 6),
        "corrective_compliance": round(_clip01(corrective), 6),
    }


def score_s5_environment_outcome(
    eval_result: RiskEvaluationResult,
    reference: RiskEvaluationResult,
) -> Dict[str, float]:
    """Subscores for deterministic risk *evaluation* vs reference."""

    def _num_score(a: Optional[float], b: Optional[float]) -> float:
        if a is None and b is None:
            return 1.0
        if a is None or b is None:
            return 0.0
        denom = max(abs(float(b)), abs(float(a)), 1e-6)
        return _clip01(1.0 - abs(float(a) - float(b)) / denom)

    cvar_s = _num_score(eval_result.cvar, reference.cvar)
    dd_s = _num_score(eval_result.drawdown, reference.drawdown)

    v_a = set(eval_result.constraint_violations or [])
    v_b = set(reference.constraint_violations or [])
    if not v_a and not v_b:
        viol_s = 1.0
    else:
        union = v_a | v_b
        viol_s = len(v_a & v_b) / len(union) if union else 1.0

    pr_a = (eval_result.metadata or {}).get("period_return")
    pr_b = (reference.metadata or {}).get("period_return")
    ret_s = _num_score(
        float(pr_a) if pr_a is not None else None,
        float(pr_b) if pr_b is not None else None,
    )

    return {
        "cvar": round(cvar_s, 6),
        "drawdown": round(dd_s, 6),
        "violation": round(viol_s, 6),
        "period_return": round(ret_s, 6),
    }

__all__ = [
    "S4_PLAN_QUALITY_KEYS",
    "S4_ENV_OUTCOME_KEYS",
    "S5_PLAN_QUALITY_KEYS",
    "S5_ENV_OUTCOME_KEYS",
    "S4S5_SCHEMA_DETERMINISTIC",
    "S4S5_SCHEMA_AGENTIC",
    "score_s4_plan_quality",
    "score_s4_environment_outcome",
    "score_s5_plan_quality",
    "score_s5_environment_outcome",
]
