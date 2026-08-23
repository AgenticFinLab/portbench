"""Bridge legacy S4/S5 outputs to dual-layer plan + environment contracts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from portbench.agent_eval.contracts import (
    S4S5_SCHEMA_DETERMINISTIC,
    ExecutionPlan,
    ExecutionResult,
    OrderIntent,
    RiskControlDecision,
    RiskEvaluationResult,
)
from portbench.sandbox.execution import (
    _implied_direction,
    _prepare_fill_books,
    simulate_execution,
)


def _snapshot_current_weights(snapshot_ctx: Any) -> Dict[str, float]:
    if snapshot_ctx is None:
        return {}
    if hasattr(snapshot_ctx, "current_weights"):
        return dict(snapshot_ctx.current_weights or {})
    if isinstance(snapshot_ctx, Mapping):
        return dict(snapshot_ctx.get("current_weights") or {})
    return {}


def _snapshot_nav(snapshot_ctx: Any, default: float = 100_000.0) -> float:
    if snapshot_ctx is None:
        return default
    if hasattr(snapshot_ctx, "portfolio_value"):
        return float(snapshot_ctx.portfolio_value)
    if isinstance(snapshot_ctx, Mapping):
        return float(snapshot_ctx.get("portfolio_value", default))
    return default


def legacy_s4_to_plan_and_result(
    s4_output: Any,
    snapshot_ctx: Any = None,
) -> Tuple[ExecutionPlan, ExecutionResult]:
    """Reconstruct an ExecutionPlan that yields the legacy executed weights.

    Orders are built from ``current_weights → executed_weights`` deltas.
    Schema tag: ``pipeline-v1-deterministic``.
    """
    executed = dict(getattr(s4_output, "executed_weights", None) or {})
    current = _snapshot_current_weights(snapshot_ctx)
    if not current and hasattr(s4_output, "orders"):
        # Best-effort: infer current ≈ executed - signed deltas from orders
        current = dict(executed)

    orders: list[OrderIntent] = []
    assets = set(current) | set(executed)
    for asset in sorted(assets):
        curr = float(current.get(asset, 0.0))
        targ = float(executed.get(asset, 0.0))
        delta = targ - curr
        if abs(delta) < 1e-6:
            continue
        direction = "buy" if delta > 0 else "sell"
        orders.append(
            OrderIntent(
                asset=asset,
                direction=direction,
                target_weight=targ,
                delta_weight=delta,
                order_type="market",
            )
        )

    plan = ExecutionPlan(
        order_direction=orders[0].direction if orders else "",
        target_scale=1.0,
        order_type="market",
        orders=orders,
        metadata={
            "target_weights": executed,
            "schema_version": S4S5_SCHEMA_DETERMINISTIC,
            "source": "legacy_s4",
        },
    )

    result = ExecutionResult(
        filled_weights={a: round(float(w), 4) for a, w in executed.items()},
        implementation_shortfall=float(getattr(s4_output, "total_cost", 0.0) or 0.0)
        / max(_snapshot_nav(snapshot_ctx), 1e-9),
        turnover=float(getattr(s4_output, "turnover", 0.0) or 0.0),
        cost=float(getattr(s4_output, "total_cost", 0.0) or 0.0),
        metadata={
            "schema_version": S4S5_SCHEMA_DETERMINISTIC,
            "source": "legacy_s4",
            "legacy_orders": [
                {
                    "asset": getattr(o, "asset", None),
                    "direction": getattr(o, "direction", None),
                    "quantity": getattr(o, "quantity", None),
                    "commission": getattr(o, "commission", None),
                    "slippage": getattr(o, "slippage", None),
                }
                for o in (getattr(s4_output, "orders", None) or [])
            ],
        },
    )
    return plan, result


def legacy_s5_to_decision_and_eval(
    s5_output: Any,
) -> Tuple[RiskControlDecision, RiskEvaluationResult]:
    """Map legacy S5Output to RiskControlDecision + RiskEvaluationResult."""
    alerts_raw = list(getattr(s5_output, "alerts", None) or [])
    alert_names: list[Any] = []
    needs_rebalance = bool(getattr(s5_output, "rebalance_needed", False))
    for a in alerts_raw:
        metric = getattr(a, "metric", None)
        if metric is not None:
            alert_names.append(str(metric))
        elif isinstance(a, Mapping):
            alert_names.append(str(a.get("metric", a)))
        else:
            alert_names.append(str(a))

    if needs_rebalance:
        action = "rebalance"
    elif any("var" in str(x).lower() for x in alert_names):
        action = "scale_down"
    else:
        action = "hold"

    decision = RiskControlDecision(
        action=action,
        alerts=alert_names,
        corrective_weights=None,
        metadata={
            "schema_version": S4S5_SCHEMA_DETERMINISTIC,
            "source": "legacy_s5",
            "rebalance_needed": needs_rebalance,
        },
    )

    violations = list(alert_names)
    eval_result = RiskEvaluationResult(
        var=float(getattr(s5_output, "portfolio_var", 0.0) or 0.0),
        cvar=None,
        drawdown=float(getattr(s5_output, "portfolio_drawdown", 0.0) or 0.0),
        constraint_violations=violations,
        metadata={
            "schema_version": S4S5_SCHEMA_DETERMINISTIC,
            "source": "legacy_s5",
            "weight_drift": float(getattr(s5_output, "weight_drift", 0.0) or 0.0),
            "final_weights": None,
        },
    )
    return decision, eval_result


def run_s4_deterministic_from_weights(
    target_weights: Mapping[str, float],
    snapshot_like: Any,
    current_weights: Mapping[str, float],
    nav: float,
    *,
    slippage_rate: float = 0.001,
    commission_rate: float = 0.0005,
) -> Tuple[ExecutionPlan, ExecutionResult]:
    """Convenience v1 path: build plan from target weights and run env.

    Order directions come from the same projected fill book the simulator
    will trade, so the reference plan is self-consistent.
    """
    original = {str(k): float(v) for k, v in dict(target_weights).items()}
    stub = ExecutionPlan(
        orders=[
            OrderIntent(asset=asset, direction="hold", target_weight=weight)
            for asset, weight in original.items()
        ],
        metadata={"target_weights": original},
    )
    current, fill_book, _ = _prepare_fill_books(stub, current_weights or {})
    orders = []
    for asset in sorted(set(current) | set(fill_book)):
        curr = float(current.get(asset, 0.0))
        targ = float(fill_book.get(asset, 0.0))
        delta = targ - curr
        orders.append(
            OrderIntent(
                asset=str(asset),
                direction=_implied_direction(delta),
                target_weight=targ,
                delta_weight=delta,
            )
        )

    plan = ExecutionPlan(
        orders=orders,
        metadata={
            "target_weights": original,
            "schema_version": S4S5_SCHEMA_DETERMINISTIC,
            "source": "run_s4_deterministic_from_weights",
        },
    )
    result = simulate_execution(
        plan,
        snapshot_like,
        current_weights,
        nav,
        slippage_rate=slippage_rate,
        commission_rate=commission_rate,
    )
    result.metadata["schema_version"] = S4S5_SCHEMA_DETERMINISTIC
    return plan, result


__all__ = [
    "legacy_s4_to_plan_and_result",
    "legacy_s5_to_decision_and_eval",
    "run_s4_deterministic_from_weights",
]
