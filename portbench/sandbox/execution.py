"""Deterministic S4/S5 environment outcomes (no LLM).

Pure functions that mirror legacy ``S4ExecutionSimulation._execute`` and
``S5RiskMonitoring._monitor`` economics so agent plans can be scored
separately from environment fill / risk evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from portbench.agent_eval.contracts import ExecutionPlan, ExecutionResult


def _price_for(snapshot_like: Any, asset: str, default: float = 100.0) -> float:
    """Return the latest available price from an object or mapping snapshot."""
    # Support live MarketSnapshot objects and serialized step-replay mappings.
    price_data = getattr(snapshot_like, "price_data", None)
    if price_data is None and isinstance(snapshot_like, Mapping):
        price_data = snapshot_like.get("price_data")
    if not price_data:
        return default
    series = price_data.get(asset) if hasattr(price_data, "get") else None
    if series is None:
        return default
    try:
        if hasattr(series, "empty") and series.empty:
            return default
        return float(series.iloc[-1])
    except Exception:
        try:
            return float(series[-1])
        except Exception:
            return default

def resolve_target_weights(
    plan: ExecutionPlan,
    current_weights: Mapping[str, float],
) -> Dict[str, float]:
    """Derive target weights from plan orders, metadata, or scale fallback."""
    current = {str(k): float(v) for k, v in dict(current_weights).items()}
    orders = plan.normalized_orders() if hasattr(plan, "normalized_orders") else []

    if orders:
        # Start from current holdings so omitted assets remain unchanged.
        targets = dict(current)
        for intent in orders:
            asset = intent.asset
            if not asset:
                continue
            if intent.target_weight is not None:
                targets[asset] = float(intent.target_weight)
            elif intent.delta_weight is not None:
                targets[asset] = float(current.get(asset, 0.0) + intent.delta_weight)
            elif intent.direction == "hold":
                targets.setdefault(asset, float(current.get(asset, 0.0)))
            # Leave unsized buy or sell orders unchanged unless metadata supplies a target.
        meta_tw = plan.metadata.get("target_weights") if plan.metadata else None
        if isinstance(meta_tw, Mapping):
            for a, w in meta_tw.items():
                targets[str(a)] = float(w)
        return targets

    meta = plan.metadata or {}
    if isinstance(meta.get("target_weights"), Mapping):
        return {str(k): float(v) for k, v in meta["target_weights"].items()}

    # Scale risky weights and let a cash-like asset absorb the residual.
    scale = float(plan.target_scale) if plan.target_scale is not None else 1.0
    if abs(scale - 1.0) < 1e-12:
        return dict(current)
    cash_key = next(
        (a for a in current if "BIL" in a or "cash" in a.lower()),
        None,
    )
    if cash_key is not None:
        # Recompute non-cash positions before assigning the remaining mass to cash.
        targets = {
            a: (w * scale if a != cash_key else 0.0) for a, w in current.items()
        }
        non_cash_sum = sum(targets.values())
        targets[cash_key] = max(0.0, 1.0 - non_cash_sum)
    else:
        targets = {a: w * scale for a, w in current.items()}
    return targets


def simulate_execution(
    plan: ExecutionPlan,
    snapshot_like: Any,
    current_weights: Mapping[str, float],
    nav: float,
    *,
    slippage_rate: float = 0.001,
    commission_rate: float = 0.0005,
) -> ExecutionResult:
    """Deterministic fill mirroring legacy S4 ``_execute`` economics.

    Linear slippage, commission on trade value, cost drag applied to a
    cash-like asset (BIL / *cash*) when present.
    """
    current = {str(k): float(v) for k, v in dict(current_weights).items()}
    target = resolve_target_weights(plan, current)
    nav_f = float(nav)
    if not nav_f > 0.0:
        raise ValueError("nav must be positive")
    if any(not 0.0 <= weight <= 1.0 for weight in target.values()):
        raise ValueError("target weights must be in [0, 1]")
    if target and abs(sum(target.values()) - 1.0) > 1e-3:
        raise ValueError("target weights must sum to one")

    # Index declared intents by asset for direction and execution-policy validation.
    intents = {
        intent.asset: intent for intent in plan.normalized_orders() if intent.asset
    }

    executed = dict(target)
    total_cost = 0.0
    total_turnover = 0.0
    order_logs: list[dict] = []

    all_assets = set(current) | set(target)
    for asset in sorted(all_assets):
        # Skip unchanged positions before computing price or cost data.
        curr = current.get(asset, 0.0)
        targ = target.get(asset, 0.0)
        delta = targ - curr
        if abs(delta) < 1e-6:
            continue

        direction = "buy" if delta > 0 else "sell"
        intent = intents.get(asset)
        if intent is not None and intent.direction not in {direction, "hold"}:
            raise ValueError(f"order direction conflicts with target for {asset}")
        if intent is not None and intent.direction == "hold":
            executed[asset] = curr
            continue
        # Convert the target-weight change into traded notional at current NAV.
        trade_value = abs(delta) * nav_f
        price = _price_for(snapshot_like, asset)
        urgency = intent.urgency if intent is not None else plan.urgency
        urgency_multiplier = {"low": 0.75, "normal": 1.0, "high": 1.25}.get(urgency)
        if urgency_multiplier is None:
            raise ValueError(f"invalid urgency: {urgency!r}")
        slip_abs = slippage_rate * urgency_multiplier
        slip_limit = intent.slip_limit if intent is not None else plan.slip_limit
        order_type = intent.order_type if intent is not None else plan.order_type
        # A limit order remains unfilled when modeled slippage exceeds its cap.
        if (
            order_type == "limit"
            and slip_limit is not None
            and slip_abs > float(slip_limit)
        ):
            executed[asset] = curr
            order_logs.append(
                {
                    "asset": asset,
                    "direction": direction,
                    "status": "unfilled_limit",
                    "slippage": slip_abs,
                    "slip_limit": float(slip_limit),
                }
            )
            continue
        if slip_limit is not None:
            slip_abs = min(slip_abs, float(slip_limit))
        # Buy and sell orders apply slippage in opposite price directions.
        slip = slip_abs * (1 if direction == "buy" else -1)
        exec_price = price * (1 + slip)
        commission = trade_value * commission_rate
        total_cost += commission + trade_value * abs(slip)
        total_turnover += abs(delta)
        order_logs.append(
            {
                "asset": asset,
                "direction": direction,
                "quantity": trade_value,
                "price": exec_price,
                "slippage": slip,
                "commission": commission,
                "order_type": order_type,
                "urgency": urgency,
                "status": "filled",
            }
        )

    cost_drag = total_cost / nav_f if nav_f else 0.0
    cash_key = next(
        (a for a in executed if "BIL" in a or "cash" in a.lower()),
        None,
    )
    if cash_key:
        # Charge total execution cost to cash without renormalizing risky assets.
        executed[cash_key] = max(0.0, executed[cash_key] - cost_drag)

    filled = {a: round(w, 4) for a, w in executed.items()}
    # Report implementation shortfall as fractional cost drag on NAV.
    shortfall = round(cost_drag, 8)

    return ExecutionResult(
        filled_weights=filled,
        implementation_shortfall=shortfall,
        turnover=round(total_turnover, 4),
        cost=round(total_cost, 4),
        metadata={
            "orders": order_logs,
            "target_weights": {a: round(w, 6) for a, w in target.items()},
            "slippage_rate": slippage_rate,
            "commission_rate": commission_rate,
            "nav": nav_f,
        },
    )

__all__ = [
    "resolve_target_weights",
    "simulate_execution",
]
