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

def _is_complete_target_book(orders: list) -> bool:
    """True when orders name explicit targets and do not use delta overlays."""
    if not orders:
        return False
    has_target = False
    for intent in orders:
        if intent.target_weight is not None:
            has_target = True
        elif intent.delta_weight is not None:
            return False
    return has_target


def _project_to_simplex(
    weights: Mapping[str, float],
    current: Mapping[str, float],
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """Clip to [0, 1] and renormalize; hold current when mass is non-positive."""
    notes: Dict[str, Any] = {"raw_sum": float(sum(weights.values()))}
    clipped = {asset: min(1.0, max(0.0, float(weight))) for asset, weight in weights.items()}
    if any(clipped[asset] != float(weights[asset]) for asset in clipped):
        notes["clipped"] = True
    total = float(sum(clipped.values()))
    if total <= 0.0:
        notes["held_current"] = True
        return dict(current), notes
    if abs(total - 1.0) > 1e-3:
        notes["renormalized"] = True
        clipped = {asset: weight / total for asset, weight in clipped.items()}
    return clipped, notes


def _cash_like_key(assets: Any) -> str | None:
    """Prefer a bill/cash sleeve so residual mass has a home."""
    names = [str(asset) for asset in assets]
    for name in names:
        if name == "BIL":
            return name
    for name in names:
        if "BIL" in name or "cash" in name.lower():
            return name
    return None


def _enforce_fill_budget(
    current: Mapping[str, float],
    target: Mapping[str, float],
    fillable: set[str],
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """Apply only fillable legs, then conserve mass against the current book.

    Unfilled sells leave leftover holdings; filled buys are scaled so they
    cannot spend mass that those sells never released. Unfilled buys leave
    residual mass in a cash-like sleeve.
    """
    assets = set(current) | set(target)
    executed = {asset: float(current.get(asset, 0.0)) for asset in assets}
    applied: Dict[str, float] = {}
    for asset in fillable:
        new_weight = float(target.get(asset, 0.0))
        applied[asset] = new_weight - float(current.get(asset, 0.0))
        executed[asset] = new_weight
    net = float(sum(applied.values()))
    notes: Dict[str, Any] = {"fill_net_delta": round(net, 8)}
    if net > 1e-12:
        # Filled buys spent more mass than filled sells released.
        buy_assets = [asset for asset, delta in applied.items() if delta > 0.0]
        buy_mass = float(sum(applied[asset] for asset in buy_assets))
        if buy_mass > 1e-12:
            scale = max(0.0, buy_mass - net) / buy_mass
            notes["buy_fill_scale"] = round(scale, 8)
            for asset in buy_assets:
                executed[asset] = float(current.get(asset, 0.0)) + applied[asset] * scale
    elif net < -1e-12:
        # Filled sells released mass that filled buys did not absorb.
        cash_key = _cash_like_key(executed)
        if cash_key is None:
            cash_key = "CASH"
            executed[cash_key] = 0.0
        executed[cash_key] = float(executed.get(cash_key, 0.0)) - net
        notes["residual_to_cash"] = round(-net, 8)
        notes["cash_key"] = cash_key
    total = float(sum(executed.values()))
    if total > 1.0 + 1e-3:
        # Last-resort simplex if buy scaling could not absorb leftover sells.
        executed = {asset: weight / total for asset, weight in executed.items()}
        notes["renormalized_after_budget"] = True
    return executed, notes


def _round_filled_weights(executed: Mapping[str, float]) -> Dict[str, float]:
    """Round to four decimals without rounding overflow above one."""
    filled = {asset: round(float(weight), 4) for asset, weight in executed.items()}
    overflow = float(sum(filled.values()) - 1.0)
    if overflow <= 1e-12:
        return filled
    donor = _cash_like_key(filled)
    if donor is None or float(filled.get(donor, 0.0)) < overflow:
        donor = max(filled, key=filled.get)
    # Keep the leftover unrounded so four-decimal rounding cannot push the book back over one.
    filled[donor] = max(0.0, float(filled[donor]) - overflow)
    return filled


def resolve_target_weights(
    plan: ExecutionPlan,
    current_weights: Mapping[str, float],
) -> Dict[str, float]:
    """Derive target weights from plan orders, metadata, or scale fallback."""
    current = {str(k): float(v) for k, v in dict(current_weights).items()}
    orders = plan.normalized_orders() if hasattr(plan, "normalized_orders") else []

    if orders:
        if _is_complete_target_book(orders):
            # Explicit target_weight legs are a full book: unlisted holdings exit.
            assets = set(current)
            for intent in orders:
                if intent.asset:
                    assets.add(intent.asset)
            targets = {asset: 0.0 for asset in assets}
            for intent in orders:
                if intent.asset and intent.target_weight is not None:
                    targets[intent.asset] = float(intent.target_weight)
        else:
            # Delta or unsized legs overlay the current book so omitted names hold.
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
    """Deterministic fill with a cash budget so partial fills stay on simplex.

    Linear slippage and commission match legacy S4 economics. Cost drag is
    charged to a cash-like sleeve when one is present.
    """
    current = {str(k): float(v) for k, v in dict(current_weights).items()}
    nav_f = float(nav)
    if not nav_f > 0.0:
        raise ValueError("nav must be positive")
    simplex_notes: Dict[str, Any] = {}
    current_total = float(sum(current.values()))
    if any(weight < 0.0 for weight in current.values()) or current_total > 1.0 + 1e-3:
        # An illegal incoming book cannot be filled; project it before applying the plan.
        clipped_current = {asset: max(0.0, weight) for asset, weight in current.items()}
        current, current_notes = _project_to_simplex(clipped_current, clipped_current)
        simplex_notes["current_projected"] = True
        simplex_notes.update(current_notes)
    target = resolve_target_weights(plan, current)
    # Project invalid books onto the simplex so a bad S4 plan scores low instead of aborting the episode.
    off_simplex = any(not 0.0 <= weight <= 1.0 for weight in target.values()) or (
        bool(target) and abs(sum(target.values()) - 1.0) > 1e-3
    )
    if off_simplex:
        target, target_notes = _project_to_simplex(target, current)
        simplex_notes.update(target_notes)

    # Index declared intents by asset for direction and execution-policy validation.
    intents = {
        intent.asset: intent for intent in plan.normalized_orders() if intent.asset
    }

    all_assets = set(current) | set(target)
    fillable: set[str] = set()
    order_logs: list[dict] = []
    intended_trade: set[str] = set()

    for asset in sorted(all_assets):
        curr = current.get(asset, 0.0)
        targ = target.get(asset, 0.0)
        delta = targ - curr
        if abs(delta) < 1e-6:
            continue
        intended_trade.add(asset)
        direction = "buy" if delta > 0 else "sell"
        intent = intents.get(asset)
        if intent is not None and intent.direction not in {direction, "hold"}:
            raise ValueError(f"order direction conflicts with target for {asset}")
        if intent is not None and intent.direction == "hold":
            continue
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
        fillable.add(asset)

    unfilled_trade = intended_trade - fillable
    if unfilled_trade:
        executed, budget_notes = _enforce_fill_budget(current, target, fillable)
        simplex_notes.update(budget_notes)
        simplex_notes["unfilled_assets"] = sorted(unfilled_trade)
    else:
        executed = {asset: float(target.get(asset, current.get(asset, 0.0))) for asset in all_assets}

    buy_fill_scale = float(simplex_notes.get("buy_fill_scale", 1.0))
    total_cost = 0.0
    total_turnover = 0.0
    for asset in sorted(fillable):
        curr = current.get(asset, 0.0)
        actual = float(executed.get(asset, curr))
        actual_delta = actual - curr
        if abs(actual_delta) < 1e-6:
            continue
        direction = "buy" if actual_delta > 0 else "sell"
        intent = intents.get(asset)
        urgency = intent.urgency if intent is not None else plan.urgency
        urgency_multiplier = {"low": 0.75, "normal": 1.0, "high": 1.25}[urgency]
        slip_abs = slippage_rate * urgency_multiplier
        slip_limit = intent.slip_limit if intent is not None else plan.slip_limit
        order_type = intent.order_type if intent is not None else plan.order_type
        if slip_limit is not None:
            slip_abs = min(slip_abs, float(slip_limit))
        # Buy and sell orders apply slippage in opposite price directions.
        slip = slip_abs * (1 if direction == "buy" else -1)
        price = _price_for(snapshot_like, asset)
        trade_value = abs(actual_delta) * nav_f
        exec_price = price * (1 + slip)
        commission = trade_value * commission_rate
        total_cost += commission + trade_value * abs(slip)
        total_turnover += abs(actual_delta)
        intended_delta = float(target.get(asset, 0.0)) - curr
        status = "filled"
        if abs(actual_delta - intended_delta) > 1e-6:
            status = "partial_fill"
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
                "status": status,
            }
        )
    if buy_fill_scale < 1.0 - 1e-12:
        simplex_notes["partial_fills"] = True

    cost_drag = total_cost / nav_f if nav_f else 0.0
    cash_key = _cash_like_key(executed)
    if cash_key:
        # Charge total execution cost to cash without renormalizing risky assets.
        executed[cash_key] = max(0.0, float(executed[cash_key]) - cost_drag)

    filled = _round_filled_weights(executed)
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
            **simplex_notes,
        },
    )

__all__ = [
    "resolve_target_weights",
    "simulate_execution",
]
