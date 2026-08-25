"""Deterministic S5 risk-control environment."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import pandas as pd

from portbench.agent_eval.contracts import RiskControlDecision, RiskEvaluationResult
from portbench.metrics.base import MetricsConfig
from portbench.metrics.risk_metrics import cvar, max_drawdown, var


def _cash_proxy(
    weights: Mapping[str, float], available_assets: Mapping[str, Any] | None = None
) -> str | None:
    """Return a tradable cash proxy from holdings or available return data."""
    for candidates in (weights, available_assets or {}):
        for asset in candidates:
            if str(asset).upper() == "BIL":
                return str(asset)
        proxy = next(
            (
                str(asset)
                for asset in candidates
                if "BIL" in str(asset).upper() or "CASH" in str(asset).upper()
            ),
            None,
        )
        if proxy is not None:
            return proxy
    return None


def _portfolio_returns(
    weights: Mapping[str, float], return_data: Mapping[str, Any]
) -> pd.Series:
    """Build aligned portfolio returns and reject malformed asset data."""
    portfolio = pd.Series(dtype=float)
    for asset, weight in weights.items():
        # Ignore assets without a usable lookback series rather than fabricating returns.
        values = return_data.get(asset) if return_data else None
        if values is None:
            continue
        series = values.dropna() if hasattr(values, "dropna") else pd.Series(values, dtype=float).dropna()
        if series.empty:
            continue
        portfolio = portfolio.add(series.astype(float) * float(weight), fill_value=0.0)
    return portfolio.sort_index()


def _risk_metrics(
    weights: Mapping[str, float], return_data: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compute the risk metrics used before and after a control action."""
    portfolio = _portfolio_returns(weights, return_data)
    config = MetricsConfig(var_confidence=0.95)
    # Require enough observations for stable tail and drawdown calculations.
    if len(portfolio) > 10:
        portfolio_var = float(var(portfolio, config))
        portfolio_cvar = float(cvar(portfolio, config))
        portfolio_drawdown = float(max_drawdown(portfolio))
    else:
        portfolio_var = 0.0
        portfolio_cvar = 0.0
        portfolio_drawdown = 0.0
    active = [
        float(value) for asset, value in weights.items() if asset.upper() != "CASH"
    ]
    # Measure drift from an equal-weight active-asset reference portfolio.
    target = 1.0 / max(len(active), 1)
    drift = max((abs(value - target) for value in active), default=0.0)
    period_return = float((1.0 + portfolio).prod() - 1.0) if len(portfolio) else None
    return {
        "var": portfolio_var,
        "cvar": portfolio_cvar,
        "drawdown": portfolio_drawdown,
        "weight_drift": float(drift),
        "period_return": period_return,
    }


def _project_to_simplex(
    weights: Mapping[str, float],
    fallback: Mapping[str, float],
) -> Dict[str, float]:
    """Clip to non-negative and renormalize; keep fallback when mass is non-positive."""
    clipped = {str(asset): max(0.0, float(weight)) for asset, weight in weights.items()}
    total = float(sum(clipped.values()))
    if total <= 0.0:
        return {str(asset): float(weight) for asset, weight in fallback.items()}
    if abs(total - 1.0) > 1e-3:
        return {asset: weight / total for asset, weight in clipped.items()}
    return clipped


def _apply_decision(
    decision: RiskControlDecision,
    weights: Mapping[str, float],
    cash_proxy: str | None = None,
) -> Dict[str, float]:
    """Apply a validated risk-control decision to portfolio weights."""
    current = {str(asset): float(weight) for asset, weight in weights.items()}
    if decision.action == "hold":
        return current
    if decision.corrective_weights:
        # Explicit corrective weights take precedence over scale-down metadata.
        return _project_to_simplex(decision.corrective_weights, current)
    if decision.action != "scale_down" or decision.scale_factor is None:
        raise ValueError("rebalance requires corrective_weights")
    factor = float(decision.scale_factor)
    # Scale every current position before moving the residual into cash.
    final = {asset: weight * factor for asset, weight in current.items()}
    cash_key = cash_proxy or _cash_proxy(final)
    if cash_key is None:
        return _project_to_simplex(final, current)
    non_cash = sum(weight for asset, weight in final.items() if asset != cash_key)
    final[cash_key] = max(0.0, 1.0 - non_cash)
    return final


def _violations(
    metrics: Mapping[str, Any], var_limit: float, drawdown_limit: float, drift_limit: float
) -> list[str]:
    """Return breached constraints for one metric snapshot."""
    violations: list[str] = []
    if float(metrics["var"]) < var_limit:
        violations.append("var_breach")
    if float(metrics["drawdown"]) < drawdown_limit:
        violations.append("drawdown")
    if float(metrics["weight_drift"]) > drift_limit:
        violations.append("weight_drift")
    return violations


def evaluate_risk(
    decision: RiskControlDecision,
    weights: Mapping[str, float],
    return_data: Mapping[str, Any],
    *,
    var_limit: float = -0.02,
    drawdown_limit: float = -0.10,
    drift_limit: float = 0.05,
) -> RiskEvaluationResult:
    """Apply a control action and report post-action risk and economics."""
    initial = {str(asset): float(weight) for asset, weight in weights.items()}
    if any(weight < 0.0 for weight in initial.values()):
        raise ValueError("weights must be non-negative")
    total_weight = sum(initial.values())
    cash_proxy = _cash_proxy(initial, return_data)
    if total_weight > 1.0 + 1e-3:
        raise ValueError("weights must not sum above one")
    if initial and total_weight < 1.0:
        if cash_proxy is None:
            initial = _project_to_simplex(initial, initial)
        else:
            initial[cash_proxy] = initial.get(cash_proxy, 0.0) + 1.0 - total_weight
    # Compare risk before and after the action using the same lookback data.
    pre = _risk_metrics(initial, return_data)
    final = _apply_decision(decision, initial, cash_proxy=cash_proxy)
    post = _risk_metrics(final, return_data)
    violations = _violations(post, var_limit, drawdown_limit, drift_limit)
    # Persist both sides of the deterministic action for plan and outcome scoring.
    return RiskEvaluationResult(
        var=round(float(post["var"]), 6),
        cvar=round(float(post["cvar"]), 6),
        drawdown=round(float(post["drawdown"]), 6),
        constraint_violations=violations,
        metadata={
            "pre_action": pre,
            "post_action": post,
            "weight_drift": round(float(post["weight_drift"]), 6),
            "final_weights": {asset: round(weight, 8) for asset, weight in final.items()},
            "action_applied": decision.action,
            "period_return": post["period_return"],
            "var_limit": var_limit,
            "drawdown_limit": drawdown_limit,
            "drift_limit": drift_limit,
            "alerts_noted": list(decision.alerts or []),
        },
    )


def summarize_pre_action_risk(
    weights: Mapping[str, float], return_data: Mapping[str, Any]
) -> Dict[str, Any]:
    """Expose the information available to the S5 agent before acting."""
    return _risk_metrics(weights, return_data)


__all__ = ["evaluate_risk", "summarize_pre_action_risk"]
