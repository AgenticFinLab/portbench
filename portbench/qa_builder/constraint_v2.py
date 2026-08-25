"""Deterministic constraint-v2 ground truth for the redesigned T3 and T4 tasks."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


TEMPLATE_VERSION = "constraint-v2"
T3_BINDING_ORDER = ("var", "es", "drawdown", "liquidity", "full_allocation")


def source_snapshot_provenance(context: Any) -> dict[str, Any]:
    """Hash the Point-in-Time observations used to build one constraint item."""
    context.validate_pit()
    latest_observation: dict[str, str] = {}
    history: dict[str, list[list[str | float]]] = {}
    for asset in sorted(context.assets):
        series = context.returns_history[asset].dropna()
        latest_observation[asset] = str(series.index.max()) if not series.empty else ""
        history[asset] = [
            [str(index), round(float(value), 12)]
            for index, value in series.items()
        ]
    payload = {
        "decision_date": str(context.decision_date),
        "assets": sorted(context.assets),
        "returns_history": history,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "source_snapshot_hash": digest,
        "pit_audit": {
            "decision_date": str(context.decision_date),
            "latest_return_observation": latest_observation,
            "validated": True,
        },
    }


def t3_solution(
    unit_risk: Mapping[str, float],
    budgets: Mapping[str, float],
    liquidity_cap: float,
) -> dict[str, Any]:
    """Solve the visible multi-constraint position-sizing rule exactly."""
    limits: dict[str, float] = {}
    for name in ("var", "es", "drawdown"):
        risk = float(unit_risk.get(name, 0.0))
        budget = float(budgets.get(name, 0.0))
        limits[name] = budget / risk if risk > 0.0 else float("inf")
    limits["liquidity"] = float(liquidity_cap)
    limits["full_allocation"] = 1.0
    position = min(limits.values())
    binding = next(name for name in T3_BINDING_ORDER if abs(limits[name] - position) <= 1e-8)
    margins = {name: round(max(0.0, limit - position), 6) for name, limit in limits.items()}
    return {
        "position_size": round(max(0.0, min(1.0, position)), 4),
        "binding_constraint": binding,
        "constraint_limits": {name: round(limit, 6) for name, limit in limits.items()},
        "constraint_margins": margins,
    }


def t4_metrics(
    weights: Mapping[str, float],
    assets: list[str],
    expected_returns: Mapping[str, float],
    covariance: list[list[float]],
    current_weights: Mapping[str, float],
) -> dict[str, float]:
    """Compute visible candidate return, variance, and one-way turnover."""
    first, second = assets
    w1 = float(weights[first])
    w2 = float(weights[second])
    mu = w1 * float(expected_returns[first]) + w2 * float(expected_returns[second])
    variance = (
        w1 * w1 * float(covariance[0][0])
        + w2 * w2 * float(covariance[1][1])
        + 2.0 * w1 * w2 * float(covariance[0][1])
    )
    turnover = 0.5 * sum(
        abs(float(weights.get(asset, 0.0)) - float(current_weights.get(asset, 0.0)))
        for asset in assets
    )
    return {
        "expected_return": round(mu, 6),
        "variance": round(variance, 8),
        "turnover": round(turnover, 6),
    }


def t4_solution(
    candidates: list[Mapping[str, Any]],
    *,
    return_floor: float,
    turnover_cap: float,
) -> dict[str, Any]:
    """Select the minimum-variance feasible candidate with frozen tie-breaks."""
    feasible: list[Mapping[str, Any]] = []
    for candidate in candidates:
        weights = dict(candidate["weights"])
        metrics = dict(candidate["metrics"])
        valid_weights = all(float(value) >= -1e-8 for value in weights.values()) and abs(
            sum(float(value) for value in weights.values()) - 1.0
        ) <= 1e-6
        valid = (
            valid_weights
            and float(metrics["expected_return"]) >= float(return_floor) - 1e-8
            and float(metrics["turnover"]) <= float(turnover_cap) + 1e-8
        )
        if valid:
            feasible.append(candidate)
    if not feasible:
        raise ValueError("constraint-v2 T4 requires at least one feasible candidate")
    selected = min(
        feasible,
        key=lambda item: (
            float(item["metrics"]["variance"]),
            -float(item["metrics"]["expected_return"]),
            float(item["metrics"]["turnover"]),
            str(item["candidate_id"]),
        ),
    )
    metrics = dict(selected["metrics"])
    binding = []
    if abs(float(metrics["expected_return"]) - float(return_floor)) <= 1e-6:
        binding.append("return_floor")
    if abs(float(metrics["turnover"]) - float(turnover_cap)) <= 1e-6:
        binding.append("turnover_cap")
    return {
        "candidate_id": str(selected["candidate_id"]),
        "weights": {str(key): round(float(value), 4) for key, value in selected["weights"].items()},
        "metrics": metrics,
        "binding_constraints": binding,
    }


__all__ = [
    "TEMPLATE_VERSION",
    "T3_BINDING_ORDER",
    "source_snapshot_provenance",
    "t3_solution",
    "t4_metrics",
    "t4_solution",
]
