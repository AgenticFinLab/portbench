"""Deterministic ground truth for fast constraint-decision QA templates."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping


TEMPLATE_VERSION = "constraint-decision-v2"
TEMPLATE_VERSIONS = frozenset({TEMPLATE_VERSION})
T3_CONSTRAINT_ORDER = ("var", "es", "drawdown", "liquidity", "cash_reserve")
T4_CONSTRAINT_ORDER = (
    "net_return_floor",
    "variance_cap",
    "turnover_cap",
    "concentration_cap",
)


def deterministic_rng(source_snapshot_hash: str, seq: int, namespace: str) -> random.Random:
    """Create a stable item generator from visible source provenance."""
    payload = f"{source_snapshot_hash}:{seq}:{namespace}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(seed)


def _is_feasible(margins: Mapping[str, float]) -> bool:
    """Return whether every verified post-trade constraint margin is non-negative."""
    return all(float(value) >= -1e-9 for value in margins.values())


def _binding_constraint(margins: Mapping[str, float], order: tuple[str, ...]) -> str:
    """Return the tightest verified constraint using a fixed deterministic tie-break."""
    return min(order, key=lambda name: (float(margins[name]), order.index(name)))


def _rank_t3(plan: Mapping[str, Any]) -> tuple[float, float, float, str]:
    """Rank feasible T3 decisions by exposure, benefit, turnover, then ID."""
    return (
        -float(plan["target_position"]),
        -float(plan["net_benefit"]),
        float(plan["turnover"]),
        str(plan["plan_id"]),
    )


def _rank_t4(candidate: Mapping[str, Any]) -> tuple[float, float, float, str]:
    """Rank feasible T4 decisions by variance, return, turnover, then ID."""
    return (
        float(candidate["portfolio_variance"]),
        -float(candidate["net_return"]),
        float(candidate["turnover"]),
        str(candidate["candidate_id"]),
    )


def _selected_id(items: list[dict[str, Any]], key, id_field: str) -> str:
    """Return the selected ID or HOLD when no candidate is feasible."""
    if not items:
        return "HOLD"
    return str(min(items, key=key)[id_field])


def t3_decision_solution(plans: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Select base and ES-stressed T3 plans from pre-verified risk margins."""
    assessments: list[dict[str, Any]] = []
    for plan in plans:
        base_margins = {
            name: round(float(dict(plan["base_margins"])[name]), 6)
            for name in T3_CONSTRAINT_ORDER
        }
        stress_margins = dict(base_margins)
        stress_margins["es"] = round(
            stress_margins["es"] - float(plan["es_stress_charge"]),
            6,
        )
        assessments.append(
            {
                "plan_id": str(plan["plan_id"]),
                "target_position": float(plan["target_position"]),
                "net_benefit": float(plan["net_benefit"]),
                "turnover": float(plan["turnover"]),
                "base_margins": base_margins,
                "stress_margins": stress_margins,
                "base_feasible": _is_feasible(base_margins),
                "stress_feasible": _is_feasible(stress_margins),
                "base_binding_constraint": _binding_constraint(
                    base_margins,
                    T3_CONSTRAINT_ORDER,
                ),
                "stress_binding_constraint": _binding_constraint(
                    stress_margins,
                    T3_CONSTRAINT_ORDER,
                ),
            }
        )
    base_feasible = [item for item in assessments if item["base_feasible"]]
    stress_feasible = [item for item in assessments if item["stress_feasible"]]
    base_selected_id = _selected_id(base_feasible, _rank_t3, "plan_id")
    stress_selected_id = _selected_id(stress_feasible, _rank_t3, "plan_id")
    by_id = {item["plan_id"]: item for item in assessments}
    return {
        "base_selected_id": base_selected_id,
        "stress_selected_id": stress_selected_id,
        "base_feasible_ids": sorted(item["plan_id"] for item in base_feasible),
        "stress_feasible_ids": sorted(item["plan_id"] for item in stress_feasible),
        "base_binding_constraint": (
            "none"
            if base_selected_id == "HOLD"
            else by_id[base_selected_id]["base_binding_constraint"]
        ),
        "stress_binding_constraint": (
            "none"
            if stress_selected_id == "HOLD"
            else by_id[stress_selected_id]["stress_binding_constraint"]
        ),
        "assessments": assessments,
    }


def t4_decision_solution(candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Select base and stressed T4 portfolios from pre-verified constraint margins."""
    assessments: list[dict[str, Any]] = []
    for candidate in candidates:
        base_margins = {
            name: round(float(dict(candidate["base_margins"])[name]), 6)
            for name in T4_CONSTRAINT_ORDER
        }
        stress_margins = dict(base_margins)
        stress_margins["net_return_floor"] = round(
            stress_margins["net_return_floor"] - float(candidate["return_stress_charge"]),
            6,
        )
        stress_margins["turnover_cap"] = round(
            stress_margins["turnover_cap"] - float(candidate["liquidity_stress_charge"]),
            6,
        )
        assessments.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "portfolio_variance": float(candidate["portfolio_variance"]),
                "net_return": float(candidate["net_return"]),
                "turnover": float(candidate["turnover"]),
                "base_margins": base_margins,
                "stress_margins": stress_margins,
                "base_feasible": _is_feasible(base_margins),
                "stress_feasible": _is_feasible(stress_margins),
                "base_binding_constraint": _binding_constraint(
                    base_margins,
                    T4_CONSTRAINT_ORDER,
                ),
                "stress_binding_constraint": _binding_constraint(
                    stress_margins,
                    T4_CONSTRAINT_ORDER,
                ),
            }
        )
    base_feasible = [item for item in assessments if item["base_feasible"]]
    stress_feasible = [item for item in assessments if item["stress_feasible"]]
    base_selected_id = _selected_id(base_feasible, _rank_t4, "candidate_id")
    stress_selected_id = _selected_id(stress_feasible, _rank_t4, "candidate_id")
    by_id = {item["candidate_id"]: item for item in assessments}
    return {
        "base_selected_id": base_selected_id,
        "stress_selected_id": stress_selected_id,
        "base_feasible_ids": sorted(item["candidate_id"] for item in base_feasible),
        "stress_feasible_ids": sorted(item["candidate_id"] for item in stress_feasible),
        "base_binding_constraint": (
            "none"
            if base_selected_id == "HOLD"
            else by_id[base_selected_id]["base_binding_constraint"]
        ),
        "stress_binding_constraint": (
            "none"
            if stress_selected_id == "HOLD"
            else by_id[stress_selected_id]["stress_binding_constraint"]
        ),
        "assessments": assessments,
    }


__all__ = [
    "TEMPLATE_VERSION",
    "TEMPLATE_VERSIONS",
    "T3_CONSTRAINT_ORDER",
    "T4_CONSTRAINT_ORDER",
    "deterministic_rng",
    "t3_decision_solution",
    "t4_decision_solution",
]
