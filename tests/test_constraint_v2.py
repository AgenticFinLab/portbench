"""Constraint-v2 QA ground-truth and deterministic scoring tests."""

from __future__ import annotations

import json
from datetime import date

import pytest

from portbench.qa_builder.base import QAConfig
from portbench.qa_builder.constraint_v2 import t3_solution, t4_metrics, t4_solution
from portbench.qa_builder.mock_data import MockDataProvider
from portbench.qa_builder.t3_position_sizing import T3PositionSizing
from portbench.qa_builder.t4_pairwise_allocation import T4PairwiseAllocation
from portbench.qa_eval.evaluator import _parse_qa_response
from portbench.qa_eval.scorer import score_response


def test_t3_solution_uses_frozen_tie_break_and_visible_limits():
    solution = t3_solution(
        {"var": 0.10, "es": 0.20, "drawdown": 0.25},
        {"var": 0.05, "es": 0.10, "drawdown": 0.20},
        0.80,
    )
    assert solution["position_size"] == 0.5
    assert solution["binding_constraint"] == "var"
    assert solution["constraint_margins"]["es"] == 0.0


def test_t3_constraint_v2_scoring_has_no_keyword_component():
    metadata = {
        "constraint_v2": {
            "position_size": 0.5,
            "binding_constraint": "var",
            "constraint_limits": {
                "var": 0.5,
                "es": 0.6,
                "drawdown": 0.8,
                "liquidity": 0.9,
                "full_allocation": 1.0,
            },
            "constraint_margins": {
                "var": 0.0,
                "es": 0.1,
                "drawdown": 0.3,
                "liquidity": 0.4,
                "full_allocation": 0.5,
            },
        }
    }
    response = json.dumps(
        {
            "position_size": 0.5,
            "binding_constraint": "var",
            "constraint_margins": metadata["constraint_v2"]["constraint_margins"],
            "rationale": "unscored free text",
        }
    )
    assert score_response("T3", "", response, template_version="constraint-v2", metadata=metadata) == 1.0


def test_t4_solution_and_scoring_use_candidates_not_hidden_optimum():
    assets = ["A", "B"]
    expected_returns = {"A": 0.10, "B": 0.04}
    covariance = [[0.04, 0.01], [0.01, 0.01]]
    current = {"A": 0.5, "B": 0.5}
    candidates = []
    for candidate_id, weights in (
        ("C1", {"A": 0.2, "B": 0.8}),
        ("C2", {"A": 0.5, "B": 0.5}),
        ("C3", {"A": 0.8, "B": 0.2}),
    ):
        candidates.append(
            {
                "candidate_id": candidate_id,
                "weights": weights,
                "metrics": t4_metrics(weights, assets, expected_returns, covariance, current),
            }
        )
    solution = t4_solution(candidates, return_floor=0.07, turnover_cap=0.4)
    assert solution["candidate_id"] == "C2"
    metadata = {
        "constraint_v2": {
            "candidates": candidates,
            "return_floor": 0.07,
            "turnover_cap": 0.4,
            **solution,
        }
    }
    response = json.dumps(
        {
            "candidate_id": "C2",
            "weights": solution["weights"],
            "calculated_metrics": candidates[1]["metrics"],
            "binding_constraints": solution["binding_constraints"],
            "rationale": "unscored free text",
        }
    )
    assert score_response(
        "T4",
        "",
        response,
        template_version="constraint-v2",
        metadata=metadata,
    ) == pytest.approx(1.0)


def test_constraint_v2_builders_are_reproducible_and_keep_solutions_hidden():
    provider = MockDataProvider(seed=42)
    config = QAConfig()
    decision_date = date(2024, 6, 3)
    t3 = T3PositionSizing(provider, config, template_version="constraint-v2")
    seen_bindings = set()
    for seq in range(5):
        context = provider.build_context(
            decision_date,
            t3._select_assets(decision_date),
            config.lookback_days,
        )
        pair = t3.build_one(context, seq)
        ground_truth = pair.metadata["constraint_v2"]
        recomputed = t3_solution(
            ground_truth["unit_risk"],
            ground_truth["budgets"],
            ground_truth["liquidity_cap"],
        )
        assert ground_truth["position_size"] == recomputed["position_size"]
        seen_bindings.add(ground_truth["binding_constraint"])
        assert "reference solution" not in pair.question.lower()
        assert pair.metadata["seed"] == config.random_seed
        assert len(pair.metadata["source_snapshot_hash"]) == 64
        assert pair.metadata["pit_audit"]["validated"] is True
    assert seen_bindings == {"var", "es", "drawdown", "liquidity", "full_allocation"}

    t4 = T4PairwiseAllocation(provider, config, template_version="constraint-v2")
    context = provider.build_context(
        decision_date,
        t4._select_assets(decision_date),
        config.lookback_days,
    )
    pair = t4.build_one(context, 1)
    ground_truth = pair.metadata["constraint_v2"]
    recomputed = t4_solution(
        ground_truth["candidates"],
        return_floor=ground_truth["return_floor"],
        turnover_cap=ground_truth["turnover_cap"],
    )
    assert ground_truth["candidate_id"] == recomputed["candidate_id"]
    assert "reference solution" not in (pair.context_summary + pair.question).lower()
    assert len(pair.metadata["source_snapshot_hash"]) == 64
    assert pair.metadata["pit_audit"]["decision_date"] == str(decision_date)


def test_constraint_v2_call_validation_rejects_incomplete_numeric_json():
    with pytest.raises(ValueError, match="position_size"):
        _parse_qa_response(
            '{"position_size": 1.2, "binding_constraint": "var", "constraint_margins": {}}',
            "constraint-v2",
            "T3",
        )
    with pytest.raises(ValueError, match="sum to one"):
        _parse_qa_response(
            json.dumps(
                {
                    "candidate_id": "C1",
                    "weights": {"A": 0.9, "B": 0.2},
                    "calculated_metrics": {
                        "expected_return": 0.1,
                        "variance": 0.2,
                        "turnover": 0.1,
                    },
                    "binding_constraints": [],
                }
            ),
            "constraint-v2",
            "T4",
        )
