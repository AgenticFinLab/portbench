"""Tests for fast T3-D and T4-D constraint-decision templates."""

from __future__ import annotations

import json
from datetime import date

import pytest

from portbench.qa_builder.base import QAConfig
from portbench.qa_builder.mock_data import MockDataProvider
from portbench.qa_builder.t3_position_sizing import T3PositionSizing
from portbench.qa_builder.t4_pairwise_allocation import T4PairwiseAllocation
from portbench.agent_eval.call_artifacts import CallRequest
from portbench.qa_eval.evaluator import _checkpoint_key, _parse_qa_response
from portbench.qa_eval.scorer import score_response


def _build_decision_pairs():
    """Build deterministic T3-D and T4-D items using only mock Point-in-Time data."""
    provider = MockDataProvider(seed=42)
    config = QAConfig()
    decision_date = date(2024, 6, 3)
    t3 = T3PositionSizing(provider, config, template_version="constraint-decision-v2")
    t4 = T4PairwiseAllocation(provider, config, template_version="constraint-decision-v2")
    t3_pair = t3.build_one(
        provider.build_context(decision_date, t3._select_assets(decision_date), config.lookback_days),
        1,
    )
    t4_pair = t4.build_one(
        provider.build_context(decision_date, t4._select_assets(decision_date), config.lookback_days),
        1,
    )
    return t3_pair, t4_pair


def _response(pair, template_id: str) -> dict:
    """Build one exact compact response from deterministic offline ground truth."""
    solution = pair.metadata["constraint_decision"]
    if template_id == "T3":
        selected_fields = {
            "base_plan_id": solution["base_selected_id"],
            "stress_plan_id": solution["stress_selected_id"],
        }
    else:
        selected_fields = {
            "base_candidate_id": solution["base_selected_id"],
            "stress_candidate_id": solution["stress_selected_id"],
        }
    return {
        **selected_fields,
        "base_feasible_ids": solution["base_feasible_ids"],
        "stress_feasible_ids": solution["stress_feasible_ids"],
        "base_binding_constraint": solution["base_binding_constraint"],
        "stress_binding_constraint": solution["stress_binding_constraint"],
    }


def test_constraint_decision_builders_use_preverified_margins_and_pit_inputs():
    t3_pair, t4_pair = _build_decision_pairs()
    t3_gt = t3_pair.metadata["constraint_decision"]
    t4_gt = t4_pair.metadata["constraint_decision"]

    assert t3_pair.metadata["display_template_id"] == "T3-D"
    assert t4_pair.metadata["display_template_id"] == "T4-D"
    assert len(t3_gt["assessments"]) == 6
    assert len(t4_gt["assessments"]) == 7
    assert t3_gt["base_selected_id"] != t3_gt["stress_selected_id"]
    assert t4_gt["base_selected_id"] != t4_gt["stress_selected_id"]
    assert "Do not recompute" in t3_pair.question
    assert "Do not recompute" in t4_pair.question
    assert t3_pair.metadata["pit_audit"]["validated"] is True
    assert t4_pair.metadata["pit_audit"]["validated"] is True


def test_constraint_decision_scoring_requires_base_and_stress_feasibility():
    t3_pair, t4_pair = _build_decision_pairs()
    for template_id, pair in (("T3", t3_pair), ("T4", t4_pair)):
        exact = _response(pair, template_id)
        assert score_response(
            template_id,
            "",
            json.dumps(exact),
            template_version="constraint-decision-v2",
            metadata=pair.metadata,
        ) == pytest.approx(1.0)
        incomplete = dict(exact)
        incomplete["stress_feasible_ids"] = []
        assert score_response(
            template_id,
            "",
            json.dumps(incomplete),
            template_version="constraint-decision-v2",
            metadata=pair.metadata,
        ) == pytest.approx(0.85)


def test_constraint_decision_validation_rejects_duplicate_feasible_ids():
    with pytest.raises(ValueError, match="must not repeat IDs"):
        _parse_qa_response(
            json.dumps(
                {
                    "base_plan_id": "P1",
                    "stress_plan_id": "P2",
                    "base_feasible_ids": ["P1", "P1"],
                    "stress_feasible_ids": ["P2"],
                    "base_binding_constraint": "var",
                    "stress_binding_constraint": "es",
                }
            ),
            "constraint-decision-v2",
            "T3",
        )


def test_constraint_decision_checkpoint_tracks_generation_contract():
    """Ensure a changed generation budget cannot reuse a QA completion checkpoint."""
    base = {
        "provider": "tencent",
        "model": "hy3-preview",
        "model_revision": "",
        "stage_id": "QA:T3",
        "system_prompt": "",
        "user_prompt": "Choose a feasible plan.",
        "response_schema": {"template_version": "constraint-decision-v2"},
        "visible_input": {"qa_id": "T3-1", "template_id": "T3"},
        "data_version": "qa-decision-v2-20260825",
    }
    request_8192 = CallRequest(
        **base,
        generation_config={"temperature": 0.0, "max_tokens": 8192, "timeout": 300},
    )
    request_4096 = CallRequest(
        **base,
        generation_config={"temperature": 0.0, "max_tokens": 4096, "timeout": 300},
    )

    assert _checkpoint_key(request_8192) != _checkpoint_key(request_4096)
