"""Schema version separation for S4/S5 ranking tables."""

from __future__ import annotations

import pytest

from portbench.agent_eval.contracts import (
    S4S5_SCHEMA_AGENTIC,
    S4S5_SCHEMA_DETERMINISTIC,
)
from portbench.metrics.plan_outcome_scores import (
    S4_ENV_OUTCOME_KEYS,
    S4_PLAN_QUALITY_KEYS,
)
from portbench.agent_eval.result_gates import (
    SchemaMixError,
    assert_homogeneous_schema_versions,
    build_ranking_rows,
)



def test_schema_version_strings_differ():
    assert S4S5_SCHEMA_DETERMINISTIC == "pipeline-v1-deterministic"
    assert S4S5_SCHEMA_AGENTIC == "pipeline-v3-collab"
    assert S4S5_SCHEMA_DETERMINISTIC != S4S5_SCHEMA_AGENTIC


def test_ranking_rejects_mixed_schema_versions():
    rows = [
        {"id": "a", "schema_version": S4S5_SCHEMA_DETERMINISTIC, "plan_quality": 0.9},
        {"id": "b", "schema_version": S4S5_SCHEMA_AGENTIC, "plan_quality": 0.8},
    ]
    with pytest.raises(SchemaMixError):
        assert_homogeneous_schema_versions(rows)
    with pytest.raises(SchemaMixError):
        build_ranking_rows(rows, score_keys=["plan_quality"])


def test_ranking_rejects_mixed_plan_and_env_keys():
    rows = [
        {
            "id": "a",
            "schema_version": S4S5_SCHEMA_DETERMINISTIC,
            "plan_quality": 0.9,
            "turnover": 0.5,
        }
    ]
    mixed = list(S4_PLAN_QUALITY_KEYS | S4_ENV_OUTCOME_KEYS)
    with pytest.raises(SchemaMixError):
        build_ranking_rows(rows, score_keys=mixed)


def test_ranking_accepts_homogeneous():
    rows = [
        {"id": "a", "schema_version": S4S5_SCHEMA_AGENTIC, "plan_quality": 0.9},
        {"id": "b", "schema_version": S4S5_SCHEMA_AGENTIC, "plan_quality": 0.8},
    ]
    out = build_ranking_rows(rows, score_keys=["plan_quality"])
    assert len(out) == 2
    assert out[0]["schema_version"] == S4S5_SCHEMA_AGENTIC


def test_prepare_ceps_ranking_rows_rejects_mixed_schema():
    from portbench.experiments.figures import prepare_ceps_ranking_rows

    rows = [
        {"id": "a", "schema_version": S4S5_SCHEMA_DETERMINISTIC, "mean_ceps": 0.2},
        {"id": "b", "schema_version": S4S5_SCHEMA_AGENTIC, "mean_ceps": 0.1},
    ]
    with pytest.raises(SchemaMixError):
        prepare_ceps_ranking_rows(rows)
