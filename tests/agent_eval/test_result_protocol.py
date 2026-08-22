"""Step-replay forbids Sharpe/CAGR/cumulative MaxDD."""

import pytest

from portbench.agent_eval.result_gates import validate_step_replay_record


def test_forbidden_metrics_rejected():
    with pytest.raises(ValueError, match="forbids"):
        validate_step_replay_record({"sharpe": 1.2})
    with pytest.raises(ValueError, match="forbids"):
        validate_step_replay_record({"cagr": 0.1})
    with pytest.raises(ValueError, match="forbids"):
        validate_step_replay_record({"cumulative_max_dd": 0.3})


def test_allowed_metrics_pass():
    validate_step_replay_record(
        {
            "stage_score": 0.8,
            "plan_quality": 0.7,
            "implementation_shortfall": 0.01,
            "single_period_return": 0.002,
        }
    )
