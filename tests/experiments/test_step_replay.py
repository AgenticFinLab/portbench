"""Step-replay runner refuses closed-loop metrics."""

from __future__ import annotations

import json
import pytest

from portbench.agent_eval.contracts import STEP_REPLAY
from portbench.agent_eval.result_gates import validate_step_replay_record
from portbench.experiments.step_replay import run_step_replay, run_step_replay_from_fixture
from tests.experiments.artifact_factory import write_legacy_episode


def test_step_replay_protocol_and_metrics():
    record = run_step_replay(
        target_weights={"A": 0.6, "B": 0.4},
        current_weights={"A": 0.5, "B": 0.5},
        snapshot_like={"price_data": {}, "portfolio_value": 100000.0},
        nav=100000.0,
        cache_key="toy",
    )
    assert record["result_protocol"] == STEP_REPLAY
    assert record["provenance"]["cache_key"] == "toy"
    validate_step_replay_record(record["metrics"])
    with pytest.raises(ValueError):
        validate_step_replay_record({**record["metrics"], "sharpe": 1.2})


def test_step_replay_fixture(tmp_path):
    # Build the source artifact inside pytest's isolated temporary directory.
    fixture = write_legacy_episode(tmp_path / "source")
    out = run_step_replay_from_fixture(fixture, tmp_path / "output")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["result_protocol"] == STEP_REPLAY
    assert "filled_weights" in data
