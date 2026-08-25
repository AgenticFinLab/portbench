"""Tests for the sparse S3 provider-output contract."""

from types import SimpleNamespace

import pytest

from portbench.agent_eval.stages import _parse_stage_payload


def test_s3_accepts_sparse_weights_that_sum_to_one():
    """Allow omitted visible assets to represent zero-weight holdings."""
    snapshot = SimpleNamespace(return_data={"SPY": object(), "BIL": object()})

    parsed = _parse_stage_payload(
        '{"weights": {"SPY": 0.7, "BIL": 0.3}}',
        stage_name="S3",
        snapshot=snapshot,
    )

    assert parsed["weights"] == {"SPY": 0.7, "BIL": 0.3}


def test_s3_rejects_unknown_or_off_simplex_sparse_weights():
    """Keep the visible-asset and simplex numeric constraints strict."""
    snapshot = SimpleNamespace(return_data={"SPY": object(), "BIL": object()})

    with pytest.raises(ValueError, match="visible assets"):
        _parse_stage_payload(
            '{"weights": {"UNKNOWN": 1.0}}',
            stage_name="S3",
            snapshot=snapshot,
        )
    with pytest.raises(ValueError, match="sum to one"):
        _parse_stage_payload(
            '{"weights": {"SPY": 0.7}}',
            stage_name="S3",
            snapshot=snapshot,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        _parse_stage_payload(
            '{"weights": {"SPY": 1.0, "BIL": 0.0}}',
            stage_name="S3",
            snapshot=snapshot,
        )
