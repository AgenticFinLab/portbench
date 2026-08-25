"""Tests for sparse S2 and S3 provider-output contracts."""

import json
from datetime import date
from types import SimpleNamespace

import pytest

from portbench.agent_eval.base import MarketSnapshot, S2Output
from portbench.agent_eval.prompts import build_s3_prompt
from portbench.agent_eval.stages import S3WeightOptimization, _parse_stage_payload


def test_s3_accepts_sparse_non_negative_allocation_scores():
    """Allow zero scores to explicitly exclude a visible asset."""
    snapshot = SimpleNamespace(return_data={"SPY": object(), "BIL": object()})

    parsed = _parse_stage_payload(
        '{"allocation_scores": {"SPY": 0.7, "BIL": 0.0}}',
        stage_name="S3",
        snapshot=snapshot,
    )

    assert parsed["allocation_scores"] == {"SPY": 0.7, "BIL": 0.0}


def test_s3_rejects_unknown_negative_or_all_zero_allocation_scores():
    """Keep the visible-asset and numeric constraints strict."""
    snapshot = SimpleNamespace(return_data={"SPY": object(), "BIL": object()})

    with pytest.raises(ValueError, match="visible assets"):
        _parse_stage_payload(
            '{"allocation_scores": {"UNKNOWN": 1.0}}',
            stage_name="S3",
            snapshot=snapshot,
        )
    with pytest.raises(ValueError, match="finite non-negative"):
        _parse_stage_payload(
            '{"allocation_scores": {"SPY": 1.0, "BIL": -0.1}}',
            stage_name="S3",
            snapshot=snapshot,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        _parse_stage_payload(
            '{"allocation_scores": {"SPY": 0.0, "BIL": 0.0}}',
            stage_name="S3",
            snapshot=snapshot,
        )


def test_s3_rejects_more_than_twelve_positive_assets():
    """Keep the S3 sparse allocation representation bounded."""
    assets = [f"A{index}" for index in range(13)]
    snapshot = SimpleNamespace(return_data={asset: object() for asset in assets})
    raw = json.dumps({"allocation_scores": {asset: 1.0 for asset in assets}})

    with pytest.raises(ValueError, match="at most 12"):
        _parse_stage_payload(raw, stage_name="S3", snapshot=snapshot)


def test_s2_accepts_sparse_signal_strength_pairs():
    """Allow omitted visible assets to take the documented hold default."""
    snapshot = SimpleNamespace(return_data={"SPY": object(), "BIL": object()})

    parsed = _parse_stage_payload(
        '{"signals": {"SPY": "buy"}, "strengths": {"SPY": 0.8}}',
        stage_name="S2",
        snapshot=snapshot,
    )

    assert parsed["signals"] == {"SPY": "buy"}
    assert parsed["strengths"] == {"SPY": 0.8}


def test_s3_prompt_uses_relative_scores_without_zero_weight_noise():
    snapshot = MarketSnapshot(
        decision_date=date(2024, 1, 2),
        price_data={},
        return_data={},
        current_weights={"SPY": 0.6, "BIL": 0.4, "TLT": 0.0},
        portfolio_value=1_000_000.0,
    )
    s2 = S2Output(
        signals={"SPY": "buy", "BIL": "hold", "TLT": "sell"},
        strengths={"SPY": 0.8, "BIL": 0.5, "TLT": 0.6},
    )

    prompt = build_s3_prompt(
        snapshot=snapshot,
        s2=s2,
        assets=["SPY", "BIL", "TLT"],
        corr_block="",
    )

    assert "Do not normalize allocation_scores" in prompt
    assert '"allocation_scores"' in prompt
    assert "SPY=0.6000, BIL=0.4000" in prompt
    assert "TLT=0.0000" not in prompt


def test_s3_normalizes_model_relative_scores_once_for_execution():
    class Adapter:
        def complete(self, prompt: str) -> str:
            return json.dumps(
                {
                    "allocation_scores": {"SPY": 2.0, "BIL": 1.0},
                    "expected_return": 0.05,
                    "expected_vol": 0.10,
                    "sharpe_estimate": 0.5,
                }
            )

    snapshot = MarketSnapshot(
        decision_date=date(2024, 1, 2),
        price_data={},
        return_data={"SPY": object(), "BIL": object()},
        current_weights={"SPY": 0.5, "BIL": 0.5},
        portfolio_value=1_000_000.0,
    )
    s2 = S2Output(
        signals={"SPY": "buy", "BIL": "hold"},
        strengths={"SPY": 0.8, "BIL": 0.5},
    )

    output = S3WeightOptimization(Adapter()).run(snapshot, s2)

    assert output.weights == {"SPY": 2.0 / 3.0, "BIL": 1.0 / 3.0}
