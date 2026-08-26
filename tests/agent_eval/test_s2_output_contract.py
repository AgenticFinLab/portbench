"""S2 non-hold signal validation tests."""

from types import SimpleNamespace

import pytest

from portbench.agent_eval.stages import _parse_stage_payload


def test_s2_ignores_hold_signals_without_strengths():
    """S2 may omit strengths for model-added hold signals."""
    snapshot = SimpleNamespace(return_data={"SPY": object(), "QQQ": object()})

    parsed = _parse_stage_payload(
        (
            '{"signals": {"SPY": "sell", "QQQ": "hold"}, '
            '"strengths": {"SPY": 0.8}, "reasoning": "Defensive."}'
        ),
        stage_name="S2",
        snapshot=snapshot,
    )

    assert parsed["signals"] == {"SPY": "sell"}
    assert parsed["strengths"] == {"SPY": 0.8}


def test_s2_preserves_paired_hold_signals():
    """S2 preserves historical paired hold outputs for reproducible scoring."""
    snapshot = SimpleNamespace(return_data={"SPY": object(), "QQQ": object()})

    parsed = _parse_stage_payload(
        (
            '{"signals": {"SPY": "sell", "QQQ": "hold"}, '
            '"strengths": {"SPY": 0.8, "QQQ": 0.5}, '
            '"reasoning": "Defensive."}'
        ),
        stage_name="S2",
        snapshot=snapshot,
    )

    assert parsed["signals"] == {"SPY": "sell", "QQQ": "hold"}
    assert parsed["strengths"] == {"SPY": 0.8, "QQQ": 0.5}


def test_s2_rejects_unpaired_non_hold_signals():
    """Every actionable S2 signal still requires a matching strength."""
    snapshot = SimpleNamespace(return_data={"SPY": object(), "QQQ": object()})

    with pytest.raises(ValueError, match="select the same assets"):
        _parse_stage_payload(
            (
                '{"signals": {"SPY": "sell", "QQQ": "buy"}, '
                '"strengths": {"SPY": 0.8}, "reasoning": "Mixed."}'
            ),
            stage_name="S2",
            snapshot=snapshot,
        )


def test_s3_canonicalizes_the_unambiguous_ics_alias():
    """S3 records the one unambiguous ICS to ICSH ticker correction."""
    snapshot = SimpleNamespace(return_data={"SPY": object(), "ICSH": object()})

    parsed = _parse_stage_payload(
        (
            '{"allocation_scores": {"SPY": 0.5, "ICS": 0.5}, '
            '"expected_return": 0.05, "expected_vol": 0.10, '
            '"sharpe_estimate": 0.5}'
        ),
        stage_name="S3",
        snapshot=snapshot,
    )

    assert parsed["allocation_scores"] == {"SPY": 0.5, "ICSH": 0.5}
    assert parsed["_artifact_normalizations"][0]["to"] == "ICSH"


def test_s3_rejects_an_ambiguous_ics_alias():
    """S3 preserves strict validation when both ticker forms are present."""
    snapshot = SimpleNamespace(return_data={"SPY": object(), "ICSH": object()})

    with pytest.raises(ValueError, match="visible assets"):
        _parse_stage_payload(
            (
                '{"allocation_scores": {"SPY": 0.5, "ICS": 0.25, "ICSH": 0.25}, '
                '"expected_return": 0.05, "expected_vol": 0.10, '
                '"sharpe_estimate": 0.5}'
            ),
            stage_name="S3",
            snapshot=snapshot,
        )
