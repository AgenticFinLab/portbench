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
