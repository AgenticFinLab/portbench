"""S1 sparse-view validation tests."""

from types import SimpleNamespace

import pytest

from portbench.agent_eval.stages import _parse_stage_payload


def test_s1_accepts_sparse_visible_asset_views():
    """S1 may omit visible assets, which the stage materializes as neutral."""
    snapshot = SimpleNamespace(return_data={"SPY": object(), "^VIX": object()})

    parsed = _parse_stage_payload(
        (
            '{"asset_views": {"SPY": 0.4}, "detected_regime": "sideways", '
            '"confidence": 0.7, "macro_summary": "Mixed signals."}'
        ),
        stage_name="S1",
        snapshot=snapshot,
    )

    assert parsed["asset_views"] == {"SPY": 0.4}


def test_s1_rejects_unknown_or_non_finite_asset_views():
    """Sparse S1 views must remain bounded to the visible asset universe."""
    snapshot = SimpleNamespace(return_data={"SPY": object()})

    with pytest.raises(ValueError, match="visible assets"):
        _parse_stage_payload(
            ('{"asset_views": {"UNKNOWN": 0.4}, ' '"confidence": 0.7}'),
            stage_name="S1",
            snapshot=snapshot,
        )
    with pytest.raises(ValueError, match="asset views"):
        _parse_stage_payload(
            ('{"asset_views": {"SPY": "not-a-number"}, ' '"confidence": 0.7}'),
            stage_name="S1",
            snapshot=snapshot,
        )
