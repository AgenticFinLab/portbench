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


def test_s1_ignores_unknown_news_asset_views():
    """Ignore asset symbols copied from news that are not investable inputs."""
    snapshot = SimpleNamespace(return_data={"SPY": object(), "BIL": object()})

    parsed = _parse_stage_payload(
        ('{"asset_views": {"SPY": 0.4, "NEWS_TICKER": 0.7}, ' '"confidence": 0.7}'),
        stage_name="S1",
        snapshot=snapshot,
    )

    assert parsed["asset_views"] == {"SPY": 0.4}


def test_s1_rejects_non_finite_visible_asset_views():
    """Visible-asset views must remain bounded numeric values."""
    snapshot = SimpleNamespace(return_data={"SPY": object()})

    with pytest.raises(ValueError, match="asset views"):
        _parse_stage_payload(
            ('{"asset_views": {"SPY": "not-a-number"}, ' '"confidence": 0.7}'),
            stage_name="S1",
            snapshot=snapshot,
        )
