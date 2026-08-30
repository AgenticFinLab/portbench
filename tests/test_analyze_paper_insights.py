"""Focused tests for the frozen-artifact paper analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_paper_insights.py"
SPEC = importlib.util.spec_from_file_location("analyze_paper_insights", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalize_weights_and_distances() -> None:
    normalized = MODULE.normalize_weights({"A": 2.0, "B": 1.0})

    assert normalized == pytest.approx({"A": 2 / 3, "B": 1 / 3})
    assert MODULE.equal_weight_distance({"A": 0.75, "B": 0.25}) == pytest.approx(0.25)
    assert MODULE.allocation_distance({"A": 1.0}, {"B": 1.0}) == pytest.approx(1.0)


def test_normalize_weights_rejects_invalid_allocations() -> None:
    with pytest.raises(ValueError, match="negative"):
        MODULE.normalize_weights({"A": 1.1, "B": -0.1})
    with pytest.raises(ValueError, match="zero"):
        MODULE.normalize_weights({"A": 0.0})


def test_binary_auroc_handles_ties_and_class_errors() -> None:
    assert MODULE.binary_auroc([0.1, 0.2, 0.2, 0.9], [0, 0, 1, 1]) == pytest.approx(0.875)
    with pytest.raises(ValueError, match="both positive and negative"):
        MODULE.binary_auroc([0.1, 0.2], [1, 1])


def test_cluster_bootstrap_keeps_model_blocks() -> None:
    frame = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "value": [0.0, 0.0, 2.0, 2.0],
        }
    )

    low, high = MODULE.cluster_bootstrap_mean(frame, "value", samples=1000, seed=7)

    assert low == pytest.approx(0.0)
    assert high == pytest.approx(2.0)


def test_ceps_preserves_signed_drawdown_direction() -> None:
    assert MODULE.ceps_from_scores([0.8, 0.6, 0.7]) == pytest.approx(0.68)
    signed_drawdowns = pd.Series([-0.05, -0.20])
    magnitudes = -signed_drawdowns

    assert list(magnitudes) == pytest.approx([0.05, 0.20])


def test_validate_expected_cells_fails_on_missing_artifact() -> None:
    incomplete = pd.DataFrame(
        [
            {"model": "a", "profile": "conservative", "scenario": "normal"},
            {"model": "a", "profile": "balanced", "scenario": "normal"},
        ]
    )

    with pytest.raises(ValueError, match="Expected 3 unique cells"):
        MODULE.validate_expected_cells(incomplete, 1, 3, 1)


def test_profile_date_pairing_fails_when_one_profile_is_missing() -> None:
    episodes = pd.DataFrame(
        [
            {
                "model": "a",
                "model_label": "A",
                "decision_date": "2024-01-01",
                "profile": "conservative",
                "risk_exposure": 0.2,
                "safe_exposure": 0.5,
                "active_distance_eqw": 0.1,
                "hhi": 0.2,
            },
            {
                "model": "a",
                "model_label": "A",
                "decision_date": "2024-01-01",
                "profile": "aggressive",
                "risk_exposure": 0.6,
                "safe_exposure": 0.1,
                "active_distance_eqw": 0.2,
                "hhi": 0.3,
            },
        ]
    )
    cells = pd.DataFrame()

    with pytest.raises(ValueError, match="Missing profile/date pair"):
        MODULE.build_profile_pairs(episodes, cells)
