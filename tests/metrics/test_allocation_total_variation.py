"""Regression tests for the S3-TV-v1 allocation score."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from portbench.agent_eval.base import MarketSnapshot, S3Output
from portbench.agent_eval.stages import S3WeightOptimization
from portbench.metrics.allocation_metrics import weight_total_variation


def test_total_variation_endpoints_and_zero_padding() -> None:
    assert weight_total_variation({"A": 1.0}, {"A": 1.0}) == pytest.approx(0.0)
    assert weight_total_variation({"A": 1.0}, {"B": 1.0}) == pytest.approx(1.0)
    assert weight_total_variation(
        {"A": 0.75, "B": 0.25, "C": 0.0},
        {"A": 0.25, "B": 0.75},
    ) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({}, "empty"),
        ({"A": 0.0}, "zero total mass"),
        ({"A": -0.1, "B": 1.1}, "negative"),
        ({"A": float("nan")}, "non-finite"),
    ],
)
def test_total_variation_rejects_invalid_allocations(
    weights: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        weight_total_variation(weights, {"A": 1.0})


def test_s3_combines_total_variation_and_correlation_equally() -> None:
    returns = {
        "A": pd.Series([0.01, 0.02, -0.01]),
        "B": pd.Series([-0.01, 0.01, 0.02]),
    }
    snapshot = MarketSnapshot(
        decision_date=date(2024, 1, 1),
        price_data={},
        return_data=returns,
        correlation_matrix=pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        ),
        asset_class_map={"A": "equities", "B": "bonds"},
    )
    stage = S3WeightOptimization(sigma=0.5)
    stage._last_snapshot = snapshot
    components = stage.score_components(
        S3Output(weights={"A": 0.75, "B": 0.25}),
        S3Output(weights={"A": 0.25, "B": 0.75}),
    )

    assert components["weight_tv"] == pytest.approx(0.5)
    assert components["accuracy"] == pytest.approx(0.5)
    assert components["correlation"] == pytest.approx(0.75)
    assert components["score"] == pytest.approx(0.625)


def test_s3_rejects_missing_finite_inter_class_pairs() -> None:
    snapshot = MarketSnapshot(
        decision_date=date(2024, 1, 1),
        price_data={},
        return_data={},
        correlation_matrix=pd.DataFrame(
            [[1.0, np.nan], [np.nan, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        ),
        asset_class_map={"A": "equities", "B": "bonds"},
    )
    stage = S3WeightOptimization(sigma=0.5)
    stage._last_snapshot = snapshot

    with pytest.raises(ValueError, match="No finite inter-class"):
        stage.score(
            S3Output(weights={"A": 0.5, "B": 0.5}),
            S3Output(weights={"A": 0.5, "B": 0.5}),
        )
