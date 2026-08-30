"""Allocation accuracy metrics for evaluating predicted portfolio weights."""

import numpy as np


def _normalize_allocation(weights: dict[str, float]) -> dict[str, float]:
    """Validate and normalize one non-negative portfolio allocation."""

    if not weights:
        raise ValueError("Allocation is empty")

    parsed = {str(asset): float(weight) for asset, weight in weights.items()}
    values = np.asarray(list(parsed.values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Allocation contains a non-finite weight")
    if (values < 0.0).any():
        raise ValueError("Allocation contains a negative weight")

    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("Allocation has zero total mass")
    return {asset: weight / total for asset, weight in parsed.items()}


def weight_total_variation(
    predicted: dict[str, float],
    actual: dict[str, float],
) -> float:
    """Return total-variation distance between two portfolio allocations."""

    predicted_norm = _normalize_allocation(predicted)
    actual_norm = _normalize_allocation(actual)
    assets = set(predicted_norm) | set(actual_norm)
    distance = 0.5 * sum(
        abs(predicted_norm.get(asset, 0.0) - actual_norm.get(asset, 0.0))
        for asset in assets
    )
    return float(np.clip(distance, 0.0, 1.0))


def weight_mae(predicted: dict[str, float], actual: dict[str, float]) -> float:
    """
    Compute the Mean Absolute Error between predicted and actual portfolio weights.

    Formula: MAE_w = (1/n) * sum(|w_pred_i - w_actual_i|)

    Both dicts must cover the same asset keys. Missing keys are treated as weight 0.

    Args:
        predicted: Dict mapping asset name -> predicted weight.
        actual:    Dict mapping asset name -> ground-truth weight.

    Returns:
        Mean absolute weight error. For normalized long-only allocations over
        n assets, the maximum is 2/n.
    """
    all_assets = set(predicted) | set(actual)
    if not all_assets:
        return 0.0

    errors = [abs(predicted.get(a, 0.0) - actual.get(a, 0.0)) for a in all_assets]
    return float(np.mean(errors))


def portfolio_return_gap(
    predicted_weights: dict[str, float],
    optimal_weights: dict[str, float],
    asset_returns: dict[str, float],
) -> float:
    """
    Compute the portfolio return gap: R_pred - R_optimal.

    A negative gap means the predicted allocation underperformed the optimal one
    over the evaluation horizon.

    Args:
        predicted_weights: Dict mapping asset -> predicted weight.
        optimal_weights:   Dict mapping asset -> ground-truth optimal weight.
        asset_returns:     Dict mapping asset -> realized return over the horizon.

    Returns:
        Return gap as a decimal. Negative = underperformance.
    """

    def _portfolio_return(
        weights: dict[str, float], returns: dict[str, float]
    ) -> float:
        return sum(weights.get(a, 0.0) * returns.get(a, 0.0) for a in returns)

    r_pred = _portfolio_return(predicted_weights, asset_returns)
    r_opt = _portfolio_return(optimal_weights, asset_returns)
    return float(r_pred - r_opt)
