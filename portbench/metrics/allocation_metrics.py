"""Allocation accuracy metrics for evaluating predicted portfolio weights."""

import numpy as np
import pandas as pd


def weight_mae(predicted: dict[str, float], actual: dict[str, float]) -> float:
    """
    Compute the Mean Absolute Error between predicted and actual portfolio weights.

    Formula: MAE_w = (1/n) * sum(|w_pred_i - w_actual_i|)

    Both dicts must cover the same asset keys. Missing keys are treated as weight 0.

    Args:
        predicted: Dict mapping asset name -> predicted weight.
        actual:    Dict mapping asset name -> ground-truth weight.

    Returns:
        Weight MAE in [0, 2] (0 = perfect, 2 = completely wrong).
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
    def _portfolio_return(weights: dict[str, float], returns: dict[str, float]) -> float:
        return sum(weights.get(a, 0.0) * returns.get(a, 0.0) for a in returns)

    r_pred = _portfolio_return(predicted_weights, asset_returns)
    r_opt = _portfolio_return(optimal_weights, asset_returns)
    return float(r_pred - r_opt)
