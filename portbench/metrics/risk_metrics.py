"""Risk metrics: volatility, max drawdown, VaR, CVaR."""

import numpy as np
import pandas as pd

from .base import MetricsConfig


def volatility(returns: pd.Series, config: MetricsConfig = None) -> float:
    """
    Compute annualized volatility (standard deviation of returns).

    Formula: sigma = std(r) * sqrt(annualization_factor)

    Args:
        returns: Daily simple returns.
        config:  MetricsConfig for annualization_factor.

    Returns:
        Annualized volatility as a decimal.
    """
    if config is None:
        config = MetricsConfig()

    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std() * np.sqrt(config.annualization_factor))


def max_drawdown(returns: pd.Series) -> float:
    """
    Compute the maximum peak-to-trough drawdown over the return series.

    Formula: MDD = max_t( (P_peak - P_t) / P_peak )

    Drawdown is returned as a negative number (e.g., -0.30 = 30% loss).

    Args:
        returns: Daily simple returns.

    Returns:
        Maximum drawdown as a negative decimal.
    """
    r = returns.dropna()
    if r.empty:
        return 0.0

    # Reconstruct a normalized price series starting at 1.0
    price = (1 + r).cumprod()
    rolling_peak = price.cummax()
    drawdown_series = (price - rolling_peak) / rolling_peak
    return float(drawdown_series.min())


def var(returns: pd.Series, config: MetricsConfig = None) -> float:
    """
    Compute the historical-simulation Value-at-Risk (VaR).

    VaR at confidence level c = -quantile(1-c) of the return distribution.
    E.g., VaR(95%) = -5th percentile return.

    Returned as a negative number (potential loss).

    Args:
        returns: Daily simple returns.
        config:  MetricsConfig with var_confidence (default 0.95).

    Returns:
        VaR as a negative decimal.
    """
    if config is None:
        config = MetricsConfig()

    r = returns.dropna()
    if r.empty:
        return 0.0

    alpha = 1.0 - config.var_confidence
    return float(np.quantile(r, alpha))


def cvar(returns: pd.Series, config: MetricsConfig = None) -> float:
    """
    Compute the Conditional VaR (CVaR / Expected Shortfall).

    CVaR = mean of returns that are <= VaR threshold.
    Returned as a negative number.

    Args:
        returns: Daily simple returns.
        config:  MetricsConfig with var_confidence.

    Returns:
        CVaR as a negative decimal.
    """
    if config is None:
        config = MetricsConfig()

    r = returns.dropna()
    if r.empty:
        return 0.0

    threshold = var(r, config)
    tail = r[r <= threshold]
    if tail.empty:
        return threshold
    return float(tail.mean())
