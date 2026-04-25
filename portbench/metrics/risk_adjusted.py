"""Risk-adjusted metrics: Sharpe, Sortino, Calmar, Information Ratio."""

import numpy as np
import pandas as pd

from .base import MetricsConfig
from .return_metrics import cagr
from .risk_metrics import max_drawdown


def sharpe_ratio(returns: pd.Series, config: MetricsConfig = None) -> float:
    """
    Compute the annualized Sharpe Ratio.

    Formula: SR = (mean(r) * annualization_factor - rf) / (std(r) * sqrt(annualization_factor))
    where rf is the annualized risk-free rate.

    Args:
        returns: Daily simple returns.
        config:  MetricsConfig with risk_free_rate and annualization_factor.

    Returns:
        Sharpe ratio (dimensionless). Returns 0.0 if volatility is zero.
    """
    if config is None:
        config = MetricsConfig()

    r = returns.dropna()
    if len(r) < 2:
        return 0.0

    ann_factor = config.annualization_factor
    excess_return = r.mean() * ann_factor - config.risk_free_rate
    vol = float(r.std() * np.sqrt(ann_factor))
    if vol == 0.0:
        return 0.0
    return float(excess_return / vol)


def sortino_ratio(returns: pd.Series, config: MetricsConfig = None) -> float:
    """
    Compute the annualized Sortino Ratio (penalizes only downside deviation).

    Formula: Sortino = (mean(r) * ann - rf) / (std(r[r<0]) * sqrt(ann))

    Args:
        returns: Daily simple returns.
        config:  MetricsConfig.

    Returns:
        Sortino ratio. Returns 0.0 if downside deviation is zero.
    """
    if config is None:
        config = MetricsConfig()

    r = returns.dropna()
    if len(r) < 2:
        return 0.0

    ann_factor = config.annualization_factor
    excess_return = r.mean() * ann_factor - config.risk_free_rate

    # Downside deviation: std of returns below zero (target = 0)
    downside = r[r < 0]
    if len(downside) == 0:
        return float("inf")  # No losing days: perfect Sortino
    downside_vol = float(downside.std() * np.sqrt(ann_factor))
    if downside_vol == 0.0:
        return 0.0
    return float(excess_return / downside_vol)


def calmar_ratio(returns: pd.Series, config: MetricsConfig = None) -> float:
    """
    Compute the Calmar Ratio: CAGR divided by absolute max drawdown.

    Formula: Calmar = CAGR / |MDD|

    Args:
        returns: Daily simple returns.
        config:  MetricsConfig.

    Returns:
        Calmar ratio. Returns 0.0 if max drawdown is zero.
    """
    if config is None:
        config = MetricsConfig()

    mdd = max_drawdown(returns)
    if mdd == 0.0:
        return 0.0
    return float(cagr(returns, config) / abs(mdd))


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    config: MetricsConfig = None,
) -> float:
    """
    Compute the Information Ratio relative to a benchmark.

    Formula: IR = mean(r - r_b) * ann / std(r - r_b) * sqrt(ann)
    where r_b is the benchmark return series.

    Args:
        returns:   Portfolio daily simple returns.
        benchmark: Benchmark daily simple returns (same index).
        config:    MetricsConfig.

    Returns:
        Information ratio. Returns 0.0 if tracking error is zero or inputs are misaligned.
    """
    if config is None:
        config = MetricsConfig()

    # Align on common dates
    aligned = pd.DataFrame({"port": returns, "bench": benchmark}).dropna()
    if len(aligned) < 2:
        return 0.0

    active = aligned["port"] - aligned["bench"]
    ann_factor = config.annualization_factor
    tracking_error = float(active.std() * np.sqrt(ann_factor))
    if tracking_error == 0.0:
        return 0.0
    return float(active.mean() * ann_factor / tracking_error)
