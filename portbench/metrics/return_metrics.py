"""Return metrics: total return and CAGR."""

import numpy as np
import pandas as pd

from .base import MetricsConfig


def total_return(returns: pd.Series) -> float:
    """
    Compute the cumulative simple return over the full return series.

    Formula: R_total = prod(1 + r_t) - 1

    Args:
        returns: Daily simple returns as a pd.Series (not log returns).

    Returns:
        Total return as a decimal (e.g., 0.15 means 15%).
    """
    if returns.empty:
        return 0.0
    return float((1 + returns.dropna()).prod() - 1)


def cagr(returns: pd.Series, config: MetricsConfig = None) -> float:
    """
    Compute the Compound Annual Growth Rate.

    Formula: CAGR = (1 + R_total)^(annualization_factor / T) - 1
    where T is the number of periods in the return series.

    Args:
        returns: Daily simple returns.
        config:  MetricsConfig for annualization_factor. Defaults to 252 if None.

    Returns:
        CAGR as a decimal.
    """
    if config is None:
        config = MetricsConfig()

    r = returns.dropna()
    if len(r) < 2:
        return 0.0

    total = float((1 + r).prod())
    n_years = len(r) / config.annualization_factor
    if total <= 0:
        return -1.0
    return float(total ** (1.0 / n_years) - 1)
