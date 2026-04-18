"""
Base classes and data structures for the PortBench metrics library.

All portfolio metrics are computed from return series (not raw prices) to ensure
consistency across asset classes with different price scales. The canonical input
is a pd.Series of daily simple or log returns with a DatetimeIndex.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricsConfig:
    """
    Configuration shared across all metric calculations.

    Attributes:
        risk_free_rate: Annualized risk-free rate (decimal). Used in Sharpe / Sortino.
        annualization_factor: Trading days per year. 252 for equities; 365 for crypto.
        var_confidence: Confidence level for VaR / CVaR (e.g., 0.95 means 5th percentile).
        benchmark_returns: Optional benchmark return series for Information Ratio.
    """

    risk_free_rate: float = 0.04          # 4% annualized
    annualization_factor: int = 252       # Trading days per year
    var_confidence: float = 0.95
    benchmark_returns: Optional[object] = None  # pd.Series when provided


@dataclass
class PortfolioMetrics:
    """
    Consolidated container for all computed portfolio metrics.

    Fields map directly to the metrics defined in docs/project-overview.md §4.3.
    None indicates the metric was not computed (e.g., IR requires a benchmark).
    """

    # --- Return metrics ---
    total_return: Optional[float] = None        # Cumulative return over period
    cagr: Optional[float] = None                # Compound Annual Growth Rate

    # --- Risk metrics ---
    volatility: Optional[float] = None          # Annualized standard deviation of returns
    max_drawdown: Optional[float] = None        # Maximum peak-to-trough decline (negative)
    var_95: Optional[float] = None              # Value-at-Risk at 95% confidence (negative)
    cvar_95: Optional[float] = None             # Conditional VaR / Expected Shortfall (negative)

    # --- Risk-adjusted metrics ---
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    information_ratio: Optional[float] = None   # Requires benchmark

    # --- Allocation accuracy (QA evaluation) ---
    weight_mae: Optional[float] = None          # Mean Absolute Error of predicted weights
    portfolio_return_gap: Optional[float] = None  # R_pred - R_optimal

    def to_dict(self) -> dict:
        """Serialize to a plain dict, omitting None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
