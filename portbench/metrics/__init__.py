"""PortBench metrics library."""

from .base import MetricsConfig, PortfolioMetrics
from .return_metrics import total_return, cagr
from .risk_metrics import volatility, max_drawdown, var, cvar
from .risk_adjusted import sharpe_ratio, sortino_ratio, calmar_ratio, information_ratio
from .allocation_metrics import weight_mae, portfolio_return_gap
from .ceps import CEPS, CEPSResult, StageScore


def compute_all(
    returns,
    config: MetricsConfig = None,
    benchmark=None,
) -> PortfolioMetrics:
    """
    Convenience function: compute all portfolio metrics in one call.

    Args:
        returns:   Daily simple returns (pd.Series).
        config:    MetricsConfig. Uses defaults if None.
        benchmark: Optional benchmark return series for Information Ratio.

    Returns:
        PortfolioMetrics with all fields populated.
    """
    if config is None:
        config = MetricsConfig()

    return PortfolioMetrics(
        total_return=total_return(returns),
        cagr=cagr(returns, config),
        volatility=volatility(returns, config),
        max_drawdown=max_drawdown(returns),
        var_95=var(returns, config),
        cvar_95=cvar(returns, config),
        sharpe_ratio=sharpe_ratio(returns, config),
        sortino_ratio=sortino_ratio(returns, config),
        calmar_ratio=calmar_ratio(returns, config),
        information_ratio=(
            information_ratio(returns, benchmark, config)
            if benchmark is not None
            else None
        ),
    )


__all__ = [
    "MetricsConfig",
    "PortfolioMetrics",
    "total_return",
    "cagr",
    "volatility",
    "max_drawdown",
    "var",
    "cvar",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "information_ratio",
    "weight_mae",
    "portfolio_return_gap",
    "CEPS",
    "CEPSResult",
    "StageScore",
    "compute_all",
]
