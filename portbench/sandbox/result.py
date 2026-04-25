"""
BacktestResult: immutable result container for a completed Sandbox backtest run.

Aggregates NAV curve, weight history, and all performance metrics computed
via portbench/metrics/ modules. When an InvestorProfile is active, also holds
per-step CEPS and alignment scores collected by BacktestEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from ..metrics.return_metrics import total_return, cagr
from ..metrics.risk_metrics import volatility, max_drawdown
from ..metrics.risk_adjusted import sharpe_ratio, sortino_ratio, calmar_ratio
from ..metrics.base import MetricsConfig


@dataclass
class BacktestResult:
    """
    Immutable result of a completed backtest run.

    Performance metrics are computed from nav_curve at construction time
    using portbench/metrics/ functions — no new metric logic here.
    """

    model_name: str
    start_date: date
    end_date: date
    initial_nav: float

    # Time series
    nav_curve: pd.Series  # index=date, values=NAV in dollars
    weight_history: pd.DataFrame  # index=date, columns=assets, values=weights
    trade_history: list[dict] = field(default_factory=list)

    # Scalar performance metrics (set by _compute_metrics on __post_init__)
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    n_rebalances: int = 0
    total_transaction_cost: float = 0.0

    # Investor profile fields (populated when BacktestEngine.profile is set)
    profile_name: Optional[str] = None
    per_step_ceps: list[float] = field(default_factory=list)
    mean_ceps: float = 0.0
    per_step_alignment: list[float] = field(default_factory=list)
    mean_profile_score: float = 0.0
    stress_passed: Optional[bool] = (
        None  # set externally by run_backtest after drawdown check
    )

    def __post_init__(self):
        self._compute_metrics()
        if self.per_step_ceps:
            self.mean_ceps = float(np.mean(self.per_step_ceps))
        if self.per_step_alignment:
            self.mean_profile_score = float(np.mean(self.per_step_alignment))

    def _compute_metrics(self):
        """Derive all scalar metrics from nav_curve using portbench/metrics/."""
        if self.nav_curve is None or len(self.nav_curve) < 2:
            return
        daily_returns = self.nav_curve.pct_change().dropna()
        cfg = MetricsConfig()
        self.total_return = total_return(daily_returns)
        self.cagr = cagr(daily_returns, cfg)
        self.sharpe_ratio = sharpe_ratio(daily_returns, cfg)
        self.sortino_ratio = sortino_ratio(daily_returns, cfg)
        self.max_drawdown = max_drawdown(daily_returns)
        self.calmar_ratio = calmar_ratio(daily_returns, cfg)
        self.volatility = volatility(daily_returns, cfg)

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict (excludes time series)."""
        d = {
            "model_name": self.model_name,
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "initial_nav": self.initial_nav,
            "final_nav": (
                round(float(self.nav_curve.iloc[-1]), 2) if len(self.nav_curve) else 0.0
            ),
            "total_return": round(self.total_return, 6),
            "cagr": round(self.cagr, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 6),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "volatility": round(self.volatility, 6),
            "n_rebalances": self.n_rebalances,
            "total_transaction_cost": round(self.total_transaction_cost, 4),
        }
        if self.profile_name is not None:
            d["profile_name"] = self.profile_name
            d["mean_ceps"] = round(self.mean_ceps, 4)
            d["mean_profile_score"] = round(self.mean_profile_score, 4)
            d["n_rebalances_with_ceps"] = len(self.per_step_ceps)
        if self.stress_passed is not None:
            d["stress_passed"] = self.stress_passed
        return d

    def summary(self) -> str:
        """Human-readable summary string."""
        final_nav = float(self.nav_curve.iloc[-1]) if len(self.nav_curve) else 0.0
        lines = [
            f"Sandbox Backtest Summary",
            f"Model:             {self.model_name}",
        ]
        if self.profile_name:
            lines.append(f"Profile:           {self.profile_name}")
        lines += [
            f"Period:            {self.start_date} → {self.end_date}",
            f"Initial NAV:       ${self.initial_nav:,.0f}",
            f"Final NAV:         ${final_nav:,.0f}",
            f"Total Return:      {self.total_return:+.2%}",
            f"CAGR:              {self.cagr:+.2%}",
            f"Sharpe Ratio:      {self.sharpe_ratio:.3f}",
            f"Sortino Ratio:     {self.sortino_ratio:.3f}",
            f"Max Drawdown:      {self.max_drawdown:.2%}",
            f"Calmar Ratio:      {self.calmar_ratio:.3f}",
            f"Annualized Vol:    {self.volatility:.2%}",
            f"Rebalances:        {self.n_rebalances}",
            f"Transaction Cost:  ${self.total_transaction_cost:,.2f}",
        ]
        if self.profile_name is not None:
            lines += [
                f"Mean CEPS:         {self.mean_ceps:.4f}",
                f"Mean Profile Score:{self.mean_profile_score:.4f}",
            ]
        if self.stress_passed is not None:
            lines.append(
                f"Stress Gate:       {'PASSED' if self.stress_passed else 'FAILED'}"
            )
        return "\n".join(lines)
