"""
Minimum Variance Portfolio baseline.

Solves the long-only minimum variance problem:

    min  w' Σ w
    s.t. sum(w) = 1,  w_i >= 0

where Σ is the empirical annualized covariance matrix estimated from
snapshot.return_data.  This is the defensive end-point of the Markowitz
efficient frontier and is a standard reference in the portfolio literature.

Unlike RiskParityBaseline (inverse-vol) and CovarianceRiskParityBaseline
(equal risk contribution), Min-Var concentrates weight on the globally
least-volatile combination, potentially producing more concentrated
allocations in low-correlation, low-vol assets.

Reference:
  Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77–91.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..agent_eval.base import MarketSnapshot
from .base import BaselineStrategy


class MinVarianceBaseline(BaselineStrategy):
    """
    Long-only minimum variance portfolio using the full covariance matrix.

    Args:
        min_periods: Minimum return observations required to include an asset.
                     Assets with fewer observations are excluded from the
                     optimization and receive zero weight.
        ridge:       Diagonal ridge (fraction of mean variance) added to the
                     covariance matrix for numerical stability.
    """

    def __init__(self, min_periods: int = 20, ridge: float = 1e-4):
        self.min_periods = min_periods
        self.ridge = ridge

    @property
    def model_name(self) -> str:
        return "min_variance"

    def allocate(self, snapshot: MarketSnapshot) -> dict[str, float]:
        if snapshot.return_data:
            assets = list(snapshot.return_data.keys())
        elif snapshot.current_weights:
            assets = list(snapshot.current_weights.keys())
        else:
            assets = list(snapshot.price_data.keys())

        usable: list[str] = []
        series: dict[str, pd.Series] = {}
        for a in assets:
            r = snapshot.return_data.get(a)
            if r is None:
                continue
            r_clean = r.dropna()
            if len(r_clean) >= self.min_periods:
                usable.append(a)
                series[a] = r_clean

        if len(usable) < 2:
            n = max(len(assets), 1)
            return {a: round(1.0 / n, 6) for a in assets}

        ret_df = pd.DataFrame(series).dropna(how="any")
        if len(ret_df) < self.min_periods:
            n = max(len(assets), 1)
            return {a: round(1.0 / n, 6) for a in assets}

        cov = ret_df.cov().values * 252.0
        n = cov.shape[0]
        cov = cov + np.eye(n) * self.ridge * float(np.mean(np.diag(cov)) or 1.0)

        w = self._solve_min_var(cov)

        weights = {a: 0.0 for a in assets}
        for i, a in enumerate(ret_df.columns):
            weights[a] = float(w[i])
        return self._normalize(weights)

    def _solve_min_var(self, cov: np.ndarray) -> np.ndarray:
        n = cov.shape[0]
        w0 = np.full(n, 1.0 / n)
        result = minimize(
            fun=lambda w: float(w @ cov @ w),
            x0=w0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
            options={"ftol": 1e-10, "maxiter": 1000},
        )
        w = result.x if result.success else w0
        w = np.maximum(w, 0.0)
        total = w.sum()
        return w / total if total > 0 else w0
