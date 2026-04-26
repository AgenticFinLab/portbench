"""
Covariance-aware Equal-Risk-Contribution (ERC) risk parity baseline.

Unlike the naive inverse-volatility variant in `risk_parity.py`, this strategy
uses the full empirical covariance matrix from `snapshot.return_data` and
solves for weights such that each asset's marginal contribution to portfolio
variance is equal:

    RC_i = w_i * (Σ w)_i / (w' Σ w)
    target: RC_i == 1/N for every i

Solved via cyclical coordinate descent on the log-loss
    L(w) = 0.5 * w' Σ w  -  (1/N) Σ_i log(w_i)
which has a unique long-only minimizer (Spinu, 2013). After convergence the
weights are renormalized to sum to 1.

Reference:
  Maillard, S., Roncalli, T., Teïletche, J. (2010).
  Spinu, F. (2013). An algorithm for computing risk parity weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..agent_eval.base import MarketSnapshot
from .base import BaselineStrategy


class CovarianceRiskParityBaseline(BaselineStrategy):
    """
    Equal-Risk-Contribution risk parity using the full covariance matrix.

    Args:
        min_periods:    Minimum return observations required to include an asset
                        in the optimization. Excluded assets receive zero weight.
        max_iter:       Maximum coordinate-descent iterations.
        tol:            Convergence tolerance on the L2 weight change between
                        iterations.
        ridge:          Diagonal ridge added to the covariance matrix for
                        numerical stability (in units of mean diagonal variance).
    """

    def __init__(
        self,
        min_periods: int = 20,
        max_iter: int = 500,
        tol: float = 1e-8,
        ridge: float = 1e-4,
    ):
        self.min_periods = min_periods
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge

    @property
    def model_name(self) -> str:
        return "risk_parity_covariance_erc"

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

        w = self._solve_erc(cov)

        weights = {a: 0.0 for a in assets}
        for i, a in enumerate(ret_df.columns):
            weights[a] = float(w[i])
        return self._normalize(weights)

    def _solve_erc(self, cov: np.ndarray) -> np.ndarray:
        """
        Cyclical coordinate descent on Spinu's log-loss objective.

        Updates each w_i in turn by solving the resulting quadratic in closed
        form, holding the other weights fixed. Always stays in the long-only
        domain.
        """
        n = cov.shape[0]
        w = np.full(n, 1.0 / n)
        target_rc = 1.0 / n

        for _ in range(self.max_iter):
            w_prev = w.copy()
            for i in range(n):
                # Solve a_i * w_i^2 + b_i * w_i - target_rc = 0
                a_i = cov[i, i]
                b_i = float(cov[i, :] @ w) - cov[i, i] * w[i]
                if a_i <= 0:
                    continue
                disc = b_i * b_i + 4.0 * a_i * target_rc
                w_i_new = (-b_i + np.sqrt(disc)) / (2.0 * a_i)
                w[i] = max(w_i_new, 1e-12)
            w = w / w.sum()
            if np.linalg.norm(w - w_prev) < self.tol:
                break

        return w
