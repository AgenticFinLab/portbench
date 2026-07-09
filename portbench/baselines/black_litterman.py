"""
Black-Litterman baseline strategy.

The Black-Litterman model combines market equilibrium (prior) returns with
investor views to produce posterior expected returns, which are then fed
into a mean-variance optimizer.

Prior:      Historical sample mean and covariance (lookback window only).
Views:      Directional views based on trailing return sign (bullish/bearish).
Tau:        Uncertainty scaling = 1 / lookback_days (auto-calibrated).
Omega:      View confidence proportional to prior variance of each viewed asset.

Reference:
  Black, F., & Litterman, R. (1992). Global portfolio optimization.
  Financial Analysts Journal, 48(5), 28–43.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..agent_eval.base import MarketSnapshot
from .base import BaselineStrategy


class BlackLittermanBaseline(BaselineStrategy):
    """
    Black-Litterman portfolio allocation strategy.

    Constructs posterior expected returns using the BL master formula:
        E[R] = [(τΣ)^-1 + P'Ω^-1 P]^-1 [(τΣ)^-1 π + P'Ω^-1 Q]

    then runs a max-Sharpe (long-only) optimization to produce weights.

    Args:
        tau:            Uncertainty scaling for the prior covariance.
                        Defaults to 1 / lookback_days when None.
        view_threshold: Minimum absolute trailing return to form a view
                        (default 0.01 = 1%).
    """

    def __init__(
        self,
        tau: Optional[float] = None,
        view_threshold: float = 0.01,
        ridge: float = 1e-4,
    ):
        self.tau = tau
        self.view_threshold = view_threshold
        self.ridge = ridge

    @property
    def model_name(self) -> str:
        return "black_litterman"

    def allocate(self, snapshot: MarketSnapshot) -> dict[str, float]:
        # 1. Collect asset return data --------------------------------
        if snapshot.return_data:
            assets = list(snapshot.return_data.keys())
        elif snapshot.current_weights:
            assets = list(snapshot.current_weights.keys())
        else:
            assets = list(snapshot.price_data.keys())

        df = pd.DataFrame({
            a: snapshot.return_data[a].dropna()
            for a in assets
            if a in snapshot.return_data and not snapshot.return_data[a].empty
        }).dropna()

        if df.shape[1] < 2 or len(df) < 2:
            return self._normalize({a: 1.0 for a in assets})

        asset_list = list(df.columns)
        n = len(asset_list)
        lookback_days = len(df)

        # 2. Prior: sample mean and covariance ------------------------
        pi = df.mean().values  # equilibrium excess returns (n,)
        Sigma = df.cov().values  # prior covariance (n x n)
        # Ridge regularization to ensure positive definiteness
        ridge_diag = np.eye(n) * self.ridge * np.mean(np.diag(Sigma))
        Sigma = Sigma + ridge_diag

        # 3. Tau ------------------------------------------------------
        tau = self.tau if self.tau is not None else (1.0 / max(lookback_days, 1))

        # 4. Views: trailing return direction -------------------------
        trailing_rets = (1 + df).prod() - 1
        P_rows = []
        Q_vals = []
        for i in range(n):
            ret = float(trailing_rets.iloc[i])
            if abs(ret) > self.view_threshold:
                row = np.zeros(n)
                row[i] = 1.0
                P_rows.append(row)
                # View magnitude = small expected excess return in the
                # direction of the trailing return.
                Q_vals.append(np.sign(ret) * 0.01)

        if not P_rows:
            # No active views — fall back to equal weight
            return self._normalize({a: 1.0 for a in asset_list})

        P = np.array(P_rows)  # (k, n) pick matrix
        Q = np.array(Q_vals)  # (k,) view vector
        k = len(Q)

        # 5. Omega: view confidence -----------------------------------
        # Omega = diag(P (tau * Sigma) P')
        Omega = np.diag(np.diag(P @ (tau * Sigma) @ P.T))
        Omega = Omega + np.eye(k) * 1e-8  # regularize

        # 6. Black-Litterman posterior --------------------------------
        # Posterior mean = [(τΣ)^-1 + P'Ω^-1 P]^-1 * [(τΣ)^-1 π + P'Ω^-1 Q]
        # Use pseudo-inverse as fallback when Σ is singular (e.g. short
        # lookback window with highly correlated assets).
        try:
            tau_Sigma_inv = np.linalg.inv(tau * Sigma)
        except np.linalg.LinAlgError:
            tau_Sigma_inv = np.linalg.pinv(tau * Sigma)
        try:
            Omega_inv = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            return self._normalize({a: 1.0 for a in asset_list})

        M = tau_Sigma_inv + P.T @ Omega_inv @ P
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return self._normalize({a: 1.0 for a in asset_list})

        posterior_mu = M_inv @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)

        # 7. Max-Sharpe optimization with posterior mu -----------------
        def neg_sharpe(w):
            ret = w @ posterior_mu
            vol = float(np.sqrt(w @ Sigma @ w + 1e-10))
            return -ret / vol

        try:
            result = minimize(
                neg_sharpe,
                np.ones(n) / n,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n,
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                options={"ftol": 1e-9, "maxiter": 500},
            )
            if result.success:
                w = np.maximum(result.x, 0.0)
                w /= w.sum()
                return self._normalize({
                    a: round(float(w[i]), 6) for i, a in enumerate(asset_list)
                })
        except Exception:
            pass

        return self._normalize({a: 1.0 for a in asset_list})
