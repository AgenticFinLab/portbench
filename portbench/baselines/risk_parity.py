"""
Risk parity baseline strategy.

Risk parity allocates portfolio weights inversely proportional to each
asset's volatility, so that each asset contributes equally to total
portfolio risk (measured by volatility).

Formula:
    w_i = (1 / σ_i) / Σ_j (1 / σ_j)

where σ_i is the rolling annualized volatility of asset i computed from
the return_data in the market snapshot.

This is the simplified "naive risk parity" (no correlation adjustment).
The full correlation-aware version requires solving an optimization problem
(Maillard et al., 2010); the naive version is used here because it is
analytically tractable, parameter-free, and easier to benchmark against.

Reference:
  Maillard, S., Roncalli, T., & Teïletche, J. (2010). The properties of
  equally weighted risk contribution portfolios. Journal of Portfolio
  Management, 36(4), 60–70.
"""

import numpy as np

from ..agent_eval.base import MarketSnapshot
from .base import BaselineStrategy


class RiskParityBaseline(BaselineStrategy):
    """
    Naive inverse-volatility risk parity strategy.

    Args:
        min_periods: Minimum number of return observations required to
                     compute volatility.  Assets with fewer observations
                     fall back to equal weight contribution.
        annualize:   If True, scale volatility to annual frequency
                     (multiplied by sqrt(252)).  Does not affect weights
                     since the factor cancels in the normalization.
    """

    def __init__(self, min_periods: int = 20, annualize: bool = True):
        self.min_periods = min_periods
        self.annualize = annualize

    @property
    def model_name(self) -> str:
        return "risk_parity_inverse_vol"

    def allocate(self, snapshot: MarketSnapshot) -> dict[str, float]:
        """
        Compute inverse-volatility weights from snapshot.return_data.

        For each asset:
          1. Compute rolling std of daily returns (after dropping NaNs).
          2. Invert to get raw weight contribution.
          3. Normalize so weights sum to 1.0.

        If an asset has insufficient data, its volatility is set to the
        cross-sectional median volatility (conservative treatment).
        """
        if snapshot.return_data:
            assets = list(snapshot.return_data.keys())
        elif snapshot.current_weights:
            assets = list(snapshot.current_weights.keys())
        else:
            assets = list(snapshot.price_data.keys())

        vols: dict[str, float] = {}
        for asset in assets:
            r = snapshot.return_data.get(asset)
            if r is None or r.dropna().shape[0] < self.min_periods:
                vols[asset] = np.nan  # Mark for backfill
            else:
                r_clean = r.dropna()
                vol = float(r_clean.std())
                if self.annualize:
                    vol *= np.sqrt(252)
                vols[asset] = max(vol, 1e-8)  # Guard against zero volatility

        # Fill missing vols with cross-sectional median
        valid_vols = [v for v in vols.values() if not np.isnan(v)]
        fallback_vol = float(np.median(valid_vols)) if valid_vols else 1.0
        vols = {a: (v if not np.isnan(v) else fallback_vol) for a, v in vols.items()}

        # Inverse-volatility weights
        raw_weights = {a: 1.0 / v for a, v in vols.items()}
        return self._normalize(raw_weights)
