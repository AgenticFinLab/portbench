"""
Mock data provider for development and unit testing.

Generates synthetic but financially realistic market data so that QA templates
and the agent evaluation pipeline can be developed and tested before real
datasets are finalized.

Synthetic data properties:
  - Prices follow Geometric Brownian Motion (GBM) with per-asset parameters
  - Returns are log-normal (GBM increments)
  - Macro indicators are deterministic functions of date
  - Market regimes are assigned based on cumulative return thresholds
  - All randomness is seeded for reproducibility

Usage:
    from portbench.qa_builder.mock_data import MockDataProvider
    provider = MockDataProvider(seed=42)
    context = provider.build_context(date(2020, 3, 15), ["SPY", "TLT"], lookback_days=60)
"""

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .base import DataProvider, MarketRegime


# ---------------------------------------------------------------------------
# Per-asset simulation parameters
# ---------------------------------------------------------------------------

_ASSET_PARAMS: dict[str, dict] = {
    # Equities
    "SPY":  {"mu": 0.10, "sigma": 0.18, "start_price": 300.0, "class": "equities"},
    "QQQ":  {"mu": 0.12, "sigma": 0.22, "start_price": 250.0, "class": "equities"},
    "EEM":  {"mu": 0.07, "sigma": 0.20, "start_price": 40.0,  "class": "equities"},
    # Bonds
    "TLT":  {"mu": 0.03, "sigma": 0.08, "start_price": 150.0, "class": "bonds"},
    "IEF":  {"mu": 0.02, "sigma": 0.05, "start_price": 110.0, "class": "bonds"},
    "LQD":  {"mu": 0.04, "sigma": 0.07, "start_price": 130.0, "class": "bonds"},
    # Commodities
    "GLD":  {"mu": 0.06, "sigma": 0.14, "start_price": 140.0, "class": "commodities"},
    "USO":  {"mu": 0.04, "sigma": 0.30, "start_price": 60.0,  "class": "commodities"},
    # Real Estate
    "VNQ":  {"mu": 0.07, "sigma": 0.16, "start_price": 85.0,  "class": "real_estate"},
    # Cryptocurrency
    "BTC":  {"mu": 0.50, "sigma": 0.80, "start_price": 8000.0, "class": "cryptocurrency"},
    "ETH":  {"mu": 0.45, "sigma": 0.90, "start_price": 200.0,  "class": "cryptocurrency"},
    # Cash
    "BIL":  {"mu": 0.02, "sigma": 0.005, "start_price": 91.0, "class": "cash"},
}

_ORIGIN_DATE = date(2015, 1, 1)   # Simulation start date


class MockDataProvider(DataProvider):
    """
    Synthetic market data provider backed by Geometric Brownian Motion.

    All price paths are deterministically generated from a seed so that the
    same (asset, date_range) always returns identical data, enabling
    reproducible test suites.

    Args:
        seed:              Random seed for GBM simulation.
        trading_days_only: If True, skip weekends (Mon–Fri only).
    """

    def __init__(self, seed: int = 42, trading_days_only: bool = True):
        self.seed = seed
        self.trading_days_only = trading_days_only
        # Pre-generate full price paths from origin to 2026-01-01
        self._cache: dict[str, pd.Series] = {}
        self._generate_all()

    # ------------------------------------------------------------------ public

    def get_price_series(self, asset: str, start: date, end: date) -> pd.Series:
        """Return close prices for asset in [start, end]."""
        series = self._get_full_series(asset)
        mask = (series.index.date >= start) & (series.index.date <= end)
        return series[mask]

    def get_return_series(self, asset: str, start: date, end: date) -> pd.Series:
        """Return daily simple returns for asset in [start, end]."""
        prices = self.get_price_series(asset, start, end)
        if prices.empty:
            return pd.Series(dtype=float)
        return prices.pct_change().dropna()

    def get_macro(self, d: date) -> dict[str, float]:
        """
        Return synthetic macro indicators at date d.
        Values are simple deterministic functions so they vary smoothly over time.
        """
        t = (d - _ORIGIN_DATE).days / 365.0  # Years since origin
        return {
            "fed_funds_rate": max(0.0, 0.02 + 0.02 * np.sin(t * np.pi)),
            "cpi_yoy": 0.02 + 0.015 * np.sin(t * 0.7 * np.pi),
            "unemployment": 0.05 - 0.01 * np.sin(t * 0.5 * np.pi),
            "gdp_growth_qoq": 0.005 + 0.003 * np.cos(t * np.pi),
            "vix": 15 + 10 * abs(np.sin(t * 1.5 * np.pi)),
        }

    def get_regime(self, d: date, asset: str = "SPY") -> MarketRegime:
        """
        Assign a market regime based on the 6-month trailing return of SPY.

        Thresholds (approximate):
          crisis:   trailing return < -20%
          bear:     trailing return < -5%
          bull:     trailing return >  10%
          sideways: otherwise
        """
        half_year_ago = d - timedelta(days=126)
        prices = self.get_price_series(asset, half_year_ago, d)
        if len(prices) < 2:
            return MarketRegime.SIDEWAYS

        ret_6m = float(prices.iloc[-1] / prices.iloc[0] - 1)
        if ret_6m < -0.20:
            return MarketRegime.CRISIS
        if ret_6m < -0.05:
            return MarketRegime.BEAR
        if ret_6m > 0.10:
            return MarketRegime.BULL
        return MarketRegime.SIDEWAYS

    def list_assets(self, asset_class: Optional[str] = None) -> list[str]:
        """List available assets, optionally filtered by asset class."""
        if asset_class is None:
            return list(_ASSET_PARAMS.keys())
        return [a for a, p in _ASSET_PARAMS.items() if p["class"] == asset_class]

    # ------------------------------------------------------------------ internals

    def _generate_all(self) -> None:
        """
        Pre-generate GBM price paths for all assets from _ORIGIN_DATE to 2026-01-01.
        Uses a per-asset seed derived from the global seed to ensure different
        assets have independent (but reproducible) price paths.
        """
        end_date = date(2026, 1, 1)
        full_index = pd.bdate_range(start=_ORIGIN_DATE, end=end_date)
        n = len(full_index)
        dt = 1.0 / 252  # Daily time step

        for i, (asset, params) in enumerate(_ASSET_PARAMS.items()):
            rng = np.random.default_rng(self.seed + i * 1000)
            mu = params["mu"]
            sigma = params["sigma"]
            s0 = params["start_price"]

            # GBM: dS = S*(mu*dt + sigma*sqrt(dt)*dW)
            drift = (mu - 0.5 * sigma ** 2) * dt
            diffusion = sigma * np.sqrt(dt)
            increments = drift + diffusion * rng.standard_normal(n)

            log_prices = np.log(s0) + np.cumsum(increments)
            prices = np.exp(log_prices)

            series = pd.Series(prices, index=full_index, name=asset)
            self._cache[asset] = series

    def _get_full_series(self, asset: str) -> pd.Series:
        if asset not in self._cache:
            raise KeyError(
                f"Asset '{asset}' not found in MockDataProvider. "
                f"Available: {list(_ASSET_PARAMS.keys())}"
            )
        return self._cache[asset]
