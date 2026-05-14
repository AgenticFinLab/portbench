"""
SnapshotBuilder: constructs a MarketSnapshot for a given decision date,
injecting the real current portfolio state (weights, NAV) from PortfolioState.

Mirrors the logic in examples/agent_eval/run_evaluation.py::build_snapshots()
but uses live portfolio state instead of equal-weight assumptions.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from ..agent_eval.base import MarketSnapshot
from ..qa_builder.base import DataProvider


class SnapshotBuilder:
    """
    Builds MarketSnapshot objects from a DataProvider and live portfolio state.

    Args:
        provider:      DataProvider (MockDataProvider or ProcessedDataProvider).
        assets:        List of asset identifiers to include.
        lookback_days: Trading days of history to include in price/return data.
    """

    def __init__(
        self,
        provider: DataProvider,
        assets: list[str],
        lookback_days: int = 60,
        asset_class_map: Optional[dict[str, str]] = None,
    ):
        self.provider = provider
        self.assets = assets
        self.lookback_days = lookback_days
        self.asset_class_map = asset_class_map

    def build(
        self,
        decision_date: date,
        current_weights: dict[str, float],
        nav: float,
        forward_days: int = 0,
    ) -> MarketSnapshot:
        """
        Build a MarketSnapshot for decision_date with live portfolio state.

        Args:
            decision_date:   The date for which to build the snapshot.
            current_weights: Actual portfolio weights at this date (post-drift).
            nav:             Current portfolio NAV.
            forward_days:    If > 0, fetch this many trading days of future returns
                             (starting from decision_date) and populate
                             future_return_data for S3 ground-truth computation.

        Returns:
            MarketSnapshot ready to pass to EvalPipeline.run_episode() or
            BaselineStrategy.allocate().
        """
        lookback_start = decision_date - timedelta(days=int(self.lookback_days * 1.5))

        price_data: dict[str, pd.Series] = {}
        return_data: dict[str, pd.Series] = {}

        for asset in self.assets:
            prices = self.provider.get_price_series(
                asset, lookback_start, decision_date
            )
            prices = prices.iloc[-self.lookback_days :]
            returns = self.provider.get_return_series(
                asset, lookback_start, decision_date
            )
            returns = returns.iloc[-self.lookback_days :]
            if prices.empty or returns.empty:
                # Asset has no observations in this window — skip silently is
                # acceptable here because some assets (e.g., crypto pre-2014)
                # legitimately don't exist yet on a given decision date.
                continue
            price_data[asset] = prices
            return_data[asset] = returns

        # Correlation matrix from return data
        corr = None
        if len(return_data) >= 2:
            ret_df = pd.DataFrame(return_data)
            corr = ret_df.corr()

        macro = self.provider.get_macro(decision_date)
        regime = self.provider.get_regime(decision_date).value

        news_text = ""
        for asset in self.assets:
            txt = self.provider.get_news(asset, decision_date)
            if txt:
                news_text = txt
                break

        # Future return data for S3 ground-truth (never exposed to LLM)
        future_return_data = None
        if forward_days > 0:
            fwd_end = decision_date + timedelta(days=int(forward_days * 1.6))
            fwd: dict[str, pd.Series] = {}
            for asset in self.assets:
                r = self.provider.get_return_series(asset, decision_date, fwd_end)
                r = r.iloc[:forward_days]
                if not r.empty:
                    fwd[asset] = r
            if fwd:
                future_return_data = fwd

        return MarketSnapshot(
            decision_date=decision_date,
            price_data=price_data,
            return_data=return_data,
            macro_data=macro,
            current_weights=current_weights,
            portfolio_value=nav,
            market_regime=regime,
            news_text=news_text,
            correlation_matrix=corr,
            asset_class_map=self.asset_class_map,
            future_return_data=future_return_data,
        )
