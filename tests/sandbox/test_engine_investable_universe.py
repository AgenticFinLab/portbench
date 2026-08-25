"""Backtest initialization must exclude assets without start-date observations."""

from datetime import date

import pandas as pd

from portbench.agent_eval.base import MarketSnapshot
from portbench.baselines.base import BaselineStrategy
from portbench.qa_builder.base import DataProvider, MarketRegime
from portbench.sandbox.engine import BacktestEngine


class _UniverseProvider(DataProvider):
    """Expose one investable asset and one asset with no observations."""

    def get_price_series(self, asset, start, end):
        if asset == "INACTIVE":
            return pd.Series(dtype=float)
        index = pd.bdate_range("2019-10-01", "2020-02-10")
        return pd.Series(100.0, index=index)

    def get_return_series(self, asset, start, end):
        prices = self.get_price_series(asset, start, end)
        if prices.empty:
            return prices
        return pd.Series(0.0, index=prices.index)

    def get_macro(self, d):
        return {}

    def get_regime(self, d, asset="ACTIVE"):
        return MarketRegime.SIDEWAYS

    def get_news(self, asset, d):
        return ""

    def list_assets(self, asset_class=None):
        return ["ACTIVE", "INACTIVE"]


class _RecordingBaseline(BaselineStrategy):
    """Return the observed current weights for an inspectable no-op rebalance."""

    model_name = "recording"

    def __init__(self):
        self.seen_current_weights = []

    def allocate(self, snapshot: MarketSnapshot) -> dict[str, float]:
        self.seen_current_weights.append(dict(snapshot.current_weights))
        return dict(snapshot.current_weights)


def test_engine_initializes_only_start_date_investable_assets():
    strategy = _RecordingBaseline()
    engine = BacktestEngine(
        strategy=strategy,
        provider=_UniverseProvider(),
        start_date=date(2020, 1, 1),
        end_date=date(2020, 2, 4),
        rebalance_freq="monthly",
        use_pipeline=False,
        progress=False,
    )

    result = engine.run()

    assert strategy.seen_current_weights == [{"ACTIVE": 1.0}]
    assert "INACTIVE" not in result.weight_history.columns
