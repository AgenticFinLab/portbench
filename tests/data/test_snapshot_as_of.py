"""SnapshotBuilder as_of_date stops series on the previous close."""

from datetime import date

import pandas as pd

from portbench.qa_builder.base import DataProvider, MarketRegime
from portbench.sandbox.snapshot_builder import SnapshotBuilder


class _StubProvider(DataProvider):
    """Prices exist on 2020-03-02; as_of 2020-03-01 must exclude that row."""

    def get_price_series(self, asset, start, end):
        idx = pd.to_datetime(["2020-02-28", "2020-03-01", "2020-03-02"])
        s = pd.Series([100.0, 101.0, 90.0], index=idx)
        mask = (s.index.date >= start) & (s.index.date <= end)
        return s[mask]

    def get_return_series(self, asset, start, end):
        prices = self.get_price_series(asset, start, end)
        return prices.pct_change().dropna()

    def get_macro(self, d):
        return {"vix": 40.0 if d >= date(2020, 3, 2) else 20.0}

    def get_regime(self, d, asset="SPY"):
        return MarketRegime.CRISIS if d >= date(2020, 3, 2) else MarketRegime.SIDEWAYS

    def get_news(self, asset, d):
        return ""

    def list_assets(self, asset_class=None):
        return ["SPY"]


def test_as_of_date_excludes_decision_day_close():
    builder = SnapshotBuilder(_StubProvider(), ["SPY"], lookback_days=5)
    snap = builder.build(
        decision_date=date(2020, 3, 2),
        current_weights={"SPY": 1.0},
        nav=1.0,
        as_of_date=date(2020, 3, 1),
    )
    last_price_day = snap.price_data["SPY"].index[-1].date()
    assert last_price_day == date(2020, 3, 1)
    assert snap.macro_data["vix"] == 20.0
    assert snap.decision_date == date(2020, 3, 2)


def test_default_includes_decision_date():
    builder = SnapshotBuilder(_StubProvider(), ["SPY"], lookback_days=5)
    snap = builder.build(
        decision_date=date(2020, 3, 2),
        current_weights={"SPY": 1.0},
        nav=1.0,
    )
    last_price_day = snap.price_data["SPY"].index[-1].date()
    assert last_price_day == date(2020, 3, 2)
    assert snap.macro_data["vix"] == 40.0
