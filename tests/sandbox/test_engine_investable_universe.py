"""Backtest initialization must exclude assets without start-date observations."""

from datetime import date

import pandas as pd

from portbench.agent_eval.base import MarketSnapshot, StageID
from portbench.baselines.base import BaselineStrategy
from portbench.qa_builder.base import DataProvider, MarketRegime
from portbench.sandbox.engine import BacktestEngine


class _UniverseProvider(DataProvider):
    """Expose one investable asset and one asset with no observations."""

    def get_price_series(self, asset, start, end):
        if asset == "INACTIVE":
            return pd.Series(dtype=float)
        index = pd.bdate_range("2019-10-01", "2020-06-10")
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


def test_engine_limits_rebalances_for_a_protocol_pilot():
    """A pilot may exercise the pipeline without evaluating an entire window."""
    strategy = _RecordingBaseline()
    engine = BacktestEngine(
        strategy=strategy,
        provider=_UniverseProvider(),
        start_date=date(2020, 1, 2),
        end_date=date(2020, 4, 30),
        rebalance_freq="monthly",
        max_rebalances=2,
        use_pipeline=False,
        progress=False,
    )

    engine.run()

    assert len(strategy.seen_current_weights) == 2


def test_engine_pilot_cap_skips_the_initial_portfolio_date():
    """A one-step pilot must select the first agent decision, not initialization."""
    strategy = _RecordingBaseline()
    engine = BacktestEngine(
        strategy=strategy,
        provider=_UniverseProvider(),
        start_date=date(2020, 1, 1),
        end_date=date(2020, 3, 31),
        rebalance_freq="monthly",
        max_rebalances=1,
        use_pipeline=False,
        progress=False,
    )

    engine.run()

    assert len(strategy.seen_current_weights) == 1


def test_engine_pit_prefix_uses_only_trailing_returns():
    """The pilot prefix must ignore the snapshot's forward-return ground truth."""
    engine = BacktestEngine(
        strategy=_RecordingBaseline(),
        provider=_UniverseProvider(),
        factual_pit_prefix_stages=["S1", "S2", "S3"],
        use_pipeline=False,
        progress=False,
    )
    snapshot = MarketSnapshot(
        decision_date=date(2020, 1, 2),
        price_data={"ACTIVE": pd.Series([100.0]), "INACTIVE": pd.Series([100.0])},
        return_data={
            "ACTIVE": pd.Series([0.0, 0.0, 0.0]),
            "INACTIVE": pd.Series([0.0, 0.0, 0.0]),
        },
        macro_data={},
        future_return_data={
            "ACTIVE": pd.Series([10.0]),
            "INACTIVE": pd.Series([-10.0]),
        },
    )

    outputs = engine._build_pit_prefix(snapshot)

    assert set(outputs) == {
        StageID.S1_MARKET_INTERPRETATION,
        StageID.S2_SIGNAL_GENERATION,
        StageID.S3_WEIGHT_OPTIMIZATION,
    }
    weights = outputs[StageID.S3_WEIGHT_OPTIMIZATION].weights
    assert weights == {"ACTIVE": 0.5, "INACTIVE": 0.5}
