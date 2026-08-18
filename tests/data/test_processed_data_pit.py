"""PiT macro lags, growth transforms, and ticker column matching."""

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from portbench.data_preprocess.cash import CashPreprocessor
from portbench.data_preprocess.base import PreprocessConfig
from portbench.qa_builder.processed_data import (
    ProcessedDataProvider,
    _MACRO_COLUMNS,
    _MACRO_LAGS,
)


def test_macro_keys_have_lags():
    assert set(_MACRO_COLUMNS) == set(_MACRO_LAGS)


def test_cpi_yoy_and_gdp_qoq_from_monthly_levels(tmp_path: Path):
    # Monthly CPI index: 100 then 110 twelve months later is +10% YoY.
    cpi_dates = pd.date_range("2019-01-01", periods=14, freq="MS")
    cpi_vals = [100.0] * 12 + [110.0, 110.0]
    (tmp_path / "fred" / "cash").mkdir(parents=True)
    pd.DataFrame({"date": cpi_dates, "value": cpi_vals}).to_csv(
        tmp_path / "fred" / "cash" / "CPIAUCSL.csv", index=False
    )
    # Quarterly GDP: 100 then 102 is +2% QoQ.
    gdp_dates = pd.date_range("2019-01-01", periods=3, freq="QS")
    pd.DataFrame({"date": gdp_dates, "value": [100.0, 102.0, 102.0]}).to_csv(
        tmp_path / "fred" / "cash" / "GDPC1.csv", index=False
    )

    cfg = PreprocessConfig(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    pre = CashPreprocessor(cfg)
    out = pre.process_numeric(datetime(2019, 1, 1), datetime(2020, 3, 1))
    assert "fred_CPIAUCSL_yoy" in out.columns
    assert "fred_GDPC1_qoq" in out.columns
    yoy = out.dropna(subset=["fred_CPIAUCSL_yoy"])["fred_CPIAUCSL_yoy"]
    assert abs(float(yoy.iloc[0]) - 0.10) < 1e-8
    qoq = out.dropna(subset=["fred_GDPC1_qoq"])["fred_GDPC1_qoq"]
    assert abs(float(qoq.iloc[0]) - 0.02) < 1e-8


def test_last_available_macro_uses_period_start_plus_lag():
    # February CPI dated 2020-02-01, ffilled through February.
    idx = pd.date_range("2020-02-01", "2020-02-29", freq="D")
    series = pd.Series(258.076, index=idx)
    lag = 45
    # Available on 2020-03-17. Before that the series is unseen.
    before = ProcessedDataProvider._last_available_macro(
        series, date(2020, 3, 16), lag
    )
    after = ProcessedDataProvider._last_available_macro(
        series, date(2020, 3, 17), lag
    )
    assert before == 0.0
    assert abs(after - 258.076) < 1e-8


def test_april_unemployment_not_visible_on_april_first():
    # April UNRATE dated 2020-04-01 should not be visible on 2020-04-01.
    idx = pd.date_range("2020-02-01", "2020-04-30", freq="MS")
    series = pd.Series([3.5, 4.4, 14.8], index=idx)
    seen = ProcessedDataProvider._last_available_macro(
        series, date(2020, 4, 1), 45
    )
    assert abs(seen - 3.5) < 1e-8


def test_ticker_hyphen_underscore_match():
    cols = {"BTC_USD_close", "SPY_close"}
    assert ProcessedDataProvider._ticker_has_columns(
        None, "BTC-USD", "cryptocurrency", cols
    )
    assert ProcessedDataProvider._ticker_has_columns(
        None, "SPY", "equities", cols
    )
    assert not ProcessedDataProvider._ticker_has_columns(
        None, "TLT", "bonds", cols
    )


def test_yahoo_coverage_rejects_short_spy(tmp_path: Path):
    from portbench.data_collect.yahoo import YahooCollector

    csv_path = tmp_path / "SPY.csv"
    pd.DataFrame(
        {"date": pd.date_range("2024-01-02", periods=10, freq="B")}
    ).to_csv(csv_path, index=False)
    collector = YahooCollector(base_dir=str(tmp_path), start_date="2015-01-01")
    assert collector._has_expected_coverage(csv_path, "SPY") is False


def test_yahoo_coverage_accepts_full_spy(tmp_path: Path):
    from portbench.data_collect.yahoo import YahooCollector

    csv_path = tmp_path / "SPY.csv"
    pd.DataFrame(
        {"date": pd.date_range("2015-01-02", periods=20, freq="B")}
    ).to_csv(csv_path, index=False)
    collector = YahooCollector(base_dir=str(tmp_path), start_date="2015-01-01")
    assert collector._has_expected_coverage(csv_path, "SPY") is True
