"""Flagship close columns must be non-null inside val and stress windows."""

import pandas as pd

from portbench.data_quality.base import QualityConfig, QualityLevel
from portbench.data_quality.numeric_quality import NumericQualityChecker


def test_flagship_spy_gap_fails_val_window():
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    close = [float("nan")] * len(dates)
    # Prices exist only in 2024, matching the truncated SPY bug.
    df = pd.DataFrame({"date": dates, "SPY_close": close})
    checker = NumericQualityChecker(QualityConfig())
    report = checker.check(df, "equities", "equities.csv", "processed")
    flagship = [
        c for c in report.checks if c.check_name.startswith("flagship_coverage_SPY")
    ]
    assert flagship
    assert any(c.level == QualityLevel.FAIL for c in flagship)


def test_flagship_spy_full_val_passes():
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    df = pd.DataFrame({"date": dates, "SPY_close": 400.0})
    checker = NumericQualityChecker(QualityConfig())
    report = checker.check(df, "equities", "equities.csv", "processed")
    val = [
        c
        for c in report.checks
        if c.check_name == "flagship_coverage_SPY_close_val"
    ]
    assert len(val) == 1
    assert val[0].level == QualityLevel.PASS
