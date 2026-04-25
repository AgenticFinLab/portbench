"""Numeric data quality checker for price and macro time series."""

from typing import Optional

import numpy as np
import pandas as pd

from .base import (
    CheckResult,
    DataQualityChecker,
    DatasetQualityReport,
    QualityLevel,
)


class NumericQualityChecker(DataQualityChecker):
    """
    Quality checks for numeric (CSV) datasets: price series and macro indicators.

    Runs the following checks per dataset:
      1. temporal_coverage   — fraction of trading days that have data
      2. stress_coverage     — coverage inside each stress period
      3. price_sanity        — no negative prices, no extreme single-day returns
      4. missing_gap_length  — longest consecutive NaN gap
      5. split_completeness  — train / val / test splits each have sufficient data
    """

    @property
    def checker_name(self) -> str:
        return "numeric"

    def check(
        self,
        df: pd.DataFrame,
        asset_class: str,
        dataset_id: str,
        source: str,
        price_cols: Optional[list[str]] = None,
    ) -> DatasetQualityReport:
        """
        Run all numeric quality checks on a DataFrame.

        Args:
            df:          DataFrame with a 'date' column and one or more numeric columns.
            asset_class: Asset class string (e.g. "equities").
            dataset_id:  Dataset identifier.
            source:      Data source (e.g. "yahoo", "kaggle").
            price_cols:  Columns that represent price levels (not returns).
                         If None, columns named 'close' / 'price' / 'value' are used.

        Returns:
            DatasetQualityReport with all check results.
        """
        report = DatasetQualityReport(
            asset_class=asset_class,
            source=source,
            dataset_id=dataset_id,
        )

        if df is None or df.empty:
            report.checks.append(
                self._make_check(
                    "data_exists",
                    value=0.0,
                    threshold=1.0,
                    level=QualityLevel.FAIL,
                    message="DataFrame is empty or None.",
                )
            )
            return report

        if "date" not in df.columns:
            report.checks.append(
                self._make_check(
                    "data_exists",
                    value=0.0,
                    threshold=1.0,
                    level=QualityLevel.FAIL,
                    message="No 'date' column found.",
                )
            )
            return report

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            report.checks.append(
                self._make_check(
                    "data_exists",
                    value=0.0,
                    threshold=1.0,
                    level=QualityLevel.FAIL,
                    message="No numeric columns found.",
                )
            )
            return report

        # Detect price columns automatically if not provided
        if price_cols is None:
            price_cols = [
                c
                for c in numeric_cols
                if any(k in c.lower() for k in ("close", "price", "value", "adj"))
            ]
            if not price_cols:
                price_cols = numeric_cols[:1]  # Fall back to first numeric column

        report.checks.append(self._check_temporal_coverage(df))
        report.checks.extend(self._check_stress_coverage(df))
        report.checks.append(self._check_split_completeness(df))
        report.checks.append(self._check_missing_gap(df, numeric_cols))
        report.checks.extend(self._check_price_sanity(df, price_cols))

        return report

    # ------------------------------------------------------------------ checks

    def _check_temporal_coverage(self, df: pd.DataFrame) -> CheckResult:
        """Fraction of business days in the full date range that have at least one row."""
        start = df["date"].min()
        end = df["date"].max()
        expected = self._trading_days(start, end)
        actual_dates = set(df["date"].dt.normalize())
        coverage = len(actual_dates.intersection(expected)) / max(len(expected), 1)

        level = self._level_by_threshold(
            coverage,
            warn_threshold=self.config.min_coverage_rate,
            fail_threshold=0.80,
            higher_is_better=True,
        )
        return self._make_check(
            "temporal_coverage",
            value=round(coverage, 4),
            threshold=self.config.min_coverage_rate,
            level=level,
            message=f"{coverage:.1%} of trading days covered ({start.date()} – {end.date()}).",
            details={
                "start": str(start.date()),
                "end": str(end.date()),
                "expected_days": len(expected),
                "actual_days": len(actual_dates),
            },
        )

    def _check_stress_coverage(self, df: pd.DataFrame) -> list[CheckResult]:
        """Coverage inside each configured stress period."""
        results = []
        actual_dates = set(df["date"].dt.normalize())

        for period_name, start, end in self.config.stress_periods:
            expected = self._trading_days(start, end)
            if len(expected) == 0:
                continue
            covered = len(actual_dates.intersection(expected))
            coverage = covered / len(expected)

            level = self._level_by_threshold(
                coverage,
                warn_threshold=self.config.min_stress_coverage,
                fail_threshold=0.70,
                higher_is_better=True,
            )
            results.append(
                self._make_check(
                    f"stress_coverage_{period_name}",
                    value=round(coverage, 4),
                    threshold=self.config.min_stress_coverage,
                    level=level,
                    message=f"{coverage:.1%} coverage in stress period '{period_name}' ({start} – {end}).",
                    details={
                        "period": period_name,
                        "start": start,
                        "end": end,
                        "expected_days": len(expected),
                        "covered_days": covered,
                    },
                )
            )

        return results

    def _check_split_completeness(self, df: pd.DataFrame) -> CheckResult:
        """All three splits (train/val/test) must contain data."""
        splits = {
            "train": (self.config.train_start, self.config.train_end),
            "val": (self.config.val_start, self.config.val_end),
            "test": (self.config.test_start, self.config.test_end),
        }
        missing = []
        details = {}
        for split_name, (start, end) in splits.items():
            mask = (df["date"] >= start) & (df["date"] <= end)
            count = mask.sum()
            details[split_name] = int(count)
            if count == 0:
                missing.append(split_name)

        if missing:
            level = QualityLevel.FAIL
            message = f"No data in split(s): {', '.join(missing)}."
        else:
            level = QualityLevel.PASS
            message = "All splits (train/val/test) contain data."

        return self._make_check(
            "split_completeness",
            value=float(len(splits) - len(missing)) / len(splits),
            threshold=1.0,
            level=level,
            message=message,
            details=details,
        )

    def _check_missing_gap(
        self, df: pd.DataFrame, numeric_cols: list[str]
    ) -> CheckResult:
        """Longest consecutive NaN run across all numeric columns."""
        max_gap = 0
        worst_col = ""
        for col in numeric_cols:
            series = df[col].isna()
            if not series.any():
                continue
            # Count consecutive True runs
            groups = series.ne(series.shift()).cumsum()
            run_lengths = series.groupby(groups).sum()
            col_max = int(run_lengths.max())
            if col_max > max_gap:
                max_gap = col_max
                worst_col = col

        threshold = self.config.max_ffill_gap_days
        level = self._level_by_threshold(
            max_gap,
            warn_threshold=threshold,
            fail_threshold=threshold * 3,
            higher_is_better=False,
        )
        if max_gap == 0:
            message = "No consecutive missing value gaps detected."
        else:
            message = (
                f"Longest consecutive NaN gap: {max_gap} rows (column: '{worst_col}')."
            )

        return self._make_check(
            "missing_gap_length",
            value=float(max_gap),
            threshold=float(threshold),
            level=level,
            message=message,
            details={"max_gap_rows": max_gap, "worst_column": worst_col},
        )

    def _check_price_sanity(
        self, df: pd.DataFrame, price_cols: list[str]
    ) -> list[CheckResult]:
        """Check for negative prices and extreme single-day returns."""
        results = []

        # Negative prices
        neg_counts = {}
        for col in price_cols:
            if col not in df.columns:
                continue
            n_neg = int((df[col] < self.config.min_price).sum())
            if n_neg:
                neg_counts[col] = n_neg

        if neg_counts:
            level = QualityLevel.FAIL
            message = f"Negative prices found in columns: {neg_counts}."
        else:
            level = QualityLevel.PASS
            message = "No negative prices detected."

        results.append(
            self._make_check(
                "price_non_negative",
                value=float(sum(neg_counts.values())),
                threshold=0.0,
                level=level,
                message=message,
                details={"negative_counts": neg_counts},
            )
        )

        # Extreme single-day returns
        extreme_counts = {}
        for col in price_cols:
            if col not in df.columns:
                continue
            prices = df[col].replace(0, np.nan).dropna()
            if len(prices) < 2:
                continue
            returns = prices.pct_change().abs()
            n_extreme = int((returns > self.config.max_single_day_return).sum())
            if n_extreme:
                extreme_counts[col] = n_extreme

        if extreme_counts:
            level = QualityLevel.WARN
            message = (
                f"Extreme daily returns (>{ self.config.max_single_day_return:.0%}) "
                f"detected: {extreme_counts}. May be splits/errors or high-volatility assets."
            )
        else:
            level = QualityLevel.PASS
            message = f"No extreme daily returns (>{self.config.max_single_day_return:.0%}) detected."

        results.append(
            self._make_check(
                "price_return_sanity",
                value=float(sum(extreme_counts.values())),
                threshold=0.0,
                level=level,
                message=message,
                details={"extreme_return_counts": extreme_counts},
            )
        )

        return results
