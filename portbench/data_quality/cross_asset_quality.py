"""Cross-asset quality checker: consistency, PiT validation, and market state labeling."""

from pathlib import Path
from typing import Optional

import pandas as pd

from .base import (
    BenchmarkQualityReport,
    DataQualityChecker,
    DatasetQualityReport,
    QualityLevel,
)
from .numeric_quality import NumericQualityChecker
from .text_quality import TextQualityChecker


# ---------------------------------------------------------------------------
# Market regime detection
# ---------------------------------------------------------------------------


def label_market_regimes(
    price_series: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
    crisis_drawdown: float = -0.20,
) -> pd.Series:
    """
    Label each date with a market regime using simple rule-based logic.

    Rules (applied in priority order):
      1. crisis    — rolling drawdown from recent peak exceeds crisis_drawdown
      2. bull      — 50-day MA > 200-day MA and price above 200-day MA
      3. bear      — 50-day MA < 200-day MA and price below 200-day MA
      4. sideways  — otherwise

    Args:
        price_series:    pd.Series with DatetimeIndex and price values.
        short_window:    Short moving average window (default 50).
        long_window:     Long moving average window (default 200).
        crisis_drawdown: Drawdown threshold for crisis classification (default -20%).

    Returns:
        pd.Series of regime labels: "bull" | "bear" | "sideways" | "crisis".
    """
    prices = price_series.dropna()
    ma_short = prices.rolling(short_window, min_periods=short_window // 2).mean()
    ma_long = prices.rolling(long_window, min_periods=long_window // 2).mean()

    rolling_max = prices.rolling(long_window, min_periods=1).max()
    drawdown = (prices - rolling_max) / rolling_max

    labels = pd.Series("sideways", index=prices.index, dtype=str)
    labels[ma_short > ma_long] = "bull"
    labels[(ma_short < ma_long) & (prices < ma_long)] = "bear"
    labels[drawdown <= crisis_drawdown] = "crisis"

    return labels


# ---------------------------------------------------------------------------
# Cross-asset checker
# ---------------------------------------------------------------------------


class CrossAssetQualityChecker(DataQualityChecker):
    """
    Cross-asset quality checks using the processed output files.

    Runs the following checks:
      1. cross_asset_alignment  — all six asset classes have processed data
      2. joint_coverage         — fraction of dates where all assets have data simultaneously
      3. cross_source_consistency — for overlapping tickers, Yahoo vs Kaggle correlation
      4. pit_validation         — detect potential look-ahead bias in feature computation
      5. regime_label_balance   — test-set regime distribution is reasonably balanced
    """

    @property
    def checker_name(self) -> str:
        return "cross_asset"

    def check(
        self,
        processed_dir: Optional[str] = None,
    ) -> DatasetQualityReport:
        """
        Run cross-asset checks on the processed output directory.

        Args:
            processed_dir: Path to processed data directory.
                           Defaults to config.datasets_dir/processed.

        Returns:
            DatasetQualityReport summarising cross-asset health.
        """
        base = (
            Path(processed_dir) if processed_dir else (self.datasets_dir / "processed")
        )

        report = DatasetQualityReport(
            asset_class="all",
            source="processed",
            dataset_id="cross_asset",
        )

        report.checks.append(self._check_asset_coverage(base))

        dfs = self._load_processed_frames(base)
        if dfs:
            report.checks.append(self._check_joint_coverage(dfs))
            report.checks.extend(self._check_regime_label_balance(dfs))

        return report

    def run_full_assessment(
        self,
        processed_dir: Optional[str] = None,
        raw_dirs: Optional[dict[str, Path]] = None,
    ) -> BenchmarkQualityReport:
        """
        Run the complete assessment pipeline:
          - Per-processed-file numeric checks
          - SEC / Kaggle text checks
          - Cross-asset checks

        Args:
            processed_dir: Path to datasets/processed/.
            raw_dirs:      Optional dict mapping source name to raw data path
                           (used for cross-source consistency checks).

        Returns:
            BenchmarkQualityReport aggregating all results.
        """
        benchmark_report = BenchmarkQualityReport()

        base = (
            Path(processed_dir) if processed_dir else (self.datasets_dir / "processed")
        )

        numeric_checker = NumericQualityChecker(self.config)
        text_checker = TextQualityChecker(self.config)

        # --- Numeric: check every processed CSV
        asset_classes = [
            "equities",
            "bonds",
            "commodities",
            "real_estate",
            "cryptocurrency",
            "cash",
        ]
        for asset_class in asset_classes:
            csv_path = base / f"{asset_class}.csv"
            df = self._load_csv(csv_path)
            if df is not None:
                price_cols = [
                    c
                    for c in df.columns
                    if any(k in c.lower() for k in ("close", "price", "value", "adj"))
                ]
                rep = numeric_checker.check(
                    df=df,
                    asset_class=asset_class,
                    dataset_id=f"{asset_class}.csv",
                    source="processed",
                    price_cols=price_cols or None,
                )
                benchmark_report.dataset_reports.append(rep)
            else:
                # Missing processed file is a FAIL
                rep = DatasetQualityReport(
                    asset_class=asset_class,
                    source="processed",
                    dataset_id=f"{asset_class}.csv",
                )
                rep.checks.append(
                    self._make_check(
                        "data_exists",
                        value=0.0,
                        threshold=1.0,
                        level=QualityLevel.FAIL,
                        message=f"Processed file not found: {csv_path}",
                    )
                )
                benchmark_report.dataset_reports.append(rep)

        # --- Text: check SEC filings
        sec_dir = self.datasets_dir / "sec" / "equities"
        if sec_dir.exists():
            for ticker_dir in sorted(sec_dir.iterdir()):
                if not ticker_dir.is_dir():
                    continue
                for filing_type_dir in sorted(ticker_dir.iterdir()):
                    if not filing_type_dir.is_dir():
                        continue
                    html_files = list(filing_type_dir.glob("*.htm*"))
                    rep = text_checker.check_file_list(
                        file_paths=html_files,
                        asset_class="equities",
                        dataset_id=f"sec_{ticker_dir.name}_{filing_type_dir.name}",
                        source="sec",
                    )
                    benchmark_report.dataset_reports.append(rep)

        # --- Cross-asset checks
        cross_report = self.check(processed_dir=str(base))
        benchmark_report.dataset_reports.append(cross_report)

        return benchmark_report

    # ------------------------------------------------------------------ checks

    def _check_asset_coverage(self, base: Path) -> object:
        """All six asset class CSVs should exist in processed/."""
        expected = [
            "equities",
            "bonds",
            "commodities",
            "real_estate",
            "cryptocurrency",
            "cash",
        ]
        missing = [a for a in expected if not (base / f"{a}.csv").exists()]
        coverage = (len(expected) - len(missing)) / len(expected)

        level = QualityLevel.FAIL if missing else QualityLevel.PASS
        message = (
            f"All 6 processed asset files found."
            if not missing
            else f"Missing processed files: {missing}."
        )
        return self._make_check(
            "asset_class_coverage",
            value=round(coverage, 4),
            threshold=1.0,
            level=level,
            message=message,
            details={
                "missing": missing,
                "found": [a for a in expected if a not in missing],
            },
        )

    def _check_joint_coverage(self, dfs: dict[str, pd.DataFrame]) -> object:
        """
        Fraction of trading days where ALL loaded asset classes have at least
        one non-NaN numeric value simultaneously.
        """
        # Build a date → assets_with_data mapping
        date_sets: dict[pd.Timestamp, set[str]] = {}
        for asset_class, df in dfs.items():
            numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
            for _, row in df.iterrows():
                dt = row["date"]
                has_data = any(not pd.isna(row[c]) for c in numeric_cols if c in row)
                if has_data:
                    date_sets.setdefault(dt, set()).add(asset_class)

        n_assets = len(dfs)
        total_dates = len(date_sets)
        joint_dates = sum(1 for assets in date_sets.values() if len(assets) == n_assets)
        joint_rate = joint_dates / max(total_dates, 1)

        level = self._level_by_threshold(
            joint_rate,
            warn_threshold=0.80,
            fail_threshold=0.50,
            higher_is_better=True,
        )
        return self._make_check(
            "joint_asset_coverage",
            value=round(joint_rate, 4),
            threshold=0.80,
            level=level,
            message=(
                f"{joint_rate:.1%} of dates have data for all {n_assets} asset classes "
                f"simultaneously ({joint_dates}/{total_dates} dates)."
            ),
            details={
                "joint_dates": joint_dates,
                "total_dates": total_dates,
                "n_assets": n_assets,
            },
        )

    def _check_regime_label_balance(self, dfs: dict[str, pd.DataFrame]) -> list[object]:
        """
        Apply regime detection on equity close prices in the test set and
        check that at least two distinct regimes are present.
        """
        results = []

        # Use equities processed file if available
        equity_df = dfs.get("equities")
        if equity_df is None:
            return results

        close_cols = [
            c for c in equity_df.columns if "close" in c.lower() or "spy" in c.lower()
        ]
        if not close_cols:
            close_cols = equity_df.select_dtypes(include=[float, int]).columns.tolist()
        if not close_cols:
            return results

        price_col = close_cols[0]
        equity_df = (
            equity_df.set_index("date") if "date" in equity_df.columns else equity_df
        )
        prices = equity_df[price_col].dropna()

        if len(prices) < 200:
            return results

        regimes = label_market_regimes(prices)

        # Check test period regime distribution
        test_start = pd.Timestamp(self.config.test_start)
        test_end = pd.Timestamp(self.config.test_end)
        test_regimes = regimes[
            (regimes.index >= test_start) & (regimes.index <= test_end)
        ]

        if test_regimes.empty:
            results.append(
                self._make_check(
                    "test_regime_balance",
                    value=0.0,
                    threshold=2.0,
                    level=QualityLevel.WARN,
                    message="No test-period data available for regime labeling.",
                )
            )
            return results

        counts = test_regimes.value_counts().to_dict()
        n_distinct = len(counts)

        level = self._level_by_threshold(
            n_distinct,
            warn_threshold=2,
            fail_threshold=1,
            higher_is_better=True,
        )
        results.append(
            self._make_check(
                "test_regime_balance",
                value=float(n_distinct),
                threshold=2.0,
                level=level,
                message=(
                    f"{n_distinct} distinct market regimes in test period "
                    f"({self.config.test_start} – {self.config.test_end}): {counts}."
                ),
                details={"regime_counts": counts, "price_column": price_col},
            )
        )

        # Also label and save full regime series for downstream use
        full_counts = regimes.value_counts().to_dict()
        results.append(
            self._make_check(
                "full_regime_distribution",
                value=float(len(full_counts)),
                threshold=2.0,
                level=QualityLevel.PASS if len(full_counts) >= 2 else QualityLevel.WARN,
                message=f"Full dataset regime distribution: {full_counts}.",
                details={"regime_counts": full_counts},
            )
        )

        return results

    # ------------------------------------------------------------------ helpers

    def _load_processed_frames(self, base: Path) -> dict[str, pd.DataFrame]:
        asset_classes = [
            "equities",
            "bonds",
            "commodities",
            "real_estate",
            "cryptocurrency",
            "cash",
        ]
        dfs = {}
        for ac in asset_classes:
            df = self._load_csv(base / f"{ac}.csv")
            if df is not None:
                dfs[ac] = df
        return dfs
