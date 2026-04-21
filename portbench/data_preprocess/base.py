"""Base classes and utilities for data preprocessing."""

import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Optional
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Six core asset classes for multi-asset portfolio management."""

    EQUITIES = "equities"
    BONDS = "bonds"
    COMMODITIES = "commodities"
    REAL_ESTATE = "real_estate"
    CRYPTOCURRENCY = "cryptocurrency"
    CASH = "cash"


class DataType(Enum):
    """Data type classification for portfolio management."""

    NUMERIC = "numeric"
    TEXT = "text"


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing."""

    # Input/Output paths
    input_dir: str = "datasets"
    output_dir: str = "datasets/processed"

    # Time alignment
    train_start: str = "2015-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2025-12-31"

    # Missing value handling
    max_ffill_days: int = 3
    # Monthly series (e.g. FRED) have 20-31 day gaps; use a larger limit for them
    monthly_ffill_days: int = 31

    # Outlier handling (winsorize percentiles)
    lower_percentile: float = 0.01
    upper_percentile: float = 0.99

    # Normalization
    rolling_window: int = 252  # Trading days for z-score

    # Data sources to scan
    sources: list = field(default_factory=lambda: ["kaggle", "fred", "yahoo", "sec"])

    # After outer-join merge, reindex to business-day calendar to drop phantom
    # weekend rows that crypto data introduces into equity/bond columns.
    # CryptocurrencyPreprocessor should set this to False for its own output.
    resample_to_business_days: bool = True


class TimeAligner:
    """Utility class for time alignment across datasets."""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.input_dir = Path(config.input_dir)

    def scan_datasets(self) -> dict[str, dict]:
        """
        Scan all datasets and extract time ranges.

        Returns:
            Dictionary with dataset info including time ranges.
        """
        datasets = {}

        for source in self.config.sources:
            source_dir = self.input_dir / source
            if not source_dir.exists():
                continue

            for asset_dir in source_dir.iterdir():
                if not asset_dir.is_dir():
                    continue

                for data_path in asset_dir.iterdir():
                    time_range = self._extract_time_range(data_path)
                    if time_range:
                        key = f"{source}/{asset_dir.name}/{data_path.name}"
                        datasets[key] = {
                            "path": data_path,
                            "source": source,
                            "asset_class": asset_dir.name,
                            "start_date": time_range[0],
                            "end_date": time_range[1],
                            "data_type": self._detect_data_type(data_path),
                        }

        # Save time ranges to JSON for inspection
        self.save_time_ranges(datasets)

        return datasets

    def _extract_time_range(self, path: Path) -> Optional[tuple[datetime, datetime]]:
        """Extract time range from a data file or directory."""
        try:
            if path.is_file() and path.suffix.lower() == ".csv":
                df = pd.read_csv(path, nrows=None, low_memory=False)
                date_col = self._find_date_column(df)
                if date_col:
                    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
                    # Remove timezone info to ensure tz-naive comparison
                    if hasattr(dates.dt, "tz") and dates.dt.tz is not None:
                        logger.warning(
                            "Stripping timezone %s from %s — ensure source data is UTC "
                            "or pre-converted to local time before ingestion.",
                            dates.dt.tz,
                            path,
                        )
                        dates = dates.dt.tz_localize(None)
                    if len(dates) > 0:
                        return dates.min().to_pydatetime(), dates.max().to_pydatetime()
            elif path.is_dir():
                # Scan CSV files in directory
                all_dates = []
                for csv_file in path.glob("*.csv"):
                    df = pd.read_csv(csv_file, nrows=None, low_memory=False)
                    date_col = self._find_date_column(df)
                    if date_col:
                        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
                        # Remove timezone info
                        if hasattr(dates.dt, "tz") and dates.dt.tz is not None:
                            logger.warning(
                                "Stripping timezone %s from %s — ensure source data is UTC "
                                "or pre-converted to local time before ingestion.",
                                dates.dt.tz,
                                csv_file,
                            )
                            dates = dates.dt.tz_localize(None)
                        all_dates.extend(dates.tolist())
                if all_dates:
                    # Ensure all dates are tz-naive
                    all_dates = [
                        (
                            d.replace(tzinfo=None)
                            if hasattr(d, "tzinfo") and d.tzinfo
                            else d
                        )
                        for d in all_dates
                    ]
                    return min(all_dates), max(all_dates)
        except Exception:
            pass
        return None

    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the date column in a DataFrame."""
        date_patterns = ["date", "time", "timestamp", "datetime", "Date", "TIME"]
        for col in df.columns:
            if any(p in col.lower() for p in ["date", "time"]):
                return col
        # Try first column
        if len(df.columns) > 0:
            try:
                pd.to_datetime(df.iloc[:, 0], errors="raise")
                return df.columns[0]
            except Exception:
                pass
        return None

    def _detect_data_type(self, path: Path) -> str:
        """Detect whether data is numeric or text."""
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in [".json", ".txt", ".html", ".htm"]:
                return "text"
            return "numeric"
        elif path.is_dir():
            # Check files in directory
            for f in path.iterdir():
                if f.suffix.lower() in [".json", ".txt", ".html", ".htm"]:
                    return "text"
        return "numeric"

    def find_common_range(
        self, datasets: dict[str, dict], data_type: str = "numeric"
    ) -> tuple[datetime, datetime]:
        """
        Find the common time range across all numeric datasets.

        Args:
            datasets: Dictionary of dataset info from scan_datasets().
            data_type: Filter by data type ("numeric" or "text").

        Returns:
            Tuple of (start_date, end_date) for common range.
        """
        filtered = {
            k: v for k, v in datasets.items() if v.get("data_type") == data_type
        }

        if not filtered:
            # Fall back to config dates
            return (
                datetime.fromisoformat(self.config.train_start),
                datetime.fromisoformat(self.config.test_end),
            )

        # Filter out datasets with bogus epoch dates (misclassified text CSVs,
        # files whose date column parsed as Unix epoch 1970-01-01, etc.)
        min_sane_date = datetime(2000, 1, 1)
        sane = {
            k: v for k, v in filtered.items()
            if v["start_date"] >= min_sane_date and v["end_date"] >= min_sane_date
        }
        if not sane:
            return (
                datetime.fromisoformat(self.config.train_start),
                datetime.fromisoformat(self.config.test_end),
            )

        start_dates = [v["start_date"] for v in sane.values()]
        end_dates = [v["end_date"] for v in sane.values()]

        common_start = max(start_dates)
        common_end = min(end_dates)

        if common_start >= common_end:
            raise ValueError(
                f"No common time range across datasets: computed range is "
                f"[{common_start.date()}, {common_end.date()}]. "
                f"Check that datasets have overlapping coverage."
            )

        return common_start, common_end

    def to_business_days(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> pd.DataFrame:
        """
        Reindex a DataFrame to a business-day (Mon–Fri) calendar.

        Call this **after** the outer-join merge of daily + crypto data and
        **before** forward-filling, so that weekend rows introduced by 24/7
        crypto data are dropped from equity/bond/commodity outputs.

        The reindex leaves weekday gaps as NaN; the caller's subsequent
        fill_missing() call fills them (with the appropriate limit).

        Args:
            df:       DataFrame with a date column.
            date_col: Name of the date column.

        Returns:
            DataFrame indexed on business days only, NaN for missing rows.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        bdays = pd.bdate_range(df.index.min(), df.index.max())
        df = df.reindex(bdays)
        df.index.name = date_col
        return df.reset_index()

    def save_time_ranges(self, datasets: dict[str, dict]) -> Path:
        """
        Save dataset time ranges to JSON file for inspection.

        Args:
            datasets: Dictionary of dataset info from scan_datasets().

        Returns:
            Path to the saved JSON file.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "time_ranges.json"

        # Convert to serializable format
        serializable = {}
        for key, info in datasets.items():
            serializable[key] = {
                "path": str(info["path"]),
                "source": info["source"],
                "asset_class": info["asset_class"],
                "start_date": info["start_date"].strftime("%Y-%m-%d"),
                "end_date": info["end_date"].strftime("%Y-%m-%d"),
                "data_type": info["data_type"],
                "days": (info["end_date"] - info["start_date"]).days,
            }

        # Add summary
        result = {
            "datasets": serializable,
            "summary": {
                "total_datasets": len(datasets),
                "by_source": {},
                "by_asset_class": {},
                "by_data_type": {},
            },
        }

        for info in serializable.values():
            # By source
            src = info["source"]
            if src not in result["summary"]["by_source"]:
                result["summary"]["by_source"][src] = 0
            result["summary"]["by_source"][src] += 1

            # By asset class
            ac = info["asset_class"]
            if ac not in result["summary"]["by_asset_class"]:
                result["summary"]["by_asset_class"][ac] = 0
            result["summary"]["by_asset_class"][ac] += 1

            # By data type
            dt = info["data_type"]
            if dt not in result["summary"]["by_data_type"]:
                result["summary"]["by_data_type"][dt] = 0
            result["summary"]["by_data_type"][dt] += 1

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  Saved time ranges to: {output_file}")
        return output_file


class NumericPreprocessor:
    """Preprocessor for numeric data (prices, returns, indicators)."""

    def __init__(self, config: PreprocessConfig):
        self.config = config

    def fill_missing(
        self, df: pd.DataFrame, date_col: str = "date", freq: str = "D"
    ) -> pd.DataFrame:
        """
        Fill missing values with forward-fill.

        Args:
            df:       Input DataFrame.
            date_col: Name of date column.
            freq:     Source data frequency — "D" for daily (limit=max_ffill_days),
                      "M" for monthly FRED series (limit=monthly_ffill_days).
                      Monthly series have 20-31 day gaps so the default 3-day
                      limit would leave them almost entirely NaN.

        Returns:
            DataFrame with filled values.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        limit = (
            self.config.monthly_ffill_days if freq == "M" else self.config.max_ffill_days
        )
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill(limit=limit)

        return df

    def winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorize outliers at specified percentiles.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with winsorized values.
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            lower = df[col].quantile(self.config.lower_percentile)
            upper = df[col].quantile(self.config.upper_percentile)
            df[col] = df[col].clip(lower=lower, upper=upper)

        return df

    def compute_log_returns(
        self, df: pd.DataFrame, price_col: str = "close"
    ) -> pd.DataFrame:
        """
        Compute log returns from price series.

        Args:
            df: Input DataFrame with price column.
            price_col: Name of price column.

        Returns:
            DataFrame with log_return column added.
        """
        df = df.copy()
        if price_col in df.columns:
            df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
        return df

    def rolling_zscore(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Apply rolling z-score normalization.

        Args:
            df: Input DataFrame.
            columns: Columns to normalize.

        Returns:
            DataFrame with normalized columns (suffixed with _zscore).
        """
        df = df.copy()
        window = self.config.rolling_window

        for col in columns:
            if col in df.columns:
                rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                rolling_std = df[col].rolling(window=window, min_periods=1).std()
                df[f"{col}_zscore"] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

        return df

    def align_to_dates(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Filter DataFrame to specified date range.

        Args:
            df: Input DataFrame.
            start_date: Start of date range.
            end_date: End of date range.
            date_col: Name of date column.

        Returns:
            Filtered DataFrame.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        # Remove timezone info to ensure tz-naive comparison
        if hasattr(df[date_col].dt, "tz") and df[date_col].dt.tz is not None:
            logger.warning(
                "Stripping timezone %s from column '%s' in align_to_dates() — "
                "ensure source data is UTC or pre-converted to local time.",
                df[date_col].dt.tz,
                date_col,
            )
            df[date_col] = df[date_col].dt.tz_localize(None)
        mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
        return df[mask].reset_index(drop=True)

    def deduplicate_columns(
        self,
        df: pd.DataFrame,
        asset_mappings: dict[str, list[str]] = None,
    ) -> pd.DataFrame:
        """
        Remove duplicate columns representing the same asset from different sources.

        Strategy:
        1. Identify columns representing the same asset (via asset_mappings or correlation)
        2. For duplicate groups, keep the column with fewer missing values
        3. Fill remaining gaps using secondary sources

        Args:
            df: Input DataFrame with potential duplicate columns.
            asset_mappings: Optional dict mapping canonical name to list of column patterns.
                           Example: {"BTC": ["BTC_USD", "bitcoin", "btc"]}

        Returns:
            DataFrame with duplicates removed.
        """
        df = df.copy()

        # Default mappings for common assets
        default_mappings = {
            "BTC": ["btc", "bitcoin", "BTC_USD", "BTC-USD"],
            "ETH": ["eth", "ethereum", "ETH_USD", "ETH-USD"],
            "GOLD": ["gold", "xau", "GLD", "XAUUSD"],
            "OIL": ["oil", "crude", "USO", "wti", "brent"],
            "SPY": ["spy", "sp500", "s&p"],
        }

        mappings = asset_mappings or default_mappings

        # Find duplicate column groups
        duplicate_groups = {}
        used_cols = set()

        for canonical, patterns in mappings.items():
            matching_cols = []
            for col in df.columns:
                if col == "date":
                    continue
                col_lower = col.lower()
                for pattern in patterns:
                    if pattern.lower() in col_lower:
                        matching_cols.append(col)
                        break

            if len(matching_cols) > 1:
                duplicate_groups[canonical] = matching_cols
                used_cols.update(matching_cols)

        # Process each duplicate group
        cols_to_drop = []
        for canonical, cols in duplicate_groups.items():
            # Find close/price columns for this asset
            close_cols = [
                c for c in cols if "close" in c.lower() or "price" in c.lower()
            ]

            if len(close_cols) > 1:
                # Rank by data quality (fewer NaN = better)
                quality_rank = [(c, df[c].isna().sum()) for c in close_cols]
                quality_rank.sort(key=lambda x: x[1])

                # Keep best, drop others
                best_col = quality_rank[0][0]
                for col, _ in quality_rank[1:]:
                    # Fill gaps in best column using secondary sources
                    df[best_col] = df[best_col].fillna(df[col])
                    cols_to_drop.append(col)

                print(
                    f"  [DEDUP] {canonical}: kept {best_col}, merged from {len(close_cols)-1} sources"
                )

        # Drop duplicate columns
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        return df


def _truncate_records_json(records: list, max_chars: int = 16000) -> str:
    """
    Serialize a list of record dicts to JSON, dropping records from the end
    until the result fits within max_chars. Always produces valid JSON.
    """
    kept = list(records)
    while kept:
        s = json.dumps(kept, ensure_ascii=False, default=str)
        if len(s) <= max_chars:
            return s
        kept.pop()
    return "[]"


# Per-source text length budgets (characters). Chosen to preserve signal
# density: long-form filings get head+tail sampling so both the business
# overview and risk-factor / MD&A closing sections survive; short news items
# get a single head slice since tail of short articles is often boilerplate.
TEXT_BUDGETS: dict[str, dict] = {
    # SEC 10-K / 10-Q: long filings. Keep opening (business / risk factors
    # usually start there) and tail (often contains MD&A or signatures with
    # forward-looking language).
    "sec":           {"strategy": "head_tail", "head": 6000, "tail": 3000},
    # Kaggle per-ticker stock news: short-to-medium articles. Head only.
    "kaggle_stock":  {"strategy": "head",      "head": 3000},
    # Kaggle crypto news items: short posts. Head only.
    "kaggle_crypto": {"strategy": "head",      "head": 2000},
}


def truncate_text_for_source(text: str, source_type: str) -> str:
    """
    Apply source-aware truncation to a single raw text record.

    - "head":      keep the first N chars (good for news where the lead covers it).
    - "head_tail": keep first H chars + sentinel + last T chars (for long filings
                   where signals cluster at both ends).

    Unknown source types fall back to a conservative head slice.
    """
    if not isinstance(text, str):
        return ""
    budget = TEXT_BUDGETS.get(source_type, {"strategy": "head", "head": 2000})
    strategy = budget["strategy"]
    if strategy == "head":
        return text[: budget["head"]]
    if strategy == "head_tail":
        h, t = budget["head"], budget["tail"]
        if len(text) <= h + t + 20:
            return text
        return text[:h] + " [...TRUNCATED...] " + text[-t:]
    return text[:2000]


class TextPreprocessor:
    """Preprocessor for text data (news, filings, reports)."""

    def __init__(self, config: PreprocessConfig):
        self.config = config

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.

        Args:
            text: Raw text string.

        Returns:
            Cleaned text string.
        """
        if not isinstance(text, str):
            return ""

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def extract_text_features(self, texts: list[str]) -> dict:
        """
        Extract basic features from text list.

        Args:
            texts: List of text strings.

        Returns:
            Dictionary of text features.
        """
        cleaned = [self.clean_text(t) for t in texts if t]
        word_counts = [len(t.split()) for t in cleaned]

        return {
            "document_count": len(cleaned),
            "total_words": sum(word_counts),
            "avg_words": np.mean(word_counts) if word_counts else 0,
            "texts": cleaned,
        }

    def aggregate_by_date(
        self, df: pd.DataFrame, date_col: str = "date", text_col: str = "text"
    ) -> pd.DataFrame:
        """
        Aggregate text data by date.

        Args:
            df: Input DataFrame with date and text columns.
            date_col: Name of date column.
            text_col: Name of text column.

        Returns:
            DataFrame with aggregated text per date.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col]).dt.date

        # Aggregate texts for each date
        aggregated = (
            df.groupby(date_col)[text_col]
            .apply(lambda x: json.dumps(list(x.dropna())))
            .reset_index()
        )
        aggregated.columns = [date_col, f"{text_col}_json"]

        return aggregated


class AssetPreprocessor(ABC):
    """Abstract base class for asset-specific preprocessing."""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.numeric_processor = NumericPreprocessor(config)
        self.text_processor = TextPreprocessor(config)

    @property
    @abstractmethod
    def asset_class(self) -> AssetClass:
        """Return the asset class this preprocessor handles."""
        pass

    @abstractmethod
    def process_numeric(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Process numeric data for this asset class.

        Args:
            start_date: Start of date range.
            end_date: End of date range.

        Returns:
            Processed numeric DataFrame.
        """
        pass

    @abstractmethod
    def process_text(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Process text data for this asset class.

        Args:
            start_date: Start of date range.
            end_date: End of date range.

        Returns:
            Processed text DataFrame with date and text_json columns.
        """
        pass

    def process(
        self, start_date: datetime, end_date: datetime
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process both numeric and text data.

        Args:
            start_date: Start of date range.
            end_date: End of date range.

        Returns:
            Tuple of (numeric_df, text_df).
        """
        numeric_df = self.process_numeric(start_date, end_date)
        text_df = self.process_text(start_date, end_date)
        return numeric_df, text_df

    def find_csv_files(self, source: str, asset: str) -> list[Path]:
        """Find all CSV files for given source and asset."""
        asset_dir = self.input_dir / source / asset
        if not asset_dir.exists():
            return []

        csv_files = []
        for item in asset_dir.iterdir():
            if item.is_file() and item.suffix.lower() == ".csv":
                csv_files.append(item)
            elif item.is_dir():
                csv_files.extend(item.glob("*.csv"))

        return csv_files

    def find_text_files(self, source: str, asset: str) -> list[Path]:
        """Find all text files for given source and asset."""
        asset_dir = self.input_dir / source / asset
        if not asset_dir.exists():
            return []

        text_files = []
        text_exts = [".json", ".txt", ".html", ".htm"]

        for item in asset_dir.iterdir():
            if item.is_file() and item.suffix.lower() in text_exts:
                text_files.append(item)
            elif item.is_dir():
                for ext in text_exts:
                    text_files.extend(item.glob(f"*{ext}"))

        return text_files

    # Split output into chunks once uncompressed bytes exceed this threshold.
    # Each chunk is written as <asset>.partNNN.csv; the loader concatenates them.
    _CHUNK_BYTES_THRESHOLD = 200 * 1024 * 1024   # 200 MB per chunk
    _CHUNK_ROWS = 1000                            # approx rows per chunk probe

    def save_output(
        self,
        numeric_df: pd.DataFrame,
        text_df: pd.DataFrame,
        suffix: str = "",
    ) -> Path:
        """
        Save processed data to output directory.

        If the merged DataFrame exceeds _CHUNK_BYTES_THRESHOLD when serialized,
        it is written as multiple part files (<asset>.part000.csv,
        <asset>.part001.csv, …) to avoid producing a single file so large
        that downstream consumers OOM when loading it whole. A companion
        manifest `<asset>.manifest.json` records the part list.

        Args:
            numeric_df: Processed numeric data.
            text_df: Processed text data.
            suffix: Optional suffix for filename.

        Returns:
            Path to the primary output (either the single CSV or the manifest).
        """
        asset_name = self.asset_class.value
        output_file = self.output_dir / f"{asset_name}{suffix}.csv"

        # Merge numeric and text data on date
        if not numeric_df.empty and not text_df.empty:
            if "date" in numeric_df.columns:
                numeric_df["date"] = pd.to_datetime(numeric_df["date"]).dt.date
            if "date" in text_df.columns:
                text_df["date"] = pd.to_datetime(text_df["date"]).dt.date

            merged = pd.merge(numeric_df, text_df, on="date", how="left")
        elif not numeric_df.empty:
            merged = numeric_df
        elif not text_df.empty:
            merged = text_df
        else:
            merged = pd.DataFrame()

        if merged.empty:
            return output_file

        # Guard against row explosion (e.g., text-records multiplying numeric rows,
        # or intraday timestamps preventing cross-row merges). Normalize the
        # date column to calendar-day granularity before dedup so rows that
        # share a trading day but differ in hour collapse correctly.
        if "date" in merged.columns:
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.date
            merged = merged.dropna(subset=["date"])
            if merged["date"].duplicated().any():
                before = len(merged)
                merged = merged.groupby("date", as_index=False).first()
                print(f"  [dedup] collapsed {before} -> {len(merged)} rows by date")

        self._clean_existing_outputs(asset_name, suffix)

        # Estimate serialized size with a small probe
        probe_rows = min(len(merged), 50)
        probe_bytes = len(
            merged.head(probe_rows).to_csv(index=False).encode("utf-8", errors="replace")
        )
        avg_row_bytes = probe_bytes / max(probe_rows, 1)
        est_total_bytes = int(avg_row_bytes * len(merged))

        if est_total_bytes <= self._CHUNK_BYTES_THRESHOLD:
            merged.to_csv(output_file, index=False)
            print(f"Saved: {output_file} ({len(merged)} rows, ~{est_total_bytes/1e6:.1f} MB)")
            return output_file

        # Chunked write
        rows_per_chunk = max(1, int(self._CHUNK_BYTES_THRESHOLD / max(avg_row_bytes, 1)))
        part_files: list[str] = []
        for i, start in enumerate(range(0, len(merged), rows_per_chunk)):
            part = merged.iloc[start:start + rows_per_chunk]
            part_path = self.output_dir / f"{asset_name}{suffix}.part{i:03d}.csv"
            part.to_csv(part_path, index=False)
            part_files.append(part_path.name)
            print(f"Saved: {part_path} ({len(part)} rows)")

        manifest_path = self.output_dir / f"{asset_name}{suffix}.manifest.json"
        manifest = {
            "asset_class": asset_name,
            "parts": part_files,
            "rows": int(len(merged)),
            "rows_per_chunk": rows_per_chunk,
            "chunk_bytes_threshold": self._CHUNK_BYTES_THRESHOLD,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Saved manifest: {manifest_path} ({len(part_files)} parts)")
        return manifest_path

    def _clean_existing_outputs(self, asset_name: str, suffix: str = "") -> None:
        """Remove stale single-file / chunked outputs before rewriting."""
        single = self.output_dir / f"{asset_name}{suffix}.csv"
        if single.exists():
            single.unlink()
        for part in self.output_dir.glob(f"{asset_name}{suffix}.part*.csv"):
            part.unlink()
        manifest = self.output_dir / f"{asset_name}{suffix}.manifest.json"
        if manifest.exists():
            manifest.unlink()
