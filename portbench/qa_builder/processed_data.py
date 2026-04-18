"""
ProcessedDataProvider — real data provider backed by datasets/processed/ CSV files.

This is the production-ready counterpart of MockDataProvider.  Once Phase 1
data collection and preprocessing are complete (i.e., `datasets/processed/`
contains the per-asset-class CSVs), swap MockDataProvider for this class to
run QA building and agent evaluation on real historical market data.

Data layout expected in datasets/processed/:
  equities.csv
  bonds.csv
  commodities.csv
  real_estate.csv
  cryptocurrency.csv
  cash.csv

Each CSV has a `date` column (YYYY-MM-DD) plus columns named
`<source>_<ticker>_<feature>` (e.g., `yahoo_SPY_close`, `fred_DGS10`).

Usage:
    from portbench.qa_builder.processed_data import ProcessedDataProvider
    provider = ProcessedDataProvider(data_dir="datasets/processed")
    context = provider.build_context(date(2020, 3, 15), ["SPY", "TLT"], lookback_days=60)
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Optional
import re

import numpy as np
import pandas as pd

from .base import DataProvider, MarketRegime


# ---------------------------------------------------------------------------
# Column name conventions in processed CSVs
# ---------------------------------------------------------------------------

# Standard close-price column suffix written by each preprocessor
_CLOSE_SUFFIX = "_close"
_RETURN_SUFFIX = "_return"

# Macro series fetched from the CASH / BONDS CSVs
_MACRO_COLUMNS = {
    "fed_funds_rate":   "fred_DFF",
    "cpi_yoy":          "fred_CPIAUCSL",
    "unemployment":     "fred_UNRATE",
    "gdp_growth_qoq":   "fred_GDPC1",
    "vix":              "yahoo_^VIX_close",
    "t10y2y_spread":    "fred_T10Y2Y",
    "breakeven_10y":    "fred_T10YIE",
    "hy_oas":           "fred_BAMLH0A0HYM2",
}

# Map from short ticker name → possible column prefixes in the processed CSV
# (the preprocessor may have stored it as yahoo_SPY or kaggle_SPY etc.)
_SOURCE_PREFIXES = ["yahoo", "kaggle", "fred"]


class ProcessedDataProvider(DataProvider):
    """
    DataProvider backed by pre-processed CSV files in datasets/processed/.

    Implements the same interface as MockDataProvider so all QA templates
    and EvalPipeline stages can be switched to real data with zero code changes.

    Args:
        data_dir:        Path to datasets/processed/ directory.
        regime_csv:      Optional path to market_regimes.csv produced by
                         label_market_regimes(). If None, regime is computed
                         on-the-fly from SPY returns.
        fill_limit:      Max consecutive NaN trading days to forward-fill (default 3).
        cache_on_init:   If True, all CSVs are loaded into memory at init time.
                         Set False for large datasets or low-memory environments.
    """

    # Asset class → CSV filename mapping
    _CLASS_TO_FILE = {
        "equities":     "equities.csv",
        "bonds":        "bonds.csv",
        "commodities":  "commodities.csv",
        "real_estate":  "real_estate.csv",
        "cryptocurrency": "cryptocurrency.csv",
        "cash":         "cash.csv",
    }

    # Hard-coded asset class membership for the standard ticker universe
    _ASSET_CLASS_MAP = {
        # Equities
        "SPY": "equities", "QQQ": "equities", "IWM": "equities",
        "VTI": "equities", "IVV": "equities", "EEM": "equities",
        "EFA": "equities", "VEA": "equities", "FXI": "equities",
        "EWJ": "equities", "ACWI": "equities",
        "XLK": "equities", "XLF": "equities", "XLE": "equities",
        "XLV": "equities", "XLI": "equities", "XLY": "equities",
        "XLP": "equities", "XLU": "equities", "XLB": "equities",
        "XLRE": "equities", "MTUM": "equities", "VLUE": "equities",
        "QUAL": "equities", "USMV": "equities", "^VIX": "equities",
        # Bonds
        "SHV": "bonds", "SHY": "bonds", "IEF": "bonds", "TLT": "bonds",
        "TLH": "bonds", "TIP": "bonds", "SCHP": "bonds", "STIP": "bonds",
        "LQD": "bonds", "HYG": "bonds", "JNK": "bonds", "VCIT": "bonds",
        "BNDX": "bonds", "EMB": "bonds", "IGOV": "bonds",
        # Commodities
        "GLD": "commodities", "IAU": "commodities", "SLV": "commodities",
        "PPLT": "commodities", "PALL": "commodities", "USO": "commodities",
        "UNG": "commodities", "BNO": "commodities", "DBA": "commodities",
        "CORN": "commodities", "WEAT": "commodities", "SOYB": "commodities",
        "DBC": "commodities", "PDBC": "commodities", "DBB": "commodities",
        "CPER": "commodities",
        # Real Estate
        "VNQ": "real_estate", "IYR": "real_estate", "SCHH": "real_estate",
        "REZ": "real_estate", "MORT": "real_estate", "INDS": "real_estate",
        "VNQI": "real_estate", "HAUZ": "real_estate",
        "XHB": "real_estate", "ITB": "real_estate",
        # Crypto
        "BTC-USD": "cryptocurrency", "ETH-USD": "cryptocurrency",
        "BNB-USD": "cryptocurrency", "SOL-USD": "cryptocurrency",
        "XRP-USD": "cryptocurrency", "ADA-USD": "cryptocurrency",
        "AVAX-USD": "cryptocurrency", "DOT-USD": "cryptocurrency",
        "LINK-USD": "cryptocurrency", "MATIC-USD": "cryptocurrency",
        # Cash
        "BIL": "cash", "SGOV": "cash", "CSHI": "cash", "ICSH": "cash",
    }

    # Paths to Kaggle text datasets (relative to repo root, same level as datasets/)
    _KAGGLE_STOCK_NEWS_DIR = Path("datasets/kaggle/equities/stock-data-with-news")
    _KAGGLE_CRYPTO_NEWS_CSV = Path("datasets/kaggle/cryptocurrency/crypto-news/cryptonews.csv")

    def __init__(
        self,
        data_dir: str = "datasets/processed",
        sec_dir: str = "datasets/sec",
        regime_csv: Optional[str] = None,
        fill_limit: int = 3,
        cache_on_init: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.sec_dir = Path(sec_dir)
        self.fill_limit = fill_limit

        # Lazily-loaded Kaggle text caches: ticker -> DataFrame
        self._kaggle_stock_cache: dict[str, "pd.DataFrame"] = {}
        self._kaggle_crypto_df: Optional["pd.DataFrame"] = None

        # Validate data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Processed data directory not found: {self.data_dir}\n"
                "Run `python examples/data_collect/get_all.py` and "
                "`python examples/data_preprocess/preprocess_all.py` first."
            )

        # Load all CSVs into memory
        self._frames: dict[str, pd.DataFrame] = {}
        if cache_on_init:
            self._load_all()

        # Load pre-computed market regimes if available
        self._regime_df: Optional[pd.DataFrame] = None
        if regime_csv:
            self._regime_df = self._load_csv(regime_csv)
        elif (self.data_dir / "market_regimes.csv").exists():
            self._regime_df = self._load_csv(str(self.data_dir / "market_regimes.csv"))

    # -----------------------------------------------------------------------
    # DataProvider interface implementation
    # -----------------------------------------------------------------------

    def get_price_series(self, asset: str, start: date, end: date) -> pd.Series:
        """
        Return daily close prices for asset in [start, end].

        Column lookup order:
          1. yahoo_<ASSET>_close
          2. kaggle_<ASSET>_close
          3. Any column whose name contains the asset ticker and ends with _close
        """
        df = self._get_frame(asset)
        col = self._find_column(df, asset, suffix="_close")
        if col is None:
            raise KeyError(f"No close-price column found for asset '{asset}'")

        series = df[col].copy()
        series = self._ffill(series)
        mask = (series.index.date >= start) & (series.index.date <= end)
        return series[mask].dropna()

    def get_return_series(self, asset: str, start: date, end: date) -> pd.Series:
        """
        Return daily simple returns for asset in [start, end].

        Prefers pre-computed _return columns; falls back to pct_change of close prices.
        """
        df = self._get_frame(asset)
        ret_col = self._find_column(df, asset, suffix="_return")
        if ret_col is not None:
            series = df[ret_col].copy()
            series = self._ffill(series)
            mask = (series.index.date >= start) & (series.index.date <= end)
            return series[mask].dropna()

        # Fallback: compute from close prices
        prices = self.get_price_series(asset, start - timedelta(days=5), end)
        return prices.pct_change().dropna()

    def get_macro(self, d: date) -> dict[str, float]:
        """
        Return macro indicators as of date d (most recent observation ≤ d).

        Reads from the cash.csv and bonds.csv frames.  Missing values are
        interpolated with the last known observation.
        """
        result = {}
        for macro_key, col_name in _MACRO_COLUMNS.items():
            # Determine which frame holds this column
            frame_key = "bonds" if col_name.startswith("fred_DGS") or col_name.startswith("fred_T10") \
                                   or col_name.startswith("fred_BAML") else "cash"
            df = self._frames.get(frame_key)
            if df is None or col_name not in df.columns:
                result[macro_key] = 0.0
                continue
            # Last observation on or before d
            candidates = df.loc[df.index.date <= d, col_name].dropna()
            result[macro_key] = float(candidates.iloc[-1]) if not candidates.empty else 0.0

        return result

    def get_regime(self, d: date, asset: str = "SPY") -> MarketRegime:
        """
        Return market regime for date d.

        Uses pre-computed market_regimes.csv if available; otherwise computes
        on-the-fly from 6-month trailing SPY return (same logic as MockDataProvider).
        """
        if self._regime_df is not None and "regime" in self._regime_df.columns:
            candidates = self._regime_df.loc[self._regime_df.index.date <= d, "regime"]
            if not candidates.empty:
                return MarketRegime(candidates.iloc[-1])

        # On-the-fly fallback
        half_year_ago = d - timedelta(days=126)
        try:
            prices = self.get_price_series("SPY", half_year_ago, d)
        except (KeyError, Exception):
            return MarketRegime.SIDEWAYS

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
        """
        Return list of available asset tickers.

        If asset_class is specified (e.g. "equities"), only return tickers
        from that class.
        """
        if asset_class:
            return [t for t, cls in self._ASSET_CLASS_MAP.items() if cls == asset_class]
        return list(self._ASSET_CLASS_MAP.keys())

    def get_news(self, asset: str, before_date: date) -> str:
        """
        Return the most recent text context strictly before before_date for asset.

        Priority order:
          1. SEC 10-K / 10-Q filings (equities only, ~20 tickers)
          2. Kaggle stock-data-with-news daily news column (99 equity tickers)
          3. Kaggle crypto-news (all crypto assets, keyed by subject keywords)

        Returns empty string if no text is available for this asset/date.
        """
        ticker = asset.upper()

        # 1. SEC filings
        sec_text = self._get_sec_news(ticker, before_date)
        if sec_text:
            return sec_text

        # 2. Kaggle per-ticker stock news (equities)
        kaggle_equity = self._get_kaggle_stock_news(ticker, before_date)
        if kaggle_equity:
            return kaggle_equity

        # 3. Kaggle crypto news (for crypto assets)
        asset_class = self._ASSET_CLASS_MAP.get(asset)
        if asset_class == "cryptocurrency":
            return self._get_kaggle_crypto_news(before_date)

        return ""

    def _get_sec_news(self, ticker: str, before_date: date) -> str:
        """Return most recent SEC 10-K/10-Q htm text before before_date."""
        ticker_dir = self.sec_dir / "equities" / ticker
        if not ticker_dir.exists():
            return ""

        candidates: list[tuple[date, Path]] = []
        for form in ("10-K", "10-Q"):
            form_dir = ticker_dir / form
            if not form_dir.exists():
                continue
            for p in form_dir.glob("*.htm"):
                m = re.match(r"^(\d{4}-\d{2}-\d{2})_", p.name)
                if not m:
                    continue
                try:
                    filing_date = date.fromisoformat(m.group(1))
                except ValueError:
                    continue
                if filing_date < before_date:
                    candidates.append((filing_date, p))

        if not candidates:
            return ""

        candidates.sort(key=lambda x: x[0], reverse=True)
        filing_date, filing_path = candidates[0]
        return self._read_htm_text(filing_path, max_chars=2000, filing_date=filing_date)

    def _get_kaggle_stock_news(self, ticker: str, before_date: date) -> str:
        """
        Return most recent news headline(s) from Kaggle stock-data-with-news
        for ticker strictly before before_date.
        """
        csv_path = self._KAGGLE_STOCK_NEWS_DIR / f"{ticker}.csv"
        if not csv_path.exists():
            return ""

        if ticker not in self._kaggle_stock_cache:
            try:
                df = pd.read_csv(csv_path, usecols=["date", "news"], parse_dates=["date"])
                df = df.dropna(subset=["news"])
                df["date"] = pd.to_datetime(df["date"]).dt.date
                self._kaggle_stock_cache[ticker] = df.sort_values("date")
            except Exception:
                return ""

        df = self._kaggle_stock_cache[ticker]
        candidates = df[df["date"] < before_date]
        if candidates.empty:
            return ""

        row = candidates.iloc[-1]
        news_text = str(row["news"])[:1500]
        news_date = row["date"]
        return f"[News {news_date}] {news_text}"

    def _get_kaggle_crypto_news(self, before_date: date) -> str:
        """
        Return the most recent crypto news headline before before_date
        from the Kaggle crypto-news dataset.
        """
        if not self._KAGGLE_CRYPTO_NEWS_CSV.exists():
            return ""

        if self._kaggle_crypto_df is None:
            try:
                df = pd.read_csv(
                    self._KAGGLE_CRYPTO_NEWS_CSV,
                    usecols=["date", "title", "text"],
                )
                df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True).dt.tz_localize(None).dt.date
                self._kaggle_crypto_df = df.sort_values("date")
            except Exception:
                return ""

        df = self._kaggle_crypto_df
        candidates = df[df["date"] < before_date]
        if candidates.empty:
            return ""

        row = candidates.iloc[-1]
        title = str(row.get("title", ""))
        text = str(row.get("text", ""))[:1200]
        news_date = row["date"]
        return f"[Crypto news {news_date}] {title}. {text}"

    @staticmethod
    def _read_htm_text(path: Path, max_chars: int = 2000, filing_date: Optional[date] = None) -> str:
        """Read an HTM file, strip tags, and return a plain-text excerpt."""
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        # Strip HTML tags with a simple regex (avoids heavy dependency on bs4)
        text = re.sub(r"<[^>]+>", " ", raw)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        prefix = f"[SEC filing {filing_date}] " if filing_date else ""
        return prefix + text[:max_chars]

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all per-asset-class CSVs into self._frames."""
        for cls, filename in self._CLASS_TO_FILE.items():
            path = self.data_dir / filename
            if path.exists():
                self._frames[cls] = self._load_csv(str(path))
            else:
                # Warn but don't fail — some asset classes may not be downloaded yet
                import warnings
                warnings.warn(
                    f"Processed file not found: {path} — {cls} data will be unavailable.",
                    UserWarning,
                    stacklevel=2,
                )

    @staticmethod
    def _load_csv(path: str) -> pd.DataFrame:
        """Load a processed CSV with DatetimeIndex."""
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        return df

    def _get_frame(self, asset: str) -> pd.DataFrame:
        """Return the DataFrame for the asset class that owns this ticker."""
        cls = self._ASSET_CLASS_MAP.get(asset)
        if cls is None:
            # Unknown ticker — try all frames
            for frame in self._frames.values():
                for prefix in _SOURCE_PREFIXES:
                    if any(asset in col for col in frame.columns):
                        return frame
            raise KeyError(
                f"Asset '{asset}' not found in ProcessedDataProvider. "
                f"Check that it is in _ASSET_CLASS_MAP or that the column "
                f"exists in a processed CSV."
            )
        if cls not in self._frames:
            # Lazy load
            path = self.data_dir / self._CLASS_TO_FILE[cls]
            if not path.exists():
                raise FileNotFoundError(
                    f"Processed file not found: {path}. "
                    "Run the data collection and preprocessing pipeline first."
                )
            self._frames[cls] = self._load_csv(str(path))
        return self._frames[cls]

    @staticmethod
    def _find_column(df: pd.DataFrame, asset: str, suffix: str) -> Optional[str]:
        """
        Find the best matching column for an asset with a given suffix.

        Search order: yahoo → kaggle → fred → any column containing asset name + suffix.
        """
        for prefix in _SOURCE_PREFIXES:
            candidate = f"{prefix}_{asset}{suffix}"
            if candidate in df.columns:
                return candidate
        # Fallback: fuzzy search (e.g., "SPY_close" without source prefix)
        asset_upper = asset.upper()
        for col in df.columns:
            if asset_upper in col.upper() and col.endswith(suffix):
                return col
        return None

    def _ffill(self, series: pd.Series) -> pd.Series:
        """Forward-fill up to fill_limit consecutive NaNs (handles non-trading days)."""
        return series.ffill(limit=self.fill_limit)
