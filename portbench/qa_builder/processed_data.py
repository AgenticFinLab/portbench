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
import json
from pathlib import Path
from typing import Optional
import re

import pandas as pd

from .base import DataProvider, MarketRegime


# ---------------------------------------------------------------------------
# Column name conventions in processed CSVs
# ---------------------------------------------------------------------------

# Standard close-price column suffix written by each preprocessor
_CLOSE_SUFFIX = "_close"
_RETURN_SUFFIX = "_return"

# Macro series fetched from the CASH / BONDS / EQUITIES CSVs.
# Each entry: indicator_name -> (csv_frame_key, column_name)
_MACRO_COLUMNS: dict[str, tuple[str, str]] = {
    # Money / rates (from cash.csv)
    "fed_funds_rate": ("cash", "fred_DFF"),
    "cpi_yoy": ("cash", "fred_CPIAUCSL"),
    "unemployment": ("cash", "fred_UNRATE"),
    "gdp_growth_qoq": ("cash", "fred_GDPC1"),
    # Yield curve / credit (from bonds.csv)
    "t10y2y_spread": ("bonds", "fred_T10Y2Y"),
    "t10y3m_spread": ("bonds", "fred_T10Y3M"),
    "breakeven_10y": ("bonds", "fred_T10YIE"),
    "hy_oas": ("bonds", "fred_BAMLH0A0HYM2"),
    "ig_oas": ("bonds", "fred_BAMLC0A0CM"),
    "ted_spread": ("bonds", "fred_TEDRATE"),
    "mortgage_30y": ("bonds", "fred_MORTGAGE30US"),
    # Equity volatility (from equities.csv — VIX lives here)
    "vix": ("equities", "^VIX_close"),
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
        "equities": "equities.csv",
        "bonds": "bonds.csv",
        "commodities": "commodities.csv",
        "real_estate": "real_estate.csv",
        "cryptocurrency": "cryptocurrency.csv",
        "cash": "cash.csv",
    }

    # Hard-coded asset class membership for the standard ticker universe
    _ASSET_CLASS_MAP = {
        # Equities
        "SPY": "equities",
        "QQQ": "equities",
        "IWM": "equities",
        "VTI": "equities",
        "IVV": "equities",
        "EEM": "equities",
        "EFA": "equities",
        "VEA": "equities",
        "FXI": "equities",
        "EWJ": "equities",
        "ACWI": "equities",
        "XLK": "equities",
        "XLF": "equities",
        "XLE": "equities",
        "XLV": "equities",
        "XLI": "equities",
        "XLY": "equities",
        "XLP": "equities",
        "XLU": "equities",
        "XLB": "equities",
        "XLRE": "equities",
        "MTUM": "equities",
        "VLUE": "equities",
        "QUAL": "equities",
        "USMV": "equities",
        "^VIX": "equities",
        # Bonds
        "SHV": "bonds",
        "SHY": "bonds",
        "IEF": "bonds",
        "TLT": "bonds",
        "TLH": "bonds",
        "TIP": "bonds",
        "SCHP": "bonds",
        "STIP": "bonds",
        "LQD": "bonds",
        "HYG": "bonds",
        "JNK": "bonds",
        "VCIT": "bonds",
        "BNDX": "bonds",
        "EMB": "bonds",
        "IGOV": "bonds",
        # Commodities
        "GLD": "commodities",
        "IAU": "commodities",
        "SLV": "commodities",
        "PPLT": "commodities",
        "PALL": "commodities",
        "USO": "commodities",
        "UNG": "commodities",
        "BNO": "commodities",
        "DBA": "commodities",
        "CORN": "commodities",
        "WEAT": "commodities",
        "SOYB": "commodities",
        "DBC": "commodities",
        "PDBC": "commodities",
        "DBB": "commodities",
        "CPER": "commodities",
        # Real Estate
        "VNQ": "real_estate",
        "IYR": "real_estate",
        "SCHH": "real_estate",
        "REZ": "real_estate",
        "MORT": "real_estate",
        "INDS": "real_estate",
        "VNQI": "real_estate",
        "HAUZ": "real_estate",
        "XHB": "real_estate",
        "ITB": "real_estate",
        # Crypto
        "BTC-USD": "cryptocurrency",
        "ETH-USD": "cryptocurrency",
        "BNB-USD": "cryptocurrency",
        "SOL-USD": "cryptocurrency",
        "XRP-USD": "cryptocurrency",
        "ADA-USD": "cryptocurrency",
        "AVAX-USD": "cryptocurrency",
        "DOT-USD": "cryptocurrency",
        "LINK-USD": "cryptocurrency",
        "MATIC-USD": "cryptocurrency",
        # Cash
        "BIL": "cash",
        "SGOV": "cash",
        "CSHI": "cash",
        "ICSH": "cash",
        # FF49 Industries (for datasets/processed_ff49/ dataset)
        "Agric": "equities", "Food": "equities", "Soda": "equities",
        "Beer": "equities", "Smoke": "equities", "Toys": "equities",
        "Fun": "equities", "Books": "equities", "Hshld": "equities",
        "Clths": "equities", "Hlth": "equities", "MedEq": "equities",
        "Drugs": "equities", "Chems": "equities", "Rubbr": "equities",
        "Txtls": "equities", "BldMt": "equities", "Cnstr": "equities",
        "Steel": "equities", "FabPr": "equities", "Mach": "equities",
        "ElcEq": "equities", "Autos": "equities", "Aero": "equities",
        "Ships": "equities", "Guns": "equities", "Gold": "equities",
        "Mines": "equities", "Coal": "equities", "Oil": "equities",
        "Util": "equities", "Telcm": "equities", "PerSv": "equities",
        "BusSv": "equities", "Hardw": "equities", "Softw": "equities",
        "Chips": "equities", "LabEq": "equities", "Paper": "equities",
        "Boxes": "equities", "Trans": "equities", "Whlsl": "equities",
        "Rtail": "equities", "Meals": "equities", "Banks": "equities",
        "Insur": "equities", "RlEst": "equities", "Fin": "equities",
        "Other": "equities",
        # SP500 Top-50 (for datasets/processed_sp500/ dataset)
        "AAPL": "equities", "MSFT": "equities", "NVDA": "equities",
        "AVGO": "equities", "ORCL": "equities", "CRM": "equities",
        "AMD": "equities", "ADBE": "equities", "QCOM": "equities",
        "TXN": "equities", "IBM": "equities", "NOW": "equities",
        "AMAT": "equities", "MU": "equities", "BRK-B": "equities",
        "JPM": "equities", "V": "equities", "MA": "equities",
        "BAC": "equities", "WFC": "equities", "GS": "equities",
        "BLK": "equities", "LLY": "equities", "UNH": "equities",
        "JNJ": "equities", "ABBV": "equities", "MRK": "equities",
        "TMO": "equities", "ABT": "equities", "AMZN": "equities",
        "TSLA": "equities", "HD": "equities", "MCD": "equities",
        "NKE": "equities", "PG": "equities", "KO": "equities",
        "PEP": "equities", "COST": "equities", "GE": "equities",
        "CAT": "equities", "UNP": "equities", "RTX": "equities",
        "HON": "equities", "GOOGL": "equities", "META": "equities",
        "NFLX": "equities", "XOM": "equities", "CVX": "equities",
        "NEE": "equities", "PLD": "equities",
    }

    # Paths to Kaggle text datasets (relative to repo root, same level as datasets/)
    _KAGGLE_STOCK_NEWS_DIR = Path("datasets/kaggle/equities/stock-data-with-news")
    _KAGGLE_CRYPTO_NEWS_CSV = Path(
        "datasets/kaggle/cryptocurrency/crypto-news/cryptonews.csv"
    )

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

        Reads from cash.csv, bonds.csv, and equities.csv as mapped in
        _MACRO_COLUMNS. Missing values default to 0.0.
        """
        result = {}
        for macro_key, (frame_key, col_name) in _MACRO_COLUMNS.items():
            df = self._frames.get(frame_key)
            if df is None or col_name not in df.columns:
                result[macro_key] = 0.0
                continue
            candidates = df.loc[df.index.date <= d, col_name].dropna()
            result[macro_key] = (
                float(candidates.iloc[-1]) if not candidates.empty else 0.0
            )

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

    def get_volume_series(self, asset: str, start: date, end: date) -> pd.Series:
        """Return daily trading volume for asset in [start, end]."""
        df = self._get_frame(asset)
        col = self._find_column(df, asset, suffix="_volume")
        if col is None:
            return pd.Series(dtype=float)
        series = df[col].copy()
        series = self._ffill(series)
        mask = (series.index.date >= start) & (series.index.date <= end)
        return series[mask].dropna()

    def get_ohlc_series(self, asset: str, start: date, end: date) -> pd.DataFrame:
        """
        Return daily OHLC DataFrame for asset in [start, end].

        Columns: open, high, low, close (float). Index: DatetimeIndex.
        """
        df = self._get_frame(asset)
        ohlc: dict[str, pd.Series] = {}
        for field_name in ("open", "high", "low", "close"):
            col = self._find_column(df, asset, suffix=f"_{field_name}")
            if col is not None:
                series = self._ffill(df[col].copy())
                mask = (series.index.date >= start) & (series.index.date <= end)
                ohlc[field_name] = series[mask]
        if not ohlc:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        result = pd.DataFrame(ohlc)
        result = result.dropna(how="all")
        return result

    def get_asset_metadata(self, asset: str) -> dict:
        """
        Return static metadata for an asset.

        For cryptocurrency assets: returns available columns such as
        launch_year, market_cap, circulating_supply, platform, cmc_rank, tvl.
        For other assets: returns asset_class.
        """
        asset_class = self._ASSET_CLASS_MAP.get(asset)
        meta: dict = {"asset_class": asset_class}

        if asset_class == "cryptocurrency":
            df = self._frames.get("cryptocurrency")
            if df is None:
                return meta
            # Crypto metadata columns written by CryptocurrencyPreprocessor
            meta_suffixes = (
                "launch_year",
                "market_cap",
                "circulating_supply",
                "platform",
                "cmc_rank",
                "tvl",
            )
            # Column names vary by Kaggle dataset name prefix; search by suffix
            for sfx in meta_suffixes:
                for col in df.columns:
                    if col.endswith(f"_{sfx}") or col == sfx:
                        # Take most recent non-null value (static field, rarely changes)
                        values = df[col].dropna()
                        if not values.empty:
                            meta[sfx] = values.iloc[-1]
                        break

        return meta

    def has_text(self, asset: str, before_date: date) -> bool:
        """
        Fast check: does any text_json record exist for this asset class strictly before before_date? Used to rank candidate dates without invoking the full get_news() pipeline (which sorts records and formats output).
        """
        cls = self._ASSET_CLASS_MAP.get(asset)
        if cls not in ("equities", "cryptocurrency"):
            return False
        df = self._frames.get(cls)
        if df is None or "text_json" not in df.columns:
            return False
        mask = (df.index.date < before_date) & df["text_json"].notna()
        return bool(mask.any())

    def get_news(self, asset: str, before_date: date) -> str:
        """
        Return the most recent text context strictly before before_date for asset.

        Priority order:
          1. Preprocessed text_json column from loaded frames (equities/cryptocurrency CSV)
             — this is the canonical source, produced by the preprocessing pipeline and
             aggregates SEC filings + Kaggle news per trading day.
          2. Raw SEC 10-K/10-Q htm files (fallback for equities not yet re-preprocessed)
          3. Raw Kaggle stock-data-with-news CSV (per-ticker fallback for equities)
          4. Raw Kaggle crypto-news CSV (fallback for cryptocurrency assets)

        Returns empty string if no text is available for this asset/date.
        """
        asset_class = self._ASSET_CLASS_MAP.get(asset)

        # 1. Preprocessed text_json from loaded frames (fastest, most complete)
        frame_key = (
            asset_class if asset_class in ("equities", "cryptocurrency") else None
        )
        if frame_key:
            text = self._get_text_from_frame(frame_key, before_date)
            if text:
                return text

        # 2. Raw SEC filings fallback (equities only)
        if asset_class == "equities":
            sec_text = self._get_sec_news(asset.upper(), before_date)
            if sec_text:
                return sec_text

            # 3. Kaggle per-ticker stock news fallback
            kaggle_text = self._get_kaggle_stock_news(asset.upper(), before_date)
            if kaggle_text:
                return kaggle_text

        # 4. Raw Kaggle crypto-news fallback
        if asset_class == "cryptocurrency":
            return self._get_kaggle_crypto_news(before_date)

        return ""

    def _get_text_from_frame(self, frame_key: str, before_date: date) -> str:
        """
        Read the most recent text_json entry from the loaded frame for frame_key,
        strictly before before_date. Returns a plain-text excerpt or empty string.
        """
        df = self._frames.get(frame_key)
        if df is None or "text_json" not in df.columns:
            return ""

        # Filter to rows before before_date that have text
        mask = (df.index.date < before_date) & df["text_json"].notna()
        candidates = df.loc[mask, "text_json"]
        if candidates.empty:
            return ""

        raw = candidates.iloc[-1]  # most recent entry
        try:
            records = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Stored as Python repr (old format) — extract plain text with regex
            texts = re.findall(r"'text':\s*'([^']{20,})'", str(raw))
            if texts:
                return f"[News {candidates.index[-1].date()}] {texts[0][:3000]}"
            return ""

        if not records:
            return ""

        # Pick the most informative record: prefer SEC over kaggle, longer text
        records.sort(
            key=lambda r: (r.get("source") == "sec", len(r.get("text", ""))),
            reverse=True,
        )
        best = records[0]
        source = best.get("source", "news")
        ticker = best.get("ticker", "")
        filing_type = best.get("type", "")
        # Respect the per-source text budgets from the preprocessing pipeline:
        #   SEC (head_tail): up to 9000 chars already trimmed by preprocess
        #   Kaggle stock:    up to 3000 chars
        #   Kaggle crypto:   up to 2000 chars
        # Do not re-truncate here — the preprocess budget is already applied.
        text = str(best.get("text", ""))
        entry_date = candidates.index[-1].date()

        if source == "sec" and ticker and filing_type:
            return f"[SEC {filing_type} {ticker} {entry_date}] {text}"
        return f"[{source.capitalize()} {entry_date}] {text}"

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
        return self._read_htm_text(filing_path, max_chars=9000, filing_date=filing_date)

    def _get_kaggle_stock_news(self, ticker: str, before_date: date) -> str:
        """
        Return most recent news from Kaggle stock-data-with-news for ticker
        strictly before before_date.
        """
        csv_path = self._KAGGLE_STOCK_NEWS_DIR / f"{ticker}.csv"
        if not csv_path.exists():
            return ""

        if ticker not in self._kaggle_stock_cache:
            try:
                df = pd.read_csv(
                    csv_path, usecols=["date", "news"], parse_dates=["date"]
                )
                df = df.dropna(subset=["news"])
                df["date"] = pd.to_datetime(df["date"]).dt.date
                self._kaggle_stock_cache[ticker] = df.sort_values("date")
            except Exception:
                return ""

        df = self._kaggle_stock_cache[ticker]
        rows = df[df["date"] < before_date]
        if rows.empty:
            return ""

        row = rows.iloc[-1]
        return f"[News {row['date']}] {str(row['news'])[:3000]}"

    def _get_kaggle_crypto_news(self, before_date: date) -> str:
        """
        Return the most recent crypto news before before_date
        from the Kaggle crypto-news dataset.
        """
        if not self._KAGGLE_CRYPTO_NEWS_CSV.exists():
            return ""

        if self._kaggle_crypto_df is None:
            try:
                df = pd.read_csv(
                    self._KAGGLE_CRYPTO_NEWS_CSV, usecols=["date", "title", "text"]
                )
                df["date"] = (
                    pd.to_datetime(df["date"], format="mixed", utc=True)
                    .dt.tz_localize(None)
                    .dt.date
                )
                self._kaggle_crypto_df = df.sort_values("date")
            except Exception:
                return ""

        rows = self._kaggle_crypto_df
        rows = rows[rows["date"] < before_date]
        if rows.empty:
            return ""

        row = rows.iloc[-1]
        title = str(row.get("title", ""))
        text = str(row.get("text", ""))[:2000]
        return f"[Crypto news {row['date']}] {title}. {text}"

    @staticmethod
    def _read_htm_text(
        path: Path, max_chars: int = 2000, filing_date: Optional[date] = None
    ) -> str:
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
        """Load all per-asset-class CSVs into self._frames.

        After loading, filters _ASSET_CLASS_MAP to only include tickers
        whose columns actually exist in the loaded data. This ensures
        dataset isolation: loading FF49 data won't advertise PortBench
        tickers that aren't present.
        """
        # Instance-level copy so filtering doesn't affect other instances
        self._ASSET_CLASS_MAP = dict(self.__class__._ASSET_CLASS_MAP)

        for cls, filename in self._CLASS_TO_FILE.items():
            df = self._load_asset_frame(cls, filename)
            if df is not None:
                self._frames[cls] = df
            else:
                import warnings

                warnings.warn(
                    f"Processed file not found for {cls} in {self.data_dir} "
                    f"— {cls} data will be unavailable.",
                    UserWarning,
                    stacklevel=2,
                )

        # Filter asset map: keep only tickers with actual column data
        all_columns = set()
        for frame in self._frames.values():
            all_columns.update(frame.columns)

        self._ASSET_CLASS_MAP = {
            ticker: cls
            for ticker, cls in self._ASSET_CLASS_MAP.items()
            if self._ticker_has_columns(ticker, cls, all_columns)
        }

    def _ticker_has_columns(
        self, ticker: str, cls: str, all_columns: set
    ) -> bool:
        """Check if a ticker has matching columns in the loaded data."""
        for suffix in ["_close", "_return"]:
            # Check with each source prefix
            for prefix in _SOURCE_PREFIXES:
                if f"{prefix}_{ticker}{suffix}" in all_columns:
                    return True
            # Check bare name (for FF49/SP500 isolated datasets)
            if f"{ticker}{suffix}" in all_columns:
                return True
        return False

    def _load_asset_frame(self, cls: str, filename: str) -> Optional[pd.DataFrame]:
        """
        Load an asset-class frame, supporting both single-file and chunked
        (manifest + partNNN.csv) outputs from the preprocess pipeline.
        """
        single = self.data_dir / filename
        stem = filename[: -len(".csv")] if filename.endswith(".csv") else filename
        manifest = self.data_dir / f"{stem}.manifest.json"

        if manifest.exists():
            info = json.loads(manifest.read_text(encoding="utf-8"))
            parts = [self.data_dir / p for p in info.get("parts", [])]
            frames = [self._load_csv(str(p)) for p in parts if p.exists()]
            if frames:
                merged = pd.concat(frames).sort_index()
                # Collapse cross-chunk duplicates if any slipped through
                if merged.index.duplicated().any():
                    merged = merged.groupby(merged.index).first()
                return merged
            return None

        if single.exists():
            return self._load_csv(str(single))

        # Orphan parts without manifest
        orphan_parts = sorted(self.data_dir.glob(f"{stem}.part*.csv"))
        if orphan_parts:
            frames = [self._load_csv(str(p)) for p in orphan_parts]
            merged = pd.concat(frames).sort_index()
            if merged.index.duplicated().any():
                merged = merged.groupby(merged.index).first()
            return merged

        return None

    @staticmethod
    def _load_csv(path: str) -> pd.DataFrame:
        """Load a processed CSV with DatetimeIndex, deduplicating any repeated dates."""
        df = pd.read_csv(path, parse_dates=["date"], low_memory=False)
        # Some preprocessed CSVs (notably equities.csv after text merge) contain
        # duplicate date rows from the text-records explosion. Collapse to first
        # occurrence per date so price/return slicing returns scalars, not vectors.
        if df["date"].duplicated().any():
            df = df.groupby("date", as_index=False).first()
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
            df = self._load_asset_frame(cls, self._CLASS_TO_FILE[cls])
            if df is None:
                raise FileNotFoundError(
                    f"Processed data not found for {cls} in {self.data_dir}. "
                    "Run the data collection and preprocessing pipeline first."
                )
            self._frames[cls] = df
        return self._frames[cls]

    @staticmethod
    def _find_column(df: pd.DataFrame, asset: str, suffix: str) -> Optional[str]:
        """
        Find the best matching column for an asset with a given suffix.

        Search order: yahoo → kaggle → fred → any column containing asset name + suffix.
        Falls back to alternative suffixes (_open if _close missing) since some
        preprocessed CSVs drop _close after deduplication.
        """
        # Try exact prefixed match first
        for prefix in _SOURCE_PREFIXES:
            candidate = f"{prefix}_{asset}{suffix}"
            if candidate in df.columns:
                return candidate

        # Build asset name variants (BTC-USD -> BTC_USD for crypto column convention)
        variants = {asset.upper(), asset.upper().replace("-", "_")}

        # Try the requested suffix first, then fall back to _open / _close interchangeably
        suffix_chain = [suffix]
        if suffix == "_close":
            suffix_chain.append("_open")
        elif suffix == "_open":
            suffix_chain.append("_close")

        for sfx in suffix_chain:
            for col in df.columns:
                col_up = col.upper()
                if col.endswith(sfx) and any(v in col_up for v in variants):
                    return col
        return None

    def _ffill(self, series: pd.Series) -> pd.Series:
        """Forward-fill up to fill_limit consecutive NaNs (handles non-trading days)."""
        return series.ffill(limit=self.fill_limit)
