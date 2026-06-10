"""S&P 500 Top-50 collector for PortBench.

Downloads OHLCV data for the top 50 S&P 500 constituents by market
capitalization, covering all major GICS sectors. Uses Yahoo Finance
(yfinance) via PortBench's existing YahooCollector infrastructure.

The output format is identical to YahooCollector (date, open, high,
low, close, volume CSVs), so the same preprocessing pipeline applies.
"""

import time
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from .base import DataCollector, AssetClass, DataType, DatasetMetadata


@dataclass
class SP500Ticker:
    """S&P 500 constituent configuration."""

    symbol: str
    sector: str
    description: str


# ---------------------------------------------------------------------------
# S&P 500 Top-50 by market cap (approx. early 2026)
#
# Covers all 11 GICS sectors:
#   Technology:        14 stocks
#   Financials:         8 stocks
#   Healthcare:         7 stocks
#   Consumer Disc:      5 stocks
#   Consumer Staples:   4 stocks
#   Industrials:        5 stocks
#   Communication:      3 stocks
#   Energy:             2 stocks
#   Utilities:          1 stock
#   Real Estate:        1 stock
# ---------------------------------------------------------------------------

SP500_TICKERS = [
    # --- Technology ---
    SP500Ticker("AAPL", "Technology", "Apple Inc."),
    SP500Ticker("MSFT", "Technology", "Microsoft Corp."),
    SP500Ticker("NVDA", "Technology", "NVIDIA Corp."),
    SP500Ticker("AVGO", "Technology", "Broadcom Inc."),
    SP500Ticker("ORCL", "Technology", "Oracle Corp."),
    SP500Ticker("CRM", "Technology", "Salesforce Inc."),
    SP500Ticker("AMD", "Technology", "Advanced Micro Devices"),
    SP500Ticker("ADBE", "Technology", "Adobe Inc."),
    SP500Ticker("QCOM", "Technology", "Qualcomm Inc."),
    SP500Ticker("TXN", "Technology", "Texas Instruments"),
    SP500Ticker("IBM", "Technology", "IBM"),
    SP500Ticker("NOW", "Technology", "ServiceNow Inc."),
    SP500Ticker("AMAT", "Technology", "Applied Materials"),
    SP500Ticker("MU", "Technology", "Micron Technology"),
    # --- Financials ---
    SP500Ticker("BRK-B", "Financials", "Berkshire Hathaway Inc."),
    SP500Ticker("JPM", "Financials", "JPMorgan Chase & Co."),
    SP500Ticker("V", "Financials", "Visa Inc."),
    SP500Ticker("MA", "Financials", "Mastercard Inc."),
    SP500Ticker("BAC", "Financials", "Bank of America Corp."),
    SP500Ticker("WFC", "Financials", "Wells Fargo & Co."),
    SP500Ticker("GS", "Financials", "Goldman Sachs Group"),
    SP500Ticker("BLK", "Financials", "BlackRock Inc."),
    # --- Healthcare ---
    SP500Ticker("LLY", "Healthcare", "Eli Lilly and Co."),
    SP500Ticker("UNH", "Healthcare", "UnitedHealth Group"),
    SP500Ticker("JNJ", "Healthcare", "Johnson & Johnson"),
    SP500Ticker("ABBV", "Healthcare", "AbbVie Inc."),
    SP500Ticker("MRK", "Healthcare", "Merck & Co."),
    SP500Ticker("TMO", "Healthcare", "Thermo Fisher Scientific"),
    SP500Ticker("ABT", "Healthcare", "Abbott Laboratories"),
    # --- Consumer Discretionary ---
    SP500Ticker("AMZN", "ConsumerDisc", "Amazon.com Inc."),
    SP500Ticker("TSLA", "ConsumerDisc", "Tesla Inc."),
    SP500Ticker("HD", "ConsumerDisc", "Home Depot Inc."),
    SP500Ticker("MCD", "ConsumerDisc", "McDonald's Corp."),
    SP500Ticker("NKE", "ConsumerDisc", "Nike Inc."),
    # --- Consumer Staples ---
    SP500Ticker("PG", "ConsumerStaples", "Procter & Gamble Co."),
    SP500Ticker("KO", "ConsumerStaples", "Coca-Cola Co."),
    SP500Ticker("PEP", "ConsumerStaples", "PepsiCo Inc."),
    SP500Ticker("COST", "ConsumerStaples", "Costco Wholesale Corp."),
    # --- Industrials ---
    SP500Ticker("GE", "Industrials", "GE Aerospace"),
    SP500Ticker("CAT", "Industrials", "Caterpillar Inc."),
    SP500Ticker("UNP", "Industrials", "Union Pacific Corp."),
    SP500Ticker("RTX", "Industrials", "RTX Corp."),
    SP500Ticker("HON", "Industrials", "Honeywell International"),
    # --- Communication Services ---
    SP500Ticker("GOOGL", "Communication", "Alphabet Inc. (Class A)"),
    SP500Ticker("META", "Communication", "Meta Platforms Inc."),
    SP500Ticker("NFLX", "Communication", "Netflix Inc."),
    # --- Energy ---
    SP500Ticker("XOM", "Energy", "Exxon Mobil Corp."),
    SP500Ticker("CVX", "Energy", "Chevron Corp."),
    # --- Utilities ---
    SP500Ticker("NEE", "Utilities", "NextEra Energy Inc."),
    # --- Real Estate ---
    SP500Ticker("PLD", "RealEstate", "Prologis Inc."),
]


class SP500Collector(DataCollector):
    """Collector for S&P 500 Top-50 stock data via Yahoo Finance.

    Downloads daily OHLCV data for the top 50 S&P 500 constituents,
    using the same format as PortBench's YahooCollector.

    Args:
        base_dir:   Base directory for storing downloaded datasets.
        start_date: Start date for data download (YYYY-MM-DD).
        end_date:   End date for data download. If None, uses today.
    """

    def __init__(
        self,
        base_dir: str = "datasets",
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
    ):
        super().__init__(base_dir)
        self.start_date = start_date
        self.end_date = end_date

    @property
    def source_name(self) -> str:
        return "sp500"

    def download(
        self,
        dataset_id: str,
        asset_class: AssetClass,
        force: bool = False,
        description: str = "",
    ) -> Path:
        """Download a single S&P 500 stock.

        Args:
            dataset_id:   Ticker symbol (e.g., "AAPL", "MSFT").
            asset_class:  Asset class (always EQUITIES for SP500).
            force:        If True, re-download even if exists.
            description:  Optional description for metadata.

        Returns:
            Path to the downloaded CSV file.
        """
        target_dir = self.get_asset_dir(asset_class)
        target_file = target_dir / f"{dataset_id}.csv"

        if not force and self._is_complete(target_file, dataset_id):
            print(f"Stock already complete: {target_file}")
            return target_file

        print(f"Downloading {dataset_id}...")

        # Download with retry (same pattern as YahooCollector)
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(dataset_id)
                df = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1} failed: {e}")
                    print(f"  Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

        if df is None or df.empty:
            raise ValueError(f"No data returned for {dataset_id}")

        # Standardize format (same as YahooCollector)
        df = df.reset_index()
        df = df.rename(columns={"Date": "date"})

        columns_map = {
            "date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = df[[c for c in columns_map if c in df.columns]]
        df = df.rename(columns=columns_map)

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.dropna()

        df.to_csv(target_file, index=False)
        print(f"  Saved to: {target_file} ({len(df)} rows)")

        # Update metadata
        self.update_metadata(
            DatasetMetadata(
                dataset_id=dataset_id,
                asset_class=asset_class.value,
                source=self.source_name,
                description=description or f"S&P 500: {dataset_id}",
                file_path=str(target_file),
                download_time=datetime.now().isoformat(),
                data_type=DataType.NUMERIC.value,
                file_format="csv",
                rows=len(df),
                columns=len(df.columns),
                start_date=df["date"].min().strftime("%Y-%m-%d"),
                end_date=df["date"].max().strftime("%Y-%m-%d"),
            )
        )

        # Rate limiting
        time.sleep(1)

        return target_file

    def download_all(self, force: bool = False) -> dict[AssetClass, list[Path]]:
        """Download all 50 S&P 500 stocks.

        Args:
            force: If True, re-download even if exists.

        Returns:
            Dict mapping asset classes to list of downloaded paths.
        """
        result: dict[AssetClass, list[Path]] = {}

        for ticker in SP500_TICKERS:
            try:
                path = self.download(
                    dataset_id=ticker.symbol,
                    asset_class=AssetClass.EQUITIES,
                    force=force,
                    description=ticker.description,
                )
                if AssetClass.EQUITIES not in result:
                    result[AssetClass.EQUITIES] = []
                result[AssetClass.EQUITIES].append(path)
            except Exception as e:
                print(f"[ERROR] Failed {ticker.symbol}: {e}")
                continue

        return result

    def list_available(self) -> dict[str, list[SP500Ticker]]:
        """List all available S&P 500 tickers grouped by sector."""
        result: dict[str, list[SP500Ticker]] = {}
        for ticker in SP500_TICKERS:
            if ticker.sector not in result:
                result[ticker.sector] = []
            result[ticker.sector].append(ticker)
        return result

    def get_symbols(self) -> list[str]:
        """Return list of all SP500 ticker symbols."""
        return [t.symbol for t in SP500_TICKERS]
