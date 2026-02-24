"""Yahoo Finance dataset collector for PortBench."""

import time
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from .base import DataCollector, AssetClass, DatasetMetadata


@dataclass
class YahooTicker:
    """Yahoo Finance ticker configuration."""

    symbol: str
    asset_class: AssetClass
    description: str


# Yahoo Finance tickers from docs/project-overview.md
YAHOO_TICKERS = [
    # Equities
    YahooTicker(
        symbol="SPY",
        asset_class=AssetClass.EQUITIES,
        description="SPDR S&P 500 ETF Trust",
    ),
    YahooTicker(
        symbol="QQQ",
        asset_class=AssetClass.EQUITIES,
        description="Invesco QQQ Trust (NASDAQ-100)",
    ),
    YahooTicker(
        symbol="IVV",
        asset_class=AssetClass.EQUITIES,
        description="iShares Core S&P 500 ETF",
    ),
    YahooTicker(
        symbol="EEM",
        asset_class=AssetClass.EQUITIES,
        description="iShares MSCI Emerging Markets ETF",
    ),
    # Bonds
    YahooTicker(
        symbol="TLT",
        asset_class=AssetClass.BONDS,
        description="iShares 20+ Year Treasury Bond ETF",
    ),
    YahooTicker(
        symbol="SHV",
        asset_class=AssetClass.BONDS,
        description="iShares Short Treasury Bond ETF",
    ),
    YahooTicker(
        symbol="IEF",
        asset_class=AssetClass.BONDS,
        description="iShares 7-10 Year Treasury Bond ETF",
    ),
    YahooTicker(
        symbol="LQD",
        asset_class=AssetClass.BONDS,
        description="iShares iBoxx Investment Grade Corporate Bond ETF",
    ),
    YahooTicker(
        symbol="HYG",
        asset_class=AssetClass.BONDS,
        description="iShares iBoxx High Yield Corporate Bond ETF",
    ),
    # Commodities
    YahooTicker(
        symbol="GLD",
        asset_class=AssetClass.COMMODITIES,
        description="SPDR Gold Shares",
    ),
    YahooTicker(
        symbol="SLV",
        asset_class=AssetClass.COMMODITIES,
        description="iShares Silver Trust",
    ),
    YahooTicker(
        symbol="USO",
        asset_class=AssetClass.COMMODITIES,
        description="United States Oil Fund",
    ),
    YahooTicker(
        symbol="DBC",
        asset_class=AssetClass.COMMODITIES,
        description="Invesco DB Commodity Index Tracking Fund",
    ),
    # Real Estate
    YahooTicker(
        symbol="VNQ",
        asset_class=AssetClass.REAL_ESTATE,
        description="Vanguard Real Estate ETF",
    ),
    YahooTicker(
        symbol="IYR",
        asset_class=AssetClass.REAL_ESTATE,
        description="iShares U.S. Real Estate ETF",
    ),
    # Cryptocurrency
    YahooTicker(
        symbol="BTC-USD",
        asset_class=AssetClass.CRYPTOCURRENCY,
        description="Bitcoin USD",
    ),
    YahooTicker(
        symbol="ETH-USD",
        asset_class=AssetClass.CRYPTOCURRENCY,
        description="Ethereum USD",
    ),
    # Cash equivalents
    YahooTicker(
        symbol="BIL",
        asset_class=AssetClass.CASH,
        description="SPDR Bloomberg 1-3 Month T-Bill ETF",
    ),
]


class YahooCollector(DataCollector):
    """Collector for Yahoo Finance data."""

    def __init__(
        self,
        base_dir: str = "datasets",
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
    ):
        """
        Initialize the Yahoo Finance collector.

        Args:
            base_dir: Base directory for storing downloaded datasets.
            start_date: Start date for data download (YYYY-MM-DD).
            end_date: End date for data download. If None, uses today.
        """
        super().__init__(base_dir)
        self.start_date = start_date
        self.end_date = end_date

    @property
    def source_name(self) -> str:
        return "yahoo"

    def download(
        self,
        dataset_id: str,
        asset_class: AssetClass,
        force: bool = False,
        description: str = "",
    ) -> Path:
        """
        Download a Yahoo Finance ticker.

        Args:
            dataset_id: Yahoo Finance ticker symbol (e.g., "SPY", "BTC-USD").
            asset_class: The asset class this ticker belongs to.
            force: If True, re-download even if exists.
            description: Optional description for metadata.

        Returns:
            Path to the downloaded CSV file.
        """
        target_dir = self.get_asset_dir(asset_class)
        target_file = target_dir / f"{dataset_id}.csv"

        if target_file.exists() and not force:
            print(f"Ticker already exists: {target_file}")
            return target_file

        print(f"Downloading {dataset_id}...")

        # Download with retry
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
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

        # Validate data
        if df is None or df.empty:
            raise ValueError(f"No data returned for ticker {dataset_id}")

        # Reset index to make date a column
        df = df.reset_index()
        df = df.rename(columns={"Date": "date"})

        # Select and rename columns to standard OHLCV format
        columns_map = {
            "date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = df[[c for c in columns_map.keys() if c in df.columns]]
        df = df.rename(columns=columns_map)

        # Convert date to datetime and drop NaN
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.dropna()

        # Save to CSV
        df.to_csv(target_file, index=False)
        print(f"  Saved to: {target_file} ({len(df)} rows)")

        # Update metadata
        start_date = df["date"].min().strftime("%Y-%m-%d") if len(df) > 0 else None
        end_date = df["date"].max().strftime("%Y-%m-%d") if len(df) > 0 else None

        self.update_metadata(
            DatasetMetadata(
                dataset_id=dataset_id,
                asset_class=asset_class.value,
                source=self.source_name,
                description=description,
                file_path=str(target_file),
                download_time=datetime.now().isoformat(),
                rows=len(df),
                columns=len(df.columns),
                start_date=start_date,
                end_date=end_date,
            )
        )

        # Rate limiting
        time.sleep(1)

        return target_file

    def download_all(self, force: bool = False) -> dict[AssetClass, list[Path]]:
        """
        Download all configured Yahoo Finance tickers.

        Args:
            force: If True, re-download even if exists.

        Returns:
            Dictionary mapping asset classes to list of downloaded paths.
        """
        result: dict[AssetClass, list[Path]] = {}

        for ticker in YAHOO_TICKERS:
            try:
                path = self.download(
                    dataset_id=ticker.symbol,
                    asset_class=ticker.asset_class,
                    force=force,
                    description=ticker.description,
                )
                if ticker.asset_class not in result:
                    result[ticker.asset_class] = []
                result[ticker.asset_class].append(path)
            except Exception as e:
                print(f"[ERROR] Failed {ticker.symbol}: {e}")
                import traceback

                traceback.print_exc()
                continue

        return result

    def download_by_asset_class(
        self, asset_class: AssetClass, force: bool = False
    ) -> list[Path]:
        """
        Download all tickers for a specific asset class.

        Args:
            asset_class: The asset class to download tickers for.
            force: If True, re-download even if exists.

        Returns:
            List of paths to downloaded files.
        """
        paths = []
        for ticker in YAHOO_TICKERS:
            if ticker.asset_class == asset_class:
                try:
                    path = self.download(
                        dataset_id=ticker.symbol,
                        asset_class=asset_class,
                        force=force,
                        description=ticker.description,
                    )
                    paths.append(path)
                except Exception:
                    continue
        return paths

    def download_ticker(
        self,
        symbol: str,
        asset_class: AssetClass,
        description: str = "",
        force: bool = False,
    ) -> Path:
        """
        Download a custom ticker not in the default list.

        Args:
            symbol: Yahoo Finance ticker symbol.
            asset_class: The asset class this ticker belongs to.
            description: Optional description.
            force: If True, re-download even if exists.

        Returns:
            Path to the downloaded CSV file.
        """
        return self.download(
            dataset_id=symbol,
            asset_class=asset_class,
            force=force,
            description=description,
        )

    def list_available(self) -> dict[AssetClass, list[YahooTicker]]:
        """
        List all available Yahoo Finance tickers by asset class.

        Returns:
            Dictionary mapping asset classes to list of ticker configs.
        """
        result: dict[AssetClass, list[YahooTicker]] = {}
        for ticker in YAHOO_TICKERS:
            if ticker.asset_class not in result:
                result[ticker.asset_class] = []
            result[ticker.asset_class].append(ticker)
        return result
