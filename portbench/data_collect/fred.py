"""FRED dataset collector for PortBench."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from .base import DataCollector, AssetClass


@dataclass
class FREDSeries:
    """FRED series configuration."""

    series_id: str
    asset_class: AssetClass
    description: str
    frequency: str = "D"  # D=Daily, M=Monthly, Q=Quarterly


# FRED series from docs/project-overview.md
# Bonds: Treasury yields, interest rates
# Macro: GDP, CPI, unemployment rate, Fed funds rate
FRED_SERIES = [
    # Bonds - Treasury Yields
    FREDSeries(
        series_id="DGS10",
        asset_class=AssetClass.BONDS,
        description="10-Year Treasury Constant Maturity Rate",
        frequency="D",
    ),
    FREDSeries(
        series_id="DGS2",
        asset_class=AssetClass.BONDS,
        description="2-Year Treasury Constant Maturity Rate",
        frequency="D",
    ),
    FREDSeries(
        series_id="DGS30",
        asset_class=AssetClass.BONDS,
        description="30-Year Treasury Constant Maturity Rate",
        frequency="D",
    ),
    FREDSeries(
        series_id="BAMLH0A0HYM2",
        asset_class=AssetClass.BONDS,
        description="ICE BofA US High Yield Index Option-Adjusted Spread",
        frequency="D",
    ),
    FREDSeries(
        series_id="BAMLC0A0CM",
        asset_class=AssetClass.BONDS,
        description="ICE BofA US Corporate Index Option-Adjusted Spread",
        frequency="D",
    ),
    # Macro - Interest Rates
    FREDSeries(
        series_id="FEDFUNDS",
        asset_class=AssetClass.CASH,
        description="Federal Funds Effective Rate",
        frequency="M",
    ),
    FREDSeries(
        series_id="DFF",
        asset_class=AssetClass.CASH,
        description="Federal Funds Effective Rate (Daily)",
        frequency="D",
    ),
    # Macro - Inflation
    FREDSeries(
        series_id="CPIAUCSL",
        asset_class=AssetClass.CASH,
        description="Consumer Price Index for All Urban Consumers",
        frequency="M",
    ),
    FREDSeries(
        series_id="CPILFESL",
        asset_class=AssetClass.CASH,
        description="Core CPI (Excluding Food and Energy)",
        frequency="M",
    ),
    # Macro - Economic Activity
    FREDSeries(
        series_id="GDP",
        asset_class=AssetClass.CASH,
        description="Gross Domestic Product",
        frequency="Q",
    ),
    FREDSeries(
        series_id="UNRATE",
        asset_class=AssetClass.CASH,
        description="Unemployment Rate",
        frequency="M",
    ),
    FREDSeries(
        series_id="PAYEMS",
        asset_class=AssetClass.CASH,
        description="Total Nonfarm Payrolls",
        frequency="M",
    ),
    # Macro - PMI and Sentiment
    FREDSeries(
        series_id="UMCSENT",
        asset_class=AssetClass.CASH,
        description="University of Michigan Consumer Sentiment",
        frequency="M",
    ),
    # Yield Curve
    FREDSeries(
        series_id="T10Y2Y",
        asset_class=AssetClass.BONDS,
        description="10-Year Treasury Minus 2-Year Treasury Spread",
        frequency="D",
    ),
    FREDSeries(
        series_id="T10Y3M",
        asset_class=AssetClass.BONDS,
        description="10-Year Treasury Minus 3-Month Treasury Spread",
        frequency="D",
    ),
]


class FREDCollector(DataCollector):
    """Collector for FRED (Federal Reserve Economic Data) datasets."""

    def __init__(
        self,
        base_dir: str = "datasets",
        api_key: Optional[str] = None,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
    ):
        """
        Initialize the FRED collector.

        Args:
            base_dir: Base directory for storing downloaded datasets.
            api_key: FRED API key. If None, reads from .env or FRED_API_KEY env variable.
            start_date: Start date for data download (YYYY-MM-DD).
            end_date: End date for data download. If None, uses today.
        """
        super().__init__(base_dir)

        # Load .env file if exists
        load_dotenv()

        # Get API key from parameter or environment
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.fred = Fred(api_key=self.api_key)
        self.start_date = start_date
        self.end_date = end_date

    @property
    def source_name(self) -> str:
        return "fred"

    def download(
        self, dataset_id: str, asset_class: AssetClass, force: bool = False
    ) -> Path:
        """
        Download a FRED series.

        Args:
            dataset_id: FRED series ID (e.g., "DGS10").
            asset_class: The asset class this series belongs to.
            force: If True, re-download even if exists.

        Returns:
            Path to the downloaded CSV file.
        """
        target_dir = self.get_asset_dir(asset_class)
        target_file = target_dir / f"{dataset_id}.csv"

        if target_file.exists() and not force:
            print(f"Series already exists: {target_file}")
            return target_file

        print(f"Downloading {dataset_id}...")

        # Download series from FRED
        series = self.fred.get_series(
            dataset_id,
            observation_start=self.start_date,
            observation_end=self.end_date,
        )

        # Validate data
        if series is None or series.empty:
            raise ValueError(f"No data returned for series {dataset_id}")

        # Convert to DataFrame and save
        df = pd.DataFrame({"date": series.index, "value": series.values})
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna()

        # Save to CSV
        df.to_csv(target_file, index=False)
        print(f"  Saved to: {target_file} ({len(df)} rows)")

        return target_file

    def download_all(self, force: bool = False) -> dict[AssetClass, list[Path]]:
        """
        Download all configured FRED series.

        Args:
            force: If True, re-download even if exists.

        Returns:
            Dictionary mapping asset classes to list of downloaded paths.
        """
        result: dict[AssetClass, list[Path]] = {}

        for series in FRED_SERIES:
            try:
                path = self.download(
                    dataset_id=series.series_id,
                    asset_class=series.asset_class,
                    force=force,
                )
                if series.asset_class not in result:
                    result[series.asset_class] = []
                result[series.asset_class].append(path)
            except Exception as e:
                print(f"[ERROR] Failed {series.series_id}: {e}")
                import traceback

                traceback.print_exc()
                continue

        return result

    def download_by_asset_class(
        self, asset_class: AssetClass, force: bool = False
    ) -> list[Path]:
        """
        Download all series for a specific asset class.

        Args:
            asset_class: The asset class to download series for.
            force: If True, re-download even if exists.

        Returns:
            List of paths to downloaded files.
        """
        paths = []
        for series in FRED_SERIES:
            if series.asset_class == asset_class:
                try:
                    path = self.download(
                        dataset_id=series.series_id,
                        asset_class=asset_class,
                        force=force,
                    )
                    paths.append(path)
                except Exception:
                    continue
        return paths

    def list_available(self) -> dict[AssetClass, list[FREDSeries]]:
        """
        List all available FRED series by asset class.

        Returns:
            Dictionary mapping asset classes to list of series configs.
        """
        result: dict[AssetClass, list[FREDSeries]] = {}
        for series in FRED_SERIES:
            if series.asset_class not in result:
                result[series.asset_class] = []
            result[series.asset_class].append(series)
        return result

    def download_series(
        self,
        series_id: str,
        asset_class: AssetClass,
        description: str = "",
        force: bool = False,
    ) -> Path:
        """
        Download a custom FRED series not in the default list.

        Args:
            series_id: FRED series ID.
            asset_class: The asset class this series belongs to.
            description: Optional description.
            force: If True, re-download even if exists.

        Returns:
            Path to the downloaded CSV file.
        """
        return self.download(
            dataset_id=series_id,
            asset_class=asset_class,
            force=force,
        )
