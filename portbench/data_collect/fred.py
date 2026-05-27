"""FRED dataset collector for PortBench."""

import os
import time
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from .base import DataCollector, AssetClass, DataType, DatasetMetadata


@dataclass
class FREDSeries:
    """FRED series configuration."""

    series_id: str
    asset_class: AssetClass
    description: str
    frequency: str = "D"  # D=Daily, M=Monthly, Q=Quarterly


# ---------------------------------------------------------------------------
# FRED series universe
#
# Selection criteria:
#   - Directly relevant to multi-asset portfolio allocation decisions
#   - Available with reasonable history back to 2015
#   - Covers the three stress periods: 2015 China shock, 2020 COVID, 2022 crypto/rate
#
# Organization:
#   BONDS       — yield curve, credit spreads, real rates
#   REAL_ESTATE — housing prices, mortgage rates, construction activity
#   COMMODITIES — commodity price indices, supply indicators
#   CASH        — monetary policy, inflation, activity, labor market
# ---------------------------------------------------------------------------

FRED_SERIES = [
    # -----------------------------------------------------------------------
    # Bonds — Nominal Treasury Yields (yield curve)
    # -----------------------------------------------------------------------
    FREDSeries(
        "DGS1MO", AssetClass.BONDS, "1-Month Treasury Constant Maturity Rate", "D"
    ),
    FREDSeries(
        "DGS3MO", AssetClass.BONDS, "3-Month Treasury Constant Maturity Rate", "D"
    ),
    FREDSeries(
        "DGS6MO", AssetClass.BONDS, "6-Month Treasury Constant Maturity Rate", "D"
    ),
    FREDSeries("DGS1", AssetClass.BONDS, "1-Year Treasury Constant Maturity Rate", "D"),
    FREDSeries("DGS2", AssetClass.BONDS, "2-Year Treasury Constant Maturity Rate", "D"),
    FREDSeries("DGS5", AssetClass.BONDS, "5-Year Treasury Constant Maturity Rate", "D"),
    FREDSeries(
        "DGS10", AssetClass.BONDS, "10-Year Treasury Constant Maturity Rate", "D"
    ),
    FREDSeries(
        "DGS30", AssetClass.BONDS, "30-Year Treasury Constant Maturity Rate", "D"
    ),
    # Bonds — Yield Curve Spreads (recession / inversion signals)
    FREDSeries("T10Y2Y", AssetClass.BONDS, "10-Year Minus 2-Year Treasury Spread", "D"),
    FREDSeries(
        "T10Y3M", AssetClass.BONDS, "10-Year Minus 3-Month Treasury Spread", "D"
    ),
    FREDSeries(
        "T5YIFR",
        AssetClass.BONDS,
        "5-Year, 5-Year Forward Inflation Expectation Rate",
        "D",
    ),
    # Bonds — TIPS / Real Yields
    FREDSeries(
        "DFII5",
        AssetClass.BONDS,
        "5-Year Treasury Inflation-Indexed Security (TIPS) Yield",
        "D",
    ),
    FREDSeries(
        "DFII10",
        AssetClass.BONDS,
        "10-Year Treasury Inflation-Indexed Security (TIPS) Yield",
        "D",
    ),
    FREDSeries(
        "DFII30",
        AssetClass.BONDS,
        "30-Year Treasury Inflation-Indexed Security (TIPS) Yield",
        "D",
    ),
    # Bonds — Breakeven Inflation (nominal minus real yield)
    FREDSeries("T5YIE", AssetClass.BONDS, "5-Year Breakeven Inflation Rate", "D"),
    FREDSeries("T10YIE", AssetClass.BONDS, "10-Year Breakeven Inflation Rate", "D"),
    # Bonds — Credit Spreads
    FREDSeries("BAMLH0A0HYM2", AssetClass.BONDS, "ICE BofA US High Yield OAS", "D"),
    FREDSeries("BAMLC0A0CM", AssetClass.BONDS, "ICE BofA US Corporate Bond OAS", "D"),
    FREDSeries(
        "BAMLC0A4CBBB", AssetClass.BONDS, "ICE BofA BBB US Corporate Bond OAS", "D"
    ),
    FREDSeries(
        "TEDRATE",
        AssetClass.BONDS,
        "TED Spread (LIBOR minus T-bill, funding stress)",
        "D",
    ),
    # Bonds — Mortgage Rates (real estate finance linkage)
    FREDSeries(
        "MORTGAGE30US", AssetClass.BONDS, "30-Year Fixed Rate Mortgage Average", "W"
    ),
    FREDSeries(
        "MORTGAGE15US", AssetClass.BONDS, "15-Year Fixed Rate Mortgage Average", "W"
    ),
    # -----------------------------------------------------------------------
    # Real Estate — Housing Prices
    # -----------------------------------------------------------------------
    FREDSeries(
        "CSUSHPINSA",
        AssetClass.REAL_ESTATE,
        "S&P/Case-Shiller US National Home Price Index (NSA)",
        "M",
    ),
    FREDSeries(
        "CSUSHPISA",
        AssetClass.REAL_ESTATE,
        "S&P/Case-Shiller US National Home Price Index (SA)",
        "M",
    ),
    FREDSeries(
        "SPCS20RSA",
        AssetClass.REAL_ESTATE,
        "S&P/Case-Shiller 20-City Composite Home Price Index (SA)",
        "M",
    ),
    FREDSeries(
        "MSPUS", AssetClass.REAL_ESTATE, "Median Sales Price of Houses Sold", "Q"
    ),
    # Real Estate — Activity Indicators
    FREDSeries(
        "HOUST",
        AssetClass.REAL_ESTATE,
        "Housing Starts: Total New Privately Owned",
        "M",
    ),
    FREDSeries(
        "PERMIT",
        AssetClass.REAL_ESTATE,
        "New Private Housing Units Authorized by Building Permits",
        "M",
    ),
    FREDSeries("HSN1F", AssetClass.REAL_ESTATE, "New One-Family Houses Sold", "M"),
    FREDSeries("EXHOSLUSM495S", AssetClass.REAL_ESTATE, "Existing Home Sales", "M"),
    # Real Estate — Affordability & Inventory
    FREDSeries(
        "MSACSR",
        AssetClass.REAL_ESTATE,
        "Monthly Supply of New Houses (months of supply)",
        "M",
    ),
    # -----------------------------------------------------------------------
    # Commodities — Price Indices
    # -----------------------------------------------------------------------
    FREDSeries(
        "DCOILWTICO",
        AssetClass.COMMODITIES,
        "Crude Oil Prices: West Texas Intermediate (WTI)",
        "D",
    ),
    FREDSeries(
        "DCOILBRENTEU", AssetClass.COMMODITIES, "Crude Oil Prices: Brent Europe", "D"
    ),
    FREDSeries(
        "DHHNGSP", AssetClass.COMMODITIES, "Henry Hub Natural Gas Spot Price", "D"
    ),
    # GOLDPMGBD228NLBM removed (invalid FRED ID); gold covered by Yahoo GLD/IAU
    FREDSeries(
        "PWHEAMTUSDM",
        AssetClass.COMMODITIES,
        "Global Price of Wheat (USD per Metric Ton)",
        "M",
    ),
    FREDSeries(
        "PMAIZMTUSDM",
        AssetClass.COMMODITIES,
        "Global Price of Maize/Corn (USD per Metric Ton)",
        "M",
    ),
    # PSOYBNUSDM removed (invalid FRED ID); soybeans covered by Yahoo SOYB
    FREDSeries(
        "PCOPPUSDM",
        AssetClass.COMMODITIES,
        "Global Price of Copper (USD per Metric Ton)",
        "M",
    ),
    # Commodities — Supply / Inventory
    # GOLDPMGBD228NLBM (invalid ID), PSOYBNUSDM (invalid), DPRODUCERAT (invalid) removed;
    # gold covered by Yahoo GLD/IAU, soybeans by Yahoo SOYB, crude production not available on FRED
    # -----------------------------------------------------------------------
    # Cash & Macro — Monetary Policy
    # -----------------------------------------------------------------------
    FREDSeries("DFF", AssetClass.CASH, "Federal Funds Effective Rate (Daily)", "D"),
    FREDSeries(
        "FEDFUNDS", AssetClass.CASH, "Federal Funds Effective Rate (Monthly)", "M"
    ),
    FREDSeries("SOFR", AssetClass.CASH, "Secured Overnight Financing Rate (SOFR)", "D"),
    FREDSeries("IOER", AssetClass.CASH, "Interest Rate on Excess Reserves", "D"),
    FREDSeries(
        "WALCL", AssetClass.CASH, "Fed Total Assets (Balance Sheet, USD millions)", "W"
    ),
    # Cash & Macro — Inflation
    FREDSeries(
        "CPIAUCSL", AssetClass.CASH, "CPI: All Urban Consumers (All Items)", "M"
    ),
    FREDSeries(
        "CPILFESL", AssetClass.CASH, "CPI: Core (Excluding Food and Energy)", "M"
    ),
    FREDSeries(
        "PCEPILFE",
        AssetClass.CASH,
        "PCE: Core Price Index (Fed's Preferred Inflation Gauge)",
        "M",
    ),
    FREDSeries("PCEPI", AssetClass.CASH, "PCE: All Items Price Index", "M"),
    # Cash & Macro — Economic Growth
    FREDSeries(
        "GDP", AssetClass.CASH, "Gross Domestic Product (Quarterly, USD billions)", "Q"
    ),
    FREDSeries("GDPC1", AssetClass.CASH, "Real GDP (Chained 2017 USD)", "Q"),
    FREDSeries("INDPRO", AssetClass.CASH, "Industrial Production Index", "M"),
    FREDSeries("TCU", AssetClass.CASH, "Capacity Utilization: Total Industry (%)", "M"),
    # Cash & Macro — Labor Market
    FREDSeries("UNRATE", AssetClass.CASH, "Unemployment Rate (%)", "M"),
    FREDSeries("PAYEMS", AssetClass.CASH, "Total Nonfarm Payrolls (thousands)", "M"),
    FREDSeries(
        "ICSA",
        AssetClass.CASH,
        "Initial Jobless Claims (weekly, leading indicator)",
        "W",
    ),
    # Cash & Macro — Sentiment & Leading Indicators
    FREDSeries(
        "UMCSENT",
        AssetClass.CASH,
        "University of Michigan Consumer Sentiment Index",
        "M",
    ),
    FREDSeries(
        "USSLIND",
        AssetClass.CASH,
        "Leading Index for the United States (Conference Board)",
        "M",
    ),
    FREDSeries(
        "BAMLH0A0HYM2EY",
        AssetClass.CASH,
        "ICE BofA US High Yield Effective Yield (risk appetite proxy)",
        "D",
    ),
    # Cash & Macro — Money Supply
    FREDSeries(
        "M2SL",
        AssetClass.CASH,
        "M2 Money Supply (seasonally adjusted, USD billions)",
        "M",
    ),
    FREDSeries("M2V", AssetClass.CASH, "Velocity of M2 Money Stock (GDP/M2)", "Q"),
    # Cash & Macro — Credit Conditions
    FREDSeries(
        "DPSACBW027SBOG",
        AssetClass.CASH,
        "Deposits at Commercial Banks (USD billions)",
        "W",
    ),
    FREDSeries(
        "TOTCI",
        AssetClass.CASH,
        "Total Consumer Credit Outstanding (USD millions)",
        "M",
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
        self,
        dataset_id: str,
        asset_class: AssetClass,
        force: bool = False,
        description: str = "",
    ) -> Path:
        """
        Download a FRED series.

        Args:
            dataset_id: FRED series ID (e.g., "DGS10").
            asset_class: The asset class this series belongs to.
            force: If True, re-download even if exists.
            description: Optional description for metadata.

        Returns:
            Path to the downloaded CSV file.
        """
        target_dir = self.get_asset_dir(asset_class)
        target_file = target_dir / f"{dataset_id}.csv"

        if not force and self._is_complete(target_file, dataset_id):
            print(f"Series already complete: {target_file}")
            return target_file

        print(f"Downloading {dataset_id}...")

        # Download series from FRED with retry
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                series = self.fred.get_series(
                    dataset_id,
                    observation_start=self.start_date,
                    observation_end=self.end_date,
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
        if series is None or series.empty:
            raise ValueError(f"No data returned for series {dataset_id}")

        # Convert to DataFrame and save
        df = pd.DataFrame({"date": series.index, "value": series.values})
        df["date"] = pd.to_datetime(df["date"])
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
                data_type=DataType.NUMERIC.value,
                file_format="csv",
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
                    description=series.description,
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
