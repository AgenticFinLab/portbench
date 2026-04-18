"""Yahoo Finance dataset collector for PortBench."""

import time
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from .base import DataCollector, AssetClass, DataType, DatasetMetadata


@dataclass
class YahooTicker:
    """Yahoo Finance ticker configuration."""

    symbol: str
    asset_class: AssetClass
    description: str


# ---------------------------------------------------------------------------
# Yahoo Finance ticker universe
#
# Selection criteria:
#   - High liquidity (AUM > $1B) for price accuracy
#   - Long history (listed before 2015) to cover all three stress periods
#   - Broad representation across asset classes and geographies
#   - Prefer ETFs over individual securities for diversification purity
#
# Coverage across six asset classes:
#   Equities:       US broad market, sector, international, factor ETFs
#   Bonds:          Treasury (short/mid/long), TIPS, corporate, high yield, international
#   Commodities:    Precious metals, energy, agriculture, base metals, broad index
#   Real Estate:    US REIT, residential, commercial, international
#   Cryptocurrency: Bitcoin, Ethereum, major altcoins
#   Cash:           T-bills, money market, short-duration
# ---------------------------------------------------------------------------

YAHOO_TICKERS = [
    # -----------------------------------------------------------------------
    # Equities — US Broad Market
    # -----------------------------------------------------------------------
    YahooTicker("SPY",  AssetClass.EQUITIES, "SPDR S&P 500 ETF Trust"),
    YahooTicker("QQQ",  AssetClass.EQUITIES, "Invesco QQQ Trust (NASDAQ-100)"),
    YahooTicker("IWM",  AssetClass.EQUITIES, "iShares Russell 2000 ETF (US Small Cap)"),
    YahooTicker("VTI",  AssetClass.EQUITIES, "Vanguard Total Stock Market ETF"),
    YahooTicker("IVV",  AssetClass.EQUITIES, "iShares Core S&P 500 ETF"),

    # Equities — International & Emerging Markets
    YahooTicker("EEM",  AssetClass.EQUITIES, "iShares MSCI Emerging Markets ETF"),
    YahooTicker("EFA",  AssetClass.EQUITIES, "iShares MSCI EAFE ETF (Developed ex-US)"),
    YahooTicker("VEA",  AssetClass.EQUITIES, "Vanguard FTSE Developed Markets ETF"),
    YahooTicker("FXI",  AssetClass.EQUITIES, "iShares China Large-Cap ETF"),
    YahooTicker("EWJ",  AssetClass.EQUITIES, "iShares MSCI Japan ETF"),
    YahooTicker("ACWI", AssetClass.EQUITIES, "iShares MSCI ACWI ETF (All-World)"),

    # Equities — US Sector ETFs (GICS sectors for sector rotation analysis)
    YahooTicker("XLK",  AssetClass.EQUITIES, "Technology Select Sector SPDR ETF"),
    YahooTicker("XLF",  AssetClass.EQUITIES, "Financial Select Sector SPDR ETF"),
    YahooTicker("XLE",  AssetClass.EQUITIES, "Energy Select Sector SPDR ETF"),
    YahooTicker("XLV",  AssetClass.EQUITIES, "Health Care Select Sector SPDR ETF"),
    YahooTicker("XLI",  AssetClass.EQUITIES, "Industrial Select Sector SPDR ETF"),
    YahooTicker("XLY",  AssetClass.EQUITIES, "Consumer Discretionary Select Sector SPDR ETF"),
    YahooTicker("XLP",  AssetClass.EQUITIES, "Consumer Staples Select Sector SPDR ETF"),
    YahooTicker("XLU",  AssetClass.EQUITIES, "Utilities Select Sector SPDR ETF"),
    YahooTicker("XLB",  AssetClass.EQUITIES, "Materials Select Sector SPDR ETF"),
    YahooTicker("XLRE", AssetClass.EQUITIES, "Real Estate Select Sector SPDR ETF"),

    # Equities — Factor ETFs
    YahooTicker("MTUM", AssetClass.EQUITIES, "iShares MSCI USA Momentum Factor ETF"),
    YahooTicker("VLUE", AssetClass.EQUITIES, "iShares MSCI USA Value Factor ETF"),
    YahooTicker("QUAL", AssetClass.EQUITIES, "iShares MSCI USA Quality Factor ETF"),
    YahooTicker("USMV", AssetClass.EQUITIES, "iShares MSCI USA Min Vol Factor ETF"),

    # -----------------------------------------------------------------------
    # Bonds — US Treasury
    # -----------------------------------------------------------------------
    YahooTicker("SHV",  AssetClass.BONDS, "iShares Short Treasury Bond ETF (0-1Y)"),
    YahooTicker("SHY",  AssetClass.BONDS, "iShares 1-3 Year Treasury Bond ETF"),
    YahooTicker("IEF",  AssetClass.BONDS, "iShares 7-10 Year Treasury Bond ETF"),
    YahooTicker("TLT",  AssetClass.BONDS, "iShares 20+ Year Treasury Bond ETF"),
    YahooTicker("TLH",  AssetClass.BONDS, "iShares 10-20 Year Treasury Bond ETF"),

    # Bonds — TIPS (Inflation-Protected)
    YahooTicker("TIP",  AssetClass.BONDS, "iShares TIPS Bond ETF (Inflation-Protected)"),
    YahooTicker("SCHP", AssetClass.BONDS, "Schwab U.S. TIPS ETF"),
    YahooTicker("STIP", AssetClass.BONDS, "iShares 0-5 Year TIPS Bond ETF (Short-term)"),

    # Bonds — Corporate
    YahooTicker("LQD",  AssetClass.BONDS, "iShares iBoxx Investment Grade Corporate Bond ETF"),
    YahooTicker("HYG",  AssetClass.BONDS, "iShares iBoxx High Yield Corporate Bond ETF"),
    YahooTicker("JNK",  AssetClass.BONDS, "SPDR Bloomberg High Yield Bond ETF"),
    YahooTicker("VCIT", AssetClass.BONDS, "Vanguard Intermediate-Term Corporate Bond ETF"),

    # Bonds — International
    YahooTicker("BNDX", AssetClass.BONDS, "Vanguard Total International Bond ETF"),
    YahooTicker("EMB",  AssetClass.BONDS, "iShares JP Morgan USD Emerging Markets Bond ETF"),
    YahooTicker("IGOV", AssetClass.BONDS, "iShares International Treasury Bond ETF"),

    # -----------------------------------------------------------------------
    # Commodities — Precious Metals
    # -----------------------------------------------------------------------
    YahooTicker("GLD",  AssetClass.COMMODITIES, "SPDR Gold Shares"),
    YahooTicker("IAU",  AssetClass.COMMODITIES, "iShares Gold Trust"),
    YahooTicker("SLV",  AssetClass.COMMODITIES, "iShares Silver Trust"),
    YahooTicker("PPLT", AssetClass.COMMODITIES, "abrdn Physical Platinum Shares ETF"),
    YahooTicker("PALL", AssetClass.COMMODITIES, "abrdn Physical Palladium Shares ETF"),

    # Commodities — Energy
    YahooTicker("USO",  AssetClass.COMMODITIES, "United States Oil Fund (WTI Crude)"),
    YahooTicker("UNG",  AssetClass.COMMODITIES, "United States Natural Gas Fund"),
    YahooTicker("BNO",  AssetClass.COMMODITIES, "United States Brent Oil Fund"),

    # Commodities — Agriculture
    YahooTicker("DBA",  AssetClass.COMMODITIES, "Invesco DB Agriculture Fund (Wheat/Corn/Soybeans/Sugar)"),
    YahooTicker("CORN", AssetClass.COMMODITIES, "Teucrium Corn Fund"),
    YahooTicker("WEAT", AssetClass.COMMODITIES, "Teucrium Wheat Fund"),
    YahooTicker("SOYB", AssetClass.COMMODITIES, "Teucrium Soybean Fund"),

    # Commodities — Broad Index & Base Metals
    YahooTicker("DBC",  AssetClass.COMMODITIES, "Invesco DB Commodity Index Tracking Fund"),
    YahooTicker("PDBC", AssetClass.COMMODITIES, "Invesco Optimum Yield Diversified Commodity Strategy ETF"),
    YahooTicker("DBB",  AssetClass.COMMODITIES, "Invesco DB Base Metals Fund (Aluminum/Zinc/Copper)"),
    YahooTicker("CPER", AssetClass.COMMODITIES, "United States Copper Index Fund"),

    # -----------------------------------------------------------------------
    # Real Estate — US REITs
    # -----------------------------------------------------------------------
    YahooTicker("VNQ",  AssetClass.REAL_ESTATE, "Vanguard Real Estate ETF (Broad US REIT)"),
    YahooTicker("IYR",  AssetClass.REAL_ESTATE, "iShares U.S. Real Estate ETF"),
    YahooTicker("SCHH", AssetClass.REAL_ESTATE, "Schwab U.S. REIT ETF"),

    # Real Estate — Sub-sector REITs
    YahooTicker("REZ",  AssetClass.REAL_ESTATE, "iShares Residential & Multisector Real Estate ETF"),
    YahooTicker("MORT", AssetClass.REAL_ESTATE, "VanEck Mortgage REIT Income ETF"),
    YahooTicker("INDS", AssetClass.REAL_ESTATE, "Pacer Benchmark Industrial Real Estate ETF"),

    # Real Estate — International
    YahooTicker("VNQI", AssetClass.REAL_ESTATE, "Vanguard Global ex-US Real Estate ETF"),
    YahooTicker("HAUZ", AssetClass.REAL_ESTATE, "Xtrackers International Real Estate ETF"),

    # Homebuilder proxies (leading indicator for residential real estate)
    YahooTicker("XHB",  AssetClass.REAL_ESTATE, "SPDR S&P Homebuilders ETF"),
    YahooTicker("ITB",  AssetClass.REAL_ESTATE, "iShares U.S. Home Construction ETF"),

    # -----------------------------------------------------------------------
    # Cryptocurrency — Major by Market Cap
    # -----------------------------------------------------------------------
    YahooTicker("BTC-USD",  AssetClass.CRYPTOCURRENCY, "Bitcoin USD"),
    YahooTicker("ETH-USD",  AssetClass.CRYPTOCURRENCY, "Ethereum USD"),
    YahooTicker("BNB-USD",  AssetClass.CRYPTOCURRENCY, "Binance Coin USD"),
    YahooTicker("SOL-USD",  AssetClass.CRYPTOCURRENCY, "Solana USD"),
    YahooTicker("XRP-USD",  AssetClass.CRYPTOCURRENCY, "Ripple USD"),
    YahooTicker("ADA-USD",  AssetClass.CRYPTOCURRENCY, "Cardano USD"),
    YahooTicker("AVAX-USD", AssetClass.CRYPTOCURRENCY, "Avalanche USD"),
    YahooTicker("DOT-USD",  AssetClass.CRYPTOCURRENCY, "Polkadot USD"),
    YahooTicker("LINK-USD", AssetClass.CRYPTOCURRENCY, "Chainlink USD"),
    YahooTicker("MATIC-USD",AssetClass.CRYPTOCURRENCY, "Polygon USD"),

    # Cryptocurrency — DeFi / Stablecoin proxies
    YahooTicker("UNI7083-USD", AssetClass.CRYPTOCURRENCY, "Uniswap USD"),
    YahooTicker("AAVE-USD",    AssetClass.CRYPTOCURRENCY, "Aave USD"),

    # -----------------------------------------------------------------------
    # Cash & Money Market Equivalents
    # -----------------------------------------------------------------------
    YahooTicker("BIL",  AssetClass.CASH, "SPDR Bloomberg 1-3 Month T-Bill ETF"),
    YahooTicker("SGOV", AssetClass.CASH, "iShares 0-3 Month Treasury Bond ETF"),
    YahooTicker("CSHI", AssetClass.CASH, "NEOS Enhanced Income Cash Alternative ETF"),
    YahooTicker("ICSH", AssetClass.CASH, "iShares Ultra Short-Term Bond ETF"),

    # Volatility index (not an investable asset but critical regime indicator)
    YahooTicker("^VIX", AssetClass.EQUITIES, "CBOE Volatility Index (VIX)"),
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

        if not force and self._is_complete(target_file, dataset_id):
            print(f"Ticker already complete: {target_file}")
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
