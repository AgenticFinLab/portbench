"""Fama-French industry portfolios collector for PortBench.

Fetches the 49 industry portfolios from Kenneth French's data library
via direct HTTP download. Data is monthly value-weighted returns from
July 1926 to present.

Each industry is saved as a CSV with synthetic price series (base=100)
to remain compatible with PortBench's BacktestEngine, which expects
OHLCV-style date-indexed price data.
"""

import io
import pickle
import zipfile
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
import requests

from .base import DataCollector, AssetClass, DataType, DatasetMetadata

# Direct download URL (no pandas-datareader dependency)
_FF49_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/49_Industry_Portfolios_CSV.zip"
)

# Cache file for raw FF49 data (avoid repeated network requests)
_FF49_CACHE = "ff49_raw_cache.pkl"

# Industry names are discovered dynamically from the data file
FF49_INDUSTRIES: list[str] = []


def _parse_ff_csv(raw_text: str) -> pd.DataFrame:
    """Parse the multi-section Fama-French CSV file.

    The file has a header block, then four data sections. We extract
    the first section: Monthly Value-Weighted Returns.

    The file format is comma-separated with the date (YYYYMM) as the
    first column, followed by 49 industry return columns.
    """
    lines = raw_text.split("\n")

    # Find the header row (contains industry names, starts with comma)
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(",") and "Agric" in stripped:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find column header row in FF49 CSV")

    # Collect data lines after the header (YYYYMM, val, val, ...)
    data_lines = []
    for i in range(header_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            # End of section (blank line before next section)
            if data_lines:
                break
            continue
        # Data lines start with a 6-digit date
        if stripped[:6].isdigit():
            data_lines.append(stripped)
        elif data_lines:
            # Hit a non-data line after collecting data — end of section
            break

    if not data_lines:
        raise ValueError("Could not find data rows in FF49 CSV")

    # Parse header to get column names
    header = lines[header_idx].strip().split(",")
    col_names = [c.strip() for c in header[1:]]  # skip first empty column

    # Parse data rows
    rows = []
    for line in data_lines:
        parts = line.split(",")
        date_str = parts[0].strip()
        values = []
        for v in parts[1:]:
            v = v.strip()
            try:
                val = float(v)
                values.append(val)
            except ValueError:
                values.append(float("nan"))
        rows.append([date_str] + values)

    # Build DataFrame
    df = pd.DataFrame(rows, columns=["date"] + col_names)
    df["date"] = df["date"].astype(str)
    df.index = pd.to_datetime(df["date"], format="%Y%m")
    df = df.drop(columns=["date"])

    # Replace FF missing value codes with NaN
    df = df.replace([-99.99, -999], float("nan"))

    return df


class FamaFrenchCollector(DataCollector):
    """Collector for Fama-French 49 industry portfolio data.

    Downloads monthly value-weighted returns from Kenneth French's data
    library and converts them to synthetic price series for compatibility
    with PortBench's BacktestEngine.

    Args:
        base_dir:    Base directory for storing downloaded datasets.
        start_date:  Filter start date (YYYY-MM-DD). Data before this is dropped.
        end_date:    Filter end date (YYYY-MM-DD). If None, uses latest available.
        base_price:  Starting price for synthetic price series (default 100.0).
    """

    def __init__(
        self,
        base_dir: str = "datasets",
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        base_price: float = 100.0,
    ):
        super().__init__(base_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.base_price = base_price

    @property
    def source_name(self) -> str:
        return "fama_french"

    def _fetch_raw_data(self) -> pd.DataFrame:
        """Fetch raw FF49 monthly value-weighted returns with local caching."""
        global FF49_INDUSTRIES

        cache_path = self.base_dir / _FF49_CACHE

        if cache_path.exists():
            print(f"Loading cached FF49 data from {cache_path}")
            with open(cache_path, "rb") as f:
                monthly_vw = pickle.load(f)
            FF49_INDUSTRIES = list(monthly_vw.columns)
            return monthly_vw

        print(f"Downloading Fama-French 49 Industry Portfolios from {_FF49_URL}")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Download ZIP (User-Agent required — Dartmouth blocks default python UA)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(_FF49_URL, headers=headers, timeout=60)
        resp.raise_for_status()

        # Extract CSV from ZIP
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")]
            if not csv_name:
                raise ValueError(f"No CSV found in ZIP. Files: {zf.namelist()}")
            with zf.open(csv_name[0]) as f:
                raw_text = f.read().decode("utf-8")

        # Parse the first section (Monthly Value-Weighted Returns)
        monthly_vw = _parse_ff_csv(raw_text)

        # Convert from percentage to decimal
        monthly_vw = monthly_vw / 100.0

        # Update the global industry list
        FF49_INDUSTRIES = list(monthly_vw.columns)
        print(f"  Found {len(FF49_INDUSTRIES)} industries: {FF49_INDUSTRIES[:5]}...")

        # Cache locally
        with open(cache_path, "wb") as f:
            pickle.dump(monthly_vw, f)

        print(f"  Cached to {cache_path}")
        return monthly_vw

    def _returns_to_prices(self, returns: pd.Series) -> pd.Series:
        """Convert a return series to a synthetic price series."""
        prices = (1 + returns).cumprod() * self.base_price
        return prices

    def download(
        self,
        dataset_id: str,
        asset_class: AssetClass,
        force: bool = False,
    ) -> Path:
        """Download a single FF49 industry portfolio.

        Args:
            dataset_id:   Industry name (e.g., "Agric", "Oil", "Banks").
            asset_class:  Asset class (always EQUITIES for FF49).
            force:        If True, re-download even if cached file exists.

        Returns:
            Path to the downloaded CSV file.
        """
        target_dir = self.get_asset_dir(asset_class)
        target_file = target_dir / f"{dataset_id}.csv"

        if not force and self._is_complete(target_file, dataset_id):
            print(f"Industry already complete: {target_file}")
            return target_file

        # Fetch full dataset
        all_returns = self._fetch_raw_data()

        if dataset_id not in all_returns.columns:
            raise ValueError(
                f"Industry '{dataset_id}' not found. "
                f"Available: {list(all_returns.columns)}"
            )

        returns = all_returns[dataset_id].dropna()

        # Filter date range
        start = pd.Timestamp(self.start_date)
        returns = returns[returns.index >= start]
        if self.end_date:
            end = pd.Timestamp(self.end_date)
            returns = returns[returns.index <= end]

        if returns.empty:
            raise ValueError(
                f"No data for '{dataset_id}' in range "
                f"[{self.start_date}, {self.end_date}]"
            )

        # Convert to synthetic price series
        prices = self._returns_to_prices(returns)

        # Build DataFrame in PortBench-compatible format
        df = pd.DataFrame({
            "date": prices.index,
            "close": prices.values,
            "return": returns.values,
        })

        # Save
        df.to_csv(target_file, index=False)
        print(f"  Saved {dataset_id}: {target_file} ({len(df)} rows)")

        # Update metadata
        self.update_metadata(
            DatasetMetadata(
                dataset_id=dataset_id,
                asset_class=asset_class.value,
                source=self.source_name,
                description=f"FF49 industry: {dataset_id} (monthly VW returns)",
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

        return target_file

    def download_all(self, force: bool = False) -> dict[AssetClass, list[Path]]:
        """Download all 49 FF industry portfolios.

        Args:
            force: If True, re-download even if cached files exist.

        Returns:
            Dict mapping asset classes to list of downloaded paths.
        """
        # Fetch once to populate FF49_INDUSTRIES
        all_returns = self._fetch_raw_data()
        industries = list(all_returns.columns)

        result: dict[AssetClass, list[Path]] = {}

        for industry in industries:
            try:
                path = self.download(
                    dataset_id=industry,
                    asset_class=AssetClass.EQUITIES,
                    force=force,
                )
                if AssetClass.EQUITIES not in result:
                    result[AssetClass.EQUITIES] = []
                result[AssetClass.EQUITIES].append(path)
            except Exception as e:
                print(f"[ERROR] Failed {industry}: {e}")
                continue

        return result

    def download_subset(
        self, industries: list[str], force: bool = False
    ) -> list[Path]:
        """Download a subset of FF49 industries.

        Args:
            industries: List of industry names to download.
            force:      If True, re-download even if cached.

        Returns:
            List of paths to downloaded files.
        """
        paths = []
        for industry in industries:
            try:
                path = self.download(
                    dataset_id=industry,
                    asset_class=AssetClass.EQUITIES,
                    force=force,
                )
                paths.append(path)
            except Exception as e:
                print(f"[ERROR] Failed {industry}: {e}")
                continue
        return paths

    def list_available(self) -> list[str]:
        """Return the list of all FF industry names (populated after first fetch)."""
        if FF49_INDUSTRIES:
            return list(FF49_INDUSTRIES)
        # Trigger fetch to discover names
        self._fetch_raw_data()
        return list(FF49_INDUSTRIES)
