"""Kaggle dataset collector for PortBench.

Uses the Kaggle CLI (kaggle datasets download) to download datasets directly
to the target directory with automatic unzipping — equivalent to:

    kaggle datasets download -d <dataset_id> -p <target_dir> --unzip

The CLI approach avoids Python import conflicts (this file is named kaggle.py)
and is faster than kagglehub's cache-and-copy strategy.

Credentials: set KAGGLE_USERNAME and KAGGLE_KEY in .env, or place
~/.kaggle/kaggle.json with {"username": ..., "key": ...}.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from dotenv import load_dotenv

from .base import DataCollector, AssetClass, DataType, DatasetMetadata

load_dotenv()


@dataclass
class KaggleDataset:
    """Kaggle dataset configuration."""

    dataset_id: str
    asset_class: AssetClass
    description: str
    data_type: DataType = DataType.NUMERIC


KAGGLE_DATASETS = [
    # Cryptocurrency - Numeric
    KaggleDataset(
        dataset_id="kaushalnandania/crypto-data-2014-2026",
        asset_class=AssetClass.CRYPTOCURRENCY,
        description="Daily OHLCV for top 50 cryptocurrencies (2014-2026)",
        data_type=DataType.NUMERIC,
    ),
    KaggleDataset(
        dataset_id="emranalbiek/crypto-quants-dataset",
        asset_class=AssetClass.CRYPTOCURRENCY,
        description="95,100+ samples covering ~5,000 cryptocurrencies from CoinMarketCap",
        data_type=DataType.NUMERIC,
    ),
    # Cryptocurrency - Text
    KaggleDataset(
        dataset_id="oliviervha/crypto-news",
        asset_class=AssetClass.CRYPTOCURRENCY,
        description="Crypto news (2021-2023) with title, text, source, subject, sentiment",
        data_type=DataType.TEXT,
    ),
    # Commodities - Numeric
    KaggleDataset(
        dataset_id="ayeshaimran1619/gold-price-dynamics-and-market-behavior",
        asset_class=AssetClass.COMMODITIES,
        description="Daily gold price data (2016-2026) with OHLCV and technical features",
        data_type=DataType.NUMERIC,
    ),
    KaggleDataset(
        dataset_id="sc231997/crude-oil-price",
        asset_class=AssetClass.COMMODITIES,
        description="Crude Oil WTI (USD/Bbl) historical data",
        data_type=DataType.NUMERIC,
    ),
    KaggleDataset(
        dataset_id="anthonygocmen/8-commodities-multi-timeframe-market-data",
        asset_class=AssetClass.COMMODITIES,
        description="7 commodities multi-timeframe: Gold, Silver, Palladium, Platinum, Brent, Natural Gas, WTI (2016-2025)",
        data_type=DataType.NUMERIC,
    ),
    # Equities - Numeric
    KaggleDataset(
        dataset_id="jacksaleeby/nasdaq100-historical-data-2000-2026-upvote",
        asset_class=AssetClass.EQUITIES,
        description="NASDAQ-100 constituents daily data (2000-2026), 514,000+ rows",
        data_type=DataType.NUMERIC,
    ),
    # Equities - Text
    KaggleDataset(
        dataset_id="ekalabyaghosh/stock-data-with-news",
        asset_class=AssetClass.EQUITIES,
        description="99 Stock data with news (1980-2026) including OHLCV and news text",
        data_type=DataType.TEXT,
    ),
    KaggleDataset(
        dataset_id="emrekaany/google-googl-financial-news-from-2000-to-today",
        asset_class=AssetClass.EQUITIES,
        description="Google stock financial news (2000-2026)",
        data_type=DataType.TEXT,
    ),
    # Real Estate - Numeric
    KaggleDataset(
        dataset_id="vincentvaseghi/us-cities-housing-market-data",
        asset_class=AssetClass.REAL_ESTATE,
        description="US housing market data from Redfin (2012-present, monthly)",
        data_type=DataType.NUMERIC,
    ),
]


def _run_kaggle_cli(dataset_id: str, target_dir: Path) -> None:
    """
    Run: kaggle datasets download -d <dataset_id> -p <target_dir> --unzip

    Credentials are forwarded from environment (KAGGLE_USERNAME / KAGGLE_KEY)
    so they override any stale ~/.kaggle/kaggle.json.

    Raises:
        FileNotFoundError: if the kaggle CLI is not on PATH.
        subprocess.CalledProcessError: if the download fails.
    """
    kaggle_cmd = shutil.which("kaggle")
    if kaggle_cmd is None:
        # Fall back to running via current Python interpreter's scripts dir
        kaggle_cmd = str(Path(sys.executable).parent / "kaggle")

    env = os.environ.copy()
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username:
        env["KAGGLE_USERNAME"] = username
    if key:
        env["KAGGLE_KEY"] = key

    cmd = [
        kaggle_cmd,
        "datasets",
        "download",
        "-d",
        dataset_id,
        "-p",
        str(target_dir),
        "--unzip",
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


class KaggleCollector(DataCollector):
    """
    Collector for Kaggle datasets.

    Calls the Kaggle CLI directly (kaggle datasets download --unzip) to
    download and unzip into the target directory in a single step.
    This avoids the kagglehub cache-and-copy overhead and the Python import
    conflict caused by this file being named kaggle.py.

    Credentials: KAGGLE_USERNAME + KAGGLE_KEY in .env, or ~/.kaggle/kaggle.json.
    """

    @property
    def source_name(self) -> str:
        return "kaggle"

    def download(
        self,
        dataset_id: str,
        asset_class: AssetClass,
        force: bool = False,
        description: str = "",
        data_type: DataType = DataType.NUMERIC,
    ) -> Path:
        """
        Download a Kaggle dataset directly to the target directory.

        Equivalent to:
            kaggle datasets download -d <dataset_id> -p <target_dir> --unzip

        Args:
            dataset_id:  Kaggle dataset identifier ("username/dataset-name").
            asset_class: The asset class this dataset belongs to.
            force:       If True, re-download even if the directory exists.
            description: Optional description for metadata.
            data_type:   NUMERIC or TEXT.

        Returns:
            Path to the directory containing the downloaded files.
        """
        dataset_name = dataset_id.split("/")[-1]
        target_dir = self.get_asset_dir(asset_class) / dataset_name

        if not force and self._is_complete(target_dir, dataset_id, min_files=1):
            print(f"Dataset already complete: {target_dir}")
            return target_dir

        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {dataset_id}...")
        _run_kaggle_cli(dataset_id, target_dir)
        print(f"  Saved to: {target_dir}")

        # Collect metadata stats
        total_rows, total_cols, document_count, total_text_length = 0, 0, 0, 0
        file_format = "csv"

        for item in sorted(target_dir.rglob("*")):
            if not item.is_file():
                continue
            suffix = item.suffix.lower()
            if suffix == ".json":
                file_format = "json"
            elif suffix == ".txt":
                file_format = "txt"

            if data_type == DataType.NUMERIC and suffix == ".csv":
                try:
                    import pandas as pd

                    df = pd.read_csv(item, nrows=0)
                    total_cols = max(total_cols, len(df.columns))
                    with open(item, "r", encoding="utf-8", errors="ignore") as f:
                        total_rows += sum(1 for _ in f) - 1
                except Exception:
                    pass
            elif data_type == DataType.TEXT:
                try:
                    if suffix == ".csv":
                        import pandas as pd

                        df = pd.read_csv(item)
                        document_count += len(df)
                        text_cols = [
                            c
                            for c in df.columns
                            if "text" in c.lower() or "content" in c.lower()
                        ]
                        if text_cols:
                            total_text_length += int(df[text_cols[0]].str.len().sum())
                    elif suffix == ".json":
                        with open(item, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                document_count += len(data)
                except Exception:
                    pass

        avg_length = (
            int(total_text_length / document_count)
            if document_count > 0 and total_text_length > 0
            else None
        )

        self.update_metadata(
            DatasetMetadata(
                dataset_id=dataset_id,
                asset_class=asset_class.value,
                source=self.source_name,
                description=description,
                file_path=str(target_dir),
                download_time=datetime.now().isoformat(),
                data_type=data_type.value,
                file_format=file_format,
                rows=total_rows if total_rows > 0 else None,
                columns=total_cols if total_cols > 0 else None,
                document_count=document_count if document_count > 0 else None,
                avg_length=avg_length,
            )
        )

        return target_dir

    def download_all(self, force: bool = False) -> dict[AssetClass, list[Path]]:
        """Download all configured Kaggle datasets."""
        result: dict[AssetClass, list[Path]] = {}
        for dataset in KAGGLE_DATASETS:
            try:
                path = self.download(
                    dataset_id=dataset.dataset_id,
                    asset_class=dataset.asset_class,
                    force=force,
                    description=dataset.description,
                    data_type=dataset.data_type,
                )
                result.setdefault(dataset.asset_class, []).append(path)
            except Exception as e:
                import traceback

                print(f"[ERROR] Failed {dataset.dataset_id}: {e}")
                traceback.print_exc()
        return result

    def download_by_asset_class(
        self, asset_class: AssetClass, force: bool = False
    ) -> list[Path]:
        """Download all datasets for a specific asset class."""
        paths = []
        for dataset in KAGGLE_DATASETS:
            if dataset.asset_class == asset_class:
                try:
                    path = self.download(
                        dataset_id=dataset.dataset_id,
                        asset_class=asset_class,
                        force=force,
                        description=dataset.description,
                        data_type=dataset.data_type,
                    )
                    paths.append(path)
                except Exception:
                    continue
        return paths

    def list_available(self) -> dict[AssetClass, list[KaggleDataset]]:
        """List all available Kaggle datasets by asset class."""
        result: dict[AssetClass, list[KaggleDataset]] = {}
        for dataset in KAGGLE_DATASETS:
            result.setdefault(dataset.asset_class, []).append(dataset)
        return result
