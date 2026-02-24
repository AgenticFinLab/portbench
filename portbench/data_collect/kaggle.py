"""Kaggle dataset collector for PortBench."""

import shutil
from pathlib import Path
from dataclasses import dataclass

import kagglehub

from .base import DataCollector, AssetClass


@dataclass
class KaggleDataset:
    """Kaggle dataset configuration."""

    dataset_id: str
    asset_class: AssetClass
    description: str


# Kaggle datasets from docs/potential-data-source.md
KAGGLE_DATASETS = [
    # Cryptocurrency
    KaggleDataset(
        dataset_id="kaushalnandania/crypto-data-2014-2026",
        asset_class=AssetClass.CRYPTOCURRENCY,
        description="Daily OHLCV for top 50 cryptocurrencies (2014-2026)",
    ),
    KaggleDataset(
        dataset_id="emranalbiek/crypto-quants-dataset",
        asset_class=AssetClass.CRYPTOCURRENCY,
        description="95,100+ samples covering ~5,000 cryptocurrencies from CoinMarketCap",
    ),
    # Commodities
    KaggleDataset(
        dataset_id="ayeshaimran1619/gold-price-dynamics-and-market-behavior",
        asset_class=AssetClass.COMMODITIES,
        description="Daily gold price data (2016-2026) with OHLCV and technical features",
    ),
    KaggleDataset(
        dataset_id="sc231997/crude-oil-price",
        asset_class=AssetClass.COMMODITIES,
        description="Crude Oil WTI (USD/Bbl) historical data",
    ),
    # Equities
    KaggleDataset(
        dataset_id="jacksaleeby/nasdaq100-historical-data-2000-2026-upvote",
        asset_class=AssetClass.EQUITIES,
        description="NASDAQ-100 constituents daily data (2000-2026), 514,000+ rows",
    ),
    # Real Estate
    KaggleDataset(
        dataset_id="vincentvaseghi/us-cities-housing-market-data",
        asset_class=AssetClass.REAL_ESTATE,
        description="US housing market data from Redfin (2012-present, monthly)",
    ),
]


class KaggleCollector(DataCollector):
    """Collector for Kaggle datasets."""

    @property
    def source_name(self) -> str:
        return "kaggle"

    def download(
        self, dataset_id: str, asset_class: AssetClass, force: bool = False
    ) -> Path:
        """
        Download a Kaggle dataset.

        Args:
            dataset_id: Kaggle dataset identifier (e.g., "username/dataset-name").
            asset_class: The asset class this dataset belongs to.
            force: If True, re-download even if exists.

        Returns:
            Path to the downloaded dataset directory.
        """
        # Extract dataset name from id
        dataset_name = dataset_id.split("/")[-1]
        target_dir = self.get_asset_dir(asset_class) / dataset_name

        if target_dir.exists() and not force:
            print(f"Dataset already exists: {target_dir}")
            return target_dir

        print(f"Downloading {dataset_id}...")

        # kagglehub downloads to cache, we copy to our target directory
        cache_path = Path(kagglehub.dataset_download(dataset_id))
        print(f"  Cache path: {cache_path}")

        # Remove existing if force
        if target_dir.exists() and force:
            shutil.rmtree(target_dir)

        # Copy from cache to target
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy files instead of directory to avoid nested structure issues
        for item in cache_path.iterdir():
            src = item
            dst = target_dir / item.name
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        print(f"  Copied to: {target_dir}")
        return target_dir

    def download_all(self, force: bool = False) -> dict[AssetClass, list[Path]]:
        """
        Download all configured Kaggle datasets.

        Args:
            force: If True, re-download even if exists.

        Returns:
            Dictionary mapping asset classes to list of downloaded paths.
        """
        result: dict[AssetClass, list[Path]] = {}

        for dataset in KAGGLE_DATASETS:
            try:
                path = self.download(
                    dataset_id=dataset.dataset_id,
                    asset_class=dataset.asset_class,
                    force=force,
                )
                if dataset.asset_class not in result:
                    result[dataset.asset_class] = []
                result[dataset.asset_class].append(path)
            except Exception as e:
                print(f"[ERROR] Failed {dataset.dataset_id}: {e}")
                import traceback

                traceback.print_exc()
                continue

        return result

    def download_by_asset_class(
        self, asset_class: AssetClass, force: bool = False
    ) -> list[Path]:
        """
        Download all datasets for a specific asset class.

        Args:
            asset_class: The asset class to download datasets for.
            force: If True, re-download even if exists.

        Returns:
            List of paths to downloaded datasets.
        """
        paths = []
        for dataset in KAGGLE_DATASETS:
            if dataset.asset_class == asset_class:
                try:
                    path = self.download(
                        dataset_id=dataset.dataset_id,
                        asset_class=asset_class,
                        force=force,
                    )
                    paths.append(path)
                except Exception:
                    continue
        return paths

    def list_available(self) -> dict[AssetClass, list[KaggleDataset]]:
        """
        List all available Kaggle datasets by asset class.

        Returns:
            Dictionary mapping asset classes to list of dataset configs.
        """
        result: dict[AssetClass, list[KaggleDataset]] = {}
        for dataset in KAGGLE_DATASETS:
            if dataset.asset_class not in result:
                result[dataset.asset_class] = []
            result[dataset.asset_class].append(dataset)
        return result
