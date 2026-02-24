"""Kaggle dataset collector for PortBench."""

import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import kagglehub

from .base import DataCollector, AssetClass, DataType, DatasetMetadata


@dataclass
class KaggleDataset:
    """Kaggle dataset configuration."""

    dataset_id: str
    asset_class: AssetClass
    description: str
    data_type: DataType = DataType.NUMERIC


# Kaggle datasets from docs/potential-data-source.md
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
    # Real Estate - Numeric
    KaggleDataset(
        dataset_id="vincentvaseghi/us-cities-housing-market-data",
        asset_class=AssetClass.REAL_ESTATE,
        description="US housing market data from Redfin (2012-present, monthly)",
        data_type=DataType.NUMERIC,
    ),
]


class KaggleCollector(DataCollector):
    """Collector for Kaggle datasets."""

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
        Download a Kaggle dataset.

        Args:
            dataset_id: Kaggle dataset identifier (e.g., "username/dataset-name").
            asset_class: The asset class this dataset belongs to.
            force: If True, re-download even if exists.
            description: Optional description for metadata.
            data_type: Type of data (NUMERIC or TEXT).

        Returns:
            Path to the downloaded dataset directory.
        """
        # Extract dataset name from id
        dataset_name = dataset_id.split("/")[-1]
        target_dir = self.get_asset_dir(asset_class) / dataset_name

        if target_dir.exists() and not force:
            print(f"Dataset already exists: {target_dir}")
            return target_dir

        print(f"Downloading {dataset_id} ({data_type.value})...")

        # kagglehub downloads to cache, we copy to our target directory
        cache_path = Path(kagglehub.dataset_download(dataset_id))
        print(f"  Cache path: {cache_path}")

        # Remove existing if force
        if target_dir.exists() and force:
            shutil.rmtree(target_dir)

        # Copy from cache to target
        target_dir.mkdir(parents=True, exist_ok=True)

        # Statistics for metadata
        total_rows = 0
        total_cols = 0
        document_count = 0
        total_text_length = 0
        file_format = "csv"

        for item in cache_path.iterdir():
            src = item
            dst = target_dir / item.name
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

                # Detect file format
                suffix = item.suffix.lower()
                if suffix in [".json"]:
                    file_format = "json"
                elif suffix in [".txt"]:
                    file_format = "txt"
                elif suffix in [".html", ".htm"]:
                    file_format = "html"

                # Process based on data type
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
                            # Try to find text column
                            text_cols = [
                                c
                                for c in df.columns
                                if "text" in c.lower() or "content" in c.lower()
                            ]
                            if text_cols:
                                total_text_length += df[text_cols[0]].str.len().sum()
                        elif suffix == ".json":
                            import json

                            with open(item, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    document_count += len(data)
                    except Exception:
                        pass

        print(f"  Copied to: {target_dir}")

        # Build metadata based on data type
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
                    description=dataset.description,
                    data_type=dataset.data_type,
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
