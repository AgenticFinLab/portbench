"""Base class for data collection."""

from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod


class AssetClass(Enum):
    """Six core asset classes for multi-asset portfolio management."""

    EQUITIES = "equities"
    BONDS = "bonds"
    COMMODITIES = "commodities"
    REAL_ESTATE = "real_estate"
    CRYPTOCURRENCY = "cryptocurrency"
    CASH = "cash"


class DataCollector(ABC):
    """Base class for data collection from various sources."""

    def __init__(self, base_dir: str = "datasets"):
        """
        Initialize the data collector.

        Args:
            base_dir: Base directory for storing downloaded datasets.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of the data source."""
        pass

    def get_asset_dir(self, asset_class: AssetClass) -> Path:
        """
        Get the directory path for a specific asset class.

        Args:
            asset_class: The asset class enum value.

        Returns:
            Path to the asset class directory.
        """
        asset_dir = self.base_dir / self.source_name / asset_class.value
        asset_dir.mkdir(parents=True, exist_ok=True)
        return asset_dir

    @abstractmethod
    def download(
        self, dataset_id: str, asset_class: AssetClass, force: bool = False
    ) -> Path:
        """
        Download a dataset.

        Args:
            dataset_id: Identifier for the dataset.
            asset_class: The asset class this dataset belongs to.
            force: If True, re-download even if exists.

        Returns:
            Path to the downloaded dataset.
        """
        pass

    @abstractmethod
    def download_all(self, force: bool = False) -> dict[AssetClass, list[Path]]:
        """
        Download all configured datasets.

        Args:
            force: If True, re-download even if exists.

        Returns:
            Dictionary mapping asset classes to list of downloaded paths.
        """
        pass

    def list_downloaded(self) -> dict[AssetClass, list[Path]]:
        """
        List all downloaded datasets by asset class.

        Returns:
            Dictionary mapping asset classes to list of dataset paths.
        """
        result = {}
        source_dir = self.base_dir / self.source_name
        if not source_dir.exists():
            return result

        for asset_class in AssetClass:
            asset_dir = source_dir / asset_class.value
            if asset_dir.exists():
                datasets = list(asset_dir.iterdir())
                if datasets:
                    result[asset_class] = datasets

        return result
