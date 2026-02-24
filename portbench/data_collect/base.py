"""Base class for data collection."""

import json
from enum import Enum
from pathlib import Path
from typing import Optional
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict


class AssetClass(Enum):
    """Six core asset classes for multi-asset portfolio management."""

    EQUITIES = "equities"
    BONDS = "bonds"
    COMMODITIES = "commodities"
    REAL_ESTATE = "real_estate"
    CRYPTOCURRENCY = "cryptocurrency"
    CASH = "cash"


class DataType(Enum):
    """Data type classification for portfolio management."""

    NUMERIC = "numeric"  # Price, returns, macro indicators -> CSV format
    TEXT = "text"  # News, filings, reports -> JSON/TXT/HTML format


@dataclass
class DatasetMetadata:
    """Metadata for a downloaded dataset."""

    dataset_id: str
    asset_class: str
    source: str
    description: str
    file_path: str
    download_time: str
    data_type: str = "numeric"  # "numeric" or "text"
    file_format: str = "csv"  # csv, json, txt, html
    rows: Optional[int] = None
    columns: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    # Text-specific metadata
    document_count: Optional[int] = None
    avg_length: Optional[int] = None


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
        self.metadata_file = self.base_dir / "metadata.json"

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

    def _load_metadata(self) -> dict:
        """Load existing metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"datasets": {}, "summary": {}}

    def _save_metadata(self, metadata: dict) -> None:
        """Save metadata to file."""
        # Update summary
        summary = {}
        for key, info in metadata.get("datasets", {}).items():
            source = info.get("source", "unknown")
            asset_class = info.get("asset_class", "unknown")

            if source not in summary:
                summary[source] = {"total": 0, "by_asset_class": {}}
            summary[source]["total"] += 1

            if asset_class not in summary[source]["by_asset_class"]:
                summary[source]["by_asset_class"][asset_class] = 0
            summary[source]["by_asset_class"][asset_class] += 1

        metadata["summary"] = summary
        metadata["last_updated"] = datetime.now().isoformat()

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def update_metadata(self, dataset_meta: DatasetMetadata) -> None:
        """
        Update metadata file with new dataset information.

        Args:
            dataset_meta: Metadata for the downloaded dataset.
        """
        metadata = self._load_metadata()

        # Use source/dataset_id as unique key
        key = f"{dataset_meta.source}/{dataset_meta.dataset_id}"
        metadata["datasets"][key] = asdict(dataset_meta)

        self._save_metadata(metadata)

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
