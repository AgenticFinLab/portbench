"""Data preprocessing module for PortBench."""

from .base import (
    AssetClass,
    DataType,
    PreprocessConfig,
    TimeAligner,
    NumericPreprocessor,
    TextPreprocessor,
    AssetPreprocessor,
)
from .equities import EquitiesPreprocessor
from .bonds import BondsPreprocessor
from .commodities import CommoditiesPreprocessor
from .real_estate import RealEstatePreprocessor
from .cryptocurrency import CryptocurrencyPreprocessor
from .cash import CashPreprocessor
from .ff49 import FF49Preprocessor
from .sp500 import SP500Preprocessor

__all__ = [
    # Base classes
    "AssetClass",
    "DataType",
    "PreprocessConfig",
    "TimeAligner",
    "NumericPreprocessor",
    "TextPreprocessor",
    "AssetPreprocessor",
    # Asset preprocessors
    "EquitiesPreprocessor",
    "BondsPreprocessor",
    "CommoditiesPreprocessor",
    "RealEstatePreprocessor",
    "CryptocurrencyPreprocessor",
    "CashPreprocessor",
    # External dataset preprocessors
    "FF49Preprocessor",
    "SP500Preprocessor",
]


def get_all_preprocessors(config: PreprocessConfig) -> list[AssetPreprocessor]:
    """
    Get all asset preprocessors with given config.

    Args:
        config: Preprocessing configuration.

    Returns:
        List of all asset preprocessors.
    """
    return [
        EquitiesPreprocessor(config),
        BondsPreprocessor(config),
        CommoditiesPreprocessor(config),
        RealEstatePreprocessor(config),
        CryptocurrencyPreprocessor(config),
        CashPreprocessor(config),
        FF49Preprocessor(config),
        SP500Preprocessor(config),
    ]


def process_all_assets(config: PreprocessConfig) -> dict[AssetClass, tuple]:
    """
    Process all assets and return results.

    Args:
        config: Preprocessing configuration.

    Returns:
        Dictionary mapping asset class to (numeric_df, text_df) tuple.
    """
    from datetime import datetime

    start = datetime.fromisoformat(config.train_start)
    end = datetime.fromisoformat(config.test_end)

    results = {}
    for preprocessor in get_all_preprocessors(config):
        print(f"\nProcessing {preprocessor.asset_class.value}...")
        try:
            numeric_df, text_df = preprocessor.process(start, end)
            results[preprocessor.asset_class] = (numeric_df, text_df)
            print(f"  Numeric: {len(numeric_df)} rows, Text: {len(text_df)} rows")
        except Exception as e:
            print(f"  [ERROR] {e}")
            results[preprocessor.asset_class] = (None, None)

    return results
