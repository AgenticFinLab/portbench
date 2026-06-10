"""Data collection module for PortBench."""

from .base import DataCollector, AssetClass, DataType, DatasetMetadata
from .kaggle import KaggleCollector, KaggleDataset, KAGGLE_DATASETS
from .fred import FREDCollector, FREDSeries, FRED_SERIES
from .yahoo import YahooCollector, YahooTicker, YAHOO_TICKERS
from .sec import SECCollector, SECCompany, SECFilingType, SEC_COMPANIES
from .fama_french import FamaFrenchCollector, FF49_INDUSTRIES
from .sp500 import SP500Collector, SP500Ticker, SP500_TICKERS

__all__ = [
    "DataCollector",
    "AssetClass",
    "DataType",
    "DatasetMetadata",
    "KaggleCollector",
    "KaggleDataset",
    "KAGGLE_DATASETS",
    "FREDCollector",
    "FREDSeries",
    "FRED_SERIES",
    "YahooCollector",
    "YahooTicker",
    "YAHOO_TICKERS",
    "SECCollector",
    "SECCompany",
    "SECFilingType",
    "SEC_COMPANIES",
    "FamaFrenchCollector",
    "FF49_INDUSTRIES",
    "SP500Collector",
    "SP500Ticker",
    "SP500_TICKERS",
]
