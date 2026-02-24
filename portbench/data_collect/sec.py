"""SEC EDGAR dataset collector for PortBench."""

import os
import re
import time
import json
import requests
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

from .base import DataCollector, AssetClass, DatasetMetadata


class SECFilingType(Enum):
    """SEC filing types."""

    FORM_10K = "10-K"  # Annual report
    FORM_10Q = "10-Q"  # Quarterly report
    FORM_8K = "8-K"  # Current report (material events)
    DEF_14A = "DEF 14A"  # Proxy statement


@dataclass
class SECCompany:
    """SEC company configuration."""

    ticker: str
    cik: str  # Central Index Key
    name: str
    asset_class: AssetClass = AssetClass.EQUITIES


# Major companies for portfolio management benchmark
SEC_COMPANIES = [
    # Large-cap Tech
    SECCompany(ticker="AAPL", cik="0000320193", name="Apple Inc."),
    SECCompany(ticker="MSFT", cik="0000789019", name="Microsoft Corporation"),
    SECCompany(ticker="GOOGL", cik="0001652044", name="Alphabet Inc."),
    SECCompany(ticker="AMZN", cik="0001018724", name="Amazon.com Inc."),
    SECCompany(ticker="NVDA", cik="0001045810", name="NVIDIA Corporation"),
    SECCompany(ticker="META", cik="0001326801", name="Meta Platforms Inc."),
    # Financial
    SECCompany(ticker="JPM", cik="0000019617", name="JPMorgan Chase & Co."),
    SECCompany(ticker="BAC", cik="0000070858", name="Bank of America Corporation"),
    SECCompany(ticker="GS", cik="0000886982", name="Goldman Sachs Group Inc."),
    SECCompany(ticker="BLK", cik="0001364742", name="BlackRock Inc."),
    # Healthcare
    SECCompany(ticker="JNJ", cik="0000200406", name="Johnson & Johnson"),
    SECCompany(ticker="UNH", cik="0000731766", name="UnitedHealth Group Inc."),
    SECCompany(ticker="PFE", cik="0000078003", name="Pfizer Inc."),
    # Consumer
    SECCompany(ticker="PG", cik="0000080424", name="Procter & Gamble Company"),
    SECCompany(ticker="KO", cik="0000021344", name="Coca-Cola Company"),
    SECCompany(ticker="WMT", cik="0000104169", name="Walmart Inc."),
    # Energy
    SECCompany(ticker="XOM", cik="0000034088", name="Exxon Mobil Corporation"),
    SECCompany(ticker="CVX", cik="0000093410", name="Chevron Corporation"),
    # Industrial
    SECCompany(ticker="CAT", cik="0000018230", name="Caterpillar Inc."),
    SECCompany(ticker="BA", cik="0000012927", name="Boeing Company"),
]

# SEC EDGAR API base URLs
SEC_EDGAR_BASE = "https://www.sec.gov"
SEC_EDGAR_DATA = "https://data.sec.gov"
SEC_EDGAR_ARCHIVES = f"{SEC_EDGAR_BASE}/cgi-bin/browse-edgar"


class SECCollector(DataCollector):
    """Collector for SEC EDGAR filings."""

    def __init__(
        self,
        base_dir: str = "datasets",
        user_agent: Optional[str] = None,
    ):
        """
        Initialize the SEC collector.

        Args:
            base_dir: Base directory for storing downloaded datasets.
            user_agent: User agent for SEC EDGAR API requests.
                       Required by SEC. Format: "Company Name email@example.com"
        """
        super().__init__(base_dir)

        # Load environment variables
        load_dotenv()

        # SEC API key (optional, for sec-api.io premium features)
        self.api_key = os.getenv("SEC_API_KEY")

        # User agent is required by SEC EDGAR
        self.user_agent = user_agent or os.getenv(
            "SEC_USER_AGENT", "PortBench Research contact@example.com"
        )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )

    @property
    def source_name(self) -> str:
        return "sec"

    def _get_company_filings(
        self,
        cik: str,
        filing_type: SECFilingType,
        count: int = 10,
    ) -> list[dict]:
        """
        Get list of filings for a company from SEC EDGAR.

        Args:
            cik: Company CIK number.
            filing_type: Type of SEC filing.
            count: Maximum number of filings to retrieve.

        Returns:
            List of filing metadata dictionaries.
        """
        # Normalize CIK (remove leading zeros for API, keep for paths)
        cik_int = int(cik)
        cik_padded = str(cik_int).zfill(10)

        # Get company submissions
        url = f"{SEC_EDGAR_DATA}/submissions/CIK{cik_padded}.json"

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1} failed: {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

        # Extract recent filings
        filings = []
        recent = data.get("filings", {}).get("recent", {})

        if not recent:
            return filings

        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        primary_documents = recent.get("primaryDocument", [])

        # Filter by filing type
        target_form = filing_type.value
        for i, form in enumerate(forms):
            if form == target_form and len(filings) < count:
                filings.append(
                    {
                        "form": form,
                        "accessionNumber": accession_numbers[i],
                        "filingDate": filing_dates[i],
                        "primaryDocument": primary_documents[i],
                        "cik": cik_padded,
                    }
                )

        return filings

    def _download_filing(
        self,
        cik: str,
        accession_number: str,
        primary_document: str,
        target_path: Path,
    ) -> bool:
        """
        Download a single SEC filing document.

        Args:
            cik: Company CIK number.
            accession_number: Filing accession number.
            primary_document: Primary document filename.
            target_path: Path to save the document.

        Returns:
            True if download successful, False otherwise.
        """
        # Format accession number for URL (remove dashes)
        accession_clean = accession_number.replace("-", "")
        cik_int = int(cik)

        # Build URL
        url = f"{SEC_EDGAR_BASE}/Archives/edgar/data/{cik_int}/{accession_clean}/{primary_document}"

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=60)
                response.raise_for_status()

                # Save content
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(response.text)

                return True

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1} failed: {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"  Failed to download: {e}")
                    return False

        return False

    def download(
        self,
        dataset_id: str,
        asset_class: AssetClass,
        force: bool = False,
        description: str = "",
    ) -> Path:
        """
        Download SEC filings for a company.

        Args:
            dataset_id: Format "TICKER:FORM_TYPE" (e.g., "AAPL:10-K").
            asset_class: The asset class (typically EQUITIES).
            force: If True, re-download even if exists.
            description: Optional description for metadata.

        Returns:
            Path to the downloaded filings directory.
        """
        # Parse dataset_id
        parts = dataset_id.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid dataset_id format: {dataset_id}. Expected 'TICKER:FORM_TYPE'"
            )

        ticker, form_type = parts

        # Find company
        company = None
        for c in SEC_COMPANIES:
            if c.ticker == ticker:
                company = c
                break

        if company is None:
            raise ValueError(f"Unknown ticker: {ticker}")

        # Parse filing type
        try:
            filing_type = SECFilingType(form_type)
        except ValueError:
            raise ValueError(
                f"Unknown filing type: {form_type}. "
                f"Supported: {[f.value for f in SECFilingType]}"
            )

        # Target directory
        target_dir = (
            self.get_asset_dir(asset_class) / ticker / form_type.replace(" ", "_")
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        existing_files = list(target_dir.glob("*.htm")) + list(
            target_dir.glob("*.html")
        )
        if existing_files and not force:
            print(f"Filings already exist: {target_dir}")
            return target_dir

        print(f"Downloading {ticker} {form_type} filings...")

        # Get filing list
        filings = self._get_company_filings(company.cik, filing_type, count=5)

        if not filings:
            raise ValueError(f"No {form_type} filings found for {ticker}")

        # Download each filing
        downloaded_count = 0
        total_size = 0

        for filing in filings:
            filing_date = filing["filingDate"]
            primary_doc = filing["primaryDocument"]
            accession = filing["accessionNumber"]

            # Create filename with date
            ext = Path(primary_doc).suffix or ".htm"
            filename = f"{filing_date}_{accession}{ext}"
            target_path = target_dir / filename

            if target_path.exists() and not force:
                print(f"  Skipping {filename} (exists)")
                downloaded_count += 1
                continue

            print(f"  Downloading {filing_date}...")
            if self._download_filing(company.cik, accession, primary_doc, target_path):
                downloaded_count += 1
                total_size += target_path.stat().st_size

            # Rate limiting (SEC requires 10 requests/second max)
            time.sleep(0.15)

        print(f"  Downloaded {downloaded_count} filings to: {target_dir}")

        # Update metadata
        self.update_metadata(
            DatasetMetadata(
                dataset_id=dataset_id,
                asset_class=asset_class.value,
                source=self.source_name,
                description=description or f"{company.name} {form_type} filings",
                file_path=str(target_dir),
                download_time=datetime.now().isoformat(),
                rows=downloaded_count,  # Number of filings
                columns=None,
            )
        )

        return target_dir

    def download_all(self, force: bool = False) -> dict[AssetClass, list[Path]]:
        """
        Download all configured SEC filings.

        Args:
            force: If True, re-download even if exists.

        Returns:
            Dictionary mapping asset classes to list of downloaded paths.
        """
        result: dict[AssetClass, list[Path]] = {}

        for company in SEC_COMPANIES:
            for filing_type in [SECFilingType.FORM_10K, SECFilingType.FORM_10Q]:
                dataset_id = f"{company.ticker}:{filing_type.value}"
                try:
                    path = self.download(
                        dataset_id=dataset_id,
                        asset_class=company.asset_class,
                        force=force,
                        description=f"{company.name} {filing_type.value}",
                    )
                    if company.asset_class not in result:
                        result[company.asset_class] = []
                    result[company.asset_class].append(path)
                except Exception as e:
                    print(f"[ERROR] Failed {dataset_id}: {e}")
                    continue

        return result

    def download_company(
        self,
        ticker: str,
        filing_types: Optional[list[SECFilingType]] = None,
        force: bool = False,
    ) -> list[Path]:
        """
        Download all filing types for a specific company.

        Args:
            ticker: Company ticker symbol.
            filing_types: List of filing types to download. If None, downloads all.
            force: If True, re-download even if exists.

        Returns:
            List of paths to downloaded filings.
        """
        if filing_types is None:
            filing_types = list(SECFilingType)

        paths = []
        for filing_type in filing_types:
            try:
                path = self.download(
                    dataset_id=f"{ticker}:{filing_type.value}",
                    asset_class=AssetClass.EQUITIES,
                    force=force,
                )
                paths.append(path)
            except Exception as e:
                print(f"[ERROR] Failed {ticker}:{filing_type.value}: {e}")

        return paths

    def download_custom(
        self,
        ticker: str,
        cik: str,
        name: str,
        filing_type: SECFilingType,
        force: bool = False,
    ) -> Path:
        """
        Download filings for a custom company not in the default list.

        Args:
            ticker: Company ticker symbol.
            cik: Company CIK number.
            name: Company name.
            filing_type: Type of filing to download.
            force: If True, re-download even if exists.

        Returns:
            Path to the downloaded filings directory.
        """
        # Temporarily add to companies list
        company = SECCompany(ticker=ticker, cik=cik, name=name)
        SEC_COMPANIES.append(company)

        try:
            return self.download(
                dataset_id=f"{ticker}:{filing_type.value}",
                asset_class=AssetClass.EQUITIES,
                force=force,
                description=f"{name} {filing_type.value}",
            )
        finally:
            SEC_COMPANIES.remove(company)

    def search_cik(self, ticker: str) -> Optional[str]:
        """
        Search for a company's CIK by ticker symbol.

        Args:
            ticker: Company ticker symbol.

        Returns:
            CIK number if found, None otherwise.
        """
        url = f"{SEC_EDGAR_DATA}/submissions/CIK{ticker}.json"

        try:
            # First try company tickers mapping
            tickers_url = f"{SEC_EDGAR_BASE}/files/company_tickers.json"
            response = self.session.get(tickers_url, timeout=30)
            response.raise_for_status()
            data = response.json()

            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry.get("cik_str", "")).zfill(10)

        except Exception as e:
            print(f"Failed to search CIK: {e}")

        return None

    def list_available(self) -> dict[str, SECCompany]:
        """
        List all available companies for SEC filings.

        Returns:
            Dictionary mapping ticker to company config.
        """
        return {c.ticker: c for c in SEC_COMPANIES}

    def list_filing_types(self) -> list[str]:
        """
        List all supported SEC filing types.

        Returns:
            List of filing type names.
        """
        return [f.value for f in SECFilingType]
