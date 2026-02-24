"""Data Collect - SEC EDGAR"""

from portbench.data_collect.sec import SECCollector, SECFilingType


def main():
    """Download SEC filings for major companies."""
    collector = SECCollector(
        base_dir="datasets",
        user_agent="PortBench Research contact@example.com",
    )

    print("Available SEC companies:")
    for ticker, company in collector.list_available().items():
        print(f"  - {ticker}: {company.name} (CIK: {company.cik})")

    print(f"\nSupported filing types: {collector.list_filing_types()}")

    print("\n" + "=" * 60)
    print("Starting download...")
    print("=" * 60 + "\n")

    # Download 10-K and 10-Q for select companies
    tickers = ["AAPL", "MSFT", "JPM", "XOM"]

    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        paths = collector.download_company(
            ticker=ticker,
            filing_types=[SECFilingType.FORM_10K, SECFilingType.FORM_10Q],
        )
        for path in paths:
            print(f"  Downloaded: {path}")

    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
