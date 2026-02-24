"""Download all datasets from all supported sources."""

from portbench.data_collect import (
    KaggleCollector,
    FREDCollector,
    YahooCollector,
    SECCollector,
)


def main():
    """Download all datasets from all collectors."""
    base_dir = "datasets"

    print("=" * 60)
    print("PortBench - Download All Datasets")
    print("=" * 60)

    # 1. Kaggle datasets (Numeric + Text)
    print("\n[1/4] Kaggle Datasets")
    print("-" * 40)
    try:
        kaggle = KaggleCollector(base_dir=base_dir)
        kaggle.download_all()
    except Exception as e:
        print(f"[ERROR] Kaggle: {e}")

    # 2. FRED series (Numeric)
    print("\n[2/4] FRED Economic Data")
    print("-" * 40)
    try:
        fred = FREDCollector(base_dir=base_dir)
        fred.download_all()
    except Exception as e:
        print(f"[ERROR] FRED: {e}")

    # 3. Yahoo Finance tickers (Numeric)
    print("\n[3/4] Yahoo Finance")
    print("-" * 40)
    try:
        yahoo = YahooCollector(base_dir=base_dir, start_date="2015-01-01")
        yahoo.download_all()
    except Exception as e:
        print(f"[ERROR] Yahoo: {e}")

    # 4. SEC filings (Text)
    print("\n[4/4] SEC EDGAR Filings")
    print("-" * 40)
    try:
        sec = SECCollector(base_dir=base_dir)
        sec.download_all()
    except Exception as e:
        print(f"[ERROR] SEC: {e}")

    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
