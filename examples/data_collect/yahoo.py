"""Data Collect - Yahoo Finance"""

from portbench.data_collect.yahoo import YahooCollector


def main():
    """Download all Yahoo Finance tickers."""
    collector = YahooCollector(
        base_dir="datasets",
        start_date="2015-01-01",
    )

    print("Available Yahoo Finance tickers:")
    for asset_class, tickers in collector.list_available().items():
        print(f"\n{asset_class.value}:")
        for t in tickers:
            print(f"  - {t.symbol}: {t.description}")

    print("\n" + "=" * 60)
    print("Starting download...")
    print("=" * 60 + "\n")

    results = collector.download_all()

    print("\n" + "=" * 60)
    print("Download Summary:")
    print("=" * 60)
    for asset_class, paths in results.items():
        print(f"\n{asset_class.value}:")
        for path in paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
