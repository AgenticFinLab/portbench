"""Data Collect - FRED"""

from portbench.data_collect.fred import FREDCollector


def main():
    """Download all FRED series."""
    # Requires FRED_API_KEY environment variable
    collector = FREDCollector(
        base_dir="datasets",
        start_date="2015-01-01",
    )

    print("Available FRED series:")
    for asset_class, series_list in collector.list_available().items():
        print(f"\n{asset_class.value}:")
        for s in series_list:
            print(f"  - {s.series_id}: {s.description} ({s.frequency})")

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
