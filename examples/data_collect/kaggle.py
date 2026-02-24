"""Data Collect - Kaggle"""

from portbench.data_collect.kaggle import KaggleCollector


def main():
    """Download all Kaggle datasets."""
    collector = KaggleCollector(base_dir="datasets")

    print("Available Kaggle datasets:")
    for asset_class, datasets in collector.list_available().items():
        print(f"\n{asset_class.value}:")
        for ds in datasets:
            print(f"  - {ds.dataset_id}: {ds.description}")

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
