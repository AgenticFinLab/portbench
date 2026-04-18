"""Preprocess all datasets for PortBench."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from portbench.data_preprocess import (
    PreprocessConfig,
    TimeAligner,
    get_all_preprocessors,
)


def main():
    """Run complete data preprocessing pipeline."""
    print("=" * 60)
    print("PortBench - Data Preprocessing")
    print("=" * 60)

    # Configuration
    config = PreprocessConfig(
        input_dir="datasets",
        output_dir="datasets/processed",
        train_start="2015-01-01",
        train_end="2022-12-31",
        val_start="2023-01-01",
        val_end="2023-12-31",
        test_start="2024-01-01",
        test_end="2025-12-31",
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Scan datasets for informational time-range report
    print("\n[Step 1] Scanning datasets...")
    aligner = TimeAligner(config)
    datasets = aligner.scan_datasets()
    print(f"  Found {len(datasets)} datasets")

    # Use config dates as the authoritative processing range.
    # find_common_range() is useful for diagnostics but can produce a
    # narrow / empty intersection when a few series are discontinued or
    # newly published — config dates define the benchmark window.
    common_start = datetime.fromisoformat(config.train_start)
    common_end = datetime.fromisoformat(config.test_end)
    print(f"  Processing range: {common_start.date()} to {common_end.date()}")

    # Log the actual data coverage for reference
    try:
        data_start, data_end = aligner.find_common_range(datasets)
        print(f"  Common data coverage: {data_start.date()} to {data_end.date()}")
    except ValueError as e:
        print(f"  Note: {e}")

    # Step 2: Process each asset class
    print("\n[Step 2] Processing assets...")
    preprocessors = get_all_preprocessors(config)

    all_numeric = {}
    all_text = {}

    for preprocessor in preprocessors:
        asset_name = preprocessor.asset_class.value
        print(f"\n--- {asset_name.upper()} ---")

        try:
            numeric_df, text_df = preprocessor.process(common_start, common_end)

            if not numeric_df.empty:
                all_numeric[asset_name] = numeric_df
                print(
                    f"  Numeric: {len(numeric_df)} rows, {len(numeric_df.columns)} cols"
                )

            if not text_df.empty:
                all_text[asset_name] = text_df
                print(f"  Text: {len(text_df)} rows")

            # Save individual asset files
            preprocessor.save_output(numeric_df, text_df)

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback

            traceback.print_exc()

    # Step 3: Summary
    print("\n[Step 3] Per-asset outputs written to:", output_dir)
    for asset_name, df in all_numeric.items():
        print(f"  {asset_name}: {df.shape[0]} rows × {df.shape[1]} cols")
    for asset_name in all_text:
        print(f"  {asset_name} (text): saved")

    # Step 4: Generate portbench.csv — six-asset common coverage window
    print("\n[Step 4] Generating portbench.csv (common coverage window)...")
    _build_portbench_csv(output_dir)

    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)


def _build_portbench_csv(processed_dir: Path) -> None:
    """
    Build portbench.csv — a single file covering the common date window where
    all six asset classes have data simultaneously.

    Reads the per-asset CSVs already written to processed_dir, finds the
    intersection of their date ranges, and outer-joins on that window.
    Each asset's columns are prefixed with the asset class name.

    Can be called standalone if per-asset files already exist.
    """
    asset_classes = ["equities", "bonds", "commodities", "real_estate", "cryptocurrency", "cash"]
    frames: dict[str, pd.DataFrame] = {}

    for ac in asset_classes:
        p = processed_dir / f"{ac}.csv"
        if not p.exists():
            print(f"  [SKIP] {ac}.csv not found — run preprocessing first")
            continue
        df = pd.read_csv(p, low_memory=False)
        if "date" not in df.columns:
            print(f"  [SKIP] {ac}.csv has no date column")
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        # Normalize to date-only (drop time component) and deduplicate
        df["date"] = df["date"].dt.normalize()
        if df["date"].duplicated().any():
            # Keep first occurrence per date (e.g. crypto has multiple rows/day from multi-source merge)
            df = df.groupby("date", as_index=False).first()
        frames[ac] = df

    if len(frames) < 6:
        print(f"  [WARN] Only {len(frames)}/6 asset files available; portbench.csv may be incomplete")

    if not frames:
        print("  [ERROR] No asset files found — skipping portbench.csv generation")
        return

    # Find common date window: intersection of each asset's [min_date, max_date]
    common_start = max(df["date"].min() for df in frames.values())
    common_end   = min(df["date"].max() for df in frames.values())

    if common_start >= common_end:
        print(f"  [ERROR] No common date window ({common_start.date()} >= {common_end.date()})")
        return

    print(f"  Common window: {common_start.date()} to {common_end.date()}")

    # Merge: start from the date spine of the asset with the densest coverage
    merged = None
    for ac, df in frames.items():
        df = df[(df["date"] >= common_start) & (df["date"] <= common_end)].copy()
        # Prefix all non-date columns with asset class name (skip if already prefixed)
        rename = {c: f"{ac}_{c}" for c in df.columns if c != "date" and not c.startswith(f"{ac}_")}
        df = df.rename(columns=rename)
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    out = processed_dir / "portbench.csv"
    merged.to_csv(out, index=False)
    print(f"  Saved: {out}  ({merged.shape[0]} rows × {merged.shape[1]} cols)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--portbench-only":
        # Regenerate portbench.csv from existing per-asset files without re-running preprocessing
        _build_portbench_csv(Path("datasets/processed"))
    else:
        main()
