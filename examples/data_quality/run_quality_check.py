"""Run the full data quality assessment for PortBench datasets."""

from portbench.data_quality import QualityConfig, CrossAssetQualityChecker


def main():
    """
    Run the complete data quality assessment pipeline.

    Steps:
      1. Check each processed asset-class CSV (numeric quality).
      2. Check SEC filing HTML files (text quality).
      3. Run cross-asset checks (joint coverage, regime balance).
      4. Print a human-readable summary.
      5. Save the full report to outputs/quality_reports/report.json.
    """
    config = QualityConfig(
        datasets_dir="datasets",
        output_dir="outputs/quality_reports",
    )

    print("=" * 60)
    print("PortBench — Data Quality Assessment")
    print("=" * 60)

    checker = CrossAssetQualityChecker(config=config)
    report = checker.run_full_assessment(
        processed_dir="datasets/processed",
    )

    report.print_summary()

    output_path = "outputs/quality_reports/report.json"
    report.save(output_path)
    print(f"Full report saved to: {output_path}")


if __name__ == "__main__":
    main()
