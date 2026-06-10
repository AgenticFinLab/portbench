"""Fama-French 49 Industry Portfolios preprocessor.

Reads raw FF49 CSVs from datasets/fama_french/equities/ and produces
a standalone processed directory compatible with ProcessedDataProvider.

Output: <output_dir>/equities.csv
Columns: date, <Industry>_close, <Industry>_return, ...

FF49 data is monthly, so prices are forward-filled to business days
to align with daily-frequency expectations.

This preprocessor produces a **self-contained** dataset directory
(e.g., datasets/processed_ff49/) that is used independently from
the default PortBench data. Results are never mixed.
"""

from datetime import datetime

import pandas as pd

from .base import AssetClass, AssetPreprocessor


class FF49Preprocessor(AssetPreprocessor):
    """Preprocessor for Fama-French 49 industry portfolio data.

    Outputs a standalone equities.csv that can be loaded by
    ProcessedDataProvider as a self-contained dataset.
    """

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.EQUITIES

    def process_numeric(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Process FF49 industry data into daily-frequency price/return series."""
        ff_dir = self.input_dir / "fama_french" / "equities"
        if not ff_dir.exists():
            print(f"  [WARN] FF49 data directory not found: {ff_dir}")
            return pd.DataFrame()

        csv_files = sorted(ff_dir.glob("*.csv"))
        if not csv_files:
            print(f"  [WARN] No CSV files found in {ff_dir}")
            return pd.DataFrame()

        all_series = {}

        for csv_file in csv_files:
            industry = csv_file.stem  # e.g., "Agric", "Banks"
            try:
                df = pd.read_csv(csv_file)

                if "date" not in df.columns or "close" not in df.columns:
                    print(f"  [WARN] {csv_file.name}: missing required columns")
                    continue

                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()

                # Filter date range
                mask = (df.index >= start_date) & (df.index <= end_date)
                df = df[mask]
                if df.empty:
                    continue

                # Resample monthly → daily (business days) via forward-fill
                daily_idx = pd.bdate_range(
                    start=df.index.min(), end=df.index.max()
                )
                close_daily = df["close"].reindex(daily_idx, method="ffill")
                ret_daily = close_daily.pct_change()

                all_series[f"{industry}_close"] = close_daily
                all_series[f"{industry}_return"] = ret_daily

            except Exception as e:
                print(f"  [WARN] Failed to process {csv_file.name}: {e}")
                continue

        if not all_series:
            return pd.DataFrame()

        merged = pd.DataFrame(all_series)
        merged.index.name = "date"
        merged = merged.reset_index()
        merged = merged.fillna(0.0)

        return merged

    def process_text(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """FF49 has no text data."""
        return pd.DataFrame(columns=["date", "text_json"])

    def process(
        self, start_date: datetime, end_date: datetime
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process and save FF49 data to equities.csv (standalone dataset)."""
        numeric_df = self.process_numeric(start_date, end_date)
        text_df = self.process_text(start_date, end_date)
        if not numeric_df.empty:
            self.save_output(numeric_df, text_df)
        return numeric_df, text_df
