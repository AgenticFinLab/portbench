"""Bonds data preprocessor."""

from datetime import datetime

import pandas as pd

from .base import AssetClass, AssetPreprocessor


class BondsPreprocessor(AssetPreprocessor):
    """Preprocessor for bonds data (treasury yields, corporate bonds)."""

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.BONDS

    def process_numeric(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Process numeric bonds data from FRED and Yahoo."""
        all_data = []

        # FRED data (treasury yields, spreads)
        fred_files = self.find_csv_files("fred", "bonds")
        for csv_file in fred_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                series_name = csv_file.stem

                # Standardize columns
                if "date" in df.columns and "value" in df.columns:
                    result = pd.DataFrame()
                    result["date"] = pd.to_datetime(df["date"])
                    result[f"fred_{series_name}"] = pd.to_numeric(
                        df["value"], errors="coerce"
                    )
                    result = self.numeric_processor.align_to_dates(
                        result, start_date, end_date
                    )
                    if not result.empty:
                        all_data.append(result)
            except Exception as e:
                print(f"  [WARN] Failed to process {csv_file}: {e}")

        # Yahoo Finance bond ETFs (TLT, SHV, etc.)
        yahoo_files = self.find_csv_files("yahoo", "bonds")
        for csv_file in yahoo_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                symbol = csv_file.stem

                result = pd.DataFrame()
                date_col = [c for c in df.columns if "date" in c.lower()][0]
                result["date"] = pd.to_datetime(df[date_col])

                for col in ["close", "Close"]:
                    if col in df.columns:
                        result[f"{symbol}_close"] = df[col]
                        result[f"{symbol}_return"] = result[
                            f"{symbol}_close"
                        ].pct_change()
                        break

                result = self.numeric_processor.align_to_dates(
                    result, start_date, end_date
                )
                if not result.empty:
                    all_data.append(result)
            except Exception as e:
                print(f"  [WARN] Failed to process {csv_file}: {e}")

        if not all_data:
            return pd.DataFrame()

        # Merge all data on date
        merged = all_data[0]
        for df in all_data[1:]:
            merged["date"] = pd.to_datetime(merged["date"])
            df["date"] = pd.to_datetime(df["date"])
            merged = pd.merge(merged, df, on="date", how="outer")

        merged = merged.sort_values("date").reset_index(drop=True)
        merged = self.numeric_processor.fill_missing(merged)
        merged = self.numeric_processor.winsorize(merged)

        # Deduplicate columns from different sources
        merged = self.numeric_processor.deduplicate_columns(merged)

        return merged

    def process_text(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Process text bonds data (central bank statements, etc.)."""
        # Bonds typically have limited text data
        # Could include FOMC statements, credit rating changes
        return pd.DataFrame(columns=["date", "text_json"])
