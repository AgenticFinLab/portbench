"""Real estate data preprocessor."""

from datetime import datetime

import pandas as pd

from .base import AssetClass, AssetPreprocessor


class RealEstatePreprocessor(AssetPreprocessor):
    """Preprocessor for real estate data (REITs, housing indices)."""

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.REAL_ESTATE

    def process_numeric(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Process numeric real estate data from Yahoo and Kaggle."""
        all_data = []

        # Yahoo Finance REIT ETFs (VNQ, IYR, etc.)
        yahoo_files = self.find_csv_files("yahoo", "real_estate")
        for csv_file in yahoo_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                symbol = csv_file.stem

                result = pd.DataFrame()
                date_col = [c for c in df.columns if "date" in c.lower()][0]
                result["date"] = pd.to_datetime(df[date_col])

                col_mapping = {
                    "open": ["open", "Open"],
                    "high": ["high", "High"],
                    "low": ["low", "Low"],
                    "close": ["close", "Close"],
                    "volume": ["volume", "Volume"],
                }

                for std_name, variants in col_mapping.items():
                    for var in variants:
                        if var in df.columns:
                            result[f"{symbol}_{std_name}"] = df[var]
                            break

                # Compute returns
                close_col = f"{symbol}_close"
                if close_col in result.columns:
                    result[f"{symbol}_return"] = result[close_col].pct_change()

                result = self.numeric_processor.align_to_dates(
                    result, start_date, end_date
                )
                if not result.empty:
                    all_data.append(result)
            except Exception as e:
                print(f"  [WARN] Failed to process {csv_file}: {e}")

        # Kaggle housing data
        kaggle_files = self.find_csv_files("kaggle", "real_estate")
        for csv_file in kaggle_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                dataset_name = csv_file.stem[:20]

                # Find date column
                date_col = None
                for col in df.columns:
                    if (
                        "date" in col.lower()
                        or "time" in col.lower()
                        or "period" in col.lower()
                    ):
                        date_col = col
                        break

                if date_col is None:
                    continue

                result = pd.DataFrame()
                result["date"] = pd.to_datetime(df[date_col], errors="coerce")
                result = result.dropna(subset=["date"])

                # Add numeric columns
                numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
                for col in numeric_cols[:5]:
                    result[f"{dataset_name}_{col}"] = df[col].values[: len(result)]

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
        """Process text real estate data."""
        return pd.DataFrame(columns=["date", "text_json"])
