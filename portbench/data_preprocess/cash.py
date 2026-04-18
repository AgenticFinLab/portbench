"""Cash/money market data preprocessor."""

from datetime import datetime

import pandas as pd

from .base import AssetClass, AssetPreprocessor


class CashPreprocessor(AssetPreprocessor):
    """Preprocessor for cash/money market data (T-Bills, money market funds)."""

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.CASH

    def process_numeric(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Process numeric cash data from FRED and Yahoo."""
        all_data = []

        # FRED data (Fed funds rate, T-Bill rates)
        fred_files = self.find_csv_files("fred", "cash")
        for csv_file in fred_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                series_name = csv_file.stem

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

        # Yahoo Finance money market ETFs (BIL, etc.)
        yahoo_files = self.find_csv_files("yahoo", "cash")
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
            merged = pd.merge(merged, df, on="date", how="outer")

        merged = merged.sort_values("date").reset_index(drop=True)
        merged = self.numeric_processor.fill_missing(merged)

        return merged

    def process_text(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Process text cash data (central bank statements)."""
        # Cash/money market has minimal text data
        return pd.DataFrame(columns=["date", "text_json"])
