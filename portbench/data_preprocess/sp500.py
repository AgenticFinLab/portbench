"""S&P 500 Top-50 preprocessor.

Reads raw SP500 OHLCV CSVs from datasets/sp500/equities/ and produces
a standalone processed directory compatible with ProcessedDataProvider.

Output: <output_dir>/equities.csv
Columns: date, <Ticker>_close, <Ticker>_return, <Ticker>_open, ...

This preprocessor produces a **self-contained** dataset directory
(e.g., datasets/processed_sp500/) that is used independently from
the default PortBench data. Results are never mixed.
"""

from datetime import datetime

import pandas as pd

from .base import AssetClass, AssetPreprocessor


class SP500Preprocessor(AssetPreprocessor):
    """Preprocessor for S&P 500 Top-50 stock data.

    Outputs a standalone equities.csv that can be loaded by
    ProcessedDataProvider as a self-contained dataset.
    """

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.EQUITIES

    def process_numeric(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Process SP500 stock data into standard price/return series."""
        sp_dir = self.input_dir / "sp500" / "equities"
        if not sp_dir.exists():
            print(f"  [WARN] SP500 data directory not found: {sp_dir}")
            return pd.DataFrame()

        csv_files = sorted(sp_dir.glob("*.csv"))
        if not csv_files:
            print(f"  [WARN] No CSV files found in {sp_dir}")
            return pd.DataFrame()

        all_data = []

        for csv_file in csv_files:
            ticker = csv_file.stem
            try:
                df = pd.read_csv(csv_file)
                processed = self._process_ohlcv(df, ticker)
                if processed.empty:
                    continue
                processed = self.numeric_processor.align_to_dates(
                    processed, start_date, end_date
                )
                all_data.append(processed)
            except Exception as e:
                print(f"  [WARN] Failed to process {csv_file.name}: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        merged = all_data[0]
        for df in all_data[1:]:
            merged["date"] = pd.to_datetime(merged["date"])
            df["date"] = pd.to_datetime(df["date"])
            merged = pd.merge(merged, df, on="date", how="outer")

        merged = merged.sort_values("date").reset_index(drop=True)
        merged = self.numeric_processor.fill_missing(merged)
        merged = self.numeric_processor.winsorize(merged)

        return merged

    def _process_ohlcv(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process OHLCV data for a single SP500 stock (no source prefix)."""
        date_col = None
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                date_col = col
                break
        if date_col is None:
            return pd.DataFrame()

        result = pd.DataFrame()
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

        close_col = f"{symbol}_close"
        if close_col in result.columns:
            result = self.numeric_processor.compute_log_returns(
                result, price_col=close_col
            )
            result = result.rename(columns={"log_return": f"{symbol}_return"})

        return result

    def process_text(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """SP500 has no text data."""
        return pd.DataFrame(columns=["date", "text_json"])

    def process(
        self, start_date: datetime, end_date: datetime
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process and save SP500 data to equities.csv (standalone dataset)."""
        numeric_df = self.process_numeric(start_date, end_date)
        text_df = self.process_text(start_date, end_date)
        if not numeric_df.empty:
            self.save_output(numeric_df, text_df)
        return numeric_df, text_df
