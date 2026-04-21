"""Equities data preprocessor."""

import json
from datetime import datetime

import pandas as pd

from .base import AssetClass, AssetPreprocessor, _truncate_records_json, truncate_text_for_source


class EquitiesPreprocessor(AssetPreprocessor):
    """Preprocessor for equities data (stocks, ETFs, indices)."""

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.EQUITIES

    def process_numeric(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Process numeric equities data from all sources."""
        all_data = []

        # Yahoo Finance data (SPY, QQQ, etc.)
        yahoo_files = self.find_csv_files("yahoo", "equities")
        for csv_file in yahoo_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                df = self._process_ohlcv(df, csv_file.stem)
                df = self.numeric_processor.align_to_dates(df, start_date, end_date)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                print(f"  [WARN] Failed to process {csv_file}: {e}")

        # Kaggle equities data
        kaggle_files = self.find_csv_files("kaggle", "equities")
        for csv_file in kaggle_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                df = self._process_ohlcv(df, csv_file.stem)
                df = self.numeric_processor.align_to_dates(df, start_date, end_date)
                if not df.empty:
                    all_data.append(df)
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

    def _process_ohlcv(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process OHLCV data for a single symbol."""
        # Find date column
        date_col = None
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                date_col = col
                break

        if date_col is None:
            return pd.DataFrame()

        # Standardize column names
        result = pd.DataFrame()
        result["date"] = pd.to_datetime(df[date_col])

        col_mapping = {
            "open": ["open", "Open", "OPEN"],
            "high": ["high", "High", "HIGH"],
            "low": ["low", "Low", "LOW"],
            "close": ["close", "Close", "CLOSE", "adj_close", "Adj Close"],
            "volume": ["volume", "Volume", "VOLUME"],
        }

        for std_name, variants in col_mapping.items():
            for var in variants:
                if var in df.columns:
                    result[f"{symbol}_{std_name}"] = df[var]
                    break

        # Compute log returns
        close_col = f"{symbol}_close"
        if close_col in result.columns:
            result = self.numeric_processor.compute_log_returns(
                result, price_col=close_col
            )
            result = result.rename(columns={"log_return": f"{symbol}_return"})

        return result

    def process_text(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Process text equities data (news, filings)."""
        all_texts = []

        # SEC filings
        sec_dir = self.input_dir / "sec" / "equities"
        if sec_dir.exists():
            for ticker_dir in sec_dir.iterdir():
                if ticker_dir.is_dir():
                    for filing_type_dir in ticker_dir.iterdir():
                        if filing_type_dir.is_dir():
                            for html_file in filing_type_dir.glob("*.htm*"):
                                try:
                                    # Extract date from filename
                                    date_str = html_file.stem.split("_")[0]
                                    date = pd.to_datetime(date_str)

                                    if start_date <= date <= end_date:
                                        with open(
                                            html_file,
                                            "r",
                                            encoding="utf-8",
                                            errors="ignore",
                                        ) as f:
                                            text = f.read()
                                        text = self.text_processor.clean_text(text)
                                        all_texts.append(
                                            {
                                                "date": date.date(),
                                                "source": "sec",
                                                "ticker": ticker_dir.name,
                                                "type": filing_type_dir.name,
                                                "text": truncate_text_for_source(
                                                    text, "sec"
                                                ),
                                            }
                                        )
                                except Exception:
                                    pass

        # Kaggle news data
        kaggle_files = self.find_csv_files("kaggle", "equities")
        for csv_file in kaggle_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                # Look for text columns
                text_cols = [
                    c
                    for c in df.columns
                    if "text" in c.lower()
                    or "news" in c.lower()
                    or "content" in c.lower()
                ]
                date_cols = [
                    c for c in df.columns if "date" in c.lower() or "time" in c.lower()
                ]

                if text_cols and date_cols:
                    df["date"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
                    df = df.dropna(subset=["date"])
                    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
                    df = df[mask]

                    for _, row in df.iterrows():
                        text = str(row[text_cols[0]])
                        if text and len(text) > 10:
                            all_texts.append(
                                {
                                    "date": row["date"].date(),
                                    "source": "kaggle",
                                    "text": truncate_text_for_source(
                                        self.text_processor.clean_text(text),
                                        "kaggle_stock",
                                    ),
                                }
                            )
            except Exception:
                pass

        if not all_texts:
            return pd.DataFrame(columns=["date", "text_json"])

        # Aggregate by date
        text_df = pd.DataFrame(all_texts)
        aggregated = (
            text_df.groupby("date").apply(lambda x: x.to_dict("records")).reset_index()
        )
        aggregated.columns = ["date", "text_json"]
        aggregated["text_json"] = aggregated["text_json"].apply(_truncate_records_json)

        return aggregated
