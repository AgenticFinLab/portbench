"""Cryptocurrency data preprocessor."""

import json
from datetime import datetime

import pandas as pd

from .base import AssetClass, AssetPreprocessor, _truncate_records_json, truncate_text_for_source


class CryptocurrencyPreprocessor(AssetPreprocessor):
    """Preprocessor for cryptocurrency data (BTC, ETH, etc.)."""

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.CRYPTOCURRENCY

    def process_numeric(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Process numeric crypto data from Yahoo and Kaggle."""
        all_data = []

        # Yahoo Finance crypto (BTC-USD, ETH-USD)
        yahoo_files = self.find_csv_files("yahoo", "cryptocurrency")
        for csv_file in yahoo_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                symbol = csv_file.stem.replace("-", "_")

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

        # Kaggle crypto data
        kaggle_files = self.find_csv_files("kaggle", "cryptocurrency")
        for csv_file in kaggle_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)

                # Find date column
                date_col = None
                for col in df.columns:
                    if "date" in col.lower() or "time" in col.lower():
                        date_col = col
                        break

                if date_col is None:
                    continue

                # Check if this is price data (has numeric columns)
                numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
                if len(numeric_cols) == 0:
                    continue

                dataset_name = csv_file.stem[:15]

                result = pd.DataFrame()
                result["date"] = pd.to_datetime(df[date_col], errors="coerce")
                result = result.dropna(subset=["date"])

                for col in numeric_cols[:30]:
                    result[f"kaggle_{dataset_name}_{col}"] = df[col].values[
                        : len(result)
                    ]

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
        """Process text crypto data (news, sentiment)."""
        all_texts = []

        # Kaggle crypto news
        kaggle_files = self.find_csv_files("kaggle", "cryptocurrency")
        for csv_file in kaggle_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)

                # Look for text columns
                text_cols = [
                    c
                    for c in df.columns
                    if any(
                        kw in c.lower()
                        for kw in ["text", "title", "content", "news", "headline"]
                    )
                ]
                date_cols = [
                    c for c in df.columns if "date" in c.lower() or "time" in c.lower()
                ]

                if not text_cols or not date_cols:
                    continue

                df["date"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
                df = df.dropna(subset=["date"])
                mask = (df["date"] >= start_date) & (df["date"] <= end_date)
                df = df[mask]

                for _, row in df.iterrows():
                    texts = []
                    for col in text_cols:
                        text = str(row.get(col, ""))
                        if text and len(text) > 5 and text != "nan":
                            texts.append(
                                truncate_text_for_source(
                                    self.text_processor.clean_text(text),
                                    "kaggle_crypto",
                                )
                            )

                    if texts:
                        all_texts.append(
                            {
                                "date": row["date"].date(),
                                "source": "kaggle",
                                "texts": texts,
                            }
                        )
            except Exception as e:
                print(f"  [WARN] Failed to process text from {csv_file}: {e}")

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
