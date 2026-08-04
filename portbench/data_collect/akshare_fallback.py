"""AKShare OHLCV fallback when Yahoo Finance is unavailable.

Writes the same schema as YahooCollector (date, open, high, low, close, volume).
Primarily uses ``ak.stock_us_daily`` (Eastmoney / Sina backends).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import pandas as pd

# Symbols that must not be rewritten via ambiguous bare tickers
# (e.g. SOL → unrelated US equity with decades of history).
_CRYPTO_BARE_BLOCKLIST = frozenset(
    {
        "BTC",
        "BNB",
        "SOL",
        "ADA",
        "AVAX",
        "DOT",
        "MATIC",
        "UNI7083",
        "UNI",
        "AAVE",
    }
)


@contextmanager
def _without_proxy():
    """Temporarily drop HTTP(S) proxies that break Eastmoney endpoints."""
    keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    )
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _candidate_symbols(symbol: str) -> list[str]:
    """Ordered AKShare symbol attempts for a Yahoo-style ticker."""
    out: list[str] = [symbol]
    if symbol.startswith("^"):
        bare = symbol.lstrip("^")
        if bare and bare not in out:
            out.append(bare)
    if symbol.endswith("-USD"):
        bare = symbol[: -len("-USD")]
        # Only try bare codes that are not known-ambiguous on US equity feeds
        if bare and bare not in _CRYPTO_BARE_BLOCKLIST and bare not in out:
            out.append(bare)
    return out


def fetch_us_ohlcv(
    symbol: str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV via AKShare ``stock_us_daily``.

    Returns a DataFrame with columns date/open/high/low/close/volume.
    Raises on total failure.
    """
    import akshare as ak

    last_err: Optional[Exception] = None
    df: Optional[pd.DataFrame] = None
    used = symbol

    with _without_proxy():
        for cand in _candidate_symbols(symbol):
            try:
                raw = ak.stock_us_daily(symbol=cand, adjust="")
            except Exception as e:
                last_err = e
                continue
            if raw is None or raw.empty:
                last_err = ValueError(f"empty frame for {cand}")
                continue
            df = raw
            used = cand
            break

    if df is None or df.empty:
        raise RuntimeError(
            f"AKShare stock_us_daily failed for {symbol}"
            + (f": {last_err}" if last_err else "")
        )

    # Normalize columns (akshare uses lowercase English for stock_us_daily)
    rename = {
        "date": "date",
        "Date": "date",
        "日期": "date",
        "open": "open",
        "Open": "open",
        "开盘": "open",
        "high": "high",
        "High": "high",
        "最高": "high",
        "low": "low",
        "Low": "low",
        "最低": "low",
        "close": "close",
        "Close": "close",
        "收盘": "close",
        "volume": "volume",
        "Volume": "volume",
        "成交量": "volume",
    }
    df = df.rename(columns={c: rename[c] for c in df.columns if c in rename})
    need = ["date", "open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"AKShare {used}: missing columns {missing}; got {list(df.columns)}"
        )
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        # yfinance end is exclusive-ish; keep inclusive calendar end for AKShare
        df = df[df["date"] <= pd.Timestamp(end_date)]

    if df.empty:
        raise ValueError(
            f"AKShare {used}: no rows in [{start_date}, {end_date}] for {symbol}"
        )

    if used != symbol:
        print(f"  [akshare] {symbol} resolved as {used}")

    return df.reset_index(drop=True)


def merge_ohlcv(
    existing: Optional[pd.DataFrame],
    new: pd.DataFrame,
) -> pd.DataFrame:
    """Union on date; ``new`` wins on overlap (fresher fallback fill)."""
    if existing is None or existing.empty:
        return new.copy()
    left = existing.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.tz_localize(None)
    cols = ["date", "open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in left.columns:
            if c == "volume":
                left[c] = 0.0
            else:
                raise ValueError(f"existing OHLCV missing {c}")
    left = left[cols]
    right = new[cols]
    out = (
        pd.concat([left, right], ignore_index=True)
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out


def is_yahoo_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "rate limited" in msg
        or "too many requests" in msg
        or "yfratelimiterror" in msg
        or type(exc).__name__ == "YFRateLimitError"
    )
