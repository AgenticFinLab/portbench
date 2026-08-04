"""AKShare OHLCV fallback — writes only to ``datasets/akshare_ext/``.

Never mutates ``datasets/yahoo/`` or ``datasets/processed/``. New rows are
stored as an overlay so the original Yahoo decade stays intact.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import AssetClass

AKSHARE_EXT_ROOT = Path("datasets") / "akshare_ext"

# Bare US-equity lookups that collide with unrelated tickers (do not strip -USD).
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


def is_yahoo_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "rate limited" in msg
        or "too many requests" in msg
        or "yfratelimiterror" in msg
        or type(exc).__name__ == "YFRateLimitError"
    )


def ext_path(
    symbol: str,
    asset_class: AssetClass,
    *,
    root: Path | str = AKSHARE_EXT_ROOT,
) -> Path:
    base = Path(root)
    d = base / asset_class.value
    d.mkdir(parents=True, exist_ok=True)
    # Windows-safe filename for ^VIX etc.
    safe = symbol.replace("^", "_")
    return d / f"{safe}.csv"


def _candidate_symbols(symbol: str) -> list[str]:
    out: list[str] = [symbol]
    if symbol.startswith("^"):
        bare = symbol.lstrip("^")
        if bare and bare not in out:
            out.append(bare)
    if symbol.endswith("-USD"):
        bare = symbol[: -len("-USD")]
        if bare and bare not in _CRYPTO_BARE_BLOCKLIST and bare not in out:
            out.append(bare)
    return out


def fetch_us_ohlcv(
    symbol: str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV via AKShare ``stock_us_daily``."""
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
    """Union on date; ``new`` wins on overlap."""
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
    out = (
        pd.concat([left[cols], new[cols]], ignore_index=True)
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out


def read_ext_end_date(
    symbol: str,
    asset_class: AssetClass,
    *,
    root: Path | str = AKSHARE_EXT_ROOT,
) -> Optional[datetime]:
    path = ext_path(symbol, asset_class, root=root)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["date"])
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if dates.empty:
            return None
        return dates.max().to_pydatetime()
    except Exception:
        return None


def save_extension(
    symbol: str,
    asset_class: AssetClass,
    df: pd.DataFrame,
    *,
    after_date: Optional[pd.Timestamp] = None,
    root: Path | str = AKSHARE_EXT_ROOT,
) -> Path:
    """
    Persist AKShare rows under ``akshare_ext/`` only.

    If ``after_date`` is set, keep only strictly newer rows (gap fill).
    Merges with any existing extension file for that symbol.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    if after_date is not None:
        out = out[out["date"] > pd.Timestamp(after_date)]
    path = ext_path(symbol, asset_class, root=root)
    existing = None
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except Exception:
            existing = None
    if out.empty and existing is not None and not existing.empty:
        return path
    if out.empty:
        raise ValueError(
            f"No AKShare rows to save for {symbol} after {after_date}"
        )
    merged = merge_ohlcv(existing, out)
    merged.to_csv(path, index=False)
    return path
