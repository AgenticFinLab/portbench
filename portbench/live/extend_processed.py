"""Build ``datasets/processed_live`` from frozen processed + akshare_ext.

Leaves ``datasets/processed`` untouched (benchmark decade). Live eval should
point ``data_dir`` at ``datasets/processed_live``.
"""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..data_collect.akshare_fallback import AKSHARE_EXT_ROOT, ext_path
from ..data_collect.base import AssetClass
from ..data_collect.yahoo import YAHOO_TICKERS

_OHLCV = ("open", "high", "low", "close", "volume")

_CLASS_FILES = {
    "equities": "equities.csv",
    "bonds": "bonds.csv",
    "commodities": "commodities.csv",
    "real_estate": "real_estate.csv",
    "cryptocurrency": "cryptocurrency.csv",
    "cash": "cash.csv",
}


def restore_yahoo_raw_from_processed(
    *,
    processed_dir: str | Path = "datasets/processed",
    yahoo_dir: str | Path = "datasets/yahoo",
    through: str = "2025-12-31",
) -> dict:
    """
    Rewrite ``datasets/yahoo/{class}/{symbol}.csv`` from processed OHLCV
    columns through ``through`` (inclusive). Use after accidental AKShare
    overwrites of yahoo raw files.
    """
    processed = Path(processed_dir)
    yahoo_root = Path(yahoo_dir)
    through_ts = pd.Timestamp(through)
    restored = 0
    skipped = 0

    for ticker in YAHOO_TICKERS:
        ac = ticker.asset_class.value
        csv_name = _CLASS_FILES.get(ac)
        if not csv_name:
            skipped += 1
            continue
        src = processed / csv_name
        if not src.exists():
            skipped += 1
            continue
        # Load once per class — cache below
        pass

    # Cache class frames
    frames: dict[str, pd.DataFrame] = {}
    for ac, fname in _CLASS_FILES.items():
        path = processed / fname
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            frames[ac] = df[df["date"] <= through_ts].copy()

    for ticker in YAHOO_TICKERS:
        ac = ticker.asset_class.value
        df = frames.get(ac)
        if df is None or df.empty:
            skipped += 1
            continue
        sym = ticker.symbol
        # processed uses ^VIX_close etc.; also try without caret
        close_col = f"{sym}_close"
        if close_col not in df.columns and sym.startswith("^"):
            close_col = f"{sym.lstrip('^')}_close"
        if close_col not in df.columns:
            skipped += 1
            continue
        stem = close_col[: -len("_close")]
        cols = {"date": df["date"]}
        ok = True
        for f in _OHLCV:
            c = f"{stem}_{f}"
            if c not in df.columns:
                if f == "volume":
                    cols[f] = 0.0
                else:
                    ok = False
                    break
            else:
                cols[f] = df[c]
        if not ok:
            skipped += 1
            continue
        out = pd.DataFrame(cols).dropna(subset=["close"])
        out = out.sort_values("date")
        out_dir = yahoo_root / ac
        out_dir.mkdir(parents=True, exist_ok=True)
        # Keep original symbol filename including ^VIX
        out_path = out_dir / f"{sym}.csv"
        out.to_csv(out_path, index=False)
        restored += 1

    return {
        "restored": restored,
        "skipped": skipped,
        "through": through,
        "yahoo_dir": str(yahoo_root),
    }


def _load_ext_ohlcv(symbol: str, asset_class: AssetClass) -> Optional[pd.DataFrame]:
    path = ext_path(symbol, asset_class)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    return df.dropna(subset=["date", "close"]).sort_values("date")


def _append_ticker_rows(
    base: pd.DataFrame,
    symbol: str,
    ext: pd.DataFrame,
    *,
    after: pd.Timestamp,
) -> pd.DataFrame:
    """Append extension OHLCV+return into a processed asset-class frame."""
    close_col = f"{symbol}_close"
    if close_col not in base.columns and symbol.startswith("^"):
        symbol = symbol.lstrip("^")
        close_col = f"{symbol}_close"
    if close_col not in base.columns:
        return base

    new = ext[ext["date"] > after].copy()
    if new.empty:
        return base

    out = base.copy()
    out["date"] = pd.to_datetime(out["date"])
    existing_dates = set(out["date"].dt.normalize())

    rows = []
    last_close = None
    if close_col in out.columns:
        hist = out.dropna(subset=[close_col])
        if len(hist):
            last_close = float(hist.iloc[-1][close_col])

    for _, r in new.iterrows():
        d = pd.Timestamp(r["date"]).normalize()
        if d in existing_dates:
            continue
        row = {c: np.nan for c in out.columns}
        row["date"] = d
        for f in _OHLCV:
            col = f"{symbol}_{f}"
            if col in out.columns and f in r.index:
                row[col] = r[f]
        ret_col = f"{symbol}_return"
        if ret_col in out.columns:
            c = float(r["close"])
            if last_close and last_close > 0 and c > 0:
                row[ret_col] = float(np.log(c / last_close))
            last_close = c
        rows.append(row)
        existing_dates.add(d)

    if not rows:
        return out
    add = pd.DataFrame(rows)
    merged = pd.concat([out, add], ignore_index=True)
    return merged.sort_values("date").reset_index(drop=True)


def build_processed_live(
    *,
    processed_dir: str | Path = "datasets/processed",
    live_dir: str | Path = "datasets/processed_live",
    akshare_root: str | Path = AKSHARE_EXT_ROOT,
    needed_through: Optional[date] = None,
    overwrite: bool = True,
) -> dict:
    """
    Copy ``processed`` → ``processed_live``, then append ``akshare_ext`` rows
    after the original max date. Does not modify ``processed``.
    """
    src = Path(processed_dir)
    dst = Path(live_dir)
    if not src.exists():
        raise FileNotFoundError(f"Missing processed dir: {src}")

    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Copy tree (CSVs + json sidecars)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)

    # Anchor cutoff = original processed max date (usually 2025-12-31)
    pb = dst / "portbench.csv"
    if pb.exists():
        dates = pd.to_datetime(
            pd.read_csv(pb, usecols=["date"])["date"], errors="coerce"
        )
        after = dates.max()
    else:
        after = pd.Timestamp("2025-12-31")

    extended_tickers = 0
    for ticker in YAHOO_TICKERS:
        ext = _load_ext_ohlcv(ticker.symbol, ticker.asset_class)
        if ext is None or ext.empty:
            continue
        ac = ticker.asset_class.value
        fname = _CLASS_FILES.get(ac)
        if not fname:
            continue
        path = dst / fname
        if not path.exists():
            continue
        base = pd.read_csv(path, low_memory=False)
        updated = _append_ticker_rows(
            base, ticker.symbol, ext, after=after
        )
        if len(updated) > len(base):
            updated.to_csv(path, index=False)
            extended_tickers += 1

    # Rebuild portbench.csv as outer-join of class frames (same as preprocess)
    class_frames = []
    for ac, fname in _CLASS_FILES.items():
        path = dst / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # prefix columns with asset class except date
        rename = {c: f"{ac}_{c}" for c in df.columns if c != "date"}
        df = df.rename(columns=rename)
        class_frames.append(df)

    if class_frames:
        merged = class_frames[0]
        for df in class_frames[1:]:
            merged = pd.merge(merged, df, on="date", how="outer")
        merged = merged.sort_values("date").reset_index(drop=True)
        if needed_through is not None:
            merged = merged[merged["date"] <= pd.Timestamp(needed_through)]
        merged.to_csv(dst / "portbench.csv", index=False)
        live_max = merged["date"].max()
    else:
        live_max = after

    meta = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "source_processed": str(src),
        "akshare_ext": str(akshare_root),
        "cutoff_exclusive_after": str(pd.Timestamp(after).date()),
        "live_max_date": str(pd.Timestamp(live_max).date()),
        "extended_tickers": extended_tickers,
        "needed_through": needed_through.isoformat() if needed_through else None,
    }
    (dst / "live_overlay_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta
