"""Thin wrappers around Yahoo/FRED collectors + live processed overlay."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


def _meta_end_date(collector, dataset_id: str) -> Optional[date]:
    entry = collector._get_metadata_entry(dataset_id) or {}
    end = entry.get("end_date")
    if not end:
        return None
    return datetime.strptime(str(end)[:10], "%Y-%m-%d").date()


def _file_end_date(path: Path) -> Optional[date]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["date"])
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if dates.empty:
            return None
        return dates.max().date()
    except Exception:
        return None


def refresh_yahoo_incremental(
    *,
    base_dir: str = "datasets",
    start_date: str = "2015-01-01",
    needed_through: Optional[date] = None,
    sleep_s: float = 0.5,
) -> dict:
    """
    Ensure coverage through ``needed_through``.

    - Prefer Yahoo → writes ``datasets/yahoo/`` only on success.
    - On Yahoo failure → AKShare overlay in ``datasets/akshare_ext/`` only
      (never mutates yahoo raw or ``datasets/processed``).
    """
    from ..data_collect.akshare_fallback import read_ext_end_date
    from ..data_collect.yahoo import YAHOO_TICKERS, YahooCollector

    yahoo = YahooCollector(base_dir=base_dir, start_date=start_date)
    needed = needed_through or date.today()
    downloaded = 0
    skipped = 0
    failed = 0
    via_akshare = 0

    for ticker in YAHOO_TICKERS:
        symbol = ticker.symbol
        target = yahoo.get_asset_dir(ticker.asset_class) / f"{symbol}.csv"
        y_end = _file_end_date(target) or _meta_end_date(yahoo, symbol)
        ak_end = read_ext_end_date(symbol, ticker.asset_class)
        if ak_end is not None:
            ak_end_d = ak_end.date() if hasattr(ak_end, "date") else ak_end
        else:
            ak_end_d = None
        effective = max(filter(None, [y_end, ak_end_d]), default=None)
        # Allow a few calendar days of EOD / weekend lag past needed_through
        if effective is not None and effective >= needed - timedelta(days=3):
            skipped += 1
            continue
        try:
            path = yahoo.download(
                dataset_id=symbol,
                asset_class=ticker.asset_class,
                force=True,
                description=ticker.description,
            )
            downloaded += 1
            if "akshare_ext" in str(path).replace("\\", "/"):
                via_akshare += 1
            time.sleep(sleep_s)
        except Exception as e:
            failed += 1
            msg = str(e)
            print(f"[live.refresh] Failed {symbol}: {msg}")
            # Back off only if the AKShare half of the error is a rate limit
            ak_part = msg.lower().split("akshare=", 1)[-1] if "akshare=" in msg.lower() else ""
            if "rate limited" in ak_part or "too many requests" in ak_part or "429" in ak_part:
                print("[live.refresh] AKShare rate-limited; sleeping 30s...")
                time.sleep(30)

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "via_akshare_ext": via_akshare,
        "needed_through": needed.isoformat(),
        "mode": "incremental_overlay",
    }


def refresh_fred_incremental(
    *,
    base_dir: str = "datasets",
    start_date: str = "2015-01-01",
    force: bool = False,
) -> dict:
    """FRED refresh; default force=False skips already-complete series."""
    from ..data_collect.fred import FREDCollector

    fred = FREDCollector(base_dir=base_dir, start_date=start_date)
    paths = fred.download_all(force=force)
    return {"fred_classes": len(paths), "force": force}


def refresh_market_data(
    *,
    base_dir: str = "datasets",
    start_date: str = "2015-01-01",
    force: bool = False,
    needed_through: Optional[date] = None,
) -> dict:
    """
    Refresh market data for live eval.

    Never rewrites ``datasets/processed``. Yahoo failures go to akshare_ext.
    ``force=True`` still only force-refreshes collectors; AKShare still
    cannot overwrite yahoo raw.
    """
    if force:
        from ..data_collect.fred import FREDCollector
        from ..data_collect.yahoo import YahooCollector

        print(
            "[live.refresh] force=True: Yahoo attempts full re-download; "
            "failures still land in akshare_ext only."
        )
        yahoo = YahooCollector(base_dir=base_dir, start_date=start_date)
        fred = FREDCollector(base_dir=base_dir, start_date=start_date)
        return {
            "yahoo": {"mode": "full", "classes": len(yahoo.download_all(force=True))},
            "fred": {"mode": "full", "classes": len(fred.download_all(force=True))},
            "force": True,
        }

    yahoo_stats = refresh_yahoo_incremental(
        base_dir=base_dir,
        start_date=start_date,
        needed_through=needed_through,
    )
    fred_stats = refresh_fred_incremental(
        base_dir=base_dir, start_date=start_date, force=False
    )
    return {"yahoo": yahoo_stats, "fred": fred_stats, "force": False}


def run_preprocess(repo_root: Optional[Path] = None) -> str:
    """Run canonical preprocess_all.py via runpy (writes datasets/processed)."""
    import runpy

    root = Path(repo_root) if repo_root else Path.cwd()
    script = root / "examples" / "data_preprocess" / "preprocess_all.py"
    if not script.exists():
        script = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "data_preprocess"
            / "preprocess_all.py"
        )
    if not script.exists():
        raise FileNotFoundError(
            f"Cannot find preprocess_all.py (looked under {root}). "
            "Run from the portbench repo root."
        )
    runpy.run_path(str(script), run_name="__main__")
    return str(root / "datasets" / "processed")


def refresh_and_preprocess(
    *,
    force: bool = False,
    skip_refresh: bool = False,
    skip_preprocess: bool = False,
    repo_root: Optional[Path] = None,
    needed_through: Optional[date] = None,
    restore_yahoo_from_processed: bool = True,
    build_live_overlay: bool = True,
    live_dir: str = "datasets/processed_live",
) -> dict:
    """
    Live data prep that preserves the benchmark processed decade.

    1. Optionally restore ``datasets/yahoo`` OHLCV from ``datasets/processed``
       (repairs accidental AKShare overwrites).
    2. Incremental Yahoo / AKShare-ext / FRED refresh.
    3. Build ``datasets/processed_live`` = processed + akshare_ext
       (does **not** rewrite ``datasets/processed``).

    ``skip_preprocess`` is kept for CLI compat; full preprocess of the
    decade is skipped by default in favor of the live overlay.
    """
    summary: dict = {
        "refreshed": False,
        "preprocessed": False,
        "processed_dir_untouched": True,
    }

    if restore_yahoo_from_processed:
        from .extend_processed import restore_yahoo_raw_from_processed

        print(
            "[live.refresh] Restoring datasets/yahoo from datasets/processed "
            "(through 2025-12-31)..."
        )
        summary["yahoo_restore"] = restore_yahoo_raw_from_processed()

    if not skip_refresh:
        summary["download"] = refresh_market_data(
            force=force, needed_through=needed_through
        )
        summary["refreshed"] = True

    # Do not run full preprocess_all into datasets/processed by default.
    if not skip_preprocess and not build_live_overlay:
        summary["processed_dir"] = run_preprocess(repo_root=repo_root)
        summary["preprocessed"] = True
        summary["processed_dir_untouched"] = False

    if build_live_overlay:
        from .extend_processed import build_processed_live

        print(
            f"[live.refresh] Building {live_dir} from processed + akshare_ext "
        )
        summary["processed_live"] = build_processed_live(
            live_dir=live_dir,
            needed_through=needed_through,
            overwrite=True,
        )
        summary["preprocessed"] = True

    return summary
