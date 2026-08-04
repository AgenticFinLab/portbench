"""Thin wrappers around existing Yahoo/FRED collectors + preprocess."""

from __future__ import annotations

import runpy
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional


def _meta_end_date(collector, dataset_id: str) -> Optional[date]:
    entry = collector._get_metadata_entry(dataset_id) or {}
    end = entry.get("end_date")
    if not end:
        return None
    return datetime.strptime(str(end)[:10], "%Y-%m-%d").date()


def refresh_yahoo_incremental(
    *,
    base_dir: str = "datasets",
    start_date: str = "2015-01-01",
    needed_through: Optional[date] = None,
    sleep_s: float = 0.5,
) -> dict:
    """
    Update Yahoo tickers that are missing or end before ``needed_through``.

    Unlike ``download_all(force=True)``, this does **not** re-download every
    ticker from scratch — that triggers Yahoo rate limits. ``YahooCollector``
    falls back to AKShare when Yahoo fails. Batch experiments never hit Yahoo;
    they only read ``datasets/processed/``.
    """
    from ..data_collect.yahoo import YAHOO_TICKERS, YahooCollector

    yahoo = YahooCollector(base_dir=base_dir, start_date=start_date)
    needed = needed_through or date.today()
    downloaded = 0
    skipped = 0
    failed = 0

    for ticker in YAHOO_TICKERS:
        symbol = ticker.symbol
        target = yahoo.get_asset_dir(ticker.asset_class) / f"{symbol}.csv"
        end = _meta_end_date(yahoo, symbol)
        complete = yahoo._is_complete(target, symbol)
        stale = (end is None) or (end < needed)
        if complete and not stale:
            skipped += 1
            continue
        try:
            yahoo.download(
                dataset_id=symbol,
                asset_class=ticker.asset_class,
                force=True,  # only for this stale/missing ticker
                description=ticker.description,
            )
            downloaded += 1
            time.sleep(sleep_s)
        except Exception as e:
            failed += 1
            msg = str(e)
            print(f"[live.refresh] Failed {symbol}: {msg}")
            # YahooCollector already falls back to AKShare; only back off if
            # both sources failed and the error still looks like Yahoo rate limit.
            if "Rate limited" in msg or "Too Many Requests" in msg:
                print(
                    "[live.refresh] Still rate-limited after fallback; "
                    "sleeping 30s before continuing..."
                )
                time.sleep(30)

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "needed_through": needed.isoformat(),
        "mode": "incremental",
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

    - force=False (default / auto-refresh): incremental Yahoo (only stale/missing)
      + FRED without re-downloading complete series.
    - force=True (--force-refresh): full re-download (slow, rate-limit prone).
    """
    if force:
        from ..data_collect.fred import FREDCollector
        from ..data_collect.yahoo import YahooCollector

        print(
            "[live.refresh] force=True: full Yahoo+FRED re-download "
            "(may hit Yahoo rate limits)."
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
    """Run canonical preprocess_all.py via runpy."""
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
) -> dict:
    """End-to-end data refresh for live eval (incremental by default)."""
    summary: dict = {"refreshed": False, "preprocessed": False}
    if not skip_refresh:
        summary["download"] = refresh_market_data(
            force=force, needed_through=needed_through
        )
        summary["refreshed"] = True
    if not skip_preprocess:
        summary["processed_dir"] = run_preprocess(repo_root=repo_root)
        summary["preprocessed"] = True
    return summary
