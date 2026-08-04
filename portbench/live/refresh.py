"""Thin wrappers around existing Yahoo/FRED collectors + preprocess."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Optional


def refresh_market_data(
    *,
    base_dir: str = "datasets",
    start_date: str = "2015-01-01",
    force: bool = True,
) -> dict:
    """Force-refresh Yahoo + FRED raw data under ``base_dir``."""
    from ..data_collect.fred import FREDCollector
    from ..data_collect.yahoo import YahooCollector

    yahoo = YahooCollector(base_dir=base_dir, start_date=start_date)
    fred = FREDCollector(base_dir=base_dir, start_date=start_date)
    yahoo_paths = yahoo.download_all(force=force)
    fred_paths = fred.download_all(force=force)
    return {
        "yahoo_classes": len(yahoo_paths),
        "fred_classes": len(fred_paths),
        "force": force,
    }


def run_preprocess(repo_root: Optional[Path] = None) -> str:
    """
    Run ``examples/data_preprocess/preprocess_all.py`` via runpy so portbench.csv
    and per-asset outputs stay consistent with the canonical script.
    """
    root = Path(repo_root) if repo_root else Path.cwd()
    script = root / "examples" / "data_preprocess" / "preprocess_all.py"
    if not script.exists():
        # Fallback when cwd is not the repo root
        script = Path(__file__).resolve().parents[2] / "examples" / "data_preprocess" / "preprocess_all.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Cannot find preprocess_all.py (looked under {root}). "
            "Run from the portbench repo root."
        )
    runpy.run_path(str(script), run_name="__main__")
    return str(root / "datasets" / "processed")


def refresh_and_preprocess(
    *,
    force: bool = True,
    skip_refresh: bool = False,
    skip_preprocess: bool = False,
    repo_root: Optional[Path] = None,
) -> dict:
    """End-to-end data refresh for live eval."""
    summary: dict = {"refreshed": False, "preprocessed": False}
    if not skip_refresh:
        summary["download"] = refresh_market_data(force=force)
        summary["refreshed"] = True
    if not skip_preprocess:
        summary["processed_dir"] = run_preprocess(repo_root=repo_root)
        summary["preprocessed"] = True
    return summary
