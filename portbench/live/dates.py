"""Resolve yesterday / today trading-day pairs for live eval."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd


def _to_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return pd.Timestamp(value).date()


def available_trading_days(
    provider,
    anchor_asset: str = "SPY",
    lookback_calendar_days: int = 400,
) -> list[date]:
    """
    Return sorted trading dates with prices for ``anchor_asset``.

    Only dates present in the provider are returned (no synthetic calendar days).
    """
    end = date.today()
    start = end - timedelta(days=lookback_calendar_days)
    series = provider.get_price_series(anchor_asset, start, end)
    if series is None or series.empty:
        # Last resort: ask for a very wide window
        series = provider.get_price_series(
            anchor_asset, date(2015, 1, 1), end
        )
    if series is None or series.empty:
        raise RuntimeError(
            f"No price history for {anchor_asset}. "
            "Run data collect + preprocess, or pass explicit dates that exist in data."
        )
    return sorted({_to_date(i) for i in series.index})


def resolve_live_dates(
    provider,
    *,
    decision_date: Optional[date] = None,
    as_of_today: Optional[date] = None,
    anchor_asset: str = "SPY",
) -> tuple[date, date]:
    """
    Resolve (yesterday/decision, today/realization) trading dates from real data.

    Defaults:
      today     = latest available trading day for anchor_asset in processed data
      yesterday = previous available trading day
    """
    days = available_trading_days(provider, anchor_asset=anchor_asset)
    if len(days) < 2:
        raise RuntimeError(
            f"Need at least 2 trading days for live eval; found {days!r} "
            f"for asset {anchor_asset}."
        )

    if as_of_today is not None:
        today = _to_date(as_of_today)
        eligible = [d for d in days if d <= today]
        if not eligible:
            raise RuntimeError(f"No trading day on or before as_of_today={today}")
        today = eligible[-1]
    else:
        today = days[-1]

    if decision_date is not None:
        yesterday = _to_date(decision_date)
        if yesterday >= today:
            raise ValueError(
                f"decision_date ({yesterday}) must be before today ({today})"
            )
        if yesterday not in days:
            eligible = [d for d in days if d < today and d <= yesterday]
            if not eligible:
                raise RuntimeError(
                    f"No trading day on or before decision_date={decision_date}"
                )
            yesterday = eligible[-1]
    else:
        prior = [d for d in days if d < today]
        if not prior:
            raise RuntimeError(f"No prior trading day before today={today}")
        yesterday = prior[-1]

    return yesterday, today


def calendar_forward_days(decision: date, realization: date) -> int:
    """Calendar-day span for SnapshotBuilder.forward_days covering realization."""
    delta = (realization - decision).days
    return max(1, delta + 2)


def iter_daily_decision_pairs(
    provider,
    start: date,
    end: date,
    *,
    anchor_asset: str = "SPY",
) -> list[tuple[date, date]]:
    """
    Daily-rebalance pairs over ``[start, end]`` (inclusive on decision dates).

    For each trading day D in the window, pair with the next trading day T > D
    (T may fall after ``end``). Skips D if no next day exists in the data.
    """
    start = _to_date(start)
    end = _to_date(end)
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    days = available_trading_days(provider, anchor_asset=anchor_asset)
    # Ensure we can see the next day after ``end``
    days = [d for d in days if d >= start]
    if len(days) < 2:
        raise RuntimeError(
            f"Need trading days covering [{start}, {end}] plus one next day; "
            f"found {days!r}. Refresh/preprocess data if the window is missing."
        )

    pairs: list[tuple[date, date]] = []
    for i, d in enumerate(days):
        if d < start or d > end:
            continue
        # next trading day after d
        nxt = None
        for cand in days[i + 1 :]:
            if cand > d:
                nxt = cand
                break
        if nxt is None:
            continue
        pairs.append((d, nxt))
    if not pairs:
        raise RuntimeError(
            f"No daily decision pairs in [{start}, {end}]. "
            "Check that processed data covers this window and the next session."
        )
    return pairs
