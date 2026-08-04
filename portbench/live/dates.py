"""Resolve trading-day calendars and live date pairs."""

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
    *,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> list[date]:
    """Return sorted trading dates with prices for ``anchor_asset``."""
    end_d = end or date.today()
    start_d = start or (end_d - timedelta(days=400))
    series = provider.get_price_series(anchor_asset, start_d, end_d)
    if series is None or series.empty:
        series = provider.get_price_series(anchor_asset, date(2015, 1, 1), end_d)
    if series is None or series.empty:
        raise RuntimeError(
            f"No price history for {anchor_asset}. "
            "Run data collect + preprocess, or pass dates that exist in data."
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
    Resolve a single (decision, realization) pair.

    Defaults: realization = latest available day; decision = previous day.
    """
    days = available_trading_days(provider, anchor_asset=anchor_asset)
    if len(days) < 2:
        raise RuntimeError(
            f"Need at least 2 trading days for live eval; found {days!r}."
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


def last_weekday_on_or_before(d: date) -> date:
    """Simple weekday floor (Sat/Sun → Friday). Not a full exchange calendar."""
    out = d
    while out.weekday() >= 5:
        out -= timedelta(days=1)
    return out


def local_data_max_date(provider, anchor_asset: str = "SPY") -> date:
    """Latest trading date present in local processed data for ``anchor_asset``."""
    days = available_trading_days(provider, anchor_asset=anchor_asset)
    return days[-1]


def coverage_needed_through(
    *,
    as_of_today: Optional[date] = None,
    range_end: Optional[date] = None,
    decision_date: Optional[date] = None,
) -> date:
    """
    Latest calendar date that local data must cover for this live run.

    - Range mode: need through ``range_end`` (plus caller may require one more
      trading day for GT; we refresh if ``range_end`` itself is missing).
    - Single-step with explicit as_of/decision: need through that date.
    - Default "today" mode: need through the last weekday on/before today.
    """
    if range_end is not None:
        return _to_date(range_end)
    if as_of_today is not None:
        return _to_date(as_of_today)
    if decision_date is not None:
        # Need decision + at least the next session for ex-post GT.
        return _to_date(decision_date) + timedelta(days=3)
    return last_weekday_on_or_before(date.today())


def is_coverage_insufficient(
    provider,
    needed_through: date,
    *,
    anchor_asset: str = "SPY",
    as_of: Optional[date] = None,
) -> bool:
    """
    True when local data is missing sessions we can reasonably expect.

    Does **not** require future calendar days or same-day EOD that vendors
    have not published yet. Target = last weekday on/before
    ``min(needed_through, as_of)``, then allow one weekday of EOD lag.
    """
    try:
        local_max = local_data_max_date(provider, anchor_asset=anchor_asset)
    except RuntimeError:
        return True
    today = _to_date(as_of) if as_of is not None else date.today()
    # Never demand dates after "today" (or after needed_through).
    target = last_weekday_on_or_before(min(_to_date(needed_through), today))
    # Vendors (Yahoo/AKShare) often lag one session behind the calendar.
    expect = last_weekday_on_or_before(target - timedelta(days=1))
    return local_max < expect
