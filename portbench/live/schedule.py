"""Rebalance schedules for live / rolling evaluation."""

from __future__ import annotations

from datetime import date
from typing import Iterable, Literal

import pandas as pd

RebalanceFreq = Literal["daily", "weekly", "monthly", "quarterly", "yearly"]

SUPPORTED_FREQUENCIES: tuple[str, ...] = (
    "daily",
    "weekly",
    "monthly",
    "quarterly",
    "yearly",
)


def _to_dates(days: Iterable[date]) -> list[date]:
    return sorted({d if isinstance(d, date) else pd.Timestamp(d).date() for d in days})


def filter_rebalance_dates(trading_days: list[date], frequency: str) -> list[date]:
    """
    Pick rebalance decision dates from a trading-day calendar.

    - daily: every trading day
    - weekly: first trading day of each ISO week
    - monthly: first trading day of each calendar month
    - quarterly: first trading day of Jan/Apr/Jul/Oct
    - yearly: first trading day of each calendar year
    """
    freq = frequency.lower().strip()
    if freq not in SUPPORTED_FREQUENCIES:
        raise ValueError(
            f"Unsupported rebalance frequency {frequency!r}. "
            f"Choose from {SUPPORTED_FREQUENCIES}."
        )
    days = _to_dates(trading_days)
    if not days:
        return []
    if freq == "daily":
        return days

    out: list[date] = []
    seen = set()
    for d in days:
        if freq == "weekly":
            key = (d.isocalendar().year, d.isocalendar().week)
        elif freq == "monthly":
            key = (d.year, d.month)
        elif freq == "quarterly":
            key = (d.year, (d.month - 1) // 3)
        else:  # yearly
            key = d.year
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def decision_realization_pairs(
    trading_days: list[date],
    *,
    start: date,
    end: date,
    frequency: str = "daily",
) -> list[tuple[date, date]]:
    """
    Build (decision_date, realization_date) pairs for a rolling live window.

    ``realization_date`` is the next rebalance date after the decision (for daily:
    the next trading day). Decisions are those rebalance dates in [start, end]
    that still have a following realization date in the calendar.
    """
    days = _to_dates(trading_days)
    # Need calendar coverage a bit past ``end`` so the last decision has a GT day.
    rebalance_all = filter_rebalance_dates(days, frequency)
    decisions = [d for d in rebalance_all if start <= d <= end]
    pairs: list[tuple[date, date]] = []
    for d in decisions:
        later = [x for x in rebalance_all if x > d]
        if not later:
            # Fall back: next raw trading day after decision
            later = [x for x in days if x > d]
        if not later:
            continue
        pairs.append((d, later[0]))
    return pairs
