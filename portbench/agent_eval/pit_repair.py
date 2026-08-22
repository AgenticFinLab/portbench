"""Deterministic Point-in-Time repair operators for the five stages."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


VERSION = "pit_repair_v2"
_FUTURE_KEYS = frozenset(
    {"future_return_data", "future_returns", "forward_returns", "forward_return_data"}
)


def _is_future_key(key: Any) -> bool:
    """Return whether a field name denotes forward information."""
    key_text = str(key).lower()
    return key_text in _FUTURE_KEYS or key_text.startswith("future_") or key_text.startswith("forward_")


def _scan_future(value: Any, path: str, seen: set[int]) -> None:
    """Recursively reject future-data fields without traversing array internals."""
    if value is None or isinstance(value, (str, bytes, int, float, bool, pd.Series, pd.DataFrame, np.ndarray)):
        return
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if _is_future_key(key):
                raise PermissionError(f"{VERSION} forbids future data at {child_path}")
            _scan_future(item, child_path, seen)
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            if _is_future_key(field.name) and getattr(value, field.name) is not None:
                raise PermissionError(f"{VERSION} forbids future data at {path}.{field.name}")
            _scan_future(getattr(value, field.name), f"{path}.{field.name}", seen)
        return
    if isinstance(value, Sequence):
        for index, item in enumerate(value):
            _scan_future(item, f"{path}[{index}]", seen)


def _raise_if_future(context: Optional[Mapping[str, Any]], **kwargs: Any) -> None:
    """Reject forward information at any nested path."""
    _scan_future(context, "context", set())
    _scan_future(kwargs, "kwargs", set())


def _lookback_returns(context: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Sequence[float]]:
    """Extract the declared lookback return mapping."""
    raw = kwargs.get("lookback_returns")
    if raw is None:
        raw = context.get("lookback_returns")
    if raw is None:
        raw = kwargs.get("returns")
    if raw is None:
        raw = context.get("returns", {})
    if not isinstance(raw, Mapping):
        raise ValueError("lookback_returns must map assets to historical returns")
    return raw


def _series(values: Sequence[float]) -> pd.Series:
    """Return finite historical observations as a float series."""
    return pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()


def repair_s1(context: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    """Return an S1-compatible view from trailing mean returns only."""
    _raise_if_future(context, **kwargs)
    returns = _lookback_returns(context or {}, **kwargs)
    views: Dict[str, float] = {}
    for asset, values in returns.items():
        recent = _series(values).tail(20)
        mean_return = float(recent.mean()) if not recent.empty else 0.0
        views[str(asset)] = float(np.clip(mean_return / 0.02, -1.0, 1.0))
    market_mean = float(np.mean(list(views.values()))) if views else 0.0
    regime = "bull" if market_mean > 0.2 else "bear" if market_mean < -0.2 else "sideways"
    return {
        "asset_views": views,
        "macro_summary": "Deterministic lookback-only repair.",
        "detected_regime": regime,
        "confidence": float(np.clip(abs(market_mean), 0.0, 1.0)),
        "repair_version": VERSION,
    }


def repair_s2(context: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    """Return S2-compatible signals derived from repaired S1 views."""
    _raise_if_future(context, **kwargs)
    s1_output = kwargs.get("s1_output") or (context or {}).get("s1_output")
    if not isinstance(s1_output, Mapping):
        s1_output = repair_s1(context, **kwargs)
    views = s1_output.get("asset_views", {})
    signals: Dict[str, str] = {}
    strengths: Dict[str, float] = {}
    for asset, raw_view in views.items():
        view = float(raw_view)
        signals[str(asset)] = "buy" if view > 0.2 else "sell" if view < -0.2 else "hold"
        strengths[str(asset)] = float(np.clip(abs(view), 0.0, 1.0))
    return {
        "signals": signals,
        "strengths": strengths,
        "reasoning": "Deterministic thresholds on repaired S1 views.",
        "repair_version": VERSION,
    }


def repair_s3(context: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    """Return S3-compatible long-only minimum-variance lookback weights."""
    _raise_if_future(context, **kwargs)
    returns = _lookback_returns(context or {}, **kwargs)
    frame = pd.DataFrame({str(asset): _series(values) for asset, values in returns.items()}).dropna(how="all")
    if frame.empty or not len(frame.columns):
        raise ValueError("S3 repair requires historical returns")
    covariance = frame.cov().fillna(0.0).to_numpy(dtype=float)
    covariance += np.eye(len(frame.columns)) * 1e-8
    inverse = np.linalg.pinv(covariance)
    raw = inverse @ np.ones(len(frame.columns), dtype=float)
    raw = np.clip(raw, 0.0, None)
    if float(raw.sum()) <= 0.0:
        raw = np.ones(len(frame.columns), dtype=float)
    raw /= raw.sum()
    weights = {asset: float(weight) for asset, weight in zip(frame.columns, raw)}
    return {
        "weights": weights,
        "expected_return": 0.0,
        "expected_vol": float(np.sqrt(max(raw @ covariance @ raw, 0.0))),
        "sharpe_estimate": 0.0,
        "repair_version": VERSION,
    }


def repair_s4(context: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    """Return a legal S4 plan that tracks the declared target weights."""
    _raise_if_future(context, **kwargs)
    target = kwargs.get("target_weights") or (context or {}).get("target_weights") or {}
    if not isinstance(target, Mapping) or not target:
        raise ValueError("S4 repair requires target_weights")
    orders = []
    current = kwargs.get("current_weights") or (context or {}).get("current_weights") or {}
    for asset, weight in target.items():
        current_weight = float(current.get(asset, 0.0)) if isinstance(current, Mapping) else 0.0
        target_weight = float(weight)
        direction = "buy" if target_weight > current_weight else "sell" if target_weight < current_weight else "hold"
        orders.append({"asset": str(asset), "direction": direction, "target_weight": target_weight})
    return {"orders": orders, "order_type": "market", "urgency": "normal", "repair_version": VERSION}


def repair_s5(context: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    """Return a lookback-only S5 risk-control decision."""
    _raise_if_future(context, **kwargs)
    returns = _lookback_returns(context or {}, **kwargs)
    recent_means = [float(_series(values).tail(10).mean()) for values in returns.values() if not _series(values).empty]
    stressed = bool(recent_means and float(np.mean(recent_means)) < -0.01)
    return {
        "action": "scale_down" if stressed else "hold",
        "alerts": ["lookback_stress"] if stressed else [],
        "scale_factor": 0.8 if stressed else None,
        "repair_version": VERSION,
    }


__all__ = ["VERSION", "repair_s1", "repair_s2", "repair_s3", "repair_s4", "repair_s5"]
