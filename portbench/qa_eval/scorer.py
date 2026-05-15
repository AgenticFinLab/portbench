"""
QA response scoring — one function per template, unified dispatch via score_response().

Each scorer takes (ground_truth_answer: str, llm_response: str) and returns float in [0, 1].
For templates with answer_numeric, specialized scorers use that value.
"""

from __future__ import annotations

import math
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

_DIRECTION_KEYWORDS = {
    "positive": "positive",
    "bullish": "positive",
    "upward": "positive",
    "increase": "positive",
    "negative": "negative",
    "bearish": "negative",
    "downward": "negative",
    "decrease": "negative",
    "flat": "flat",
    "neutral": "flat",
    "sideways": "flat",
}


def _extract_direction(text: str) -> Optional[str]:
    text_lower = text.lower()
    for keyword, direction in _DIRECTION_KEYWORDS.items():
        if keyword in text_lower:
            return direction
    return None


def _extract_float(text: str) -> Optional[float]:
    # Match patterns like: -12.5%, 0.034, 12.5 %, -3.14
    matches = re.findall(r"[-+]?\d*\.?\d+\s*%?", text)
    for m in matches:
        try:
            m = m.strip()
            if m.endswith("%"):
                return float(m[:-1].strip()) / 100.0
            return float(m)
        except ValueError:
            continue
    return None


def _extract_weights(text: str, assets: list[str]) -> Optional[dict[str, float]]:
    if not text or not assets:
        return None

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    def _parse_val(v) -> Optional[float]:
        """Parse numeric or string weight value; handles % suffix and >1 percentage form."""
        if isinstance(v, (int, float)):
            val = float(v)
            return val / 100.0 if val > 1.5 else val
        if isinstance(v, str):
            s = v.strip()
            is_pct = s.endswith("%")
            try:
                val = float(s.rstrip("%").strip())
                return val / 100.0 if (is_pct or val > 1.5) else val
            except ValueError:
                return None
        return None

    weights: dict[str, float] = {}

    # ── Try JSON parsing ──────────────────────────────────────────────────────
    try:
        import json as _json
        data = _json.loads(text)

        # Unwrap nested {"weights": {...}} or {"weights": [...]}
        if isinstance(data, dict) and "weights" in data:
            inner = data["weights"]
            if isinstance(inner, dict):
                data = inner
            elif isinstance(inner, list) and len(inner) >= len(assets):
                for i, asset in enumerate(assets):
                    v = _parse_val(inner[i])
                    if v is not None:
                        weights[asset] = v
                return weights if len(weights) == len(assets) else None

        if isinstance(data, dict):
            for asset in assets:
                for key in (
                    asset, asset.lower(),
                    f"w_{asset}", f"w_{asset.lower()}",
                    f"{asset}_weight_pct", f"{asset.lower()}_weight_pct",
                ):
                    if key in data:
                        v = _parse_val(data[key])
                        if v is not None:
                            weights[asset] = v
                            break
        elif isinstance(data, list) and len(data) >= len(assets):
            for i, asset in enumerate(assets):
                v = _parse_val(data[i])
                if v is not None:
                    weights[asset] = v

        if len(weights) == len(assets):
            return weights
    except (ValueError, TypeError):
        pass

    # ── Regex fallback — handles plain text and partially-structured responses ─
    text_lower = text.lower()
    for asset in assets:
        if asset in weights:
            continue
        # Match w_ASSET or ASSET, optional surrounding quotes, then value with optional %
        pattern = rf"(?:w_)?{re.escape(asset.lower())}['\"]?\s*[:\-=]\s*['\"]?([-+]?\d*\.?\d+)\s*(%?)"
        m = re.search(pattern, text_lower)
        if m:
            val = float(m.group(1))
            if m.group(2) == "%" or val > 1.5:
                val /= 100.0
            weights[asset] = val

    if weights:
        return weights

    # ── Last resort: ordered percentages — "0%, 100%" ────────────────────────
    pct_nums = re.findall(r"([-+]?\d*\.?\d+)\s*%", text)
    if len(pct_nums) >= len(assets):
        for i, asset in enumerate(assets):
            weights[asset] = float(pct_nums[i]) / 100.0
        return weights

    return None


def _extract_rebalance_decision(text: str) -> Optional[str]:
    text_lower = text.strip().lower()
    if "rebalance" in text_lower or re.search(r"\byes\b", text_lower):
        return "rebalance"
    if "hold" in text_lower or "no rebalance" in text_lower or re.search(r"\bno\b", text_lower):
        return "hold"
    return None


_REGIME_KEYWORDS = {
    "bull": "bull",
    "bullish": "bull",
    "bear": "bear",
    "bearish": "bear",
    "sideways": "sideways",
    "flat": "sideways",
    "range-bound": "sideways",
    "crisis": "crisis",
    "crash": "crisis",
}


def _extract_regime(text: str) -> Optional[str]:
    text_lower = text.lower()
    for keyword, regime in _REGIME_KEYWORDS.items():
        if keyword in text_lower:
            return regime
    return None


_ALLOC_DIRECTIONS = {"increase", "decrease", "neutral"}


def _extract_alloc_directions(text: str) -> dict[str, str]:
    out = {}

    # Try JSON parsing first (handles {"equities": "increase", ...} and nested part_b)
    try:
        import json as _json
        data = _json.loads(re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`"))
        if isinstance(data, dict):
            # Unwrap {"part_b": {...}} nesting
            if "part_b" in data and isinstance(data["part_b"], dict):
                data = data["part_b"]
            for cls in ["equities", "bonds", "commodities", "real_estate", "cryptocurrency", "cash"]:
                val = data.get(cls, "")
                if val in _ALLOC_DIRECTIONS:
                    out[cls] = val
            if out:
                return out
    except (ValueError, TypeError):
        pass

    # Regex fallback — handles both plain text and JSON key formats
    text_lower = text.lower()
    for cls in ["equities", "bonds", "commodities", "real_estate", "cryptocurrency", "cash"]:
        pattern = rf"{cls}['\"]?\s*[:\-=]\s*['\"]?(\w+)"
        match = re.search(pattern, text_lower)
        if match and match.group(1) in _ALLOC_DIRECTIONS:
            out[cls] = match.group(1)
    return out


# ---------------------------------------------------------------------------
# Per-template scorers
# ---------------------------------------------------------------------------

def _score_t1(gt_answer: str, response: str, **kw) -> float:
    gt_dir = _extract_direction(gt_answer)
    pred_dir = _extract_direction(response)
    if gt_dir is None or pred_dir is None:
        return 0.0
    return 1.0 if gt_dir == pred_dir else 0.0


def _score_t2(gt_answer: str, response: str, answer_numeric: float = None, **kw) -> float:
    if answer_numeric is None:
        answer_numeric = _extract_float(gt_answer)
    pred = _extract_float(response)
    if answer_numeric is None or pred is None:
        return 0.0
    if abs(answer_numeric) < 1e-9:
        return 1.0 if abs(pred) < 1e-6 else 0.0
    rel_err = abs(pred - answer_numeric) / abs(answer_numeric)
    return max(0.0, 1.0 - rel_err)


def _score_t3(gt_answer: str, response: str, answer_numeric: float = None, **kw) -> float:
    return _score_t2(gt_answer, response, answer_numeric=answer_numeric)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return max(0.0, dot / (norm_a * norm_b))


def _score_t4(gt_answer: str, response: str, assets: list[str] = None, **kw) -> float:
    if not assets or len(assets) < 2:
        return 0.0
    gt_weights = _extract_weights(gt_answer, assets)
    pred_weights = _extract_weights(response, assets)
    if not gt_weights or not pred_weights:
        return 0.0
    gt_vec = [gt_weights.get(a, 0.0) for a in assets]
    pred_vec = [pred_weights.get(a, 0.0) for a in assets]
    return _cosine_similarity(gt_vec, pred_vec)


def _score_t5(gt_answer: str, response: str, assets: list[str] = None, **kw) -> float:
    return _score_t4(gt_answer, response, assets=assets)


def _score_t6(gt_answer: str, response: str, **kw) -> float:
    gt_dec = _extract_rebalance_decision(gt_answer)
    pred_dec = _extract_rebalance_decision(response)
    if gt_dec is None or pred_dec is None:
        return 0.0
    return 1.0 if gt_dec == pred_dec else 0.0


def _score_t7(gt_answer: str, response: str, **kw) -> float:
    # Part A: regime detection (0.5)
    gt_regime = _extract_regime(gt_answer)
    pred_regime = _extract_regime(response)
    regime_score = 0.5 if (gt_regime and pred_regime and gt_regime == pred_regime) else 0.0

    # Part B: allocation directions (0.5)
    gt_dirs = _extract_alloc_directions(gt_answer)
    pred_dirs = _extract_alloc_directions(response)
    if gt_dirs:
        matches = sum(1 for cls in gt_dirs if gt_dirs.get(cls) == pred_dirs.get(cls))
        alloc_score = 0.5 * matches / len(gt_dirs)
    else:
        alloc_score = 0.0

    return regime_score + alloc_score


_SCORERS = {
    "T1": _score_t1,
    "T2": _score_t2,
    "T3": _score_t3,
    "T4": _score_t4,
    "T5": _score_t5,
    "T6": _score_t6,
    "T7": _score_t7,
}


def score_response(
    template_id: str,
    gt_answer: str,
    llm_response: str,
    answer_numeric: float = None,
    assets: list[str] = None,
) -> float:
    """
    Score an LLM response against a ground-truth answer.

    Args:
        template_id: "T1" through "T7"
        gt_answer: Ground-truth answer string from QAPair.answer
        llm_response: Raw LLM output
        answer_numeric: Optional numeric ground truth (for T2/T3)
        assets: Optional asset list (for T4/T5 weight extraction)

    Returns:
        float in [0, 1]
    """
    scorer = _SCORERS.get(template_id)
    if scorer is None:
        return 0.0
    try:
        return scorer(
            gt_answer, llm_response,
            answer_numeric=answer_numeric, assets=assets,
        )
    except Exception:
        return 0.0
