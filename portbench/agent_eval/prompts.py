"""
Centralized prompt templates for the S1–S3 LLM stages.

Design rules:
  - Each builder takes only the runtime values it needs and returns a string.
  - All prompts end with the same hard JSON-format contract (`format_contract`).
  - Without tools, the contract tells the model to emit a single JSON object,
    no prose, no code fences, starts with `{` and ends with `}`.
  - With tools, the model may call tools first; the *final* assistant message
    must still be exactly one JSON object. This is the same contract enforced
    by _call_with_json_retry's correction message in stages.py.

If a model still violates the contract on first try, _call_with_json_retry
re-issues the prompt with an even stronger reminder appended.
"""

from __future__ import annotations

import re
from typing import Iterable

from .base import MarketSnapshot, S1Output, S2Output


# ---------------------------------------------------------------------------
# Shared format contract — appended at the end of every prompt
# ---------------------------------------------------------------------------

FORMAT_CONTRACT = """\
OUTPUT FORMAT — STRICT:
  1. Reply with EXACTLY ONE JSON object and nothing else.
  2. The very first character of your reply MUST be `{` and the very last
     character MUST be `}`. No leading whitespace, no trailing newline text.
  3. Do NOT wrap the object in markdown code fences (no ```json, no ```).
  4. Do NOT add any explanation, preamble, or commentary before/after.
  5. Use double quotes for all keys and string values; numbers must be raw
     JSON numbers (no quotes, no `%`, no commas).
  6. Every key listed in the schema below MUST appear exactly once.
"""

TOOL_FORMAT_CONTRACT = """\
OUTPUT FORMAT — STRICT:
  1. You MAY call the provided tools first to inspect Point-in-Time data.
  2. After any tool rounds, your FINAL assistant message must be EXACTLY ONE
     JSON object and nothing else.
  3. The very first character of that final message MUST be `{` and the very
     last character MUST be `}`. No leading whitespace, no trailing newline text.
  4. Do NOT wrap the object in markdown code fences (no ```json, no ```).
  5. Do NOT add any explanation, preamble, or commentary before/after the JSON.
  6. Use double quotes for all keys and string values; numbers must be raw
     JSON numbers (no quotes, no `%`, no commas).
  7. Every key listed in the schema below MUST appear exactly once.
"""


def format_contract(use_tools: bool = False) -> str:
    """Return the JSON output contract, optionally allowing a tool round first."""
    return TOOL_FORMAT_CONTRACT if use_tools else FORMAT_CONTRACT


def _schema_block(schema_lines: Iterable[str]) -> str:
    """Render a JSON schema as a fenced-looking block inside the prompt."""
    return "JSON SCHEMA:\n{\n" + "\n".join(schema_lines) + "\n}"


def _quote(s: str) -> str:
    """Wrap a string in double quotes — keeps backslashes out of f-strings."""
    return '"' + s + '"'


# ---------------------------------------------------------------------------
# Regime display mapping — neutral labels for prompt text
#
# Internal regime labels (used for scoring/storage) are kept as-is.
# Only the text that goes into LLM prompts is remapped.
# ---------------------------------------------------------------------------

_REGIME_DISPLAY: dict[str, str] = {
    "crisis": "high-volatility",
}


def _display_regime(regime: str | None) -> str:
    """Map an internal regime label to its prompt-safe display string."""
    if regime is None:
        return "unknown"
    return _REGIME_DISPLAY.get(regime.lower(), regime)


# ---------------------------------------------------------------------------
# News text sanitiser — strip event-specific crisis language
#
# News articles from stress periods can contain words that trigger content
# safety filters in some LLMs.  We replace the surface form while keeping
# the factual market signal intact (price data tells the model what happened).
# Applied uniformly across all models to preserve cross-model comparability.
# ---------------------------------------------------------------------------

_NEWS_REPLACEMENTS: list[tuple[str, str]] = [
    (r"\bcrash(?:ed|es|ing)?\b", "sharp decline"),
    (r"\bcollapse(?:d|s|ing)?\b", "sharp correction"),
    (r"\bcrisis\b", "high-volatility period"),
    (r"\bbankruptcy\b", "financial difficulty"),
    (r"\bbankrupt(?:ed|ing)?\b", "in financial difficulty"),
    (r"\bmeltdown\b", "significant drawdown"),
    (r"\bpanic(?:ked|king|s)?\b", "rapid"),
    (r"\bcatastrophe\b", "adverse event"),
    (r"\bdisaster\b", "adverse event"),
    (r"\bdefault(?:ed|ing|s)?\b", "payment difficulty"),
]

_NEWS_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), repl) for pat, repl in _NEWS_REPLACEMENTS
]


def _sanitize_news(text: str) -> str:
    """Replace crisis-specific terms in news text with neutral equivalents."""
    for pattern, replacement in _NEWS_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# S1 — Market Interpretation
# ---------------------------------------------------------------------------


def build_s1_prompt(
    snapshot: MarketSnapshot,
    assets: list[str],
    price_context: str,
    macro_block: str,
    corr_block: str,
    trailing_days: int,
    use_tools: bool = False,
) -> str:
    """Prompt for Stage 1: structured asset views + regime + macro summary."""
    news_block = (
        f"\nRECENT NEWS / FILINGS:\n{_sanitize_news(snapshot.news_text[:3000])}\n"
        if snapshot.news_text
        else ""
    )
    schema = _schema_block(
        [
            '  "asset_views": { "ASSET": <float in [-1, 1]> },',
            '  "detected_regime": "<bull|bear|sideways|high-volatility>",',
            '  "confidence": <float in [0, 1]>,',
            '  "macro_summary": "<one sentence>"',
        ]
    )
    return f"""You are a portfolio manager analyzing market conditions on {snapshot.decision_date}.

MARKET DATA (trailing {trailing_days} trading days):
{price_context}

{macro_block}

{corr_block}

Current market regime context: {_display_regime(snapshot.market_regime)}
{news_block}
TASK: Interpret the market data and provide structured asset views.

Return a sparse set of material asset views. Any visible asset omitted from
"asset_views" is deterministically interpreted as neutral (0.0). Only use
visible asset identifiers and do not invent assets. You may include an
explicit neutral 0.0 view, but do not need to list every neutral asset.

For each reported asset, assign a sentiment score in [-1.0, +1.0]:
  +1.0 = strongly bullish (expect strong outperformance)
   0.0 = neutral
  -1.0 = strongly bearish (expect significant underperformance)

Identify the overall market regime: one of "bull", "bear", "sideways", "high-volatility".

{schema}

{format_contract(use_tools)}"""


# ---------------------------------------------------------------------------
# S2 — Signal Generation
# ---------------------------------------------------------------------------


def build_s2_prompt(
    snapshot: MarketSnapshot,
    s1: S1Output,
    assets: list[str],
    use_tools: bool = False,
) -> str:
    """Prompt for Stage 2: discrete buy/hold/sell signals + strengths."""
    views_str = "\n".join(f"  {a}: view={v:+.3f}" for a, v in s1.asset_views.items())
    news_block = (
        f"\nRecent news / filings:\n{_sanitize_news(snapshot.news_text)}\n"
        if snapshot.news_text
        else ""
    )
    schema = _schema_block(
        [
            '  "signals": {"SELECTED_ASSET": "<buy|sell>", ...},',
            '  "strengths": {"SELECTED_ASSET": <float in [0, 1]>, ...},',
            '  "reasoning": "<one sentence>"',
        ]
    )
    return f"""You are a portfolio manager on {snapshot.decision_date}.

Stage 1 market interpretation produced these asset views (scale: -1=bearish, +1=bullish):
{views_str}

Detected market regime: {_display_regime(s1.detected_regime)}
Macro summary: {s1.macro_summary}
{news_block}
TASK: Convert each asset view into an actionable trading signal.

Rules:
  - view >  0.15: consider "buy"
  - view < -0.15: consider "sell"
  - otherwise:    consider "hold"

Use your judgement to refine signals based on regime and macro context.
Signal strength should reflect conviction (0.0 = low, 1.0 = high).
Return only non-hold signals. Omitted visible assets are interpreted as hold with strength 0.5.
The signals and strengths objects must contain exactly the same selected assets.

{schema}

{format_contract(use_tools)}"""


# ---------------------------------------------------------------------------
# S3 — Weight Optimization
# ---------------------------------------------------------------------------


def build_s3_prompt(
    snapshot: MarketSnapshot,
    s2: S2Output,
    assets: list[str],
    corr_block: str,
    use_tools: bool = False,
) -> str:
    """Prompt for Stage 3: sparse relative allocation scores."""
    signals_str = "\n".join(
        f"  {a}: signal={s2.signals[a]}, strength={s2.strengths.get(a, 0.5):.2f}"
        for a in assets
    )
    current_w_str = ", ".join(
        f"{asset}={weight:.4f}"
        for asset, weight in snapshot.current_weights.items()
        if abs(float(weight)) > 1e-8
    ) or "none"
    corr_section = f"\n{corr_block}\n" if corr_block else ""
    schema = _schema_block(
        [
            '  "allocation_scores": {"SELECTED_ASSET": <positive decimal>, ...},',
            '  "expected_return": <annualized decimal, e.g. 0.08>,',
            '  "expected_vol": <annualized decimal, e.g. 0.12>,',
            '  "sharpe_estimate": <decimal>',
        ]
    )
    return f"""You are a portfolio manager on {snapshot.decision_date}.

Stage 2 signals:
{signals_str}

Current portfolio weights: {current_w_str}
Portfolio NAV: ${snapshot.portfolio_value:,.0f}
Market regime: {_display_regime(snapshot.market_regime)}
{corr_section}
TASK: Select a sparse portfolio allocation based on the signals above.

Constraints:
  - allocation_scores must be finite, strictly positive decimal numbers
  - Do not normalize allocation_scores; the deterministic environment normalizes their relative magnitudes into target weights
  - Return only selected assets in allocation_scores; omitted visible assets receive a target weight of 0.0
  - Select at most 12 assets
  - "sell" signals should receive reduced weight (ideally 0.0)
  - "buy" signals should receive increased weight
  - Minimize unnecessary turnover from current weights

{schema}

{format_contract(use_tools)}"""


# ---------------------------------------------------------------------------
# JSON-format correction suffix (used by _call_with_json_retry)
# ---------------------------------------------------------------------------


def build_format_correction_suffix(last_error: str) -> str:
    """Suffix appended to the prompt on a JSON-parse retry."""
    return (
        "\n\n=== FORMAT CORRECTION ===\n"
        f"Your previous response could not be parsed as JSON (error: {last_error}).\n"
        "Re-emit the SAME content but obey every rule of OUTPUT FORMAT above.\n"
        "Specifically: start with `{`, end with `}`, no markdown, no prose."
    )
