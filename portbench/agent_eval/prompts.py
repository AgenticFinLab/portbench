"""
Centralized prompt templates for the S1–S3 LLM stages.

Design rules:
  - Each builder takes only the runtime values it needs and returns a string.
  - All prompts end with the same hard JSON-format contract (FORMAT_CONTRACT).
  - The contract is intentionally repetitive: it tells the model to emit a
    single JSON object, no prose, no code fences, starts with `{` and ends
    with `}`. This is the same contract enforced by _call_with_json_retry's
    correction message in stages.py.

If a model still violates the contract on first try, _call_with_json_retry
re-issues the prompt with an even stronger reminder appended.
"""

from __future__ import annotations

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


def _schema_block(schema_lines: Iterable[str]) -> str:
    """Render a JSON schema as a fenced-looking block inside the prompt."""
    return "JSON SCHEMA:\n{\n" + "\n".join(schema_lines) + "\n}"


def _quote(s: str) -> str:
    """Wrap a string in double quotes — keeps backslashes out of f-strings."""
    return '"' + s + '"'


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
) -> str:
    """Prompt for Stage 1: structured asset views + regime + macro summary."""
    news_block = (
        f"\nRECENT NEWS / FILINGS:\n{snapshot.news_text[:3000]}\n"
        if snapshot.news_text
        else ""
    )
    asset_view_fields = ", ".join(f"{_quote(a)}: <float in [-1, 1]>" for a in assets)
    schema = _schema_block(
        [
            f'  "asset_views": {{ {asset_view_fields} }},',
            '  "detected_regime": "<bull|bear|sideways|crisis>",',
            '  "confidence": <float in [0, 1]>,',
            '  "macro_summary": "<one sentence>"',
        ]
    )
    return f"""You are a portfolio manager analyzing market conditions on {snapshot.decision_date}.

MARKET DATA (trailing {trailing_days} trading days):
{price_context}

{macro_block}

{corr_block}

Current market regime context: {snapshot.market_regime or "unknown"}
{news_block}
TASK: Interpret the market data and provide structured asset views.

For each asset, assign a sentiment score in [-1.0, +1.0]:
  +1.0 = strongly bullish (expect strong outperformance)
   0.0 = neutral
  -1.0 = strongly bearish (expect significant underperformance)

Identify the overall market regime: one of "bull", "bear", "sideways", "crisis".

{schema}

{FORMAT_CONTRACT}"""


# ---------------------------------------------------------------------------
# S2 — Signal Generation
# ---------------------------------------------------------------------------


def build_s2_prompt(
    snapshot: MarketSnapshot,
    s1: S1Output,
    assets: list[str],
) -> str:
    """Prompt for Stage 2: discrete buy/hold/sell signals + strengths."""
    views_str = "\n".join(f"  {a}: view={v:+.3f}" for a, v in s1.asset_views.items())
    news_block = (
        f"\nRecent news / filings:\n{snapshot.news_text}\n"
        if snapshot.news_text
        else ""
    )
    sig_fields = ", ".join(f'{_quote(a)}: "<buy|hold|sell>"' for a in assets)
    str_fields = ", ".join(f"{_quote(a)}: <float in [0, 1]>" for a in assets)
    schema = _schema_block(
        [
            f'  "signals": {{ {sig_fields} }},',
            f'  "strengths": {{ {str_fields} }},',
            '  "reasoning": "<one sentence>"',
        ]
    )
    return f"""You are a portfolio manager on {snapshot.decision_date}.

Stage 1 market interpretation produced these asset views (scale: -1=bearish, +1=bullish):
{views_str}

Detected market regime: {s1.detected_regime}
Macro summary: {s1.macro_summary}
{news_block}
TASK: Convert each asset view into an actionable trading signal.

Rules:
  - view >  0.15: consider "buy"
  - view < -0.15: consider "sell"
  - otherwise:    consider "hold"

Use your judgement to refine signals based on regime and macro context.
Signal strength should reflect conviction (0.0 = low, 1.0 = high).

{schema}

{FORMAT_CONTRACT}"""


# ---------------------------------------------------------------------------
# S3 — Weight Optimization
# ---------------------------------------------------------------------------


def build_s3_prompt(
    snapshot: MarketSnapshot,
    s2: S2Output,
    assets: list[str],
    corr_block: str,
) -> str:
    """Prompt for Stage 3: portfolio weight allocation summing to 1.0."""
    signals_str = "\n".join(
        f"  {a}: signal={s2.signals[a]}, strength={s2.strengths.get(a, 0.5):.2f}"
        for a in assets
    )
    current_w_str = ", ".join(
        f"{a}={w:.3f}" for a, w in snapshot.current_weights.items()
    )
    corr_section = f"\n{corr_block}\n" if corr_block else ""
    weight_fields = ", ".join(f"{_quote(a)}: <float in [0, 1]>" for a in assets)
    schema = _schema_block(
        [
            f'  "weights": {{ {weight_fields} }},',
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
Market regime: {snapshot.market_regime or "unknown"}
{corr_section}
TASK: Allocate portfolio weights based on the signals above.

Constraints:
  - All weights must be in [0.0, 1.0]
  - Weights must sum to exactly 1.0
  - "sell" signals should receive reduced weight (ideally 0.0)
  - "buy" signals should receive increased weight
  - Minimize unnecessary turnover from current weights

{schema}

{FORMAT_CONTRACT}"""


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
