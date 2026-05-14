"""
Concrete implementations of the five pipeline stages.

Each stage:
  1. Implements compute_ground_truth() — deterministic rule-based ideal answer
  2. Implements run()                 — two paths:
       a. MockAgentAdapter: adds calibrated noise to ground truth (for testing)
       b. Real LLM adapter: builds a prompt, calls adapter.complete(), parses JSON
  3. Implements score()              — stage-appropriate distance metric

Prompt design principles:
  - Each prompt provides a JSON schema for the expected response
  - Prompts include all numerically relevant context from MarketSnapshot
  - Parsing is robust: wraps JSON extraction in a try/except that falls back
    to the ground truth rather than crashing the pipeline

To use a real LLM:
    from portbench.agent_eval.llm_adapters import AnthropicAdapter
    adapter = AnthropicAdapter(model="claude-opus-4-6")
    pipeline = build_default_pipeline(adapter)
    result = pipeline.run_episode(snapshot)
"""

import json
import re
from typing import Any, Optional
import numpy as np
import pandas as pd

from .base import (
    AgentAdapter,
    MarketSnapshot,
    PipelineStage,
    RiskAlert,
    S1Output, S2Output, S3Output, S4Output, S5Output, TradeOrder,
    StageID,
)
from .mock_agent import MockAgentAdapter
from .prompts import (
    build_s1_prompt,
    build_s2_prompt,
    build_s3_prompt,
    build_format_correction_suffix,
)
from .tools import get_tools
from ..metrics.risk_metrics import var, max_drawdown
from ..metrics.base import MetricsConfig


# ---------------------------------------------------------------------------
# Shared prompt utilities
# ---------------------------------------------------------------------------

def _recover_truncated_json(text: str) -> Optional[dict]:
    """
    Last-resort recovery for truncated LLM output (hit max_tokens mid-JSON).
    Scans for complete "key":"value" or "key":number pairs inside any named
    sub-dict and returns whatever could be extracted, keyed by sub-dict name.
    Returns None if nothing useful is found.
    """
    result: dict = {}
    # Find each named sub-dict: "fieldname": {  ...
    for m in re.finditer(r'"(\w+)"\s*:\s*\{', text):
        field = m.group(1)
        sub = text[m.end():]
        # Extract complete pairs only: "KEY":"val" or "KEY":number
        pairs: dict = {}
        for pm in re.finditer(r'"([^"\\]+)"\s*:\s*(?:"([^"\\]*)"|([-\d.]+))', sub):
            # Stop once we hit the closing brace of this sub-dict
            if "}" in sub[: pm.start()]:
                break
            k = pm.group(1)
            v: Any = pm.group(2) if pm.group(2) is not None else float(pm.group(3))
            pairs[k] = v
        if pairs:
            result[field] = pairs
    return result or None


def _extract_json(text: str) -> dict:
    """
    Extract the first JSON object from a model response string.

    Handles common LLM formatting patterns:
      - Bare JSON:                       {"key": "value"}
      - Markdown code block:             ```json\n{...}\n```
      - JSON embedded in prose:          "Here is the answer: {...} Done."
      - Body without outer braces:       '"asset_views": {...}, "regime": "bull"'
      - Multiple sibling JSON fragments: picks the first balanced object
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    if not text:
        raise ValueError("Empty model response")

    # Case A: text begins with `{` — find the FIRST balanced object
    if text.lstrip().startswith("{"):
        s = text.lstrip()
        depth = 0
        in_str = False
        esc = False
        start = None
        for i, ch in enumerate(s):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    return json.loads(s[start : i + 1])
        # Fall through to greedy regex
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            return json.loads(match.group())
        # Case A-recovery: JSON was truncated (e.g. hit max_tokens).
        # Extract whatever complete sub-dicts are present so downstream stages
        # can fill in defaults for the missing keys rather than failing entirely.
        recovered = _recover_truncated_json(s)
        if recovered:
            return recovered

    # Case B: looks like the body without outer braces — wrap and try
    if '"' in text and ":" in text:
        try:
            return json.loads("{" + text.rstrip(",").rstrip() + "}")
        except json.JSONDecodeError:
            pass

    # Case C: greedy fallback
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in model response: {text[:200]}")


_JSON_RETRY_LIMIT = 2  # additional attempts after the first parse failure


def _max_sharpe_weights(assets: list[str], future_return_data: dict) -> dict[str, float]:
    """
    Compute max-Sharpe weights for the given assets using realized future returns.
    Falls back to equal-weight if optimization fails or data is insufficient.
    """
    from scipy.optimize import minimize

    data = {a: future_return_data[a] for a in assets if a in future_return_data and not future_return_data[a].empty}
    eq = {a: round(1.0 / len(assets), 4) for a in assets}
    if len(data) < 2:
        return eq

    df = pd.DataFrame(data).dropna()
    if len(df) < 2:
        return eq

    mu = df.mean().values
    cov = df.cov().values
    n = len(mu)
    asset_list = list(data.keys())

    def neg_sharpe(w):
        ret = w @ mu
        vol = float(np.sqrt(w @ cov @ w + 1e-10))
        return -ret / vol

    try:
        result = minimize(
            neg_sharpe,
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if result.success:
            w = np.maximum(result.x, 0.0)
            w /= w.sum()
            weights = {a: round(float(w[i]), 4) for i, a in enumerate(asset_list)}
            for a in assets:
                weights.setdefault(a, 0.0)
            return weights
    except Exception:
        pass
    return eq


class StageRefusalError(RuntimeError):
    """
    Raised when an LLM declines to respond to a stage prompt due to content
    safety filters or policy restrictions.  The pipeline catches this and
    substitutes a neutral fallback output, recording the refusal in
    EpisodeResult.refused_stages with a forced stage score of 0.0.
    """


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

_REFUSAL_PATTERNS: list[re.Pattern] = [
    # Chinese refusals (common content-safety responses)
    re.compile(r"无法给到", re.IGNORECASE),
    re.compile(r"无法回答", re.IGNORECASE),
    re.compile(r"无法提供", re.IGNORECASE),
    re.compile(r"无法处理", re.IGNORECASE),
    re.compile(r"无法.*内容", re.IGNORECASE),
    re.compile(r"不(能|可以).{0,10}(回答|提供|处理|回复)", re.IGNORECASE),
    re.compile(r"对不起.*无法", re.IGNORECASE),
    re.compile(r"抱歉.*无法", re.IGNORECASE),
    # English refusals
    re.compile(r"i('m|\s+am)?\s+(not\s+able|unable)\s+to\s+(provide|answer|respond|assist)", re.IGNORECASE),
    re.compile(r"i\s+can'?t\s+(provide|answer|respond|give|assist)", re.IGNORECASE),
    re.compile(r"i\s+cannot\s+(provide|answer|respond|give|assist)", re.IGNORECASE),
    re.compile(r"i\s+(?:must\s+)?decline", re.IGNORECASE),
]


def _is_refusal(raw: str) -> bool:
    """
    Return True if the model output looks like a content-safety refusal rather
    than a legitimate (possibly malformed) JSON response.

    Two conditions trigger detection:
    1. A known refusal keyword pattern is matched, OR
    2. The response is very short (< 120 chars) and contains no `{` character,
       making it structurally impossible to be valid JSON.
    """
    if any(p.search(raw) for p in _REFUSAL_PATTERNS):
        return True
    # Heuristic: too short to be JSON and contains no opening brace
    if len(raw.strip()) < 120 and "{" not in raw:
        return True
    return False


def _call_with_json_retry(adapter, prompt: str, use_tools: bool, stage_name: str) -> tuple:
    """
    Call the LLM and parse a JSON object, retrying on parse failure.

    Only LLM-format errors trigger retries. Network / API / HTTP errors
    propagate immediately so the experiment fails loudly.
    Raises StageRefusalError (a subclass of RuntimeError) if the model output
    matches a content-safety refusal pattern — callers should catch this
    separately from generic RuntimeError to apply fallback logic.
    """
    last_err: Optional[Exception] = None
    last_raw: str = ""
    for attempt in range(_JSON_RETRY_LIMIT + 1):
        if attempt == 0:
            full_prompt = prompt
        else:
            full_prompt = prompt + build_format_correction_suffix(str(last_err))
        if use_tools:
            raw = adapter.complete_with_tools(full_prompt, get_tools())
        else:
            raw = adapter.complete(full_prompt)
        last_raw = raw
        # Detect content-safety refusals immediately — no point retrying
        if _is_refusal(raw):
            raise StageRefusalError(
                f"{stage_name} model refused to respond. "
                f"Last raw output: {raw!r}"
            )
        try:
            return _extract_json(raw), raw
        except (json.JSONDecodeError, ValueError) as exc:
            last_err = exc
            continue
    raise RuntimeError(
        f"{stage_name} JSON parse failed after {_JSON_RETRY_LIMIT + 1} attempts. "
        f"Last error: {last_err}. Last raw output: {last_raw!r}"
    )


def _format_price_context(snapshot: MarketSnapshot, max_assets: int = 8) -> str:
    """
    Format price/return data from a snapshot into a compact string for prompt injection.
    Limits to max_assets to keep prompts within token budget.
    """
    lines = []
    assets = list(snapshot.return_data.keys())[:max_assets]
    for asset in assets:
        r = snapshot.return_data[asset].dropna()
        if r.empty:
            continue
        trailing_ret = float((1 + r).prod() - 1)
        vol = float(r.std() * (252 ** 0.5))
        last_price = float(snapshot.price_data[asset].dropna().iloc[-1]) \
            if asset in snapshot.price_data and not snapshot.price_data[asset].empty else 0.0
        lines.append(
            f"  {asset}: price={last_price:.2f}, trailing_return={trailing_ret:+.2%}, "
            f"ann_vol={vol:.2%}"
        )
    macro = snapshot.macro_data
    macro_str = ", ".join(f"{k}={v:.3f}" for k, v in list(macro.items())[:5])
    return "\n".join(lines) + f"\nMacro: {macro_str}"

def _format_macro_context(snapshot: MarketSnapshot) -> str:
    """Format all macro indicators as a human-readable section for LLM prompts."""
    if not snapshot.macro_data:
        return ""
    labels = {
        "fed_funds_rate": "Fed Funds Rate (%)",
        "cpi_yoy":        "CPI YoY index",
        "unemployment":   "Unemployment (%)",
        "gdp_growth_qoq": "GDP growth QoQ (%)",
        "t10y2y_spread":  "10Y-2Y spread (pp)",
        "t10y3m_spread":  "10Y-3M spread (pp)",
        "breakeven_10y":  "10Y breakeven inflation (%)",
        "hy_oas":         "HY OAS (pp)",
        "ig_oas":         "IG OAS (pp)",
        "ted_spread":     "TED spread (pp)",
        "mortgage_30y":   "30Y mortgage rate (%)",
        "vix":            "VIX",
    }
    lines = ["MACRO INDICATORS:"]
    for key, label in labels.items():
        val = snapshot.macro_data.get(key)
        if val is not None and val != 0.0:
            lines.append(f"  {label}: {val:.2f}")
    # Include any unlabeled indicators
    known = set(labels.keys())
    for key, val in snapshot.macro_data.items():
        if key not in known and val is not None and val != 0.0:
            lines.append(f"  {key}: {val:.4f}")
    return "\n".join(lines)


def _format_correlation(snapshot: MarketSnapshot, max_assets: int = 6) -> str:
    """Format the pairwise return correlation matrix as a compact table string.

    When snapshot.asset_class_map is set, the table is augmented with two extra
    blocks that surface the *intra-class* (within e.g. equities) and *inter-class*
    (e.g. equities vs bonds) correlation structure separately, since those two
    layers have different portfolio implications (concentration vs hedging).
    """
    if snapshot.correlation_matrix is None or snapshot.correlation_matrix.empty:
        return ""
    cm = snapshot.correlation_matrix
    assets = list(cm.columns[:max_assets])
    cm_view = cm.loc[assets, assets]
    lines = ["PAIRWISE RETURN CORRELATIONS (trailing window):"]
    header = "         " + "".join(f"{a:>10}" for a in assets)
    lines.append(header)
    for row_asset in assets:
        row = f"{row_asset:<9}" + "".join(
            f"{cm_view.loc[row_asset, col]:>10.2f}" for col in assets
        )
        lines.append(row)

    if snapshot.asset_class_map:
        intra = snapshot.get_intra_class_correlation()
        if intra:
            lines.append("")
            lines.append("INTRA-CLASS AVERAGE CORRELATION (concentration risk per class):")
            for ac, sub in sorted(intra.items()):
                vals = sub.values[~np.eye(len(sub), dtype=bool)]
                if len(vals):
                    lines.append(f"  {ac:<14}: mean={float(np.mean(vals)):+.2f}  n={len(sub)}")
        inter = snapshot.get_inter_class_correlation()
        if not inter.empty:
            lines.append("")
            lines.append("INTER-CLASS CORRELATION (hedging/diversification across classes):")
            classes = list(inter.columns)
            lines.append("         " + "".join(f"{c[:8]:>10}" for c in classes))
            for ci in classes:
                row = f"{ci[:8]:<9}" + "".join(
                    f"{inter.loc[ci, cj]:>10.2f}" for cj in classes
                )
                lines.append(row)
    return "\n".join(lines)




class S1MarketInterpretation(PipelineStage):
    """
    Stage 1: Market Information Interpretation.

    Ground truth: rule-based sentiment scoring using trailing returns.
      - asset_view = sign(trailing_return) * min(|trailing_return| / 0.10, 1.0)
      - detected_regime = from snapshot.market_regime

    Scoring: mean absolute error of asset views, normalized to [0, 1].
    """

    def __init__(self, adapter: AgentAdapter = None, use_tools: bool = False):
        self.adapter = adapter or MockAgentAdapter()
        self.use_tools = use_tools

    @property
    def stage_id(self) -> StageID:
        return StageID.S1_MARKET_INTERPRETATION

    def compute_ground_truth(self, snapshot: MarketSnapshot) -> S1Output:
        views = {}
        for asset, returns in snapshot.return_data.items():
            r = returns.dropna()
            if r.empty:
                views[asset] = 0.0
                continue
            trailing = float((1 + r).prod() - 1)   # Cumulative return over window
            # Normalize: ±10% trailing return → ±1.0 view
            views[asset] = float(np.clip(trailing / 0.10, -1.0, 1.0))

        regime = snapshot.market_regime or "sideways"
        return S1Output(
            asset_views=views,
            detected_regime=regime,
            confidence=0.8,
            macro_summary=f"Macro snapshot at {snapshot.decision_date}: {snapshot.macro_data}",
        )

    def run(self, snapshot: MarketSnapshot, prior_output=None) -> S1Output:
        gt = self.compute_ground_truth(snapshot)
        if isinstance(self.adapter, MockAgentAdapter):
            # Add Gaussian noise to ground-truth views
            noisy_views = {
                a: float(np.clip(v + self.adapter._rng.normal(0, self.adapter.noise_level), -1, 1))
                for a, v in gt.asset_views.items()
            }
            return S1Output(
                asset_views=noisy_views,
                detected_regime=gt.detected_regime,
                confidence=max(0.0, gt.confidence - self.adapter.noise_level * 0.3),
                macro_summary=gt.macro_summary,
                raw_llm_output=self.adapter.complete(""),
            )

        # ----------------------------------------------------------------
        # Real LLM path
        # ----------------------------------------------------------------
        assets = list(snapshot.return_data.keys())
        context = _format_price_context(snapshot)
        macro_block = _format_macro_context(snapshot)
        corr_block = _format_correlation(snapshot)
        trailing_days = len(next(iter(snapshot.return_data.values()), pd.Series()))
        prompt = build_s1_prompt(
            snapshot=snapshot,
            assets=assets,
            price_context=context,
            macro_block=macro_block,
            corr_block=corr_block,
            trailing_days=trailing_days,
        )

        self._last_prompt = prompt
        try:
            parsed, raw = _call_with_json_retry(self.adapter, prompt, self.use_tools, "S1")
        except StageRefusalError as exc:
            return S1Output(
                asset_views={a: 0.0 for a in assets},
                detected_regime="sideways",
                confidence=0.0,
                macro_summary="[refused]",
                raw_llm_output=str(exc),
                refused=True,
            )
        try:
            views_src = parsed.get("asset_views", parsed)
            return S1Output(
                asset_views={a: float(np.clip(views_src.get(a, 0.0), -1, 1))
                             for a in assets},
                detected_regime=str(parsed.get("detected_regime", gt.detected_regime)),
                confidence=float(np.clip(parsed.get("confidence", 0.5), 0, 1)),
                macro_summary=str(parsed.get("macro_summary", "")),
                raw_llm_output=raw,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"S1 parsed JSON missing required fields: {exc}. Parsed: {parsed!r}"
            ) from exc

    def score(self, actual: S1Output, ground_truth: S1Output) -> float:
        """Score = 1 - mean absolute error of asset views (views in [-1, 1] range)."""
        if not ground_truth.asset_views:
            return 1.0
        errors = [
            abs(actual.asset_views.get(a, 0.0) - v)
            for a, v in ground_truth.asset_views.items()
        ]
        mae = np.mean(errors) / 2.0   # Normalize by max possible error (2.0)
        return float(np.clip(1.0 - mae, 0.0, 1.0))


# ---------------------------------------------------------------------------
# S2 – Signal Generation
# ---------------------------------------------------------------------------

class S2SignalGeneration(PipelineStage):
    """
    Stage 2: Investment Signal Generation.

    Ground truth: convert asset views from S1 ground truth to discrete signals.
      view >  0.2 → "buy"
      view < -0.2 → "sell"
      otherwise   → "hold"

    Scoring: fraction of assets with correct signal direction.
    """

    def __init__(self, adapter: AgentAdapter = None, use_tools: bool = False):
        self.adapter = adapter or MockAgentAdapter()
        self.use_tools = use_tools

    @property
    def stage_id(self) -> StageID:
        return StageID.S2_SIGNAL_GENERATION

    def compute_ground_truth(self, snapshot: MarketSnapshot) -> S2Output:
        # Derive from S1 ground truth
        s1_stage = S1MarketInterpretation()
        s1_gt = s1_stage.compute_ground_truth(snapshot)
        return self._views_to_signals(s1_gt)

    def _views_to_signals(self, s1: S1Output) -> S2Output:
        signals, strengths = {}, {}
        for asset, view in s1.asset_views.items():
            if view > 0.2:
                signals[asset] = "buy"
            elif view < -0.2:
                signals[asset] = "sell"
            else:
                signals[asset] = "hold"
            strengths[asset] = float(abs(view))
        return S2Output(signals=signals, strengths=strengths)

    def run(self, snapshot: MarketSnapshot, prior_output: S1Output = None) -> S2Output:
        s1 = prior_output if prior_output is not None else S1MarketInterpretation(self.adapter).run(snapshot)
        if isinstance(self.adapter, MockAgentAdapter):
            gt = self._views_to_signals(S1MarketInterpretation().compute_ground_truth(snapshot))
            # Randomly flip some signals based on noise_level
            noisy_signals = {}
            for asset, sig in gt.signals.items():
                if self.adapter._rng.random() < self.adapter.noise_level:
                    noisy_signals[asset] = self.adapter._rng.choice(["buy", "hold", "sell"])
                else:
                    noisy_signals[asset] = sig
            return S2Output(signals=noisy_signals, strengths=gt.strengths, raw_llm_output=self.adapter.complete(""))

        # ----------------------------------------------------------------
        # Real LLM path
        # ----------------------------------------------------------------
        assets = list(s1.asset_views.keys())
        prompt = build_s2_prompt(snapshot=snapshot, s1=s1, assets=assets)

        self._last_prompt = prompt
        try:
            parsed, raw = _call_with_json_retry(self.adapter, prompt, self.use_tools, "S2")
        except StageRefusalError as exc:
            return S2Output(
                signals={a: "hold" for a in assets},
                strengths={a: 0.5 for a in assets},
                reasoning="[refused]",
                raw_llm_output=str(exc),
                refused=True,
            )
        try:
            valid_signals = {"buy", "hold", "sell"}
            signals = {
                a: parsed["signals"].get(a, "hold")
                if parsed["signals"].get(a, "hold") in valid_signals else "hold"
                for a in assets
            }
            strengths = {
                a: float(np.clip(parsed.get("strengths", {}).get(a, 0.5), 0, 1))
                for a in assets
            }
            return S2Output(
                signals=signals,
                strengths=strengths,
                reasoning=str(parsed.get("reasoning", "")),
                raw_llm_output=raw,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"S2 parsed JSON missing required fields: {exc}. Parsed: {parsed!r}"
            ) from exc

    def score(self, actual: S2Output, ground_truth: S2Output) -> float:
        """Score = fraction of assets with correct signal."""
        if not ground_truth.signals:
            return 1.0
        correct = sum(
            1 for a, sig in ground_truth.signals.items()
            if actual.signals.get(a) == sig
        )
        return float(correct / len(ground_truth.signals))


# ---------------------------------------------------------------------------
# S3 – Weight Optimization
# ---------------------------------------------------------------------------

class S3WeightOptimization(PipelineStage):
    """
    Stage 3: Portfolio Weight Optimization.

    Ground truth: equal-weight among "buy" signals, 0 for "sell".
    Scores using weight MAE normalized to [0, 1].
    """

    def __init__(self, adapter: AgentAdapter = None, use_tools: bool = False):
        self.adapter = adapter or MockAgentAdapter()
        self.use_tools = use_tools
        self._last_snapshot: Optional[MarketSnapshot] = None

    @property
    def stage_id(self) -> StageID:
        return StageID.S3_WEIGHT_OPTIMIZATION

    def compute_ground_truth(self, snapshot: MarketSnapshot) -> S3Output:
        s2_gt = S2SignalGeneration().compute_ground_truth(snapshot)
        if snapshot.future_return_data:
            buy_assets = [a for a, sig in s2_gt.signals.items() if sig == "buy"]
            if not buy_assets:
                buy_assets = list(s2_gt.signals.keys())
            weights = _max_sharpe_weights(buy_assets, snapshot.future_return_data)
            for a in s2_gt.signals:
                weights.setdefault(a, 0.0)
            return S3Output(weights=weights)
        return self._signals_to_weights(s2_gt, snapshot)

    def _signals_to_weights(self, s2: S2Output, snapshot: MarketSnapshot) -> S3Output:
        buy_assets = [a for a, sig in s2.signals.items() if sig == "buy"]
        if not buy_assets:
            # All hold/sell: equal weight across all assets
            buy_assets = list(s2.signals.keys())
        n = len(buy_assets)
        weights = {a: round(1.0 / n, 4) for a in buy_assets}
        # Assets with sell/hold get 0 weight (unless all are hold/sell → equal)
        for a in s2.signals:
            if a not in weights:
                weights[a] = 0.0
        return S3Output(weights=weights)

    def run(self, snapshot: MarketSnapshot, prior_output: S2Output = None) -> S3Output:
        self._last_snapshot = snapshot  # cache for correlation-awareness scoring
        s2 = prior_output if prior_output is not None else S2SignalGeneration(self.adapter).run(snapshot)
        if isinstance(self.adapter, MockAgentAdapter):
            gt = self._signals_to_weights(
                S2SignalGeneration().compute_ground_truth(snapshot), snapshot
            )
            # Add noise: perturb weights and renormalize
            noisy = {
                a: max(0, w + float(self.adapter._rng.normal(0, self.adapter.noise_level * 0.2)))
                for a, w in gt.weights.items()
            }
            total = sum(noisy.values())
            if total > 0:
                noisy = {a: round(w / total, 4) for a, w in noisy.items()}
            return S3Output(weights=noisy, raw_llm_output=self.adapter.complete(""))

        # ----------------------------------------------------------------
        # Real LLM path
        # ----------------------------------------------------------------
        assets = list(s2.signals.keys())
        corr_block = _format_correlation(snapshot)
        prompt = build_s3_prompt(
            snapshot=snapshot, s2=s2, assets=assets, corr_block=corr_block
        )

        self._last_prompt = prompt
        try:
            parsed, raw = _call_with_json_retry(self.adapter, prompt, self.use_tools, "S3")
        except StageRefusalError as exc:
            # Hold current weights — no rebalancing when the model refuses
            current = dict(snapshot.current_weights) if snapshot.current_weights else {}
            if not current:
                n = len(assets)
                current = {a: round(1.0 / n, 4) for a in assets}
            return S3Output(
                weights=current,
                expected_return=0.0,
                expected_vol=0.0,
                sharpe_estimate=0.0,
                raw_llm_output=str(exc),
                refused=True,
            )
        try:
            raw_weights = {a: float(parsed["weights"].get(a, 0.0)) for a in assets}
            # Clip negatives and renormalize to enforce constraints
            raw_weights = {a: max(0.0, w) for a, w in raw_weights.items()}
            total = sum(raw_weights.values())
            if total <= 0:
                raise ValueError("All weights are zero or negative")
            weights = {a: round(w / total, 4) for a, w in raw_weights.items()}
            return S3Output(
                weights=weights,
                expected_return=float(parsed.get("expected_return", 0.0)),
                expected_vol=float(parsed.get("expected_vol", 0.0)),
                sharpe_estimate=float(parsed.get("sharpe_estimate", 0.0)),
                raw_llm_output=raw,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"S3 parsed JSON missing required fields: {exc}. Parsed: {parsed!r}"
            ) from exc

    def score(self, actual: S3Output, ground_truth: S3Output) -> float:
        """
        Composite score = 70% weight accuracy + 30% correlation awareness.

        Weight accuracy: 1 - weight_mae / 2 (normalized, max error = 2.0)

        Correlation awareness (30%) is split into two components when an
        asset_class_map is available on the snapshot:
          - 15%: intra-class concentration penalty — high weight on tickers
                 inside a class with high mutual correlation is penalized
                 (concentration in correlated names is not real diversification).
          - 15%: inter-class hedging credit — distributing weight across asset
                 classes whose returns are weakly / negatively correlated is
                 rewarded (cross-class diversification).
        When no asset_class_map is available, the full 30% falls back to the
        original variance-ratio score.

        Falls back to 100% weight accuracy if the snapshot is not available
        or fewer than 2 assets have sufficient return data.
        """
        from ..metrics.allocation_metrics import weight_mae
        mae = weight_mae(actual.weights, ground_truth.weights)
        accuracy_score = float(np.clip(1.0 - mae / 2.0, 0.0, 1.0))

        snap = getattr(self, "_last_snapshot", None)
        if snap is None:
            return accuracy_score

        if snap.asset_class_map:
            intra = self._intra_class_diversification_score(actual.weights, snap)
            inter = self._inter_class_hedging_score(actual.weights, snap)
            return float(0.70 * accuracy_score + 0.15 * intra + 0.15 * inter)

        corr_score = self._correlation_awareness_score(
            actual.weights, ground_truth.weights, snap
        )
        return float(0.70 * accuracy_score + 0.30 * corr_score)

    def _intra_class_diversification_score(
        self,
        actual_weights: dict,
        snapshot: "MarketSnapshot",
    ) -> float:
        """
        Penalize concentrating weight inside a single asset class with high
        internal correlation.

        For each class with >= 2 represented assets:
            class_weight = sum of actual weights on that class
            avg_intra_corr = average off-diagonal correlation inside the class
            penalty_c = class_weight * max(avg_intra_corr, 0)
        Score = 1 - sum(penalty_c), clipped to [0, 1].
        Returns 1.0 when the asset_class_map yields no usable groups.
        """
        intra = snapshot.get_intra_class_correlation()
        if not intra:
            return 1.0
        penalty = 0.0
        for sub in intra.values():
            members = list(sub.columns)
            class_w = sum(float(actual_weights.get(a, 0.0)) for a in members)
            if class_w <= 0:
                continue
            vals = sub.values[~np.eye(len(sub), dtype=bool)]
            if not len(vals):
                continue
            avg_corr = float(np.mean(vals))
            penalty += class_w * max(avg_corr, 0.0)
        return float(np.clip(1.0 - penalty, 0.0, 1.0))

    def _inter_class_hedging_score(
        self,
        actual_weights: dict,
        snapshot: "MarketSnapshot",
    ) -> float:
        """
        Reward spreading weight across asset classes that are weakly / negatively
        correlated with each other.

        Aggregate weights per class, then compute the weighted average of the
        off-diagonal entries of the inter-class correlation matrix:
            avg_xclass_corr = sum_{i!=j} w_i * w_j * corr(class_i, class_j)
                              / sum_{i!=j} w_i * w_j
        Score = clip((1 - avg_xclass_corr) / 2, 0, 1)
        (correlation +1 → 0, 0 → 0.5, -1 → 1.0).
        Returns 1.0 when fewer than two classes carry weight.
        """
        inter = snapshot.get_inter_class_correlation()
        if inter.empty:
            return 1.0
        class_weights: dict[str, float] = {}
        for a, w in actual_weights.items():
            ac = snapshot.asset_class_map.get(a) if snapshot.asset_class_map else None
            if ac is None or ac not in inter.columns:
                continue
            class_weights[ac] = class_weights.get(ac, 0.0) + float(w)
        active = [c for c, w in class_weights.items() if w > 0]
        if len(active) < 2:
            return 1.0
        num = 0.0
        den = 0.0
        for ci in active:
            for cj in active:
                if ci == cj:
                    continue
                w_pair = class_weights[ci] * class_weights[cj]
                num += w_pair * float(inter.loc[ci, cj])
                den += w_pair
        if den <= 0:
            return 1.0
        avg_corr = num / den
        return float(np.clip((1.0 - avg_corr) / 2.0, 0.0, 1.0))

    def _correlation_awareness_score(
        self,
        actual_weights: dict,
        gt_weights: dict,
        snapshot: "MarketSnapshot",
    ) -> float:
        """
        Score how well the weights reflect the cross-asset correlation structure.

        Computes realized portfolio variance for both actual and ground-truth
        weights using the empirical covariance matrix from snapshot.return_data.
        A lower-variance actual portfolio (vs. gt) scores higher, since it
        indicates the model found a more diversified allocation.

        Returns a score in [0, 1].
        """
        df = pd.DataFrame({a: s for a, s in snapshot.return_data.items() if not s.empty})
        if df.shape[1] < 2 or len(df) < 10:
            return 1.0  # Insufficient data, skip this component

        cov = df.cov() * 252  # Annualized covariance
        assets = list(cov.columns)

        def port_var(w_dict):
            w = np.array([w_dict.get(a, 0.0) for a in assets])
            return float(w @ cov.values @ w)

        var_actual = port_var(actual_weights)
        var_gt     = port_var(gt_weights)

        if var_gt <= 0:
            return 1.0

        # Reward lower variance: score = 1 if actual ≤ gt, penalizes proportionally if higher
        ratio = var_actual / var_gt
        return float(np.clip(2.0 - ratio, 0.0, 1.0))


# ---------------------------------------------------------------------------
# S4 – Execution Simulation
# ---------------------------------------------------------------------------

class S4ExecutionSimulation(PipelineStage):
    """
    Stage 4: Trade Execution Simulation.

    Simulates a realistic execution layer with slippage and commission.
    The 'agent' here is deterministic (no LLM needed) — it is included
    as a pipeline stage to model execution costs affecting the final portfolio.

    Slippage model: linear market impact proportional to trade size.
    Commission: fixed rate per trade value.
    """

    SLIPPAGE_RATE = 0.0010   # 10 bps
    COMMISSION_RATE = 0.0005  # 5 bps

    def __init__(self, adapter: AgentAdapter = None):
        self.adapter = adapter or MockAgentAdapter()

    @property
    def stage_id(self) -> StageID:
        return StageID.S4_EXECUTION_SIMULATION

    def compute_ground_truth(self, snapshot: MarketSnapshot) -> S4Output:
        # Ground truth: execute S3 ground-truth weights with zero slippage
        s3_gt = S3WeightOptimization().compute_ground_truth(snapshot)
        return self._execute(s3_gt, snapshot, slippage_rate=0.0)

    def _execute(
        self, s3: S3Output, snapshot: MarketSnapshot, slippage_rate: float = SLIPPAGE_RATE
    ) -> S4Output:
        current_w = snapshot.current_weights
        target_w = s3.weights
        nav = snapshot.portfolio_value

        orders = []
        executed = dict(target_w)
        total_cost = 0.0
        total_turnover = 0.0

        all_assets = set(current_w) | set(target_w)
        for asset in all_assets:
            curr = current_w.get(asset, 0.0)
            targ = target_w.get(asset, 0.0)
            delta = targ - curr
            if abs(delta) < 1e-6:
                continue

            direction = "buy" if delta > 0 else "sell"
            trade_value = abs(delta) * nav

            # Get approximate current price from price_data
            price_series = snapshot.price_data.get(asset)
            price = float(price_series.iloc[-1]) if (price_series is not None and not price_series.empty) else 100.0

            slippage = slippage_rate * (1 if direction == "buy" else -1)
            exec_price = price * (1 + slippage)
            commission = trade_value * self.COMMISSION_RATE
            total_cost += commission + trade_value * abs(slippage)
            total_turnover += abs(delta)

            orders.append(TradeOrder(
                asset=asset, direction=direction, quantity=trade_value,
                price=exec_price, slippage=slippage, commission=commission,
            ))

        # Adjust executed weights for cost drag
        cost_drag = total_cost / nav
        cash_key = next((a for a in executed if "BIL" in a or "cash" in a.lower()), None)
        if cash_key:
            executed[cash_key] = max(0, executed[cash_key] - cost_drag)

        return S4Output(
            orders=orders,
            executed_weights={a: round(w, 4) for a, w in executed.items()},
            total_cost=round(total_cost, 4),
            turnover=round(total_turnover, 4),
        )

    def run(self, snapshot: MarketSnapshot, prior_output: S3Output = None) -> S4Output:
        s3 = prior_output if prior_output is not None else S3WeightOptimization(self.adapter).run(snapshot)
        return self._execute(s3, snapshot)

    def score(self, actual: S4Output, ground_truth: S4Output) -> float:
        """Score based on how closely executed weights match ground truth."""
        from ..metrics.allocation_metrics import weight_mae
        mae = weight_mae(actual.executed_weights, ground_truth.executed_weights)
        return float(np.clip(1.0 - mae / 2.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# S5 – Risk Monitoring
# ---------------------------------------------------------------------------

class S5RiskMonitoring(PipelineStage):
    """
    Stage 5: Risk Monitoring.

    Computes portfolio-level risk metrics and triggers alerts when thresholds
    are breached. Thresholds are conservative defaults; adjust in QAConfig.
    """

    VAR_LIMIT = -0.02          # 1-day VaR(95%) threshold: -2%
    DRAWDOWN_LIMIT = -0.10     # Drawdown limit: -10%
    DRIFT_LIMIT = 0.05         # Max weight drift: 5%

    def __init__(self, adapter: AgentAdapter = None):
        self.adapter = adapter or MockAgentAdapter()

    @property
    def stage_id(self) -> StageID:
        return StageID.S5_RISK_MONITORING

    def compute_ground_truth(self, snapshot: MarketSnapshot) -> S5Output:
        # Use S4 ground-truth weights to compute portfolio returns
        s4_gt = S4ExecutionSimulation().compute_ground_truth(snapshot)
        return self._monitor(s4_gt.executed_weights, snapshot)

    def _monitor(self, weights: dict[str, float], snapshot: MarketSnapshot) -> S5Output:
        # Compute weighted portfolio returns
        port_returns = pd.Series(dtype=float)
        for asset, w in weights.items():
            r = snapshot.return_data.get(asset)
            if r is None or r.empty:
                continue
            port_returns = port_returns.add(r.dropna() * w, fill_value=0)

        cfg = MetricsConfig(var_confidence=0.95)
        port_var = float(var(port_returns, cfg)) if len(port_returns) > 10 else 0.0
        port_dd = float(max_drawdown(port_returns)) if len(port_returns) > 10 else 0.0

        # Weight drift vs equal-weight target
        n = max(len(weights), 1)
        target_w = 1.0 / n
        drift = float(max(abs(w - target_w) for w in weights.values())) if weights else 0.0

        alerts = []
        if port_var < self.VAR_LIMIT:
            alerts.append(RiskAlert("var_breach", port_var, self.VAR_LIMIT, "warning", "reduce"))
        if port_dd < self.DRAWDOWN_LIMIT:
            alerts.append(RiskAlert("drawdown", port_dd, self.DRAWDOWN_LIMIT, "critical", "rebalance"))
        if drift > self.DRIFT_LIMIT:
            alerts.append(RiskAlert("weight_drift", drift, self.DRIFT_LIMIT, "warning", "rebalance"))

        rebalance_needed = any(a.action == "rebalance" for a in alerts)

        return S5Output(
            portfolio_var=round(port_var, 6),
            portfolio_drawdown=round(port_dd, 6),
            weight_drift=round(drift, 6),
            alerts=alerts,
            rebalance_needed=rebalance_needed,
        )

    def run(self, snapshot: MarketSnapshot, prior_output: S4Output = None) -> S5Output:
        weights = prior_output.executed_weights if prior_output else {}
        if not weights and snapshot.current_weights:
            weights = snapshot.current_weights
        return self._monitor(weights, snapshot)

    def score(self, actual: S5Output, ground_truth: S5Output) -> float:
        """
        Score based on:
          - Correct rebalance_needed decision (50%)
          - VaR / drawdown accuracy (50%)
        """
        decision_score = 1.0 if actual.rebalance_needed == ground_truth.rebalance_needed else 0.0

        var_err = abs(actual.portfolio_var - ground_truth.portfolio_var) / max(abs(ground_truth.portfolio_var), 1e-6)
        dd_err = abs(actual.portfolio_drawdown - ground_truth.portfolio_drawdown) / max(abs(ground_truth.portfolio_drawdown), 1e-6)
        numeric_score = float(np.clip(1.0 - (var_err + dd_err) / 2.0, 0.0, 1.0))

        return float(0.5 * decision_score + 0.5 * numeric_score)
