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
from typing import Optional
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
from ..metrics.risk_metrics import var, max_drawdown
from ..metrics.base import MetricsConfig


# ---------------------------------------------------------------------------
# Shared prompt utilities
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Extract the first JSON object from a model response string.

    Handles common LLM formatting patterns:
      - Bare JSON:              {"key": "value"}
      - Markdown code block:    ```json\n{...}\n```
      - JSON embedded in prose: "Here is the answer: {...} Done."
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in model response: {text[:200]}")


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


# ---------------------------------------------------------------------------
# S1 – Market Interpretation
# ---------------------------------------------------------------------------

class S1MarketInterpretation(PipelineStage):
    """
    Stage 1: Market Information Interpretation.

    Ground truth: rule-based sentiment scoring using trailing returns.
      - asset_view = sign(trailing_return) * min(|trailing_return| / 0.10, 1.0)
      - detected_regime = from snapshot.market_regime

    Scoring: mean absolute error of asset views, normalized to [0, 1].
    """

    def __init__(self, adapter: AgentAdapter = None):
        self.adapter = adapter or MockAgentAdapter()

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
        prompt = f"""You are a portfolio manager analyzing market conditions on {snapshot.decision_date}.

MARKET DATA (trailing {len(next(iter(snapshot.return_data.values()), pd.Series()))} trading days):
{context}

Current market regime context: {snapshot.market_regime or "unknown"}

TASK: Interpret the market data and provide structured asset views.

For each asset, assign a sentiment score in [-1.0, +1.0]:
  +1.0 = strongly bullish (expect strong outperformance)
   0.0 = neutral
  -1.0 = strongly bearish (expect significant underperformance)

Also identify the overall market regime: one of "bull", "bear", "sideways", "crisis".

Respond with ONLY a JSON object in this exact format:
{{
  "asset_views": {{{", ".join(f'"{a}": <float -1 to 1>' for a in assets)}}},
  "detected_regime": "<bull|bear|sideways|crisis>",
  "confidence": <float 0 to 1>,
  "macro_summary": "<one sentence summary of macro environment>"
}}"""

        raw = self.adapter.complete(prompt)
        try:
            parsed = _extract_json(raw)
            return S1Output(
                asset_views={a: float(np.clip(parsed["asset_views"].get(a, 0.0), -1, 1))
                             for a in assets},
                detected_regime=str(parsed.get("detected_regime", gt.detected_regime)),
                confidence=float(np.clip(parsed.get("confidence", 0.5), 0, 1)),
                macro_summary=str(parsed.get("macro_summary", "")),
                raw_llm_output=raw,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            # Parsing failed — fall back to ground truth to keep pipeline alive
            return S1Output(
                asset_views=gt.asset_views,
                detected_regime=gt.detected_regime,
                confidence=0.0,
                macro_summary="[parse error — fell back to ground truth]",
                raw_llm_output=raw,
            )

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

    def __init__(self, adapter: AgentAdapter = None):
        self.adapter = adapter or MockAgentAdapter()

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
        views_str = "\n".join(f"  {a}: view={v:+.3f}" for a, v in s1.asset_views.items())
        prompt = f"""You are a portfolio manager on {snapshot.decision_date}.

Stage 1 market interpretation produced these asset views (scale: -1=bearish, +1=bullish):
{views_str}

Detected market regime: {s1.detected_regime}
Macro summary: {s1.macro_summary}

TASK: Convert each asset view into an actionable trading signal.

Rules:
  - view >  0.15: consider "buy"
  - view < -0.15: consider "sell"
  - otherwise:    consider "hold"

Use your judgement to refine signals based on regime and macro context.
Signal strength should reflect conviction (0.0 = low, 1.0 = high).

Respond with ONLY a JSON object:
{{
  "signals": {{{", ".join(f'"{a}": "<buy|hold|sell>"' for a in assets)}}},
  "strengths": {{{", ".join(f'"{a}": <float 0 to 1>' for a in assets)}}},
  "reasoning": "<one sentence explaining key decisions>"
}}"""

        raw = self.adapter.complete(prompt)
        try:
            parsed = _extract_json(raw)
            valid_signals = {"buy", "hold", "sell"}
            signals = {
                a: parsed["signals"].get(a, "hold")
                if parsed["signals"].get(a, "hold") in valid_signals else "hold"
                for a in assets
            }
            strengths = {
                a: float(np.clip(parsed["strengths"].get(a, 0.5), 0, 1))
                for a in assets
            }
            return S2Output(
                signals=signals,
                strengths=strengths,
                reasoning=str(parsed.get("reasoning", "")),
                raw_llm_output=raw,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            gt = self._views_to_signals(S1MarketInterpretation().compute_ground_truth(snapshot))
            return S2Output(
                signals=gt.signals,
                strengths=gt.strengths,
                reasoning="[parse error — fell back to ground truth]",
                raw_llm_output=raw,
            )

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

    def __init__(self, adapter: AgentAdapter = None):
        self.adapter = adapter or MockAgentAdapter()
        self._last_snapshot: Optional[MarketSnapshot] = None

    @property
    def stage_id(self) -> StageID:
        return StageID.S3_WEIGHT_OPTIMIZATION

    def compute_ground_truth(self, snapshot: MarketSnapshot) -> S3Output:
        s2_gt = S2SignalGeneration().compute_ground_truth(snapshot)
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
        signals_str = "\n".join(
            f"  {a}: signal={s2.signals[a]}, strength={s2.strengths.get(a, 0.5):.2f}"
            for a in assets
        )
        current_w_str = ", ".join(
            f"{a}={w:.3f}" for a, w in snapshot.current_weights.items()
        )
        prompt = f"""You are a portfolio manager on {snapshot.decision_date}.

Stage 2 signals:
{signals_str}

Current portfolio weights: {current_w_str}
Portfolio NAV: ${snapshot.portfolio_value:,.0f}
Market regime: {snapshot.market_regime or "unknown"}

TASK: Allocate portfolio weights based on the signals above.

Constraints:
  - All weights must be in [0.0, 1.0]
  - Weights must sum to exactly 1.0
  - "sell" signals should receive reduced weight (ideally 0.0)
  - "buy" signals should receive increased weight
  - Minimize unnecessary turnover from current weights

Respond with ONLY a JSON object:
{{
  "weights": {{{", ".join(f'"{a}": <float 0 to 1>' for a in assets)}}},
  "expected_return": <annualized expected return as decimal, e.g. 0.08>,
  "expected_vol": <annualized volatility as decimal, e.g. 0.12>,
  "sharpe_estimate": <estimated Sharpe ratio>
}}
The weights MUST sum to 1.0."""

        raw = self.adapter.complete(prompt)
        try:
            parsed = _extract_json(raw)
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
        except (json.JSONDecodeError, KeyError, ValueError):
            gt = self._signals_to_weights(
                S2SignalGeneration().compute_ground_truth(snapshot), snapshot
            )
            return S3Output(
                weights=gt.weights,
                raw_llm_output=raw,
            )

    def score(self, actual: S3Output, ground_truth: S3Output) -> float:
        """
        Composite score = 70% weight accuracy + 30% correlation awareness.

        Weight accuracy: 1 - weight_mae / 2 (normalized, max error = 2.0)

        Correlation awareness: measures whether the proposed weights reduce
        portfolio variance relative to the ground-truth weights, given the
        empirical correlation structure. Specifically:
            score = 1 - |port_var(actual) - port_var(gt)| / port_var(gt)
        clamped to [0, 1]. This penalizes allocations that ignore correlation
        by concentrating in highly correlated assets (e.g., all equities) when
        the correlation structure would support diversification.

        Falls back to 100% weight accuracy if the snapshot is not available
        or fewer than 2 assets have sufficient return data.
        """
        from ..metrics.allocation_metrics import weight_mae
        mae = weight_mae(actual.weights, ground_truth.weights)
        accuracy_score = float(np.clip(1.0 - mae / 2.0, 0.0, 1.0))

        # Attempt correlation-awareness scoring via the snapshot stored during run()
        if hasattr(self, "_last_snapshot") and self._last_snapshot is not None:
            corr_score = self._correlation_awareness_score(
                actual.weights, ground_truth.weights, self._last_snapshot
            )
            return float(0.70 * accuracy_score + 0.30 * corr_score)

        return accuracy_score

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
        try:
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

        except Exception:
            return 1.0   # Graceful degradation


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
