"""
Base classes for the end-to-end agent evaluation pipeline.

The pipeline models the five sequential stages of portfolio management:

    S1: Market Information Interpretation
    S2: Investment Signal Generation
    S3: Portfolio Weight Optimization
    S4: Trade Execution Simulation
    S5: Risk Monitoring

Each stage is an independent unit that:
  - Receives a typed StageInput
  - Calls an LLM (or baseline) to produce output
  - Returns a typed StageOutput
  - Is scored against a ground-truth StageOutput

The EvalPipeline chains the stages sequentially, passing each stage's output
as the next stage's input. This architecture makes it easy to:
  - Run isolated single-stage evaluation (ablation)
  - Run full end-to-end evaluation
  - Inject ground-truth at any stage to measure downstream impact (oracle mode)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Stage identifiers
# ---------------------------------------------------------------------------


class StageID(Enum):
    S1_MARKET_INTERPRETATION = "S1"
    S2_SIGNAL_GENERATION = "S2"
    S3_WEIGHT_OPTIMIZATION = "S3"
    S4_EXECUTION_SIMULATION = "S4"
    S5_RISK_MONITORING = "S5"


# ---------------------------------------------------------------------------
# Stage I/O containers
# ---------------------------------------------------------------------------


@dataclass
class MarketSnapshot:
    """
    Raw market data snapshot fed into the pipeline at each evaluation step.

    This is the only external input; all subsequent stage inputs are derived
    from prior stage outputs plus this snapshot.

    Attributes:
        decision_date:      Date of the portfolio decision.
        price_data:         Dict mapping asset -> pd.Series of recent close prices.
        return_data:        Dict mapping asset -> pd.Series of recent daily returns.
        macro_data:         Dict of macro indicator name -> scalar value.
        news_text:          News / filing text available as of decision_date.
        current_weights:    Dict mapping asset -> current portfolio weight (before rebalancing).
        portfolio_value:    Current portfolio NAV in dollars.
        market_regime:      Optional pre-labeled regime (used for evaluation scoring).
        correlation_matrix: pd.DataFrame (assets × assets) of pairwise return correlations
                            computed from return_data. Populated automatically by
                            get_correlation() on first access.
                            Captures the cross-asset correlation structure that multi-asset
                            portfolio decisions must account for — both cross-class correlations
                            (e.g., equity–bond flight-to-quality dynamics) and intra-class
                            correlations (e.g., sector concentration within equities).
                            Used in S3 CorrelationAwarenessScore and T4/T5/T7 templates.
        asset_class_map:    Optional dict mapping asset id -> asset class string
                            (e.g. {"SPY": "equities", "TLT": "bonds"}). When set,
                            enables get_intra_class_correlation() / get_inter_class_correlation()
                            so downstream code can reason about *intra-class* (e.g. sector
                            concentration within equities) vs *inter-class* (e.g. stock-bond
                            hedging) correlation structure separately.
        future_return_data: Optional dict mapping asset -> pd.Series of forward returns
                            (from decision_date to decision_date + rebalance_period).
                            Used exclusively in S3 ground-truth computation to derive
                            max-Sharpe optimal weights. Must never be shown to the LLM.
    """

    decision_date: date
    price_data: dict[str, pd.Series]
    return_data: dict[str, pd.Series]
    macro_data: dict[str, float] = field(default_factory=dict)
    news_text: str = ""
    current_weights: dict[str, float] = field(default_factory=dict)
    portfolio_value: float = 100_000.0
    market_regime: Optional[str] = None
    correlation_matrix: Optional[pd.DataFrame] = None
    asset_class_map: Optional[dict[str, str]] = None
    future_return_data: Optional[dict[str, "pd.Series"]] = None

    def get_correlation(self) -> pd.DataFrame:
        """
        Return the pairwise Pearson correlation matrix, computing it on first call.

        The matrix is cached in self.correlation_matrix after the first computation.
        Returns an empty DataFrame if fewer than 2 assets have return data.
        """
        if self.correlation_matrix is not None:
            return self.correlation_matrix
        df = pd.DataFrame({a: s for a, s in self.return_data.items() if not s.empty})
        if df.shape[1] < 2:
            return pd.DataFrame()
        self.correlation_matrix = df.corr(method="pearson")
        return self.correlation_matrix

    def get_intra_class_correlation(self) -> dict[str, pd.DataFrame]:
        """
        Return per-asset-class correlation sub-matrices.

        Requires self.asset_class_map. Each key is an asset class with at least
        two member assets present in the correlation matrix; the value is the
        sub-matrix restricted to that class's tickers. Empty dict if the map is
        missing or no class has >= 2 members with data.
        """
        if not self.asset_class_map:
            return {}
        cm = self.get_correlation()
        if cm.empty:
            return {}
        groups: dict[str, list[str]] = {}
        for a in cm.columns:
            ac = self.asset_class_map.get(a)
            if ac:
                groups.setdefault(ac, []).append(a)
        return {ac: cm.loc[members, members] for ac, members in groups.items() if len(members) >= 2}

    def get_inter_class_correlation(self) -> pd.DataFrame:
        """
        Return the asset-class × asset-class correlation matrix obtained by
        averaging cross-class pairwise correlations. Diagonal entries are
        the within-class average correlation (excluding self-pairs).

        Empty DataFrame if asset_class_map is missing or fewer than 2 classes
        are represented.
        """
        if not self.asset_class_map:
            return pd.DataFrame()
        cm = self.get_correlation()
        if cm.empty:
            return pd.DataFrame()
        groups: dict[str, list[str]] = {}
        for a in cm.columns:
            ac = self.asset_class_map.get(a)
            if ac:
                groups.setdefault(ac, []).append(a)
        if len(groups) < 2:
            return pd.DataFrame()
        classes = sorted(groups)
        out = pd.DataFrame(index=classes, columns=classes, dtype=float)
        for ci in classes:
            for cj in classes:
                vals = []
                for a in groups[ci]:
                    for b in groups[cj]:
                        if a == b:
                            continue
                        vals.append(float(cm.loc[a, b]))
                out.loc[ci, cj] = sum(vals) / len(vals) if vals else float("nan")
        return out


@dataclass
class S1Output:
    """
    Output of Stage 1 (Market Interpretation).

    Contains structured market views derived from raw data + text.

    Attributes:
        asset_views:     Dict mapping asset -> sentiment score in [-1, +1].
                         +1 = strongly bullish, -1 = strongly bearish, 0 = neutral.
        macro_summary:   Short free-text summary of macro environment.
        detected_regime: Model's detected market regime string.
        confidence:      Overall confidence of the interpretation in [0, 1].
        raw_llm_output:  Full LLM response text (for debugging / CEPS analysis).
    """

    asset_views: dict[str, float] = field(default_factory=dict)
    macro_summary: str = ""
    detected_regime: str = "unknown"
    confidence: float = 0.5
    raw_llm_output: str = ""
    refused: bool = False  # True when model declined to answer


@dataclass
class S2Output:
    """
    Output of Stage 2 (Signal Generation).

    Converts S1 views into actionable directional signals.

    Attributes:
        signals:     Dict mapping asset -> signal in {"buy", "hold", "sell"}.
        strengths:   Dict mapping asset -> signal strength in [0, 1].
        reasoning:   Free-text reasoning for each signal.
    """

    signals: dict[str, str] = field(default_factory=dict)  # "buy" | "hold" | "sell"
    strengths: dict[str, float] = field(default_factory=dict)  # [0, 1]
    reasoning: str = ""
    raw_llm_output: str = ""
    refused: bool = False


@dataclass
class S3Output:
    """
    Output of Stage 3 (Weight Optimization).

    Optimal portfolio weights derived from S2 signals.

    Attributes:
        weights:          Dict mapping asset -> portfolio weight. Must sum to 1.
        expected_return:  Estimated portfolio annualized return.
        expected_vol:     Estimated portfolio annualized volatility.
        sharpe_estimate:  Estimated Sharpe ratio.
    """

    weights: dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    expected_vol: float = 0.0
    sharpe_estimate: float = 0.0
    raw_llm_output: str = ""
    refused: bool = False


@dataclass
class TradeOrder:
    """A single trade order generated by the execution stage."""

    asset: str
    direction: str  # "buy" | "sell"
    quantity: float  # Dollar amount
    price: float  # Execution price (post-slippage)
    slippage: float  # Slippage applied (as fraction)
    commission: float  # Commission paid in dollars


@dataclass
class S4Output:
    """
    Output of Stage 4 (Execution Simulation).

    Attributes:
        orders:            List of TradeOrder objects.
        executed_weights:  Actual weights after execution (may differ from S3 due to costs).
        total_cost:        Total transaction cost (commissions + slippage) in dollars.
        turnover:          Total portfolio turnover (sum of |Δw_i|).
    """

    orders: list[TradeOrder] = field(default_factory=list)
    executed_weights: dict[str, float] = field(default_factory=dict)
    total_cost: float = 0.0
    turnover: float = 0.0
    raw_llm_output: str = ""


@dataclass
class RiskAlert:
    """A triggered risk alert from Stage 5."""

    metric: str  # e.g., "drawdown", "var_breach", "weight_drift"
    value: float  # Observed value
    threshold: float  # Limit that was breached
    severity: str  # "warning" | "critical"
    action: str  # Recommended action: "rebalance" | "reduce" | "hold"


@dataclass
class S5Output:
    """
    Output of Stage 5 (Risk Monitoring).

    Attributes:
        portfolio_var:     1-day VaR at 95% confidence.
        portfolio_drawdown: Current drawdown from peak.
        weight_drift:      Max absolute deviation of current from target weights.
        alerts:            List of triggered RiskAlert objects.
        rebalance_needed:  True if any critical alert was triggered.
    """

    portfolio_var: float = 0.0
    portfolio_drawdown: float = 0.0
    weight_drift: float = 0.0
    alerts: list[RiskAlert] = field(default_factory=list)
    rebalance_needed: bool = False
    raw_llm_output: str = ""


# ---------------------------------------------------------------------------
# Pipeline stage interface
# ---------------------------------------------------------------------------


class PipelineStage(ABC):
    """
    Abstract base class for a single evaluation pipeline stage.

    Each stage receives market data + prior stage output, calls the agent
    (LLM or baseline), and returns a typed output object.

    Subclasses must implement:
      - stage_id: StageID enum value
      - run(): main execution logic
      - score(): compare actual output to ground truth → float in [0, 1]
    """

    @property
    @abstractmethod
    def stage_id(self) -> StageID:
        pass

    @abstractmethod
    def run(
        self,
        snapshot: MarketSnapshot,
        prior_output: Optional[Any] = None,
    ) -> Any:
        """
        Execute this stage and return a typed output object.

        Args:
            snapshot:     Market data snapshot.
            prior_output: Output from the immediately preceding stage (or None for S1).

        Returns:
            Typed stage output (S1Output, S2Output, …, S5Output).
        """
        pass

    @abstractmethod
    def score(self, actual: Any, ground_truth: Any) -> float:
        """
        Score the actual output against a ground-truth output.

        Args:
            actual:       Output produced by the agent.
            ground_truth: Ideal output computed from true market data.

        Returns:
            Score in [0, 1]. 1 = perfect match, 0 = complete failure.
        """
        pass

    def compute_ground_truth(self, snapshot: MarketSnapshot) -> Any:
        """
        Compute the ideal ground-truth output for this stage given the snapshot.

        Used in oracle mode (inject ground truth at any stage) and for scoring.
        Default implementation returns None; override in each concrete stage.
        """
        return None


# ---------------------------------------------------------------------------
# LLM agent adapter interface
# ---------------------------------------------------------------------------


class AgentAdapter(ABC):
    """
    Abstract adapter for calling an LLM agent.

    Concrete implementations:
      - MockAgentAdapter   (for development / unit tests)
      - OpenAIAdapter      (GPT-4o etc.)
      - AnthropicAdapter   (Claude etc.)
      - HuggingFaceAdapter (open-source models)

    All adapters expose a single `complete(prompt: str) -> str` method so that
    pipeline stages are model-agnostic.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier (used in result filenames)."""
        pass

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """
        Call the LLM and return the raw text response.

        Args:
            prompt: Full prompt string (including system + user messages).

        Returns:
            Model's response as a plain string.
        """
        pass

    def complete_with_tools(self, prompt: str, tools: list) -> str:
        """
        Call the LLM with tool-calling support.

        The default implementation ignores tools and falls back to complete().
        Cloud adapters (AnthropicAdapter, OpenAIAdapter, LiteLLMAdapter) override
        this with a native multi-turn tool execution loop.

        Args:
            prompt: Full prompt string.
            tools:  List of ToolSpec objects available to the model.

        Returns:
            Model's final text response after all tool calls are resolved.
        """
        return self.complete(prompt)


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """
    Full evaluation result for one market snapshot episode.

    Attributes:
        decision_date:  Date of the episode.
        stage_outputs:  Dict mapping StageID → actual stage output.
        gt_outputs:     Dict mapping StageID → ground-truth stage output.
        stage_scores:   Dict mapping StageID → score in [0, 1].
        errors:         Dict mapping StageID → error message (if stage failed).
    """

    decision_date: date
    stage_outputs: dict[StageID, Any] = field(default_factory=dict)
    gt_outputs: dict[StageID, Any] = field(default_factory=dict)
    stage_scores: dict[StageID, float] = field(default_factory=dict)
    errors: dict[StageID, str] = field(default_factory=dict)
    refused_stages: list[str] = field(default_factory=list)  # stage names that returned a refusal fallback

    def to_stage_score_list(self):
        """Convert to a list of StageScore objects for CEPS computation."""
        from ..metrics.ceps import StageScore

        ordered = [
            StageID.S1_MARKET_INTERPRETATION,
            StageID.S2_SIGNAL_GENERATION,
            StageID.S3_WEIGHT_OPTIMIZATION,
            StageID.S4_EXECUTION_SIMULATION,
            StageID.S5_RISK_MONITORING,
        ]
        return [
            StageScore(
                stage_id=sid.value,
                stage_name=sid.name,
                score=self.stage_scores.get(sid, 0.0),
                ground_truth=self.gt_outputs.get(sid),
                actual_output=self.stage_outputs.get(sid),
                error_details=(
                    {"error": self.errors.get(sid, "")} if sid in self.errors else {}
                ),
            )
            for sid in ordered
        ]


class EvalPipeline:
    """
    Chains five pipeline stages and runs end-to-end agent evaluation.

    Usage:
        pipeline = EvalPipeline(stages=[s1, s2, s3, s4, s5])
        result = pipeline.run_episode(snapshot)
        ceps_result = CEPS().compute(result.to_stage_score_list())

    With logging (records every prompt, response, score to disk):
        pipeline.enable_logging(output_dir="outputs/eval_logs", model_name="claude-opus-4-6")
        result = pipeline.run_episode(snapshot)  # logs written automatically

    Oracle mode:
        Set inject_gt_at=StageID.S3 to inject ground-truth S3 output, measuring
        only the impact of S4 and S5 errors on downstream performance.
    """

    def __init__(self, stages: list[PipelineStage]):
        """
        Args:
            stages: List of PipelineStage objects, ordered S1 → S5.
        """
        self.stages = {s.stage_id: s for s in stages}
        self._logger = None  # Set by enable_logging()

    def enable_logging(
        self,
        output_dir: str = "outputs/eval_logs",
        model_name: str = "unknown",
        run_id: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> "EvalPipeline":
        """
        Enable persistent logging of all evaluation interactions.

        Creates a new EvalLogger that writes every prompt, raw LLM response,
        parsed output, ground truth, and score to disk under output_dir/{run_id}/.

        Args:
            output_dir:  Root directory for log files (default: outputs/eval_logs).
            model_name:  Model identifier included in log metadata.
            run_id:      Optional explicit run ID. Defaults to timestamp-based ID.
            config:      Arbitrary config dict stored in run_meta.json.

        Returns:
            self (for chaining: pipeline.enable_logging(...).run_episode(...))
        """
        from .eval_logger import EvalLogger

        self._logger = EvalLogger(
            run_id=run_id,
            output_dir=output_dir,
            model_name=model_name,
            config=config,
        )
        return self

    def finalize_logging(self) -> Optional[str]:
        """
        Write run summary and close the logger.

        Call after all episodes are complete to write run_summary.json.
        Returns the path to the log directory, or None if logging was not enabled.
        """
        if self._logger is not None:
            path = self._logger.close()
            return str(path)
        return None

    def run_episode(
        self,
        snapshot: MarketSnapshot,
        inject_gt_at: Optional[StageID] = None,
    ) -> EpisodeResult:
        """
        Run the full pipeline for one market snapshot.

        Args:
            snapshot:     Market data snapshot for this episode.
            inject_gt_at: If set, replace all stage outputs from this stage
                          onwards with ground-truth values (oracle mode).

        Returns:
            EpisodeResult with per-stage scores and outputs.
            If logging is enabled, interaction details are also written to disk.
        """
        result = EpisodeResult(decision_date=snapshot.decision_date)
        ordered_ids = [
            StageID.S1_MARKET_INTERPRETATION,
            StageID.S2_SIGNAL_GENERATION,
            StageID.S3_WEIGHT_OPTIMIZATION,
            StageID.S4_EXECUTION_SIMULATION,
            StageID.S5_RISK_MONITORING,
        ]

        # Collect prompts and raw responses for the logger
        prompts: dict[StageID, str] = {}
        raw_responses: dict[StageID, str] = {}
        latencies_ms: dict[StageID, float] = {}

        prior_output = None
        using_oracle = False
        episode_start = datetime.now()

        for sid in ordered_ids:
            stage = self.stages.get(sid)
            if stage is None:
                continue

            # Compute ground truth for this stage (always, for scoring)
            gt = stage.compute_ground_truth(snapshot)
            result.gt_outputs[sid] = gt

            # Switch to oracle mode at inject_gt_at boundary
            if inject_gt_at is not None and sid == inject_gt_at:
                using_oracle = True

            try:
                stage_start = datetime.now()

                if using_oracle and gt is not None:
                    actual = gt
                else:
                    actual = stage.run(snapshot, prior_output)

                latencies_ms[sid] = (
                    datetime.now() - stage_start
                ).total_seconds() * 1000

                result.stage_outputs[sid] = actual
                result.stage_scores[sid] = (
                    stage.score(actual, gt) if gt is not None else 1.0
                )
                # If the stage returned a refusal fallback, override score to 0
                if getattr(actual, "refused", False):
                    result.refused_stages.append(sid.name)
                    result.stage_scores[sid] = 0.0
                prior_output = actual

                # Extract prompt and raw response from stage output if present
                if hasattr(actual, "raw_llm_output") and actual.raw_llm_output:
                    raw_responses[sid] = actual.raw_llm_output
                # Prompts are injected by stages that support logging (see stages.py)
                if hasattr(stage, "_last_prompt"):
                    prompts[sid] = stage._last_prompt

            except Exception as exc:
                result.errors[sid] = str(exc)
                if self._logger is not None:
                    duration_ms = (datetime.now() - episode_start).total_seconds() * 1000
                    self._logger.log_episode(
                        result=result,
                        prompts=prompts,
                        raw_responses=raw_responses,
                        latencies_ms=latencies_ms,
                        ceps_score=0.0,
                        duration_ms=duration_ms,
                    )
                raise RuntimeError(
                    f"Pipeline stage {sid} failed: {exc}"
                ) from exc

        # Write episode log if logging is enabled
        if self._logger is not None:
            duration_ms = (datetime.now() - episode_start).total_seconds() * 1000
            self._logger.log_episode(
                result=result,
                prompts=prompts,
                raw_responses=raw_responses,
                latencies_ms=latencies_ms,
                ceps_score=0.0,  # Filled in by run_evaluation.py after CEPS computation
                duration_ms=duration_ms,
            )

        return result
