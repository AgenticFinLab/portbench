"""
BacktestEngine: stateful backtest loop for the PortBench Sandbox.

Drives a portfolio through historical data, calling the strategy on each
rebalance date and updating PortfolioState with transaction costs and
daily mark-to-market between rebalances.

Compatible with both:
  - LLM agents (use_pipeline=True): full S1→S5 pipeline on each rebalance date
  - Baseline strategies (use_pipeline=False): direct allocate() call, no API cost

When an InvestorProfile is provided, the profile description is prepended to
each snapshot's news_text, and per-step CEPS and profile alignment scores are
collected and passed into BacktestResult.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from tqdm.auto import tqdm

from ..agent_eval.base import AgentAdapter, EvalPipeline, StageID
from ..agent_eval import build_default_pipeline
from ..agent_eval.investor_profiles import InvestorProfile, ProfileAlignmentScorer
from ..baselines.base import BaselineStrategy
from ..qa_builder.base import DataProvider
from .portfolio import PortfolioState
from .result import BacktestResult
from .snapshot_builder import SnapshotBuilder


_REBALANCE_FREQS = {
    "weekly": "W-FRI",
    "monthly": "BMS",  # Business Month Start
    "quarterly": "QS-JAN",  # Quarter Start
}

_REBALANCE_FORWARD_DAYS = {
    "weekly": 5,
    "monthly": 21,
    "quarterly": 63,
}


class BacktestEngine:
    """
    Stateful backtest engine that drives a strategy through a historical period.

    Args:
        strategy:        AgentAdapter (LLM) or BaselineStrategy to evaluate.
        provider:        DataProvider for historical price/return/macro data.
        assets:          Asset list (defaults to provider.list_assets()).
        start_date:      Backtest start date.
        end_date:        Backtest end date.
        rebalance_freq:  One of "weekly", "monthly", "quarterly".
        initial_nav:     Starting portfolio NAV in dollars.
        initial_weights: Optional initial portfolio weights. If None, uses
                         equal-weight across all assets.
        lookback_days:   Trading days of history passed to each snapshot.
        use_pipeline:    If True, route LLM through full S1→S5 EvalPipeline.
                         If False, call BaselineStrategy.allocate() directly.
        profile:         Optional InvestorProfile to inject into each snapshot and
                         score against. When set, per-step alignment scores and CEPS
                         are collected into BacktestResult.
        asset_class_map: Required when profile is set. Dict mapping ticker → asset
                         class string (e.g. {"SPY": "equities", "TLT": "bonds"}).
    """

    def __init__(
        self,
        strategy: AgentAdapter,
        provider: DataProvider,
        assets: Optional[list[str]] = None,
        start_date: date = date(2024, 1, 1),
        end_date: date = date(2024, 12, 31),
        rebalance_freq: str = "monthly",
        initial_nav: float = 1_000_000.0,
        initial_weights: Optional[dict[str, float]] = None,
        lookback_days: int = 60,
        use_pipeline: bool = True,
        use_tools: bool = False,
        profile: Optional[InvestorProfile] = None,
        asset_class_map: Optional[dict[str, str]] = None,
        snapshot_dump_dir: Optional[str] = None,
        propagation_weight: float = 0.1,
        step_cache_dir: Optional[str] = None,
        on_rebalance: Optional[Callable] = None,
        progress: bool = True,
        oracle_mode: str = "ex_post",
        architecture_id: Optional[str] = None,
        provider_name: str = "",
        data_version: str = "",
        code_commit: str = "",
        resource_budget=None,
        intervention_spec: Optional[dict] = None,
        call_max_attempts: int = 3,
        retry_failed_calls: bool = False,
        schema_version: str = "pipeline-v3-collab",
        call_artifact_dir: Optional[str] = None,
    ):
        self.strategy = strategy
        self._snapshot_dump_dir: Optional[str] = snapshot_dump_dir
        self.provider = provider
        self.assets = assets or provider.list_assets()
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_freq = rebalance_freq
        self._forward_days = _REBALANCE_FORWARD_DAYS.get(rebalance_freq, 21)
        self.initial_nav = initial_nav
        self.initial_weights = initial_weights
        self.lookback_days = lookback_days
        self.use_pipeline = use_pipeline
        self._propagation_weight = propagation_weight
        self._on_rebalance = on_rebalance
        self._progress = progress
        self._profile = profile
        self._architecture_id = architecture_id
        self._schema_version = str(schema_version)
        self._data_version = data_version
        self._code_commit = code_commit
        self._intervention_spec: dict = dict(intervention_spec or {})
        self._rebalance_records: list[tuple] = []
        self.closed_loop_interventions: list[dict] = []
        self._alignment_scorer = (
            ProfileAlignmentScorer(asset_class_map)
            if (profile is not None and asset_class_map)
            else None
        )

        self._snapshot_builder = SnapshotBuilder(
            provider, self.assets, lookback_days, asset_class_map=asset_class_map
        )
        self._pipeline_log_dir: Optional[str] = None

        # Build pipeline once (reused across all rebalance steps)
        self._pipeline: Optional[EvalPipeline] = None
        if use_pipeline:
            self._pipeline = build_default_pipeline(
                strategy,
                use_tools=use_tools,
                profile=profile,
                oracle_mode=oracle_mode,
                architecture_id=architecture_id,
                cache_dir=(
                    str(Path(step_cache_dir) / "stage_cache")
                    if step_cache_dir and architecture_id is not None
                    else None
                ),
                memory_path=(
                    str(Path(step_cache_dir) / "memory.json")
                    if step_cache_dir and architecture_id is not None
                    else None
                ),
                budget=resource_budget,
                provider=provider_name,
                profile_name=profile.name if profile is not None else "",
                data_version=data_version,
                code_commit=code_commit,
                call_max_attempts=call_max_attempts,
                retry_failed_calls=retry_failed_calls,
                schema_version=schema_version,
                call_artifact_dir=call_artifact_dir,
            )
            spec = dict(self._intervention_spec)
            spec.setdefault("propagation_weight", self._propagation_weight)
            self._pipeline.configure_interventions(spec)

        # Per-rebalance collection (populated in _get_target_weights)
        self._episode_results = []
        self._per_step_ceps: list[float] = []
        self._per_step_alignment: list[float] = []
        self._resource_audit: list[dict] = []
        self._n_refused_steps: int = 0  # episodes with ≥1 refused stage

        # Step-level weight cache: {date_str → {weights, step_ceps, step_alignment}}
        # Loaded from disk on init; written after every successful LLM rebalance.
        self._step_cache_dir: Optional[Path] = (
            Path(step_cache_dir) if step_cache_dir else None
        )
        self._step_cache: dict[str, dict] = {}
        if self._step_cache_dir is not None:
            cache_file = self._step_cache_dir / "step_cache.json"
            if cache_file.exists():
                try:
                    import json as _json
                    self._step_cache = _json.loads(
                        cache_file.read_text(encoding="utf-8")
                    )
                except Exception:
                    self._step_cache = {}

    def enable_pipeline_logging(self, output_dir: str, run_id: str) -> None:
        """Persist every S1-S5 prompt/response to disk (mirrors EvalPipeline.enable_logging)."""
        if self._pipeline is not None:
            self._pipeline.enable_logging(
                output_dir=output_dir,
                model_name=self.strategy.model_name,
                config={
                    "sandbox_run_id": run_id,
                    "architecture_id": self._architecture_id or "legacy",
                    "result_protocol": "closed-loop",
                },
            )
            self._pipeline_log_dir = output_dir

    def run(self) -> BacktestResult:
        """
        Execute the full backtest and return a BacktestResult.

        Loop structure:
          For each business day d in [start_date, end_date]:
            - If d is a rebalance date: get target weights, execute rebalance
            - Else: mark portfolio to market using daily returns
        """
        # Build the initial portfolio only from assets investable on the start date.
        initial_snapshot = self._snapshot_builder.build(
            self.start_date,
            current_weights={},
            nav=self.initial_nav,
        )
        initial_assets = list(initial_snapshot.return_data)
        if not initial_assets:
            raise RuntimeError("No investable assets are available on the start date")

        if self.initial_weights:
            unavailable_initial_assets = sorted(
                asset
                for asset, weight in self.initial_weights.items()
                if float(weight) > 1e-12 and asset not in initial_snapshot.return_data
            )
            if unavailable_initial_assets:
                raise ValueError(
                    "Initial weights contain assets outside the start-date "
                    f"investable universe: {', '.join(unavailable_initial_assets)}"
                )
            init_weights = {
                asset: float(self.initial_weights.get(asset, 0.0))
                for asset in initial_assets
            }
            total_initial_weight = sum(init_weights.values())
            if total_initial_weight <= 0:
                raise ValueError("Initial weights must assign positive total weight")
            init_weights = {
                asset: weight / total_initial_weight
                for asset, weight in init_weights.items()
            }
        else:
            equal_weight = 1.0 / len(initial_assets)
            init_weights = {asset: equal_weight for asset in initial_assets}
        portfolio = PortfolioState(
            nav=self.initial_nav,
            weights=init_weights,
        )
        portfolio.nav_history.append((self.start_date, self.initial_nav))

        # Generate rebalance dates
        freq_key = _REBALANCE_FREQS.get(self.rebalance_freq, "BMS")
        rebalance_dates = {
            ts.date()
            for ts in pd.bdate_range(self.start_date, self.end_date, freq=freq_key)
        }

        # All business days for mark-to-market
        all_bdays = [ts.date() for ts in pd.bdate_range(self.start_date, self.end_date)]

        weight_rows: list[dict] = []
        total_cost = 0.0
        n_rebalances_done = 0

        n_rebalances_total = sum(1 for d in all_bdays if d in rebalance_dates)
        pbar = tqdm(
            total=n_rebalances_total,
            desc=f"backtest {self.strategy.model_name} {self.start_date}→{self.end_date}",
            unit="reb",
            leave=False,
            disable=not self._progress,
        )

        for d in all_bdays:
            if d == self.start_date:
                weight_rows.append({"date": d, **portfolio.weights})
                continue

            if d in rebalance_dates:
                # --- Rebalance step ---
                snapshot = self._snapshot_builder.build(
                    d, portfolio.weights, portfolio.nav,
                    forward_days=self._forward_days,
                )

                if not snapshot.return_data:
                    weight_rows.append({"date": d, **portfolio.weights})
                    pbar.update(1)
                    continue

                target_weights = self._get_target_weights(snapshot)

                prices = {
                    asset: float(series.iloc[-1])
                    for asset, series in snapshot.price_data.items()
                    if not series.empty
                }

                trade_record = portfolio.rebalance(target_weights, prices, d)
                total_cost += trade_record["total_cost"]
                n_rebalances_done += 1
                pbar.update(1)

                # Fire on_rebalance callback for incremental saving
                if self._on_rebalance is not None:
                    try:
                        self._on_rebalance(
                            date=d,
                            weights=dict(portfolio.weights),
                            nav=portfolio.nav,
                            rebalance_count=n_rebalances_done,
                        )
                    except Exception:
                        pass  # don't let callback errors break backtest
            else:
                # --- Daily mark-to-market ---
                daily_returns = self._get_daily_returns(d)
                if daily_returns:
                    portfolio.mark_to_market(daily_returns, d)
                else:
                    portfolio.nav_history.append((d, portfolio.nav))

            weight_rows.append({"date": d, **portfolio.weights})

        pbar.close()

        # Build result time series
        nav_series = pd.Series(
            {d: nav for d, nav in portfolio.nav_history},
            name="nav",
        )
        nav_series.index = pd.to_datetime(nav_series.index)
        nav_series = nav_series[~nav_series.index.duplicated(keep="last")]

        weight_df = pd.DataFrame(weight_rows).set_index("date")
        weight_df.index = pd.to_datetime(weight_df.index)

        if self._pipeline is not None and self._pipeline_log_dir:
            self._pipeline.finalize_logging()

        self._run_closed_loop_interventions()

        return BacktestResult(
            model_name=self.strategy.model_name,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_nav=self.initial_nav,
            nav_curve=nav_series,
            weight_history=weight_df,
            trade_history=portfolio.trade_history,
            n_rebalances=len(portfolio.trade_history),
            total_transaction_cost=round(total_cost, 4),
            profile_name=self._profile.name if self._profile else None,
            per_step_ceps=list(self._per_step_ceps),
            per_step_alignment=list(self._per_step_alignment),
            n_refused_steps=self._n_refused_steps,
            refused_rate=round(
                self._n_refused_steps / max(1, len(portfolio.trade_history)), 4
            ),
            architecture_id=self._architecture_id or "legacy",
            result_protocol="closed-loop",
            schema_version=(
                self._schema_version if self._architecture_id is not None else "pipeline-v1"
            ),
            data_version=self._data_version,
            code_commit=self._code_commit,
            resource_audit=list(self._resource_audit),
        )

    def _get_target_weights(self, snapshot) -> dict[str, float]:
        """Get target weights; inject profile, collect CEPS + alignment when configured."""
        self._dump_snapshot(snapshot)
        if self.use_pipeline and self._pipeline is not None:
            date_key = str(snapshot.decision_date)

            # ── Step cache hit: skip LLM call, restore cached metrics ──────
            cacheable_architecture = self._architecture_id is None or (
                self._architecture_id == "SA"
                and self._schema_version == "pipeline-v4-sa-causal"
            )
            if cacheable_architecture and date_key in self._step_cache:
                cached = self._step_cache[date_key]
                if cached.get("step_ceps") is not None:
                    self._per_step_ceps.append(float(cached["step_ceps"]))
                if cached.get("step_alignment") is not None:
                    self._per_step_alignment.append(float(cached["step_alignment"]))
                return dict(cached["weights"])

            # ── LLM call ────────────────────────────────────────────────────
            if self._profile is not None:
                prefix = f"[INVESTOR PROFILE] {self._profile.description}\n\n"
                snapshot = dataclasses.replace(
                    snapshot, news_text=prefix + snapshot.news_text
                )

            result = self._pipeline.run_episode(snapshot)
            self._episode_results.append(result)
            self._rebalance_records.append((snapshot, result))
            self._resource_audit.append(
                {
                    "decision_date": date_key,
                    "architecture_id": result.architecture_id,
                    "usage": dict(result.resource_usage),
                    "provenance": dict(result.provenance),
                    "collaboration_trace": list(result.collaboration_trace),
                }
            )
            if result.refused_stages:
                self._n_refused_steps += 1

            # Collect per-step CEPS
            step_ceps: Optional[float] = None
            ssl = result.to_stage_score_list()
            if ssl:
                from ..metrics.ceps import CEPS
                ceps_result = CEPS(self._propagation_weight).compute(ssl)
                self._per_step_ceps.append(ceps_result.ceps_score)
                step_ceps = ceps_result.ceps_score

            # Collect per-step profile alignment
            step_alignment: Optional[float] = None
            if self._alignment_scorer is not None and self._profile is not None:
                step_alignment = self._alignment_scorer.score(result, self._profile)
                self._per_step_alignment.append(step_alignment)

            # Extract weights
            weights: Optional[dict] = None
            s5_output = result.stage_outputs.get(StageID.S5_RISK_MONITORING)
            if s5_output is not None and getattr(s5_output, "final_weights", None):
                weights = s5_output.final_weights
            if weights is None:
                s4_output = result.stage_outputs.get(StageID.S4_EXECUTION_SIMULATION)
                if s4_output is not None and s4_output.executed_weights:
                    weights = s4_output.executed_weights
            s3_output = result.stage_outputs.get(StageID.S3_WEIGHT_OPTIMIZATION)
            if weights is None and s3_output is not None and s3_output.weights:
                weights = s3_output.weights

            if weights is not None:
                if cacheable_architecture:
                    self._write_step_cache(
                        date_key,
                        weights,
                        step_ceps,
                        step_alignment,
                        snapshot=snapshot,
                        result=result,
                    )
                return weights

        if isinstance(self.strategy, BaselineStrategy):
            return self.strategy.allocate(snapshot)

        raise RuntimeError(
            "Pipeline produced no usable target weights and strategy is not a "
            "BaselineStrategy — refusing to fall back to equal-weight."
        )

    def _run_closed_loop_interventions(self) -> None:
        """Fork NAV from the first valid rebalance; write separately from CEPS."""
        spec = self._intervention_spec
        if not spec.get("enabled") or not spec.get("closed_loop"):
            return
        if not self._rebalance_records:
            return
        from ..agent_eval.intervention import run_offline_closed_loop

        stages = list(spec.get("stages") or ["S1", "S2", "S3", "S4", "S5"])
        operator = str(spec.get("operator") or "repair")
        weight = float(spec.get("propagation_weight") or self._propagation_weight)
        branches = []
        for stage_id in stages:
            branches.append(
                run_offline_closed_loop(
                    self._rebalance_records,
                    stage_id=stage_id,
                    operator=operator,
                    intervention_index=0,
                    pipeline=self._pipeline,
                    propagation_weight=weight,
                )
            )
        self.closed_loop_interventions = branches

    def _write_step_cache(
        self,
        date_key: str,
        weights: dict,
        step_ceps: Optional[float],
        step_alignment: Optional[float],
        *,
        snapshot=None,
        result=None,
    ) -> None:
        """Atomically persist one completed rebalance step for recovery."""
        if self._step_cache_dir is None:
            return
        self._step_cache[date_key] = {
            "weights": weights,
            "step_ceps": step_ceps,
            "step_alignment": step_alignment,
            "episode": self._episode_checkpoint(snapshot, result),
        }
        self._step_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._step_cache_dir / "step_cache.json"
        try:
            import json as _json
            temporary = cache_file.with_suffix(".tmp")
            temporary.write_text(
                _json.dumps(self._step_cache, indent=2, default=str),
                encoding="utf-8",
            )
            temporary.replace(cache_file)
        except Exception:
            pass  # non-fatal: step is lost from cache but run continues

    @staticmethod
    def _episode_checkpoint(snapshot, result) -> dict:
        """Capture state and stage outputs needed to audit a completed episode."""
        if snapshot is None or result is None:
            return {}
        stage_outputs = {}
        for stage_id, output in result.stage_outputs.items():
            try:
                stage_outputs[stage_id.value] = dataclasses.asdict(output)
            except TypeError:
                stage_outputs[stage_id.value] = str(output)
        call_keys = {
            call_id: str(record.get("call_key", ""))
            for call_id, record in dict(result.provenance or {}).items()
            if record.get("call_key")
        }
        return {
            "decision_date": str(snapshot.decision_date),
            "nav_before": float(snapshot.portfolio_value),
            "weights_before": dict(snapshot.current_weights),
            "stage_outputs": stage_outputs,
            "stage_scores": {
                stage_id.value: float(score)
                for stage_id, score in result.stage_scores.items()
            },
            "call_keys": call_keys,
            "ceps_score": float(result.ceps_score),
        }

    def _dump_snapshot(self, snapshot) -> None:
        """Persist MarketSnapshot for the current rebalance date if dumping is enabled."""
        if not self._snapshot_dump_dir:
            return
        import json
        from pathlib import Path

        out = Path(self._snapshot_dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "decision_date": str(snapshot.decision_date),
            "portfolio_value": float(snapshot.portfolio_value),
            "current_weights": dict(snapshot.current_weights),
            "market_regime": snapshot.market_regime,
            "macro_data": {k: float(v) for k, v in (snapshot.macro_data or {}).items()},
            "assets": list(snapshot.return_data.keys()),
            "trailing_returns": {
                a: float((1 + s.dropna()).prod() - 1)
                for a, s in snapshot.return_data.items()
                if not s.empty
            },
            "news_text_preview": (snapshot.news_text or "")[:500],
        }
        (out / f"{snapshot.decision_date}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    def _get_daily_returns(self, d: date) -> dict[str, float]:
        """
        Fetch single-day returns for all assets.

        Returns a dict of asset → daily return for date d.
        Assets with empty series are skipped (no observation that day);
        any other error is propagated so data issues fail loudly.
        """
        from datetime import timedelta

        returns = {}
        prev = d - timedelta(days=7)  # wide window to ensure we get prev business day
        for asset in self.assets:
            series = self.provider.get_return_series(asset, prev, d)
            if not series.empty:
                returns[asset] = float(series.iloc[-1])
        return returns
