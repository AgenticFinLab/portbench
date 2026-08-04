"""LiveEvalRunner: rolling live eval with dual-oracle CEPS."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from ..agent_eval import build_default_pipeline
from ..agent_eval.base import StageID
from ..agent_eval.investor_profiles import PROFILES
from ..agent_eval.mock_agent import MockAgentAdapter
from ..qa_builder.processed_data import ProcessedDataProvider
from ..sandbox.snapshot_builder import SnapshotBuilder
from .dates import (
    available_trading_days,
    calendar_forward_days,
    coverage_needed_through,
    is_coverage_insufficient,
    local_data_max_date,
    resolve_live_dates,
)
from .dual_score import dual_score
from .refresh import refresh_and_preprocess
from .schedule import SUPPORTED_FREQUENCIES, decision_realization_pairs


@dataclass
class LiveEvalResult:
    decision_date: date
    today: date  # realization / GT end date
    provider: str
    model: str
    profile: str
    recommended_weights: dict[str, float]
    scores: dict[str, dict]
    output_dir: str
    meta: dict = field(default_factory=dict)
    rebalance: str = "daily"


@dataclass
class LiveRangeResult:
    start: date
    end: date
    rebalance: str
    provider: str
    model: str
    profile: str
    episodes: list[LiveEvalResult]
    summary_path: str
    output_dir: str


def _default_assets(provider: ProcessedDataProvider) -> list[str]:
    assets: list[str] = []
    for cls in (
        "equities",
        "bonds",
        "commodities",
        "real_estate",
        "cryptocurrency",
        "cash",
    ):
        try:
            assets.extend(provider.list_assets(cls) or [])
        except Exception:
            continue
    seen = set()
    out = []
    for a in assets:
        if a not in seen:
            seen.add(a)
            out.append(a)
    if not out:
        out = ["SPY", "TLT", "GLD", "VNQ", "BTC-USD", "BIL"]
    return out


def _serialize_stage_outputs(stage_outputs: dict) -> dict:
    out = {}
    for sid, val in stage_outputs.items():
        key = sid.value if hasattr(sid, "value") else str(sid)
        if hasattr(val, "__dict__"):
            d = {
                k: v
                for k, v in vars(val).items()
                if not k.startswith("_")
            }
            out[key] = d
        else:
            out[key] = str(val)
    return out


def _extract_llm_io(pipeline, stage_outputs: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Collect full per-stage prompts and raw LLM responses (no truncation)."""
    prompts: dict[str, str] = {}
    responses: dict[str, str] = {}
    stages = getattr(pipeline, "stages", {}) or {}
    for sid, stage in stages.items():
        key = sid.value if hasattr(sid, "value") else str(sid)
        prompt = getattr(stage, "_last_prompt", None)
        if prompt:
            prompts[key] = str(prompt)
        out = stage_outputs.get(sid)
        raw = getattr(out, "raw_llm_output", None) if out is not None else None
        if raw:
            responses[key] = str(raw)
    return prompts, responses


def _trim_future_to_realization(snapshot, decision: date, realization: date):
    """Keep future_return_data rows in (decision, realization]."""
    if not snapshot.future_return_data:
        return
    trimmed = {}
    for a, s in snapshot.future_return_data.items():
        s2 = s.copy()
        try:
            idx_dates = s2.index.map(
                lambda x: x.date() if hasattr(x, "date") else x
            )
            s2 = s2[(idx_dates > decision) & (idx_dates <= realization)]
        except Exception:
            s2 = s2.iloc[:1]
        if not s2.empty:
            trimmed[a] = s2
    if trimmed:
        snapshot.future_return_data = trimmed


class LiveEvalRunner:
    """
    Live / rolling evaluation.

    Single step: decision_date sees only PiT data; realization day(s) fill
    future_return_data for ex-post scoring. Dual-score lookback + ex_post.

    Range mode: rebalance in {daily, weekly, monthly, quarterly, yearly}.
    """

    def __init__(
        self,
        *,
        data_dir: str = "datasets/processed",
        sec_dir: str = "datasets/sec",
        output_root: str = "outputs/live",
        lookback_days: int = 60,
        initial_nav: float = 1_000_000.0,
        propagation_weight: float = 0.1,
    ):
        self.data_dir = data_dir
        self.sec_dir = sec_dir
        self.output_root = Path(output_root)
        self.lookback_days = lookback_days
        self.initial_nav = initial_nav
        self.propagation_weight = propagation_weight

    def _load_provider(self) -> ProcessedDataProvider:
        return ProcessedDataProvider(data_dir=self.data_dir, sec_dir=self.sec_dir)

    def _ensure_data_coverage(
        self,
        *,
        needed_through: date,
        force_refresh: bool = False,
        auto_refresh: bool = True,
        skip_preprocess: bool = False,
    ) -> tuple[ProcessedDataProvider, dict]:
        """
        Ensure local processed data covers ``needed_through``.

        If coverage is missing (or ``force_refresh``), download Yahoo/FRED and
        re-preprocess, then reload the provider.
        """
        meta: dict = {"auto_refreshed": False, "forced_refresh": False}
        live_dir = (
            self.data_dir
            if "processed_live" in self.data_dir.replace("\\", "/")
            else "datasets/processed_live"
        )

        def _try_load():
            try:
                return self._load_provider()
            except Exception:
                # Fall back to benchmark processed for coverage probe only
                if Path(self.data_dir) != Path("datasets/processed"):
                    return ProcessedDataProvider(
                        data_dir="datasets/processed", sec_dir=self.sec_dir
                    )
                raise

        data = _try_load()
        try:
            insufficient = is_coverage_insufficient(data, needed_through)
        except Exception:
            insufficient = True

        if force_refresh or (auto_refresh and insufficient):
            try:
                local_max = local_data_max_date(data)
            except RuntimeError:
                local_max = None
            reason = "force_refresh" if force_refresh else "local_data_stale"
            print(
                f"[live] Local data max={local_max}, need through {needed_through} "
                f"({reason}). Yahoo→akshare_ext overlay + build {live_dir} "
                "(datasets/processed untouched)..."
            )
            refresh_and_preprocess(
                force=bool(force_refresh),
                skip_refresh=False,
                skip_preprocess=False,
                needed_through=needed_through,
                build_live_overlay=True,
                live_dir=live_dir,
            )
            data = self._load_provider()
            meta["auto_refreshed"] = not force_refresh
            meta["forced_refresh"] = bool(force_refresh)
            meta["local_max_before"] = (
                local_max.isoformat() if local_max else None
            )
            try:
                meta["local_max_after"] = local_data_max_date(data).isoformat()
            except RuntimeError:
                meta["local_max_after"] = None

            if is_coverage_insufficient(data, needed_through):
                after = meta.get("local_max_after")
                raise RuntimeError(
                    f"After refresh, local data still ends at {after}, "
                    f"but need through {needed_through}. "
                    "Check network / FRED_API_KEY / market holiday / EOD lag."
                )
        elif not skip_preprocess and insufficient:
            refresh_and_preprocess(
                force=False,
                skip_refresh=False,
                skip_preprocess=False,
                needed_through=needed_through,
                build_live_overlay=True,
                live_dir=live_dir,
            )
            data = self._load_provider()
            meta["auto_refreshed"] = True

        return data, meta

    def _build_adapter(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        mock: bool = False,
        baseline: Optional[str] = None,
    ):
        from ..experiments.providers import build_adapter, build_baseline

        if baseline:
            return build_baseline(baseline), "baseline", baseline
        if mock or provider == "mock":
            return MockAgentAdapter(noise_level=0.15, seed=42), "mock", "mock"
        if not provider:
            raise ValueError("provider is required when not using mock/baseline")
        adapter = build_adapter(provider, model=model)
        return adapter, provider, model or provider

    def _run_one_pair(
        self,
        *,
        data: ProcessedDataProvider,
        builder: SnapshotBuilder,
        adapter,
        provider_name: str,
        model_name: str,
        profile: str,
        decision: date,
        realization: date,
        weights: dict[str, float],
        rebalance: str,
        run_tag: str,
    ) -> LiveEvalResult:
        fwd = calendar_forward_days(decision, realization)
        snapshot = builder.build(
            decision, weights, self.initial_nav, forward_days=fwd
        )
        _trim_future_to_realization(snapshot, decision, realization)
        if snapshot.future_return_data is None:
            raise RuntimeError(
                f"No future_return_data from {decision} toward {realization}. "
                "Refresh data or shrink the end date so a next period exists."
            )

        prof = PROFILES.get(profile) or PROFILES["balanced"]
        pipeline = build_default_pipeline(
            adapter, profile=prof, oracle_mode="lookback"
        )
        episode = pipeline.run_episode(snapshot)
        scores = dual_score(
            snapshot,
            episode.stage_outputs,
            profile=prof,
            propagation_weight=self.propagation_weight,
            oracle_modes=["lookback", "ex_post"],
        )
        s3_out = episode.stage_outputs.get(StageID.S3_WEIGHT_OPTIMIZATION)
        recommended = dict(getattr(s3_out, "weights", {}) or {})

        out_dir = (
            self.output_root
            / run_tag
            / provider_name
            / model_name.replace("/", "-")
            / profile
            / decision.isoformat()
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "decision_date": decision.isoformat(),
            "realization_date": realization.isoformat(),
            "rebalance": rebalance,
            "provider": provider_name,
            "model": model_name,
            "profile": profile,
            "n_assets": len(weights),
            "future_return_assets": list(snapshot.future_return_data.keys()),
            "notes": [
                "Model sees only information available on decision_date.",
                "future_return_data through realization_date is used only for ex_post scoring.",
            ],
        }
        (out_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        (out_dir / "episode_trace.json").write_text(
            json.dumps(
                _serialize_stage_outputs(episode.stage_outputs),
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        prompts, responses = _extract_llm_io(pipeline, episode.stage_outputs)
        (out_dir / "prompts.json").write_text(
            json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "llm_responses.json").write_text(
            json.dumps(responses, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "recommended_weights.json").write_text(
            json.dumps(recommended, indent=2), encoding="utf-8"
        )
        (out_dir / "scores_lookback.json").write_text(
            json.dumps(scores["lookback"], indent=2), encoding="utf-8"
        )
        (out_dir / "scores_ex_post.json").write_text(
            json.dumps(scores["ex_post"], indent=2), encoding="utf-8"
        )

        return LiveEvalResult(
            decision_date=decision,
            today=realization,
            provider=provider_name,
            model=model_name,
            profile=profile,
            recommended_weights=recommended,
            scores=scores,
            output_dir=str(out_dir),
            meta=meta,
            rebalance=rebalance,
        )

    def run(
        self,
        *,
        provider: str = "mock",
        model: Optional[str] = None,
        profile: str = "balanced",
        decision_date: Optional[date] = None,
        as_of_today: Optional[date] = None,
        mock: bool = False,
        force_refresh: bool = False,
        auto_refresh: bool = True,
        skip_refresh: bool = True,
        skip_preprocess: bool = True,
        assets: Optional[list[str]] = None,
        rebalance: str = "daily",
        baseline: Optional[str] = None,
    ) -> LiveEvalResult:
        """Single decision → next-period realization (default: yesterday → today)."""
        needed = coverage_needed_through(
            as_of_today=as_of_today,
            decision_date=decision_date,
        )
        # skip_refresh=False means user asked to refresh (legacy CLI flag)
        data, cov_meta = self._ensure_data_coverage(
            needed_through=needed,
            force_refresh=force_refresh or (not skip_refresh),
            auto_refresh=auto_refresh,
            skip_preprocess=skip_preprocess,
        )
        decision, realization = resolve_live_dates(
            data,
            decision_date=decision_date,
            as_of_today=as_of_today,
        )
        asset_list = assets or _default_assets(data)
        n = len(asset_list)
        weights = {a: round(1.0 / n, 6) for a in asset_list}
        acm = None
        acm_path = Path(self.data_dir) / "asset_class_map.json"
        if acm_path.exists():
            try:
                acm = json.loads(acm_path.read_text(encoding="utf-8"))
            except Exception:
                acm = None
        builder = SnapshotBuilder(
            data, asset_list, lookback_days=self.lookback_days, asset_class_map=acm
        )
        adapter, provider_name, model_name = self._build_adapter(
            provider=provider, model=model, mock=mock, baseline=baseline
        )
        result = self._run_one_pair(
            data=data,
            builder=builder,
            adapter=adapter,
            provider_name=provider_name,
            model_name=model_name,
            profile=profile,
            decision=decision,
            realization=realization,
            weights=weights,
            rebalance=rebalance,
            run_tag="single",
        )
        result.meta["data_coverage"] = cov_meta
        return result

    def run_range(
        self,
        *,
        start: date,
        end: date,
        rebalance: str = "daily",
        provider: str = "mock",
        model: Optional[str] = None,
        profile: str = "balanced",
        mock: bool = False,
        force_refresh: bool = False,
        auto_refresh: bool = True,
        skip_refresh: bool = True,
        skip_preprocess: bool = True,
        assets: Optional[list[str]] = None,
        carry_weights: bool = True,
        baseline: Optional[str] = None,
    ) -> LiveRangeResult:
        """
        Rolling live eval over [start, end] at the given rebalance frequency.

        For each rebalance date D in the window, the model decides using data ≤ D;
        ex-post GT uses returns from D to the next rebalance date (daily ⇒ next
        trading day). This simulates "we ran live eval every period in the window"
        without waiting for a full month of wall-clock time.
        """
        freq = rebalance.lower().strip()
        if freq not in SUPPORTED_FREQUENCIES:
            raise ValueError(
                f"rebalance must be one of {SUPPORTED_FREQUENCIES}, got {rebalance!r}"
            )

        # Buffer so the last decision still has a realization day for ex-post GT.
        # Coverage check floors to available sessions / EOD lag — do not require
        # future calendar dates that markets have not traded yet.
        needed = coverage_needed_through(range_end=end + timedelta(days=5))
        data, cov_meta = self._ensure_data_coverage(
            needed_through=needed,
            force_refresh=force_refresh or (not skip_refresh),
            auto_refresh=auto_refresh,
            skip_preprocess=skip_preprocess,
        )

        # Pull enough calendar to cover GT after ``end``
        cal_start = start - timedelta(days=40)
        cal_end = end + timedelta(days=40)
        days = available_trading_days(
            data, start=cal_start, end=cal_end
        )
        pairs = decision_realization_pairs(
            days, start=start, end=end, frequency=freq
        )
        if not pairs:
            raise RuntimeError(
                f"No (decision, realization) pairs for {start}..{end} "
                f"at rebalance={freq}. Check that processed data covers the window "
                f"plus one period after the last decision."
            )

        asset_list = assets or _default_assets(data)
        n = len(asset_list)
        weights = {a: round(1.0 / n, 6) for a in asset_list}
        acm = None
        acm_path = Path(self.data_dir) / "asset_class_map.json"
        if acm_path.exists():
            try:
                acm = json.loads(acm_path.read_text(encoding="utf-8"))
            except Exception:
                acm = None
        builder = SnapshotBuilder(
            data, asset_list, lookback_days=self.lookback_days, asset_class_map=acm
        )
        adapter, provider_name, model_name = self._build_adapter(
            provider=provider, model=model, mock=mock, baseline=baseline
        )

        tag = f"{freq}_{start.isoformat()}_{end.isoformat()}"
        episodes: list[LiveEvalResult] = []
        rows = []
        for decision, realization in pairs:
            result = self._run_one_pair(
                data=data,
                builder=builder,
                adapter=adapter,
                provider_name=provider_name,
                model_name=model_name,
                profile=profile,
                decision=decision,
                realization=realization,
                weights=weights,
                rebalance=freq,
                run_tag=tag,
            )
            episodes.append(result)
            rows.append(
                {
                    "decision_date": decision.isoformat(),
                    "realization_date": realization.isoformat(),
                    "ceps_lookback": result.scores["lookback"]["ceps"],
                    "ceps_ex_post": result.scores["ex_post"]["ceps"],
                    "stage_scores_lookback": result.scores["lookback"]["stage_scores"],
                    "stage_scores_ex_post": result.scores["ex_post"]["stage_scores"],
                }
            )
            if carry_weights and result.recommended_weights:
                weights = dict(result.recommended_weights)

        out_root = (
            self.output_root
            / tag
            / provider_name
            / model_name.replace("/", "-")
            / profile
        )
        out_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "rebalance": freq,
            "provider": provider_name,
            "model": model_name,
            "profile": profile,
            "n_episodes": len(episodes),
            "mean_ceps_lookback": round(
                sum(r["ceps_lookback"] for r in rows) / len(rows), 4
            )
            if rows
            else None,
            "mean_ceps_ex_post": round(
                sum(r["ceps_ex_post"] for r in rows) / len(rows), 4
            )
            if rows
            else None,
            "data_coverage": cov_meta,
            "episodes": rows,
            "notes": [
                "Simulated rolling live eval over a historical window.",
                "Supports daily/weekly/monthly/quarterly/yearly rebalance.",
                "Main paper uses monthly; daily is for faster live-capability demos.",
                "If requested dates exceed local data, Yahoo/FRED are auto-refreshed.",
            ],
        }
        summary_path = out_root / "range_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return LiveRangeResult(
            start=start,
            end=end,
            rebalance=freq,
            provider=provider_name,
            model=model_name,
            profile=profile,
            episodes=episodes,
            summary_path=str(summary_path),
            output_dir=str(out_root),
        )
