"""LiveEvalRunner: yesterday decision + today GT, dual-oracle CEPS."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

from ..agent_eval import build_default_pipeline
from ..agent_eval.base import StageID
from ..agent_eval.investor_profiles import PROFILES
from ..agent_eval.mock_agent import MockAgentAdapter
from ..qa_builder.processed_data import ProcessedDataProvider
from ..sandbox.snapshot_builder import SnapshotBuilder
from .dates import (
    calendar_forward_days,
    iter_daily_decision_pairs,
    resolve_live_dates,
)
from .dual_score import dual_score
from .refresh import refresh_and_preprocess


@dataclass
class LiveEvalResult:
    decision_date: date
    today: date
    provider: str
    model: str
    profile: str
    recommended_weights: dict[str, float]
    scores: dict[str, dict]
    output_dir: str
    meta: dict = field(default_factory=dict)


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
    # Deduplicate, keep order
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
                if not k.startswith("_") and k != "raw_llm_output"
            }
            # Keep a short raw snippet for debugging
            raw = getattr(val, "raw_llm_output", None)
            if raw:
                d["raw_llm_output_preview"] = str(raw)[:500]
            out[key] = d
        else:
            out[key] = str(val)
    return out


class LiveEvalRunner:
    """
    End-to-end live eval:

      refresh → snapshot(yesterday, forward=today) → one LLM episode
      → dual score (lookback + ex_post) → write outputs/live/...
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
        skip_refresh: bool = True,
        skip_preprocess: bool = True,
        assets: Optional[list[str]] = None,
    ) -> LiveEvalResult:
        if force_refresh or not skip_refresh:
            refresh_and_preprocess(
                force=True,
                skip_refresh=False,
                skip_preprocess=skip_preprocess,
            )
        elif not skip_preprocess:
            refresh_and_preprocess(skip_refresh=True, skip_preprocess=False)

        data = ProcessedDataProvider(data_dir=self.data_dir, sec_dir=self.sec_dir)
        yesterday, today = resolve_live_dates(
            data,
            decision_date=decision_date,
            as_of_today=as_of_today,
        )
        asset_list = assets or _default_assets(data)
        n = len(asset_list)
        weights = {a: round(1.0 / n, 6) for a in asset_list}

        # Load optional asset_class_map
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
        fwd = calendar_forward_days(yesterday, today)
        snapshot = builder.build(
            yesterday, weights, self.initial_nav, forward_days=fwd
        )
        # Ensure future window includes today when available
        if snapshot.future_return_data:
            # Truncate each series to the first trading day after decision (= today ideally)
            trimmed = {}
            for a, s in snapshot.future_return_data.items():
                # keep rows strictly after decision_date
                s2 = s.copy()
                try:
                    s2 = s2[s2.index.map(lambda x: x.date() if hasattr(x, "date") else x) > yesterday]
                except Exception:
                    pass
                if not s2.empty:
                    trimmed[a] = s2.iloc[:1]
            if trimmed:
                snapshot.future_return_data = trimmed

        if snapshot.future_return_data is None:
            raise RuntimeError(
                f"No future_return_data from {yesterday} toward {today}. "
                "Refresh market data so today's returns are available, or pass "
                "--as-of-today / --decision-date explicitly."
            )

        # Safety: prompts must not receive future_return_data (pipeline already
        # omits it; we still record the keys for the meta file).
        future_keys = list(snapshot.future_return_data.keys())

        prof = PROFILES.get(profile) or PROFILES["balanced"]
        if mock or provider == "mock":
            adapter = MockAgentAdapter(noise_level=0.15, seed=42)
            provider_name, model_name = "mock", "mock"
        else:
            from ..experiments.providers import build_adapter

            adapter = build_adapter(provider, model=model)
            provider_name, model_name = provider, model or provider

        # Run once with lookback pipeline (oracle only affects GT scoring inside
        # run_episode; we re-score both modes from saved outputs below).
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
            / provider_name
            / model_name.replace("/", "-")
            / yesterday.isoformat()
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "decision_date": yesterday.isoformat(),
            "today": today.isoformat(),
            "provider": provider_name,
            "model": model_name,
            "profile": profile,
            "n_assets": len(asset_list),
            "future_return_assets": future_keys,
            "notes": [
                "Model sees only information available on decision_date (yesterday).",
                "future_return_data (today) is used only for ex_post scoring.",
                "Does not remove all training-data leakage risk.",
            ],
        }
        (out_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        (out_dir / "episode_trace.json").write_text(
            json.dumps(_serialize_stage_outputs(episode.stage_outputs), indent=2, default=str),
            encoding="utf-8",
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
            decision_date=yesterday,
            today=today,
            provider=provider_name,
            model=model_name,
            profile=profile,
            recommended_weights=recommended,
            scores=scores,
            output_dir=str(out_dir),
            meta=meta,
        )

    def run_range(
        self,
        *,
        start: date,
        end: date,
        provider: str = "mock",
        model: Optional[str] = None,
        profile: str = "balanced",
        mock: bool = False,
        force_refresh: bool = False,
        skip_refresh: bool = True,
        skip_preprocess: bool = True,
        assets: Optional[list[str]] = None,
    ) -> list[LiveEvalResult]:
        """
        Daily rebalance over ``[start, end]`` decision dates.

        Each day D is scored with the next trading day as ex-post GT.
        Refresh/preprocess runs at most once before the loop.
        """
        if force_refresh or not skip_refresh:
            refresh_and_preprocess(
                force=True,
                skip_refresh=False,
                skip_preprocess=skip_preprocess,
            )
        elif not skip_preprocess:
            refresh_and_preprocess(skip_refresh=True, skip_preprocess=False)

        data = ProcessedDataProvider(data_dir=self.data_dir, sec_dir=self.sec_dir)
        pairs = iter_daily_decision_pairs(data, start, end)
        results: list[LiveEvalResult] = []
        for decision, realization in pairs:
            results.append(
                self.run(
                    provider=provider,
                    model=model,
                    profile=profile,
                    decision_date=decision,
                    as_of_today=realization,
                    mock=mock,
                    force_refresh=False,
                    skip_refresh=True,
                    skip_preprocess=True,
                    assets=assets,
                )
            )

        # Write a small window summary next to the model folder
        if results:
            provider_name = results[0].provider
            model_name = results[0].model.replace("/", "-")
            summary_dir = self.output_root / provider_name / model_name
            summary_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "decision_date": r.decision_date.isoformat(),
                    "realization_date": r.today.isoformat(),
                    "profile": r.profile,
                    "ceps_lookback": r.scores["lookback"]["ceps"],
                    "ceps_ex_post": r.scores["ex_post"]["ceps"],
                }
                for r in results
            ]
            summary = {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "n_days": len(rows),
                "profile": profile,
                "mean_ceps_lookback": round(
                    sum(x["ceps_lookback"] for x in rows) / len(rows), 4
                ),
                "mean_ceps_ex_post": round(
                    sum(x["ceps_ex_post"] for x in rows) / len(rows), 4
                ),
                "days": rows,
            }
            tag = f"{start.isoformat()}_{end.isoformat()}_{profile}"
            (summary_dir / f"window_summary_{tag}.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        return results
