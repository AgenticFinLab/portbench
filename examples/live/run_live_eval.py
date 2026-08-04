"""
Thin CLI for PortBench live / rolling evaluation.

Reads configs/live/default.yaml by default (models/profiles aligned with
configs/experiments/default.yaml). CLI flags override YAML.

Examples:
    python examples/live/run_live_eval.py
    python examples/live/run_live_eval.py --provider dashscope --model qwen3.7-max --profile balanced
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.live import SUPPORTED_FREQUENCIES, LiveEvalRunner

_DEFAULT_CONFIG = Path("configs/live/default.yaml")


def _parse_date(value):
    if not value:
        return None
    if hasattr(value, "year"):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Live config must be a mapping: {path}")
    return raw


def _resolve_jobs(cfg: dict, args) -> list[dict]:
    profiles = list(cfg.get("profiles") or ["balanced"])
    if args.profile is not None:
        profiles = [args.profile]

    # CLI single-model override
    if args.mock or args.provider is not None or args.model is not None:
        provider = args.provider or ("mock" if args.mock else "mock")
        model = args.model
        mock = bool(args.mock or provider == "mock")
        return [
            {
                "provider": provider,
                "model": model,
                "baseline": None,
                "mock": mock,
                "profile": prof,
            }
            for prof in profiles
        ]

    models = cfg.get("models") or []
    jobs = []
    for m in models:
        if not isinstance(m, dict):
            continue
        for prof in profiles:
            jobs.append(
                {
                    "provider": m.get("provider"),
                    "model": m.get("model"),
                    "baseline": m.get("baseline"),
                    "mock": bool(m.get("mock", False)),
                    "profile": prof,
                }
            )
    if jobs:
        return jobs
    return [
        {
            "provider": "mock",
            "model": None,
            "baseline": None,
            "mock": True,
            "profile": "balanced",
        }
    ]


def _job_label(job: dict) -> str:
    if job.get("baseline"):
        return f"baseline/{job['baseline']}/{job['profile']}"
    if job.get("mock"):
        return f"mock/mock/{job['profile']}"
    return f"{job.get('provider')}/{job.get('model')}/{job['profile']}"


def main() -> int:
    p = argparse.ArgumentParser(
        description="PortBench live / rolling eval (YAML + CLI overrides)"
    )
    p.add_argument("--config", default=str(_DEFAULT_CONFIG))
    p.add_argument("--provider", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--rebalance", default=None, choices=list(SUPPORTED_FREQUENCIES))
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--decision-date", default=None)
    p.add_argument("--as-of-today", default=None)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--no-auto-refresh", action="store_true")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--sec-dir", default=None)
    p.add_argument("--output-root", default=None)
    args = p.parse_args()

    cfg = _load_yaml(Path(args.config))
    rebalance = (
        args.rebalance if args.rebalance is not None else cfg.get("rebalance", "daily")
    )
    start = args.start if args.start is not None else cfg.get("start")
    end = args.end if args.end is not None else cfg.get("end")
    decision_date = (
        args.decision_date
        if args.decision_date is not None
        else cfg.get("decision_date")
    )
    as_of_today = (
        args.as_of_today if args.as_of_today is not None else cfg.get("as_of_today")
    )
    data_dir = (
        args.data_dir
        if args.data_dir is not None
        else cfg.get("data_dir", "datasets/processed")
    )
    sec_dir = (
        args.sec_dir if args.sec_dir is not None else cfg.get("sec_dir", "datasets/sec")
    )
    output_root = (
        args.output_root
        if args.output_root is not None
        else cfg.get("output_root", "outputs/live")
    )
    force_refresh = bool(args.force_refresh or cfg.get("force_refresh", False))
    auto_refresh = False if args.no_auto_refresh else bool(cfg.get("auto_refresh", True))
    on_error = str(cfg.get("on_error", "isolate"))

    jobs = _resolve_jobs(cfg, args)
    print(f"[live] config={args.config}")
    print(
        f"[live] jobs={len(jobs)} rebalance={rebalance} start={start} end={end} "
        f"on_error={on_error}"
    )

    runner = LiveEvalRunner(
        data_dir=data_dir,
        sec_dir=sec_dir,
        output_root=output_root,
        lookback_days=int(cfg.get("lookback_days", 60)),
        initial_nav=float(cfg.get("initial_nav", 1_000_000)),
        propagation_weight=float(cfg.get("propagation_weight", 0.1)),
    )

    batch_rows = []
    refreshed_once = False

    for i, job in enumerate(jobs):
        label = _job_label(job)
        print(f"\n[live] ({i+1}/{len(jobs)}) {label}")
        common = dict(
            provider=job.get("provider") or "mock",
            model=job.get("model"),
            baseline=job.get("baseline"),
            profile=job["profile"],
            mock=bool(job.get("mock")),
            force_refresh=force_refresh and not refreshed_once,
            auto_refresh=auto_refresh and not refreshed_once,
            skip_refresh=True,
            skip_preprocess=True,
        )
        try:
            if start or end:
                if not (start and end):
                    p.error("Range mode requires both start and end")
                result = runner.run_range(
                    start=_parse_date(start),
                    end=_parse_date(end),
                    rebalance=rebalance,
                    **common,
                )
            else:
                result = runner.run(
                    decision_date=_parse_date(decision_date),
                    as_of_today=_parse_date(as_of_today),
                    rebalance=rebalance,
                    **common,
                )
            refreshed_once = True
            if hasattr(result, "summary_path"):
                summary = json.loads(
                    Path(result.summary_path).read_text(encoding="utf-8")
                )
                print(
                    f"  episodes={summary.get('n_episodes')} "
                    f"mean_lookback={summary.get('mean_ceps_lookback')} "
                    f"mean_ex_post={summary.get('mean_ceps_ex_post')}"
                )
                batch_rows.append(
                    {
                        "job": label,
                        "ok": True,
                        "summary_path": result.summary_path,
                        "mean_ceps_lookback": summary.get("mean_ceps_lookback"),
                        "mean_ceps_ex_post": summary.get("mean_ceps_ex_post"),
                        "n_episodes": summary.get("n_episodes"),
                    }
                )
            else:
                print(
                    f"  CEPS lookback={result.scores['lookback']['ceps']} "
                    f"ex_post={result.scores['ex_post']['ceps']}"
                )
                batch_rows.append(
                    {
                        "job": label,
                        "ok": True,
                        "output_dir": result.output_dir,
                        "ceps_lookback": result.scores["lookback"]["ceps"],
                        "ceps_ex_post": result.scores["ex_post"]["ceps"],
                    }
                )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            batch_rows.append({"job": label, "ok": False, "error": str(exc)})
            traceback.print_exc()
            refreshed_once = True
            if on_error == "fail_fast":
                return 1

    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    tag = "batch"
    if start and end:
        tag = f"{rebalance}_{start}_{end}"
    batch_path = out / f"{tag}_batch_summary.json"
    batch_path.write_text(
        json.dumps(
            {
                "config": args.config,
                "rebalance": rebalance,
                "start": start,
                "end": end,
                "n_jobs": len(jobs),
                "n_ok": sum(1 for r in batch_rows if r.get("ok")),
                "n_failed": sum(1 for r in batch_rows if not r.get("ok")),
                "jobs": batch_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[live] batch summary → {batch_path}")
    print(
        f"[live] ok={sum(1 for r in batch_rows if r.get('ok'))} "
        f"failed={sum(1 for r in batch_rows if not r.get('ok'))}"
    )
    return 0 if all(r.get("ok") for r in batch_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
