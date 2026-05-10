"""
BatchRunner: sweep (model × profile × stress scenario) and persist all artifacts.

Directory layout (new):
  EXPERIMENTS/{rebalance}/{provider}/{model}/{timestamp}/{profile}/{scenario}/

One "run" = one model across all profiles (same timestamp directory).
Parallelism: parallel_experiments controls concurrent model runs.
             workers_per_experiment controls concurrent stress scenarios per profile.

Failure isolation: per (provider, model_name). Failure in one model does not
abort others. Errors are written to errors.jsonl inside the run_dir.
"""

from __future__ import annotations

import json
import logging
import matplotlib
matplotlib.use("Agg")
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from tqdm.auto import tqdm

from ..agent_eval.base import AgentAdapter
from ..agent_eval.investor_profiles import PROFILES
from ..agent_eval.stress_scenarios import STRESS_SCENARIOS
from ..qa_builder.mock_data import MockDataProvider
from ..qa_builder.processed_data import ProcessedDataProvider
from ..sandbox import BacktestEngine
from . import paths
from .config import ExperimentConfig, ModelSpec
from .figures import (
    render_dataset_correlation_figures,
    render_experiment_figures,
    render_batch_comparison_figures,
)
from .providers import (
    build_adapter,
    build_baseline,
    build_mock,
    spec_provider_name,
    spec_model_name,
)


_STRESS_BY_NAME = {s.name: s for s in STRESS_SCENARIOS}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_strategy(
    spec: ModelSpec, noise: float, seed: int, timeout: float = 120.0
) -> AgentAdapter:
    kind = spec.kind()
    if kind == "baseline":
        return build_baseline(spec.baseline)  # type: ignore[arg-type]
    if kind == "mock":
        return build_mock(noise=noise, seed=seed)
    return build_adapter(spec.provider, spec.model, timeout=timeout)  # type: ignore[arg-type]


def _resolve_model_name(spec: ModelSpec) -> str:
    """Resolve the actual model name string (reading from env if needed)."""
    if spec.baseline or spec.mock:
        return spec_model_name(spec)
    from .providers import PROVIDER_REGISTRY, _env
    model = spec.model
    if not model:
        prov_spec = PROVIDER_REGISTRY[spec.provider.lower()]
        model = _env(prov_spec.env_prefix, "MODEL") or "default"
    return model.replace("/", "_").replace(":", "_")


def _build_provider(cfg: ExperimentConfig):
    if cfg.data_provider == "mock":
        return MockDataProvider(seed=cfg.seed)
    if cfg.data_provider == "processed":
        if not (Path(cfg.data_dir) / "equities.csv").exists():
            raise RuntimeError(
                f"ProcessedDataProvider requested but {cfg.data_dir}/equities.csv missing. "
                "No mock fallback in batch mode — set data_provider: mock explicitly."
            )
        return ProcessedDataProvider(data_dir=cfg.data_dir, sec_dir=cfg.sec_dir)
    raise ValueError(f"Unknown data_provider: {cfg.data_provider!r}")


def _build_asset_class_map(provider) -> dict[str, str]:
    classes = ["equities", "bonds", "commodities", "real_estate", "cryptocurrency", "cash"]
    out: dict[str, str] = {}
    for cls in classes:
        try:
            for a in provider.list_assets(cls):
                out[a] = cls
        except Exception:
            continue
    return out


def _make_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"portbench.exp.{name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(f"[{name}] %(message)s"))
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Per-scenario and per-profile runners
# ---------------------------------------------------------------------------

def _run_one_scenario(
    cfg: ExperimentConfig,
    spec: ModelSpec,
    adapter: AgentAdapter,
    provider,
    asset_class_map: dict[str, str],
    profile_obj,
    scenario_name: str,
    out_dir: Path,
):
    """Run one stress scenario; persist artifacts; return (result, passed).

    If backtest_result.json already exists in out_dir, the scenario is skipped
    and its result is loaded from disk (scenario-level resume).
    """
    from ..sandbox.result import BacktestResult as _BR

    # ── Scenario-level cache hit ────────────────────────────────────────────
    cached_result_path = out_dir / "backtest_result.json"
    if cached_result_path.exists():
        try:
            data = json.loads(cached_result_path.read_text(encoding="utf-8"))
            result = _BR.from_dict(data)
            passed = bool(data.get("stress_passed", False))
            return result, passed
        except Exception:
            pass  # corrupt cache — fall through and re-run

    scenario = _STRESS_BY_NAME[scenario_name]
    use_pipeline = spec.kind() != "baseline"
    pipeline_log_dir = (
        out_dir / "pipeline_logs"
        if (cfg.logging.save_pipeline_logs and use_pipeline)
        else None
    )
    snapshot_dir = (out_dir / "snapshots") if cfg.logging.save_snapshots else None
    step_cache_dir = (out_dir / "step_cache") if use_pipeline else None

    engine = BacktestEngine(
        strategy=adapter,
        provider=provider,
        start_date=scenario.start,
        end_date=scenario.end,
        rebalance_freq="weekly",
        initial_nav=cfg.initial_nav,
        use_pipeline=use_pipeline,
        use_tools=cfg.use_tools and use_pipeline,
        profile=profile_obj,
        asset_class_map=asset_class_map,
        snapshot_dump_dir=str(snapshot_dir) if snapshot_dir else None,
        propagation_weight=cfg.propagation_weight,
        step_cache_dir=str(step_cache_dir) if step_cache_dir else None,
    )
    if pipeline_log_dir is not None:
        engine.enable_pipeline_logging(
            output_dir=str(pipeline_log_dir),
            run_id=scenario_name,
        )

    result = engine.run()
    passed = abs(result.max_drawdown) <= profile_obj.max_drawdown_tolerance
    result.stress_passed = passed
    paths.save_backtest_result(result, out_dir)
    return result, passed


def _run_normal(
    cfg: ExperimentConfig,
    spec: ModelSpec,
    adapter: AgentAdapter,
    provider,
    asset_class_map: dict[str, str],
    profile_obj,
    out_dir: Path,
):
    from ..sandbox.result import BacktestResult as _BR

    # ── Scenario-level cache hit ────────────────────────────────────────────
    cached_result_path = out_dir / "backtest_result.json"
    if cached_result_path.exists():
        try:
            data = json.loads(cached_result_path.read_text(encoding="utf-8"))
            return _BR.from_dict(data)
        except Exception:
            pass  # corrupt cache — re-run

    use_pipeline = spec.kind() != "baseline"
    pipeline_log_dir = (
        out_dir / "pipeline_logs"
        if (cfg.logging.save_pipeline_logs and use_pipeline)
        else None
    )
    snapshot_dir = (out_dir / "snapshots") if cfg.logging.save_snapshots else None
    step_cache_dir = (out_dir / "step_cache") if use_pipeline else None

    engine = BacktestEngine(
        strategy=adapter,
        provider=provider,
        start_date=cfg.normal_period.start,
        end_date=cfg.normal_period.end,
        rebalance_freq=cfg.rebalance,
        initial_nav=cfg.initial_nav,
        use_pipeline=use_pipeline,
        use_tools=cfg.use_tools and use_pipeline,
        profile=profile_obj,
        asset_class_map=asset_class_map,
        snapshot_dump_dir=str(snapshot_dir) if snapshot_dir else None,
        propagation_weight=cfg.propagation_weight,
        step_cache_dir=str(step_cache_dir) if step_cache_dir else None,
    )
    if pipeline_log_dir is not None:
        engine.enable_pipeline_logging(
            output_dir=str(pipeline_log_dir),
            run_id="normal",
        )
    result = engine.run()
    paths.save_backtest_result(result, out_dir)
    return result


def _run_profile(
    cfg: ExperimentConfig,
    spec: ModelSpec,
    adapter: AgentAdapter,
    provider,
    asset_class_map: dict[str, str],
    profile_name: str,
    p_dir: Path,
    logger: logging.Logger,
    model_label: str = "",
) -> dict:
    """Run Phase A + Phase B for one (model, profile). Returns summary dict."""
    profile_obj = PROFILES[profile_name]
    logger.info("Starting profile=%s", profile_name)
    logger.info(
        "Profile constraints: max_equity=%s min_bond_cash=%s max_dd=%s",
        profile_obj.max_equity_weight,
        profile_obj.min_bond_cash_weight,
        profile_obj.max_drawdown_tolerance,
    )

    stress_summaries: list[dict] = []
    all_passed = True
    scenarios = cfg.resolved_stress_scenarios()
    logger.info("Phase A: %d stress scenarios", len(scenarios))

    workers = max(1, min(cfg.workers_per_experiment, len(scenarios)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _run_one_scenario,
                cfg, spec, adapter, provider, asset_class_map, profile_obj,
                sc_name,
                p_dir / f"stress_{sc_name}",
            ): sc_name
            for sc_name in scenarios
        }
        results: dict[str, tuple] = {}
        for fut in as_completed(futs):
            sc_name = futs[fut]
            results[sc_name] = fut.result()  # raises → caller catches

    for sc_name in scenarios:
        result, passed = results[sc_name]
        status = "PASSED" if passed else "FAILED"
        logger.info(
            "  %s: %s drawdown=%.2f%% tol=%.0f%%",
            sc_name, status,
            result.max_drawdown * 100,
            profile_obj.max_drawdown_tolerance * 100,
        )
        stress_summaries.append({
            "scenario": sc_name,
            "passed": passed,
            "max_drawdown": round(result.max_drawdown, 4),
            "tolerance": profile_obj.max_drawdown_tolerance,
            "total_return": round(result.total_return, 4),
        })
        all_passed = all_passed and passed

    normal_dict = None
    if cfg.run_normal and all_passed:
        logger.info(
            "Phase B: normal backtest %s → %s",
            cfg.normal_period.start, cfg.normal_period.end,
        )
        normal_result = _run_normal(
            cfg, spec, adapter, provider, asset_class_map, profile_obj,
            p_dir / "normal",
        )
        normal_dict = normal_result.to_dict()
        logger.info(
            "  return=%+.2f%% sharpe=%.3f ceps=%.4f alignment=%.4f",
            normal_result.total_return * 100,
            normal_result.sharpe_ratio,
            normal_result.mean_ceps,
            normal_result.mean_profile_score,
        )
    elif not cfg.run_normal:
        logger.info("Phase B: skipped (run_normal=false)")
    else:
        logger.info("Phase B: skipped (stress gate FAILED)")

    if cfg.logging.save_figures:
        try:
            render_experiment_figures(p_dir, model_label or profile_name, profile_name, logger=logger)
        except Exception as exc:
            logger.warning("figure rendering failed: %s", exc)

    return {
        "stress_gate_passed": all_passed,
        "stress_results": stress_summaries,
        "normal": normal_dict,
    }


# ---------------------------------------------------------------------------
# Per-model runner (all profiles for one model, one timestamp)
# ---------------------------------------------------------------------------

def _run_one_model(
    cfg: ExperimentConfig,
    spec: ModelSpec,
    provider,
    asset_class_map: dict[str, str],
    prov_name: str,
    model_name: str,
    timestamp: str,
    errors_lock: threading.Lock,
    already_done: list[str] | None = None,
) -> dict:
    """
    Run all pending profiles for one (provider, model_name) and persist artifacts.

    already_done: profiles that completed in a previous partial run; they are
                  skipped and their results are loaded from the existing
                  run_summary.json so the final summary stays coherent.

    Returns {profile_name: summary_dict} for ALL profiles (old + new).
    """
    r_dir = paths.run_dir(cfg.output_root, cfg.rebalance, prov_name, model_name, timestamp)
    r_dir.mkdir(parents=True, exist_ok=True)

    logger = _make_logger(
        f"{prov_name}/{model_name}",
        r_dir / "runner.log",
    )

    already_done_set = set(already_done or [])
    pending = [p for p in cfg.profiles if p not in already_done_set]

    if already_done_set:
        logger.info(
            "Resuming run ts=%s — skipping %d already-complete profile(s): %s",
            timestamp, len(already_done_set), ", ".join(sorted(already_done_set)),
        )
    else:
        logger.info("Model run started: provider=%s model=%s ts=%s", prov_name, model_name, timestamp)

    # Pre-load results for already-completed profiles from existing run_summary.json
    profile_results: dict[str, dict] = {}
    if already_done_set:
        existing_summary = r_dir / "run_summary.json"
        if existing_summary.exists():
            try:
                existing = json.loads(existing_summary.read_text(encoding="utf-8"))
                profile_results = existing.get("profiles", {})
            except Exception:
                pass  # summary missing or corrupt — completed profiles have no in-memory results

    adapter = _build_strategy(spec, noise=cfg.noise, seed=cfg.seed, timeout=cfg.timeout)

    errors_path = r_dir / "errors.jsonl"
    completed = list(already_done or [])

    for profile_name in pending:
        p_dir = r_dir / profile_name
        p_dir.mkdir(parents=True, exist_ok=True)
        log_path = p_dir / "experiment.log"
        p_logger = _make_logger(f"{prov_name}/{model_name}/{profile_name}", log_path)

        try:
            summary = _run_profile(
                cfg, spec, adapter, provider, asset_class_map,
                profile_name, p_dir, p_logger,
                model_label=f"{prov_name}/{model_name}",
            )
            profile_results[profile_name] = summary
            completed.append(profile_name)

            # Update checkpoint after each completed profile
            (r_dir / "checkpoint.json").write_text(
                json.dumps({"completed": completed, "updated_at": datetime.now().isoformat()},
                           indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            tb = traceback.format_exc()
            p_logger.error("Profile failed: %s", exc)
            with errors_lock:
                paths.append_jsonl(errors_path, {
                    "timestamp": datetime.now().isoformat(),
                    "provider": prov_name,
                    "model": model_name,
                    "profile": profile_name,
                    "error": str(exc),
                    "traceback": tb,
                })
            paths.write_error(p_dir, {"error": str(exc), "traceback": tb})
            if cfg.on_error == "fail_fast":
                raise

    return profile_results


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------

def _write_run_summary(
    r_dir: Path,
    prov_name: str,
    model_name: str,
    timestamp: str,
    rebalance: str,
    profile_results: dict,
    elapsed: float,
) -> None:
    n_done = len(profile_results)
    out = {
        "provider": prov_name,
        "model_name": model_name,
        "rebalance": rebalance,
        "run_id": timestamp,
        "n_completed": n_done,
        "elapsed_seconds": round(elapsed, 2),
        "profiles": profile_results,
    }
    (r_dir / "run_summary.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# BatchRunner
# ---------------------------------------------------------------------------

class BatchRunner:
    def __init__(self, cfg: ExperimentConfig, raw_yaml: Optional[str] = None):
        self.cfg = cfg
        self._raw_yaml = raw_yaml

    def dry_run(self) -> list[dict]:
        """Return the (provider, model, profile, scenario) matrix without running."""
        out = []
        scenarios = self.cfg.resolved_stress_scenarios()
        for spec in self.cfg.models:
            prov = spec_provider_name(spec)
            model = _resolve_model_name(spec)
            for profile in self.cfg.profiles:
                for sc in scenarios:
                    out.append({"provider": prov, "model": model, "profile": profile, "scenario": sc})
                if self.cfg.run_normal:
                    out.append({"provider": prov, "model": model, "profile": profile, "scenario": "normal"})
        return out

    def run(self) -> dict:
        cfg = self.cfg
        rebal_dir = paths.rebalance_dir(cfg.output_root, cfg.rebalance)
        rebal_dir.mkdir(parents=True, exist_ok=True)

        # Save config snapshot in the rebalance dir
        cfg_snapshot = rebal_dir / "_last_run_config.yaml"
        if self._raw_yaml:
            cfg_snapshot.write_text(self._raw_yaml, encoding="utf-8")

        # Dataset-level figures (shared, independent of model or rebalance)
        if cfg.logging.save_figures:
            render_dataset_correlation_figures(
                output_dir=paths.dataset_figures_dir(cfg.output_root),
                processed_dir=Path(cfg.data_dir),
            )

        data_provider = _build_provider(cfg)
        asset_class_map = _build_asset_class_map(data_provider)

        # Resolve (provider, model_name) for every spec
        model_specs: list[tuple[ModelSpec, str, str]] = []  # (spec, prov_name, model_name)
        for spec in cfg.models:
            prov_name = spec_provider_name(spec)
            model_name = _resolve_model_name(spec)
            model_specs.append((spec, prov_name, model_name))

        # Decide which models to run vs. resume vs. fully reuse
        # Each entry: (spec, prov, model, timestamp, already_done_profiles)
        to_run: list[tuple[ModelSpec, str, str, str, list[str]]] = []
        reused: list[tuple[str, str, str]] = []  # (prov, model, timestamp)
        run_timestamps: dict[tuple, str] = {}  # (prov, model) → timestamp

        for spec, prov_name, model_name in model_specs:
            if cfg.reuse_latest:
                ts = paths.find_best_run(
                    cfg.output_root, cfg.rebalance, prov_name, model_name, cfg.profiles
                )
                if ts:
                    r_dir = paths.run_dir(cfg.output_root, cfg.rebalance, prov_name, model_name, ts)
                    done = paths.get_completed_profiles(r_dir, cfg.profiles)
                    if set(done) >= set(cfg.profiles):
                        # All profiles complete — fully reuse this run
                        print(
                            f"[reuse]  {prov_name}/{model_name}: "
                            f"all {len(cfg.profiles)} profiles complete, using run {ts}"
                        )
                        reused.append((prov_name, model_name, ts))
                        run_timestamps[(prov_name, model_name)] = ts
                        continue
                    else:
                        # Partial run — resume from same timestamp
                        missing = [p for p in cfg.profiles if p not in done]
                        print(
                            f"[resume] {prov_name}/{model_name}: "
                            f"{len(done)}/{len(cfg.profiles)} profiles done in run {ts}, "
                            f"continuing: {', '.join(missing)}"
                        )
                        to_run.append((spec, prov_name, model_name, ts, done))
                        continue
            # No existing run (or reuse_latest=False) — fresh timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            to_run.append((spec, prov_name, model_name, timestamp, []))

        # ── Run pending models ──────────────────────────────────────────────
        errors_lock = threading.Lock()
        n_total = len(to_run)
        n_done = 0
        n_failed = 0

        # Pre-populate timestamps for reused runs
        for prov_name, model_name, ts in reused:
            run_timestamps[(prov_name, model_name)] = ts

        pbar = tqdm(
            total=n_total,
            desc=f"batch [{cfg.rebalance}]",
            unit="model",
            dynamic_ncols=True,
        )

        def _task(spec, prov_name, model_name, timestamp, already_done):
            nonlocal n_done, n_failed
            t0 = time.time()
            r_dir = paths.run_dir(cfg.output_root, cfg.rebalance, prov_name, model_name, timestamp)
            try:
                profile_results = _run_one_model(
                    cfg, spec, data_provider, asset_class_map,
                    prov_name, model_name, timestamp, errors_lock,
                    already_done=already_done,
                )
                _write_run_summary(
                    r_dir, prov_name, model_name, timestamp, cfg.rebalance,
                    profile_results, time.time() - t0,
                )
                run_timestamps[(prov_name, model_name)] = timestamp
                n_done += 1
            except Exception as exc:
                tb = traceback.format_exc()
                with errors_lock:
                    paths.append_jsonl(r_dir / "errors.jsonl", {
                        "timestamp": datetime.now().isoformat(),
                        "provider": prov_name,
                        "model": model_name,
                        "error": str(exc),
                        "traceback": tb,
                    })
                n_failed += 1
                if cfg.on_error == "fail_fast":
                    raise
            finally:
                pbar.update(1)
                pbar.set_postfix(done=n_done, failed=n_failed)

        max_workers = max(1, cfg.parallel_experiments)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(_task, spec, prov, model, ts, done): (prov, model)
                for spec, prov, model, ts, done in to_run
            }
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception:
                    if cfg.on_error == "fail_fast":
                        pbar.close()
                        raise

        pbar.close()

        # ── Save env metadata ───────────────────────────────────────────────
        import subprocess
        import sys as _sys
        try:
            git_hash = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip() or "unknown"
        except Exception:
            git_hash = "unknown"
        (rebal_dir / "_env_meta.json").write_text(
            json.dumps({
                "rebalance": cfg.rebalance,
                "git_hash": git_hash,
                "python": _sys.version,
                "updated_at": datetime.now().isoformat(),
            }, indent=2),
            encoding="utf-8",
        )

        # ── Batch-level comparison figures ──────────────────────────────────
        if cfg.logging.save_figures and run_timestamps:
            try:
                render_batch_comparison_figures(
                    rebal_dir,
                    run_timestamps={
                        f"{p}/{m}": ts for (p, m), ts in run_timestamps.items()
                    },
                    output_root=cfg.output_root,
                    rebalance=cfg.rebalance,
                )
                print(f"[figures] comparison figures → {rebal_dir / 'comparison_figures'}")
            except Exception as exc:
                print(f"[figures] batch comparison failed: {exc}")

        # ── QA evaluation ───────────────────────────────────────────────────
        if cfg.run_qa:
            from ..qa_eval.evaluator import QAEvaluator
            try:
                qa_eval = QAEvaluator(cfg)
                qa_summary = qa_eval.run()
            except Exception as exc:
                print(f"[QA] evaluation failed: {exc}")

        return {
            "rebalance": cfg.rebalance,
            "n_completed": n_done,
            "n_reused": len(reused),
            "n_resumed": sum(1 for *_, done in to_run if done),
            "n_failed": n_failed,
            "run_timestamps": {f"{p}/{m}": ts for (p, m), ts in run_timestamps.items()},
        }
