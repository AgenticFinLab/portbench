"""
BatchRunner: sweep (model × profile × stress scenario) and persist all artifacts.

Failure isolation granularity: (model, profile). A failure in one combination
writes error.json + appends to errors.jsonl, then continues.

Each (model, profile) experiment runs:
  Phase A — stress gate: every scenario in cfg.resolved_stress_scenarios()
  Phase B — normal backtest: only when all stress scenarios passed AND cfg.run_normal
"""

from __future__ import annotations

import json
import logging
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
from .figures import render_dataset_correlation_figures, render_experiment_figures
from .providers import build_adapter, build_baseline, build_mock, model_label


_STRESS_BY_NAME = {s.name: s for s in STRESS_SCENARIOS}


def _build_strategy(spec: ModelSpec, noise: float, seed: int) -> AgentAdapter:
    kind = spec.kind()
    if kind == "baseline":
        return build_baseline(spec.baseline)  # type: ignore[arg-type]
    if kind == "mock":
        return build_mock(noise=noise, seed=seed)
    return build_adapter(spec.provider, spec.model)  # type: ignore[arg-type]


def _spec_label(spec: ModelSpec) -> str:
    kind = spec.kind()
    if kind == "baseline":
        return f"baseline-{spec.baseline}"
    if kind == "mock":
        return "mock"
    return model_label(spec.provider, spec.model)  # type: ignore[arg-type]


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
    classes = [
        "equities",
        "bonds",
        "commodities",
        "real_estate",
        "cryptocurrency",
        "cash",
    ]
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
    """Run one stress scenario; persist artifacts; return (result, passed)."""
    scenario = _STRESS_BY_NAME[scenario_name]
    use_pipeline = spec.kind() != "baseline"
    pipeline_log_dir = (
        out_dir / "pipeline_logs"
        if (cfg.logging.save_pipeline_logs and use_pipeline)
        else None
    )
    snapshot_dir = (out_dir / "snapshots") if cfg.logging.save_snapshots else None

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
    )
    if pipeline_log_dir is not None:
        engine.enable_pipeline_logging(
            output_dir=str(pipeline_log_dir),
            run_id=f"{scenario_name}",
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
    use_pipeline = spec.kind() != "baseline"
    pipeline_log_dir = (
        out_dir / "pipeline_logs"
        if (cfg.logging.save_pipeline_logs and use_pipeline)
        else None
    )
    snapshot_dir = (out_dir / "snapshots") if cfg.logging.save_snapshots else None

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
    )
    if pipeline_log_dir is not None:
        engine.enable_pipeline_logging(
            output_dir=str(pipeline_log_dir),
            run_id="normal",
        )
    result = engine.run()
    paths.save_backtest_result(result, out_dir)
    return result


def _run_profile_experiment(
    cfg: ExperimentConfig,
    spec: ModelSpec,
    adapter: AgentAdapter,
    provider,
    asset_class_map: dict[str, str],
    profile_name: str,
    label: str,
):
    """Run Phase A + Phase B for a single (model, profile). Returns summary dict."""
    profile_obj = PROFILES[profile_name]
    p_dir = paths.profile_dir(cfg.output_root, cfg.batch_id, label, profile_name)
    p_dir.mkdir(parents=True, exist_ok=True)
    logger = _make_logger(f"{label}/{profile_name}", p_dir / "experiment.log")
    logger.info("Starting (model=%s, profile=%s)", label, profile_name)
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
                cfg,
                spec,
                adapter,
                provider,
                asset_class_map,
                profile_obj,
                sc_name,
                paths.stress_dir(
                    cfg.output_root, cfg.batch_id, label, profile_name, sc_name
                ),
            ): sc_name
            for sc_name in scenarios
        }
        results: dict[str, tuple] = {}
        for fut in as_completed(futs):
            sc_name = futs[fut]
            results[sc_name] = fut.result()  # may raise → caller catches

    for sc_name in scenarios:
        result, passed = results[sc_name]
        status = "PASSED" if passed else "FAILED"
        logger.info(
            "  %s: %s drawdown=%.2f%% tol=%.0f%%",
            sc_name,
            status,
            result.max_drawdown * 100,
            profile_obj.max_drawdown_tolerance * 100,
        )
        stress_summaries.append(
            {
                "scenario": sc_name,
                "passed": passed,
                "max_drawdown": round(result.max_drawdown, 4),
                "tolerance": profile_obj.max_drawdown_tolerance,
                "total_return": round(result.total_return, 4),
            }
        )
        all_passed = all_passed and passed

    normal_dict = None
    if cfg.run_normal and all_passed:
        logger.info(
            "Phase B: normal backtest %s → %s",
            cfg.normal_period.start,
            cfg.normal_period.end,
        )
        n_dir = paths.normal_dir(cfg.output_root, cfg.batch_id, label, profile_name)
        normal_result = _run_normal(
            cfg, spec, adapter, provider, asset_class_map, profile_obj, n_dir
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
            render_experiment_figures(p_dir, label, profile_name, logger=logger)
        except Exception as exc:
            logger.warning("figure rendering failed: %s", exc)

    return {
        "stress_gate_passed": all_passed,
        "stress_results": stress_summaries,
        "normal": normal_dict,
    }


def _write_checkpoint(output_root: str, batch_id: str, completed_keys: set[str]) -> None:
    ckpt = paths.checkpoint_file(output_root, batch_id)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "completed": sorted(completed_keys),
                "updated_at": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _save_batch_config(cfg: ExperimentConfig, raw_yaml: Optional[str]) -> None:
    bd = paths.batch_dir(cfg.output_root, cfg.batch_id)
    bd.mkdir(parents=True, exist_ok=True)
    if raw_yaml:
        (bd / "batch_config.yaml").write_text(raw_yaml, encoding="utf-8")
    else:
        # Reconstruct YAML from the dataclass for in-memory configs
        from dataclasses import asdict

        (bd / "batch_config.yaml").write_text(
            yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8"
        )


class BatchRunner:
    def __init__(self, cfg: ExperimentConfig, raw_yaml: Optional[str] = None):
        self.cfg = cfg
        self._raw_yaml = raw_yaml
        self._completed_keys: set[str] = set()
        self._checkpoint_warned = False

        bd = paths.batch_dir(cfg.output_root, cfg.batch_id)
        ckpt = paths.checkpoint_file(cfg.output_root, cfg.batch_id)

        if bd.exists():
            if cfg.resume and ckpt.exists():
                data = json.loads(ckpt.read_text(encoding="utf-8"))
                self._completed_keys = set(data.get("completed", []))
                print(
                    f"[resume] batch_id='{cfg.batch_id}' — "
                    f"{len(self._completed_keys)} completed pairs will be skipped"
                )
            elif not cfg.resume:
                print(
                    f"[WARNING] Output directory already exists: {bd}\n"
                    "  Results will be overwritten. "
                    "Set resume: true to skip completed experiments instead."
                )

    def dry_run(self) -> list[dict]:
        """Return the (model, profile, scenario) matrix without running anything."""
        out = []
        scenarios = self.cfg.resolved_stress_scenarios()
        for spec in self.cfg.models:
            label = _spec_label(spec)
            for profile in self.cfg.profiles:
                for sc in scenarios:
                    out.append({"model": label, "profile": profile, "scenario": sc})
                if self.cfg.run_normal:
                    out.append(
                        {"model": label, "profile": profile, "scenario": "normal"}
                    )
        return out

    def run(self) -> dict:
        cfg = self.cfg
        _save_batch_config(cfg, self._raw_yaml)
        bd = paths.batch_dir(cfg.output_root, cfg.batch_id)
        errors_path = bd / "errors.jsonl"
        errors_lock = threading.Lock()
        checkpoint_lock = threading.Lock()

        # Capture environment metadata for reproducibility
        import subprocess
        import sys as _sys
        try:
            git_hash = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5
            ).stdout.strip() or "unknown"
        except Exception:
            git_hash = "unknown"
        env_meta = {
            "batch_id": cfg.batch_id,
            "git_hash": git_hash,
            "python": _sys.version,
            "created_at": datetime.now().isoformat(),
        }
        bd.mkdir(parents=True, exist_ok=True)
        (bd / "env_meta.json").write_text(
            json.dumps(env_meta, indent=2), encoding="utf-8"
        )

        provider = _build_provider(cfg)
        asset_class_map = _build_asset_class_map(provider)

        if cfg.logging.save_figures:
            render_dataset_correlation_figures(
                output_dir=bd / "_dataset_figures",
                processed_dir=Path(cfg.data_dir),
            )

        comparison_root: dict[str, dict[str, dict]] = {}
        comparison_lock = threading.Lock()
        t0 = time.time()
        n_total = len(cfg.models) * len(cfg.profiles)
        n_done = 0
        n_failed = 0

        pbar = tqdm(
            total=n_total,
            desc=f"batch {cfg.batch_id}",
            unit="exp",
            dynamic_ncols=True,
        )

        # Build adapters upfront; adapter build failures skip all profiles for that model
        adapter_map: dict[str, object] = {}
        for spec in cfg.models:
            label = _spec_label(spec)
            try:
                adapter_map[label] = _build_strategy(spec, noise=cfg.noise, seed=cfg.seed)
                comparison_root[label] = {}
            except Exception as exc:
                tb = traceback.format_exc()
                with errors_lock:
                    paths.append_jsonl(
                        errors_path,
                        {
                            "timestamp": datetime.now().isoformat(),
                            "model": label,
                            "profile": "*",
                            "stage": "build_adapter",
                            "error": str(exc),
                            "traceback": tb,
                        },
                    )
                paths.write_error(
                    paths.model_dir(cfg.output_root, cfg.batch_id, label),
                    {"stage": "build_adapter", "error": str(exc), "traceback": tb},
                )
                n_failed += len(cfg.profiles)
                pbar.update(len(cfg.profiles))
                pbar.set_postfix(done=n_done, failed=n_failed)
                if cfg.on_error == "fail_fast":
                    pbar.close()
                    raise

        def _run_task(spec: ModelSpec, label: str, profile_name: str) -> None:
            nonlocal n_done, n_failed
            ck = f"{label}:{profile_name}"
            if ck in self._completed_keys:
                print(f"[SKIP] {ck} (already completed)")
                pbar.update(1)
                return

            p_dir = paths.profile_dir(cfg.output_root, cfg.batch_id, label, profile_name)
            try:
                summary = _run_profile_experiment(
                    cfg, spec, adapter_map[label], provider, asset_class_map,
                    profile_name, label,
                )
                with comparison_lock:
                    comparison_root[label][profile_name] = summary
                    n_done += 1
                with checkpoint_lock:
                    self._completed_keys.add(ck)
                    _write_checkpoint(cfg.output_root, cfg.batch_id, self._completed_keys)
            except Exception as exc:
                tb = traceback.format_exc()
                with errors_lock:
                    paths.append_jsonl(
                        errors_path,
                        {
                            "timestamp": datetime.now().isoformat(),
                            "model": label,
                            "profile": profile_name,
                            "stage": "run_experiment",
                            "error": str(exc),
                            "traceback": tb,
                        },
                    )
                paths.write_error(
                    p_dir,
                    {"stage": "run_experiment", "error": str(exc), "traceback": tb},
                )
                with comparison_lock:
                    n_failed += 1
                if cfg.on_error == "fail_fast":
                    raise
            finally:
                pbar.update(1)
                pbar.set_postfix(done=n_done, failed=n_failed)

        # Submit all (model, profile) tasks; parallelism controlled by parallel_experiments
        tasks = [
            (spec, _spec_label(spec), profile_name)
            for spec in cfg.models
            if _spec_label(spec) in adapter_map
            for profile_name in cfg.profiles
        ]
        max_workers = max(1, cfg.parallel_experiments)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_run_task, s, l, p): (l, p) for s, l, p in tasks}
            for fut in as_completed(futs):
                label, profile_name = futs[fut]
                try:
                    fut.result()
                except Exception:
                    if cfg.on_error == "fail_fast":
                        pbar.close()
                        raise

        pbar.close()

        # Per-model profile_comparison.json (written after all profiles are done)
        for label in list(comparison_root.keys()):
            self._write_profile_comparison(label, comparison_root[label])

        summary = self._write_batch_summary(
            comparison_root, n_done, n_failed, time.time() - t0
        )

        # QA evaluation (if enabled)
        if cfg.run_qa:
            from ..qa_eval.evaluator import QAEvaluator
            try:
                qa_eval = QAEvaluator(cfg)
                qa_summary = qa_eval.run()
                summary["qa"] = qa_summary
            except Exception as exc:
                print(f"[QA] evaluation failed: {exc}")
                summary["qa"] = {"error": str(exc)}

        return summary

    # ------------------------------------------------------------------
    def _write_profile_comparison(self, label: str, by_profile: dict) -> None:
        cfg = self.cfg
        normal_returns = [
            v["normal"]["total_return"]
            for v in by_profile.values()
            if v.get("normal") is not None
        ]
        adaptation = float(np.std(normal_returns)) if len(normal_returns) > 1 else 0.0
        out = {
            "model_label": label,
            "batch_id": cfg.batch_id,
            "profiles": by_profile,
            "adaptation_score": round(adaptation, 4),
        }
        path = (
            paths.model_dir(cfg.output_root, cfg.batch_id, label)
            / "profile_comparison.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    def _write_batch_summary(
        self, comparison_root: dict, n_done: int, n_failed: int, elapsed: float
    ) -> dict:
        cfg = self.cfg
        rows: list[dict] = []
        for label, profiles in comparison_root.items():
            for pname, payload in profiles.items():
                base = {
                    "model": label,
                    "profile": pname,
                    "stress_gate_passed": payload["stress_gate_passed"],
                }
                for s in payload["stress_results"]:
                    rows.append({**base, "phase": "stress", **s})
                if payload.get("normal") is not None:
                    normal_row = dict(payload["normal"])
                    per_step = normal_row.pop("per_step_ceps", [])
                    normal_row["std_ceps"] = round(float(np.std(per_step)), 6) if per_step else 0.0
                    rows.append({**base, "phase": "normal", **normal_row})
        summary = {
            "batch_id": cfg.batch_id,
            "n_completed": n_done,
            "n_failed": n_failed,
            "elapsed_seconds": round(elapsed, 2),
            "rows": rows,
        }
        bd = paths.batch_dir(cfg.output_root, cfg.batch_id)
        (bd / "batch_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        return summary
