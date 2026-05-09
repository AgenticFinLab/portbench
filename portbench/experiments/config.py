"""
ExperimentConfig: typed schema for batch experiments + YAML loader.

YAML schema example:

    batch_id: tencent_vs_dashscope_apr25
    data_provider: processed                 # processed | mock
    data_dir: datasets/processed
    sec_dir: datasets/sec
    rebalance: monthly                       # weekly | monthly | quarterly
    initial_nav: 1000000
    workers_per_experiment: 3
    parallel_experiments: 1
    seed: 42
    noise: 0.2

    models:
      - provider: dashscope                  # model omitted → use DASHSCOPE_MODEL
      - provider: tencent
        model: hunyuan-pro
      - baseline: equal_weight
      - mock: true                           # uses MockAgentAdapter

    profiles: [conservative, balanced, aggressive]
    stress_scenarios: all                    # all | [name1, name2]
    run_normal: true
    normal_period:
      start: 2024-01-01
      end: 2024-12-31

    logging:
      save_pipeline_logs: true
      save_snapshots: true
      save_figures: true

    on_error: isolate                        # isolate | fail_fast
    use_tools: false                         # true = S1/S2/S3 call complete_with_tools()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

import yaml


@dataclass
class ModelSpec:
    """One element of the `models:` list."""

    provider: Optional[str] = None  # for LLM providers
    model: Optional[str] = None  # optional override of {PREFIX}_MODEL
    baseline: Optional[str] = None  # for baseline strategies
    mock: bool = False  # MockAgentAdapter

    def kind(self) -> str:
        if self.baseline:
            return "baseline"
        if self.mock:
            return "mock"
        if self.provider:
            return "llm"
        raise ValueError(f"Invalid ModelSpec: {self!r}")


@dataclass
class LoggingConfig:
    save_pipeline_logs: bool = True
    save_snapshots: bool = True
    save_figures: bool = True


@dataclass
class NormalPeriod:
    start: date = date(2024, 1, 1)
    end: date = date(2024, 12, 31)


@dataclass
class QAConfig:
    dataset_path: str = "datasets/qa_dataset"
    split: str = "test"  # train | val | test | all
    templates: list[str] = field(
        default_factory=lambda: ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    )
    max_pairs_per_template: int = 50
    parallel_questions: int = 4
    save_responses: bool = True


@dataclass
class ExperimentConfig:
    batch_id: str
    models: list[ModelSpec] = field(default_factory=list)
    profiles: list[str] = field(
        default_factory=lambda: ["conservative", "balanced", "aggressive"]
    )
    stress_scenarios: Union[str, list[str]] = "all"  # "all" or list of names
    run_normal: bool = True
    normal_period: NormalPeriod = field(default_factory=NormalPeriod)

    data_provider: str = "processed"
    data_dir: str = "datasets/processed"
    sec_dir: str = "datasets/sec"
    rebalance: str = "monthly"
    initial_nav: float = 1_000_000.0
    workers_per_experiment: int = 3
    parallel_experiments: int = 1
    seed: int = 42
    noise: float = 0.2
    use_tools: bool = False
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    on_error: str = "isolate"
    output_root: str = "EXPERIMENTS"
    propagation_weight: float = 0.1  # CEPS cascade penalty weight
    resume: bool = False  # skip already-completed (model, profile) pairs
    run_qa: bool = False  # run QA dataset evaluation alongside sandbox
    qa: QAConfig = field(default_factory=QAConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return ExperimentConfig.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict) -> "ExperimentConfig":
        models = [ModelSpec(**m) for m in (raw.get("models") or [])]
        if not models:
            raise ValueError("ExperimentConfig.models must be a non-empty list")

        normal_raw = raw.get("normal_period") or {}
        np_obj = NormalPeriod(
            start=_to_date(normal_raw.get("start", "2024-01-01")),
            end=_to_date(normal_raw.get("end", "2024-12-31")),
        )

        log_raw = raw.get("logging") or {}
        log_obj = LoggingConfig(
            **{
                k: v
                for k, v in log_raw.items()
                if k in {"save_pipeline_logs", "save_snapshots", "save_figures"}
            }
        )

        batch_id_raw = raw.get("batch_id") or "{models}_{date}"
        batch_id = _expand_batch_id(batch_id_raw, models)

        return ExperimentConfig(
            batch_id=batch_id,
            models=models,
            profiles=list(
                raw.get("profiles") or ["conservative", "balanced", "aggressive"]
            ),
            stress_scenarios=raw.get("stress_scenarios", "all"),
            run_normal=bool(raw.get("run_normal", True)),
            normal_period=np_obj,
            data_provider=raw.get("data_provider", "processed"),
            data_dir=raw.get("data_dir", "datasets/processed"),
            sec_dir=raw.get("sec_dir", "datasets/sec"),
            rebalance=raw.get("rebalance", "monthly"),
            initial_nav=float(raw.get("initial_nav", 1_000_000.0)),
            workers_per_experiment=int(raw.get("workers_per_experiment", 3)),
            parallel_experiments=int(raw.get("parallel_experiments", 1)),
            seed=int(raw.get("seed", 42)),
            noise=float(raw.get("noise", 0.2)),
            logging=log_obj,
            on_error=raw.get("on_error", "isolate"),
            output_root=raw.get("output_root", "EXPERIMENTS"),
            use_tools=bool(raw.get("use_tools", False)),
            propagation_weight=float(raw.get("propagation_weight", 0.1)),
            resume=bool(raw.get("resume", False)),
            run_qa=bool(raw.get("run_qa", False)),
            qa=_parse_qa_config(raw.get("qa") or {}),
        )

    def resolved_stress_scenarios(self) -> list[str]:
        """Return the list of scenario names to run (resolves 'all')."""
        from ..agent_eval.stress_scenarios import STRESS_SCENARIOS

        if self.stress_scenarios == "all":
            return [s.name for s in STRESS_SCENARIOS]
        if not isinstance(self.stress_scenarios, list):
            raise ValueError(
                f"stress_scenarios must be 'all' or a list, got {self.stress_scenarios!r}"
            )
        return list(self.stress_scenarios)


def _to_date(value) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d").date()
    raise ValueError(f"Cannot parse date: {value!r}")


def _expand_batch_id(template: str, models: list[ModelSpec]) -> str:
    """
    Expand batch_id template variables:
      {models}  — abbreviated model labels joined by "_" (max 3 models shown)
      {date}    — YYYYMMDD
      {time}    — HHMMSS

    Examples:
      "{models}_{date}"          → "ark-doubao_tencent_20250509"
      "exp_{date}_{time}"        → "exp_20250509_143022"
      "ablation_{models}_{date}" → "ablation_ark-doubao_20250509"
    """
    if "{" not in template:
        return template

    now = datetime.now()

    def _label(spec: ModelSpec) -> str:
        if spec.baseline:
            return spec.baseline.replace("_", "-")
        if spec.mock:
            return "mock"
        # provider + shortened model: take last segment after - or /
        provider = (spec.provider or "").lower()
        model = (spec.model or "").lower()
        if model:
            # take first two dash-segments for readability: doubao-seed-2-0-pro → doubao-seed
            parts = model.split("-")
            short = "-".join(parts[:2]) if len(parts) >= 2 else parts[0]
            short = short[:16]
            return f"{provider}-{short}"
        return provider

    labels = [_label(m) for m in models]
    if len(labels) > 3:
        models_str = "_".join(labels[:3]) + f"_and{len(labels) - 3}more"
    else:
        models_str = "_".join(labels)

    # Replace special chars unsafe for directory names
    models_str = models_str.replace("/", "-").replace(":", "-").replace(" ", "-")

    return (
        template
        .replace("{models}", models_str)
        .replace("{date}", now.strftime("%Y%m%d"))
        .replace("{time}", now.strftime("%H%M%S"))
    )


def _parse_qa_config(raw: dict) -> QAConfig:
    return QAConfig(
        dataset_path=raw.get("dataset_path", "datasets/qa/qa_dataset.jsonl"),
        split=raw.get("split", "test"),
        templates=list(raw.get("templates", ["T1", "T2", "T3", "T4", "T5", "T6", "T7"])),
        max_pairs_per_template=int(raw.get("max_pairs_per_template", 50)),
        parallel_questions=int(raw.get("parallel_questions", 4)),
        save_responses=bool(raw.get("save_responses", True)),
    )
