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
    complete_with_tools()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union


@dataclass
class GenerationConfig:
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class ResourceBudgetConfig:
    """Configure optional per-episode token and request ceilings.

    Set a ceiling to 0 to record usage without aborting the episode.
    """

    max_tokens_per_episode: int = 32000
    max_requests_per_episode: int = 24
    config_version: str = "iso-token-v3"


import yaml


@dataclass
class ModelSpec:
    """One element of the `models:` list."""

    provider: Optional[str] = None  # for LLM providers
    model: Optional[str] = None  # optional override of {PREFIX}_MODEL
    baseline: Optional[str] = None  # for baseline strategies
    mock: bool = False  # MockAgentAdapter
    temperature: Optional[float] = None  # overrides global generation.temperature
    max_tokens: Optional[int] = None  # overrides global generation.max_tokens
    parallel_questions: Optional[int] = None  # overrides global qa.parallel_questions for this model
    architecture_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.architecture_id is None:
            return
        from ..agent_eval.contracts import ARCHITECTURE_IDS

        # Fail during config loading instead of after a paid provider call.
        if self.architecture_id not in ARCHITECTURE_IDS:
            raise ValueError(f"unknown architecture_id={self.architecture_id!r}")

    def kind(self) -> str:
        if self.baseline:
            return "baseline"
        if self.mock:
            return "mock"
        if self.provider:
            return "llm"
        raise ValueError(f"Invalid ModelSpec: {self!r}")


@dataclass
class InterventionConfig:
    """Causal stage interventions applied after each factual episode.

    operator:
      repair  — replace the stage output with the Point-in-Time ground-truth reconstruction
      perturb — shock the model's factual output with the built-in +10% stance amplification
    mode:
      offline — no extra LLM; deterministic suffix; CEPS / stage-score deltas only (not NAV)
      online  — re-run the downstream agent; same CEPS/score-delta family, more expensive
    closed_loop:
      true — also fork NAV from the first rebalance of the window (portfolio-path effect).
              This NAV fork is always the cheap offline suffix, even when mode is online.
              Episode-level CEPS interventions still follow ``mode``.
    """

    enabled: bool = False
    stages: list[str] = field(
        default_factory=lambda: ["S1", "S2", "S3", "S4", "S5"]
    )
    operator: str = "repair"
    mode: str = "offline"
    closed_loop: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.operator not in {"repair", "perturb"}:
            raise ValueError(f"interventions.operator must be repair or perturb, got {self.operator!r}")
        if self.mode not in {"offline", "online"}:
            raise ValueError(f"interventions.mode must be offline or online, got {self.mode!r}")
        if self.closed_loop is None:
            self.closed_loop = self.mode == "offline"
        unknown = [stage for stage in self.stages if stage not in {"S1", "S2", "S3", "S4", "S5"}]
        if unknown:
            raise ValueError(f"unsupported intervention stages: {unknown}")

    def to_spec(self, propagation_weight: float = 0.1) -> dict:
        return {
            "enabled": bool(self.enabled),
            "stages": list(self.stages),
            "operator": self.operator,
            "mode": self.mode,
            "closed_loop": bool(self.closed_loop),
            "propagation_weight": float(propagation_weight),
        }


def _parse_intervention_config(raw: dict) -> InterventionConfig:
    if not raw:
        return InterventionConfig()
    closed = raw.get("closed_loop")
    return InterventionConfig(
        enabled=bool(raw.get("enabled", False)),
        stages=list(raw.get("stages") or ["S1", "S2", "S3", "S4", "S5"]),
        operator=str(raw.get("operator") or "repair"),
        mode=str(raw.get("mode") or "offline"),
        closed_loop=None if closed is None else bool(closed),
    )


@dataclass
class LoggingConfig:
    save_pipeline_logs: bool = True
    save_snapshots: bool = True
    save_figures: bool = True


@dataclass
class NormalPeriod:
    start: date = date(2024, 1, 1)
    end: date = date(2024, 12, 31)
    label: str = ""  # e.g. "bull_2024", "bear_2022"


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
    # "full" = original prompts; "restricted" = T4/T5 prompts with covariance info stripped;
    # "both" = run full first, then restricted (results stored in T4_restricted/ T5_restricted/)
    info_level: str = "full"
    # false = legacy T3/T4 (paper numbers); true = redesigned templates + explanation scoring
    t3t4_redesign: bool = False
    t3t4_numeric_weight: float = 0.7
    t3t4_explanation_weight: float = 0.3
    template_version: str = "legacy"
    scorer_version: str = "legacy-v1"
    call_max_attempts: int = 3
    retry_failed_calls: bool = False
    freeze_manifest: str = ""


@dataclass
class ExperimentConfig:
    # batch_id is a human-readable label stored in run metadata; no longer used for directory naming
    batch_id: str = ""
    models: list[ModelSpec] = field(default_factory=list)
    profiles: list[str] = field(
        default_factory=lambda: ["conservative", "balanced", "aggressive"]
    )
    stress_scenarios: Union[str, list[str]] = "all"  # "all" or list of names
    run_normal: bool = True
    normal_periods: list[NormalPeriod] = field(
        default_factory=lambda: [NormalPeriod()]
    )
    max_rebalances_per_window: int = 0
    factual_pit_prefix_stages: list[str] = field(default_factory=list)
    legacy_stage_reuse_root: str = ""
    oracle_mode: str = "ex_post"  # "ex_post" | "lookback" | "equal_weight"
    experiment_tag: str = ""  # appended to output_root for experiment isolation
                              # e.g. "rebuttal_lookback" → EXPERIMENTS_rebuttal_lookback/

    data_provider: str = "processed"
    data_version: str = "processed-v1"
    pipeline_schema_version: str = "pipeline-v3-collab"
    sa_only: bool = False
    call_artifact_root: str = ""
    required_gate: str = ""
    data_dir: str = "datasets/processed"
    sec_dir: str = "datasets/sec"
    rebalance: str = "monthly"
    initial_nav: float = 1_000_000.0
    workers_per_experiment: int = 3
    parallel_experiments: int = 1
    seed: int = 42
    noise: float = 0.2
    use_tools: bool = False
    timeout: float = 120.0  # per-request timeout in seconds for LLM calls
    call_max_attempts: int = 3
    retry_failed_calls: bool = False
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    on_error: str = "isolate"
    output_root: str = "EXPERIMENTS"
    propagation_weight: float = 0.1  # CEPS cascade penalty weight
    reuse_latest: bool = False  # reuse the most complete existing run per model
    run_sandbox: bool = True  # run the S1-S5 backtest matrix
    run_qa: bool = False  # run QA dataset evaluation alongside sandbox
    qa: QAConfig = field(default_factory=QAConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    resource_budget: ResourceBudgetConfig = field(default_factory=ResourceBudgetConfig)
    sigma_ablation_values: list = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    interventions: InterventionConfig = field(default_factory=InterventionConfig)

    def __post_init__(self):
        if self.max_rebalances_per_window < 0:
            raise ValueError("max_rebalances_per_window must be non-negative")
        allowed_prefix = ["S1", "S2", "S3"]
        if self.factual_pit_prefix_stages != allowed_prefix[: len(self.factual_pit_prefix_stages)]:
            raise ValueError("factual_pit_prefix_stages must be an S1-S3 prefix")
        if self.legacy_stage_reuse_root and self.factual_pit_prefix_stages:
            raise ValueError(
                "legacy_stage_reuse_root cannot be combined with factual_pit_prefix_stages"
            )
        # Concurrent scenarios would mix provider usage deltas across threads.
        if any(model.architecture_id for model in self.models) and self.workers_per_experiment != 1:
            raise ValueError(
                "architecture experiments require workers_per_experiment=1 for exact usage attribution"
            )
        if self.experiment_tag:
            self.output_root = f"{self.output_root}_{self.experiment_tag}"
        if self.sa_only:
            from ..agent_eval.contracts import PIPELINE_V4_SA_CAUSAL

            if self.pipeline_schema_version != PIPELINE_V4_SA_CAUSAL:
                raise ValueError("sa_only experiments require pipeline-v4-sa-causal")
            if self.use_tools:
                raise ValueError("sa_only experiments require use_tools=false")
            if self.workers_per_experiment != 1:
                raise ValueError("sa_only experiments require workers_per_experiment=1")
            non_sa = [model for model in self.models if model.architecture_id != "SA"]
            if non_sa:
                raise ValueError("sa_only experiments require every model architecture_id=SA")

    @staticmethod
    def from_yaml(path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return ExperimentConfig.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict) -> "ExperimentConfig":
        models = [ModelSpec(**m) for m in (raw.get("models") or [])]
        if not models:
            raise ValueError("ExperimentConfig.models must be a non-empty list")

        # Support "normal_periods" (list) with fallback to "normal_period" (single)
        normal_periods_raw = raw.get("normal_periods")
        if normal_periods_raw:
            np_objs = [
                NormalPeriod(
                    start=_to_date(np_raw.get("start", "2024-01-01")),
                    end=_to_date(np_raw.get("end", "2024-12-31")),
                    label=np_raw.get("label", ""),
                )
                for np_raw in normal_periods_raw
            ]
        else:
            normal_raw = raw.get("normal_period") or {}
            np_objs = [
                NormalPeriod(
                    start=_to_date(normal_raw.get("start", "2024-01-01")),
                    end=_to_date(normal_raw.get("end", "2024-12-31")),
                    label=normal_raw.get("label", ""),
                )
            ]

        log_raw = raw.get("logging") or {}
        log_obj = LoggingConfig(
            **{
                k: v
                for k, v in log_raw.items()
                if k in {"save_pipeline_logs", "save_snapshots", "save_figures"}
            }
        )

        gen_raw = raw.get("generation") or {}
        gen_obj = GenerationConfig(
            temperature=float(gen_raw.get("temperature", 0.0)),
            max_tokens=int(gen_raw.get("max_tokens", 4096)),
        )
        budget_raw = raw.get("resource_budget") or {}
        # Parse the budget as integers so runtime accounting has one canonical type.
        budget_obj = ResourceBudgetConfig(
            max_tokens_per_episode=int(budget_raw.get("max_tokens_per_episode", 32000)),
            max_requests_per_episode=int(budget_raw.get("max_requests_per_episode", 24)),
            config_version=str(budget_raw.get("config_version", "iso-token-v3")),
        )

        batch_id_raw = raw.get("batch_id") or ""
        batch_id = _expand_batch_id(
            batch_id_raw, models, raw.get("rebalance", "monthly")
        )

        return ExperimentConfig(
            batch_id=batch_id,
            models=models,
            profiles=list(
                raw.get("profiles") or ["conservative", "balanced", "aggressive"]
            ),
            stress_scenarios=raw.get("stress_scenarios", "all"),
            run_normal=bool(raw.get("run_normal", True)),
            normal_periods=np_objs,
            max_rebalances_per_window=int(raw.get("max_rebalances_per_window", 0)),
            factual_pit_prefix_stages=list(raw.get("factual_pit_prefix_stages") or []),
            legacy_stage_reuse_root=str(raw.get("legacy_stage_reuse_root") or ""),
            oracle_mode=str(raw.get("oracle_mode", "ex_post")),
            experiment_tag=str(raw.get("experiment_tag", "")),
            data_provider=raw.get("data_provider", "processed"),
            data_version=str(raw.get("data_version", "processed-v1")),
            pipeline_schema_version=str(raw.get("pipeline_schema_version", "pipeline-v3-collab")),
            sa_only=bool(raw.get("sa_only", False)),
            call_artifact_root=str(raw.get("call_artifact_root", "")),
            required_gate=str(raw.get("required_gate", "")),
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
            timeout=float(raw.get("timeout", 120.0)),
            call_max_attempts=int(raw.get("call_max_attempts", 3)),
            retry_failed_calls=bool(raw.get("retry_failed_calls", False)),
            propagation_weight=float(raw.get("propagation_weight", 0.1)),
            reuse_latest=bool(raw.get("reuse_latest", False)),
            run_sandbox=bool(raw.get("run_sandbox", True)),
            run_qa=bool(raw.get("run_qa", False)),
            qa=_parse_qa_config(raw.get("qa") or {}),
            generation=gen_obj,
            resource_budget=budget_obj,
            sigma_ablation_values=list(
                raw.get("sigma_ablation_values", [0.0, 0.25, 0.5, 0.75, 1.0])
            ),
            interventions=_parse_intervention_config(raw.get("interventions") or {}),
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


def _expand_batch_id(
    template: str, models: list[ModelSpec], rebalance: str = "monthly"
) -> str:
    """
    Expand batch_id template variables:
      {models}    — abbreviated model labels joined by "_" (max 3 models shown)
      {rebalance} — rebalance frequency (e.g. "monthly", "weekly", "quarterly")
      {date}      — YYYYMMDD
      {time}      — HHMMSS

    Examples:
      "{models}_{rebalance}_{date}_{time}" → "ark-doubao_monthly_20250509_143022"
      "exp_{date}_{time}"                  → "exp_20250509_143022"
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
        template.replace("{models}", models_str)
        .replace("{rebalance}", rebalance)
        .replace("{date}", now.strftime("%Y%m%d"))
        .replace("{time}", now.strftime("%H%M%S"))
    )


def _parse_qa_config(raw: dict) -> QAConfig:
    return QAConfig(
        dataset_path=raw.get("dataset_path", "datasets/qa/qa_dataset.jsonl"),
        split=raw.get("split", "test"),
        templates=list(
            raw.get("templates", ["T1", "T2", "T3", "T4", "T5", "T6", "T7"])
        ),
        max_pairs_per_template=int(raw.get("max_pairs_per_template", 50)),
        parallel_questions=int(raw.get("parallel_questions", 4)),
        save_responses=bool(raw.get("save_responses", True)),
        info_level=str(raw.get("info_level", "full")),
        t3t4_redesign=bool(raw.get("t3t4_redesign", False)),
        t3t4_numeric_weight=float(raw.get("t3t4_numeric_weight", 0.7)),
        t3t4_explanation_weight=float(raw.get("t3t4_explanation_weight", 0.3)),
        template_version=str(raw.get("template_version", "legacy")),
        scorer_version=str(raw.get("scorer_version", "legacy-v1")),
        call_max_attempts=int(raw.get("call_max_attempts", 3)),
        retry_failed_calls=bool(raw.get("retry_failed_calls", False)),
        freeze_manifest=str(raw.get("freeze_manifest", "")),
    )
