"""
EvalLogger — persistent logging for LLM evaluation runs.

Captures every prompt, raw LLM response, parsed output, ground truth,
and per-stage score to disk so that evaluation runs can be fully replayed
and analysed post-hoc.

Log structure on disk:
    outputs/eval_logs/{run_id}/
        run_meta.json          — run-level metadata (model, timestamps, config)
        episodes/
            {date}_{seq}.json  — one file per episode, all stages inside
        errors.jsonl           — any stage-level errors across all episodes

Usage (automatic — integrated into EvalPipeline):
    pipeline = build_default_pipeline(adapter)
    pipeline.enable_logging(output_dir="outputs/eval_logs")
    result = pipeline.run_episode(snapshot)
    # Logs written automatically after each episode

Usage (manual):
    from portbench.agent_eval.eval_logger import EvalLogger
    logger = EvalLogger(run_id="my_run", output_dir="outputs/eval_logs")
    logger.log_episode(episode_result, prompts, raw_responses)
    logger.close()
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .base import EpisodeResult, StageID


# ---------------------------------------------------------------------------
# Per-stage interaction record
# ---------------------------------------------------------------------------


@dataclass
class StageLog:
    """
    Complete record of a single stage call within one episode.

    Attributes:
        stage_id:       Stage identifier string (e.g., "S1").
        decision_date:  Episode date.
        prompt:         Full prompt sent to the LLM (empty for mock/deterministic stages).
        raw_response:   Raw text returned by the LLM (or mock stub).
        parsed_output:  Dict representation of the parsed stage output.
        ground_truth:   Dict representation of the ground-truth output.
        score:          Numeric score in [0, 1].
        latency_ms:     LLM call latency in milliseconds (0 if not measured).
        error:          Error message if the stage failed, else empty string.
        timestamp:      ISO-format timestamp when this stage was called.
    """

    stage_id: str
    decision_date: str
    prompt: str = ""
    raw_response: str = ""
    parsed_output: dict = field(default_factory=dict)
    ground_truth: dict = field(default_factory=dict)
    score: float = 0.0
    latency_ms: float = 0.0
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EpisodeLog:
    """
    Complete record of one evaluation episode (all five stages).

    Attributes:
        episode_id:    Unique ID (date + sequence number).
        decision_date: Date of the portfolio decision.
        model_name:    Adapter model identifier.
        stages:        List of StageLog objects in pipeline order.
        ceps_score:    Episode-level CEPS score (filled in after scoring).
        duration_ms:   Total wall-clock time for the episode in milliseconds.
        timestamp:     ISO-format episode start time.
    """

    episode_id: str
    decision_date: str
    model_name: str
    stages: list[StageLog] = field(default_factory=list)
    ceps_score: float = 0.0
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Logger class
# ---------------------------------------------------------------------------


class EvalLogger:
    """
    Writes evaluation logs to disk in a structured directory layout.

    One JSON file per episode, plus a run-level metadata file and an error log.
    All files are human-readable JSON for easy inspection and replay.

    Args:
        run_id:      Unique identifier for this evaluation run.
                     Defaults to a timestamp-based ID.
        output_dir:  Root directory for log files.
        model_name:  Model identifier (used in filenames and metadata).
        config:      Arbitrary dict of run configuration (adapter params, dataset info, etc.)
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        output_dir: str = "outputs/eval_logs",
        model_name: str = "unknown",
        config: Optional[dict] = None,
    ):
        self.run_id = (
            run_id or datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        )
        self.model_name = model_name
        self.output_dir = Path(output_dir) / self.run_id
        self.episodes_dir = self.output_dir / "episodes"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(exist_ok=True)

        self._episode_count = 0
        self._start_time = datetime.now()

        # Write run-level metadata immediately
        meta = {
            "run_id": self.run_id,
            "model_name": model_name,
            "start_time": self._start_time.isoformat(),
            "output_dir": str(self.output_dir),
            "config": config or {},
        }
        self._write_json(self.output_dir / "run_meta.json", meta)

    def log_episode(
        self,
        result: EpisodeResult,
        prompts: dict[StageID, str],
        raw_responses: dict[StageID, str],
        latencies_ms: Optional[dict[StageID, float]] = None,
        ceps_score: float = 0.0,
        duration_ms: float = 0.0,
    ) -> Path:
        """
        Write a complete episode record to disk.

        Args:
            result:        EpisodeResult from EvalPipeline.run_episode().
            prompts:       Dict mapping StageID → prompt string sent to LLM.
            raw_responses: Dict mapping StageID → raw LLM response text.
            latencies_ms:  Optional dict mapping StageID → call latency in ms.
            ceps_score:    Episode CEPS score (computed externally).
            duration_ms:   Total episode wall-clock time.

        Returns:
            Path to the written episode JSON file.
        """
        self._episode_count += 1
        episode_id = f"{result.decision_date}_{self._episode_count:04d}"

        stage_logs = []
        for sid in [
            StageID.S1_MARKET_INTERPRETATION,
            StageID.S2_SIGNAL_GENERATION,
            StageID.S3_WEIGHT_OPTIMIZATION,
            StageID.S4_EXECUTION_SIMULATION,
            StageID.S5_RISK_MONITORING,
        ]:
            actual = result.stage_outputs.get(sid)
            gt = result.gt_outputs.get(sid)

            stage_logs.append(
                StageLog(
                    stage_id=sid.value,
                    decision_date=str(result.decision_date),
                    prompt=prompts.get(sid, ""),
                    raw_response=raw_responses.get(sid, ""),
                    parsed_output=self._to_dict(actual),
                    ground_truth=self._to_dict(gt),
                    score=result.stage_scores.get(sid, 0.0),
                    latency_ms=latencies_ms.get(sid, 0.0) if latencies_ms else 0.0,
                    error=result.errors.get(sid, ""),
                )
            )

        episode_log = EpisodeLog(
            episode_id=episode_id,
            decision_date=str(result.decision_date),
            model_name=self.model_name,
            stages=stage_logs,
            ceps_score=ceps_score,
            duration_ms=duration_ms,
        )

        # Serialize and write
        path = self.episodes_dir / f"{episode_id}.json"
        self._write_json(path, self._episode_to_dict(episode_log))

        # Append errors to errors.jsonl if any stage failed
        if result.errors:
            self._append_errors(result)

        return path

    def close(self) -> Path:
        """
        Write run summary and return the output directory path.

        Call this after all episodes are logged to finalize the run record.
        """
        end_time = datetime.now()
        summary = {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "start_time": self._start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_s": (end_time - self._start_time).total_seconds(),
            "n_episodes": self._episode_count,
        }
        path = self.output_dir / "run_summary.json"
        self._write_json(path, summary)
        return self.output_dir

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        """Write dict as formatted JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)

    def _append_errors(self, result: EpisodeResult) -> None:
        """Append stage errors to the rolling errors.jsonl file."""
        errors_path = self.output_dir / "errors.jsonl"
        with open(errors_path, "a", encoding="utf-8") as f:
            for sid, err_msg in result.errors.items():
                record = {
                    "run_id": self.run_id,
                    "decision_date": str(result.decision_date),
                    "stage_id": sid.value,
                    "error": err_msg,
                    "timestamp": datetime.now().isoformat(),
                }
                f.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def _to_dict(obj: Any) -> dict:
        """
        Convert a stage output dataclass to a plain dict for JSON serialization.
        Returns an empty dict if obj is None or not serializable.
        """
        if obj is None:
            return {}
        try:
            return asdict(obj)
        except (TypeError, Exception):
            # asdict fails for objects with non-dataclass fields (e.g., pd.Series)
            # Fall back to __dict__ with string conversion of unserializable values
            result = {}
            for k, v in (obj.__dict__ if hasattr(obj, "__dict__") else {}).items():
                try:
                    json.dumps(v, default=str)
                    result[k] = v
                except Exception:
                    result[k] = str(v)
            return result

    @staticmethod
    def _episode_to_dict(ep: EpisodeLog) -> dict:
        """Serialize EpisodeLog to a plain dict."""
        return {
            "episode_id": ep.episode_id,
            "decision_date": ep.decision_date,
            "model_name": ep.model_name,
            "ceps_score": ep.ceps_score,
            "duration_ms": ep.duration_ms,
            "timestamp": ep.timestamp,
            "stages": [
                {
                    "stage_id": s.stage_id,
                    "decision_date": s.decision_date,
                    "score": s.score,
                    "latency_ms": s.latency_ms,
                    "error": s.error,
                    "timestamp": s.timestamp,
                    "prompt": s.prompt,
                    "raw_response": s.raw_response,
                    "parsed_output": s.parsed_output,
                    "ground_truth": s.ground_truth,
                }
                for s in ep.stages
            ],
        }
