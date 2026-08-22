"""Scan legacy EXPERIMENTS artifacts and grade cache reuse readiness.

Grade A: full S1-S5 episode artifacts.
Grade B: partial stages usable for step-replay of present stages.
Grade C: logs only or incomplete; citation or manual repair required.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence


STAGE_NAMES = ("S1", "S2", "S3", "S4", "S5")


def _usable_stage(item: dict) -> bool:
    """Require the payload needed for deterministic replay and audit."""
    stage_id = str(item.get("stage_id", "")).upper()
    parsed = item.get("parsed_output")
    if stage_id not in STAGE_NAMES or not isinstance(parsed, dict):
        return False
    if item.get("score") is None:
        return False
    if stage_id in {"S1", "S2", "S3"}:
        if not str(item.get("prompt", "")).strip() or not str(item.get("raw_response", "")).strip():
            return False
    if stage_id == "S1":
        return isinstance(parsed.get("asset_views"), dict) and bool(parsed["asset_views"])
    if stage_id == "S2":
        return isinstance(parsed.get("signals"), dict) and bool(parsed["signals"])
    if stage_id == "S3":
        return isinstance(parsed.get("weights"), dict) and bool(parsed["weights"])
    if stage_id == "S4":
        return isinstance(parsed.get("executed_weights"), dict) and bool(parsed["executed_weights"])
    return "portfolio_var" in parsed and "portfolio_drawdown" in parsed


@dataclass
class EpisodeCacheRecord:
    """One graded episode discovered under an experiments root."""

    episode_id: str
    path: str
    stages_present: List[str] = field(default_factory=list)
    grade: str = "C"
    has_pipeline_log: bool = False
    notes: str = ""


def _stages_from_episode_json(path: Path) -> List[str]:
    """Parse PortBench pipeline episode JSON and return stage ids present."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    stages = obj.get("stages")
    if not isinstance(stages, list):
        return []
    found: List[str] = []
    for item in stages:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("stage_id", "")).upper()
        if sid in STAGE_NAMES and sid not in found:
            if _usable_stage(item):
                found.append(sid)
    return found


def _detect_stages(episode_dir: Path) -> List[str]:
    """Return stage ids present as files, nested dirs, or episode JSON."""
    found: List[str] = []
    if not episode_dir.is_dir():
        return found
    names = {p.name.lower() for p in episode_dir.iterdir()}
    for stage in STAGE_NAMES:
        key = stage.lower()
        hit = any(
            key == n
            or n.startswith(key + ".")
            or n.startswith(key + "_")
            or n.startswith("stage_" + key)
            or ("stage" + key) in n.replace("-", "").replace("_", "")
            for n in names
        )
        if hit:
            found.append(stage)

    # Merge stages declared inside episode JSON files in this directory.
    for p in episode_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".json":
            for sid in _stages_from_episode_json(p):
                if sid not in found:
                    found.append(sid)
    return found


def _grade(stages: Sequence[str], has_pipeline_log: bool) -> str:
    """Assign A/B/C from stage coverage."""
    present = set(stages)
    if present.issuperset(STAGE_NAMES):
        return "A"
    if present.intersection({"S1", "S2", "S3"}) and present.intersection({"S4", "S5"}):
        return "B"
    if present or has_pipeline_log:
        return "C"
    return "C"


def iter_episode_json_files(root: Path) -> List[Path]:
    """Find episode JSON files with a top-level stages list."""
    out: List[Path] = []
    if not root.exists():
        return out
    for p in root.rglob("*.json"):
        if "__pycache__" in p.parts:
            continue
        name = p.name.lower()
        if name in {
            "run_summary.json",
            "run_meta.json",
            "checkpoint.json",
            "backtest_result.json",
            "trade_history.json",
        }:
            continue
        try:
            head = p.read_text(encoding="utf-8", errors="ignore")[:2000]
        except OSError:
            continue
        if '"stages"' in head and '"stage_id"' in head:
            out.append(p)
    return out


def iter_episode_dirs(root: Path) -> List[Path]:
    """Yield directories that look like episode or pipeline_logs containers."""
    if not root.exists():
        return []
    preferred: List[Path] = []
    for name in ("pipeline_logs", "episodes", "episode_logs"):
        cand = root / name
        if cand.is_dir():
            preferred.append(cand)
    search_roots = preferred or [root]
    seen = set()
    out: List[Path] = []
    for base in search_roots:
        for p in base.rglob("*"):
            if not p.is_dir():
                continue
            if "__pycache__" in p.parts:
                continue
            child_names = {c.name.lower() for c in p.iterdir()}
            looks_episode = any(
                n.endswith(".json")
                or n.startswith("s1")
                or n.startswith("stage")
                or n == "pipeline_log.json"
                or n.startswith("episode")
                for n in child_names
            )
            if looks_episode and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def scan_legacy_cache(root: str | Path) -> List[EpisodeCacheRecord]:
    """Scan root for legacy episode artifacts and return graded records."""
    root_p = Path(root)
    records: List[EpisodeCacheRecord] = []
    if not root_p.exists():
        return records

    # Prefer per-episode JSON files used by PortBench pipeline_logs.
    json_eps = iter_episode_json_files(root_p)
    if json_eps:
        for ep_json in json_eps:
            stages = _stages_from_episode_json(ep_json)
            grade = _grade(stages, has_pipeline_log=True)
            notes = ""
            if not stages:
                notes = "episode json missing usable stage payloads"
            records.append(
                EpisodeCacheRecord(
                    episode_id=ep_json.stem,
                    path=str(ep_json),
                    stages_present=stages,
                    grade=grade,
                    has_pipeline_log=True,
                    notes=notes,
                )
            )
        return records

    for ep in iter_episode_dirs(root_p):
        stages = _detect_stages(ep)
        has_log = any(
            p.name.lower() in {"pipeline_log.json", "pipeline_logs.json", "log.json"}
            or "pipeline_log" in p.name.lower()
            for p in ep.iterdir()
            if p.is_file()
        )
        grade = _grade(stages, has_log)
        records.append(
            EpisodeCacheRecord(
                episode_id=ep.name,
                path=str(ep),
                stages_present=stages,
                grade=grade,
                has_pipeline_log=has_log,
                notes="" if stages else "no stage artifacts detected",
            )
        )
    return records


def write_manifest(records: Sequence[EpisodeCacheRecord], out_path: str | Path) -> Path:
    """Write read-only JSONL manifest of graded cache records."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(asdict(rec), sort_keys=True, ensure_ascii=False) + "\n")
    return out


def summarize(records: Sequence[EpisodeCacheRecord]) -> Dict[str, Any]:
    """Build a dry-run summary dict with counts and grade histogram."""
    hist = {"A": 0, "B": 0, "C": 0}
    stage_hist = {s: 0 for s in STAGE_NAMES}
    for r in records:
        hist[r.grade] = hist.get(r.grade, 0) + 1
        for s in r.stages_present:
            stage_hist[s] = stage_hist.get(s, 0) + 1
    return {
        "episode_count": len(records),
        "grade_histogram": hist,
        "stages_present_counts": stage_hist,
    }


__all__ = [
    "STAGE_NAMES",
    "EpisodeCacheRecord",
    "scan_legacy_cache",
    "write_manifest",
    "summarize",
    "iter_episode_dirs",
]
