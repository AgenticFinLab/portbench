"""Append-only preregistration manifest for window expansions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SA_FACTUAL_ONLY = "SA_factual_only"


@dataclass
class PreregistrationManifest:
    """A single preregistration / expansion record."""

    # original | expansion
    record_type: str
    window_id: str
    original_window: Optional[Dict[str, Any]] = None
    new_window: Optional[Dict[str, Any]] = None
    reason: str = ""
    based_on: str = SA_FACTUAL_ONLY
    treatment_results_seen: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PreregistrationWriter:
    """Append-only JSONL writer for preregistration manifests."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, record: PreregistrationManifest) -> None:
        if record.record_type == "expansion":
            if record.based_on != SA_FACTUAL_ONLY:
                raise ValueError(
                    f"expansion requires based_on={SA_FACTUAL_ONLY!r}, got {record.based_on!r}"
                )
            if record.treatment_results_seen:
                raise PermissionError(
                    "refusing expansion: treatment_results_seen=True; "
                    "windows may only expand from SA factual before any treatment"
                )
        line = json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read_all(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not self.path.exists():
            return rows
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


__all__ = [
    "SA_FACTUAL_ONLY",
    "PreregistrationManifest",
    "PreregistrationWriter",
]
