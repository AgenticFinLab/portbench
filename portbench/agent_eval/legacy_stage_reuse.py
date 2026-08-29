"""Prompt-exact reuse of archived S1-S3 single-agent responses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .base import MarketSnapshot, S1Output, S2Output, S3Output, StageID
from .prompts import build_s1_prompt, build_s2_prompt, build_s3_prompt
from .stages import _format_correlation, _format_macro_context, _format_price_context


VERSION = "legacy-prompt-hash-v1"


def _sha256_text(value: str) -> str:
    """Return a stable digest for a prompt or response string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_coordinates(path: Path, root: Path) -> tuple[str, str, str, str] | None:
    """Read provider, model, profile, and date from the legacy layout."""
    parts = path.relative_to(root).parts
    if len(parts) < 10 or parts[0] not in {"weekly", "monthly", "quarterly"}:
        return None
    if parts[6] != "pipeline_logs" or parts[-2] != "episodes":
        return None
    decision_date = path.stem.split("_", 1)[0]
    if len(decision_date) != 10:
        return None
    model = parts[2]
    if model.endswith("__SA"):
        model = model[: -len("__SA")]
    return parts[1], model, parts[4], decision_date


def _stage_records(episode: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the archived S1-S3 stage records keyed by their stage id."""
    records: dict[str, dict[str, Any]] = {}
    for item in episode.get("stages") or []:
        if not isinstance(item, dict):
            continue
        stage_id = str(item.get("stage_id", "")).upper()
        if stage_id in {"S1", "S2", "S3"} and stage_id not in records:
            records[stage_id] = item
    return records


def _usable(record: dict[str, Any] | None) -> bool:
    """Require the archived replay payload for one LLM-generated stage."""
    return bool(
        isinstance(record, dict)
        and str(record.get("prompt", "")).strip()
        and str(record.get("raw_response", "")).strip()
        and isinstance(record.get("parsed_output"), dict)
    )


def _s1_output(record: dict[str, Any]) -> S1Output:
    """Rehydrate a validated archived S1 output without calling a provider."""
    parsed = record["parsed_output"]
    views = parsed.get("asset_views")
    if not isinstance(views, dict):
        raise ValueError("archived S1 output has no asset_views mapping")
    return S1Output(
        asset_views={str(asset): float(value) for asset, value in views.items()},
        macro_summary=str(parsed.get("macro_summary", "")),
        detected_regime=str(parsed.get("detected_regime", "unknown")),
        confidence=float(parsed.get("confidence", 0.5)),
        raw_llm_output=str(record["raw_response"]),
        refused=bool(parsed.get("refused", False)),
    )


def _s2_output(record: dict[str, Any]) -> S2Output:
    """Rehydrate a validated archived S2 output without calling a provider."""
    parsed = record["parsed_output"]
    signals = parsed.get("signals")
    strengths = parsed.get("strengths")
    if not isinstance(signals, dict) or not isinstance(strengths, dict):
        raise ValueError("archived S2 output has no signals or strengths mapping")
    return S2Output(
        signals={str(asset): str(value) for asset, value in signals.items()},
        strengths={str(asset): float(value) for asset, value in strengths.items()},
        reasoning=str(parsed.get("reasoning", "")),
        raw_llm_output=str(record["raw_response"]),
        refused=bool(parsed.get("refused", False)),
    )


def _s3_output(record: dict[str, Any]) -> S3Output:
    """Rehydrate a validated archived S3 output without calling a provider."""
    parsed = record["parsed_output"]
    weights = parsed.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("archived S3 output has no weights mapping")
    return S3Output(
        weights={str(asset): float(value) for asset, value in weights.items()},
        expected_return=float(parsed.get("expected_return", 0.0)),
        expected_vol=float(parsed.get("expected_vol", 0.0)),
        sharpe_estimate=float(parsed.get("sharpe_estimate", 0.0)),
        raw_llm_output=str(record["raw_response"]),
        refused=bool(parsed.get("refused", False)),
    )


def _prompt_s1(snapshot: MarketSnapshot) -> str:
    """Build the exact current S1 request text without invoking an adapter."""
    assets = list(snapshot.return_data)
    trailing_days = len(next(iter(snapshot.return_data.values()))) if assets else 0
    return build_s1_prompt(
        snapshot=snapshot,
        assets=assets,
        price_context=_format_price_context(snapshot),
        macro_block=_format_macro_context(snapshot),
        corr_block=_format_correlation(snapshot),
        trailing_days=trailing_days,
        use_tools=False,
    )


def _prompt_s2(snapshot: MarketSnapshot, s1: S1Output) -> str:
    """Build the exact current S2 request text without invoking an adapter."""
    return build_s2_prompt(
        snapshot=snapshot,
        s1=s1,
        assets=list(s1.asset_views),
        use_tools=False,
    )


def _prompt_s3(snapshot: MarketSnapshot, s2: S2Output) -> str:
    """Build the exact current S3 request text without invoking an adapter."""
    return build_s3_prompt(
        snapshot=snapshot,
        s2=s2,
        assets=list(s2.signals),
        corr_block=_format_correlation(snapshot),
        use_tools=False,
    )


@dataclass(frozen=True)
class LegacyCandidate:
    """Reference one archived episode eligible for prompt-level matching."""

    source_path: Path

    def load(self) -> dict[str, Any]:
        """Load one source episode only when its date is needed."""
        return json.loads(self.source_path.read_text(encoding="utf-8"))


@dataclass
class LegacyReuseResolution:
    """Describe reused outputs and per-stage prompt-match decisions."""

    outputs: dict[StageID, object] = field(default_factory=dict)
    prompts: dict[StageID, str] = field(default_factory=dict)
    sources: dict[StageID, str] = field(default_factory=dict)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    decisions: dict[str, dict[str, int]] = field(default_factory=dict)


class LegacyStageReuseStore:
    """Index a legacy archive and reuse only byte-identical S1-S3 prompts."""

    def __init__(
        self,
        source_root: str | Path,
        *,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.source_root = Path(source_root)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._source_config_path = self.source_root / "monthly" / "_last_run_config.yaml"
        self._source_config_hash = ""
        self._index: dict[tuple[str, str, str, str], list[LegacyCandidate]] = {}
        self._validate_source_configuration()
        self._build_index()

    def _validate_source_configuration(self) -> None:
        """Require the declared archived generation configuration to match exactly."""
        if not self._source_config_path.exists():
            raise RuntimeError(
                "legacy S1-S3 reuse requires the archived _last_run_config.yaml"
            )
        source_text = self._source_config_path.read_text(encoding="utf-8")
        source = yaml.safe_load(source_text) or {}
        generation = source.get("generation") or {}
        if float(generation.get("temperature", float("nan"))) != self.temperature:
            raise RuntimeError("legacy source temperature does not match the current run")
        if int(generation.get("max_tokens", -1)) != self.max_tokens:
            raise RuntimeError("legacy source max_tokens does not match the current run")
        self._source_config_hash = _sha256_text(source_text)

    def _build_index(self) -> None:
        """Index source paths without retaining their large prompts in memory."""
        for path in sorted(self.source_root.rglob("*.json")):
            coordinates = _source_coordinates(path, self.source_root)
            if coordinates is None:
                continue
            self._index.setdefault(coordinates, []).append(LegacyCandidate(path))

    def resolve(
        self,
        snapshot: MarketSnapshot,
        *,
        provider: str,
        model: str,
        profile: str,
    ) -> LegacyReuseResolution:
        """Reuse the longest S1-S3 prefix whose reconstructed prompts match exactly."""
        key = (provider, model, profile, str(snapshot.decision_date))
        candidates = self._index.get(key, [])
        resolution = LegacyReuseResolution()
        s1_prompt = _prompt_s1(snapshot)
        s1_matches: list[tuple[LegacyCandidate, dict[str, Any], S1Output]] = []
        for candidate in candidates:
            try:
                stages = _stage_records(candidate.load())
                record = stages.get("S1")
                if _usable(record) and record["prompt"] == s1_prompt:
                    s1_matches.append((candidate, stages, _s1_output(record)))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        resolution.decisions["S1"] = {
            "candidates": len(candidates),
            "prompt_matches": len(s1_matches),
        }
        if not s1_matches:
            return resolution

        s2_matches: list[tuple[LegacyCandidate, dict[str, Any], S1Output, S2Output]] = []
        for candidate, stages, s1 in s1_matches:
            try:
                record = stages.get("S2")
                prompt = _prompt_s2(snapshot, s1)
                if _usable(record) and record["prompt"] == prompt:
                    s2_matches.append((candidate, stages, s1, _s2_output(record)))
            except (ValueError, TypeError):
                continue
        resolution.decisions["S2"] = {
            "candidates": len(s1_matches),
            "prompt_matches": len(s2_matches),
        }
        if not s2_matches:
            candidate, stages, s1 = s1_matches[0]
            self._record(resolution, StageID.S1_MARKET_INTERPRETATION, candidate, stages["S1"], s1_prompt, s1)
            return resolution

        s3_matches: list[
            tuple[LegacyCandidate, dict[str, Any], S1Output, S2Output, S3Output]
        ] = []
        for candidate, stages, s1, s2 in s2_matches:
            try:
                record = stages.get("S3")
                prompt = _prompt_s3(snapshot, s2)
                if _usable(record) and record["prompt"] == prompt:
                    s3_matches.append((candidate, stages, s1, s2, _s3_output(record)))
            except (ValueError, TypeError):
                continue
        resolution.decisions["S3"] = {
            "candidates": len(s2_matches),
            "prompt_matches": len(s3_matches),
        }
        if s3_matches:
            candidate, stages, s1, s2, s3 = s3_matches[0]
            self._record(resolution, StageID.S1_MARKET_INTERPRETATION, candidate, stages["S1"], s1_prompt, s1)
            self._record(resolution, StageID.S2_SIGNAL_GENERATION, candidate, stages["S2"], _prompt_s2(snapshot, s1), s2)
            self._record(resolution, StageID.S3_WEIGHT_OPTIMIZATION, candidate, stages["S3"], _prompt_s3(snapshot, s2), s3)
            return resolution

        candidate, stages, s1, s2 = s2_matches[0]
        self._record(resolution, StageID.S1_MARKET_INTERPRETATION, candidate, stages["S1"], s1_prompt, s1)
        self._record(resolution, StageID.S2_SIGNAL_GENERATION, candidate, stages["S2"], _prompt_s2(snapshot, s1), s2)
        return resolution

    def _record(
        self,
        resolution: LegacyReuseResolution,
        stage_id: StageID,
        candidate: LegacyCandidate,
        record: dict[str, Any],
        prompt: str,
        output: object,
    ) -> None:
        """Attach one verified source stage to the current episode resolution."""
        prompt_hash = _sha256_text(prompt)
        raw_hash = _sha256_text(str(record["raw_response"]))
        source_id = f"{VERSION}:{prompt_hash}"
        resolution.outputs[stage_id] = output
        resolution.prompts[stage_id] = prompt
        resolution.sources[stage_id] = source_id
        resolution.provenance.append(
            {
                "stage_id": stage_id.value,
                "source_path": str(candidate.source_path),
                "source_config": str(self._source_config_path),
                "source_config_hash": self._source_config_hash,
                "source_prompt_hash": prompt_hash,
                "source_raw_response_hash": raw_hash,
                "generation": {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                "policy_version": VERSION,
            }
        )


__all__ = ["LegacyReuseResolution", "LegacyStageReuseStore", "VERSION"]
