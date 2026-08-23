"""Controlled 2x2x2 architecture runtime with caching and resource budgets."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from portbench.agent_eval.base import AgentAdapter
from portbench.agent_eval.canonical import build_stage_cache_key, canonical_json, sha256_hex
from portbench.agent_eval.contracts import (
    ARCHITECTURE_IDS,
    PIPELINE_V3_COLLAB,
    ArchitectureId,
    ProvenanceSource,
    ResultProvenance,
)
from portbench.agent_eval.replay_adapter import FACTUAL_NAMESPACE, ReplayAdapter


@dataclass(frozen=True)
class ArchitectureSpec:
    """Define one controlled topology, memory, and tools cell."""

    architecture_id: str
    shared_agent: bool
    memory_enabled: bool
    tools_enabled: bool


def _parse_id(architecture_id: str) -> ArchitectureSpec:
    """Build an architecture specification from its registered ID."""
    # Reject free-form names so every experiment maps to a preregistered cell.
    if architecture_id not in ARCHITECTURE_IDS:
        raise ValueError(f"unknown architecture_id={architecture_id!r}")
    return ArchitectureSpec(
        architecture_id=architecture_id,
        shared_agent=architecture_id.startswith("SA"),
        memory_enabled=architecture_id in {
            ArchitectureId.SA_M.value,
            ArchitectureId.SA_MT.value,
            ArchitectureId.MA_M.value,
            ArchitectureId.MA_MT.value,
        },
        tools_enabled=architecture_id in {
            ArchitectureId.SA_T.value,
            ArchitectureId.SA_MT.value,
            ArchitectureId.MA_T.value,
            ArchitectureId.MA_MT.value,
        },
    )


ARCHITECTURE_REGISTRY: Dict[str, ArchitectureSpec] = {
    architecture_id: _parse_id(architecture_id) for architecture_id in ARCHITECTURE_IDS
}

MA_ROLE_NAMES: tuple[str, ...] = ("analyst", "signal", "optimizer", "executor", "risk")
STAGE_ROLES: Mapping[str, str] = {
    "S1": "analyst",
    "S2": "signal",
    "S3": "optimizer",
    "S4": "executor",
    "S5": "risk",
}

AGENT_PROTOCOL_VERSION = "collab-protocol-v1"


@dataclass(frozen=True)
class AgentMessage:
    """Represent one Point-in-Time-safe message between logical agents."""

    message_id: str
    sender: str
    recipient: str
    episode_id: str
    decision_date: str
    stage_id: str
    round_id: str
    message_type: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a stable JSON-compatible envelope with a payload digest."""
        # Copy the payload before hashing so caller-side mutation cannot alter the envelope.
        payload = dict(self.payload)
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "episode_id": self.episode_id,
            "decision_date": self.decision_date,
            "stage_id": self.stage_id,
            "round_id": self.round_id,
            "message_type": self.message_type,
            "payload": payload,
            "payload_hash": sha256_hex(canonical_json(payload)),
        }


class MessageBus:
    """Route immutable episode-scoped messages and retain an audit trace."""

    def __init__(self) -> None:
        self._messages: List[AgentMessage] = []

    def reset(self) -> None:
        """Clear messages at an episode boundary."""
        self._messages = []

    def publish(self, message: AgentMessage) -> None:
        """Validate and publish a message without exposing forward data."""
        from portbench.agent_eval.pit_repair import _raise_if_future

        # Enforce PiT at the communication boundary, before any inbox can observe the payload.
        _raise_if_future(message.payload)
        self._messages.append(message)

    def inbox(self, recipient: str) -> List[Dict[str, Any]]:
        """Return all messages addressed to one logical agent."""
        return [message.to_dict() for message in self._messages if message.recipient == recipient]

    def inbox_hash(self, recipient: str) -> str:
        """Hash the complete ordered inbox for cache isolation."""
        return sha256_hex(canonical_json(self.inbox(recipient)))

    def trace(self) -> List[Dict[str, Any]]:
        """Return the complete ordered collaboration trace."""
        return [message.to_dict() for message in self._messages]


@dataclass
class AgentNode:
    """Own one logical agent's identity, private memory, and inbox."""

    agent_id: str
    role_prompt: str
    memory: "MemoryStore"
    bus: MessageBus

    def render_context(self, memory_enabled: bool) -> str:
        """Render identity, private memory, and current inbox for a call."""
        # Identity remains present even when persistent memory is disabled.
        blocks = [f"[AGENT IDENTITY] {self.role_prompt}"]
        if memory_enabled:
            memory_block = self.memory.render()
            if memory_block:
                blocks.append(memory_block)
        inbox = self.bus.inbox(self.agent_id)
        # Serialize the full ordered inbox because message order is part of the protocol state.
        if inbox:
            blocks.append("[INBOX]\n" + canonical_json(inbox))
        return "\n".join(blocks)


@dataclass
class MARoleMessage:
    """Serialize one controlled role-decomposition message."""

    role: str
    episode_id: str
    decision_date: str
    payload: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    rationale_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return the frozen role message envelope."""
        if self.role not in MA_ROLE_NAMES:
            raise ValueError(f"role must be one of {MA_ROLE_NAMES}")
        return {
            "role": self.role,
            "episode_id": self.episode_id,
            "decision_date": self.decision_date,
            "payload": dict(self.payload),
            "confidence": float(self.confidence),
            "rationale_tags": list(self.rationale_tags),
        }


class MARoleAdapter:
    """Create structured messages for a single controlled role."""

    def __init__(self, role: str) -> None:
        if role not in MA_ROLE_NAMES:
            raise ValueError(f"role must be one of {MA_ROLE_NAMES}")
        self.role = role

    def emit(
        self,
        *,
        episode_id: str,
        decision_date: str,
        payload: Optional[Mapping[str, Any]] = None,
        confidence: float = 0.0,
        rationale_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Emit the frozen role message envelope."""
        return MARoleMessage(
            role=self.role,
            episode_id=episode_id,
            decision_date=decision_date,
            payload=dict(payload or {}),
            confidence=confidence,
            rationale_tags=list(rationale_tags or []),
        ).to_dict()


def build_ma_role_adapters() -> Dict[str, MARoleAdapter]:
    """Return adapters for all controlled roles."""
    return {role: MARoleAdapter(role) for role in MA_ROLE_NAMES}


@dataclass(frozen=True)
class IsoTokenBudget:
    """Define optional per-episode token and request ceilings.

    A non-positive ceiling disables that check. Usage is still recorded.
    """

    max_tokens_per_episode: int = 32000
    max_requests_per_episode: int = 24
    config_version: str = "iso-token-v3"


class BudgetExceeded(RuntimeError):
    """Raised when an episode exceeds its frozen resource budget."""


@dataclass
class ResourceUsage:
    """Record exact and estimated usage without mixing the two fields."""

    token_exact: int = 0
    token_est: int = 0
    request_count: int = 0
    tool_call_count: int = 0
    cache_hit_count: int = 0
    logical_call_count: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible usage record."""
        return asdict(self)


class ResourceLedger:
    """Enforce one resource budget independently for each episode."""

    def __init__(self, budget: IsoTokenBudget) -> None:
        self.budget = budget
        self._lock = threading.RLock()
        self.usage = ResourceUsage()

    def begin_episode(self) -> None:
        """Reset usage at an episode boundary."""
        with self._lock:
            self.usage = ResourceUsage()

    def record(
        self,
        *,
        token_exact: int = 0,
        token_est: int = 0,
        request_count: int = 0,
        tool_call_count: int = 0,
        cache_hit_count: int = 0,
        logical_call_count: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        """Add usage and fail only when a positive ceiling is exceeded."""
        with self._lock:
            # Keep provider, cache, logical-call, and latency counters in one atomic update.
            self.usage.token_exact += int(token_exact)
            self.usage.token_est += int(token_est)
            self.usage.request_count += int(request_count)
            self.usage.tool_call_count += int(tool_call_count)
            self.usage.cache_hit_count += int(cache_hit_count)
            self.usage.logical_call_count += int(logical_call_count)
            self.usage.latency_ms += float(latency_ms)
            # Estimated tokens are used only when the provider omits exact usage.
            counted_tokens = self.usage.token_exact + self.usage.token_est
            token_cap = self.budget.max_tokens_per_episode
            if token_cap > 0 and counted_tokens > token_cap:
                raise BudgetExceeded(
                    f"episode tokens {counted_tokens} exceed {token_cap}"
                )
            request_cap = self.budget.max_requests_per_episode
            if request_cap > 0 and self.usage.request_count > request_cap:
                raise BudgetExceeded(
                    f"episode requests {self.usage.request_count} exceed {request_cap}"
                )


class MemoryStore:
    """Persist bounded cross-episode memory and support branch copy-on-write."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        entries: Optional[List[Dict[str, str]]] = None,
        read_only: bool = False,
        max_entries: int = 10,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.read_only = read_only
        self.max_entries = max_entries
        self.entries: List[Dict[str, str]] = list(entries or [])
        # Load only the bounded tail so old runs cannot grow prompts without limit.
        if self.path is not None and self.path.exists() and not entries:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = list(payload.get("entries", []))[-self.max_entries :]

    def state_hash(self) -> str:
        """Return the content hash of the current memory snapshot."""
        return sha256_hex(canonical_json(self.entries))

    def render(self) -> str:
        """Return a bounded prompt block containing previous decisions."""
        if not self.entries:
            return ""
        lines = [f"{item['stage_id']} {item['decision_date']}: {item['response']}" for item in self.entries]
        return "\n[CROSS-EPISODE MEMORY]\n" + "\n".join(lines)

    def append(self, stage_id: str, decision_date: str, response: str) -> None:
        """Append one bounded response and atomically persist writable memory."""
        if self.read_only:
            raise PermissionError("memory snapshot is read-only")
        compact = " ".join(str(response).split())[:1000]
        # Store a compact response rather than the full provider transcript.
        self.entries.append(
            {"stage_id": str(stage_id), "decision_date": str(decision_date), "response": compact}
        )
        self.entries = self.entries[-self.max_entries :]
        if self.path is None:
            return
        # Replace the memory file atomically to avoid partial state after interruption.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
        temp_path.write_text(
            json.dumps({"entries": self.entries}, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temp_path, self.path)

    def clone(self, *, path: str | Path | None = None, read_only: bool = False) -> "MemoryStore":
        """Create an isolated branch from the current memory snapshot."""
        return MemoryStore(
            path,
            entries=[dict(item) for item in self.entries],
            read_only=read_only,
            max_entries=self.max_entries,
        )


def _usage_snapshot(adapter: Any) -> ResourceUsage:
    """Read cumulative provider usage when the adapter exposes it."""
    getter = getattr(adapter, "resource_usage", None)
    if callable(getter):
        raw = getter()
        return ResourceUsage(
            token_exact=int(raw.get("token_exact", 0)),
            token_est=int(raw.get("token_est", 0)),
            request_count=int(raw.get("request_count", 0)),
            tool_call_count=int(raw.get("tool_call_count", 0)),
        )
    return ResourceUsage()


class ArchitectureStageAdapter(AgentAdapter):
    """Apply role, memory, cache, and budget controls to one stage."""

    def __init__(
        self,
        runtime: "ArchitectureRuntime",
        stage_id: str,
        *,
        agent_id: str,
        call_id: str,
        round_id: str = "main",
    ) -> None:
        self.runtime = runtime
        self.stage_id = stage_id
        self.agent_id = agent_id
        self.call_id = call_id
        self.round_id = round_id

    @property
    def model_name(self) -> str:
        return self.runtime.base_adapter.model_name

    def _augment(self, prompt: str) -> str:
        """Add only the controls enabled for the registered architecture."""
        blocks: List[str] = []
        # SA cells share one memory stream across all five stages.
        if self.runtime.spec.shared_agent:
            if self.runtime.spec.memory_enabled:
                memory = self.runtime.memory.render()
                if memory:
                    blocks.append(memory)
        else:
            # MA cells render only the selected node's private state and inbox.
            blocks.append(
                self.runtime.agent_nodes[self.agent_id].render_context(
                    self.runtime.spec.memory_enabled
                )
            )
        blocks.append(prompt)
        return "\n".join(blocks)

    def _memory(self) -> MemoryStore:
        """Return the memory store owned by this logical caller."""
        if self.runtime.spec.shared_agent:
            return self.runtime.memory
        return self.runtime.agent_nodes[self.agent_id].memory

    def _cache_key(self, prompt: str, toolset_hash: str) -> str:
        """Build a complete key for the exact prompt and runtime state."""
        base = self.runtime.base_adapter
        system_prompt = str(getattr(base, "_system_prompt", ""))
        generation = {
            "temperature": getattr(base, "_temperature", None),
            "max_tokens": getattr(base, "_max_tokens", None),
            "timeout": getattr(base, "_timeout", None),
            "budget_version": self.runtime.budget.config_version,
        }
        # The prompt digest covers identity, memory, inbox, and the stage task.
        prompt_hash = sha256_hex(prompt)
        return build_stage_cache_key(
            model=self.model_name,
            model_revision=str(getattr(base, "_model_revision", "")),
            provider=self.runtime.provider,
            stage_id=self.stage_id,
            stage_schema_version=PIPELINE_V3_COLLAB,
            system_prompt_hash=sha256_hex(system_prompt),
            user_prompt_hash=prompt_hash,
            structured_stage_input_hash=prompt_hash,
            generation_config=generation,
            toolset_hash=toolset_hash,
            tool_result_hash="",
            memory_state_hash=self._memory().state_hash() if self.runtime.spec.memory_enabled else "",
            market_snapshot_hash=self.runtime.snapshot_hash,
            profile=self.runtime.profile,
            decision_date=self.runtime.decision_date,
            data_version=self.runtime.data_version,
            code_commit=self.runtime.code_commit,
            architecture_id=self.runtime.spec.architecture_id,
            memory_enabled=self.runtime.spec.memory_enabled,
            tools_enabled=self.runtime.spec.tools_enabled,
            intervention_id=self.runtime.intervention_id,
            agent_id=self.agent_id,
            agent_protocol_version=AGENT_PROTOCOL_VERSION,
            collaboration_round=self.round_id,
            inbox_hash=self.runtime.message_bus.inbox_hash(self.agent_id),
            message_bundle_hash=sha256_hex(canonical_json(self.runtime.message_bus.trace())),
        )

    def _record_new_call(self, before: ResourceUsage, prompt: str, response: str) -> ResultProvenance:
        """Record exact provider deltas or a clearly labeled estimate."""
        after = _usage_snapshot(self.runtime.base_adapter)
        # Convert cumulative provider counters into this call's incremental cost.
        exact_delta = max(0, after.token_exact - before.token_exact)
        request_delta = max(0, after.request_count - before.request_count) or 1
        tool_delta = max(0, after.tool_call_count - before.tool_call_count)
        # Fall back to an explicit estimate without mixing it into token_exact.
        estimated_delta = 0 if exact_delta else max(1, (len(prompt) + len(response)) // 4)
        self.runtime.ledger.record(
            token_exact=exact_delta,
            token_est=estimated_delta,
            request_count=request_delta,
            tool_call_count=tool_delta,
        )
        return ResultProvenance(
            source=ProvenanceSource.NEW_API_CALL.value,
            cache_key="",
            schema_version=PIPELINE_V3_COLLAB,
            data_version=self.runtime.data_version,
            code_commit=self.runtime.code_commit,
            memory_hash=self._memory().state_hash() if self.runtime.spec.memory_enabled else "",
            token_exact=exact_delta or None,
            token_est=estimated_delta or None,
            request_count=request_delta,
            tool_call_count=tool_delta,
        )

    def _complete(self, prompt: str, tools: Optional[list] = None) -> str:
        """Execute one controlled call and update cache, usage, and memory."""
        # JSON correction retries reuse call_id and therefore count as one logical task.
        self.runtime.record_logical_call(self.call_id, self.agent_id)
        augmented = self._augment(prompt)
        toolset_hash = sha256_hex(
            canonical_json([{"name": tool.name, "schema": tool.input_schema} for tool in tools])
        ) if tools else ""
        key = self._cache_key(augmented, toolset_hash)
        before = _usage_snapshot(self.runtime.base_adapter)
        captured: Dict[str, ResultProvenance] = {}
        started = time.perf_counter()

        def call() -> str:
            # Provider-native tools may create extra requests inside the base adapter.
            if tools is not None:
                response = self.runtime.base_adapter.complete_with_tools(augmented, tools)
            else:
                response = self.runtime.base_adapter.complete(augmented)
            captured["provenance"] = self._record_new_call(before, augmented, str(response))
            return str(response)

        def provenance_factory() -> ResultProvenance:
            return captured["provenance"]

        # Tool calls are Point-in-Time deterministic given the snapshot; cache them
        # with the same content-addressed key as plain completions (includes toolset_hash).
        response, provenance = self.runtime.cache.complete_or_call(
            key,
            call,
            namespace=self.runtime.cache_namespace,
            schema_version=PIPELINE_V3_COLLAB,
            provenance_factory=provenance_factory,
        )
        if provenance.source == ProvenanceSource.CURRENT_CACHE.value:
            self.runtime.ledger.record(cache_hit_count=1)
            self.runtime.record_agent_usage(self.agent_id, cache_hit_count=1)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.runtime.ledger.record(latency_ms=latency_ms)
        # Historical cache provenance remains auditable but contributes zero current provider cost.
        fresh = provenance.source == ProvenanceSource.NEW_API_CALL.value
        self.runtime.record_agent_usage(
            self.agent_id,
            token_exact=int(provenance.token_exact or 0) if fresh else 0,
            token_est=int(provenance.token_est or 0) if fresh else 0,
            request_count=int(provenance.request_count) if fresh else 0,
            tool_call_count=int(provenance.tool_call_count) if fresh else 0,
            latency_ms=latency_ms,
        )
        record = asdict(provenance)
        record.update(
            {
                "stage_id": self.stage_id,
                "agent_id": self.agent_id,
                "round_id": self.round_id,
            }
        )
        self.runtime.provenance[self.call_id] = record
        # Persist only the calling node's response in memory-enabled cells.
        if self.runtime.spec.memory_enabled:
            self._memory().append(self.call_id, self.runtime.decision_date, str(response))
        return str(response)

    def complete(self, prompt: str) -> str:
        """Complete a stage call without tools."""
        return self._complete(prompt)

    def complete_with_tools(self, prompt: str, tools: list) -> str:
        """Complete a stage call with provider-native tools."""
        if not self.runtime.spec.tools_enabled:
            raise RuntimeError("tools are disabled for this architecture")
        return self._complete(prompt, tools=tools)


class ArchitectureRuntime:
    """Own shared resources for one architecture and backtest branch."""

    def __init__(
        self,
        base_adapter: AgentAdapter,
        architecture_id: str,
        *,
        cache_dir: str | Path | None = None,
        memory_path: str | Path | None = None,
        budget: Optional[IsoTokenBudget] = None,
        provider: str = "",
        profile: str = "",
        data_version: str = "",
        code_commit: str = "",
        cache_namespace: str = FACTUAL_NAMESPACE,
        intervention_id: str = "",
        memory_store: Optional[MemoryStore] = None,
    ) -> None:
        self.base_adapter = base_adapter
        self.spec = get_architecture(architecture_id)
        self.budget = budget or IsoTokenBudget()
        self.ledger = ResourceLedger(self.budget)
        self.cache = ReplayAdapter(cache_dir)
        self.memory = memory_store or MemoryStore(memory_path)
        self.message_bus = MessageBus()
        self.schema_version = PIPELINE_V3_COLLAB
        self.provider = provider
        self.profile = profile
        self.data_version = data_version
        self.code_commit = code_commit
        self.cache_namespace = cache_namespace
        self.intervention_id = intervention_id
        self.episode_id = ""
        self.decision_date = ""
        self.snapshot_hash = ""
        self.provenance: Dict[str, Dict[str, Any]] = {}
        self.agent_usage: Dict[str, ResourceUsage] = {}
        self._logical_calls_seen: set[str] = set()
        # Create independent nodes before binding each main stage adapter.
        self.agent_nodes = self._build_agent_nodes(memory_path)
        self._adapters = {
            stage_id: ArchitectureStageAdapter(
                self,
                stage_id,
                agent_id=STAGE_ROLES[stage_id] if not self.spec.shared_agent else "shared",
                call_id=f"{stage_id}-main",
            )
            for stage_id in STAGE_ROLES
        }

    def _build_agent_nodes(self, memory_path: str | Path | None) -> Dict[str, AgentNode]:
        """Create five persistent nodes with independent memories for MA cells."""
        if self.spec.shared_agent:
            return {}
        # Use one sibling file per role so persistent memories cannot alias.
        memory_root = Path(memory_path).with_suffix("") if memory_path is not None else None
        prompts = {
            "analyst": "You are the market analyst. Interpret only Point-in-Time market evidence.",
            "signal": "You are the signal agent. Convert analyst evidence into investment signals.",
            "optimizer": "You are the portfolio optimizer. Propose and revise long-only portfolio weights.",
            "executor": "You are the execution agent. Assess implementability, turnover, and trading cost.",
            "risk": "You are the risk agent. Assess portfolio concentration, drawdown, VaR, and constraints.",
        }
        nodes: Dict[str, AgentNode] = {}
        for role in MA_ROLE_NAMES:
            # Memory-less MA cells still receive an in-memory store for a uniform node contract.
            node_path = memory_root / f"{role}.json" if memory_root is not None else None
            nodes[role] = AgentNode(role, prompts[role], MemoryStore(node_path), self.message_bus)
        return nodes

    def begin_episode(self, episode_id: str, decision_date: str, snapshot_hash: str) -> None:
        """Reset episode-scoped usage and provenance."""
        self.episode_id = str(episode_id)
        self.decision_date = str(decision_date)
        self.snapshot_hash = str(snapshot_hash)
        # Persistent memory survives episodes; messages and cost counters do not.
        self.provenance = {}
        self.agent_usage = {}
        self._logical_calls_seen = set()
        self.message_bus.reset()
        self.ledger.begin_episode()

    def stage_adapter(self, stage_id: str) -> ArchitectureStageAdapter:
        """Return the controlled adapter assigned to one stage."""
        return self._adapters[stage_id]

    def collaboration_adapter(
        self,
        *,
        call_id: str,
        agent_id: str,
        round_id: str,
    ) -> ArchitectureStageAdapter:
        """Create an adapter for one explicit collaboration call."""
        if self.spec.shared_agent:
            raise RuntimeError("collaboration adapters are available only for MA architectures")
        return ArchitectureStageAdapter(
            self,
            "S3",
            agent_id=agent_id,
            call_id=call_id,
            round_id=round_id,
        )

    def publish(
        self,
        *,
        sender: str,
        recipient: str,
        stage_id: str,
        round_id: str,
        message_type: str,
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Publish one validated message and return its serialized envelope."""
        # Sequence numbers make the message order explicit within an episode.
        sequence = len(self.message_bus.trace()) + 1
        message = AgentMessage(
            message_id=f"{self.episode_id}:{sequence}",
            sender=sender,
            recipient=recipient,
            episode_id=self.episode_id,
            decision_date=self.decision_date,
            stage_id=stage_id,
            round_id=round_id,
            message_type=message_type,
            payload=dict(payload),
        )
        self.message_bus.publish(message)
        return message.to_dict()

    def record_stage_output(self, stage_id: str, output: Any) -> None:
        """Route each main stage output to its declared downstream agent."""
        if self.spec.shared_agent:
            return
        from dataclasses import asdict as output_asdict, is_dataclass

        payload = output_asdict(output) if is_dataclass(output) else dict(output)
        # The deterministic router defines the only automatic main-stage edges.
        routes = {
            "S1": ("analyst", "signal", "market_interpretation"),
            "S2": ("signal", "optimizer", "investment_signals"),
            "S3": ("optimizer", "executor", "final_weights"),
            "S4": ("executor", "risk", "execution_result"),
        }
        if stage_id in routes:
            sender, recipient, message_type = routes[stage_id]
            self.publish(
                sender=sender,
                recipient=recipient,
                stage_id=stage_id,
                round_id="main",
                message_type=message_type,
                payload=payload,
            )

    def record_agent_usage(self, agent_id: str, **deltas: Any) -> None:
        """Accumulate usage independently for each logical agent."""
        usage = self.agent_usage.setdefault(agent_id, ResourceUsage())
        for name, value in deltas.items():
            if name == "latency_ms":
                usage.latency_ms += float(value)
            else:
                setattr(usage, name, int(getattr(usage, name)) + int(value))

    def record_logical_call(self, call_id: str, agent_id: str) -> None:
        """Count one logical task even when JSON correction needs provider retries."""
        # Provider retries share a call identifier and must not inflate agent count.
        if call_id in self._logical_calls_seen:
            return
        self._logical_calls_seen.add(call_id)
        self.ledger.record(logical_call_count=1)
        self.record_agent_usage(agent_id, logical_call_count=1)

    def clone_memory_from(self, other: "ArchitectureRuntime") -> None:
        """Copy memory state into an isolated intervention branch."""
        # Clone stores rather than sharing references so branch writes never reach factual state.
        if self.spec.shared_agent:
            self.memory = other.memory.clone()
            return
        for role in MA_ROLE_NAMES:
            self.agent_nodes[role].memory = other.agent_nodes[role].memory.clone()

    def collaboration_trace(self) -> List[Dict[str, Any]]:
        """Return the episode's ordered message trace."""
        return self.message_bus.trace()

    def episode_audit(self) -> Dict[str, Any]:
        """Return resource usage and stage provenance for logging."""
        return {
            "architecture_id": self.spec.architecture_id,
            "budget": asdict(self.budget),
            "usage": self.ledger.usage.to_dict(),
            "provenance": dict(self.provenance),
            "agent_usage": {
                agent_id: usage.to_dict() for agent_id, usage in self.agent_usage.items()
            },
            "collaboration_trace": self.collaboration_trace(),
        }


def get_architecture(architecture_id: str) -> ArchitectureSpec:
    """Return one registered architecture specification."""
    if architecture_id not in ARCHITECTURE_REGISTRY:
        raise KeyError(architecture_id)
    return ARCHITECTURE_REGISTRY[architecture_id]


__all__ = [
    "ArchitectureSpec",
    "ARCHITECTURE_REGISTRY",
    "MA_ROLE_NAMES",
    "AGENT_PROTOCOL_VERSION",
    "AgentMessage",
    "MessageBus",
    "AgentNode",
    "MARoleMessage",
    "MARoleAdapter",
    "build_ma_role_adapters",
    "IsoTokenBudget",
    "BudgetExceeded",
    "ResourceUsage",
    "ResourceLedger",
    "MemoryStore",
    "ArchitectureStageAdapter",
    "ArchitectureRuntime",
    "get_architecture",
]
