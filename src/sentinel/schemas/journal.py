"""Journal event schema, v0. (PLAN.md §4)

One rollout = one append-only JSONL file in the rollout directory; each line is
one event. SQLite indexes rollouts/metrics; this file is the truth.

Envelope invariants (must survive every future schema version):
- `seq` is global append order, assigned by the controller at append time. With
  concurrent writers (real-time episodes), seq is *the order things were
  journaled*; `ts_wall` across streams is the evidence for true causality.
- `stream` says who originated the event: a participant role name, or the
  reserved streams "env" / "controller".
- `category` implements §4.1: "decision" events are controller-owned and
  replayable by fold; "observation" events are evidence about a world we don't
  control — fold may read them, never re-execute them.
- `episode` groups events between a session_start/session_end pair (index,
  0-based). None outside any episode. Designed in from day one even though
  Phase 1 rollouts have exactly one episode and one writer.
- Write-ahead intent (§4.2): an `action` is journaled BEFORE dispatch and an
  `action_result` after, paired by `intent_id`. A dangling `action` on recovery
  means the effect is ambiguous — that ambiguity is recorded, not resolved.
- `ts_sim` is the fictional clock (two-scale clock, §5); equals `ts_wall` when
  the task's clock mode is "real".

Heavyweight payloads (full model request/response bodies, large observations)
may live in sidecar files in the rollout directory, referenced by `*_ref`
fields, so the journal itself stays greppable.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

SCHEMA_VERSION = 0

Category = Literal["decision", "observation"]

# Reserved stream names; anything else is a participant role.
STREAM_ENV = "env"
STREAM_CONTROLLER = "controller"


class ParticipantSpec(BaseModel):
    """What acted. For local models this is the full measured tuple (§3, §11):
    never just a model name."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["chat_model", "installed_agent", "callable", "human"]
    model: str | None = None          # e.g. "qwen2.5-coder-14b"
    quant: str | None = None          # e.g. "Q4_K_M", "Q8_0", "fp16"
    backend: str | None = None        # e.g. "vllm", "llama.cpp", "api:anthropic"
    context_limit: int | None = None
    sampling: dict[str, Any] | None = None  # temperature, top_p, ...
    scaffold: str | None = None       # versioned scaffold id (prompt template,
                                      # polling policy, agent loop), §3
    extra: dict[str, Any] = Field(default_factory=dict)


class EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    v: int = SCHEMA_VERSION
    seq: int
    rollout_id: str
    stream: str
    episode: int | None = None
    ts_wall: datetime
    ts_sim: datetime


# --- rollout lifecycle -------------------------------------------------------


class RolloutStarted(EventBase):
    """First line of every journal: what this rollout is."""

    type: Literal["rollout_started"] = "rollout_started"
    category: Literal["decision"] = "decision"
    task_name: str
    task_version: str
    task_hash: str | None = None      # hash of the task dir for provenance
    bindings: dict[str, ParticipantSpec]  # role -> participant
    experiment: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)  # resolved run config


class Provisioned(EventBase):
    """Environment is up. `handle` is what recovery re-attaches to (§4.1):
    container id, VM ref, session credentials ref — evidence, not rebuildable."""

    type: Literal["provisioned"] = "provisioned"
    category: Literal["observation"] = "observation"
    backend: str                      # "in_process" | "podman" | "kata" | "vm" | "chat"
    handle: dict[str, Any]
    image: str | None = None          # pinned image ref/digest


class RolloutFinished(EventBase):
    type: Literal["rollout_finished"] = "rollout_finished"
    category: Literal["decision"] = "decision"
    status: Literal["completed", "crashed", "env_lost", "aborted", "timeout"]
    detail: str | None = None


# --- episodes (§5) -----------------------------------------------------------


class SessionStart(EventBase):
    type: Literal["session_start"] = "session_start"
    category: Literal["decision"] = "decision"
    trigger: str                      # "initial" | "scheduled:<event name>" | "resume" | "manual"
    roles: list[str]                  # participants live in this episode


class SessionEnd(EventBase):
    type: Literal["session_end"] = "session_end"
    category: Literal["decision"] = "decision"
    reason: Literal["completed", "timeout", "crash", "interrupted", "budget"]


class ClockTick(EventBase):
    """Simulated-clock advance (between-episode fast-forward in two-scale mode)."""

    type: Literal["clock_tick"] = "clock_tick"
    category: Literal["decision"] = "decision"
    sim_from: datetime
    sim_to: datetime
    reason: str = ""                  # why time moved, incl. any drawn value


# --- interaction -------------------------------------------------------------


class Observation(EventBase):
    """What a participant was shown / could see. `stream` = the observing role."""

    type: Literal["observation"] = "observation"
    category: Literal["observation"] = "observation"
    kind: str                         # env-family-typed: "board", "terminal", "inbox", ...
    data: dict[str, Any]
    data_ref: str | None = None       # sidecar file for heavyweight payloads


class Action(EventBase):
    """A participant's chosen action — journaled BEFORE dispatch (write-ahead
    intent, §4.2). `stream` = the acting role."""

    type: Literal["action"] = "action"
    category: Literal["decision"] = "decision"
    intent_id: str
    kind: str                         # env-family-typed: "move", "command", "send_message", ...
    data: dict[str, Any]
    idempotency_key: str | None = None


class ActionResult(EventBase):
    """What actually happened when the action hit the world — journaled after.
    status="ambiguous" is a legal, honest terminal state for an intent."""

    type: Literal["action_result"] = "action_result"
    category: Literal["observation"] = "observation"
    intent_id: str
    status: Literal["ok", "error", "ambiguous"]
    data: dict[str, Any] = Field(default_factory=dict)
    data_ref: str | None = None


class ToolCall(EventBase):
    """A tool invocation inside a participant's scaffold (as opposed to an
    Action on the environment). `stream` = the calling role."""

    type: Literal["tool_call"] = "tool_call"
    category: Literal["observation"] = "observation"
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float | None = None


class ModelCall(EventBase):
    """One LLM request/response, captured uniformly at the gateway (§6).
    `stream` = the role whose adapter made the call."""

    type: Literal["model_call"] = "model_call"
    category: Literal["observation"] = "observation"
    participant: ParticipantSpec
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    request_ref: str | None = None    # sidecar: full request body
    response_ref: str | None = None   # sidecar: full response body


class EnvEvent(EventBase):
    """Something happened in the environment. Scheduled injections fired by the
    daemon are decisions (the controller chose to fire); spontaneous
    environment happenings are observations."""

    type: Literal["env_event"] = "env_event"
    category: Category = "observation"
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)
    scheduled: bool = False           # True => came from task [[events]]


# --- durability --------------------------------------------------------------


class Checkpoint(EventBase):
    """kind="controller": serialized controller state (resume point for fold).
    kind="environment": pointer to an env snapshot (podman commit, qcow2, ZFS)."""

    type: Literal["checkpoint"] = "checkpoint"
    category: Literal["decision"] = "decision"
    kind: Literal["controller", "environment"]
    ref: str | None = None            # snapshot id / sidecar path
    state: dict[str, Any] | None = None


# --- scoring (§10) -----------------------------------------------------------


class VerifierResult(EventBase):
    type: Literal["verifier_result"] = "verifier_result"
    category: Literal["observation"] = "observation"
    reward: float
    passed: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ProbeResult(EventBase):
    """Functional probe (§10 layer 3): artifacts scored by using them."""

    type: Literal["probe_result"] = "probe_result"
    category: Literal["observation"] = "observation"
    probe: str
    score: float
    details: dict[str, Any] = Field(default_factory=dict)


class CheckResult(EventBase):
    """Trajectory check (§10 layer 2): a deterministic predicate over the
    journal itself — "never called tool X", "<= N steps", "answered the
    interrupt". Journal-based, so it re-runs offline over a finished rollout
    without touching the environment."""

    type: Literal["check_result"] = "check_result"
    category: Literal["observation"] = "observation"
    check: str                        # check id from task.toml
    kind: str                         # predicate kind, e.g. "tool_not_called"
    passed: bool
    detail: str = ""                  # why, in one line
    evidence: list[int] = Field(default_factory=list)  # journal seqs

class JudgeResult(EventBase):
    """Judge emits per-item scores + triggered penalties; the harness computes
    `score` via Rubric.compute_score (weighted sum × penalty factors). The
    arithmetic is never delegated to the judge model."""

    type: Literal["judge_result"] = "judge_result"
    category: Literal["observation"] = "observation"
    judge: ParticipantSpec
    rubric: str | None = None         # rubric file path (task-relative)
    method: Literal["pairwise_vs_reference", "absolute"] = "pairwise_vs_reference"
    # One `sentinel judge` invocation = one pass. Metrics fold the LAST pass
    # only, and identity has to be explicit: re-judging with the same model and
    # the same rubric path produces judgments that are otherwise
    # indistinguishable from the previous pass's, and silently averaging across
    # them mixes measurements taken under different conditions.
    pass_id: str | None = None
    rubric_sha256: str | None = None  # the rubric's CONTENT, not just its path
    item_scores: dict[str, float]     # rubric item id -> [0, 1]
    penalties: dict[str, float] = Field(default_factory=dict)  # triggered id -> factor
    score: float                      # final computed score
    rationale_ref: str | None = None


Event = Annotated[
    Union[
        RolloutStarted,
        Provisioned,
        RolloutFinished,
        SessionStart,
        SessionEnd,
        ClockTick,
        Observation,
        Action,
        ActionResult,
        ToolCall,
        ModelCall,
        EnvEvent,
        Checkpoint,
        VerifierResult,
        ProbeResult,
        CheckResult,
        JudgeResult,
    ],
    Field(discriminator="type"),
]

_adapter: TypeAdapter[Event] = TypeAdapter(Event)


def parse_event_line(line: str | bytes) -> Event:
    return _adapter.validate_json(line)


def dump_event_line(event: Event) -> str:
    """One JSONL line, no trailing newline. Compact and key-stable."""
    return json.dumps(
        event.model_dump(mode="json", exclude_none=True), separators=(",", ":")
    )
