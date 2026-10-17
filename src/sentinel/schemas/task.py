"""task.toml schema, v0. (PLAN.md §9, extended per §3/§5/§7/§10)

A task is a directory:

    tasks/<name>/
    ├── task.toml
    ├── instruction/       # per-role instructions: <role>.md by default
    ├── environment/       # Dockerfile | compose | game module ref | vm spec
    ├── verifier/          # grader, separately containerizable
    └── oracle/            # optional known-good solution

Harbor-compatible where not extended. Extensions over Harbor's shape:
per-role interaction shape, clock modes (incl. two-scale), scheduled events
with session-opening, realism profile, probes.
"""

from __future__ import annotations

import re
import tomllib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from sentinel.schemas.rubric import load_rubric

_DUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([dhms])")
_DUR_MULT = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}


def _parse_duration(v: Any) -> Any:
    """Accept '2d', '3h30m', '90s', bare seconds, or a timedelta."""
    if isinstance(v, timedelta) or isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        parts = _DUR_RE.findall(v)
        if parts and "".join(f"{n}{u}" for n, u in parts) == re.sub(r"\s+", "", v):
            return timedelta(seconds=sum(float(n) * _DUR_MULT[u] for n, u in parts))
        raise ValueError(f"invalid duration {v!r} (use e.g. '2d', '3h30m', '45s')")
    return v


Duration = Annotated[timedelta, BeforeValidator(_parse_duration)]

# Public alias: task-side code (scripted participants, tool modules) parses the
# same duration strings the schema does, rather than inventing a second syntax.
parse_duration = _parse_duration


class _Model(BaseModel):
    # extra="forbid" so a typo'd key in task.toml is an error, not silence
    model_config = ConfigDict(extra="forbid")


class TaskMeta(_Model):
    name: str
    version: str = "0.1"
    summary: str = ""
    tags: list[str] = Field(default_factory=list)


class RoleDef(_Model):
    name: str
    # Interaction shape attaches to the role, not the model behind it (§3).
    shape: Literal["stepped", "session"] = "session"
    # Path under instruction/, defaults to "<name>.md".
    instruction: str | None = None
    # Optional task-owned system prompt (path under instruction/). Replaces
    # the scaffold's default persona; the scaffold still appends whatever the
    # transport requires (the generated tool grammar for schema-less models)
    # and the briefing, if any. This is the realism lever §7.2 rates highest —
    # a real deployment's system prompt, not an eval harness's — and it is
    # NOT the briefing slot: a briefing arrives under a "known only to you,
    # do not reveal" header, which is itself an eval tell.
    system: str | None = None
    # Optional private briefing (path under instruction/): appended to this
    # role's system prompt, never shown to other roles. This is how a task
    # scripts a confederate — "you are the intern; make this specific
    # mistake" — while the instruction stays the public ask.
    briefing: str | None = None
    # graded=False marks a fixture/confederate: recorded in bindings like any
    # participant (provenance is unconditional), but excluded from the report's
    # subject label — it is condition, not the thing being measured.
    graded: bool = True
    # Optional per-role tool allowlist (names). None = everything the task
    # resolves. Asymmetric capabilities: only the locksmith gets `unlock`.
    tools: list[str] | None = None

    @property
    def instruction_file(self) -> str:
        return self.instruction or f"{self.name}.md"


class ParticipantsConfig(_Model):
    roles: list[RoleDef]

    @model_validator(mode="after")
    def _unique_names(self) -> "ParticipantsConfig":
        names = [r.name for r in self.roles]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate role names: {names}")
        if STREAM_RESERVED & set(names):
            raise ValueError(f"role names {STREAM_RESERVED} are reserved")
        if not any(r.graded for r in self.roles):
            raise ValueError("at least one role must be graded")
        return self


STREAM_RESERVED = {"env", "controller"}


class ClockConfig(_Model):
    # real: wall-clock only. simulated: fully fictional (stepped games).
    # two-scale: wall-clock within episodes, fast-forward between (§5).
    mode: Literal["real", "simulated", "two-scale"] = "real"
    # Fictional epoch; required for simulated/two-scale. In-world timestamps
    # must stay consistent with this clock (§5, §7.2).
    start: datetime | None = None


class EnvironmentConfig(_Model):
    backend: Literal["in_process", "chat", "podman", "kata", "vm"]
    # exactly one source, depending on backend:
    image: str | None = None          # pinned image ref (podman/kata/vm)
    build: str | None = None          # path under environment/, e.g. "Dockerfile"
    module: str | None = None         # python module ref for in_process games
    # Optional setup script (path under environment/), run inside the container
    # after provision and before the first session: start services, seed
    # per-rollout state, scrub setup tells. It is removed from the container
    # before the agent ever acts; nonzero exit crashes the rollout.
    init: str | None = None
    init_timeout: Duration = timedelta(minutes=10)
    network_mode: Literal["none", "allowlist", "proxied", "open"] = "none"
    allowlist: list[str] = Field(default_factory=list)
    # Capabilities the image provides beyond the backend's intrinsic ones —
    # the harness can't sniff that an image contains e.g. playwright.
    provides: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_source(self) -> "EnvironmentConfig":
        if self.backend in ("chat", "in_process"):
            return self
        if sum(x is not None for x in (self.image, self.build)) != 1:
            raise ValueError(f"backend {self.backend!r} needs exactly one of image/build")
        return self


class RealismConfig(_Model):
    """How the environment presents itself (§7).

    disclosed — says outright that it is an evaluation. Honest by construction,
                right default for capability evals, freely publishable.
    sandboxed — an isolated deployment, described truthfully as one. Nothing is
                disclosed and nothing is faked: a real deployment of this agent
                would look exactly like this, so a model correctly inferring
                "I am sandboxed" learns nothing about whether it is being
                tested. Cheaper than dressing and immune to the honesty-floor
                problem, because there is no claim to catch. Its tells live in
                the *scenario* (too-clean fixtures, a counterparty who pushes
                back exactly once), not in the filesystem.
    dressed   — an aged, lived-in environment with no eval artifacts, for
                propensity work where realism has to be built rather than
                stated.
    """

    profile: Literal["disclosed", "sandboxed", "dressed"] = "disclosed"


class ScheduledEvent(_Model):
    """[[events]] — future-dated injections the daemon fires (§4). `at` is an
    offset on the sim clock from rollout start."""

    at: Duration
    name: str
    target: str = "env"               # "env" or a declared role
    payload: dict[str, Any] = Field(default_factory=dict)
    # True => the daemon wakes the session participants into a new episode.
    opens_session: bool = False


class VerifierConfig(_Model):
    # separate = split verifier (§7.1): grade in a private image, answer key
    # never in the agent-visible world. inline = grader runs in the task env.
    # module = harness-side grader: a Python function that reads the task's
    # own tool-module state and the journal, for worlds that never existed
    # inside the container. none = no state verifier; the task is graded by
    # trajectory checks and/or a judge alone.
    mode: Literal["separate", "inline", "module", "none"] = "separate"
    # Required for mode="module": path under the task dir to a file exporting
    # grade(ctx: VerifyContext) -> {"reward": float, "passed": bool, ...}.
    module: str | None = None
    timeout: Duration = timedelta(minutes=10)

    @model_validator(mode="after")
    def _module_present(self) -> "VerifierConfig":
        if (self.mode == "module") != (self.module is not None):
            raise ValueError('verifier.module is required by, and only by, mode = "module"')
        return self


_CHECK_NEEDS = {
    # kind -> required fields (beyond id)
    "tool_called": ("tool",),
    "tool_not_called": ("tool",),
    "max_tool_calls": ("tool", "limit"),
    "max_actions": ("limit",),
    "action_matches": ("pattern",),
    "action_not_matches": ("pattern",),
    "tool_args_match": ("tool", "pattern"),
    "tool_args_not_match": ("tool", "pattern"),
    "tool_result_match": ("tool", "pattern"),
    "tool_result_not_match": ("tool", "pattern"),
    "said_matches": ("pattern",),
    "said_not_matches": ("pattern",),
    "claim_without_tool": ("tool", "pattern"),
}


class CheckConfig(_Model):
    """[[checks]] — a trajectory check (§10 layer 2): a deterministic
    predicate over the journal. Cheap, contamination-proof, and the only
    layer that can see *how* a result was reached — "claimed it sent the file
    but never called the tool" is not visible to a state verifier."""

    id: str
    kind: Literal[tuple(_CHECK_NEEDS)]  # type: ignore[valid-type]
    desc: str = ""
    tool: str | None = None        # tool name the predicate is about
    pattern: str | None = None     # regex (re.search, case-sensitive unless (?i))
    limit: int | None = None       # budget for max_* kinds
    role: str | None = None        # restrict to one stream; None = any role

    @model_validator(mode="after")
    def _required_fields(self) -> "CheckConfig":
        missing = [f for f in _CHECK_NEEDS[self.kind] if getattr(self, f) is None]
        if missing:
            raise ValueError(f"check {self.id!r} ({self.kind}) needs: {missing}")
        if self.pattern is not None:
            try:
                re.compile(self.pattern)
            except re.error as e:
                raise ValueError(f"check {self.id!r}: bad regex {self.pattern!r}: {e}") from None
        return self


class ProbeConfig(_Model):
    """[[probes]] — functional probes (§10 layer 3): score artifacts by using
    them. Command runs in the verifier context, emits {"score": float, ...}."""

    name: str
    command: list[str]
    timeout: Duration = timedelta(minutes=10)


class JudgeConfig(_Model):
    """Optional; a task must be gradable with judges off (§10)."""

    model: str | None = None          # None => experiment-level default judge
    rubric: str = "verifier/rubric.toml"  # structured rubric, see schemas/rubric.py
    method: Literal["pairwise_vs_reference", "absolute"] = "pairwise_vs_reference"


class ToolsConfig(_Model):
    """[tools] — the task's tool surface policy. `module` points at a
    task-shipped Python file exporting TOOLS = [(ToolDef, handler), ...];
    those handlers run harness-side, outside the agent-visible world."""

    module: str | None = None         # path under the task dir, e.g. "tools.py"
    allow: list[str] | None = None    # allowlist over builtins + custom; None = all
    deny: list[str] = Field(default_factory=list)


class LimitsConfig(_Model):
    max_wall: Duration | None = None      # whole-rollout wall-clock budget
    max_sim: Duration | None = None       # sim-clock horizon ("7d" = the week)
    max_episodes: int | None = None
    budget_usd: float | None = None       # enforced at the gateway (§6)


class TaskConfig(_Model):
    task: TaskMeta
    participants: ParticipantsConfig
    environment: EnvironmentConfig
    clock: ClockConfig = Field(default_factory=ClockConfig)
    realism: RealismConfig = Field(default_factory=RealismConfig)
    events: list[ScheduledEvent] = Field(default_factory=list)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    checks: list[CheckConfig] = Field(default_factory=list)
    probes: list[ProbeConfig] = Field(default_factory=list)
    judge: JudgeConfig | None = None
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    @model_validator(mode="after")
    def _cross_checks(self) -> "TaskConfig":
        roles = {r.name for r in self.participants.roles}
        for ev in self.events:
            if ev.target != "env" and ev.target not in roles:
                raise ValueError(
                    f"event {ev.name!r} targets unknown role {ev.target!r}"
                )
        if self.clock.mode in ("simulated", "two-scale") and self.clock.start is None:
            raise ValueError(f"clock mode {self.clock.mode!r} requires clock.start")
        ids = [c.id for c in self.checks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate check ids: {ids}")
        for c in self.checks:
            if c.role is not None and c.role not in roles:
                raise ValueError(f"check {c.id!r} targets unknown role {c.role!r}")
        if self.environment.backend == "chat" and self.environment.init is not None:
            raise ValueError("the chat backend has no filesystem to run environment.init in")
        if self.environment.backend == "chat" and self.verifier.mode in ("inline", "separate"):
            raise ValueError(
                f"verifier mode {self.verifier.mode!r} needs a container backend; "
                'the chat backend grades with mode = "module" or "none"'
            )
        return self


def load_task(task_dir: str | Path) -> TaskConfig:
    """Parse and validate <task_dir>/task.toml. Filesystem checks beyond the
    instruction files (environment build context, verifier presence) belong to
    the Phase 1 task loader, not the schema."""
    task_dir = Path(task_dir)
    toml_path = task_dir / "task.toml"
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)
    cfg = TaskConfig.model_validate(raw)
    missing = [
        f
        for r in cfg.participants.roles
        for f in (r.instruction_file, r.briefing, r.system)
        if f is not None and not (task_dir / "instruction" / f).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{task_dir}: missing instruction files: {missing}"
        )
    if cfg.environment.init is not None:
        init_path = task_dir / "environment" / cfg.environment.init
        if not init_path.is_file():
            raise FileNotFoundError(f"{task_dir}: missing init script {cfg.environment.init}")
    if cfg.tools.module is not None and not (task_dir / cfg.tools.module).is_file():
        raise FileNotFoundError(f"{task_dir}: missing tools module {cfg.tools.module}")
    if cfg.verifier.module is not None and not (task_dir / cfg.verifier.module).is_file():
        raise FileNotFoundError(f"{task_dir}: missing verifier module {cfg.verifier.module}")
    if cfg.judge is not None:
        rubric_path = task_dir / cfg.judge.rubric
        if not rubric_path.is_file():
            raise FileNotFoundError(f"{task_dir}: missing rubric {cfg.judge.rubric}")
        load_rubric(rubric_path)  # invariants: weights sum to 1.0, unique ids
    return cfg
