"""One rollout end-to-end (PLAN §12 Phase 1): load task -> journal ->
provision -> orchestrated episodes -> verify -> check -> index. The runner
supports session roles on the podman and chat backends with a real clock and
no scheduled events; anything else fails loudly rather than half-running. How the roles
share the rollout is the orchestrator's policy (orchestrate.py) — a measured
variable, recorded like everything else.
"""

from __future__ import annotations

import hashlib
import random
import secrets
import traceback
from datetime import datetime, timezone
from pathlib import Path

from sentinel.checks import emit_checks
from sentinel.envs.chat import ChatEnv
from sentinel.envs.podman import PodmanEnv
from sentinel.host import host_fingerprint
from sentinel.index import Index
from sentinel.journal.reader import iter_events, replay_check
from sentinel.journal.writer import JournalWriter, RealClock, TwoScaleClock
from sentinel.metrics import compute_metrics
from sentinel.orchestrate import RolePlan, make_orchestrator
from sentinel.participants.base import SessionParticipant
from sentinel.report import team_label
from sentinel.tools import build_toolbox, load_task_tool_module, task_tools_of
from sentinel.schemas.journal import (
    EnvEvent,
    Provisioned,
    RolloutFinished,
    RolloutStarted,
)
from sentinel.schemas.task import TaskConfig, load_task
from sentinel.verify import run_verifier


# Backends the v1 runner can drive. "chat" is the containerless tier: no
# provisioning, no shell, tasks that live entirely in prompts and harness-side
# tools — cheap enough that pass^k across a quant sweep is an afternoon.
_BACKENDS = {"podman": PodmanEnv, "chat": ChatEnv}


def _check_v1_supported(cfg: TaskConfig) -> None:
    for role in cfg.participants.roles:
        if role.shape != "session":
            raise NotImplementedError("v1 runner: session roles only")
    if cfg.environment.backend not in _BACKENDS:
        raise NotImplementedError(
            f"v1 runner: backends {sorted(_BACKENDS)}, got {cfg.environment.backend!r}"
        )
    if cfg.events:
        raise NotImplementedError("v1 runner: scheduled events need the Phase 3 daemon")
    if cfg.clock.mode not in ("real", "two-scale"):
        raise NotImplementedError(
            f"v1 runner: clock modes real|two-scale, got {cfg.clock.mode!r} "
            "(fully simulated clocks arrive with stepped games)"
        )


def _run_init(task_dir: Path, cfg: TaskConfig, env: PodmanEnv, writer: JournalWriter) -> None:
    """Reset phase: run the task's init script, then erase the mechanism —
    /sentinel is wiped before the agent ever sees the filesystem."""
    init = cfg.environment.init
    if not init:
        return
    env.copy_in(str(task_dir / "environment" / init), "/sentinel/init.sh")
    result = env.open_session().exec(
        "sh /sentinel/init.sh; rc=$?; rm -rf /sentinel; exit $rc",
        timeout=cfg.environment.init_timeout.total_seconds(),
    )
    writer.emit(
        EnvEvent, stream="env", category="decision", name="environment_init",
        payload={
            "script": init,
            "exit_code": result.exit_code,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-2000:],
        },
    )
    if result.exit_code != 0:
        raise RuntimeError(
            f"init script {init!r} exited {result.exit_code}: {result.stderr.strip()[:500]}"
        )


def _make_plans(
    task_dir: Path,
    cfg: TaskConfig,
    participants: SessionParticipant | dict[str, SessionParticipant],
    capabilities: frozenset[str],
    seed: int = 0,
) -> tuple[list[RolePlan], object | None]:
    roles = cfg.participants.roles
    if not isinstance(participants, dict):
        if len(roles) != 1:
            raise ValueError(
                f"task has roles {[r.name for r in roles]}; pass a role->participant dict"
            )
        participants = {roles[0].name: participants}
    role_names = {r.name for r in roles}
    if set(participants) != role_names:
        raise ValueError(
            f"bindings {sorted(participants)} do not match task roles {sorted(role_names)}"
        )
    # imported once and kept: the module IS the task's world for harness-side
    # tools, and the verifier must grade that same object (sentinel.tools)
    tools_module = (
        load_task_tool_module(task_dir, cfg.tools.module) if cfg.tools.module else None
    )
    task_tools = (
        task_tools_of(tools_module, task_dir / cfg.tools.module) if tools_module else []
    )
    plans = []
    for role in roles:
        briefing = (
            (task_dir / "instruction" / role.briefing).read_text()
            if role.briefing else None
        )
        system = (
            (task_dir / "instruction" / role.system).read_text()
            if role.system else None
        )
        toolbox = build_toolbox(
            capabilities=capabilities, task_tools=task_tools,
            allow=cfg.tools.allow, deny=cfg.tools.deny, role_allow=role.tools,
        )
        if role.shape == "session" and not toolbox.entries and capabilities:
            # zero tools in a world that HAS capabilities means a policy
            # mistake; zero tools on the chat backend is the point
            raise ValueError(
                f"role {role.name!r} resolves to zero tools "
                f"(capabilities={sorted(capabilities)}); a session role cannot act"
            )
        plans.append(
            RolePlan(
                role=role,
                participant=participants[role.name],
                instruction=(task_dir / "instruction" / role.instruction_file).read_text(),
                briefing=briefing,
                system=system,
                toolbox=toolbox,
                # per-role stream, derived from the rollout seed and the role
                # name: reproducible, and independent across roles
                rng=random.Random(f"{seed}:{role.name}"),
            )
        )
    return plans, tools_module


def run_rollout(
    task_dir: str | Path,
    participants: SessionParticipant | dict[str, SessionParticipant],
    *,
    runs_root: str | Path = "runs",
    experiment: str | None = None,
    orchestration: str = "solo-v1",
    seed: int = 0,
) -> dict:
    task_dir = Path(task_dir)
    cfg = load_task(task_dir)
    _check_v1_supported(cfg)
    orch = make_orchestrator(orchestration)
    env_cls = _BACKENDS[cfg.environment.backend]
    env_capabilities = frozenset(
        {*env_cls.capabilities, *cfg.environment.provides}
    )
    plans, tools_module = _make_plans(
        task_dir, cfg, participants, env_capabilities, seed=seed
    )

    rollout_id = (
        f"{cfg.task.name}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(4)}"
    )
    runs_root = Path(runs_root)
    rollout_dir = runs_root / rollout_id
    index = Index(runs_root / "index.db")
    env = env_cls(task_dir, cfg.environment, name=f"sentinel-{rollout_id}")

    specs = {p.role.name: p.participant.spec() for p in plans}
    bindings_doc = {
        name: {**spec.model_dump(exclude_none=True), "graded": plan.role.graded}
        for (name, spec), plan in zip(specs.items(), plans)
    }
    subject = team_label(bindings_doc)
    config: dict = {
        "host": host_fingerprint(),
        "orchestration": orch.id,
        "seed": seed,
        "clock": cfg.clock.mode,
        "roles": {
            p.role.name: {
                "graded": p.role.graded,
                "tools": p.toolbox.names(),
                **(
                    {
                        "briefing_file": p.role.briefing,
                        "briefing_sha256": hashlib.sha256(p.briefing.encode()).hexdigest()[:16],
                    }
                    if p.briefing is not None
                    else {}
                ),
                **(
                    {
                        "system_file": p.role.system,
                        "system_sha256": hashlib.sha256(p.system.encode()).hexdigest()[:16],
                    }
                    if p.system is not None
                    else {}
                ),
            }
            for p in plans
        },
    }

    clock = (
        TwoScaleClock(cfg.clock.start) if cfg.clock.mode == "two-scale" else RealClock()
    )
    status = "crashed"
    summary: dict = {"rollout_id": rollout_id, "rollout_dir": str(rollout_dir)}
    with JournalWriter(rollout_dir, rollout_id, clock=clock) as writer:
        started = writer.emit(
            RolloutStarted, stream="controller",
            task_name=cfg.task.name, task_version=cfg.task.version,
            bindings=specs, experiment=experiment, config=config,
        )
        index.rollout_started(
            rollout_id,
            task_name=cfg.task.name, task_version=cfg.task.version,
            bindings=bindings_doc, subject=subject, orchestration=orch.id,
            journal_path=str(writer.path),
            started_wall=started.ts_wall.isoformat(), experiment=experiment,
        )
        for key, value in config["host"].items():
            if value is not None:
                index.meta(rollout_id, f"host.{key}", value)
        for p in plans:
            if p.role.briefing:
                index.meta(rollout_id, f"role.{p.role.name}.briefing", p.role.briefing)
            if p.role.system:
                index.meta(rollout_id, f"role.{p.role.name}.system", p.role.system)
            index.meta(rollout_id, f"role.{p.role.name}.tools", ",".join(p.toolbox.names()))
        try:
            handle = env.provision()
            writer.emit(
                Provisioned, stream="controller", backend=env.backend,
                handle=handle, image=env.image,
            )
            _run_init(task_dir, cfg, env, writer)
            t0 = datetime.now(timezone.utc)
            orch.run(env, plans, writer, rollout_dir)
            summary["wall_seconds"] = (datetime.now(timezone.utc) - t0).total_seconds()
            run_verifier(task_dir, cfg, env, writer, rollout_dir, tools_module)
            if cfg.checks:
                # layer 2 (§10): deterministic predicates over the journal we
                # just wrote. Re-runnable offline; `sentinel check` does exactly
                # this against a finished rollout.
                emit_checks(cfg.checks, rollout_dir, writer)
            status = "completed"
        except Exception as e:
            summary["error"] = f"{type(e).__name__}: {e}"
            (rollout_dir / "crash.txt").write_text(traceback.format_exc())
        finally:
            env.teardown()
            finished = writer.emit(
                RolloutFinished, stream="controller", status=status,
                detail=summary.get("error"),
            )
            index.rollout_finished(rollout_id, status, finished.ts_wall.isoformat())

    try:
        counts = replay_check(writer.path)
    except ValueError as e:
        # a crashed rollout may legally end on a dangling intent (§4.2) — the
        # ambiguity is recorded, not resolved; only a completed rollout that
        # fails replay is a framework invariant violation worth dying for
        if status == "completed":
            raise
        summary["replay_error"] = str(e)
        counts = {}
        for ev in iter_events(writer.path):
            counts[ev.type] = counts.get(ev.type, 0) + 1
    summary.update(status=status, events=counts)
    if status == "completed":
        metrics = compute_metrics(writer.path)
        if metrics:
            for name, value in metrics.items():
                index.metric(rollout_id, name, value)
            summary["metrics"] = metrics
    index.close()
    return summary
