"""Orchestration policies: how roles share a rollout. The orchestration id is
a measured variable, exactly like a participant's scaffold (§3) — experiments
hold the team fixed and vary the harness, so ids are versioned and parametric
("turns-v1:r3").

solo-v1  — one role, one session: the degenerate case (v1 behavior).
turns-v1 — round-robin: each role gets one session per round, sequentially,
           in task.toml role order. Roles communicate through the environment
           itself (files); every turn opens a fresh conversation, so the
           world — not the context window — is the memory.
turns-v2 — turns with persistent per-role context: a role's later rounds
           continue its own conversation instead of starting over.
relay-v2 — agent-directed turn order via the structured hand_off tool: each
           turn ends with hand_off(to, message); the message is delivered to
           the named agent. to="none", or ending without a handoff, ends the
           rollout. Combined with role briefings this tests information
           asymmetry: knowing whom to ping can itself be secret knowledge.
           (v1 parsed "NEXT: <role>" from prose — retired: it conflated
           coordination failures with magic-string formatting failures.)
relay-v3 — relay with persistent per-role context (the realistic default:
           agents remember their own prior turns). Stateless variants stay
           available as controls — memory-in-context vs memory-in-world is
           an experimental axis, not an accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sentinel.journal.writer import JournalWriter
from sentinel.participants.base import (
    RolloutContext,
    SessionParticipant,
    TurnResult,
)
from sentinel.schemas.journal import EnvEvent, SessionEnd, SessionStart
from sentinel.schemas.task import RoleDef


@dataclass
class RolePlan:
    """One role, resolved: who plays it, what it is told, what it can use."""

    role: RoleDef
    participant: SessionParticipant
    instruction: str
    briefing: str | None = None
    system: str | None = None      # task-owned system prompt (RoleDef.system)
    rng: object | None = None      # this role's seeded random.Random
    # Gated tool surface (sentinel.tools.Toolbox); None = participant default.
    toolbox: object | None = None


_RESUME_TEMPLATE = (
    "Round {round} of {rounds}. Your original instruction:\n\n{instruction}\n\n"
    "You and your teammates have already been working in this environment — "
    "inspect the current state of the files before acting, then continue the "
    "work. If there is nothing left for you to do, reply with a short status "
    "and no command."
)

_RESUME_PERSISTENT_TEMPLATE = (
    "Round {round} of {rounds}. Continue your work — your context from "
    "earlier rounds is retained, but teammates may have changed the "
    "environment since your last turn. If there is nothing left for you to "
    "do, reply with a short status and no command."
)


def _run_session(
    env, plan: RolePlan, writer: JournalWriter, rollout_dir: Path,
    *, episode: int, trigger: str, instruction: str, peers: list[str] | None = None,
    conversation: object | None = None,
) -> "TurnResult":
    writer.episode = episode
    writer.emit(
        SessionStart, stream="controller", trigger=trigger, roles=[plan.role.name]
    )
    ctx = RolloutContext(
        writer=writer, role=plan.role.name, instruction=instruction,
        rollout_dir=rollout_dir, briefing=plan.briefing, system=plan.system,
        peers=peers, rng=plan.rng,
        conversation=conversation,
        tools=plan.toolbox.resolve(peers=peers) if plan.toolbox else None,
    )
    result = plan.participant.run(env.open_session(), ctx)
    writer.emit(SessionEnd, stream="controller", reason="completed")
    return result


class SoloOrchestrator:
    id = "solo-v1"

    def run(
        self, env, plans: list[RolePlan], writer: JournalWriter, rollout_dir: Path
    ) -> None:
        if len(plans) != 1:
            raise NotImplementedError("solo-v1 orchestrates exactly one role")
        _run_session(
            env, plans[0], writer, rollout_dir,
            episode=0, trigger="initial", instruction=plans[0].instruction,
        )
        writer.episode = None


class TurnsOrchestrator:
    def __init__(self, rounds: int = 2, persistent: bool = False):
        self.rounds = rounds
        self.persistent = persistent
        self.id = f"turns-v{2 if persistent else 1}:r{rounds}"

    def run(
        self, env, plans: list[RolePlan], writer: JournalWriter, rollout_dir: Path
    ) -> None:
        conversations: dict[str, object] = {}
        episode = 0
        for rnd in range(self.rounds):
            for plan in plans:
                if rnd == 0:
                    instruction = plan.instruction
                elif self.persistent:
                    instruction = _RESUME_PERSISTENT_TEMPLATE.format(
                        round=rnd + 1, rounds=self.rounds
                    )
                else:
                    instruction = _RESUME_TEMPLATE.format(
                        round=rnd + 1, rounds=self.rounds, instruction=plan.instruction
                    )
                result = _run_session(
                    env, plan, writer, rollout_dir,
                    episode=episode,
                    trigger="initial" if rnd == 0 else "resume",
                    instruction=instruction,
                    conversation=conversations.get(plan.role.name) if self.persistent else None,
                )
                if self.persistent:
                    conversations[plan.role.name] = result.conversation
                episode += 1
        writer.episode = None


_RELAY_PROTOCOL = (
    "\n\n## Turn protocol\n"
    "You are agent `{name}`, one of several agents taking turns in this "
    "environment: {roles}. Only one agent acts at a time. End your turn with "
    "the hand_off mechanism your tools provide, naming who acts next — or "
    "'none' when the team's work is complete. Your handoff message is "
    "delivered to the agent you name."
)

_RELAY_PING = (
    "Agent `{from_role}` just finished a turn and handed off to you. Their "
    "message:\n\n{message}\n\n---\n\n"
    "Your original instruction:\n\n{instruction}{protocol}"
)

_RELAY_PING_PERSISTENT = (
    "Agent `{from_role}` handed off to you. Their message:\n\n{message}\n\n"
    "Continue — your context from your earlier turns is retained, but "
    "teammates may have changed the environment since your last turn."
)

_MESSAGE_CAP = 4000


class RelayOrchestrator:
    """Agent-directed turn order under a hard episode budget. Ending a turn
    without a handoff, or with an unknown target, ends the rollout — the team
    stopping is a legal outcome, journaled as the final handoff."""

    def __init__(self, max_episodes: int = 8, persistent: bool = False):
        self.max_episodes = max_episodes
        self.persistent = persistent
        self.id = f"relay-v{3 if persistent else 2}:e{max_episodes}"

    def run(
        self, env, plans: list[RolePlan], writer: JournalWriter, rollout_dir: Path
    ) -> None:
        by_name = {p.role.name.lower(): p for p in plans}
        peers = [p.role.name for p in plans]
        names = ", ".join(peers)
        conversations: dict[str, object] = {}
        plan = plans[0]
        trigger = "initial"
        message: str | None = None
        from_role: str | None = None
        for episode in range(self.max_episodes):
            conversation = (
                conversations.get(plan.role.name) if self.persistent else None
            )
            protocol = _RELAY_PROTOCOL.format(name=plan.role.name, roles=names)
            if from_role is None:
                instruction = plan.instruction + protocol
            elif conversation is not None:
                # this role has context: just the ping, not the whole preamble
                instruction = _RELAY_PING_PERSISTENT.format(
                    from_role=from_role, message=message
                )
            else:
                instruction = _RELAY_PING.format(
                    from_role=from_role, message=message,
                    instruction=plan.instruction, protocol=protocol,
                )
            result = _run_session(
                env, plan, writer, rollout_dir,
                episode=episode, trigger=trigger, instruction=instruction, peers=peers,
                conversation=conversation,
            )
            if self.persistent:
                conversations[plan.role.name] = result.conversation
            handoff = result.handoff
            target = handoff.to.lower() if handoff and handoff.to else None
            writer.emit(
                EnvEvent, stream=plan.role.name, category="decision", name="handoff",
                payload={
                    "to": handoff.to if handoff else None,
                    "requested": handoff is not None,
                    "valid": target in by_name,
                },
            )
            if target not in by_name:
                break  # explicit "none", no handoff, or unknown: the team is done
            from_role, plan = plan.role.name, by_name[target]
            message = (handoff.message or result.final or "")[-_MESSAGE_CAP:]
            trigger = f"ping:{from_role}"
        writer.episode = None


def make_orchestrator(spec: str):
    """Parse an orchestration id: "solo-v1", "turns-v1|v2[:r<N>]",
    "relay-v2|v3[:e<N>]". turns-v2 and relay-v3 carry persistent per-role
    context; the earlier versions are the stateless controls."""
    if spec == "solo-v1":
        return SoloOrchestrator()
    for prefix, persistent in (("turns-v1", False), ("turns-v2", True)):
        if spec == prefix:
            return TurnsOrchestrator(persistent=persistent)
        if spec.startswith(prefix + ":r"):
            return TurnsOrchestrator(int(spec[len(prefix) + 2:]), persistent=persistent)
    for prefix, persistent in (("relay-v2", False), ("relay-v3", True)):
        if spec == prefix:
            return RelayOrchestrator(persistent=persistent)
        if spec.startswith(prefix + ":e"):
            return RelayOrchestrator(int(spec[len(prefix) + 2:]), persistent=persistent)
    raise ValueError(f"unknown orchestration {spec!r}")
