"""Orchestrator parsing, team labels, relay handoffs, and the
journal->index reindex path."""

from pathlib import Path

import pytest

from sentinel.journal import JournalWriter, iter_events
from sentinel.orchestrate import (
    RelayOrchestrator,
    RolePlan,
    SoloOrchestrator,
    TurnsOrchestrator,
    make_orchestrator,
)
from sentinel.participants.base import Handoff, TurnResult
from sentinel.reindex import reindex
from sentinel.report import build_report, team_label
from sentinel.schemas.task import RoleDef
from sentinel.schemas.journal import (
    Action,
    ActionResult,
    ModelCall,
    ParticipantSpec,
    RolloutFinished,
    RolloutStarted,
    SessionEnd,
    SessionStart,
    VerifierResult,
)


def test_make_orchestrator() -> None:
    assert isinstance(make_orchestrator("solo-v1"), SoloOrchestrator)
    assert make_orchestrator("turns-v1").rounds == 2
    assert make_orchestrator("turns-v1:r3").rounds == 3
    assert make_orchestrator("turns-v1:r3").id == "turns-v1:r3"
    assert isinstance(make_orchestrator("relay-v2"), RelayOrchestrator)
    assert make_orchestrator("relay-v2:e5").max_episodes == 5
    assert make_orchestrator("relay-v3:e5").persistent is True
    assert make_orchestrator("relay-v2:e5").persistent is False
    assert make_orchestrator("turns-v2:r3").persistent is True
    assert make_orchestrator("turns-v2:r3").id == "turns-v2:r3"
    with pytest.raises(ValueError, match="unknown orchestration"):
        make_orchestrator("vibes-v1")


class _FakeEnv:
    def open_session(self):
        return None  # scripted participants never touch the session


class _Scripted:
    """SessionParticipant that replays canned TurnResults and records what
    instruction text each of its turns received."""

    def __init__(self, turns: list[TurnResult]):
        self.turns = list(turns)
        self.seen: list[str] = []

    def spec(self):
        from sentinel.schemas.journal import ParticipantSpec

        return ParticipantSpec(kind="callable")

    def run(self, session, ctx) -> TurnResult:
        self.seen.append(ctx.instruction)
        return self.turns.pop(0)


def _relay_plans(turns: dict[str, list[TurnResult]]) -> dict[str, RolePlan]:
    return {
        name: RolePlan(
            role=RoleDef(name=name), participant=_Scripted(t), instruction=f"do {name}"
        )
        for name, t in turns.items()
    }


def test_relay_follows_handoffs(tmp_path: Path) -> None:
    plans = _relay_plans({
        "alpha": [TurnResult(final="fragment placed.", handoff=Handoff("charlie", "over to you"))],
        "bravo": [TurnResult(final="done here.", handoff=Handoff(None, "all wrapped up"))],
        "charlie": [TurnResult(final="appended.", handoff=Handoff("bravo"))],
    })
    with JournalWriter(tmp_path / "r", "r") as w:
        RelayOrchestrator(max_episodes=8).run(_FakeEnv(), list(plans.values()), w, tmp_path)

    events = list(iter_events(w.path))
    starts = [e for e in events if e.type == "session_start"]
    assert [s.roles for s in starts] == [["alpha"], ["charlie"], ["bravo"]]
    assert [s.trigger for s in starts] == ["initial", "ping:alpha", "ping:charlie"]
    handoffs = [e.payload for e in events if e.type == "env_event" and e.name == "handoff"]
    assert handoffs == [
        {"to": "charlie", "requested": True, "valid": True},
        {"to": "bravo", "requested": True, "valid": True},
        {"to": None, "requested": True, "valid": False},
    ]
    # the pinged agent got the handoff message and its own instruction
    charlie_seen = plans["charlie"].participant.seen[0]
    assert "over to you" in charlie_seen and "do charlie" in charlie_seen
    assert "Turn protocol" in plans["alpha"].participant.seen[0]


def test_relay_stops_at_episode_budget(tmp_path: Path) -> None:
    plans = _relay_plans({
        "a": [TurnResult(handoff=Handoff("b"))] * 2,
        "b": [TurnResult(handoff=Handoff("a"))] * 2,
    })
    with JournalWriter(tmp_path / "r", "r") as w:
        RelayOrchestrator(max_episodes=4).run(_FakeEnv(), list(plans.values()), w, tmp_path)
    starts = [e for e in iter_events(w.path) if e.type == "session_start"]
    assert [s.roles[0] for s in starts] == ["a", "b", "a", "b"]


def test_relay_v3_carries_conversation(tmp_path: Path) -> None:
    """A role's second turn receives its first turn's conversation token and
    a lean ping (no re-sent instruction)."""
    plans = _relay_plans({
        "a": [
            TurnResult(handoff=Handoff("b", "over to b"), conversation="a-conv-1"),
            TurnResult(handoff=Handoff(None), conversation="a-conv-2"),
        ],
        "b": [TurnResult(handoff=Handoff("a", "back to a"), conversation="b-conv-1")],
    })
    seen_conversations: list[object] = []
    for p in plans.values():
        original = p.participant.run
        def wrapped(session, ctx, _orig=original):
            seen_conversations.append(ctx.conversation)
            return _orig(session, ctx)
        p.participant.run = wrapped
    with JournalWriter(tmp_path / "r", "r") as w:
        RelayOrchestrator(max_episodes=6, persistent=True).run(
            _FakeEnv(), list(plans.values()), w, tmp_path
        )
    # a's turn 1: fresh; b's turn 1: fresh; a's turn 2: resumes a-conv-1
    assert seen_conversations == [None, None, "a-conv-1"]
    a_turn2 = plans["a"].participant.seen[1]
    assert "back to a" in a_turn2
    assert "original instruction" not in a_turn2  # context already has it
    assert RelayOrchestrator(persistent=True).id == "relay-v3:e8"


def test_relay_ends_on_missing_handoff(tmp_path: Path) -> None:
    plans = _relay_plans({
        "a": [TurnResult(final="I never handed off")],
        "b": [TurnResult(final="unused")],
    })
    with JournalWriter(tmp_path / "r", "r") as w:
        RelayOrchestrator().run(_FakeEnv(), list(plans.values()), w, tmp_path)
    events = list(iter_events(w.path))
    assert sum(1 for e in events if e.type == "session_start") == 1
    handoff = next(e for e in events if e.type == "env_event")
    assert handoff.payload == {"to": None, "requested": False, "valid": False}


def test_team_label() -> None:
    solo = {"assistant": {"kind": "chat_model", "model": "m1", "graded": True}}
    assert team_label(solo) == "m1"
    pair = {
        "worker": {"model": "m1", "graded": True},
        "reviewer": {"model": "claude-sonnet-5", "graded": True},
    }
    assert team_label(pair) == "m1 + claude-sonnet-5"
    confederate = {
        "assistant": {"model": "claude-sonnet-5", "graded": True},
        "intern": {"model": "m-weak", "graded": False},
    }
    assert team_label(confederate) == "claude-sonnet-5 [+intern=m-weak]"


def _write_rollout_journal(runs: Path, rid: str) -> None:
    """A completed two-role rollout journal, as run_rollout would write it."""
    spec_a = ParticipantSpec(kind="chat_model", model="m1")
    spec_b = ParticipantSpec(kind="chat_model", model="m-weak")
    with JournalWriter(runs / rid, rid) as w:
        w.emit(
            RolloutStarted, stream="controller", task_name="t", task_version="0.1",
            bindings={"assistant": spec_a, "intern": spec_b}, experiment="exp",
            config={
                "orchestration": "turns-v1:r1",
                "host": {"hostname": "hosty", "cores": 8},
                "roles": {
                    "assistant": {"graded": True},
                    "intern": {"graded": False, "briefing_file": "sabotage.md"},
                },
            },
        )
        w.episode = 0
        w.emit(SessionStart, stream="controller", trigger="initial", roles=["assistant"])
        w.emit(ModelCall, stream="assistant", participant=spec_a,
               prompt_tokens=100, completion_tokens=50, latency_ms=1000.0, cost_usd=0.01)
        w.emit(Action, stream="assistant", intent_id="i1", kind="command", data={"command": "ls"})
        w.emit(ActionResult, stream="assistant", intent_id="i1", status="ok")
        w.emit(SessionEnd, stream="controller", reason="completed")
        w.episode = 1
        w.emit(SessionStart, stream="controller", trigger="initial", roles=["intern"])
        w.emit(ModelCall, stream="intern", participant=spec_b,
               prompt_tokens=10, completion_tokens=5, latency_ms=500.0)
        w.emit(SessionEnd, stream="controller", reason="completed")
        w.episode = None
        w.emit(VerifierResult, stream="env", reward=0.75, passed=True, details={})
        w.emit(RolloutFinished, stream="controller", status="completed")


def test_reindex_rebuilds_from_journals(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_rollout_journal(runs, "t-abc")
    assert reindex(runs) == 1

    report = build_report(runs, "exp")
    assert len(report) == 1
    assert report[0]["participant"] == "m1 [+intern=m-weak]"
    assert report[0]["orch"] == "turns-v1:r1"
    assert report[0]["reward"] == 0.75

    from sentinel.index import Index

    index = Index(runs / "index.db")
    meta = dict(index.query("SELECT key, value FROM meta WHERE rollout_id='t-abc'"))
    metrics = dict(index.query("SELECT name, value FROM metrics WHERE rollout_id='t-abc'"))
    rows = index.query("SELECT orchestration, subject FROM rollouts")
    index.close()
    assert meta["host.hostname"] == "hosty"
    assert meta["role.intern.briefing"] == "sabotage.md"
    assert rows == [("turns-v1:r1", "m1 [+intern=m-weak]")]
    assert metrics["model_calls"] == 2
    assert metrics["cost_usd"] == 0.01
    # per-role attribution via stream
    assert metrics["role.assistant.shell_commands"] == 1
    assert metrics["role.intern.completion_tokens"] == 5
    assert metrics["wall_seconds"] >= 0


def test_reindex_is_idempotent(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_rollout_journal(runs, "t-abc")
    reindex(runs)
    assert reindex(runs) == 1  # drops and rebuilds cleanly
