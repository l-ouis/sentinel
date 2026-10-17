"""Two-scale clock: real time inside a turn, compressed time between turns.

The measurement this exists for: a simulated person takes minutes to reply,
and a sweep must not spend those minutes. Delays are drawn from a per-role
seeded RNG and journaled, so repeat 3 of model A and repeat 3 of model B wait
exactly the same minutes.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from sentinel.journal.reader import iter_events, replay_check
from sentinel.participants.chat_model import ChatModelParticipant
from sentinel.participants.scripted import ScriptedParticipant, draw_delay
from sentinel.run import run_rollout
from tests.stub_openai import StubOpenAI

_TASK = """
[task]
name = "clocktest"

[[participants.roles]]
name = "peer"
graded = false

[[participants.roles]]
name = "assistant"

[clock]
mode = "two-scale"
start = 2026-09-07T09:00:00

[environment]
backend = "chat"

[verifier]
mode = "none"
"""

_SCRIPT = """
[[steps]]
delay = "3m"
say = "morning — can we meet this week?"
to = "assistant"

[[steps]]
delay = ["2m", "15m"]
say = "sorry, was in a lecture. that works."
to = "assistant"

[[steps]]
end = true
"""

START = datetime(2026, 9, 7, 9, 0)


def _task(tmp_path: Path) -> Path:
    d = tmp_path / "task"
    (d / "instruction").mkdir(parents=True)
    (d / "users").mkdir()
    (d / "task.toml").write_text(_TASK)
    (d / "instruction" / "peer.md").write_text("scripted")
    (d / "instruction" / "assistant.md").write_text("help them find a time")
    (d / "users" / "peer.toml").write_text(_SCRIPT)
    return d


def _run(tmp_path: Path, seed: int = 0, runs: Path | None = None) -> dict:
    task = _task(tmp_path)
    with StubOpenAI([], final="how about thursday at 4?", handoff=("peer", "thu 4?")) as stub:
        return run_rollout(
            task,
            {
                "peer": ScriptedParticipant(task / "users" / "peer.toml"),
                "assistant": ChatModelParticipant(
                    base_url=stub.base_url, model="stub", probe=False
                ),
            },
            runs_root=runs or tmp_path / "runs",
            orchestration="relay-v3:e6", seed=seed,
        )


def _ticks(summary: dict) -> list:
    return [
        e for e in iter_events(Path(summary["rollout_dir"]) / "journal.jsonl")
        if e.type == "clock_tick"
    ]


def test_sim_time_advances_without_spending_real_time(tmp_path: Path) -> None:
    summary = _run(tmp_path)
    assert summary["status"] == "completed", summary

    ticks = _ticks(summary)
    assert len(ticks) == 2  # one per delayed step; the closing step has no delay
    assert (ticks[0].sim_to - ticks[0].sim_from) == timedelta(minutes=3)
    drawn = ticks[1].sim_to - ticks[1].sim_from
    assert timedelta(minutes=2) <= drawn <= timedelta(minutes=15)

    # the whole rollout cost seconds of real time, but moved minutes of sim time
    events = list(iter_events(Path(summary["rollout_dir"]) / "journal.jsonl"))
    wall = (events[-1].ts_wall - events[0].ts_wall).total_seconds()
    sim = (events[-1].ts_sim - events[0].ts_sim).total_seconds()
    assert wall < 30
    assert sim >= 5 * 60
    assert events[0].ts_sim.replace(tzinfo=None) >= START


def test_the_drawn_value_is_journaled_not_just_the_seed(tmp_path: Path) -> None:
    """§4.1: record RNG outputs. A replay reads what was rolled; it never
    re-rolls, or the record would disagree with what happened."""
    summary = _run(tmp_path)
    reasons = [t.reason for t in _ticks(summary)]
    assert "fixed 3m" in reasons[0]
    assert "drew" in reasons[1] and "s from [2m, 15m]" in reasons[1]
    said = [
        e for e in iter_events(Path(summary["rollout_dir"]) / "journal.jsonl")
        if e.type == "env_event" and e.name == "scripted_utterance"
    ]
    assert said[1].payload["delay_s"] > 0
    assert said[0].payload["sim_time"].startswith("2026-09-07T09:03")


def test_same_seed_pairs_the_comparison(tmp_path: Path) -> None:
    a = _run(tmp_path / "a", seed=7)
    b = _run(tmp_path / "b", seed=7)
    c = _run(tmp_path / "c", seed=8)

    def draws(s):
        return [t.sim_to - t.sim_from for t in _ticks(s)]

    assert draws(a) == draws(b)      # two models, repeat 7: identical waits
    assert draws(a) != draws(c)      # a different repeat really is different


def test_journal_still_replays(tmp_path: Path) -> None:
    summary = _run(tmp_path)
    counts = replay_check(Path(summary["rollout_dir"]) / "journal.jsonl")
    assert counts["clock_tick"] == 2


def test_advancing_a_real_clock_fails_loudly(tmp_path: Path) -> None:
    from sentinel.journal.writer import JournalWriter

    with JournalWriter(tmp_path / "r", "r1") as writer:
        with pytest.raises(RuntimeError, match="two-scale"):
            writer.advance_sim(timedelta(minutes=5))


def test_draw_delay_shapes() -> None:
    import random

    rng = random.Random(0)
    assert draw_delay({}, rng) == (None, "")
    fixed, how = draw_delay({"delay": "90s"}, rng)
    assert fixed == timedelta(seconds=90) and how == "fixed 90s"
    drawn, how = draw_delay({"delay": ["1m", "2m"]}, rng)
    assert timedelta(minutes=1) <= drawn <= timedelta(minutes=2)
    assert how.startswith("drew ")


def test_bad_delay_syntax_is_rejected_at_load(tmp_path: Path) -> None:
    script = tmp_path / "s.toml"
    script.write_text('[[steps]]\ndelay = ["2m"]\nsay = "hi"\n')
    with pytest.raises(ValueError, match="min, max"):
        ScriptedParticipant(script)
    script.write_text('[[steps]]\ndelay = "soon"\nsay = "hi"\n')
    with pytest.raises(ValueError, match="invalid duration"):
        ScriptedParticipant(script)
