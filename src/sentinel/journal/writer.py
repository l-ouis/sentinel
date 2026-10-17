"""Append-only journal writer (PLAN §4).

One writer per rollout. Assigns global seq at append time (the append order IS
the global order; wall timestamps carry causality across streams), stamps both
clocks, and fsyncs every line — the journal is the truth, losing tail lines to
a crash would lie about what was dispatched.

v1 runs a real clock only (ts_sim == ts_wall); the Clock seam is where
simulated/two-scale modes land in Phase 2/3.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sentinel.schemas.journal import (
    SCHEMA_VERSION,
    ClockTick,
    Event,
    dump_event_line,
)


def _next_seq(path: Path) -> int:
    """Resume the global sequence from an existing journal. Post-hoc scorers
    (judge, checks) append to a finished rollout's journal, and crash recovery
    reopens a live one — either way the append order must keep counting, or
    replay_check sees a seq gap."""
    if not path.exists():
        return 0
    last = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)["seq"]
    return last + 1


class RealClock:
    def now(self) -> tuple[datetime, datetime]:
        now = datetime.now(timezone.utc)
        return now, now


class TwoScaleClock:
    """Real time within episodes, explicit fast-forward between them (§5).

        sim = start + (wall elapsed) + accumulated advances

    Generation time therefore flows into simulated time — a slow model really
    does keep the other party waiting, and pretending otherwise would delete a
    property worth measuring. The gaps *between* turns are the compressible
    part: a simulated user who "takes eleven minutes to reply" costs eleven
    simulated minutes and no real ones.
    """

    def __init__(self, start: datetime):
        self.start = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        self._wall0 = datetime.now(timezone.utc)
        self._offset = timedelta(0)

    def now(self) -> tuple[datetime, datetime]:
        wall = datetime.now(timezone.utc)
        return wall, self.start + (wall - self._wall0) + self._offset

    def advance(self, delta: timedelta) -> None:
        if delta < timedelta(0):
            raise ValueError("simulated time only moves forward")
        self._offset += delta


class JournalWriter:
    def __init__(
        self,
        rollout_dir: str | Path,
        rollout_id: str,
        clock: RealClock | TwoScaleClock | None = None,
    ):
        self.rollout_dir = Path(rollout_dir)
        self.rollout_id = rollout_id
        self.sidecar_dir = self.rollout_dir / "sidecars"
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        self.sidecar_dir.mkdir(exist_ok=True)
        self.path = self.rollout_dir / "journal.jsonl"
        self._clock = clock or RealClock()
        self._lock = threading.Lock()  # concurrent streams are designed in
        self._seq = _next_seq(self.path)
        self._f = open(self.path, "a", encoding="utf-8")
        self.episode: int | None = None  # orchestrator sets around episodes

    def emit(self, event_cls: type, *, stream: str, **fields: Any) -> Event:
        """Build an event with the envelope filled in, append it, fsync."""
        with self._lock:
            ts_wall, ts_sim = self._clock.now()
            event = event_cls(
                v=SCHEMA_VERSION,
                seq=self._seq,
                rollout_id=self.rollout_id,
                stream=stream,
                episode=self.episode,
                ts_wall=ts_wall,
                ts_sim=ts_sim,
                **fields,
            )
            self._f.write(dump_event_line(event) + "\n")
            self._f.flush()
            os.fsync(self._f.fileno())
            self._seq += 1
            return event

    def now_sim(self) -> datetime:
        """The in-world clock. Task tools read this rather than the host's
        date, so "this week" means the fixture's week."""
        return self._clock.now()[1]

    def advance_sim(self, delta: timedelta, *, reason: str = "") -> None:
        """Fast-forward the simulated clock and journal the jump. The reason
        carries any drawn value, so a random delay is a recorded fact rather
        than something a replay would have to re-roll (§4.1)."""
        advance = getattr(self._clock, "advance", None)
        if advance is None:
            raise RuntimeError(
                "this rollout runs on a real clock; set [clock] mode = "
                '"two-scale" (and a start) to fast-forward simulated time'
            )
        with self._lock:
            sim_from = self._clock.now()[1]
            advance(delta)
        # sim_to is sim_from + the jump, not a re-read of the clock: the tick
        # documents the fast-forward itself, not the fast-forward plus the
        # microseconds of real time that elapsed while journaling it
        self.emit(
            ClockTick, stream="controller",
            sim_from=sim_from, sim_to=sim_from + delta, reason=reason,
        )

    def write_sidecar(self, name: str, content: str) -> str:
        """Store a heavyweight payload; returns the journal-relative ref."""
        path = self.sidecar_dir / name
        path.write_text(content, encoding="utf-8")
        return f"sidecars/{name}"

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JournalWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
