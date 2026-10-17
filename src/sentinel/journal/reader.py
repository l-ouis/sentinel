"""Journal reading + the replay check (Phase 1 exit criterion: 'its journal
replays' — every line parses against the schema and seq is gapless)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from sentinel.schemas.journal import Event, parse_event_line


def iter_events(path: str | Path) -> Iterator[Event]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield parse_event_line(line)


def replay_check(path: str | Path) -> dict[str, int]:
    """Validate the whole journal; returns event-type counts. Raises on any
    unparseable line, seq gap, or dangling action intent (§4.2)."""
    counts: dict[str, int] = {}
    open_intents: set[str] = set()
    expected_seq = 0
    for ev in iter_events(path):
        if ev.seq != expected_seq:
            raise ValueError(f"seq gap: expected {expected_seq}, got {ev.seq}")
        expected_seq += 1
        counts[ev.type] = counts.get(ev.type, 0) + 1
        if ev.type == "action":
            open_intents.add(ev.intent_id)
        elif ev.type == "action_result":
            open_intents.discard(ev.intent_id)
    if open_intents:
        raise ValueError(f"dangling action intents (ambiguous effects): {open_intents}")
    return counts
