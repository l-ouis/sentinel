"""ScriptedParticipant: a participant whose turns are written down in advance.

A user-simulator is just another participant (PLAN §3), and for most
measurements it should not be a model. When the question is "is Q4 of this
model as helpful as Q8", every token the *user* improvises is variance in the
answer — you end up comparing user simulators. A scripted user makes the
condition identical across every rollout in a sweep, costs nothing, and never
rate-limits.

It is a fixture, not a subject: bind it to a role declared `graded = false` so
the report labels it as condition rather than folding it into the subject.

Script file (TOML, task-relative), one turn per step:

    [[steps]]
    say = "can you get my project notes onto my laptop?"
    to  = "assistant"          # relay handoff target; omit for turns-*

    [[steps]]
    tool = "set_device_state"  # optional: act on the world before speaking
    args = { device = "laptop", online = true }
    say  = "ok it's on now"
    to   = "assistant"

    [[steps]]
    delay = ["2m", "15m"]      # how long they "took" to reply, in sim time
    say = "sorry, was in a lecture — thursday 4 works"
    to = "assistant"

    [[steps]]
    end = true                 # or just run out of steps

Running out of steps ends the turn with no handoff, which ends a relay — the
scripted user leaving is a legal, journaled ending.

`delay` fast-forwards the simulated clock before the step speaks (task needs
[clock] mode = "two-scale"): a person who takes eleven minutes to answer costs
eleven simulated minutes and no real ones. A two-element list is a uniform
draw from this role's seeded RNG, and the drawn value is journaled — a replay
reads what was rolled instead of re-rolling it (§4.1). Seeding by rollout
means model A and model B at repeat 3 wait exactly the same eleven minutes,
so the comparison stays paired.
"""

from __future__ import annotations

import hashlib
import random
import tomllib
from datetime import timedelta
from pathlib import Path
from typing import Any

from sentinel.envs.base import Session
from sentinel.participants.base import Handoff, RolloutContext, TurnResult
from sentinel.schemas.journal import EnvEvent, ParticipantSpec
from sentinel.schemas.task import parse_duration
from sentinel.tools import ToolContext

SCAFFOLD_ID = "scripted-v1"


def load_script(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{path}: script needs at least one [[steps]] entry")
    allowed = {"say", "to", "tool", "args", "end", "delay"}
    for i, step in enumerate(steps):
        extra = set(step) - allowed
        if extra:
            raise ValueError(f"{path}: step {i} has unknown keys {sorted(extra)}")
        if not (step.get("say") or step.get("tool") or step.get("end")):
            raise ValueError(f"{path}: step {i} does nothing (needs say, tool, or end)")
        delay = step.get("delay")
        if delay is not None:
            if isinstance(delay, list) and len(delay) != 2:
                raise ValueError(
                    f"{path}: step {i} delay must be a duration or a "
                    '[min, max] pair, e.g. ["2m", "15m"]'
                )
            for part in delay if isinstance(delay, list) else [delay]:
                _as_timedelta(part)  # validate the duration syntax now, not mid-run
    return steps


def _as_timedelta(value: Any) -> timedelta:
    parsed = parse_duration(value)
    return parsed if isinstance(parsed, timedelta) else timedelta(seconds=float(parsed))


def draw_delay(step: dict, rng: random.Random) -> tuple[timedelta | None, str]:
    """This step's simulated wait, and a one-line record of how it was chosen."""
    spec = step.get("delay")
    if spec is None:
        return None, ""
    if isinstance(spec, list):
        lo, hi = (_as_timedelta(x) for x in spec)
        seconds = round(rng.uniform(lo.total_seconds(), hi.total_seconds()))
        return timedelta(seconds=seconds), f"drew {seconds}s from [{spec[0]}, {spec[1]}]"
    delay = _as_timedelta(spec)
    return delay, f"fixed {spec}"


class ScriptedParticipant:
    def __init__(self, script_path: str | Path, *, label: str | None = None):
        self.script_path = Path(script_path)
        self.steps = load_script(self.script_path)
        self.label = label or f"script:{self.script_path.stem}"
        self._sha = hashlib.sha256(self.script_path.read_bytes()).hexdigest()[:16]
        self._index = 0

    def spec(self) -> ParticipantSpec:
        # the script IS the participant: its hash is provenance, and a changed
        # script is a different experimental condition, not the same fixture
        return ParticipantSpec(
            kind="callable", model=self.label, scaffold=SCAFFOLD_ID,
            extra={"script": self.script_path.name, "script_sha256": self._sha,
                   "steps": len(self.steps)},
        )

    def run(self, session: Session, ctx: RolloutContext) -> TurnResult:
        if self._index >= len(self.steps):
            return self._finish(ctx, "exhausted")
        step = self.steps[self._index]
        self._index += 1

        # they "took" this long to reply: simulated minutes, no real ones
        delay, how = draw_delay(step, ctx.rng or random.Random(0))
        if delay is not None:
            ctx.advance_sim(delay, reason=f"{ctx.role} reply delay ({how})")

        if step.get("end"):
            return self._finish(ctx, "end")

        if step.get("tool"):
            if ctx.tools is None:
                raise RuntimeError(
                    f"{self.script_path}: step calls {step['tool']!r} but this role "
                    "resolved no tools"
                )
            tctx = ToolContext(session=session, ctx=ctx)
            ctx.tools.dispatch(step["tool"], dict(step.get("args") or {}), tctx)

        say = str(step.get("say") or "")
        to = step.get("to")
        ctx.writer.emit(
            EnvEvent, stream=ctx.role, category="decision", name="scripted_utterance",
            payload={"step": self._index - 1, "say": say, "to": to,
                     "tool": step.get("tool"),
                     "delay_s": delay.total_seconds() if delay else 0.0,
                     "sim_time": ctx.now_sim().isoformat()},
        )
        handoff = Handoff(to=to, message=say) if to else None
        return TurnResult(final=say or None, handoff=handoff)

    def _finish(self, ctx: RolloutContext, reason: str) -> TurnResult:
        ctx.writer.emit(
            EnvEvent, stream=ctx.role, category="decision", name="scripted_end",
            payload={"reason": reason, "steps_run": self._index},
        )
        return TurnResult(final=None, handoff=Handoff(to=None, message=""))
