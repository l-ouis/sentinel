"""Trajectory checks (PLAN §10 layer 2): deterministic predicates over the
journal.

Verifiers see the world's final state; judges see the trajectory but cost
money and vary. Checks sit between them — free, exactly reproducible, and the
only layer that can see *how* a result was reached. The one that earns its
keep first is `claim_without_tool`: "said it sent the file, never called the
tool" is invisible to a state verifier and expensive to ask a judge.

Checks never touch reward. A task's reward stays verifier-owned so that
turning checks on or off cannot silently move the headline number; each check
lands as its own metric (`check.<id>`) plus the `checks_passed` fraction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from sentinel.journal.writer import JournalWriter
from sentinel.schemas.journal import CheckResult
from sentinel.schemas.task import CheckConfig
from sentinel.transcript import build_transcript, read_journal


@dataclass
class CheckOutcome:
    id: str
    kind: str
    passed: bool
    detail: str = ""
    evidence: list[int] = field(default_factory=list)


def _tool_calls(events: list[dict], name: str, role: str | None) -> list[dict]:
    return [
        e for e in events
        if e.get("type") == "tool_call" and e.get("name") == name
        and (role is None or e.get("stream") == role)
    ]


def _actions(events: list[dict], role: str | None) -> list[dict]:
    return [
        e for e in events
        if e.get("type") == "action" and (role is None or e.get("stream") == role)
    ]


def _said(entries, role: str | None) -> list:
    return [
        e for e in entries
        if e.kind == "assistant" and (role is None or e.stream == role)
    ]


def _seqs(items) -> list[int]:
    return [i.get("seq", 0) if isinstance(i, dict) else i.seq for i in items][:20]


def evaluate(check: CheckConfig, rollout_dir: str | Path, events: list[dict]) -> CheckOutcome:
    kind, role = check.kind, check.role
    rx = re.compile(check.pattern) if check.pattern else None
    scope = f" by {role}" if role else ""

    def out(passed: bool, detail: str, evidence=()) -> CheckOutcome:
        return CheckOutcome(check.id, kind, passed, detail, list(evidence))

    if kind in ("tool_called", "tool_not_called", "max_tool_calls"):
        calls = _tool_calls(events, check.tool, role)
        n = len(calls)
        if kind == "tool_called":
            return out(n > 0, f"{check.tool} called {n}x{scope}", _seqs(calls))
        if kind == "tool_not_called":
            return out(n == 0, f"{check.tool} called {n}x{scope}", _seqs(calls))
        return out(n <= check.limit, f"{check.tool} called {n}x (limit {check.limit})", _seqs(calls))

    if kind == "max_actions":
        acts = _actions(events, role)
        return out(len(acts) <= check.limit, f"{len(acts)} actions (limit {check.limit})", _seqs(acts[check.limit:]))

    if kind in ("action_matches", "action_not_matches"):
        hits = [a for a in _actions(events, role) if rx.search((a.get("data") or {}).get("command", ""))]
        want = kind == "action_matches"
        return out(
            bool(hits) == want,
            f"{len(hits)} commands match /{check.pattern}/{scope}",
            _seqs(hits),
        )

    if kind in (
        "tool_args_match", "tool_args_not_match",
        "tool_result_match", "tool_result_not_match",
    ):
        field_name = "args" if kind.startswith("tool_args") else "result"
        hits = [
            c for c in _tool_calls(events, check.tool, role)
            if rx.search(json.dumps(c.get(field_name) or {}, ensure_ascii=False))
        ]
        # The negated forms are how a task says "this string must never reach
        # the counterparty": the delivered text of a structured tool call
        # (hand_off's message) lives in args, not in the model's reply, so
        # said_not_matches alone cannot see it.
        want = not kind.endswith("not_match")
        return out(
            bool(hits) == want,
            f"{len(hits)} {check.tool} {field_name} match /{check.pattern}/",
            _seqs(hits),
        )

    entries = build_transcript(rollout_dir, events)
    if kind in ("said_matches", "said_not_matches"):
        hits = [e for e in _said(entries, role) if rx.search(e.text)]
        want = kind == "said_matches"
        return out(
            bool(hits) == want,
            f"{len(hits)} replies match /{check.pattern}/{scope}",
            _seqs(hits),
        )

    if kind == "claim_without_tool":
        claims = [e for e in _said(entries, role) if rx.search(e.text)]
        calls = _tool_calls(events, check.tool, role)
        # honest by default: no claim is not a failure, and a claim backed by
        # a real call is exactly what we want to see
        passed = not claims or bool(calls)
        return out(
            passed,
            f"{len(claims)} claims match /{check.pattern}/, {check.tool} called {len(calls)}x",
            _seqs(claims + calls),
        )

    raise ValueError(f"unknown check kind {kind!r}")  # schema should have caught it


def run_checks(
    checks: list[CheckConfig], rollout_dir: str | Path, events: list[dict] | None = None
) -> list[CheckOutcome]:
    rollout_dir = Path(rollout_dir)
    if events is None:
        events = read_journal(rollout_dir / "journal.jsonl")
    return [evaluate(c, rollout_dir, events) for c in checks]


def emit_checks(
    checks: list[CheckConfig], rollout_dir: str | Path, writer: JournalWriter
) -> list[CheckOutcome]:
    """Run the task's checks over the journal so far and append the results to
    it. Safe to re-run offline: each pass appends a fresh set of check_result
    events, and metrics are recomputed from the last one."""
    outcomes = run_checks(checks, rollout_dir)
    for o in outcomes:
        writer.emit(
            CheckResult, stream="env", check=o.id, kind=o.kind,
            passed=o.passed, detail=o.detail, evidence=o.evidence,
        )
    return outcomes
