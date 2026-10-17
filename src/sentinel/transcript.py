"""The rollout as a conversation: journal + sidecars folded into role-attributed
entries (PLAN §4 — every scorer is a journal consumer, none of them touches the
environment again).

Two consumers today: the judge, which reads a rendered transcript, and the
trajectory checks, which read the same entries as structured data. Both work
offline over a finished rollout, so re-scoring never re-runs anything.

Entries carry the `seq` of the event they came from, which is what makes
evidence anchoring possible — a judge cites #seq and the claim is checkable
against the journal line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Judgments are never fed back into a transcript: re-judging a rollout must not
# see the previous judge's verdict, or the second opinion is not independent.
_SKIP_TYPES = {"judge_result"}


@dataclass
class Entry:
    seq: int
    kind: str          # system | user | assistant | tool | command | output | verifier | check
    stream: str        # role name, or "env" / "controller"
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


def load_sidecar(rollout_dir: Path, ref: str | None) -> dict | None:
    if not ref:
        return None
    try:
        return json.loads((Path(rollout_dir) / ref).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def assistant_text(resp: dict | None) -> str | None:
    """The model's visible reply, from either sidecar shape: an
    OpenAI-compatible response body or claude CLI --output-format json."""
    if not resp:
        return None
    if "choices" in resp:
        choices = resp.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content")
    return resp.get("result")


def read_journal(path: str | Path) -> list[dict]:
    """Journal lines as raw dicts — scorers care about fields, not classes,
    and an unknown future event type must not break an old scorer."""
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _prompt_entries(
    rollout_dir: Path, ev: dict, seen: dict[str, int]
) -> list[Entry]:
    """The system/user messages this model call saw that no earlier call in
    the same stream did. Diffing (rather than 'first call only') is what makes
    relay pings and later user turns visible in a persistent conversation."""
    stream = ev.get("stream") or ""
    req = load_sidecar(rollout_dir, ev.get("request_ref"))
    if not req:
        return []
    seq = ev.get("seq", 0)
    entries: list[Entry] = []
    if req.get("messages") is not None:            # OpenAI shape
        messages = req["messages"]
        for m in messages[seen.get(stream, 0):]:
            if m.get("role") in ("system", "user") and m.get("content"):
                entries.append(Entry(seq, m["role"], stream, str(m["content"])))
        seen[stream] = len(messages)
    else:                                          # claude CLI shape
        argv = req.get("argv") or []
        if stream not in seen and "--system-prompt" in argv:
            entries.append(Entry(seq, "system", stream, argv[argv.index("--system-prompt") + 1]))
        if req.get("prompt"):
            entries.append(Entry(seq, "user", stream, str(req["prompt"])))
        seen[stream] = seen.get(stream, 0) + 1
    return entries


def build_transcript(rollout_dir: str | Path, events: list[dict] | None = None) -> list[Entry]:
    rollout_dir = Path(rollout_dir)
    events = events if events is not None else read_journal(rollout_dir / "journal.jsonl")
    seen: dict[str, int] = {}
    entries: list[Entry] = []
    for ev in events:
        t, seq, stream = ev.get("type"), ev.get("seq", 0), ev.get("stream") or ""
        if t in _SKIP_TYPES:
            continue
        if t == "model_call":
            entries.extend(_prompt_entries(rollout_dir, ev, seen))
            text = assistant_text(load_sidecar(rollout_dir, ev.get("response_ref")))
            if text and text.strip():
                entries.append(
                    Entry(seq, "assistant", stream, text.strip(),
                          {"completion_tokens": ev.get("completion_tokens"),
                           "latency_ms": ev.get("latency_ms")})
                )
        elif t == "tool_call":
            args = ev.get("args") or {}
            entries.append(
                Entry(seq, "tool", stream, json.dumps(args, ensure_ascii=False),
                      {"name": ev.get("name"), "result": ev.get("result"),
                       "error": ev.get("error")})
            )
            # The result is a separate entry, not metadata. A scorer that
            # cannot see what a tool RETURNED cannot judge whether a claim was
            # grounded in it — three judges independently marked a correct run
            # down for "no tool output backing this claim" when the output was
            # simply never rendered. Shell actions always had this via
            # action_result; custom tools did not.
            if ev.get("result") is not None or ev.get("error"):
                body = (
                    f"[error] {ev['error']}" if ev.get("error")
                    else json.dumps(ev["result"], ensure_ascii=False)
                )
                entries.append(
                    Entry(seq, "output", stream, body,
                          {"tool": ev.get("name"), "error": ev.get("error")})
                )
        elif t == "action":
            data = ev.get("data") or {}
            entries.append(
                Entry(seq, "command", stream, str(data.get("command") or json.dumps(data)),
                      {"kind": ev.get("kind")})
            )
        elif t == "action_result":
            data = ev.get("data") or {}
            text = str(data.get("stdout") or "")
            if (data.get("stderr") or "").strip():
                text += ("\n" if text else "") + "[stderr] " + data["stderr"]
            entries.append(
                Entry(seq, "output", stream, text, {"exit_code": data.get("exit_code")})
            )
        elif t == "env_event" and ev.get("name") == "scripted_utterance":
            # a scripted participant speaks: it makes no model call, so this is
            # the only record of what the other roles were actually told
            payload = ev.get("payload") or {}
            entries.append(
                Entry(seq, "user", stream, str(payload.get("say") or ""),
                      {"scripted": True, "to": payload.get("to")})
            )
        elif t == "verifier_result":
            entries.append(
                Entry(seq, "verifier", stream,
                      json.dumps(ev.get("details") or {}, ensure_ascii=False),
                      {"reward": ev.get("reward"), "passed": ev.get("passed")})
            )
        elif t == "check_result":
            entries.append(
                Entry(seq, "check", stream, ev.get("detail") or "",
                      {"check": ev.get("check"), "passed": ev.get("passed")})
            )
    return entries


def blind_map(events: Iterable[dict]) -> dict[str, str]:
    """model name -> role alias, from the rollout's bindings. Judges are
    measurably swayed by which model they think they are reading; blinding
    costs nothing and removes the confound."""
    for ev in events:
        if ev.get("type") == "rollout_started":
            out = {}
            for role, spec in (ev.get("bindings") or {}).items():
                model = (spec or {}).get("model")
                if model:
                    out[model] = f"the {role}"
            return out
    return {}


def _blind(text: str, mapping: dict[str, str]) -> str:
    for model, alias in mapping.items():
        if model:
            text = text.replace(model, alias)
    return text


def render(
    entries: list[Entry],
    *,
    blind: dict[str, str] | None = None,
    entry_cap: int = 4000,
    include: Iterable[str] = ("system", "user", "assistant", "tool", "command", "output"),
    show_seq: bool = True,
) -> str:
    """Plain-text transcript for a judge prompt. Long tool output is clipped in
    the middle: heads and tails carry the signal, and an unbounded paste of
    build logs would crowd out the behaviour being judged."""
    include = set(include)
    lines: list[str] = []
    for e in entries:
        if e.kind not in include:
            continue
        text = e.text
        if len(text) > entry_cap:
            half = entry_cap // 2
            text = f"{text[:half]}\n[... {len(text) - entry_cap} chars clipped ...]\n{text[-half:]}"
        if blind:
            text = _blind(text, blind)
        label = e.kind if e.kind in ("system", "user") else f"{e.stream}/{e.kind}"
        if e.kind == "tool":
            label += f" {e.meta.get('name')}"
        elif e.kind == "output":
            label += (
                f" <- {e.meta['tool']}" if e.meta.get("tool")
                else f" exit={e.meta.get('exit_code')}"
            )
        prefix = f"[#{e.seq}] " if show_seq else ""
        lines.append(f"{prefix}{label}:\n{text}")
    return "\n\n".join(lines)
