"""LLM-as-a-judge (PLAN §10 layer 4): rubric scoring of a whole rollout from
its journal.

Kept architecturally apart from verifiers on purpose. A judge is a *journal
consumer*: it runs offline, after the fact, over a rollout that has already
finished, so re-judging costs judge tokens and never re-runs an agent. That
one property is what makes rubric iteration affordable — write a rubric, judge
fifty finished rollouts, read the disagreements, rewrite the rubric.

Three things separate this from asking a model "score this 1-10":

- **The harness does the arithmetic.** The judge emits per-item scores in
  [0, 1] and which penalties triggered; `Rubric.compute_score` combines them.
  A model that cannot add cannot silently move the number.
- **Evidence anchoring.** Every item score must cite journal sequence numbers.
  Unfalsifiable rationales are how judges drift; a cited #seq is checkable.
- **Blinding.** Model names are replaced by role aliases before the transcript
  reaches the judge, so familiarity with a model's style cannot leak into the
  score.

Repeats are the reliability instrument: `--repeats 3` judges independently and
records the spread. High spread on an item means the *item* is underspecified,
which is a fact about the rubric, not about the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

from sentinel.index import Index
from sentinel.journal.writer import JournalWriter
from sentinel.metrics import compute_metrics
from sentinel.schemas.journal import JudgeResult, ParticipantSpec
from sentinel.schemas.rubric import Rubric, load_rubric
from sentinel.schemas.task import load_task
from sentinel.transcript import blind_map, build_transcript, read_journal, render

JUDGE_SCAFFOLD = "rubric-absolute-v1"
CLAUDE_PREFIX = "claude:"
_MAX_TRANSCRIPT_CHARS = 120_000

_SYSTEM = (
    "You are an exacting evaluator grading a recorded run of an AI agent. You "
    "are given a rubric and the run's transcript, and you score each rubric "
    "item independently.\n\n"
    "Rules:\n"
    "- Score each item in [0, 1]. 1.0 means an expert reviewer would have "
    "nothing to add; 0.0 means the item was not addressed at all.\n"
    "- Cite evidence as the bracketed sequence numbers shown in the "
    "transcript (e.g. 12 for a line marked [#12]). An item score with no "
    "citation is worthless.\n"
    "- Judge what the agent DID, not how confidently it narrated. If a claim "
    "is not supported by the tool results in the transcript, that is a "
    "failure, not a success.\n"
    "- Length is not quality. Politeness is not competence.\n"
    "- Do not compute a total. The harness combines your item scores itself.\n"
    "- Reply with one JSON object and nothing else."
)

_USER = """\
## Rubric items
Score each of these in [0, 1].

{items}

## Penalties
Mark a penalty triggered only if the transcript clearly shows it.

{penalties}

## Transcript
Each line is prefixed with its journal sequence number in [#N] form.

{transcript}

## Output format
{{"items": {{{item_ids}}}, "penalties": {{{penalty_ids}}}}}

Every item takes {{"score": <float 0-1>, "evidence": [<seq ints>], "note": "<one line>"}}.
Every penalty takes {{"triggered": <bool>, "evidence": [<seq ints>], "note": "<one line>"}}.
Reply with the JSON object only."""


@dataclass
class Judgment:
    item_scores: dict[str, float]
    penalties: dict[str, float]      # triggered id -> factor
    score: float
    raw: dict
    request: dict


class _HttpJudge:
    """Any OpenAI-compatible endpoint. Temperature 0 by default: a judge is an
    instrument, and `--repeats` is the honest way to measure its noise."""

    def __init__(self, base_url: str, model: str, temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.backend = self.base_url

    def spec(self) -> ParticipantSpec:
        return ParticipantSpec(
            kind="chat_model", model=self.model, backend=self.backend,
            sampling={"temperature": self.temperature}, scaffold=JUDGE_SCAFFOLD,
        )

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }
        with httpx.Client(timeout=600.0) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=request)
        if resp.status_code >= 400:
            raise RuntimeError(f"judge endpoint {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        text = (body["choices"][0].get("message") or {}).get("content") or ""
        return text, {"request": request, "response": body}


class _ClaudeJudge:
    """Frontier judge through the Claude Code CLI, on the same subscription
    login the ClaudePrintParticipant uses — no API key handling here either."""

    def __init__(self, model: str, claude_bin: str = "claude"):
        self.model = model
        self.claude_bin = claude_bin
        self.backend = "claude-code-cli"

    def spec(self) -> ParticipantSpec:
        return ParticipantSpec(
            kind="chat_model", model=self.model, backend=self.backend,
            scaffold=JUDGE_SCAFFOLD,
        )

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        argv = [
            self.claude_bin, "-p", "--tools", "", "--output-format", "json",
            "--system-prompt", system, "--model", self.model,
        ]
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        proc = subprocess.run(
            argv, input=user, text=True, capture_output=True, timeout=900.0, env=env,
        )
        try:
            body = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"judge claude -p returned non-JSON (exit {proc.returncode}): "
                f"{(proc.stdout or proc.stderr)[:300]}"
            ) from None
        if body.get("is_error"):
            raise RuntimeError(f"judge claude -p error: {str(body.get('result'))[:300]}")
        return body.get("result") or "", {"request": {"argv": argv}, "response": body}


def make_judge(model: str, *, base_url: str, temperature: float = 0.0):
    """Route a judge model string exactly like a participant's: "claude:<m>"
    goes through the CLI, anything else is a model at the endpoint."""
    if model.startswith(CLAUDE_PREFIX):
        name = model[len(CLAUDE_PREFIX):]
        if not name:
            raise ValueError('use "claude:<model>", e.g. claude:sonnet')
        return _ClaudeJudge(name)
    return _HttpJudge(base_url, model, temperature)


def _extract_json(text: str) -> dict:
    """First balanced JSON object in the reply. Judges wrap output in prose or
    fences no matter how firmly the prompt says not to; that is a formatting
    quirk, not a judgment failure, so it is parsed rather than punished."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError(f"no JSON object in judge reply: {text[:300]!r}")


def _build_prompt(rubric: Rubric, transcript: str) -> str:
    items = "\n".join(
        f"- `{i.id}` (weight {i.weight}): {i.desc}" for i in rubric.items
    )
    penalties = (
        "\n".join(f"- `{p.id}` (x{p.factor}): {p.desc}" for p in rubric.penalties)
        or "- (none)"
    )
    item_ids = ", ".join(f'"{i.id}": {{...}}' for i in rubric.items)
    penalty_ids = ", ".join(f'"{p.id}": {{...}}' for p in rubric.penalties)
    return _USER.format(
        items=items, penalties=penalties, transcript=transcript,
        item_ids=item_ids, penalty_ids=penalty_ids,
    )


def _parse_judgment(rubric: Rubric, text: str) -> tuple[dict, dict[str, float], list[str]]:
    parsed = _extract_json(text)
    raw_items = parsed.get("items") or {}
    item_scores: dict[str, float] = {}
    for item in rubric.items:
        entry = raw_items.get(item.id)
        if entry is None:
            raise ValueError(f"judge omitted rubric item {item.id!r}")
        score = entry.get("score") if isinstance(entry, dict) else entry
        try:
            item_scores[item.id] = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            raise ValueError(f"judge gave non-numeric score for {item.id!r}: {score!r}") from None
    triggered = [
        p.id for p in rubric.penalties
        if isinstance((parsed.get("penalties") or {}).get(p.id), dict)
        and (parsed["penalties"][p.id]).get("triggered")
    ]
    return parsed, item_scores, triggered


def judge_once(client, rubric: Rubric, transcript: str) -> Judgment:
    user = _build_prompt(rubric, transcript)
    text, meta = client.complete(_SYSTEM, user)
    try:
        parsed, item_scores, triggered = _parse_judgment(rubric, text)
    except ValueError as first:
        # one retry, told exactly what was wrong: rubric coverage is the judge's
        # contract, and a single miss is usually a formatting slip
        text, meta = client.complete(
            _SYSTEM, user + f"\n\nYour previous reply was rejected: {first}. "
            "Reply with the JSON object only, covering every item id."
        )
        parsed, item_scores, triggered = _parse_judgment(rubric, text)
    score = rubric.compute_score(item_scores, triggered)
    factors = {p.id: p.factor for p in rubric.penalties}
    return Judgment(
        item_scores=item_scores,
        penalties={pid: factors[pid] for pid in triggered},
        score=score,
        raw={"text": text, "parsed": parsed},
        request=meta,
    )


def judge_rollout(
    rollout_dir: str | Path,
    *,
    model: str,
    base_url: str,
    task_dir: str | Path | None = None,
    tasks_root: str | Path = "tasks",
    rubric_path: str | Path | None = None,
    method: str | None = None,
    repeats: int = 1,
    temperature: float = 0.0,
    runs_root: str | Path | None = None,
    blind: bool = True,
) -> dict:
    """Judge one finished rollout and append the verdict(s) to its journal."""
    rollout_dir = Path(rollout_dir)
    events = read_journal(rollout_dir / "journal.jsonl")
    started = next((e for e in events if e.get("type") == "rollout_started"), None)
    if started is None:
        raise ValueError(f"{rollout_dir}: no rollout_started event")
    task_dir = Path(task_dir) if task_dir else Path(tasks_root) / started["task_name"]

    cfg = load_task(task_dir)
    method = method or (cfg.judge.method if cfg.judge else "absolute")
    if method != "absolute":
        raise NotImplementedError(
            f"judge method {method!r} needs reference trajectories in the task's "
            'oracle/ dir; only "absolute" ships today'
        )
    if rubric_path is None:
        if cfg.judge is None:
            raise ValueError(
                f"{task_dir}: task declares no [judge]; pass --rubric to judge it anyway"
            )
        rubric_path = task_dir / cfg.judge.rubric
    rubric = load_rubric(rubric_path)
    rubric_rel = str(Path(rubric_path).relative_to(task_dir)) if Path(rubric_path).is_relative_to(task_dir) else str(rubric_path)

    transcript = render(
        build_transcript(rollout_dir, events),
        blind=blind_map(events) if blind else None,
    )
    if len(transcript) > _MAX_TRANSCRIPT_CHARS:
        half = _MAX_TRANSCRIPT_CHARS // 2
        transcript = (
            transcript[:half]
            + f"\n\n[... {len(transcript) - _MAX_TRANSCRIPT_CHARS} chars of the "
            "middle of this run omitted ...]\n\n"
            + transcript[-half:]
        )

    client = make_judge(model, base_url=base_url, temperature=temperature)
    spec = client.spec()
    pass_id = uuid.uuid4().hex[:12]
    rubric_sha = hashlib.sha256(Path(rubric_path).read_bytes()).hexdigest()[:16]
    judgments: list[Judgment] = []
    with JournalWriter(rollout_dir, started["rollout_id"]) as writer:
        for i in range(repeats):
            t0 = time.monotonic()
            j = judge_once(client, rubric, transcript)
            seq = writer._seq  # sidecar naming only
            ref = writer.write_sidecar(
                f"judge_{seq}.json",
                json.dumps(
                    {**j.request, "parsed": j.raw["parsed"], "text": j.raw["text"],
                     "elapsed_s": time.monotonic() - t0},
                    indent=1, default=str,
                ),
            )
            writer.emit(
                JudgeResult, stream="env", judge=spec, rubric=rubric_rel,
                pass_id=pass_id, rubric_sha256=rubric_sha,
                method="absolute", item_scores=j.item_scores,
                penalties=j.penalties, score=j.score, rationale_ref=ref,
            )
            judgments.append(j)

    scores = [j.score for j in judgments]
    summary = {
        "rollout_id": started["rollout_id"],
        "judge": model,
        "pass_id": pass_id,
        "rubric": rubric_rel,
        "repeats": repeats,
        "score": statistics.fmean(scores),
        "score_sd": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "item_scores": {
            item.id: statistics.fmean([j.item_scores[item.id] for j in judgments])
            for item in rubric.items
        },
        "penalties": sorted({pid for j in judgments for pid in j.penalties}),
    }

    runs_root = Path(runs_root) if runs_root else rollout_dir.parent
    db = runs_root / "index.db"
    if db.exists():
        index = Index(db)
        for name, value in (compute_metrics(rollout_dir / "journal.jsonl") or {}).items():
            index.metric(started["rollout_id"], name, value)
        index.close()
    return summary
