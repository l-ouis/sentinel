"""Experiment = task x participants x repeats (PLAN §2). Sequential by
default — a single local endpoint serves one model at a time; `parallel` runs
rollouts concurrently, which suits claude: participants (no shared endpoint;
each rollout is its own container + CLI session). Beware CPU-bound tasks:
concurrent rollouts contend for cores, so wall-time metrics blur."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from sentinel.participants import CLAUDE_PREFIX, SCRIPT_PREFIX, make_participant
from sentinel.run import run_rollout
from sentinel.schemas.task import load_task


def _warmup(base_url: str, model: str) -> None:
    """JIT-loading endpoints (LM Studio) pay the model-load cost on the first
    request; take it here so rollout latency metrics measure generation."""
    try:
        httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 1},
            timeout=600.0,
        )
    except httpx.HTTPError:
        pass  # the rollout itself will surface real endpoint failures


def _subject_role(task_dir: str | Path, fixtures: dict[str, str]) -> str:
    """The one role the sweep varies: every other role must be pinned as a
    fixture. Varying two roles at once measures the pair, not the model."""
    roles = [r.name for r in load_task(task_dir).participants.roles]
    unknown = set(fixtures) - set(roles)
    if unknown:
        raise ValueError(f"fixture roles {sorted(unknown)} are not task roles {roles}")
    free = [r for r in roles if r not in fixtures]
    if len(free) != 1:
        raise ValueError(
            f"an experiment varies exactly one role; roles {roles} minus fixtures "
            f"{sorted(fixtures)} leaves {free}. Bind the others with --bind role=spec."
        )
    return free[0]


def run_experiment(
    task_dir: str | Path,
    *,
    name: str,
    models: list[str],
    repeats: int,
    base_url: str,
    temperature: float | None = None,
    runs_root: str | Path = "runs",
    parallel: int = 1,
    fixtures: dict[str, str] | None = None,
    orchestration: str = "solo-v1",
    on_rollout=None,
) -> list[dict]:
    for model in dict.fromkeys(models):
        if not model.startswith((CLAUDE_PREFIX, SCRIPT_PREFIX)):
            _warmup(base_url, model)  # claude CLI has no JIT model load

    fixtures = fixtures or {}
    subject = _subject_role(task_dir, fixtures) if fixtures else None
    lock = threading.Lock()

    def one(model: str, repeat: int) -> dict:
        # built per rollout, never shared: a scripted fixture carries its
        # position in the script, so reusing one would resume mid-conversation
        participant = make_participant(
            model, base_url=base_url, temperature=temperature, task_dir=task_dir
        )
        bound = (
            participant
            if not fixtures
            else {
                subject: participant,
                **{
                    role: make_participant(
                        spec, base_url=base_url, temperature=None, task_dir=task_dir
                    )
                    for role, spec in fixtures.items()
                },
            }
        )
        summary = run_rollout(
            task_dir, bound, runs_root=runs_root, experiment=name,
            orchestration=orchestration,
        )
        summary["model"] = model
        summary["repeat"] = repeat
        if on_rollout:
            with lock:
                on_rollout(summary)
        return summary

    jobs = [(model, i) for model in models for i in range(repeats)]
    if parallel <= 1:
        return [one(model, i) for model, i in jobs]
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(one, model, i) for model, i in jobs]
        return [f.result() for f in futures]
