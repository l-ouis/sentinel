"""Verifier runner (PLAN §10 layer 1).

Modes:
- inline    the task's verifier/ dir is copied to /sentinel/verifier inside the
            task container, run.sh executes there and writes reward.json next
            to itself: {"reward": <float>, "passed": <bool>, ...}.
- module    harness-side grader: a task-shipped Python file exporting
            grade(ctx: VerifyContext) -> the same dict. For worlds that never
            existed inside a container — a task's tool module holding simulated
            device state, say — where there is nothing in the filesystem to
            inspect. It reads the SAME tool-module object the participants
            acted on, so it grades the world they actually changed.
- none      no state verifier; the task is graded by trajectory checks and/or a
            judge. An honest option, not a degenerate one: "was this answer
            helpful" has no file to stat.
- separate  split grading image (§7.1) — v1.1 follow-up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinel.journal.writer import JournalWriter
from sentinel.schemas.journal import VerifierResult
from sentinel.schemas.task import TaskConfig

_GUEST_DIR = "/sentinel/verifier"


@dataclass
class VerifyContext:
    """What a harness-side grader is handed. The journal is included because a
    module verifier may legitimately grade final state AND read the record of
    how it got there — it just may never touch the live environment."""

    task_dir: Path
    rollout_dir: Path
    journal_path: Path
    tools: Any = None          # the task's imported tool module, if it has one
    env: Any = None            # the live environment, for backends that have one


def _emit(writer: JournalWriter, reward: dict) -> dict:
    writer.emit(
        VerifierResult,
        stream="env",
        reward=float(reward["reward"]),
        passed=reward.get("passed"),
        details={k: v for k, v in reward.items() if k not in ("reward", "passed")},
    )
    return reward


def _run_module_verifier(
    task_dir: Path, cfg: TaskConfig, rollout_dir: Path, tools_module: Any
) -> dict:
    import importlib.util

    path = task_dir / cfg.verifier.module
    spec = importlib.util.spec_from_file_location(
        f"sentinel_task_verifier_{task_dir.name.replace('-', '_')}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grade = getattr(module, "grade", None)
    if not callable(grade):
        raise ValueError(f"{path}: must export grade(ctx) -> {{'reward': float, ...}}")
    result = grade(
        VerifyContext(
            task_dir=task_dir,
            rollout_dir=rollout_dir,
            journal_path=rollout_dir / "journal.jsonl",
            tools=tools_module,
        )
    )
    if not isinstance(result, dict) or "reward" not in result:
        raise ValueError(f"{path}: grade() must return a dict with a 'reward' key")
    (rollout_dir / "reward.json").write_text(json.dumps(result, default=str))
    return result


def run_verifier(
    task_dir: Path,
    cfg: TaskConfig,
    env,
    writer: JournalWriter,
    rollout_dir: Path,
    tools_module: Any = None,
) -> dict | None:
    mode = cfg.verifier.mode
    if mode == "none":
        return None
    if mode == "module":
        return _emit(writer, _run_module_verifier(task_dir, cfg, rollout_dir, tools_module))
    if mode != "inline":
        raise NotImplementedError("verifier mode 'separate' lands in v1.1; use 'inline'")

    env.copy_in(str(task_dir / "verifier"), _GUEST_DIR)
    session = env.open_session()
    result = session.exec(
        f"sh {_GUEST_DIR}/run.sh", timeout=cfg.verifier.timeout.total_seconds()
    )
    if result.exit_code != 0:
        raise RuntimeError(
            f"verifier exited {result.exit_code}: {result.stderr.strip()[:500]}"
        )
    dest = rollout_dir / "reward.json"
    env.copy_out(f"{_GUEST_DIR}/reward.json", str(dest))
    return _emit(writer, json.loads(dest.read_text()))
