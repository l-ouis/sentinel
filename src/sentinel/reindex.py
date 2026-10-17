"""Rebuild runs/index.db from the journals. The journal directory is the
truth (§4); the index is derived and disposable — which is what makes index
schema changes free: change the schema, reindex, done. Metrics go through the
same compute_metrics as the live runner, so the numbers cannot drift."""

from __future__ import annotations

import json
from pathlib import Path

from sentinel.index import Index
from sentinel.metrics import compute_metrics
from sentinel.report import team_label


def reindex(runs_root: str | Path) -> int:
    """Drop and rebuild the index; returns the number of rollouts indexed."""
    runs_root = Path(runs_root)
    db = runs_root / "index.db"
    if db.exists():
        db.unlink()
    index = Index(db)
    n = 0
    for journal_path in sorted(runs_root.glob("*/journal.jsonl")):
        events = []
        with open(journal_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        started = next((e for e in events if e["type"] == "rollout_started"), None)
        if started is None:
            continue
        rid = started["rollout_id"]
        config = started.get("config") or {}
        roles_cfg = config.get("roles") or {}
        bindings = {
            role: {
                **{k: v for k, v in spec.items() if v is not None},
                "graded": roles_cfg.get(role, {}).get("graded", True),
            }
            for role, spec in (started.get("bindings") or {}).items()
        }
        index.rollout_started(
            rid,
            task_name=started["task_name"], task_version=started["task_version"],
            bindings=bindings, subject=team_label(bindings) or "?",
            orchestration=config.get("orchestration", "solo-v1"),
            journal_path=str(journal_path),
            started_wall=started["ts_wall"], experiment=started.get("experiment"),
        )
        finished = next(
            (e for e in reversed(events) if e["type"] == "rollout_finished"), None
        )
        if finished:
            index.rollout_finished(rid, finished["status"], finished["ts_wall"])
        for key, value in (config.get("host") or {}).items():
            if value is not None:
                index.meta(rid, f"host.{key}", value)
        for role, rc in roles_cfg.items():
            if rc.get("briefing_file"):
                index.meta(rid, f"role.{role}.briefing", rc["briefing_file"])
            if rc.get("system_file"):
                index.meta(rid, f"role.{role}.system", rc["system_file"])
            if rc.get("tools"):
                index.meta(rid, f"role.{role}.tools", ",".join(rc["tools"]))
        if finished and finished["status"] == "completed":
            metrics = compute_metrics(journal_path)
            if metrics:
                for name, value in metrics.items():
                    index.metric(rid, name, value)
        n += 1
    index.close()
    return n
