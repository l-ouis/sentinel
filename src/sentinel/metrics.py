"""Rollout metrics derived from the journal alone — the live runner and
`sentinel reindex` call the same function, so they can never disagree.
Multi-role rollouts additionally get per-role breakdowns namespaced
"role.<name>.<metric>", attributed via each event's stream."""

from __future__ import annotations

from pathlib import Path

from sentinel.journal.reader import iter_events


def _fold_judgments(judgments: list) -> dict[str, float]:
    """Only the most recent judging pass counts. Judgments append to the
    journal forever (a rollout judged in March and re-judged in June keeps
    both), so metrics take the trailing run of judgments that agree on judge
    model, rubric, and method — i.e. exactly one `sentinel judge` invocation's
    repeats — and average those. Older passes stay in the journal as history."""
    if not judgments:
        return {}
    def key(ev) -> tuple:
        # pass_id is the real identity; the rest is the fallback for judgments
        # written before it existed
        if ev.pass_id:
            return ("pass", ev.pass_id)
        return ((ev.judge.model or ""), ev.rubric or "", ev.method,
                ev.rubric_sha256 or "")
    last = key(judgments[-1])
    group = []
    for ev in reversed(judgments):
        if key(ev) != last:
            break
        group.append(ev)
    scores = [ev.score for ev in group]
    out = {
        "judge.score": sum(scores) / len(scores),
        "judge.n": float(len(group)),
        "judge.score_sd": (
            (sum((x - sum(scores) / len(scores)) ** 2 for x in scores) / len(scores)) ** 0.5
            if len(scores) > 1 else 0.0
        ),
    }
    for item in group[0].item_scores:
        vals = [ev.item_scores.get(item, 0.0) for ev in group]
        out[f"judge.item.{item}"] = sum(vals) / len(vals)
    # fraction of the pass's judgments that triggered each penalty: 1 of 3 is
    # a disagreement worth seeing, not a verdict
    triggered: dict[str, int] = {}
    for ev in group:
        for pid in ev.penalties:
            triggered[pid] = triggered.get(pid, 0) + 1
    for pid, count in triggered.items():
        out[f"judge.penalty.{pid}"] = count / len(group)
    return out


def compute_metrics(journal_path: str | Path) -> dict[str, float] | None:
    """All numeric metrics for one rollout; None if nothing scored it at all
    (no verifier, no checks, no judge) — an ungraded rollout has no result to
    report, only a trajectory."""
    reward: float | None = None
    passed = False
    model_calls = actions = 0
    prompt = completion = 0
    gen_ms = cost = 0.0
    first_start = last_end = None
    per_role: dict[str, dict[str, float]] = {}
    checks: dict[str, bool] = {}   # last result per check id: re-runs append
    judgments: list = []

    def role(stream: str) -> dict[str, float]:
        return per_role.setdefault(
            stream,
            {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
             "shell_commands": 0, "cost_usd": 0.0},
        )

    for ev in iter_events(journal_path):
        if ev.type == "session_start":
            first_start = first_start or ev.ts_wall
        elif ev.type == "session_end":
            last_end = ev.ts_wall
        elif ev.type == "model_call":
            model_calls += 1
            prompt += ev.prompt_tokens or 0
            completion += ev.completion_tokens or 0
            gen_ms += ev.latency_ms or 0.0
            cost += ev.cost_usd or 0.0
            r = role(ev.stream)
            r["model_calls"] += 1
            r["prompt_tokens"] += ev.prompt_tokens or 0
            r["completion_tokens"] += ev.completion_tokens or 0
            r["cost_usd"] += ev.cost_usd or 0.0
        elif ev.type == "action":
            actions += 1
            role(ev.stream)["shell_commands"] += 1
        elif ev.type == "verifier_result":
            reward = ev.reward
            passed = bool(ev.passed)
        elif ev.type == "check_result":
            checks[ev.check] = ev.passed
        elif ev.type == "judge_result":
            judgments.append(ev)

    if reward is None and not checks and not judgments:
        return None
    wall = (
        (last_end - first_start).total_seconds()
        if first_start and last_end
        else 0.0
    )
    metrics: dict[str, float] = {
        "wall_seconds": wall,
        "model_calls": model_calls,
        "shell_commands": actions,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "tok_per_sec": completion / (gen_ms / 1000) if gen_ms > 0 else 0.0,
        "cost_usd": cost,
    }
    if reward is not None:
        # reward stays verifier-owned: checks and judges add columns, they
        # never move the headline number (PLAN §10 keeps the layers apart)
        metrics["reward"] = float(reward)
        metrics["passed"] = 1.0 if passed else 0.0
    if checks:
        for name, ok in checks.items():
            metrics[f"check.{name}"] = 1.0 if ok else 0.0
        metrics["checks_passed"] = sum(checks.values()) / len(checks)
        metrics["checks_n"] = float(len(checks))
    metrics.update(_fold_judgments(judgments))
    if len(per_role) > 1:
        for name, vals in per_role.items():
            for k, v in vals.items():
                metrics[f"role.{name}.{k}"] = v
    return metrics
