"""Experiment report: aggregate metrics per participant from the SQLite index.

pass^k (τ-bench): probability that k independently sampled rollouts ALL pass.
Unbiased estimator from n runs with s passes: C(s,k)/C(n,k).
"""

from __future__ import annotations

from pathlib import Path

from sentinel.index import Index


def pass_hat_k(n: int, s: int, k: int) -> float:
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    p = 1.0
    for i in range(k):
        if s - i <= 0:
            return 0.0
        p *= (s - i) / (n - i)
    return p


def participant_label(spec: dict) -> str:
    parts = [spec.get("model") or spec.get("kind", "?")]
    if spec.get("quant"):
        parts.append(spec["quant"])
    return " ".join(parts)


def team_label(bindings: dict[str, dict]) -> str:
    """Report-row label for a rollout's team. Graded roles ARE the subject;
    fixtures (graded=False) are appended as condition — a subject measured
    against a different fixture is a different experimental condition."""
    graded = [participant_label(s) for s in bindings.values() if s.get("graded", True)]
    fixtures = [
        f"{role}={participant_label(s)}"
        for role, s in bindings.items()
        if not s.get("graded", True)
    ]
    label = " + ".join(graded)
    if fixtures:
        label += " [+" + ", ".join(fixtures) + "]"
    return label


def build_report(runs_root: str | Path, experiment: str | None = None) -> list[dict]:
    index = Index(Path(runs_root) / "index.db")
    where, params = ("WHERE r.experiment = ?", (experiment,)) if experiment else ("", ())
    rows = index.query(
        f"""
        SELECT r.rollout_id, r.subject, r.orchestration, r.status, m.name, m.value
        FROM rollouts r LEFT JOIN metrics m ON m.rollout_id = r.rollout_id
        {where} ORDER BY r.started_wall
        """,
        params,
    )
    index.close()

    # group by (team, harness): the same team under a different orchestration
    # is a different experimental condition and must not blend
    by_condition: dict[tuple[str, str], dict[str, dict]] = {}
    for rollout_id, subject, orchestration, status, mname, mvalue in rows:
        cell = by_condition.setdefault((subject, orchestration), {}).setdefault(
            rollout_id, {"status": status}
        )
        if mname is not None:
            cell[mname] = mvalue

    # columns only appear when some rollout actually carries them: a
    # judge-graded task has no reward, a verifier-graded one has no judge
    # score, and printing 0.000 for "not measured" is a lie either way
    scored = [c for rs in by_condition.values() for c in rs.values()]
    has_verifier = any("passed" in c for c in scored)
    has_checks = any("checks_passed" in c for c in scored)
    has_judge = any("judge.score" in c for c in scored)

    report = []
    for (label, orchestration), rollouts in by_condition.items():
        n = len(rollouts)
        completed = [r for r in rollouts.values() if r["status"] == "completed"]
        s = sum(1 for r in completed if r.get("passed", 0.0) >= 1.0)

        def mean(name: str) -> float | None:
            vals = [r[name] for r in completed if name in r]
            return sum(vals) / len(vals) if vals else None

        row = {
            "participant": label,
            "orch": orchestration,
            "n": n,
            "crashed": n - len(completed),
        }
        if has_verifier:
            # None, not 0.0, when nothing completed: a mean over an empty set
            # is undefined, and printing 0 reads as "attempted and scored zero"
            # when the truth is "never produced a number". pass^k below still
            # counts a crash as a non-pass — failing to run IS a failure of the
            # (model, serving stack) tuple, which is what we measure (§3).
            row["reward"] = mean("reward")
            row["pass^1"] = pass_hat_k(n, s, 1) if n else 0.0
            row["pass^5"] = pass_hat_k(n, s, 5) if n >= 5 else None
        if has_checks:
            row["checks"] = mean("checks_passed")
        if has_judge:
            row["judge"] = mean("judge.score")
            row["judge_sd"] = mean("judge.score_sd")
        row.update(
            {
                "tok/s": mean("tok_per_sec"),
                "wall_s": mean("wall_seconds"),
                "cmds": mean("shell_commands"),
            }
        )
        report.append(row)
    return report


def format_report(report: list[dict]) -> str:
    if not report:
        return "no rollouts found"
    cols = list(dict.fromkeys(c for row in report for c in row))
    widths = {
        c: max(len(c), *(len(_fmt(row.get(c))) for row in report)) for c in cols
    }
    lines = ["  ".join(c.ljust(widths[c]) for c in cols)]
    lines.append("  ".join("-" * widths[c] for c in cols))
    for row in report:
        lines.append("  ".join(_fmt(row.get(c)).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def _fmt(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    return str(v)
