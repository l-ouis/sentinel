"""Sentinel CLI: validate / run / experiment / judge / check / report."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sentinel.schemas.task import load_task

app = typer.Typer(no_args_is_help=True, add_completion=False)

DEFAULT_BASE_URL = "http://localhost:1234/v1"  # LM Studio


@app.callback()
def main() -> None:
    """Sentinel — an evaluation framework for autonomous agents."""


@app.command()
def validate(task_dirs: list[Path]) -> None:
    """Validate task directories against the v0 task.toml schema."""
    failed = False
    for d in task_dirs:
        try:
            cfg = load_task(d)
        except Exception as e:
            failed = True
            typer.secho(f"FAIL {d}: {e}", fg="red")
            continue
        roles = ", ".join(f"{r.name}({r.shape})" for r in cfg.participants.roles)
        typer.secho(
            f"ok   {d}: {cfg.task.name} v{cfg.task.version} — "
            f"roles [{roles}], clock {cfg.clock.mode}, "
            f"backend {cfg.environment.backend}, verifier {cfg.verifier.mode}, "
            f"{len(cfg.events)} events, {len(cfg.checks)} checks, "
            f"{len(cfg.probes)} probes, judge {'on' if cfg.judge else 'off'}",
            fg="green",
        )
    raise typer.Exit(1 if failed else 0)


@app.command()
def run(
    task_dir: Path,
    model: str | None = typer.Option(
        None,
        help='Model name as the endpoint knows it, or "claude:<model>" '
        "(e.g. claude:sonnet) to go through the Claude Code CLI",
    ),
    bind: list[str] = typer.Option(
        [],
        help="Multi-role: repeatable role=model (e.g. --bind worker=claude:sonnet, "
        "--bind user=script:users/student.toml for a scripted fixture)",
    ),
    orchestration: str = typer.Option(
        "solo-v1", help='Harness: "solo-v1", "turns-v1", "turns-v1:r<rounds>"'
    ),
    base_url: str = typer.Option(DEFAULT_BASE_URL),
    temperature: float | None = typer.Option(None),
    max_turns: int = typer.Option(20),
    exec_timeout: float = typer.Option(900.0, help="Per-shell-command timeout (s)"),
    runs_dir: Path = typer.Option(Path("runs")),
    experiment: str | None = typer.Option(None),
) -> None:
    """Run one rollout of a task (OpenAI-compatible endpoint or claude CLI)."""
    from sentinel.participants import make_participant
    from sentinel.run import run_rollout

    if bool(model) == bool(bind):
        raise typer.BadParameter("pass exactly one of --model or --bind role=model ...")

    def make(m: str):
        return make_participant(
            m, base_url=base_url, temperature=temperature,
            max_turns=max_turns, exec_timeout=exec_timeout, task_dir=task_dir,
        )

    if bind:
        pairs = [b.partition("=") for b in bind]
        if any(not role or not m for role, _, m in pairs):
            raise typer.BadParameter(f"malformed --bind (want role=model): {bind}")
        participants: dict | object = {role: make(m) for role, _, m in pairs}
    else:
        participants = make(model)
    summary = run_rollout(
        task_dir, participants, runs_root=runs_dir,
        experiment=experiment, orchestration=orchestration,
    )
    typer.echo(json.dumps(summary, indent=2, default=str))
    raise typer.Exit(0 if summary["status"] == "completed" else 1)


@app.command()
def experiment(
    task_dir: Path,
    name: str = typer.Option(..., help="Experiment name (report key)"),
    model: list[str] = typer.Option(..., help="Repeatable: one per model to compare"),
    repeat: int = typer.Option(1),
    bind: list[str] = typer.Option(
        [],
        help="Pin the non-subject roles: repeatable role=spec "
        "(e.g. --bind user=script:users/student.toml). The one unpinned role "
        "is the subject the sweep varies.",
    ),
    orchestration: str = typer.Option("solo-v1"),
    base_url: str = typer.Option(DEFAULT_BASE_URL),
    temperature: float | None = typer.Option(None),
    runs_dir: Path = typer.Option(Path("runs")),
    parallel: int = typer.Option(
        1,
        help="Concurrent rollouts. Suits claude: models; a single local "
        "endpoint serves one model at a time, and CPU-bound tasks contend.",
    ),
) -> None:
    """Run task x models x repeats, then print the report."""
    from sentinel.experiment import run_experiment
    from sentinel.report import build_report, format_report

    def progress(s: dict) -> None:
        typer.echo(
            f"[{s['model']} #{s['repeat']}] {s['rollout_id']}: {s['status']}"
            + (f" reward={s['metrics']['reward']}" if "metrics" in s else "")
        )

    pairs = [b.partition("=") for b in bind]
    if any(not role or not spec for role, _, spec in pairs):
        raise typer.BadParameter(f"malformed --bind (want role=spec): {bind}")
    run_experiment(
        task_dir, name=name, models=list(model), repeats=repeat,
        base_url=base_url, temperature=temperature, runs_root=runs_dir,
        parallel=parallel, fixtures={role: spec for role, _, spec in pairs},
        orchestration=orchestration, on_rollout=progress,
    )
    typer.echo(format_report(build_report(runs_dir, name)))


def _resolve_rollouts(
    runs_dir: Path, rollouts: list[Path], experiment: str | None
) -> list[Path]:
    """Rollout dirs from explicit paths or from an experiment name. Only
    completed rollouts are scored in bulk — a crashed run's journal is a
    trajectory, not a result."""
    if rollouts and experiment:
        raise typer.BadParameter("pass rollout dirs or --experiment, not both")
    if rollouts:
        return [Path(r) for r in rollouts]
    if not experiment:
        raise typer.BadParameter("pass at least one rollout dir, or --experiment")
    from sentinel.index import Index

    index = Index(runs_dir / "index.db")
    rows = index.query(
        "SELECT rollout_id FROM rollouts WHERE experiment = ? AND status = 'completed' "
        "ORDER BY started_wall",
        (experiment,),
    )
    index.close()
    return [runs_dir / row[0] for row in rows]


@app.command()
def judge(
    rollouts: list[Path] = typer.Argument(None, help="Rollout dirs to judge"),
    model: str = typer.Option(
        ..., help='Judge model — "claude:<model>" for the CLI, else an endpoint model. '
        "Use a stronger model than the ones under test, and never one of them."
    ),
    experiment: str | None = typer.Option(None, help="Judge every completed rollout of it"),
    repeats: int = typer.Option(
        1, help="Independent judgments per rollout; >1 reports the spread"
    ),
    rubric: Path | None = typer.Option(None, help="Override the task's rubric file"),
    task_dir: Path | None = typer.Option(None, help="Override the task dir lookup"),
    tasks_dir: Path = typer.Option(Path("tasks")),
    base_url: str = typer.Option(DEFAULT_BASE_URL),
    temperature: float = typer.Option(0.0),
    blind: bool = typer.Option(True, help="Replace model names with role aliases"),
    runs_dir: Path = typer.Option(Path("runs")),
) -> None:
    """Score finished rollouts against a task rubric with an LLM judge.

    Judges are journal consumers: this re-reads finished rollouts and appends
    its verdicts, so a rubric can be rewritten and re-run without re-running a
    single agent.
    """
    from sentinel.judge import judge_rollout

    targets = _resolve_rollouts(runs_dir, rollouts or [], experiment)
    if not targets:
        typer.secho("no rollouts to judge", fg="yellow")
        raise typer.Exit(1)
    failures = 0
    for target in targets:
        try:
            summary = judge_rollout(
                target, model=model, base_url=base_url, task_dir=task_dir,
                tasks_root=tasks_dir, rubric_path=rubric, repeats=repeats,
                temperature=temperature, runs_root=runs_dir, blind=blind,
            )
        except Exception as e:
            failures += 1
            typer.secho(f"FAIL {target.name}: {type(e).__name__}: {e}", fg="red")
            continue
        spread = f" ±{summary['score_sd']:.3f}" if repeats > 1 else ""
        typer.secho(
            f"{summary['rollout_id']}: judge.score={summary['score']:.3f}{spread} "
            + " ".join(f"{k}={v:.2f}" for k, v in summary["item_scores"].items())
            + (f" penalties={','.join(summary['penalties'])}" if summary["penalties"] else ""),
            fg="green",
        )
    raise typer.Exit(1 if failures else 0)


@app.command()
def check(
    rollouts: list[Path] = typer.Argument(None, help="Rollout dirs to check"),
    experiment: str | None = typer.Option(None, help="Check every completed rollout of it"),
    task_dir: Path | None = typer.Option(None, help="Override the task dir lookup"),
    tasks_dir: Path = typer.Option(Path("tasks")),
    runs_dir: Path = typer.Option(Path("runs")),
) -> None:
    """Re-run a task's trajectory checks over finished rollouts (they also run
    automatically at the end of each rollout). Free and deterministic: edit the
    checks in task.toml, re-run this, and the metrics update."""
    from sentinel.checks import emit_checks
    from sentinel.index import Index
    from sentinel.journal.writer import JournalWriter
    from sentinel.metrics import compute_metrics
    from sentinel.transcript import read_journal

    targets = _resolve_rollouts(runs_dir, rollouts or [], experiment)
    index = Index(runs_dir / "index.db")
    for target in targets:
        events = read_journal(target / "journal.jsonl")
        started = next(e for e in events if e["type"] == "rollout_started")
        cfg = load_task(task_dir or tasks_dir / started["task_name"])
        if not cfg.checks:
            typer.secho(f"{target.name}: task declares no [[checks]]", fg="yellow")
            continue
        with JournalWriter(target, started["rollout_id"]) as writer:
            outcomes = emit_checks(cfg.checks, target, writer)
        for name, value in (compute_metrics(target / "journal.jsonl") or {}).items():
            index.metric(started["rollout_id"], name, value)
        passed = sum(o.passed for o in outcomes)
        typer.secho(
            f"{started['rollout_id']}: {passed}/{len(outcomes)} checks passed"
            + "".join(
                f"\n    {'ok  ' if o.passed else 'FAIL'} {o.id}: {o.detail}"
                for o in outcomes
            ),
            fg="green" if passed == len(outcomes) else "yellow",
        )
    index.close()


@app.command()
def web(
    runs_dir: Path = typer.Option(Path("runs")),
    tasks_dir: Path = typer.Option(Path("tasks")),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8710),
) -> None:
    """Serve the local web console (live status over runs, tasks, and reports)."""
    from sentinel.web import serve

    serve(runs_dir, tasks_dir, host=host, port=port)


@app.command()
def reindex(runs_dir: Path = typer.Option(Path("runs"))) -> None:
    """Rebuild the SQLite index from the journals (journal is the truth)."""
    from sentinel.reindex import reindex as _reindex

    typer.echo(f"reindexed {_reindex(runs_dir)} rollouts")


@app.command()
def report(
    runs_dir: Path = typer.Option(Path("runs")),
    experiment: str | None = typer.Option(None),
) -> None:
    """Aggregate metrics per participant (optionally for one experiment)."""
    from sentinel.report import build_report, format_report

    typer.echo(format_report(build_report(runs_dir, experiment)))


if __name__ == "__main__":
    app()
