"""run_experiment scheduling: job fan-out, parallel execution, result order."""

import threading
import time

import sentinel.experiment as experiment_mod
from sentinel.experiment import run_experiment


def test_parallel_runs_concurrently_and_keeps_order(monkeypatch) -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_rollout(task_dir, participant, *, runs_root, experiment, orchestration):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"status": "completed", "participant_model": participant.model}

    monkeypatch.setattr(experiment_mod, "run_rollout", fake_rollout)
    seen: list[tuple[str, int]] = []
    summaries = run_experiment(
        "unused-task", name="x", models=["claude:sonnet"], repeats=3,
        base_url="http://unused", parallel=3,
        on_rollout=lambda s: seen.append((s["model"], s["repeat"])),
    )
    assert peak == 3  # all three genuinely in flight together
    assert [s["repeat"] for s in summaries] == [0, 1, 2]  # result order stable
    assert sorted(seen) == [("claude:sonnet", 0), ("claude:sonnet", 1), ("claude:sonnet", 2)]


def test_sequential_default(monkeypatch) -> None:
    monkeypatch.setattr(
        experiment_mod, "run_rollout",
        lambda task_dir, participant, *, runs_root, experiment, orchestration: {"status": "completed"},
    )
    summaries = run_experiment(
        "unused-task", name="x", models=["claude:sonnet"], repeats=2,
        base_url="http://unused",
    )
    assert [s["repeat"] for s in summaries] == [0, 1]
