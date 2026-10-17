from pathlib import Path

import pytest

from sentinel.journal import JournalWriter, iter_events, replay_check
from sentinel.report import pass_hat_k
from sentinel.schemas.journal import Action, ActionResult, RolloutFinished, RolloutStarted


def _write_journal(tmp_path: Path, with_result: bool = True) -> Path:
    with JournalWriter(tmp_path / "r1", "r1") as w:
        w.emit(RolloutStarted, stream="controller", task_name="t", task_version="0.1", bindings={})
        w.emit(Action, stream="assistant", intent_id="i1", kind="command", data={"command": "ls"})
        if with_result:
            w.emit(ActionResult, stream="assistant", intent_id="i1", status="ok")
        w.emit(RolloutFinished, stream="controller", status="completed")
        return w.path


def test_write_and_replay(tmp_path: Path) -> None:
    path = _write_journal(tmp_path)
    events = list(iter_events(path))
    assert [e.seq for e in events] == [0, 1, 2, 3]
    counts = replay_check(path)
    assert counts == {
        "rollout_started": 1, "action": 1, "action_result": 1, "rollout_finished": 1,
    }


def test_replay_flags_dangling_intent(tmp_path: Path) -> None:
    path = _write_journal(tmp_path, with_result=False)
    with pytest.raises(ValueError, match="dangling"):
        replay_check(path)


def test_pass_hat_k() -> None:
    assert pass_hat_k(10, 10, 5) == 1.0
    assert pass_hat_k(10, 0, 1) == 0.0
    assert pass_hat_k(4, 2, 1) == pytest.approx(0.5)
    # C(2,2)/C(4,2) = 1/6
    assert pass_hat_k(4, 2, 2) == pytest.approx(1 / 6)
