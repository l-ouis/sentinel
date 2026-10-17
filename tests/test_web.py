import json
from pathlib import Path

import pytest

from sentinel.index import Index
from sentinel.journal import JournalWriter
from sentinel.schemas.journal import Action, ActionResult, RolloutStarted
from sentinel.web import build_state, rollout_detail


def _seed_runs(runs: Path) -> None:
    bindings = {"assistant": {"kind": "chat_model", "model": "m1", "graded": True}}
    index = Index(runs / "index.db")
    index.rollout_started(
        "t-1", task_name="t", task_version="0.1",
        bindings=bindings, subject="m1",
        journal_path=str(runs / "t-1" / "journal.jsonl"),
        started_wall="2026-08-12T00:00:00+00:00", experiment="exp-a",
    )
    index.rollout_finished("t-1", "completed", "2026-08-12T00:01:00+00:00")
    index.metric("t-1", "reward", 1.0)
    index.metric("t-1", "passed", 1.0)
    index.rollout_started(
        "t-2", task_name="t", task_version="0.1",
        bindings=bindings, subject="m1",
        journal_path=str(runs / "t-2" / "journal.jsonl"),
        started_wall="2026-08-12T00:02:00+00:00", experiment="exp-b",
    )
    index.close()
    with JournalWriter(runs / "t-2", "t-2") as w:
        w.emit(RolloutStarted, stream="controller", task_name="t", task_version="0.1", bindings={})
        w.emit(Action, stream="assistant", intent_id="i1", kind="command", data={"command": "ls"})
        w.emit(ActionResult, stream="assistant", intent_id="i1", status="ok",
               data={"exit_code": 0, "stdout": "a\nb\n"})


def test_build_state(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _seed_runs(runs)
    state = build_state(runs, tmp_path / "tasks")

    assert state["counts"] == {"completed": 1, "running": 1}
    assert state["experiments"] == ["exp-a", "exp-b"]
    by_id = {r["rollout_id"]: r for r in state["rollouts"]}
    assert by_id["t-1"]["metrics"] == {"reward": 1.0, "passed": 1.0}
    # the running rollout gets a live journal peek
    assert by_id["t-2"]["live_events"] == 3
    assert by_id["t-2"]["live_last"].startswith("ok exit=0")
    assert [r["participant"] for r in state["report"]] == ["m1"]


def test_build_state_experiment_filter(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _seed_runs(runs)
    state = build_state(runs, tmp_path / "tasks", experiment="exp-a")
    assert [r["rollout_id"] for r in state["rollouts"]] == ["t-1"]


def test_build_state_empty(tmp_path: Path) -> None:
    state = build_state(tmp_path / "runs", tmp_path / "tasks")
    assert state["rollouts"] == [] and state["report"] == [] and state["tasks"] == []


def test_rollout_detail(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _seed_runs(runs)
    (runs / "t-2" / "crash.txt").write_text("Traceback: boom")
    detail = rollout_detail(runs, "t-2")
    assert detail["status"] == "running"
    assert detail["total_events"] == 3
    assert detail["events"][1]["text"] == "command: ls"
    assert "ok exit=0" in detail["events"][2]["text"]
    assert "boom" in detail["crash"]


def test_host_fingerprint() -> None:
    from sentinel.host import host_fingerprint

    h = host_fingerprint()
    assert h["hostname"] and h["cores"] >= 1
    assert set(h) == {"hostname", "os", "cpu", "cores", "ram_gb", "gpu"}


def test_rollout_detail_includes_meta(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _seed_runs(runs)
    index = Index(runs / "index.db")
    index.meta("t-1", "host.hostname", "boxy")
    index.meta("t-1", "host.cores", 16)
    index.close()
    detail = rollout_detail(runs, "t-1")
    assert detail["meta"] == {"host.hostname": "boxy", "host.cores": "16"}


def test_sanitize_output() -> None:
    from sentinel.web import _sanitize_output

    # ANSI colors stripped, progress-bar CR overwrites resolved to final state
    assert _sanitize_output("\x1b[32mok\x1b[0m done") == "ok done"
    assert _sanitize_output(" 10%\r 50%\r100%|done\nnext") == "100%|done\nnext"
    assert _sanitize_output("plain\ntext") == "plain\ntext"


def test_rollout_detail_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        rollout_detail(tmp_path, "../../etc/passwd")


def _journal_with_sidecars(runs: Path, rid: str, request: dict, response: dict) -> None:
    from sentinel.schemas.journal import (
        Action, ActionResult, ModelCall, ParticipantSpec, RolloutStarted,
        ToolCall, VerifierResult,
    )

    spec = ParticipantSpec(kind="chat_model", model="m")
    with JournalWriter(runs / rid, rid) as w:
        w.emit(RolloutStarted, stream="controller", task_name="t", task_version="0.1", bindings={})
        req_ref = w.write_sidecar("model_call_1_request.json", json.dumps(request))
        resp_ref = w.write_sidecar("model_call_1_response.json", json.dumps(response))
        w.emit(ModelCall, stream="assistant", participant=spec, prompt_tokens=10,
               completion_tokens=5, latency_ms=100.0, request_ref=req_ref, response_ref=resp_ref)
        w.emit(Action, stream="assistant", intent_id="i1", kind="command", data={"command": "ls"})
        w.emit(ActionResult, stream="assistant", intent_id="i1", status="ok",
               data={"exit_code": 1, "stdout": "out", "stderr": "boom"})
        w.emit(ToolCall, stream="assistant", name="hand_off",
               args={"to": "bravo", "message": "your turn"})
        w.emit(VerifierResult, stream="env", reward=0.5, passed=False, details={"recall": 0.5})


def test_transcript_openai_shape(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _journal_with_sidecars(
        runs, "t-3",
        {"messages": [{"role": "system", "content": "sys prompt"},
                      {"role": "user", "content": "the instruction"}]},
        {"choices": [{"message": {"content": "on it"}}]},
    )
    t = rollout_detail(runs, "t-3")["transcript"]
    assert [(e["kind"], e.get("text")) for e in t] == [
        ("system", "sys prompt"),
        ("user", "the instruction"),
        ("assistant", "on it"),
        ("command", "ls"),
        ("output", "out\n[stderr]\nboom"),
        ("tool", "→ bravo: your turn"),
        ("verifier", '{\n "recall": 0.5\n}'),
    ]
    assert t[2]["completion_tokens"] == 5
    assert t[4]["exit_code"] == 1
    assert t[5]["name"] == "hand_off" and t[5]["stream"] == "assistant"


def test_transcript_claude_shape_strips_executed_fences(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _journal_with_sidecars(
        runs, "t-4",
        {"argv": ["claude", "-p"], "prompt": "the instruction"},
        {
            "result": "on it\n```sh\nls\n```\nnote: later I might run\n```sh\nrm x\n```\n"
                      '```handoff\n{"to": "bravo", "message": "go"}\n```',
            "session_id": "s",
        },
    )
    t = rollout_detail(runs, "t-4")["transcript"]
    kinds = [(e["kind"], e.get("text")) for e in t[:4]]
    # chronological, whole messages: the executed fence is stripped (it
    # renders as the tool call entry that follows); the second, unexecuted
    # sh block stays visible as commentary
    assert kinds == [
        ("user", "the instruction"),
        ("assistant", "on it\n\nnote: later I might run\n```sh\nrm x\n```"),
        ("command", "ls"),
        ("output", "out\n[stderr]\nboom"),
    ]
