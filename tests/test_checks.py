"""Trajectory checks over a real journal (PLAN §10 layer 2)."""

from pathlib import Path

import pytest

from sentinel.checks import evaluate, run_checks
from sentinel.participants.chat_model import ChatModelParticipant
from sentinel.participants.scripted import ScriptedParticipant
from sentinel.run import run_rollout
from sentinel.schemas.task import CheckConfig
from tests.stub_openai import StubOpenAI

TASK = Path(__file__).parent.parent / "tasks" / "smoke-chat"
SCRIPT = TASK / "users" / "student.toml"


@pytest.fixture(scope="module")
def rollout(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("checks")
    move = {
        "name": "send_file",
        "args": {"name": "notes.md", "from_device": "laptop", "to_device": "phone"},
    }
    with StubOpenAI([move], final="Copied notes.md over to your phone.") as stub:
        summary = run_rollout(
            TASK,
            {
                "user": ScriptedParticipant(SCRIPT),
                "assistant": ChatModelParticipant(base_url=stub.base_url, model="stub"),
            },
            runs_root=tmp, orchestration="relay-v3:e6",
        )
    assert summary["status"] == "completed", summary
    return Path(summary["rollout_dir"])


def check(**kw) -> CheckConfig:
    return CheckConfig(**kw)


def test_tool_called_and_not_called(rollout: Path) -> None:
    assert evaluate(check(id="a", kind="tool_called", tool="send_file"), rollout, _ev(rollout)).passed
    assert not evaluate(check(id="b", kind="tool_called", tool="list_files"), rollout, _ev(rollout)).passed
    assert evaluate(check(id="c", kind="tool_not_called", tool="list_files"), rollout, _ev(rollout)).passed


def test_budgets(rollout: Path) -> None:
    events = _ev(rollout)
    assert evaluate(check(id="d", kind="max_tool_calls", tool="send_file", limit=1), rollout, events).passed
    assert not evaluate(check(id="e", kind="max_tool_calls", tool="send_file", limit=0), rollout, events).passed
    assert evaluate(check(id="f", kind="max_actions", limit=0), rollout, events).passed  # no shell


def test_args_and_results(rollout: Path) -> None:
    events = _ev(rollout)
    assert evaluate(
        check(id="g", kind="tool_args_match", tool="send_file", pattern="notes\\.md"), rollout, events
    ).passed
    assert evaluate(
        check(id="h", kind="tool_result_match", tool="send_file", pattern='"sent": true'), rollout, events
    ).passed
    assert not evaluate(
        check(id="i", kind="tool_args_match", tool="send_file", pattern="thesis"), rollout, events
    ).passed


def test_negated_args_and_results(rollout: Path) -> None:
    """The negated forms are the only way to assert on what a structured tool
    call DELIVERED — hand_off's message never reaches the model's reply text,
    so said_not_matches cannot see it."""
    events = _ev(rollout)
    assert evaluate(
        check(id="l", kind="tool_args_not_match", tool="send_file", pattern="thesis"), rollout, events
    ).passed
    assert not evaluate(
        check(id="m", kind="tool_args_not_match", tool="send_file", pattern="notes\\.md"), rollout, events
    ).passed
    assert evaluate(
        check(id="n", kind="tool_result_not_match", tool="send_file", pattern='"sent": false'), rollout, events
    ).passed


def test_said_matches_reads_the_sidecar_text(rollout: Path) -> None:
    events = _ev(rollout)
    said = evaluate(check(id="j", kind="said_matches", pattern="(?i)copied"), rollout, events)
    assert said.passed and said.evidence
    assert not evaluate(
        check(id="k", kind="said_matches", pattern="zzz-not-in-transcript"), rollout, events
    ).passed


def test_claim_without_tool_is_honest_by_default(rollout: Path) -> None:
    events = _ev(rollout)
    # claim made AND backed by the call -> pass
    assert evaluate(
        check(id="l", kind="claim_without_tool", tool="send_file", pattern="(?i)copied"),
        rollout, events,
    ).passed
    # a claim about a tool that was never called -> fail
    assert not evaluate(
        check(id="m", kind="claim_without_tool", tool="list_files", pattern="(?i)copied"),
        rollout, events,
    ).passed
    # no claim at all -> pass, without needing the call
    assert evaluate(
        check(id="n", kind="claim_without_tool", tool="list_files", pattern="(?i)nonsense"),
        rollout, events,
    ).passed


def test_role_scoping(rollout: Path) -> None:
    events = _ev(rollout)
    assert evaluate(
        check(id="o", kind="tool_called", tool="send_file", role="assistant"), rollout, events
    ).passed
    assert not evaluate(
        check(id="p", kind="tool_called", tool="send_file", role="user"), rollout, events
    ).passed


def test_run_checks_reads_the_journal_itself(rollout: Path) -> None:
    outcomes = run_checks([check(id="q", kind="tool_called", tool="send_file")], rollout)
    assert [o.id for o in outcomes] == ["q"] and outcomes[0].passed


def test_schema_rejects_incomplete_checks() -> None:
    with pytest.raises(ValueError):
        CheckConfig(id="x", kind="tool_called")           # no tool
    with pytest.raises(ValueError):
        CheckConfig(id="y", kind="said_matches", pattern="(unclosed")


def _ev(rollout: Path) -> list[dict]:
    from sentinel.transcript import read_journal

    return read_journal(rollout / "journal.jsonl")
