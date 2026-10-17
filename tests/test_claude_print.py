"""ClaudePrintParticipant against a stub `claude` binary: the shell-loop
protocol (sh block -> exec -> feedback -> resume), journal mapping, and the
model-string routing in make_participant."""

import json
import os
import stat
from pathlib import Path

import pytest

from sentinel.envs.base import ExecResult
from sentinel.journal import JournalWriter, iter_events
from sentinel.participants import (
    ChatModelParticipant,
    ClaudePrintParticipant,
    make_participant,
)
from sentinel.participants.base import RolloutContext

_STUB = """#!/usr/bin/env python3
import json, os, sys
if "--version" in sys.argv:
    print("9.9.9 (stub)"); sys.exit(0)
d = os.environ["STUB_DIR"]
n = len([f for f in os.listdir(d) if f.startswith("call_")])
with open(f"{d}/call_{n}.json", "w") as f:
    json.dump({"argv": sys.argv[1:], "stdin": sys.stdin.read()}, f)
responses = json.load(open(f"{d}/responses.json"))
print(json.dumps(responses[min(n, len(responses) - 1)]))
"""


def _response(result: str, session_id: str = "sid-1", **over) -> dict:
    body = {
        "type": "result", "subtype": "success", "is_error": False,
        "result": result, "session_id": session_id,
        "usage": {"input_tokens": 100, "output_tokens": 20,
                  "cache_read_input_tokens": 50, "cache_creation_input_tokens": 0},
        "duration_api_ms": 1500.0, "total_cost_usd": 0.001,
    }
    body.update(over)
    return body


@pytest.fixture
def stub(tmp_path: Path, monkeypatch) -> Path:
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    bin_path = stub_dir / "claude"
    bin_path.write_text(_STUB)
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("STUB_DIR", str(stub_dir))
    return stub_dir


class FakeSession:
    def __init__(self):
        self.commands: list[str] = []

    def exec(self, command: str, timeout: float = 120.0) -> ExecResult:
        self.commands.append(command)
        return ExecResult(0, f"ran:{command}", "", 1.0)


def _ctx(tmp_path: Path) -> RolloutContext:
    writer = JournalWriter(tmp_path / "r1", "r1")
    return RolloutContext(
        writer=writer, role="assistant", instruction="do the thing",
        rollout_dir=tmp_path / "r1",
    )


def _run(stub: Path, tmp_path: Path, responses: list[dict], peers: list[str] | None = None):
    (stub / "responses.json").write_text(json.dumps(responses))
    participant = ClaudePrintParticipant(model="sonnet", claude_bin=str(stub / "claude"))
    session = FakeSession()
    ctx = _ctx(tmp_path)
    ctx.peers = peers
    result = participant.run(session, ctx)
    ctx.writer.close()
    calls = [
        json.loads((stub / f"call_{i}.json").read_text())
        for i in range(len([f for f in os.listdir(stub) if f.startswith("call_")]))
    ]
    return result, session, ctx, calls


def test_shell_loop(stub: Path, tmp_path: Path) -> None:
    result, session, ctx, calls = _run(stub, tmp_path, [
        _response("on it\n```sh\necho hi\n```"),
        _response("all done"),
    ])
    assert result.final == "all done"
    assert session.commands == ["echo hi"]

    # turn 1: instruction, no resume; turn 2: exec feedback, resumed session
    assert calls[0]["stdin"] == "do the thing"
    assert "--resume" not in calls[0]["argv"]
    assert calls[1]["argv"][calls[1]["argv"].index("--resume") + 1] == "sid-1"
    feedback = json.loads(calls[1]["stdin"])
    assert feedback == {"exit_code": 0, "stdout": "ran:echo hi", "stderr": ""}
    # the CLI is invoked as a pure chat transport: all built-in tools off
    assert calls[0]["argv"][calls[0]["argv"].index("--tools") + 1] == ""

    events = list(iter_events(ctx.writer.path))
    assert [e.type for e in events] == ["model_call", "action", "action_result", "model_call"]
    mc = events[0]
    assert mc.prompt_tokens == 150  # input + cache_read + cache_creation
    assert mc.completion_tokens == 20
    assert mc.latency_ms == 1500.0
    assert mc.cost_usd == 0.001
    assert mc.participant.model == "sonnet"
    assert mc.participant.backend == "claude-code-cli"
    assert mc.participant.extra["cli_version"] == "9.9.9 (stub)"


def test_only_first_block_runs(stub: Path, tmp_path: Path) -> None:
    result, session, _, calls = _run(stub, tmp_path, [
        _response("```sh\necho one\n```\nand then\n```bash\necho two\n```"),
        _response("done"),
    ])
    assert session.commands == ["echo one"]
    assert json.loads(calls[1]["stdin"])["note"] == "only the first tool block was executed"


def test_empty_reply_gets_nudged(stub: Path, tmp_path: Path) -> None:
    result, session, _, calls = _run(stub, tmp_path, [
        _response(""),
        _response("summary of work"),
    ])
    assert result.final == "summary of work"
    assert session.commands == []
    assert "no tool block" in calls[1]["stdin"]  # the nudge text


def test_handoff_block(stub: Path, tmp_path: Path) -> None:
    result, session, ctx, calls = _run(stub, tmp_path, [
        _response('```sh\necho prep\n```\n'
                  '```handoff\n{"to": "blake", "message": "your turn"}\n```'),
    ], peers=["avery", "blake"])
    # the sh block ran first, then the turn ended with the structured handoff
    assert session.commands == ["echo prep"]
    assert result.handoff.to == "blake"
    assert result.handoff.message == "your turn"
    # handoff protocol only appears in the system prompt when peers exist
    system = calls[0]["argv"][calls[0]["argv"].index("--system-prompt") + 1]
    assert "```handoff" in system
    events = list(iter_events(ctx.writer.path))
    tool = next(e for e in events if e.type == "tool_call")
    assert tool.name == "hand_off" and tool.args["to"] == "blake"


def test_conversation_token_resumes_cli_session(stub: Path, tmp_path: Path) -> None:
    result, _, ctx, calls = _run(stub, tmp_path, [_response("done", session_id="sid-9")])
    assert result.conversation == "sid-9"
    # a later turn passing the token back resumes that CLI session
    ctx2 = _ctx(tmp_path / "second")
    ctx2.conversation = "sid-9"
    (stub / "responses.json").write_text(json.dumps([_response("continuing")]))
    participant = ClaudePrintParticipant(model="sonnet", claude_bin=str(stub / "claude"))
    participant.run(FakeSession(), ctx2)
    ctx2.writer.close()
    calls = sorted(f for f in os.listdir(stub) if f.startswith("call_"))
    argv = json.loads((stub / calls[-1]).read_text())["argv"]
    assert argv[argv.index("--resume") + 1] == "sid-9"


def test_handoff_none_without_peers(stub: Path, tmp_path: Path) -> None:
    result, _, _, calls = _run(stub, tmp_path, [
        _response('done ```handoff\n{"to": "blake", "message": "x"}\n```'),
    ])
    assert result.handoff is None  # no peers: block is just text
    system = calls[0]["argv"][calls[0]["argv"].index("--system-prompt") + 1]
    assert "```handoff" not in system


def test_cli_error_raises(stub: Path, tmp_path: Path) -> None:
    (stub / "responses.json").write_text(json.dumps(
        [_response("Not logged in · Please run /login", is_error=True)]
    ))
    participant = ClaudePrintParticipant(model="sonnet", claude_bin=str(stub / "claude"))
    ctx = _ctx(tmp_path)
    with pytest.raises(RuntimeError, match="Not logged in"):
        participant.run(FakeSession(), ctx)
    ctx.writer.close()


def test_make_participant_routing() -> None:
    p = make_participant("claude:sonnet", base_url="http://x")
    assert isinstance(p, ClaudePrintParticipant) and p.model == "sonnet"
    q = make_participant("qwen2.5-coder", base_url="http://x", temperature=0.2)
    assert isinstance(q, ChatModelParticipant) and q.model == "qwen2.5-coder"
    with pytest.raises(ValueError, match="claude:<model>"):
        make_participant("claude:", base_url="http://x")
    with pytest.raises(ValueError, match="temperature"):
        make_participant("claude:sonnet", base_url="http://x", temperature=0.5)
