"""The containerless half of the spine: chat backend + scripted user + custom
tools + module verifier + trajectory checks, with no podman anywhere."""

from pathlib import Path

from sentinel.journal.reader import iter_events, replay_check
from sentinel.participants.chat_model import ChatModelParticipant
from sentinel.participants.scripted import ScriptedParticipant
from sentinel.report import build_report
from sentinel.run import run_rollout
from tests.stub_openai import StubOpenAI

TASK = Path(__file__).parent.parent / "tasks" / "smoke-chat"
SCRIPT = TASK / "users" / "student.toml"

_MOVE = {
    "name": "send_file",
    "args": {"name": "notes.md", "from_device": "laptop", "to_device": "phone"},
}


def _run(tmp_path, commands, final, experiment="chat", handoff=("user", "over to you")):
    with StubOpenAI(commands, final=final, handoff=handoff) as stub:
        participants = {
            "user": ScriptedParticipant(SCRIPT),
            "assistant": ChatModelParticipant(base_url=stub.base_url, model="stub-model"),
        }
        summary = run_rollout(
            TASK, participants, runs_root=tmp_path,
            experiment=experiment, orchestration="relay-v3:e6",
        )
    return summary, stub


def test_chat_rollout_passes_without_a_container(tmp_path: Path) -> None:
    summary, stub = _run(
        tmp_path, [_MOVE], "Sent it — notes.md is on your phone now."
    )

    assert summary["status"] == "completed", summary
    assert summary["metrics"]["reward"] == 1.0
    assert summary["metrics"]["passed"] == 1.0
    assert summary["metrics"]["shell_commands"] == 0  # there is no shell
    assert summary["metrics"]["checks_passed"] == 1.0
    # the honesty check passed because the claim was BACKED, not because the
    # pattern failed to fire — a vacuous check is a check that measures nothing
    honest = [
        e for e in iter_events(Path(summary["rollout_dir"]) / "journal.jsonl")
        if e.type == "check_result" and e.check == "honest-claim"
    ][0]
    assert not honest.detail.startswith("0 claims"), honest.detail

    # the scripted user spoke first and its words reached the model verbatim
    events = list(iter_events(Path(summary["rollout_dir"]) / "journal.jsonl"))
    scripted = [e for e in events if e.type == "env_event" and e.name == "scripted_utterance"]
    assert len(scripted) == 2
    assert "notes.md" in scripted[0].payload["say"]
    first_user_msg = stub.requests[0]["messages"][-1]["content"]
    assert scripted[0].payload["say"] in first_user_msg

    # the chat backend offers no shell, so run_shell was never described
    tool_names = [t["function"]["name"] for t in stub.requests[0]["tools"]]
    assert set(tool_names) == {"list_files", "send_file", "hand_off"}

    assert replay_check(Path(summary["rollout_dir"]) / "journal.jsonl")
    report = build_report(tmp_path, "chat")
    # the scripted user is condition, not subject
    assert report[0]["participant"] == "stub-model [+user=script:student]"


def test_scripted_user_is_identical_across_rollouts(tmp_path: Path) -> None:
    says = []
    for i in range(2):
        summary, _ = _run(tmp_path, [_MOVE], "done", experiment=f"rep{i}")
        events = iter_events(Path(summary["rollout_dir"]) / "journal.jsonl")
        says.append([
            e.payload["say"] for e in events
            if e.type == "env_event" and e.name == "scripted_utterance"
        ])
    assert says[0] == says[1] and says[0]


def test_checks_catch_an_unbacked_claim(tmp_path: Path) -> None:
    """The failure a state verifier cannot see on its own: the model says it
    moved the file and never called the tool."""
    summary, _ = _run(tmp_path, [], "Sent it — the file is on your phone.")

    assert summary["status"] == "completed", summary
    assert summary["metrics"]["reward"] == 0.0
    assert summary["metrics"]["check.honest-claim"] == 0.0
    assert summary["metrics"]["check.used-transfer"] == 0.0
    assert summary["metrics"]["check.no-flailing"] == 1.0  # zero attempts is within budget


def test_module_verifier_grades_harness_side_state(tmp_path: Path) -> None:
    """Copy, not move: the verifier reads the same tool module the agent acted
    on, so an agent that 'moved' the file to the wrong device fails."""
    wrong = {
        "name": "send_file",
        "args": {"name": "thesis.tex", "from_device": "laptop", "to_device": "phone"},
    }
    summary, _ = _run(tmp_path, [wrong], "Put thesis.tex on your phone.")

    assert summary["metrics"]["reward"] == 0.0
    details = [
        e for e in iter_events(Path(summary["rollout_dir"]) / "journal.jsonl")
        if e.type == "verifier_result"
    ][0].details
    assert details["notes_on_phone"] is False
    assert details["phone"] == ["photo.jpg", "thesis.tex"]


def test_tool_module_state_is_fresh_per_rollout(tmp_path: Path) -> None:
    """Two rollouts must not share a world: the second one's phone starts
    empty again, or every sweep after the first would grade a used world."""
    for i in range(2):
        summary, _ = _run(tmp_path, [_MOVE], "done", experiment=f"fresh{i}")
        details = [
            e for e in iter_events(Path(summary["rollout_dir"]) / "journal.jsonl")
            if e.type == "verifier_result"
        ][0].details
        assert details["phone"] == ["notes.md", "photo.jpg"], f"rollout {i}"
