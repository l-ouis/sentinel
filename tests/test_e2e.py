"""Full-spine end-to-end test: real Podman container, stub model server.
Proves journal -> env -> participant -> verifier -> index -> report blind."""

import shutil
import subprocess
from pathlib import Path

import pytest

from sentinel.journal.reader import iter_events
from sentinel.participants.chat_model import ChatModelParticipant
from sentinel.report import build_report
from sentinel.run import run_rollout
from tests.stub_openai import StubOpenAI

TASK = Path(__file__).parent.parent / "tasks" / "smoke-shell"

podman_available = shutil.which("podman") and (
    subprocess.run(["podman", "info"], capture_output=True).returncode == 0
)
pytestmark = pytest.mark.skipif(not podman_available, reason="podman unavailable")


def test_smoke_rollout_passes(tmp_path: Path) -> None:
    # no mkdir: /work existing proves the init script ran, and /sentinel being
    # gone proves it erased itself before the agent acted
    commands = [
        "test -d /work && ! test -e /sentinel && printf 'hello sentinel' > /work/hello.txt"
    ]
    with StubOpenAI(commands) as stub:
        participant = ChatModelParticipant(base_url=stub.base_url, model="stub-model")
        summary = run_rollout(TASK, participant, runs_root=tmp_path, experiment="smoke")

    assert summary["status"] == "completed", summary
    assert summary["metrics"]["reward"] == 1.0
    assert summary["metrics"]["passed"] == 1.0
    assert summary["metrics"]["shell_commands"] == 1
    assert summary["metrics"]["completion_tokens"] == 40  # 2 calls x 20

    # journal shape: the minimal happy path, in order
    types = [e.type for e in iter_events(Path(summary["rollout_dir"]) / "journal.jsonl")]
    assert types == [
        "rollout_started", "provisioned", "env_event", "session_start",
        "env_event",  # endpoint probe: how this model is being served
        "model_call", "action", "action_result", "model_call",
        "session_end", "verifier_result", "rollout_finished",
    ]

    report = build_report(tmp_path, "smoke")
    assert len(report) == 1
    assert report[0]["participant"] == "stub-model"
    assert report[0]["pass^1"] == 1.0


def test_smoke_pair_turns(tmp_path: Path) -> None:
    """turns-v1: worker botches the file, checker inspects and fixes it."""
    pair_task = Path(__file__).parent.parent / "tasks" / "smoke-pair"
    worker_cmds = ["printf 'hello sentinal' > /work/hello.txt"]  # (sic) botched
    checker_cmds = [
        "cat /work/hello.txt",
        "printf 'hello sentinel' > /work/hello.txt",
    ]
    with StubOpenAI(worker_cmds) as worker, StubOpenAI(checker_cmds) as checker:
        participants = {
            "worker": ChatModelParticipant(base_url=worker.base_url, model="stub-worker"),
            "checker": ChatModelParticipant(base_url=checker.base_url, model="stub-checker"),
        }
        summary = run_rollout(
            pair_task, participants, runs_root=tmp_path,
            experiment="pair", orchestration="turns-v1:r1",
        )

    assert summary["status"] == "completed", summary
    assert summary["metrics"]["reward"] == 1.0
    # per-role attribution: the checker did the extra work
    assert summary["metrics"]["role.worker.shell_commands"] == 1
    assert summary["metrics"]["role.checker.shell_commands"] == 2

    events = list(iter_events(Path(summary["rollout_dir"]) / "journal.jsonl"))
    sessions = [e for e in events if e.type == "session_start"]
    assert [s.roles for s in sessions] == [["worker"], ["checker"]]
    assert [s.episode for s in sessions] == [0, 1]

    report = build_report(tmp_path, "pair")
    assert report[0]["participant"] == "stub-worker + stub-checker"


def test_relay_code_e2e(tmp_path: Path) -> None:
    """relay-v2 + briefings: alpha must ping charlie (not round-robin bravo),
    each agent appends its secret fragment, bravo closes the relay."""
    relay_task = Path(__file__).parent.parent / "tasks" / "relay-code"
    stubs = {
        "alpha": StubOpenAI(
            ["printf '52' > /work/code.txt"],
            final="first fragment in.", handoff=("charlie", "fragment one is placed"),
        ),
        "bravo": StubOpenAI(
            ["cat /work/code.txt", "printf '99' >> /work/code.txt"],
            final="code complete.", handoff=("none", "done"),
        ),
        "charlie": StubOpenAI(
            ["printf '81' >> /work/code.txt"],
            final="appended mine.", handoff=("bravo", "two fragments in"),
        ),
    }
    for s in stubs.values():
        s.__enter__()
    try:
        participants = {
            name: ChatModelParticipant(base_url=s.base_url, model=f"stub-{name}")
            for name, s in stubs.items()
        }
        summary = run_rollout(
            relay_task, participants, runs_root=tmp_path,
            experiment="relay", orchestration="relay-v2:e6",
        )
    finally:
        for s in stubs.values():
            s.__exit__()

    assert summary["status"] == "completed", summary
    assert summary["metrics"]["reward"] == 1.0, summary["metrics"]
    events = list(iter_events(Path(summary["rollout_dir"]) / "journal.jsonl"))
    order = [e.roles[0] for e in events if e.type == "session_start"]
    assert order == ["alpha", "charlie", "bravo"]
    # each agent's secret briefing reached only its own system prompt
    assert summary["metrics"]["role.charlie.shell_commands"] == 1


def test_calendar_negotiation_e2e(tmp_path: Path) -> None:
    """Plumbing check with scripted agents: propose, agree, both record."""
    task = Path(__file__).parent.parent / "tasks" / "calendar-negotiation"
    stubs = {
        "avery": StubOpenAI(
            ["printf 'Thu 14:00' > /work/avery.txt"],
            final="Recorded on avery's side.",
            handoff=("blake", "Thu 14:00 works for avery — please record it too."),
        ),
        "blake": StubOpenAI(
            ["printf 'Thu 14:00' > /work/blake.txt"],
            final="Confirmed for blake.", handoff=("none", "locked in"),
        ),
    }
    for s in stubs.values():
        s.__enter__()
    try:
        participants = {
            name: ChatModelParticipant(base_url=s.base_url, model=f"stub-{name}")
            for name, s in stubs.items()
        }
        summary = run_rollout(
            task, participants, runs_root=tmp_path,
            experiment="cal", orchestration="relay-v2:e6",
        )
    finally:
        for s in stubs.values():
            s.__exit__()

    assert summary["status"] == "completed", summary
    assert summary["metrics"]["reward"] == 1.0, summary["metrics"]
    report = build_report(tmp_path, "cal")
    assert report[0]["participant"] == "stub-avery + stub-blake"


def test_vault_crack_e2e(tmp_path: Path) -> None:
    """Custom task tool end-to-end: the scout (no unlock tool) relays its
    half; the operator assembles the code and opens the harness-side lock."""
    task = Path(__file__).parent.parent / "tasks" / "vault-crack"
    stubs = {
        "scout": StubOpenAI(
            [], final="Relaying my half.",
            handoff=("operator", "First three digits: 473. Your turn."),
        ),
        "operator": StubOpenAI(
            [
                {"name": "unlock", "args": {"code": "473000"}},  # botched once
                {"name": "unlock", "args": {"code": "473952"}},
            ],
            final="Vault open.", handoff=("none", "done"),
        ),
    }
    for s in stubs.values():
        s.__enter__()
    try:
        participants = {
            name: ChatModelParticipant(base_url=s.base_url, model=f"stub-{name}")
            for name, s in stubs.items()
        }
        summary = run_rollout(
            task, participants, runs_root=tmp_path,
            experiment="vault", orchestration="relay-v2:e4",
        )
    finally:
        for s in stubs.values():
            s.__exit__()

    assert summary["status"] == "completed", summary
    assert summary["metrics"]["reward"] == 1.0, summary["metrics"]

    # per-role tool gating was advertised, not just enforced: the scout's
    # requests never offered unlock, the operator's did
    scout_tools = [t["function"]["name"] for t in stubs["scout"].requests[0]["tools"]]
    op_tools = [t["function"]["name"] for t in stubs["operator"].requests[0]["tools"]]
    assert "unlock" not in scout_tools and "unlock" in op_tools

    # both attempts journaled with results
    events = list(iter_events(Path(summary["rollout_dir"]) / "journal.jsonl"))
    attempts = [e for e in events if e.type == "tool_call" and e.name == "unlock"]
    assert [a.result["unlocked"] for a in attempts] == [False, True]


def test_smoke_rollout_fails_cleanly(tmp_path: Path) -> None:
    commands = ["printf 'wrong contents' > /work/hello.txt"]
    with StubOpenAI(commands) as stub:
        participant = ChatModelParticipant(base_url=stub.base_url, model="stub-model")
        summary = run_rollout(TASK, participant, runs_root=tmp_path)

    assert summary["status"] == "completed"  # rollout ran fine; the task failed
    assert summary["metrics"]["reward"] == 0.0
    assert summary["metrics"]["passed"] == 0.0
