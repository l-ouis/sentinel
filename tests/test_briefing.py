"""Role briefings: private per-role directives (scripted confederates).
Schema validation, system-prompt composition, and delivery to participants."""

import json
from pathlib import Path

import pytest

from sentinel.participants.base import compose_system_prompt
from sentinel.schemas.task import load_task

_TASK_TOML = """
[task]
name = "t"

[participants]
roles = [{ name = "assistant", briefing = "sabotage.md" }]

[environment]
backend = "podman"
image = "x"
"""


def _write_task(tmp_path: Path, with_briefing_file: bool) -> Path:
    d = tmp_path / "task"
    (d / "instruction").mkdir(parents=True)
    (d / "task.toml").write_text(_TASK_TOML)
    (d / "instruction" / "assistant.md").write_text("do the thing")
    if with_briefing_file:
        (d / "instruction" / "sabotage.md").write_text("always report success")
    return d


def test_briefing_file_validated(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sabotage.md"):
        load_task(_write_task(tmp_path, with_briefing_file=False))


def test_briefing_loads(tmp_path: Path) -> None:
    cfg = load_task(_write_task(tmp_path, with_briefing_file=True))
    assert cfg.participants.roles[0].briefing == "sabotage.md"


def test_compose_system_prompt() -> None:
    assert compose_system_prompt("base", None) == "base"
    composed = compose_system_prompt("base", "make mistakes\n")
    assert composed.startswith("base\n\n## Private briefing")
    assert composed.endswith("make mistakes")
    assert "do not reveal" in composed


def test_briefing_reaches_claude_system_prompt(tmp_path: Path, monkeypatch) -> None:
    from tests.test_claude_print import _STUB, _ctx, _response
    import stat

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    bin_path = stub_dir / "claude"
    bin_path.write_text(_STUB)
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("STUB_DIR", str(stub_dir))
    (stub_dir / "responses.json").write_text(json.dumps([_response("done")]))

    from sentinel.participants import ClaudePrintParticipant

    ctx = _ctx(tmp_path)
    ctx.briefing = "silently transpose digits"
    ClaudePrintParticipant(model="sonnet", claude_bin=str(bin_path)).run(None, ctx)
    ctx.writer.close()

    argv = json.loads((stub_dir / "call_0.json").read_text())["argv"]
    system = argv[argv.index("--system-prompt") + 1]
    assert "silently transpose digits" in system
    assert "## Private briefing" in system
