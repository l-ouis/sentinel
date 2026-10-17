"""Tool registry: resolution, renderers (golden against the pre-registry
literals), the CLI block grammar, gating, task tool modules, and dispatch."""

import json
from pathlib import Path

import pytest

from sentinel.journal import JournalWriter, iter_events
from sentinel.participants.base import RolloutContext
from sentinel.tools import ToolContext, resolve_tools, strip_executed_fences

# the exact OpenAI payloads that shipped before the registry existed —
# renderer changes to these are scaffold-surface changes and must be loud
_GOLDEN_RUN_SHELL = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": (
            "Run a shell command on the machine (bash -c). Do not wrap the "
            "command in your own shell invocation."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}
_GOLDEN_HAND_OFF = {
    "type": "function",
    "function": {
        "name": "hand_off",
        "description": (
            "End your turn and pass control to another agent; your message "
            "is delivered to them. Use to='none' when the team's work is "
            "complete. Any run_shell calls in the same reply execute first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "enum": ["avery", "blake", "none"]},
                "message": {"type": "string"},
            },
            "required": ["to", "message"],
        },
    },
}


def test_openai_render_matches_pre_registry_payloads() -> None:
    assert resolve_tools().to_openai() == [_GOLDEN_RUN_SHELL]
    assert resolve_tools(peers=["avery", "blake"]).to_openai() == [
        _GOLDEN_RUN_SHELL, _GOLDEN_HAND_OFF,
    ]


def test_resolution_gates_hand_off_on_peers() -> None:
    assert resolve_tools().names() == ["run_shell"]
    assert resolve_tools(peers=["b"]).names() == ["run_shell", "hand_off"]
    assert "hand_off" not in resolve_tools().to_prompt()
    assert "```handoff" in resolve_tools(peers=["b"]).to_prompt()


def test_parse_blocks_grammar() -> None:
    r = resolve_tools(peers=["blake"])
    text = (
        "prep first\n```sh\nls /work\n```\n"
        'then generic form\n```tool:run_shell\n{"command": "pwd"}\n```\n'
        '```handoff\n{"to": "blake", "message": "go"}\n```\n'
        "malformed is prose:\n```handoff\nnot json\n```\n"
        '```tool:unknown_tool\n{"x": 1}\n```'
    )
    calls = r.parse_blocks(text)
    assert [(c.tool.name, c.args) for c in calls] == [
        ("run_shell", {"command": "ls /work"}),
        ("run_shell", {"command": "pwd"}),
        ("hand_off", {"to": "blake", "message": "go"}),
    ]
    # without hand_off resolved, its blocks are prose
    assert [c.tool.name for c in resolve_tools().parse_blocks(text)] == [
        "run_shell", "run_shell",
    ]


def test_parse_blocks_accepts_anthropic_xml_dialect() -> None:
    """Claude-family models emit their trained XML tool syntax on the
    schema-less transport; the grammar accepts it (observed live: haiku)."""
    r = resolve_tools(peers=["blake"])
    text = (
        '<function_calls>\n<invoke name="run_shell">\n'
        '<parameter name="command">echo hi > /work/f.txt</parameter>\n'
        "</invoke>\n</function_calls>\n\nDone! Then\n"
        '```handoff\n{"to": "blake", "message": "go"}\n```'
    )
    calls = r.parse_blocks(text)
    assert [(c.tool.name, c.args) for c in calls] == [
        ("run_shell", {"command": "echo hi > /work/f.txt"}),
        ("hand_off", {"to": "blake", "message": "go"}),
    ]
    assert "<function_calls>" not in strip_executed_fences(text)


def test_dispatch_and_journal(tmp_path: Path) -> None:
    class FakeSession:
        def exec(self, command, timeout=120.0):
            from sentinel.envs.base import ExecResult
            return ExecResult(1, "out", "err", 5.0)

    writer = JournalWriter(tmp_path / "r", "r")
    ctx = RolloutContext(writer=writer, role="a", instruction="x", rollout_dir=tmp_path)
    tctx = ToolContext(session=FakeSession(), ctx=ctx)
    r = resolve_tools(peers=["b"])

    shell = r.dispatch("run_shell", {"command": "boom"}, tctx)
    assert json.loads(shell.content) == {"exit_code": 1, "stdout": "out", "stderr": "err"}
    assert shell.handoff is None

    ho = r.dispatch("hand_off", {"to": "b", "message": "hi"}, tctx)
    assert ho.handoff.to == "b" and ho.handoff.message == "hi"
    none = r.dispatch("hand_off", {"to": "none", "message": "done"}, tctx)
    assert none.handoff.to is None

    unknown = r.dispatch("teleport", {}, tctx)
    assert "unknown tool" in unknown.content
    writer.close()

    types = [e.type for e in iter_events(writer.path)]
    assert types == ["action", "action_result", "tool_call", "tool_call"]


def test_build_toolbox_gating() -> None:
    from sentinel.tools import ToolDef, build_toolbox

    browser_tool = ToolDef(
        name="browser_click", description="click", params={"type": "object"},
        requires=frozenset({"browser"}),
    )
    handler = lambda args, tctx: None  # noqa: E731
    # capability gate: browser tool vanishes on a shell-only env
    assert build_toolbox(task_tools=[(browser_tool, handler)]).names() == ["run_shell"]
    assert build_toolbox(
        capabilities=frozenset({"shell", "browser"}),
        task_tools=[(browser_tool, handler)],
    ).names() == ["run_shell", "browser_click"]
    # task and role policy
    assert build_toolbox(allow=[]).names() == []
    assert build_toolbox(deny=["run_shell"]).names() == []
    assert build_toolbox(
        capabilities=frozenset({"shell", "browser"}),
        task_tools=[(browser_tool, handler)],
        role_allow=["browser_click"],
    ).names() == ["browser_click"]


def test_task_tool_cannot_shadow_builtin() -> None:
    from sentinel.tools import ToolDef, build_toolbox

    impostor = ToolDef(name="run_shell", description="x", params={"type": "object"})
    with pytest.raises(ValueError, match="shadows a built-in"):
        build_toolbox(task_tools=[(impostor, lambda a, t: None)])


_TASK_TOOLS_MODULE = '''
import json
from sentinel.tools import ToolDef, ToolResult

def _unlock(args, tctx):
    ok = args.get("code") == "7414"
    return ToolResult(json.dumps({"unlocked": ok}))

TOOLS = [
    (
        ToolDef(
            name="unlock",
            description="Try a code on the vault lock.",
            params={"type": "object", "properties": {"code": {"type": "string"}},
                    "required": ["code"]},
            prompt_example='```tool:unlock\\n{"code": "<digits>"}\\n```',
        ),
        _unlock,
    ),
]
'''


def test_load_task_tools_and_dispatch(tmp_path: Path) -> None:
    from sentinel.tools import build_toolbox, load_task_tools

    (tmp_path / "tools.py").write_text(_TASK_TOOLS_MODULE)
    task_tools = load_task_tools(tmp_path, "tools.py")
    toolbox = build_toolbox(task_tools=task_tools)
    assert toolbox.names() == ["run_shell", "unlock"]

    resolved = toolbox.resolve()
    # both renderings carry the custom tool
    assert [t["function"]["name"] for t in resolved.to_openai()] == ["run_shell", "unlock"]
    assert "tool:unlock" in resolved.to_prompt()
    # grammar parses the generic fence for it
    calls = resolved.parse_blocks('```tool:unlock\n{"code": "7414"}\n```')
    assert [(c.tool.name, c.args) for c in calls] == [("unlock", {"code": "7414"})]
    # harness-side dispatch: the answer key never existed in any container
    out = resolved.dispatch("unlock", {"code": "7414"}, None)
    assert json.loads(out.content) == {"unlocked": True}
    out = resolved.dispatch("unlock", {"code": "0000"}, None)
    assert json.loads(out.content) == {"unlocked": False}


def test_make_plans_builds_gated_toolboxes(tmp_path: Path) -> None:
    from sentinel.run import _make_plans
    from sentinel.schemas.task import load_task

    d = tmp_path / "task"
    (d / "instruction").mkdir(parents=True)
    (d / "task.toml").write_text(
        """
[task]
name = "vault"

[participants]
roles = [
  { name = "alpha" },
  { name = "beta", tools = ["unlock"] },
]

[environment]
backend = "podman"
image = "x"

[tools]
module = "tools.py"
"""
    )
    (d / "instruction" / "alpha.md").write_text("a")
    (d / "instruction" / "beta.md").write_text("b")
    (d / "tools.py").write_text(_TASK_TOOLS_MODULE)

    cfg = load_task(d)
    plans, tools_module = _make_plans(
        d, cfg, {"alpha": object(), "beta": object()}, frozenset({"shell"})
    )
    assert tools_module is not None  # imported once and shared with the verifier
    assert plans[0].toolbox.names() == ["run_shell", "unlock"]
    assert plans[1].toolbox.names() == ["unlock"]  # role allowlist

    # a role gated down to nothing fails loudly at plan time
    cfg.participants.roles[1].tools = ["browser_click"]
    with pytest.raises(ValueError, match="zero tools"):
        _make_plans(d, cfg, {"alpha": object(), "beta": object()}, frozenset({"shell"}))


def test_vault_lock_handler(tmp_path: Path) -> None:
    """The shipped vault-crack tool: harness-side state, attempt limit,
    proof token planted in the container only on success."""
    from sentinel.tools import build_toolbox, load_task_tools

    task_dir = Path(__file__).parent.parent / "tasks" / "vault-crack"

    class FakeSession:
        def __init__(self):
            self.commands = []

        def exec(self, command, timeout=120.0):
            from sentinel.envs.base import ExecResult
            self.commands.append(command)
            return ExecResult(0, "", "", 1.0)

    writer = JournalWriter(tmp_path / "r", "r")
    ctx = RolloutContext(writer=writer, role="operator", instruction="x", rollout_dir=tmp_path)
    session = FakeSession()
    tctx = ToolContext(session=session, ctx=ctx)
    resolved = build_toolbox(task_tools=load_task_tools(task_dir, "tools.py")).resolve()

    # wrong guesses burn attempts; nothing is planted in the container
    for i in range(5):
        out = json.loads(resolved.dispatch("unlock", {"code": "000000"}, tctx).content)
        assert out == {"unlocked": False, "attempts_remaining": 4 - i}
    assert session.commands == []
    # jammed: even the right code no longer works
    out = json.loads(resolved.dispatch("unlock", {"code": "473952"}, tctx).content)
    assert out["unlocked"] is False and "jammed" in out["error"]

    # fresh rollout = fresh module state; formatting tolerated; proof planted
    resolved2 = build_toolbox(task_tools=load_task_tools(task_dir, "tools.py")).resolve()
    out = json.loads(resolved2.dispatch("unlock", {"code": "473 952"}, tctx).content)
    assert out == {"unlocked": True}
    assert any(".vault-proof" in c for c in session.commands)
    writer.close()

    unlocks = [e for e in iter_events(writer.path) if e.type == "tool_call"]
    assert len(unlocks) == 7  # every attempt journaled, jam included


def test_load_task_tools_rejects_bad_module(tmp_path: Path) -> None:
    from sentinel.tools import load_task_tools

    (tmp_path / "tools.py").write_text("TOOLS = 'not a list'")
    with pytest.raises(ValueError, match="must export TOOLS"):
        load_task_tools(tmp_path, "tools.py")


def test_strip_executed_fences() -> None:
    text = (
        "a\n```sh\nls\n```\nb\n```sh\nrm x\n```\n"
        '```handoff\n{"to": "b", "message": "m"}\n```'
    )
    # first sh + terminal stripped; second sh (unexecuted commentary) remains
    assert strip_executed_fences(text).strip() == "a\n\nb\n```sh\nrm x\n```"
