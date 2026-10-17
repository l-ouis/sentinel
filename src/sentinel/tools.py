"""Tool registry: single source of truth for the tools agents can use.

A tool is defined once — name, description, JSON Schema, handler — and
rendered per transport: an OpenAI function-calling entry for HTTP chat
models, or a fenced-block grammar section for the claude CLI (which accepts
no tool schemas). Availability is resolved per rollout, so nothing is
described to a model that isn't callable and nothing callable is
undescribed.

Resolution pipeline: built-ins + the task's custom tool module, gated by
environment capabilities (ToolDef.requires), narrowed by task allow/deny and
per-role allowlists into a Toolbox, then finalized per session with
orchestration injections (hand_off when the orchestrator provides peers).
Task-specific tools run harness-side: their logic, state, and secrets never
exist in the agent-visible world — which is what makes schema-enforced locks
and oracles possible (an in-container script could simply be read).

Handlers own their journaling: world-touching tools go through the
write-ahead action/action_result protocol (journal_exec, §4.2);
scaffold-internal tools emit tool_call events. A `terminal` tool ends the
participant's turn (hand_off).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from sentinel.envs.base import Session
from sentinel.participants.base import Handoff, RolloutContext, journal_exec
from sentinel.schemas.journal import ToolCall


@dataclass
class ToolContext:
    """Everything a handler may need at dispatch time."""

    session: Session | None
    ctx: RolloutContext
    exec_timeout: float = 900.0
    output_cap: int = 8000


@dataclass
class ToolResult:
    content: str                    # what the model sees next turn
    handoff: Handoff | None = None  # terminal tools return control info


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    params: dict                        # JSON Schema, object-typed
    requires: frozenset = frozenset()   # env capabilities (gating lands in phase 2)
    terminal: bool = False              # ends the turn
    fence: str | None = None            # CLI sugar tag ("sh", "handoff"); generic is tool:<name>
    fence_raw_param: str | None = None  # sugar fence body is raw text for this param
    prompt_example: str = ""            # invocation snippet shown to CLI models


Handler = Callable[[dict, ToolContext], ToolResult]


@dataclass
class ParsedCall:
    tool: ToolDef
    args: dict


# One grammar for the CLI transport: ```tool:<name> with JSON args, plus
# sugar tags for ergonomics (models write ```sh natively; fighting that
# costs measured performance).
_FENCE_RE = re.compile(
    r"```(sh|bash|shell|handoff|tool:[A-Za-z0-9_-]+)\s*\n(.*?)```", re.S
)
_SUGAR = {"sh": "run_shell", "bash": "run_shell", "shell": "run_shell", "handoff": "hand_off"}

# Claude-family models are trained on Anthropic's XML tool syntax and emit it
# spontaneously on the schema-less CLI transport, where it would otherwise be
# inert text. It is well-formed and unambiguous, so the grammar accepts it as
# a dialect — rejecting it would measure format luck, not capability.
_XML_INVOKE_RE = re.compile(r"<invoke name=\"([A-Za-z0-9_-]+)\">(.*?)</invoke>", re.S)
_XML_PARAM_RE = re.compile(r"<parameter name=\"([A-Za-z0-9_-]+)\">(.*?)</parameter>", re.S)


@dataclass
class ResolvedTools:
    """The tool surface one role actually has, ready to render and dispatch."""

    tools: list[tuple[ToolDef, Handler]]

    def get(self, name: str) -> ToolDef | None:
        return next((d for d, _ in self.tools if d.name == name), None)

    def names(self) -> list[str]:
        return [d.name for d, _ in self.tools]

    def to_openai(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": d.name,
                    "description": d.description,
                    "parameters": d.params,
                },
            }
            for d, _ in self.tools
        ]

    def to_prompt(self) -> str:
        """The generated "## Tools" section for schema-less transports."""
        lines = [
            "\n\n## Tools",
            "Invoke a tool by writing a fenced block in your reply, exactly in "
            "the form shown for each tool below. Only the first non-terminal "
            "tool block per reply runs; I execute it and send you the result. "
            "A terminal tool (it ends your turn) may accompany it — any other "
            "tool in the same reply runs first.",
        ]
        for d, _ in self.tools:
            terminal = " (terminal: ends your turn)" if d.terminal else ""
            lines.append(f"\n### {d.name}{terminal}\n{d.description}\nInvocation:\n{d.prompt_example}")
        return "\n".join(lines)

    def parse_blocks(self, text: str) -> list[ParsedCall]:
        """Tool invocations in a CLI reply, in order: fenced blocks (the
        canonical form) and the Anthropic XML dialect. Unresolved tool names
        and malformed bodies are skipped — they are prose, not calls."""
        found: list[tuple[int, ParsedCall]] = []
        for m in _FENCE_RE.finditer(text):
            tag, body = m.group(1), m.group(2)
            name = _SUGAR.get(tag) or tag[len("tool:"):]
            tool = self.get(name)
            if tool is None:
                continue
            if tool.fence_raw_param and tag in ("sh", "bash", "shell"):
                args: dict = {tool.fence_raw_param: body.strip()}
            else:
                try:
                    args = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if not isinstance(args, dict):
                    continue
            found.append((m.start(), ParsedCall(tool, args)))
        for m in _XML_INVOKE_RE.finditer(text):
            tool = self.get(m.group(1))
            if tool is None:
                continue
            args = {p.group(1): p.group(2).strip() for p in _XML_PARAM_RE.finditer(m.group(2))}
            found.append((m.start(), ParsedCall(tool, args)))
        return [call for _, call in sorted(found, key=lambda x: x[0])]

    def dispatch(self, name: str, args: dict, tctx: ToolContext) -> ToolResult:
        for d, handler in self.tools:
            if d.name == name:
                return handler(args, tctx)
        return ToolResult(json.dumps({"error": f"unknown tool {name!r}"}))


# --- built-in tools ----------------------------------------------------------


_RUN_SHELL = ToolDef(
    name="run_shell",
    description=(
        "Run a shell command on the machine (bash -c). Do not wrap the "
        "command in your own shell invocation."
    ),
    params={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
    requires=frozenset({"shell"}),
    fence="sh",
    fence_raw_param="command",
    prompt_example="```sh\nls -la /work\n```",
)


def _run_shell_handler(args: dict, tctx: ToolContext) -> ToolResult:
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return ToolResult(json.dumps({"error": "run_shell needs a command string"}))
    result = journal_exec(
        tctx.session, tctx.ctx, command,
        timeout=tctx.exec_timeout, output_cap=tctx.output_cap,
    )
    return ToolResult(
        json.dumps(
            {
                "exit_code": result.exit_code,
                "stdout": result.stdout[-tctx.output_cap:],
                "stderr": result.stderr[-tctx.output_cap:],
            }
        )
    )


def _hand_off_def(peers: list[str]) -> ToolDef:
    return ToolDef(
        name="hand_off",
        description=(
            "End your turn and pass control to another agent; your message "
            "is delivered to them. Use to='none' when the team's work is "
            "complete. Any run_shell calls in the same reply execute first."
        ),
        params={
            "type": "object",
            "properties": {
                "to": {"type": "string", "enum": [*peers, "none"]},
                "message": {"type": "string"},
            },
            "required": ["to", "message"],
        },
        terminal=True,
        fence="handoff",
        prompt_example=(
            '```handoff\n{"to": "<agent name or none>", '
            '"message": "<delivered to that agent>"}\n```'
        ),
    )


def _hand_off_handler(args: dict, tctx: ToolContext) -> ToolResult:
    to = str(args.get("to") or "none")
    message = str(args.get("message") or "")
    tctx.ctx.writer.emit(
        ToolCall, stream=tctx.ctx.role, name="hand_off",
        args={"to": to, "message": message},
    )
    return ToolResult(
        json.dumps({"delivered": True}),
        handoff=Handoff(to=None if to.lower() == "none" else to, message=message),
    )


_BUILTINS: list[tuple[ToolDef, Handler]] = [(_RUN_SHELL, _run_shell_handler)]


@dataclass
class Toolbox:
    """One role's gated tool surface, before orchestration injections.
    Built once per rollout; resolve() finalizes it per session."""

    entries: list[tuple[ToolDef, Handler]]
    # Task-level [tools] deny list, kept so it can also veto the injected
    # hand_off (see resolve).
    deny: frozenset[str] = frozenset()

    def names(self) -> list[str]:
        return [d.name for d, _ in self.entries]

    def resolve(self, *, peers: list[str] | None = None) -> ResolvedTools:
        tools = list(self.entries)
        # hand_off is injected by orchestration, not chosen by the task, so
        # per-role allowlists must NOT strip it — a role given only its own
        # tools still has to be able to end its turn. An explicit task-level
        # deny does remove it, which is how a task ships its own terminal
        # messaging tool ("send_message") without the model being offered two
        # ways to talk and having to guess which one is real.
        if peers and "hand_off" not in self.deny:
            tools.append((_hand_off_def(peers), _hand_off_handler))
        return ResolvedTools(tools)


def build_toolbox(
    *,
    capabilities: frozenset[str] = frozenset({"shell"}),
    task_tools: list[tuple[ToolDef, Handler]] = (),
    allow: list[str] | None = None,
    deny: list[str] = (),
    role_allow: list[str] | None = None,
) -> Toolbox:
    """Gate the tool surface: (built-ins + task tools) filtered by what the
    environment can actually do, then by task and role policy."""
    for d, _ in task_tools:
        if any(d.name == b.name for b, _ in _BUILTINS):
            raise ValueError(f"task tool {d.name!r} shadows a built-in tool")
    entries = [
        (d, h)
        for d, h in [*_BUILTINS, *task_tools]
        if d.requires <= capabilities
    ]
    if allow is not None:
        entries = [(d, h) for d, h in entries if d.name in allow]
    if deny:
        entries = [(d, h) for d, h in entries if d.name not in deny]
    if role_allow is not None:
        entries = [(d, h) for d, h in entries if d.name in role_allow]
    return Toolbox(entries, deny=frozenset(deny))


def load_task_tool_module(task_dir, module_rel: str):
    """Import a task's custom tool module (harness-side code, task-authored).

    Import it exactly ONCE per rollout and share the module object: task tools
    keep their world in module state (the vault's attempt counter, a simulated
    device's power state), so importing twice would hand the participants and
    the verifier two different worlds."""
    import importlib.util
    from pathlib import Path

    path = Path(task_dir) / module_rel
    spec = importlib.util.spec_from_file_location(
        f"sentinel_task_tools_{Path(task_dir).name.replace('-', '_')}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_tools_of(module, path) -> list[tuple[ToolDef, Handler]]:
    """The TOOLS export of an already-imported task tool module."""
    tools = getattr(module, "TOOLS", None)
    if not isinstance(tools, list) or not all(
        isinstance(t, tuple) and len(t) == 2 and isinstance(t[0], ToolDef) and callable(t[1])
        for t in tools
    ):
        raise ValueError(
            f"{path}: must export TOOLS = [(ToolDef, handler), ...]"
        )
    return tools


def load_task_tools(task_dir, module_rel: str) -> list[tuple[ToolDef, Handler]]:
    """Convenience for callers that need only the tools (tests, one-offs)."""
    from pathlib import Path

    module = load_task_tool_module(task_dir, module_rel)
    return task_tools_of(module, Path(task_dir) / module_rel)


def resolve_tools(*, peers: list[str] | None = None) -> ResolvedTools:
    """Default surface (shell-capable env, no policies) — the fallback for
    participants invoked outside a full rollout, and for tests."""
    return build_toolbox().resolve(peers=peers)


# --- display helpers (transcript viewer) --------------------------------------

_EXEC_FENCE_RE = re.compile(
    r"```(?:sh|bash|shell|tool:(?!hand_off\b)[A-Za-z0-9_-]+)\s*\n.*?```"
    r"|<function_calls>.*?</function_calls>",
    re.S,
)
_TERMINAL_FENCE_RE = re.compile(r"```(?:handoff|tool:hand_off)\s*\n.*?```", re.S)


def strip_executed_fences(text: str) -> str:
    """Remove the first executed (non-terminal) block and the first terminal
    block from a CLI reply — they render as structured transcript entries;
    remaining blocks are unexecuted commentary and stay visible."""
    text = _EXEC_FENCE_RE.sub("", text, count=1)
    return _TERMINAL_FENCE_RE.sub("", text, count=1)
