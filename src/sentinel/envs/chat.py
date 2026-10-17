"""Chat environment: no world at all (PLAN §6 — the tier below in-process).

Some tasks have no filesystem and no container: the participant is asked a
question, or acts on a world that lives entirely in the task's harness-side
tool module. Provisioning a container for those buys nothing and costs
seconds per rollout — which is the difference between a pass^8 sweep across
quants being routine and being an evening.

It provides no capabilities, so capability gating (ToolDef.requires) removes
shell-dependent tools automatically: a chat-backend role sees exactly the
task's own tools, or none at all.

Grading: nothing can be inspected inside a container that does not exist, so
these tasks use verifier mode "module" (harness-side state) or "none"
(trajectory checks and/or a judge). The task schema enforces that.
"""

from __future__ import annotations

from typing import Any

from sentinel.envs.base import ExecResult
from sentinel.schemas.task import EnvironmentConfig

_NO_SHELL = "the chat backend has no shell; this task's roles act through tools only"


class ChatSession:
    """A session with nothing behind it. Handlers that need a shell are gated
    out before dispatch, so reaching exec() is a harness bug, not a model
    mistake — it raises rather than returning a fake nonzero exit."""

    def exec(self, command: str, timeout: float = 120.0) -> ExecResult:
        raise RuntimeError(_NO_SHELL)


class ChatEnv:
    backend = "chat"
    capabilities: frozenset[str] = frozenset()

    def __init__(self, task_dir: object, cfg: EnvironmentConfig, name: str):
        self.cfg = cfg
        self.name = name
        self.image = None

    def provision(self) -> dict[str, Any]:
        return {"backend": "chat"}

    def open_session(self) -> ChatSession:
        return ChatSession()

    def copy_in(self, src: str, dest: str) -> None:
        raise RuntimeError(_NO_SHELL)

    def copy_out(self, src: str, dest: str) -> None:
        raise RuntimeError(_NO_SHELL)

    def teardown(self) -> None:
        return None
