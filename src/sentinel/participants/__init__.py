"""Participant registry.

Imports are lazy on purpose: `sentinel.tools` imports `participants.base`, and
every concrete participant imports `sentinel.tools` back. Pulling the concrete
classes in at package-import time made that a genuine cycle — whichever of the
two was imported first won, and `import sentinel.tools` first lost. Module-level
__getattr__ (PEP 562) keeps the public names while breaking the cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sentinel.participants.base import RolloutContext, SessionParticipant

if TYPE_CHECKING:  # for type checkers only; never at runtime
    from sentinel.participants.chat_model import ChatModelParticipant
    from sentinel.participants.claude_print import ClaudePrintParticipant
    from sentinel.participants.scripted import ScriptedParticipant

CLAUDE_PREFIX = "claude:"
SCRIPT_PREFIX = "script:"

_LAZY = {
    "ChatModelParticipant": "sentinel.participants.chat_model",
    "ClaudePrintParticipant": "sentinel.participants.claude_print",
    "ScriptedParticipant": "sentinel.participants.scripted",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_participant(
    model: str,
    *,
    base_url: str,
    temperature: float | None = None,
    max_turns: int = 20,
    exec_timeout: float = 900.0,
    task_dir: str | Path | None = None,
) -> SessionParticipant:
    """Route a model string to its transport: "claude:<model>" goes through
    the Claude Code CLI (subscription login), "script:<path>" is a scripted
    fixture (path relative to the task dir); anything else is a model name at
    the OpenAI-compatible endpoint."""
    from sentinel.participants.chat_model import ChatModelParticipant
    from sentinel.participants.claude_print import ClaudePrintParticipant
    from sentinel.participants.scripted import ScriptedParticipant

    if model.startswith(SCRIPT_PREFIX):
        rel = model[len(SCRIPT_PREFIX):]
        if not rel:
            raise ValueError('use "script:<path>", e.g. script:users/student.toml')
        path = Path(task_dir or ".") / rel
        if not path.is_file():
            raise FileNotFoundError(f"script {path} not found")
        return ScriptedParticipant(path, label=model)
    if model.startswith(CLAUDE_PREFIX):
        name = model[len(CLAUDE_PREFIX):]
        if not name:
            raise ValueError(
                'use "claude:<model>", e.g. claude:sonnet or claude:claude-haiku-4-5'
            )
        if temperature is not None:
            raise ValueError("claude -p exposes no sampling controls; drop --temperature")
        return ClaudePrintParticipant(
            model=name, max_turns=max_turns, exec_timeout=exec_timeout
        )
    return ChatModelParticipant(
        base_url=base_url, model=model, temperature=temperature,
        max_turns=max_turns, exec_timeout=exec_timeout,
    )


__all__ = [
    "CLAUDE_PREFIX",
    "SCRIPT_PREFIX",
    "ChatModelParticipant",
    "ClaudePrintParticipant",
    "RolloutContext",
    "ScriptedParticipant",
    "SessionParticipant",
    "make_participant",
]
