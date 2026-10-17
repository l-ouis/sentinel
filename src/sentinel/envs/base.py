"""Environment/provisioner protocol (PLAN §5/§6). v1 implements the Podman
backend; kata/vm/in_process slot in behind the same interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float


@runtime_checkable
class Session(Protocol):
    def exec(self, command: str, timeout: float = 120.0) -> ExecResult: ...


@runtime_checkable
class Environment(Protocol):
    backend: str
    # What tools this world can support (gates ToolDef.requires): a shell
    # container provides {"shell"}; a playwright-equipped one adds "browser".
    capabilities: frozenset[str]

    def provision(self) -> dict[str, Any]:
        """Bring the world up; returns the re-attachable handle (§4.1)."""
        ...

    def open_session(self) -> Session: ...

    def copy_in(self, src: str, dest: str) -> None: ...

    def copy_out(self, src: str, dest: str) -> None: ...

    def teardown(self) -> None: ...
