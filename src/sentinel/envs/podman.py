"""Podman container environment (PLAN §6 default tier: daemonless, rootless,
OCI-compatible). The container outlives our process — the handle is the
container id, which is what recovery re-attaches to."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from sentinel.envs.base import ExecResult
from sentinel.schemas.task import EnvironmentConfig


def _run(args: list[str], timeout: float = 600.0) -> subprocess.CompletedProcess:
    # errors="replace": command output is world evidence, not ours to validate —
    # a stray non-utf8 byte must not crash the harness mid-intent (§4.2)
    return subprocess.run(
        args, capture_output=True, text=True, errors="replace", timeout=timeout
    )


def _check(args: list[str], timeout: float = 600.0) -> str:
    proc = _run(args, timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])}... failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


class PodmanSession:
    def __init__(self, container_id: str):
        self.container_id = container_id
        self._shell: str | None = None

    def shell(self) -> str:
        """bash when the image has it — models write bash-isms, and forcing
        them to nest shells invites quoting failures — else sh."""
        if self._shell is None:
            probe = _run(
                ["podman", "exec", self.container_id, "sh", "-c", "command -v bash"],
                timeout=30.0,
            )
            self._shell = "bash" if probe.returncode == 0 else "sh"
        return self._shell

    def exec(self, command: str, timeout: float = 120.0) -> ExecResult:
        start = time.monotonic()
        try:
            proc = _run(
                ["podman", "exec", self.container_id, self.shell(), "-lc", command],
                timeout=timeout,
            )
            exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            exit_code = 124
            stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = f"command timed out after {timeout}s"
        return ExecResult(exit_code, stdout, stderr, (time.monotonic() - start) * 1000)


class PodmanEnv:
    backend = "podman"
    capabilities = frozenset({"shell"})

    def __init__(self, task_dir: str | Path, cfg: EnvironmentConfig, name: str):
        if cfg.network_mode not in ("none", "open"):
            raise NotImplementedError(f"network_mode {cfg.network_mode!r} not in v1")
        self.task_dir = Path(task_dir)
        self.cfg = cfg
        self.name = name
        self.container_id: str | None = None
        self.image: str | None = None

    def provision(self) -> dict[str, Any]:
        self.image = self.cfg.image or self._build()
        args = ["podman", "run", "-d", "--name", self.name]
        if self.cfg.network_mode == "none":
            args += ["--network", "none"]
        args += [self.image, "sleep", "infinity"]
        self.container_id = _check(args)
        return {"container_id": self.container_id, "image": self.image}

    def _build(self) -> str:
        context = self.task_dir / "environment"
        tag = f"sentinel-task-{self.task_dir.name}"
        _check(
            ["podman", "build", "-t", tag, "-f", str(context / self.cfg.build), str(context)],
            timeout=1800.0,
        )
        return tag

    def open_session(self) -> PodmanSession:
        assert self.container_id, "provision() first"
        return PodmanSession(self.container_id)

    def copy_in(self, src: str, dest: str) -> None:
        parent = str(Path(dest).parent)
        self.open_session().exec(f"mkdir -p {parent}")
        _check(["podman", "cp", src, f"{self.container_id}:{dest}"])

    def copy_out(self, src: str, dest: str) -> None:
        _check(["podman", "cp", f"{self.container_id}:{src}", dest])

    def teardown(self) -> None:
        if self.container_id:
            _run(["podman", "rm", "-f", self.container_id], timeout=60.0)
            self.container_id = None
