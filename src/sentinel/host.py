"""Host fingerprint: what machine a rollout ran on. Local-model metrics
(tok/s, wall time) are meaningless without this — the same model+quant is a
different measured participant on different hardware. Best-effort everywhere:
an unknown field is None, never a crash."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from functools import lru_cache


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError):
        return None


def _gpu() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return ", ".join(out.stdout.strip().splitlines())
    except OSError:
        pass
    try:  # AMD/Intel or headless-NVIDIA fallback: first VGA/3D device
        out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10)
        for line in out.stdout.splitlines():
            if "VGA" in line or "3D controller" in line:
                return line.split(":", 2)[-1].strip()
    except OSError:
        pass
    return None


@lru_cache(maxsize=1)
def host_fingerprint() -> dict:
    return {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": _cpu_model(),
        "cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": _gpu(),
    }
