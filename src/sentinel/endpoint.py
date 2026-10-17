"""What the inference server is actually doing to your prompt.

A local endpoint sits between the messages sentinel sends and the tokens the
model sees: the model's **chat template** decides how roles are wrapped, and
several templates inject a default system prompt of their own when the system
slot is empty (the Qwen family is the usual example). Servers add another
layer — LM Studio carries a per-model preset whose system prompt and sampling
defaults can apply to API requests, and which a person can change by clicking
something in a GUI three weeks before the run.

There is no settled academic convention here; the harnesses that care mostly
apply the template themselves and log it. Sentinel's position follows §3: the
template and any server-side default are **part of the measured tuple**, like
quant. So probe for them, record what comes back, and let two runs that differ
only by a silently-edited preset show up as different conditions instead of
mysterious numbers.

Two practical consequences for task authors:

1. **Always send an explicit system message.** Most templates only inject a
   default when the system slot is empty, so filling it is what stops an
   invisible third party from writing part of your prompt. Sentinel always
   sends one — either the scaffold's or the task's (RoleDef.system).
2. **Full control means bypassing the template**, i.e. rendering it yourself
   and posting to /v1/completions. That is a bigger change than it looks
   (every model's template differs) and is not what this module does; it
   records the template rather than replacing it.

Everything here is best-effort. Endpoints disagree about what they expose, and
none of these routes is guaranteed: a probe that fails records that it failed
and the rollout proceeds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import httpx

_PROBE_TIMEOUT = 3.0
_cache: dict[tuple[str, str], "EndpointInfo"] = {}


@dataclass
class EndpointInfo:
    """What we could learn. Every field is optional on purpose."""

    source: str = "unavailable"          # which route answered
    context_limit: int | None = None
    quant: str | None = None
    chat_template: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def template_sha256(self) -> str | None:
        if self.chat_template is None:
            return None
        return hashlib.sha256(self.chat_template.encode()).hexdigest()[:16]

    def to_spec_extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {"endpoint_probe": self.source}
        if self.template_sha256:
            extra["chat_template_sha256"] = self.template_sha256
        extra.update(self.details)
        return extra


def _get(client: httpx.Client, url: str) -> Any | None:
    try:
        resp = client.get(url)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def _from_props(body: Any) -> EndpointInfo | None:
    """llama.cpp server /props: the one route that reliably hands back the
    chat template it is applying."""
    if not isinstance(body, dict) or "chat_template" not in body:
        return None
    settings = body.get("default_generation_settings") or {}
    n_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
    return EndpointInfo(
        source="llama.cpp:/props",
        context_limit=n_ctx if isinstance(n_ctx, int) else None,
        chat_template=body.get("chat_template"),
        details={
            k: body[k] for k in ("model_path", "build_info", "total_slots") if k in body
        },
    )


def _from_model_list(body: Any, model: str, source: str) -> EndpointInfo | None:
    """A models listing (OpenAI-compatible, or LM Studio's richer REST one).
    Field names vary by server and version, so every key is treated as
    optional and anything unrecognised is ignored rather than guessed at."""
    if not isinstance(body, dict):
        return None
    entries = body.get("data")
    if not isinstance(entries, list):
        return None
    entry = next(
        (e for e in entries if isinstance(e, dict) and e.get("id") == model), None
    )
    if entry is None:
        return None
    context = next(
        (
            entry[k]
            for k in ("loaded_context_length", "max_context_length", "max_model_len")
            if isinstance(entry.get(k), int)
        ),
        None,
    )
    details = {
        k: entry[k]
        for k in ("arch", "publisher", "compatibility_type", "state", "type")
        if isinstance(entry.get(k), (str, int, float, bool))
    }
    # capabilities is a list ("tool_use", ...) and is worth keeping: a model
    # served without tool_use will fail a tool task for a reason that has
    # nothing to do with the model's ability
    caps = entry.get("capabilities")
    if isinstance(caps, list) and all(isinstance(c, str) for c in caps):
        details["capabilities"] = ",".join(caps)
    quant = entry.get("quantization")
    return EndpointInfo(
        source=source,
        context_limit=context,
        quant=quant if isinstance(quant, str) else None,
        details=details,
    )


def probe_endpoint(base_url: str, model: str) -> EndpointInfo:
    """Best-effort description of how `model` is being served. Cached per
    (endpoint, model): this is configuration, and it does not change under us
    mid-rollout."""
    key = (base_url.rstrip("/"), model)
    if key in _cache:
        return _cache[key]

    v1 = key[0]
    root = v1[: -len("/v1")] if v1.endswith("/v1") else v1
    info = EndpointInfo()
    with httpx.Client(timeout=_PROBE_TIMEOUT) as client:
        found = _from_props(_get(client, f"{root}/props"))
        # LM Studio's own REST API carries quant and context length, which the
        # OpenAI-compatible listing usually does not
        lmstudio = _from_model_list(
            _get(client, f"{root}/api/v0/models"), model, "lmstudio:/api/v0/models"
        )
        openai = _from_model_list(
            _get(client, f"{v1}/models"), model, "openai:/v1/models"
        )
    for candidate in (found, lmstudio, openai):
        if candidate is None:
            continue
        if info.source == "unavailable":
            info = candidate
        else:  # merge: keep the first answer, fill gaps from later ones
            info.context_limit = info.context_limit or candidate.context_limit
            info.quant = info.quant or candidate.quant
            info.chat_template = info.chat_template or candidate.chat_template
            info.details = {**candidate.details, **info.details}
            info.source = f"{info.source}+{candidate.source}"
    _cache[key] = info
    return info
