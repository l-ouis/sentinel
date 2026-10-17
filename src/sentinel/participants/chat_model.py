"""ChatModelParticipant: any OpenAI-compatible endpoint (LM Studio, llama.cpp
server, vLLM, Ollama) driving a session through tools.

Tool definitions, rendering, and dispatch live in the sentinel.tools
registry; this scaffold renders the resolved set as an OpenAI tools array,
loops until the model stops calling tools (or a terminal tool ends the
turn), and feeds results back. The scaffold id is part of the measured
ParticipantSpec (§3) — polling behavior, prompt shape, and truncation
policy all live here, versioned.

Journal mapping per turn:
  model_call   (observation)  one request/response at the endpoint, usage+latency
  action       (decision)     shell command, journaled BEFORE exec (§4.2)
  action_result(observation)  exit code + captured output after
Full request/response bodies go to sidecars, referenced from the model_call.
"""

from __future__ import annotations

import json
import time

import httpx

from sentinel.endpoint import probe_endpoint
from sentinel.envs.base import Session
from sentinel.participants.base import (
    Handoff,
    RolloutContext,
    TurnResult,
    compose_system_prompt,
)
from sentinel.schemas.journal import EnvEvent, ModelCall, ParticipantSpec
from sentinel.tools import ToolContext, resolve_tools

SCAFFOLD_ID = "tool-loop-v4"  # v4: bounded retries on engine-side failures
_MAX_NUDGES = 2
_NUDGE = (
    "You replied with no visible content and no tool call. Use your tools to "
    "do the work; when finished, reply with a short summary of what you did."
)
_OUTPUT_CAP = 8000  # chars of tool output fed back per command

_NOT_TERMINAL = type("_NT", (), {"terminal": False})()  # unknown tools sort first


# OPEN QUESTION, deliberately not answered by guesswork: reasoning models
# return their analysis channel alongside the reply (LM Studio calls it
# `reasoning` for gpt-oss; other servers use `reasoning_content`), and this
# scaffold drops it when replaying prior turns. Echoing it back was tried
# against gpt-oss-20b and showed no benefit over 5 trials per policy
# (none 5/5, last-only 5/5, all 5/5 on a simplified conversation; on the
# captured hard case none 5/5 vs all 4/5) — well inside noise either way.
# It is a scaffold variable worth an actual experiment rather than a hunch,
# which is what SCAFFOLD_ID versioning exists for.

# LM Studio reports engine-side failures as 400s wrapping a 500 body, so status
# code alone cannot separate "your request is malformed" from "the engine fell
# over". These markers mean the server is telling us it broke, not us.
# Matched against the raw body, where LM Studio's inner JSON arrives
# backslash-escaped — so these are quote-free on purpose.
_RETRYABLE_MARKERS = ("server_error", "fetch failed", "predict request failed")
_MAX_RETRIES = 2
_RETRY_BACKOFF_S = 1.5


def _tc_args(tc: dict) -> dict:
    try:
        args = json.loads(tc["function"]["arguments"])
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}

_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a Linux machine via the "
    "run_shell tool. Use it to accomplish what the user asks. When you are "
    "done, reply with a short summary and stop calling tools."
)
# Tools, but no shell: describing a Linux machine here would be describing
# something that does not exist, which is both a tell and an invitation to
# call a tool the role was never given.
_SYSTEM_PROMPT_TOOLS = (
    "You are a helpful assistant. Use the tools available to you to do what "
    "the user asks, working from what the tools actually return rather than "
    "what you expect them to return. When you are done, reply with a short "
    "summary and stop calling tools."
)
# The chat backend gives a role no tools at all (PLAN §6): the shell prompt
# would then describe a machine that does not exist — a tell, and a prompt for
# hallucinated tool calls. The prompt shape is part of the measured scaffold,
# which is why SCAFFOLD_ID moved to v3 when this branch was added.
_SYSTEM_PROMPT_NO_TOOLS = (
    "You are a helpful assistant. Answer the user directly and concretely. "
    "You have no tools and no machine to act on: say what you would do, or "
    "ask for what you need, rather than pretending to have done anything."
)


class ChatModelParticipant:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        temperature: float | None = None,
        max_turns: int = 20,
        request_timeout: float = 600.0,
        exec_timeout: float = 900.0,  # long enough for in-rollout transcription
        probe: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_turns = max_turns
        self.request_timeout = request_timeout
        self.exec_timeout = exec_timeout
        self.probe = probe

    def spec(self) -> ParticipantSpec:
        sampling = {"temperature": self.temperature} if self.temperature is not None else None
        # quant, context limit and chat-template identity are part of the
        # measured tuple (§3/§11), not plumbing — ask the server rather than
        # trusting that a preset has not been clicked since the last run
        info = probe_endpoint(self.base_url, self.model) if self.probe else None
        return ParticipantSpec(
            kind="chat_model", model=self.model, backend=self.base_url,
            quant=info.quant if info else None,
            context_limit=info.context_limit if info else None,
            sampling=sampling, scaffold=SCAFFOLD_ID,
            extra=info.to_spec_extra() if info else {},
        )

    def run(self, session: Session, ctx: RolloutContext) -> TurnResult:
        resolved = ctx.tools if ctx.tools is not None else resolve_tools(peers=ctx.peers)
        tools = resolved.to_openai()
        tctx = ToolContext(
            session=session, ctx=ctx,
            exec_timeout=self.exec_timeout, output_cap=_OUTPUT_CAP,
        )
        if ctx.conversation is not None:
            # persistent context: continue the prior turns' conversation
            messages: list[dict] = ctx.conversation  # type: ignore[assignment]
            messages.append({"role": "user", "content": ctx.instruction})
        else:
            if "run_shell" in resolved.names():
                base = _SYSTEM_PROMPT
            elif tools:
                base = _SYSTEM_PROMPT_TOOLS
            else:
                base = _SYSTEM_PROMPT_NO_TOOLS
            system = compose_system_prompt(
                base, ctx.briefing, override=ctx.system
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": ctx.instruction},
            ]
        self._journal_endpoint(ctx)
        nudges = 0
        with httpx.Client(timeout=self.request_timeout) as client:
            for _turn in range(self.max_turns):
                msg = self._complete(client, messages, ctx, tools)
                assistant: dict = {"role": "assistant", "content": msg.get("content")}
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    assistant["tool_calls"] = tool_calls
                messages.append(assistant)
                # non-terminal calls first, terminal last, regardless of the
                # order the model emitted them — matches the advertised
                # "commands in the same reply execute first" semantics
                ordered = sorted(
                    tool_calls,
                    key=lambda tc: (resolved.get(tc["function"]["name"]) or _NOT_TERMINAL).terminal,
                )
                handoff: Handoff | None = None
                for tc in ordered:
                    result = resolved.dispatch(
                        tc["function"]["name"], _tc_args(tc), tctx
                    )
                    # every call gets a response so the transcript stays
                    # protocol-valid if this conversation is resumed later
                    messages.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": result.content}
                    )
                    if result.handoff is not None:
                        handoff = result.handoff
                if handoff is not None:
                    return TurnResult(
                        final=msg.get("content"), handoff=handoff, conversation=messages
                    )
                if not tool_calls:
                    # some models stall: all tokens spent reasoning, nothing
                    # emitted — nudge instead of accepting silence as "done"
                    if not (msg.get("content") or "").strip() and nudges < _MAX_NUDGES:
                        nudges += 1
                        messages.append({"role": "user", "content": _NUDGE})
                        continue
                    return TurnResult(final=msg.get("content"), conversation=messages)
        return TurnResult(conversation=messages)

    def _post_with_retries(
        self, client: httpx.Client, request: dict, ctx: RolloutContext
    ) -> httpx.Response:
        """One completion, retried on server-side failures.

        Safe to retry precisely because a completion is an observation, not a
        world-touching action (§4.2): a failed request dispatched nothing, so
        there is no ambiguous effect to reason about. Every attempt is
        journaled — an endpoint that needs two tries per call is a worse
        deployment than one that needs none, and that should show up in the
        data rather than being smoothed away.
        """
        last: str = ""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    json=request,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            except httpx.HTTPError as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code < 400:
                    return resp
                last = f"HTTP {resp.status_code}: {resp.text[:300]}"
                retryable = resp.status_code >= 500 or (
                    resp.status_code == 400
                    and any(m in resp.text for m in _RETRYABLE_MARKERS)
                )
                if not retryable:
                    break
            if attempt < _MAX_RETRIES:
                ctx.writer.emit(
                    EnvEvent, stream=ctx.role, category="observation",
                    name="endpoint_retry",
                    payload={"model": self.model, "attempt": attempt + 1, "error": last},
                )
                time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
        raise RuntimeError(f"endpoint failed for model {self.model!r}: {last}")

    def _journal_endpoint(self, ctx: RolloutContext) -> None:
        """Record what the server said about itself, once per role. The full
        chat template goes to a sidecar (it is long and unchanging); the spec
        carries only its hash, so every model_call stays greppable."""
        if not self.probe or getattr(ctx, "_endpoint_journaled", False):
            return
        ctx._endpoint_journaled = True  # type: ignore[attr-defined]
        info = probe_endpoint(self.base_url, self.model)
        ref = None
        if info.chat_template:
            ref = ctx.writer.write_sidecar(
                f"chat_template_{ctx.role}.jinja", info.chat_template
            )
        ctx.writer.emit(
            EnvEvent, stream=ctx.role, category="observation", name="endpoint_probe",
            payload={
                "model": self.model, "source": info.source,
                "quant": info.quant, "context_limit": info.context_limit,
                "chat_template_sha256": info.template_sha256,
                "chat_template_ref": ref, **info.details,
            },
        )

    def _complete(
        self, client: httpx.Client, messages: list[dict], ctx: RolloutContext, tools: list[dict]
    ) -> dict:
        # omit `tools` entirely when empty: some endpoints reject [] outright,
        # and others take it as "you have a tool, guess its name"
        request = {"model": self.model, "messages": messages}
        if tools:
            request["tools"] = tools
        if self.temperature is not None:
            request["temperature"] = self.temperature
        start = time.monotonic()
        resp = self._post_with_retries(client, request, ctx)
        latency_ms = (time.monotonic() - start) * 1000
        body = resp.json()
        seq = ctx.writer._seq  # sidecar naming only
        req_ref = ctx.writer.write_sidecar(f"model_call_{seq}_request.json", json.dumps(request))
        resp_ref = ctx.writer.write_sidecar(f"model_call_{seq}_response.json", json.dumps(body))
        usage = body.get("usage") or {}
        ctx.writer.emit(
            ModelCall, stream=ctx.role, participant=self.spec(),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms, request_ref=req_ref, response_ref=resp_ref,
        )
        return body["choices"][0]["message"]
