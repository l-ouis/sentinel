"""ClaudePrintParticipant: frontier models through the Claude Code CLI in
print mode (`claude -p`), riding whatever login the CLI already has
(subscription) — sentinel never sees or handles an API key.

The CLI is used strictly as a chat transport: `--tools ""` disables every
built-in tool and `--system-prompt` replaces Claude Code's own. Print mode
takes text, not tool schemas, so tools are expressed through the registry's
fenced-block grammar (sentinel.tools): the "## Tools" section is generated
from the same definitions HTTP models get as schemas, and replies are parsed
by the same shared grammar. Conversation state lives in the CLI session,
resumed by id each turn, so a turn sends only the new message.

NOTE: --bare would strip hooks/plugins for a cleaner measurement, but it also
strips the settings that carry login on some setups (observed 2026-08-12:
"Not logged in" with --bare, fine without) — so we run without it and accept
the local Claude Code config as part of the measured scaffold.

Journal mapping per turn, same shape as chat_model:
  model_call    one `claude -p` invocation: usage, API latency, reported cost
  action        first sh block in the reply, journaled BEFORE exec (§4.2)
  action_result exit code + capped output, fed back as the next message
A reply with no sh block ends the loop; that text is the final answer.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

from sentinel.envs.base import Session
from sentinel.participants.base import (
    RolloutContext,
    TurnResult,
    compose_system_prompt,
)
from sentinel.schemas.journal import ModelCall, ParticipantSpec
from sentinel.tools import ToolContext, resolve_tools

SCAFFOLD_ID = "claude-print-v4"  # v4: toolless (chat-backend) prompt variant
_OUTPUT_CAP = 8000  # chars of command output fed back per turn
_MAX_NUDGES = 2

_SYSTEM_PROMPT = (
    "You are a helpful assistant working on a Linux machine through the "
    "tools listed below. Work step by step: invoke a tool, read the result "
    "I send back, then continue. Anything outside tool blocks is commentary. "
    "When the work is done, reply with a short summary and no tool block."
)
_SYSTEM_PROMPT_NO_TOOLS = (
    "You are a helpful assistant. Answer the user directly and concretely. "
    "You have no tools and no machine to act on: say what you would do, or "
    "ask for what you need, rather than pretending to have done anything."
)
_NUDGE = (
    "You replied with no visible content and no tool block. Use the tools "
    "to do the work; when finished, reply with a short summary."
)


class ClaudePrintParticipant:
    def __init__(
        self,
        *,
        model: str,
        max_turns: int = 20,
        request_timeout: float = 600.0,
        exec_timeout: float = 900.0,
        claude_bin: str = "claude",
    ):
        self.model = model
        self.max_turns = max_turns
        self.request_timeout = request_timeout
        self.exec_timeout = exec_timeout
        self.claude_bin = claude_bin
        self._cli_version: str | None = None

    def spec(self) -> ParticipantSpec:
        return ParticipantSpec(
            kind="chat_model", model=self.model, backend="claude-code-cli",
            scaffold=SCAFFOLD_ID, extra={"cli_version": self._version()},
        )

    def _version(self) -> str:
        if self._cli_version is None:
            try:
                out = subprocess.run(
                    [self.claude_bin, "--version"],
                    capture_output=True, text=True, timeout=30,
                )
                self._cli_version = out.stdout.strip() or "unknown"
            except OSError:
                self._cli_version = "unknown"
        return self._cli_version

    def run(self, session: Session, ctx: RolloutContext) -> TurnResult:
        resolved = ctx.tools if ctx.tools is not None else resolve_tools(peers=ctx.peers)
        # the generated tool grammar is transport machinery, not framing: a
        # task that replaced it would leave this model unable to call anything
        transport = resolved.to_prompt() if resolved.tools else ""
        base = _SYSTEM_PROMPT if resolved.tools else _SYSTEM_PROMPT_NO_TOOLS
        system = compose_system_prompt(
            base, ctx.briefing, override=ctx.system, transport=transport
        )
        tctx = ToolContext(
            session=session, ctx=ctx,
            exec_timeout=self.exec_timeout, output_cap=_OUTPUT_CAP,
        )
        prompt = ctx.instruction
        # persistent context: the conversation token is the CLI session id —
        # resuming it replays the role's whole history across turns
        session_id: str | None = ctx.conversation  # type: ignore[assignment]
        nudges = 0
        for _turn in range(self.max_turns):
            body = self._call(prompt, ctx, resume=session_id, system=system)
            session_id = body.get("session_id") or session_id
            text = body.get("result") or ""
            calls = resolved.parse_blocks(text)
            exec_calls = [c for c in calls if not c.tool.terminal]
            terminal_calls = [c for c in calls if c.tool.terminal]
            feedback: str | None = None
            if exec_calls:
                result = resolved.dispatch(exec_calls[0].tool.name, exec_calls[0].args, tctx)
                feedback = result.content
                if len(exec_calls) > 1:
                    noted = json.loads(feedback)
                    noted["note"] = "only the first tool block was executed"
                    feedback = json.dumps(noted)
            if terminal_calls:
                # any non-terminal call in the same reply ran first
                result = resolved.dispatch(
                    terminal_calls[0].tool.name, terminal_calls[0].args, tctx
                )
                return TurnResult(
                    final=text, handoff=result.handoff, conversation=session_id
                )
            if not exec_calls:
                if not text.strip() and nudges < _MAX_NUDGES:
                    nudges += 1
                    prompt = _NUDGE
                    continue
                return TurnResult(final=text or None, conversation=session_id)
            prompt = feedback
        return TurnResult(conversation=session_id)

    def _call(
        self, prompt: str, ctx: RolloutContext, resume: str | None, system: str = _SYSTEM_PROMPT
    ) -> dict:
        argv = [
            self.claude_bin, "-p", "--tools", "", "--output-format", "json",
            "--system-prompt", system, "--model", self.model,
        ]
        if resume:
            argv += ["--resume", resume]
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)  # subscription login, never API billing
        start = time.monotonic()
        proc = subprocess.run(
            argv, input=prompt, text=True, capture_output=True,
            timeout=self.request_timeout, cwd=ctx.rollout_dir, env=env,
        )
        wall_ms = (time.monotonic() - start) * 1000
        try:
            body = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"claude -p returned non-JSON (exit {proc.returncode}): "
                f"{(proc.stdout or proc.stderr)[:300]}"
            ) from None
        seq = ctx.writer._seq  # sidecar naming only
        req_ref = ctx.writer.write_sidecar(
            f"model_call_{seq}_request.json",
            json.dumps({"argv": argv, "prompt": prompt}),
        )
        resp_ref = ctx.writer.write_sidecar(f"model_call_{seq}_response.json", proc.stdout)
        if body.get("is_error"):
            raise RuntimeError(f"claude -p error: {str(body.get('result'))[:300]}")
        usage = body.get("usage") or {}
        prompt_tokens = sum(
            usage.get(k) or 0
            for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        )
        ctx.writer.emit(
            ModelCall, stream=ctx.role, participant=self.spec(),
            prompt_tokens=prompt_tokens or None,
            completion_tokens=usage.get("output_tokens"),
            latency_ms=body.get("duration_api_ms") or wall_ms,
            cost_usd=body.get("total_cost_usd"),  # informational under subscription
            request_ref=req_ref, response_ref=resp_ref,
        )
        return body
