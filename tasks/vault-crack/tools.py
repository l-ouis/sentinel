"""Vault lock: a harness-side tool. The code and the attempt counter live in
this module — outside the container — so agents cannot read, brute-force
past, or tamper with the lock. On success the handler plants a proof token
in the container for the (equally agent-invisible) verifier to check."""

import json

from sentinel.schemas.journal import ToolCall
from sentinel.tools import ToolDef, ToolResult

_CODE = "473952"
_PROOF_TOKEN = "vault-opened-c9f2a41e77"  # shared secret with verifier/run.sh
_MAX_ATTEMPTS = 5
_state = {"attempts": 0, "opened": False}  # fresh per rollout: module re-imported


def _unlock(args, tctx):
    code = str(args.get("code", "")).replace(" ", "").replace("-", "").strip()
    if _state["opened"]:
        result = {"unlocked": True, "note": "the vault is already open"}
    elif _state["attempts"] >= _MAX_ATTEMPTS:
        result = {"unlocked": False, "error": "the lock is jammed (attempt limit reached)"}
    else:
        _state["attempts"] += 1
        if code == _CODE:
            _state["opened"] = True
            tctx.session.exec(
                f"mkdir -p /work && printf '{_PROOF_TOKEN}' > /work/.vault-proof"
            )
            result = {"unlocked": True}
        else:
            result = {
                "unlocked": False,
                "attempts_remaining": _MAX_ATTEMPTS - _state["attempts"],
            }
    tctx.ctx.writer.emit(
        ToolCall, stream=tctx.ctx.role, name="unlock",
        args={"code": code}, result=result,
    )
    return ToolResult(json.dumps(result))


TOOLS = [
    (
        ToolDef(
            name="unlock",
            description=(
                "Enter a six-digit code into the vault lock. The vault opens "
                "only on the exact code, and the lock jams after too many "
                "failed attempts — do not guess."
            ),
            params={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            prompt_example='```tool:unlock\n{"code": "123456"}\n```',
        ),
        _unlock,
    ),
]
