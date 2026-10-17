"""Two simulated devices, harness-side.

The world is this module's `_state` dict. Nothing about it exists inside any
container (there is no container), so it cannot be read, edited, or faked from
the agent's side — the same asymmetry vault-crack gets from its lock, applied
to something mundane. The verifier grades this same module object, because the
runner imports it exactly once per rollout.
"""

import json

from sentinel.schemas.journal import ToolCall
from sentinel.tools import ToolDef, ToolResult

_state = {  # fresh per rollout: the module is re-imported for each one
    "laptop": ["notes.md", "thesis.tex", "photo.jpg"],
    "phone": ["photo.jpg"],
}
DEVICES = tuple(_state)


def _journal(tctx, name, args, result):
    tctx.ctx.writer.emit(
        ToolCall, stream=tctx.ctx.role, name=name, args=args, result=result,
    )
    return ToolResult(json.dumps(result))


def _list_files(args, tctx):
    device = str(args.get("device", "")).lower()
    result = (
        {"device": device, "files": sorted(_state[device])}
        if device in _state
        else {"error": f"unknown device {device!r}; known devices: {list(DEVICES)}"}
    )
    return _journal(tctx, "list_files", {"device": device}, result)


def _send_file(args, tctx):
    name = str(args.get("name", "")).strip()
    src = str(args.get("from_device", "")).lower()
    dst = str(args.get("to_device", "")).lower()
    if src not in _state or dst not in _state:
        result = {"sent": False, "error": f"unknown device; known devices: {list(DEVICES)}"}
    elif name not in _state[src]:
        result = {"sent": False, "error": f"{name!r} is not on the {src}"}
    elif name in _state[dst]:
        result = {"sent": True, "note": f"{name} was already on the {dst}"}
    else:
        _state[dst].append(name)
        result = {"sent": True, "device": dst, "files": sorted(_state[dst])}
    return _journal(
        tctx, "send_file", {"name": name, "from_device": src, "to_device": dst}, result
    )


TOOLS = [
    (
        ToolDef(
            name="list_files",
            description="List the files on one of the user's devices ('laptop' or 'phone').",
            params={
                "type": "object",
                "properties": {"device": {"type": "string", "enum": list(DEVICES)}},
                "required": ["device"],
            },
            prompt_example='```tool:list_files\n{"device": "laptop"}\n```',
        ),
        _list_files,
    ),
    (
        ToolDef(
            name="send_file",
            description=(
                "Copy one file from one of the user's devices to the other. "
                "Fails if the file is not on the source device."
            ),
            params={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "from_device": {"type": "string", "enum": list(DEVICES)},
                    "to_device": {"type": "string", "enum": list(DEVICES)},
                },
                "required": ["name", "from_device", "to_device"],
            },
            prompt_example=(
                '```tool:send_file\n'
                '{"name": "notes.md", "from_device": "laptop", "to_device": "phone"}\n```'
            ),
        ),
        _send_file,
    ),
]
