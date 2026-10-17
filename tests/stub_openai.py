"""A scriptable stand-in for LM Studio: a minimal OpenAI-compatible
/chat/completions server. Behavior: while the conversation contains no tool
result yet, answer with the configured tool call(s); afterwards, answer with a
final message. Lets the entire spine run blind (no real model)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class StubOpenAI:
    def __init__(
        self,
        commands: list[str],
        final: str = "done!",
        handoff: tuple[str, str] | None = None,  # (to, message): hand_off tool call
        terminal: tuple[str, dict] | None = None,  # (tool name, args): any terminal tool
    ):
        self.commands = commands
        self.final = final
        # a task that ships its own terminal messaging tool denies hand_off, so
        # the stub has to be able to end a turn through that tool instead
        self.terminal = terminal or (
            ("hand_off", {"to": handoff[0], "message": handoff[1]}) if handoff else None
        )
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append(body)
                n_tool_msgs = sum(1 for m in body["messages"] if m["role"] == "tool")
                if n_tool_msgs < len(stub.commands):
                    spec = stub.commands[n_tool_msgs]
                    # a str is run_shell sugar; a dict is any tool: {"name", "args"}
                    if isinstance(spec, str):
                        name, args = "run_shell", {"command": spec}
                    else:
                        name, args = spec["name"], spec["args"]
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{n_tool_msgs}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args),
                                },
                            }
                        ],
                    }
                else:
                    message = {"role": "assistant", "content": stub.final}
                    if stub.terminal:
                        name, args = stub.terminal
                        message["tool_calls"] = [
                            {
                                "id": "call_terminal",
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)},
                            }
                        ]
                resp = json.dumps(
                    {
                        "id": "stub",
                        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "StubOpenAI":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
