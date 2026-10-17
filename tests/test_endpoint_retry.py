"""Bounded retries on engine-side failures.

Retrying a completion is safe precisely because it is an observation, not a
world-touching action (§4.2): a failed request dispatched nothing, so there is
no ambiguous effect. Retrying a *tool call* would be a different question.

The failure this exists for, observed against LM Studio: gpt-oss-20b
intermittently emits output the server's harmony grammar rejects, and LM Studio
reports it as a 400 whose body wraps an engine 500. Status code alone cannot
tell that apart from "your request is malformed", so the body is inspected —
and it arrives backslash-escaped, which an earlier version of the matcher got
wrong and silently never retried.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from sentinel.journal import JournalWriter
from sentinel.journal.reader import iter_events
from sentinel.participants import ChatModelParticipant
from sentinel.participants.base import RolloutContext

# exactly what LM Studio returns: an engine 500 wrapped in a 400, escaped
_ENGINE_500 = json.dumps(
    {"error": 'Engine protocol predict stream returned an error: '
              '{"code":500,"message":"The model produced output that does not '
              'match the expected peg-native format","type":"server_error"}'}
)
_BAD_REQUEST = json.dumps({"error": "unknown model 'nope'"})


class FlakyEndpoint:
    """Fails `failures` times with `body`, then answers normally."""

    def __init__(self, failures: int, body: str = _ENGINE_500):
        self.failures = failures
        self.body = body
        self.calls = 0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: object) -> None:
                pass

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["Content-Length"]))
                stub.calls += 1
                if stub.calls <= stub.failures:
                    payload, code = stub.body.encode(), 400
                else:
                    payload, code = json.dumps({
                        "choices": [{"index": 0, "message": {
                            "role": "assistant", "content": "recovered"},
                            "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }).encode(), 200
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "FlakyEndpoint":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()


def _ctx(tmp_path: Path) -> RolloutContext:
    return RolloutContext(
        writer=JournalWriter(tmp_path / "r1", "r1"), role="assistant",
        instruction="go", rollout_dir=tmp_path / "r1",
    )


def _retries(ctx: RolloutContext) -> list:
    ctx.writer.close()
    return [
        e for e in iter_events(ctx.rollout_dir / "journal.jsonl")
        if e.type == "env_event" and e.name == "endpoint_retry"
    ]


def test_transient_engine_failure_is_retried_and_journaled(tmp_path: Path) -> None:
    with FlakyEndpoint(failures=2) as ep:
        p = ChatModelParticipant(base_url=ep.base_url, model="m", probe=False)
        ctx = _ctx(tmp_path)
        result = p.run(None, ctx)
    assert result.final == "recovered"
    assert ep.calls == 3
    retries = _retries(ctx)
    # the flakiness is recorded, not smoothed away
    assert [r.payload["attempt"] for r in retries] == [1, 2]
    assert "server_error" in retries[0].payload["error"]


def test_retries_are_bounded(tmp_path: Path) -> None:
    with FlakyEndpoint(failures=99) as ep:
        p = ChatModelParticipant(base_url=ep.base_url, model="m", probe=False)
        ctx = _ctx(tmp_path)
        with pytest.raises(RuntimeError, match="endpoint failed"):
            p.run(None, ctx)
    assert ep.calls == 3  # one attempt plus two retries, then give up


def test_a_real_client_error_is_not_retried(tmp_path: Path) -> None:
    """A malformed request is our fault and will fail identically forever;
    retrying it just triples the time to the error message."""
    with FlakyEndpoint(failures=99, body=_BAD_REQUEST) as ep:
        p = ChatModelParticipant(base_url=ep.base_url, model="m", probe=False)
        ctx = _ctx(tmp_path)
        with pytest.raises(RuntimeError, match="unknown model"):
            p.run(None, ctx)
    assert ep.calls == 1
    assert _retries(ctx) == []
