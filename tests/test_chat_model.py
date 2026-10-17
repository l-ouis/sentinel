"""ChatModelParticipant conversation persistence against the stub server."""

from pathlib import Path

from sentinel.journal import JournalWriter
from sentinel.participants import ChatModelParticipant
from sentinel.participants.base import RolloutContext
from tests.stub_openai import StubOpenAI


def _ctx(tmp_path: Path, **kw) -> RolloutContext:
    writer = JournalWriter(tmp_path / "r1", "r1")
    return RolloutContext(
        writer=writer, role="assistant", instruction="turn one",
        rollout_dir=tmp_path / "r1", **kw,
    )


def test_conversation_carries_messages(tmp_path: Path) -> None:
    with StubOpenAI([], final="reply one") as stub:
        p = ChatModelParticipant(base_url=stub.base_url, model="m")
        ctx = _ctx(tmp_path)
        r1 = p.run(None, ctx)
        assert r1.final == "reply one"
        ctx2 = RolloutContext(
            writer=ctx.writer, role="assistant", instruction="turn two",
            rollout_dir=ctx.rollout_dir, conversation=r1.conversation,
        )
        p.run(None, ctx2)
        ctx.writer.close()
    msgs = stub.requests[-1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"] == "turn one"
    assert msgs[3]["content"] == "turn two"


def test_handoff_conversation_stays_protocol_valid(tmp_path: Path) -> None:
    with StubOpenAI([], final="passing the baton", handoff=("blake", "go")) as stub:
        p = ChatModelParticipant(base_url=stub.base_url, model="m")
        ctx = _ctx(tmp_path, peers=["blake"])
        r = p.run(None, ctx)
        ctx.writer.close()
    assert r.handoff.to == "blake"
    # the hand_off tool call got a synthetic response, so resuming this
    # conversation later won't violate the tools protocol
    assert r.conversation[-1]["role"] == "tool"
    assert r.conversation[-2]["tool_calls"][0]["function"]["name"] == "hand_off"
