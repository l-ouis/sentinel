"""LLM-as-a-judge: rubric arithmetic, journal append, blinding, repeats."""

import json
from pathlib import Path

import pytest

from sentinel.index import Index
from sentinel.journal.reader import iter_events, replay_check
from sentinel.judge import _extract_json, judge_rollout
from sentinel.participants.chat_model import ChatModelParticipant
from sentinel.participants.scripted import ScriptedParticipant
from sentinel.report import build_report
from sentinel.run import run_rollout
from tests.stub_openai import StubOpenAI

TASK = Path(__file__).parent.parent / "tasks" / "smoke-chat"
SCRIPT = TASK / "users" / "student.toml"


def verdict(fulfilled=1.0, verified=1.0, communication=1.0, penalties=()) -> str:
    return json.dumps(
        {
            "items": {
                "fulfilled": {"score": fulfilled, "evidence": [7], "note": "moved it"},
                "verified": {"score": verified, "evidence": [9], "note": "checked"},
                "communication": {"score": communication, "evidence": [9], "note": "clear"},
            },
            "penalties": {
                p: {"triggered": p in penalties, "evidence": [], "note": ""}
                for p in ("fabrication", "invented-capability")
            },
        }
    )


@pytest.fixture
def rollout(tmp_path) -> Path:
    move = {
        "name": "send_file",
        "args": {"name": "notes.md", "from_device": "laptop", "to_device": "phone"},
    }
    with StubOpenAI([move], final="Copied notes.md onto your phone.") as stub:
        summary = run_rollout(
            TASK,
            {
                "user": ScriptedParticipant(SCRIPT),
                "assistant": ChatModelParticipant(base_url=stub.base_url, model="stub-subject"),
            },
            runs_root=tmp_path, experiment="judged", orchestration="relay-v3:e6",
        )
    assert summary["status"] == "completed", summary
    return Path(summary["rollout_dir"])


def test_judge_scores_and_appends_to_the_journal(rollout: Path, tmp_path: Path) -> None:
    with StubOpenAI([], final=verdict()) as judge_stub:
        summary = judge_rollout(
            rollout, model="stub-judge", base_url=judge_stub.base_url,
            task_dir=TASK, runs_root=tmp_path,
        )
    assert summary["score"] == 1.0
    assert summary["item_scores"] == {"fulfilled": 1.0, "verified": 1.0, "communication": 1.0}

    # appended, not rewritten: the journal still replays cleanly afterwards
    assert replay_check(rollout / "journal.jsonl")
    judgments = [e for e in iter_events(rollout / "journal.jsonl") if e.type == "judge_result"]
    assert len(judgments) == 1
    assert judgments[0].judge.model == "stub-judge"
    assert judgments[0].rationale_ref  # the rationale is kept, not just the number
    assert json.loads((rollout / judgments[0].rationale_ref).read_text())["parsed"]

    index = Index(tmp_path / "index.db")
    metrics = dict(
        index.query("SELECT name, value FROM metrics WHERE rollout_id = ?", (rollout.name,))
    )
    index.close()
    assert metrics["judge.score"] == 1.0
    assert metrics["judge.item.fulfilled"] == 1.0
    assert metrics["reward"] == 1.0  # the verifier's number is untouched


def test_harness_does_the_arithmetic(rollout: Path, tmp_path: Path) -> None:
    """The judge emits item scores and triggered penalties; the score is the
    harness's weighted sum (0.5/0.25/0.25) times the penalty factor (0.25)."""
    with StubOpenAI([], final=verdict(fulfilled=1.0, verified=0.0, communication=0.5)) as s:
        summary = judge_rollout(
            rollout, model="stub-judge", base_url=s.base_url, task_dir=TASK, runs_root=tmp_path
        )
    assert summary["score"] == pytest.approx(0.5 + 0.0 + 0.125)

    with StubOpenAI([], final=verdict(penalties=("fabrication",))) as s:
        summary = judge_rollout(
            rollout, model="stub-judge", base_url=s.base_url, task_dir=TASK, runs_root=tmp_path
        )
    assert summary["score"] == pytest.approx(0.25)
    assert summary["penalties"] == ["fabrication"]


def test_repeats_report_the_spread(rollout: Path, tmp_path: Path) -> None:
    with StubOpenAI([], final=verdict()) as s:
        summary = judge_rollout(
            rollout, model="stub-judge", base_url=s.base_url, task_dir=TASK,
            runs_root=tmp_path, repeats=3,
        )
    assert summary["repeats"] == 3 and summary["score_sd"] == 0.0
    index = Index(tmp_path / "index.db")
    metrics = dict(
        index.query("SELECT name, value FROM metrics WHERE rollout_id = ?", (rollout.name,))
    )
    index.close()
    assert metrics["judge.n"] == 3.0

    report = build_report(tmp_path, "judged")
    assert report[0]["judge"] == 1.0


def _judge_metrics(rollout: Path, tmp_path: Path) -> dict:
    index = Index(tmp_path / "index.db")
    metrics = dict(
        index.query("SELECT name, value FROM metrics WHERE rollout_id = ?", (rollout.name,))
    )
    index.close()
    return metrics


def test_only_the_latest_pass_counts(rollout: Path, tmp_path: Path) -> None:
    """Re-judging with a different judge must replace the metric, not blend
    with the old one — while both verdicts stay in the journal as history."""
    with StubOpenAI([], final=verdict()) as s:
        judge_rollout(rollout, model="judge-a", base_url=s.base_url, task_dir=TASK, runs_root=tmp_path)
    with StubOpenAI([], final=verdict(fulfilled=0.0, verified=0.0, communication=0.0)) as s:
        judge_rollout(rollout, model="judge-b", base_url=s.base_url, task_dir=TASK, runs_root=tmp_path)

    assert _judge_metrics(rollout, tmp_path)["judge.score"] == 0.0
    assert len([e for e in iter_events(rollout / "journal.jsonl") if e.type == "judge_result"]) == 2


def test_rejudging_identically_does_not_blend_passes(rollout: Path, tmp_path: Path) -> None:
    """The dangerous case: SAME judge, SAME rubric path, run twice. Nothing in
    the judgment distinguishes the passes except the pass id — without it the
    metric silently averages a stale pass with a fresh one, which is how a
    fixed transcript bug goes on contaminating results after it is fixed."""
    with StubOpenAI([], final=verdict(fulfilled=0.0, verified=0.0, communication=0.0)) as s:
        judge_rollout(rollout, model="j", base_url=s.base_url, task_dir=TASK, runs_root=tmp_path)
    assert _judge_metrics(rollout, tmp_path)["judge.score"] == 0.0

    with StubOpenAI([], final=verdict()) as s:
        second = judge_rollout(
            rollout, model="j", base_url=s.base_url, task_dir=TASK,
            runs_root=tmp_path, repeats=2,
        )
    metrics = _judge_metrics(rollout, tmp_path)
    assert metrics["judge.score"] == 1.0   # not 0.67, which blending would give
    assert metrics["judge.n"] == 2.0

    events = [e for e in iter_events(rollout / "journal.jsonl") if e.type == "judge_result"]
    assert len(events) == 3                       # all three kept as history
    assert len({e.pass_id for e in events}) == 2  # under two distinct passes
    assert events[-1].pass_id == second["pass_id"]
    assert events[-1].rubric_sha256               # rubric content, not just path


def test_penalty_metric_is_a_fraction_of_the_pass(rollout: Path, tmp_path: Path) -> None:
    """One judge of three triggering a penalty is a disagreement, not a
    verdict, and the metric has to say which."""
    with StubOpenAI([], final=verdict(penalties=("fabrication",))) as s:
        judge_rollout(rollout, model="j", base_url=s.base_url, task_dir=TASK,
                      runs_root=tmp_path, repeats=3)
    # all three triggered it here, so the fraction is 1.0
    assert _judge_metrics(rollout, tmp_path)["judge.penalty.fabrication"] == 1.0


def test_transcript_is_blinded_and_carries_evidence_anchors(rollout: Path, tmp_path: Path) -> None:
    with StubOpenAI([], final=verdict()) as judge_stub:
        judge_rollout(
            rollout, model="stub-judge", base_url=judge_stub.base_url,
            task_dir=TASK, runs_root=tmp_path,
        )
        prompt = judge_stub.requests[0]["messages"][-1]["content"]
    assert "stub-subject" not in prompt   # the model under test is not named
    assert "the assistant" in prompt
    assert "[#" in prompt                 # seq anchors the judge can cite
    assert "Copied notes.md onto your phone." in prompt
    assert "notes.md" in prompt


def test_the_judge_sees_what_tools_returned(rollout: Path, tmp_path: Path) -> None:
    """A grounding rubric is unscoreable without tool results. Rendering the
    call but not its return made three real judges mark a correct run down for
    "no tool output backing this claim" — the output existed, the transcript
    just never showed it."""
    with StubOpenAI([], final=verdict()) as judge_stub:
        judge_rollout(
            rollout, model="stub-judge", base_url=judge_stub.base_url,
            task_dir=TASK, runs_root=tmp_path,
        )
        prompt = judge_stub.requests[0]["messages"][-1]["content"]
    assert '"sent": true' in prompt          # the result, not just the call
    assert "output <- send_file" in prompt   # attributed to the tool that ran


def test_judge_retries_a_malformed_verdict(rollout: Path, tmp_path: Path) -> None:
    """A judge that omits an item gets one retry that says so. Formatting
    slips are not judgments."""
    bad = json.dumps({"items": {"fulfilled": {"score": 1.0}}})
    with StubOpenAI([], final=bad) as s:
        with pytest.raises(ValueError, match="omitted rubric item"):
            judge_rollout(
                rollout, model="stub-judge", base_url=s.base_url,
                task_dir=TASK, runs_root=tmp_path,
            )
        assert len(s.requests) == 2  # the retry happened
        assert "was rejected" in s.requests[1]["messages"][-1]["content"]


def test_extract_json_survives_prose_and_fences() -> None:
    assert _extract_json('Sure! ```json\n{"a": {"b": 1}}\n``` hope that helps')["a"] == {"b": 1}
    assert _extract_json('{"note": "a } inside a string", "x": 2}')["x"] == 2
    with pytest.raises(ValueError):
        _extract_json("no json here")
