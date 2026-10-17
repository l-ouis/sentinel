"""The index is derived and disposable (§4) — but a schema change has to say
so out loud, not surface as a column-count error from inside an INSERT."""

import sqlite3
from pathlib import Path

import pytest

from sentinel.index import Index, StaleIndexError

_OLD_SCHEMA = """
CREATE TABLE rollouts (
    rollout_id   TEXT PRIMARY KEY,
    task_name    TEXT NOT NULL,
    task_version TEXT NOT NULL,
    experiment   TEXT,
    participant  TEXT NOT NULL,
    status       TEXT NOT NULL,
    started_wall TEXT NOT NULL,
    finished_wall TEXT,
    journal_path TEXT NOT NULL
);
"""


def test_stale_index_names_the_fix(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    conn = sqlite3.connect(db)
    conn.executescript(_OLD_SCHEMA)
    conn.commit()
    conn.close()

    with pytest.raises(StaleIndexError) as e:
        Index(db)
    assert "orchestration" in str(e.value)   # says what is missing
    assert "sentinel reindex" in str(e.value)  # and what to do about it


def test_fresh_index_opens_and_reopens(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    index = Index(db)
    index.rollout_started(
        "r1", task_name="t", task_version="0.1", bindings={}, subject="s",
        journal_path="j", started_wall="2026-01-01T00:00:00",
    )
    index.close()
    reopened = Index(db)  # the schema check must pass on our own tables
    assert reopened.query("SELECT rollout_id FROM rollouts") == [("r1",)]
    reopened.close()
