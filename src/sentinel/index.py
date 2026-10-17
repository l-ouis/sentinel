"""SQLite index over rollouts and metrics (PLAN §4: SQLite holds the index and
results; the journal directory holds the heavyweight truth)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rollouts (
    rollout_id   TEXT PRIMARY KEY,
    task_name    TEXT NOT NULL,
    task_version TEXT NOT NULL,
    experiment   TEXT,
    orchestration TEXT NOT NULL,         -- harness id, e.g. "solo-v1"
    bindings     TEXT NOT NULL,          -- {role: ParticipantSpec JSON + graded}
    subject      TEXT NOT NULL,          -- report label: graded roles [+fixtures]
    status       TEXT NOT NULL,          -- running | completed | crashed | ...
    started_wall TEXT NOT NULL,
    finished_wall TEXT,
    journal_path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    rollout_id TEXT NOT NULL,
    name       TEXT NOT NULL,
    value      REAL NOT NULL,
    PRIMARY KEY (rollout_id, name)
);
CREATE TABLE IF NOT EXISTS meta (
    rollout_id TEXT NOT NULL,
    key        TEXT NOT NULL,   -- dotted namespace, e.g. "host.hostname"
    value      TEXT NOT NULL,
    PRIMARY KEY (rollout_id, key)
);
"""


# Expected column order of `rollouts`, checked on open. The index is derived
# and disposable (§4), so a schema change costs a reindex — but only if the
# mismatch is reported as such instead of surfacing as a column-count error
# from deep inside an INSERT.
_ROLLOUT_COLUMNS = [
    "rollout_id", "task_name", "task_version", "experiment", "orchestration",
    "bindings", "subject", "status", "started_wall", "finished_wall",
    "journal_path",
]


class StaleIndexError(RuntimeError):
    """The index on disk was written by an older schema."""


class Index:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.db_path)
        self._db.executescript(_SCHEMA)
        self._db.commit()
        self._check_schema()

    def _check_schema(self) -> None:
        found = [row[1] for row in self._db.execute("PRAGMA table_info(rollouts)")]
        if found == _ROLLOUT_COLUMNS:
            return
        self._db.close()
        missing = [c for c in _ROLLOUT_COLUMNS if c not in found]
        raise StaleIndexError(
            f"{self.db_path} was built by an older index schema"
            + (f" (missing: {', '.join(missing)})" if missing else "")
            + ". The journals are the truth and the index is derived — rebuild "
            f"it with:  sentinel reindex --runs-dir {self.db_path.parent}"
        )

    def rollout_started(
        self,
        rollout_id: str,
        *,
        task_name: str,
        task_version: str,
        bindings: dict[str, dict],
        subject: str,
        journal_path: str,
        started_wall: str,
        orchestration: str = "solo-v1",
        experiment: str | None = None,
    ) -> None:
        self._db.execute(
            "INSERT INTO rollouts VALUES (?,?,?,?,?,?,?,?,?,NULL,?)",
            (rollout_id, task_name, task_version, experiment, orchestration,
             json.dumps(bindings), subject, "running", started_wall, journal_path),
        )
        self._db.commit()

    def rollout_finished(self, rollout_id: str, status: str, finished_wall: str) -> None:
        self._db.execute(
            "UPDATE rollouts SET status = ?, finished_wall = ? WHERE rollout_id = ?",
            (status, finished_wall, rollout_id),
        )
        self._db.commit()

    def metric(self, rollout_id: str, name: str, value: float) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO metrics VALUES (?,?,?)",
            (rollout_id, name, float(value)),
        )
        self._db.commit()

    def meta(self, rollout_id: str, key: str, value: object) -> None:
        """Non-numeric rollout facts (provenance, host, config)."""
        self._db.execute(
            "INSERT OR REPLACE INTO meta VALUES (?,?,?)",
            (rollout_id, key, str(value)),
        )
        self._db.commit()

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        return self._db.execute(sql, params).fetchall()

    def close(self) -> None:
        self._db.close()
