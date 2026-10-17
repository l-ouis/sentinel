"""Deterministic grade: notes.typ exists -> typst compiles it -> content
recall against the blessed oracle transcript.

Until oracle_transcript.txt is blessed (one whisper run over the real mp3s,
reviewed and committed next to this file), grading is compile-only.
"""

import json
import re
import subprocess
from pathlib import Path

NOTES = Path("/root/notes/notes.typ")
ORACLE = Path(__file__).parent / "oracle_transcript.txt"
RECALL_THRESHOLD = 0.6


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 3}


def main() -> None:
    out: dict = {"reward": 0.0, "passed": False}
    if not NOTES.is_file():
        out["detail"] = "notes.typ missing"
        print(json.dumps(out))
        return
    proc = subprocess.run(
        ["typst", "compile", str(NOTES), "/tmp/notes.pdf"],
        capture_output=True, text=True,
    )
    out["compiles"] = proc.returncode == 0
    if not out["compiles"]:
        out["detail"] = proc.stderr[-500:]
        print(json.dumps(out))
        return
    if ORACLE.is_file():
        oracle = content_words(ORACLE.read_text())
        notes = content_words(NOTES.read_text())
        recall = len(oracle & notes) / len(oracle) if oracle else 0.0
        out["recall"] = round(recall, 3)
        out["reward"] = round(recall, 3)
        out["passed"] = recall >= RECALL_THRESHOLD
    else:
        out["reward"] = 1.0
        out["passed"] = True
        out["detail"] = "no oracle transcript yet: compile-only grading"
    print(json.dumps(out))


main()
