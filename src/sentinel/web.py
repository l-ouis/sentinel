"""Local web console: live status over runs/index.db, journals, and tasks.

One plain HTML page (black on white, no frameworks) that polls two JSON
endpoints. stdlib http.server only — no new dependencies; state building is
pure functions over the same SQLite index and journal files the CLI uses.
Read-only by design: the console never writes to the index or journals.
"""

from __future__ import annotations

import json
import re
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from sentinel.tools import strip_executed_fences

_RID_RE = re.compile(r"^[A-Za-z0-9._-]+$")  # rollout ids; rejects path traversal
_ROLLOUT_CAP = 500  # newest rollouts served to the page
_EVENT_CAP = 500  # newest journal events served per rollout


def _query(db_path: Path, sql: str, params: tuple = ()) -> list[tuple]:
    """Read-only query; a missing or locked-out index is just an empty console."""
    if not db_path.exists():
        return []
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return db.execute(sql, params).fetchall()
    finally:
        db.close()


def _clip(s: str | None, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " …"


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?")


def _sanitize_output(text: str) -> str:
    """Terminal control junk -> readable text: drop ANSI escape sequences and
    resolve carriage-return overwrites (progress bars) to their final state."""
    text = _ANSI_RE.sub("", text)
    return "\n".join(
        line.rsplit("\r", 1)[-1] if "\r" in line else line
        for line in text.split("\n")
    )


def _summarize_event(ev: dict) -> str:
    """One greppable line per journal event for the detail view."""
    t = ev.get("type")
    d = ev.get("data") or {}
    if t == "action":
        return f"{ev.get('kind')}: {_clip(d.get('command') or json.dumps(d), 300)}"
    if t == "action_result":
        bits = [f"{ev.get('status')} exit={d.get('exit_code')}"]
        for name in ("stdout", "stderr"):
            if (d.get(name) or "").strip():
                bits.append(f"{name}: {_clip(d[name], 160)}")
        return " | ".join(bits)
    if t == "model_call":
        p = ev.get("participant") or {}
        secs = (ev.get("latency_ms") or 0) / 1000
        return (
            f"{p.get('model')} {ev.get('prompt_tokens')}→"
            f"{ev.get('completion_tokens')} tok, {secs:.1f}s"
        )
    if t == "verifier_result":
        return (
            f"reward={ev.get('reward')} passed={ev.get('passed')} "
            f"{_clip(json.dumps(ev.get('details') or {}), 300)}"
        )
    if t == "probe_result":
        return f"{ev.get('probe')} score={ev.get('score')}"
    if t == "check_result":
        return (
            f"{ev.get('check')} {'pass' if ev.get('passed') else 'FAIL'}: "
            f"{_clip(ev.get('detail'), 200)}"
        )
    if t == "judge_result":
        return f"score={ev.get('score')} items={json.dumps(ev.get('item_scores'))}"
    if t == "env_event":
        return f"{ev.get('name')} {_clip(json.dumps(ev.get('payload') or {}), 200)}"
    if t == "session_start":
        return f"trigger={ev.get('trigger')} roles={','.join(ev.get('roles') or [])}"
    if t == "session_end":
        return f"reason={ev.get('reason')}"
    if t == "rollout_started":
        return f"{ev.get('task_name')} v{ev.get('task_version')} experiment={ev.get('experiment')}"
    if t == "rollout_finished":
        detail = ev.get("detail")
        return str(ev.get("status")) + (f" — {_clip(detail, 300)}" if detail else "")
    if t == "provisioned":
        return f"{ev.get('backend')} image={ev.get('image')}"
    if t == "tool_call":
        err = ev.get("error")
        return f"{ev.get('name')} " + (f"error: {_clip(err, 120)}" if err else "ok")
    known = {"v", "seq", "rollout_id", "stream", "episode", "ts_wall", "ts_sim", "type", "category"}
    return _clip(json.dumps({k: v for k, v in ev.items() if k not in known}), 200)


def _read_journal(path: Path) -> list[dict]:
    """Tolerant journal read for display: a half-written last line (a rollout
    mid-flight) is skipped, not an error — replay_check stays the strict one."""
    events = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return events


def _scan_tasks(tasks_root: Path) -> list[dict]:
    from sentinel.schemas.task import load_task

    tasks = []
    if not tasks_root.is_dir():
        return tasks
    for d in sorted(p for p in tasks_root.iterdir() if (p / "task.toml").exists()):
        try:
            cfg = load_task(d)
            tasks.append(
                {
                    "name": cfg.task.name,
                    "version": cfg.task.version,
                    "roles": [f"{r.name}({r.shape})" for r in cfg.participants.roles],
                    "backend": cfg.environment.backend,
                    "events": len(cfg.events),
                    "probes": len(cfg.probes),
                }
            )
        except Exception as e:
            tasks.append({"name": d.name, "error": f"{type(e).__name__}: {e}"})
    return tasks


def build_state(
    runs_root: str | Path, tasks_root: str | Path, experiment: str | None = None
) -> dict:
    """Everything the page shows, in one JSON-able dict."""
    runs_root, tasks_root = Path(runs_root), Path(tasks_root)
    db = runs_root / "index.db"

    where, params = ("WHERE r.experiment = ?", (experiment,)) if experiment else ("", ())
    rows = _query(
        db,
        f"""
        SELECT r.rollout_id, r.task_name, r.experiment, r.subject, r.orchestration,
               r.status, r.started_wall, r.finished_wall, m.name, m.value
        FROM rollouts r LEFT JOIN metrics m ON m.rollout_id = r.rollout_id
        {where} ORDER BY r.started_wall DESC, r.rollout_id DESC
        """,
        params,
    )
    rollouts: dict[str, dict] = {}
    for rid, task, exp, subject, orch, status, started, finished, mname, mvalue in rows:
        cell = rollouts.setdefault(
            rid,
            {
                "rollout_id": rid,
                "task": task,
                "experiment": exp,
                "subject": subject,
                "orchestration": orch,
                "status": status,
                "started": started,
                "finished": finished,
                "metrics": {},
            },
        )
        if mname is not None:
            cell["metrics"][mname] = mvalue
        if len(rollouts) > _ROLLOUT_CAP:
            break

    # Live peek: running rollouts get their journal length + last event line.
    for cell in rollouts.values():
        if cell["status"] == "running":
            events = _read_journal(runs_root / cell["rollout_id"] / "journal.jsonl")
            cell["live_events"] = len(events)
            cell["live_last"] = _summarize_event(events[-1]) if events else ""

    counts = dict(
        _query(db, "SELECT status, COUNT(*) FROM rollouts GROUP BY status")
    )
    per_task = {
        row[0]: {"rollouts": row[1], "running": row[2], "completed": row[3], "passed": row[4]}
        for row in _query(
            db,
            """
            SELECT r.task_name, COUNT(*), SUM(r.status = 'running'),
                   SUM(r.status = 'completed'), SUM(COALESCE(m.value, 0) >= 1.0)
            FROM rollouts r
            LEFT JOIN metrics m ON m.rollout_id = r.rollout_id AND m.name = 'passed'
            GROUP BY r.task_name
            """,
        )
    }
    tasks = _scan_tasks(tasks_root)
    for t in tasks:
        t.update(per_task.get(t["name"], {"rollouts": 0, "running": 0, "completed": 0, "passed": 0}))

    report: list[dict] = []
    if db.exists():
        from sentinel.report import build_report

        report = build_report(runs_root, experiment)

    experiments = [
        r[0]
        for r in _query(
            db,
            "SELECT DISTINCT experiment FROM rollouts "
            "WHERE experiment IS NOT NULL ORDER BY experiment",
        )
    ]
    return {
        "counts": counts,
        "experiments": experiments,
        "tasks": tasks,
        "report": report,
        "rollouts": list(rollouts.values()),
    }


# One source of truth for reading sidecars: the viewer and the scorers must
# never disagree about what the model actually said.
from sentinel.transcript import assistant_text as _assistant_text  # noqa: E402
from sentinel.transcript import load_sidecar as _load_sidecar  # noqa: E402


def _transcript_entries(rollout_dir: Path, events: list[dict]) -> list[dict]:
    """The rollout as a conversation: instruction, model replies (from the
    sidecar bodies), commands, and outputs, in journal order. Feedback/tool
    messages are omitted — the action_result entry already is that content."""
    entries: list[dict] = []
    # Strictly chronological (journal order IS generation-then-execution
    # order): each assistant message renders whole, then its tool call, then
    # the tool response. Per role, the first model call also surfaces the
    # system prompt + instruction that role actually received.
    seen_streams: set[str] = set()
    for ev in events:
        t = ev.get("type")
        if t == "model_call":
            stream = ev.get("stream") or ""
            if stream not in seen_streams:
                seen_streams.add(stream)
                req = _load_sidecar(rollout_dir, ev.get("request_ref"))
                if req and req.get("messages"):  # OpenAI shape
                    for m in req["messages"]:
                        if m.get("role") in ("system", "user") and m.get("content"):
                            entries.append(
                                {"kind": m["role"], "stream": stream, "text": m["content"]}
                            )
                elif req:  # claude CLI shape: system prompt lives in argv
                    argv = req.get("argv") or []
                    if "--system-prompt" in argv:
                        entries.append({
                            "kind": "system", "stream": stream,
                            "text": argv[argv.index("--system-prompt") + 1],
                        })
                    if req.get("prompt"):
                        entries.append(
                            {"kind": "user", "stream": stream, "text": req["prompt"]}
                        )
            resp = _load_sidecar(rollout_dir, ev.get("response_ref"))
            text = _assistant_text(resp)
            if resp and "result" in resp and text:
                # claude CLI shape: the executed fences ARE the tool calls,
                # rendered as the typed entries that follow — strip, don't echo
                text = strip_executed_fences(text)
            if text and text.strip():
                entries.append(
                    {
                        "kind": "assistant",
                        "stream": stream,
                        "text": text.strip(),
                        "prompt_tokens": ev.get("prompt_tokens"),
                        "completion_tokens": ev.get("completion_tokens"),
                        "latency_ms": ev.get("latency_ms"),
                    }
                )
        elif t == "tool_call":
            args = ev.get("args") or {}
            if ev.get("name") == "hand_off":
                text = f"→ {args.get('to')}"
                if args.get("message"):
                    text += f": {args['message']}"
            else:
                text = json.dumps(args)
            if ev.get("error"):
                text += f"\n[error] {ev['error']}"
            entries.append(
                {"kind": "tool", "name": ev.get("name"), "stream": ev.get("stream"), "text": text}
            )
            # what the tool returned, as its own entry — reading a transcript
            # of calls with no results tells you what was attempted, not what
            # happened (the same gap that misled the judge)
            if ev.get("result") is not None and ev.get("name") != "hand_off":
                entries.append(
                    {"kind": "output", "exit_code": None,
                     "text": json.dumps(ev["result"], indent=1)}
                )
        elif t == "action":
            entries.append(
                {"kind": "command", "stream": ev.get("stream"),
                 "text": (ev.get("data") or {}).get("command", "")}
            )
        elif t == "action_result":
            d = ev.get("data") or {}
            text = _sanitize_output(d.get("stdout") or "")
            if (d.get("stderr") or "").strip():
                text += ("\n" if text else "") + "[stderr]\n" + _sanitize_output(d["stderr"])
            entries.append(
                {"kind": "output", "exit_code": d.get("exit_code"), "text": text}
            )
        elif t == "env_event" and ev.get("name") == "scripted_utterance":
            payload = ev.get("payload") or {}
            entries.append(
                {"kind": "user", "stream": ev.get("stream"),
                 "text": str(payload.get("say") or "")}
            )
        elif t == "verifier_result":
            entries.append(
                {
                    "kind": "verifier",
                    "reward": ev.get("reward"),
                    "passed": ev.get("passed"),
                    "text": json.dumps(ev.get("details") or {}, indent=1),
                }
            )
        elif t == "rollout_finished":
            entries.append(
                {"kind": "finished", "status": ev.get("status"), "text": ev.get("detail") or ""}
            )
    return entries


def rollout_detail(runs_root: str | Path, rollout_id: str) -> dict:
    """Everything the /rollout/<id> page shows: index metadata, the transcript
    (journal + sidecar bodies), the summarized event log, and the crash tail."""
    runs_root = Path(runs_root)
    if not _RID_RE.match(rollout_id):
        raise KeyError(rollout_id)
    rollout_dir = runs_root / rollout_id
    events = _read_journal(rollout_dir / "journal.jsonl")
    rows = _query(
        runs_root / "index.db",
        "SELECT status, bindings, subject, orchestration, experiment, task_name, "
        "started_wall, finished_wall FROM rollouts WHERE rollout_id = ?",
        (rollout_id,),
    )
    status, bindings, subject, orchestration, experiment, task, started, finished = (
        rows[0] if rows else (None,) * 8
    )
    metrics = dict(
        _query(
            runs_root / "index.db",
            "SELECT name, value FROM metrics WHERE rollout_id = ?",
            (rollout_id,),
        )
    )
    meta = dict(
        _query(
            runs_root / "index.db",
            "SELECT key, value FROM meta WHERE rollout_id = ?",
            (rollout_id,),
        )
    )
    crash = ""
    crash_path = rollout_dir / "crash.txt"
    if crash_path.exists():
        crash = crash_path.read_text()[-3000:]
    return {
        "rollout_id": rollout_id,
        "status": status,
        "task": task,
        "experiment": experiment,
        "subject": subject,
        "orchestration": orchestration,
        "bindings": json.loads(bindings) if bindings else None,
        "started": started,
        "finished": finished,
        "metrics": metrics,
        "meta": meta,
        "dir": str(rollout_dir),
        "total_events": len(events),
        "transcript": _transcript_entries(rollout_dir, events),
        "events": [
            {
                "seq": ev.get("seq"),
                "ts": ev.get("ts_wall"),
                "stream": ev.get("stream"),
                "type": ev.get("type"),
                "text": _summarize_event(ev),
            }
            for ev in events[-_EVENT_CAP:]
        ],
        "crash": crash,
    }


_CSS = """
  body { background: #fff; color: #000; margin: 1.5rem;
         font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  h1 { font-size: 15px; margin: 0 0 2px; }
  h2 { font-size: 13px; margin: 1.6em 0 0.3em; }
  table { border-collapse: collapse; }
  th, td { text-align: left; padding: 1px 14px 1px 0; vertical-align: top;
           border-bottom: 1px solid #e0e0e0; white-space: nowrap; }
  th { border-bottom: 1px solid #000; }
  td.num, th.num { text-align: right; }
  td.wrap, span.wrap { white-space: pre-wrap; word-break: break-word; max-width: 60em; }
  .dim { color: #666; }
  #err { display: none; border: 1px solid #000; padding: 3px 8px; margin: 8px 0; }
  select { font: inherit; }
  pre { margin: 4px 0; white-space: pre-wrap; word-break: break-word; }
"""

_JS_HELPERS = """
const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const num = (v, d = 2) => v == null ? '\\u2013'
  : Number.isInteger(v) ? String(v) : Number(v).toFixed(d);
const hhmmss = iso => iso ? iso.slice(11, 19) : '\\u2013';
const rolloutLink = id => `<a href="/rollout/${encodeURIComponent(id)}">${esc(id)}</a>`;
function showError(e) {
  const el = document.getElementById('err');
  el.style.display = 'block';
  el.textContent = 'fetch failed: ' + e;
}
function clearError() { document.getElementById('err').style.display = 'none'; }
"""

_PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sentinel</title>
<style>__CSS__</style>
<h1>sentinel</h1>
<div><span id="top" class="dim">loading…</span>
  &nbsp; experiment: <select id="exp"><option value="">(all)</option></select></div>
<div id="err"></div>
<div id="running"></div>
<h2>report</h2><div id="report"></div>
<h2>tasks</h2><div id="tasks"></div>
<h2>rollouts</h2><div id="rollouts"></div>
<script>
'use strict';
__HELPERS__
let exp = new URLSearchParams(location.search).get('experiment') || '';
let state = null, lastRender = '';

const elapsed = iso => {
  const s = Math.max(0, (Date.now() - Date.parse(iso)) / 1000) | 0;
  return `${(s / 60) | 0}m${String(s % 60).padStart(2, '0')}s`;
};

function table(cols, rows) {
  // cols: [ [label, class?] ]; rows: array of arrays of pre-escaped html
  let h = '<table><tr>' + cols.map(c =>
    `<th class="${c[1] || ''}">${c[0]}</th>`).join('') + '</tr>';
  for (const r of rows) h += '<tr' + (r.attrs || '') + '>' + r.cells.map((v, i) =>
    `<td class="${cols[i][1] || ''}">${v}</td>`).join('') + '</tr>';
  return h + '</table>';
}

function render() {
  const c = state.counts || {};
  const total = Object.values(c).reduce((a, b) => a + b, 0);
  document.getElementById('top').textContent =
    `${total} rollouts — ${c.running || 0} running, ${c.completed || 0} completed, ` +
    `${c.crashed || 0} crashed — updated ${new Date().toTimeString().slice(0, 8)}`;
  const key = JSON.stringify(state);
  if (key === lastRender) return;  // keep tables stable (text selection) when nothing changed
  lastRender = key;

  const sel = document.getElementById('exp');
  const want = ['', ...state.experiments];
  if ([...sel.options].map(o => o.value).join('\\n') !== want.join('\\n')) {
    sel.innerHTML = want.map(e =>
      `<option value="${esc(e)}">${e ? esc(e) : '(all)'}</option>`).join('');
  }
  sel.value = exp;

  const running = state.rollouts.filter(r => r.status === 'running');
  document.getElementById('running').innerHTML = !running.length ? '' :
    '<h2>running now</h2>' + table(
      [['rollout'], ['task'], ['model'], ['started'], ['elapsed'], ['events', 'num'], ['last event']],
      running.map(r => ({
        cells: [rolloutLink(r.rollout_id), esc(r.task), esc(r.subject),
                hhmmss(r.started), elapsed(r.started), num(r.live_events),
                `<span class="wrap">${esc(r.live_last)}</span>`],
      })));

  document.getElementById('report').innerHTML = !state.report.length
    ? '<span class="dim">no rollouts</span>'
    : table(
      [['participant'], ['orch'], ['n', 'num'], ['crashed', 'num'], ['reward', 'num'],
       ['pass^1', 'num'], ['pass^5', 'num'], ['tok/s', 'num'], ['wall_s', 'num'], ['cmds', 'num']],
      state.report.map(r => ({ cells: [
        esc(r.participant), esc(r.orch), num(r.n), num(r.crashed), num(r.reward, 3),
        num(r['pass^1'], 3), num(r['pass^5'], 3), num(r['tok/s'], 1),
        num(r.wall_s, 1), num(r.cmds, 1),
      ]})));

  document.getElementById('tasks').innerHTML = table(
    [['task'], ['version'], ['roles'], ['backend'], ['rollouts', 'num'],
     ['completed', 'num'], ['passed', 'num']],
    state.tasks.map(t => ({ cells: t.error
      ? [esc(t.name), '', `<span class="wrap">config error: ${esc(t.error)}</span>`, '', '', '', '']
      : [esc(t.name), esc(t.version), esc(t.roles.join(', ')), esc(t.backend),
         num(t.rollouts), num(t.completed), num(t.passed)],
    })));

  let h = '<table><tr><th>started</th><th>rollout</th><th>task</th><th>experiment</th>' +
    '<th>participant</th><th>orch</th><th>status</th><th class="num">reward</th>' +
    '<th class="num">wall_s</th><th class="num">cmds</th><th class="num">tok p+c</th>' +
    '<th class="num">tok/s</th></tr>';
  for (const r of state.rollouts) {
    const m = r.metrics || {};
    const cells = [
      esc((r.started || '').slice(0, 19).replace('T', ' ')), rolloutLink(r.rollout_id),
      esc(r.task), esc(r.experiment || '–'),
      esc(r.subject), esc(r.orchestration),
      esc(r.status) + (r.status !== 'completed' ? '' : m.passed >= 1 ? ' ✓' : ' ✗'),
      num(m.reward, 3), num(m.wall_seconds, 1), num(m.shell_commands),
      m.prompt_tokens == null ? '–' : `${num(m.prompt_tokens)}+${num(m.completion_tokens)}`,
      num(m.tok_per_sec, 1),
    ];
    h += '<tr>' + cells.map(v => `<td>${v}</td>`).join('') + '</tr>';
  }
  document.getElementById('rollouts').innerHTML = h + '</table>';
}

async function tick() {
  try {
    const r = await fetch('/api/state?experiment=' + encodeURIComponent(exp));
    if (!r.ok) throw new Error('HTTP ' + r.status);
    state = await r.json();
    clearError();
    render();
  } catch (e) {
    showError(e);
  }
}

document.getElementById('exp').addEventListener('change', e => {
  exp = e.target.value;
  history.replaceState(null, '', exp ? '/?experiment=' + encodeURIComponent(exp) : '/');
  tick();
});
tick();
setInterval(tick, 2000);
</script>
""".replace("__CSS__", _CSS).replace("__HELPERS__", _JS_HELPERS)


_ROLLOUT_PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rollout</title>
<style>__CSS__
  .entry { margin: 14px 0; }
  .lbl { font-weight: 700; margin-bottom: 2px; }
  .tag { display: block; font-weight: 700; }
  .tag.bash { color: #0550ae; }
  .tag.output { color: #777; }
  .tag.tool { color: #7a3fa8; }
  pre.tool { border-left: 2px solid #7a3fa8; padding: 3px 8px; margin: 2px 0; }
  .ok { color: #1a7f37; }
  .bad { color: #b02a2a; }
  pre.cmd { border-left: 2px solid #0550ae; background: #f6f6f6;
            padding: 3px 8px; margin: 2px 0; }
  pre.out { color: #444; border-left: 2px solid #d0d0d0; padding: 3px 8px; margin: 2px 0; }
  pre.out.fail { border-left-color: #b02a2a; }
</style>
<div><a href="/">&larr; sentinel</a> &middot; <a id="tlink" href="#">timeline</a></div>
<h1 id="rid"></h1>
<div id="meta" class="dim">loading…</div>
<div id="err"></div>
<div id="transcript"></div>
<div id="crash"></div>
<h2>event log</h2><div id="events"></div>
<script>
'use strict';
__HELPERS__
const rid = decodeURIComponent(location.pathname.split('/').pop());
document.getElementById('rid').textContent = rid;
document.title = rid;
document.getElementById('tlink').href = '/rollout/' + encodeURIComponent(rid) + '/timeline';
let lastRender = '';

function entryHtml(e) {
  const who = e.stream ? ` (${esc(e.stream)})` : '';
  switch (e.kind) {
    case 'system':
      return `<details class="entry dim"><summary class="lbl">system${who}</summary>` +
        `<pre>${esc(e.text)}</pre></details>`;
    case 'user':
      return `<div class="entry"><div class="lbl">user${who}</div><pre>${esc(e.text)}</pre></div>`;
    case 'assistant': {
      const info = ` <span class="dim">${num(e.prompt_tokens)}→${num(e.completion_tokens)}` +
        ` tok, ${((e.latency_ms || 0) / 1000).toFixed(1)}s</span>`;
      return `<div class="entry"><div class="lbl">assistant${who}${info}</div><pre>${esc(e.text)}</pre></div>`;
    }
    case 'tool':
      return `<div class="entry"><span class="tag tool">tool call · ${esc(e.name)}${who}</span>` +
        `<pre class="tool">${esc(e.text)}</pre></div>`;
    case 'command':
      return `<div class="entry"><span class="tag bash">tool call · run_shell</span>` +
        `<pre class="cmd">${esc(e.text)}</pre></div>`;
    case 'output': {
      const tag = e.exit_code
        ? `<span class="tag bad">tool response · exit ${num(e.exit_code)}</span>`
        : `<span class="tag output">tool response</span>`;
      const body = e.text ? esc(e.text) : '<span class="dim">(no output)</span>';
      return `<div class="entry">${tag}<pre class="out${e.exit_code ? ' fail' : ''}">${body}</pre></div>`;
    }
    case 'verifier': {
      const verdict = e.passed
        ? `<span class="ok">passed</span>` : `<span class="bad">failed</span>`;
      return `<div class="entry"><div class="lbl">verifier — reward ${num(e.reward, 3)}` +
        ` · ${verdict}</div><pre>${esc(e.text)}</pre></div>`;
    }
    case 'finished': {
      const cls = e.status === 'completed' ? 'ok' : 'bad';
      return `<div class="entry"><div class="lbl">rollout <span class="${cls}">${esc(e.status)}</span></div>` +
        (e.text ? `<pre>${esc(e.text)}</pre>` : '') + `</div>`;
    }
  }
  return '';
}

function render(d) {
  const key = JSON.stringify(d);
  if (key === lastRender) return;
  lastRender = key;
  const m = d.metrics || {};
  const bits = [d.task, d.subject, d.orchestration, d.experiment, d.status];
  if (m.reward != null) bits.push(`reward ${num(m.reward, 3)} (${m.passed >= 1 ? 'passed' : 'failed'})`);
  if (m.wall_seconds != null) bits.push(`${num(m.wall_seconds, 1)}s`);
  const host = d.meta && d.meta['host.hostname'];
  if (host) bits.push(`on ${host}${d.meta['host.gpu'] ? ' (' + d.meta['host.gpu'] + ')' : ''}`);
  bits.push(`${d.total_events} events`, d.dir);
  const roles = Object.entries(d.bindings || {}).map(([role, s]) =>
    `${role}: ${s.model || s.kind}${s.graded === false ? ' (fixture)' : ''}`);
  document.getElementById('meta').innerHTML = esc(bits.filter(Boolean).join(' · ')) +
    (roles.length > 1 ? `<br>${esc(roles.join(' · '))}` : '');

  document.getElementById('transcript').innerHTML =
    (d.transcript || []).map(entryHtml).join('') ||
    '<div class="dim">no transcript yet</div>';

  document.getElementById('crash').innerHTML = d.crash
    ? `<h2>crash.txt</h2><pre>${esc(d.crash)}</pre>` : '';

  let h = '<table><tr><th class="num">seq</th><th>time</th><th>stream</th><th>type</th><th>summary</th></tr>';
  for (const e of d.events) {
    h += `<tr><td class="num">${num(e.seq)}</td><td>${hhmmss(e.ts)}</td><td>${esc(e.stream)}</td>` +
      `<td>${esc(e.type)}</td><td><span class="wrap">${esc(e.text)}</span></td></tr>`;
  }
  document.getElementById('events').innerHTML = h + '</table>';
}

async function tick() {
  let delay = 2000;
  try {
    const r = await fetch('/api/rollout/' + encodeURIComponent(rid));
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    clearError();
    render(d);
    if (d.status && d.status !== 'running') return;  // terminal: stop polling
  } catch (e) {
    showError(e);
    delay = 5000;
  }
  setTimeout(tick, delay);
}
tick();
</script>
""".replace("__CSS__", _CSS).replace("__HELPERS__", _JS_HELPERS)


_TIMELINE_PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>timeline</title>
<style>__CSS__
  #wrap { overflow-x: auto; }
  #grid { display: grid; align-items: start; column-gap: 12px; row-gap: 2px; }
  .lanehead { position: sticky; top: 0; background: #fff; z-index: 2;
              border-bottom: 1px solid #000; font-weight: 700; padding: 2px 4px; }
  .time { color: #666; white-space: nowrap; padding: 2px 6px 2px 0;
          position: sticky; left: 0; background: #fff; z-index: 1; }
  .card { border: 1px solid #e0e0e0; border-left-width: 3px; padding: 2px 6px;
          white-space: pre-wrap; word-break: break-word; }
  .card .k { font-weight: 700; }
  .card.model_call { border-left-color: #000; }
  .card.action { border-left-color: #0550ae; }
  .card.action_result { border-left-color: #bbb; color: #444; }
  .card.action_result.fail { border-left-color: #b02a2a; }
  .card.tool_call { border-left-color: #7a3fa8; }
  .card.verifier_result.ok { border-left-color: #1a7f37; }
  .card.verifier_result.bad { border-left-color: #b02a2a; }
  .card.lifecycle { background: #f6f6f6; border-left-color: #e0e0e0; }
  .gap { color: #999; text-align: center; padding: 1px 0; }
</style>
<div><a id="back" href="#">&larr; rollout</a> &middot; <a href="/">sentinel</a></div>
<h1 id="rid"></h1>
<div id="meta" class="dim">loading…</div>
<div id="err"></div>
<div id="wrap"><div id="grid"></div></div>
<script>
'use strict';
__HELPERS__
const rid = decodeURIComponent(location.pathname.split('/')[2]);
document.getElementById('rid').textContent = rid + ' — timeline';
document.title = rid + ' · timeline';
document.getElementById('back').href = '/rollout/' + encodeURIComponent(rid);
let lastRender = '';
const LIFECYCLE = new Set(['rollout_started', 'provisioned', 'session_start',
                           'session_end', 'rollout_finished', 'checkpoint']);
const GAP_MS = 5000;

function cardClass(e) {
  let cls = e.type;
  if (LIFECYCLE.has(e.type)) cls += ' lifecycle';
  if (e.type === 'action_result' && !/exit=0\\b/.test(e.text)) cls += ' fail';
  if (e.type === 'verifier_result') cls += /passed=True/.test(e.text) ? ' ok' : ' bad';
  return cls;
}

function render(d) {
  const key = JSON.stringify(d);
  if (key === lastRender) return;
  lastRender = key;
  const m = d.metrics || {};
  document.getElementById('meta').textContent = [
    d.task, d.subject, d.orchestration, d.status,
    m.reward != null ? `reward ${num(m.reward, 3)}` : null,
    `${d.total_events} events`,
  ].filter(Boolean).join(' · ');

  const roles = [];
  for (const e of d.events) {
    if (e.stream && e.stream !== 'env' && e.stream !== 'controller' &&
        !roles.includes(e.stream)) roles.push(e.stream);
  }
  const grid = document.getElementById('grid');
  grid.style.gridTemplateColumns =
    `76px minmax(200px, 320px) repeat(${roles.length}, minmax(280px, 460px))`;

  let h = '<div class="lanehead time" style="grid-row:1;grid-column:1">time</div>' +
    '<div class="lanehead" style="grid-row:1;grid-column:2">env / controller</div>';
  roles.forEach((r, i) =>
    h += `<div class="lanehead" style="grid-row:1;grid-column:${i + 3}">${esc(r)}</div>`);

  let row = 2, prev = null;
  for (const e of d.events) {
    const t = Date.parse(e.ts);
    if (prev != null && t - prev > GAP_MS) {
      h += `<div class="gap" style="grid-row:${row};grid-column:2 / -1">` +
        `··· +${Math.round((t - prev) / 1000)}s ···</div>`;
      row++;
    }
    prev = t;
    const col = (!e.stream || e.stream === 'env' || e.stream === 'controller')
      ? 2 : roles.indexOf(e.stream) + 3;
    h += `<div class="time" style="grid-row:${row};grid-column:1">${hhmmss(e.ts)}</div>`;
    h += `<div class="card ${cardClass(e)}" style="grid-row:${row};grid-column:${col}">` +
      `<span class="k">${esc(e.type)}</span> ${esc(e.text)}</div>`;
    row++;
  }
  grid.innerHTML = h;
}

async function tick() {
  let delay = 2000;
  try {
    const r = await fetch('/api/rollout/' + encodeURIComponent(rid));
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    clearError();
    render(d);
    if (d.status && d.status !== 'running') return;
  } catch (e) {
    showError(e);
    delay = 5000;
  }
  setTimeout(tick, delay);
}
tick();
</script>
""".replace("__CSS__", _CSS).replace("__HELPERS__", _JS_HELPERS)


class _Handler(BaseHTTPRequestHandler):
    runs_root: Path
    tasks_root: Path

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        url = urlparse(self.path)
        try:
            if url.path == "/":
                self._send(200, _PAGE.encode(), "text/html; charset=utf-8")
            elif re.fullmatch(r"/rollout/[^/]+", url.path):
                self._send(200, _ROLLOUT_PAGE.encode(), "text/html; charset=utf-8")
            elif re.fullmatch(r"/rollout/[^/]+/timeline", url.path):
                self._send(200, _TIMELINE_PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                q = parse_qs(url.query)
                experiment = (q.get("experiment") or [""])[0] or None
                self._json(build_state(self.runs_root, self.tasks_root, experiment))
            elif url.path.startswith("/api/rollout/"):
                rid = unquote(url.path[len("/api/rollout/"):])
                self._json(rollout_detail(self.runs_root, rid))
            else:
                self._send(404, b"not found", "text/plain")
        except KeyError:
            self._send(404, b"unknown rollout", "text/plain")
        except Exception as e:  # surface, don't kill the thread
            self._send(500, f"{type(e).__name__}: {e}".encode(), "text/plain")

    def _json(self, obj: dict) -> None:
        self._send(200, json.dumps(obj, default=str).encode(), "application/json")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # quiet: polling would flood the terminal
        pass


def serve(
    runs_root: str | Path,
    tasks_root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8710,
) -> None:
    handler = type(
        "Handler",
        (_Handler,),
        {"runs_root": Path(runs_root), "tasks_root": Path(tasks_root)},
    )
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"sentinel console on http://{host}:{port} (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
