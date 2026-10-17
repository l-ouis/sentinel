# Sentinel

An evaluation framework for autonomous agents, built from scratch.

Status: pre-v1. Schemas and run data are disposable until v1.

## Why

Existing harnesses (Inspect AI, Harbor) assume the subject is one LLM, one
rollout fits in one process, and the core speaks chat messages. Sentinel is
built around three things those assumptions break:

- **Heterogeneous participants.** LLMs, scripted user-simulators, in-process
  bots, and humans all implement the same `Participant` protocol. A simulated
  user and an opposing agent are the same mechanism.
- **Multi-agent, multi-role tasks.** A task declares roles; a rollout binds a
  participant to each. Information and tool access can differ per role.
- **An append-only event journal per rollout.** Every scorer (verifier,
  check, judge, metric) is a journal consumer, so grading can be re-run
  offline without re-running an agent.

Full design notes are in [PLAN.md](PLAN.md).

## Layout

```
src/sentinel/        harness: run, orchestrate, verify, check, judge, web console
  participants/      chat_model (OpenAI-compatible), claude_print (Claude Code CLI), scripted
  envs/              podman (container shell), chat (tool-only, no filesystem)
  schemas/           pydantic models for task.toml, journal events, rubrics
tasks/<name>/        one directory per task (see below)
tests/               pytest suite; stub_openai.py fakes an endpoint
runs/                rollout journals, gitignored
```

A task directory:

```
task.toml            roles, environment, tools, verifier, checks, judge, limits
instruction/         per-role instructions, briefings, system prompts
environment/         Containerfile or init.sh for podman tasks
tools.py             optional harness-side custom tools (state the agent never sees)
users/*.toml         scripted user turns
verifier/            run.sh / grade.py and rubric.toml
```

## Scoring stack

Each layer is independent and lands as its own metric.

1. **Verifier.** Deterministic, inspects environment state, produces reward.
   Modes: `inline` (in the container), `module` (harness-side, reads the
   task's tool state), `none`.
2. **Checks.** Deterministic trajectory assertions over the journal:
   tool called, tool args match, said / did not say, max tool calls. They
   never touch reward.
3. **Judge.** LLM rubric scoring of the blinded transcript. The judge emits
   per-item scores and triggered penalties; the harness computes the total.
   `--repeats N` measures rubric variance.
4. **Metrics.** Tokens, cost, latency, pass^k, aggregated per experiment.

## Usage

```sh
uv sync
uv run sentinel validate tasks/*

# single-role task against a local OpenAI-compatible endpoint (default: LM Studio)
uv run sentinel run tasks/smoke-shell --model qwen2.5-coder

# multi-role, mixed participants
uv run sentinel run tasks/calendar-guard --orchestration relay-v3:e12 \
    --bind devin=script:users/devin.toml --bind assistant=claude:sonnet

# grade afterwards, offline
uv run sentinel check runs/<rollout>
uv run sentinel judge runs/<rollout> --model claude:sonnet --repeats 3

# compare models on one task
uv run sentinel experiment tasks/<task> --name <exp> --model a --model b

# browse results
uv run sentinel web
```

Participant specs: a bare name is a model on the endpoint, `claude:<model>`
runs through the Claude Code CLI, `script:<path>` is a scripted user.

Orchestrations: `solo-v1`, `turns-v1|v2[:r<N>]`, `relay-v2|v3[:e<N>]`. The
v2/v3 variants keep persistent per-role context across turns.

## Tasks

The tasks shipped here are placeholders and examples. They exist to exercise
the harness, not as a benchmark; expect them to be rewritten or dropped.

| Task | Shape |
|---|---|
| smoke-shell, smoke-pair, smoke-chat | harness smoke tests |
| hello-rewrite | single-role text task |
| transcribe-typst | container task with real tooling (whisper, typst) |
| relay-code, vault-crack | multi-agent, split secrets, per-role tool gating |
| calendar-negotiation | two agents across private calendars |
| calendar-guard | propensity: decline a slot without leaking why; judge-graded |
| notetaker-week | long-horizon simulated-user task |

## Tests

```sh
uv run pytest
```
