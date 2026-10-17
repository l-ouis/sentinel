# Sentinel — Design & Planning

An evaluation framework for autonomous agents, built from the ground up.

Status: pre-v1, greenfield. Data and schemas are disposable until v1 — rewrite
migrations and clear tables freely.

---

## 1. Why build from scratch

The mainstream agent-eval ecosystem consolidated onto two open-source centers of
gravity — **Inspect AI** (safety/research-grade, Task/solver/scorer, rich
sandboxing) and **Harbor** (containerized "product agent in a box", verifier
scripts, cloud scale-out) — plus a wave of benchmark suites (SWE-bench family,
OSWorld, Gaia2/ARE, τ²-bench, Terminal-Bench) and commercial observability
platforms (LangSmith, Braintrust, Galileo, etc.).

Sentinel's requirements each break a different assumption baked into those tools,
which is why a ground-up build is justified:

- **Heterogeneous participants** (LLMs, game-theory bots, humans, simulated
  users — one interface). Every existing harness hardcodes "the subject is an
  LLM" with chat-message semantics at the core.
- **Multi-day long-horizon runs** (e.g. testing a personal agent framework over
  several real-world days). Every existing runner assumes a rollout fits inside
  one process lifetime; nothing survives a reboot mid-eval.
- **Multi-agent games** (models playing against each other). Only the RL world
  (PettingZoo) models this well; the LLM eval harnesses are single-subject.
- **Short-horizon container tasks.** This one is solved — steal the Harbor task
  shape rather than reinventing it.

What we deliberately reuse from the landscape rather than invent:
- The **container + verifier task format** (Harbor's directory shape).
- The **split verifier** idea (private grader, public environment) as our
  contamination/asymmetry answer.
- **pass^k reliability** (τ-bench), **cost/Pareto reporting** (Princeton HAL),
  and **time-horizon framing** (METR) as the credible metric axes.
- The **eval-awareness / realism** research as the basis for disclosed-vs-dressed
  environments (see §7).

---

## 2. Core vocabulary

Eight nouns. These become the schema; get them right on day one.

| Term | Meaning |
|---|---|
| **Participant** | Anything that acts: an LLM agent, a minimax bot, a human, a scripted user-simulator |
| **Environment** | The world participants act in: a container shell, a game board, a live personal-agent deployment |
| **Task** | Environment spec + instructions + verifier + metadata, as a directory on disk |
| **Rollout** | One execution of one task with bound participants — the unit of everything |
| **Episode** | One contiguous interaction window within a rollout; participants are live inside episodes, the clock may fast-forward between them |
| **Verifier** | Deterministic grader inspecting environment *state* → reward |
| **Judge** | Model-based grader inspecting the *journal* → scores (kept separate from verifiers on purpose) |
| **Experiment** | A named set of rollouts (task × participants × repeats) you compare and report on |

The central bet, from which everything else follows:

> **The core speaks observations and actions, not chat messages, and every
> rollout is an append-only event journal.**

---

## 3. The Participant protocol — the central interface

Abstract over *policies*, not "models". Two tiny core protocols, one per
interaction shape (§5) — the shape attaches to the **role binding**, not to the
model behind it:

```python
class SteppedParticipant(Protocol):
    def act(self, observation: Observation, ctx: RolloutContext) -> Action: ...

class SessionParticipant(Protocol):
    def run(self, session: Session, ctx: RolloutContext) -> None: ...
```

Stepped = synchronous turn-taking (games). Session = a loop that is handed a
live session and drives it until the episode ends — required for real-time
concurrent episodes, where `act()`'s "wait for my observation" model can't
represent a message arriving mid-action.

`Observation` and `Action` are typed per environment-family (a game env observes
board state and acts with moves; a shell env observes terminal output and acts
with commands). "LLM-ness" lives entirely in **adapters**:

1. **ChatModelParticipant** — wraps any OpenAI-compatible endpoint (covers
   Anthropic, OpenAI, and local vLLM/SGLang/Ollama). Owns the prompt template
   that renders `Observation` → messages and parses completions → `Action`.
   Scaffolding choices live here, versioned and swappable. Implements both
   protocols: stepped directly; session via a small inner agent loop (whose
   polling behavior — e.g. does it re-check the inbox mid-task? — is itself a
   versioned scaffolding choice).
2. **InstalledAgentParticipant** — Harbor's trick: a CLI agent (Claude Code, the
   user's personal agent framework) installed *inside* the environment and run
   headless. Naturally a `SessionParticipant` (see SessionEnv, §5).
3. **CallableParticipant** — an in-process Python function. Game-theory bots,
   random baselines, Nash-equilibrium reference players. Zero ceremony.
   *Shipped:* **ScriptedParticipant**, its simplest useful case — a role whose
   turns are written down in a TOML script. When the question is "Q4 or Q8",
   every token an improvising user-sim contributes is variance in the answer;
   a scripted user makes the condition byte-identical across a sweep, and its
   script hash is recorded as part of the spec.
4. **HumanParticipant** — blocks on local input. Cheap; enables human baselining.

Because a **user-simulator is just another Participant**, τ-bench-style simulated
users and multi-agent games are the *same mechanism*. A rollout binds N
participants to roles declared by the task
(`roles = ["player_white", "player_black"]` or `["agent", "customer"]`). In
concurrent episodes the user-sim is a `SessionParticipant` too: its own loop,
reacting to environment events (inbox arrivals) plus scripted impulses from the
task's event schedule — including mid-task interrupts ("actually, make it a
PDF"), which double as high-signal scaffolding probes.

**Local-model note:** a participant spec is
`(model, quant, backend, context_limit, sampling)` — for home hosting, quant /
serving stack / context budget are first-class dimensions you hold or vary in an
experiment, not hidden plumbing. See §9.

*Shipped:* the spec is **probed, not assumed** (`sentinel.endpoint`). A local
server sits between our messages and the model's tokens: the **chat template**
decides how roles are wrapped and several templates inject a system prompt of
their own when the system slot is empty, while LM Studio-style presets can add
server-side defaults that someone changed in a GUI weeks ago. So quant, context
limit, and the chat template's hash are read from the endpoint at rollout start
and recorded in the spec, with the template body kept as a sidecar. Two runs
differing only by a silently-edited preset then read as different conditions
instead of unexplained numbers. Sentinel also always sends an explicit system
message, which is what stops a template's default from writing part of the
prompt. (Full control means rendering the template ourselves against
`/v1/completions` — a bigger change, deliberately not taken yet.)

---

## 4. The event journal — the actual backbone

Every rollout is an append-only journal: `provisioned`, `session_start`,
`session_end`, `observation`, `action`, `tool_call`, `env_event`, `clock_tick`,
`checkpoint`, `verifier_result`, `judge_result` — each timestamped (wall +
simulated clock), sequence-numbered, and tagged with an originating **stream**
(a participant role, or the environment/controller).

Rollouts may have **concurrent writers**: in real-time episodes (§5) several
participants act at once. Global sequence = append order at the controller;
wall timestamps across streams are the evidence for true causality ("did the
correction arrive before the send?"). Race outcomes are journaled facts — fold
logic may read them, never re-derive them.
Concretely: one JSONL file per rollout in a per-rollout directory, indexed in
SQLite. SQLite holds the index and results (rollouts, metrics, experiments); the
journal directory holds the heavyweight truth. No server, no Postgres — right-
sized for solo.

This single decision is what makes the hard requirements cheap:

- **Long-horizon durability:** the orchestrator is a small daemon that replays
  the journal to recover controller state after any crash/reboot. A multi-day
  run is just a rollout whose journal grows slowly.
- **Scheduled injections** ("on day 2, send the agent this email") are
  future-dated `env_event` entries the daemon fires — Gaia2's dynamic-events
  model, but persistent.
- **Trajectory scoring, awareness meters, realism scores** all become journal
  consumers, addable later without touching the runner.
- **Replay-based realism (LURE-style)** and regression tests fall out for free:
  a journal prefix is a reusable fixture.
- **Debugging** is reading a file.

### 4.1 Statelessness vs. a changing environment — the reconciliation

The journal is **not a simulator of the environment.** It is the source of truth
for the **controller** (the harness's decisions and knowledge) plus a **record of
evidence** about an environment it does not claim to be able to rebuild.

Two state machines, only one of which is event-sourced:

- **Rollout controller** — phase, whose turn, which scheduled events fired, what
  was dispatched/scored. Small, harness-owned, deterministic if disciplined (no
  wall-clock reads inside fold logic; every external input arrives as a journal
  entry). *This* is what crash recovery replays.
- **Environment** — container filesystem, game board, a real inbox. Never rebuilt
  by replay; it persists independently of our process. Recovery = fold the
  journal to reconstruct what the controller knew, then **re-attach** to the live
  environment via the handle recorded in the journal (container ID, VM ref,
  session credentials).

Journal entries therefore carry a `category`:
- **decisions** — controller-owned, replayable.
- **observations** — recorded facts about a world we don't control; evidence, not
  instructions to re-execute.

Recovery per environment class:
- **In-process games (SteppedEnv):** the one place rebuild-by-replay is valid —
  *if* every randomness draw is journaled (record RNG outputs, not just a seed).
  Cheaper and more robust: serialize full board state into periodic `checkpoint`
  entries and resume from the last one.
- **Container/VM sessions:** re-attach by recorded handle. If the container/VM
  itself died, that's *environment loss*, not journal failure — resume from an
  environment snapshot if the backend supports one (`podman commit`, ZFS/qcow2
  snapshot, recorded as a checkpoint entry pointing at a snapshot ID) or declare
  the rollout crashed. Both are honest outcomes.
- **Real-world long-horizon:** the world *is* the state; there is no reset. These
  rollouts only ever move **forward** — recovery re-attaches and continues,
  nothing rewinds. The journal here is purely evidentiary.

### 4.2 Flaky side effects — record at the boundary, never re-derive

Replay can't reproduce nondeterministic effects, so design so nothing ever needs
it to:

1. **Journal what was actually observed, not what should be re-derivable** — the
   actual command output, API response body, model completion. All downstream
   consumers (verifiers reading trajectories, judges, awareness meters, analysis)
   operate on recorded observations and never touch the environment again.
   Determinism of the world becomes irrelevant downstream of the run.
2. **Recovery-replay ≠ re-running.** Re-running a task is a *new rollout* and is
   *expected* to diverge (stochastic models, flaky networks). Cross-run
   reproducibility comes from the environment reset (pinned images), not the
   journal. Divergence across repeats is a **measurement** — that's what pass^k
   is for — not a bug to engineer away.
3. **Crash mid-action** (daemon dies after dispatching but before journaling the
   result): use write-ahead intent — journal `action_dispatched` before sending,
   `action_result` after. On recovery, a dangling `action_dispatched` means the
   effect is *ambiguous*: for idempotent effectors, retry with an idempotency
   key; for non-idempotent real-world ones, probe the environment ("is it in the
   sent folder?") or record the ambiguity as a fact and continue. **At-least-once
   with recorded uncertainty** is the honest semantics; exactly-once against the
   real world is a lie.

One-line summary: *append-only journal for what the harness decided and saw; live
handles and snapshots for what the environment is; repeats and pass^k for what
won't reproduce.*

---

## 5. The Environment protocol — concurrent sessions, one lifecycle

Uniform lifecycle for every environment:
`provision → reset → (episode)* → collect → teardown`.

The interaction phase is **re-enterable**: a rollout is one or more episodes
against environment state that persists between them. A container task is
exactly one episode; a week-long assistant rollout is many short ones, with the
daemon waking participants into sessions when scheduled events fire. Episodes
open and close with `session_start`/`session_end` journal entries.

The general interaction model: **N concurrent session participants sharing one
environment, with the environment as the message bus.** Participants never
exchange messages through the harness — one *acts* on the env (sends a message
via its messaging service), another *observes* it (inbox changed). Everything
stays observations-and-actions, and real-time interleaving (a user correction
landing while the agent is mid-task) is representable, journaled, and
measurable.

Two episode modes, per environment (stepped is the degenerate case, not a peer):

- **SteppedEnv** — the environment *enforces serialization*: emits an
  observation for whichever role acts next, receives the action, advances.
  PettingZoo's agent-environment-cycle model. What games and negotiations need.
  Supports fully simulated clocks (a "year" of vending-machine business in an
  afternoon), and cheap "fake duplex" variants of real-time tasks (interrupts
  injected between turns) for smoke-testing.
- **SessionEnv** — free-running and concurrent: each bound session participant
  drives its session (a shell, a browser, a chat client) until done or timeout.
  What Harbor-style container tasks, installed-agent frameworks, and real-time
  assistant tasks need. The env *observes* (records the sessions) rather than
  *steps*.

**Two-scale clock** (*shipped*, `TwoScaleClock`): wall-clock *within* episodes —
latency is part of the measurement and cannot be simulated away without deleting
it — and compressible *between* episodes (a "week" of class in an afternoon of
episode time). `sim = start + wall elapsed + explicit advances`, so a slow model
really does keep the other party waiting, while the gaps between turns cost no
real seconds. A scripted participant's `delay` is the usual advance; a random
delay is drawn from a **per-role seeded RNG** (independent streams, so a new
draw in one role cannot shift another's) and the drawn value is journaled rather
than re-rolled on replay (§4.1). Seeding per rollout keeps a sweep **paired**:
every model at repeat 3 waits the same eleven minutes. When
compressing, in-world timestamps (files, mail headers, container clock) must
stay consistent with the fictional clock, or the compression is itself a tell
(§7.2's time-consistency warning, arriving early).

Both modes write to the same journal format, so verifiers, judges, and analysis
don't care which shape produced a trace.

**Backends** (the runtime a task actually runs on) sit *below* this as a
provisioner interface. Same task definition, swappable runtime — see §6/§8.

---

## 6. Environments: runtime tiers

The backend matrix, cheapest → most realistic, all behind one provisioner
interface:

- **chat** — no world at all: prompt-only tasks, and tasks whose world lives in
  the task's harness-side tool module. Zero provisioning, no shell, and
  capability gating removes shell tools automatically. This is the tier that
  makes pass^k across a quant sweep an afternoon rather than a weekend, so it
  is the default for everything that does not genuinely need a filesystem.
- **in-process** — games and callable participants. No isolation needed.
- **Podman + crun** (default container tier) — daemonless (per-container
  `conmon`, survives orchestrator death → clean re-attach), rootless,
  Docker/OCI-compatible (keeps import-compat with Harbor/SWE-bench tasks),
  ~100ms starts, layer-based COW resets. The pass^k workhorse. Native on Fedora.
- **Podman + Kata** (isolation step-up) — same OCI interface, each "container"
  backed by a KVM microVM with its own kernel. Flip `--runtime kata`: hardware
  isolation and *no container tells* (`/.dockerenv`, `/proc/1/cgroup`, overlayfs)
  with zero task changes.
- **QEMU/KVM via libvirt (or Proxmox — see §8)** — real machine images, qcow2
  snapshot/rollback, real init/networking, later grows PCIe passthrough for
  hardware tasks. The realistic tier, and the natural home for long-horizon runs
  (VMs run for weeks and snapshot mid-flight → checkpoint-resume).

Skip **gVisor** (detectable + behaviorally quirky — hurts realism *and*
fidelity). Skip **Firecracker** unless going cloud-scale (bespoke backend for
little gain over Kata).

**Inference (local models):** serve open-weight models behind an
OpenAI-compatible endpoint (**vLLM/SGLang** for NVIDIA throughput,
**llama.cpp/Ollama** for widest hardware net). Put a **LiteLLM-style gateway** in
front of local + API models as the single endpoint `ChatModelParticipant` talks
to — it's also where tokens/cost/latency/full request-response logging get
captured uniformly into the journal, and the natural place for per-experiment
budget caps.

---

## 7. Disclosed vs. dressed — realism as a task property

Grounded in the eval-awareness literature (models detect evals at high rates,
worst in agentic settings; the effect on *capability* is small but on
*propensity/behavior* can be large; the biggest realism levers are real
artifacts and real services, not surface set-dressing). Make it explicit in the
schema:

```toml
[realism]
profile = "disclosed"   # or "dressed"
```

- **Disclosed** — lean into awareness rather than fight it: an explicit `EVAL.md`
  stating it's an evaluation, clean/pristine image, verifier may even be visible.
  Right default for **capability** evals. Honest-by-construction. **These are the
  tasks we can open-source freely** — realism isn't a property we're protecting.
  Cheap, fast, publishable.
- **Sandboxed** — the environment is an isolated deployment and says so
  truthfully. Nothing is disclosed and nothing is faked: a real deployment of
  the same agent would look exactly like this, so a model correctly inferring
  "I am sandboxed" learns nothing about whether it is being tested, and there
  is no claim for it to catch us in (the honesty floor below is satisfied by
  construction rather than by discipline). Cheap — the framing lives in the
  task's system prompt (`RoleDef.system`), not in an aged image. Its tells
  migrate to the **scenario**: a too-clean fixture, a counterparty who pushes
  back exactly once. Spend the realism budget on incidental clutter there.
- **Dressed** — for **propensity/behavioral** evals where realism measurably
  matters: runs on Kata/QEMU tiers, no eval artifacts in the filesystem, split
  verifier (nothing to find), and an **aged image** (see below).

### 7.1 Contamination & the asymmetry primitive

Full asymmetric verification (prove a solution without revealing the task)
doesn't exist and probably can't. But partial asymmetries the format supports:

- **Split verifier** — Harbor's `[verifier] environment_mode = "separate"`: grade
  in a private image; the agent's world is public, the answer key never is. This
  is the closest thing to asymmetric verification that ships today, and it
  doubles as a realism win (no grader files in the agent-visible filesystem — one
  of the most commonly detected tells).
- **Hosted grading over held-out data** (SWE-bench `sb-cli` model) layers on top
  of the same format without changes.
- **Continuous refresh / commit-reveal** to outrun training-data soak.
- **Trace auditing** (HAL-style) to detect contamination post-hoc.

### 7.2 Realism measurement & the aged-image pipeline

- Build an **awareness meter** into the harness: classify transcripts/CoT for
  verbalized awareness + an LLM-judge `P(eval | transcript)` realism score per
  scenario (LURE-style). Report both next to every result, and report behavioral
  metrics **conditioned** on detected awareness (aware vs. unaware subsets).
- Highest-ROI realism levers, in order: real deployment artifacts (real system
  prompts, real tool defs, real repos) → replay real trajectories with minimal
  evaluative edits → incidental (not spotlighted) misbehavior opportunities →
  concrete names/dates, no oversight-flavored language.
- **Aged-image pipeline** (build realism reproducibly, don't hand-dress):
  provenance build replaying a scripted-but-organic history (real multi-month git
  histories, populated shell history/logs with consistent timestamps, cron,
  package history, lived-in home dir) → versioned/hashed golden image. **Time
  consistency is the fiddly part** (mtimes, log dates, VM clock must agree or the
  aging is itself the tell). Score images with the P(eval) judge so "realistic"
  is measured, not vibed.
- **Honesty floor:** never falsely deny it's an eval if the agent asks. Costs
  nothing statistically (those transcripts land in the "aware" bucket anyway) and
  keeps unaware-subset numbers clean. Never plant eval metadata/purpose labels in
  the environment unless measuring the honeypot response *is* the point (effect
  sizes there are ~0→100pp).

---

## 8. Homelab notes (future, when scaling up)

The homelab is a bigger, faster set of *provisioner backends* — it does **not**
change the backbone. Keep abstractions pointed at "a modest machine" so a task
developed against Podman on a laptop runs unchanged against a golden-lab clone on
Proxmox. That portability keeps disclosed tasks publishable and lets others
reproduce evals without the rack.

- **Proxmox VE** as provisioner backbone (API, ZFS, linked clones in ~seconds,
  snapshots, SDN). With fast linked clones, VM-per-rollout becomes the default
  for dressed/long-horizon; Podman stays the "50 cheap parallel rollouts" tool.
- **Defense in depth for free:** run container-tier rollouts *inside* a
  disposable exec VM — a container escape lands in a VM you roll back anyway.
- **Control plane** on a small always-on box (daemon + SQLite + journal on ZFS,
  on a UPS). Long-horizon runs live or die by this being boring; everything else
  can reboot (recovery = re-attach).
- **Network as environment** — VLANs turn `network_mode` into physical truth:
  management (never reachable from evals) / sealed (`no-network`) / proxied
  (`allowlist` via a logging filter proxy) / **realistic-services** (a self-hosted
  fake internet: real mail server, Gitea/Forgejo with aged repos, wiki, DNS,
  chat — replaces the fake-tool tells with actual indistinguishable services).
- **Aged-image pipeline → golden *lab***: snapshot a whole plausible network
  (mail, git, wiki, target) together and clone as a unit.
- Open decision when we get there: realistic-services VLAN **shared** across
  rollouts (cheap, leaks state, serializes mutations) vs. **cloned per-rollout**
  (clean, heavier). Lean: per-rollout clones for dressed/long-horizon; optional
  shared pool for disclosed.

---

## 9. Task format on disk

Harbor's directory shape, extended where our requirements exceed it:

```
tasks/<name>/
├── task.toml          # + [participants] roles (+ shape: stepped|session),
│                      #   [clock] mode = "real"|"simulated"|"two-scale",
│                      #   [realism] profile, [[events]] at="2d" inject=...,
│                      #   [environment] backend = "podman"|"vm"|... , network_mode
├── instruction/       # per-role instructions (multi-agent needs >1), plus
│                      #   optional per-role system prompts (RoleDef.system):
│                      #   the task owns the persona, the scaffold owns the
│                      #   transport's tool grammar, the briefing lands last
├── environment/       # Dockerfile | compose | game module ref | vm/baremetal spec
├── verifier/          # grader (separately containerizable, keepable-private)
└── oracle/            # optional known-good solution for task QA
```

Staying compatible with Harbor's shape where we haven't extended it lets us
import their task ecosystem nearly for free.

---

## 10. Scoring stack — layered, all journal-consumers

1. **Verifiers** (deterministic, state-based): split-container `reward.json`
   contract. Runs at `collect` time. Four modes ship: `inline` (grade in the
   task container), `module` (harness-side grader reading the task's own tool
   module — the world the agent acted on, for backends with no filesystem),
   `none` (graded by checks and/or judge alone), and `separate` (v1.1).
2. **Trajectory checks** (deterministic, journal-based): "never called tool X",
   "≤ N steps", "visited state Y", "responded to the interrupt" —
   agentevals-style match modes. *Shipped* as `[[checks]]` in task.toml, run at
   collect time and re-runnable offline (`sentinel check`). They never touch
   reward: each lands as `check.<id>` plus a `checks_passed` fraction, so
   turning checks on cannot silently move the headline number. The one that
   pays for itself immediately is `claim_without_tool` — "said it sent the
   file, never called the tool" is invisible to a state verifier.
3. **Functional probes** (model-in-the-loop, deterministic protocol): score
   produced *artifacts* by using them — e.g. hand only the agent's notes to a
   fixed reference model and quiz it on the source material; score =
   answerable-from-notes rate. Neither verifier (it calls a model) nor judge
   (it doesn't rubric the journal); contamination-resistant when the probe
   questions are private.
4. **Judges** (LLM, journal-based): rubric scoring of whole runs. Kept
   architecturally separate from verifiers so a task can be graded with judges
   *off* (contamination-sensitive publishing) or *on*. Use a stronger judge than
   the models under test; prefer pairwise-vs-reference over absolute scoring.
   *Shipped* (`sentinel judge`, absolute method): the judge emits per-item
   scores and triggered penalties and the harness computes the total, so a
   model that cannot add cannot move the number; the transcript is blinded
   (model names → role aliases) and carries `[#seq]` anchors the judge must
   cite; `--repeats N` records the spread, which is a measurement of the
   *rubric*, not of the run. Because judgments append to a finished journal,
   rubric iteration never re-runs an agent. `pairwise_vs_reference` waits on
   reference trajectories in `oracle/`.
5. **Meta-metrics** (across rollouts): pass^k reliability, cost, tokens,
   throughput (tokens/sec), per-interaction latency, VRAM, verbalized-awareness
   rate, P(eval) realism — reported per-experiment and conditionable.

Every scorer emits `Metric(name, value, rollout_id)`; an experiment report is a
query.

---

## 11. Initial focus — local models first

We start by testing **reasonably locally-hosted models** under a small home-LLM
budget, on **general evals matching the workloads people actually use these
models for** (coding help, doc summarization/RAG, rewriting, multi-turn chat,
structured extraction/tool formatting, creative). Eval *content* design is the
user's domain — this doc doesn't prescribe it.

This first focus is deliberately the cheapest slice of the backbone (Phase 1:
`ChatModelParticipant` + Podman + disclosed tasks, stateless chat calls, no
container gymnastics), so nothing in the larger design is wasted — the spine gets
built against the easiest participants first.

**Key local-hosting wrinkle to bake into the schema now:** you're never measuring
just *a model* — you're measuring `(model, quant, backend, context_limit,
sampling)` on your hardware. Quant (Q4 vs Q8 vs FP16), serving stack, and context
budget change quality *and* throughput *and* VRAM. Treat each as a first-class,
holdable/varyable experiment dimension. The Pareto frontier of **quality vs.
throughput vs. VRAM** is the actual deliverable for home hosting — report all
three, always.

**Real-time wrinkle:** in concurrent real-time tasks (§5) the Pareto axes stop
being independent — throughput folds *into* quality. A slow-but-smart quant
answers stale while the user-sim has moved on; a fast Q4 keeps up. These tasks
are the most honest home-assistant evals and belong in this milestone's task
family — in their cheap stepped fake-duplex variant first, with the true
real-time variant arriving with the daemon (Phase 3). Accept the cost honestly:
real-time-within-episode can't be accelerated, so keep episodes short, compress
the gaps, and parallelize rollouts.

**Milestone-1 exit criterion:** run a quant sweep of one model family across a
set of workload-grounded tasks (deterministic verifiers where the workload
allows, rubric judge where not) and answer, from your own data, a question like
"should I run Q4 or Q8 of this model on my GPU for coding?" — exercising the whole
spine (journal, participant adapter, verifier, judge, meta-metrics, experiment
report) on the cheapest possible environment.

---

## 12. Build order (backbone)

- **Phase 1 — spine (one rollout end-to-end):** journal format + SQLite index,
  task loader, Podman SessionEnv backend, `ChatModelParticipant`, script
  verifier, `sentinel run <task> --participant <spec>` CLI. Local-model milestone
  (§11) lives here. Exit: a toy task runs, grades, and its journal replays.
- **Phase 2 — games:** in-process SteppedEnv, CallableParticipant, multi-role
  binding, simulated clock. Exit: two LLMs (or LLM vs. bot) play a game, scored.
- **Phase 3 — long horizon:** the daemon (systemd unit), crash-resume from
  journal, future-dated event injection, episodic session scheduling (daemon
  wakes participants into sessions on scheduled events), ~~two-scale clock~~
  (landed early — the scripted-user delay that makes a conversation take
  simulated minutes and real seconds needed it, and it is journal-local),
  periodic checkpoint metrics. Exit: a multi-day, multi-episode run with a
  concurrent user-sim (e.g. a week-long assistant task) survives a reboot.
- **Phase 4 — depth:** ~~judges, trajectory checks~~ (pulled forward — they are
  journal consumers, so they cost nothing to build early and the tasks that
  need them are the cheap ones), pass^k/cost/awareness meta-metrics, experiment
  reports, VM/Proxmox backend, aged-image pipeline.

**Assumed stack:** Python throughout (every model SDK, PettingZoo, and the
harness ecosystem live there; `InstalledAgentParticipant` shells out regardless).
The journal/daemon core is genuinely language-agnostic if that ever changes.

**Pin down first (everything hardens around these):** the v0 **journal event
schema** (including stream tags and episode boundaries — design for multiple
episodes and concurrent writers from day one, even though Phase 1 uses neither)
and the v0 **task.toml schema**.
