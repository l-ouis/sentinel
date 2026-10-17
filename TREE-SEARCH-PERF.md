# Tree-search rollouts: making them fast on the Strix Halo

Working doc for a session on the 128 GB Strix Halo box. Goal: find out how fast a
pyramid-shaped rollout tree can run locally, and what to change to get there.

Everything below is **modeled, not measured**. The point of the session is to
replace these numbers with real ones.

---

## 1. What we're building

Snapshot the world before each graded model turn, sample `n` continuations from
each snapshot, recurse. Leaves get scored; node values are the mean over their
subtree; the advantage of a turn is `V(child) − V(parent)`.

Because siblings share an identical prefix, the branch at each node is a
**randomized assignment**, not an observation — sibling comparison gives a causal
estimate of what that turn did to the outcome, conditional on the prefix. That's
the whole reason for the shape.

Reference workload — `calendar-guard` at `n=8`, 3 graded turns:

| Level | Branches | Assistant turns run |
|---|---|---|
| turn 1 | 8 | 8 |
| turn 2 | 64 | 64 |
| turn 3 | 512 | 512 |

584 assistant turns, ~1,560 model calls, 512 leaves.

Token profile, measured from `runs/calendar-guard-20260820-231718-6ba81bf4`
(8 model calls, prompt ramping 1.6k → 7.0k, 4.6k output):

- **7.96M input tokens** naive
- **0.394M output tokens**
- **~0.50M input tokens** if prefixes are reused — see §2

---

## 2. The only number that really matters

The tree's 8M input tokens are almost entirely re-reads of shared prefixes.
Tokens the tree has *never seen* — model outputs, tool results, Devin's scripted
lines — total about **0.50M**. That's a **16× reduction in prefill work** if the
serving stack reuses KV across siblings, and nothing if it doesn't.

Modeled wall clock per tree:

| Stack | Decode | Prefill | Tree |
|---|---|---|---|
| Strix Halo, no prefix reuse | ~157 tok/s → 42 min | ~500 tok/s → 265 min | **~5.1 hrs** |
| Strix Halo, with prefix reuse | ~157 tok/s → 42 min | ~500 tok/s → 17 min | **~59 min** |
| Strix Halo, matured vLLM/SGLang | ~361 tok/s → 18 min | ~2,500 tok/s → 3 min | **~22 min** |
| 2× RTX 5090 (for scale) | ~4,048 tok/s → 1.6 min | ~24,000 tok/s → 0.3 min | ~2 min |

Assumptions: Q4 weights ~17 GB, MoE decode reading the full expert set per step
at batch, 60–75% bandwidth efficiency, batch 16.

**A slower engine that reuses prefixes beats a faster one that doesn't.** 16×
beats 2×. Do not optimize kernels before confirming cache reuse.

Holding the whole tree's KV resident costs **50 GB at fp16, 25 GB at q8_0**.
128 GB of unified memory means this fits with room to spare — that's the
architectural advantage of this box, and the thing 2× 5090 (64 GB) cannot do.

---

## 3. Probes, in priority order

### Probe 0 — Establish the baseline

Before anything: what is actually serving, and how fast?

```
# what's running
ps aux | grep -iE 'llama|vllm|ollama|sglang'
rocm-smi --showmeminfo vram 2>/dev/null; amdgpu_top -d 2>/dev/null | head -40
# llama.cpp
llama-bench -m <qwen3-30b-a3b-q4>.gguf -p 512,2048,6144 -n 128 -r 3
```

Record, for Qwen3-30B-A3B at Q4:
- single-stream decode tok/s
- prefill tok/s at 6k context (this is the number the whole tree hinges on)
- model load time and resident memory

### Probe 1 — Does prefix reuse work? *(the decision probe)*

The single measurement that separates a 1-hour tree from a 5-hour one.

Send one ~6k-token prompt, then 8 different short continuations that share that
exact prefix. `llama-server` returns a `timings` object per response — watch
`prompt_n` and `prompt_ms`.

```
llama-server -m <model>.gguf --host 0.0.0.0 --port 8080 \
  --ctx-size 32768 --parallel 4 --slot-save-path /var/tmp/llama-slots

# request 1: full 6k prefix
# requests 2..8: same prefix + different 20-token suffix, cache_prompt: true
```

**Success:** requests 2–8 report `prompt_n` ≈ only the new tokens, and
`prompt_ms` collapses by an order of magnitude.

**Failure:** `prompt_n` ≈ 6000 every time — the whole prefix is recomputed and
you are in the 5-hour column.

Confirm `cache_prompt` defaults on in this build; set it explicitly either way.

### Probe 2 — Slot save / restore

`cache_prompt` only reuses what is *currently resident in that slot*, so it
rewards depth-first, sibling-consecutive traversal and punishes anything else.
Slot save/restore lifts that restriction — it's KV-level fork.

```
POST /slots/{id}?action=save&filename=node-<path>.bin
POST /slots/{id}?action=restore&filename=node-<path>.bin
POST /slots/{id}?action=erase
```

Measure:
- save latency and file size at 6k context (expect ~634 MB at fp16, ~317 MB at q8_0)
- restore latency
- whether a restored slot actually skips prefill (check `prompt_n` again)

73 branch points × ~317 MB ≈ 23 GB — trivial against 1.2 TB free. At NVMe speeds
the whole tree's save/restore I/O should be a couple of minutes.

Verify the endpoint shapes against this llama.cpp build; the API moves.

### Probe 3 — Parallel slots and batch throughput

MoE decode gets *more* efficient with batching (the same expert reads amortize
over more sequences), up to the point where every expert is touched each step.
Find the knee.

```
--parallel 1,2,4,8,16  # measure aggregate decode tok/s at each
```

Record aggregate tok/s vs slot count, and where it stops scaling. That number
sets the harness's concurrency cap. Note each slot needs its own KV allocation —
this is where 128 GB pays off again.

### Probe 4 — KV cache quantization

```
--cache-type-k q8_0 --cache-type-v q8_0    # then q4_0
```

Measure three things, not two:
- memory per token (expect 96 KiB → 48 KiB → 24 KiB)
- throughput change
- **output quality change** — this one matters more than usual. KV quant
  perturbs outputs, and in a tree that perturbation is indistinguishable from
  the model's own decision variance. It inflates the noise floor the whole
  attribution rests on. If q8_0 measurably shifts outputs, take the memory hit.

### Probe 5 — How much of the 128 GB can the iGPU actually claim?

APU unified memory is not automatically all available to the GPU.

```
# check current GTT ceiling
dmesg | grep -i 'amdgpu.*\(gtt\|vram\)'
cat /sys/module/ttm/parameters/pages_limit 2>/dev/null
# kernel params if it's capped low:
#   amdgpu.gttsize=<MB>  ttm.pages_limit=<pages>
```

This gates Probe 8. If the iGPU can only claim ~half of system RAM, the bf16
reference model doesn't fit and that plan changes.

### Probe 6 — ROCm vs Vulkan backend

On RDNA3.5, llama.cpp's Vulkan backend is frequently *faster* than the HIP/ROCm
one. Build or install both and run Probe 0 against each. Cheap to test, and it
may be a free 1.5–2×.

### Probe 7 — vLLM / SGLang feasibility *(time-boxed, only if 1 and 2 fail)*

If llama.cpp gives no prefix reuse, the fallback is an engine with paged
attention. Cap this at a day.

- **SGLang / RadixAttention** is the better conceptual fit than vLLM — it is
  literally a radix tree over KV prefixes, which is the shape of this experiment.
  RDNA3 support is thinner than CDNA; check first.
- **vLLM** on gfx1151 needs `PYTORCH_ROCM_ARCH=gfx1151` builds of torch and vLLM,
  the Triton attention path (CK/AITER target CDNA), and Triton fused-MoE.
- gfx1151 has **no native FP8 matmul** — fp8 weights and fp8 KV are off the
  table. Plan on Q4 AWQ/GPTQ with dequant kernels.

Getting it to *run* is plausible in a day or two. Getting it *fast* is months of
kernel engineering and is explicitly out of scope — see §6.

### Probe 8 — bf16 reference model

61 GB of weights for bf16 Qwen3-30B-A3B. If it loads and still leaves KV room,
this box can anchor the quantization sweep (`scripted.py`'s stated motivation:
"is Q4 of this model as helpful as Q8") against an unquantized reference. No
64 GB dual-GPU rig can do that. Confirm it actually loads and measure its
decode rate — it will be roughly 3.5× slower than Q4 on bandwidth alone.

---

## 4. Harness-side work — independent of the serving stack

These are worth doing regardless of what the probes find.

- **Cache-aware traversal order.** Depth-first, siblings consecutive. Turns
  `cache_prompt` from useless into most of the 16×, with no engine changes.
  Currently the orchestrator has no concept of tree order at all.
- **Snapshot as a tuple, not a filesystem.** A branch point is: task module state
  (`ToolsConfig.module`), every participant's conversation history, journal
  offset, `TwoScaleClock` offset, each role's seeded RNG state, and — if using
  Probe 2 — the KV slot file. Miss the RNG state and siblings silently share
  draws they shouldn't. This wants to be one explicit `checkpoint()` on the
  Environment protocol, implemented per backend.
- **Journal lineage.** `Event` has `rollout_id` and gapless `seq` but no parent.
  Needs `parent_rollout_id` + `fork_seq`, and `_next_seq` must resume from the
  fork point rather than 0.
- **Judge repeats = 1 for tree runs.** `calendar-guard` specifies `--repeats 3`.
  At 512 leaves that's 13.8M input tokens — *more than the entire generation
  cost*, and it roughly triples the run. The 3 repeats exist to beat down judge
  variance on a single rollout; averaging 512 leaves already does that.
- **Non-uniform fan-out.** Turn 3 is 7.0M of the 7.96M input, because 512
  executions of the longest prompt dominates everything above it. `n=8/8/4`
  costs about half of `n=8/8/8` and barely degrades the estimates that matter —
  depth-3 nodes were the noisy ones anyway.
- **Quantized clock mode.** `TwoScaleClock` derives sim time from wall elapsed,
  deliberately. In a tree that means siblings drift apart in *simulated* time
  purely from generation-speed jitter — and for `calendar-guard` specifically,
  landing in a different simulated hour is outcome-changing, not cosmetic. Add a
  mode where each turn advances sim time by a fixed or replayed increment. Keep
  the realistic behaviour for headline runs.
- **Null-branch calibration.** Fan out `n=8` where the *identical* model turn is
  replayed and only the simuser and env vary. The spread in final scores is the
  noise floor. Every pivotality claim has to clear it. Cheap, and it's what makes
  "this turn is pivotal" a real statement rather than a ranking of noise.

---

## 5. Statistics to get right

- **Leaves are not independent samples.** 512 leaves from a shared trunk have far
  fewer effective samples than 512. `metrics.py` computing pass rates over leaves
  as if they were independent rollouts will produce confidence intervals that are
  badly too narrow. Use a clustered / nested-variance estimator.
- **Estimate quality is inverted.** The root's V averages 512 leaves; a depth-3
  node's V averages 8. So the noisiest estimates sit where the most comparisons
  happen (64 depth-3 nodes). Ranking turns by raw sibling variance will put noise
  at the top. Apply empirical-Bayes shrinkage across nodes at the same depth, and
  FDR control before calling anything pivotal.
- **Binary scores will hurt.** With 8 children, a pass/fail outcome gives node
  estimates on a 9-point grid and almost no signal. Use the graded rubric score
  from `judge.py` — here that's the difference between depth-3 resolving anything
  and not.
- **Headline metric.** By the law of total variance applied down the tree, total
  outcome variance decomposes into how much is decided at turn 1, turn 2, turn 3,
  and what's irreducible at the leaf. *"This task is decided at turn 2"* is a far
  more useful one-line summary of a task than a pass rate.

---

## 6. Explicitly not worth doing

- **Writing ROCm/WMMA attention kernels.** Closing 500 → 2,500 tok/s prefill is
  dedicated GPU kernel engineering with a profiler in hand, over months. The
  payoff is ~60 min → ~20 min per tree. The 16× from prefix reuse is available
  for a day of integration work; take that first and stop.
- **Buying GPUs.** 2× RTX 5090 would be ~30× faster and could not hold the bf16
  reference model at all. Renting an 80 GB card is ~$1–2 per tree if a specific
  run ever needs to be fast.
- **Naive BFS traversal.** Breadth-first with no cache awareness throws away the
  entire prefix-reuse win even on an engine that supports it.
- **fp8 anything on gfx1151.** No native FP8 matmul. Don't design around it.

---

## 7. Decision tree for the session

```
Probe 1: does cache_prompt reuse the prefix?
├── YES → implement DFS sibling-consecutive traversal in the orchestrator.
│         Expect ~1 hr/tree. Then Probe 2 to lift the ordering constraint,
│         Probe 3 to set concurrency, Probe 4 for memory headroom.
│         Stop there. Ship it.
└── NO  → Probe 2 (slot save/restore) is the fallback — it gives explicit fork
          even without implicit reuse.
          ├── works → same as above, ~1 hr/tree
          └── fails → Probe 7, time-boxed to one day. If SGLang/vLLM won't
                      build on gfx1151, accept ~5 hr trees (run them overnight)
                      or rent for the runs that need to be fast.
```

---

## 8. Numbers to bring back

Fill these in — they replace every modeled figure above.

| Measurement | Value |
|---|---|
| Decode tok/s, Q4, single stream | |
| Decode tok/s, Q4, aggregate at best `--parallel` | |
| Best `--parallel` before throughput plateaus | |
| Prefill tok/s at 6k context | |
| Prefill tok/s on a cache hit (Probe 1, requests 2–8) | |
| Slot save latency / file size at 6k | |
| Slot restore latency | |
| KV bytes/token at fp16 / q8_0 / q4_0 | |
| Max GTT the iGPU can claim | |
| bf16 30B: loads? decode tok/s? | |
| Vulkan vs ROCm backend delta | |
| **Projected wall clock for one 512-leaf tree** | |
