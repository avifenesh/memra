# Serving exactness

The numeric contract of `memra-server`: which streams are byte-identical, which differences are
documented near-tie classes, and the gate and receipt behind each rule. It is for an operator who
compares outputs across runs and configurations, and for a contributor who changes a prime, decode
or cache path. The operator guide is [SERVING.md](../SERVING.md); gate commands are in
[TESTING.md](../TESTING.md); cache behavior is in [prompt-cache.md](prompt-cache.md).

## Terms

| Term | Meaning |
|---|---|
| numeric program | The kernels and reduction orders that produce a row's logits. Two correct programs can differ in the last bits of a float. |
| cold twin | The same request on a fresh server: no cache hit, no pool resume, no peers. |
| near tie | A step whose top-1 and top-2 logits are close enough that a last-bit difference flips the greedy argmax. Everything after the flipped token follows the new prefix. |
| prime grid | Multiples of `gdn_chunk_size()`: 32 tokens by default (`MEMRA_GDN_CHUNK`, clamped to 32..128 in steps of 32), the fold grid of the chunked GDN prefill scan. |
| `PRIME_MIN_T` | 16. The smallest call the batched prime runs. A shorter prompt or tail goes token by token through `decode_step`. |
| recurrent cache | A cache that carries GDN conv and SSM state (the Qwen hybrids and every DFlash target). An attention-only cache carries none. |

## The isolation contract

Greedy serving is **isolated-identical** under concurrent load at the defaults: a request's output
tokens are byte-identical whether it decodes alone, joins a batch mid-stream, or shares the tick
with peers at any other depth. Two properties carry it:

- Every decode width runs one numeric program ([below](#one-numeric-program-per-request)).
- A co-resident session changes no bit of another session's logits. Batched attention derives each
  session's split partition from its own KV length (the one-partition law): a batch either shares
  one split rung or every row takes the per-sequence loop. A peer at any other depth, across a
  `fa_split_keys` rung boundary, at B=2..8 over 300 steps, changes zero bits
  ([`research/iso-gap-20260807/`](../../research/iso-gap-20260807/)).

Sampled draws are batch-invariant too: device draws at B=N equal the same rows' draws at B=1.

The staggered-depth divergence once measured at byte 1347 of a 768-token request
([`research/spec-gate-20260806/`](../../research/spec-gate-20260806/), `logs/exact/`) was the
solo-to-batched program flip at the co-residence boundary, not batch composition. The one-class
decode default removed that flip.

Gates:

- `decode-batch-gate` config mode: B=1 against B=N rows bit-identical (gate1), batchmates do not
  change a stream (gate2), device argmax, sampled draws and lean logits match (gate3).
- fast-gate `isogap` / `isogapc`: solo against co-resident logits across a split-rung boundary
  placed per rig; the canary feeds the peer a wrong token and must be caught.
- `tools/request-fault-gate.py`: peers of an injected per-request fault finish byte-equal to
  control.

Commands and where each runs: [TESTING.md](../TESTING.md#kernel-and-numeric-programs) and
[TESTING.md](../TESTING.md#serving).

## One numeric program per request

The rule: a code path that can produce tokens for the same request under two numeric programs is a
defect, unless the transition is forbidden or proven bit-identical. The inventory of program pairs
that applied it is [`research/twoprog-20260813/RESULTS.md`](../../research/twoprog-20260813/RESULTS.md).

**Decode.** B=1 and B>=2 run the same generic batched body, so a peer arriving or leaving never
changes a session's program ([ONE-CLASS-DECODE-DEFAULT](../decisions/ONE-CLASS-DECODE-DEFAULT.md)).
The CUDA-graph session serve door is removed. The m=1 fused eager trunk is an opt-in for
fixed-solo measurement: `MEMRA_SERVE_B1FAST=1` sends a solo tick through it. Use it only when
admission guarantees the request cannot gain a peer. It stays off even when set for every
GatedDeltaNet plan, Gemma 4, the sliding-gated MoE (Step) program, SwiGLU-OAI trunks and the
exact-verify scope (`b1_fast_plan_eligible`), and on an unsplit trunk for penalized rows. Under a
pipeline split it takes a stage-scoped eager path. The one-class default gives up
the fast path's solo win, measured at +8.33% (Qwen3.5-9B NVFP4) and +5.19% (Qwen3.6-27B NVFP4)
decode-only at c=1 on the RTX 5090 Laptop, N=5, 2026-08-05
([`research/servepath-p2-20260805/`](../../research/servepath-p2-20260805/)). Step-3.7-Flash
keeps a B=1 entry into the same batched kernels with two device copies removed and no arithmetic
changed: +4.4% at c=1, byte-identical
([`research/eagerpar-20260810/`](../../research/eagerpar-20260810/)).

**Pipeline split.** The split adds zero deviation: `decode-batch-gate --mode pp` compares every f32
logit of every row with the split off and on, 0 differing bits on seven configs (both placement
orders, an uneven cut, four stages on two cards, the 64-layer 27B, B=12 and 16)
([`research/pp2-batch-20260806/`](../../research/pp2-batch-20260806/)). Step-3.7-Flash PP-2
decodes in one numeric class at every batch width, and a PP-N B=1 tick fails closed rather than
fall back to the eager class ([`research/p0iso-20260810/`](../../research/p0iso-20260810/),
[`research/b1fix-20260810/`](../../research/b1fix-20260810/)).

**Speculative and plain.** A greedy speculative stream equals plain greedy decode in the class that
model's plain decode is served in
([the oracle](../TESTING.md#the-oracle-a-speculative-route-is-exact-against-memra1146)). A session
demoted to plain under load writes the bytes it would have written batched from the start (the
mode-switch gate, `research/spec-gate-20260806/exactness.py`). Route behavior is in
[speculative.md](speculative.md).

**Cross-request prefill.** The concat prime does not move a row. The prefill MoE router and
shared-expert gate run decode's m-invariant kernels, and the 8x8 batched router twin
(`MEMRA_ROUTER_BATCH`, default on) is bit-identical per row
([`research/concat-prime-exact-20260802/`](../../research/concat-prime-exact-20260802/),
[`research/fast-router-20260802/`](../../research/fast-router-20260802/)). Before that fix a MoE
request's expert selection changed with its co-arrivals. A continuation chunk (the second and later
chunks of a long cold prompt) joins a concat prime only when every trunk operation of the plan has a
carried-prime implementation (the `carried_prime` column of `op_registry`, manifest
`CARRIED_PRIME`). Routed-MoE and Gemma plans prime continuation chunks alone: carried MoE batches
selected EOS at token 26 in a Qwen3.6-35B-A3B mixed c=4 cell where serial priming and one whole
fresh batch both reached 60/60 ([`research/coldfix-20260812/`](../../research/coldfix-20260812/)).
Gates: `tools/prime-batch-exact-gate.sh`, `tools/prime-tick-exact-gate.sh`.

**The sub-floor tail.** A prompt under `PRIME_MIN_T` primes token by token, which is its one
program. A longer prompt never sends a 1..15-token tail through `decode_step`: the prime take folds
such a remainder into the call before it (`prefill_tick_take`, `batch_row_take`), and every
boundary serve chooses steps down one grid unit to leave at least 16 tokens after it
(`grid_align_boundary_within`). On dense GDN trunks a cold prime with one stop for a capture or
checkpoint runs as one walk with the capture tapped inside it
([PRIME-TAIL-FOLD](../decisions/PRIME-TAIL-FOLD.md)).

## The isolation law: non-tools traffic bypasses the tools renderer

A chat request with no `tools`, the default think mode, no `reasoning_effort`, no tool turns and no
replayed reasoning renders through the legacy `apply_chat_template` path and skips the tool-call
parser (`plain_chat_render_path`). Its stream is byte-identical to a build without the tools
surface. Tools traffic is generation-identical for the identical rendered prompt, and the
tools-capable renderer reproduces the legacy bytes on plain requests
(`tools_renderer_matches_legacy_when_plain`). One carve-out: a template carrying the Qwen3.8
reasoning-effort ladder always renders through the tools-capable renderer, because only that
renderer injects the template's default effort on an unset request. Without it the same unset
request rendered two ways, depending on whether a `tools` array came along. Receipts:
[`research/serve-tools-20260802/`](../../research/serve-tools-20260802/),
[`research/integrate-cache-20260802/`](../../research/integrate-cache-20260802/). Request shapes
are in [API-SURFACES.md](../API-SURFACES.md).

## The exact-16 decode chunk tier

The batched tick decodes sessions in waves. With `MEMRA_DECODE_BATCH_CAP` unset, the wave cap per
model is (`chunk_cap_for`):

| Model class | Wave cap |
|---|---|
| exact-16 tier, no pipeline split | 32 when the boot wave probe finds one 32-row wave faster than two 16-row waves; else 16 (also when the probe cannot run or `MEMRA_MAX_SESSIONS` <= 16) |
| exact-16 tier under a pipeline split | 16 |
| Step (sliding-gated MoE), Gemma 4 | 8; the variable narrows only |
| GLM-5.3 hyper trunks | 15 (`PRIME_MIN_T - 1`, the shared-expert decode-exact knee); narrows only |
| everything else | 8 |

An explicit `MEMRA_DECODE_BATCH_CAP` (1..32) is a measurement door and skips the probe.

A model is in the tier (`decode_batch_exact16_ok`) when every matmul of the batched step has a
kernel class that is bit-identical per (token, row) to the m=1 launch at 9..16 rows: Q4_0, Q4_K,
Q5_K, Q6_K, Q8_0 (base layout; the q8rp mirror is not required), NVFP4, FP8 E4M3 and block-128 FP8.
Float matmuls, MLA and KDA mixers, routed-MoE FFNs and the Gemma 4 and SwiGLU-OAI programs refuse
it. `MEMRA_EXACT16_WHY=1` names the first refusing tensor. `MEMRA_EXACT16_MOE=1` admits MoE FFNs as
a measurement door: byte-safe, but slower than two 8-row waves on the checkpoints measured.

The tier is a scope, not only a width. Outside it the m>=16 GEMM/MMQ arms and the m=9..15 dp4a tail
shift logits at step 0 (maxdiff about 0.13 to 0.23), so a wide wave without the scope is another
program. A 17..32-row wave is per-row bit-identical to two 16-row waves: NVFP4 runs 32-column
kernels and other classes run as two 16-column halves. An exact tensor-core form of the 32-row
NVFP4 tier (`nvfp4_mmvq_tc`, bit-identical to the dp4a tier) is picked by the same probe;
`MEMRA_MMVQ_TC=0/1` forces it. Both choices are speed only. Pooled speculative verify stays at 16
rows.

Receipts: [`research/batched-tick-inc3-20260801/`](../../research/batched-tick-inc3-20260801/)
(gate2 bit-checked at chunk 12 and 16, steps 32 and 160; +12% aggregate over chunk 8 at 32 sessions
on the RTX 5090 Laptop), [`research/batch-spec-20261101/`](../../research/batch-spec-20261101/)
(`GB10-PLAIN-DECODE.md`, `LEVER-LEDGER.md`: one 32-row wave 3.8% faster per tick on GB10 at depth
2048, the tensor-core form 13.8% faster, tokens identical).

## First-token cross-config drift (batched prime)

Serving primes a prompt with batched prefill GEMMs (`prime_cache`). The tokenwise oracle stream
(`decode_step`, m=1) is another program by design. On a near-tie prompt the first generated token
can differ between the two, and the rest follows the new prefix. This is a documented class, not a
wrong path.

Measured on six models and 144 prompts, 2026-08-02
([`research/prime-gate-coverage-20260802/`](../../research/prime-gate-coverage-20260802/)):

- 10 of 144 first tokens flipped, each at a tokenwise top-1/top-2 margin of 0.70 or less.
- The batched prime was bit-deterministic. `forward_last` sided with it in 8 of the 10 flips, so
  the tokenwise program is usually the outlier.
- Dense Q8_0 models flipped 0 of 48. One Gemma prefill row moved between processes (cuBLASLt
  algorithm selection) and was bit-deterministic within a process.
- A flip can be visible: a Qwen3.6-35B pp512 probe greedily emitted a newline and EOS at 2 tokens
  where the tokenwise stream wrote 128.

`MEMRA_PRIME_TOKENWISE=1` pins the oracle stream at prefill cost. The run-gen batched-prime check
and `prime-gate` bound the class: structured divergence fails, a near-tie flip within
`MEMRA_PRIME_GATE_MARGIN` (default 1.0) reports `FLIP-NEARTIE`.

A cache hit is not a new instance of this class. An entry stores the bytes of whichever prime ran,
so a hit is bit-identical to the run that computed the prefix (16/16 partial-prefix and 16/16
full-prefix cached against fresh,
[`research/prompt-cache-20260802/`](../../research/prompt-cache-20260802/)). Comparing a hit with a
fresh stream primed in another configuration inherits this class.

## Chunked prefill is split-stable

Prefill is byte-identical across `MEMRA_PRIME_CHUNK` values, with no door. Every chunk, the first
included, quantizes its K/V into the cache and attends through the quantized-cache kernel
(`full_attn_prime_fa_dispatch`, quantize-then-attend), so the chunk size cannot decide where a
precision edge falls. `MEMRA_PRIME_CHUNK` is a memory and fairness knob.

Until 2026-08-05 chunk 0 attended over f32 K/V and later chunks over the quantized cache, so output
moved with the chunk size from the first chunk boundary; the prefill GEMM itself is m-invariant
([`research/chunk-invariance-20260805/VERDICT.md`](../../research/chunk-invariance-20260805/VERDICT.md),
[`research/chunkinv-flip-20260805/`](../../research/chunkinv-flip-20260805/)).

The property is per architecture, because an attention family can re-enter the class through its
own prime program:

- **Step-3.7-Flash SWA prefill.** The SWA arm keys on the request's end position
  (`seq_end = cache.pos + t + queued_after`), not the chunk's or the call's, and the windowed FA
  view offset aligns down to the 32-key tile. Neither the split inside a call nor the tick
  segmentation across calls moves a row
  ([`research/step35-chunkfix-20260807/`](../../research/step35-chunkfix-20260807/),
  [`research/tick-seg-20260807/`](../../research/tick-seg-20260807/),
  [`research/pp-prefill-20260807/`](../../research/pp-prefill-20260807/)).
- **The generic chunked prime** declares chunk and tick invariance per operation (the
  `chunked_prime` column of `op_registry`, manifest `CHUNKED_PRIME`). A plan with an operation
  outside it primes monolithically. The Gemma family consults the column today, and Gemma 4 SWA
  prefill runs one windowed program at every length
  ([GEMMA-FA-ONE-PROGRAM](../decisions/GEMMA-FA-ONE-PROGRAM.md)).

Gates (fast-gate ids; each canary changes the program and must fail):

| Ids | Asserts | Canary |
|---|---|---|
| `chunkinv`, `chunkinvc` | prefill byte-identical at chunks 2048/64/32 (`tools/chunk-invariance-gate.sh`) | `MEMRA_PRIME_F32CHUNK0=1` restores the f32 first chunk |
| `chunkinv35`, `chunkinv35c` | the same on Step-3.7-Flash past its 512-token window | `MEMRA_STEP35_SWA_TKV=1` |
| `tickinv35`, `tickinv35c` | a prompt split across calls by tick budgets and LCP splits (`tools/tick-invariance-gate.sh`) | `MEMRA_PRIME_CALLLOCAL=1` |
| `primegrid`, `primegridc` | the prime-grid law below (`tools/prime-grid-gate.sh`) | `MEMRA_GDN_CHUNK=64` |

## The prime-grid law

The third segmentation axis: splits across prime calls on a recurrent trunk. The engine law, gated
by `tools/prime-grid-gate.sh`:

- A prompt primed as two `prime_cache` calls split at `L` is bit-identical to the monolithic prime
  when `L % gdn_chunk_size() == 0`, whatever the calls' internal chunk boundaries.
- An off-grid split diverges from exactly row `L` onward. Rows before `L` are bit-identical, and
  greedy flips land only on near-tie margins.
- Under the sequential GDN scan (`MEMRA_GDN_CHUNKED=0`) every split is exact. That pins the
  mechanism: the chunked WY scan segments per call, so an off-grid call start shifts the fold grid
  and materializes recurrent state at a point the monolithic program never computes. This is lawful
  floating-point behavior of the WY form and cannot be fixed at arbitrary split points.

What serve controls is where it stops. Every boundary serve chooses rounds down to the grid
(`grid_align_boundary_within`): the plain and spec checkpoint stops, the prefix-cache LCP-split
capture, the first-message capture and the prompt-end seed. A boundary-stopped prime, a checkpoint
resume and a whole-entry restore then reproduce the cold monolithic bytes, at a cost of at most 31
re-primed tokens per resume. Attention-only trunks are split-invariant already; the rounding applies
to them too, so there is one law. The decision and its extensions:
[PRIME-GRID-LAW](../decisions/PRIME-GRID-LAW.md).

Measured on the RTX 5090 Laptop (Qwen3.8-27B, spec off, greedy,
[`research/spill-b-20260919/DAY17.md`](../../research/spill-b-20260919/DAY17.md)): a twelve-turn
chain restored from off-grid prompt-end entries flipped a near tie at turn 10; the same
12,350-token prompt restored from on-grid entries reproduced the cold bytes; under the sequential
scan the chain was identical 12 of 12.

### The capture law

One capture law for every site the server captures at (`seed_capture_boundary`):

- A published entry's length is `capture_len(P)` for a prompt of `P` tokens: `P` when `P` is a
  multiple of 32, otherwise the largest multiple of 32 below `P` that leaves at least 16 tokens
  after it. The prime stops there, the entry is snapshotted, and the rest primes from a grid start.
  A grid-entry hit reports `cached_tokens == capture_len(P_previous)`; the operator view is in
  [prompt-cache.md](prompt-cache.md). A plain session's off-grid prompt end attaches to that entry
  as a [prompt-end tail](prompt-cache.md#prompt-end-tails): the state the cold prime ended in,
  restored only by a prompt equal to the entry's key plus the tail's tokens. Such a hit reports
  `cached_tokens == P`.
- A prompt of 65 to 79 tokens has no grid entry at or above the 64-token floor and publishes
  nothing: `[prefix-cache] seed REFUSED (grid)`, counted in `prefix_cache_seed_grid_refusals`. A
  prime path that cannot stop on the boundary refuses with the same line rather than publish an
  off-grid entry.
- A cold interactive session whose seed boundary lies inside its prompt joins the concat prime with
  its row's take ending on the boundary, publishes after that call, and primes the rest batched with
  its peers' remainders (`batch_row_take`,
  [`research/c8-serving-20261107/`](../../research/c8-serving-20261107/)). The LCP split and the
  plain-affinity checkpoint still prime alone.
- Speculative sessions: a cold spec session captures at the seed's grid boundary, with a prime stop
  of its own beside the affinity boundary. A restored spec session republishes at the render-stable
  boundary ahead of what it restored, or at the seed's grid boundary when none lies ahead. The
  engine's prompt-end republish fires only when the prompt end is on the grid. Spec-on and spec-off
  therefore restore the same entry and prime the same suffix from the same grid start
  ([`research/spill-b-20260919/DAY19.md`](../../research/spill-b-20260919/DAY19.md)).
- DFlash: a cold prime with no render-stable cut captures at the seed's grid boundary
  (`dspark_cold_capture`). The prompt end stays the cut only when it is on the grid or the prompt is
  too short for a grid entry. The price is a two-call cold prime for an off-grid prompt: +0.15 s on
  a 2,992-token first turn on the RTX 5090 Laptop (3.54 to 3.69 s, mean of 2 runs per arm,
  2026-11-07, [`research/plain-restore-grid-20261107/`](../../research/plain-restore-grid-20261107/)).

### The restore grid law

An entry that carries recurrent state restores exactly only as a whole cover of the prompt, or as a
strict prefix that ends on the grid and leaves at least `PRIME_MIN_T` tokens to prime
(`prefix_restore_exact`). Every probe that names the entry a hit restores applies it: the device
and host-tier lookups, the seed dedupe (`deepest_covering`), the retained-prefix admission plan,
the DFlash restore and the DFlash cold-drafter conversion. The longest qualifying entry serves,
else the request primes cold. The hit site prints `[prefix-cache] restore grid law: entry <N> of
<P> prompt tokens refused ...` (throttled) and counts `prefix_cache_restore_grid_refusals`; the
cold-drafter conversion counts `dspark_cold_drafter_grid_refusals_total`. Attention-only entries are
split-invariant and keep every whole-entry hit.

Evidence: the warm-against-cold pair cells in
[`research/plain-restore-grid-20261107/`](../../research/plain-restore-grid-20261107/)
(`grid_pair_cells.py`, every warm answer byte-equal to its cold twin), and the CPU test
`the_device_prefix_probes_apply_the_restore_grid_law` in the default test set
([`research/grid-law-cpu-test-20261107/`](../../research/grid-law-cpu-test-20261107/)).

### The pool grid law: pool rewinds

A parked session's rows past its own prompt were written by decode and verify, off the grid. On a
recurrent trunk, a resume at the parked end primes the next turn from state no cold prime computes,
and a later turn can leave the cold text. Each pool therefore rewinds a recurrent session to an
on-grid point a cold prime also computes and primes the rest, or primes cold. Attention-only trunks
resume where they stand.

- **Plain continuation pool** (memra#1091). An exact-extension hit rewinds to the session's grid
  checkpoint when the prompt matches the committed tokens through it and extends it by at least 16
  rows (`plain_pool_resume_at`, `grid_rewind_plain`). Otherwise it logs `[kv-reuse] grid-rewind:
  plain declined (...)` and primes cold. The pool asks this rule first
  (`grid_rewind_plain_refusal`), so park compaction never grows an entry the rewind then drops.
- **MTP spec pool** (memra#1093). An exact or text hit rewinds to the session's turn checkpoint
  when that checkpoint is on the grid, the prompt matches the committed tokens through it, and at
  least 16 rows remain (`spec_pool_grid_refusal`). A text hit that rewinds primes the prompt's own
  tokens. The affinity tier takes only such a checkpoint. A retiring session with no on-grid
  checkpoint is freed, not parked (`spec-park SKIPPED (grid law)`).
- **DSpark pool** (memra#1093). A retiring DFlash session parks rewound to the last prime-boundary
  capture `g` it published itself (`dspark_park_rewind`): trunk KV cut to `g`, the recurrent state
  copied from that prefix entry (matched by id), the drafter cut to `g`. That is the state a cold
  prime of those `g` tokens leaves. The exact token tier (`dspark_exact_pick`, never the text tier)
  resumes it as a new stream and primes the rest from `g`. With no capture of its own, the entry
  evicted, the drafter window compacted away, `MEMRA_KV_PAUSE_DEMOTE` on, or the restore switch on,
  the session is freed at retire (`dspark-park SKIPPED (grid law)`).
- **Growing a rewound session.** The exact tier takes a rewound session at any cap: after every
  cheaper refusal the resume reallocates it at the request's cold allocation with the state carried
  over (`HybridModel::dspark_grow_session`), so an agentic prompt that grows every turn under one
  `max_tokens` still resumes. A failed grow or seat serves cold and drops the entry. The
  cache-preferred veto yields to the entry the probe would take.
- **The DSpark restore switch.** `MEMRA_DSPARK_PREFIX_RESTORE` defaults on for GB10 and the RTX PRO
  6000 Max-Q, off elsewhere; `1` or `0` decides. With it on, a retiring DFlash session is always
  freed: the prefix restore serves the next turn from an on-grid entry that carries the draft tail.

Rules shared by every pool:

- A resume that primes a suffix is a new request's stream: its own sampler, seed, cap, and Philox
  counters at (0, 0) as on a cold session. Carrying the parked counters made a sampled warm turn
  leave the cold text even from an on-grid checkpoint.
- A sampled request never resumes at a parked end with nothing to prime
  (`sampled_parked_end_resume_refused`, memra#1154): it would emit the token the parked request
  drew. An empty-suffix greedy continuation keeps the parked counters.
- The plain-affinity and spec-affinity rewinds obey the same 16-row floor. A 1..15-row suffix would
  prime token by token, a program the cold prime does not run.
- Sessions with no plain checkpoint prime such a turn cold: glm5_next hyper trunks, prompts with no
  turn markers and no `session_id` or `user`, DFlash sessions demoted to plain under load, and boots
  with `MEMRA_AFFINITY=0` or PP host bounce.

Measured on the RTX 5090 Laptop at the device default (counter definitions in
[metrics.md](metrics.md)):

| Change | Result | Receipt |
|---|---|---|
| plain pool rewind | about +0.1 s TTFT on a 3k-token turn; the on-grid seed serves the bulk | [`pool-resume-grid-20261107`](../../research/pool-resume-grid-20261107/), [`pool-resume-cost-20261107`](../../research/pool-resume-cost-20261107/) |
| DSpark park rewind | turn 3 resumes at 3,072 and primes 72 rows; 9 of 9 pair cells byte-equal to cold; TTFT 2.462 to 0.275 s | [`pool-rewind-dspark-20261108`](../../research/pool-rewind-dspark-20261108/) |
| DSpark grow | 15 of 15 pair cells byte-equal; turn 2 TTFT 2.319 to 0.273 s; cap-refused agentic second turns 2.6-3.0 s to 0.45-0.83 s | [`pool-grow-dspark-20261108`](../../research/pool-grow-dspark-20261108/) |
| spec pool rewind | 22 of 22 three-turn pair cells byte-equal (12 of 22 on the base); +0.1 s TTFT where the base resumed at the parked end (0.120 to 0.216 s) | [`spec-pool-grid-20261109`](../../research/spec-pool-grid-20261109/) |
| plain-affinity floor | an edited 175-token resend byte-equal 3 of 3; TTFT 0.327 to 0.217 s, medians of 3 | [`affinity-floor-20261109`](../../research/affinity-floor-20261109/) |

Counters: `continuation_pool_grid_rewinds`, `continuation_pool_grid_declines`,
`spec_pool_grid_rewinds`, `spec_pool_grid_declines`, `spec_pool_grid_park_skips`,
`dspark_pool_grid_park_rewinds`, `dspark_pool_grid_park_skips`, `dspark_pool_grid_rewind_resumes`,
`dspark_pool_grid_rewind_grows`, `dspark_pool_grid_declines`.

### What the grid law still does not promise

- A verbatim-extension continuation resume of an attention-only cache keeps decode-computed rows
  that a cold prefill never reproduces: bounded logit perturbation, flips only at near ties
  (`concat-prime-probe primepath --hist` measures it). The two doors that tried to remove this
  residual are deleted
  ([FLAGS.md](../FLAGS.md#removed-doors-2026-11-04-the-verbatim-extension-resume-arms-memra_resume_grid_rewind-and-memra_resume_exact)).
- The `PRIME_MIN_T` floor applies to recurrent entries only. An attention-only hit that leaves a
  1..15-token suffix primes it token by token.
- The same-window prefix fanout (`prefix_fanout_groups`) takes the raw in-batch LCP as its capture
  length. Covered by inspection only.
- glm5_next KDA entries (captured only under `MEMRA_PREFIX_LATENT=1`) count as recurrent although
  KDA prefill is a sequential scan with no fold grid, so their off-grid prompt-end captures serve
  only whole covers.

The valid assertions everywhere: per-program determinism, grid-aligned byte identity, and across
programs, divergence confined to the split row with near-tie-only flips.

Gates: `tools/prefix-restore-identity-gate.py` (every restore point gives the cold completion and
lands on the grid), `tools/prefix-newest-turn-fits-gate.py` (its V5 identity and V6 grid clauses)
and `tools/spec-on-cache-hit-gate.sh qwen`. Each was red on the pre-fix binary and green on the fix
on the RTX 5090 Laptop and one RTX PRO 6000 Blackwell
([`DAY18.md`](../../research/spill-b-20260919/DAY18.md),
[`DAY19.md`](../../research/spill-b-20260919/DAY19.md)). Commands:
[TESTING.md](../TESTING.md#prefix-cache).

## Near-tie classes

Each class is two correct programs disagreeing on a near tie. A wide-margin flip, divergence before
a split row, or a nondeterministic program is a defect, not a member of a class.

| Class | Programs | Status and bound |
|---|---|---|
| batched prime against the tokenwise oracle | `prime_cache` and `decode_step` | reported by run-gen and `prime-gate`; `FLIP-NEARTIE` within `MEMRA_PRIME_GATE_MARGIN`, structured divergence fails |
| prefill against decode argmax | `forward_last` and decode | the [argmax margin gate](../TESTING.md#argmax-margin-gate-floors-stored-rows-recalibration) |
| cache hit against a fresh prime of another configuration | the entry's prime and the fresh one | the first row's class; a hit against its own prime config is bit-identical |
| attention-only verbatim-extension resume | decode-written rows and a cold prefill | bounded perturbation, near-tie flips only |
| off-grid prime split | two calls at `L % 32 != 0` and the monolithic prime | engine law; serve chooses no such split except the fanout; `primegrid` confines it to row `L` with flips at margin below 0.5 |
| chunked against sequential GDN scan | `MEMRA_GDN_CHUNKED=1` and `=0` | accepted when the chunked scan became the default; run-gen and run-spec batteries |
| opt-in eager B=1 against the batched body | `MEMRA_SERVE_B1FAST=1` | measurement door only; `decode-batch-gate` strict mode |
| speculative verify shape on an unbounded schema | verify width changes the summation order in a whitespace tail | byte-identical with the shape held fixed; bound the schema ([`constrained-full-20260803`](../../research/constrained-full-20260803/)) |
| CLI against server on a safetensors checkpoint | prefill GEMM and batched decode against the CLI's chain | `tools/serve-st-gate.sh`: the parting step is a near tie within `MEMRA_PRIME_GATE_MARGIN`, and a boot on the CLI's config reproduces the CLI stream |
| sampled speculative draws at another K | Philox events drawn per round | by design; a sampled request takes the K its cold twin takes ([`spec-k-resume-20261109`](../../research/spec-k-resume-20261109/)) |

## Serving-contract guarantees, and the receipt for each

The standing list of what the serving contract promises and the measurement behind each item.
Changes are in the [releases](https://github.com/avifenesh/memra/releases).

- Greedy streams are isolated-identical under concurrent load at the defaults, and the decode
  program does not change with batch width
  ([iso-gap](../../research/iso-gap-20260807/), [servepath-p2](../../research/servepath-p2-20260805/)).
- Step-3.7-Flash PP-2 decodes in one numeric class at every batch width: a greedy request returns
  the same bytes whether it decodes solo, joins a batch mid-generation or starts batched
  ([isolation](../../research/p0iso-20260810/RESULTS.md),
  [fix](../../research/b1fix-20260810/RESULTS.md)).
- Prefill is byte-identical across chunk sizes and tick segmentation on the shipped architectures
  ([chunk invariance](../../research/chunk-invariance-20260805/VERDICT.md),
  [tick segmentation](../../research/tick-seg-20260807/)).
- A restored or resumed turn on a recurrent cache writes its cold twin's bytes: captures, restores
  and pool rewinds obey the prime-grid law
  ([restore grid](../../research/plain-restore-grid-20261107/),
  [spec pool](../../research/spec-pool-grid-20261109/),
  [DSpark pool](../../research/pool-grow-dspark-20261108/)).
- A cache hit is bit-identical to the run that computed the prefix
  ([receipt](../../research/prompt-cache-20260802/)).
- Serving fails closed on exposure: a non-loopback bind without a key source refuses to boot,
  `/metrics` and `/yield/metrics` require bearer auth whenever keys are configured or the bind is
  public, key comparisons are constant-time, and keyring rewrites are atomic
  ([receipt](../../research/sec1-20260811/RESULTS.md)).
- Release builds retire prefix pins, and client cache salts are bounded and validated
  ([receipt](../../research/pinfix-20260809/RESULTS.md)).
- The GGUF parser rejects truncated and malformed files with contextual errors instead of panicking
  the GPU worker; every byte-prefix of a valid file is covered by fixture tests
  ([receipt](../../research/ggufhard-20260811/RESULTS.md)).
- The GPU Gumbel sampler keeps uniforms below 1.0, so no rare `+inf` winner injects an arbitrary
  token into a long sampled generation ([receipt](../../research/longdepth-20260809/RESULTS.md)).
- Explicit `max_tokens` is enforced exactly: token events, usage and scheduler accounting follow
  visible token ids, not speculative rounds ([receipt](../../research/honesty-20260809/RESULTS.md)).
- Plain-decode session affinity checkpoints and resumes rewritten conversation histories, so later
  turns prime the new suffix instead of the whole history
  ([receipt](../../research/affinity-20260809/RESULTS.md)).
- A new admission can end an in-flight speculative burst at its next round boundary; request-sized
  KV allocation stays VRAM-aware ([receipt](../../research/admission-20260806/VERDICT.md)).
- The Step-3.7-Flash SWA ring, armed by default for that family (`MEMRA_SWA_RING` forces it on or
  off), right-sizes sliding-window KV: the 262k session KV component drops 3.6x, from 2 to 12
  concurrent 262k sessions before the first defer, and lapped checkpoints decline safely
  ([receipt](../../research/kv256-20260809/RESULTS.md),
  [validation](../../research/ringval-20260810/RESULTS.md)).
- A host-staged PP boundary (`MEMRA_PP_HOST_BOUNCE=1`, default off) serves byte-correct on hosts
  whose GPU peer copy reports success but does not preserve bytes, a failure a capability flag or
  bandwidth test does not reveal. Byte-probe peer paths when provisioning
  ([receipt](../../research/hostbounce-20260810/RESULTS.md)).
- Graceful shutdown joins the GPU worker thread before exit, so a new server cannot boot while the
  old worker still holds device state (`tools/health-fault-gate.sh`, its SIGTERM drain arm).
- `/v1/models` advertises the parameters each loaded model supports and renders prices and input
  modalities from its `MEMRA_MODEL_METADATA` entry ([model-metadata.md](model-metadata.md)).
