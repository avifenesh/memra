# qwen4_exp decode PROFILE-9 — devtwin: the host twins move device-side

Lane: devtwin (owner-sequenced from three closed lanes: PROFILE-7 §2's verify
decomposition — ~12 ms/round of host-twin bubbles ≈ 1/3 of verify wall, 48 router dtoh
boundaries; mtp11's audit — the draft step's host twins serialize the round, deferred
readback flat until they move; and the round-3 graphs doctrine — whole-step graphs
structurally impossible BECAUSE of the host boundaries). Box: sbox-eval Frankfurt,
2× RTX PRO 6000 Blackwell 96 GB; artifact `~/data/q48fn-nvfp4`; ship admission
everywhere (dev1 + K=5 + adapt k_lo=1 + pmin 0.3). Census: devtwin/CENSUS.md (work
item 1) — 64 blocking dtoh per forward: 48 router + 12 idx_proj + 4 PLE.
Receipts: `devtwin/` (box pulls from ~/realgate/devtwin). Baselines this measures
against: PROFILE-7 ship battery (thinkon 75.5 tok/s spec, raw 115-120) and the mtp11
defer ladder (the re-measure baseline PROFILE-8 §7 banked for exactly this lane).

## 1. What moved, in three stages (both seams now default ON — §6)

- **Stage 1 — device MoE router** (`qwen4exp_route_topk_f32`, seam `routerdev`): the
  full `host_route_softmax_topk` program on device — softmax over 512 (order-sensitive
  reductions sequential on one thread, host op order verbatim; exp through double, the
  one op not bit-pinned to host libm), top-10 under the pinned tie rule (weight desc,
  index asc), renorm with the 6.1035156e-5 floor — writing sel/w(/tok-map) straight
  into the grouped-NVFP4 dispatch and the graph-driver slots. Kills the 48 blocking
  router dtoh + host top-k + selection h2d per forward. Engages: t==1 decode (eager +
  graph driver), merged verify columns, zero-draft steps. Host twin kept: per-expert
  prefill executor, TP2 (host expert ids by construction).
- **Stage 2 — the draft's device route** (`qmatvec_bf16w_sel_f32`): the card-1
  DeviceBf16 draft bank consumed HOST expert ids in per-slot launch offsets — the
  reason mtp11's deferred readback could not move the chain (PROFILE-8 §4). The sel
  kernel reads expert ids from the device route (per-row program off_into-VERBATIM =>
  bit-identical; 3 launches replace ~30/step), so the draft chain step's router
  boundary dies with the same `routerdev` seam.
- **Stage 3 — device indexer raw-key cache** (`copy_rows_col_f32`, seam `idxcache`):
  below the QSA selection horizon ((base_pos+t)/block <= budget — position < 2051 on
  real geometry, i.e. EVERY shape in the ship battery) the per-layer idx_proj dtoh
  existed only to feed the host raw-key cache for a possible future scored row. The
  k-part rows now append d2d and the host cache materializes LAZILY at the first
  scored chunk (same bytes dtoh'd later — bit-identical by construction; flips pay
  debt both directions; rewind clamps the device rows). Kills the 12 idx_proj blocking
  dtoh per forward (+1 per draft chain step) and the [t×640]f32 dtoh per prefill chunk.

NOT moved, deliberately: the PLE host n-gram gather (102 GB host table — host BY
DESIGN; its 4 gate-dot dtoh remain, 1 layer); the indexer top-k past the horizon (the
score-slab dtoh + host `top_blocks_ascending` engage only at position >= 2051 — the
yarn lane's domain, zero rounds in this battery's shapes; census §4). TP2 keeps the
HOST router twin by construction (it consumes host expert ids), but its indexer path WAS
touched — it now pays the `idxcache` host-cache debt on entry, because a single-card
prefill can leave that cache lagging and the state migration is one-way — so **tp2-gate
was re-run with both seams armed: 24/24 argmax, worst rel 3.0e-5** (`tp2-gate-dt4-tp2.tsv`).

## 2. Exactness receipts (the contract: selection set+order EXACT, weights ULP-documented)

- Kernel oracle (tiny arm 0f, `gate_route_kernel`, real 512/10 geometry + tie
  batteries — boundary-straddling duplicate groups, all-equal, underflow/subnormal
  ties, geometry edges, batched rows + tok map): selection ids+order EXACT; worst
  weight ULP **0 on the rig**, **0 on the box** (tiny-gate-routerdev-box.tsv).
- bf16 sel-matvec oracle (bf16 oracle sel mode): BIT-IDENTICAL to the per-slot
  off_into chain, duplicate slots, shared-x + slot-x strides, rig + box.
- **Live cross-surface audit** (`MEMRA_Q4E_ROUTER_AUDIT=1`, the dsv4 sigrouter
  precedent): every device route in a real run recomputed on host from the same
  logits and hard-compared. Box, phase 1: **252,778 rows, ZERO selection
  mismatches, worst weight ULP 3** (bound 8) — the exp-twin (double-exp rounded to
  f32) differs from this box's glibc expf by <= ~1 ulp on some live inputs, which
  compounds to <= 3 ulp through the two divisions and moved NO selection. Rig glibc
  matched at ULP 0 — the bound is per-host, which is why the audit rides every gate
  battery instead of being a one-time proof.
- Law gates with the seams armed (+audit), box: verify-bit **24/24 bit-identical**;
  spec-gate byte identity **raw 4/4, thinkon 4/4, long 6/6** at 256 tokens;
  seam-gate 24 decode rows **24/24 argmax, worst KL 0.00000, worst rel 1.7e-5**;
  **tp2-gate 24/24**; greedy-vs-banked-goldens divergence pattern IDENTICAL to the mtp11
  host-router run (raw: -1/8/-1/48 — the pre-existing stale-golden forks reproduce
  EXACTLY, i.e. the device router moved no greedy chain on those prompts; thinkon
  goldens were stale at token 0 in mtp11 too — not a devtwin artifact).
- Tiny fixture: 26/26 arms PASS with `routerdev,idxcache` armed (rig + box), incl.
  mtp-spec byte identity with the DRAFT device-routed at tiny geometry (8 experts,
  top-2 — the tiny draft bank is DeviceBf16, so stage 2 is exercised end-to-end) and
  the horizon-crossing chunked arms (mtp-spec-ring chunk 8 / prefill-extend chunk 5
  cross the tiny horizon at position 11, exercising idxcache's device-append, lazy
  catch-up boundary, and scored path inside byte-identity/tolerance arms).

## 2b. The route kernel took three versions, and the A/B caught what the oracles could not

Corpus-worthy sequence (all receipts in devtwin/):

- **v1** ran every phase on thread 0 over GLOBAL memory. Every exactness gate was
  green (the oracles measure bits, not microseconds); the FIRST perf row —
  plain-decode `--ab-seam routerdev --ab-moe 5x128` — showed the seam **+12.5%
  SLOWER** (host 14.90 vs dev 16.76 ms/token, ab-routerdev-dt-plain-V1KERNEL.tsv),
  ~39 us per route launch. The battery was STOPPED at that row rather than spending
  GPU-hours measuring a known-slow kernel.
- **v2** staged the softmax slab in smem and parallelized the fmaxf fold (associative
  — any order bit-exact): 16.76 -> 16.46 ms. Barely moved — wrong diagnosis.
- A standalone **phase bisect** (rig microbench) attributed 48 of the 54 us to the
  top-k: a serial insertion over thread-LOCAL arrays with DYNAMIC indexing, which
  nvcc spills to local memory — every `top_w[pos-1]` in the scan was a DRAM-backed
  load on a serial dependence chain. exp-through-double was 3 us, the order-bound
  sequential sum 2.4 us: the "obviously slow" phases were fine, the boring array was
  the wall.
- **v3** replaces the insertion with k rounds of block-wide strict MIN over distinct
  u64 keys `(~bits(w) << 32) | idx` — ascending key order IS the host comparator
  (weight desc via total_cmp, index asc) on the non-negative weight domain, so a
  total order enumerated smallest-first is bit-exact under ANY evaluation order, ties
  included. 54 -> **9.0 us/launch** on the throttled rig (13.4 serialized); oracle +
  tiny arms re-green at every step.

Law reinforced: an exactness gate proves bits, only an interleaved A/B proves the
lever — run the FIRST perf row before committing the fleet's hours (and the
`ab-moe` plain-decode row is the cheapest first row this route has).

## 3. Per-change A/B — plain decode, box, interleaved x5 (`--ab-moe 5x128`)

The plain-decode row is the cheapest instrument that reaches the graph-driver route, and
it carried the whole diagnosis. All rows: rep-0 chains IDENTICAL (`first_divergence -1`),
within-arm spread <= 0.3% (this box is very stable at 128 steps x 5).

| arm | decode graphs | ms/token | tok/s | vs its own host arm |
|---|---|---|---|---|
| host router (shipped default) | ON | 14.91 | 67.1 | 1.000 |
| `routerdev` **v1 kernel** | ON | 16.76 | 59.7 | **0.890** |
| `routerdev` v3 kernel | ON | 16.46 | 60.8 | **0.906** |
| `routerdev` v3 + `MEMRA_Q4E_ROUTE_SYNC=1` | ON | **14.26** | 70.1 | **1.046** |
| host router | OFF | 15.13 | 66.1 | 1.000 |
| `routerdev` v3 | OFF | **13.97** | **71.6** | **1.083** |
| `idxcache` alone | ON | 14.55 | 68.7 | **1.024** |

### The finding: the seam's graphs-ON cost is the MISSING SYNC, not the kernel

v1 -> v3 cut the kernel 54 -> 9 us/launch (rig microbench) and the graphs-ON row moved
16.76 -> 16.46: essentially nothing. Restoring the host arm's per-layer stream sync
while keeping the device route (the `ROUTE_SYNC` diagnostic) moved the SAME kernel from
16.46 to **14.26** — a 2.2 ms/token swing bought by ADDING a sync back. So the host
twin's dtoh was doing double duty: it was the semantic boundary AND a per-layer throttle
on the graph-replay queue. Remove it and 96 graph replays + 48 route kernels per token
enqueue unthrottled; the launch path degrades far past what the removed dtoh cost.

Corroboration from the section profiler (eager, graphs disabled by the profiler —
`profile-dt3-prof-{off,on}.tsv`): the device router is a clean win where no graph queue
exists — `moe.router` **1.775 -> 1.175 ms/token** (48 calls), attributed total
18.90 -> 18.21. Same direction as the graphs-OFF A/B (15.13 -> 13.97).

Read at this point in the diagnosis, the best configuration looked like "device router
with decode graphs OFF" (13.97 vs 14.91, 1.067x). §3a supersedes that: with the STACK
(router + indexer) the graph setting stops mattering (13.57 ON vs 13.60 OFF) and the
stack wins either way, so the shipped configuration keeps `graph` at its default ON and
no pairing with the graph seam is required. What survives from this row is the doctrine
inversion: decode graphs were impossible BECAUSE of the host boundary, and with the
boundary gone they are merely optional (they only ever bought +1.3%, PROFILE-2).

### 3a. The COMBINED stack inverts the router's graphs-ON sign (honest open item)

| arm (plain decode, x5) | decode graphs | ms/token | tok/s | vs host |
|---|---|---|---|---|
| `routerdev` alone | ON | 16.46 | 60.8 | 0.906 (LOSES, reproduced twice) |
| `idxcache` alone | ON | 14.55 | 68.7 | 1.024 |
| **`devtwin` (both)** | **ON** | **13.57** | **73.7** | **1.099** |
| `devtwin` (both) | OFF | 13.60 | 73.5 | 1.112 (host arm 15.12) |

So the router's marginal effect flips sign with the indexer seam present: **+1.55 ms
alone, -0.98 ms on top of idxcache** — and the combined stack needs NO graph pairing
(it wins with the shipped `graph` default ON, 13.57 vs 13.60 off: a wash).

### Mechanism, MEASURED (phase-windowed nsys pair, `nsys/dt6-{routeronly,both}_*`)

Same window (10 warm decode steps, `--profiler-window`), routerdev-only vs the stack:

| | GPU kernel total | `cuMemcpyDtoHAsync` calls | median per call | `cuGraphLaunch` |
|---|---|---|---|---|
| routerdev only | 31.14 ms | 170 | **821 us** | 840 calls, 6.81 ms |
| stack (both) | 31.25 ms | 50 | **21.5 us** | 840 calls, 6.71 ms |

**GPU work is identical (31.14 vs 31.25 ms) and graph launches are identical** — so the
cost is pure host-side stalling, and it is not the graph machinery. What changed is what
the SURVIVING dtoh calls cost: the same 2.5 KB idx_proj readbacks that take 21.5 us with
the stack armed take **821 us** (38x) with only the router moved, because each one now
drains a queue the removed router syncs used to keep shallow. Pipeline bubbles, priced:

- host router (64 syncs/token): shallow queue, 64 small drains -> 14.91.
- routerdev only (16 syncs/token): 4 layers of work queued between syncs; each of the 12
  QSA dtoh drains it, then the GPU idles while the CPU refills -> 16.46.
- routerdev + `ROUTE_SYNC` (64 syncs again): shallow queue restored -> 14.26.
- the stack (4 PLE syncs/token): ONE bubble per token instead of twelve -> 13.57.

So bubble cost is not monotone in sync COUNT — it is (number of blocking syncs) x (queue
depth at each) — which is exactly why removing HALF the boundaries is worse than removing
either none or nearly all of them. Transferable rule: **remove host boundaries in whole
groups; a partial removal can pay more in drain-bubbles than the boundaries it deleted.**

Named residual from the same capture: `qwen4exp_route_topk_f32` is 480 instances /
5.33 ms in the window = **~0.53 ms/token of GPU time** (11.1 us/launch on the box, the
rig microbench's 9.0 us confirmed) against the 1.775 ms/token host-router section it
replaced. A cheaper route (folding it into the MoE-tail prologue, or a wider grid) is
worth up to that 0.53 ms.

## 3b. Spec-loop A/B — the product shape (box, `--router-ab`, x3 escalated to x5 on every cell)

Ship admission (dev1 + K=5 + adapt k_lo=1 + pmin 0.3), 256 tokens/arm, combined
`devtwin` stack. **`earliest_chain_divergence = -1` on EVERY cell**: across 256-token
generations on every shape the device-routed run emitted the byte-identical token
stream as the host-twin run — stronger than the ULP contract requires, and the reason
this table is a like-for-like comparison rather than two different generations.

| shape | host twins ms/tok | devtwin ms/tok | tok/s (host -> dev) | speedup |
|---|---|---|---|---|
| **thinkon** (the model's DEFAULT render) | 13.331 | **11.417** | 75.0 -> **87.6** | **1.168** |
| thinkoff | 9.755 | 8.307 | 102.5 -> **120.4** | **1.174** |
| efflow | 12.862 | 11.088 | 77.8 -> **90.2** | **1.160** |
| raw goldens (bench-only shape) | 8.764 | 7.341 | 114.1 -> **136.2** | **1.194** |
| long agentic (724-token prompt) | 16.340 | 14.648 | 61.2 -> **68.3** | **1.116** |

Within-arm spreads 2.7-4.4% (receipted per arm); every verdict is 1.12-1.19x, i.e.
3-6x the pooled spread — outside the noise band that made mtp11's rows unclaimable.

Why the spec loop shows the win cleanly and needs no graph discussion: under an armed
verify `graphs_mode` is OFF by construction (the mtp11 wide-capture fix), so spec
rounds never take the graph-replay path at all — the verify chunks and zero-draft
steps run eager, exactly where the section profiler measured `moe.router`
1.775 -> 1.175 ms/token.

### K ladder (thinkon, combined stack)

| K | host ms/tok | devtwin | speedup |
|---|---|---|---|
| 1 | 13.358 | 11.738 | 1.138 |
| 2 | 12.977 | 11.321 | 1.146 |
| 5 (ship) | 13.331 | 11.417 | 1.168 |

The win GROWS with K (more verify columns and chain steps per round = more former host
boundaries), the opposite of mtp11's deferred-readback decay — consistent with the
boundaries being the thing that moved.

Note the structural reason the spec loop should behave like the graphs-OFF row: under an
armed verify, `graphs_mode` is OFF by construction (the mtp11 wide-capture fix), so a
spec run's t==1 steps and verify chunks never take the graph-replay path — the
unthrottled-queue effect above does not apply there.

Phases: combined `devtwin` (routerdev+idxcache) per shape at ship K=5
(thinkon/thinkoff/efflow/raw/long-724); K ladders 1,2,3,5,8; the defer re-measure
against the mtp11 banked ladder (PROFILE-8 §4's unlock condition is now satisfied: the
draft step's router AND indexer dtoh are device-side); sampled probe.

## 4. Graphs doctrine note answered — measurement first, capture second

The round-3 doctrine ("whole-step graphs structurally impossible BECAUSE routing is a
host twin") is now UNBLOCKED: with `routerdev` the graph-driver route is a pure device
launch between the interior and tail replays, so interior+route+tail can capture as
one span (and multi-layer runs of the 3 consecutive PLE-free GDN layers between QSA
layers). Deliberately NOT built in this round: the sync REMOVAL itself ships without
any capture change, vgraph measured launch-issue FLAT on this box (0.9992x,
PROFILE-6), and the trunk's own 84-graph receipt was +1.3% — so bigger capture sets
are the follow-up ONLY if the A/B rows show residual issue-bound wall after the
boundaries die. KERNELS.md's "do not re-propose graph work without a new mechanism"
law stands; the new mechanism this lane added is the boundary removal, and it is
measured on its own.

## 4. The mtp11 defer seam re-measured from its banked baseline (PROFILE-8 §7's condition)

PROFILE-8 kept `defer`/`defer_guard_sync` OFF and banked its ladder as "the baseline to
re-measure from IF the host-twin lane later removes the router/indexer syncs from the
draft step". That condition is now satisfied (stage 2 moved the draft's route; stage 3
its indexer dtoh), so the re-measure ran with the devtwin stack armed (thinkon, x5):

| K | host chain | defer | defer/host | defer-gsync | gsync/host | mtp11 defer/host |
|---|---|---|---|---|---|---|
| 1 | 11.744 | 11.673 | 1.006 | 11.569 | **1.015** | 1.009 |
| 5 (ship) | 11.342 | 11.416 | 0.994 | 11.319 | **1.002** | 0.993 |
| 5 (2nd cell) | 11.497 | 11.460 | 1.003 | 11.330 | **1.015** | 0.993 |

Counter + chain identity PASS in every rep (the truncation-semantics receipt). The
ceiling did rise as PROFILE-8 predicted — `defer` moved 0.993 -> 1.003 at ship K and
`gsync` sits positive at every rung (up to +1.5%) — but the verdicts still sit inside
the 3.2-3.9% within-arm spread, so **both defer seams stay OFF**: this lane changes
PROFILE-8's verdict from "flat/negative" to "positive but unclaimable", and the next
re-measure needs a quieter box or a bigger effect, not another arm.

## 5. Sampled probe (serving law) and the greedy twin, devtwin armed

Vendor defaults (temp 1.0, top_p 0.95, top_k 20), thinkon, K=5 ship admission:
**SPEC-ENGAGEMENT rounds=150, rounds_with_accepts=71, accepted=105/199 drafted**,
13.38 ms/token = **74.73 tok/s sampled** (`spec-sampled-k5-dt4-sampled.tsv`). Greedy
A/B in the same load: plain 14.48 vs spec **11.63 ms/token (85.97 tok/s, 1.245x)** —
the interleaved thinkon row (87.6) reproduces through the sampled load.

## 6. Verdict, defaults, and the honest 200 tok/s answer

**Defaults FLIPPED ON, together, on receipts** (new-flags law: decision + reasons +
both arms + rollback + receipts in the same change): `ROUTER_DEV_DEFAULT = true`,
`IDX_CACHE_DEFAULT = true`, FLAGS rows rewritten, kill switches
`MEMRA_Q4E_SEAMS=routerdev=0` / `idxcache=0` intact. Why ON: every measured surface
wins (spec 1.116-1.194x across five shapes, plain decode 1.099x graphs-ON /
1.112x graphs-OFF, K ladder 1.14-1.18x), the chains are byte-identical over
256-token generations, and all three rule gates plus tp2-gate are green with the seams
armed. **They flip TOGETHER**: `routerdev` alone with decode graphs ON measured 0.906x
(§3a), so the FLAGS rows name the pairing explicitly — enabling half this stack is a
measured regression.

Serving numbers at ship admission (spec ON, dev1, adapt k_lo=1 + pmin 0.3), vs the
PROFILE-7 ship battery this lane inherited:

| shape | PROFILE-7 ship battery | devtwin | plain decode today |
|---|---|---|---|
| **thinkon** (default render) | 75.5 | **87.6** | 67.1 -> 73.7 |
| thinkoff | 101.9 | **120.4** | |
| efflow | 78.5 | **90.2** | |
| long agentic (724) | 61.3 | **68.3** | |
| raw (bench shape only) | 115.1 | **136.2** | |

Baseline note (so the raw row is unambiguous): 115.1 is PROFILE-7's ship-battery raw at
the admission policy; the older **119.97 tok/s** figure is mtp10's FIXED-K=5 raw before
admission (PROFILE-7 §entry state), and the mtp11 re-measure of the same shape sat at
114.1 (8.764 ms/token, the host arm of this lane's raw cell). Against either, the
devtwin raw row (136.2) is the highest number this artifact has produced.

**Vs the owner's 200 tok/s target: NOT crossed.** The honest read: the thinking shape
(what real traffic renders) is at **87.6 tok/s**, 2.28x short; the friendliest bench
shape is at 136.2, 1.47x short. What this lane did buy is the largest single-round jump
the qwen4_exp decode has had since admission (+16% on the default render) and the
retirement of the host-boundary class that three prior lanes named as their blocker.

Residual levers, in measured order:

1. **The remaining 4 blocking dtoh per forward are PLE's gate dots** (1 layer, census
   P2/P3) — small, and the 102 GB host n-gram table keeps that layer host-bound by
   design. Not a lever; the census's other 60 are gone.
2. **The route kernel's own GPU time: ~0.53 ms/token** (48 x 11.1 us, nsys §3a) — fold
   it into the MoE-tail prologue or widen its grid. The graphs-ON interaction that
   looked unexplained is now MEASURED (§3a: drain bubbles, not graph machinery), so this
   is what is left of that thread.
3. **mt-kernel bandwidth efficiency on the weight-shared dense sections** —
   PROFILE-7's lever (2), still open and now the largest verify slice: with the twins
   gone, `hyper.read` (3.36 ms/token, 96 calls) and `moe.sel_grouped` (2.74) top the
   section profile.
4. **The defer seams** (§4): positive at every rung, unclaimable on this box's spread.
5. **The indexer top-k past the horizon** (score-slab dtoh + host `top_blocks_ascending`)
   — zero rounds in these shapes (< 2051 fill), a yarn-lane item for long contexts.
6. **TP2 t-generic verify** — unchanged bound <= ~9%, and TP2 still routes on the host
   twin (untouched by design; its gate is green with the seams armed).

## 7. The default flip was PROVEN to engage, not assumed

A default is a claim about what runs when nobody passes a flag, so it is verified the
way this repo verifies everything else — by running with NO env arming at all and
asserting an OUTCOME, not liveness (`run5-*`, receipts beside this file):

- tiny gate: **26/26 PASS** with no `MEMRA_Q4E_SEAMS` set.
- box, no seam env, ship admission: verify-bit **24/24 bit-identical**
  (`verify-bit-gate-dt5-defaults.tsv`), spec-gate **byte identity at 256 tokens**
  (`spec-gate-k5-dt5-defaults.tsv`).
- **`# router-audit rows=129004 worst_w_ulp=3`** — the live host-twin audit counted
  129k device routes in a run that armed nothing, which is the positive proof that the
  flipped default actually engages (a silent no-op default would have reported rows=0,
  the failure mode the audit's row counter exists to catch).

No claim, price, or roster row changes: qwen4_exp is not served; nothing here is a
customer-visible number.
