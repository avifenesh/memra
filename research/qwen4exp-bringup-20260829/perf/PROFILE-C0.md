# qwen4_exp PROFILE-C0 — the MEMORY side at depth: what the read patterns cost, and the
# kvq sign-flip verdict

Lane: the memory/expert lever lane (cache-friendly layout + expert speculation), one of three
parallel lever lanes against the owner's 262,144 target. Agent A owns indexer selection and
graphs at depth; agent B owns host-side prefetch and thread stickiness. This file owns the KV
block layout and its read path, the pooled-key / indexer-key LAYOUT, and the MoE expert path.

Box: the round-2/round-3 lane boxes, **2x RTX PRO 6000 Blackwell Server Edition, 97,887 MiB,
600 W** each. Provider, region, instance class and instance ids are fleet state and live in
darklanes, not here. Every timing row below names its cache arm; the static receipts in §2 are
host-side compilations and name their target architecture instead.

Resolved geometry this file reasons about, from ARCH.md (re-derived, not assumed): 48 text
layers (12 QSA + 36 GDN), hidden 2560, QSA **24 Q heads x 256, 2 KV heads x 256** so
`kv_dim = 512`, indexer MQA 4 Q heads x 128 with **1 shared K head x 128**, micro-block
`block_size = 4`, **budget 512 blocks = 2,048 tokens**, MoE 512 experts top-10 + 1 shared,
expert intermediate 640.

## THE VERDICT (the sentences to read alone)

**The kvq depth penalty is a LAYOUT/READ-PATTERN ARTIFACT, not the cost of quantization.** The
quantized block-list attention issues **2x the f32 twin's KV-cache load instructions per K
element** while reading 3.76x fewer bytes, and phase 1 is thread-per-position — so every load
replays 32 ways into 32 distinct sectors and the byte saving can never land. Measured in the
sm_120 SASS: the score-phase inner loop is **120 instructions per 8 K elements against the f32
twin's 37**, of which 8 loads per 8 elements are a *redundant re-load of the same fp16 block
scale*. Hoisting that one load to once per 32-element block takes the loop to **52** and brings
KV-cache load instructions to **parity with f32**. Fixable, not a tax.

**The two levers this lane owns are the #1 and #2 terms of the deep decode profile.**
`qsa.sdpa` is the largest section at every depth measured (10.3 ms, 27-30%) and is FLAT in
depth; `qsa.idx_host` is the second-largest term that GROWS with depth (2.5 -> 3.0 -> 3.2 ms
over 100k -> 131k -> 150k, extrapolating to ~5.7 ms at 262,144), and its cost is a
512-byte-strided pooled read that moves 1,024 B to use 128 B.

**Expert speculation is a RECEIPTED NON-LEVER and is dropped, not deferred** — there is no
staging latency to hide at this window at all, for a structural reason rather than a small
number. See §4.

## 1. The memory profile at depth (banked instrument, no new box time)

Source: `../round2-box-receipts/kvq2/ladder-r2prof-step-idxsel.tsv`, header
`# cache kv_quant=q8_0/q5_1 idxq=q8 golden_pin=false seams_env=idxsel` — ship defaults with
agent A's device top-k armed, i.e. the post-cliff regime that is the only relevant baseline.
`--profile` decode sections, one step per rung. **Absolutes are sync-bounded, eager (`prof::on`
disables decode graphs) and inflated; shares and the depth SLOPE are the signal.**

| section | 100,000 | 131,072 | 150,000 | slope | owner |
|---|---|---|---|---|---|
| **`qsa.sdpa`** | **10.3 ms (29.7%)** | **10.3 (27.9%)** | **10.3 (27.3%)** | **FLAT** | **this lane** |
| `ple.host_ngram_gather` | 4.9 (14.1%) | 6.5 (17.7%) | 7.3 (19.5%) | **+0.048 ms/k** | agent B |
| `hyper.read` | 3.2 (9.3%) | 3.2 (8.7%) | 3.2 (8.5%) | flat | — |
| **`qsa.idx_host`** | **2.5 (7.3%)** | **3.0 (8.1%)** | **3.2 (8.4%)** | **+0.014 ms/k** | **this lane** |
| `moe.sel_grouped` | 2.6 (7.7%) | 2.6 (7.2%) | 2.6 (7.0%) | flat | this lane |
| `gdn.proj` | 2.6 (7.5%) | 2.6 (7.0%) | 2.6 (6.8%) | flat | — |
| `gdn.norm_gate_out` | 1.4 | 1.4 | 1.3 | flat | — |
| `moe.shared` | 1.2 | 1.2 | 1.2 | flat | this lane |
| `moe.router` | 1.1 (3.2%) | 1.1 (3.1%) | 1.1 (2.9%) | flat | this lane |
| `hyper.write` | 0.9 | 0.9 | 0.9 | flat | — |
| `qsa.proj` | 0.9 | 0.9 | 0.9 | flat | — |
| `lm_head` | 0.8 | 0.8 | 0.8 | flat | — |
| profiled token | ~34.7 ms | ~36.9 | ~37.7 | | |

### 1a. Why `qsa.sdpa` is flat, and why that makes a shallow A/B valid at 262k

The QSA selection budget is **512 blocks x block_size 4 = 2,048 tokens**, so the block-list
`count` SATURATES at ~2,052 rows from roughly 8k of fill onward and never grows again. The
measurement consequence is the useful one: **the block-list attention read is depth-invariant
above saturation**, which the banked table confirms to the tenth of a millisecond across a 1.5x
depth range. An A/B of the KV read pattern is therefore valid at 32,768 — a ~3 minute prefill —
and transfers to 262,144, where a per-arm fill is ~23 minutes. This lane's KV cells are sized
on that, and the claim is stated so it can be falsified: one confirmation rung at 262,144 is
owed before any KV read-path number is quoted at the target window.

`qsa.idx_host` is the opposite case — O(n_blocks) by construction, so its lever needs depth and
gets no shallow proxy.

### 1b. Extrapolation to the target window (stated as extrapolation, not measured)

Holding the flat sections and extending the two linear ones to 262,144:
`ple.host_ngram_gather` ~12.7 ms, `qsa.idx_host` ~5.7 ms, flat remainder ~17.0 ms, total
~45.7 ms profiled. Shares would be ple ~28%, **`qsa.sdpa` ~22.5%**, **`qsa.idx_host` ~12.5%**.
So the two sections this lane owns are **~35% of the profiled decode token at the target
window**. No 262,144 decode profile with `idxsel` armed exists yet; that rung is owed.

## 2. Static receipt: the read patterns, counted in SASS (no GPU, no box)

`nvcc 13.1, -arch=sm_120 -cubin` on the rig, `cuobjdump -sass`, inner loops identified by
backward branch and counted programmatically. The box's RTX PRO 6000 Blackwell and the rig's
5090 are both sm_120, so this compiles the shipping instruction stream. Static counts are per
UNROLLED iteration (the score loop unrolls by 8); they are instruction-stream facts, not
timings, and they are exactly the right instrument for a transaction-count question.

### 2a. Score phase (K dots) — the sign-flip mechanism

Per 8 K elements of the innermost loop:

| kernel | instrs | KV-cache loads | `q[d]` loads | **redundant fp16 scale loads** |
|---|---|---|---|---|
| `sdpa_blocklist_f32` (the f32 twin) | **37** | 8 (`LDG.E`) | 8 | — |
| `q4e_sdpa_blocklist_q8q5` (ship) | **120** | 8 (`LDG.E.S8`) | 8 | **8 (`LDG.E.U16`)** |
| `q4e_sdpa_blocklist_q8q5_hoist` (new) | **52** | 8 (`LDG.E.S8`) | 8 | **0** (1 per 32-elem block) |

`q4e_deq_q8` recomputes the block pointer from the element index, so the fp16 block scale is
re-loaded **once per element** — 32 redundant loads per 32-element block. Phase 1 is
thread-per-position (`for i = tid; i < count; i += blockDim.x`), so lanes of a warp sit on 32
DIFFERENT selected tokens, `k_tok_bytes = 544 B` apart. Every load instruction therefore replays
32 ways into 32 distinct 32-byte sectors, and the transaction count is set by the load
INSTRUCTION count, not by the byte count.

That is the whole sign flip, and it is arithmetic rather than a hypothesis:

- q8_0 reads **544 B** per token per layer where f32 reads 2,048 B — **3.76x fewer bytes**.
- q8_0 issues **2 KV-cache load instructions per K element** where f32 issues **1** — **2x the
  transactions**.
- So the quantized cache moves *more* sector traffic than the f32 cache it replaces, plus a
  3.2x heavier instruction stream in the hottest loop of the largest decode section.

It also explains why the original flip receipt had the OPPOSITE sign (+1.3% at a shallow fill,
KVQ-CELL round 1): below selection saturation almost no rows are read, so phase 1 barely runs
and only the smaller byte count shows. **-7.4% at 100k and +1.3% shallow are the same kernel
measured on different sides of selection saturation.**

### 2b. V phase (weighted V) — a receipted DEAD END

The same hoist was written for phase 3 (the per-thread block offset, packed-byte index and
nibble shift are loop invariants `q4e_deq_q5` recomputes once per selected position) and the
SASS rejected it. Per selected position: the base loop issues `{1 U16, 1 U8, 0.5 LDG.E}` and the
hoisted form issued `{1 U8, 2 LDG.E}`. The base already lets the compiler merge `d` and `m` into
one 32-bit load; spelling the reads out separately cost that merge.

Phase 3 also had no transaction win available: it is `for d = tid; d < head_dim; d += blockDim.x`
with the position loop inside, so adjacent lanes take adjacent `e` and a warp covers one 24-byte
q5_1 block in ONE sector. **Phase 3 is already coalesced, and q5_1 beats f32 there** (24 B per
block per position against f32's 128 B for the same 32 dims). The kvq penalty is concentrated
entirely in phase 1. `kvhoist` is therefore deliberately a ONE-VARIABLE seam, and phase 3 in the
new kernel is the old phase 3 verbatim — confirmed by the SASS, whose phase-3 loops are
identical between the two kernels (122 / 264 instructions with identical `LDG` breakdowns).

### 2c. Pooled-key reads — 8x sector amplification on the one array whose size IS the context

`qsa_index_score_f32` is thread-per-block over the pooled plane:

```
const float* k = pooled + (size_t)block * head_dim;   // head_dim = 128
for (int d = 0; d < head_dim; d++) dot = __fadd_rn(dot, __fmul_rn(qh[d], k[d]));
```

Lane L holds block `block0 + L`, so lanes are `head_dim * 4 = 512 B` apart. A warp's single
`k[d]` load touches **32 distinct sectors and moves 1,024 B to use 128 B — 8x amplification**,
for every element, every head, every row. At the 262,144 target the pooled plane is 65,536
blocks x 128 floats = **33.5 MB read per scored row per layer**, x 12 QSA layers.

`qsa_index_score_f32_t` reads a **dim-major** plane, `pooled_t[d*pitch + block]`: the same 32
lanes read 32 consecutive floats — 128 B, 4 sectors, zero waste. Same loop order, same explicit
`__fmul_rn`/`__fadd_rn`/`__fdiv_rn` (the kernel's existing comment records that `-fmad=true`
contraction here flipped near-tie top-k picks by 1 ULP; those intrinsics are preserved
verbatim). Inner-loop instruction cost of the transposed addressing: 46 instructions per 8
elements against 37 — about one extra address op per element, traded for 8x less sector traffic.

## 3. The levers, their exactness class, and their state

| seam | what moves | class | default | state |
|---|---|---|---|---|
| `kvhoist` | fp16 K block scale hoisted out of the score element loop | **BIT-IDENTICAL** (same product, same `acc +=` order, phases 2-3 verbatim) | **OFF** by design | implemented, static receipt banked, oracle + A/B owed |
| `poolT` | pooled device mirror gains a dim-major plane; score kernel reads it | **BIT-IDENTICAL** (only the address changes) | **OFF** by design | implemented, oracle + A/B owed |
| `kvsplit3` | split block-list attention into scores / softmax / weightedV kernels so phase 1 gets a 17x larger grid | BIT-IDENTICAL (per-phase orders unchanged) | — | designed, §5 |
| expert speculation | L2 prefetch of the predicted expert set during the router GEMV | bit-UNCHANGED by construction (a prefetch is a hint) | — | **priced, not built — §4** |

Both implemented seams are **mid-run flippable** (no allocation latch), which matters at these
depths: agent B's `set_seam` lets one filled state carry both arms of an interleaved A/B instead
of paying a fresh 23-minute prefill per arm. `poolT` maintains BOTH layouts unconditionally so
that (a) the append cost is identical in both arms and the measurement isolates the read
pattern, and (b) there is no stale-plane failure mode when an arm sits OFF while rows are
appended — a stale pooled plane scores stale keys, which reads as plausible output rather than
as a failure. Instrument cost stated: one extra pooled plane (33.5 MB at the target geometry,
1.6% of the ~2 GB free there) plus a transpose over the delta (512 rows per prefill chunk, 0-1
per decode step). The losing layout goes away in the commit that records the verdict.

## 3a. The two cells, the measurement rules they had to be rewritten under, and one WITHDRAWN row

Both A/B cells use the within-fill interleaved instrument (`--ladder-ab-seam`, lead-flipped, x3
with automatic escalation), which is what makes a decode-only lever affordable at these depths:
one prefill carries both arms instead of one prefill per arm.

**A row this lane produced and then withdrew, because the withdrawal is the more useful receipt.**
The first `kvhoist` cell read, at a 32,768 fill on the ship defaults with `idxsel` armed:

```
# ladder-rung  32768  prefill 169.8 s (16 chunks)  26.2 ms median  37.69 tok/s  spread 0.19% (x3)
# ladder-ab-verdict rung=32768 seam=kvhoist off_ms=26.16 on_ms=25.44 speedup=1.0284x delta_pct=2.76%
```

+2.76%, x3, within-arm spread 0.19%, no escalation owed. **It is WITHDRAWN as suspect and is not
banked as a number.** It was taken with the lock wrapped around the timed ROUNDS only, so the
prefills of all three lever lanes ran unlocked and a sibling lane was computing on the other card
throughout. The interleaved design with lead-flip means the *direction* survives contention (both
arms eat the same neighbour in the same proportion), so it stands as a signal that the lever is
real and positive; the *magnitude* does not, and quoting 2.76% would be quoting a number nobody
can defend. Re-measured under `flock -x` around the ENTIRE invocation.

Three measurement rules changed mid-cell and all three are now obeyed by `q4e-qC2.sh`:

- **`flock -x` around load + prefill + timing**, not just the timed rounds. This also fixes the
  VRAM race for free: nobody allocates while somebody else holds the lock.
- **`MEMRA_Q4E_MEASURE_LOCK` stays unset** in these cells — one lock mode per cell, because a
  shell holding the lock whose child instrument then requests it on the same path blocks on its
  own ancestor forever.
- **`CUDA_VISIBLE_DEVICES=1`.** Two cards hold at most TWO lanes by construction: a loaded model
  is 89,971 MiB and a filled 262k rung peaks at 95,805 of 97,887, so a third lane cannot have a
  card. Card 0 went back to the indexer/graphs lane (released at 0 MiB with no compute app — the
  kill was by exact pid, never by basename, which is a known orphaned-VRAM trap here) at the cost
  of a 131,072 rung that died at fill 114,832 after ~7 minutes of prefill. Correct trade.

Cell shapes, sized to be short turns rather than one long monopoly:

| cell | rungs | seam | why that depth |
|---|---|---|---|
| `C1b-kvhoist` | 32,768 | `kvhoist` | valid SHALLOW because `qsa.sdpa` is depth-invariant (§1a); ~10 min exclusive instead of ~25 |
| `C2b-poolT` | 32,768 + 131,072, one fill | `poolT` | the pooled score is O(n_blocks), so the FALSIFIABLE prediction is that the delta GROWS between the rungs; a flat delta kills the lever |

Script banked at `../round2-box-receipts/bin/q4e-qC2.sh`. **STATUS: both cells are PARKED in the
`flock -x` queue behind the other two lever lanes, unmeasured.** Both cards were full at the time
of writing (3,596 and 1,252 MiB free against a 89,971 MiB load), which is the two-lane capacity
ceiling, not a fault. The cells are unattended and self-reporting: they will guard, measure and
write their own verdicts whenever a card frees.

The queue carries a **capacity guard**, because winning the lock is not the same as having a card:
a lock holder can release the lock and remain resident, and the cell would then spend its scarce
turn OOMing. It waits up to 600 s for 91,000 MiB free on card 1 and exits 90 loudly instead of
failing obscurely. **Both of its arms were executed rather than written** — with the real
`nvidia-smi` parse, `need=100` returns rc=0 and `need=91000` returns rc=1 on both devices — because
a guard proven only on its red arm may be unconditionally red, which is the same defect as
unconditionally green, and this lane has had three diagnostics-that-were-themselves-silent in one
night.

**What is owed, stated plainly:** neither seam has a defensible perf number. `kvhoist` has one
withdrawn, contended signal (+2.76%, direction only). `poolT` has none. Both have their exactness
receipts and their static-analysis case; neither may be quoted as a perf result, and neither may
have its default flipped, until these cells land.

## 4. Expert speculation: a RECEIPTED NON-LEVER (dropped, not deferred)

The owner's lever is "speculatively stage the predicted expert set while the router computes,
correct on mismatch, output bit-unchanged". The structure is right; at this model's geometry the
prize is not there, and the number that says so comes from the banked profile plus the resident
bank layout, at no GPU cost.

**There is nothing to stage.** The serving expert bank is `BankHalf::Nvfp4` — fully VRAM-resident
`[512, 1280, 2560]` gate_up + `[512, 2560, 640]` down per layer, read directly at
`((e*out_f + o) * in_f/2)` by `launch_nvfp4_sel_matvec`. No gather buffer, no per-token H2D, no
streaming. (`HostBf16` does upload per forward, and is documented in-tree as "never a serving
configuration".) So "staging" can only mean warming L2, not moving weights.

**The window is 1.1 ms and the prize is inside it.** The only latency a speculative prefetch can
hide is the router GEMV it overlaps: `moe.router` is **1.1 ms, 3.1%** of the profiled decode
token, flat in depth. That is a hard ceiling — a prefetch cannot save more time than the
computation it hides behind. The bytes agree that this is generous: 10 selected experts x
~2.76 MB (NVFP4 codes + UE4M3 scales) = ~27.6 MB per layer, x 48 layers = **~1.33 GB per decode
token**, which is ~0.74 ms at the card's bandwidth against `moe.sel_grouped`'s measured 2.6 ms —
the matvec is already within ~3.5x of its byte roofline, so the cold-miss stall available to
hide is a fraction of 1.1 ms. **Realistic ceiling well under 1% of the token.**

**And the decision would need a new sync it does not have.** With `routerdev` armed the route
never leaves the card, so a host-side prefetch decision requires a readback — and the only
readbacks available (`MEMRA_Q4E_ROUTER_AUDIT`, `MEMRA_Q4E_ROUTE_SYNC`) are documented
diagnostics, not serving arms. A device-side prefetch keyed off the previous token's parked `sel`
is the only sound form, and it buys at most that sub-1%.

### The structural reason, which is stronger than the arithmetic

The numbers above bound the prize. The reason it is not a prize at all is structural, and it is
worth stating in the form that kills the lever rather than merely shrinks it:

**Speculative staging hides STAGING latency. At the 262,144 window there is no staging.** The
expert bank is fully device-resident, so the "prediction" would be predicting which resident rows
a matvec is about to read — and a correct prediction saves nothing, because the read was never
waiting on a transfer. The mechanism the owner's lever describes requires a bank that is NOT
wholly resident on the computing card, i.e. a peer-resident or host-resident bank. On this model
that means **TP2** — and TP2 is a measured DEPTH REGRESSION in this lane (card 0 carries all state
growth, +2,784 MiB post-load, and it OOMs during the fill below 100k while one card reaches
~731k), so it cannot reach the target window at all. The lever's precondition and the product's
target are mutually exclusive here.

**Verdict: DROPPED as a receipted dead end, not deferred.** Two consequences to carry forward so
this is not re-derived:

- **It becomes live again only if a peer- or host-resident expert bank ever becomes serveable at
  this window.** That is the condition to re-open on, and nothing weaker.
- **The expert-placement lane keeps its value, for a DIFFERENT reason.** Placement is not a
  latency-hiding lever and must not inherit this one's rationale. Its value is residency and
  cross-card fan-out: PROFILE-10 measured **99.93% of layer-tokens touching BOTH cards** under an
  even split, with the peer taking 51.40% of dispatched expert slots, so co-activation placement
  reduces how much crosses, not how well a guess is hidden. Same trace input, different claim.

The per-layer hit rate is still collected, and is still worth collecting for placement — but it is
now explicitly NOT a perf-lever input. `moe-hit-rate.py` (beside this file's receipts) reads the
shared trace and reports per-layer hit rate plus co-activated pairs; it computes the hit rate over
`t == 1` DECODE lines only, because a prefill line carries many tokens' picks concatenated with no
delimiter and has no well-defined "previous token".

**What IS worth collecting, and is cheap: the per-layer hit rate.** It is the input the owner's
co-activation placement doctrine needs, independent of whether a prefetch ever ships. One honest
gap found while pricing this and worth stating because it silently blanks the placement lane's
input: `trace_moe_routes` (the shared `MEMRA_MOE_TRACE` format, `<layer> <t> <csv>`) is called
**only from the TP2 paths**. The single-card device-routed default emits NOTHING — `route_topk_device`
does the `ROUTER_AUDIT` readback and never calls the tracer — and the single-card host-routed
`moe_forward` has no call either. The box's `~/realgate/traces/` is empty for exactly this
reason. Wiring the existing tap onto the audit readback (diagnostic-gated, zero new syncs when
the audit is already armed) is the cheap fix and does not duplicate agent A's tap; it makes it
fire at all.

## 5. What the residual profile says next

1. **`qsa.sdpa`'s remaining cost is GRID SIZE, and it is bit-identically fixable.** At decode the
   kernel launches `grid = (n_head, T) = (24, 1)` with `block = (128,1,1)` — **24 CTAs, 3,072
   threads, for the whole card.** The important half is that this is not a resource limit it could
   grow into: measured from the sm_120 cubin, `q4e_sdpa_blocklist_q8q5` uses `REG:44` and
   `SHARED:1024` static plus `max_count*8` = 16.4 KB dynamic at a saturated selection, which would
   allow many CTAs per SM. **The kernel occupies at most 24 SMs of a card that has far more, and
   the binding constraint is the grid, not registers or smem.** On top of that, phase 2 (max / exp
   / normalize over ~2,052 entries) runs on `tid == 0` alone while 127 threads idle.

   Phase 1's ~49,000 independent dots (24 heads x ~2,052 positions) are what a larger grid would
   absorb. Splitting the kernel into scores / softmax / weightedV lets phase 1 take
   `grid = (24, ceil(2052/128))` = **408 CTAs**, and phase 3 can additionally split over dims
   (`grid = (24, 256/64)` = 96 CTAs, one dim per thread) — **with every per-element accumulation
   order unchanged in both**, so the bar stays bit-identity rather than a band. Phase 3's sum over
   POSITIONS is the hard floor: it must stay sequential per dim, so it cannot be split further than
   one CTA per dim group. Cost is a global scores buffer (`T x n_head x max_count` floats) and two
   extra launches per layer per token. This is the designed `kvsplit3` lever, it is fully specified
   here, and it is the next thing to build once the two implemented seams have verdicts —
   deliberately NOT started ahead of them, so a third unmeasured seam does not pile up behind two.
2. **A split-plane q8_0 K cache remains the only way to widen the loads.** With the scale hoisted,
   phase 1 still issues one 1-byte load per element because `blk + 2` is 4-byte aligned for only
   half the blocks (`(34b+2) mod 4` alternates 2, 0), so `int4` loads are illegal on the 34-byte
   block layout. A quants plane (`[T, 512]` int8, 128-byte aligned per head slice) plus a scale
   plane (`[T, 16]` half) makes them legal and is a pure permutation. It is deprioritised behind
   `kvsplit3` for one reason: the layout latches at state allocation, so it cannot ride the
   mid-run A/B and each arm costs a full fill.
3. **The 262,144 rung with `idxsel` armed is owed** — every number at the target window in this
   file is an extrapolation from 100k/131k/150k and is labelled as one.
4. **`moe.sel_grouped` at 2.6 ms against a ~0.74 ms byte roofline** is a ~3.5x gap in this lane's
   territory that nothing above addresses. It is flat in depth, so it is not a 262k-specific bug,
   but it is the largest MoE term and the next MoE question after the trace tap is wired.
</content>
