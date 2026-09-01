# qwen4_exp round 2, work item 3 — the long-context ladder and the 1M affordability verdict

Boxes: the round-2 lane boxes, all **2x RTX PRO 6000 Blackwell Server Edition, 97,887 MiB,
600 W** (same class as round 1, so round-1 numbers are the comparison). Artifact
`q48fn-yarn1m` (hardlink twin of `q48fn-nvfp4`, `rope_type=yarn factor=3.814697265625
original=262144 mpe=1000000`). Corpus `corpus_commit=69dc19a410fc91756fc5df1d4714f251c2fe71aa`,
1,150,000 tokens from 297 files — quote that commit beside any continuation row; round-1's
corpus was 295 files at a different commit, so continuation coherence is comparable within a
corpus commit and not across one.

## SCOPE CHANGE 2026-08-31 (owner): the target is 262,144, and 1M is DROPPED

The product context is **262,144 — the model's native window** — and the goal is best
performance THERE, not maximum reachable depth. No 600k rung, no 1M rung, no YaRN long-context
extension work. YaRN stays a **banked capability** (round-1 wiring gates green, factor-1.0
byte-identical on the real checkpoint) and is **not a target**; nothing in this file should be
read as a reason to resume 1M.

The 1M measurements below are kept because they are what closed the question, not because 1M is
being pursued: they are the receipts behind "1M is not reachable on this hardware", and §2's
ceiling arithmetic is what makes 262k affordable at all.

## THE 262k TABLE (the deliverable), ship defaults, one card

`kvq` K=`q8_0` / V=`q5_1`, `idxq q8`, device router + idxcache, yarn factor 3.8147, chunk 2048,
`corpus_commit=84a9d5b6a`. Both rows self-evidencing via
`# cache kv_quant=q8_0/q5_1 idxq=q8 golden_pin=false`.

| depth | tok/s | ms/token (mean/med/p90) | prefill wall | chunks | card-0 VRAM | free | card 1 | spread |
|---|---|---|---|---|---|---|---|---|
| 100,000 | **33.62** | 29.7 / 29.7 / 29.8 | 561.4 s | 49 | 92,957 MiB | 4,930 MiB | 3 MiB | 0.09% (x3) |
| **262,144** | **15.21** | **65.7 / 65.0 / 66.6** | **4,779.1 s** | 128 | **95,805 MiB** | **2,082 MiB** | 3 MiB | 2.56% (**x5**) |

Both `looped=false`, host RSS ~167,8xx MiB. The 262k row's spread was 2.56%, over the 0.5%
threshold, so the instrument **escalated to x5 by itself** — medians
`[64.9, 64.2, 65.1, 65.9, 65.2]` — which is the interleaved protocol working rather than being
remembered.

**Two things this table says that round 1 did not.**

1. **Decode is NOT flat in depth over the target window.** Round 1's headline was near-flat
   decode (22.3 ms at 4k -> 28.1 ms at 100k, +26% over 24x depth) on the bounded-attention
   argument. From 100k to 262k, decode goes **29.7 -> 65.7 ms, 2.2x for 2.6x depth** — close to
   linear. The flatness result is real but **scope-bounded to <=100k**, which is where it was
   measured; at the native window the model costs 4.4 tok/s per 100k of context.
2. **Prefill is the dominant cost at 262k, and it is superlinear.** 561 s -> 4,779 s for 2.6x
   the tokens is **8.5x the wall**. That is the ~131k step-up (§4a) landing squarely inside the
   product window, which is why it is now a first-class perf bug and not a deep-context
   curiosity.

**The KV trade at 262k, stated honestly:** at this depth kvq is **memory-REQUIRED, not merely
memory-preferred** — the f32 arm does not allocate a 262,144-token state at all (§2), so there
is no f32 twin to compare against and the ~7% decode cost measured at 100k has no alternative
at the target window. "Memory-required vs perf-optimal" resolves to: required. What remains
measurable at 262k is the *other* cache knob, `idxq` (q8 vs bf16), and that arm is not yet run.

## 1. The ship-default ceiling, measured by state allocation

Single card, ascending. State allocation is the honest first gate: a capacity that cannot be
allocated cannot be filled, and it refuses in seconds instead of after an hour of prefill.

| capacity | card-0 at state-alloc | free | verdict |
|---|---|---|---|
| 131,072 | **91,475 MiB** | 6,412 MiB | allocates |
| 262,144 | **92,883 MiB** | 5,004 MiB | allocates |
| 600,000 | **96,467 MiB** | 1,420 MiB | allocates |
| 1,000,000 | — | — | **OUT_OF_MEMORY at state alloc** |

`(96,467 - 89,971) / 600,000` = **11.08 KiB/token**, matching KVQ-CELL's analytic ~11.1
KiB/token to three digits. Ceiling from that rate: `7,916 MiB / 11.08 KiB` ≈ **731,000 tokens**.

## 2. What kvq actually buys: the ceiling goes 165k -> 731k

The same probe on the f32 arm (`MEMRA_Q4E_SEAMS=kvq=0`):

| capacity | f32 card-0 at state-alloc | verdict |
|---|---|---|
| 131,072 | 96,243 MiB | allocates (and reproduces round 1's 96,243 for 131,288 exactly) |
| 160,000 | — | OUT_OF_MEMORY |
| 200,000 | — | OUT_OF_MEMORY |
| 262,144 | — | OUT_OF_MEMORY |

f32 rate: `(96,243 - 89,971) / 131,072` = **49.0 KiB/token** — round 1 measured 48.9. So:

| arm | KiB/token | single-card ceiling |
|---|---|---|
| f32 | 49.0 | **~165k tokens** |
| kvq q8_0/q5_1 + idxq q8 | 11.08 | **~731k tokens** |

**kvq multiplies usable context depth by ~4.4x.** That is the round-2 headline and it is a
real product number: it moves this model from "cannot serve 262k" to "can serve 600k on one
card". It still does not reach 1M.

## 3. TP2 makes depth WORSE, and the mechanism is measured

`--tp2 --ladder-tp2` at rung 100000, chunk 2048:

```
# vram post-load          0, 92755 MiB | 1, 40211 MiB
# ladder state-allocated  0, 95283 MiB | 1, 42611 MiB
# ladder-progress fill=16384  elapsed_s=63.7   0, 96435 MiB | 1, 43603 MiB
# ladder-progress fill=32768  elapsed_s=130.4  0, 96563 MiB | 1, 43603 MiB
# ladder-progress fill=49152  elapsed_s=203.1  0, 96819 MiB | 1, 43603 MiB
# ladder-progress fill=65536  elapsed_s=278.7  0, 96851 MiB | 1, 43603 MiB
Error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

Two measured facts explain it:

1. **TP2 post-load costs card 0 2,784 MiB MORE than single-card** (92,755 vs 89,971), so it
   starts with 5,132 MiB free instead of 7,916.
2. **Card 1 is FLAT at 43,603 MiB from fill=16384 onward** while card 0 carries every
   additional byte. On this program the TP2 split does not move the growing cache off the
   binding card; it only adds shard staging to it.

Card 1 sits with **~54 GiB free that long context cannot use.** So TP2's ceiling is *below*
100k while the single card's is ~731k. For depth, TP2 is a regression, and the round-1
prediction that TP2 residency was the route to 1M is **not borne out** — it was reasoned from
"trunk halves ~44 GiB + local KV halves ~23.9 GiB = ~68 GiB/card", and the measured shard is
not that shape (card 0 keeps the full resident bank; card 1 gets a 40 GiB half-bank copy).

## 4. The timed rungs at 100,000 tokens — and kvq's sign FLIPS with depth

Both arms at the same depth on the same card class. This is the row work item 3 owed, and it
changes what this lane should say about kvq.

| arm | tok/s | ms/token (mean/med/p90) | prefill wall | card-0 VRAM | free | spread |
|---|---|---|---|---|---|---|
| **kvq q8_0/q5_1 + idxq q8 (ship default)** | **33.62** | 29.7 / 29.7 / 29.8 | 561.4 s | **92,957 MiB** | **4,930 MiB** | 0.09% |
| f32 (round 2, pre-pin-fix) | 36.30 | 27.6 / 27.6 / 27.7 | 523.3 s | 97,213 MiB | 674 MiB | 0.44% |
| f32 (round 1) | 35.56 | 28.1 / 28.1 / 28.2 | 501.5 s | 96,381 MiB | 1,506 MiB | — |

Both round-2 rows: `looped=false`, host RSS ~167,8xx MiB, 49 chunks, `rounds=3x12`; spreads
0.09% and 0.44%, both under 0.5%, so the interleaved x3 stands and no x5 escalation is owed.
The ship-default row's header carries `# cache kv_quant=q8_0/q5_1 idxq=q8 golden_pin=false`, so
it is self-evidencing. Corpus commits differ between the two rows (84a9d5b6a vs 69dc19a41), so
the continuation TEXT differs; depth, token count and chunking are identical.

**FINDING — kvq is FASTER shallow and ~7% SLOWER at depth.** KVQ-CELL's flip receipt records
"the quantized cache measures FASTER: 13.36-13.39 vs 13.53-13.57 ms/token". At a 100,000-token
fill the sign reverses: **33.62 vs 36.30 tok/s, -7.4%**, prefill wall -7.3% (561.4 vs 523.3 s).
The mechanism is consistent with the design: dequant cost scales with the number of KV rows
READ per token, and the block-list path reads up to ~2,052 rows/token at depth, while the flip
was measured at a shallow fill where almost none are read. The flip DECISION is not wrong — but
its perf claim is **scope-bounded to shallow fills** and must not be quoted at depth.

**The trade, stated as a trade:** kvq costs ~7% decode and ~7% prefill at 100k, and buys
**4,256 MiB of card-0 headroom at that depth** plus a **4.4x higher ceiling** (~165k -> ~731k).
For a long-context product that is plainly the right side of the trade; for a short-prompt
product it is not free, and not the "free win" the shallow row implies.

## 4a. FINDING: prefill cost per chunk STEPS UP ~4.8x at ~131k depth

From the 262,144 rung's own `# ladder-progress` lines (chunk 2048, ship defaults, one card),
per-16,384-token segment wall:

| fill reached | cumulative s | segment s | s per 16k |
|---|---|---|---|
| 16,384 | 84.4 | 84.4 | 84 |
| 32,768 | 169.8 | 85.4 | 85 |
| 49,152 | 259.9 | 90.1 | 90 |
| 65,536 | 352.6 | 92.7 | 93 |
| 81,920 | 449.2 | 96.6 | 97 |
| 98,304 | 548.5 | 99.3 | 99 |
| 114,688 | 650.9 | 102.4 | 102 |
| **131,072** | **756.3** | **105.4** | **105** |
| **147,456** | **1231.8** | **475.5** | **475** |
| **163,840** | **1714.1** | **482.3** | **482** |

Below ~131k the segment wall grows gently and almost linearly (84 -> 105 s per 16k across an 8x
depth increase, +25%). Between 131,072 and 147,456 it **jumps 4.5x to 475 s**, and the next
segment confirms the new level at 482 s rather than continuing to climb. So this is a **step**,
not a curve.

Why it matters: round 1's headline for this model was that decode is near-flat in depth (22.3
ms at 4k -> 28.1 ms at 100k) because QSA attention is bounded. That result is about DECODE, and
round 1 only ever prefilled to 100,000 — i.e. it never crossed this step. **Prefill is not flat
in depth, and the discontinuity sits just past the deepest depth any previous receipt in this
lane reached.** A 262k prefill is therefore ~77 minutes rather than the ~25 the sub-131k rate
predicts, and a 600k prefill extrapolates to ~3.6 hours at the post-step rate, not ~56 minutes.
That changes the economics of deep-context serving far more than the decode rate does.

Not yet attributed — stated as measured, with candidates named rather than guessed at: the step
lands suspiciously close to 2^17, and the plausible mechanisms are the QSA selection horizon
crossing into non-full rows (the `longatt` AUTO engagement condition), the indexer's pooled-key
device mirror crossing a sizing boundary, or host-side raw-key cache growth. Distinguishing
them wants a `--profile` section run at 120k and 150k, which is one cheap cell and is the first
thing to run on the next box.

## 4c. DIAGNOSIS of the step: it is HOST-SIDE indexer work (`qsa.idx_host`), not GPU attention

`--ladder 100000,131072,150000 --profile 1` profiles the first chunk (t=2048) of each segment,
so the three columns are at fills of ~0, ~100,000 and ~131,072 — straddling the step.
Receipt: `kvq2/ladder-r2prof-step.tsv`.

| section | fill ~0 | fill ~100,000 | fill ~131,072 |
|---|---|---|---|
| `moe.sel_grouped` | 2598.8 ms | 2512.4 ms | 2563.4 ms |
| `moe.router` | 1417.3 ms | 1513.2 ms | 1478.4 ms |
| `hyper.read` | 1387.0 ms | 1321.0 ms | 1357.3 ms |
| `gdn.proj` | 1010.2 ms | 984.5 ms | 990.4 ms |
| `gdn.conv_scan` | 878.0 ms | 863.0 ms | 863.5 ms |
| `gdn.norm_gate_out` | 373.5 ms | 347.7 ms | 353.7 ms |
| `qsa.proj` | 267.8 ms | 261.3 ms | 261.3 ms |
| `moe.shared` | 265.2 ms | 256.0 ms | 259.2 ms |
| `qsa.sdpa` | 930.2 ms | **2006.0 ms** | 2153.7 ms |
| **`qsa.idx_host`** | **absent** | **2709.9 ms (20.9%)** | **51235.1 ms (83.0%)** |

**Every GPU section is FLAT across the step.** Eight of ten sections move by less than 4%
between fill 100k and 131k. Only two move at all:

- `qsa.sdpa` steps 930 -> 2006 ms and then holds — this is the expected bounded-attention
  transition as rows stop being structurally full and the block-list path engages. It is a 2.2x
  step on a 10%-share section, so it explains almost none of the wall.
- **`qsa.idx_host` goes 2,709.9 -> 51,235.1 ms, an 18.9x jump for a 1.31x depth increase, and
  becomes 83% of the entire chunk.** That is the step.

`qsa.idx_host` is the **host** half of the QSA indexer. Per the `set_idx_dev` FLAGS row the
device scorer computes the block scores on the GPU, and then **"the pinned top-k runs on the
host over the dtoh'd scores"**. The block count grows linearly with fill, and this is host work
that the GPU waits on, which is exactly the shape of the observed cliff: no VRAM change, no GPU
section change, and a wall that steps and then holds.

### The geometry, read from the code — and a correction

An earlier note in this file guessed the threshold sat "close to 2^17". **That was wrong, and the
code says so.** ARCH.md gives the QSA indexer micro-block size as **4** with a budget of **512
blocks = 2,048 tokens**, and the fast-path condition is
`all_full = (base_pos + t) / block_size <= budget_blocks`. So:

- The structural fast path holds only while `base_pos + t <= 2048`. The very first t=2048 chunk
  is exactly at the boundary, which is why `qsa.idx_host` is **absent** in the fill~0 column.
- From fill ~2k onward, **every chunk takes the scoring path**, for all 12 QSA layers. That is
  confirmed by `calls=12` on `qsa.idx_host` in both deep columns.
- Per chunk the work is `t` rows x `complete = fill/4` blocks, so it is **O(fill) per chunk and
  quadratic across a prefill** — exactly what the `set_idx_dev` comment says the device scorer
  was introduced to attack ("the host twin is O(context) per token per layer ... quadratic across
  a long prefill").

So the *general* growth is explained and expected. What is NOT explained is the **sharpness**:
2,709.9 -> 51,235.1 ms is **18.9x for 1.31x more blocks**, where the quadratic model predicts
~1.7x. Two mechanisms in the code could add a step but neither is nearly big enough on its own:
the score slab is sub-batched at 32 M floats, so the batch count goes 2 -> 3 across this range
(1.5x), and each sub-batch ends in a **blocking `dtoh` of up to 128 MB** (256 -> 384 MB total).
The residual factor of ~12x is unexplained.

### The bracket run: it IS a discontinuity, and it sits between 120,000 and 131,072

`--ladder 110000,120000,140000 --profile 1` (`kvq2/ladder-r2prof-bracket.tsv`) adds two more
`qsa.idx_host` points. Combined with the first run, four fills:

| fill of the profiled chunk | `qsa.idx_host` | share | growth vs previous |
|---|---|---|---|
| ~0 | **absent** (structural fast path) | — | — |
| ~100,000 | 2,709.9 ms | 20.9% | — |
| ~110,000 | 2,914.7 ms | 22.4% | 1.08x for 1.10x depth |
| ~120,000 | 3,198.7 ms | 24.0% | 1.10x for 1.09x depth |
| **~131,072** | **51,235.1 ms** | **83.0%** | **16.0x for 1.09x depth** |

**Below 131,072 the growth is linear in depth and unremarkable** (+18% across 100k -> 120k, for
+20% depth — the expected O(fill)-per-chunk behavior). Then it jumps **16x for a 9% depth
increase.** That is a genuine discontinuity, not a steep curve, and it is now bracketed to within
11,072 tokens by four measured points rather than inferred from two.

The decode cliff sits in the same place, from the same two runs' rung rows:

| depth | decode ms/token | tok/s | spread |
|---|---|---|---|
| 110,000 | 30.6 | 32.68 | 0.35% (x3) |
| 120,000 | 31.2 | 32.01 | 0.32% (x3) |
| **131,072** | **57.4** | **17.41** | 2.74% (x5) |
| 140,000 | 55.4 | 18.04 | 0.93% (x5) |
| 262,144 | 65.7 | 15.21 | 2.56% (x5) |

So the whole 262k window is priced by one boundary: **below it ~32 tok/s, above it ~15-18 tok/s.**

### The leading hypothesis, with its arithmetic — and it is UNTESTED

131,072 is not a round number by accident: `131,072 / 4 = 32,768` blocks, and the score slab is
sub-batched at **32 M floats**, so rows per sub-batch is `32,000,000 / complete_blocks`:

- at `complete = 30,000` (fill 120,000): 1,066 rows/batch -> `ceil(2048/1066)` = **2 sub-batches**
- at `complete = 32,768` (fill 131,072): **1,024 rows/batch exactly** -> `2048/1024` = **2**, and
  one block more makes it **3**

**The cliff coincides exactly with the 2 -> 3 sub-batch transition.** Each sub-batch does
`e.uninit(n * batch_max)` — a fresh device allocation of up to 128 MB — and ends in a **blocking
`dtoh`**. At these depths card 0 has only ~2-4 GB free, so a third large transient allocation per
layer per chunk (x12 layers x 128 chunks) is a plausible allocator-thrash trigger. A 1.5x work
increase producing a 16x wall is the signature of allocation or synchronization behaviour, not of
arithmetic.

### The hypothesis was TESTED and it is DEAD

`MEMRA_Q4E_IDX_SCORE_CAP_MF` (new, default 32 = today's behaviour exactly, FLAGS row in the same
commit) makes the cap tunable. At **128** mega-floats, fill 131,072 needs
`134,217,728 / 32,768 = 4,096` rows per sub-batch, so all 2,048 scored rows fit in **ONE**
sub-batch instead of two-going-on-three — the exact condition the hypothesis said was causing the
cliff. Receipt `kvq2/ladder-r2prof-cap128.tsv`:

| fill | `qsa.idx_host`, cap 32 (default) | `qsa.idx_host`, cap 128 | change |
|---|---|---|---|
| ~100,000 | 2,709.9 ms (20.9%) | 2,510.0 ms (20.0%) | **-7.4%** |
| ~131,072 | 51,235.1 ms (83.0%) | **49,001.0 ms (82.7%)** | **-4.4%** |

**The cliff did not move.** Eliminating the sub-batch transition entirely bought 4.4%, not the 16x
the hypothesis required. Decode agrees: 55.5 ms/token at 131,072 under cap 128 against 57.4 ms
under cap 32 — 18.02 vs 17.41 tok/s, a 3.5% difference on a 1.9x cliff.

So the sub-batch-count / transient-allocation mechanism is **ruled out**, despite an arithmetic
boundary that matched the measured cliff location to within one block. That coincidence was a
coincidence, and it is worth recording as one: it is exactly the kind of match that gets written
up as a root cause without a test.

It also rules out the *fallback* suspect named earlier in the same breath: with cap 128 the
blocking-`dtoh` count per chunk went **down** (one sub-batch instead of two or three) and the wall
did not move, so `dtoh` count is not the mechanism either.

**What the test did buy:** a real, if small, win at every depth — **-7.4% on `qsa.idx_host` at
100k and -4.4% at 131k** for a one-constant change that alters no arithmetic (identical scores,
identical selected sets; only the batching of a device launch moves). That is worth keeping on its
own merits, but it is not the cliff.

**Remaining suspects, none tested:** the host `top_blocks_ascending` top-512 per row (2,048 rows x
12 layers x `complete` scores per chunk — ~3.2 GB of host reads per chunk at 131k, which could be
crossing a host memory-bandwidth or CPU-cache boundary), the host pooled-key extension, or growth
behaviour in the host-side `Vec`s (`pooled_keys` is a `Vec<f32>` that reaches 16.7 M floats /
67 MB at exactly 2^17 x 128). Distinguishing these wants host-side profiling (perf/flamegraph on
the host thread), not another GPU cell — which is the right next step and is cheap.

**The fix direction is unchanged and does not depend on which suspect wins:** the device scorer
computes the block scores on the GPU and then dtoh's them so the **host** can run top-512 per row.
Doing the selection on the device removes the entire `qsa.idx_host` section, which is 83% of a
deep prefill chunk and the difference between ~32 and ~15-18 tok/s across the 262k window.

**Independent of the step, one thing is already safe to act on:** the device scorer computes block
scores on the GPU only to **dtoh the slab and run the top-512 per row on the host**. Finishing the
selection on the device removes both the transient slab and the blocking dtoh, which is the same
fix whichever way the hypothesis resolves — and it is the difference between ~32 and ~15 tok/s
across most of the 262k window.

**The same cliff is in DECODE.** From the same run's rung rows: 30.1 ms/token at 100,000 ->
**57.4 ms at 131,072** (1.9x for 1.31x depth) -> 58.3 ms at 150,000, and 65.7 ms at 262,144. So
decode steps between 100k and 131k too, then grows slowly. The 262k decode number is a
*post-cliff* number, which means **the cliff, not depth as such, is what sets performance across
most of the product window.**

| depth | decode ms/token | tok/s | spread |
|---|---|---|---|
| 100,000 | 30.1 | 33.18 | 0.30% (x3) |
| 131,072 | **57.4** | 17.41 | 2.74% (**x5**) |
| 150,000 | 58.3 | 17.15 | 1.62% (**x5**) |
| 262,144 | 65.7 | 15.21 | 2.56% (**x5**) |

Note the instrument escalated to x5 on its own at all three deep rungs. Absolute ms under
`--profile` are sync-bounded and inflated — shares and ratios are the signal, and the rung rows
above are from unprofiled timing.

**Why this is the top perf item for the 262k window:** if the host top-k were not on the
critical path, the sub-131k rate (~105 s per 16k tokens) extrapolated to 262,144 gives a prefill
wall of roughly **28 minutes instead of the measured 79.7**, and the decode cliff sits at ~2x.
The optimization target for the native window is `qsa.idx_host`, and it is a host-side
selection problem, not a kernel problem.

## 4b. The f32 rung detail: 100,000 tokens

| field | round 2 | round 1 (f32, single card) |
|---|---|---|
| prefill wall | **523.3 s** (49 chunks) | 501.5 s cumulative |
| decode mean / median / p90 | **27.6 / 27.6 / 27.7 ms** | 28.1 / 28.1 / 28.2 ms |
| **tok/s** | **36.30** | 35.56 |
| looped | false | false |
| card-0 VRAM at rung | 97,213 MiB (674 MiB free) | 96,381 MiB (1,506 free) |
| host RSS | 167,812 MiB | — |
| A/B | `rounds=3x12 medians=[27.7, 27.5, 27.6] spread=0.44%` | — |

Spread **0.44% < 0.5%**, so the interleaved x3 stands and no escalation to x5 is owed under
the protocol. Continuation ids banked in the receipt; `looped=false` at the rung.

**HONESTY NOTE — this row is the f32 arm.** It ran before the golden-pin defect was found
(see §5), so despite being a ladder run it measured `kv_quant=f32`. That is why its card-0
VRAM (97,213) is *higher* than round 1's at the same depth rather than lower: it is an f32
row, and the extra ~832 MiB is the round-2 binary's other caches. It is kept as the f32 twin of
the ship-default row in §4. The ceiling numbers in §1-§2 are unaffected: they were all measured after
the fix, and their receipts carry `# cache kv_quant=q8_0/q5_1 idxq=q8`.

## 5. Two instrument defects this work item found

### The gate binary was an f32-ONLY instrument

`qwen4exp_real_gate` pinned the cache seams to f32 unless `MEMRA_Q4E_SEAMS` named them, and
that pin was **unconditional** — which its own comment did not say (it scopes itself to
"reference-parity comparisons vs the transformers goldens"). So the ladder, the spec-at-depth
cells and the TP2 gates all silently measured f32 while reporting themselves as ship defaults.

Caught by a decisive probe, not by reading: the 131k state allocated at 96,243 MiB, *exactly*
round 1's f32 number, so either kvq bought nothing or kvq was not on. Forcing the seam both
ways answered it in two minutes:

```
default (no env)         131,072  state-alloc  96,243 MiB   <- the f32 number
MEMRA_Q4E_SEAMS=kvq=0    131,072  state-alloc  96,243 MiB
MEMRA_Q4E_SEAMS=kvq      131,072  state-alloc  91,475 MiB   <- 4,768 MiB less
```

Fixed two ways: the pin is scoped to runs that make a golden comparison, and **every receipt
header now carries `# cache kv_quant=... idxq=... golden_pin=... seams_env=...`**. The second
is the real fix — the pin was a bug, but what let it survive and get written up wrongly is
that an f32 run and a ship-default run produced identical-looking headers. Every
golden-comparison receipt is byte-identical across the change, so nothing banked earlier is
invalidated.

### The ascending ladder cannot bank partial rungs

`--ladder 100000,262144,600000,1000000` OOM'd **at state allocation with nothing banked**: the
ladder computes `cap = max_rung + ...` and allocates ONE state at the deepest rung's capacity
before the first rung runs (the rungs are cumulative fills of that one state). So the
reclaim-safety property the resume order was counting on — "ascending, so a reclaim keeps the
rungs already banked" — does not hold when the top rung does not fit. Worked around by one
invocation per rung. The nested-continuation design is worth keeping, but delivering its
stated property needs per-rung allocation.

Related, and sharper given this lane's reclaim rate: the ladder flushes a rung's row only on
rung **completion**, so a rung that takes longer than the mean time between reclaims banks
nothing. The 262,144 rung was lost exactly this way — its receipt has the header (with the
correct ship-default `# cache` line) and no row, after ~23 minutes of prefill.

## 6. Still owed on this work item

Stated plainly rather than implied:

- **262,144 and 600,000 timed rungs** (decode tok/s, prefill wall, continuation coherence) on
  the ship defaults. Both are proven to ALLOCATE; neither has been filled and timed. Blocked
  on hardware: six reclaims in this lane on 2026-08-31, and each replacement needs a ~174 GB
  mint download plus a build before it can measure, which has been exceeding the boxes'
  lifetimes.
- ~~A ship-default tok/s row at any depth~~ **DONE**: **33.62 tok/s at a 100,000-token fill**
  (§4), with the f32 twin beside it.
- The **flat-workspace-in-depth** check the resume order asked for is partially answered: on
  the ship defaults the per-token cost is flat at 11.08 KiB/token across 131k/262k/600k
  state allocations (a linear fit through three points with no growth term), and the TP2 fill
  rows show card-0 growth of 128-256 MiB per 16k-token chunk with no acceleration. Neither is
  the full per-chunk held-VRAM curve across a deep fill.

## 5a. Gate battery for the SCORE_CAP commit, and the 262k-depth gate item

Per-commit battery on the box with `MEMRA_Q4E_IDX_SCORE_CAP_MF` present at its default (32):

| gate | result |
|---|---|
| tiny gate, all arms | 0 failures, rc=0 |
| hidden goldens | **BYTE-IDENTICAL** to the round-2 baseline |
| verify-bit 24 | **BYTE-IDENTICAL**; `rows=24 mismatched=0 policy=bit-identity pass=true` |
| spec byte-identity 256, raw | `pass=true`; differs only in the two wall-clock columns (7.57 -> 7.54 ms, 14.14 -> 14.12) |
| `--tp2-gate 24` | **BYTE-IDENTICAL**; 24/24 argmax, worst_rel 3.016e-5 |
| `--tp2-prefill-gate 8` (calibrated) | **BYTE-IDENTICAL**; prime 1.383e-5 / band 1.4e-4, decode 1.574e-5 / band 1.6e-4, pass=true |

Four byte-identical receipts is the expected result and the point of running them: a cap change
alters no arithmetic (identical scores, identical selected sets — only the batching of a device
launch moves), so anything other than byte identity here would have been a bug in the plumbing.

**The 262k-DEPTH gate re-run is NOT done, and it is not a small item.** The owner order is that
gates re-run at the target depth so the correctness claim matches the shipped window. Every gate
above runs against the 10-token goldens probe, because that is what the banked transformers
goldens cover — there is no golden oracle at 262k and there cannot be one (round 1: "transformers
cannot run 500k here"). So a depth gate has to be built out of the self-consistency instruments
rather than out of goldens:

- **spec byte-identity at depth** is the one that ports directly: `--ladder-spec` runs
  `spec_generate_ext` at a rung and the plain-vs-spec chain comparison is same-config, so it needs
  no oracle. This is work item 4's instrument and it is committed (`--ladder-spec-shape`) but never
  run. At 262k it costs one ~80-minute prefill per shape.
- **verify-bit at depth** needs a variant that seeds its plain/verify rows from a deep state rather
  than from `goldens.input_ids`; the comparison itself is oracle-free.
- **tp2-gate / class gate at depth** need the TP2 route to reach 262k at all, and it does not
  (§3) — so at the target window these two are currently **not applicable**, which is itself the
  honest gate result rather than a gap.

Named as owed, with the cost and the blocker for each, rather than implied to be covered by the
shallow battery above.
