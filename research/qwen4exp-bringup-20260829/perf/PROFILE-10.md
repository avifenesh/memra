# qwen4_exp PROFILE-10 — round 2: TP2-prefill class gate, KV-quant at depth, and the
# instrument defects that had to be fixed before any of it could be read

Boxes: the round-2 lane boxes, **2x RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, 600 W**
each — the same card class as every prior receipt in this lane, which is what makes round-1
timings the comparison. Provider, region, instance class and instance ids are fleet state and
live in darklanes, not here.

Receipts: `../round2-box-receipts/` (`BASELINE.md`, `TP2-CLASS-GATE.md`, `LADDER.md`, raw TSVs
in `kvq2/`, per-arm stdout in `logs/`).

## 1. Box baseline: NO DELTA, and four arms byte-identical

The fourth box in this lane reproduced round 1 on eight arms with **different binaries**
(`qwen4exp_real_gate` c8b1af69d vs round 1's aa6ec1d17). Four arms did not merely land inside
a tolerance — they wrote **the same bytes** on a different physical machine: the
real-checkpoint hidden goldens (every data row, argmax 10/10, and the same 89,971 MiB
post-load), spec byte-identity at 256 tokens for **both** raw and thinkon (down to the accept
histogram), and `--tp2-gate 24` (24/24 argmax, worst_rel 3.016e-5, same 92,755/40,211 TP2 VRAM
split). Greedy first-divergence reproduced both cache patterns: `-1/8/-1/48` (kvq0) and
`-1/8/-1/26` (kvq1). Full table in `BASELINE.md`.

Two reading rules came out of it:

- **A row-count delta is not a value delta.** The tiny gate went 287 -> 302 rows, which is
  eight arms ADDED since the round-1 receipt (kvq, idxq x4, kvq-spec, idxq-spec, trunk-diet)
  with zero round-1 arm keys absent. The value-level compare on shared keys — 16 keys, **0
  differences** — is the honest instrument; a raw row-count compare would have reported a
  delta and been wrong.
- **The router audit's row count is shape-dependent, not a metric.** This baseline read
  `router-audit rows=768 worst_w_ulp=1` where PROFILE-9 read `rows=129004 worst_w_ulp=3`. The
  first is a 10-token probe with `--verify-bit-gate 8` (768 = 48 layers x 16 rows), the second
  is spec-gate 256 x 4 prompts. Both prove engagement (`rows=0` is the silent-no-op failure the
  counter exists to catch); neither is comparable to the other, and nor is the ulp bound.

## 2. The TP2-prefill class gate is calibrated — after three defects

Detail in `TP2-CLASS-GATE.md`. The resume order was "run `--tp2-class-calibrate`, read the
measured worst, set the constants". The gate could not measure what it claimed to.

**Defect 1: `HeadMode::All` did not exist in `forward_tp2`.** It allocated one row and copied
`plane[(t-1)*hidden]` regardless of `HeadMode`, so `All` was silently identical to `LastRow`.
The PRIME regime's whole purpose is "compare EVERY row of a t>=2 forward" — the exact defect
the two-regime gate was written to fix — and it could not. **It surfaced only because the gate
length-checks the two logits vectors before comparing** (`single-card produced 2483200 logits,
TP2 248320`). Corpus-worthy: the loud failure that saved this was a *length* assert, not a
value assert. Fixed by threading `rows` through `tp2_seg_exit` (both `gate_read_inner` and
`launch_qmatvec_bf16w` already take a row count — the kernel's grid y-dim *is* `t`), with
`tp2-gate 24` byte-identical afterwards as the proof that `rows == 1` is byte-for-byte the old
code.

**Defect 2: the band and the measurement were in different units.** The gate divided by
`max(|a|, 1e-6)`; `--tp2-gate`'s cited 3.0e-5, every other receipt in this lane, and the glm5
4.85e-5 the placeholders were borrowed from all use `compare()`, which floors at **1.0**. The
comment claimed the 1e-6 floor meant "a near-zero logit cannot flatter the comparison" — the
sign is backwards, a near-zero denominator catastrophically *penalizes*, and over a
248,320-wide vocab there are always near-zero logits. First calibration run: `prime_worst_rel
2.865e4` on a row whose worst *absolute* difference was 3.975e0 with a matching top-1. That is
a category error, not a loose bar. The 1e-6 form survives as the `elem_rel` diagnostic column.

**Defect 3: PRIME straddled two variables and the wrong one dominated.** `HeadMode::All`
selects the **per-expert** MoE executor while TP2's `tp2_moe_rows` is **grouped**, so prime
measured the executor difference (2e-3..4.1e-3) rather than the TP2 split (chunked, grouped on
both sides: **1.4e-5**) — two orders, independently corroborated by the tiny gate's own
`prefill-extend` arm pricing the executor difference alone at 1.865e-4. Fixed with
`set_prefill_grouped_all` (FLAGS.md row, **default OFF**, scoped ON around the prime forward
only), with hidden-goldens and verify-bit byte-identical as the proof that OFF is byte-for-byte
the old behavior.

### The calibrated bands

Measured green worst, 19 rows, argmax 19/19, tape OK, peer engaged:

| regime | rows | green worst | placeholder | **calibrated band** |
|---|---|---|---|---|
| prime (t>=2) | 10 all-rows + 1 chunked | **1.383e-5** | 2e-4 (borrowed) | **1.4e-4** |
| decode (t==1) | 8 | **1.574e-5** | 3e-4 (borrowed) | **1.6e-4** |
| red floor | — | — | 1e-3 | 1e-3 (kept) |

Both tighter than the placeholders — calibrate downward, never up.

**Finding: decode is NOT tighter than prime here.** The placeholder's comment reasoned it must
be (no batched-GEMM width variance at t==1); measured, decode's 1.574e-5 exceeds prime's
1.383e-5. The expert-half join reorder alone puts t==1 in the same order and the width variance
does not dominate. Second place in one gate where prose about the numerics was confidently
wrong in a checkable way — which is the argument for calibrating rather than reasoning.

**`decode_byte_identical` = FALSE**, reported as a measured field rather than assumed as the
bar, exactly as designed: our program is an expert-half split with a join, not glm5's
column-parallel-over-gather.

### The band is a bar

| arm | prime worst | vs green worst | tape | argmax | peer_slots |
|---|---|---|---|---|---|
| green | 1.383e-5 | 1x | ok | 19/19 | 6908 |
| `skip-peer-moe` | **9.930e0** | ~7.2e5 x | BROKEN | 13/19 | 6999 |
| `peer-local-ids` | **1.003e1** | ~7.3e5 x | BROKEN | 11/19 | 6987 |
| `reverse-peer-weights` | **8.271e0** | ~6.0e5 x | BROKEN | 13/19 | 6937 |

Four orders between the band and the *least* wrong red. `peer-local-ids` is both the most
plausible real bug (right magnitudes, wrong experts) and the most damaging by argmax — the arm
to keep if one ever has to be dropped.

### Per-rank engagement: glm5's DERIVED fractions, measured here

```
peer_slots=6908  home_slots=6532  peer_slot_fraction=0.5140
layer_tokens=1344  both_card_rows=1343  both_card_fraction=0.9993  engaged=true
```

**Measured for this geometry (512 experts, even split): 99.93% of layer-tokens touch BOTH
cards**, and the peer takes 51.40% of dispatched expert slots. NOT comparable to glm5's derived
99.3% (different expert count and top-k, and derived — see ROUND2-STATUS). Under an even split
essentially every token pays a cross-card join, which is precisely the number the co-activation
placement lane exists to reduce, and the even arm is its control.

## 3. Depth: kvq buys 4.4x, TP2 costs depth, 1M is out of reach

Full tables in `LADDER.md`. Headlines:

| arm | KiB/token | single-card ceiling |
|---|---|---|
| f32 | 49.0 (round 1: 48.9) | ~165k |
| kvq q8_0/q5_1 + idxq q8 | **11.08** | **~731k** |

600,000 tokens allocate on one card at 96,467 MiB (1,420 free); **1,000,000 OOMs at state
allocation**. TP2 is a depth *regression*: +2,784 MiB on card 0 post-load and card 1 flat at
43,603 MiB while card 0 carries all growth, so it OOMs during the fill below 100k while one
card reaches ~731k. Card 1's ~54 GiB free is unusable for context.

### 3a. Timed at 100,000 tokens, both arms — and kvq's perf sign FLIPS with depth

| arm | tok/s | ms/token | prefill wall | card-0 VRAM | free | spread |
|---|---|---|---|---|---|---|
| **kvq q8_0/q5_1 + idxq q8 (ship default)** | **33.62** | 29.7 / 29.7 / 29.8 | 561.4 s | **92,957 MiB** | **4,930 MiB** | 0.09% |
| f32 twin, same depth | 36.30 | 27.6 / 27.6 / 27.7 | 523.3 s | 97,213 MiB | 674 MiB | 0.44% |
| f32, round 1 | 35.56 | 28.1 / 28.1 / 28.2 | 501.5 s | 96,381 MiB | 1,506 MiB | — |

Both round-2 rows `looped=false`, 49 chunks, `rounds=3x12`, spreads under 0.5% so x3 stands.

**KVQ-CELL's flip receipt says "the quantized cache measures FASTER: 13.36-13.39 vs 13.53-13.57
ms/token". At depth the sign REVERSES: -7.4% decode, -7.3% prefill wall.** Dequant cost scales
with the number of KV rows READ per token, and the block-list path reads up to ~2,052 rows/token
at depth while the flip's shallow fill reads almost none. **The flip DECISION stands on memory
grounds** — 4,256 MiB of headroom at 100k and a 4.4x ceiling — **but the perf claim is now
DEPTH-SCOPED to short context and must never be quoted at depth.** That scoping is written into
the `set_kv_quant` row in docs/FLAGS.md, because that row cited the shallow number as part of the
flip justification and a stale perf justification must not ride.

### 3b. FINDING: prefill cost per chunk STEPS UP ~4.8x at ~131k depth

Per-16,384-token prefill segment wall from the 262,144 rung (chunk 2048, ship defaults, one card):

| fill | s per 16k | | fill | s per 16k |
|---|---|---|---|---|
| 16,384 | 84 | | 114,688 | 102 |
| 32,768 | 85 | | **131,072** | **105** |
| 49,152 | 90 | | **147,456** | **475** |
| 65,536 | 93 | | **163,840** | **482** |
| 81,920 | 97 | | 180,224 | 497 |
| 98,304 | 99 | | | |

Below ~131k the segment wall grows +25% across an 8x depth increase — gently, near-linearly.
Between 131,072 and 147,456 it jumps **4.5x**, then HOLDS at ~475-497 s rather than continuing to
climb. It is a **step, not a curve**.

Round 1's headline for this model is that decode is near-flat in depth (22.3 ms at 4k -> 28.1 ms
at 100k) because QSA attention is bounded. That is about **decode**, and round 1 only ever
prefilled to 100,000 — it never crossed this step. **Prefill is NOT flat in depth**, and the
discontinuity sits just past the deepest depth any previous receipt in this lane reached. A 262k
prefill is ~77 min against the ~25 the sub-131k rate predicts; 600k extrapolates to ~3.6 h at the
post-step rate rather than ~56 min. For deep-context serving economics that dominates the decode
rate.

Not attributed — measured, with candidates named rather than guessed: the step lands close to
2^17, and the plausible mechanisms are the QSA selection horizon crossing into non-full rows (the
`longatt` AUTO engagement condition), the indexer pooled-key device mirror crossing a sizing
boundary, or host-side raw-key cache growth. A `--profile` section run at 120k and 150k separates
them: one cheap cell, and the first thing to run on the next box.

## 4. The defect that makes §3's timing row f32, and the fix that matters

`qwen4exp_real_gate` pinned the cache seams to f32 unless `MEMRA_Q4E_SEAMS` named them, and the
pin was **unconditional** — which its own comment did not say, since it scopes itself to
reference-parity golden comparisons. So the ladder, the spec-at-depth cells and the TP2 gates
all measured f32 while reporting themselves as ship defaults.

Caught by a decisive probe: the 131k state allocated at 96,243 MiB, *exactly* round 1's f32
number, so either kvq bought nothing or kvq was not on. Forcing the seam both ways settled it
in two minutes (96,243 / 96,243 / **91,475**).

Two fixes; the second is the one that matters. The pin is now scoped to golden comparisons —
and **every receipt header carries `# cache kv_quant=... idxq=... golden_pin=... seams_env=...`**.
The pin was a bug, but what let it survive and get written up wrongly is that an f32 run and a
ship-default run produced identical-looking headers. A receipt that cannot state its own cache
arm cannot be read. Every golden-comparison receipt is byte-identical across the change.

## 5. Instrument note: the ascending ladder cannot bank partial rungs

`--ladder 100000,262144,600000,1000000` OOM'd at state allocation with **nothing banked**: the
ladder allocates ONE state at the deepest rung's capacity before the first rung runs. The
reclaim-safety property the resume order counted on does not hold when the top rung does not
fit. Worse given this lane's reclaim rate, a rung's row flushes only on rung *completion*, so
the 262,144 rung was lost after ~23 minutes of prefill with its header banked and no row.

## 6. Open, and stated rather than implied

- **262,144 and 600,000 timed rungs** on the ship defaults; both proven to ALLOCATE, neither
  filled and timed. Blocked on hardware: six spot reclaims in this lane on 2026-08-31, each
  replacement needing a ~174 GB mint download plus a build before it can measure, which has
  been exceeding the boxes' lifetimes.
- **A ship-default tok/s row at any depth** (§3's 36.30 is f32).
- **Work item 4, spec at depth** (32k/100k per shape at ship admission, card-1 draft): the
  instrument is built and committed — `--ladder-spec-shape` appends a shape pack's
  chat-template render to the END of a deep corpus fill, so the fed sequence is
  `[deep document][task turn]`, which is the long agentic shape the question is about, with the
  corpus head trimmed so total fill == rung. **Never compiled or run** — no measurement.
- **Work item 5, router traces** in the shared `MEMRA_MOE_TRACE` format: not collected.
- The **per-chunk held-VRAM curve** across a deep fill. Partially answered: 11.08 KiB/token is
  flat across three state allocations with no growth term, and the TP2 fill rows grow 128-256
  MiB per 16k chunk with no acceleration. Neither is the full curve.
