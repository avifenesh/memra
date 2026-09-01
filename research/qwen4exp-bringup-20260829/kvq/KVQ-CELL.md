# qwen4_exp KV-quant + TP2-prefill cell (kvq lane, 2026-08-31)

Owner decisions binding this lane: KV cache default **K=q8_0, V=q5_1** (asymmetric —
K feeds the score dots + rope, V errors average under attention weighting); the default
ships ON only with this lane's receipts attached (flags law). Activation quantization
stays owner-retired; KV STORAGE precision is this assignment. Box: sbox-eval Frankfurt,
2x RTX PRO 6000 Blackwell 96 GB; artifact `~/data/q48fn-nvfp4` (+ the yarn-1M hardlink
override for the deep rungs); branch qwen4exp-bringup-20260829.

## What the seams are

- `kvq` (`MEMRA_Q4E_SEAMS=kvq`, `set_kv_quant`): QSA KV rows stored q8_0 (K, 34 B/32
  elems) / q5_1 (V, 24 B/32 elems) — **48.9 -> ~11.1 KiB/token** vs the f32 cache whose
  47.8 GiB at 1M was the yarn cell's blocker. STORAGE-only: attention math runs f32 on
  dequanted values; the quantized cache always takes the block-list attention form
  (`q4e_sdpa_blocklist_q8q5`, in-place dequant). Format latches PER STATE at alloc.
- `idxq` (`MEMRA_Q4E_SEAMS=idxq[=bf16|=0]`, `set_idxq`): the indexer RAW-key cache
  (128-dim pre-norm/pre-rope keys; host + idxcache device mirror) stored q8_0 or bf16.
  Consumed only through fp32 mean-pooling (dequant at read, pooling op order identical),
  so output moves ONLY through the selection set. Precision picked by measurement.
- `trunk_f32_diet` (`--trunk-diet`): frees the trunk f32 originals whose bf16 twins
  serve under the ship seams (QSA/GDN projections, gates, router, shared-expert mats,
  lm head). Guards err loudly if a dropped original would be read.
- TP2 prefill (`forward_tp2`/`prefill_extend_tp2`/`alloc_state_tp2`, `--ladder-tp2`,
  `--tp2-prefill-gate`): chunked prefill ON the TP2 route — KV/state fill sharded-local
  (the yarn cell measured remote KV at 18x decode collapse), selection once on card 0,
  MoE split by expert half, join adds in fixed rank order. TP2 decode switched from the
  dense-mask attend (O(context) bytes + t_kv mask h2d per layer per token) to the same
  selection/block-list halves.

## Kernel oracles (rig + box tiny gate: 33 arms, failures=0 both hosts)

- Append-quantize vs host twins: BYTE-identical (q8_0 K rows, q5_1 V rows; adversarial
  battery = zeros / constant / half-ulp ties / subnormal-scale rows at widths 512 + 40).
  **First run caught a real contract hole: device `lrintf(inf)` is UNDEFINED on
  subnormal-amax blocks while the host twin saturates — fixed with an `isfinite` guard
  on BOTH sides (the one deliberate divergence from the flash appender program).**
- Row-dequant vs host twins: BIT-identical.
- **Fused quantized block-list attention vs the "dequant rows then `sdpa_blocklist_f32`"
  composition: BIT-IDENTICAL over 18,432 values** (real geometry, full+stride
  selections) — the storage contract's load-bearing oracle.
- Indexer q8/bf16 appenders vs host cache twins: BYTE-identical; **system-level pin:
  idxcache ON vs OFF decode rows BIT-IDENTICAL under q8 AND bf16** (host- and
  device-quantized rows interleave freely).
- kvq fixture arms: determinism PASS, spec byte-identity same-config PASS; envelope vs
  f32 twin abs 1.5e-3, argmax flips 0/25 (tiny REPORT; the real-checkpoint envelope is
  the box gate). trunk-diet arm: dir-bf16 pre/post rows BIT-IDENTICAL.

## Box round 1 — real checkpoint (receipts in ~/realgate/kvq, mirrored here)

Defaults regression (the refactor moved nothing at kvq=0): verify-bit **24/24
bit-identical**, spec-gate raw + thinkon **byte identity**, tp2-gate **24/24 argmax,
worst rel 3.0e-5**, tiny 33/33.

kvq armed (K=q8_0/V=q5_1):

| gate | result |
|---|---|
| goldens logits argmax | **10/10** (`hidden-gate-kvq1-gold.tsv`) |
| greedy first-divergence vs banked goldens | **-1 / 8 / -1 / 26** vs the f32 arm's -1/8/-1/48 — the same near-tie fork class; prompt 3 forks at 26 instead of 48 (one additional near-tie flip; mint-gate reporting style, stated not hidden) |
| verify-bit (kvq both arms) | **24/24 bit-identical** |
| spec byte-identity >= 256 tokens, SAME-config (raw + thinkon) | **PASS** |
| seam-gate 24 decode rows, f32 vs kvq (the CROSS-config envelope) | argmax 22/24, worst KL 0.583 (median class ~0.02), worst abs 5.3 on rows with ref_absmax 14-36 — the honest cross-config envelope: two near-tie argmax flips at steps 1 and 10; this is a REAL quantization (mint class), not an accumulation-class seam |

idxq armed:

| gate | result |
|---|---|
| seam-gate 24 decode rows, f32 vs q8 raw-key cache | **BIT-ZERO envelope, 24/24 argmax, worst_abs 0.000e0** — below the selection horizon the raw-key cache provably cannot move output except through the selection, and it did not |
| spec byte-identity raw (q8 both arms) | see round1.log phase 3b |

## Round-1 results (banked from ~/realgate/kvq, analyzed 2026-08-31)

- Within-config exactness, all green: spec byte-identity 6/6 PASS (kvq/idxq/diet arms,
  raw + thinkon, ≥256 tokens), verify-bit 24/24 ×3, envelope 24/24 argmax at worst_rel
  3.016e-5 (hidden-gate-kvq0-defaults).
- idxq q8 vs f32: **BIT-ZERO** (seam-gate-idxq-idxq1.tsv: 24/24, worst_abs 0.000e0,
  KL 0) — the selection provably did not move. idxq default = q8 by this measurement.
- kvq cross-config envelope (seam-gate-kvq-kvq1.tsv, OFF vs ON on the same fed tokens —
  an ENVELOPE claim, not a same-config gate): 22/24 argmax, worst_abs 5.291 on
  |logit|≈12-15, worst KL 0.58257 at step 1. The two flipped rows are NEAR-TIES: step 1
  top1 271 vs 74455; step 10 flips between the eos pair 248045/248046. Stated
  mint-gate-style; the quant is not byte-transparent and was never claimed to be.
- Greedy vs the transformers goldens, valid (raw) instrument: kvq0 forks -1/8/-1/48,
  kvq1 forks -1/8/-1/26 — same class (two full 64/64 both arms).
- **Broken instrument found**: greedy-gate-kvq{0,1}-thinkon fork at step 0 on BOTH arms
  incl. f32 — the thinkon goldens do not match the gate's thinkon render (instrument
  mismatch, not a model defect). Not quoted as a quality receipt; repair or drop the arm.
- A/B (ab-kvq-kvq-plain, interleaved, fresh state per arm): q8q5 13.36-13.39 vs f32
  13.53-13.57 ms/token — the quantized cache is ~1.3% FASTER (fewer bytes read) at
  ~4.4× smaller KV (48.9 → ~11.1 KiB/token).
- Depth ladder with kvq+idxq (ladder-kvqidx-ladder.tsv, ×3 rounds, spreads ≤0.17%):
  4k = 41.44 tok/s, 32k = 38.13, 100k = 32.11; card0 92.5-93.0 GiB (single-card arm).
- **Defaults FLIPPED with these receipts** (owner decision executed per the flags law):
  `KV_QUANT_DEFAULT=true` (K=q8_0, V=q5_1), `IDXQ_MODE` init q8. Gate binaries pin the
  f32 exactness-instrument arms for reference-parity comparisons (the tiny gate caught
  the leak: 2 rows out of tolerance under flipped defaults, receipt in the flip commit);
  explicit --ab-seam / MEMRA_Q4E_SEAMS arms still control their own state. Tiny gate at
  flipped defaults + pin: 263 rows, 0 failures.

## Round 2 — BLOCKED ON HARDWARE (third spot reclaim, 2026-08-31)

Full state: **`ROUND2-STATUS.md`** beside this file. Short version:

- The replacement box (round-2 box 3, same card class) lived **42 minutes** and was
  preempted for lack of preemptible capacity. No replacement hardware provisioned from the
  lane — capacity hunting is the owner's path (provider/region live in darklanes).
- **Box baseline (work item 1) reproduced round 1 with no delta on the five arms it
  reached**: tiny gate all-PASS, hidden goldens **identical to every printed digit** on all
  10 rows with 10/10 argmax, greedy raw `-1/8/-1/48` (the kvq0 pattern), verify-bit 24/24
  bit-identical, spec byte-identity raw PASS. Same card class, and the numbers behave like
  it.
- Not reached: greedy-kvq rows, spec thinkon, tp2-gate, the audit row count — and work
  items 2-4 (TP2 class-gate calibration, the 1M ladder, spec at depth) entirely. **No
  number is claimed for any of those.**
- Phase 0's TSVs died with the box: they were read over ssh instead of scp'd. On a spot
  box a receipt is banked when it is on the rig, not when it is written — copy per ARM,
  not per phase.
- Engine work banked and rig-built (`8a1b7348b`, gates not yet run on a box): pluggable
  per-layer expert placement reading the shared `memra-ep-map-v1`
  (`MEMRA_Q4E_EP_MAP`, default OFF = even split, bit-identical control by construction);
  MoE route traces in the shared `MEMRA_MOE_TRACE` format; per-rank engagement counters;
  and a two-regime TP2-prefill class gate that replaced a bar which was calibrated against
  nothing (flat `max_rel <= 0.01`, ~50x looser than the same class's calibrated band) and
  which compared only the chunked prefill's LAST ROW — a t==1-shaped read of a t>=2
  program. **Its band constants are placeholders until the calibration run lands.**

STILL PENDING (round 2): tp2-gate + the tp2-prefill CLASS gate at the TP2-prefill tip
(calibration first, then the three RED arms), the TP2 ladder (4k/32k/100k then
100k/262k/600k/1M on the yarn artifact), spec at depth (32k/100k/250k, ship K=5, card-1
draft), and route traces banked by shape and depth for the placement lane.

## The 1M budget this lane is built against (analytic, to be replaced by the ladder)

Per token per QSA layer: K 544 B + V 384 B = 928 B x12 layers = **10.9 GiB at 1M**
(vs 47.8 GiB f32); TP2 halves ~5.45 GiB/card. Indexer device caches at 1M: raw q8
~1.6 GiB + pooled f32 ~1.5 GiB (card 0). Card 0 ~= trunk (89.97 GiB - diet ~6) + 5.45 +
3.1 + workspace ~0.8 ~= 93.5 GiB of 96. Tight; the ladder's VRAM table is the verdict.

---

# ROUND 2 (2026-08-31): kvq at depth, and a defect that hid it

Full tables in `../round2-box-receipts/LADDER.md`; the box baseline that gates all of it is
`../round2-box-receipts/BASELINE.md`.

## kvq's payoff is DEPTH, and it is 4.4x

Measured by state allocation on one card (97,887 MiB, trunk 89,971 MiB, 7,916 MiB free):

| arm | KiB/token | single-card ceiling | 262,144 | 600,000 | 1,000,000 |
|---|---|---|---|---|---|
| f32 | **49.0** (round 1: 48.9) | ~165k | OOM | OOM | OOM |
| kvq q8_0/q5_1 + idxq q8 | **11.08** | **~731k** | 92,883 MiB | 96,467 MiB | OOM |

`(96,467 - 89,971) / 600,000` = 11.08 KiB/token, matching this cell's analytic ~11.1 to three
digits. So the ship default moves the model from "cannot serve 262k" to "can serve 600k on one
card". It does not reach 1M (see YARN-CELL round 2).

## The defect: the gate binary was an f32-ONLY instrument

`qwen4exp_real_gate` pins the cache seams to f32 unless `MEMRA_Q4E_SEAMS` names them. The pin
is right for reference-parity runs — this cell's own note that the no-env hidden/greedy rows
are the **kvq0** rows depends on it — but it was **unconditional**, which its comment did not
say. Every non-golden run of the binary therefore measured f32 while reporting itself as
running the ship defaults: the long-context ladder, the spec-at-depth cells, the TP2 gates.

It cost a full 100,000-token ladder rung (~9 minutes of prefill) that was written up as "kvq
ship defaults" and was f32. Caught by a decisive probe rather than by reading — the 131k state
allocated at **96,243 MiB, exactly round 1's f32 number**, so either kvq bought nothing or kvq
was not on:

```
default (no env)         131,072  state-alloc  96,243 MiB   <- the f32 number
MEMRA_Q4E_SEAMS=kvq=0    131,072  state-alloc  96,243 MiB
MEMRA_Q4E_SEAMS=kvq      131,072  state-alloc  91,475 MiB   <- 4,768 MiB less
```

Fixed twice over: the pin is scoped to runs that make a golden comparison (`--goldens` /
`--prompts`), and **every receipt header now carries
`# cache kv_quant=... idxq=... golden_pin=... seams_env=...`**. The second is the real fix.
The pin was a bug; what let it survive and get published wrongly is that an f32 run and a
ship-default run produced identical-looking headers. A receipt that cannot state its own cache
arm cannot be read.

Controls: every golden-comparison receipt is BYTE-IDENTICAL across the change (hidden goldens,
verify-bit 24, tp2-gate 24, tp2-prefill class gate), so no banked receipt in this cell is
invalidated and the kvq0/kvq1 greedy patterns above still stand.

**Reading rule for this cell, from now on:** a receipt without a `# cache` line predates
2026-08-31 and its cache arm must be inferred from whether it passed `--goldens`/`--prompts`
(f32) or not (f32 as well, because of the unconditional pin). In practice: **every ladder or
perf row in this lane banked before 2026-08-31 is an f32-cache row.**

## kvq's PERF sign flips with depth — the flip receipt is shallow-scoped

The flip decision above records "the quantized cache measures FASTER: 13.36-13.39 vs
13.53-13.57 ms/token". Measured at a **100,000-token fill** on the same card class, the sign
REVERSES:

| arm | tok/s @ 100k | ms/token | prefill wall | card-0 VRAM | free | spread |
|---|---|---|---|---|---|---|
| kvq q8_0/q5_1 + idxq q8 | **33.62** | 29.7 | 561.4 s | **92,957 MiB** | **4,930 MiB** | 0.09% |
| f32 twin, same depth | 36.30 | 27.6 | 523.3 s | 97,213 MiB | 674 MiB | 0.44% |

**-7.4% decode and -7.3% prefill wall at depth.** Mechanism, consistent with the design:
dequant cost scales with the number of KV rows READ per token, and the block-list path reads up
to ~2,052 rows/token at depth, whereas the flip was measured at a shallow fill where almost
none are read. Both spreads are under 0.5%, so x3 stands.

The flip DECISION stands — at 100k the same seam buys 4,256 MiB of card-0 headroom and a 4.4x
higher depth ceiling, which is the right trade for a long-context product. What must change is
the CLAIM's scope: **"kvq is faster" is true only at shallow fills and must never be quoted at
depth.** For a short-prompt product the seam is a ~7% cost, not a free win.
