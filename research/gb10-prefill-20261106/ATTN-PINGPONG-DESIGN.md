# Prime attention ping-pong on GB10 (round 4 design, 2026-11-08)

Design note before code, as the GEMM v3 lane did; the measurement at the end rejected it (ledger
row 47). Round 4's lever came from the profile instead (`MTP-FILL-DESIGN.md`).

## Why attention

The fresh 131k profile on main (`profile-r4-main-131k/`, README round 4) splits the cold prime
into the F8F4 v3 GEMM tile, prime attention and a tail of DRAM-floor passes. Attention is the part
whose share grows with T: causal attention is 196,608 x T^2 FLOP, so at 259,951 tokens it is about
13.3 PFLOP and, at h2p's 85 TFLOPS, about 156 s of the 322 s cold TTFT. The GEMM tile is linear in T,
and its exact program is pinned near its measured loop ceiling by rows 33-45. Every other part
left is at its DRAM floor (`research/gb10-prefill-next-20261107/README.md`).

The attention ceiling on this device is about 102 TFLOPS for the bf16 m16n8k16 forms the kernel
issues (`research/gb10-prefill-attn-20261107/ceilings/`), FA2-class BF16 at serving shapes is 82.3,
and h2p runs 85. Each TFLOPS h2p gains is worth about 1.8 s at 260k and 0.47 s at 131k.

## Two candidates, one dropped on SASS

1. **Skip the O rescale when it is the identity.** Rejected before building. The idea: when no row
   of a warp raises its running max, alpha is `exp2f(0) = 1.0f` and the 128 multiplies per thread per
   tile change nothing. The shipped SASS shows they do not exist as multiplies: nvcc contracts
   `O *= alpha` and the later `O += C` into one `FFMA O, alpha, O, C` per element (h2p's tile loop:
   130 FFMA, 128 HMMA, 38 FADD, 70 FMUL). So the program's per-element update is
   `O = fma(alpha, O, C)`, the skip would replace an FFMA by an FADD at the same issue cost, and the
   first probe copy, which split the update across a branch, broke the contraction (FMUL then FADD,
   two roundings): it was not even exact. Lesson: read the SASS of the operation sequence before
   claiming a skip.
2. **Warp-pair ping-pong** (built). h2p has 8 warps, two query heads of one KV head; warps `wq` (head
   A) and `wq + 4` (head B) share a scheduler. A named barrier skews head B one QK behind head A at
   the start, then the warps drift, held only by the three-stage K/V ring. Each warp alternates an
   MMA block (PV of tile i - 1, then QK of tile i: 128 HMMA) and a softmax with no MMA. When both
   warps of a scheduler are in their MMA blocks they share the pipe, finish close together, and
   then both sit in the softmax with the tensor pipe idle. ncu has the pipe 78.7% active.

## The ping-pong (FA3's warpgroup schedule at warp-pair granularity)

Per scheduler pair, two named barriers (`GO_A = 2 + wq`, `GO_B = 6 + wq`, 64 threads each):

- head A: QK(0), arrive GO_B; then per tile: softmax(i), wait GO_A, PV(i) and QK(i + 1), arrive GO_B;
  after the last PV, one more arrive on GO_B;
- head B: wait GO_B before QK(0); then per tile: arrive GO_A after QK(i), softmax(i), wait GO_B, PV(i).

So the MMA blocks strictly alternate, A, B, A, B, and each warp's softmax runs under its partner's
block. One warp's block alone reaches the pipe (`ceil_attn.cu`: 2 chains per warp, 96.7 TFLOPS at 4
warps per SM), and the softmax (about 400 instructions with MUFU and shuffle latency) is shorter than
the partner's 128-HMMA block, so the pipe should stay busy. Arrivals and waits pair up exactly
(A arrives GO_B nt + 1 times, B waits on it nt + 1 times; B arrives GO_A nt times, A waits nt times),
and a warp cannot lap its partner, so the barrier phases never alias. The ring needs no change: a
warp refilling a stage waits for all eight warps to release the tile before it, which the
alternation has already ordered.

Variants in the probe (`attn-r4/probe/attn_pp.cuh`, generated from the shipped body text, so every
other line is h2p's):

| arm | barriers |
|---|---|
| pp0 | h2p's body as shipped (control; same SASS as `fa_prefill_qw_h2p`) |
| ppw | per warp pair, 64-thread barriers 2-9 |
| ppg | the two 4-warp head groups, 256-thread barriers 2-3 (couples schedulers) |
| ppl | ppw with the row-sum shuffles and the l update moved after the wait, into the MMA block |

Exactness: barriers order execution only. Every row still runs h2p's tiles in order with the same
QK chains, softmax expressions, `exp2f`, bf16 P rounding, `fma(alpha, O, C)` update and PV chains
(the SASS keeps h2p's 130 FFMA in every arm: the wait sits before the O update, not between its
multiply and add). O must be bit-identical to h2p; the probe compares every float at four depths
and the attention lane's 21 ragged shapes, kernel-check's ARC B cells cover the engine.

## Measurement plan

1. Probe on one unit: T 4,096 at depths 4k / 16k / 64k / 128k, 3 reps, both orders, every arm
   against h2p; ragged shapes byte-compared.
2. If an arm clears h2p by more than the same-binary spread (the attention lane's h2p/h2p repeat:
   0.1 TFLOPS), it goes into `fa_prefill_qw_h2p_body` in place (no new door: it is h2p's schedule,
   bit-identical, and `MEMRA_PRIME_KV_H2P=0` stays the rollback to h2).
3. Served cold TTFT, B V V B on one unit, fresh boot per cell, 32k / 131k / 260k per boot; the second
   unit confirms; the 16k argmax-margin gate; kernel-check ALL GREEN with the ARC B cells.

## Result (probe, unit 1, `attn-r4/probe-runs/`)

Every arm is exact (0 mismatching floats at 4 depths and at the 21 ragged shapes), and every arm
loses:

| arm | 4k | 16k | 64k | 128k | 262k (ragged run) |
|---|---:|---:|---:|---:|---:|
| h2p (main) | 62.47 / 62.37 | 78.18 / 78.59 | 84.16 / 84.20 | 85.15 / 84.77 | |
| pp0 (control) | 62.56 | 77.75 | 83.82 | 85.19 | |
| ppw | 45.63 / 45.66 | 53.80 / 53.76 | 56.60 / 56.80 | 57.18 / 57.35 | 57.57 |
| ppg | 44.73 | 53.11 | 55.95 | 56.63 | 56.88 |
| ppl | 46.57 / 46.61 | 52.24 / 55.26 | 58.18 / 58.43 | 58.89 / 57.69 | 59.10 |
| ppv (PV blocks alternate, QK free; 56 B stack) | 57.79 | 73.40 | 77.18 | 77.82 | 78.60 |

TFLOPS on the causal work, T 4,096, median of 3 (two values: the arm's two passes). Strict
alternation costs a third: one warp alone does not keep the tensor pipe busy (its MMA chains are
latency-bound), so h2p's free drift, where both warps of a scheduler are often in MMA blocks at
once, beats a schedule that guarantees the softmax overlap. Alternating only the PV blocks still
loses 9%. Attention's remaining 17% to the 102 TFLOPS ceiling is the softmax phase with only two
warps per scheduler, and the 255-register body leaves no room for a third.
