# TP/EP decode levers, second round: fused MoE on the partition, and two refuted schedules (memra #710, 2026-09-26/27)

Model: `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`. Program:
the served TP/EP default after `../levers-20260926/` (PDL chain, vocab-parallel head, push joins).
Hardware: two 2x RTX PRO 6000 Blackwell Server Edition pairs, named here by the pair each run used.
Scored work is one campaign at a time under `/tmp/memra-gpu.lock`.

Each lever is a scheduling change. Every gate compares bits: the long gate's `PROGRAM_SHA256`
(`fbce1a0492d69635` on main) covers 304 replayed steps from a 400-token prefix. It hashes every
step's token, logits bits, and cache and hidden digests.

## The post-lever step (where the time went)

With PDL off, from an nsys trace of the served plain c1 stream, 510 steps on the SE pair: 13.96 ms
per step in the capture window, with a per-device kernel sum of 12.3 ms.

| part | per step per rank | launches | note |
|---|---|---|---|
| dense FP8 GEMV | 3.14 ms | 366 | 3.52 GB per rank at about 1.12 TB/s; each launch pays about 2 us of ramp over its bytes |
| MoE stream visitor | 1.96 ms | 130 | gate, up and down of the rank's experts, about 0.87 TB/s |
| dots (compressor, router, head half) | 1.17 ms | 170 | the head half is 348 us at 1.52 TB/s |
| push expert reduce | 0.70 ms | 43 | mostly one rank waiting for the other's experts: 4.6 us in a balanced layer, 48 us in an unbalanced one |
| HC finish | 0.63 ms | 87 | one 128-thread block per call, about 7 us |
| push row gathers | 0.53 / 0.36 ms | 88 | |
| the routed chain's small kernels | about 1 ms | about 12 per layer | act_quant, route count/prefix/scatter, mirrors, scale_rows, SwiGLU, scatter |

The MoE visitor's rate follows from its access pattern. Each warp keeps about 1.7 KB of 64-byte
row pieces in flight, 8 rows apart, and on TP/EP a rank's three-odd experts put about 768 warps on
188 SMs.

## Fused one-token MoE on the TP/EP partition (adopted)

Lane `lane/dsv4-moe-fused-tpep-20260926`. The fused pair (#694) ran PP-2 only. It now has a
partition form:
- the table holds a rank's experts `[first, first + n)` of the global bank;
- selections and macro scales keep global ids;
- another rank's slot exits at entry;
- down writes only the rank's contribution rows and leaves the slot sum to the caller, after the
  rank-order join, where the chain's sum already runs.

A TP/EP one-token step takes it in place of the grouped chain, about 14 launches per layer, and
also skips the separate x activation quantization. The pair streams 256 k per stage, so each row
piece is 128 bytes.

**Correctness** (second SE pair, `raw/se2-fused-s2j/`):
- `cuda_fused_partition_moe_is_the_tp_ep_chain_bit_for_bit` compares the contribution plane with
  `execute_matrix_local`'s bit for bit on both halves of a bank:
  - slots on both ranks;
  - every slot on one rank (the other rank's plane stays zero);
  - duplicated experts;
  - each red fixture's fault bit;
  - a partition that asks for the in-kernel sum refuses 40004.
  It passes with the full-bank fused test, the deferred-partition test and the TP/EP local-only
  test.
- The long gate's `PROGRAM_SHA256` is `fbce1a0492d69635` with the partition form engaged.
- `dsv4_rows_gate` TP/EP and `dsv4_kv_split_gate` pass.
- The DSpark TP/EP gate's plain arm takes 27520 fused dispatches, and its proposal shas equal
  main's on the pod.

**Long-gate replay**, interleaved M F F M: 11.32 to 11.41 ms per token against 12.38 to 12.52 for
main, -9.0%.

**Served**, the one-row form: second SE pair, one boot per row, order M F F M M F F M M F, N=5
(`raw/se2-fused-s2j/`):

| cell | main | fused | delta |
|---|---|---|---|
| greedy c1 | 77.48 (77.25..77.73), decode 81.01 | 85.16 (84.80..85.34), decode 89.32 | +9.91% |
| sampled c1 | 77.26 | 85.62 | +10.82% |
| greedy c2 | 104.02 | 105.63 | +1.55% |
| greedy c4 | 133.21 | 133.40 | +0.14% |
| c2, 1500-word context | 19.18 (TTFT 10.38 s) | 19.42 (10.25 s) | +1.25% |

- Every fused row is above every main row at c1.
- Every cell's texts are identical across both arms, the sampled ones included.
- c2 and c4 barely move: a B-row step of 2 to 4 rows still ran the chain in this form.

The multi-row form (next section) covers those steps.

**The multi-row form.** Slot p reads token row p / topk, so one launch covers 1 to 16 rows. That is
every step the stream visitors covered: one token, the four-lane B-row steps and the DSpark
verify rounds.

The multi-row visitor puts the rows of one expert in one pass. The fused pair runs each slot
alone, and an MMA output row depends only on its own A row, so each slot keeps the chain's bits.
The component test adds a three-row case with experts repeated across rows on both halves of the
bank. The full-bank form stays one row, because its in-kernel slot sum is one row.

**Correctness, multi-row form** (second SE pair, `raw/se2-fused-rows-s2m/`):
- the component tests pass, the three-row case exact on both ranks;
- the long gate's `PROGRAM_SHA256` is `fbce1a0492d69635`;
- `dsv4_rows_gate` TP/EP passes: every B-row step bit-identical to its solo step across join, leave
  and row moves, and the sampled B-row graph draws equal the eager ones;
- `dsv4_kv_split_gate` passes;
- the DSpark TP/EP gate gives main's proposal shas, with batched equal to sequential over 3206 cache
  classes. It still reported `[FAIL]` on one finding, "MROW STREAM ENGAGEMENT: stream ON but the
  batched arm took 0 dispatches". The partition form now takes the verify rows the multi-row
  visitor took, so the claim was stale, not the program: every bit verdict passed. The gate now
  claims fused dispatches for the batched arm on TP/EP (`7ffe5281c`); re-run below.

**Rebased on main** (the DSpark round lane and the route fix under it; `raw/se2-rebased-s2n/`):
- the component tests pass, including the three-row case;
- `PROGRAM_SHA256` is `fbce1a0492d69635`;
- the rows and split gates pass;
- the DSpark TP/EP gate with the engagement fix reads `GPU DSPARK GATE [PASS]`. The batched arm
  takes 15480 fused dispatches and 0 multi-row visitor ones, with main's proposal shas.

**Served, multi-row form**, same pair, one boot per row, plain M F F M M F then DSpark M F F M M F,
N=3:

| cell | main | fused | delta |
|---|---|---|---|
| greedy c1 | 77.36 (76.57..77.84), decode 80.93 | 84.49 (84.45..84.87), decode 88.86 | +9.22% |
| sampled c1 | 77.36 | 84.80 | +9.62% |
| greedy c2 | 103.75 (103.31..105.18) | 116.45 (115.24..116.65) | +12.24% |
| greedy c4 | 130.48 (103.99..131.70) | 146.44 (95.26..146.64) | +12.2% on the medians |
| c2, 1500-word context | 19.10 (TTFT 10.32 s) | 19.16 (10.46 s) | +0.3% |
| DSpark greedy c1 | 88.99 (86.18..89.45) | 95.63 (94.57..95.81) | +7.46% |
| DSpark sampled c1 | 75.16 | 79.43 | +5.68% |

Every text in every cell is identical across both arms.

**The c4 cell on this pair is bimodal in both arms.** One main row and one fused row ran every step
of the cell about 1.5x slower: ITL p10 36 ms against 23 ms, and in one row the slowdown began
mid-cell. Nothing in the server logs differs between fast and slow boots. The same pair showed
this before (`../levers-20260926/`, two of five push boots). It is recorded as the pair's own
noise, and the c4 row wants a confirmation on the first pair.

**Confirmed on the first pair after the merge.** Main before the merge (`df006602e`) against main
with it (`5420d34bd`), one boot per row A B B A A B, N=3 (`raw/se-merge-v6d/`):

| cell | before | after | delta |
|---|---|---|---|
| greedy c1 | 78.15 (78.13..78.24), decode 82.00 | 85.67 (85.66..85.76), decode 90.03 | +9.62% |
| sampled c1 | 79.26 | 86.43 | +9.05% |
| greedy c2 | 106.92 | 119.33 | +11.61% |
| greedy c4 | 134.39 (98.67..136.10) | 151.08 (150.64..151.16) | +12.42% |
| c2, 1500-word context | 21.24 (TTFT 9.20 s) | 21.68 (9.20 s) | +2.07% |

One before-row ran its c4 cell at 98.67. The slow-c4 boot is rarer on this pair, but it is not
the second pair's alone.

## Sibling projections in one launch (adopted)

Lane `lane/dsv4-dense-pair-20260927`. A dense-fast launch pays about 2 us of ramp on top of its
bytes, and the TP/EP step issued four pairs back to back over the same activation rows:
- the shared expert's gate and up;
- wq_a and wkv;
- each compressor's kv and gate dots, in the one-row step and the hoisted B-row step.

The FP8 dense-fast and BF16 dots bodies are now device functions. The single kernels call them
with their block index. The pair kernels run matrix a's body on blocks `[0, nblk_a)` and matrix
b's on the rest, so both outputs keep the bits of their own launches. The pair launchers take the
pair only when both matrices admit the dense-fast transport at M = 1..8, and otherwise make the
two ordinary calls. wkv now projects with wq_a: nothing between them touches its output. About
130 fewer launches per step.

**Correctness** (second SE pair, `raw/se2-pair-s2p/`):
- the dense-fast component gate's 32 pair cases pass: both outputs equal the single launches at
  M = 1, 2, 4 and 6 on the four pair shapes, and each capture holds one pair node;
- the long gate's `PROGRAM_SHA256` is `fbce1a0492d69635`;
- the TP/EP rows gate passes;
- the DSpark TP/EP gate passes with main's proposal shas;
- every served text is identical across arms.

**Long-gate replay**, F P P F: 11.06 to 11.19 ms per token against 11.31 to 11.44, -2.2%.

**Served**, one boot per row F P P F F P, N=3, F being the fused lane:

| cell | fused | fused + pairs | delta |
|---|---|---|---|
| greedy c1 | 85.04 (84.60..85.29), decode 88.97 | 86.51 (86.37..87.11), decode 90.79 | +1.73% |
| sampled c1 | 85.09 | 86.72 | +1.92% |
| greedy c2 | 116.64 | 117.58 | +0.81% |
| greedy c4 | 145.22 | 149.89 | +3.22% |
| c2, 1500-word context | 23.58 (TTFT 8.19 s) | 24.04 (8.00 s) | +1.95% |

Every pair row is above every fused row at c1.

## Refuted: loading weights before the PDL wait

The weights and block scales of the dense-fast FP8 GEMV, the BF16 dots and the HC split partial
are checkpoint constants. The lane opened each of these kernels with a prologue that loaded them
before `griddepcontrol.wait`, so they could stream while the predecessor ran. A new
`MEMRA_PDL_PRE_WAIT(p, ...)` declaration, checked by `tools/check-pdl-chain.py`, named the only
pointers the prologue could read. The patch is `raw/se-prologue-v5v/prologue.patch`.

It was bit-identical:
- the long gate's hash matched with PDL on and off;
- the dense-fast component gate passed, extended to wo_b's 8192-long row and to rows longer than
  the prologue;
- the rows, split and DSpark gates passed;
- every greedy text was identical.

It was slower. SE pair, one boot per row, order M P P M M P P M M P, N=5
(`raw/se-prologue-v5v/`):

| cell | main | prologue | delta |
|---|---|---|---|
| greedy c1 | 77.82 (77.58..78.17) | 76.45 (76.30..76.54) | -1.76% |
| sampled c1 | 79.04 | 77.50 | -1.95% |
| greedy c2 | 106.58 | 104.13 | -2.30% |
| greedy c4 | 135.00 | 135.13 | +0.10% |

Every prologue row is below every main row at c1 and c2. The loads the prologue moves ahead of
the wait compete with a predecessor that is itself streaming weights, so they cannot shorten the
chain. What they cost is not isolated: a register-resident prologue, blocks resident while they
wait, or the rewritten loop's schedule. The lane is not merged.

## Refuted: a deeper MoE stream ring

The one-token and multi-row visitors stream each warp's rows through a 4-stage ring of 128-k
chunks. The ring depth is a compile-time constant, so each variant is a box-local build that
changes only that define (`raw/se2-ring-s2i/variant-*.diff`). The long gate's replay ms per token
was measured three reps per run, order M S4 S8 S12 S12 S8 S4 M:

| build | replay ms/token (6 reps) | vs main |
|---|---|---|
| main | 12.39 .. 12.52 | |
| S4 (committed depth, with gate and up in one launch) | 12.37 .. 12.50 | flat |
| S8 | 12.60 .. 12.72 | +1.6% |
| S12 | 12.68 .. 12.86 | +2.3% |

Every build keeps the program hash. More 64-byte requests in flight make the stream slower, not
faster, which points at the access pattern, not at the queue depth.

**S4 is not merged either.** It put gate and up in one launch of the stream visitors and chained
the multi-row visitor on PDL (`raw/se2-ring-s2i/gu-launch.patch`). It was flat at c1, and on
TP/EP the fused pair's multi-row form now takes every step those visitors took, so only the PP-2
rollback would still run them.

## Refuted: more output rows per dense-fast block

The FP8 dense-fast GEMV puts two output rows (256 threads) in a block, and every block reads the
whole activation row. More rows per block would share those reads through L1. Box-local builds
changed only the rows constant, for the one-token and grouped launches and for the M-row
launches (`raw/se-rows-v6a/`). Long-gate replay ms per token, order Z A B C C B A Z, first-pod
builds of the fused lane:

| build (rows, M-row rows) | replay ms/token |
|---|---|
| Z (2, 2), committed | 11.12 .. 11.26 |
| A (4, 4) | 11.09 .. 11.25, flat |
| B (8, 4) | 11.67 .. 11.90, +5% |
| C (8, 2) | 11.68 .. 12.05, +5% |

Every build keeps the program hash. Eight rows halve the grid, and the single-wave kernels lose
more to the smaller grid than they gain from shared reads. The constants are not merged.

## The fused pair's ring

The pair streams KC k per stage through a STAGES-deep ring; the committed setting is (256, 2). On
the second SE pair, box-local builds, order M F G H H G F M (`raw/se2-fused-ring-s2k/`):

| build | replay ms/token |
|---|---|
| main | 12.37 .. 12.50 |
| F (256, 2), committed | 11.32 .. 11.39 |
| G (256, 3) | 11.24 .. 11.31, -0.6% |
| H (512, 2) | 11.25 .. 11.35, -0.6% |

The ring is not what holds the pair near 0.9 TB/s. The next guess was the CTA count: with about
three local experts, the committed 4 warps per projection give about 192 live CTAs of 8 warps on
188 SMs, so the SMs that hold two CTAs would set the kernel time. Wider grids of narrower CTAs
refute it. On the first pair, order F I J K K J I F (`raw/se-fused-cta-v6c/`):

| build (warps per projection, KC, stages) | replay ms/token |
|---|---|
| F (4, 256, 2), committed | 11.12 .. 11.27 |
| I (2, 256, 3) | 11.40 .. 11.52 |
| K (2, 512, 2) | 11.50 .. 11.64 |
| J (1, 256, 3) | 12.60 .. 12.76 |

Each CTA computes its row's x mirror before it streams, so halving the warps per CTA doubles the
mirrors and halves the warps that share each one. Nsight Compute on the committed pair
(`raw/se-ncu-v6b/`) reads 18% of peak warps active, 27% of SM throughput and 22% of L2
throughput. The committed setting stays.
