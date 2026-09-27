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

## Register-resident row kernels, and wq_b paired with the indexer's (adopted)

Lane `lane/dsv4-small-regs-20260927`, stacked on the sibling pairs.

**Register-resident row kernels.** The q-LoRA norm pack and the per-head RMS are one-block-per-row
kernels on the decode chain. Both reloaded x for their scale pass and reduced through
`block_sum_f32`'s seven-barrier tree. At 128 threads they now load every owned element (and
weight) before the first add, keep them for the scale pass, and reduce through
`dsv4_block_sum128_f32`. That tree pairs exactly as `block_sum_f32` does at 128 threads, the
pattern `dsv4_rmsnorm_f32acc_regs` already uses.

**wq_b paired with the indexer's.** On the 21 ratio-4 layers, wq_b and the indexer's wq_b read the
same q-LoRA rows, so they now share one pair launch (the previous section's kernel).

**Served** (second SE pair, `raw/se2-small-s2q/`), register kernels only against the sibling-pair
tree, one boot per row P S S P P S, N=3:

| cell | pairs | pairs + register kernels | delta |
|---|---|---|---|
| greedy c1 | 86.99 (86.44..87.18), decode 91.10 | 88.36 (87.59..88.44), decode 92.62 | +1.57% |
| sampled c1 | 87.42 | 88.50 | +1.24% |
| greedy c2 | 118.61 | 119.37 | +0.64% |
| greedy c4 | 150.37 | 152.44 | +1.38% |

**Long-gate replay**, P S S P: 10.92 to 10.99 ms per token against 11.10 to 11.22.

Every gate keeps `PROGRAM_SHA256` `fbce1a0492d69635`, and every served text is identical across
arms.

The wq_b pair on top (`raw/se2-small2-s2r/`):
- `PROGRAM_SHA256` unchanged;
- the rows and DSpark gates pass;
- long-gate replay S T T S: 10.86 to 10.99 against 10.91 to 11.01 ms per token, about -0.3%, the
  size its 21 launches predict.

## The shared expert on the rank with fewer routed slots (adopted)

Lane `lane/dsv4-shared-owner-main-20260927`. Raw data is in `raw/se2-shared-owner-s2x/`: the queue
script, the summary, every gate and served row.

**Why.**
- TP/EP ran the shared expert on both ranks, after the expert join. That is 25.2 MB of FP8
  weights per layer (gate, up and down at 2048 x 4096), about 1.08 GB per step per rank: a
  third of the dense GEMV bytes.
- The routed experts split between the ranks as Binomial(6, 0.5), so one rank waits in the join
  for the other in most layers.
- The shared expert reads only the MoE input, which is replicated. Which rank computes it does
  not change a bit.

**What changed.**
1. **The owner word.** One warp of the fused gate/up launch counts the step's routed slots on
   each rank. It writes the rank's owner word: 1 on the rank with fewer slots, rank 0 on a tie.
2. **The owner's shared expert.** After the fused down, the owner rank runs the shared expert:
   - the bf16 pack of the MoE input;
   - the gate/up pair;
   - SwiGLU with its bf16 pack, in one launch;
   - down, into the rows after the routed ones in the contribution plane.
   The other rank runs the same four launches, and each exits at entry
   (`memra_dsv4_gemv_fp8_m_gated`, `_pair_gated`, `memra_dsv4_cvt_bf16_gated`,
   `memra_dsv4_swiglu_bf16_gated`). Every block runs the ungated launch's body.
3. **The join and the tail.** The expert join carries the shared rows too. The tail adds them
   where it used to recompute the shared expert. That add is `y + (sh + 0.0)`, and a dense-fast
   output cannot be -0.0 because its partials start at +0.0 and only add, so y keeps its bits.
4. **Fallback.** A step of more than 8 rows, or one where a launch would leave the dense-fast
   transport, keeps the replicated shared expert.

**Expected size.** One routed expert is about 15.5 us per layer at the pair's rate, and the
shared expert about 22.5 us. Over the Binomial(6, 0.5) split, the critical path falls from
about 83.6 us per layer to about 68 us, less the gated no-op launches on the heavier rank:
about 0.5 ms per step.

**Correctness** (second SE pair):
- The fused partition fixture asserts the owner word on both ranks of every clean case.
- The dense-fast component gate's `gated_case` compares each gated launch with its ungated
  launch at M = 1, 2, 4 and 8 on both cards. It also checks that a clear word moves no output
  byte: `PASS gated ... bits=1 owner_off_untouched=1`, eight lines.
- The long gate's `PROGRAM_SHA256` is `fbce1a0492d69635`.
- The TP/EP rows gate, the KV split gate and the DSpark TP/EP gate pass. DSpark's proposal shas
  are the pair's.
- Every served request's text is identical in all six rows.

**Long gate.** Replay ms/token, order M S S M M S: main 10.84 .. 11.01 against 10.38 .. 10.59,
**-4.2%**.

**Served.** cells-pdl, one boot per row, order M S S M M S, N=3 per arm:

| cell | main agg tok/s | lane agg tok/s | delta |
|---|---|---|---|
| greedy c1 | 88.52 / 88.84 / 88.76 | 91.96 / 92.39 / 92.07 | **+3.9%** |
| sampled c1 | 88.26 / 89.37 / 89.33 | 91.79 / 92.79 / 91.95 | +3.6% |
| greedy c2 | 118.81 / 120.37 / 120.16 | 122.73 / 124.15 / 122.19 | +2.7% |
| greedy c4 | 150.91 / 153.56 / 151.96 | 112.33 / 157.55 / 152.96 | see below |
| greedy c2, 2k prompt | 23.81 / 23.87 / 24.10 | 23.88 / 23.25 / 24.14 | flat |

TPOT p50 at c1 falls from 10.74 .. 10.76 ms to 10.31 .. 10.35 ms. One lane boot (r2) ran its c4
cell at 112 tok/s with TPOT p50 34 ms. Its c1 and c2 cells were in line, and the other two lane
boots ran c4 at 157.6 and 153.0. Slow-c4 boots have been seen on main on both pairs. This one
is kept, and the rebased rows below add c4 rows. Thermal: median power 267 .. 271 W while the
cards work, SM clock median 2400 MHz, max 50 C.

**Rebased on main `c566d2096`** (with the Sinkhorn warp and the prefill tile), same pair
(`raw/se2-shared-owner-rebased-s2y/`):
- The fused partition fixtures pass, and so do the dense-fast gate's eight gated cases.
- The long gate hash is `fbce1a0492d69635`.
- The TP/EP rows gate, the KV split gate and the DSpark TP/EP gate pass.
- Long gate, order M S S M: 10.65 .. 10.77 ms/token against 10.17 .. 10.27, **-4.7%**.
- Served cells-pdl, order M S S M, N=2:

| cell | main agg tok/s | lane agg tok/s | delta |
|---|---|---|---|
| greedy c1 | 91.18 / 90.92 (decode 95.3 / 95.1) | 94.66 / 94.14 (decode 99.1 / 98.9) | **+3.7%** |
| sampled c1 | 91.41 / 91.31 | 94.73 / 93.97 | +3.3% |
| greedy c2 | 122.53 / 122.22 | 124.71 / 125.35 | +2.0% |
| greedy c4 | 153.85 / 155.82 | 157.81 / 158.67 | +2.2% |
| greedy c2, 2k prompt | 24.63 / 26.04 | 25.63 / 25.48 | flat |

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

## What the fused pair's mirrors cost, and two refuted ways to recover it

**Probe** (second SE pair, `raw/se2-mirror-probe-s2s/`). These are box-local builds of main that skip
the mirrors, using `mirror_hack.py` with the variant diffs kept. The outputs are wrong by design,
so only the long gate's replay ms per token is read. Order M X1 X2 X2 X1 M:

| build | replay ms/token |
|---|---|
| main | 11.10 .. 11.18 |
| X1, no x mirror in the gate/up launch | 11.02 .. 11.14 |
| X2, no x mirror and no h mirror in the down launch | 10.76 .. 10.86 |

Every down CTA of a slot rebuilds the same h mirror from the f32 row, and that costs about
0.3 ms per step.

**Refuted: the h mirror published once per slot** (`raw/se2-hmirror-s2t/`, code in
`h-mirror.patch`).
- Design: the last gate/up CTA of each slot to finish its h columns builds the mirror once,
  from the row through L2. It writes the swizzled halves and the row scale to global. Down CTAs
  load them instead of rebuilding.
- The bits are the same by construction: `PROGRAM_SHA256 fbce1a0492d69635`. The fused-pair
  component tests pass, with a standalone publish kernel for the fixtures that edit h between
  launches. The TP/EP rows gate passes, and so does the DSpark TP/EP gate with the pair's
  proposal shas.
- Speed: it is slower. Long gate, order M H H M M H: main 10.87 .. 11.00 ms/token against
  10.94 .. 11.03.
- Served cells-pdl, same order, N=3:
  - greedy c1: 88.32 .. 88.77 tok/s on main against 87.83 .. 87.95, -0.7%;
  - sampled c1: 88.53 .. 88.98 against 87.88 .. 88.29;
  - c2 and c4 flat within row noise.
- Why: the publish is a serial tail at the end of the gate/up launch that nothing overlaps. It
  costs more than the parallel rebuild it replaces, whose loads overlap each down CTA's first
  weight stage.

**Refuted: register-held mirror groups with exact reciprocal divides**
(`raw/se2-mirror-regs-s2v/`, `m2-mirror-regs.patch` on main, `h2-on-h.patch` on the published
form).
- Design: each warp keeps its groups' float4 values in registers across the mirror's two passes,
  with every load in flight at once. The divisions by the power-of-two group and row scales
  become multiplies by their exact reciprocals. `x * (1 / 2^k)` is the same correctly rounded
  value as `x / 2^k`.
- The bits are the same: `fbce1a0492d69635` on every build.
- Long gate, order M M2 H H2 H2 H M2 M:

| build | replay ms/token |
|---|---|
| M, main | 10.86 .. 10.95 |
| M2, main with the held groups | 11.05 .. 11.15, +1.8% |
| H, the published mirror | 10.91 .. 11.01 |
| H2, the published mirror with the held groups | 10.95 .. 11.02 |

The held groups push the gate/up kernel from 47 registers to 57 (55 in H2): `cuobjdump
-res-usage` on the gate binaries. That takes its 256-thread CTAs from five per SM to four, which
costs more than the saved loads. The down kernel stays at 48. Neither form is merged. Recovering
the probe's 0.3 ms needs a mirror that adds no work to the gate/up launch's critical path and
no registers to it.

## Refuted: dense-fast blocks prefetching their weight rows into L2 before the PDL wait

Weights never depend on the predecessor. A dense-fast block that becomes resident while a
latency-bound predecessor runs (an HC finish, a norm) could start its stream from L2. The probe
issues `prefetch.global.L2` over each block's weight rows ahead of `griddepcontrol.wait`. It covers
the single, pair and dots launches (`raw/se2-l2-prefetch-s2w/l2-prefetch.patch`). A prefetch
moves no value, and the hash stays `fbce1a0492d69635`.

Long gate, second SE pair, order M P P M M P: main 10.85 .. 11.04 ms/token against 10.82 .. 10.95.
That is flat. Few blocks are resident early enough to matter. Where they are, the predecessor's
own stream already holds the bandwidth. Not merged.

## The fused pair on one card: where its time goes, and a refuted wider CTA

`tools/dsv4-moe-fused-bench.cu` times the pair on one card at the TP/EP partition shape: 128
local experts of 256, 6 slots per row, 3 of them local. Each launch takes a fresh expert set, so
the weights stream from DRAM.

Its variants instantiate the kernel templates directly:
- a stage-major packed copy of the weights (each warp's stage contiguous), checked bit for bit
  against the standard kernel;
- decomposition twins that stream without computing, or compute without streaming, with and
  without the x mirror;
- other ring depths and CTA widths.

Receipts: `raw/se2-moe-bench-s2z/`, `raw/se2-moe-bench2-s2za/`, `raw/se2-moe-bench3-s2zb/`,
`raw/se2-moe-bench4-s2zc/`. Second SE pair. The pair has no performance-counter access, so the
Nsight run in s2z refused.

**One token row**, gate/up launch, us:

| arm | us | TB/s over the weights |
|---|---|---|
| the committed kernel | 28.6 | 0.99 |
| its skeleton (no epilogue) | 27.9 | 1.02 |
| no x mirror | 22.8 | 1.24 |
| streaming only, no mirror | 20.7 | 1.37 |
| compute only, no loads | 18.7 | 1.51 |
| the stage-major packed copy | 29.7 | 0.95 |

At one row the launch is compute and stream in about equal parts, overlapped well, plus about
5 us of x mirror that does not overlap. The packed layout loses 3%, which refutes the access
pattern as the limit. At four rows the same kernel reads 1.17 TB/s, and the down launch 0.95.

**Refuted: 16-warp CTAs.** On the bench, 16-warp CTAs cut both launches at one row:
- gate/up from 28.6 to 25.2 us (8 warps per projection);
- down from 17.5 to 15.2 us (16 warps).

Fewer CTAs rebuild each mirror. In the program they lose. Long gate, second SE pair, order
M W W M M W, the bits the same (`fbce1a0492d69635`, `raw/se2-moe-wide-s2zd/`, code in
`wide-cta.patch`):
- main: 10.60 .. 10.70 ms/token;
- lane: 10.74 .. 10.87, +1.4%.

One served row per arm agrees: greedy c1 91.00 against 90.30. The bench times each launch alone.
In the chain, the wider CTAs leave half the SMs idle at one row (96 CTAs of 512 threads against
192 of 256), and that costs more than the mirrors it saves. Not merged.
