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

**Long-gate replay:** 11.33 ms per token, against 12.39 to 12.51 for main on the same pod.

**Served:** pending (`raw/se2-fused-s2j/`, order M F F M M F F M M F).

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
