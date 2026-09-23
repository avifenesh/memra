# DSv4 indexer scorer: dispatch knee (#451)

Question: the tiled indexer scorer (`dsv4_indexer_score_tiled_kernel`) is bit-identical to the
ordered scalar scorer and much faster at long context, but the door
`MEMRA_DSV4_INDEXER_SCORE` kept it default-OFF. Where does tiled start to win, and can the
default pick it there?

Answer: yes. Tiled wins for one query row from about 1150 candidates (4.6k tokens of context at
the ratio-4 compressor) and for more rows from about 8192 row-candidates. The default is now a
knee dispatch on those two thresholds; `scalar` and `tiled` stay as forced seams.

## Rig and binary

- One RTX PRO 6000 Blackwell Workstation Edition, driver 610.57.04, 600 W limit, 3090 MHz max
  SM clock. Single card, no co-tenant.
- Tool: `tools/dsv4-indexer-tiled-gate.cu` at 6ecb3ffdc (runs 1 and 2) and 73b235479 (run 3,
  adds 16 and 32 rows), built `nvcc -O3 -arch=sm_120a -fmad=false -Xcompiler=-ffp-contract=off`
  exactly like the engine TU. `--knee` times both arms per cell: 5 interleaved repeats
  (arm order alternates per repeat), 5 launches each, CUDA events. Every timed cell is also a
  full bit comparison of scalar vs tiled outputs, mask and tail guards.
- One-row cells use the fixed-limit f32acc scorer (plain decode: `lim0 = nb`). Multi-row
  cells use the absolute-position `pos_m` scorer (DSpark verify at 6 rows, chunked prefill at
  64 to 512 rows) with `pos0 = 4*nb - rows`.

## Correctness

- Base gate: PASS, 6,178,334 comparisons (`raw/gate.log`).
- Teeth: one output ulp flipped is caught, rc=1 "score mismatch at 0" (`raw/teeth.log`).
- Knee sweeps: runs 1 and 2 PASS 29,903,495 comparisons each, run 3 PASS 31,608,503.

## Knee (medians of the per-run medians, 3 runs; runs 1 and 2 have no 16 or 32 row cells)

| rows | candidates | scalar µs | tiled µs | scalar/tiled | runs | knee picks |
|---|---|---|---|---|---|---|
| 1 | 129 | 6.5 | 21.4 | 0.30 | 3 | scalar |
| 1 | 192 | 8.1 | 21.4 | 0.38 | 3 | scalar |
| 1 | 256 | 8.4 | 21.2 | 0.40 | 3 | scalar |
| 1 | 384 | 10.9 | 21.5 | 0.51 | 3 | scalar |
| 1 | 512 | 11.2 | 21.5 | 0.52 | 3 | scalar |
| 1 | 768 | 17.4 | 21.9 | 0.79 | 3 | scalar |
| 1 | 1024 | 19.9 | 21.9 | 0.91 | 3 | scalar |
| 1 | 1536 | 29.6 | 21.9 | 1.35 | 3 | tiled |
| 1 | 2048 | 35.5 | 21.9 | 1.62 | 3 | tiled |
| 1 | 3072 | 53.4 | 21.9 | 2.44 | 3 | tiled |
| 1 | 4096 | 68.2 | 22.0 | 3.10 | 3 | tiled |
| 1 | 8192 | 133.5 | 22.0 | 6.08 | 3 | tiled |
| 1 | 16384 | 263.5 | 22.4 | 11.77 | 3 | tiled |
| 1 | 32768 | 524.0 | 32.2 | 16.27 | 3 | tiled |
| 1 | 65536 | 1037.6 | 44.0 | 23.56 | 3 | tiled |
| 6 | 129 | 6.4 | 21.9 | 0.29 | 3 | scalar |
| 6 | 192 | 7.8 | 21.9 | 0.35 | 3 | scalar |
| 6 | 256 | 7.8 | 22.0 | 0.35 | 3 | scalar |
| 6 | 384 | 8.2 | 22.0 | 0.37 | 3 | scalar |
| 6 | 512 | 10.0 | 22.2 | 0.45 | 3 | scalar |
| 6 | 768 | 13.7 | 22.2 | 0.62 | 3 | scalar |
| 6 | 1024 | 16.9 | 22.2 | 0.76 | 3 | scalar |
| 6 | 1536 | 22.7 | 22.3 | 1.02 | 3 | tiled |
| 6 | 2048 | 28.0 | 22.2 | 1.26 | 3 | tiled |
| 6 | 4096 | 52.6 | 30.4 | 1.73 | 3 | tiled |
| 6 | 8192 | 102.3 | 40.3 | 2.54 | 3 | tiled |
| 6 | 16384 | 202.5 | 71.2 | 2.85 | 3 | tiled |
| 16 | 129 | 10.7 | 22.1 | 0.48 | 1 | scalar |
| 16 | 192 | 14.2 | 22.0 | 0.65 | 1 | scalar |
| 16 | 256 | 18.2 | 22.2 | 0.82 | 1 | scalar |
| 16 | 384 | 20.7 | 22.3 | 0.93 | 1 | scalar |
| 16 | 512 | 22.7 | 22.2 | 1.02 | 1 | tiled |
| 16 | 768 | 29.7 | 22.3 | 1.33 | 1 | tiled |
| 16 | 1024 | 37.4 | 22.3 | 1.68 | 1 | tiled |
| 16 | 1536 | 53.4 | 30.5 | 1.75 | 1 | tiled |
| 16 | 2048 | 69.8 | 32.1 | 2.18 | 1 | tiled |
| 16 | 4096 | 135.7 | 41.5 | 3.27 | 1 | tiled |
| 16 | 8192 | 268.6 | 80.7 | 3.33 | 1 | tiled |
| 16 | 16384 | 533.3 | 143.4 | 3.72 | 1 | tiled |
| 32 | 129 | 20.1 | 22.2 | 0.91 | 1 | scalar |
| 32 | 192 | 26.0 | 22.2 | 1.17 | 1 | scalar |
| 32 | 256 | 28.5 | 22.3 | 1.28 | 1 | tiled |
| 32 | 384 | 33.3 | 22.3 | 1.50 | 1 | tiled |
| 32 | 512 | 39.5 | 22.3 | 1.77 | 1 | tiled |
| 32 | 768 | 55.0 | 30.4 | 1.81 | 1 | tiled |
| 32 | 1024 | 71.8 | 32.1 | 2.24 | 1 | tiled |
| 32 | 1536 | 104.6 | 40.4 | 2.59 | 1 | tiled |
| 32 | 2048 | 136.2 | 41.5 | 3.28 | 1 | tiled |
| 32 | 4096 | 269.4 | 80.6 | 3.34 | 1 | tiled |
| 32 | 8192 | 533.1 | 143.6 | 3.71 | 1 | tiled |
| 32 | 16384 | 1060.5 | 284.2 | 3.73 | 1 | tiled |
| 64 | 129 | 29.3 | 22.3 | 1.31 | 3 | tiled |
| 64 | 192 | 37.5 | 22.3 | 1.69 | 3 | tiled |
| 64 | 256 | 44.5 | 22.3 | 2.00 | 3 | tiled |
| 64 | 384 | 59.6 | 30.5 | 1.96 | 3 | tiled |
| 64 | 512 | 74.7 | 32.1 | 2.32 | 3 | tiled |
| 64 | 768 | 106.0 | 40.3 | 2.63 | 3 | tiled |
| 64 | 1024 | 138.7 | 41.5 | 3.34 | 3 | tiled |
| 64 | 1536 | 205.4 | 71.5 | 2.87 | 3 | tiled |
| 64 | 2048 | 270.1 | 80.5 | 3.35 | 3 | tiled |
| 64 | 4096 | 533.9 | 144.1 | 3.70 | 3 | tiled |
| 64 | 8192 | 1063.0 | 284.8 | 3.73 | 3 | tiled |
| 64 | 16384 | 2147.0 | 568.9 | 3.77 | 3 | tiled |
| 256 | 129 | 67.9 | 41.9 | 1.62 | 3 | tiled |
| 256 | 192 | 104.7 | 42.7 | 2.45 | 3 | tiled |
| 256 | 256 | 142.0 | 42.7 | 3.32 | 3 | tiled |
| 256 | 384 | 210.7 | 72.2 | 2.92 | 3 | tiled |
| 256 | 512 | 278.5 | 81.7 | 3.41 | 3 | tiled |
| 256 | 768 | 412.2 | 121.8 | 3.39 | 3 | tiled |
| 256 | 1024 | 541.8 | 144.7 | 3.75 | 3 | tiled |
| 256 | 1536 | 803.8 | 221.8 | 3.62 | 3 | tiled |
| 256 | 2048 | 1068.2 | 285.5 | 3.74 | 3 | tiled |
| 256 | 4096 | 2140.1 | 571.3 | 3.75 | 3 | tiled |
| 256 | 8192 | 4300.8 | 1140.4 | 3.77 | 3 | tiled |
| 256 | 16384 | 8851.8 | 2356.4 | 3.76 | 3 | tiled |
| 512 | 129 | 92.0 | 53.8 | 1.71 | 3 | tiled |
| 512 | 192 | 161.1 | 70.3 | 2.29 | 3 | tiled |
| 512 | 256 | 235.6 | 82.1 | 2.87 | 3 | tiled |
| 512 | 384 | 379.1 | 121.9 | 3.11 | 3 | tiled |
| 512 | 512 | 520.0 | 145.6 | 3.57 | 3 | tiled |
| 512 | 768 | 789.9 | 222.0 | 3.56 | 3 | tiled |
| 512 | 1024 | 1049.7 | 285.2 | 3.68 | 3 | tiled |
| 512 | 1536 | 1571.0 | 424.5 | 3.70 | 3 | tiled |
| 512 | 2048 | 2102.6 | 565.5 | 3.72 | 3 | tiled |
| 512 | 4096 | 4257.9 | 1131.9 | 3.76 | 3 | tiled |
| 512 | 8192 | 8869.2 | 2358.6 | 3.76 | 3 | tiled |
| 512 | 16384 | 17829.9 | 4726.8 | 3.77 | 3 | tiled |

The rule `rows == 1 ? nb >= 1152 : rows * nb >= 8192` picks the faster arm in 90 of 91
cells. The miss is 32 rows at 192 candidates (tiled 1.17x faster, about 4 µs), left as is
rather than adding a third threshold.

## What the default changes

- Short context serving (under about 4.6k tokens, nb < 1152) keeps the scalar scorer on plain
  decode and DSpark verify, so the served program there is unchanged.
- Long context decode takes tiled: at 32k tokens (nb 8192) one row costs 22 µs instead of
  133.5 µs per indexer launch, at 128k (nb 32768) 32.4 µs instead of 524 µs.
- Chunked prefill takes tiled from its first chunk at 64 or more rows: 3.7x at nb >= 1024.
- Full-token replay stays pinned to scalar (`scalar_only` at the verify call site); a forced
  `tiled` still refuses the replay program.
- A program the tiled kernel cannot run (host math, non-f32x, indexer shape other than 64x128)
  resolves the knee to scalar at load; only a forced `tiled` refuses there.

## Served evidence

Pending on a 2x PRO 6000 pair: long-context identity (32k and 128k prompts, greedy, lane vs
main binary, same hashes) and TTFT/decode rows. Rows land here before the PR merges.
