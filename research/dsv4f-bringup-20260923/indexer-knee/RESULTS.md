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

2x RTX PRO 6000 WS (450 W), lane `798e12674` (`idx`) against main `6978f5fac` (`base`), one boot
per row in the order `idx base base idx idx base base idx`, N=4 per arm per mode, plain then
DSpark (`raw/served/q-pairL.sh`). Cells (`raw/served/cells-long.txt`): two greedy requests at
about 32k prompt tokens (22,400 words) and one at about 64k (44,800 words), 256 output tokens
each, text kept and hashed. `MEMRA_TIMEOUT_MS_MAX=900000` lifts the 90 s default ceiling, the
documented measurement-cell override. The first run of this campaign is void and kept as
`raw/served/long-void-408/`: every long request hit the default ceiling (HTTP 408) before its
first token.

### Plain


#### greedy-32k

| row | arm | decode p50 tok/s | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | ok |
|---|---|---|---|---|---|---|
| r1 | idx | 44.90 | 22.27 / 22.31 / 22.31 | 110,278 / 110,362 | 115,956 | 2/2 |
| r2 | base | 40.81 | 24.50 / 24.55 / 24.55 | 115,550 / 115,900 | 121,799 | 2/2 |
| r3 | base | 40.80 | 24.51 / 24.55 / 24.55 | 115,713 / 115,983 | 121,962 | 2/2 |
| r4 | idx | 44.89 | 22.28 / 22.32 / 22.32 | 110,774 / 110,983 | 116,455 | 2/2 |
| r5 | idx | 44.85 | 22.30 / 22.36 / 22.36 | 110,833 / 111,023 | 116,519 | 2/2 |
| r6 | base | 40.80 | 24.51 / 24.55 / 24.55 | 115,799 / 116,011 | 122,049 | 2/2 |
| r7 | base | 40.81 | 24.51 / 24.54 / 24.55 | 115,697 / 116,020 | 121,946 | 2/2 |
| r8 | idx | 44.89 | 22.28 / 22.32 / 22.32 | 110,825 / 111,058 | 116,506 | 2/2 |

| arm | N | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|
| idx | 4 | 44.89 | 22.28 | 110,800 |
| base | 4 | 40.80 | 24.51 | 115,705 |

Delta, lane vs base: decode +10.01%, TTFT -4.24%.

Hashes: identical across every row of both arms; 2 requests, first 9a8aa4bc 17f9f998.

#### greedy-64k

| row | arm | decode p50 tok/s | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | ok |
|---|---|---|---|---|---|---|
| r1 | idx | 43.56 | 22.96 / 22.96 / 22.96 | 230,334 / 230,334 | 236,188 | 1/1 |
| r2 | base | 35.93 | 27.83 / 27.83 / 27.83 | 249,586 / 249,586 | 256,683 | 1/1 |
| r3 | base | 35.93 | 27.83 / 27.83 / 27.83 | 249,268 / 249,268 | 256,366 | 1/1 |
| r4 | idx | 43.55 | 22.96 / 22.96 / 22.96 | 230,644 / 230,644 | 236,499 | 1/1 |
| r5 | idx | 43.56 | 22.96 / 22.96 / 22.96 | 230,615 / 230,615 | 236,469 | 1/1 |
| r6 | base | 35.93 | 27.83 / 27.83 / 27.83 | 249,699 / 249,699 | 256,797 | 1/1 |
| r7 | base | 35.92 | 27.84 / 27.84 / 27.84 | 250,021 / 250,021 | 257,120 | 1/1 |
| r8 | idx | 43.56 | 22.96 / 22.96 / 22.96 | 230,622 / 230,622 | 236,476 | 1/1 |

| arm | N | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|
| idx | 4 | 43.56 | 22.96 | 230,618 |
| base | 4 | 35.93 | 27.83 | 249,643 |

Delta, lane vs base: decode +21.24%, TTFT -7.62%.

Hashes: identical across every row of both arms; 1 requests, first e84068e3.

Thermal idx: power median 324..330 W, peak 633 W, SM clock 2287..2865 MHz, max temp 64 C, 8234 samples.

Thermal base: power median 335..342 W, peak 609 W, SM clock 2265..2865 MHz, max temp 64 C, 8833 samples.

### DSpark (`MEMRA_DSV4_DRAFTER=dspark`)


#### greedy-32k

| row | arm | decode p50 tok/s | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | ok |
|---|---|---|---|---|---|---|
| r1 | idx | 53.00 | 18.87 / 20.89 / 21.07 | 114,258 / 114,583 | 119,070 | 2/2 |
| r2 | base | 43.23 | 23.13 / 25.60 / 25.82 | 119,221 / 119,451 | 125,120 | 2/2 |
| r3 | base | 43.23 | 23.13 / 25.60 / 25.82 | 119,345 / 119,682 | 125,243 | 2/2 |
| r4 | idx | 52.99 | 18.87 / 20.89 / 21.07 | 114,232 / 114,569 | 119,044 | 2/2 |
| r5 | idx | 52.99 | 18.87 / 20.90 / 21.08 | 114,272 / 114,584 | 119,084 | 2/2 |
| r6 | base | 43.22 | 23.14 / 25.60 / 25.82 | 117,464 / 118,577 | 123,364 | 2/2 |
| r7 | base | 43.22 | 23.14 / 25.61 / 25.83 | 117,714 / 118,356 | 123,614 | 2/2 |
| r8 | idx | 52.99 | 18.87 / 20.89 / 21.07 | 113,957 / 114,068 | 118,769 | 2/2 |

| arm | N | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|
| idx | 4 | 52.99 | 18.87 | 114,245 |
| base | 4 | 43.22 | 23.14 | 118,467 |

Delta, lane vs base: decode +22.60%, TTFT -3.56%.

Hashes: identical across every row of both arms; 2 requests, first 9a8aa4bc 17f9f998.

#### greedy-64k

| row | arm | decode p50 tok/s | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | ok |
|---|---|---|---|---|---|---|
| r1 | idx | 47.36 | 21.12 / 21.12 / 21.12 | 237,538 / 237,538 | 242,922 | 1/1 |
| r2 | base | 32.39 | 30.87 / 30.87 / 30.87 | 256,947 / 256,947 | 264,820 | 1/1 |
| r3 | base | 32.38 | 30.88 / 30.88 / 30.88 | 258,058 / 258,058 | 265,932 | 1/1 |
| r4 | idx | 47.37 | 21.11 / 21.11 / 21.11 | 237,892 / 237,892 | 243,275 | 1/1 |
| r5 | idx | 47.37 | 21.11 / 21.11 / 21.11 | 236,801 / 236,801 | 242,184 | 1/1 |
| r6 | base | 32.37 | 30.89 / 30.89 / 30.89 | 258,011 / 258,011 | 265,888 | 1/1 |
| r7 | base | 32.38 | 30.88 / 30.88 / 30.88 | 255,463 / 255,463 | 263,338 | 1/1 |
| r8 | idx | 47.35 | 21.12 / 21.12 / 21.12 | 231,518 / 231,518 | 236,903 | 1/1 |

| arm | N | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |
|---|---|---|---|---|
| idx | 4 | 47.36 | 21.11 | 237,170 |
| base | 4 | 32.38 | 30.88 | 257,479 |

Delta, lane vs base: decode +46.27%, TTFT -7.89%.

Hashes: identical across every row of both arms; 1 requests, first e84068e3.

Thermal idx: power median 330..339 W, peak 680 W, SM clock 2295..2865 MHz, max temp 66 C, 8335 samples.

Thermal base: power median 338..348 W, peak 571 W, SM clock 2235..2865 MHz, max temp 65 C, 9036 samples.

## Verdict

**The knee default wins at long context on both served modes, with the same text.** Plain decode
is 10.0% faster at 32k (40.80 -> 44.89 tok/s) and 21.2% faster at 64k (35.93 -> 43.56). DSpark
decode is 22.6% faster at 32k (43.22 -> 52.99) and 46.3% faster at 64k (32.38 -> 47.36): a
DSpark round scores the block list of every verify row, so it runs the indexer several times per
committed step where plain runs it once. TTFT falls 4% at 32k and 8% at 64k: the
chunked prefill's indexer takes tiled from its first 64-row chunk. Every request of every row of
both arms hashed the same, in both modes, and plain and DSpark agree with each other
(`9a8aa4bc 17f9f998` at 32k, `e84068e3` at 64k).

Short context is unchanged by construction: under about 4.6k tokens the knee keeps the scalar
scorer. The lane's served DSpark identity gate passed on the same pair (tape 416,
`../kernel-ab-short/summary.txt`, `idx dspark-served`); no short served rows were scored for this
lane. The prefill itself stays slow at about
290 tokens per second whatever the scorer (memra #700).
