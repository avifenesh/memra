# DSv4 prefill tiles: the FP8 GEMV's and the compressor dots' own arithmetic over token x output tiles (#700, #472)

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`,
kernel rows on one RTX PRO 6000 Blackwell Server Edition, served rows on 2x RTX PRO 6000 Server
Edition, 2026-09-24.

## What changed

At prefill widths (`m > DSV4_TMAX`) the dense FP8 GEMV family ran one CTA per output row over
32-row chunks, and every CTA re-read all of x. That made it activation-stream bound.
`dsv4_gemm_fp8_tile_kernel<8, 8>` gives each 128-thread CTA 8 token rows by 8 output rows:
- Each weight element is decoded once and used for 8 tokens.
- Each activation is used for 8 outputs.
- Every output keeps the GEMV's arithmetic: the same 128 k-slices, 8 elements ascending, the same
  `e4m3 * scale` product and the same halving tree (shared memory for offsets 64 and 32,
  shuffles for 16 to 1). The bits cannot move.

`dsv4_dots_f32acc_tile_kernel<8, 8>` does the same for the compressor dots (f32 activations,
BF16 or f32 weights). Decode never takes either tile (t=1).

## Correctness (one card, `raw/run2.sh`)

- `DSV4_DENSE_TILE EXACT cases=106 shapes=9 widths=[33, 48, 64, 100, 256, 512] red_arm=1`
- `DSV4_DOTS_TILE EXACT cases=40`, over BF16 and f32 weights at the compressor latents

Served text is identical on every request of all six rows (below).

## Kernel timing (m=512, device time per call, tile against the loop)

| projection (n x k) | loop ms | tile ms | speedup |
|---|---|---|---|
| wq_a 1024 x 4096 | 0.590 | 0.336 | 1.76x |
| wq_b 32768 x 1024 | 8.890 | 3.785 | 2.35x |
| wo_b 4096 x 8192 | 2.73 | 2.34 | 1.17-1.23x |
| wkv 512 x 4096 | 0.45 | 0.17 | 2.56-2.69x |
| shared expert 2048 x 4096 | 1.050 | 0.655 | 1.60x |
| compressor dots, bf16, 1024 / 512 / 256 x 4096 | 0.755 / 0.591 / 0.600 | 0.338 / 0.178 / 0.100 | 2.2x / 3.3x / 6.0x |

Tile-shape sweep, deleted arms: 16x4, 16x8 and 32x4 lose to 8x8 on 4 of 5 dense shapes. 16x8 wins
only wq_b (2.47x against 2.35x), and one 5% kernel gain did not justify a per-shape dispatch
rule. Sweep lines are in the lane history (`5feffa64f`).

## Served TTFT (q-v3l, lane `7e15b241d` against main `a8d0121d9`, one boot per row, L B B L L B, N=3)

`raw/ttft/`, cells `raw/cells-ttft.txt`, serial route (`MEMRA_DSV4_SESSIONS=1`), 64 generated
tokens:

| cell | lane TTFT p50 | base TTFT p50 | delta |
|---|---|---|---|
| greedy 8k prompt | 22.91 s | 28.53 s | **-19.7%** |
| greedy 32k prompt | 94.29 s | 117.45 s | **-19.7%** |

Decode in the same rows moves under 1.1%, within row noise (2 and 1 requests of 64 tokens). The
decode step never reaches the tile. Thermal: power median 190..230 W, SM clock 2280..2430 MHz,
max temperature 56 C.

## Register diet, 2026-09-27 (TP/EP served default)

Lane `lane/dsv4-prefill-tile-occ-20260927`, rebased on main `359e850d0`. Raw:
`raw/occ-20260927/` (queue scripts, summaries, per-run kernel sums, variant diffs, served rows).

Nsight Compute on the 8x8 FP8 tile (`raw/occ-20260927/ncu-tile-v6f/`): 214 registers, two
blocks per SM (16.67% theoretical occupancy, 14.31% achieved). The tile holds TN*8 decoded
weights while it adds into TT*TN accumulators. Four compile-time knobs, each bit-inert:
- TN, outputs per block;
- SUB, elements per pass inside each 8-element chunk (every accumulator still takes its chunk's
  elements in ascending order);
- DEC, the E4M3 decode: the shared-memory table, or `cvt.rn.f16x2.e4m3x2` after clearing the
  two NaN codes to +0, which is `dsv4_e4m3`'s value for them and exact for every other code;
- MINB, the blocks per SM the launch bounds ask for.

**Kernel sweep.** One card, nsys over `dsv4_gemm_tile_gpu` (every output bit-checked each run),
FP8 tile kernel time summed over its 318 launches, box-local builds, each order mirrored:

| build (TT, TN, SUB, DEC, MINB) | tile ms |
|---|---|
| B (8, 8, 8, table, 1), the merged tile | 411.2 .. 421.2 |
| C (8, 8, 4, cvt, 3) | 316.9 .. 321.2 |
| D (8, 8, 2, cvt, 4) | 312.2 .. 314.5 |
| H (8, 8, 4, table, 3) | 331.9 .. 332.1 |
| G (16, 4, 4, cvt, 3) | 397.6 .. 399.2 |
| **E (8, 4, 8, cvt, 4)** | **241.5 .. 245.2** |
| L (8, 4, 8, table, 4) | 290.9 .. 291.8 |
| I (8, 4, 4, cvt, 5) | 281.1 .. 283.6 |
| J (8, 2, 8, cvt, 6) | 237.7 .. 239.0 |
| K (4, 4, 8, cvt, 6) | 252.7 .. 252.8 |
| M (16, 2, 8, cvt, 4) | 339.3 .. 340.1 |

E is 128 registers at four blocks per SM. The cvt decode is worth 17% on its own (E against L).
Served, E and J tie (below), so E is committed: it keeps twice J's activation reuse per block.
Against main's binary directly (`raw/occ-20260927/tile-vs-main-v6o/`, order M T T M): 422.3 ..
428.7 ms against 243.1 .. 243.2 ms, **-43%**.

The dots tile took the same sweep (`dots-tile-v6j/`): (8, 8, 1) 56.0 .. 57.7 ms, (8, 4, 4)
59.4 .. 61.5, (8, 2, 6) 65.3 .. 66.4, (4, 4, 6) 61.9 .. 62.2. The committed (8, 8, 1) equals
main's dots tile: 57.1 .. 57.3 ms against 57.2 .. 58.4.

**Served TTFT, three builds** (`tile-ttft-v6i/`, first SE pair, B E J J E B B E J, one boot per
row, cells `raw/occ-20260927/cells-ttft.txt`, c1 greedy, 64 generated tokens):

| build | 8k TTFT p50 | 32k TTFT p50 |
|---|---|---|
| B | 17,485 .. 17,496 ms | 73,159 .. 73,178 ms |
| E | 14,336 .. 14,347 ms | 60,867 .. 60,880 ms |
| J | 14,343 .. 14,355 ms | 60,992 .. 61,000 ms |

**Correctness on the rebased lane** (`tile-rebased-v6k/`):
- The tile bit test: `DSV4_DENSE_TILE EXACT cases=106 shapes=9 widths=[33, 48, 64, 100, 256,
  512] red_arm=1` and `DSV4_DOTS_TILE EXACT cases=40` (`tile-test-ignored/`).
- The long gate's `PROGRAM_SHA256` is `fbce1a0492d69635` over 304 steps from a 400-token
  prefix, which the tile prefills.
- `dsv4_rows_gate` TP/EP, the KV split gate and the DSpark TP/EP gate pass. The DSpark proposal
  shas equal main's on that pair.

**Served A/B against main** (same pair, M T T M M T, N=3, one boot per row):

| cell | main | lane | delta |
|---|---|---|---|
| greedy 8k TTFT p50 | 17,678 .. 17,705 ms | 14,341 .. 14,368 ms | **-18.9%** |
| greedy 32k TTFT p50 | 73,927 .. 73,935 ms | 60,965 .. 60,971 ms | **-17.5%** |
| decode p50 at 8k | 74.56 .. 74.94 tok/s | 74.79 .. 75.02 tok/s | flat |

Every request's text is identical in all six rows. Thermal: median power 260 .. 263 W (main)
and 294 .. 299 W (lane) while the cards work, SM clock median 2392 .. 2422 MHz, max 54 C.

## What this does not fix

Prefill runs about 560 tok/s at 8k and 530 at 32k on the TP/EP default after the register diet. The exact program pins every output to the GEMV's reduction
order, which keeps it on CUDA cores. A tensor-core prefill (#472's CUTLASS path) changes the
numeric class, so it is an owner decision, not a rewrite this lane can qualify as exact.
