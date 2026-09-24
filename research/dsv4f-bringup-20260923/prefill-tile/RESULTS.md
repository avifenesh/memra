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
| wq_a-like 1024 x 4096 | 0.590 | 0.336 | 1.76x |
| wq_b 32768 x 1024 | 8.890 | 3.785 | 2.35x |
| wo 4096 x 8192 | 2.73 | 2.34 | 1.17-1.23x |
| 512 x 4096 | 0.45 | 0.17 | 2.56-2.69x |
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

## What this does not fix

Prefill runs about 360 tok/s at 8k. The exact program pins every output to the GEMV's reduction
order, which keeps it on CUDA cores. A tensor-core prefill (#472's CUTLASS path) changes the
numeric class, so it is an owner decision, not a rewrite this lane can qualify as exact.
