# GB10 prime attention: the h2p mbarrier ring by default (2026-11-07)

**Status:** decided on one device class (unified-memory 120a family): served A/B on DGX Spark unit
1, probe and exactness battery on both units. Discrete cards keep db: the laptop RTX 5090 probe
(db 26.7, h2 33.5, h2p 37.5 TFLOPS at depth 131,072, bit-identical) is kernel-only; a discrete
default needs its own served A/B on a 5090 and a PRO 6000.

## Question

On GB10 the quantized-KV prime runs `fa_prefill_qw_h2` (two query heads of one KV head per 8-warp
CTA, double-buffered K/V planes). At a 131,051-token served prime it took 50.8 s of 181.4 s at 66
TFLOPS, against a bf16 `mma.sync` ceiling of about 102 TFLOPS measured on the device for the forms
the kernel issues. Which exact change closes the gap?

## Decision

`MEMRA_PRIME_KV_H2P` resolves like `MEMRA_PRIME_KV_H2`: `1`/`0` decide on any device, unset is on
for a unified-memory 120a-family device. Inside the h2 route it selects `fa_prefill_qw_h2p`: h2's
row program on a three-stage K/V ring handed over by per-stage full/empty mbarriers, with the
second head's warps one QK behind the first's, and `ldmatrix.shared` on precomputed per-lane
offsets. Every row keeps h2's tiles, chains, softmax expressions and roundings, so the output is
bit-identical to h2, db and t3. `MEMRA_PRIME_KV_H2P=0` restores h2.

## Evidence

Device probe, the shipped `fa_prefill_qw_h2p` against h2 on unit 1
(`research/gb10-prefill-attn-20261107/probe-runs/probe-r4-unit1.log`, T = 4,096 rows, median of 3
after a warmup, the first of two passes; one run at 262,144 from `edge-h2p-unit1.log`; O compared
with h2's bit for bit):

| depth | h2 | h2p |
|---:|---:|---:|
| 4,096 | 3.907 ms (52.8 TFLOPS) | 3.296 ms (62.6) |
| 16,384 | 22.668 ms (63.7) | 18.420 ms (78.4) |
| 65,536 | 94.979 ms (67.3) | 75.627 ms (84.5) |
| 131,072 | 191.235 ms (67.9) | 152.726 ms (85.0) |
| 262,144 | 384.410 ms (68.1) | 305.673 ms (85.7) |

0 mismatching floats at every depth and at 21 ragged shapes (fresh primes, partial final CTAs,
the served 131k final chunk 4,075 x 131,051). The probe copy measured the same on unit 2
(85.4 TFLOPS at 131,072). Served A/B: see the lane README.

Served cold TTFT, Qwen3.8-27B NVFP4+Q5_K GGUF with the masked MTP draft at 262,144 context, base
main `6f6d816757` against the lane head `6f59cac641` (the same code as the landing commit), cells
B H H B B H on unit 1, one fresh boot per cell
(`research/gb10-prefill-attn-20261107/served-unit1-final/`):

| prompt tokens | main (h2) | h2p | change |
|---:|---|---|---:|
| 32,784 | 32.638, 32.574, 32.556 s | 32.060, 32.056, 31.974 s | -1.59% (medians) |
| 131,051 | 167.899, 168.249, 167.839 s | 158.237, 158.136, 158.400 s | -5.75% |
| 259,951 | 433.306, 432.708, 432.590 s | 394.011, 393.874, 393.564 s | -8.97% |

Every h2p boot is faster than every main boot at every length. The answer text is identical in all
six boots (and to the earlier rounds' answers), needles found, no request error, MemAvailable floor
66.56-66.67 GiB in every cell. Exactness battery on the landing code: kernel-check ALL GREEN in h2p,
h2 and db modes with every ARC B cell bitdiff=0, argmax-margin flips=0 bad=0, run-spec K=1..8 PASS,
`tests/prime_kv_h2_gpu.rs` passing in both modes, and h2p/h2 prime-path logits byte-identical at
16,219 and 33,890 tokens.

## Rejected

Each measured on the device by `research/gb10-prefill-attn-20261107/probe/` (T = 4,096 rows,
depth 131,072, exact against h2 unless noted):

- One CTA barrier per tile (sb2): +0.6%. Not what limits the kernel.
- Q k-tiles in shared memory to free registers (qh8 +3.5%, qh12 +5.2% on h2's barriers), and FA2's
  single K and V planes with all of Q in shared memory (fa2s, -3.6%). Not measured on top of the
  ring: three K/V stages take 96 KB, and Q k-tiles 12..15 add 16 KB, past the 99 KB per-block
  limit.
- Skipping the mask compares on interior tiles (-3.2% on top of h2p) and passing the zero C
  operand explicitly (flat; ptxas already uses `RZ`).
- A later start for head B (after head A's first softmax: flat; after its first tile: -8%) and each
  warp issuing QK(i+1) before softmax(i) (-34%: tile i+1's copy loses a third of a tile of slack
  while head B lags, and the extra score tile spills).
- Split-KV: a different float program (partial softmax states merged by rescaling).
- A persistent BF16 workspace (no per-chunk rewrite of the dequantized prefix): about 1.4 s at
  131k for 17 GB of workspace at 260k; not built.
