# GB10 prime attention: the h2p mbarrier ring by default (2026-11-07)

**Status:** pending the served A/B on GB10 (unified-memory 120a family). Measured so far on one
DGX Spark unit (probe, ncu, exactness battery). Discrete cards keep db: the laptop RTX 5090 probe
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

Device probe on one GB10 (`research/gb10-prefill-attn-20261107/`, T = 4,096 rows, median of 3
after a warmup, a single run at 262,144; O compared with h2's bit for bit):

| depth | h2 | h2p |
|---:|---:|---:|
| 4,096 | 3.966 ms (52.0 TFLOPS) | 3.289 ms (62.7) |
| 16,384 | 22.669 ms (63.7) | 18.224 ms (79.2) |
| 65,536 | 94.687 ms (67.5) | 75.419 ms (84.7) |
| 131,072 | 191.313 ms (67.9) | 152.090 ms (85.4) |
| 262,144 | 384.505 ms (68.1) | 305.657 ms (85.7) |

0 mismatching floats at every depth. Served A/B: see the lane README.

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
- Split-KV: a different float program (partial softmax states merged by rescaling).
- A persistent BF16 workspace (no per-chunk rewrite of the dequantized prefix): about 1.4 s at
  131k for 17 GB of workspace at 260k; not built.
