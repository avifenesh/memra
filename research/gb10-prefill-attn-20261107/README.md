# Qwen3.8-27B prime attention on GB10: the h2p ring (2026-11-07)

Device: two DGX Spark units (GB10, compute capability 12.1, 48 SMs, unified LPDDR5x), SM clock
2,177 MHz under load (clock cap active), CUDA 13.0 on the device, memra built `121a` on the
device. Model: the Qwen3.8-27B NVFP4+Q5_K GGUF with its masked MTP draft, served at 262,144
context, the same files, prompts and harness as `research/gb10-prefill-20261106/` (unit 1 and
unit 2 keep that lane's names). Base: main `c2e04719b6`, where the 131k profile put prime
attention at 50.8 s of 181.4 s (`fa_prefill_qw_h2`, 66 TFLOPS).

## Architecture (GGUF metadata + `crates/memra-gguf` qwen35 plan)

- 65 blocks: 64 trunk layers + 1 MTP block. `full_attention_interval` 4: layers 3, 7, ..., 63
  are full attention (16 layers), the other 48 are Gated DeltaNet (16 key / 48 value heads x 128
  state, conv 4, inner 6,144).
- Full attention: 24 query heads, 4 KV heads (GQA 6), head dim 256 for K and V, per-head
  q/k RMS norm, an output gate (the q projection is 5,120 -> 12,288 = query + gate), partial RoPE
  (64 of 256 dims, base 1e7, sections 11/11/10/0), scale 1/16.
- KV cache: q8_0 K (34 B per 32) and q5_1 V (24 B per 32): 1,088 + 768 = 1,856 B per token per
  layer, 31,552 B per token over the 16 + 1 attention layers.
- Prime: 4,096-row chunks. Per layer and chunk the new rows are quantized into the cache, the
  whole prefix `[0, T_kv)` is dequantized once into a BF16 workspace (`fa_dequant_kv_ws_bf16`,
  4 KB per token for K and V), and the attention kernel reads the workspace: f32 Q, BF16 K/V,
  bf16 `mma.sync m16n8k16` with f32 accumulation, 32-key tiles, online softmax with `exp2f`, P
  rounded to bf16 per element, O rescaled by alpha and accumulated from a fresh two-MMA PV chain
  per tile. On GB10 that kernel is h2: two query heads of one KV head in an 8-warp CTA (64 rows
  per head), double-buffered K/V planes (64 KB), one CTA per SM at 255 registers.

Work at the three served lengths (16 layers, 4,096-row chunks; `ceilings/derive.txt`):

| prompt tokens | causal attention | executed by the 32-key tiles | at 102 TFLOPS | BF16 workspace written | quantized KV read | K/V streamed L2 -> SM |
|---:|---:|---:|---:|---:|---:|---:|
| 32,784 | 0.211 PFLOP | +0.49% | 2.1 s | 11.8 GB | 5.4 GB | 1.66 TB |
| 131,051 | 3.377 PFLOP | +0.08% | 33.1 s | 141.7 GB | 64.2 GB | 26.4 TB |
| 259,951 | 13.29 PFLOP | +0.04% | 130.3 s | 558.2 GB | 252.9 GB | 103.8 TB |

Each CTA streams its KV head's prefix once (1 KB per key for K and V) and does 128 KFLOP per
key, so attention is compute-bound: about 128 FLOP per L2 byte, and the 48 concurrent CTAs of a
wave share one or two KV heads in L2 (ncu: 97.5% L2 hit rate, DRAM negligible). The DRAM-bound
part is the workspace pass, about 1.4 s at 131k.

## Ceiling (`probe/ceil_attn.cu`, `ceilings/ceil-attn.log`, unit 2)

| form, bf16 m16n8k16 -> f32, one CTA per SM | 4 warps/SM | 8 warps/SM |
|---|---:|---:|
| 8 independent accumulators | 84.8 | 101.4 TFLOPS |
| 4 dependent 16-deep chains per warp (the QK shape) | 98.9 | 106.8 |
| 2 chains per warp | 96.7 | 106.0 |
| 4 chains, B operand by swizzled `ldmatrix` before each MMA pair (QK as issued) | 89.6 | 102.1 |

Dependent chains and `ldmatrix`-fed operands do not lower the tensor pipe at 8 warps per SM: the
ceiling for the attention MMAs is about 102 TFLOPS. The kernel's own program, measured with the
real h2 body and its stages taken away (`probe/attn_cand.cuh` diagnostics, T = 4,096 rows at
depth 131,072):

| h2 body | TFLOPS |
|---|---:|
| as shipped (staging, two CTA barriers per tile) | 67.8 |
| no staging and no barriers (tile 0 reread) | 77.5 |
| no staging, no softmax | 96.4 |

So of the 34 TFLOPS gap to the ceiling about 19 is the softmax not overlapping the MMAs and
about 10 is staging and barriers. ncu on h2 (`ncu/r1-h2*`): tensor pipe 62.8% active, issue 37%,
9.56 instructions per MMA (MOV 23%, IADD3 9%), stalls `wait` 38%, `math` 21%, `short_scoreboard`
9%, two spilled values reloaded every tile. The two CTA barriers per tile keep all eight warps in
phase, so warps w and w + 4 (one per head, the pair that shares a scheduler) reach their softmax
together; in the softmax phase neither feeds the tensor pipe.

## Arms (probe, `probe/probe_attn.cu`)

The probe runs each kernel on the model's attention shape (24/4 heads x 256, causal, f32 Q,
BF16 workspace from a seeded generator) at T = 4,096 rows and four depths, median of 3 after a
warmup, and compares every candidate's O with h2's bit for bit. TFLOPS count the causal work.
Every arm below is exact (0 mismatching floats at every depth); `probe-runs/probe-r{1,2,3}.log`.

| arm | what changes | 4k | 16k | 64k | 128k |
|---|---|---:|---:|---:|---:|
| h2 (main) | | 52.7 | 63.3 | 67.1 | 67.8 |
| h2c | h2's body copied (control) | 52.7 | 63.4 | 67.3 | 67.9 |
| sb2 | one CTA barrier per tile | 52.8 | 63.6 | 67.4 | 68.2 |
| ad | precomputed ldmatrix offsets, K lanes ordered for consecutive B registers (generic addressing) | 51.8 | 65.4 | 68.7 | 69.3 |
| sbad | sb2 + ad | 54.0 | 65.9 | 69.1 | 69.5 |
| qh8 | sb2 + Q k-tiles 8..15 from smem (frees 32 registers) | 53.9 | 65.3 | 69.5 | 70.1 |
| qh12 | sb2 + Q k-tiles 12..15 from smem | 54.6 | 66.4 | 70.6 | 71.3 |
| qh8db | h2 barriers + qh8 | 51.2 | 63.9 | 67.8 | 68.3 |
| fa2s | all Q in smem, FA2's single K and V planes and stagger | 53.4 | 62.2 | 64.8 | 65.4 |
| pp | three-stage ring, per-stage full/empty mbarriers, head B one QK behind head A | 54.7 | 66.5 | 72.8 | 73.9 |
| pp2 | pp with a precomputed-address producer | 54.9 | 67.4 | 71.7 | 71.8 |
| ppad | pp2 + ad | 58.7 | 72.3 | 76.2 | 76.6 |
| ppadm | ppad + no mask compares on interior tiles | 59.9 | 72.2 | 75.8 | 76.6 |
| ppm | pp2 + no mask compares | 54.8 | 65.9 | 70.0 | 70.7 |
| ad, `.shared` | ad with `ldmatrix.shared` on 32-bit offsets (immediates fold, no 64-bit address pairs) | 54.0 | 66.0 | 70.0 | 70.8 |
| sbad, `.shared` | | 54.3 | 67.0 | 70.5 | 71.0 |
| **ppad, `.shared` (h2p)** | the ring + skew + `.shared` addressing | **62.7** | **79.2** | **84.7** | **85.4** |
| ppadm, `.shared` | + no interior mask compares | 59.6 | 75.9 | 81.1 | 82.6 |
| ppadz, `.shared` | + zero C operand on each chain's first MMA | 62.7 | 79.8 | 84.7 | 85.1 |

(Rows after "ad, `.shared`" are round 3; earlier rows use the generic-address `ldmatrix`.)

What moved it:

- The skew. Freed registers (qh8/qh12, fa2s) and fewer barriers (sb2) were worth 0-5%; letting
  the head-B warps run one QK behind head A (pp) was worth 9%, because one head's softmax then
  overlaps the other head's MMAs.
- The addressing. `ldmatrix` without a state space takes a generic address: ptxas built a 64-bit
  pair (`IADD` + `MOV R+1, RZ`) per load. With `.shared` and per-lane 32-bit offsets the swizzle
  folds into four offsets plus immediates (`LDSM [R9+0x2000]`), and executed instructions per MMA
  fall from h2's 9.56 to 8.24 (ncu; MOV from 23% to 8% of the mix); one spilled value is reloaded
  per tile instead of two. On its own the addressing gives h2 +4%; with the skew, +15% on top of pp.
- The mask-free interior tiles and the explicit zero C operand did nothing or worse (ptxas
  already uses `RZ` for the zero accumulator); not kept.

ncu on h2p at depth 65,536 (`ncu/r3-ppad*`): tensor pipe 78.7% active (h2 62.8%), 8.24
instructions per MMA, 78.2 ms against h2's 98.4 ms under ncu's clock.

Ragged and long shapes (`probe-runs/edge.log`): T in {128, 130, 192, 1,000, 1,024, 4,033, 4,096}
with T_kv from T to 262,144, tails not multiples of 32 or 64: h2p equals h2 bit for bit in all
19 cells; h2p is faster in every cell except the launch-bound T = 128/130 ones (+2 us), and at
depth 262,144 it runs 305.7 ms against 384.5 ms (85.7 against 68.1 TFLOPS).

Laptop RTX 5090 (82 SMs, the GeForce half-rate f32-accumulate MMA), same probe built `120a`,
T = 4,096 rows (`probe-runs/probe-rtx5090-laptop.log`): at depth 131,072 db 26.7, h2 33.5 and
h2p 37.5 TFLOPS, all bit-identical. Discrete cards are not changed by this lane: they keep db,
and a default there needs its own served A/B on a 5090 and a PRO 6000.

## The landed kernel (`fa_prefill_qw_h2p`)

The h2 row program on a three-stage ring of K/V tiles (3 x 32 KB) in an 8-warp CTA:

- each stage has a full mbarrier (256 arrivals, one `cp.async.mbarrier.arrive.noinc` per thread
  after its copies) and an empty mbarrier (8 arrivals, one per warp after its PV); a thread
  refills a stage with tile t + 2 after the empty barrier says every warp released tile t - 1,
  so the copy has a whole tile to land and warps can drift by up to a tile;
- head B's warps wait on a named barrier until head A's warps finish their first QK, so the two
  warps of each scheduler run out of phase from the first tile on;
- the producer copies rows warp + 8j, chunk lane, whose swizzle is `lane ^ warp` for all four
  rows, with zero-filling `cp.async` (`src-size` 0) for rows past the tail;
- K and V are read by `ldmatrix.shared` on per-lane offsets; K's lanes are ordered so each B
  operand lands in two consecutive registers.

Every row sees h2's tiles in order, the same QK chains, the same softmax expressions, the same
bf16 P rounding, the same O rescale and the same PV chains, and the same bytes reach the same MMA
operand slots: O is bit-identical to h2 (and to db and t3, which h2 already matched).

`MEMRA_PRIME_KV_H2P` resolves like `MEMRA_PRIME_KV_H2`: `1`/`0` decide, unset follows the device
(on for a unified-memory 120a-family device). It acts only inside the h2 route, so
`MEMRA_PRIME_KV_H2=0` (db) and `MEMRA_PRIME_KV_T3=1` keep their meaning. 98,368 B of dynamic shared
memory, one CTA per SM, 255 registers, 8 B stack (the `_prime_table` twin 24 B).

## Exactness

First battery, unit 2, on `b6604e1b90` (the h2p commit on `c2e04719b6`; `gates-unit2-b6604e1b90/`):

- kernel-check ALL GREEN (115 cells, 5 skipped) in the device default (h2p), `MEMRA_PRIME_KV_H2P=0`
  (h2) and `MEMRA_PRIME_KV_H2=0` (db); the ARC B cells (fa_prefill_view_ws against the inline-dequant
  kernel, now including 4,096 x 36,864) are bitdiff=0 in all three.
- argmax-margin gate (board-2048): flips=0 bad=0, PASS. run-spec K=1..8 self-consistency PASS.
- `run-gen` `MEMRA_PP_ONLY` + `MEMRA_PP_LOGITS` at 16,219 and 33,890 tokens: base, h2p and h2 dumps
  byte-identical, and the three timings equal. That instrument does not reach the prime: the GGUF
  arm of `MEMRA_PP_ONLY` runs the monolithic `forward_last`, not `prime_cache` (the FLAGS row said
  otherwise). `MEMRA_PP_PRIME=1` (this lane) runs the warmups, the timed reps and the dump through
  `prime_cache`, so the final battery byte-compares the prime path itself.
- Probe: every candidate equals h2 bit for bit at every depth and at 19 ragged shapes up to depth
  262,144.

## Served

PENDING: the landing A/B runs h2p rebased on main after memra#1077 (the F8F4 GEMM v2), against that
main, fresh boot per cell, B H H B B H on one unit and a confirmation on the other.

- `served-unit2-r0/`: the first campaign (base `c2e04719b6` against `b6604e1b90`) was stopped after
  its first cell to free the unit for #1077's landing evidence. Its one base cell: 36.233 / 182.199
  / 460.018 s at 32,784 / 131,051 / 259,951 tokens, needles found, the round-2 answers.
- `served-unit2-contaminated/`: an earlier start of that campaign ran beside the gate battery
  (the queue's `waitfor --grep RE --file LOG` returned at once; the blocking form is
  `waitfor --grep RE LOG`), so its cells shared the GPU with kernel-check and the argmax probe
  (858.7 s at 260k). Not a measurement; kept as a record.

## Levers considered and not built

- Split-KV or a different chunk-by-prefix schedule: splitting a row's keys changes the combine
  (two partial softmax states merged by rescaling), so it is a different float program; the chunk
  size does not change attention's numbers but row 15 of the 2026-11-06 ledger rejected 8,192 on
  peer latency and attention's per-chunk efficiency is already flat past 16k depth.
- A different softmax or rescale order: the probe kept h2's order; the mask-free variant (the one
  reorder that is a provable no-op) measured slower.
- SGLang's 128k win (FA2 on BF16 K/V gathered from its FP8 cache instead of FA2's in-kernel FP8
  path) is the design memra's prime already has: the q8_0/q5_1 prefix is dequantized once per
  layer and chunk into a BF16 workspace and the MMA kernel reads BF16. memra's exact program is
  defined on that workspace, so there is no further equivalent to port.
- Fusing the KV write or cast into the attention prologue: the workspace pass costs about 1.4 s
  at 131k and 5 s at 260k (the prefix is rewritten every chunk). Dequantizing in the attention
  producer instead repeats it in every CTA (768 per layer-chunk); a persistent per-layer
  workspace written once per row would save the rewrite but holds 16 x 4 KB per token (17 GB at
  260k). Not built; about 0.8% / 1.2% of TTFT.
