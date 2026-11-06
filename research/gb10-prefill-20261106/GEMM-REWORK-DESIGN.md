# F8F4 prefill GEMM on GB10: structural rework (design, not built)

Status: proposal for the owner's go. Nothing here is implemented. Numbers are measured on one GB10
(SM 2,177 MHz under the clock cap) unless marked as an estimate.

> **Built 2026-11-06** (measured as `b6d10d41c7`; `LEVER-LEDGER.md` rows 16-26,
> `docs/decisions/GB10-F8F4-GEMM-V2.md`). Kept as designed: the per-call pooled image, the
> pipelined mainloop without ALU staging, the grouped raster, the numeric program. Changed by
> measurement: the 16 warps per SM come from one 16-warp 256 x 128 CTA (the two 8-warp 128 x 128
> CTAs reached 128 registers and 36 KB but ran 4-9% slower on the FFN shapes), the activation needed its own layout
> (reading the 144 B `block_e4m3_mmq` blocks cost 45% excess L2 sectors), and smem rows are
> XOR-swizzled rather than padded. The design text below is unchanged.

## Where it stands

At 131,051 tokens on the round-2 stack (`0cb9642b98`, `round2-profile-128k/`), the trunk GEMMs are
80.3 s of 181.4 s of GPU time: 6,800 launches of `mul_mat_q_nvfp4_f8f4` (8,192-row chunks),
6.376 PFLOP, 79 TFLOPS. The e4m3 activation quantize adds 9.3 s (the fused silu quantize carries
the down projection's activation read; `LEVER-LEDGER.md` rows 11-12).

ncu at the gate shape (4.1k tokens x 17,408 x 5,120, 9.13 ms, 80 TFLOPS): one CTA of 8 warps per
SM (187-200 registers, 94.7 KB smem), tensor pipe 36.7% active, issue slots 41% busy, 17.3 warp
instructions per MMA, no eligible warp on 61% of cycles. Stalls per issued instruction: math-pipe
throttle 1.00, fixed-latency wait 0.94, MIO throttle 0.57, barrier 0.27. Three loader micro-edits
were exact but spilled (254-255 registers) or were flat (`LEVER-LEDGER.md` rows 3-6): the tile
has no register or smem headroom left.

## The program that must not change

For output row i (weight) and column j (token), with k32 blocks b in ascending order:

    s += (dA[i,b] * dB[j,b]) * C_b,   C_b = mma.sync m16n8k32 kind::mxf8f6f4.block_scale (ue8m0 = 1)
                                      over A = e4m3(weight), B = e4m3(activation), fresh accumulator

where the e4m3 weight bytes and dA come from the existing ratio fold of the NVFP4 GGUF block
(`dequant_tiles_nvfp4_f8f4`: s32 = max of the pair's UE4M3 scales, values cvt.rn.satfinite to
e4m3 after multiplying by s_sub / s32), and B, dB from the existing e4m3 activation quantizer.
Any tile shape, warp layout, CTA raster, staging schedule or pre-pass that feeds the same bytes to
the same instruction and keeps this per-element sequence is bit-identical. The four rewrites this
lane measured (rows 3-6) confirm it: 24 run-gen processes produced one logits hash.

## Where the cycles go now (per 128 x 128 x 256 step, per SM sub-partition, 2 warps)

| work | count | cost |
|---|---|---|
| MMA m16n8k32 block-scale | 256 | 4,096 tensor cycles at 16.0 cycles each |
| epilogue: 4 FMUL (dA*dB) + 4 FFMA per MMA | 2,048 | one FP32 issue per cycle |
| weight dequant (FP4 to e4m3 ratio fold, LUT build, gather, smem stores) | ~1,000 issue cycles | serialized with the MMA phase by `__syncthreads`; nothing overlaps it at one CTA per SM |
| operand loads (B 2 LDS, dB 1 LDS per MMA; A ldmatrix) and addressing | ~3-4 per MMA | MIO throttle 0.57 |

That is ~17 instructions per 16 tensor cycles: issue-bound in the MMA phase and idle tensor pipe in
the dequant phase, with 1.9 warps per scheduler to hide the MMA-to-FFMA latency.

## Design

1. **Dequantize once.** A pre-pass converts each NVFP4 weight into the tile's smem image in global
   memory (per 256 values: 64 ints of e4m3 + 8 f32 s32 + 4 pad = 304 B, 1.19 B per weight) with
   the existing device functions, so the bytes are those the current loader writes to smem.
   Two placements:
   - **per GEMM call, pooled scratch** (largest weight, gate/up: 106 MB): every chunk reads all
     13.68 GB of trunk NVFP4 weights and writes 28.89 GB of images, 0.198 s per chunk at 215 GB/s,
     so 6.3 s at 131k with 4,096-token chunks, 3.2 s at 8,192, 1.6 s at 16,384;
   - **resident images** built at load: 0 s per chunk, +28.9 GB of device memory (the 256K
     MemAvailable floor would fall from 66.8 GiB to about 38 GiB).
2. **Mainloop without ALU staging.** Both operands arrive as e4m3 bytes plus per-32 scales, so the
   loop is cp.async (or TMA) of the A image and the B tile into a 3-4 stage ring, ldmatrix for A
   and for B (the B fragment for m16n8k32 is exactly an ldmatrix.x2 of two 8 x 16-byte slices),
   the MMA and the fixed epilogue. One `__syncthreads` per stage, no xr staging plane, no dequant
   registers. smem per stage at K = 128: A 18 KB + B 18 KB; three stages 108 KB is over the 99 KB
   opt-in limit, so K = 64 per stage with four stages (72 KB) or K = 128 with two (72 KB).
   Instruction budget: 1 MMA + 8 FP32 + ~1.5 ldmatrix + ~0.5 addressing = ~11 per MMA, inside the
   16-cycle tensor interval, which is what the per-k32 program allows at all (ceiling below).
   The register budget stays ~180 (accumulators 64 + A/B fragments + scales), one 8-warp CTA per
   SM; the dequant registers that pushed the micro-edits over the cap no longer exist.
3. **L2-ordered raster.** Group G weight tiles with all token tiles of the chunk: weights stream
   from DRAM about once per GEMM and the activation tile set is re-read 136 / G times per gate GEMM
   from L2 (GB10 L2: 24 MiB, measured by `ceilings/ceil_probe2.txt`). With images at 1.19 B per weight, a naive weight-major raster would
   re-read 106 MB per token tile from DRAM (32 x 106 MB = 3.4 GB per gate GEMM at 4,096 tokens),
   so the raster is part of the design, not a tuning knob.

## Expected seconds at 131k, from measured ceilings

`ceilings/ceil_probe2.txt` (round 2) adds the tile's exact epilogue form to the probe: per MMA,
four distinct dA*dB products and four FFMAs, with scales that change every k32 step so nothing can
be hoisted (the first probe's "f8resc" row hoisted the product and overstated the ceiling at 175).

| warps per SM | raw block-scale MMA | tile epilogue form | GEMM at that ceiling (6.376 PFLOP) | at 85% of it |
|---:|---:|---:|---:|---:|
| 8 (today's tile: 1 CTA x 8 warps) | 201.3 TFLOPS | 113.3 TFLOPS | 56.3 s | 66.2 s |
| 16 | 202.8 | 132.0 | 48.3 s | 56.8 s |
| 32 | 204.6 | 138.1 | 46.2 s | 54.3 s |

Today the tile runs 79 TFLOPS, 70% of the 8-warp ceiling. Removing the dequant phase and the
operand-load overhead at the same occupancy recovers at most 80.3 - 66.2 = 14 s, minus the image
pre-pass. The epilogue is latency-bound, so the occupancy is what moves the ceiling: the design
only pays well at 16 warps per SM, which needs <= 128 registers per thread and <= 50 KB smem per
CTA (two 8-warp CTAs). Pre-converted operands make that budget possible: no dequant registers, no
xr staging plane, K = 64 per stage at 18 KB, two stages = 37 KB per CTA.

| design | GEMM | pre-pass (per-call images, 8,192-token chunks) | saving vs 80.3 s |
|---|---:|---:|---:|
| 8 warps per SM, 85% of ceiling | 66.2 s | +3.2 s | ~11 s |
| 16 warps per SM, 85% of ceiling | 56.8 s | +3.2 s | ~20 s |
| 16 warps per SM, at ceiling | 48.3 s | +3.2 s | ~29 s |
| resident images instead of the pre-pass | | 0 s, +28.9 GB device memory | +3.2 s more |

Expected: about 20 s at 131k, 181.1 -> ~161 s TTFT (-11%), with the range 11-29 s set by the
occupancy the kernel reaches. The epilogue is fixed by the numeric program, so no pipeline beats
the tile-epilogue row. The FP4 x FP4 GEMM SGLang runs (15.8 s for the MLP at 131k) is a different
program and out of scope.

## What it takes

- A v2 F8F4 GEMM next to the current tile (`cu/mmq_nvfp4_w4a8.cu` or its own TU): the image
  pre-pass kernel, the pipelined mainloop, the grouped raster, the GEMM-only FFI entry shape that
  round 2 already added (`memra_mmq_nvfp4_f8f4_preq` takes a pre-quantized activation).
- Rust: a pooled image scratch per call (or a resident image per weight at load), the route keyed
  on the device like the F8F4 default, an env rollback seam.
- Gates: kernel-check bit-identity of v2 against the current F8F4 tile at every trunk shape
  (in 5,120 / 6,144 / 17,408, out 1,024 to 17,408) for T in {16, 130, 1,024, 4,096}; the run-gen
  logits byte compare; served fingerprints; argmax-margin and run-spec K=1..8.
- Tuning: stage count and K per stage, the raster group G, and the register budget that admits
  two 8-warp CTAs per SM (the deciding knob; ncu "Block Limit Registers"), measured by the
  run-gen GEMM bench and ncu.
- Size: about 1-2 agent-days including tuning. Risk: if the kernel cannot hold 128 registers per
  thread with a 128 x 128 tile, it stays at 8 warps per SM and the saving is ~11 s; a 128 x 64 tile
  halves the accumulators (32 per thread) at the cost of reading each weight image twice as often
  from L2.
