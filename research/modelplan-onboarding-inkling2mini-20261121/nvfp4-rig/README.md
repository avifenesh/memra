# nvfp4-rig: the default numeric class of Inkling 2 Mini on the rig (lane M)

Coordinator decision 8 (2026-11-21), from lane F's per-layer KL receipts
(`~/.cache/inkling2mini/klsens/results/tables-screen.md`): the class `nvfp4-rig` is

- the checkpoint's declared activation-quant experts (NVFP4 W4A4 on layers 0-19);
- the MXFP8 expert banks of layers 20-23 converted to NVFP4, running the same NVFP4 W4A4 program;
- an NVFP4 output projection (`unembed`) and NVFP4 attention projections (q, k, v, o, output gate,
  relative-bias projection) on every physical layer, the loop core 6-17 included (owner ruling
  2026-11-21 ~14:25: "convert the loop attention too, measure it"); these are weight-only: f32
  inputs, f32 accumulation;
- unit-scale e4m3 K and V on all 48 cache slots (the f32 value after the key clip and the value norm,
  rounded to nearest, ties to even, saturating at 448), as lane F measured it (`klh.fp8_kv` on the
  output of `k_proj` / `v_proj`, which is what the cache stores).

Decision 8's first set kept the loop core's attention in BF16; it stays as the measurement class
`nvfp4-rig-outside-loop` (the same overlay, the loop-core attention planes not read), so both sets
are measured side by side. The checkpoint as stored stays the named class `shipped`. A request runs
one class.

## The mint

`ink2-mint mint CKPT OUT --artifact HF_ID@REV` (`crates/memra-gguf/src/bin/ink2_mint.rs`, library
`memra_gguf::model_packs::inkling2::nvfp4_rig`): CPU only, deterministic (bytes independent of the
thread count), no external engine. The quantizer is lane F's `klh.nvfp4_weight_quant` ported
operation for operation: per-16 e4m3 block scales chosen from 16 MSE candidates, a per-tensor (per
expert for a bank) scale `t = amax / 2688`. Expert banks are stored in the checkpoint's own NVFP4
layout (`_weight`, `_weight_scale`, `_weight_amax`, `_weight_outer_scale_denominator` = 2688), dense
tensors in the modelopt layout (`.weight`, `.weight_scale`, `.weight_scale_2` = t).

Overlay of the pinned checkpoint (`~/models/inkling2-mini.nvfp4-rig/`, read-only):
153 tensors, `overlay.safetensors` 2,916,121,412 bytes, sha256
`cb47da1a33e683a6664b48e1b937b818a696ee761d15723d1f39f7241d1acc04`; the manifest
(`overlay-manifest.json`, a copy) binds the source file's sha256 (`65e60f51...`, the pinned
weights hash), size and header hash, the `hf_quant_config.json` hash, and the sha256 of every source
tensor it replaces and of every overlay tensor. Mint: `mint.log` (339 s, 12 threads). Weight
relative MSE 6.7e-3 on every tensor (`manifest.json` `rel_mse`), lane F's own figure for NVFP4.

`ink2-mint verify` (`mint-verify.log`): the loader's binding check, both full-file hashes, and a
re-mint of all 153 tensors byte-identical.

Proof that the port is lane F's quantizer:

- `golden_quant.py` runs `klh.nvfp4_weight_quant` unmodified (KLH_DEV=cpu) on synthetic edge cases
  (zeros, signed zeros, e2m1 tie points, 1e-30 values, mixed row scales, per-expert 3-D) and on slices
  of the shipped checkpoint (BF16 attention and unembed rows, MXFP8 layer-20 experts dequantized
  exactly). The repository fixture `crates/memra-gguf/src/model_packs/inkling2/fixtures/nvfp4-rig-golden.bin`
  is its short run; `quantizer_matches_lane_f_goldens`: 0 code bytes, 0 scale bytes, 0 tensor scales
  and 0 dequantized values differ on every case. The long run (2.6 M elements) also matched with 0
  differences (`INK2_RIG_GOLDEN=... cargo test -p memra-gguf nvfp4_rig -- --include-ignored`).
- `golden_full.py`: the same quantizer on CUDA over all 153 full tensors, hashed and compared with the
  manifest (`gpu-run/golden-full-cuda.json`).

## Loader and contract

The plan carries the class (`Inkling2Numerics::weight_class`, `minted`, `kv` per slot,
`ExpertNumericsPlan::minted`); `nvfp4_rig_plan` builds it, `plan_with_expert_activation` stays the
shipped one. `expected_census_for(config, plan)` moves the minted tensors to their NVFP4 layout, so
the tensor contract of an `nvfp4-rig` plan binds only the checkpoint merged with its overlay, and the
shipped plan refuses the merged census (`nvfp4_rig_binds_only_the_checkpoint_merged_with_its_overlay`).
`bind_overlay` refuses a manifest of another class, format or quantizer, a tensor list other than the
class's set, another base file size or header, another quantization config, and any source or
overlay tensor whose sha256 differs (`overlay_binds_its_checkpoint_and_refuses_every_mismatch`:
one flipped overlay byte, one flipped byte of a replaced source tensor, another config hash, a short
manifest). On the real checkpoint: `ink2-gpu CKPT --contract-check` binds 1170 tensors under
`nvfp4-rig` and under `nvfp4-rig-outside-loop`, every red arm refused.

## Reference executor and tiny gate

`memra_reference::inkling2`: minted dense tensors read as `fl32(code * fl32(s * t))` (the values lane
F's arms ran), minted banks through the declared NVFP4 program, e4m3 K/V per the plan's slot format.
Tiny variant `nvfp4-rig`: the tiny checkpoint with its overlay minted at run time, pinned to an
independent oracle (`tiny_rig_oracle.py`: lane E's activation-quant simulation on an f32 stream, the
overlay dequantized in numpy, K/V rounded to e4m3 by a forward hook). Reference vs oracle: max abs
2.2e-6 over 12,000 logits (the declared variant's band, 1e-5); `memra model verify tiny --against
inkling2` passes with the variant. Red arms: f32 KV, and the plan over the base checkpoint without
its overlay (refused).

## GPU (sm_120a)

- Dense NVFP4 projections: the route's GEMM / GEMV with the NVFP4 format and one "expert" whose
  factor is `scale_2` over a denominator of 1 (`DenseW`). Not bit-exact to the f64 reference (an f32
  fused multiply-add chain per 16 inputs); bound: `|err| <= k * 2^-23 * sum |w x|`
  (`ink2g-gemm-nvfp4-dense`); rows alone are bit-equal to rows in a call.
- Layer 20-23 experts: the shipped NVFP4 W4A4 path (lane D's kernels), unchanged.
- e4m3 KV: `ink2g_heads_kernel<uint8_t>` stores by `cvt.rn.satfinite.e4m3x2.f32`;
  `ink2g_attn_kernel<uint8_t>` reads back exactly. `ink2g-kv-e4m3` (bytes equal to round to nearest
  even of the f32 values) and `ink2g-attn-e4m3` (bit-identical to the f32 kernel over the decoded
  values).

## Lease results (2026-11-21 16:47-17:17, `gpu_run.sh`, `gpu-run/`; head a6b8a7f69e, binaries in `gpu-run/binaries.sha256`)

- Clocks: clampcheck OK before every timed take; telemetry (`telemetry.csv`, 7,024 samples at 250 ms)
  holds no P2 or 11,001 MHz sample.
- Kernel cells (`kernel-cells.log`): all `ink2g` cells pass. `ink2g-gemm-nvfp4-dense` max|err|/sum|p|
  5.3e-8 (bound 2.7e-4), rows alone bit-equal. `ink2g-kv-e4m3` 0 bytes differ. `ink2g-attn-e4m3`
  0 bits differ.
- Route under nvfp4-rig: calls of 26 vs calls of 1 over 57 rows, every slot row and logit row bit-equal
  (`chunkings-rig.log`). Greedy 32/32 equal to the p1 tape. Resident weights 14.33 GiB.
- Full-tensor golden: on the CPU, lane F's quantizer equals the overlay on all 145 dense tensors
  (`golden-full-cpu-dense.json`, 0 differences). On CUDA (`gpu-run/golden-full-cuda.json`) the 8
  expert banks and 107 dense tensors are equal; 38 dense tensors differ because CUDA torch divides a
  tensor by a scalar as a multiply by the reciprocal, so `t = amax / 2688` moves by one ulp (t_ok=False
  on exactly those). The mint keeps the stated division.
- KL, decode and memory for both sets: `docs/decisions/INKLING2-NVFP4-RIG.md`. The route reproduces lane
  F's harness. All-layer nvfp4-rig on rawdoc: KL real 0.0279 (route) / 0.0267 (harness), dNLL +0.0195 /
  +0.0150. Outside-loop: 0.0212 / 0.0196, +0.0097 / +0.0102. Shipped route: 0.0127, +0.0010. Decode on
  today's kernels: shipped 5.0, outside-loop 4.4, nvfp4-rig 3.3 tok/s. Max context: 73.5k (shipped) to
  538k (nvfp4-rig).
- Per-layer arms `w:attn:L0`..`L23` on rawdoc and chatwrap (lane F's harness): `drive_perlayer.sh`,
  `gpu-run/perlayer-*.json`.
