# B200 kernel twins, dry phase

Status: hardware closure complete on one non-production NVIDIA B200. NVFP4 W4A8 is qualified on
the pinned Qwen3.5-9B artifact; W4A4 and block-FP8 remain explicit non-default arms.

Hardware-qualified source base: `origin/main` at
`0b346597c0743e71febe1e76d66124aec538bda4`. Public PR base:
`fae70b0aaf7fc8ee6780a9a699a899a44b5ceade`. Branch:
`lane/b200-kernel-twins-dry-20260901`. Toolchain: CUDA 13.1, nvcc 13.1.115.

## NVFP4 outcome

The `sm_100a` build now contains two NVFP4 prefill programs without changing checkpoint bytes or
adding an external runtime/kernel dependency:

1. Default W4A8: the existing accuracy-safe FP4-to-int8 loader and `mma.sync.m16n8k16.s8` kernel
   now compile as the real translation unit on B200. Optional `MEMRA_MMQ_F8F4=1` compiles through
   its already-gated bit-identical plain-E4M3 form; only the faster sm_120a instruction form is
   architecture-specific.
2. `MEMRA_MMQ=1` W4A4: `mmq_fp4.cu` now has a 128x128x64
   `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X` twin. It consumes the same
   `block_nvfp4` weights and `block_fp4_mmq` activation scratch as the `sm_120a` path. Scale factors
   are staged to TMEM, accumulation stays FP32, and the existing C ABI and residual correction
   remain intact.

The W4A4 form is synchronous per K=64 slice on purpose. This dry phase establishes the exact
program and production ABI. TMA staging, deeper commit groups, tile selection, and any default
decision belong to the later real-B200 window.

## Block-FP8 outcome

`mmq_fp8_blk.cu` now compiles as a real `sm_100a` translation unit. The dense path uses a
128x128x128 `tcgen05`/TMEM twin. The checkpoint's `[128x128]` scale grid and the activation
quantizer's per-128 scale remain f32: each K block is computed with identity UE8M0 factors inside
tcgen05, read back, and folded as `(weight_scale * activation_scale) * partial` in ascending block
order. No scale is rounded into UE8M0 and no weight byte is re-quantized.

The expert-grouped ABI remains present through the legal plain E4M3 `mma.sync` form on SM100. The
dense tcgen05 kernel uses 99,456 bytes of dynamic shared memory to retain the outer f32 accumulation
order and currently compiles at 254 registers, zero local memory. This is a correctness-first dry
form; the real box must price a lower-register/TMA pipeline after exactness.

On `sm_100a`, literal `MEMRA_FP8_MMQ=1` is required. Unset or `0` keeps the established dequant
fallback, because unmeasured hardware behavior does not default on.

## Dry receipts

- `bash research/b200-kernel-twins-dry-20260901/check-layouts.sh`: PASS. The exact helpers compiled
  into both production kernels map all 4,096 NVFP4 bytes and 16,384 FP8 bytes bijectively, keep the
  4X scale atom collision-free across all 512 bytes, and bound the 1X scale atom to its 128 live
  positions.
- `bash research/b200-kernel-twins-dry-20260901/check-nvfp4.sh`: PASS. W4A4 exports all four
  `memra_mmq_nvfp4*` symbols and contains `UTCOMMA.4X`, `UTCCP`, `UTCBAR`, and TMEM
  allocate/deallocate SASS. W4A8 exports its ABI and contains `IMMA.16816.S8.S8`.
- W4A4 production-kernel static resources: 55 registers, 10,260 bytes shared memory, zero local
  memory (`cuobjdump --dump-resource-usage`).
- `bash research/b200-kernel-twins-dry-20260901/check-fp8.sh`: PASS. The dense ABI contains
  `UTCQMMA`, `UTCCP`, `UTCBAR`, and TMEM allocation/deallocation; grouped ABI and
  `HMMA.16816.F32` are both present.
- `research/glm5-b200-prep-20260901/compile-census.sh`: build-faithful arm A PASS for every
  `sm_100a` translation unit. Remaining arm-B failures are deliberate negative controls that remove
  the F8F4/FP8 SM100 selections, plus the separate native-FP4 fatbin and FA3 sources.
- `MEMRA_CUDA_ARCH=100a cargo build -p memra-engine --bins`: PASS, including static archive and
  final binary linkage.
- `.github/workflows/ci.yml` runs the layout, NVFP4, FP8, installer-admission, and literal-1 FP8
  policy gates inside the `100a` release-arch cell, so an ABI-compatible stub cannot silently
  replace the real static twins while CI stays green.

These dry receipts prove the address formulas are bounded and collision-free. The hardware
receipts below separately prove how silicon interprets the descriptors, TMEM addresses,
accumulation, tails, and output values.

## Real-B200 closure, 2026-09-01

Hardware: one Nebius preemptible NVIDIA B200, compute capability 10.0, 180 GB, CUDA 13.1.115,
driver 580.173.02. Host role was labeled `research-non-production`. Tested source was clean commit
`69a2eb3684e14c64dc01516dbaddc1b261ffd1ca`.

Sealed receipt namespaces:

- `receipts/b200-phase0-20260901T1745Z/`: source hashes, B200 identity, release build, NVFP4 exact
  oracle, block-FP8 exact/random/tail oracle, and fast kernel census.
- `receipts/b200-model-nvfp4-20260901T1758Z/`: pinned Qwen3.5-9B artifact lock and required-cell
  model manifests for default W4A8 and F8F4.
- `receipts/b200-e2e-nvfp4-20260901T1804Z/`: real-prompt generation, true raw-layout W4A4 gates,
  K=1..8, sampled serving, concurrency 4, context/model refusal, rollback, and performance.
- `receipts/b200-fp8-q38-20260901T1820Z/`: pinned official Qwen3.8-27B-FP8 shard manifest,
  prime-path dispatch, fallback comparison, K=1..8, sampled serving, and concurrency 2.
- `receipts/b200-autodetect-20260901T1900Z/`: exact closure commit `af2fc06aaa85`, clean B200
  source rebuild with no architecture override, and the `compute_cap 10.0 -> sm_100a` receipt.
- `receipts/b200-pr-tip-20260902T0000Z/`: exact pre-publication hardware tip `229e0c296309`,
  clean automatic `sm_100a` release build, NVFP4 zero-mismatch oracle, block-FP8
  exact/random/tail oracle, and 92-cell fast kernel census.
- `receipts/b200-pr-tip-models-20260902T0028Z/`: the same exact PR tip against the immutable
  Qwen3.5-9B NVFP4 and Qwen3.8-27B-FP8 artifacts. The required real-tensor kernel sweep ended
  `ALL GREEN (111 cells, 13 skipped)`; default W4A8, opt-in raw W4A4, and explicit FP8 all passed
  K=1..8 self-consistency. Vendor-default sampled serving engaged speculation for both models.
- `receipts/b200-pr-tip-perf-ab-20260902T0112Z/`: settlement for the local-CI cross-day
  qwen9b throughput tripwire. Exact-parent control `0b346597c` and candidate `229e0c296` ran
  interleaved A/B, N=5 each, under one exclusive GPU lock. Medians were 132.66 and 132.30 tok/s,
  respectively, a -0.27% delta and `NO_CODE_REGRESSION` verdict.

The public PR preserves every engine-source hash in that hardware tip's
`source-manifest.sha256`. Its publish-history cleanup only replaces two raw f32 logits vectors
with `pp-logits.sha256`, so arbitrary binary receipt bytes cannot impersonate public-boundary
provider or credential patterns. The rerun preserves the same support-state decision.
Qwen3.5-9B NVFP4 W4A8 remains `NativeQualified`; raw W4A4 stays explicit. Block FP8 remains
explicit `NativeReference` and the implicit kernel-check arm refuses by policy before the
explicit qualification rerun is allowed.

### NVFP4 verdict

- Phase 0: `NVFP4-MMQ-EXACT ALL PASS`, zero mismatches across all three shapes.
- Model manifests: both default W4A8 and F8F4 ended `ALL GREEN (111 cells, 13 skipped)`; skips are
  unrelated missing-model cells, not required NVFP4 cells.
- Real model: default W4A8, F8F4, raw-layout W4A8, and true raw-layout W4A4 produced the same
  64-token greedy continuation. True W4A4 requires `MEMRA_RP=0 MEMRA_MMQ=1`; an RP weight always
  selects W4A8. K=1..8 self-consistency passed on the true W4A4 arm.
- Customer shape: Qwen's vendor-default sampled request returned 96 tokens with 108 drafted,
  57 accepted, and `usage.spec.acceptance_rate=0.5278`. Four concurrent sampled requests returned
  HTTP 200 and engaged speculation. The declared 8,192-token prompt policy refused a 9,010-token
  request with `context_length_exceeded`; unknown model and spec-off rollback gates passed.
- Performance at pp1483: default split-plane W4A8 mean 9,928.1 tok/s; raw W4A8 mean 9,134.3;
  raw W4A4 mean 4,758.6. W4A4 is **0.521x** raw W4A8 and stays explicit. F8F4 is also opt-in.

The pinned Qwen3.5-9B NVFP4 artifact advances to `NativeQualified` on this single-B200 topology.
This does not qualify every NVFP4 checkpoint or a multi-B200 topology.

### Block-FP8 verdict

- Synthetic gate: nine exact/random/ragged shapes green, exact cells bit-identical, 254/254 legal
  E4M3 codes exercised, and zero NaNs.
- Pinned official `Qwen/Qwen3.8-27B-FP8` revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`: all LFS hashes verified; coherent first light;
  K=1..8 self-consistency passed; vendor-default sampled serving returned 64 tokens with 35
  drafted and 28 accepted; concurrency 2 returned HTTP 200 with speculation engaged.
- Prime-path engagement: pp1483 recorded 1,200 FP8-MMQ dispatches, zero bad shape/scale/NaN
  refusals. The B200 twin measured 530.9 tok/s versus 3,071.1 for the established fallback =
  **0.173x**. Teacher-forced NLL was 20.790810 versus 20.621361; the last-logit argmax matched but
  top-20 overlap was 17/20.

Block FP8 advances only to `NativeReference`. `MEMRA_FP8_MMQ=1` remains explicit on B200; the
fallback is the performance and quality choice until a tuned twin earns new receipts.

### Default decisions

- Source builds auto-detect B200 compute capability 10.0 as `sm_100a`.
- NVFP4 stays W4A8 by default. W4A4 remains `MEMRA_RP=0 MEMRA_MMQ=1`.
- F8F4 remains opt-in.
- B200 block-FP8 MMQ remains literal `MEMRA_FP8_MMQ=1` opt-in.
- The release installer still refuses B200 because no `sm_100a` prebuilt is published.

## Real-B200 gate, rerun contract

Run in this order on an explicitly non-production B200:

```bash
research/b200-kernel-twins-dry-20260901/run-box-phase0.sh --plan

MEMRA_B200_HOST_ROLE=research-non-production \
MEMRA_B200_EXPECTED_SHA=<approved-40-hex-sha> \
MEMRA_B200_RECEIPT_DIR=/absolute/new/receipt/directory \
MEMRA_B200_DEVICE_LIST=0 \
  research/b200-kernel-twins-dry-20260901/run-box-phase0.sh run
```

Phase 0 fails closed on host role, exact clean source, CUDA 13.1, CC 10.0 identity, existing GPU
processes, the canonical lock, and receipt reuse. It then runs the exact synthetic
`nvfp4_mmq_check`, `fp8_mmq_check`, and fast `kernel-check` cells on every requested card and seals
the raw logs and source hashes.

The first model-backed command after phase 0 is pinned too. It refuses if any NVFP4 sub-arm is
skipped or absent:

```bash
target/release/kernel-check /absolute/immutable/model.gguf \
  --require-manifest \
  research/b200-kernel-twins-dry-20260901/kernel-check-nvfp4.cells
```

Run it once in default W4A8 mode and once with `MEMRA_MMQ_F8F4=1`; the W4A4 static cells execute in
the same battery. Artifact-specific `run-gen`, `run-spec`, and serving commands are selected only
after the box and immutable checkpoint identities are known.

1. A 128x128x64 synthetic oracle over random and poison-pattern E2M1/UE4M3 blocks, including zero,
   sign, scale-quadrant, K-chain, token-tail, and output-tail cells.
2. `kernel-check` NVFP4 W4A8 and W4A4 arms against the same immutable tensor source; every red must
   bite. Treat W4A8 and W4A4 as different numeric programs.
3. `run-gen` argmax and `run-spec` K=1..8 on the affected NVFP4 models, including the widened W4A4
   reject corpus and per-model acceptance rows required by the prefill-KV law.
4. Only after correctness: interleaved W4A8/W4A4 and synchronous/TMA cells with raw logs and SASS
   hashes. A winner may become a B200-specific default only from those receipts.
5. Block-FP8: run the exact integer/power-of-two cell, random E4M3 drift bound, NaN refusal,
   ragged K/N/M tails, dense-vs-host oracle, and grouped-vs-dense cells before any model battery.
   Then run the affected FP8 ModelPlan/checkpoint/serve gates and per-model acceptance rows.

Any changed engine source, different artifact bytes, different CUDA binary, or different topology
invalidates these receipts and reruns the applicable battery. No path is `NativeTuned` from this
lane: NVFP4 W4A8 is `NativeQualified` on the named tuple and block-FP8 is `NativeReference`.
