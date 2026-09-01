# FlashInfer SM12x NVFP4 zero-output finding

Date: 2026-08-14

Status: **SOURCE-ONLY; NO-GO FOR RUNTIME PROMOTION**

## Scope

This is a CPU-only, public-source qualification of the earlier FlashInfer SM12x/SM120
zero-output finding. It used no GPU, no private Darklanes material, and no benchmark. The
worktree baseline was `138cf4acfa970d9e3e40c77dfa7c7caf497be0cc`.

The package contains only public upstream facts and a future admission gate:

- [`sources.lock.json`](sources.lock.json) pins the official primary sources and immutable
  source revisions inspected.
- [`affected-shapes.json`](affected-shapes.json) preserves the corrected end-to-end symptom,
  the maintainer's no-GEMM root-cause matrix, and the merged regression matrix separately.
- [`FUTURE-GATE.md`](FUTURE-GATE.md) defines the correctness-first requalification order.

## Corrected verdict

The prior handoff, "an open SM120 shape-dependent `mm_fp4` zero-output bug blocks
promotion," is stale in two important ways:

1. [FlashInfer issue #3398](https://github.com/flashinfer-ai/flashinfer/issues/3398) is
   closed, not open. The fix merged on 2026-06-03 in
   [PR #3497](https://github.com/flashinfer-ai/flashinfer/pull/3497) and shipped in
   [v0.6.13](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.13).
2. The two GEMM backends were not the root cause. A FlashInfer maintainer reproduced the
   failure without any GEMM and isolated it to the CUDA `nvfp4_quantize` path. In v0.6.12,
   a BF16/FP16 `global_scale` could be read byte-wise as FP32 because the native dtype
   guard was disabled. That produced all-zero scale buffers for some inputs and
   systematically under-scaled values for others. Both `b12x` and `cutlass` then consumed
   the same corrupt packed inputs faithfully.

PR #3497 made two matching changes:

- normalize `global_scale` onto the input device as FP32 in the Python wrapper;
- restore the native FP32 input check before casting the scale pointer to `float *`.

It also added a no-GEMM regression test that checks for nonzero scale buffers and checks
dequantized magnitude, not cosine alone. The fix commit is an ancestor of current
[v0.6.17](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.17).

This correction clears the historical GEMM indictment. It does **not** admit current
FlashInfer into memra: no current release was exercised on either target rig in this lane,
and v0.6.17 contains later SM12x FP4 numerical and dense-kernel changes. Current source is
a candidate for the future gate, not evidence for a default change.

## Exact observed shapes

All tables below use `K=2048`; the paired GEMM has `N=8192`. The complete numeric record is
in [`affected-shapes.json`](affected-shapes.json).

### Corrected reporter run: end-to-end symptom

After restoring the correct `compute_120f` target, the reporter observed the same result
from `mm_fp4(backend="b12x")` and `mm_fp4(backend="cutlass")`:

| Result | M values |
|---|---|
| nonzero output | 1, 8, 2048, 3072, 4096 |
| all-zero output | 32, 128, 256, 512, 1024, 1536 |

That test quantized A and B immediately before GEMM. It therefore records an end-to-end
symptom, not a GEMM-only result. The issue body's earlier table was produced while a local
patch still forced `compute_120`; the reporter explicitly superseded it, so this package
does not use that table as gate evidence.

### Maintainer run: quantize/dequantize with no GEMM

The maintainer then isolated `nvfp4_quantize` on tensors `[M, 2048]`. Both the `128x4` and
linear scale layouts behaved identically:

| CUDA quantizer result | M values |
|---|---|
| scale buffer 100% zero | 384, 512, 1024 |
| nonzero but magnitude wrong | 256, 1280, 1536, 2048, 3072, 4096 |

For the nonzero CUDA rows, cosine remained about `0.9955`, while relative L2 error was
`0.802` to `0.951` and median `abs(dequant/input)` was only `0.049` to `0.198`. A
cosine-only test therefore labeled direction-preserving, badly under-scaled output "OK."
The CuTe-DSL control was correct at every tested M, with relative L2 about `0.095` and
median magnitude ratio about `0.994` to `0.999`.

These M values are observations, not the causal support boundary. The causal condition was
the non-FP32 global scale being reinterpreted as FP32. Shape and input values changed the
misread bytes and the visible symptom, which is why the two public reproductions have
different zero sets.

## Artifact generation versus runtime

The layer distinction is mandatory:

| Layer | What #3398 establishes | Posture |
|---|---|---|
| Offline artifact generation that called FlashInfer v0.6.12 CUDA `nvfp4_quantize` with a non-FP32 global scale | Packed scale factors can be corrupt before inference starts. | Reject or regenerate from the pinned source after offline dequant validation. |
| An artifact produced by another quantizer | #3398 says nothing about its correctness. | Require its own manifest, hashes, and dequant oracle; do not infer guilt or safety from #3398. |
| Runtime activation/input quantization through the affected helper | A valid weight artifact does not protect the request; temporary activation scales can be corrupt. | Gate every runtime quantization call shape and scale dtype. |
| `mm_fp4` fed independently validated packed values and scale factors | The maintainer explicitly cleared both GEMM backends for this incident. | Test `b12x` and `cutlass` separately against a BF16 oracle; do not use #3398 as proof that either backend is correct in general. |

A zero result observed after loading an artifact is not enough to locate the defect. The
gate must first dequantize the stored/generated representation without GEMM, then run GEMM
with independently validated inputs, and only then test the combined path.

## Source currency

The latest release found at the recorded fetch time was v0.6.17. It includes the v0.6.13
dtype fix, later b12x FP4 numerical fixes
([PR #3932](https://github.com/flashinfer-ai/flashinfer/pull/3932)), and a refresh of the
SM120 dense `mm_fp4` kernel
([PR #4253](https://github.com/flashinfer-ai/flashinfer/pull/4253)). Those changes make a
fresh correctness run worthwhile, but they also make the old v0.6.12 observation
insufficient to qualify current code.

## No-go and default posture

- **Hard no-go:** FlashInfer v0.6.12 CUDA `nvfp4_quantize` with any non-FP32 global scale.
- **Hard no-go:** any artifact produced through that path unless its packed values and
  scales pass the offline gate or it is regenerated from the pinned source.
- **No-go for promotion:** current FlashInfer `b12x`, `cutlass`, or fused-MoE FP4 paths
  until [`FUTURE-GATE.md`](FUTURE-GATE.md) passes on the designated SM120 rigs.
- **No timing before correctness:** a fast all-zero or magnitude-wrong path is a failed
  path, not a benchmark result.
- **Default unchanged:** this package changes no code, dispatch, flag, artifact, or
  runtime default. The documented Marlin W4A16 MoE/default posture remains in force.

## Verification performed

The source audit re-fetched the issue, all comments, PR metadata/files/commits, release
records, tag commits, and immutable source blobs from the official
`flashinfer-ai/flashinfer` repository. It confirmed:

- issue #3398 is closed at the PR #3497 merge time;
- PR #3497 merge commit is
  `d8cb7553f5da653e3cbe9efe0d9d0db8d81c4070`;
- v0.6.13 contains the Python FP32 normalization, restored native type check, and
  regression test;
- v0.6.17 descends from that merge commit;
- the v0.6.12 source lacks both guards while v0.6.13 and v0.6.17 contain them.

No claim here is a target-rig runtime result.
