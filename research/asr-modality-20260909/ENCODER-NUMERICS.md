# Whisper encoder numerical stages

Source checkpoint: `ivrit-ai/whisper-large-v3@766847c9795b3b5cc0d42f8476199c711d5cee21`.
One deterministic synthetic 2-second waveform, 16 kHz, padded normally to 30 seconds.
HF reference: transformers 5.16.1, torch 2.14.0, one CPU thread, eager attention.
This is operator/reference evidence, not WER, GPU performance or serving qualification.

## Passed frontend

Native log-mel `[128,3000]`: max abs **5.6743622e-5**, limit **1e-3**.
Receipt: `stage1-mel.json`. The fixture and numerical assertion run in hosted CI.

## Encoder diagnosis

The first pre-fix FP32 block above 1e-2 was layer 20: **0.010009765625** at
`[1482,695]`. The final normalized output was **0.0017070770**. Shorter FP32 dot
reductions and stable normalization helped but did not clear the later 1e-3 final gate.
All diagnostic captures remain under separate names; no failure was dropped.

The actual mismatch was the GELU numerical program. The native implementation used
scalar `erff`. The pinned HF CPU vector implementation uses the published five-term
Abramowitz-Stegun 7.1.26 erf approximation with FP32 fused Horner evaluation. On identical
layer-20 FC1 inputs, the native scalar-erf GELU differed by max **4.7683716e-7**, mean
**6.8554776e-8**, across millions of values. Those small systematic differences propagated.
The owned Rust implementation of the published formula reduces the mean to **4.6528124e-9**.
No upstream kernel is vendored, linked or called by the native execution path.

Sources: [pinned PyTorch GELU](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/cpu/Gelu.h),
[pinned vector erf program](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cpu/vec/vec512/vec512_float.h),
[NIST handbook](https://www.nist.gov/mathematics-statistics/handbook-mathematical-functions-abramowitz-and-stegun).

After the GELU correction, native FP32 final max abs is **0.0002613067626953125**, below
**1e-3**. Layer 20 is **0.00079345703125**, and every intermediate is below 1e-3.
Receipt: `stage2-encoder-f32-gelu.json`.

The final reference implementation retains four fixed FP32 dot partial sums, blockwise
FP32 Welford normalization and eight FP32 softmax partial sums. The unsuccessful longer
matrix experiments were removed. The reduced-precision LayerNorm affine follows the pinned
HF evaluation order; it is not reassociated into the FP32 expression.

## FP16 comparison rule

HF's own strict-FP16 encoder differs from HF FP32 by **0.27716827392578125** on this same
fixture. It first exceeds 1e-2 at layer 5. The original native strict-FP16 implementation
had max abs **0.375** against HF FP16 before the normalization/GELU correction.
Those failures are retained in the matching JSON receipts.

Owner amendment, 2026-09-09: after native FP32 is within 1e-3 of HF FP32, compare native
FP16 to HF FP16 using the measured same-fixture HF FP16-vs-FP32 max abs as its stage gate.
`tools/check_whisper_stages.py` derives the floor from hash-verified controls and refuses
mismatched PCM, source revision, config or shape. No manually enlarged native-error threshold.
The corrected FP16 final max abs is **0.25** versus HF FP16, below the derived
**0.27716827392578125** floor. Receipt: `stage2-encoder-f16-softmax-order.json`.
The live HF build selects AVX2. The native reference now uses its 8-lane moment grouping
and half-softmax reduction order, scalar tail placement and reciprocal multiplication.
The first LayerNorm isolated probe differed in 420/1,920,000 entries, max 0.000244140625;
the material remaining mismatch was the half-softmax numerical order. No mixed-precision
substitution or manually enlarged threshold was used.

A mixed FP16-weight/FP32-activation diagnostic reached **0.0357141495** against HF FP32,
but still missed its gate. That mode was removed when the owner selected same-class FP16
comparison. Its receipt and isolated layer-20 attention/FFN controls remain as evidence.

## Scope

All rig execution was explicitly owner-authorized tiny CPU work under nice, one process/thread.
No GPU, local workspace CI, general battery or smoke server. Source shard hashes were verified.
Hosted CI gates each committed stage; pushes use `MEMRA_SKIP_PERF_CI=1`.
The complete ASR program is still unqualified: decoder policy, pinned rental oracles,
checkpoint/serve parity, RNNT and target-device execution remain separate gates.

## Sealed encoder stage

Final F32 receipt: `stage2-encoder-f32-final.json`, max abs **0.0002231597900390625**,
limit **1e-3**. F16 receipt: `stage2-encoder-f16-softmax-order.json`, max abs **0.25**,
limit **0.27716827392578125** versus HF F16. Both bind the exact same native binary.
All eight focused speech tests pass, including the fixed HF vector-GELU fixture.
No further F16 precision experiment is active. The next implementation stage is cached decoding.
