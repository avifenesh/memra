# Spark-X2.5-4B

Support state: **NativeReference**. The `spark25` ModelPlan pack and native reference
executor run the checkpoint. Optimized numerical and complete serving qualification
remain open, so this card describes the evaluation configuration.

## Pinned artifact and program

- Source: [XHToken/Spark-X2.5-4B-FP8](https://huggingface.co/XHToken/Spark-X2.5-4B-FP8),
  revision `e186ce2d5a0442e1e42cad45e0c65907b84fc34c`.
- Matrix weights retain the checkpoint's E4M3 codes and 128×128 scale grid. Activations
  follow its declared dynamic per-128 FP8 quantization. Unquantized BF16 tensors remain
  source-preserved with `MEMRA_FULL_PREC=1`.
- Architecture: 36 dense layers, head dimension 256, separate per-head sigmoid attention
  gate, exact-erf GELU, and three 512-token sliding layers for each full-attention layer.
  The pack preserves the separate RoPE programs and fused-QKV tensor row slices.
- The current optimized cache uses q8_0 keys and q5_1 values with full-history backing.
  Step-3.7's family-specific serving defaults are not inherited by this dense pack.
- The checkpoint declares 1,048,576 positions. That declaration is not a qualified
  serving capacity on the 32 GB evaluation GPU.

## Evidence and remaining qualification

The four-token source-FP32 checkpoint oracle passes with maximum absolute logit error
0.00018501282 and matching argmax. Source tensors are explicitly dequantized for that
oracle; it does not represent an FP8 production runtime. The tokenizer/reference suites
and FP8 slicing/dispatch component tests pass. The hd256 windowed-prefill path now
completes instead of hitting the former hd128-only assertion.

The strict optimized-logit comparison remains outside its 0.05 band against the repaired
offline FP8 runtime control. Sampled tool calls can complete, but the eight-turn sampled
surface check was not fully green. These results do not promote the model to
`NativeQualified` or `NativeTuned`.

The evaluated hardware is the non-serving [RTX 5090](../rigs/rtx-5090.md), `sm_120a`.
Exact binary bindings, numerical classes, control repairs and gate results are in
[the native lane receipts](../../research/bfcl-native-20260908/RESULTS.md).
