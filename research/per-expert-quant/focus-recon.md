# FOCUS NVFP4 format/repack reconnaissance

Research snapshot: 2026-08-12. Paper: FOCUS v1, arXiv 2608.01847 (2026-08-03).
Upstream implementation checked at AngelSlim `main`
`67394ce55f6b6cfae702575fbc3c7a05c13fdd74`.

## Verdict

**Class (a): quantizer-side only; no GGUF block or inference-kernel change.** The learned FOCUS
coefficient itself cannot be represented in memra's 36-byte NVFP4 block, but it is not supposed to
be represented there. It exists only while quantizing the pinned BF16 weights. It changes which
E2M1 codes are selected; FOCUS then discards it and stores those codes together with an ordinary,
hardware-compliant E4M3 dequantization scale. Both outputs already fit the current GGUF layout.

This needs one terminology guard. It is "free at repack time" only in the deployment-format sense:
a FOCUS-aware quantizer can emit the same bytes that the existing repack path consumes. It is **not**
a free post-hoc transformation of an already-quantized NVFP4 artifact. A pure code/scale repack has
no BF16 weights or optimization state from which to discover different assignments. For this lane,
FOCUS therefore belongs at **pinned BF16 source -> scored-artifact quantization**, before the final
GGUF byte packing. It is future-arm/v2-plan material and does not modify
[`arms.lock.json`](arms.lock.json) or reinterpret any existing arm.

## What FOCUS learns

FOCUS makes explicit that the divisor used to choose a four-bit code need not equal the multiplier
used later to reconstruct it. In the paper's notation:

```text
S_dq       = Q_fmt(S_fp)
S_q        = S_dq * c
q          = Q_E2M1(W / S_q)
W_hat      = q * S_dq
```

- The original weights are frozen. The baseline learns a full-precision (FP16) scale copy `S_fp`
  through a straight-through estimator; its quantized value `S_dq` remains E4M3 for NVFP4 (E8M0
  is the MXFP4 case). For NVFP4, the paper freezes the tensor-wide FP32 global scale and learns the
  local scale.
- Coupled-Relaxation Scaling adds a full-precision coefficient `c`, initialized to one. `c` is
  multiplied into `S_q`, the offline quantization divisor, but **not** into `S_dq`, the deployed
  multiplier.
- Dual-Granularity Scaling uses one `c` per eight-value sub-block. An NVFP4 hardware scale group is
  16 values, so it has two transient coefficients but still one deployed E4M3 scale.
- At export, only `q` and `S_dq` survive. The full-precision scale copy and every relaxation
  coefficient are discarded. Thus `c` is folded into the *choice of stored codes*, not into a new
  inference-time scale. End-to-end scale learning can also select a different `S_dq`, but it remains
  an ordinary value in the existing E4M3 field.

These points are stated in the paper's method equations (4), (6)-(8), especially §§Coupled-
Relaxation Scaling and Dual-Granularity Scaling, and in its output-model description
([FOCUS v1 PDF](https://arxiv.org/pdf/2608.01847v1)). The current AngelSlim exporter corroborates
the boundary in executable form: `quant_max_scale` affects the code divisor and is discarded, while
the exporter writes only packed weights, E4M3 local scales, and the standard global scale
([pinned packing code](https://github.com/Tencent/AngelSlim/blob/67394ce55f6b6cfae702575fbc3c7a05c13fdd74/angelslim/compressor/quant/modules/nvfp4/packing.py#L40-L146),
[pinned export code](https://github.com/Tencent/AngelSlim/blob/67394ce55f6b6cfae702575fbc3c7a05c13fdd74/angelslim/compressor/qat/export/nvfp4.py#L108-L168)). The exporter deliberately reloads the frozen base weights and
the learned scale state; it warns against quantizing fake-quantized weights a second time
([pinned FOCUS guide](https://github.com/Tencent/AngelSlim/blob/67394ce55f6b6cfae702575fbc3c7a05c13fdd74/docs/source/features/quantization/focus_fp4.md#L209-L234)).

### Quantization-time cost

"Zero inference overhead" does not mean zero artifact-build cost. FOCUS runs end-to-end
forward/backward optimization with frozen weights. The paper uses KL-Top (`k=1000`), AdamW, one
epoch, global batch 32, 1,248 WikiText2 training samples of length 2,048, and for NVFP4 learning
rates `5e-3` for the scale and `1e-3` for the coefficient. On one H20, its reported complete
quantization runs use 77 GB / 19 minutes for Qwen3-4B and 85 GB / 29 minutes for Qwen3-8B. Those are
dense W4A4 paper measurements, not estimates for Hy3 expert-only quantization. They show that the
runtime representation is free while the quantization procedure is a real GPU optimization job
([FOCUS v1, Experimental Settings and Quantization Cost](https://arxiv.org/pdf/2608.01847v1)).

## Mapping to memra's GGUF NVFP4 bytes

memra's physical block groups 64 values into 36 bytes: four positive UE4M3 scale bytes `d[4]`, one
per 16 values, followed by 32 bytes `qs[32]` containing 64 E2M1 nibbles
([type geometry](../../crates/memra-gguf/src/lib.rs#L106-L107),
[layout contract](../../crates/memra-gguf/src/nvfp4_repack.rs#L1-L28)). The repository calls the
positive scale encoding UE4M3; this is the NVFP4 E4M3 scale field. Its doubled-codebook/half-scale
GGUF convention cancels algebraically and reconstructs standard `E2M1 * E4M3` values.

| FOCUS quantity | Lifetime | Existing memra representation | Consequence |
|---|---|---|---|
| E2M1 code `q` | Inference | 64 nibbles in `qs[32]` | Fits unchanged. `c` can cause different nibble choices. |
| Dequant scale `S_dq` | Inference | Four UE4M3 bytes `d[4]`, each governing 16 values | Fits unchanged. A learned scale must still round to this field. |
| DGS coefficient `c^k` | Quantization only | None; two per 16-value group, hence eight transient coefficients per 64-value GGUF block | Correctly discarded; no field is missing. |
| NVFP4 tensor-wide FP32 scale | Inference when used | Existing sibling `.scale`/post-matmul macro-scale support; Hy3 overlay experts currently default to 1.0 | At most future exporter/manifest plumbing, not a block or kernel change. |

The scalar reference dequantizer reads exactly `d[4]` and `qs[32]`
([`dequant.rs`](../../crates/memra-gguf/src/dequant.rs#L479-L503)). The CUDA decode path does the
same: it obtains the scale byte and packed nibbles, performs the integer dot, and multiplies by the
decoded UE4M3 value; there is no coefficient lookup
([generic element dequant](../../crates/memra-engine/cu/qmatvec.cu#L324-L335),
[dp4a consumer](../../crates/memra-engine/cu/qmatvec.cu#L4699-L4748)). Existing model loading also
already supports a standard NVFP4 tensor macro-scale outside the block, including per-expert
macros, while ordinary GGUF experts use 1.0
([2-D loading](../../crates/memra-engine/src/model.rs#L567-L583),
[expert contract](../../crates/memra-engine/src/model.rs#L1288-L1304),
[per-expert macro loader](../../crates/memra-engine/src/model.rs#L2036-L2065)).

The E8M0 restriction in the motivating FOCUS example belongs to **MXFP4**, not this NVFP4 arm.
For memra NVFP4 the analogous hard constraint is the discrete UE4M3/E4M3 dequant-scale byte. FOCUS
relaxes neither runtime constraint: only the temporary quantization divisor has full precision and
finer granularity.

## Where the mechanism would enter the Hy3 pipeline

The current artifact builder is a source quantizer despite its `repack` name. For each BF16/F16/F32
expert row, `quantize_nvfp4_rows()` computes absmax/6, rounds it to UE4M3, assigns nearest E2M1
codes, and writes the four scale bytes plus 32 packed bytes
([current quantizer](../../tools/prepare_mixed_expert_repack.py#L268-L303),
[call site](../../tools/prepare_mixed_expert_repack.py#L715-L727)). That encoder is the output
boundary a future FOCUS implementation would replace or feed; its 36-byte contract does not move.

By contrast, `repack_modelopt_to_gguf()` receives codes and E4M3 scale bytes that already exist. It
copies the scale bytes verbatim and only changes nibble order
([pure repack](../../crates/memra-gguf/src/nvfp4_repack.rs#L208-L249)). Running FOCUS there would be
too late. The required future sequence is:

```text
pinned Hy3 BF16 source at 716aa7241bd6d95896be4ebfc761162a9c4d49ef
  -> future-only FOCUS scale/coefficient optimization on non-public training data
  -> re-read the same frozen BF16 tensors plus the frozen learned state
  -> emit ordinary d[4] + qs[32] NVFP4 blocks (and existing macro sidecar if selected)
  -> hash the exact quantizer, training state, source, and output artifact
  -> only then run the normal private sensitivity and public-evaluation gates
```

The sensitivity tool currently promises to reuse the artifact builder's exact quantizers
([contract](../../tools/build_hy3_quant_sensitivity.py#L1-L7),
[quant/dequant path](../../tools/build_hy3_quant_sensitivity.py#L105-L147)). A future arm must keep
that identity: it cannot score the current absmax encoder and build with FOCUS, or vice versa.

## Scope and effort

**Implementation effort: M.** There is no runtime-format, dequant-kernel, cache, or spill-path work.
The medium work is quantizer-side: reproduce or adapt the end-to-end optimizer for Hy3 experts,
bind its training data and state hashes, export from the frozen BF16 source, support the existing
optional macro scale deliberately, and make sensitivity scoring consume the identical emitted
bytes. Paper quality transfer is unproven because its evidence is dense all-linear W4A4 on 1B-8B
models, whereas this lane quantizes routed Hy3 expert weights and uses memra's own activation/runtime
path.

Two proposals would cross into **class (b)** and are out of scope: storing the full-precision
coefficient for inference, or storing a non-E4M3 dequant scale in `d`. Neither is required by
FOCUS. Applying the method to already-quantized public MLX experts would instead be **inapplicable**;
the pinned BF16 checkpoint remains the only valid source.
