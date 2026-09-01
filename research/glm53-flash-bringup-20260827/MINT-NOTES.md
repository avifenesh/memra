# GLM-5.3-Flash NVFP4 mint — decision record (2026-08-27)

Owner directive: "mint with nvidia tools, should be the leading ones" → NVIDIA
TensorRT Model Optimizer (nvidia-modelopt). Files: `mint-nvfp4.py` (the mint),
`mint-run.sh` (on-box launcher). NOT yet run — no GPU box in this session.

## The recipe in one paragraph

**NVFP4 W4A16 weight-only** (e2m1 weights, per-16 e4m3 block scales + per-tensor
f32 macro-scale, activations untouched), **zero calibration**, streamed
tensor-by-tensor from the BF16 twin
`zai-org/GLM-5.3-Flash-BF16 @ f12e0fe1f6b2ea274c11a569582edfd99d993c5e` (656 GB)
into a modelopt/compressed-tensors HF safetensors checkpoint (~204 GB computed —
see Size). Quantization math is `NVFP4QTensor.quantize` from
`nvidia-modelopt==0.46.0` — the exact function modelopt's own
`export_hf_checkpoint` → `to_quantized_weight` calls — so the emitted bytes are
identical to an official export; only the *packaging* (shard writing, index,
config) is done by the mint script, mirroring the 0.46.0 export code verbatim.

## Version pin

- `nvidia-modelopt==0.46.0` (PyPI latest as of 2026-08-27). W4A16 NVFP4
  (`W4A16_NVFP4_CFG`, qformat `w4a16_nvfp4`) landed in **0.45 (2026-07-02)**
  per CHANGELOG.rst; `W4A16_NVFP4_CFG` confirmed present at the 0.46.0 tag
  (modelopt/torch/quantization/config.py:1736).
- Import-time deps discovered by local dry-run: `torch`, `safetensors`,
  `requests`, `huggingface_hub` (`import modelopt.torch` hard-requires the last
  two). mint-run.sh preflights all of them.

## Quant config (semantics of mtq.W4A16_NVFP4_CFG, applied per-tensor)

From the 0.46.0 config YAMLs (`configs/ptq/presets/model/w4a16_nvfp4.yaml` →
`units/w4_nvfp4.yaml` → `numerics/nvfp4.yaml`):

```
weight_quantizer: num_bits: e2m1, block_sizes: {-1: 16, type: dynamic, scale_bits: e4m3}
input_quantizer:  DISABLED (weight-only)
algorithm: max   (moot here — no activation quantizers to calibrate)
```

Per tensor W [out, in]:
- `weight_scale_2` (f32 scalar) = amax(|W|) / (6 × 448)   (E2M1_MAX × E4M3_MAX)
- `weight_scale` (e4m3, [out, in/16]) = block_amax / (6 × weight_scale_2), zeros → 1.0
- `weight` (u8, [out, in/2]) = e2m1 codes, element 2i in the LOW nibble
(nvfp4_tensor.py @0.46.0: `get_weights_scaling_factor_2`,
`get_weights_scaling_factor`, `quantize` — packing line
`packed = (q[..., 1::2] << 4) | q[..., 0::2]`.)

**Trap avoided**: `NVFP4QTensor.quantize(..., try_tensorrt=True)` on a box with
tensorrt_llm installed returns CUTLASS-**swizzled** scales, not the modelopt
layout. The mint uses the default `try_tensorrt=False` and asserts the scale
shape/dtype `[out, in/16] float8_e4m3fn` on every tensor.

## Calibration: NONE — and why that is honest

W4A16 NVFP4 is weight-only with *dynamic* block scales derived entirely from the
weight tensor itself; modelopt's own docs state it "does not require a
calibration forward pass" (CHANGELOG 0.45; `modelopt_recipes/ptq.md` weight-only
schemes; `model_calib.py` `weight_only_quantize` runs when `forward_loop=None`).
memra's W4A4 kernels quantize activations dynamically at runtime, and
`nvfp4_repack.rs` explicitly marks `input_scale` UNUSED — so no activation
statistics are needed in the checkpoint at all.

**Consequence: the chat-template / reasoning_effort question is moot for this
mint.** No prompts are rendered, no forward passes run. (If a future mint moves
to W4A4 static activation scales or AWQ, calibration prompts must be rendered
through `chat_template.jinja` with reasoning_effort pinned — that requirement
re-activates then.)

## Why streaming per-tensor, not `mtq.quantize` + `export_hf_checkpoint`

Surfacing a deviation from the task's "via modelopt's HF export path"
instruction, with receipts — the model-level path **cannot load this checkpoint
on this box**:

1. **656 GB bf16 exceeds VRAM+RAM** (2×96 GB + 384–499 GB). The only modelopt
   path that fits memory (`init_quantized_weights` low-memory mode, compress-on-
   load) loads via `accelerate.load_checkpoint_and_dispatch`
   (plugins/accelerate.py @0.46.0), which does **raw key matching** — it never
   applies transformers' checkpoint-key conversion.
2. **The glm5_next modeling renames KDA keys**: it builds ONE fused
   `self_attn.conv1d` (`conv_dim = 3*qkv_dim`) while the checkpoint stores
   `q_conv1d`/`k_conv1d`/`v_conv1d` separately, and nests `f_a_proj`/`f_b_proj`
   under `self_attn.forget_gate.*` while the checkpoint stores them flat
   (modular_glm5_next-ref.py:616-637 vs CENSUS.md). Under raw key matching those
   KDA weights silently never load.
3. **The modeling drops the MTP layer entirely**:
   `_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", r"layers\.\d+\.shared_head\."]`
   (modular_glm5_next-ref.py:1235) and `range(config.num_hidden_layers)` builds
   only layers 0–44 — a model-level export would lose NextN, which memra needs
   for the future MTP/spec route.
   (Plain `from_pretrained`, which DOES convert keys, cannot fit; additionally
   `init_quantized_weights` @0.46.0 patches only `AutoModelForCausalLM`, and
   this arch is `Glm5NextForConditionalGeneration`.)

NVIDIA's own precedent for exactly this situation and this model lineage is
`examples/deepseek/deepseek_v4/quantize_to_nvfp4.py`: stream the source
checkpoint, quantize the target tensors to NVFP4, write HF shards directly.
The mint follows that pattern with modelopt's `NVFP4QTensor.quantize` as the
only quantization math. Memory: one tensor at a time (largest ~1.3 GB), so RAM
is never a constraint and both GPUs stay free.

## Precision split (owner-pinned; == the vendor's own FP8 split)

QUANTIZE (37,338 tensors): 37,152 routed-expert projections (288 × 43 layers ×
gate/up/down) + 129 shared-expert projections + 9 dense-MLP projections (layers
0–2) + 48 MLA projections (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`,
`o_proj` on the 12 DSA layers = 3,7,…,43 + MTP layer 45).

KEEP bf16 (1,432 tensors): all KDA tensors (q/k/v_proj, q/k/v_conv1d, b_proj,
f_a/f_b, g_a/g_b, o_norm, A_log, dt_bias, **o_proj on the 34 KDA layers**),
`kv_b_proj`, `indexer.*`, `mlp.gate.*` (router + e_score_correction_bias),
`hc_*` (mHC), all norms, `embed_tokens`, `lm_head`, `model.visual.*`, MTP
scaffolding (`eh_proj`/`enorm`/`hnorm`/`shared_head.norm`).

`o_proj` is the one name shared across both attention types; the split is
layer-indexed from config `linear_attn_config.kda_layers`/`full_attn_layers`
and cross-checked against per-layer `q_proj` vs `q_a_proj` presence in the
checkpoint index.

**Receipt (local test, 2026-08-27)**: simulating the BF16 twin tensor list from
the banked FP8 index (76,108 names minus 37,338 `*_scale_inv`), the classifier
labels all 38,770 tensors with zero unclassified, and the quantize set is
**set-identical to the vendor's own FP8 `weight_scale_inv` set** (37,338 = 37,338,
symmetric difference empty).

## Fail-loud gates in the mint

- Census: total tensor count must be exactly 38,770 and quantize count exactly
  37,338, or the mint stops before writing anything.
- Every tensor must classify as exactly one of QUANTIZE/KEEP; any unknown name
  raises (`UNCLASSIFIED tensor (extend the classifier deliberately...)`).
- Per quantized tensor: source dtype BF16, 2-D, `in_features % 64 == 0` (memra
  block_nvfp4 QK; also satisfies modelopt's %16) — asserted from shard headers
  BEFORE any quantization work.
- Per quantized output: packed `uint8 [out, in/2]`, scale
  `float8_e4m3fn [out, in/16]`, scale_2 scalar f32; scale bytes asserted free of
  the sign bit (memra decodes them as UNSIGNED ue4m3) and of the 0x7F NaN code.
- Cross-implementation dequant spot check (every 500th quantized tensor):
  an independent reimplementation of **memra's consumer math** from
  nvfp4_repack.rs (std e2m1 code × raw-ue4m3 bit-decoded scale × scale_2) must
  agree with modelopt's own `NVFP4QTensor.dequantize` within rtol 1e-5
  (1-ulp association noise ~4e-7; a nibble/scale error is O(1)).
- Conservation: every source tensor consumed exactly once; final pass re-reads
  the output shard headers and asserts the complete triple for every quantized
  stem, byte-shape identity for every kept tensor, no stray scales, exact
  output-tensor count.

**Drill receipts (local, mini synthetic checkpoint, modelopt 0.46.0 + torch
2.13 cpu, final script)**: full end-to-end mint → `MINT-DONE` (35 triples + 113
kept, 3 shards, verify OK; spot-check median rel err ≈ 0.10 = expected e2m1
noise on gaussian weights). Corruption drills, all through the SHIPPED
`spot_check`/classifier, all fire: nibble swap → gate raises; single scale-byte
exponent-bit flip in the CPU copy (what memra would read) → gate raises (15/512
elements diverge, caught); unknown tensor name → raises; MLA name on KDA layer
→ raises; o_proj split correct on layers 0/3/45-analog.

## Output artifact

```
~/models/glm53-nvfp4/
  model-XXXXX-of-YYYYY.safetensors   ~10 GB shards
  model.safetensors.index.json
  config.json          + quantization_config (compressed-tensors form, below)
  hf_quant_config.json   legacy modelopt form
  tokenizer*, chat_template.jinja, preprocessor/generation configs (copied)
```

`hf_quant_config.json` (mirrors `get_quant_config` + `process_layer_quant_config`
@0.46.0): `{"producer": {"name": "modelopt", "version": "0.46.0"},
"quantization": {"quant_algo": "W4A16_NVFP4", "kv_cache_quant_algo": null,
"group_size": 16, "exclude_modules": [ ...1,000+ explicit module names... ]}}`.

`config.json → quantization_config` (mirrors `convert_hf_quant_config_format`
@0.46.0 for W4A16_NVFP4): weights-only `config_groups.group_0` (`num_bits: 4,
type: float, group_size: 16, targets: ["Linear"]`), `ignore` = same explicit
list, `"quant_algo": "W4A16_NVFP4"`, `"quant_method": "modelopt"`.
(exclude/ignore uses explicit module names, not modelopt's prefix-wildcard
summarizer — deliberate: unambiguous for memra ingest; embeddings included.)

## Size (computed from config dims, not the 165 GB slogan)

Quantized ≈ 314.4B params (experts 311.7B + shared 1.08B + dense 0.45B + MLA
1.21B) → packed 157.2 GB + scales 19.7 GB = **176.9 GB**. Kept bf16 ≈ 13.6B
params → **~27 GB**. **Artifact total ≈ 204 GB** (10⁹ bytes; ≈190 GiB).

⚠ The BRINGUP "~165 GB → 2×96 GB PP fit" target is optimistic: 204 GB > 192 GB
of combined VRAM. Even with embed/lm_head/vision (~6–9 GB) host-resident,
weights alone are ~195 GB before KV/state/activations. Surfacing, not
scope-adjusting: closing the gap (e.g. also-NVFP4 for KDA projections — the
vendor kept them high-precision, so that needs its own quality gate — or a
partial-host-resident expert tier) is an owner call.

## How memra ingests it

`crates/memra-gguf/src/nvfp4_repack.rs`: the FP4 code nibbles and ue4m3 scale
bytes copy through VERBATIM (memra's doubled-e2m1 codebook × its 0.5-halved
ue4m3 decode cancel exactly); only the within-row nibble grouping is repacked
(sequential 2-per-byte → 64-elem GGUF sub-blocks). `weight_scale_2` becomes the
sibling `<stem>.scale` post-matmul macro-scale. `input_scale` does not exist in
this artifact and is marked UNUSED by the repack header — memra quantizes
activations dynamically.

**⚠ Detection string**: this artifact says `"quant_algo": "W4A16_NVFP4"`, NOT
`"NVFP4"`. memra's existing modelopt detection (config.rs:2539-2540, step3p7-
scoped) matches `quant_method == "modelopt" && quant_algo == "NVFP4"` — the
glm5_next pack's ingest must accept `W4A16_NVFP4` (tensor layout is identical;
the only difference is the absent `input_scale`). Do not "fix" this by writing
a false `NVFP4` label into the artifact.

## Running it

```
./mint-run.sh                # nohup + log, prints tail command
grep -E 'MINT-DONE|MINT-FAILED' ~/mint-nvfp4-*.log
```

Runtime is IO-bound: read 656 GB + write ~204 GB + 37,338 small quantize ops
(GPU-accelerated when a card is visible, CPU fine). Expect low single-digit
hours on box NVMe; RAM stays in the tens of GB (shard buffer ~10 GB + one
tensor). Neither GPU is meaningfully occupied — the box can keep other lanes.

After the mint: the artifact gates argmax-vs-reference before any serving or
publish (TRAP:convert-direct-q8; BRINGUP "NVFP4 mint" section).

## Open risks

1. **BF16 twin index assumed = FP8 index minus `*_scale_inv`** (same names,
   BF16 dtypes). Asserted at runtime (38,770 / 37,338 / per-tensor dtype); if
   the twin differs the mint stops loudly — investigate, don't loosen.
2. `weight_scale_2` written as 0-dim f32 (modelopt export convention); verify
   accepts `[]` or `[1]`. If memra's HF reader requires 1-dim, flatten at ingest.
3. `kv_a_proj_with_mqa` out-dim taken from the checkpoint (not assumed);
   size math above estimates it — artifact total may drift a GB or two.
4. Artifact ≈ 204 GB vs the 165 GB placement target (above) — owner decision
   pending on how Phase-2 placement absorbs it.
5. Revision pin: the script records `f12e0fe...` but cannot verify a local dir's
   HF revision; the download lane's pin-verify receipt is the authority.
6. modelopt 0.46.0 quantize emits per-block scales with `zeros → 1.0`
   (get_weights_scaling_factor); an all-zero row therefore dequantizes to zero
   via zero codes, not zero scales — consistent with memra decode either way.

## Citations

Verified at the **0.46.0 tag** (2026-08-27) — everything the mint's bytes and
formats depend on:

- W4A16 NVFP4 config + numerics (fetched at ref 0.46.0, identical to main):
  https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0.46.0/modelopt_recipes/configs/ptq/presets/model/w4a16_nvfp4.yaml
  https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0.46.0/modelopt_recipes/configs/ptq/units/w4_nvfp4.yaml
  https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0.46.0/modelopt_recipes/configs/numerics/nvfp4.yaml
  (+ `W4A16_NVFP4_CFG` at config.py:1736 of the tag)
- Quantize math + packing:
  https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0.46.0/modelopt/torch/quantization/qtensor/nvfp4_tensor.py
- Export format writers replicated:
  https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0.46.0/modelopt/torch/export/quant_utils.py
  (get_quantization_format → W4A16_NVFP4 when input quantizer disabled;
  get_quant_config; process_layer_quant_config)
  https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0.46.0/modelopt/torch/export/convert_hf_config.py
  https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0.46.0/modelopt/torch/export/unified_export_hf.py
- `weight_only_quantize` / `forward_loop is None` path present at the tag:
  https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0.46.0/modelopt/torch/quantization/model_calib.py (line 179)
- `init_quantized_weights` (the rejected model-level path):
  https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0.46.0/modelopt/torch/quantization/plugins/accelerate.py
- Plus the empirical receipt: the 0.46.0 micro end-to-end run above exercised
  quantize/packing/scales on the installed wheel itself.

Read from **main** only (background/rationale, not byte-affecting):

- https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/CHANGELOG.rst
  (W4A16 in 0.45; low-memory 0.31; compress 0.29)
- https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/hf_ptq/README.md
  and hf_ptq.py (the low_memory_mode flow analyzed and rejected)
- https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/deepseek/README.md
  (DeepSeek V4 routed-expert NVFP4 streaming precedent, quantize_to_nvfp4.py)

Consumer contract: /home/avifenesh/projects/wt-glm53/crates/memra-gguf/src/nvfp4_repack.rs (header)
