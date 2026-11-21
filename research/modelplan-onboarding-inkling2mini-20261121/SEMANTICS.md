# Inkling 2 Mini: program semantics and open facts

Read from `config.json`, `hf_quant_config.json`, the safetensors headers, llama.cpp at
`29b52d398b29` (`inkling2.cpp`, `inkling2-loop.cpp`) and transformers PR 51050 at
`6173a2f679ff`. The module math below is common to both numeric classes. They differ only at the expert
GEMMs: the declared class, which is the default and the support target, quantizes the expert
inputs (see Stored quantization); the float-activation class, a bring-up and oracle step, does
not. Nothing here was executed by memra.

## Trunk

- 24 physical layers, hidden 2304, 24 query heads, 8 KV heads, head dim 128, RMS eps 1e-6.
- No RoPE. Position enters through a relative-position bias and, on full layers, a log scale.
- Full attention on layers 5, 11, 17, 23. The other 20 are sliding-window, window 1024.
- Token embeddings pass through their own RMS norm (`embed_norm`). Image embeddings would not.
- Head: block-residual mix with `output_query`, RMS norm, untied `unembed [201024, 2304]`
  accumulated in f32 (`fwd_output_dtype: float32`), softcap `30 * tanh(x / 30)`, then
  truncation to the 200,058 unpadded ids.

## Loop

- Core layers 6 to 17 (`core_start 6`, `core_end 18`) run 3 passes. Cache slots: 24 + 2 x 12 = 48.
  Slot `s >= 24` uses physical layer `6 + (s - 24) % 12`. Each slot owns its KV and conv state.
- Execution order: layers 0-5, pass 0 over layers 6-17 (slots 6-17), pass 1 (slots 24-35),
  pass 2 (slots 36-47), then layers 18-23 (slots 18-23).
- Every pass starts again from the prelude output (layer 5) and the prelude's block list.
- Passes 1 and 2 read a message from the previous pass: a depth softmax over that pass's block
  outputs after the first core block plus its final output, with 4 queries per core layer
  (q, k, v, mlp). The checkpoint stores them as `middle_cycle_cross_pass_queries [24, 2304]`
  (rows `2i` = q and `2i + 1` = mlp of core layer `i`) and
  `middle_cycle_cross_pass_kv_queries [24, 2304]` (rows `2i` = k, `2i + 1` = v).
- Each message enters through a rank-64 controller (`controllers.{attn,k,v,mlp}` on layers
  6-17): `g = down(cat(rms(x), rms(m)))`, output `x - tanh(forget g) * x + tanh(gain g) * m`.
  `forget` and `gain` run in f32.
- The message bank is per token. The loop adds no cross-token state.

## Attention

- Inputs: a sliding layer mixes one input and norms it with `attn_norm`. A full layer mixes
  three inputs (q, k, v; `attn_queries` shape `[3, 2304]`) and norms each with its own
  `{q,k,v}_input_norm`. In passes 1 and 2 the three inputs of a sliding layer differ too, since
  each takes its own controller before the shared norm.
- `attn_res.attn_queries.N` belongs to layer `N + 1`; layer 0 has no attention mix.
- Q: `wq_du`, per-channel scale `q_l2_clip.scale [24, 128]`, then an L2 clip per head at
  22.627 (`x * min(1, t / (|x| + 1e-6))`). K: `wk_dv`, short conv, L2 clip without a scale.
  V: `wv_dv`, short conv, weightless RMS norm.
- R: `wr_du` gives `[24 heads, 16]`, scaled by `rel_l2_clip.scale` and clipped at 8.
- Full layers multiply Q and R by `tau = 1 + 0.1 * ln(max(1, (pos + 1) / 8192))`.
- Bias: `bias[h, d] = (R_h @ rel_logits_proj.proj [16, 1024])[d]` for distance
  `0 <= d < 1024`, zero outside.
- Sink: a per-query logit `attn_sink_proj(x) [24]` joins the softmax denominator.
- Scale `1 / sqrt(128)`. The output is multiplied by `sigmoid(gate_attn(x))`, a separate
  per-dimension gate `[3072, 2304]` on the query input, then `wo_ud`, then a short conv.

## Short convolutions

- Kernel 4, depthwise, causal, with a residual: `(x + conv(x)) / sqrt(k + 1)`. Four per layer:
  K, V, attention output, MLP output.
- Full layers add input-dependent taps on K and V: `weight_proj [32, 2304]` gives 4 taps for
  each of the 8 KV heads, the static taps are `static_w [4, 1024]`, and the divisor becomes
  `sqrt(2k + 1)`.
- State per slot: the last 3 inputs of 1024 + 1024 + 2304 + 2304 = 6656 channels.

## Block attention residual

- Block size 6. The first layer of a block pushes its input onto the block list and starts a new
  stream: its attention output is not added to the input.
- Every sublayer input is a mix: `softmax(q . rms(states) / sqrt(hidden))` over the earlier block
  outputs plus the current state, summing the normalized states (`attn_res_normalize_values`).

## MoE

- 128 experts, top 8, no shared expert, expert width 1152.
- Router in f32: `sigmoid(W x)`, top-k on score plus `gate.bias`, weights normalized over the
  selected set (eps 1e-20), times `route_scale` 8.
- Expert: `softcap(g, 4) * sigmoid(g) * softcap(u, 25)` with `softcap(x, c) = c * tanh(x / c)`,
  then down, then the routed weight, accumulated in f32. `w13` holds gate and up interleaved
  on the output dimension.

## Stored quantization

- NVFP4 (layers 0-19): U8 payload, two e2m1 codes per byte with the even element in the low
  nibble; one e4m3 scale per 16 inputs per row; per-expert factor
  `amax / outer_scale_denominator`. The denominator read from bytes is 2688.
- MXFP8 (layers 20-23): e4m3 values with one u8 exponent per 32 inputs per row, value
  `2^(s - 127)`. `hf_quant_config.json` names `weight_block_shape [32, 32]`; the stored grid
  is one exponent per row per 32 inputs.
- `hf_quant_config.json` (`quant_method: inkling2`) is the `parameter_groups` form. The NVFP4
  group declares `required_activation_quantization` (`nvfp4_e2m1`, block 16, per-token amax,
  e4m3 scale, `fouroversix_mae` selection); the MXFP8 group declares `mxfp8_e4m3`, block 32,
  ue8m0. This declared program is the family's default numeric class. Both reference
  implementations run float activations instead, so neither is an oracle for it; where the
  activation quantizer sits (which expert inputs, before or after the routed weight) and the
  exact four-over-six rule come from the vLLM image.

## Drafter (`mtp.safetensors`)

54 BF16 tensors: 3 blocks (`q_proj`, `k_proj`, `v_proj`, `k_proj_ctx`, `v_proj_ctx`,
`o_proj`, `q_norm`, `k_norm`, `k_norm_ctx`, dense SwiGLU 9216, three norms), a Markov head
(`markov_w1 [201024, 256]`, `rnn_proj [768, 5120]`, `hidden_out [2304, 256]`), `fuse_weight
[4, 4]`, `mask_embed`, `norm`. Config: block 8, RoPE theta 1e6, local layers 0 and 2 with
window 1024, `markov_head_type: rnn_hidden`, `target_layer_ids [5, 23, 41, 48]`. Neither
reference implements it.

## Open facts

| # | Fact | How it closes |
|---|---|---|
| 1 | The exact activation-quantization program of the default class: which expert inputs are quantized, the per-token amax, the `fouroversix_mae` selection rule, rounding ties | extraction from the vLLM image; the float-activation class stays a bring-up oracle |
| 2 | The drafter program: the meaning of slots `[5, 23, 41, 48]` (48 read as the final hidden), `fuse_weight`, the RNN head, the `_ctx` projections | extraction from the vLLM image |
| 3 | Whether the sliding window counts the current token (transformers mask against llama.cpp `n_swa`) | tiny fixture run past the window |
| 4 | The sink form: llama.cpp subtracts the sink from the bias and uses a zero sink; on paper this equals the PR's form | tiny fixture |
| 5 | NVFP4 nibble order and e4m3 scale decode | one expert dequantized against the PR's op, bit for bit |
| 6 | Whether the MXFP8 scale grid is `[1, 32]` everywhere, as the shapes say, against the declared `[32, 32]` | the same check on a layer 20-23 expert |
| 7 | Whether single-term BF16 GEMM inputs hold the argmax | checkpoint parity on the GPU class |
| 8 | Top-k tie order in the router | pinned to index order; checked against the oracle |
