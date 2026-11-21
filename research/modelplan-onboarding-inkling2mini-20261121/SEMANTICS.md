# Inkling 2 Mini: program semantics and open facts

Read from `config.json`, `hf_quant_config.json`, the safetensors headers, llama.cpp at
`29b52d398b29` (`inkling2.cpp`, `inkling2-loop.cpp`), transformers PR 51050 at `6173a2f679ff`
and the model code of the vendor's serving image (the files listed in `sources.lock.json`).
The reference executor (memra-reference `inkling2`) runs this program on CPU; what its tests
prove is listed under [Proven by the reference tests](#proven-by-the-reference-tests). No tuned
surface runs any of it.

There are two numeric classes. The **declared** class is the program `hf_quant_config.json`
requires: the routed experts take quantized inputs. It is the default and the support target.
The **float-activation** class runs the experts on float inputs; it is a bring-up and oracle
step. Every section below is common to both classes except [MoE](#moe), which states each class
on its own: the two differ in the expert inputs, in where the routing weight is applied and in
how the routing weights are normalized.

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

Common to both classes:

- 128 experts, top 8, no shared expert, expert width 1152.
- Router scores in f32: `sigmoid(W x)`. Selection is top-k on score plus `gate.bias`; the
  routing weights are the selected raw sigmoids, normalized over the selected set and multiplied
  by `route_scale` 8.
- Expert activation: `softcap(g, 4) * sigmoid(g) * softcap(u, 25)` with
  `softcap(x, c) = c * tanh(x / c)`. `w13` holds gate and up; the combined output of the 8
  experts is accumulated in f32.

Declared class (read from the vendor image's code; the default):

- The token's row is quantized once (NVFP4 on layers 0-19, MXFP8 on 20-23) and that quantized
  row feeds the gate/up GEMM of each of its 8 experts.
- The routing weight multiplies the activation output BEFORE the down projection, and the
  product is quantized again, per routed row, before the down GEMM. With a quantized down input
  this placement changes the result, so it is part of the class.
- The routing weights are normalized with no epsilon. Selection ties go to the lower expert
  index.
- Everything outside the two expert GEMMs is BF16 with f32 accumulation.

Float-activation class (the transformers PR and llama.cpp; bring-up only):

- Float inputs against exactly dequantized expert weights.
- The routing weight multiplies AFTER the down projection.
- llama.cpp normalizes the routing weights with an epsilon of 1e-20.

In exact arithmetic the two placements of the routing weight are equal; under input
quantization or BF16 rounding they are not. Which rows of `w13` are gate and which are up is
proven below. The declared class's placement is implemented by the activation-quant reference
and pinned by its goldens, a port of the serving image's ops; it stays open against a serving
binary.

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

## Read from reference code, not yet proven

These come from the model code of the vendor's serving image (the only public source of the
declared numeric class and the drafter). Each one is typed in the plan as the reading a test has
to confirm, and becomes a program fact only when a test proves it. The MoE readings (where the
expert inputs are quantized, the routing weight before the down projection, no normalization
epsilon, ties to the lower index) are in [MoE](#moe) under the declared class; the plan fields
`routed_weight_before_down` and `gate_up_interleaved` carry two of them and say so in their
doc comments.

- Drafter taps `[5, 23, 41, 48]` count layer visits in execution order and read the INPUT of
  the visit (the running residual stream): as cache slots they are the inputs of slots 5, 29 and
  47, plus the final normed hidden. The drafter has a learned mask embedding, a fuse matrix over
  the taps, context K/V projections, a recurrent Markov head and no confidence head, so it is not
  the `DsparkPlan` program.

## Proven by the reference tests

Each line is a test of `crates/memra-reference/src/inkling2/tests.rs` against one module of the
float-activation implementation run alone in f64, with the misreading run against the same
record as a red arm. The table of tests, records and red arms is in
[`reference/README.md`](reference/README.md#facts-proven-one-test-each).

- `w13` rows are interleaved: row `2j` is gate `j`, row `2j + 1` is up `j`.
- The sliding window counts the current token: the query sees itself and `window - 1` positions
  back. A sliding layer's table ends its window; a full layer gives keys at or past its table a
  zero bias.
- The sink is one more logit per query and head in the softmax denominator, with no value row.
  Subtracting the sink from the bias with a zero sink is the same function only when it comes
  off the bias of every attended key, the zero-bias keys past the table included.
- Static short-convolution taps run oldest input first, so the last tap weights the current
  token. The static taps of a full layer's K and V convolutions run the other way, indexed by
  lag with the current token first, like the dynamic taps. The state of a convolution is its
  last three inputs.
- `tau` is 1 through position `floor - 1` and `1 + alpha * ln((pos + 1) / floor)` from position
  `floor` on, applied to Q and R of full layers after their clips.
- The message queries of core layer `i` are rows `2i` (q) and `2i + 1` (mlp) of
  `middle_cycle_cross_pass_queries` and rows `2i` (k) and `2i + 1` (v) of
  `middle_cycle_cross_pass_kv_queries`.
- NVFP4: the even element of a byte is its low nibble; the value is
  `val(code) * e4m3(scale) * amax / denominator`, equal to the PR's dequantizer bit for bit in
  f32. MXFP8: `e4m3(value) * 2^(scale - 127)`.
- On float inputs the routing weight before or after the down projection is one function up to
  rounding.
- The whole trunk: on a tiny checkpoint in the real layout the executor's logits equal the PR
  code's in f64 to the bit after rounding to f32, as one prefill chunk and under three schedules
  that split the request at and around the window, the extent and the floor
  ([`reference/README.md`](reference/README.md#what-the-tiny-gate-runs)).

Router ties are not provable against the PR, whose `torch.topk` leaves the order unspecified.
The executor takes the lower expert index, the serving image's rule.

## Closed from the bytes

- **The MXFP8 scale grid is one exponent per row per 32 inputs.** `hf_quant_config.json`
  declares `weight_block_shape [32, 32]` for the MXFP8 group. If the stored planes were a 32x32
  grid repeated over rows, every aligned group of 32 rows would hold 32 identical rows.
  [`mxfp8_scale_grid.py`](mxfp8_scale_grid.py) reads all 8 scale planes of layers 20-23:
  none of the 73,728 aligned groups is uniform. In the four `w13` planes every one of the 31
  other rows of every group differs from the group's first row; in the four `w2` planes all but
  5 of 1,142,784 do ([`mxfp8-scale-grid.tsv`](mxfp8-scale-grid.tsv); exponents 113 to 118). So the stored scales have no 32-row block, and the program dequantizes with the
  stored per-row exponents. The reader types this as one explicit reading
  (`WeightBlock::Mxfp8Declared32x32StoredRow32`): `[32, 32]` is accepted only on an MXFP8 group
  and only with that meaning, any other shape is refused, and the census pins the stored grid on
  its own (`MXFP8_E4M3`, block 1x32). What the first 32 of the declaration meant to its writer
  is not known and does not enter the program.

## Open facts

| # | Fact | How it closes |
|---|---|---|
| 1 | The exact activation-quantization program of the default class: which expert inputs are quantized, the per-token amax, the `fouroversix_mae` selection rule, rounding ties | extraction from the vLLM image; the float-activation class stays a bring-up oracle |
| 2 | The drafter program, read from the image's code (above) and not yet typed as an executable plan | a drafter plan with its own reference implementation and oracle |
| 3 | closed: see "Proven by the reference tests" (the window edge) | |
| 4 | closed: see "Proven by the reference tests" (the sink form, and when the two spellings are equal) | |
| 5 | closed: see "Proven by the reference tests" (NVFP4 nibble order and e4m3 decode) | |
| 6 | closed: see "Closed from the bytes" (the MXFP8 scale grid) | |
| 7 | Whether single-term BF16 GEMM inputs hold the argmax | checkpoint parity on the GPU class |
| 8 | Top-k tie order in the router, read as lower index first, and the declared class's normalization without an epsilon. The PR cannot close it: its `torch.topk` leaves tie order unspecified | captured vectors of the declared class |
| 9 | closed: see "Proven by the reference tests" (the interleave direction) | |
| 10 | Whether the routing weight enters before the down projection in the declared class | captured vectors of the declared class |
