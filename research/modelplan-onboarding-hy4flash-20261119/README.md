# Hy4-flash NVFP4 onboarding, step 0: sources and semantics (2026-11-19)

Verdict: the artifact, its BF16 source and the two reference implementations are pinned and each
pin resolves to its recorded revision (`sources/verify-pins.jsonl`). The checkpoint has 2,429
header tensors in 50 shards, 161,078,199,806 tensor bytes. The Engram layer-index discrepancy is a
naming convention, not a fault: `engram_layer_ids` is 1-based and the tensors sit at the 0-based
decoder layer `id - 1`; both reference implementations and the table sizes agree. No memra code
changes in this step, no pack exists yet, and no GPU was used.

## Pins

`sources.lock.json`; `python3 -I sources/verify_pins.py sources.lock.json` re-checks every entry
against the HF and GitHub APIs and exits 1 on a mismatch.

| id | pin | role |
|---|---|---|
| checkpoint | `Avifenesh/Hy4-flash-NVFP4@f9eac06cf022cfd3dc2e8ed423922466b2adcd87` | the artifact onboarded here |
| bf16-source | `tencent/Hy4-flash@d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959` | source of every checkpoint byte |
| transformers-pr-50903 | head `b865850d05326cf20e306b31f14b0027de802da3` (open) | semantic reference, CPU oracle for CheckpointParity |
| vllm-pr-68296 | head `8cbb80efa3974d10bb09c87580627bd874a3606b` (merged as `b067b6f6fa`) | second reading of the same program |

The lock also records the git blob of every reference file read for this note, and the sha256 of
the checkpoint's `config.json`, `hf_quant_config.json`, `quant_config.json`, index, tokenizer and
template. The checkpoint `config.json` equals the source's plus one `quantization_config` block.

## Census

`census/header-census.tsv` is every header tensor (name, dtype, shape, shard, bytes), read from
the pinned revision by HTTP range requests (`sources/hf_headers.py`); `census/shards.tsv` holds
each shard's header length and sha256. sha256 of the census file:
`26526e1f45a0bbcb648a1ceeaf950592dadc53b15b4647aa262fb907a1e67c72`.

| class | tensors | bytes |
|---|---:|---:|
| routed experts (layers 1-41, stacked `[512, ...]`) | 369 | 83,575,953,408 |
| Engram (4 layers) | 168 | 66,191,773,024 |
| trunk, everything else | 1,307 | 8,172,682,528 |
| embedding, head, final norm, exit hyper-head | 6 | 1,484,986,388 |
| vision tower (`vit.*`) | 441 | 933,258,720 |
| DFly2 drafter (`model.dfly_layers.*`) | 138 | 719,545,738 |

By dtype: 1,332 BF16, 669 F32, 283 F8_E4M3, 141 U8, 4 I32.

## Architecture (config.json `text_config`, transformers `modeling_hy_v4_flash.py`)

- 42 decoder layers, hidden 3072, vocab 120,832, untied F32-computed head (`enable_lm_head_fp32`),
  `max_position_embeddings` 1,048,576.
- Attention is gated MLA on every layer: `q_a_proj` 3072 to 1024, `q_a_layernorm`, `q_b_proj` to
  heads x (192 nope + 64 rope); `kv_a_proj_with_mqa` 3072 to 512 + 64, `kv_a_layernorm`,
  `kv_b_proj` 512 to heads x (192 + 256 value). Softmax scale `256^-0.5`, a learned per-head sink
  in the denominator (`learnable_sink_param`, F32), an elementwise sigmoid output gate
  (`linear_gate`, heads x 256 from the layer input), `o_proj`, then `post_proj_norm` (RMS).
  RoPE rotates the trailing 64 dims as adjacent pairs; it is M-RoPE (`mrope_section`
  [12, 10, 10], interleaved), which reduces to plain RoPE on text because all three axes read the
  same position.
- Two layer types (`swa_pattern`, `layer_types`): 12 indexed layers (0, 1, 5, 9, ..., 41) with
  32 heads and `rope_theta` 1e7, and 30 sliding-window layers with 64 heads, window 512
  (`window_size` [511, 0]) and `rope_theta` 1e4. The head count is per layer type, so
  `kv_b_proj` is `[14336, 512]` on indexed layers and `[28672, 512]` on sliding ones.
- The indexed layers run the compressed indexer (`compressed_indexer.*`): 16 heads of 128 over
  `q_resid`, token keys `token_key_norm(token_key_projection(x))` cached per token, pooled in
  groups of 4 by an F32 softmax gate (`key_value_projection`, `key_gate_projection`,
  `intra_group_ape`, `key_norm`), RoPE on the trailing 64 dims at the query position and at each
  group's first position, ReLU scores weighted by `head_weight_projection`; a query keeps the 511
  best closed groups plus its own (2,048 tokens). With at most 2,048 keys the layer attends densely.
- Residual: 4 hyper-connection streams without Sinkhorn (`enable_ihc`). Per sublayer,
  `hc_pre.hc_fn [8, 4*3072]` mixes the RMS-normalized flat streams into 4 pre and 4 post logits;
  `pre = sigmoid(l*scale0 + base) + eps` collapses the streams, `post = 2.0 * sigmoid(...) + eps`
  (`hc_magnitude`) writes the sublayer output back to each stream, all in F32. The exit collapses
  with `hc_head` (4 weights, `sigmoid(mix*scale + base) + eps`) before `model.norm`.
- Sandwich norms: `input_layernorm` before attention, `post_attention_layernorm` before the MLP,
  `post_proj_norm` after `o_proj`, and an RMS norm on the MLP output (`mlp.post_fc2_norm` on the
  dense layer 0, `mlp.combine_output_norm` on MoE layers).
- MLP: layer 0 dense SwiGLU 13,824. Layers 1-41: sigmoid router over 512 experts with
  `e_score_correction_bias` steering selection only, top-16, renormalized, x 3.998. Routed experts
  run in a 1,536 latent (`routed_expert_down_proj` 3072 to 1536, experts 1536 to 1536 to 1536,
  `routed_expert_up_proj` back to 3072); SwiGLU clamps gate from above and up to +-10 before
  `silu(gate) * up`; one shared SwiGLU expert (1,536) on the full-width input.
- DFly2 drafter: 5 blocks (`model.dfly_layers.0-4`), mask token 120009, reads the target KV and
  indexer top-k of layers 25/29/33/37/41 (0-based; all five are indexed layers).

## Engram layer index: config [2, 12, 22, 32], tensors at layers 1, 11, 21, 31

Both are right. `engram_layer_ids` names 1-based layers; the module lives in the 0-based decoder
layer `id - 1` and runs there before that layer's attention hyper-connection.

- transformers `HYV4FlashDecoderLayer.__init__`: `if layer_idx + 1 in config.engram_layer_ids:
  self.engram_layer = HYV4FlashEngramLayer(config, layer_idx + 1)`.
- vLLM `model.py`: `self.layers[layer_id - 1].engram_layer = FlashEngram(..., layer_id, ...)`,
  and the loader maps `model.layers.N.engram_layer.*` to `layer_id = N + 1`, refusing a tensor
  whose `N + 1` is not in `engram_layer_ids`.
- The 1-based id is semantic, not a label: it seeds the hash multipliers
  (`np.random.default_rng(engram_seed + 10007 * layer_id)`, both implementations) and keys
  `engram_layer_head_vocab_sizes`. The data agrees: the table of `layers.1` has 164,994,380 rows,
  the sum of the 16 prime moduli under key `"2"`; `layers.11`, `.21`, `.31` match keys `"12"`,
  `"22"`, `"32"` (164,999,008, 165,004,086, 165,008,942).

A pack that seeds or sizes with the 0-based index hashes every n-gram into the wrong buckets
without any shape error, so PR 1 carries a negative test for exactly that offset.

## The in-checkpoint lookup_table

`model.layers.{1,11,21,31}.engram_layer.lookup_table`, I32 `[120818]` (`engram_lookup_table_size`).
It maps token ids to a normalized vocabulary before hashing. transformers declares it as a
persistent buffer initialized to `arange`, so the checkpoint value replaces the default; vLLM
loads it and refuses unless all Engram layers carry the same map.
`sources/lookup_tables.py` range-reads all four (`sources/lookup-tables-nvfp4.json`, and the BF16
source in `sources/lookup-tables-bf16-source.json`): the four are byte-identical and identical to
the source (sha256 `87c7b2e1c2fc...`), values 0..92,619 (92,620 distinct, the compressed
vocabulary), an identity prefix for ids 0-63 (id 64 is the first that maps elsewhere), and the
pad id 120002 maps to 91,804. The table is load-bearing twice: it maps the ids, and it bounds the
hash multipliers. Both implementations draw them as `integers(0, int64::MAX // V // 2) * 2 + 1`
with `V = max(lookup_table) + 1`, 92,620 here (transformers `_init_weights` redraws them after
the table loads; vLLM `load_lookup`), not with the table length 120,818. The `arange` default
would hash a different program on both counts.

The four Engram RMS norms (`lookup_per_head_norms`, `norm_key`, `norm_query`, the weightless
`unified_rmsnorm`) are `torch.nn.RMSNorm` with `eps=None`. Corrected in step 5: this step read
that as `torch.finfo(input dtype).eps`, 2^-7 on BF16 activations, from the module's
documentation. ATen's default is the epsilon of the op-math type instead, `f32::EPSILON`
(2^-23) for BF16 inputs, which is also the value vLLM writes out (`finfo(float32).eps`), so the
two implementations agree. memra carries the rule in the plan rather than a number.

Ids 120818-120831 (14 tokenizer placeholders, `<｜hy_place▁holder▁no▁800｜>` to `..813｜>`) lie
past the table. transformers reads them as the pad id's entry; vLLM extends the map with the
identity, so the two implementations differ on those 14 ids. memra follows transformers, the
CheckpointParity oracle.

## Quantization classes (hf_quant_config.json, ModelOpt schema)

`quant_algo: MIXED_PRECISION`, 279 `quantized_layers` entries:

| class | modules | tensors |
|---|---|---|
| `W4A16_NVFP4`, group 16 (141) | `mlp.experts.{gate,up,down}_proj` of layers 1-41, DFly2 `mlp.*` and `hidden_correction.*` | `weight` U8 (e2m1 pairs), `weight_scale` F8_E4M3 per 16, `weight_scale_2` F32 (`[512]` per expert when stacked, `[]` otherwise) |
| `FP8_PB_WO`, group 128 (138) | `self_attn.{q_b_proj,o_proj,linear_gate}` of layers 1-41 and DFly2 0-4 | `weight` F8_E4M3, `weight_scale_inv` F32 `[ceil(out/128), ceil(in/128)]`, multiplied in |
| FP8 per row (in neither list) | the four Engram tables | `weight` F8_E4M3 `[rows, 96]`, `weight_scale` F32 `[rows]` |

`exclude_modules` lists the BF16 and F32 modules (18 wildcards, layer 0 entirely). Two details a
reader must not normalize away:

- gate and up are separate tensors, each with its own `weight_scale_2` (the source's fused
  `gate_up_proj` was split row-wise), so a loader that fuses them must keep two per-expert scales.
- The Engram tables are FP8 with one F32 scale per row, which no ModelOpt `quant_algo` expresses;
  the configs say so in a `comment` field. transformers reads the BF16 source's table through a
  per-row FP8 round trip (`_fp8_round_trip`, `engram_fp8_embedding: true`); whether that round
  trip of the BF16 row equals this checkpoint's stored `q * scale` bit for bit is a
  CheckpointParity question, not settled here.

No activation, KV-cache or input scales exist: activations stay BF16 (W4A16).

## Scope decisions (owner rule: technical calls are the lane's; recorded 2026-11-19)

1. Topology: one GB10 first, Engram tables disk-backed (161.08 GB of tensors do not fit 128 GB
   unified memory resident); a two-node pipeline only after a measured need.
2. Served context: 256k first; 1M later if memory allows. KV is 576 latent values per token per
   layer plus 128 index-key values on the 12 indexed layers.
3. Vision: out of scope. The tower's 441 tensors stay contract-owned (a missing or reshaped one
   fails the census), the plan does not execute them, and requests with image parts are refused.
4. Format: the artifact stays weight-only NVFP4 (W4A16); no requantization in this lane.

## Next

PR 1: `Arch::HyV4Flash`, its config and the `hy_v4_flash` pack: typed plan, tensor contract over
all 2,429 tensors (FP8 per-row Engram tables included) and the per-layer `hf_quant_config.json`
reader, with the Config and TensorCensus gates on the real headers.

# Step 1: config, pack and census (2026-11-20)

Verdict: the new `hy_v4_flash` pack parses the pinned config by name, compiles a typed plan for
it and binds every one of the 2,429 header tensors. Config, TokenizerTemplate and TensorCensus pass
at inspect on the pinned revision, read over HTTP with no weights downloaded. The pack's support
stays none: no executor runs the Hy4-flash operations, so TinyParity, CheckpointParity,
RewriteParity and Serve are pending. No GPU was used.

## Inspect

```bash
memra model inspect Avifenesh/Hy4-flash-NVFP4@f9eac06cf022cfd3dc2e8ed423922466b2adcd87 \
  --against hy_v4_flash --out inspect
# family=hy_v4_flash tensors=2005 artifacts=inspect
```

`inspect/artifact.lock`: `config_sha256=c88f950d8050...` (the `config.json` bytes, also the pack
fixture), `census_sha256=406b097584513b88...` (`inspect/tensor-census.tsv.gz` unpacks to the
hashed text), `plan_sha256=a5c4fdf4264eb087...` (step 4 adds the indexer's tie order to the plan
and refreshes `inspect/`: now `a4a43c267029e5b9...`), `binding=passed`, `tokenizer=passed`
(`dialect=artifact`: the template hash only; the hy_v4 dialect and its parity are the next step).

The census has 2,005 rows: each quantized weight row carries its scale planes (424 of the 2,429
header tensors).

| rows | bytes | storage |
|---:|---:|---|
| 141 | 83,841,374,280 | NVFP4: U8 e2m1 pairs, F8_E4M3 `weight_scale` per 16, F32 `weight_scale_2` |
| 4 | 66,000,641,600 | FP8 e4m3 with one F32 `weight_scale` per row: the Engram tables |
| 1,332 | 6,735,494,946 | BF16 |
| 138 | 4,463,828,992 | FP8 e4m3 with F32 `weight_scale_inv` per 128x128 block |
| 386 | 34,926,900 | F32: hyper-connection parameters, router biases, trunk sinks, indexer pooling |
| 4 | 1,933,088 | I32: the Engram lookup tables |

## What the pack does

- `config::hy_v4_flash` reads every `text_config` field the transformers program reads, by name,
  and refuses a switch that selects another program (Sinkhorn hyper-connections, a softmax router,
  a fine indexer pass, an output or gate LoRA, an isolated U+FFFD namespace, a window other than
  `sliding_window - 1` back) or a missing field. `layer_types` and `swa_pattern` must agree, the
  indexer aliases must equal the canonical `index_*` fields, and the drafter's target layers must
  be indexed layers. `engram_lookup_table_size` is read when declared; the artifact's config does
  not declare it, so the configuration class default (120818) applies and the census checks the
  tensor's shape against it. `o_lora_rank: null` and `gate_lora_rank: null` read as absent for
  this family only.
- `modelopt_quant` reads the per-layer ModelOpt map from `hf_quant_config.json` and from
  `config.json`'s `quantization_config`, both through the strict JSON decoder (a key listed twice
  refuses). The pack compiles from the `config.json` copy; `memra model inspect` (local and remote
  sources) refuses unless the artifact's `hf_quant_config.json` is present and is the same map.
  Only `W4A16_NVFP4` (group 16) and `FP8_PB_WO` (group 128) are accepted. The census is derived
  from the map: a quantizable module the map neither quantizes nor excludes, a map entry no module
  consumes, or a module the plan keeps BF16 (layer 0's dense MLP included) that the map quantizes
  refuses the config.
- The plan: `AttentionPlan::HyV4` (gated MLA, indexed with an `IndexPoolPlan` or sliding with its
  window, head geometry per layer type), `MlpPlan::LatentMoe` (dense SwiGLU on layer 0),
  whose routed experts run pre-clamped SwiGLU and whose shared expert runs plain SwiGLU,
  `ResidualTopology::HyV4` (independent hyper-connections, the MLP output norm, Engram on the
  layer `id - 1`, seeded with the 1-based id, multipliers bounded by `max(lookup_table) + 1`, norm
  epsilon from the input dtype), `StatePlan::HyV4`, and `ModelPlan::hy_v4_flash`
  (stream copy, `hc_head` exit, F32 head). Nine new `OperationKind`s, all registry rows with no
  surface; `ModelPlan::compile`, the reference executor, the hybrid engine and the generic
  parallel loader refuse the family by name.
- The census: the safetensors reader now labels an FP8 weight with one F32 `weight_scale` per row
  as block `[1, in]` (`QuantConstraint::Fp8RowF32`) instead of the 128x128 default, and reads a
  standalone I32 tensor (`QuantConstraint::I32`), which binds only through an explicit `I32`
  requirement, never as a canonical weight. Every other family's census is unchanged.
- The DFly2 drafter and the vision tower are contract-owned and not in the text plan. The DFly2
  MLP width (9,216) is not in the config; the vendor runtime takes it from the checkpoint, so the
  pack pins it as a constant that the census checks.

## Tests

`crates/memra-gguf/src/model_packs/hy_v4_flash/tests.rs`, on the pinned `config.json`,
`hf_quant_config.json` and a header-only census fixture (`fixtures/census-classes.tsv`, generated
by `census_classes.py` from `census/header-census.tsv`, which it must expand back to exactly):

- geometry of the pinned config; both ModelOpt maps agree (141 + 138 modules);
- the typed plan per layer and its operation list;
- the census gate: all 2,429 tensors derived with names, dtypes and shapes, every header byte in
  one census row, the contract binds, the loader boundary (`bind_census`) accepts it;
- negative: up's `weight_scale_2` removed (auxiliary mismatch), one fused `gate_up_proj` bank
  (missing tensors), a map that adds a fused bank beside gate and up (names a module the program
  lacks) or replaces them by it (gate unlisted);
- negative: the Engram ids read 0-based (refused at the plan through layer 0's exclusion
  wildcard, and, with that wildcard narrowed, refused by the real headers while the narrowed map
  alone binds) or one layer late (the real headers refuse it), and shifted ids against the true
  size table (refused by name);
- negative: an FP8 group other than 128, a quantized layer-0 dense MLP, an `hf_quant_config.json`
  that is absent, differs on one module or repeats a key;
- negative: one head count for both layer types (shape mismatch on the real headers), and a
  `swa_pattern` that contradicts `layer_types`;
- every program switch above refuses by name.

# Open questions

| question | owner step |
|---|---|
| Resolved in step 10. DFly2 dense MLP width: 9,216 in the checkpoint, not in the config. Answer: the width is the checkpoint's. The pinned vLLM program builds the drafter's MLPs from the gate/up/down shapes it reads (`HYV4DFlyModel.materialize_ffn`), refuses a declared `dfly_intermediate_size` that disagrees and widths that differ between layers; the 13,824 default of `_derive_dfly_config` is read only for dummy weights. All five layers are 9,216 wide in the NVFP4 artifact and in its BF16 source. | was PR 10 (DFly2) |
| Resolved in step 5. Whether transformers' per-row FP8 round trip of a BF16 Engram row equals this checkpoint's stored `q * weight_scale` bit for bit. Answer: yes in F32, on all 8,192 sampled rows, because the artifact was written by the same rule; no under BF16 activations, where transformers computes the scale in BF16 (`engram/`). memra reads `q * weight_scale` and runs no second round trip. | was CheckpointParity |
| Resolved in step 5. The Engram norm epsilon. Answer: `nn.RMSNorm(eps=None)` adds ATen's op-math epsilon, `f32::EPSILON` (2^-23) on BF16 activations, not `finfo(bfloat16).eps`; vLLM's explicit F32 epsilon is the same value. The plan says `HyV4EngramNormEpsilon::OpMathF32`. | was CheckpointParity |

# Step 2: tokenizer, chat template and tool-call parser (2026-11-20)

Verdict: memra renders the artifact's `chat_template.jinja` byte for byte as transformers 5.23.0
does on all 22 render fixtures and refuses the two invalid efforts as the template does, its
tokenizer gives the `tokenizers` library's ids on every render and on a 132-case raw corpus in
both encode modes, and its Hy4 tool-call parser gives the vLLM parser's calls and typed
arguments on 19 of 24 parse fixtures; the other 5 are recorded divergences, each checked as
memra's own behavior with its reason. No GPU was used.

## Oracles and fixtures

`chat/gen_fixtures.py` (run under `uv` with transformers 5.23.0, tokenizers 0.23.3, jinja2 3.1.6)
writes everything under `chat/`:

- `artifact/`: `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` of the pinned
  revision (sha256 as in `sources.lock.json`; the script refuses other bytes).
- `fixtures/renders.json`: 24 cases through `render_jinja_template`, the call
  `apply_chat_template` makes: the three efforts (`no_think`, `high`, `max`) and the default
  (`high`), system and no system, empty turns, reasoning history before and after the last user
  turn, `reasoning` over `reasoning_content`, a later system turn, tools with and without a
  system (the synthesized system turn), `defer_loading` and `strict`, tool calls with typed
  arguments, tool responses reordered by call id and left in order when the ids do not match,
  preserved thinking with tools, `no_think` with history reasoning, unicode and escapes, and the
  two efforts the template raises on (`low`, `medium`). Each render carries its ids
  (`add_special_tokens=False`).
- `corpus.tsv` / `ref-ids.tsv`: the deepseek-v3 split's adversarial raw cases from
  `research/step-sku-20260807/corpus.tsv`, seven Hy4 control-token cases and every render, ids in
  both modes.
- `fixtures/parses.json`: 24 completions through vLLM's `HYV4ToolExtractor.extract_tool_calls`
  (strict, as `HYV4ToolParser` builds it; the source is fetched at the pinned PR head and its git
  blob recorded, `295456c7699d...`), with its vLLM-only imports stubbed. Where memra differs, the
  case carries a `memra` record: what memra returns and why.

## What changed

- `memra-tokenizer`: `chat::apply_hy4_template` reproduces the template; `template_is_hy4`
  detects it by its `<｜hy_start{}｜>` and `<｜reasoning_mode{}｜>` format strings, and both
  render entries dispatch to it before the qwen arm (the tools prompt carries `<tools>`).
  `hy4_reasoning_effort` refuses what the template refuses; a thinking-off request renders
  `no_think` whatever effort it names (as the HY3 arm does), an explicit effort wins otherwise,
  and the default is the template's `high`. Tool-definition keys are written raw, as the
  template's `tool_to_json` writes them. Image parts are refused: this engine runs no Hy4 vision
  program.
- `memra-server`: `ToolStreamParser::hy4`, the HY3 seam and call framing with the Hy4 call body
  (name up to the first `<arg_key:opensource>`, no `<tool_sep:opensource>`), with vLLM's strict
  rules: any whitespace between calls and between argument pairs, a key's close directly
  followed by its value's open, a repeated key keeping its first position with the last value.
  HY3 parsing is unchanged. Not wired to a serve route yet: Hy4-flash has none.
- Tokenizer: no change. The pre-tokenizer is the deepseek-v3 split memra already implements, and
  the parity gates pass on the existing code.

Recorded differences from vLLM's parser, each checked by the parse gate:

- text after the closing `</tool_calls:opensource>` stays content in memra (vLLM drops it);
- a bare `<tool_call:opensource>` without the `<tool_calls:opensource>` wrapper is content in
  memra (vLLM restores the wrapper);
- memra parses every `<tool_calls:opensource>` block (vLLM the first);
- a name that contains a tag (a HY3 `<tool_sep:opensource>` body) is refused by memra (vLLM
  keeps it in the name);
- argument coercion follows memra's request schema, which reads a plain-string `type`: list and
  `anyOf` types coerce nothing, and a declared non-string value is parsed as JSON only (vLLM
  also reads list and `anyOf` types and falls back to `ast.literal_eval`). The schema plumbing
  belongs to the serve-route step.

Not covered by the renderer: a tool message keyed by `id` rather than `tool_call_id` (memra's
turn carries `tool_call_id` only), `tool_search_output` content parts, and numbers whose spelling
`json.dumps` would normalize (`1e-3` stays as written; this predates Hy4 and is shared with the
GLM and Gemma arms). The render harness picks `reasoning` over `reasoning_content` as the
template does; memra's request type carries `reasoning_content` only.

# Step 3: reference attention, residual and MoE operations (2026-11-20)

Verdict: memra-reference implements the Hy4-flash gated MLA attention, the independent
hyper-connection and its write-back, the dense SwiGLU MLP, the latent MoE with its router, the
exit hyper-head and the decoder layer that composes them, in f64, from the pack's typed layer
plan. On a tiny 4-layer config every module matches the pinned transformers code run in float64
to at most 3.8e-15, and the 4-layer stack through `hc_head` and the final norm matches the
model's last hidden state to 6.8e-15 (`modules/gate.log`). Five red arms each move the output
past 1e-6. The pooled indexer and Engram are not in this step; the indexed layers run below
the indexer budget, where the vendor program attends densely. No GPU was used.

## Oracle

`modules/gen_modules.py` runs transformers PR #50903, installed
from b865850d0532 (the modeling and configuration files checked against their git blobs) on a
tiny config (hidden 16, 2 indexed layers with 2 heads and 2 sliding layers with 4 heads and a
window of 3, q_lora 12, kv_lora 8, nope 6, rope 8, value 5, 6 experts top-2 in a latent of 8,
4 streams; 7 tokens, under the indexer budget of 8) in float64 and records every module's
inputs and outputs and all weights to `modules/modules.json`.

The pinned code casts to float32 inside RMSNorm, RoPE, the router and the hyper-connections.
The oracle process maps the modeling module's `torch.float32` and `torch.float` and
`Tensor.float` to float64, so both sides run in float64 and the bound can be 1e-12 relative to
each output's magnitude. Without the `torch.float` mapping, RoPE's inverse frequencies stayed in
float32 and the attention differed by 1.6e-9. The epsilons are powers of two (2^-17, 2^-20)
because the typed plan carries them as f32: with 1e-5 the f32-rounded plan epsilon moved the
norms by 1.3e-12. Both were measurement artefacts of the oracle, not program differences.

## What the gate covers

`crates/memra-reference/tests/hy_v4_flash_modules.rs`, compiled from
`model_packs::hy_v4_flash::layer_plan` on the tiny config:

- attention on both layer types (sinks, elementwise gate, post-projection norm, interleaved
  RoPE at each type's theta; text positions on all three M-RoPE axes reduce to plain RoPE);
- both hyper-connections per layer (pre and post weights), both norms, `hc_head`;
- the latent MoE on the three sparse layers, its banks split from the module's fused
  `gate_up_proj` the way the artifact stores them;
- each decoder layer, and the whole stack to the last hidden state;
- red arms: the sink dropped, a sliding layer read as full attention, the write-back magnitude
  set to 1, gate and up swapped, the router's correction bias ignored.

# Step 4: the pooled indexer (2026-11-20)

Verdict: memra-reference implements `HYV4FlashCompressedIndexer` in f64 from the typed
`IndexPoolPlan`: token keys, the gated F32 pooling, the RoPE'd group logits, the group top-k and
the recalled-key mask, wired into the attention of the indexed layers. Past the budget, on a tiny
config with 41 tokens, the reference selects exactly the oracle's groups and keys for every query
of both indexed layers, ties included, and matches the pinned transformers code in float64 to at
most 9.8e-15 on every module and the 4-layer stack (`indexpool/gate.log`). Four red arms move the
attention output by 1.2 to 3.9. No GPU was used.

## The program

Per indexed layer, with `x` the attention input and `q_resid = q_a_layernorm(q_a_proj(x))`:

- token keys `token_key_norm(token_key_projection(x))` (`[tokens, 128]` on the artifact), the
  part the cache keeps per token;
- each complete group of 4 keys pools to `key_norm(sum_slot value(k) * softmax_slot(gate(k) +
  intra_group_ape))`: `key_value_projection` and `key_gate_projection` are F32 `[128, 128]`,
  `intra_group_ape` F32 `[4, 128]`, the softmax runs over the 4 slots per dim; a trailing partial
  group is not pooled;
- queries `query_projection(q_resid)` (16 heads of 128) and the pooled keys rotate their trailing
  64 dims (the indexed layers' RoPE, theta 1e7), queries at their position, a group at its first
  token's;
- the group logit is `sum_h hw_h * relu(q_h . k)` with `hw = head_weight_projection(x) * 16^-0.5 *
  128^-0.5`; a group that has not closed before the query is `-inf`;
- a query keeps the `2048 / 4 - 1 = 511` best groups plus its own and attends to their tokens
  under the causal mask. No fine (per-token) selection follows (`skip_fine_select`, refused by
  the config). At or below 2,048 keys the layer attends densely and the indexer is skipped.

## Tie order

The ReLU makes exact ties common: a closed group no head scores above zero has logit 0 (52 and
31 such query-group pairs on the two tiny layers). Which tied groups the top-k keeps changes the
attended keys, so the order is part of the program and the plan carries it
(`IndexPoolPlan::ties`, `GroupTieOrder::LowerIndexFirst`; the plan hash moves, `inspect/` is
refreshed, nothing else in it changes).

The rule is CUDA `torch.topk`'s, where the vendor program runs. PyTorch v2.15.0
(`31a77fc9c0a7`), `aten/src/ATen/native/cuda/TensorTopK.cu`: both the single-block
(`sbtopk::gatherTopK`) and the multi-block (`mbtopk::gatherTopK`) kernel first write every value
strictly above the k-th value, then the values equal to it in ascending input index until k are
written (a prefix count per warp or per block keeps that order). Values compare by
`TopKTypeConfig<float>::convert` (`SortingRadixSelect.cuh`), an order-preserving integer key. `TensorTopK.cpp` sorts the
result afterwards when `sorted=True`, which reorders but does not change the set. Values compare
by their radix key, so `-0.0` ranks below `0.0`; within one query all zero logits carry the same
sign (the head weights are per query), which the oracle asserts. The full-sort and warp
merge-sort paths are ROCm-only (`should_use_sort` returns false on CUDA). CPU `topk` leaves ties
unspecified.

## Oracle

`indexpool/gen_indexpool.py` runs the pinned code (blobs checked) on the step 3 tiny config with
`index_head_dim` 12 (8 rope dims), `index_topk` 12 and 41 tokens: past the budget, 10 closed
groups, 2 kept plus the query's own. It maps float32 to float64 as in step 3 and replaces
`topk` in the process by a stable descending sort, which selects the CUDA set. It records the
token keys, group keys, group logits, selected groups and the visible-key mask of each indexed
layer, every module's inputs and outputs, and the last hidden state (`indexpool/indexpool.json`,
sha256 `3805658873f1eacb...`, reproduced bit for bit with `OMP_NUM_THREADS=1`; `indexpool/gen.log`).
Other thread counts change the summation order of a few BLAS reductions: at 24 threads 16
values move by at most 2.2e-16, the selection does not. The `topk` replacement is process-wide,
so the MoE router's top-k sees it too; it selects the same set there (no ties among router
scores) and matches the lower-index rule `latent_moe` already follows. Under
the flipped tie order 5 and 4 of the 41 queries select different closed groups; under the CPU
kernel 3 and 3.

## What the gate covers

`crates/memra-reference/tests/hy_v4_flash_indexpool.rs` (the step 3 helpers moved to
`tests/hy_v4_flash_common/`):

- token keys, pooled group keys and the finite group logits to 1e-12, the `-inf` pattern exact;
- the selected groups per query and the visible-key mask exact; 381 of 861 causal keys stay
  visible, and 6 and 4 queries tie at their last kept group;
- attention on all four layers, each decoder layer, and the stack through `hc_head` and the
  final norm;
- red arms, each moving the indexed attention by more than 1e-6 (layer 0, layer 1): pool size 2
  (3.8, 2.2), the `intra_group_ape` omitted (2.8, 1.2), the flipped tie order (1.5, 2.0) and
  dense causal attention past the budget (3.9, 2.1). The last is step 3's program, which refused
  to run past the budget.
- the config refuses a zero budget (`index_topk` 0 passes the divisibility and alias checks but
  holds not even the query's own group).

The step 3 gate runs unchanged in substance: its indexed layers stay below the budget, and an
indexed layer now carries its indexer weights.

## For the whole-trunk and checkpoint steps

- The selection is discrete. In f64 against f64 it is exact; the vendor program computes the
  pooling in F32 and the scores from BF16 projections, so near-tied non-zero logits can flip
  between a BF16 run and memra's, and a flipped group moves the attention by O(1) on that query.
  Compare selections first and attribute output differences to selection flips before reading
  any band.
- Exact zero ties do not depend on precision: they follow the tie order, which every memra
  surface must implement as the plan states.

# Step 5: the Engram memory (2026-11-20)

Verdict: memra hashes the artifact's n-grams bit for bit as the pinned transformers code does,
on the artifact's own config and lookup table: the NumPy multipliers of all four Engram layers,
the n-gram tokens, the mask and every hash id. memra-reference implements the float half of
`HYV4FlashEngramLayer` in f64 from the stored FP8 rows and matches the pinned code to 3.3e-16 on
the module and 2.0e-14 on a 4-layer stack with one Engram layer past the indexer budget
(`engram/gate.log`). Both open questions of step 0 on Engram are answered from the pinned code
and measured; one corrects the plan. No GPU was used.

## The program

Per Engram layer (config id `L`, decoder layer `L - 1`), from the plan and the config:

- Token ids map through the checkpoint's `lookup_table` (120,818 entries). The 14 placeholder
  ids 120818-120831 past it read the pad id's entry, as transformers does (vLLM extends the map
  with the identity).
- The multipliers are NumPy `default_rng(engram_seed + 10007 * L).integers(0, int64::MAX // V //
  2, size=3) * 2 + 1` with `V = max(lookup_table) + 1 = 92,620` (`dsv41_forward`'s PCG64 port,
  now seeded by a caller: `numpy_odd_multipliers`).
- Each position takes the 3 tokens ending there; a slot before the start of its document reads
  the pad id. Orders 2 and 3 XOR the first 2 or 3 products, and each order's 8 heads take the
  value modulo their own prime (`engram_layer_head_vocab_sizes`, 16 primes per layer summing to
  the table's rows). Row = hash + the head's offset in the one table.
- A row is its 96 e4m3 codes times its F32 `weight_scale`; each of the 16 heads is normalized by
  its own `lookup_per_head_norms`, and the 1,536-wide memory feeds 4 per-stream key projections
  (`key_projs`, `norm_key`, against `norm_query` of that stream) and one value projection
  (`value_proj`, the weightless `unified_rmsnorm`, `unified_pre_gamma`). Stream `s` gets
  `sigmoid(sqrt(max(|g|, 1e-6)) * sign(g)) * v + unified_per_hc_scales.s * v`. A position whose
  token is the pad id, the image placeholder (120120) or the video placeholder (120124) adds
  nothing (`HYV4FlashModel.build_engram_inputs`; the plan's `masked_token_ids`, from the
  top-level `image_token_id` and `video_token_id`). The result is added to the streams before
  the attention hyper-connection.

Host code: `crates/memra-gguf/src/hy_v4_flash_engram.rs` (n-gram tokens, mask, compressed
vocabulary, multipliers, hash ids, head offsets). Reference: `engram` in
`crates/memra-reference/src/hy_v4_flash.rs`, wired into `decoder_layer`.

## FP8 rows: the round trip question

`engram/fp8_rows.py` range-reads 8 blocks of 256 consecutive rows from each of the four tables of
the BF16 source and of the artifact (8,192 rows, none all-zero) and runs the pinned
`_fp8_round_trip` on them (`engram/fp8-rows.json`, `fp8-rows.log`):

| check | rows |
|---|---:|
| artifact scale = `f32(amax) / 448` of the source row, bit for bit | 8,192 / 8,192 |
| artifact codes = the source row rounded to nearest even, saturated at 448, by value | 8,192 / 8,192 |
| the same, byte for byte (4 rows hold a zero the producer writes as +0 and torch as -0) | 8,188 / 8,192 |
| pinned round trip of the source row in F32 = stored `q * scale` in F32, bit for bit | 8,192 / 8,192 |
| pinned round trip on BF16 activations = stored row in BF16, bit for bit | 808 / 8,192 |
| round trip of the stored row = the stored row (F32; and F64 on the exact `q * scale`) | 8,192 / 8,192 |
| largest code of the row is 448 | 8,192 / 8,192 |

On BF16 activations transformers computes the scale in BF16, up to 0.33% off the stored F32
scale, and rounds the result to BF16: 70.5% of the elements agree, and an element whose code
crosses a rounding boundary moves by up to 11.5%. So the artifact's rows are the F32 program, and
memra reads them as stored with no second round trip (which would be the identity).

## The norm epsilon

`torch.nn.RMSNorm(eps=None)` adds the epsilon of the op-math type, not of the input dtype:
PyTorch v2.15.0 `aten/src/ATen/native/layer_norm.cpp` (`rms_norm_composite`,
`eps.value_or(std::numeric_limits<float>::epsilon())` for BF16, F16 and F32 inputs) and the fused
CUDA kernel in `cuda/layer_norm_kernel.cu` (the same default on the accumulate type).
`engram/eps_check.py` measures it on torch 2.15.0 (`eps-check.log`): on BF16 and F16 inputs the
module equals the formula at `finfo(float32).eps` exactly and differs from the one at
`finfo(input dtype).eps` by 2.1 and 0.63. The plan's rule is now
`HyV4EngramNormEpsilon::OpMathF32` (`value()` is `f32::EPSILON`), the plan hash moves and
`inspect/` is refreshed (`plan_sha256=fd44fd62f4c4ed0a...`; the only diff is the four
`norm_epsilon` lines). The mask's `masked_token_ids` (pad, image and video ids) move it again:
`plan_sha256=47a593754e167e86...`, with the four `masked_token_ids` lists in `model-plan.txt` and
`image_token_id`/`video_token_id` in `normalized-config.txt` as the only other diff. With the artifact's row RMS around 0.85 (0.43 at the lowest sampled), the
step 0 reading would have moved each per-head norm by about 0.5%.

## Oracle

`engram/gen_engram.py` (pinned blobs checked; `OMP_NUM_THREADS=1`) records two parts of
`engram/engram.json`:

- `artifact_hash`: the pinned config class on the pinned `config.json` (sha256 checked), the
  lookup table range-read from the artifact (sha256 checked), and the pinned
  `draw_hash_multipliers`, `build_ngram_tokens`, `build_engram_inputs` (the mask) and
  `hash_ngrams` run as unbound methods on stand-in objects. 210 tokens: the identity prefix and
  the first ids past it, random ids, the pad id, all 14 placeholders, a repeated trigram, the
  image and video placeholders and a packed second document. It keeps the table
  entries those tokens read, not the table.
- `tiny`: the step 4 tiny model (41 tokens past the indexer budget) with `engram_layer_ids` [2],
  2 orders of 2 heads of 4 dims, primes 29, 31, 37, 41, a 60-entry map onto 40 ids on a 64-token
  vocabulary (ids 60-63 lie past it), seed 3, the pad id at one position. The four Engram norms
  get `eps = 2^-23`, the program's value (in a float64 run ATen's default would be the F64
  epsilon). The table is recorded as the round trip's codes and scales; `q * scale` equals the
  pinned round trip's output exactly (asserted).

## What the gate covers

`crates/memra-reference/tests/hy_v4_flash_engram.rs`:

- On the artifact's config, compiled by the pack: the four Engram plans sit in layers 1, 11, 21,
  31 with ids 2, 12, 22, 32, 16 heads of 96 and the op-math epsilon; the multipliers, n-gram
  tokens, mask and hash ids equal the pinned ones exactly. The lookup table outside the recorded
  entries is a sentinel, so a hash that read any other entry would fail.
- Hash red arms, each changing the hash ids: multipliers bounded by the table length (and those
  equal the pinned draw at that bound), the 0-based seed, placeholders read as themselves, the
  two orders' moduli swapped. A pad-only mask differs from the pinned mask, and a zero modulus
  with the same total is refused.
- On the tiny model: the hash ids, the Engram residual (masked tokens exactly zero), every decoder
  layer and the stack to the last hidden state.
- Float red arms, each moving the Engram residual past 1e-6: the 2^-7 epsilon (0.16), stream 0's
  key projection for every stream (0.32), the per-stream scales dropped (0.46), head 0's norm
  for every head (1.7), the mask ignored (0.73).
- Every compared value must be finite (a NaN fails the comparison), and a short Engram weight
  is refused rather than truncated.

## For the whole-trunk and checkpoint steps

- Hashing is integer and exact. Any hash difference is a bug, never a tolerance.
- The f64 oracle must set the Engram norms' epsilon to 2^-23; left at `None`, a float64 run adds
  the F64 epsilon and moves the norms by about `6e-8 / ms` relative, past a 1e-12 bound.
- A transformers run with a BF16 model re-quantizes every Engram row with a BF16 scale (0.33% per
  row, absorbed by the per-head norm) and moves codes across rounding boundaries (29.5% of the
  elements, up to 11.5% each, not absorbed). Against memra, which reads the stored F32 rows, that
  is a BF16-oracle artefact: run the CheckpointParity oracle's Engram path in F32 (or mapped to
  F64), which reproduces the stored rows bit for bit, or attribute the difference to it.
- vLLM, the second reading, loads FP8 tables with a BF16 row scale (`weight_scale_inv`) and runs
  no round trip on a BF16 table, and zeroes a token whose hash ids are all negative where
  transformers masks the pad id and the image and video placeholders. Neither is the oracle.

# Step 6: the whole trunk, TinyParity (2026-11-20)

Verdict: memra-reference runs the whole Hy4-flash text program, token ids to logits, as a
prefill chunk and then teacher-forced decode steps over per-layer caches. On the tiny fixture it
matches the pinned transformers code to the bit: the main request (prefill past the 512-token
window and the indexer budget) and two variants split below each limit, 43,328 logits, and every
indexer selection and Engram row of all 71 forward calls. `memra model verify tiny --against
hy_v4_flash` passes, the pack is `NativeReference` (record `hy_v4_flash-reference`), and seven
wrong programs fail the gate. No GPU was used; CheckpointParity on the artifact is the next step.

## The program

`crates/memra-reference/src/hy_v4_flash_trunk.rs`, from the typed plan: the embedding copied to
the 4 streams, each decoder layer (`decoder_layer_cached`: Engram first where the plan puts it,
then the attention and MLP sublayers through their hyper-connections), `hc_head`, `model.norm`,
the head in f64, logits rounded once to f32. The attention of steps 3 and 4 now runs over an
`MlaCache` (normalized latent, rotated rope key, the latent expanded by `kv_b_proj`, the indexer
token keys, the positions), so one program serves the prefill and each decode step: a decode
query attends the cached keys; an indexed layer pools the cached token keys and runs its
indexer once the cache holds more than `index_topk` keys, as transformers' cached forward does
(`kv_length <= index_topk` attends densely); a sliding layer reads the last `sliding_window`
keys. The Engram n-grams of a step reach back into the token history
(`prepare_inputs_for_generation`). The step 3 to 5 module gates run unchanged on the refactor.

## Fixture

`crates/memra-gguf/src/model_packs/hy_v4_flash/fixtures/tiny-config.json` (the pack's
`TINY_CONFIG`) is the artifact's `config.json` at hidden 64 with its own switches and constants:
window 512, `rope_theta` 1e7 and 1e4, `routed_scaling_factor` 3.998, `rms_norm_eps` 1e-5,
`hc_eps` 1e-6. Six layers in the artifact's order (indexed 0, 1 and 5, sliding 2 to 4, dense
layer 0), 2 indexed and 4 sliding heads with value widths 16 and 40, an indexer of 2 heads of 16
with a budget of 256 (64 groups of 4), 8 experts top-3 in a latent of 32 with the SwiGLU limit at
2 (the artifact's 10 would never clamp at this width), Engram at the config ids 2 and 4 (decoder
layers 1 and 3) with 4 heads of 8 over a 28-entry lookup table onto 20 ids, on a 32-token
vocabulary whose ids 28-31 lie past the table. Its `quantization_config` is the artifact's
ModelOpt map at that width: FP8 128x128 blocks on `q_b_proj`, `o_proj` and `linear_gate` of
layers 1-5, NVFP4 routed experts, layer 0 and everything else BF16 or F32. A pack test pins
that the tiny config keeps the artifact's switches, constants, layer order and formats.

The fixture's tensors are the pack's census of the text program (the tower and the drafter
dropped), seeded and stored in the artifact's classes: BF16, F32, the I32 lookup tables, FP8
blocks with an F32 `weight_scale_inv`, NVFP4 (U8 e2m1 pairs, low nibble first, an e4m3 scale per
16 inputs, an F32 `weight_scale_2` per expert), and the Engram tables' FP8 rows with the
producer's F32 row scale (`f32(amax) / 448`, step 5). The fixture picks power-of-two FP8 block
scales and `weight_scale_2`, so the dequantized weights are exact in the reference's f32
storage. The tokens are a seeded walk over the vocabulary with the pad id and the image and
video placeholders as real tokens, ids past the table and a repeated 16-token span.

| run | tokens | prefill | decode steps | what the split exercises | oracle sha256 |
|---|---:|---:|---:|---|---|
| main | 544 | 536 | 8 | prefill past the window (512) and the budget (256) | `bcb7ce13` |
| `split250` | 280 | 250 | 30 | dense prefill; the indexer starts in decode at 257 cached keys | `0f52565e` |
| `split500` | 530 | 500 | 30 | prefill past the budget; the window starts dropping keys in decode at slot 512 | `0184aa1c` |

## Oracle

`tiny/make_tiny_oracle.py` runs `HYV4FlashTextModel` from the pinned code (modeling,
configuration, conversion mapping, cache and masking files blob-checked) in float64 over the
checkpoint image `verify tiny` writes (`tiny-checkpoint*/`: `model.safetensors`, `config.json`,
`fixture.json`). It dequantizes every tensor exactly in float64, renames it by the pinned
`conversion_mapping` rules for `hy_v4_flash` and concatenates the gate and up banks into
`gate_up_proj`; it redraws the Engram multipliers from the loaded table as `_init_weights` does.
The request runs as `generate` runs it: one prefill, one `DynamicCache`, then one step per token
with `ngram_tokens` and `engram_mask` from the pinned `build_engram_inputs` over the whole
history. Substitutions, as in steps 3 to 5: float32 islands mapped to float64, top-k under the
CUDA tie rule, the Engram norms at `eps = 2^-23`, and the constants the program computes with in
F32 (`rms_norm_eps`, `hc_eps`, `routed_scaling_factor`, `swiglu_limit`, `hc_magnitude`) fed as
their f32 values. The pinned `_fp8_round_trip` stays on; on the stored rows it is the identity,
which the run asserts. The TSVs reproduce bit for bit at 1 and at 8 BLAS threads
(`tiny/generator-run*.json`, `tiny/hashes.tsv`; the image `verify tiny` writes and the one the
oracle read hash the same).

Beside each TSV, `tiny/trace-<run>.json` records per forward call the recalled groups of every
indexed layer (the top-k indices plus the query's own group) and the table rows of every Engram
layer, as FNV-1a 64 digests. `crates/memra-reference/tests/hy_v4_flash_tiny.rs` compares them
first, before any logit: all 192 indexed-layer calls and 142 Engram-layer calls are equal.

## Tolerance

The pack's `checkpoint_parity` band is 1e-5 absolute plus 1e-5 relative, argmax required. It is
set from what two correct f64 runs of this program may differ by, measured in this lane:

- the f32 constants read as decimal literals: `tiny/raw-constants.log` runs the oracle with
  `--raw-constants`, which moves 3,906 of 17,408 logits by at most 2.4e-7 absolute (the
  1e-8-relative move of steps 3 to 5 after the f32 rounding; 895 of them, near zero, move by
  more than one ulp, the largest by 574 ulps of a small logit);
- the RoPE phase formula (transformers `1 / base^(2i/d)`, memra `base^(-2i/d)`):
  `tiny/rope-phase.log` gives 5.7e-14 rad below 544 positions and 1.2e-10 rad at 1M on the
  artifact's 64 rotary dims; on the fixture's 8 dims the two are identical;
- summation order: about 1e-15 relative; the oracle at 8 threads equals the one at 1.

The band is at least 40 times the largest of these (154 times at the fixture's largest logits,
|logit| 2.7) and four orders under the smallest wrong program below. Selections are not inside any band: they
are compared exactly, first. The native runner's CheckpointParity band is its own numeric class,
to be set from the source program's spread between two backends when that runner lands.

## What the gate covers

- `memra model verify tiny --against hy_v4_flash --out OUT` (and `ci:verify-tiny`): the main
  fixture and both variants, each run twice (bit-deterministic) and compared to its pinned
  oracle (`crates/memra-cli/tests/fixtures/tiny-oracle/hy_v4_flash*.tsv`); receipts
  `tiny/tiny-gate.tsv` and `tiny/gates.txt`.
- memra-cli `hy_v4_flash_tiny_gate_matches_the_transformers_oracle_and_refuses_wrong_programs`:
  the reference's main run matches the pinned oracle within the band (to the bit on the lane's
  machines, `tiny/hashes.tsv`), and each wrong program fails the gate (largest logit move,
  `tiny/redarms.log`):

| wrong program | max move |
|---|---:|
| sliding window off | 0.72 |
| attention sink off | 0.77 |
| indexer budget 128 | 2.11 |
| indexer budget 2048 (dense here) | 2.20 |
| Engram seeded with the 0-based layer index | 2.08 |
| Engram placed one layer late (at the config id) | 2.43 |
| router correction bias ignored | 1.80 |

- memra-reference `hy_v4_flash_tiny`: the selections and Engram rows above, a budget one group
  short changes the selection digests on every indexed layer, and a transposed weight with the
  right element count is refused.
- the memra and transformers programs are both split-invariant here: each variant's logits on
  the shared positions equal the main run's bit for bit, in memra and in the oracle.

## For the native runner (step 7)

- The cache layout the reference keeps per layer is the one to port: the normalized latent and
  the rotated rope key per token (576 values on the artifact), the indexer token keys on indexed
  layers (128); a sliding layer attends its last 512 keys, the current one included, so it needs
  the 511 before. The pooled group keys are recomputed from
  the token keys each step, as transformers does; a runner that caches them must close a group
  exactly when its fourth key arrives.
- The indexer switches on per request when the cache passes `index_topk`; prefill and decode
  then agree (the variants prove it on both sides of the switch).
- CheckpointParity runs this oracle's loader on the artifact's own shards: the dequantization
  here (FP8 blocks with `weight_scale_inv`, NVFP4 with `weight_scale_2` per expert, FP8 Engram
  rows) is exact in float64 there too; the reference's f32 storage is exact only because the
  fixture picks power-of-two scales, so the native runner's dequant must not round through f32
  where the artifact's scales are not powers of two.

# Step 7: the native checkpoint runner, CheckpointParity (2026-11-20)

Verdict: memra's first native runner of the artifact, `hy4f-run` (memra-engine `hy4f_gpu`), runs
the whole text program on one GB10 over the artifact's own shards and matches the pinned
transformers program on CPU in f64 with exact dequantization on three requests, 4,263 scored
positions: every indexer selection (25,344 + 768 queries) and every Engram row (344 calls) equal,
every router pick (174,783 queries over 41 MoE layers) equal, the argmax equal at every position, and
the largest logit difference 0.0097 against a class band of 0.12 set from the transformers
program's own spread between its f64 and f32 backends.
`memra model verify checkpoint --against hy_v4_flash` passes (`verified=Checkpoint`), and three
wrong programs fail it. The router turned out to be a second discrete surface: at one near tie
the pinned program's own f32 backend picks another expert and lands 16 logits from its f64 run,
so the gate compares router picks before logits too. Correctness first: the runner is slow by design (about 5 minutes for the
39-token chat turn, 8 for 2,112 tokens); the GPU route is the next step.

## The runner

`crates/memra-engine/src/hy4f_gpu.rs`, `src/bin/hy4f_run.rs`, kernels `cu/hy4f_gpu.cu`:

- the program of the typed plan, numeric class `hy-v4-flash-source-program`: the transformers
  program computed in f32;
- the weights stay in their stored formats and are never formed as rounded dequantized values:
  the GEMM kernel multiplies the e2m1 or e4m3 codes into the inputs, the FP8 128x128 block scale
  (`weight_scale_inv`) or the NVFP4 group scale multiplies each partial sum, and the expert's
  `weight_scale_2` the finished sum. BF16 and F32 weights are read as stored;
- the GEMMs and the attention core (sink in the denominator, an explicit key list per query) run
  on the GPU in f32; norms, RoPE, hyper-connections, router, indexer pooling and scoring and the
  Engram gather and gates run on the host in f64 over f32 activations;
- one layer at a time, its weights read from the shards and dropped after it (no resident
  model, about 20 GB of host memory at most); the Engram tables stay on disk and each row is read
  with `pread`;
- a request is scored teacher-forced in one layer-by-layer pass: the prefill call
  `tokens[..decode_from]`, then one decode call per later token. Every operation acts per token
  except attention, and each kernel computes a row on its own, so the pass gives each query the
  keys its call holds: a prefill query the causal prefix of the prefill, a decode query at `t`
  the `t + 1` cached keys. An indexed layer runs its indexer for a query whose call holds more
  than `index_topk` (2,048) keys, over that call's closed groups; a sliding layer reads its last
  512 keys. The cache layout the reference keeps (576 latent and rope values a token a layer,
  128 indexer key values on the 12 indexed layers) is what the next step carries;
- the router picks from its F32 selection scores (sigmoid plus the correction bias, rounded to
  f32 as the program computes them), under the CUDA tie order;
- it writes a `memra-checkpoint-tape-v2` directory: every position's f32 logits, every indexer
  call's recalled groups and group logits, every MoE layer's experts and selection scores per
  call, every Engram call's table rows as an FNV-1a 64 digest.

## The oracle

`checkpoint/make_checkpoint_oracle.py` runs `HYV4FlashTextModel` from transformers PR #50903 at
b865850d0532 (the five source files blob-checked, no line edited) on the artifact's shards on
CPU, as the tiny oracle of step 6 does: exact f64 dequantization, the pinned `conversion_mapping`
renames, gate and up concatenated in that order, float32 islands mapped to float64, the CUDA
top-k tie rule, the Engram norms at 2^-23 and the F32 constants at their f32 values. A
dequantized model would be about 334 GB, so the trunk is built on the meta device and assigned
(about 53 GB in f64), each routed expert is dequantized when the eager experts loop reads it, and
the Engram rows are gathered from the memory-mapped shards. The rows are read as stored,
`f32(e4m3 * row scale)`; the pinned `_fp8_round_trip` runs on them in F32 and is the identity,
which the run asserts. The requests run as `generate` runs them: one prefill, one
`DynamicCache`, one step per later token with `build_engram_inputs` over the history (pad, image
120120 and video 120124 masked). The router is recorded through a copy of the pinned
`HYV4FlashTopkRouter.forward` (one group, so the group mask keeps every expert) that also writes
its picks and scores. `--dtype f32` runs the same program in f32, and `--force-selections TAPE`
takes every router and indexer pick from the f64 tape while recording its own: the second
backend the band is set from. The f64 tapes are deterministic: rerun after adding the router
records, the logits of all three requests are byte-identical to the first runs. The review's
last rounds added checks to the forcing path after the forced runs (the forcing tape's request
and hashes, exact query geometry, each record consumed where it is applied); they assert and
change no computed value.

## Requests

`checkpoint/gen_probes.py` encodes them with the artifact's template (`reasoning_effort`
"no_think") and tokenizer (`checkpoint/requests/`); the pack pins each by its token ids' sha256,
length and prefill end, and its first request is the pack's declared probe:

| request | positions | prefill | decode calls | what it crosses |
|---|---:|---:|---:|---|
| `chat` | 39 | 36 | 3 | "What is the capital of France? Answer in one word." answered "Paris." |
| `long` | 2,112 | 2,104 | 8 | the indexer budget in the prefill (every prefill query runs the indexer) |
| `long-split` | 2,112 | 2,040 | 72 | a dense prefill; the indexer switches on in the decode at 2,049 keys |

`long` asks for a one-sentence summary of the first 7,884 characters of memra's
`docs/ONBOARDING.md` (`checkpoint/probe-doc.txt`) and forces the answer. Both programs are
split-invariant across the budget: the native `long` and `long-split` logits are equal bit for
bit at all 2,112 positions, and the oracle's at 2,038 (the other 74 within 2.4e-7).

## Results

`checkpoint/receipts/`: the gate's own run (`verify/final`: `run.log`, `checkpoint-parity.tsv`,
`gates.txt`, `artifact.lock`) and the three red arms (`verify/red-*`), the comparator's report per
pair (`native-vs-oracle-*.json`, `band-*.json`, `red-*.json`), every tape's `tape.json` (gzipped, `tapes/`) and the
sha256 of its arrays (`arrays.sha256`), the binaries' sha256 (`bins.sha256`) and the layer
diagnosis of the band (`layer-divergence-long-head.txt`). The arrays are full logits (up to 1 GB a
tape) and stay off the repository. `band-*-free.json` compare tapes from before the router was
recorded and so report failed on missing router records; their logit numbers are the ones below.

| request | indexer calls / queries | router calls / queries | picks differ | Engram calls | argmax equal | max abs diff (prefill / decode) | max KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `chat` | 0 / 0 | 164 / 1,599 | 0 | 16 | 39 / 39 | 3.4e-5 / 2.5e-5 | 2.3e-11 |
| `long` | 108 / 25,344 | 369 / 86,592 | 0 | 36 | 2,112 / 2,112 | 0.0097 / 3.7e-5 | 5.5e-7 |
| `long-split` | 768 / 768 | 2,993 / 86,592 | 0 | 292 | 2,112 / 2,112 | 0.0097 / 5.8e-4 | 5.5e-7 |

No pick differs, so no near-tie flip needed attributing. Against the f64 oracle the native
router scores move by at most 8.6e-4 and the group logits by 2.3e-3. (The first native runs,
which picked experts from f64 router scores, gave 0.0151 on `long`; the program's scores are
F32, and reading them so moved the native logits closer.)

## Band

The class band is set from the pinned program alone, never from the runner: its f32 backend
against its f64 backend on the same requests (`checkpoint/receipts/band-*.json`).

| backend pair | request | picks differ | argmax equal | max abs | per-position max, p50 / p99 |
|---|---|---:|---:|---:|---:|
| f32 vs f64, free selections | `chat` | 0 | 39 / 39 | 2.9e-4 | |
| f32 vs f64, free selections | `long` | 59 indexer queries (router picks not recorded in that run) | 2,062 / 2,112 | 16.0 | |
| f32 vs f64, selections forced to f64's | `chat` | (forced) | 39 / 39 | 1.6e-4 | |
| f32 vs f64, selections forced to f64's | `long` | (forced) | 2,112 / 2,112 | 0.0567 | 1.5e-4 / 3.1e-3 |

The free f32 run diverges from position 163 on. Rerun on the request's first 200 tokens with
every layer's output and router recorded (`--dump-layers`), the f32 layers stay within 7.9e-4 of
f64 through layer 39 at every position and pick the same experts, until MoE layer 40, where position 163's 16th and 17th expert scores differ by
5.4e-7 and f32 takes the other expert; that position's layer output then moves by 1.04 and the
divergence spreads through attention to every later position. A band wide enough for that
would admit any program. Forced to the f64 run's picks, the f32 backend measures what the band
is for, the continuous numeric spread along the same selections: 0.0567. The band is twice
that, 0.12 absolute (the factor the deepseek_v41 class uses), 12 times the native runner's
0.0097 and 20 times under the smallest wrong program refused on logits (2.49).

The selections get their own near-tie bands, from the same forced run: the f32 backend's router
selection scores (|s| <= 1.01) sit within 4.43e-3 of f64's and its group logits (median 2.36)
within 1.18e-2, so a pick that differs from the oracle's passes only when every moved entry
scores within 0.009 (router) or 0.024 (indexer) of the oracle's last kept one. Left free, the
f32 backend's own router picks differ from f64's on 11 of 86,592 queries of `long`, all at gaps
inside 0.009 (the largest 7.9e-4). The band does not cover what follows a flip: a correct
backend that flips at a near tie can still leave the logit band, and then the gate fails it.

## Red arms

Each wrong program runs through the same gate (`checkpoint/red-runner.sh` adds `--red ARM`) and
fails it; a red-arm tape never passes even when its numbers would. With router picks gated, the
first two now stop at a selection, before any logit is read; the logit moves are from the arms' first
runs, before the router was recorded and while the runner still picked experts from f64
scores, so its green runs then sat at 0.0151 on `long` (`checkpoint/compare_tapes.py`,
`receipts/red-*.json`; those tapes carry no router records, so the comparator also reports them
as missing):

| wrong program | request | where the gate stops | max logit move | argmax differs |
|---|---|---|---:|---:|
| Engram off | `chat` | router picks, MoE layer 1, position 0, not a near tie | 17.6 | 12 of 39 |
| quantized weights multiplied as bf16 of their dequantized value | `chat` | router picks, MoE layer 28, position 18, not a near tie | 2.49 | 2 of 39 |
| indexer ties in descending index | `long` | indexer picks: not the native logits' own top-k under the plan's tie order (24,528 of 25,344 queries differ from the oracle's) | 0.0151 | 0 |

The tie order moves no logit on these requests: the queries it changes take groups whose logit
is `-inf` (not yet closed), which attend nothing. Only the selection gate sees it, which is why
selections are compared before logits. Forming the f32 dequantized weight (`f32(code * scale *
scale_2)`) instead of the code-times-input order would move a weight by at most 2^-24 relative,
below this class's own spread, so no gate can or needs to tell it apart in f32; what the gate
refuses is a lower-precision weight.

## What the gate covers

- `memra model verify checkpoint <artifact dir> --against hy_v4_flash --oracle <tape set>
  --native-runner hy4f-run`: the tape set of `crates/memra-cli/src/checkpoint_tape.rs`. Both
  tapes are checked against the compiled plan first (vocabulary, layers, every call's indexer
  records with their query, group and pick counts, one Engram record per Engram layer, array
  sha256), then selections (the native pick must be its own logits' top-k under the plan's tie
  order; a pick that differs from the oracle's passes only as a near tie of the oracle's
  logits), Engram rows exactly, then argmax and band at every position. The artifact lock and the
  receipt carry the request set's digest. A forced or red-arm tape never passes.
- memra-cli unit tests of the gate (`checkpoint_tape::tests`, 9), the pack tests that bind the
  pins to the declared probe and the budget.

## For the GPU route (step 8)

- Carry the cache instead of rescoring the request: the latent and rope key per token, the
  indexer token keys on the indexed layers; close a pooled group when its fourth key arrives.
- Keep the weights resident (experts and Engram as planned in step 0) and move the host
  operations to the device; BF16 activations and tensor-core GEMMs are a new numeric class with
  its own band from the same oracle.
- The f64 oracle tapes of these three requests are the fixed reference: rerun only the native
  side.

# Step 8: the GPU route on one GB10, CheckpointParity of its numeric class (2026-11-20)

Verdict: `hy4f-gpu` (memra-engine `hy4f_route`) runs the artifact on one GB10 with the weights
resident in their stored formats, a carried cache and every operation on the GPU, and passes
`memra model verify checkpoint --against hy_v4_flash` against the same f64 oracle tapes as
step 7: 4,263 positions of the three requests, every own indexer and router pick its own scores'
top-k under the plan's tie order, every selection score within its band of the oracle's, 38
router picks that differ from the oracle's all near ties, every Engram row equal, the argmax
equal at every position and the largest logit difference 0.054 against a class band of 0.111.
Five wrong programs fail the same command. The class is not plain BF16 activations: one position
of `long` amplifies input perturbations about a thousandfold, and every program that rounds its
linear inputs to BF16 once (this route, the pinned program's own BF16 backend, and a backend
that rounds only the linear inputs) lands 13 to 22 logits from f64 there and moves 16 to 51
argmaxes. The route carries each GEMM input as two BF16 terms instead. The first throughput
reading: 225 tokens/s prefill on a 2,104-token prompt and 5.4 tokens/s decode at c=1; nothing
is tuned.

## The route

`crates/memra-engine/src/hy4f_route.rs`, `src/bin/hy4f_gpu.rs`, kernels `cu/hy4f_route.cu`:

- numeric class `hy-v4-flash-bf16x2-activations`. Every weight GEMM takes its input as two BF16
  terms, `hi = bf16(x)` and `lo = bf16(x - hi)` (16 significant bits), and multiplies both on the
  tensor cores (`mma.sync` m16n8k16, F32 accumulation, hi then lo at each 16-wide K step) by the
  weight as stored: BF16 as read; FP8 E4M3 codes, exact in BF16, with the 128x128 F32 block
  scale on each 128-input partial sum; NVFP4 e2m1 codes times their E4M3 group scale (at most six
  significant bits, exact in BF16) with the expert's `weight_scale_2` on the finished sum. No
  rounded dequantized weight exists. The program's F32 islands stay F32: router logits and the
  head are F32 GEMMs over F32 inputs (step 7's kernel), and the hyper-connection streams, norms,
  RoPE (angles in f64), softmax, router scores, indexer pooling and group logits and the Engram
  gates are F32 kernels;
- the weights are uploaded once, 87.4 GiB of device memory (93.4 GB read with `pread`, the page
  cache of each tensor dropped after it), in about 50 to 85 s; the runner refuses unless
  MemAvailable covers them with 16 GiB to spare, and the runs here ran under a 114 GB
  `MemoryMax` scope. The Engram tables (66 GB) stay on disk: each call hashes its n-grams on the
  host with step 5's code and reads each row and its scale with `pread`; the GPU decodes,
  normalizes and gates them;
- the carried cache: per layer the normalized latent and the rotated rope key (576 F32 values a
  token; a 512-row ring on sliding layers, so a query reads its 511 previous keys and its own),
  and on the 12 indexed layers the indexer's token key (128 F32) and every closed group's pooled
  key (128 F32), pooled when the group's fourth key arrives. MLA runs absorbed over the latent:
  `q_lat = kv_b_nope^T q_nope` per head, score `q_lat . latent + q_rope . k_rope`, the sink in the
  max and the denominator, output `kv_b_value (sum p * latent)`;
- the indexer, router and Engram run on the GPU: group logits of the groups closed before the
  query (`-inf` for the rest of the call's groups), a radix-select top-k under the CUDA tie order
  (lower index first), the recall with the own group and its key rows; sigmoid scores plus the
  correction bias, top-16, weights normalized and scaled; a grouped expert GEMM over a device
  tile table, combined in ascending expert order;
- a call scores tokens `p0..p0 + t` against the cache: the prefill call, then one decode call per
  later token, teacher-forced for the gate. Every kernel computes a row on its own in an order
  that does not depend on the other rows or on the tile shape (no split K; the 16-row decode tile
  runs the same per-element sequence as the 64-row prefill tile; attention walks a query's keys in
  fixed 256-key chunks merged in order). Measured: `long` (2,104-token prefill, 8 decode calls) and
  `long-split` (2,040-token prefill, 72 decode calls) give the same logits bit for bit at all
  2,112 positions, free and forced;
- `--force-selections TAPE` applies the oracle tape's router and indexer picks (the router's
  weights follow the forced experts from the route's own scores) while the route records its own
  picks and scores. Free and forced runs of `long` are equal bit for bit at every position before
  the route's first own pick that differs from the oracle's (position 426), and diverge there
  (`gpu/free_vs_forced.py`).

## Why two BF16 terms

The class was first built with single BF16 inputs and a BF16 cache, then measured against the
f64 tapes with selections forced to the oracle's (`gpu/receipts/spreads.jsonl`):

| program, selections forced to f64's | request | max abs | argmax equal | router score spread |
|---|---|---:|---:|---:|
| the route, single BF16 inputs, BF16 cache | `chat` | 0.248 | 39 / 39 | 0.0135 |
| the route, single BF16 inputs, BF16 cache | `long` | 13.3 (pos 756) | 2,096 / 2,112 | 0.864 |
| pinned program, `bf16` backend (BF16 weights, activations and streams) | `long` | 18.9 (pos 756) | 2,061 / 2,112 | 0.888 |
| pinned program, `bf16a` backend (only the linear inputs at BF16) | `long` | 21.6 (pos 756) | 2,092 / 2,112 | 0.513 |
| pinned program, `bf16x2a` backend (linear inputs as two BF16 terms) | `long` | 0.0555 (pos 1083) | 2,112 / 2,112 | 0.0029 |
| pinned program, `f32` backend (step 7) | `long` | 0.0567 (pos 1083) | 2,112 / 2,112 | 0.0044 |

At position 756 the router's selection scores move by 2e-6 at layer 1 in the f32 backend and by
2e-3 at layer 39; the BF16 backends start at about 1e-3 and reach 0.5 to 0.9
(`gpu/receipts/position-756.txt`). The amplification belongs to the program at that position, so
no single-BF16-input program can hold the argmax at every position against f64, and a band wide
enough to admit one (2 x 21.6) would admit any wrong program. With two BF16 terms the backend sits
where the f32 backend does. Forced, the single-BF16 route's own router picks differ from the
oracle's on 6,172 of 86,592 queries of `long`; the two-term route's on 19, all near ties. The decision record is
`docs/decisions/HY4F-GPU-ACTIVATION-FORMAT.md`.

## Band

From the pinned program's `bf16x2a` backend, never from the route: the step 7 `f32` backend with
the input of every linear layer the program runs at its activation dtype (every `nn.Linear`
outside `_keep_in_fp32_modules_strict`, 516 modules, and the routed experts' two `F.linear`s)
carried as `hi + lo` BF16, the weights the exact dequantized values in F32, its picks forced to
the f64 tapes' (`checkpoint/make_checkpoint_oracle.py --dtype bf16x2a`). Against f64 it differs
by at most 0.0555 on `long` and 3.1e-4 on `chat`, argmax equal everywhere; its router selection
scores sit within 2.89e-3 and its group logits within 1.12e-2. Each band is twice that: logits
0.111 absolute, router 0.0058, indexer 0.0224. The route runs forced, as that backend ran
(`TapeClassGate::forced_selections`): its own picks are still checked (own top-k, scores within
the band, a differing pick only at a near tie), and the logit band then measures the class's
numeric spread along the same selections. Left free, a near-tie flip moves later positions by up
to 16.7 logits, as it does for the f32 backend free (step 7).

## Results

`memra model verify checkpoint <artifact> --against hy_v4_flash --oracle <step 7 f64 tapes>
--native-runner hy4f-gpu` (`gpu/receipts/verify/final`):

| request | indexer calls / queries | router calls / queries | picks differ (all near ties) | Engram calls | argmax equal | max abs (prefill / decode) | router score / group logit spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| `chat` | 0 / 0 | 164 / 1,599 | 0 | 16 | 39 / 39 | 1.2e-3 / 3.9e-4 | 4.7e-5 / - |
| `long` | 108 / 25,344 | 369 / 86,592 | 19 router | 36 | 2,112 / 2,112 | 0.054 / 5.1e-4 | 1.6e-3 / 7.1e-3 |
| `long-split` | 768 / 768 | 2,993 / 86,592 | 19 router | 292 | 2,112 / 2,112 | 0.054 / 4.1e-3 | 1.6e-3 / 1.9e-3 |

## Red arms

Each through the same command (`checkpoint/red-runner.sh` adds `--red ARM`); each fails before
any logit is read (`gpu/receipts/verify/red-*`):

| wrong program | where the gate stops |
|---|---|
| Engram off | `chat`, router score, layer 1, position 0: 0.018 from the oracle's, band 0.0058 |
| quantized weights multiplied as bf16 of their dequantized value | `chat`, router score, layer 34, position 32: 0.0060, band 0.0058 |
| indexer ties in descending index | `long`, layer 0, position 0: the recall is not its own logits' top-k under the plan's tie order |
| a decode call writes its cache row one row late | `chat`, decode position 36, router score, layer 1: 0.0097 |
| a group pooled at its third key | `long`, decode position 2108, layer 0, group 526: group logit 0.894 from the oracle's, band 0.0224 |

The dequantized-weight arm is the narrowest: on `chat` it leaves the logits within the band
(0.107) and only the router score band refuses it: the first score past the band is 0.0060, the
largest 0.0083. On `long`, run forced
outside the gate, it moves the logits by 9.2 at position 756 (6 argmaxes of 2,112) and the router
scores by up to 0.37, 64 times the band (`gpu/receipts/spreads.jsonl`). Run forced outside the
gate on `long-split`, the late cache row moves its decode logits by 17.7 (10 argmaxes), while the
early-pooled group leaves the logits where the green run has them (0.054): the forced picks never
read that group, and only its group logits (4.5 from the oracle's) and 11 own recalls show it,
which is why the gate reads the selection scores. The
selection-score bands are what keep the narrow wrong programs out; they were added to the gate in
this step and hold step 7's class too (its runner's spreads, 8.6e-4 and 2.3e-3, sit inside its
bands of 0.009 and 0.024).

## The gate

- The pack names each native class's terms (`ClassParityGate::tape`, `TapeClassGate`): the
  oracle class it is scored against, its router and indexer selection bands, and whether it runs
  forced. `hy-v4-flash-source-program` (step 7) runs free against itself;
  `hy-v4-flash-bf16x2-activations` runs forced against the source program's f64 tapes. The
  artifact lock carries the terms.
- The runner names its class (`RUNNER --numeric-class`); a runner that names none is scored as
  its oracle's class. The gate passes a forced class `--force-selections <oracle tape>`, and the
  native tape must name that tape's sha256. An oracle tape is never forced.
- New in this step for every class: each native router selection score and each finite group
  logit must sit within the class's band of the oracle's, before the near-tie attribution.
- Step 7's class through the new gate: `hy4f-run` passes it free (`gpu/receipts/verify/pr7-class`), its numbers unchanged from step 7: max abs 0.0097, router scores within 8.6e-4 and group logits within 2.3e-3 of the oracle's.

## Throughput, a first reading

One GB10, c=1, greedy, one warm-up then three timed runs of a prefill and 63 decode calls
(`hy4f-gpu --bench 3`, `gpu/receipts/bench-*.jsonl`), GPU at 56 to 60 C after the runs, SM
clock read 2,184 MHz (maximum 3,003):

| prompt | prefill | prefill tokens/s | decode tokens/s | decode step p50 / p99 |
|---|---:|---:|---:|---:|
| `chat`, 36 tokens | 0.43 s | 84 | 5.9 to 6.0 | 166 / 168 to 210 ms |
| `long`, 2,104 tokens | 9.30 to 9.34 s | 225 to 226 | 5.4 | 185 / 187 to 216 ms |

A measurement only, N = 3 a cell. Nothing is tuned: single-stage GEMM tiles, SIMT attention, one
launch per operation, the Engram rows read synchronously each call. The weight read alone bounds
decode near 24 tokens/s on this box (step 0).

## For the serve route (step 9)

- The serve route runs free (its own picks); the gate's forced run covers the apply path only up
  to the first flip, which the free-vs-forced check pins bit for bit.
- The cache is F32: 33.8 KB a token on the indexed layers (8.9 GB at 256k) plus the sliding
  rings (35 MB); a prefill call holds its whole prompt's scratch, so long prompts need chunking.
  Each scratch growth and each call's host output is admitted against MemAvailable with 4 GiB to
  spare and refused otherwise; a call that fails leaves the route refusing until `reset` starts a
  new request (a failed call may have rewritten a ring row an earlier position owns).
- The NativeQualified record, the model card's serving row and the serve gates are step 9's.

# Step 9: the serve route on one GB10, NativeQualified (2026-11-20)

Verdict: memra-server serves Hy4-flash on one GB10 (DGX Spark, sm_121a) through its own
`hy_v4_flash` route over the GPU route of step 8, running free on its own router and indexer picks,
and `memra model verify checkpoint` and `memra model verify serve --against hy_v4_flash` pass on the
pinned artifact with that binary (`verified=Checkpoint`, `verified=Serve`). Served greedy answers
equal the route's own greedy ids token for token on the three checkpoint requests, every
teacher-forced decode position served as its own prompt equals the step 8 tape, chunked prefills
give the same logits as whole ones bit for bit (chunks of 7 to 2,048 tokens, across the 2,048-key
indexer budget and the 512-key window), and two to six requests served at once answer byte for byte
as alone. The pack is `NativeQualified` on this topology (record `hy_v4_flash-gpu-1x-gb10`). A
correctness route, not a fast one: one request decodes at about 6 tokens per second; nothing is
tuned.

## The route

`crates/memra-server/src/hy4f_serve.rs`, wired beside the deepseek-v4.1 route in `worker.rs` (a
checkpoint directory whose `model_type` is `hy_v4_flash` loads here and never through
`HybridModel`):

- **Program.** The step 8 class, `hy-v4-flash-bf16x2-activations`, with the route's own picks. A
  request's prompt is scored in prefill calls of at most `MEMRA_HY4F_PREFILL_CHUNK` tokens (default
  2,048) on its own session, then one decode call per generated token. Every kernel computes a row
  on its own, so a chunk boundary changes no value (measured below), and the transition from prefill
  to decode is the same program.
- **Sessions.** `hy4f_route::Session` holds a request's cache (576 F32 a token a layer, a 512-row
  ring on the 30 sliding layers, the indexer's token and group keys on the 12 indexed layers) and
  its history; the weights and the call scratch are shared. `Route::call_in` swaps a session in for
  one call. Each step runs the oldest prefilling request's next chunk, then one decode call per
  generating request, each a call of its own, so a long prompt does not stall the others' decode
  for longer than one chunk.
- **Load.** The whole checkpoint binds to the pack's contract first (2,005 contract entries, the
  drafter's and the vision tower's among them, each present with its shape), then the weights upload in their
  stored formats: 87.5 GiB resident in 83 to 91 s. The Engram tables (66 GB) stay on disk; their rows
  are read through the checkpoint's own read-only mapping (`madvise` random), so they live in the
  page cache, which the kernel reclaims. A warm-up of two chunks and a decode call grows the call
  scratch to its chunk-sized maximum (3.04 GiB) and checks the logits are finite.
- **Memory book (unified memory).** The ceiling is MemAvailable after the weights and the warm-up,
  less the route's 4 GiB headroom: 20.9 to 21.0 GB here. A session books its cache at its capacity
  (prompt plus token budget: 35.4 MB plus 35,328 bytes a position) and the one call scratch that
  grows with the context (the indexer's group logits of a chunk) plus a 1 GiB margin. A request
  that cannot fit with the route idle is refused as a context-length error naming the largest
  session that fits; one that fits alone waits until running requests finish; then a live
  MemAvailable check (the new session's cache and the scratch growth still owed to the longest
  session) and every allocation's own check refuse rather than run the box out of memory. The
  served context is the largest session the book holds with the route idle: 531,000 positions here
  (the config declares 1,048,576), so the 256k scope target fits one session with room for a second
  of about 270k.
- **Pool.** The server's engine pins the device pool's release threshold high, so a finished
  session's cache would stay mapped and out of MemAvailable. The route trims the pool when a session
  ends (`Dev::trim_pool`). The first full gate run found this: the third of three 212k-token
  sessions was refused with 9.95 GB available and nothing running (`serve/receipts/gate-65d5e8d00f-pool/`).
- **Sampling and stops.** Greedy is the first of equal maxima (the tapes' rule); temperature,
  top-p, top-k, seeds and penalties draw through the DSv4 route's position-keyed host sampler;
  `min_p` is refused. Stops: the end-of-sentence id (unless `ignore_eos`), stop ids, stop strings
  (the Hy4 calls' close `</tool_calls:opensource>` stays in the stream so the parser closes the
  block), the budget and the session's capacity.
- **Chat surface.** The HTTP layer renders with step 2's Hy4 template and parses with
  `ToolStreamParser::hy4` (`ModelCaps::hy4`, set with `hy3`, which carries the shared seam and
  framing). `reasoning_effort` takes `no_think` (also `none`), `high` (the template's default) and
  `max` (also `xhigh`); `low` and `medium` ask for reasoning and render `high`; any other level is a
  400. Every chat request arms the parser with the think open unless reasoning is off.
- **Policy contract** (`route_contract::RouteContract::hy4f_thread`): every surface implemented,
  prime fairness by the chunk-a-step schedule.
- **Not served, by name:** `response_format`, image and video parts, `min_p`, DFly2 drafts,
  prefix reuse.

## Gates

On one GB10 (sm_121a, 128 GB unified memory, CUDA 13.0), every run alone on the box, the binaries of
`2e915e65a9` (`serve/receipts/verify-2e915e65a9/bins.sha256`):

| gate | result |
|---|---|
| `memra model verify checkpoint <artifact> --against hy_v4_flash --native-runner hy4f-gpu` | `verified=Checkpoint`, numbers equal to step 8's receipt line for line (max abs 0.054, argmax at all 4,263 positions) |
| `memra model verify serve <artifact> --against hy_v4_flash --native-runner memra-server` | `verified=Serve`; Config, TokenizerTemplate, TensorCensus, CheckpointParity and Serve passed in the bundle (`serve/receipts/verify-2e915e65a9/gates.txt`) |
| `memra model verify tiny --against hy_v4_flash` and `ci:verify-tiny`'s test on the same tree | `verified=Tiny`, every oracle hash equal to step 6's; the test passed (`serve/receipts/verify-2e915e65a9-tiny/`) |
| the route after the session split, against step 8's tapes | `long` whole and with prefill chunks of 500, 511, 1,000 and 2,048, `long-split` with chunks of 7 and 512: all 2,112 positions' logits have step 8's sha256 (`36feabb7...`); `chat` forced whole and in chunks of 5: step 8's (`78376ab8...`) (`serve/receipts/route/`) |
| render | `/v1/tokenize` of the probes' user turns (reasoning off) equals the requests' prompt ids up to the generation prompt |
| route | served greedy `/v1/completions` on the three requests' prompts equals `hy4f-gpu --gen` (chat 1 token and EOS, long 33 and EOS, long-split 72) |
| forced | 23 teacher-forced positions (every decode position of `chat` and `long`; `long-split` at 2,047 to 2,050 and every eighth to 2,112, across the budget) served as prompts: each token equals step 8's free tape. The free route's tape equals the f64 oracle's argmax at every one of these positions except 2,072 of `long-split`, where the free route had already left the oracle at its first own near-tie pick (position 426) |
| concurrent | c = 2, 3, 4 and 6 requests at once (the 2,104-token `long` prefill among them): every answer's ids equal its solo ids |
| stream | SSE deltas equal the blocking answer, finish reason and `[DONE]` present |
| efforts | `no_think`/`none` answer without reasoning, `high` and `max` with it; `low`/`medium` equal `high`, `xhigh` equals `max`; `turbo` is a 400 |
| tools | a tool call round trip with reasoning off (`get_weather`, JSON arguments, the follow-up with the tool's answer served) and a call with reasoning on |
| sampled | seed 1017, T 0.8, top-p 0.95, top-k 40: the same answer twice alone and once beside three others, unlike greedy |
| stops, penalties, refusals | a stop string cuts before its first occurrence; `max_tokens` 5 stops at the budget; penalties repeat; an image part and `response_format` are 400s |
| cancel | a streamed client that leaves after 3 of 400 tokens: its session ends at the next step and the next request runs alone |
| admission | three requests with 212,400-token budgets on the 531,000-position context: two admitted at once, the third deferred, then admitted, all three the full greedy answer |
| long | a needle at the middle of a 30,934-token prompt (15 times the indexer budget) found, greedy, 191 s |
| `MEMRA_HY4F_PREFILL_CHUNK=500` | the same served gates with 500-token chunks |

## Readings (measurement only)

One GB10, the binaries of `2e915e65a9`, through the server (`serve/scripts/bench.py`: memra's own
stream shape, `prompt_ids`, greedy, `ignore_eos`, one warm-up then three samples a workload, N = 3;
GPU at 61 to 65 C after each sample, SM clock read 2,177 to 2,184 MHz;
`serve/receipts/gate-2e915e65a9/bench.jsonl`). TTFT is request to first token; prefill tokens/s is the
prompt over TTFT; decode tokens/s is per request after its first token; aggregate is every request's
tokens over the sample's wall time. Medians, ranges in brackets:

| workload | prompt / output tokens | TTFT | prefill tok/s | decode tok/s per request | aggregate tok/s |
|---|---:|---:|---:|---:|---:|
| chat, c = 1 | 36 / 128 | 0.43 s | 83 | 5.96 | 5.89 |
| long, c = 1 | 2,104 / 64 | 10.26 s | 205 | 5.42 | 2.92 |
| chat, c = 4 | 36 / 128 | 1.16 s [0.43 to 2.22] | 33 [16 to 84] | 1.49 | 5.89 |
| long, c = 4 | 2,104 / 32 | 26.1 s [10.3 to 42.7] | 84 [49 to 205] | 0.86 [0.61 to 1.46] | 2.00 |

The route runs one call at a time, so four requests share the decode rate of one (each about a
quarter) and the c = 4 prefills queue one 2,048-token chunk a step behind each other. The 2,104-token
prefill reads 205 tokens/s through the server in two calls (step 8's bench, one call: 225). The
30,934-token needle prompt took 191 s end to end, about 162 tokens/s. Noted for the speed lane, not
done here: running the generating requests' decode rows through one call (the deepseek-v4.1 route's
batched step) is the lever for aggregate throughput, and needs its own batched-vs-solo gate.

## What is left

- Speed (a separate lane): batched decode across sessions (calls run one at a time, so aggregate
  throughput is flat in the number of requests), tensor-core attention, fused and staged GEMMs,
  launch count, overlapped Engram reads, CUDA graphs. A batched-vs-solo gate comes with any batching.
- DFly2 drafts (step 10), with the open question of the drafter's dense MLP width: 9,216 in the
  checkpoint's tensors, not in the config, where the vLLM derivation defaults it to
  `intermediate_size` (13,824) before reading the checkpoint (step 1's open questions).
- The sm_120a record and the device battery cells for this stack (the landing runs them).
- Prefix reuse, `response_format`, vision, logprobs.

## Commands

`serve/scripts/`: `serve_gate.sh TAG [gate|bench|both] [ENV=...]` boots the server, runs
`chat_gate.py` and `bench.py` against it and stops it; the lane's other scripts are in
`serve/receipts/scripts/` with the run box's paths.
