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
`unified_rmsnorm`) are `torch.nn.RMSNorm` with `eps=None`, so transformers uses
`torch.finfo(input dtype).eps`: 2^-7 on BF16 activations. vLLM uses the F32 epsilon
(`finfo(float32).eps`) for the per-head and unified norms. memra follows transformers, the
oracle, and carries the rule in the plan rather than a number.

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
| DFly2 dense MLP width: 9,216 in the checkpoint, not in the config. The vLLM derivation defaults `dfly_intermediate_size` to `intermediate_size` (13,824) and then materializes the width from the checkpoint. Confirm against the DFly2 program before pinning it beyond the census. | PR 10 (DFly2) |
| Whether transformers' per-row FP8 round trip of a BF16 Engram row equals this checkpoint's stored `q * weight_scale` bit for bit. | CheckpointParity |
| The Engram norm epsilon is `finfo(input dtype).eps` in transformers (2^-7 on BF16); vLLM uses the F32 epsilon. The oracle must run BF16 activations for the band to mean anything. | CheckpointParity |

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
