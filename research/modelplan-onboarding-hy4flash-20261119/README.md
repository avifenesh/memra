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
  best closed groups plus its own (2,048 tokens). Below 2,048 keys the layer attends densely.
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
vocabulary), identity only for ids 0-63, and the pad id 120002 maps to 91,804. The table is
load-bearing: the `arange` default would hash a different program.

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
