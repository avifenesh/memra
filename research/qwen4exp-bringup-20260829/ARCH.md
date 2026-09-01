# qwen4_exp geometry (phase 1-2) — Qwen/Qwen3.8-Flash-Next @ de4b8e4d

Source: config.json + safetensors header census (raw/census-summary.txt, 1658 tensors,
359,999,963,128 bytes, all BF16 except two I64 index tables). Every claim below is a
shape read from headers, not an assumption. Nothing transfers from qwen3_5 (27B) by
analogy — LAW:no-generic-support.

## Trunk layout

48 text layers, pattern index%4==3 → QSA else GDN (12 QSA, 36 GDN); MoE after every
layer. hidden 2560. **Residual stream is 4×2560=10240** (gated residual / hyper-
connections, hc_count 4): every sublayer reads the wide stream via a rank-320 mix
(input_mix_weight_down [320,10240] → up [10240,320]) plus hc_norm [10240], and writes
back via block_inject_weight [4,10240]. One global hyper_connection_mixer (same shapes)
at trunk level. lm_head/embed are 2560-wide — the wide stream is mixed down at entry/exit.

## Per-class geometry table

| class | count | tensors (shape) | resolved geometry |
|---|---|---|---|
| GDN linear_attn | 36 | in_proj_qkv [10240,2560]; in_proj_z [6144,2560]; in_proj_a,b [48,2560]; conv1d [10240,1,4]; A_log,dt_bias [48]; norm [128]; out_proj [2560,6144] | QK: 16 heads × 128 ×2 = 4096; V: 48 heads × 128 = 6144 (qkv fused 10240); z = output gate on V; a/b per-V-head data-dependent scalars; conv k=4 over fused qkv; RMSNormGated per head_dim 128; ssm state f32 |
| QSA self_attn | 12 | q_proj [12288,2560]; k_proj,v_proj [512,2560]; o_proj [2560,6144]; q_norm,k_norm [256] | 24 Q heads × 256 = 6144, q_proj carries 2× → fused per-head output gate (sigmoid, output_gate_type); 2 KV heads × 256; n_rot = 0.25×256 = **64** (partial rotary — same trap class as the 27B n_rot bug), rope_theta 1e7, mrope interleaved [11,11,10] |
| QSA indexer | 12 | index_qk_proj [640,2560]; q_layernorm,k_layernorm [128] | MQA: 4 Q heads ×128 + 1 shared K head ×128, fused proj; micro-block size 4 (compress_ratio), budget 512 blocks = 2048 tokens |
| MoE | 48 | gate [512,2560]; experts.gate_up_proj [512,1280,2560]; experts.down_proj [512,2560,640]; shared_expert g/u [640,2560], d [2560,640]; shared_expert_gate [1,2560] | 512 experts, top-10 + 1 shared, intermediate 640; experts are fused 3D tensors (Qwen4ExpTextExperts module, NOT nn.Linear) |
| PLE / ngram (layer 2 only) | 1 | ngram_embedding: 128 shards × [2,500,012,160] (=320,001,536 rows, 102.4 GB bf16); ngram_heads_offsets, ngram_heads_vocab_sizes I64 [16]; layer_multipliers I64 [3]; key_proj [10240,2560]; value_proj [2560,2560]; conv1d [10240,1,4]; norm_conv/key/query [10240] | 16 ngram heads (8 per ngram size × {bigram,trigram}), per-head vocab in the I64 tables; gather-only 160-dim rows → value_proj to 2560; keyed against the wide stream. Offload-friendly by design |
| MTP | 1 | fc_embedding, fc_hidden [2560,2560]; pre_fc_norm_embedding [2560], pre_fc_norm_hidden [10240]; + one FULL decoder layer (QSA + indexer + 512-expert MoE + both hyper_connections) + own hyper_connection_mixer | 1-layer draft head; shares embed/lm_head (no dedicated embeddings); reads the wide (10240) trunk stream + token embedding; rope_theta 1e7 |
| Vision | 27 blocks | qkv fused [3456,1152]+bias, proj [1152,1152]; mlp fc [4304]; patch_embed Conv3d [1152,3,2,16,16]; pos_embed [2304,1152]; merger fc1 [4608,4608] → fc2 [2560,4608] | ViT hidden 1152, 16 heads, patch 16, temporal 2, spatial merge 2 (4×1152=4608 into merger); no deepstack |
| Embeddings | | embed_tokens, lm_head [248320,2560], untied | vocab 248,320 (padded) |

## Serving-relevant consequences (to verify on hardware, not assume)

- KV cache exists on 12 QSA layers only: 2 KV heads × 256 × 2 × bf16 ≈ 24 KB/token
  (+ indexer K ~3 KB/token compressed). 262K ≈ 6.3 GB; 1M (YaRN) ≈ 24 GB.
- GDN state per seq: 36 layers × 48 V heads × 128×128 f32 ≈ constant ~340 MB/seq — the
  prefix-cache/extension-shapes law from the 27B likely applies HARDER here (36 vs 48 GDN
  layers but wider state); re-derive, don't assume.
- QSA replaces full attention: decode reads ≤2048 tokens of KV regardless of context —
  long-context decode cost is flat by construction. Indexer selection is the new
  correctness-critical path (top-k over micro-blocks; tie-break semantics TBD phase 3).
- New math vs anything we have: gated residual (4-branch wide stream), QSA indexer
  (micro-block top-k), ngram gather + PLE conv block, fused-3D expert matmuls. Router
  is softmax top-10 renormalized (Qwen3NextTopKRouter — corrected 2026-08-29, see
  SEMANTICS.md; earlier "sigmoid router" here was wrong), shared-expert gate sigmoid. GDN kernel geometry differs from 27B (48V/16QK/128 vs
  27B's shapes) — kernel-check pins required per class.

## Phase log

- 2026-08-29 phase 0: lane frozen (branch qwen4exp-bringup-20260829 @ 46aa1aa475,
  artifact rev de4b8e4d pinned, raw metadata banked).
- 2026-08-29 phase 1: census banked (raw/census.tsv.gz, census-summary.txt); tokenizer/
  template inspected: ChatML markers, tools present, enable_thinking/preserve_thinking/
  reasoning_effort template kwargs, eos [248046,248044]; defaults audit: generation_config
  do_sample t=1.0 top_p .95 top_k 20 (thinking-mode); card instruct rec t=.7 top_p .80
  presence 1.5; reasoning_effort default xhigh.
- 2026-08-29 phase 2: this geometry table. transformers main (5.16.0.dev0) instantiates
  Qwen4ExpForConditionalGeneration on meta ✓ (oracle route available for goldens).
