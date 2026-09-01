# GLM-5.3-Flash tensor census (zai-org/GLM-5.3-Flash @ main, 2026-08-27)

76,108 tensors, 328.3 GB FP8 e4m3 (block scales as `*_scale_inv`; BF16 twin repo
exists: zai-org/GLM-5.3-Flash-BF16). Source: model.safetensors.index.json +
config.json banked beside this file. Counts below are exact.

## Topology facts (from counts, not the card)

- 45 decoder layers + **1 MTP (NextN) layer = 46** carrying attention norms
  (`input_layernorm` ×46, `post_attention_layernorm` ×46, `o_proj` ×46). MTP layer
  is DeepSeek-shaped: `eh_proj`, `enorm`, `hnorm`, `shared_head.norm` (×1 each).
- **KDA (linear attention) ×34**: `A_log`, `dt_bias` (decay, mamba-class),
  `q/k/v_proj` + `q/k/v_conv1d` (short conv 4), `b_proj`, low-rank gate pairs
  `f_a_proj`/`f_b_proj` and `g_a_proj`/`g_b_proj`, `o_norm`. All KDA projections
  are **BF16** (no scale_inv) — matches `modules_to_not_convert`.
- **MLA+indexer (DSA) ×12** = 11 in the main stack (layers 3,7,...,43) + 1 in the
  MTP layer (`index_share_for_mtp_iteration: true`): `q_a_proj`/`q_b_proj` (+
  `q_a_layernorm`), `kv_a_proj_with_mqa`/`kv_b_proj` (+ `kv_a_layernorm`), indexer
  = `wk`, `wq_b`, `weights_proj`, `k_norm(.weight/.bias)`,
  `index_kpool_compress_ape`, `index_kpool_compress_gate`. MLA projections are
  FP8 EXCEPT `kv_b_proj` (BF16 — it is absorbed/expanded at runtime).
  NoPE: qk_rope_head_dim=0, `mla_use_nope: true` — no rotary in the MLA path,
  AND NONE IN THE INDEXER EITHER (corrected 2026-08-30): the 5.3 reference's
  `Glm5NextTextIndexer.forward` (modular_glm5_next-ref.py:771) computes
  `q = wq_b(q_resid)` and `k = k_norm(wk(x))` with no rotary application, and the
  file never reads the config key `indexer_rope_interleave` — the key ships in
  config.json but is dead for this architecture (the only rotary in the reference
  is the vision tower's `Glm5NextVisionRotaryEmbedding`). The earlier text here
  inferred an indexer rope from the config key; the reference never applies one.
- **MoE ×43** (42 sparse decoder + MTP layer; first 3 dense): 288 experts ×
  gate/up/down (FP8) = 12,384 per proj; router `gate.weight` +
  `e_score_correction_bias` (noaux_tc, sigmoid scoring, routed_scaling 2.5);
  1 shared expert per layer (FP8). Dense layers: plain gate/up/down (FP8).
- **mHC (hyper-connections) ×45 per stream-pair**: `hc_attn_{base,fn,scale}` +
  `hc_ffn_{base,fn,scale}` — hc_mult 4, sinkhorn_iters 20, eps 1e-6. BF16
  (in modules_to_not_convert as hyper_connection).
- **Vision** 24 blocks (qkv fused, q/k norm, SwiGLU mlp, norm1/2) + patch_embed,
  downsample, merger (gate/up/down + proj + post_projection_norm), post_layernorm.
  All BF16.
- `lm_head` + `embed_tokens` BF16, not tied. vocab 154,880 padded; triple EOS
  [154820, 154827, 154829].

## Serving-relevant config pins

hidden 4096; 64 heads; qk_nope 256, v_head 256, kv_lora 512, q_lora 1536;
KDA: 64 heads × 128, gate_lower_bound −5.0; indexer: 32 heads × 128, topk 2048,
kpool 4 compress + always-select-tail; ctx 1,048,576; MoE dim 2048, 8 active + 1
shared; reasoning_effort ∈ {low, high, max}, DEFAULT max (thinking-default —
TRAP:reasoning-effort-unpinned-decode-cell applies to every perf cell);
vendor sampling: temp 1.0, top_p 0.95, no top_k; chat_template.jinja shipped
(clear_thinking defaults false).

## Quantization split (FP8 checkpoint)

FP8 e4m3 dynamic-activation: all MoE experts + shared experts + dense MLPs + MLA
q_a/q_b/kv_a/o_proj. BF16 kept: every KDA tensor, kv_b_proj, mHC, embeddings,
lm_head, vision, norms, router gates. `modules_to_not_convert` in config names
attn_mha/attn_mqa/dt_bias/hyper_connection/lm_head/mapping_proj/embed_tokens.
