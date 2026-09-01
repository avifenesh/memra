# glm5_next ModelPlan design (increment 2, pinned before code)

Verified against modular_glm5_next-ref.py (banked) + CENSUS.md. Decisions:

## AttentionPlan
- NEW variant `AttentionPlan::KimiDeltaNet(KimiDeltaNetPlan)`:
  { num_heads: 64, head_dim: 128, conv_kernel: 4, gate_lower_bound: -5.0 }.
  Symmetric heads (q=k=v = 64x128) — deliberately NOT GatedDeltaNetPlan (asym
  key/value heads, scalar-per-head decay); KDA decay is PER-CHANNEL via low-rank
  f_a(hidden)->head_dim->f_b->qkv + dt_bias, g = lower_bound * sigmoid(exp(A_log)*g);
  beta = sigmoid(b_proj) per head; q/k l2norm fp32 (+eps inside sqrt, FLA);
  output = o_proj(RMSNormGated_SIGMOID(core, gate=g_b(g_a(x)))).
  Checkpoint stores q/k/v convs split; loader fuses to one grouped conv (3*qkv).
- MLA layers: `MlaAttentionPlan::LatentKv` { query_heads 64, q_lora 1536,
  kv_lora 512, qk_head_dim 256, rope_head_dim 0 (NoPE — RopePlan::None class),
  value_head_dim 256, sparse_index: Own{heads 32, head_dim 128, top_k 2048,
  kpool: Some(KpoolPlan{pool 4, always_select_tail: true})} }.
  SparseIndexPlan::Own gains `kpool: Option<KpoolPlan>` (kpool compression:
  learned softmax over gate_scores + APE inside each pool of 4; ReLU-scored;
  top-(top_k/pool) pools expanded to token ids; incomplete tail appended raw).
  indexer_rope_interleave is config fact only — reference forward applies NO rope
  in the indexer (NoPE model); do not add a plan field until an oracle disagrees.

## Layer mix + state
- compile_attention branches per-layer on cfg.glm5.is_kda_layer(il): 34 KDA /
  11 MLA at [3,7,...,43]. All DSA layers own their indexer (indexer_types all
  "full"; census: 12 indexer sets = 11 trunk + 1 MTP).
- StatePlan: KDA -> Recurrent { conv_width 24576 (3*64*128), conv_kernel 4,
  state_width 64*128*128 } (state fp32 by contract — mamba_ssm_dtype float32);
  MLA -> LatentKvCache { width 512 } (rope 0 adds nothing).

## Residual topology
- ResidualTopology::HyperConnections { streams 4, epsilon 1e-6,
  sinkhorn_iterations 20 } + NEW collapse knob: glm5_next collapses streams at
  model exit with an UNWEIGHTED MEAN (Glm5NextTextHyperHead), unlike dsv4's
  weighted collapse. Add `collapse: HcCollapse::{Weighted, Mean}` (default
  Weighted for dsv4 — check dsv4_forward.rs actual exit math before naming).
  Per-layer application (both hc_attn and hc_ffn): out = post ⊗ branch_out +
  combᵀ · streams; sinkhorn(20, eps) normalizes mixing weights; entry = broadcast
  expand x4.

## MoE / MLP
- RouterPlan::Sigmoid { normalize_selected: true, scaling_factor: 2.5,
  selection_bias: true } (noaux_tc == DeepSeek-V3 recipe, e_score_correction_bias
  = exp_probs_b — existing end-to-end path).
- Activation: clamped swiglu, ASYMMETRIC: gate.clamp(max=+10) (no lower bound!),
  up.clamp(±10), silu(gate)*up. VERIFY memra's SwiGluClamped matches this exact
  clamp shape (dsv4 lineage); if memra clamps gate from below too, new variant.
- Shared expert: SharedMlpPlan { intermediate 2048 (= moe_int * n_shared), gated }.
- Dense layers 0..2: DenseMlpPlan intermediate 12288.

## MTP
- num_nextn_predict_layers 1 -> automatic MtpBlockPlan; its LayerPlan attention =
  MLA LatentKv + Own indexer (12th indexer set exists in census).
  index_share_for_mtp_iteration=true refers to sharing across MTP steps, not
  base-vs-MTP (MTP has own tensors) — verify against transformers generate path
  when the spec route lands; irrelevant for NativeReference.

## Norms
- All RMSNorm eps 1e-5 (rms_norm_eps) EXCEPT KDA o_norm gated (eps 1e-6? —
  o_norm constructed with layer_norm_epsilon = rms_norm_eps 1e-5; recheck at
  impl); indexer k_norm has BIAS (k_norm.bias in census — LayerNorm not RMSNorm?
  verify wk norm class in GlmMoeDsa base before wiring).
- Triple EOS [154820,154827,154829]; head_dim top-level = 0 quirk (ignore).

## Verify-items RESOLVED (2026-08-27, before implementation)
- Activation: memra SwiGluClamped = silu(gate).min(limit) * up.clamp(±limit)
  (reference lib.rs:4785) — POST-activation clamp. GLM5.3 clamps PRE-activation:
  silu(gate.clamp(max=limit)) * up.clamp(±limit), gate one-sided. NEW variant
  `ActivationPlan::SwiGluPreClamped { limit }`.
- HC: per-layer math identical to dsv4 (same transformers class); ONLY the final
  collapse differs — dsv4 hc_head = sigmoid-gated (dsv4_forward.rs:789), glm5 =
  unweighted mean. Knob: HcCollapse::{GatedHead, Mean} on
  ResidualTopology::HyperConnections.
- Indexer k_norm = LayerNorm WITH bias (GlmMoeDsaIndexer <- DeepseekV32Indexer;
  census k_norm.bias). Not RMSNorm.
