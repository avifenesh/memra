# qwen4_exp forward semantics — extracted from transformers modular_qwen4_exp.py (main, fetched 2026-08-29)

Source of truth for reference-executor arms. Line refs are into
`transformers/src/transformers/models/qwen4_exp/modular_qwen4_exp.py` (fetched at
transformers 5.16.0.dev0-era main; bank a copy beside this file if it drifts).
Corrections to earlier lane docs are marked ⚠.

## Layer stack (Qwen4ExpTextDecoderLayer.forward, L796-833)

```
if ple: hidden += PLE(hidden, ple_input_ids)            # wide stream, layers with PLE only
mixed, hyper, inj = attn_hyper_connection(hidden)        # read gate (wide→2560)
mixed = linear_attn(mixed) | self_attn(mixed)            # token mixer on 2560
hidden = hyper + outer(mixed, inj)                       # write gate back to wide
mixed, hyper, inj = mlp_hyper_connection(hidden)
mixed = moe(mixed)
hidden = hyper + outer(mixed, inj)
```

- Entry (L1012): `hidden = embed_tokens(ids).repeat(1,1,hc_count)` — wide stream = 4 copies.
- Exit (L1025): `hidden = hyper_connection_mixer(hidden)` (use_combine=False → returns the
  2560 mixed read only). ⚠ There is NO separate final norm module — the mixer's grouped
  hc_norm is the exit normalization; census has no `model.language_model.norm`.

## Gated residual (Qwen4ExpTextGatedResidual, L530-558)

- `hc_norm` = **grouped RMSNorm**: normalize each 2560-wide group of the 10240 vector
  independently (group_size = hidden_size; Qwen4ExpTextRMSNorm L298-309).
- read: `w = sigmoid(up(silu(down(normed)/hc_count)))` → [10240] → view [4,2560];
  `mixed = mean_over_streams(w * normed_streams)` → [2560].
- write: `inj = 2*sigmoid(block_inject(normed)/hc_count)` → [4] scalars;
  `out = hyper_input(PRE-norm) + (block_out[...,None,:]*inj[...,:,None]).flatten`
  (L825-826: injection = block_out ⊗ inj across the 4 streams).
- RMSNorm here is **(1+weight)-centered**: `_init_weights` zero-inits RMSNorm weights
  (L860-861) and the class docstring says "RMSNorm here does (1 + weight)" — ⚠ verify the
  parent Qwen3_5RMSNorm applies `(1+w)*x̂` and mirror exactly.

## QSA indexer (Qwen4ExpTextQSAIndexer, L367-473)

- `index_qk_proj(hidden)` [640] → split q = 4 heads × 128, k = 1 × 128.
- q: per-head RMSNorm(128) → rope with the MAIN rotary cos/sin, **partial**: rotary_dim =
  head_dim×partial_rotary_factor = 64 applied to dims [0..64) of the 128-d index head
  (apply_rotary_pos_emb L329-364 splits rope/nope at cos width).
- k: cached **RAW** — pre-norm, pre-rope (update_indexer L411). Indexer cache cost:
  128 dims/token/QSA-layer.
- Per query token: visible tokens (mask row) in order → complete blocks of compress_ratio=4;
  pooled key = fp32 mean of the 4 raw keys → k_layernorm → rope at the position of the
  block's FIRST token (group_starts, L439-444).
- score(block) = Σ_{h=1..4} relu(q_h · k_block) / sqrt(128), fp32 (L446-449).
- top-k blocks, k = min(512, n_complete_blocks) (torch.topk largest; ⚠ tie order is
  torch-impl-defined — our kernel must pin a deterministic tie rule and gate it against
  this reference on tie-free fixtures, dsv4-lane lesson).
- Selected = chosen blocks' tokens ∪ the incomplete TAIL block's tokens (always visible,
  L456-457). Max selected = budget + ratio − 1 = 2051.
- Output = boolean/float overlay mask; attention then runs **dense with the combined mask**
  (Qwen4ExpTextAttention.forward L491-496: `causal ∧ indexer` / additive for eager).
  Reference arm in memra = full attention + this mask. Efficient kernel = gather, later.
- Decode consequence: each new token attends ≤2051 tokens regardless of context length.

## Attention (QSA layers, Qwen4ExpTextAttention, L476-507)

Inherits Qwen3_5Attention: fused q|gate in q_proj (24 heads × 256 × 2), sigmoid per-head
output gate (family convention), q_norm/k_norm RMSNorm(256), partial rope 64, theta 1e7,
2 KV heads. KV cached normally (24 KB/token over the 12 QSA layers).

## GDN layers (L321-326)

`Qwen4ExpTextGatedDeltaNet(Qwen3_5GatedDeltaNet)` — the qwen3_5 GDN math EXACTLY
(same in_proj_qkv/a/b/z, conv k=4, A_log/dt_bias, chunk/recurrent delta rule), with ONE
delta: the gated output RMSNorm activation = `output_gate_type` = **sigmoid**, not silu
(L324-326; config L201-203 allows {sigmoid, silu}). ⚠ single-line numeric divergence from
the 27B — do not reuse its pinned outputs.

## MoE (L510-527)

⚠ CORRECTION to earlier lane docs: router = `Qwen3NextTopKRouter` = **softmax over 512
logits, top-10, renormalize (norm_topk_prob=True)** — NOT sigmoid-scored. Shared expert:
`Qwen4ExpTextMLP` (SwiGLU, intermediate 640) gated by `sigmoid(shared_expert_gate(x))`
(Qwen3NextSparseMoeBlock convention). Experts fused 3D (gate_up [512,1280,2560],
down [512,2560,640]).

## PLE / n-gram (L561-778)

- `ple_layer_ids` is **ONE-indexed** (config doc L91-92): `[2]` → module `layers.1`
  (0-based). PLE allowed only on linear_attention layers (validate L256-263).
- N-gram ids (Qwen4ExpTextNGramEmbedding.forward L658-703): token history = 2 previous
  tokens (cached as conv_state idx 2; fresh context pads with EOS) ++ current ids;
  `shifted[j]` = tokens shifted right by j with **EOS-segment reset** (positions whose
  segment (since last EOS) is shorter than the shift read EOS, L642-656);
  for n∈{2,3}: `mixed = XOR_{j<n}(shifted[j] * layer_multipliers[j])` (int64 wraparound);
  per head h of that n-gram's 8: `id = mixed mod head_vocab_sizes[h] + head_offsets[h]`.
  16 head vocab sizes = consecutive primes ≥ 20,000,000; multipliers/sizes/offsets ship
  as checkpoint buffers (census I64 [3]/[16]/[16]) — LOAD, never re-derive.
- Gather: 16 ids → [16,160] rows → flatten [2560].
- PLE block (L758-778): `key = grouped-RMSNorm(key_proj(emb))` [4,2560];
  `query = grouped-RMSNorm(hidden_wide)` [4,2560];
  `gate_s = (key·query)/sqrt(2560)` per stream → signed sqrt
  (`abs.clamp_min(1e-6).sqrt()*sign`) → `sigmoid`;
  `gated_value[s] = gate_s * value_proj(emb)` → flatten [10240];
  `out = gated_value + silu(depthwise_conv1d(grouped-RMSNorm(gated_value)))` with kernel 4,
  **dilation 3**, causal left-pad 9, own conv cache (state idx 1, len 9);
  added to the wide stream BEFORE attn_hyper_connection (L806-809).
- Cache states on the PLE layer: idx0 GDN conv, idx1 PLE conv (9), idx2 token history (2).

## Rope / positions

- Rotary = Qwen3_5TextRotaryEmbedding, 3-axis position_ids (mrope [11,11,10] interleaved,
  rows 1-3 of a 4-row position tensor; row 0 = text positions for masks, L963-977). For
  TEXT-ONLY input all axes are equal → degenerates to plain partial rope (64 dims, theta
  1e7). Engine text path may use plain rope; vision path needs the real 3-axis (net-new).
- Indexer consumes FULL-history cos/sin (cache binds position_ids, L979-985).

## MTP

⚠ transformers modeling has NO MTP module — the mtp.* namespace is checkpoint-only.
RESOLVED 2026-08-29 from SGLang PR #36497 (sgl-project/sglang @ 99c9362e,
srt/models/qwen4_exp_mtp.py, banked in raw/):

- MTP model = Qwen4ExpModel with num_hidden_layers=1, layer_types=["full_attention"]
  (QSA, own indexer weights), full_attention_interval=1, ple_layer_ids=[] (no PLE).
- Input fusion (_fuse_residual_linear_shared): the spec chain hands the DRAFT the trunk's
  WIDE hidden state [10240] (spec_info.hidden_states). Then:
  e = fc_embedding(GemmaRMSNorm_2560(embed(token)));
  h = GemmaRMSNorm_10240(wide_hidden) viewed [4,2560]; per-stream fc_hidden(h);
  fused wide input = (e broadcast over streams + fc_hidden(h)) flattened to [10240].
- GemmaRMSNorm = zero-centered (1+w) — matches the family's RMSNorm convention.
- Exit: the 1-layer model applies its own hyper_connection_mixer (mtp.hyper_connection_mixer
  in census) → [2560] → SHARED lm_head (loaded from the target's head). The model also
  returns the post-layer wide state, which becomes spec_info.hidden_states for the NEXT
  draft step (multi-step trained; the wide state is the K>1 carrier).
- The cookbook confirms the MTP's full-attention layer runs QSA too, "which is what keeps
  speculative acceptance high"; SGLang tests a shared-indexer optimization
  (test_qsa_mtp_shared_indexer.py) — kernel-level, not semantics.
- Owner mint decision context: MTP experts are NVFP4 in our artifact (trunk-matching).

## Loading notes

- 128 ngram shards `shard_0..127` [2500012,160] concatenate on dim 0 → [320001536,160]
  (config docstring L108-110); HF skips its device placement entirely
  (`_no_placement_params`, L847) — same constraint our loader must respect (host-resident
  or sharded across devices; TP plan shards it colwise on dim 1, L142-144).
- `number_of_conv_states` = 3 when PLE present else 1 (L180) — cache layout is per-layer
  keyed, worth mirroring in engine cache design.
- `layer_types` in the shipped config says `full_attention`; the config class REWRITES it
  to `qwen_sparse_attention` when indexer fields are set (L188-192). Engine should treat
  full_attention + indexer_* present ⇒ QSA.
