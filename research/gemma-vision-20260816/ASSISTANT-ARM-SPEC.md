# Gemma 4 31B official MTP assistant — memra arm spec (2026-08-16)

Owner funded ("every possible improvement") the arm to use Google's OFFICIAL MTP
assistant `google/gemma-4-31B-it-assistant` as the gemma-4-31B drafter, aiming to beat
the dspark baseline (code 0.739 / 190.4 tok/s, prose 0.549 / 132.6 tok/s — receipts/).

## BUILD-TIME CORRECTIONS (2026-08-16, second pass — supersede the sizing below)

1. **The arm ALREADY EXISTS: `gemma_spec.rs` (landed 2026-07-10).** The "days-class,
   two new primitives" sizing below was made without finding it — post-compaction
   blindness, the exact failure the orientation law warns about. `GemmaDraft` +
   `gemma4_draft_step`/`generate_spec_gemma` implement precisely this architecture:
   Q-only dual-geometry (hd 256 SWA / hd 512 global) draft attention over the MAIN
   model's KV (`gemma4_draft_kv_target`: last own-KV layer per class), main-embed
   `× sqrt(n_backbone)` + concat pre-projection, plain tied head (no softcap),
   post-projection h chain. Verified against llama.cpp's gemma4-assistant wiring in
   July; measured then at 0.845-0.883 acceptance on a 31B-class trunk. First-shot
   bring-up on this lane (5090, QAT Q4_0 trunk + QAT Q8_0 assistant): acceptance
   0.573, 2.30× (41.8→96.3 tok/s), 128/128 byte-exact.
2. **Concat glue PINNED from HF source** (transformers 5.14.1,
   `generation/candidate_generator.py`, Gemma4 assistant `get_candidates`):
   `inputs_embeds = torch.cat([target_embedding(last_token_id), last_hidden_state], -1)`
   — EMBEDDING half first (scaled: `Gemma4TextScaledWordEmbedding` applies
   `× sqrt(backbone_hidden)` inside the module), HIDDEN half second. Round seed:
   backbone `hidden_states[-1]`; in-loop: the assistant's own `post_projection`
   output. `position_ids` CONSTANT (`seq_len-1`) across the draft round;
   `use_cache=False`; draft token = plain `logits.argmax`. memra's July arm matches
   all of this. ONE convention divergence: HF 5.14.1 records `hidden_states` per
   decoder layer, so its round seed is PRE-final-norm; memra/llama.cpp seed with the
   POST-output_norm hidden (h_nextn). p0 acceptance 0.84 properly paired says the
   convention performs; noted as a tuning experiment, not a defect.
3. **Weight-lineage pairing law (measured, load-bearing).** The on-disk
   `gemma-4-31B-it-Q8_0-MTP.gguf` is the QAT-lineage assistant, NOT gg-hf-am bf16
   (byte gate: layer_scalars 0.130/0.566/0.613/0.490 vs 0.146/0.578/0.613/0.520;
   output_norm max|Δ| 10.2). Cross-pairing costs half the win: on the QAT Q4_0 trunk
   the QAT head reads 0.573 acceptance, the official bf16 head 0.344. Drafters must
   ship lineage-matched to their trunk: QAT head ↔ QAT trunk, official bf16 head
   (fresh GGUF `gemma-4-31B-it-official-{F16,Q8_0}-MTP.gguf`, byte-parity-gated vs
   gg-hf-am: layer_scalars EXACT, output_norm 1024/1024) ↔ the NVFP4mix
   official-weights artifact.
4. Remaining truth of the original recon: the CENSUS corrections stand (no centroid
   tensors, layer-3 global hd 512 — both mirrored in the GGUF metadata:
   `key_length 512 / key_length_swa 256`, kv heads `[16,16,16,4]`), and the
   scratch-KV dflash law exception is real — it is simply already carved out by
   `gemma_spec.rs`, not a new build.

**Status: spec locked from HF source; ARM EXISTS (July); A/B vs dspark running.**
The sizing below is kept for the record of what the census gate caught.

Sources (verbatim): `transformers/models/gemma4_assistant/modeling_gemma4_assistant.py`
(249 lines, read in full), the checkpoint `config.json`, and the safetensors header
(48 tensors, range-read). Line refs below are into that modeling file.

## Checkpoint census (model.safetensors, 939 MB bf16)

```
model.embed_tokens.weight            [262144, 1024]   # tied to lm_head
pre_projection.weight                [1024, 10752]    # 10752 = 2*backbone_hidden(5376)
post_projection.weight               [5376, 1024]     # assistant hidden -> backbone dim
model.norm.weight                    [1024]
masked_embedding.centroids.weight    [2048, 1024]     # centroid head (see below)
masked_embedding.token_ordering      [262144] i64     # buffer: canonical vocab permutation
# 4 decoder layers (layer_types = [swa, swa, swa, full]):
model.layers.N.self_attn.q_proj.weight   [8192, 1024]   # 32 heads x 256 head_dim; Q ONLY
model.layers.N.self_attn.q_norm.weight   [256]
model.layers.N.self_attn.o_proj.weight   [1024, 8192]
model.layers.N.input_layernorm / post_attention_layernorm /
   pre_feedforward_layernorm / post_feedforward_layernorm  [1024]  # sandwich norms
model.layers.N.layer_scalar               [1]           # per-layer residual scalar
model.layers.N.mlp.{gate,up}_proj.weight  [8192, 1024]
model.layers.N.mlp.down_proj.weight       [1024, 8192]
```
No k_proj / no v_proj — `attention_k_eq_v=true`, `num_kv_shared_layers=4` (ALL layers).

## CENSUS-GATE CORRECTIONS (2026-08-16, before any code)

The deterministic census gate (assistant_census_check.py) caught two spec errors I would
otherwise have built wrong — the exact silent-wrong class the norm-fold finding taught this
lane:

1. **Decode is a PLAIN TIED lm_head — NOT centroid decode.** `use_ordered_embeddings=false`
   on this checkpoint; the centroid/token_ordering tensors do not exist (48 tensors total).
   Primitive #3 below is a CLASS capability this 31B checkpoint does not use. **This drops
   the arm from three new primitives to TWO** — memra already has tied-embedding lm_head.
   Big de-risk; a centroid primitive would have been built for nothing.
2. **Layer 3 (the full_attention layer) uses global geometry:** head_dim 512, so
   q_proj [16384,1024], q_norm [512], o_proj [1024,16384]. Layers 0-2 (sliding) use
   head_dim 256 ([8192,1024]). The KV-share attention arm must carry BOTH geometries.

The two-primitive arm (KV-share attention + 2×backbone concat) stands; centroid decode
struck.

## The three new primitives (why this is not the dflash path)

memra's dflash/MTP law: draft head owns its OWN scratch KV (§D.6). This checkpoint breaks
all three of that law's assumptions:

1. **Target-KV-share attention.** The assistant computes only Q from its own hidden; K
   and V come from the BACKBONE (target 31B) KV cache — `shared_kv_states` in forward
   (mga.py:167,181). The draft attention kernel must read the target's stored K/V buffers,
   not compute or own them. `k_eq_v=true` ⇒ K and V are the SAME tensor (backbone stores
   one, assistant reads it as both). Two KV geometries: SWA layers read the backbone's
   sliding KV (16 kv heads, head_dim 256, window 1024); the final full-attention layer
   reads the backbone's global KV (4 kv heads, global_head_dim 512). This is a NEW draft
   attention arm keyed to the backbone's per-layer-type KV — the single biggest build.

2. **2×backbone concat pre-projection.** `inputs_embeds` into forward is [B, L, 10752] =
   two concatenated backbone-hidden(5376) vectors, projected to assistant hidden 1024
   (mga.py:126,177). This is the MTP concat, but its EXACT construction (which two 5376
   vectors, in which order, normed how) lives in HF's assisted-generation candidate
   generator, NOT in the checkpoint or modeling file. **PIN THIS FROM
   `transformers/generation/candidate_generator.py` (the Gemma4/assisted path) AT BUILD
   TIME — do not guess.** This is the norm-fold-class silent-wrong trap for this arm: a
   wrong concat order forwards fluently and only shows as low acceptance, never an error.

3. **[STRUCK for this checkpoint — use_ordered_embeddings=False; plain tied lm_head.
   Kept only as the class capability others in the family (E2B/E4B) may enable.]
   Centroid-masked decode (mga.py:42-87).** NOT a cheap lm_head and NOT an
   approximation-with-quality-loss in the usual sense — it is: project hidden→2048
   centroid logits (`centroids` linear), take top-32 centroids, gather their vocab members
   via the `token_ordering` permutation (2048 clusters × 128 tokens/cluster = full 262144
   vocab partitioned), score ONLY those 32×128=4096 candidate tokens by dot with the tied
   embed_tokens rows, scatter into a full-vocab logit vector with the rest masked to
   min-1. CONSEQUENCE FOR ACCEPTANCE: the draft argmax is exact IFF the true argmax's
   token sits in one of the top-32 centroids; otherwise the draft proposes from a
   restricted 4096-token set. Fine for a draft (verify catches misses), but it means the
   acceptance-vs-dspark comparison is the whole point and cannot be assumed to win — it
   must be measured. A memra centroid-decode primitive (top-k over 2048 + gather-scatter
   over the ordering buffer) is required; a plain lm_head substitute would be WRONG
   (different token distribution).

Also: bidirectional/position-invariant masks over the shared KV (mga.py:205-247, the
flip-on-kv-axis SWA trick) — the draft attends the past KV as a bidirectional block since
q_len is small; the mask construction is spelled out and must be reproduced.

## memra mapping (deterministic, gate-able now — see census-check)

| checkpoint tensor | memra draft slot | note |
|---|---|---|
| pre_projection.weight | new: draft.pre_proj [1024,10752] | fresh slot, no existing analog |
| post_projection.weight | new: draft.post_proj [5376,1024] | |
| layers.N.self_attn.q_proj | draft.layer[N].wq | Q-only; NO wk/wv. L0-2: [8192,1024] (hd 256); L3: [16384,1024] (global hd 512) |
| layers.N.self_attn.q_norm | draft.layer[N].q_norm | L0-2: [256]; L3: [512] |
| layers.N.self_attn.o_proj | draft.layer[N].wo | L0-2: [1024,8192]; L3: [1024,16384] |
| layers.N.{4 norms} | draft.layer[N].{sandwich norms} | gemma sandwich; norm-fold law: MEASURE fold per-artifact, panic on unrecognized (already the house guard) |
| layers.N.layer_scalar | draft.layer[N].residual_scalar | per-layer, [1] |
| layers.N.mlp.{gate,up,down} | draft.layer[N].{ffn_gate,up,down} | GEGLU (gelu_pytorch_tanh) |
| embed_tokens / lm_head (tied) | draft.embed (=lm_head) | tied; PLAIN lm_head decode (no centroid on this checkpoint) |

## Sizing (honest)

Days-class, TWO fresh CUDA/loader pieces + the concat-glue pin + measurement (centroid
struck by the census gate):
1. Loader `Arch::Gemma4Assistant` + config parse + census-parity (gate lands NOW —
   assistant_census_check.py green: 48 tensors, layer-3 global geometry, no k/v, no centroid).
2. Target-KV-share draft attention arm (SWA head_dim 256 for L0-2 + global head_dim 512 for
   L3, reading the backbone's per-type KV; k_eq_v ⇒ K and V are one tensor). Biggest piece.
3. Pin the pre-projection concat from candidate_generator.py; wire the draft round to feed
   backbone hidden + build the 2×5376 concat. Decode = existing tied lm_head (no new head).
4. Gate ladder (house law): loader census gate (green); greedy logit parity of one draft
   step vs the HF paired-forward oracle (assistant_oracle.py, needs both checkpoints on a
   box); acceptance A/B vs dspark interleaved x5; exactness is free (verify makes the
   served output identical regardless — the draft only moves speed/acceptance).

Bar to justify shipping over dspark: acceptance > dspark's 0.549 prose / 0.739 code at
comparable draft cost. If the KV-share attention + centroid decode cost more per draft
step than the acceptance gain buys, dspark stays. Decide by measurement.
