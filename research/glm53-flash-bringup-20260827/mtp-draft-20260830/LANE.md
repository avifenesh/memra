# glm5_next native MTP, draft side (lane/glm5-mtp-remint, 2026-08-30)

Lane pivot record. This lane opened as "re-mint the NVFP4 artifact to ADD the MTP/NextN
tensors the first mint dropped". The premise was false — the artifact carries the full
MTP layer; the DFlash2 probe's "no mtp/nextn tensors" finding was a name-grep miss
(`CORRECTION-artifact-has-mtp.md` beside this file, verified against the published
index). The lane pivoted to what is actually missing: **MTP execution** — the tensors
loaded nowhere and nothing executed the NextN layer.

Why native MTP matters: it is the license-clean speculative draft. Upstream measures
native MTP acceptance 3.71-5.06 and 1.36-2.05x decode at concurrency 1 on this model
(engine-survey-20260829/ENGINE-SURVEY.md §4.1); the faster DFlash2 drafter is
CC BY-NC-ND and cannot serve customers.

## The refusal map (what stood between the checkpoint and an executed MTP step)

Censused before building, each refusal named:

| layer | state before this lane |
|---|---|
| checkpoint | COMPLETE — 2,631 `layers.45.*` tensors incl. eh_proj/enorm/hnorm/shared_head.norm + 288-expert MoE (NVFP4) + MLA (NVFP4) + indexer (BF16) |
| `ModelPlan` | COMPLETE — `mtp_blocks[0]` compiles: MLA + own k-pool indexer (`compile_attention` mtp arm), MoE (`layer_uses_moe` answers past the trunk vec), `ResidualTopology::Serial` (NextN carries no hc_* tensors) |
| tensor contract | COMPLETE — `add_mtp_glue` HfSafetensors arm names `model.layers.45.{enorm,hnorm,eh_proj}` + the `shared_head.norm.weight` alias; census gate binds them |
| `Cache` | COMPLETE — `new_inner` iterates `0..cfg.n_layer` (=46) chaining `plan.mtp_blocks`, so the MTP latent + indexer planes were ALREADY allocated at il=45 |
| ggml->HF map | **REFUSED (silent)** — no glm5_next row for `nextn.{enorm,hnorm,eh_proj,shared_head_norm}`; `src.has("blk.45.nextn.eh_proj.weight")` = false; embedded-MTP loop `break`s at offset 0, loading NO head, no log line |
| reference | **REFUSED (loud)** — `execute_embedded` passed the raw `[tokens, streams*hidden]` hc stream stack as `trunk_hidden`; `execute_mtp` errored "HyperConnections MTP fusion" on every hc plan |
| engine forward | **REFUSED (panic)** — `mtp_head_forward_dev` op 6: `Mixer::Mla(_) => mla_path_unimplemented("MTP head forward")`; also every spec entry point (`generate_spec`, `generate_spec_eagle`, `generate_spec_dflash`) refuses hc trunks via `refuse_hyper` |
| worker routing | fails closed — `mtp_spec_capable` requires `MTP_SPEC.capabilities(plan).speculative.supported`, and `mtp_spec_support` includes neither `HyperConnections` nor the KDA/MLA/kpool operations, so a loaded head can NOT route customers into `refuse_hyper` |

## What this lane built (draft side only)

1. **hf_mapping** (`crates/memra-gguf/src/hf_mapping.rs`): four plain rows in the
   glm5_next arm — `nextn.enorm.weight -> enorm.weight`, `hnorm`, `eh_proj`,
   `nextn.shared_head_norm.weight -> shared_head.norm.weight` (glm5 norms are verbatim,
   no +1 fold). `nextn.shared_head_head.weight` deliberately unmapped: the checkpoint
   ships no private MTP head; absent IS the trunk-lm_head fallback. Pinned by
   `glm5_next_mtp_block_resolves_through_the_engine_map` (28 rows, contract-complete,
   red-proven: without the rows it fails on `blk.2.nextn.enorm.weight`).
2. **Reference** (`crates/memra-reference/src/lib.rs`): `collapse_trunk_hidden` — ONE
   trunk-exit collapse shared by the LM-head projection and the MTP fusion input, so
   `execute_mtp` receives the same collapsed PRE-output_norm hidden the engine hands
   over as `h_seed` (MTP-PLAN §A). Red-proven: reverting the pass reproduces the
   "HyperConnections MTP fusion" error under the new host gate.
3. **Load flag** (`crates/memra-engine/src/hybrid.rs`): `MEMRA_GLM5_MTP`, default OFF —
   glm5_next skips the embedded-MTP load unless set. Deliberate: the head is a full MoE
   layer (a trunk-layer of VRAM + load time) and nothing consumes it in serving yet.
   OFF arm = today's serving byte-identical (the head was already never loaded).
   FLAGS.md row lands in the same commit.
4. **Engine forward** (`crates/memra-engine/src/spec.rs`):
   `mtp_head_forward_mla_cached` — one NextN draft step for the MLA-mixer MTP class,
   same op chain as `mtp_head_forward_dev` (enorm/hnorm -> concat -> eh_proj ->
   attn_norm -> attention -> +residual -> post_attn_norm -> MoE -> +residual ->
   shared_head norm -> trunk lm_head), with the attention arm riding `mla_attn_cached`
   on the plan's own MTP latent+indexer plane (il = 45) instead of the full-attn
   `MtpScratch`. Err-not-assert throughout: non-MLA mixer, Dense FFN, missing plane,
   and position gaps all return named errors.

## Gate (glm5_mtp_head_gpu; 5090, TF32 off, flock-serialized, 2026-08-30)

Fixture: 2-trunk-layer hc glm5_next mini config through the real parser/pack
(dense f32 trunk so the gate's error budget isolates the MTP block) + 1 NextN block
(MLA + k-pool indexer + 4-expert MoE with the real routing constants). MTP expert banks
served **NVFP4** (the artifact's real class for these tensors; the engine's expert
loader refuses F32 banks); the reference reads the roundtrip of the exact bytes the
engine decodes, so the measured residual is the expert kernel's own
quantized-activation arithmetic.

| gate | result |
|---|---|
| 1 (host) plan wiring + reference collapse fix | PASS (red-proven: pre-fix errors "HyperConnections MTP fusion") |
| 2 GREEN teacher-forced walk, 48 rows, engine vs `execute_mtp` | PASS — worst row **1.144e-3** (TOL 5e-3, measured-and-pinned) |
| 3 RED eh_proj served TRANSPOSED | bites — worst row **4.165e0** |
| 4 RED h_seed off by one row | bites — worst row **6.723e-1** |
| 5 flag default OFF loads no head; `=1` loads MLA+MoE head | PASS |
| 6 position gap refuses with a named Err | PASS |

Green-to-red separation: 3 orders of magnitude. The walk exercises the draft plane's
k-pool indexer IN the sparse regime (48 rows over a 16-raw-token budget) and the
incremental one-row-per-step cache accumulation.

Repro:
```
cargo test -p memra-gguf --lib hf_mapping                                  # name map pins
cargo test -p memra-engine --test glm5_mtp_head_gpu                        # host gate
NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
  cargo test -p memra-engine --test glm5_mtp_head_gpu -- --ignored --test-threads=1
```

## Interface contract for the verify arc (stated precisely, as built)

- **h_seed**: `[n_embd]` f32 device buffer = the trunk's COLLAPSED PRE-output_norm
  hidden of the position whose next token is being drafted (MTP-PLAN §A). This is
  exactly what `prime_cache` / `decode_step` already return for hc models (the
  `hiddens` stack rows / the decode `h_seed`); `MEMRA_SPEC_HPOST` flips producer and
  carrier to post-norm together, same as the qwen35 path.
- **e_tok**: the token at the seeded position's SUCCESSOR (the token the trunk just
  sampled/accepted). Oracle pairing: `mtp_head_forward_mla_cached(depth 0, ids[i],
  h[i], pos i)` reproduces `execute_mtp` row `i`.
- **logits**: `[n_vocab]` f32 device, FULL vocab (no private head, no d2t — the draft
  projects through the trunk `lm_head`).
- **carrier**: `[n_embd]` pre-shared_head_norm hidden (post-norm under
  `MEMRA_SPEC_HPOST`) — the chain seed for multi-depth drafting; glm5_next declares
  one head, `index_share_for_mtp_iteration: true` refers to reuse across MTP
  iterations of that one block.
- **state**: the MTP block's latent+indexer plane lives IN the model `Cache` at
  il = n_trunk (45); `mla_attn_cached` appends exactly ONE row per draft step and
  advances `latent[45].len` itself. `mtp_pos` must equal the plane's current length
  (enforced, named Err). Rollback on rejection = latent-plane len reset (+ the
  resident pool-key `ready` rewind the trunk kpool planes already do on rewind);
  the plane is append-only otherwise.
- **routing safety**: worker `mtp_spec_capable` stays false for this plan (MTP_SPEC
  manifest has no hc/KDA/MLA/kpool operations), so loading the head cannot expose
  customers to `refuse_hyper`. The verify arc must extend the manifest deliberately
  when its paths are gated, not flip it here.

## Deliberately NOT built here (the verify arc's scope)

- T-parallel verify over the hc trunk, acceptance/rollback, `generate_spec` wiring,
  and the KDA/kpool state rollback under rejected drafts.
- Any serving exposure: the flag stays OFF; no roster, perf, or acceptance claim.
- Prefill of the MTP plane over the prompt (the draft-side gate teacher-forces from
  position 0; upstream runs the MTP layer over the prompt to warm its cache — that
  belongs to the verify arc's scheduling).
- Upstream-shape acceptance numbers: 3.71-5.06 is upstream's measurement on their
  stack, quoted for context only, never a claim about ours.
