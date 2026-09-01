# CORRECTION: the NVFP4 artifact DOES carry the MTP/NextN tensors (2026-08-30)

The DFlash2 probe's receipts (`lane/glm5-dflash2-probe`,
`research/glm53-flash-bringup-20260827/dflash2-probe-20260829/RECEIPTS.md`) state:

> Our NVFP4 artifact carries NO mtp/nextn tensors (checked
> `model.safetensors.index.json`), so that route needs a re-mint with the MTP head
> before it can even be probed.

**That claim is false, and the re-mint it motivated is unnecessary.** The check was a
name grep for the literal substrings `mtp` / `nextn`, which this checkpoint never uses:
glm5_next spells its MTP layer `model.language_model.layers.45.*` (the appended NextN
layer; the modeling's `_keys_to_ignore_on_load_unexpected` in the reference file names
`layers\.45\.` for exactly this reason).

## Verified against the published index directly (2026-08-30)

`model.safetensors.index.json` fetched from `Avifenesh/GLM-5.3-Flash-NVFP4 @ main`:

- **113,446 tensors total** = 1,432 kept + 3 x 37,338 NVFP4 triples — the COMPLETE
  BF16-twin census (CENSUS.md: 38,770 source tensors, 37,338 quantized). Nothing was
  dropped by the mint.
- **2,631 `model.language_model.layers.45.*` tensors**, and layer 45 IS the MTP/NextN
  layer with its signature glue present: `eh_proj.weight`, `enorm.weight`,
  `hnorm.weight`, `shared_head.norm.weight`, plus its full 288-expert MoE (as NVFP4
  triples), its MLA projections (NVFP4), and its own k-pool indexer set (BF16).
  Arithmetic check: 288 experts x 3 projections x 3 (triple) = 2,592, + 9 shared-expert
  triple + 12 MLA triple + 18 kept glue/norm/indexer tensors = 2,631. Exact.
- All 347 `model.visual.*` tensors are present too.

The mint receipts agree end to end: `../mint-receipts/mint-log-summary.txt` shows
`[verify] OK: 37338 NVFP4 triples + 1432 kept tensors, 190.7 GB total`, and the
`hf_quant_config.json` / `nvfp4-config.json` ignore lists name the layer-45 keeps
(`layers.45.eh_proj`, `layers.45.mlp.gate`, `layers.45.self_attn.indexer.*`,
`layers.45.kv_b_proj`). The mint's classifier handled layer 45 explicitly
(mint-nvfp4.py: MLA quant suffixes on the MTP layer, MTP scaffolding in
`ANYLAYER_KEEP_SUFFIXES`).

## What was actually missing: EXECUTION, in three refusals

1. **ggml->HF name map**: the four `nextn.*` glue names had no glm5_next row in
   `hf_mapping.rs::resolve_ggml`, so `src.has("blk.45.nextn.eh_proj.weight")` answered
   false and the embedded-MTP loader silently loaded NO head (`break` at offset 0).
2. **Reference**: `execute_mtp` was handed the raw `[tokens, streams*hidden]` hc stream
   stack and refused every hyper-connections plan ("HyperConnections MTP fusion").
3. **Engine forward**: `mtp_head_forward_dev`'s attention dispatch had
   `Mixer::Mla(_) => mla_path_unimplemented(...)`.

All three are fixed in this lane (`lane/glm5-mtp-remint`); see `LANE.md` beside this
file. Do not cite the probe's "no mtp/nextn tensors" line further; a probe that greps
for family-conventional names must enumerate the checkpoint's own spelling first
(here: the appended layer index, `layers.{num_hidden_layers}`).
