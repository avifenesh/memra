# memra code map for the glm5_next door (Explore sweep, 2026-08-27)

Key findings (full details in the sweep; line numbers @ main 941c774a68):

- Dispatch: worker.rs:6100-6203 — dir + is_dsv4_dir() → Dsv4Gpu; every other dir →
  HybridModel::load_from_source. Doors keyed on config.json.
- Arch table: memra-gguf/src/config.rs:62-85 from_hf_model_type; glm_moe_dsa →
  Arch::GlmDsa EXISTS (GLM-5.2 line, loader-only). attention_gate_kind (117-140)
  is exhaustive — new Arch variant fails compile until declared.
- ModelPlan: model_plan.rs:10-28; already has MlaAttentionPlan::{LatentKv,
  CompressedKv}, SparseIndexPlan::{None,Own,SharedFromPrevious},
  GatedDeltaNetPlan, RouterPlan::Sigmoid{selection_bias}, ResidualTopology::
  HyperConnections{streams,epsilon,sinkhorn_iterations}, MtpBlockPlan.
  compile_attention (989-1090) currently all-MLA when cfg.mla present — NO
  per-layer linear/MLA mix branch. ArchGeometryTable needed (config.rs:297-330).
- Packs: model_packs/mod.rs; glm_dsa/mod.rs:8-118 is the template (NativeReference,
  parity 0.005 + require_argmax, tiny_plan). Registration = pack entry, no trait.
- memra-reference (crates/memra-reference/src/lib.rs, 5753 lines) ALREADY has:
  hyper-connections+sinkhorn (246-300, 2493-2610, 2994-3100), GDN (744, 2666,
  3061), MLA LatentKv (794-870) / CompressedKv (878-960), sparse index Own/Shared
  (965-1010), router variants (1097), dspark drafter blocks.
- MLA CUDA forward: hybrid.rs:1822-1837 mla_forward_unimplemented() PANICS —
  glm-dsa is loader-only; serving needs the MLA kernel family (mla.rs is the CPU
  oracle, MlaDims::GLM52). DSA indexer scorer exists in cu/dsv4_gpu.cu:844.
- GDN kernels: cu/hybrid.cu (2608 lines; chunk family gdn_chunk_* 424-520,
  scan 389). hybrid.cu:325 has the only "KDA" mention (comment). NO KDA impl.
- MoE sigmoid noaux_tc: DONE end-to-end (RouterPlan::Sigmoid, exp_probs_b
  selection bias = e_score_correction_bias, cu/moe_router.cu:597).
- FP8 e4m3 + weight_scale_inv 128x128 block dequant at load: source.rs:1033-1050
  + 998-1005 (Qwen/DeepSeek lineage) — GLM FP8 should ride this. Raw-FP8
  st_dtype_to_ggml panics; the sibling-scale path is the entry.
- MTP: automatic from nextn_predict_layers (model_plan.rs:493-509); MtpHead.mixer
  is a Mixer — MLA-mixer MTP loads free (hits same missing CUDA forward).
- Vision: vision.rs qwen tower pattern; glm tower = new but standard.
- CLI: memra model scaffold/inspect/verify (memra-cli); generates census, plan,
  gates, capture-hf-oracle.py. docs/ONBOARDING.md phase table; gate ports
  18300-18399; execution_manifest.rs fails closed on missing kernel programs.

## Phase-1 work list (NativeReference), in order
1. Arch::Glm5Next + from_hf_model_type + parse + attention_gate_kind.
2. Glm5NextConfig typed struct (per-layer mixer list, KDA geometry, hc knobs,
   indexer kpool fields, MoE, nextn) — panic-with-field-name law.
3. model_plan: KDA plan type (extend GatedDeltaNetPlan or new KdaPlan: per-channel
   low-rank forget gate, beta head, fused conv, sigmoid gated o_norm, l2norm),
   per-layer mixed compile_attention, HyperConnections residual, StatePlan split,
   NoPE MLA (rope dim 0), kpool indexer plan fields.
4. tensor_contract additions (Kda*, kpool ape/gate) + hf_mapping names.
5. model_packs/glm5_next (tiny_plan: 1 KDA + 1 DSA layer + hc + MoE + MTP).
6. memra-reference: KDA op (chunk not needed — recurrent f32 is fine for
   reference) + kpool-compressed relu indexer + always-tail.
7. memra model scaffold/inspect/verify loop; argmax parity vs the pinned oracle
   bank from the bench box.
