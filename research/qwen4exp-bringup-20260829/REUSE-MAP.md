# qwen4_exp reuse-vs-new map (phase 3 scoping, surveyed 2026-08-29)

Survey of this tree @ 46aa1aa475. Anchors verified by the survey agent; re-verify at edit time.

| Component | Verdict | Anchor |
|---|---|---|
| model_type→Arch dispatch | EXTEND: add `Arch::Qwen4Exp` + `qwen4_exp`/`qwen4_exp_text` strings; `attention_gate_kind()` is compile-enforced (FusedQ — q_proj carries 2× width) | config.rs:8-146 |
| from_hf config parse | REUSE + family sub-config (indexer_*, hc_*, ngram_*, ple_*, full_attention_interval already read at config.rs:1516) | config.rs:1201+,2179-2237 |
| ModelPack + 7-gate ladder | COPY model_packs/qwen35/mod.rs (54 lines), register in PACKS | model_packs/mod.rs:117 |
| GDN 3:1 layer pattern | REUSE ArchGeometryTable::qwen35 pattern ((il+1)%interval==0 → full; MTP tail full) — qwen4_exp matches interval 4 exactly | config.rs:327-368,1659 |
| GDN plan + reference | REUSE (GatedDeltaNetPlan model_plan.rs:1111-1147; reference gated_delta_net lib.rs:4015). Geometry differs (48V/16QK/128) — kernel-check pins per class |
| QSA micro-block indexer | NEW plan arm. Closest prior art: dsv4 indexer — SAME ratio-4 blocks, SAME topk 512, per-block score+causal mask+topk (dsv4_forward.rs:967-1094, dsv4_gpu.rs). qwen4_exp differs: MQA 4Q/1K fused index_qk_proj + q/k layernorm, budget 2048 tokens, no 128-tok window ring (verify), no Hadamard/FP4-act (verify vs HF impl) |
| Tensor contract | REUSE builders: fused 3D experts = gemma4 SplitExpertGateUp shape (tensor_contract.rs:1038-1054) + hy3 stacked banks (1812); add qwen4_exp names |
| NVFP4 ST ingestion | REUSE AS-IS: modelopt schema source.rs:1087-1118, stacked expert banks find_nvfp4_stacked_native source.rs:1501 |
| MTP head | REUSE: compile auto-builds from num_nextn_predict_layers (model_plan.rs:493-509); runtime MtpHead hybrid.rs:2151. qwen4_exp MTP = full-attn block ✓ same as qwen35 declaration; NEW: its fc_embedding/fc_hidden are separate 2560×2560 (not concat eh_proj [2n,n]) — verify mapping |
| Router + shared_expert_gate | ⚠ CORRECTED 2026-08-29: router is SOFTMAX top-10 renormalized (norm_topk_prob) per SEMANTICS.md, not sigmoid — REUSE RouterPlan::Softmax; SharedMlpPlan.gated (sigmoid) 268-271 |
| Gated residual (hc 4-branch) | EXTEND ResidualTopology (model_plan.rs:310-323): Sinkhorn HyperConnections exists (dsv4, execute_hyper_layer lib.rs:2984); qwen4_exp = NEW gated variant: rank-320 input_mix down/up + block_inject [4,10240] + hc_norm, hc_count 4, elementwise read gate + per-branch scalar write gate |
| n-gram / PLE | NET-NEW (absent). Nearest relative: DSpark Markov head (tensor_contract.rs:113-123). 16-head 20M-vocab gather + conv/key/value proj block at layer idx 1 (0-based); 51B table, host-resident + gather-on-CPU candidate (mint surgery proved the pattern) |
| mrope interleaved [11,11,10] | NET-NEW (RopePlan has no axis concept, model_plan.rs:163; vision.rs:10 states no M-RoPE). Text-side positions likely plain for text-only serving — verify HF impl before building |
| Vision tower | REUSE pattern (vision.rs:24-35 consts are qwen3_5 dims — same depth 27/hidden 1152/patch 16/merge 2 in qwen4_exp config; pos_embed 2304 vs 48×48=2304 ✓); mrope question above gates it |
| Reference executor | EXTEND execute_layer (lib.rs:2578-2700) + deterministic_fixture (lib.rs:191); unsupported ops fail loudly by design |

Bring-up order (per docs/ONBOARDING.md phases 2-4, sized by the above):
1. Arch variant + pack + geometry table + from_hf sub-configs (mostly mechanical)
2. Tensor contract rows (census-gated from raw/census.tsv.gz)
3. Reference arms: gated-residual layer executor, QSA (dense fallback first: full attention over budget window? NO — QSA must be exact from day one, top-k is the semantic program), ngram/PLE block, MTP mapping
4. Goldens vs transformers main (5.16.0.dev0) on the mint box (bf16 CPU/offload forward = oracle)
