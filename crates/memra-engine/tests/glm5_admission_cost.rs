//! ADMISSION-COST gates for the glm5_next latent/workspace accounting
//! (lane/glm5-gpf-workspace, 2026-08-30). CPU-only: every function under test is pure
//! arithmetic over a compiled `ModelPlan` — no CUDA context is created, so these run in
//! ordinary CI, not behind the rig lock.
//!
//! WHAT THIS HOLDS DOWN. The 262k 2-card cell
//! (`research/glm53-flash-bringup-20260827/262k-2card-20260830/LANE.md`) banked the
//! receipt line `request cost: ... = 0 B/token x ctx + 155MB fixed`: glm5_next's
//! per-token admission coefficient was literally zero because
//! `cache_bytes_per_token_for_plan` matched only `LayerKind::FullAttention` KV planes and
//! this family has none — its 34 KDA layers are `Recurrent` (correctly 0/token) and its
//! 11 MLA layers are `StatePlan::LatentKvCache`, which the sum silently skipped. Admission
//! therefore admitted prompts the device could never serve, and the failure surface was a
//! mid-stream engine OOM. The fix charges the latent plane exactly as `Cache::new_inner`
//! allocates it; these tests pin the formula on the SAME mini glm5_next config the chunked
//! prime and ppN gates load, so a plan-compilation drift moves this gate too.
//!
//! The server-side half (the cost model refusing the 262k cell's own rungs, red arm
//! included) lives in `memra-server`'s worker tests:
//! `hyper_prefill_workspace_makes_admission_see_the_262k_wall`.

use memra_engine::hybrid_forward::{HyperPrimeWorkspaceShape, hyper_prime_call_rows};
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::{ModelPlan, StatePlan};

/// The mini glm5_next config the chunked-prime and hyper-ppn gates use: layer 0 KDA
/// (`Recurrent`), layer 1 DSA (MLA `LatentKvCache` + k-pool indexer), kv_lora_rank 16,
/// index_head_dim 8, index_kpool 4.
fn mini_config_json() -> String {
    r#"{
      "model_type": "glm5_next_text",
      "num_hidden_layers": 2,
      "num_nextn_predict_layers": 0,
      "hidden_size": 128,
      "intermediate_size": 64,
      "vocab_size": 32,
      "max_position_embeddings": 4096,
      "rms_norm_eps": 1e-05,
      "hidden_act": "silu",
      "swiglu_limit": 1e30,
      "tie_word_embeddings": true,
      "hc_mult": 4,
      "hc_eps": 1e-06,
      "hc_sinkhorn_iters": 20,
      "mhc": true,
      "layer_types": ["linear_attention", "deepseek_sparse_attention"],
      "mlp_layer_types": ["dense", "dense"],
      "first_k_dense_replace": 2,
      "indexer_types": ["full", "full"],
      "linear_attn_config": {
        "num_heads": 1,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [0],
        "full_attn_layers": [1]
      },
      "num_attention_heads": 1,
      "num_key_value_heads": 1,
      "q_lora_rank": 16,
      "kv_lora_rank": 16,
      "qk_head_dim": 16,
      "qk_nope_head_dim": 16,
      "qk_rope_head_dim": 0,
      "v_head_dim": 16,
      "mla_use_nope": true,
      "index_n_heads": 1,
      "index_head_dim": 8,
      "index_topk": 8,
      "index_kpool": 4,
      "index_kpool_always_select_tail": true,
      "index_kpool_compress": true,
      "indexer_rope_interleave": true,
      "index_share_for_mtp_iteration": true,
      "n_routed_experts": 4,
      "num_experts_per_tok": 2,
      "moe_intermediate_size": 32,
      "n_shared_experts": 1,
      "scoring_func": "sigmoid",
      "topk_method": "noaux_tc",
      "routed_scaling_factor": 2.5,
      "norm_topk_prob": true,
      "n_group": 1,
      "topk_group": 1,
      "head_dim": 0,
      "attention_bias": false,
      "moe_router_dtype": "float32",
      "dtype": "bfloat16"
    }"#
    .to_string()
}

fn mini_config() -> ModelConfig {
    ModelConfig::from_hf(&HfConfig::parse(&mini_config_json()))
}

fn mini_plan(config: &ModelConfig) -> ModelPlan {
    memra_gguf::model_packs::for_config(config)
        .expect("glm5_next model pack matches the mini config")
        .compile_plan(config)
        .expect("mini glm5_next plan compiles")
}

/// Non-vacuity first (the wiring-assertions-match-prose law): the plan really carries a
/// `LatentKvCache` layer with the widths the formula below multiplies, so the gate cannot
/// pass by matching nothing.
#[test]
fn the_mini_plan_carries_the_latent_layer_the_formula_charges() {
    let config = mini_config();
    let plan = mini_plan(&config);
    let latent: Vec<_> = plan
        .layers
        .iter()
        .filter_map(|l| match l.state {
            StatePlan::LatentKvCache { width, index_width } => Some((width, index_width)),
            _ => None,
        })
        .collect();
    assert_eq!(
        latent,
        vec![(16, 16)],
        "one MLA layer, latent width = kv_lora_rank 16, index_width = 2 * index_head_dim 8"
    );
    assert_eq!(
        config.glm5.as_ref().map(|g| g.index_kpool),
        Some(4),
        "the pool the resident-key term divides by really is the config's"
    );
}

/// The latent coefficient mirrors `Cache::new_inner` + the engine's lazy pool-key plane:
/// `width * 4` (f32 latent row) + `index_head_dim * 4 / pool` (one resident key per pool).
/// AND it is the WHOLE coefficient for this family — the pre-lane FullAttention-only sum was
/// zero, which is the banked 0 B/token hole. Both halves asserted so neither can silently
/// regress: the formula (green) and the fact the old arithmetic missed it (the red the cell
/// paid for).
#[test]
fn latent_plan_layers_charge_the_cache_they_allocate() {
    let config = mini_config();
    let plan = mini_plan(&config);
    let n = plan.layers.len();
    let expected_latent = 16 * 4 + (16 / 2) * 4 / 4; // 64 + 8 = 72 B/token
    assert_eq!(
        memra_engine::cache::latent_kv_bytes_per_token_for_plan(&config, &plan, 0, n),
        expected_latent,
    );
    let total = memra_engine::cache::cache_bytes_per_token_for_plan(&config, &plan, 0, n);
    assert_eq!(
        total, expected_latent,
        "for glm5_next the latent term IS the whole coefficient: the FullAttention-only \
         share is the pre-lane 0 B/token (the 262k cell's banked receipt line)"
    );
    // The per-stage split contract PP admission relies on: the KDA layer's range charges
    // nothing, the MLA layer's range charges everything.
    assert_eq!(
        memra_engine::cache::cache_bytes_per_token_for_plan(&config, &plan, 0, 1),
        0,
        "the Recurrent (KDA) layer stays 0 B/token"
    );
    assert_eq!(
        memra_engine::cache::cache_bytes_per_token_for_plan(&config, &plan, 1, n),
        expected_latent,
    );
}

/// The workspace charge is CHUNK-bounded: past one prime chunk, the per-call term stops
/// growing with the request (the whole point of the ppN chunk walk this lane ports), while
/// the ctx-coupled score plane and the prompt-long hiddens keep scaling — exactly the terms
/// that stay per-request by design. Naked schedule (no MEMRA_PRIME_CHUNK / PP env in CI):
/// fixed 4096-token ranges with a sub-PRIME_MIN_T tail merged.
#[test]
fn the_admission_workspace_charge_is_chunk_bounded() {
    let shape = HyperPrimeWorkspaceShape {
        // The arm-3 workspace is charged only while MEMRA_B200_PRIME_V2 is open; this fixture
        // pins the door-shut arithmetic, so it is 0 here by construction.
        bgemm_workspace_bytes: 0,
        chunk_token_bytes: 1_000,
        prompt_bytes_per_token: 10,
        kpool_score_pool: 4,
        n_layers: 45,
        gdn_grid: false,
    };
    let rows_16k = hyper_prime_call_rows(16_384, 45, false);
    let rows_262k = hyper_prime_call_rows(262_144, 45, false);
    assert!(
        rows_16k <= 4096 + 16 && rows_262k <= 4096 + 16,
        "the naked schedule bounds every call at one chunk (+ tail merge): got {rows_16k} / {rows_262k}"
    );
    let at_16k = shape.admission_bytes(16_384);
    let at_262k = shape.admission_bytes(262_144);
    // Identity, term by term: chunk workspace + score plane + hiddens.
    assert_eq!(
        at_16k,
        1_000 * rows_16k + rows_16k * (16_384 / 4) * 4 + 10 * 16_384
    );
    assert_eq!(
        at_262k,
        1_000 * rows_262k + rows_262k * (262_144 / 4) * 4 + 10 * 262_144
    );
    // The chunk term did NOT scale 16x with the request; only the ctx-coupled terms did.
    assert_eq!(
        rows_16k, rows_262k,
        "call rows are the chunk, not the request"
    );
}
