//! Reference-oracle acceptance gate for glm5_next's FUSED MoE epilogue — the sigmoid
//! `noaux_tc` router, the PRE-clamped SwiGLU and the per-expert macro fold, collapsed into one
//! launch pair per token-layer (`MEMRA_MOE_FUSED_EPI`).
//!
//! Truth is `memra_reference::execute`, the unfused f32 executor. Every arm below is accepted
//! only by agreeing with it. This model's bring-up defects were all found by the reference
//! disagreeing with the engine, never by the engine looking healthy
//! (`research/glm53-flash-bringup-20260827/BRINGUP.md`).
//!
//! WHY THIS GATE EXISTS, BEFORE THE FUSION. glm5_next is denied every fused MoE arm the engine
//! has, by three independent predicates, and each one is denying a REAL semantic difference:
//!
//!   * `cfg.sigmoid_router().is_none()` — `moe_ffn_pairs` / `moe_ffn_dev` / the grouped-decode
//!     pair all route through the fused SOFTMAX router (`moe_router_topk`), which has no
//!     `e_score_correction_bias` and normalizes to 1 instead of `routed_scaling_factor` 2.5.
//!     Letting a sigmoid arch in silently picks different experts (the M3 gate-MISMATCH
//!     74602-vs-92 lesson, `hybrid_forward.rs:6733`).
//!   * `!cfg.swiglu_clamped_at(il)` — every fused epilogue in the family hardcodes plain
//!     `silu(gate)*up`. glm5_next is `ActivationPlan::SwiGluPreClamped`:
//!     `silu(gate.min(l)) * up.clamp(-l, l)`. Feeding a PRE limit to a POST epilogue compiles,
//!     runs, and returns plausible-but-wrong logits (`hybrid_forward.rs:7937`).
//!   * `no_exp_macros` — the compressed-tensors NVFP4 class carries a per-expert
//!     `weight_scale_2` that lives OUTSIDE the block bytes. Dropping it is a ~3e4x error that is
//!     fluent and invisible (measured garbage, 2026-07-16).
//!
//! A fused arm for this arch has to get all three right at once. So the gate is written first,
//! and it is written to fail on each of them separately.
//!
//! WHAT IS COMPARED. The whole model's logits — which is the MoE block's complete output
//! (routed sum x `routed_scaling_factor`, PLUS the shared expert) carried through the mHC
//! residual to the head. Comparing only the routed sum would make `shared-expert-dropped`
//! structurally impossible to turn red, so the comparison point is the model, not the branch.
//!
//! THE FOUR RED ARMS, and why each one is a wrong answer someone would actually ship:
//!   * `post-for-pre-clamp` — step35's `min(silu(gate), l) * clamp(up, +-l)`. This is what a
//!     one-line "make the fused epilogue read glm5's limit" fix produces, and it is the exact
//!     semantic difference the brief names.
//!   * `plain-swiglu` — no clamp at all: what every fused epilogue in the family does today,
//!     i.e. what happens if the deny predicate is simply deleted.
//!   * `shared-expert-dropped` — the fused pair covers the ROUTED experts only; the shared
//!     expert is a separate branch that must still run and still be summed in.
//!   * `softmax-for-sigmoid` — the router half, denied by the same predicate the fusion has to
//!     relax.
//! Each is a plan mutation read by the REFERENCE only; the engine always runs the real plan, so
//! every distance is measured against the SAME engine output the passing assertion uses
//! (GATE:pin-against-truth — the truth arm is anchored outside the feature under test).
//!
//! A FIFTH RED ARM CANNOT BE EXPRESSED AS A PLAN MUTATION. The per-expert macro scale is a
//! checkpoint artifact, not a `ModelPlan` knob, so `macro-scale-dropped` is minted into the
//! FIXTURE instead: `wrong_macro_plane_fails_the_gate` perturbs one `weight_scale_2` in the
//! bytes the engine loads and requires the comparison against the unperturbed reference to
//! break.
//!
//! NON-VACUITY IS ENFORCED, NOT ASSUMED, in three places:
//!   * `the_epilogue_program_actually_binds` measures all four plan mutations on the REFERENCE
//!     ALONE, with no CUDA. A fixture that drifted into agreement fails on a GPU-less machine
//!     rather than turning the GPU gates into tautologies. This is also the arm that makes the
//!     gate's RED evidence bankable without a GPU.
//!   * `LIMIT` is 0.75, not the shipped 10.0, and `MLP_NORM_GAIN` scales each layer's pre-MLP
//!     RMSNorm gamma so both projections cross +-`LIMIT` in both signs. At L=10 the PRE and POST
//!     forms differ by at most `L*(1-sigmoid(L))` = 4.5e-4 per element, small enough to hide
//!     inside TOL; at L=0.75 the same bound is 0.23. The FORM is what is under test, and the
//!     (limit, gain) pair was picked by MEASURING five candidates on the smallest red arm — see
//!     `LIMIT`.
//!   * `the_macro_plane_is_live` pins that the macro scales are not all 1.0 —
//!     `HostExps::stacked_macros` returns `None` for an all-ones vector, which would drop the
//!     macro plane entirely and leave the macro arm gated against a no-op.
//!
//! AND THE GREEN ARM IS DISPATCH-BOUND. `fused_epilogue_matches_the_reference` asserts on
//! `memra_engine::moe_fused_epilogue_dispatches()`, a counter incremented at the fused arm's own
//! invocation — not on a comment, not on liveness (LAW:wiring-assertions-match-prose). It measured
//! 51 on the workload below, against 0 for the unfused control.
//!
//! THE GATE HAS FAILED, ON PURPOSE, ON THE ENGINE AND NOT ONLY ON THE REFERENCE. The four plan
//! mutations never touch the engine, so they cannot by themselves show that a WRONG FUSION is
//! caught. Two were therefore built and run (receipts:
//! `research/glm53-flash-bringup-20260827/moe-epilogue-receipts/gpu-gates-RED-*.txt`):
//!   * the fused kernel's epilogue swapped to step35's POST form — `fused` 1.747e-1,
//!     `the_two_arms_agree` 1.251e-1 with 32/32 elements differing in bits;
//!   * the fused arm's macro scales replaced with 1.0 — `fused` 1.624e-1, `the_two_arms_agree`
//!     7.698e-2 with 32/32 elements differing in bits.
//! In both the unfused control stayed at its 4.822e-3 floor and PASSED, so each failure is
//! attributable to the fused arm rather than to the harness.
//!
//! NOTE ON THE OBSERVATION MODES. `MEMRA_MOE_STATS` / `MEMRA_MOE_TRACE` /
//! `MEMRA_MOE_WEIGHT_TRACE` / `MEMRA_MOE_INPUT_TRACE_DIR` set `observe_routes` in
//! `moe_ffn_inner`, which DIVERTS dispatch to the host-routed path. Proving the fused arm ran by
//! setting one of them would test the wrong arm. Hence the dedicated counter.
//!
//! FIXTURE SHAPE. Real glm5_next routing constants (`routed_scaling_factor` 2.5,
//! `norm_topk_prob`, sigmoid `noaux_tc`, 1 shared expert, PRE-clamped SwiGLU), shrunk only in
//! width. Routed banks are REAL NVFP4 blocks in memra's internal `block_nvfp4` layout, with a
//! live `<stem>.scale` macro plane — the class glm5_next actually serves on, and the only expert
//! encoding `q8_expert_supported` admits among the ones this fixture could mint (Q8_0 is not on
//! that list, so a Q8_0 bank would never reach a fused q8 arm at all). The reference reads the
//! NVFP4-GRID-SNAPPED values times their macro, so the weight encoding costs no parity and the
//! only floor left is the q8_1 ACTIVATION quantization the expert matvecs do.
//!
//! SCOPE — what this gate does NOT prove. It runs a 2-layer fixture, not the 190.7 GB artifact.
//! It proves the fused epilogue is the same PROGRAM as the reference for this arch's router,
//! clamp and macro plane. It proves nothing about throughput, about launch counts (those are a
//! source/profile claim, and the rig is exactness-only by law), or about any layer width other
//! than the fixture's.
//!
//! GPU-gated (`#[ignore]`); rig law = exactness only, never timing. Run under
//! `flock /tmp/memra-5090.lock` with `NVIDIA_TF32_OVERRIDE=0` and `-- --ignored`.

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::{ActivationPlan, MlpPlan, ModelPlan, RouterPlan};
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, LayerTensor, OutputHead, TensorContract, TensorId,
    TensorMatch,
};
use memra_reference::{ReferenceTensor, deterministic_fixture};
use std::borrow::Cow;
use std::collections::BTreeMap;

const VOCAB: u32 = 32;
const HIDDEN: usize = 128;

/// GLM-5.3-Flash's real routing shape, shrunk only in width.
const EXPERTS: usize = 8;
const TOP_K: usize = 3;
const SCALING: f32 = 2.5;
const NORM: bool = true;

/// `moe_intermediate_size`. NVFP4 requires `in_f % 64 == 0` on EVERY projection, and `down`'s
/// `in_f` IS this dimension — the router gate's 32 would make the bank unrepackable.
const MOE_FF: usize = 64;

/// The clamp under gate. Deliberately NOT glm5_next's shipped 10.0: at limit L the PRE and POST
/// forms differ by at most `L*(1 - sigmoid(L))` per element, 4.5e-4 at L=10 — small enough to
/// hide inside this gate's tolerance. At L=0.75 the same bound is 0.23. The FORM is under test;
/// the shipped limit is pinned by `crates/memra-gguf/src/config.rs`'s own parser tests.
///
/// 0.75 with `MLP_NORM_GAIN` 2.0 was CHOSEN BY MEASUREMENT over four other pairs, on the
/// `post-for-pre-clamp` margin (the smallest of the five red arms and therefore the one that sets
/// the gate's resolution): 1.5/6.0 -> 7.743e-2, 0.4/2.0 -> 8.357e-2, 0.4/6.0 -> 1.172e-1,
/// 0.75/6.0 -> 1.550e-1, 0.75/2.0 -> 1.575e-1. The low gain also buys the tolerance: see `TOL`.
const LIMIT: f32 = 0.75;

/// Pre-MLP RMSNorm gamma multiplier — the non-vacuity construction. RMSNorm renormalizes the
/// residual stream, so inflating the embedding would do nothing; the gain has to ride the weight
/// that feeds the FFN. `the_epilogue_program_actually_binds` is what proves it is large enough,
/// and `TOL` records what makes it small enough.
const MLP_NORM_GAIN: f32 = 2.0;

/// Selection-bias amplifier, so `softmax-for-sigmoid` is not gated against a router whose bias
/// never changes a pick. Same construction and same reason as `glm5_routed_router_gpu.rs`.
const ROUTER_BIAS_GAIN: f32 = 12.0;

/// SLRU slots, chosen to sit between the two bounds that matter, so BOTH properties hold at once:
///   * at least `TOP_K * 3` = 9, the fused arm's fail-closed capacity floor — below it
///     `moe_fused_epi_token_q8` refuses (an admission would evict one of the token's own blocks)
///     and the arm would never fire, silently turning every GPU gate here into an unfused run;
///   * below the layer's live block count `EXPERTS * 3` = 24, so evictions RECUR after warm-up
///     and the arm is exercised with real MISSES rather than only on a warm layer — the property
///     that distinguishes it from gdec, which cannot fire on a miss at all.
/// `MoeCache::new` floors the count at 8.
const SLOTS: usize = 12;

/// Scale-relative bound, CALIBRATED FROM THE MEASURED UNFUSED CONTROL — the shipped sequential
/// program, which this lane did not touch and which `glm5_routed_router_gpu.rs` and
/// `swiglu_preclamp_gpu.rs` gate independently. Its worst row over the whole workload is
/// **4.822e-3** (5090, TF32 off, 2026-08-28, `decode step 2`), and the fused arm measures the
/// IDENTICAL number, bit for bit, so the distance is the fixture's floor and not the fusion's.
///
/// The floor is the q8_1 ACTIVATION quantization the expert matvecs do against an f32 reference;
/// the weights cost nothing because both sides read the NVFP4-grid-snapped values. It is an order
/// above `glm5_routed_router_gpu.rs`'s 1.858e-4 because this fixture deliberately drives the
/// clamp: `MLP_NORM_GAIN` amplifies the MoE branch against the residual stream, so the branch's
/// own error shows up amplified in the logits. Measured directly — at gain 6.0 the same control
/// sat at 1.918e-2, and dropping to 2.0 moved it to the current figure while every mutation
/// stayed above 1.5e-1.
///
/// 1e-2 is 2.1x the measured control and 5.3x below the smallest mutation ROW
/// (`softmax-for-sigmoid`, closest row 5.301e-2). The upper separation is what decides whether
/// the gate binds; the lower margin is deliberately thin, because a control that drifts up is a
/// finding. Calibrate downward, never upward: if it drifts, find out why rather than raising
/// this number.
const TOL: f32 = 1e-2;

/// Floor for "this wrong epilogue program fails by a wide margin". Measured margins are printed
/// by `the_epilogue_program_actually_binds`; this floor must sit well below the smallest of them
/// and well above TOL.
const MUTATION_FLOOR: f32 = 3e-2;

/// Per-expert `weight_scale_2` macro scales, 0.5 .. 0.8 and 1.2 .. 1.5.
///
/// Deliberately NOT all 1.0: `stacked_macros` returns `None` for an all-ones vector, which would
/// drop the macro plane entirely and leave the macro arm unable to bind.
///
/// And deliberately WIDE. The first version of this plane was 0.85 .. 1.06, and
/// `wrong_macro_plane_fails_the_gate` MEASURED 2.549e-2 against a 3.0e-2 floor on it — the
/// mutation was real but too small to clear the bar, because dropping a 0.97 scale is a 3% error
/// on one expert's gate rows. Every scale here sits at least 0.2 clear of 1.0, so flattening the
/// plane is a 1.25x-to-2x error on EVERY expert, while each value stays inside a range a real
/// `weight_scale_2` occupies.
fn macro_scales() -> Vec<f32> {
    (0..EXPERTS)
        .map(|e| {
            // Two bands with 1.0 excluded from both: a linear ramp through 1.0 puts some expert
            // at ~1.0, and flattening the plane would then be a no-op for that expert.
            if e < EXPERTS / 2 {
                0.5 + 0.1 * e as f32
            } else {
                1.2 + 0.1 * (e - EXPERTS / 2) as f32
            }
        })
        .collect()
}

/// GPU tests serialize on one device, and these arms additionally mutate PROCESS-GLOBAL env vars
/// the engine reads at load time and per forward.
fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// cuBLASLt f32 compute rides TF32 on Blackwell by default — right for serving, wrong for a
/// parity gate. The driver reads this at CUDA init, so it must be set before the first
/// `Engine::new` in the process; `call_once` serializes every test thread behind the write.
fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call has been made and no Engine handed out in this process yet,
            // and call_once serializes every test thread behind this write.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
    });
}

/// A glm5_next trunk with one dense layer and one routed-MoE layer, expressed the only way the
/// engine will accept: a real `config.json` through the real `HfConfig`/`ModelConfig` path,
/// compiled by the real glm5_next model pack. `HybridModel::load_from_source` compiles the plan
/// from `src.config()`, so a hand-built `ModelPlan` could not reach it.
///
/// `head_dim` is 128 because that is the only width `memra_kda_scan_s128` is instantiated for.
/// The MLA/DSA fields are required by the glm5_next config parser and are inert here: no layer
/// in `layer_types` selects them.
fn mini_config_json() -> String {
    format!(
        r#"{{
      "model_type": "glm5_next_text",
      "num_hidden_layers": 2,
      "num_nextn_predict_layers": 0,
      "hidden_size": {HIDDEN},
      "intermediate_size": 64,
      "vocab_size": 32,
      "max_position_embeddings": 512,
      "rms_norm_eps": 1e-05,
      "hidden_act": "silu",
      "swiglu_limit": {LIMIT},
      "tie_word_embeddings": true,
      "hc_mult": 4,
      "hc_eps": 1e-06,
      "hc_sinkhorn_iters": 20,
      "mhc": true,
      "layer_types": ["linear_attention", "linear_attention"],
      "mlp_layer_types": ["dense", "sparse"],
      "first_k_dense_replace": 1,
      "indexer_types": ["full", "full"],
      "linear_attn_config": {{
        "num_heads": 1,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [0, 1],
        "full_attn_layers": []
      }},
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
      "n_routed_experts": {EXPERTS},
      "num_experts_per_tok": {TOP_K},
      "moe_intermediate_size": {MOE_FF},
      "n_shared_experts": 1,
      "scoring_func": "sigmoid",
      "topk_method": "noaux_tc",
      "routed_scaling_factor": {SCALING},
      "norm_topk_prob": {NORM},
      "n_group": 1,
      "topk_group": 1,
      "head_dim": 0,
      "attention_bias": false,
      "moe_router_dtype": "float32",
      "dtype": "bfloat16"
    }}"#
    )
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

// ---------------------------------------------------------------------------------------------
// The wrong epilogue programs. Only the REFERENCE reads these — the GPU always runs the real
// plan — so each answers "what would the reference say if the engine had fused THIS?" against
// one fixed set of weights.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mutation {
    /// step35's POST form, `min(silu(gate), l) * clamp(up, +-l)`. What a one-line "teach the
    /// fused epilogue glm5's limit" fix produces, and the exact pre-vs-post semantic difference
    /// this family is prone to.
    PostClamp,
    /// No clamp at all — what every fused epilogue in the family hardcodes today, i.e. what
    /// happens if the `!swiglu_clamped_at` deny predicate is simply relaxed.
    PlainSwiglu,
    /// The shared expert dropped. The fused pair covers the ROUTED experts only; the shared
    /// branch must still run and still be summed into the block's output.
    NoSharedExpert,
    /// The router half: softmax scores, no `e_score_correction_bias`, weights summing to 1
    /// instead of `routed_scaling_factor`.
    SoftmaxRouter,
}

impl Mutation {
    fn label(self) -> &'static str {
        match self {
            Mutation::PostClamp => "post-for-pre-clamp",
            Mutation::PlainSwiglu => "plain-swiglu (no clamp)",
            Mutation::NoSharedExpert => "shared-expert-dropped",
            Mutation::SoftmaxRouter => "softmax-for-sigmoid",
        }
    }

    /// The same plan with exactly one property of the MoE epilogue replaced.
    fn apply(self, plan: &ModelPlan) -> ModelPlan {
        let mut mutated = plan.clone();
        let mut seen = false;
        for layer in &mut mutated.layers {
            match (&mut layer.mlp, self) {
                (MlpPlan::Moe(moe), Mutation::PostClamp) => {
                    moe.activation = ActivationPlan::SwiGluClamped { limit: LIMIT };
                    seen = true;
                }
                (MlpPlan::Moe(moe), Mutation::PlainSwiglu) => {
                    moe.activation = ActivationPlan::Silu;
                    seen = true;
                }
                (MlpPlan::Moe(moe), Mutation::NoSharedExpert) => {
                    assert!(
                        moe.shared.is_some(),
                        "the mini plan must carry a shared expert, or this mutation is a no-op"
                    );
                    moe.shared = None;
                    seen = true;
                }
                (MlpPlan::Moe(moe), Mutation::SoftmaxRouter) => {
                    moe.router = RouterPlan::Softmax;
                    seen = true;
                }
                _ => {}
            }
        }
        assert!(seen, "the mini plan must carry a routed-MoE layer");
        mutated
    }
}

/// The macro-plane mutation, which no `ModelPlan` knob can express: it is a checkpoint artifact.
/// Minted into the bytes the ENGINE loads, and measured against the unperturbed reference.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FixtureMutation {
    /// The honest fixture.
    None,
    /// The gate bank's WHOLE `weight_scale_2` plane flattened to 1.0 — an epilogue that never
    /// folds the gate macro at all. That is the shape of the real failure (a kernel either folds
    /// or it does not; it does not forget one expert), and it is also what the engine SEES: an
    /// all-ones plane makes `HostExps::stacked_macros` answer `None`, so the macro plane is gone
    /// rather than merely wrong. The reference still carries it, so the comparison must break.
    ///
    /// Measured on the earlier one-expert form of this mutation: 2.549e-2 at the original
    /// 0.85..1.06 plane and 2.100e-2 after widening it — both under the 3.0e-2 floor. Perturbing
    /// a single scale is simply not the failure this class produces, and it could not clear the
    /// bar. The whole-plane drop is both stronger and more honest.
    MacroDropped,
}

// ---------------------------------------------------------------------------------------------
// Fixture.
// ---------------------------------------------------------------------------------------------

/// The three stacked routed-expert slabs, and each one's `in_f` — the row length the NVFP4
/// blocking must align to. Returns `None` for every other tensor.
fn expert_bank_in_f(id: &TensorId) -> Option<usize> {
    match id {
        TensorId::Layer {
            tensor: LayerTensor::MoeExpertGateBank | LayerTensor::MoeExpertUpBank,
            ..
        } => Some(HIDDEN),
        TensorId::Layer {
            tensor: LayerTensor::MoeExpertDownBank,
            ..
        } => Some(MOE_FF),
        _ => None,
    }
}

fn is_expert_bank(id: &TensorId) -> bool {
    expert_bank_in_f(id).is_some()
}

/// NVFP4 round trip, row by row: mint the block memra actually stores, then read it back.
/// Snapping the fixture ONTO that grid keeps the reference and the GPU reading one set of
/// numbers once the bank is encoded — otherwise 4-bit weight quantization error would sit on top
/// of the epilogue this gate is trying to measure. Rows are `in_f` long and `in_f % 64 == 0`, so
/// the 64-element NVFP4 blocking never straddles a row boundary.
fn snap_to_nvfp4(data: &mut [f32], in_f: usize) {
    assert_eq!(in_f % 64, 0, "NVFP4 requires in_f % 64 == 0, got {in_f}");
    assert_eq!(
        data.len() % in_f,
        0,
        "expert bank of {} elements is not a whole number of {in_f}-wide rows",
        data.len()
    );
    for row in data.chunks_exact_mut(in_f) {
        let bytes = memra_gguf::nvfp4_repack::f32_to_nvfp4(row);
        let back = memra_gguf::nvfp4_repack::dequant_gguf_row(&bytes, in_f);
        assert_eq!(back.len(), in_f);
        row.copy_from_slice(&back);
    }
}

/// The weights the SOURCE serves: grid-snapped expert banks (what the NVFP4 blocks encode
/// exactly), amplified pre-MLP norms, amplified router selection bias.
fn engine_fixture(plan: &ModelPlan) -> BTreeMap<TensorId, ReferenceTensor> {
    let mut weights = deterministic_fixture(plan)
        .expect("deterministic glm5_next fixture")
        .weights;

    // Non-vacuity: drive both FFN projections across +-LIMIT in both signs.
    for layer in &plan.layers {
        let id = TensorId::Layer {
            index: layer.index,
            tensor: LayerTensor::PreMlpNorm,
        };
        let gamma = weights
            .get_mut(&id)
            .unwrap_or_else(|| panic!("fixture has no pre-MLP norm for layer {}", layer.index));
        for v in &mut gamma.data {
            *v *= MLP_NORM_GAIN;
        }
    }

    // Non-vacuity: make `e_score_correction_bias` change WHO is picked, so the router half of
    // the fusion is gated against a bias that is doing work.
    let bias_id = router_bias_id(plan);
    let bias = weights
        .get_mut(&bias_id)
        .expect("fixture mints e_score_correction_bias for a selection-bias router");
    for v in &mut bias.data {
        *v *= ROUTER_BIAS_GAIN;
    }

    for (id, tensor) in weights.iter_mut() {
        if let Some(in_f) = expert_bank_in_f(id) {
            snap_to_nvfp4(&mut tensor.data, in_f);
        }
    }
    weights
}

/// The weights the REFERENCE reads: the engine fixture with each routed expert's macro scale
/// folded into its rows, because that is what the engine computes — the block bytes carry the
/// grid-snapped value and `HostExps::macros` multiplies the per-expert `weight_scale_2` back in
/// after the matmul. The reference has no macro plane, so the product has to be in its numbers.
///
/// `mutation` mirrors what the SOURCE will do to the macro plane, so the reference stays the
/// honest program in every arm: under `MacroDropped` the engine folds a 1.0 the reference still
/// carries, and the comparison must break.
fn reference_fixture(
    plan: &ModelPlan,
    engine: &BTreeMap<TensorId, ReferenceTensor>,
) -> BTreeMap<TensorId, ReferenceTensor> {
    let macros = macro_scales();
    let mut weights = engine.clone();
    let mut folded = 0usize;
    for (id, tensor) in weights.iter_mut() {
        let Some(_) = expert_bank_in_f(id) else {
            continue;
        };
        assert_eq!(
            tensor.data.len() % EXPERTS,
            0,
            "expert bank {id:?} is not divisible into {EXPERTS} experts"
        );
        let per_expert = tensor.data.len() / EXPERTS;
        for (e, chunk) in tensor.data.chunks_exact_mut(per_expert).enumerate() {
            for v in chunk.iter_mut() {
                *v *= macros[e];
            }
        }
        folded += 1;
    }
    assert_eq!(
        folded, 3,
        "expected the gate/up/down routed banks to fold macros, folded {folded}"
    );
    let _ = plan;
    weights
}

fn router_bias_id(plan: &ModelPlan) -> TensorId {
    for layer in &plan.layers {
        if matches!(layer.mlp, MlpPlan::Moe(_)) {
            return TensorId::Layer {
                index: layer.index,
                tensor: LayerTensor::MoeRouterBias,
            };
        }
    }
    panic!("the mini plan must carry a routed-MoE layer");
}

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
    ggml_type: GgmlType,
}

/// Serves the fixture under the contract's ggml names. Must answer `config()`:
/// `HybridModel::load_from_source*` compiles the plan from it.
struct FixtureSource {
    config: ModelConfig,
    tensors: BTreeMap<String, OwnedTensor>,
}

impl TensorSource for FixtureSource {
    fn config(&self) -> ModelConfig {
        self.config.clone()
    }
    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        let t = self.tensors.get(name)?;
        Some(TensorView {
            bytes: Cow::Borrowed(&t.bytes),
            ggml_type: t.ggml_type,
            ne: t.ne.clone(),
        })
    }
}

/// Build the source. Routed banks are minted as REAL NVFP4 blocks in memra's internal
/// `block_nvfp4` layout (36 B per 64 elements) — the same layout `repack_modelopt_to_gguf`
/// produces — and each bank gets its `<stem>.scale` sibling (F32, one value per expert), which
/// is how `HostExps::stacked_macros` finds the macro plane.
fn fixture_source(
    config: &ModelConfig,
    plan: &ModelPlan,
    weights: &BTreeMap<TensorId, ReferenceTensor>,
    mutation: FixtureMutation,
) -> FixtureSource {
    let contract = TensorContract::for_plan(
        plan,
        CheckpointDialect::Gguf,
        ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        },
    )
    .expect("contract for the mini glm5_next plan");
    assert!(
        contract.requirements.iter().any(|r| matches!(
            r.id,
            TensorId::Layer {
                tensor: LayerTensor::MoeRouterBias,
                ..
            }
        )),
        "the contract must require e_score_correction_bias, or the bias is never served"
    );

    let mut tensors: BTreeMap<String, OwnedTensor> = BTreeMap::new();
    let mut bank_stems: Vec<String> = Vec::new();

    for req in contract
        .requirements
        .iter()
        .filter(|r| r.required || weights.contains_key(&r.id))
    {
        let tensor = weights
            .get(&req.id)
            .unwrap_or_else(|| panic!("reference fixture is missing {:?}", req.id));
        let elements: usize = req.shape.iter().map(|&d| d as usize).product();
        assert_eq!(
            elements,
            tensor.data.len(),
            "fixture {:?} has {} elements, contract requires {elements}",
            req.id,
            tensor.data.len()
        );
        let expert_bank = is_expert_bank(&req.id);
        let (bytes, ggml_type) = if expert_bank {
            (
                memra_gguf::nvfp4_repack::f32_to_nvfp4(&tensor.data),
                GgmlType::NVFP4,
            )
        } else {
            (
                tensor.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                GgmlType::F32,
            )
        };
        let names = match req.match_mode {
            TensorMatch::OneOf => &req.names[..1],
            TensorMatch::All => req.names.as_slice(),
        };
        for name in names {
            if expert_bank && let Some(stem) = name.strip_suffix(".weight") {
                bank_stems.push(stem.to_string());
            }
            tensors.insert(
                name.clone(),
                OwnedTensor {
                    bytes: bytes.clone(),
                    ne: req.shape.clone(),
                    ggml_type,
                },
            );
        }
    }

    assert_eq!(
        bank_stems.len(),
        3,
        "expected exactly gate/up/down routed-expert banks, got {bank_stems:?}"
    );

    for stem in &bank_stems {
        let mut macros = macro_scales();
        if mutation == FixtureMutation::MacroDropped && stem.ends_with("ffn_gate_exps") {
            // The failure this class is prone to: the gate macro plane never folded at all.
            macros.fill(1.0);
        }
        tensors.insert(
            format!("{stem}.scale"),
            OwnedTensor {
                bytes: macros.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ne: vec![EXPERTS as u64],
                ggml_type: GgmlType::F32,
            },
        );
    }

    FixtureSource {
        config: config.clone(),
        tensors,
    }
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared slices differ in length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn scale_of(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6)
}

fn relative(got: &[f32], want: &[f32]) -> f32 {
    maxdiff(got, want) / scale_of(want)
}

/// Compare a whole run against the reference: print EVERY row's distance first, then assert on
/// the worst. Asserting row by row aborts on the first failure and hides the shape — which is
/// exactly what happened on this gate's first GPU run, where one printed row could not say
/// whether the distance was the fused arm's or the fixture's.
fn check_all(label: &str, got: &[(String, Vec<f32>)], want: &[Vec<f32>]) {
    assert_eq!(got.len(), want.len(), "{label}: row count mismatch");
    let mut worst = 0.0f32;
    let mut worst_row = String::new();
    for ((name, g), w) in got.iter().zip(want) {
        assert!(
            g.iter().all(|v| v.is_finite()),
            "{label} {name}: GPU output has non-finite values"
        );
        let rel = relative(g, w);
        println!("{label} {name}: relative maxdiff {rel:.3e} (tol {TOL:.1e})");
        if rel > worst {
            worst = rel;
            worst_row = name.clone();
        }
    }
    println!("{label}: WORST {worst:.3e} on `{worst_row}` (tol {TOL:.1e})");
    assert!(
        worst <= TOL,
        "{label}: worst GPU vs reference relative maxdiff {worst:.3e} on `{worst_row}` \
         (tol {TOL:.1e})"
    );
}

fn tokens(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as u32) % VOCAB
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Host-only gates. No CUDA. These are the ones that carry the RED evidence: they fail on a
// GPU-less machine the moment a mutation stops binding.
// ---------------------------------------------------------------------------------------------

/// The plan the engine will fuse must be the plan this gate thinks it is fusing: PRE-clamped
/// SwiGLU, sigmoid `noaux_tc` routing, a live shared expert, and a macro-carrying expert class.
/// Every fused-arm predicate in `hybrid_forward` keys off these, so pinning them is half of what
/// is under test.
#[test]
fn the_plan_declares_the_epilogue_under_gate() {
    let config = mini_config();
    let plan = mini_plan(&config);

    let MlpPlan::Moe(moe) = &plan.layers[1].mlp else {
        panic!("layer 1 must be the routed-MoE branch");
    };
    assert_eq!(
        moe.activation,
        ActivationPlan::SwiGluPreClamped { limit: LIMIT },
        "the fused epilogue must be gated against the PRE form; SwiGluClamped is step35's POST \
         form and produces plausible-but-wrong logits above the limit"
    );
    assert_eq!(
        moe.router,
        RouterPlan::Sigmoid {
            normalize_selected: NORM,
            scaling_factor: SCALING,
            selection_bias: true,
        },
        "the fused epilogue must route the noaux_tc recipe: sigmoid scores, selection-only bias, \
         sum-normalized selected weights, x routed_scaling_factor"
    );
    assert!(
        moe.shared.is_some(),
        "the fixture must carry a shared expert, or `shared-expert-dropped` cannot turn red"
    );
    assert_eq!(moe.expert_count as usize, EXPERTS);
    assert_eq!(moe.experts_per_token as usize, TOP_K);

    // The engine-side accessors every dispatch predicate consults, on the same config.
    assert_eq!(
        config.sigmoid_router(),
        Some((SCALING, NORM)),
        "ModelConfig::sigmoid_router() must answer glm5_next's routed_scaling_factor and \
         norm_topk_prob — the fused arm's predicate keys off it"
    );
    // The slot pin is part of the gate, not a tuning knob: outside this window the GPU arms
    // below are either impossible (the fused arm refuses) or vacuous (no misses ever recur).
    assert!(
        SLOTS >= TOP_K * 3,
        "SLOTS={SLOTS} is below the fused arm's capacity floor of {} — it would refuse every \
         token and the GPU gates would silently measure the unfused loop",
        TOP_K * 3
    );
    assert!(
        SLOTS < EXPERTS * 3,
        "SLOTS={SLOTS} covers the layer's whole live set of {} blocks — no eviction would ever \
         recur and the fused arm would only ever be gated on a warm layer",
        EXPERTS * 3
    );
    assert!(
        config.swiglu_clamped_at(1),
        "swiglu_clamped_at must be TRUE for the routed layer — it is the predicate that denies \
         every plain-SiLU fused epilogue in the family"
    );
}

/// NON-VACUITY, and the gate's RED evidence. Each wrong epilogue program must disagree with the
/// vendor one by a wide margin on THIS fixture, measured on the reference alone. If any mutation
/// collapses toward the truth here, the GPU mutation assertions below stop binding — and it
/// fails on a GPU-less machine rather than passing silently.
#[test]
fn the_epilogue_program_actually_binds() {
    let config = mini_config();
    let plan = mini_plan(&config);
    let engine_weights = engine_fixture(&plan);
    let weights = reference_fixture(&plan, &engine_weights);
    let ids = tokens(8, 0x_E9_10_6E);
    let truth = memra_reference::execute(&plan, &weights, &ids)
        .expect("reference execute")
        .logits;

    for mutation in [
        Mutation::PostClamp,
        Mutation::PlainSwiglu,
        Mutation::NoSharedExpert,
        Mutation::SoftmaxRouter,
    ] {
        let got = memra_reference::execute(&mutation.apply(&plan), &weights, &ids)
            .expect("reference execute")
            .logits;
        let rel = relative(&got, &truth);
        let label = mutation.label();
        println!("binding[{label}]: {rel:.3e} (floor {MUTATION_FLOOR:.1e})");
        assert!(
            rel >= MUTATION_FLOOR,
            "[{label}] the wrong epilogue program is only {rel:.3e} from the right one — this \
             mutation does not bind, so the GPU gate below is vacuous for it. Raise \
             MLP_NORM_GAIN or lower LIMIT for the clamp arms."
        );
    }
}

/// NON-VACUITY (macro plane). `HostExps::stacked_macros` returns `None` for an all-ones vector,
/// which silently deletes the macro plane and would leave `wrong_macro_plane_fails_the_gate`
/// gated against a no-op. Also pins that the mutation actually changes a value.
#[test]
fn the_macro_plane_is_live() {
    let macros = macro_scales();
    assert_eq!(macros.len(), EXPERTS);
    assert!(
        macros.iter().any(|&m| m != 1.0),
        "an all-ones macro plane is dropped by stacked_macros — the macro arm would be vacuous"
    );
    // The mutation flattens the whole plane to 1.0, so EVERY expert must depart from 1.0 or the
    // drop is a partial no-op on the experts that were already there.
    assert!(
        macros.iter().all(|&m| (m - 1.0).abs() > 0.05),
        "every macro scale must sit clear of 1.0, or MacroDropped is a no-op for some experts"
    );
    println!("macro plane: {macros:?}");
}

/// NON-VACUITY (clamp). The limit only binds where activations cross it. Measured on the
/// reference's own MoE-layer activations, with no CUDA: the PRE and POST forms must disagree on
/// a real fraction of elements, or the pre/post half of this gate is a tautology.
#[test]
fn the_clamp_actually_binds() {
    let config = mini_config();
    let plan = mini_plan(&config);
    let engine_weights = engine_fixture(&plan);
    let weights = reference_fixture(&plan, &engine_weights);
    let ids = tokens(8, 0x_E9_10_6E);

    let truth = memra_reference::execute(&plan, &weights, &ids)
        .expect("reference execute")
        .logits;
    let post = memra_reference::execute(&Mutation::PostClamp.apply(&plan), &weights, &ids)
        .expect("reference execute")
        .logits;
    let plain = memra_reference::execute(&Mutation::PlainSwiglu.apply(&plan), &weights, &ids)
        .expect("reference execute")
        .logits;

    let post_rel = relative(&post, &truth);
    let plain_rel = relative(&plain, &truth);
    println!(
        "clamp non-vacuity at LIMIT={LIMIT} gain={MLP_NORM_GAIN}: post {post_rel:.3e}, \
         plain {plain_rel:.3e} (floor {MUTATION_FLOOR:.1e})"
    );
    assert!(
        post_rel >= MUTATION_FLOOR && plain_rel >= MUTATION_FLOOR,
        "the clamp does not bind on this fixture — raise MLP_NORM_GAIN or lower LIMIT, or the \
         GPU gates are vacuous in this state"
    );
}

// ---------------------------------------------------------------------------------------------
// GPU gates.
// ---------------------------------------------------------------------------------------------

/// One arm's environment. The fused arm is OFF by default (`docs/FLAGS.md`
/// `MEMRA_MOE_FUSED_EPI`), so both arms are pinned explicitly rather than inherited.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// The shipped program: the per-expert sequential loop, one `ffn_act_lim` launch per expert.
    Unfused,
    /// The fused epilogue under gate.
    Fused,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Arm::Unfused => "unfused sequential loop",
            Arm::Fused => "fused MoE epilogue",
        }
    }
}

/// Where the routed expert weights live. This is the PRODUCT question, not a flag A/B: full
/// two-card expert residency is the serving config now, and it makes `dev_exps` present on every
/// stage engine, which makes `slab_local` `Some`. The fused epilogue's first landing keyed on
/// `slab_local.is_none()` and was therefore DENIED OUTRIGHT under residency — an A/B on the
/// serving config would have read 0 dispatches and looked like "no effect". Both placements are
/// gated here so that cannot recur silently.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Placement {
    /// Host-resident banks, bounded GPU hot set. Pointers are SLRU slot addresses; the arm admits
    /// a token's working set, then re-verifies, and can fall through.
    Slru,
    /// Every routed expert in a device-resident slab. Pointers are slab base + ex*stride; there is
    /// no admission, no eviction and no fall-through, so engagement must be TOTAL.
    Slab,
}

impl Placement {
    fn label(self) -> &'static str {
        match self {
            Placement::Slru => "SLRU hot-set",
            Placement::Slab => "resident slabs",
        }
    }
}

struct Run {
    rows: Vec<(String, Vec<f32>)>,
    dispatches: u64,
    /// `(hits, misses, staged_bytes, n_slots)` from the SLRU. `None` means no cache was ever
    /// built, which for these arms is a fixture failure, not a mode.
    cache: Option<(u64, u64, u64, usize)>,
}

/// Token-layer opportunities the workload presents to the MoE arm: one MoE layer, and every
/// token of every prefill width plus the prime and each decode step.
const OPPORTUNITIES: u64 = 1 + 3 + 8 + 65 + 6 + 6;

/// Every arm must have run the SLRU under RECURRING pressure, not merely warm. Asserted rather
/// than left to the `SLOTS` construction: the sibling residency gate's header records that the
/// same by-construction argument silently produced a no-op cache on its first run.
fn check_cache_pressure_at(label: &str, run: &Run, placement: Placement) {
    if placement == Placement::Slab {
        assert!(
            run.cache.is_none(),
            "{label}: an SLRU was built on the resident-slab arm ({:?}) — the two placements are \
             secretly the same program and the slab provenance is untested",
            run.cache
        );
        println!(
            "{label}: no SLRU (resident slabs), fused dispatches {}/{OPPORTUNITIES}",
            run.dispatches
        );
        return;
    }
    check_cache_pressure(label, run);
}

fn check_cache_pressure(label: &str, run: &Run) {
    let (hits, misses, staged, slots) = run.cache.unwrap_or_else(|| {
        panic!(
            "{label}: no SLRU was built — MEMRA_MOE_CACHE/RESIDENT pins \
             did not take, and the arm ran on resident slabs instead of the staged path"
        )
    });
    println!(
        "{label}: SLRU hits={hits} misses={misses} staged={staged}B slots={slots}; \
         fused dispatches {}/{OPPORTUNITIES} token-layer opportunities",
        run.dispatches
    );
    assert!(
        misses > 0,
        "{label}: the SLRU took {misses} misses — nothing was ever staged, so the arm was gated \
         on an all-resident layer and its whole difference from gdec went untested"
    );
    assert!(
        hits > 0,
        "{label}: the SLRU took {hits} hits — the cache is thrashing completely and the \
         hit/miss MIX this arm is supposed to serve was never exercised"
    );
    assert_eq!(
        slots, SLOTS,
        "{label}: MEMRA_MOE_SLOTS did not take ({slots} != {SLOTS}); the eviction pressure this \
         gate depends on is not the pressure it thinks it pinned"
    );
}

/// Run the whole workload under one arm. A FRESH `Engine` and a FRESH model load per arm are
/// mandatory, not hygiene: expert residency and pinned-vs-paged host buffers are decided at LOAD
/// time and the SLRU is per-Engine, so reusing either would silently compare an arm to itself.
fn run_arm(arm: Arm, mutation: FixtureMutation) -> Run {
    run_arm_at(arm, mutation, Placement::Slru)
}

fn run_arm_at(arm: Arm, mutation: FixtureMutation, placement: Placement) -> Run {
    force_true_f32();
    // SAFETY: every caller holds `gpu_guard()`, which is the only thing in this binary that
    // touches these vars, and no other thread is running engine code.
    unsafe {
        match placement {
            // SLRU residency, slots pinned below the layer's 24 live blocks, so the arm is
            // exercised with recurring misses rather than only on a fully warm layer.
            Placement::Slru => {
                std::env::set_var("MEMRA_MOE_CACHE", "1");
                std::env::set_var("MEMRA_MOE_RESIDENT", "0");
                std::env::set_var("MEMRA_MOE_SLOTS", SLOTS.to_string());
            }
            // No cache at all: the load-time planner slabs this fixture's tiny expert set, which
            // is what the real artifact gets from full two-card residency. `moe_cache_stats()`
            // must then answer `None`, and that is asserted rather than assumed — it is the only
            // thing proving the two placements are different programs.
            Placement::Slab => {
                std::env::set_var("MEMRA_MOE_CACHE", "0");
                std::env::remove_var("MEMRA_MOE_RESIDENT");
                std::env::remove_var("MEMRA_MOE_SLOTS");
            }
        }
        match arm {
            Arm::Unfused => std::env::set_var("MEMRA_MOE_FUSED_EPI", "0"),
            Arm::Fused => std::env::set_var("MEMRA_MOE_FUSED_EPI", "1"),
        }
    }

    let config = mini_config();
    let plan = mini_plan(&config);
    let engine_weights = engine_fixture(&plan);
    let source = fixture_source(&config, &plan, &engine_weights, mutation);
    let engine = Engine::new(0).expect("CUDA engine on device 0");
    let model = HybridModel::load_from_source_without_mtp(&engine, &source)
        .expect("mini glm5_next model loads from the contract");

    let before = memra_engine::moe_fused_epilogue_dispatches();
    println!(
        "arm[{} / {}]: MEMRA_MOE_FUSED_EPI={} fixture={mutation:?}",
        arm.label(),
        placement.label(),
        match arm {
            Arm::Unfused => "0",
            Arm::Fused => "1",
        }
    );
    let mut rows: Vec<(String, Vec<f32>)> = Vec::new();

    // Prefill widths that cross the KDA scan's chunk size and the MoE dispatch's MOE_DEV_MAX_T
    // seam, then the prime + decode seam, whose t=1 MoE dispatch is a different arm again.
    for &n in &[1usize, 3, 8, 65] {
        let ids = tokens(n, 0x_E9_10_6E ^ n as u64);
        let got = model.forward(&engine, &ids).expect("GPU routed prefill");
        rows.push((format!("prefill T={n}"), got));
    }

    let prompt = 6usize;
    let steps = 6usize;
    let ids = tokens(prompt + steps, 0x_E9_10_6E_DEC0);
    let mut cache = memra_engine::cache::Cache::new_planned(&engine, &model.cfg, &plan, 64)
        .expect("cache for the mini glm5_next model");
    let (primed, _seed, _hiddens) = model
        .prime_cache(&engine, &ids[..prompt], &mut cache, 0)
        .expect("GPU routed prime");
    rows.push(("prime last row".to_string(), primed));
    for step in 0..steps {
        let got = model
            .decode_step(&engine, ids[prompt + step], &mut cache)
            .expect("GPU routed decode step");
        rows.push((format!("decode step {step}"), got));
    }

    Run {
        rows,
        dispatches: memra_engine::moe_fused_epilogue_dispatches() - before,
        cache: engine.moe_cache_stats(),
    }
}

/// Reference logits for the same workload `run_arm` produces, row for row.
fn reference_rows(
    plan: &ModelPlan,
    weights: &BTreeMap<TensorId, ReferenceTensor>,
) -> Vec<Vec<f32>> {
    let vocab = VOCAB as usize;
    let mut rows = Vec::new();
    for &n in &[1usize, 3, 8, 65] {
        let ids = tokens(n, 0x_E9_10_6E ^ n as u64);
        rows.push(
            memra_reference::execute(plan, weights, &ids)
                .expect("reference execute")
                .logits,
        );
    }
    let prompt = 6usize;
    let steps = 6usize;
    let ids = tokens(prompt + steps, 0x_E9_10_6E_DEC0);
    let full = memra_reference::execute(plan, weights, &ids)
        .expect("reference execute")
        .logits;
    rows.push(full[(prompt - 1) * vocab..prompt * vocab].to_vec());
    for step in 0..steps {
        let row = prompt + step;
        rows.push(full[row * vocab..(row + 1) * vocab].to_vec());
    }
    rows
}

/// GATE A — the fused epilogue against the reference oracle, end to end, and it must have
/// actually dispatched. The dispatch assertion is anchored on the counter the fused arm
/// increments at its own call site, not on a flag being set and not on the model merely running
/// (LAW:wiring-assertions-match-prose). Until the fusion lands this test is RED on that
/// assertion, which is the intended state.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn fused_epilogue_matches_the_reference() {
    let _gpu = gpu_guard();
    let config = mini_config();
    let plan = mini_plan(&config);
    let want = reference_rows(&plan, &reference_fixture(&plan, &engine_fixture(&plan)));

    let run = run_arm(Arm::Fused, FixtureMutation::None);
    check_all("fused", &run.rows, &want);
    check_cache_pressure("fused", &run);
    assert!(
        run.dispatches > 0,
        "MEMRA_MOE_FUSED_EPI=1 produced {} fused-epilogue dispatches — the arm never fired, so \
         this gate measured the unfused loop and proves nothing about the fusion",
        run.dispatches
    );
    println!("fused-epilogue dispatches: {}", run.dispatches);
}

/// GATE B — the unfused arm against the same oracle. This is the control: it proves the fixture,
/// the macro fold and the tolerance are honest independently of the fusion, so a GATE A failure
/// is attributable to the fused arm and not to the harness. It must ALSO report zero fused
/// dispatches, which is what proves the two arms are different programs.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn unfused_control_matches_the_reference() {
    let _gpu = gpu_guard();
    let config = mini_config();
    let plan = mini_plan(&config);
    let want = reference_rows(&plan, &reference_fixture(&plan, &engine_fixture(&plan)));

    let run = run_arm(Arm::Unfused, FixtureMutation::None);
    check_all("unfused", &run.rows, &want);
    check_cache_pressure("unfused", &run);
    assert_eq!(
        run.dispatches, 0,
        "MEMRA_MOE_FUSED_EPI=0 still took the fused arm {} times — the rollback seam does not \
         roll back, and the two arms are the same program",
        run.dispatches
    );
}

/// GATE C — the two arms against each other. Same weights, same workload, same engine: the only
/// difference is the epilogue's dispatch. Reports the exact-bit disagreement count alongside the
/// relative distance, so a future claim of bit-identity is measured rather than asserted from
/// the kernel's provenance.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn the_two_arms_agree() {
    let _gpu = gpu_guard();
    let unfused = run_arm(Arm::Unfused, FixtureMutation::None);
    let fused = run_arm(Arm::Fused, FixtureMutation::None);
    check_cache_pressure("arms/unfused", &unfused);
    check_cache_pressure("arms/fused", &fused);
    assert!(
        fused.dispatches > 0,
        "the fused arm never fired — this comparison is arm-vs-itself"
    );
    assert_eq!(unfused.dispatches, 0, "the unfused arm took the fused path");
    assert_eq!(unfused.rows.len(), fused.rows.len());
    for ((name, a), (_, b)) in unfused.rows.iter().zip(&fused.rows) {
        let bits = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let rel = relative(b, a);
        println!(
            "arms {name}: relative maxdiff {rel:.3e} (tol {TOL:.1e}), {bits}/{} elements differ \
             in bits",
            a.len()
        );
        assert!(
            rel <= TOL,
            "arms {name}: fused vs unfused relative maxdiff {rel:.3e} (tol {TOL:.1e})"
        );
    }
}

/// MUTATION CHECK — each wrong epilogue program must fail GATE A by a wide margin, on the SAME
/// GPU output the passing assertion uses. If any of these ever lands inside TOL, GATE A is
/// worthless.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn wrong_epilogue_programs_fail_the_gate() {
    let _gpu = gpu_guard();
    let config = mini_config();
    let plan = mini_plan(&config);
    let weights = reference_fixture(&plan, &engine_fixture(&plan));

    let run = run_arm(Arm::Fused, FixtureMutation::None);
    assert!(
        run.dispatches > 0,
        "the fused arm never fired — these mutations would be measured against the unfused loop"
    );

    // Control: the GPU really does match the vendor program on this run, so the distances below
    // are the mutations' and not a broken harness's.
    check_all(
        "mutation control",
        &run.rows,
        &reference_rows(&plan, &weights),
    );

    for mutation in [
        Mutation::PostClamp,
        Mutation::PlainSwiglu,
        Mutation::NoSharedExpert,
        Mutation::SoftmaxRouter,
    ] {
        let wrong = reference_rows(&mutation.apply(&plan), &weights);
        let label = mutation.label();
        let worst = run
            .rows
            .iter()
            .zip(&wrong)
            .map(|((_, got), expect)| relative(got, expect))
            .fold(0.0f32, f32::max);
        let closest = run
            .rows
            .iter()
            .zip(&wrong)
            .map(|((_, got), expect)| relative(got, expect))
            .fold(f32::INFINITY, f32::min);
        println!(
            "mutation[{label}]: relative maxdiff min {closest:.3e} max {worst:.3e} \
             (tol {TOL:.1e}, floor {MUTATION_FLOOR:.1e})"
        );
        assert!(
            closest >= MUTATION_FLOOR,
            "[{label}] is only {closest:.3e} away from the GPU's output on its closest row — \
             this gate does not bind"
        );
    }
}

/// MUTATION CHECK (macro plane) — the one property no `ModelPlan` knob can express. One
/// per-expert `weight_scale_2` is dropped to 1.0 in the bytes the ENGINE loads; the reference
/// still carries it, so the comparison must break. This is the failure mode the NVFP4 expert
/// class is specifically prone to, and it is fluent and invisible in the output.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn wrong_macro_plane_fails_the_gate() {
    let _gpu = gpu_guard();
    let config = mini_config();
    let plan = mini_plan(&config);
    let want = reference_rows(&plan, &reference_fixture(&plan, &engine_fixture(&plan)));

    let run = run_arm(Arm::Fused, FixtureMutation::MacroDropped);
    assert!(
        run.dispatches > 0,
        "the fused arm never fired — the macro mutation would be measured against the unfused \
         loop"
    );
    let worst = run
        .rows
        .iter()
        .zip(&want)
        .map(|((_, got), expect)| relative(got, expect))
        .fold(0.0f32, f32::max);
    println!(
        "mutation[macro-scale-dropped]: worst relative maxdiff {worst:.3e} (tol {TOL:.1e}, \
         floor {MUTATION_FLOOR:.1e})"
    );
    assert!(
        worst >= MUTATION_FLOOR,
        "dropping a per-expert weight_scale_2 moved the fused output by only {worst:.3e} — the \
         macro fold is not under gate"
    );
}

// ---------------------------------------------------------------------------------------------
// RESIDENT-SLAB placement. This is the serving config: full two-card expert residency puts every
// routed expert in a device slab, and the fused epilogue's first landing was DENIED there by its
// own `slab_local.is_none()` predicate. These gates exist so that denial cannot come back
// silently, and so the slab provenance ships gated rather than assumed-equivalent.
// ---------------------------------------------------------------------------------------------

/// GATE D — the fused epilogue on RESIDENT SLABS, against the reference oracle, with TOTAL
/// engagement. Unlike the SLRU arm this one cannot fall through: a slab holds every expert by
/// construction, so there is no admission that could move a pointer and no capacity floor to
/// refuse on. `OPPORTUNITIES`-out-of-`OPPORTUNITIES` is therefore the bar, not `> 0` — anything
/// less means a predicate is denying the arm somewhere and the serving config is not running it.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn fused_epilogue_on_resident_slabs_matches_the_reference() {
    let _gpu = gpu_guard();
    let config = mini_config();
    let plan = mini_plan(&config);
    let want = reference_rows(&plan, &reference_fixture(&plan, &engine_fixture(&plan)));

    let run = run_arm_at(Arm::Fused, FixtureMutation::None, Placement::Slab);
    check_all("fused/slab", &run.rows, &want);
    check_cache_pressure_at("fused/slab", &run, Placement::Slab);
    assert_eq!(
        run.dispatches, OPPORTUNITIES,
        "the resident-slab arm took the fused epilogue {}/{OPPORTUNITIES} times. A slab has every \
         expert by construction, so every token-layer must engage; a shortfall means a predicate \
         is denying the arm on the placement the product actually serves on",
        run.dispatches
    );
}

/// GATE E — the unfused control on resident slabs. Attribution control for GATE D, and it must
/// report zero fused dispatches so the rollback seam is proven on this placement too.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn unfused_control_on_resident_slabs_matches_the_reference() {
    let _gpu = gpu_guard();
    let config = mini_config();
    let plan = mini_plan(&config);
    let want = reference_rows(&plan, &reference_fixture(&plan, &engine_fixture(&plan)));

    let run = run_arm_at(Arm::Unfused, FixtureMutation::None, Placement::Slab);
    check_all("unfused/slab", &run.rows, &want);
    check_cache_pressure_at("unfused/slab", &run, Placement::Slab);
    assert_eq!(
        run.dispatches, 0,
        "MEMRA_MOE_FUSED_EPI=0 still took the fused arm {} times on resident slabs",
        run.dispatches
    );
}

/// GATE F — the two arms against each other ON SLABS, and the slab arm against the SLRU arm.
///
/// The second comparison is the one worth having. The slab and SLRU arms fill the SAME kernel
/// pair from different pointer sources, so they are a provenance-only pair and their outputs must
/// agree; if they ever diverge, one of the two pointer computations is wrong and the fixture-width
/// bit-identity of either arm alone would not have caught it.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn slab_and_slru_fused_arms_agree() {
    let _gpu = gpu_guard();
    let unfused_slab = run_arm_at(Arm::Unfused, FixtureMutation::None, Placement::Slab);
    let fused_slab = run_arm_at(Arm::Fused, FixtureMutation::None, Placement::Slab);
    let fused_slru = run_arm_at(Arm::Fused, FixtureMutation::None, Placement::Slru);
    assert_eq!(
        fused_slab.dispatches, OPPORTUNITIES,
        "slab arm did not fully engage"
    );
    assert!(fused_slru.dispatches > 0, "SLRU arm never fired");
    assert_eq!(
        unfused_slab.dispatches, 0,
        "slab control took the fused path"
    );

    for ((name, a), (_, b)) in unfused_slab.rows.iter().zip(&fused_slab.rows) {
        let bits = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let rel = relative(b, a);
        println!(
            "slab arms {name}: relative maxdiff {rel:.3e} (tol {TOL:.1e}), {bits}/{} bits differ",
            a.len()
        );
        assert!(rel <= TOL, "slab arms {name}: {rel:.3e} > tol {TOL:.1e}");
    }
    for ((name, a), (_, b)) in fused_slru.rows.iter().zip(&fused_slab.rows) {
        let bits = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let rel = relative(b, a);
        println!(
            "fused slru-vs-slab {name}: relative maxdiff {rel:.3e} (tol {TOL:.1e}), \
             {bits}/{} bits differ",
            a.len()
        );
        assert!(
            rel <= TOL,
            "fused slru-vs-slab {name}: {rel:.3e} > tol {TOL:.1e} — the two pointer provenances \
             feed the same kernel pair, so they must agree"
        );
    }
}

/// MUTATION CHECK on slabs — the same four wrong programs must fail GATE D by a wide margin. A
/// gate that binds on one placement and not the other is half a gate.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn wrong_epilogue_programs_fail_the_slab_gate() {
    let _gpu = gpu_guard();
    let config = mini_config();
    let plan = mini_plan(&config);
    let weights = reference_fixture(&plan, &engine_fixture(&plan));
    let run = run_arm_at(Arm::Fused, FixtureMutation::None, Placement::Slab);
    assert_eq!(
        run.dispatches, OPPORTUNITIES,
        "slab arm did not fully engage"
    );
    check_all(
        "slab mutation control",
        &run.rows,
        &reference_rows(&plan, &weights),
    );

    for mutation in [
        Mutation::PostClamp,
        Mutation::PlainSwiglu,
        Mutation::NoSharedExpert,
        Mutation::SoftmaxRouter,
    ] {
        let wrong = reference_rows(&mutation.apply(&plan), &weights);
        let closest = run
            .rows
            .iter()
            .zip(&wrong)
            .map(|((_, got), expect)| relative(got, expect))
            .fold(f32::INFINITY, f32::min);
        println!(
            "slab mutation[{}]: closest row {closest:.3e} (tol {TOL:.1e}, floor {MUTATION_FLOOR:.1e})",
            mutation.label()
        );
        assert!(
            closest >= MUTATION_FLOOR,
            "[{}] is only {closest:.3e} from the slab arm's output — this gate does not bind",
            mutation.label()
        );
    }
}
