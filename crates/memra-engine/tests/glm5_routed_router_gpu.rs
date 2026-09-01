//! GPU-vs-reference gate for glm5_next's ROUTED-expert MoE arm — the sigmoid / `noaux_tc`
//! router (`RouterPlan::Sigmoid`), end to end.
//!
//! Truth is the vendor program, `DeepseekV3MoE.route_tokens_to_experts` (glm5_next's
//! `Glm5NextTextTopkRouter`/`Glm5NextTextMoE` subclass it and override nothing routing-related —
//! `research/glm53-flash-bringup-20260827/modular_glm5_next-ref.py:350`):
//!
//! ```text
//! scores            = sigmoid(router_logits)                 # f32
//! scores_for_choice = scores + e_score_correction_bias       # SELECTION ONLY
//! topk_indices      = topk(scores_for_choice, k)             # bias decides WHO
//! topk_weights      = scores.gather(topk_indices)            # un-biased scores decide HOW MUCH
//! topk_weights     /= topk_weights.sum() + 1e-20             # norm_topk_prob
//! topk_weights     *= routed_scaling_factor                  # 2.5 -> the weights sum to 2.5
//! ```
//!
//! (`n_group`/`topk_group` are both 1 on the GLM-5.3-Flash checkpoint, so the vendor's
//! group-masking step is the identity; neither memra side implements it, and that is a scope
//! line for this family's config, not a gap this gate covers.)
//!
//! WHY THIS GATE EXISTS. `memra_reference::route_experts` and the engine's
//! `moe_route_sigmoid_host`/`moe_router_sigmoid_topk_f32` both implement exactly that program —
//! and glm5_next reached NEITHER of them. `ModelConfig::sigmoid_router()` carried arms for
//! m3/hy3/mla/step35 and none for `cfg.glm5`, while the HF/safetensors path sets `mla: None`, so
//! the accessor answered `None` for every glm5_next model. Every dispatch predicate in
//! `hybrid_forward` keys off that accessor, so the routed branch rode the SOFTMAX router instead:
//! softmax scores in place of sigmoid, `e_score_correction_bias` ignored, and weights normalized
//! to sum 1 instead of 2.5. The plan was right the whole time — `model_plan::router` emits
//! `RouterPlan::Sigmoid { normalize_selected, scaling_factor, selection_bias: true }` for
//! glm5_next — so the checkpoint's `exp_probs_b` was loaded, uploaded, and never read. It
//! compiled, it ran, and it produced plausible-but-wrong logits.
//!
//! MEASURED, before and after (5090, TF32 off, this fixture, 2026-08-28). BEFORE the fix:
//! `the_plan_and_the_engine_agree_on_the_glm5_router` fails with `sigmoid_router() == None` vs
//! `Some((2.5, true))`; routed prefill T=1 is 1.820e-1 from the reference and T=8 is 1.463e-1;
//! prime's last row is 3.902e-2. AFTER: 1.858e-4 / 6.271e-5 / 1.612e-4 / 1.569e-4 at
//! T = 1/3/8/65, and the prime+decode seam is inside TOL — three orders down, at the q8
//! activation-quantization floor.
//!
//! FOUR PROGRAMS ARE PINNED, not one. Each is a wrong answer a narrower fix would have produced,
//! and each is measured against the SAME GPU output the passing assertion uses:
//!   * SOFTMAX routing — the bug itself (what `sigmoid_router() == None` selected), 1.449e-1;
//!   * the selection bias never applied — 1.385e-1;
//!   * normalization skipped — weights sum to `sum(sigmoid) * 2.5` instead of 2.5, 1.875e-1;
//!   * scaling factor dropped — weights sum to 1, 1.699e-1.
//!
//! A fifth belongs to the router alone and no `RouterPlan` knob can express it, so it is pinned
//! on the router's own oracle instead of end to end (`the_selected_weights_carry_no_bias`):
//! folding the bias into the COMBINING weights rather than the selection, 4.058e-1.
//!
//! NON-VACUITY IS ENFORCED, NOT ASSUMED. A selection bias only binds where it changes WHO is
//! picked, and `deterministic_fixture` mints `e_score_correction_bias` at scale 0.05 — small
//! next to the spread of `sigmoid(logits)`. `ROUTER_BIAS_GAIN` scales it to a size where biased
//! and un-biased top-k disagree; `the_router_program_actually_binds` measures the four mutations
//! on the reference alone, with no CUDA, so a fixture that drifted back into agreement fails on a
//! GPU-less machine rather than turning the GPU gates into tautologies.
//! `the_selection_bias_actually_changes_the_picks` pins the same thing at the router itself,
//! counting tokens whose biased top-k differs from their un-biased top-k.
//!
//! THE CLAMP IS DELIBERATELY INERT HERE. `swiglu_limit` is the shipped 10.0 and the pre-MLP norms
//! are un-scaled, so no activation reaches it and any failure is the router's.
//! `swiglu_preclamp_gpu.rs` is the gate that makes the clamp bind.
//!
//! BOTH ROUTER ARMS ARE COVERED. `moe_route_sigmoid_cfg` defaults to the device kernel
//! (`moe_router_sigmoid_topk_f32`); `MEMRA_SIG_ROUTER=0` restores the host oracle. Every number
//! above was measured twice, once per arm, and they agree to the printed digits — so this gate
//! binds the rollback seam as well as the default. Re-run the =0 arm the same way, with the env
//! var set.
//!
//! GPU-gated (`#[ignore]`); rig law = exactness only, never timing. Run under
//! `flock /tmp/memra-5090.lock` with `NVIDIA_TF32_OVERRIDE=0` and `-- --ignored`.

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::{MlpPlan, ModelPlan, RouterPlan};
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, LayerTensor, OutputHead, TensorContract, TensorId,
    TensorMatch,
};
use memra_reference::{ReferenceTensor, deterministic_fixture};
use std::borrow::Cow;
use std::collections::BTreeMap;

const VOCAB: u32 = 32;

/// The checkpoint's own routing shape, shrunk only where the fixture must be small.
/// `SCALING` and `NORM` are GLM-5.3-Flash's real `routed_scaling_factor` / `norm_topk_prob`
/// (`research/glm53-flash-bringup-20260827/glm-config.json`), NOT scaled-down stand-ins: the
/// accessor returning the checkpoint's values is half of what is under test.
const EXPERTS: usize = 8;
const TOP_K: usize = 3;
const SCALING: f32 = 2.5;
const NORM: bool = true;

/// Selection-bias amplifier — the non-vacuity construction (see the module header).
/// `the_selection_bias_actually_changes_the_picks` is what proves this number is large enough.
const ROUTER_BIAS_GAIN: f32 = 12.0;

/// Scale-relative bound for the routed branch. Looser than `swiglu_preclamp_gpu`'s 2e-5 for one
/// measured reason: the routed expert matvecs quantize their ACTIVATIONS to q8_1 while the
/// reference multiplies in f32, and unlike the expert weights (snapped onto the Q8_0 grid by
/// `snap_to_q8_0`, so both sides read the same numbers) that error cannot be constructed away.
/// Calibrated from the measured post-fix figures (worst 1.858e-4 over T = 1/3/8/65 plus the
/// prime+decode seam), with ~5x headroom — calibrate downward, never upward.
const TOL: f32 = 1e-3;

/// Floor for "this wrong router program fails by a wide margin". The measured end-to-end margins
/// are 1.449e-1 (softmax), 1.385e-1 (no selection bias), 1.875e-1 (no normalization) and 1.699e-1
/// (no scaling factor), so this floor sits ~4.6x below the smallest of them and 30x above TOL.
const MUTATION_FLOOR: f32 = 3e-2;

/// GPU tests serialize on one device: the model, its cache and the reference stack are all live
/// at once, and cargo runs test fns in parallel by default.
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
/// The MLA/DSA fields are required by the glm5_next config parser and are inert: no layer in
/// `layer_types` selects them.
fn mini_config_json() -> String {
    format!(
        r#"{{
      "model_type": "glm5_next_text",
      "num_hidden_layers": 2,
      "num_nextn_predict_layers": 0,
      "hidden_size": 128,
      "intermediate_size": 64,
      "vocab_size": 32,
      "max_position_embeddings": 512,
      "rms_norm_eps": 1e-05,
      "hidden_act": "silu",
      "swiglu_limit": 10.0,
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
      "moe_intermediate_size": 32,
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

/// The wrong router programs. Only the REFERENCE reads these — the GPU always runs the real plan
/// — so each answers "what would the reference say if the engine had implemented THIS router?"
/// against one fixed set of weights.
#[derive(Clone, Copy)]
enum Mutation {
    /// The bug: softmax scores, no selection bias, weights summing to 1.
    Softmax,
    /// `e_score_correction_bias` never applied: selection falls back to the raw sigmoid scores.
    NoSelectionBias,
    /// `norm_topk_prob` dropped: weights become `sigmoid(logit) * 2.5`, summing to whatever.
    NoNormalize,
    /// `routed_scaling_factor` dropped: normalized weights summing to 1.
    NoScaling,
}

impl Mutation {
    fn label(self) -> &'static str {
        match self {
            Mutation::Softmax => "softmax-for-sigmoid",
            Mutation::NoSelectionBias => "no-selection-bias",
            Mutation::NoNormalize => "no-normalization",
            Mutation::NoScaling => "no-scaling-factor",
        }
    }

    fn router(self) -> RouterPlan {
        match self {
            Mutation::Softmax => RouterPlan::Softmax,
            Mutation::NoSelectionBias => RouterPlan::Sigmoid {
                normalize_selected: NORM,
                scaling_factor: SCALING,
                selection_bias: false,
            },
            Mutation::NoNormalize => RouterPlan::Sigmoid {
                normalize_selected: false,
                scaling_factor: SCALING,
                selection_bias: true,
            },
            Mutation::NoScaling => RouterPlan::Sigmoid {
                normalize_selected: NORM,
                scaling_factor: 1.0,
                selection_bias: true,
            },
        }
    }
}

/// The same plan with the MoE layer's router replaced.
fn with_router(plan: &ModelPlan, router: RouterPlan) -> ModelPlan {
    let mut mutated = plan.clone();
    let mut seen = false;
    for layer in &mut mutated.layers {
        if let MlpPlan::Moe(moe) = &mut layer.mlp {
            moe.router = router.clone();
            seen = true;
        }
    }
    assert!(seen, "the mini plan must carry a routed-MoE layer");
    mutated
}

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
    ggml_type: GgmlType,
}

/// Serves the reference fixture's own numbers under the contract's ggml names, so the reference
/// and the GPU read ONE set of weights. Must answer `config()`: `HybridModel::load_from_source`
/// compiles the plan from it.
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

fn fixture_source(
    config: &ModelConfig,
    plan: &ModelPlan,
    weights: &BTreeMap<TensorId, ReferenceTensor>,
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
    let mut tensors = BTreeMap::new();
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
        // `HostExps` rejects F32 expert slabs, so the three MoE banks ride Q8_0 — the encoding
        // `micro_gguf`'s fixtures use. `routing_fixture` has already snapped their values onto
        // that grid, so the bytes the GPU dequantizes ARE the numbers the reference reads and the
        // weight encoding costs no parity. The expert matvec's q8_1 ACTIVATION quantization is a
        // separate, unavoidable floor — see TOL.
        let (bytes, ggml_type) = if is_expert_bank(&req.id) {
            (
                memra_gguf::nvfp4_repack::f32_to_q8_0(&tensor.data),
                GgmlType::Q8_0,
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
    FixtureSource {
        config: config.clone(),
        tensors,
    }
}

/// The three stacked MoE expert slabs, the only tensors the engine refuses to load as F32.
fn is_expert_bank(id: &TensorId) -> bool {
    matches!(
        id,
        TensorId::Layer {
            tensor: LayerTensor::MoeExpertGateBank
                | LayerTensor::MoeExpertUpBank
                | LayerTensor::MoeExpertDownBank,
            ..
        }
    )
}

/// Q8_0 round trip: `d = amax/127` per 32-element block, `q = round(v/d)`, back to `d*q`.
/// Snapping the fixture ONTO that grid keeps the reference and the GPU reading one set of numbers
/// once the bank is encoded — otherwise the weight quantization error would sit on top of the
/// routing this gate is trying to measure.
fn snap_to_q8_0(data: &mut [f32]) {
    for blk in data.chunks_exact_mut(32) {
        let amax = blk.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        if d <= 0.0 {
            continue;
        }
        let id = 1.0 / d;
        for v in blk.iter_mut() {
            *v = d * ((*v * id).round_ties_even() as i32 as i8 as f32);
        }
    }
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

/// The fixture: expert banks snapped onto the Q8_0 grid they are served on, and the selection
/// bias amplified by `ROUTER_BIAS_GAIN` (the non-vacuity construction). Expert banks are left at
/// their generated values — every routed expert must compute something DIFFERENT, or which
/// experts get picked stops mattering and the selection half of the router goes ungated.
fn routing_fixture(plan: &ModelPlan) -> BTreeMap<TensorId, ReferenceTensor> {
    let mut weights = deterministic_fixture(plan)
        .expect("deterministic glm5_next fixture")
        .weights;
    let bias_id = router_bias_id(plan);
    let bias = weights
        .get_mut(&bias_id)
        .expect("fixture mints e_score_correction_bias for a selection-bias router");
    for v in &mut bias.data {
        *v *= ROUTER_BIAS_GAIN;
    }
    for (id, tensor) in weights.iter_mut() {
        if is_expert_bank(id) {
            snap_to_q8_0(&mut tensor.data);
        }
    }
    weights
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

fn check(name: &str, got: &[f32], want: &[f32]) {
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{name}: GPU output has non-finite values"
    );
    let rel = relative(got, want);
    assert!(
        rel <= TOL,
        "{name}: GPU vs reference relative maxdiff {rel:.3e} (tol {TOL:.1e})"
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

struct Harness {
    engine: Engine,
    model: HybridModel,
    plan: ModelPlan,
    weights: BTreeMap<TensorId, ReferenceTensor>,
}

impl Harness {
    fn new() -> Self {
        force_true_f32();
        let config = mini_config();
        let plan = mini_plan(&config);
        let weights = routing_fixture(&plan);
        let source = fixture_source(&config, &plan, &weights);
        let engine = Engine::new(0).expect("CUDA engine on device 0");
        let model = HybridModel::load_from_source_without_mtp(&engine, &source)
            .expect("mini glm5_next model loads from the contract");
        Self {
            engine,
            model,
            plan,
            weights,
        }
    }

    /// Reference logits under the plan's own router (the vendor truth).
    fn reference_logits(&self, tokens: &[u32]) -> Vec<f32> {
        memra_reference::execute(&self.plan, &self.weights, tokens)
            .expect("reference execute")
            .logits
    }

    fn reference_logits_mutated(&self, mutation: Mutation, tokens: &[u32]) -> Vec<f32> {
        reference_logits_mutated(&self.plan, &self.weights, mutation, tokens)
    }
}

fn reference_logits_mutated(
    plan: &ModelPlan,
    weights: &BTreeMap<TensorId, ReferenceTensor>,
    mutation: Mutation,
    tokens: &[u32],
) -> Vec<f32> {
    memra_reference::execute(&with_router(plan, mutation.router()), weights, tokens)
        .expect("reference execute")
        .logits
}

// ---------------------------------------------------------------------------------------------
// Host-only gates. These need no CUDA and they are the ones that fail on the bug itself.
// ---------------------------------------------------------------------------------------------

/// GATE 1 — THE BUG. The plan and the engine must describe the SAME router. `model_plan::router`
/// has always emitted `RouterPlan::Sigmoid` for glm5_next; `ModelConfig::sigmoid_router()` is the
/// accessor every `hybrid_forward` dispatch predicate consults, and it answered `None` for this
/// arch — so the routed branch ran softmax. This asserts both halves against the checkpoint's own
/// numbers.
#[test]
fn the_plan_and_the_engine_agree_on_the_glm5_router() {
    let config = mini_config();
    let plan = mini_plan(&config);

    let MlpPlan::Moe(moe) = &plan.layers[1].mlp else {
        panic!("layer 1 must be the routed-MoE branch");
    };
    assert_eq!(
        moe.router,
        RouterPlan::Sigmoid {
            normalize_selected: NORM,
            scaling_factor: SCALING,
            selection_bias: true,
        },
        "the plan must declare the noaux_tc recipe: sigmoid scores, selection-only bias, \
         sum-normalized selected weights, x routed_scaling_factor"
    );
    assert_eq!(moe.expert_count as usize, EXPERTS);
    assert_eq!(moe.experts_per_token as usize, TOP_K);

    // The engine-side accessor, on the same config. `None` here is the bug: it sends every
    // glm5_next routed layer down the softmax arms of `moe_ffn_inner`.
    assert_eq!(
        config.sigmoid_router(),
        Some((SCALING, NORM)),
        "ModelConfig::sigmoid_router() must answer glm5_next's routed_scaling_factor and \
         norm_topk_prob — every routed-MoE dispatch predicate in hybrid_forward keys off it, and \
         `None` silently selects the SOFTMAX router (no sigmoid, no e_score_correction_bias, \
         weights summing to 1 instead of {SCALING})"
    );
}

/// The router probe vector. Chosen so the bias REVERSES the picks: un-biased top-3 is {2, 0, 5},
/// biased top-3 is {1, 4, 0} — two of three experts differ, so "bias is selection-only" is not
/// vacuous on it. Independent of the model fixture on purpose: this is the router in isolation.
fn probe_row() -> (Vec<f32>, Vec<f32>) {
    (
        vec![0.4, -0.2, 0.9, 0.1, -1.0, 0.3],
        vec![0.0, 0.5, -0.6, 0.0, 0.4, 0.0],
    )
}

const PROBE_TOP_K: usize = 3;

/// GATE 2 — THE ROUTER PROGRAM ITSELF, on hand-picked logits, against a hand-computed vendor
/// value. This is the isolating measurement: it compares the engine's own routing oracle
/// (`moe_route_sigmoid_host_public`, the same math the device kernel reproduces) with the four
/// steps of `DeepseekV3MoE.route_tokens_to_experts`, and with the softmax program the engine ran
/// before the fix — reporting the per-token weight SUM, which is the quantity that made the
/// constant-expert experiment diverge (2.5 vs 1.0).
#[test]
fn the_engine_router_is_the_vendor_program() {
    let (logits, bias) = probe_row();
    let n_expert = logits.len();
    let n_used = PROBE_TOP_K;

    let sigmoid = |x: f32| 1.0f32 / (1.0 + (-x).exp());
    let scores: Vec<f32> = logits.iter().copied().map(sigmoid).collect();

    // --- vendor semantics, computed here and nowhere else ---
    let choice: Vec<f32> = scores.iter().zip(&bias).map(|(s, b)| s + b).collect();
    let mut order: Vec<usize> = (0..n_expert).collect();
    order.sort_by(|&a, &b| choice[b].total_cmp(&choice[a]).then(a.cmp(&b)));
    let want_ids: Vec<usize> = order[..n_used].to_vec();
    let mut want_w: Vec<f32> = want_ids.iter().map(|&i| scores[i]).collect();
    let denominator: f32 = want_w.iter().sum::<f32>() + 1e-20;
    for w in &mut want_w {
        *w = *w / denominator * SCALING;
    }

    // Un-biased picks, to prove the bias is doing work in this vector.
    let mut plain_order: Vec<usize> = (0..n_expert).collect();
    plain_order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let unbiased_ids: Vec<usize> = plain_order[..n_used].to_vec();
    assert_ne!(
        want_ids, unbiased_ids,
        "the probe vector must be one where the selection bias changes the picks"
    );

    // --- the engine's own oracle ---
    let (sel, w) = HybridModel::moe_route_sigmoid_host_public(
        &logits,
        1,
        n_expert,
        n_used,
        Some(&bias),
        SCALING,
        NORM,
        None,
    )
    .expect("engine sigmoid routing oracle");
    let got_ids: Vec<usize> = sel.iter().map(|&i| i as usize).collect();

    assert_eq!(
        got_ids, want_ids,
        "engine selection {got_ids:?} != vendor selection {want_ids:?} \
         (scores+bias picks WHO; un-biased picks would be {unbiased_ids:?})"
    );
    let weight_diff = maxdiff(&w, &want_w);
    println!(
        "router[vendor]: ids {want_ids:?} weights {want_w:?} sum {:.6}\n\
         router[engine]: ids {got_ids:?} weights {w:?} sum {:.6} (maxdiff {weight_diff:.3e})",
        want_w.iter().sum::<f32>(),
        w.iter().sum::<f32>(),
    );
    assert!(
        weight_diff <= 1e-6,
        "engine routing weights differ from the vendor expression by {weight_diff:.3e}"
    );

    // The weight SUM is the isolating quantity. Under `norm_topk_prob` the selected weights sum
    // to exactly `routed_scaling_factor` regardless of which experts were picked — which is why
    // the constant-expert experiment (every expert computing the same value) still diverged:
    // the softmax program the engine ran normalizes to 1.
    let engine_sum: f32 = w.iter().sum();
    assert!(
        (engine_sum - SCALING).abs() <= 1e-5,
        "normalized+scaled weights must sum to {SCALING}, got {engine_sum}"
    );

    // The pre-fix program, computed independently: softmax over all experts, top-k on the
    // softmax probabilities (no bias), renormalized to sum 1.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&x| x / total).collect();
    let mut soft_order: Vec<usize> = (0..n_expert).collect();
    soft_order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
    let soft_ids: Vec<usize> = soft_order[..n_used].to_vec();
    let soft_den: f32 = soft_ids.iter().map(|&i| probs[i]).sum();
    let soft_w: Vec<f32> = soft_ids.iter().map(|&i| probs[i] / soft_den).collect();
    let soft_sum: f32 = soft_w.iter().sum();
    println!(
        "router[softmax, the pre-fix program]: ids {soft_ids:?} weights {soft_w:?} sum {soft_sum:.6}"
    );
    assert!(
        (soft_sum - 1.0).abs() <= 1e-5,
        "the softmax program normalizes to 1 — that IS the divergence"
    );
    assert_ne!(
        soft_ids, want_ids,
        "on this vector the softmax program must also pick different experts"
    );
}

/// NON-VACUITY (router). The amplified `e_score_correction_bias` must actually change WHO is
/// picked on the fixture's own router weights, or the "bias is selection-only" property is
/// gated against a no-op. No CUDA: a fixture that drifted back into agreement fails on any
/// machine.
#[test]
fn the_selection_bias_actually_changes_the_picks() {
    let config = mini_config();
    let plan = mini_plan(&config);
    let weights = routing_fixture(&plan);
    let bias_id = router_bias_id(&plan);
    let bias = &weights.get(&bias_id).expect("bias tensor").data;
    assert_eq!(bias.len(), EXPERTS);
    let bias_amplitude = bias.iter().fold(0f32, |m, &v| m.max(v.abs()));

    // Sweep the router over a spread of plausible logit rows: the fixture's router weights times
    // a deterministic set of unit-ish hidden vectors is what the real forward feeds it, and the
    // property under test ("bias decides WHO") only needs the score spread to be comparable to
    // the bias.
    let mut flipped = 0usize;
    let rows = 64usize;
    for row in 0..rows {
        let mut s = (row as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let logits: Vec<f32> = (0..EXPERTS)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect();
        let (biased, _) = HybridModel::moe_route_sigmoid_host_public(
            &logits,
            1,
            EXPERTS,
            TOP_K,
            Some(bias),
            SCALING,
            NORM,
            None,
        )
        .expect("biased routing");
        let (plain, _) = HybridModel::moe_route_sigmoid_host_public(
            &logits, 1, EXPERTS, TOP_K, None, SCALING, NORM, None,
        )
        .expect("un-biased routing");
        let mut a = biased.clone();
        let mut b = plain.clone();
        a.sort_unstable();
        b.sort_unstable();
        if a != b {
            flipped += 1;
        }
    }
    println!(
        "bias non-vacuity: amplitude {bias_amplitude:.4} (gain {ROUTER_BIAS_GAIN}), \
         {flipped}/{rows} rows have biased top-{TOP_K} != un-biased top-{TOP_K}"
    );
    assert!(
        flipped * 4 >= rows,
        "the selection bias flips only {flipped}/{rows} rows — raise ROUTER_BIAS_GAIN, or the \
         'bias is selection-only' half of this gate is vacuous"
    );
}

/// NON-VACUITY (end to end). The four wrong router programs must disagree with the vendor one by
/// a wide margin on THIS fixture, measured on the reference alone. If a mutation collapses toward
/// the truth here, every GPU mutation assertion below stops binding — and it fails on a GPU-less
/// machine rather than passing silently.
#[test]
fn the_router_program_actually_binds() {
    let config = mini_config();
    let plan = mini_plan(&config);
    let weights = routing_fixture(&plan);
    let ids = tokens(8, 0x8_0A17);
    let truth = memra_reference::execute(&plan, &weights, &ids)
        .expect("reference execute")
        .logits;

    for mutation in [
        Mutation::Softmax,
        Mutation::NoSelectionBias,
        Mutation::NoNormalize,
        Mutation::NoScaling,
    ] {
        let got = reference_logits_mutated(&plan, &weights, mutation, &ids);
        let rel = relative(&got, &truth);
        let label = mutation.label();
        println!("binding[{label}]: {rel:.3e} (floor {MUTATION_FLOOR:.1e})");
        assert!(
            rel >= MUTATION_FLOOR,
            "[{label}] the wrong router program is only {rel:.3e} from the right one — this \
             mutation does not bind, so the GPU gate below is vacuous for it"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// GPU gates.
// ---------------------------------------------------------------------------------------------

/// GATE 3 — ROUTED EXPERTS, end to end, stateless prefill, at lengths that cross the KDA scan's
/// chunk size and the MoE dispatch's `MOE_DEV_MAX_T` seam. This is the comparison
/// `swiglu_preclamp_gpu`'s header could not make: before the fix the routed branch ran the
/// softmax router and T=1 measured 1.820e-1; after it, 1.858e-4 — the q8-activation floor.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn routed_prefill_matches_the_reference() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    for &n in &[1usize, 3, 8, 65] {
        let ids = tokens(n, 0x8_0A17 ^ n as u64);
        let want = h.reference_logits(&ids);
        let got = h
            .model
            .forward(&h.engine, &ids)
            .expect("GPU routed prefill");
        println!(
            "routed prefill T={n}: relative maxdiff {:.3e} (tol {TOL:.1e})",
            relative(&got, &want)
        );
        check(&format!("routed prefill T={n}"), &got, &want);
    }
}

/// GATE 4 — ROUTED EXPERTS through the prime + decode seam, whose MoE dispatch is a different arm
/// (`moe_ffn_dev`/sequential at t=1) from prefill's. Both must consult the same router.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn routed_prime_then_decode_matches_the_reference() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let prompt = 6usize;
    let steps = 4usize;
    let ids = tokens(prompt + steps, 0x8_0A17_DEC0);
    let vocab = VOCAB as usize;
    let want = h.reference_logits(&ids);

    let mut cache = memra_engine::cache::Cache::new_planned(&h.engine, &h.model.cfg, &h.plan, 64)
        .expect("cache for the mini glm5_next model");
    let (primed, _seed, _hiddens) = h
        .model
        .prime_cache(&h.engine, &ids[..prompt], &mut cache, 0)
        .expect("GPU routed prime");
    check(
        "routed prime last row",
        &primed,
        &want[(prompt - 1) * vocab..prompt * vocab],
    );
    for step in 0..steps {
        let row = prompt + step;
        let got = h
            .model
            .decode_step(&h.engine, ids[row], &mut cache)
            .expect("GPU routed decode step");
        check(
            &format!("routed decode step {step}"),
            &got,
            &want[row * vocab..(row + 1) * vocab],
        );
    }
}

/// MUTATION CHECK — each wrong router program must fail the exact gate by a wide margin, on the
/// SAME GPU output the passing assertion uses. `softmax-for-sigmoid` is the bug this lane fixes;
/// the other three are the wrong answers a narrower fix would have produced. If any of these ever
/// lands inside TOL, GATE 3 is worthless.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn wrong_router_programs_fail_the_gate() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let ids = tokens(8, 0x8_0A17_9057);
    let got = h
        .model
        .forward(&h.engine, &ids)
        .expect("GPU routed prefill");

    // Control: the GPU really does match the vendor router on this run, so the distances below
    // are the mutations' and not a broken harness's.
    check("routed mutation control", &got, &h.reference_logits(&ids));

    for mutation in [
        Mutation::Softmax,
        Mutation::NoSelectionBias,
        Mutation::NoNormalize,
        Mutation::NoScaling,
    ] {
        let rel = relative(&got, &h.reference_logits_mutated(mutation, &ids));
        let label = mutation.label();
        println!("mutation[{label}]: relative maxdiff {rel:.3e} (tol {TOL:.1e})");
        assert!(
            rel >= MUTATION_FLOOR,
            "[{label}] is only {rel:.3e} away from the GPU's output — this gate does not bind"
        );
    }
}

/// MUTATION CHECK (router weights) — the one property no `RouterPlan` knob can express, so it
/// cannot be gated end to end through the reference: the selection bias must NOT reach the
/// combining weights. `topk_weights = scores.gather(topk_indices)` in the vendor program reads
/// `scores`, not `scores_for_choice`. Measured on the probe vector against the engine's own
/// routing oracle, alongside the two arithmetic steps that follow it.
#[test]
fn the_selected_weights_carry_no_bias() {
    let (logits, bias) = probe_row();
    let n_expert = logits.len();
    let n_used = PROBE_TOP_K;
    let sigmoid = |x: f32| 1.0f32 / (1.0 + (-x).exp());
    let scores: Vec<f32> = logits.iter().copied().map(sigmoid).collect();

    let (sel, got) = HybridModel::moe_route_sigmoid_host_public(
        &logits,
        1,
        n_expert,
        n_used,
        Some(&bias),
        SCALING,
        NORM,
        None,
    )
    .expect("engine sigmoid routing oracle");
    let ids: Vec<usize> = sel.iter().map(|&i| i as usize).collect();

    // Same selection in every arm — only the WEIGHT program differs, so each distance below is
    // attributable to one step of the recipe and nothing else.
    let normalize_scale = |mut w: Vec<f32>| {
        let d: f32 = w.iter().sum::<f32>() + 1e-20;
        for x in &mut w {
            *x = *x / d * SCALING;
        }
        w
    };
    let variants: [(&str, Vec<f32>); 3] = [
        (
            "bias folded into the weights",
            normalize_scale(ids.iter().map(|&i| scores[i] + bias[i]).collect()),
        ),
        (
            "no sum-normalization",
            ids.iter().map(|&i| scores[i] * SCALING).collect(),
        ),
        ("no scaling factor", {
            let mut w: Vec<f32> = ids.iter().map(|&i| scores[i]).collect();
            let d: f32 = w.iter().sum::<f32>() + 1e-20;
            for x in &mut w {
                *x /= d;
            }
            w
        }),
    ];
    let scale = scale_of(&got);
    for (label, wrong) in variants {
        let rel = maxdiff(&got, &wrong) / scale;
        let sum: f32 = wrong.iter().sum();
        println!(
            "router-mutation[{label}]: weights {wrong:?} sum {sum:.6}, relative maxdiff vs the \
             vendor program {rel:.3e}"
        );
        assert!(
            rel >= 1e-2,
            "[{label}] is only {rel:.3e} from the vendor weights — the router gate does not bind"
        );
    }
}
