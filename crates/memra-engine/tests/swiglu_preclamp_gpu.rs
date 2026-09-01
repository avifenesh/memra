//! GPU-vs-reference gate for glm5_next's PRE-activation clamped SwiGLU
//! (`ActivationPlan::SwiGluPreClamped`).
//!
//! Truth is `memra_reference::execute`, whose `activate_pair` runs
//! `silu(gate.min(limit)) * up.clamp(-limit, limit)` — the vendor `Glm5NextTextMLP.forward` /
//! `Glm5NextTextExperts._apply_gate` shape. The candidate is the whole loaded `HybridModel`: the
//! fixture serves the contract tensor names through a `TensorSource`, so config parse, plan
//! compile, the `clamp_exp_at`/`clamp_shexp_at` accessors, the fused-epilogue deny predicate and
//! the kernel are all under gate together.
//!
//! WHY THIS GATE EXISTS. Before it, the plan declared `SwiGluPreClamped { limit: 10.0 }` for every
//! glm5_next MLP while the engine read its clamp from step35-only accessors, so every FFN silently
//! ran plain `silu(gate)*up`: it compiled, it ran, and it produced plausible-but-wrong logits. Two
//! arithmetic mistakes are pinned, not one:
//!   * the PLAIN form — the bug itself;
//!   * the POST form — step35's `min(silu(gate), l) * clamp(up, ±l)`, the wrong answer a one-line
//!     "make the accessors return glm5's limit" fix would have produced.
//!     Both are measured against the SAME GPU output the passing assertions use, so a gate that
//!     stopped binding fails loudly instead of passing three ways.
//!
//! NON-VACUITY IS ENFORCED, NOT ASSUMED. A limit only binds where activations cross it, and on
//! small random weights the shipped 10.0 never would. `MLP_NORM_GAIN` scales each layer's pre-MLP
//! RMSNorm gamma — RMSNorm renormalizes the residual stream, so inflating the embedding would do
//! nothing; the gain has to ride the weight that feeds the FFN — which drives BOTH projections
//! across ±`LIMIT` in both signs. `the_limit_actually_binds` measures the divergence between the
//! three activation forms on the reference alone, with no CUDA, so a fixture that drifted back
//! under the limit fails on a GPU-less machine rather than turning these gates into tautologies.
//!
//! BRANCH COVERAGE, and the one structural limit. glm5_next applies one clamp to three FFN
//! branches, and each takes different engine code:
//!   * DENSE MLP — `dense_*`, exact against the reference (TOL).
//!   * SHARED expert — `shared_expert_*`, exact: the MoE fixture's ROUTED banks are zeroed, so
//!     the layer's output is the shared expert's alone.
//!   * ROUTED experts — `the_routed_branch_selects_the_preclamped_kernel` plus
//!     `the_preclamped_kernel_matches_its_oracle`: the accessor/deny predicates that pick the
//!     dispatch, and the kernel that dispatch lands on. The end-to-end routed comparison lives in
//!     `glm5_routed_router_gpu.rs` — see the block below.
//!
//! WHY THE ROUTED BRANCH IS GATED NEXT DOOR. When this file was written the routed-expert MoE arm
//! did not agree with `memra_reference` on a glm5_next fixture INDEPENDENTLY of the SwiGLU clamp,
//! so no absolute or ratio bound on the whole-model logits could attribute a failure to the
//! activation. Measured then (5090, TF32 off, 2026-08-28), T=8, relative to the pre-clamped
//! reference:
//!   * clamp neutralized (`swiglu_limit` 1e30, both sides plain): 2.3e-1;
//!   * every routed expert weight set to one Q8_0-exact constant, which removes both the q8
//!     quantization error and any top-k selection disagreement: 7.5e-2.
//!     Neither number moved with the activation form and both scaled with `routed_scaling_factor`,
//!     which is what named the router rather than the clamp. ROOT-CAUSED 2026-08-28:
//!     `ModelConfig::sigmoid_router()` had no `cfg.glm5` arm, so it answered `None` for every
//!     glm5_next model and the routed branch rode the SOFTMAX router — no sigmoid, no
//!     `e_score_correction_bias`, weights normalized to 1 instead of `routed_scaling_factor` 2.5.
//!     (The constant-expert figure is exactly that: with every expert computing the same value the
//!     routed output is `sum(weights) * expert`, so 1.0-vs-2.5 survives even when selection cannot
//!     matter.) The fix is the accessor arm; the end-to-end routed gate, its four wrong-router
//!     mutations and the vendor-semantics receipts are `glm5_routed_router_gpu.rs`. This file keeps
//!     its dispatch-and-kernel routed coverage — the two gates are complementary, and this one still
//!     zeroes the routed banks so the shared branch can be gated at TOL. (The house's other
//!     reference-parity gates — `hyper_connections_gpu`, `kda_fixture_gpu` — are dense-only for
//!     adjacent reasons.)
//!
//! Mixers are KDA under glm5_next's mHC residual; both have their own gates (`kda_fixture_gpu`,
//! `hyper_connections_gpu`), so a failure here is the activation's.
//!
//! GPU-gated (`#[ignore]`); rig law = exactness only, never timing. Run under
//! `flock /tmp/memra-5090.lock` with `NVIDIA_TF32_OVERRIDE=0` and `-- --ignored`.

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::{ActivationPlan, MlpPlan, ModelPlan};
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, LayerTensor, OutputHead, TensorContract, TensorId,
    TensorMatch,
};
use memra_reference::{ReferenceTensor, deterministic_fixture};
use std::borrow::Cow;
use std::collections::BTreeMap;

const HIDDEN: usize = 128;
const VOCAB: u32 = 32;

/// The clamp under gate. Deliberately NOT glm5_next's shipped 10.0: at limit L the PRE and POST
/// forms differ by at most `L*(1 - sigmoid(L))` per element, which is 4.5e-4 at L=10 — small
/// enough to hide inside this gate's tolerance, so a post-for-pre substitution could slip through.
/// At L=1.5 the same bound is 0.27. The FORM is what is under test; the shipped limit is a config
/// value pinned by `crates/memra-gguf/src/config.rs`'s own parser tests.
const LIMIT: f32 = 1.5;

/// Pre-MLP RMSNorm gamma multiplier — the non-vacuity construction (see the module header).
/// `the_limit_actually_binds` is what proves this number is large enough.
const MLP_NORM_GAIN: f32 = 6.0;

/// Scale-relative bound for the branches the reference can match exactly, same shape and same
/// calibration discipline as `hyper_connections_gpu`'s: the reference sums on the host in
/// declaration order while the GPU runs cuBLASLt GEMMs and warp-tree reductions, so bit-identity
/// is not the bar. Calibrate downward, never upward.
const TOL: f32 = 2e-5;

/// Floor for "this wrong activation form fails by a wide margin", on the branches gated at TOL.
/// Measured margins are 4.7e-1 (post) and 1.0e0 (plain), so this floor sits ~50x below the
/// smaller of them and ~500x above TOL.
const MUTATION_FLOOR: f32 = 1e-2;

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

/// Which FFN branches a fixture exercises.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    /// Two dense-MLP layers. The only shape whose FFN the f32 reference can match exactly.
    Dense,
    /// Layer 0 dense, layer 1 MoE with a shared expert. `zero_routed` blanks the routed banks so
    /// the MoE layer's output is the shared expert's alone (see `crossing_fixture`).
    Moe { zero_routed: bool },
}

impl Shape {
    fn label(self) -> &'static str {
        match self {
            Shape::Dense => "dense",
            Shape::Moe { zero_routed: true } => "shared expert",
            Shape::Moe { zero_routed: false } => "routed",
        }
    }
}

/// A glm5_next trunk, expressed the only way the engine will accept: a real `config.json` through
/// the real `HfConfig`/`ModelConfig` path, compiled by the real glm5_next model pack.
/// `HybridModel::load_from_source` compiles the plan from `src.config()`, so a hand-built
/// `ModelPlan` could not reach it.
///
/// `head_dim` is 128 because that is the only width `memra_kda_scan_s128` is instantiated for.
/// The MLA/DSA fields are required by the glm5_next config parser and are inert: no layer in
/// `layer_types` selects them.
fn mini_config_json(shape: Shape) -> String {
    let (mlp_layer_types, first_k_dense_replace) = match shape {
        Shape::Dense => (r#"["dense", "dense"]"#, 2),
        Shape::Moe { .. } => (r#"["dense", "sparse"]"#, 1),
    };
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
      "swiglu_limit": {LIMIT},
      "tie_word_embeddings": true,
      "hc_mult": 4,
      "hc_eps": 1e-06,
      "hc_sinkhorn_iters": 20,
      "mhc": true,
      "layer_types": ["linear_attention", "linear_attention"],
      "mlp_layer_types": {mlp_layer_types},
      "first_k_dense_replace": {first_k_dense_replace},
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
    }}"#
    )
}

fn mini_config(shape: Shape) -> ModelConfig {
    ModelConfig::from_hf(&HfConfig::parse(&mini_config_json(shape)))
}

fn mini_plan(config: &ModelConfig) -> ModelPlan {
    memra_gguf::model_packs::for_config(config)
        .expect("glm5_next model pack matches the mini config")
        .compile_plan(config)
        .expect("mini glm5_next plan compiles")
}

/// The same plan with every MLP's activation replaced. Only the REFERENCE reads these variants —
/// the GPU always runs the real plan — so each one answers "what would the reference say if the
/// engine had implemented THIS form?" against one fixed set of weights.
fn with_activation(plan: &ModelPlan, activation: &ActivationPlan) -> ModelPlan {
    let mut mutated = plan.clone();
    for layer in &mut mutated.layers {
        match &mut layer.mlp {
            MlpPlan::Dense(dense) => dense.activation = activation.clone(),
            MlpPlan::Moe(moe) => moe.activation = activation.clone(),
        }
    }
    mutated
}

fn post_form(plan: &ModelPlan) -> ModelPlan {
    with_activation(plan, &ActivationPlan::SwiGluClamped { limit: LIMIT })
}

fn plain_form(plan: &ModelPlan) -> ModelPlan {
    with_activation(plan, &ActivationPlan::Silu)
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
        // `micro_gguf`'s fixtures use. `crossing_fixture` has already snapped their values onto
        // that grid, so the bytes the GPU dequantizes ARE the numbers the reference reads and the
        // weight encoding costs no parity. The expert matvec's q8_1 ACTIVATION quantization is a
        // separate, unavoidable floor — see the module header.
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
/// activation form this gate is trying to measure.
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

/// The fixture, with each layer's pre-MLP norm gamma scaled by `MLP_NORM_GAIN` (the non-vacuity
/// construction) and the expert banks snapped onto the Q8_0 grid they are served on.
/// `Shape::Moe { zero_routed: true }` instead blanks the routed banks: routing still runs and
/// still weights its picks, but every routed expert contributes zero on BOTH sides, leaving the
/// MoE layer's output equal to the shared expert's — which is how the shared branch gets gated at
/// TOL despite the routed branch's q8 floor.
fn crossing_fixture(plan: &ModelPlan, shape: Shape) -> BTreeMap<TensorId, ReferenceTensor> {
    let mut weights = deterministic_fixture(plan)
        .expect("deterministic glm5_next fixture")
        .weights;
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
    let zero_routed = matches!(shape, Shape::Moe { zero_routed: true });
    for (id, tensor) in weights.iter_mut() {
        if !is_expert_bank(id) {
            continue;
        }
        if zero_routed {
            tensor.data.fill(0.0);
        } else {
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
    fn new(shape: Shape) -> Self {
        force_true_f32();
        let config = mini_config(shape);
        let plan = mini_plan(&config);
        let weights = crossing_fixture(&plan, shape);
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

    /// Reference logits under the plan's own activation (the pre-clamped truth).
    fn reference_logits(&self, tokens: &[u32]) -> Vec<f32> {
        self.reference_logits_as(&self.plan, tokens)
    }

    fn reference_logits_as(&self, plan: &ModelPlan, tokens: &[u32]) -> Vec<f32> {
        memra_reference::execute(plan, &self.weights, tokens)
            .expect("reference execute")
            .logits
    }
}

/// The plans really do declare glm5_next's pre-clamped SwiGLU on all three MLP branches, and the
/// two shapes really do cover dense / routed / shared. No CUDA — if this fails, every assertion
/// below is measuring something other than what it claims.
#[test]
fn the_mini_plans_declare_preclamped_swiglu_on_dense_routed_and_shared() {
    let want = ActivationPlan::SwiGluPreClamped { limit: LIMIT };

    let dense_plan = mini_plan(&mini_config(Shape::Dense));
    assert_eq!(dense_plan.layers.len(), 2);
    assert_eq!(dense_plan.hidden_size as usize, HIDDEN);
    for layer in &dense_plan.layers {
        let MlpPlan::Dense(dense) = &layer.mlp else {
            panic!("dense shape layer {} must be a dense MLP", layer.index);
        };
        assert_eq!(dense.activation, want, "dense MLP activation");
    }

    let config = mini_config(Shape::Moe { zero_routed: false });
    let moe_plan = mini_plan(&config);
    assert!(matches!(moe_plan.layers[0].mlp, MlpPlan::Dense(_)));
    let MlpPlan::Moe(moe) = &moe_plan.layers[1].mlp else {
        panic!("moe shape layer 1 must be the MoE branch");
    };
    assert_eq!(moe.activation, want, "routed + shared expert activation");
    assert!(
        moe.shared.is_some(),
        "the MoE layer must carry a shared expert, or the shared branch is ungated"
    );

    // The engine-side accessors and the fused-epilogue deny predicate, on the same config.
    use memra_gguf::config::SwigluClamp;
    for il in 0..2u32 {
        assert_eq!(
            config.clamp_exp_at(il),
            Some(SwigluClamp::Pre(LIMIT)),
            "routed clamp at layer {il}"
        );
        assert_eq!(
            config.clamp_shexp_at(il),
            Some(SwigluClamp::Pre(LIMIT)),
            "shared/dense clamp at layer {il}"
        );
        assert!(
            config.swiglu_clamped_at(il),
            "layer {il} MUST deny the fused plain-SiLU epilogues"
        );
    }
    assert!(config.swiglu_clamped_anywhere());
}

/// NON-VACUITY. The three activation forms must disagree on THIS fixture by a wide margin, which
/// is only true where activations actually cross ±LIMIT. Runs on the reference alone, so a
/// fixture that drifted back under the limit fails here — on any machine, GPU or not — instead of
/// leaving every gate below passing three ways.
#[test]
fn the_limit_actually_binds() {
    for shape in [Shape::Dense, Shape::Moe { zero_routed: false }] {
        let plan = mini_plan(&mini_config(shape));
        let weights = crossing_fixture(&plan, shape);
        let ids = tokens(8, 0xC1A_B14D);

        let run = |p: &ModelPlan| {
            memra_reference::execute(p, &weights, &ids)
                .expect("reference execute")
                .logits
        };
        let pre = run(&plan);
        let pre_vs_post = relative(&run(&post_form(&plan)), &pre);
        let pre_vs_plain = relative(&run(&plain_form(&plan)), &pre);
        let label = shape.label();
        println!(
            "binding[{label}]: pre-vs-post {pre_vs_post:.3e}, pre-vs-plain {pre_vs_plain:.3e} \
             (floor {MUTATION_FLOOR:.1e})"
        );
        assert!(
            pre_vs_post >= MUTATION_FLOOR,
            "[{label}] the clamp does not bind: pre and post forms agree to {pre_vs_post:.3e}. \
             Raise MLP_NORM_GAIN or lower LIMIT — the GPU gates are vacuous in this state."
        );
        assert!(
            pre_vs_plain >= MUTATION_FLOOR,
            "[{label}] the clamp does not bind: pre and plain forms agree to {pre_vs_plain:.3e}"
        );
    }
}

/// GATE 1 — DENSE MLP, stateless prefill, at lengths that cross the KDA scan's chunk size.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn dense_prefill_matches_the_reference() {
    let _gpu = gpu_guard();
    let h = Harness::new(Shape::Dense);
    for &n in &[1usize, 3, 8, 65] {
        let ids = tokens(n, 0x9C1A ^ n as u64);
        let want = h.reference_logits(&ids);
        let got = h.model.forward(&h.engine, &ids).expect("GPU dense prefill");
        check(&format!("dense prefill T={n}"), &got, &want);
    }
}

/// GATE 2 — DENSE MLP, prime then decode. The decode seam has its own FFN dispatch (`decode.rs`'s
/// `ffn_swiglu_decode` and the fused q8 fast path it must decline), so prefill parity does not
/// imply it.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn dense_prime_then_decode_matches_the_reference() {
    let _gpu = gpu_guard();
    let h = Harness::new(Shape::Dense);
    let prompt = 6usize;
    let steps = 4usize;
    let ids = tokens(prompt + steps, 0xC1A_DEC0);
    let vocab = VOCAB as usize;
    let want = h.reference_logits(&ids);

    // `new_planned`, not `new`: the KDA layers' recurrent state and conv ring are allocated from
    // the ModelPlan's StatePlan, not from the config alone.
    let mut cache = memra_engine::cache::Cache::new_planned(&h.engine, &h.model.cfg, &h.plan, 64)
        .expect("cache for the mini glm5_next model");
    let (primed, _seed, _hiddens) = h
        .model
        .prime_cache(&h.engine, &ids[..prompt], &mut cache, 0)
        .expect("GPU dense prime");
    check(
        "dense prime last row",
        &primed,
        &want[(prompt - 1) * vocab..prompt * vocab],
    );
    for step in 0..steps {
        let row = prompt + step;
        let got = h
            .model
            .decode_step(&h.engine, ids[row], &mut cache)
            .expect("GPU dense decode step");
        check(
            &format!("dense decode step {step}"),
            &got,
            &want[row * vocab..(row + 1) * vocab],
        );
    }
}

/// GATE 3 — SHARED expert. The MoE layer's routed banks are zero on both sides, so what this
/// compares is the shared expert's own FFN — a different dispatch from the dense MLP's (the
/// `gate_shexp`/`up_shexp`/`down_shexp` arms, including the fused dual-matvec ones that have to
/// decline the PRE form).
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn shared_expert_matches_the_reference() {
    let _gpu = gpu_guard();
    let h = Harness::new(Shape::Moe { zero_routed: true });
    // T=65 is load-bearing: the pairs arm only fires above `MOE_DEV_MAX_T`, so a T<=8 sweep never
    // exercises the deny chain that keeps a clamped layer out of it. At 65 the layer must reach
    // the staged prefill arm's `ffn_act_lim`, and a leak trips `moe_ffn_pairs`'s
    // `debug_assert!(!swiglu_clamped_anywhere())` in a debug build instead of returning quietly
    // wrong logits.
    for &n in &[1usize, 3, 8, 65] {
        let ids = tokens(n, 0x5AED ^ n as u64);
        let want = h.reference_logits(&ids);
        let got = h.model.forward(&h.engine, &ids).expect("GPU shexp prefill");
        check(&format!("shared expert T={n}"), &got, &want);
    }
}

/// GATE 3b — SHARED expert through the decode seam and its own fused shexp matvec arms.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn shared_expert_decode_matches_the_reference() {
    let _gpu = gpu_guard();
    let h = Harness::new(Shape::Moe { zero_routed: true });
    let prompt = 5usize;
    let steps = 3usize;
    let ids = tokens(prompt + steps, 0x5AED_DEC0);
    let vocab = VOCAB as usize;
    let want = h.reference_logits(&ids);
    let mut cache = memra_engine::cache::Cache::new_planned(&h.engine, &h.model.cfg, &h.plan, 64)
        .expect("cache for the mini glm5_next model");
    let (primed, _seed, _hiddens) = h
        .model
        .prime_cache(&h.engine, &ids[..prompt], &mut cache, 0)
        .expect("GPU shexp prime");
    check(
        "shared expert prime last row",
        &primed,
        &want[(prompt - 1) * vocab..prompt * vocab],
    );
    for step in 0..steps {
        let row = prompt + step;
        let got = h
            .model
            .decode_step(&h.engine, ids[row], &mut cache)
            .expect("GPU shexp decode step");
        check(
            &format!("shared expert decode step {step}"),
            &got,
            &want[row * vocab..(row + 1) * vocab],
        );
    }
}

/// GATE 4 — the ROUTED branch's DISPATCH. `moe_ffn_*` sources its routed-expert limit from
/// `clamp_exp_at` and hands it to `ffn_act_lim`, whose match is exhaustive over `SwigluClamp`;
/// every fused routed epilogue (pairs, dev, grouped-decode, the slab-local pair) is keyed off
/// `!swiglu_clamped_at`. Pinning both predicates pins which kernel the routed branch runs, which
/// is the half of routed coverage a whole-model comparison cannot supply here (module header).
#[test]
fn the_routed_branch_selects_the_preclamped_kernel() {
    use memra_gguf::config::SwigluClamp;
    let config = mini_config(Shape::Moe { zero_routed: false });
    let il = 1u32; // the MoE layer
    assert_eq!(
        config.clamp_exp_at(il),
        Some(SwigluClamp::Pre(LIMIT)),
        "routed experts must resolve glm5_next's PRE form, never step35's Post"
    );
    assert!(
        config.swiglu_clamped_at(il),
        "the routed fused epilogues (pairs/dev/gdec/slab) are plain SiLU and MUST be denied"
    );
    assert!(
        config.swiglu_clamped_anywhere(),
        "the no-`il` seams' whole-model assertion must also see the clamp"
    );
}

/// GATE 5 — the KERNEL all three branches share, against a CPU oracle of the vendor expression,
/// on inputs that cross the limit in every direction: gate far above it, gate far below zero
/// (where the ONE-sided clamp must not touch it), up beyond +limit and beyond -limit.
///
/// The mechanism probe is the point: at `gate > limit` the pre- and post-clamp kernels must
/// DISAGREE, which is what makes every "wrong form" assertion in this file mean something. It
/// mirrors the `swiglu_clamped` cell in `src/bin/kernel_check.rs`.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn the_preclamped_kernel_matches_its_oracle() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");

    // Deterministic sweep over [-4*LIMIT, 4*LIMIT] on both operands, so every quadrant of the
    // clamp (gate above / below, up above / below / inside) is populated.
    let n = 4096usize;
    let span = 4.0 * LIMIT;
    let gate: Vec<f32> = (0..n)
        .map(|i| span * (2.0 * (i as f32 / (n - 1) as f32) - 1.0))
        .collect();
    let up: Vec<f32> = (0..n)
        .map(|i| span * (2.0 * (((i * 7) % n) as f32 / (n - 1) as f32) - 1.0))
        .collect();
    assert!(
        gate.iter().any(|&g| g > LIMIT) && gate.iter().any(|&g| g < -LIMIT),
        "the sweep must cross the gate clamp in both directions"
    );
    assert!(
        up.iter().any(|&u| u > LIMIT) && up.iter().any(|&u| u < -LIMIT),
        "the sweep must cross the up clamp in both directions"
    );

    let gd = e.htod(&gate).expect("gate htod");
    let ud = e.htod(&up).expect("up htod");
    let mut dd = e.zeros(n).expect("dst alloc");
    e.swiglu_preclamped_mul_scaled(&gd, &ud, 1.0, 1.0, LIMIT, &mut dd, n)
        .expect("pre-clamped kernel");
    let got = e.dtoh(&dd).expect("dst dtoh");

    // Vendor expression: silu(min(gate, limit)) * clamp(up, ±limit).
    let want: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| {
            let x = g.min(LIMIT);
            (x / (1.0 + (-x).exp())) * u.clamp(-LIMIT, LIMIT)
        })
        .collect();
    let rel = relative(&got, &want);
    println!("preclamp kernel vs oracle: relative maxdiff {rel:.3e}");
    assert!(
        rel <= 1e-6,
        "pre-clamped kernel vs CPU oracle {rel:.3e} — the kernel is not the vendor expression"
    );

    // Mechanism probe: the post-clamp kernel is a DIFFERENT program above the limit.
    let mut pd = e.zeros(n).expect("post dst alloc");
    e.swiglu_clamped_mul_scaled(&gd, &ud, 1.0, 1.0, LIMIT, &mut pd, n)
        .expect("post-clamped kernel");
    let post = e.dtoh(&pd).expect("post dtoh");
    let sep = relative(&post, &want);
    println!("preclamp vs postclamp on the same inputs: relative maxdiff {sep:.3e}");
    assert!(
        sep >= 1e-2,
        "the two clamp kernels agree to {sep:.3e} — one of them is not doing what its name says"
    );
}

/// MUTATION CHECK — swapping PRE for step35's POST form must fail the exact gates by a wide
/// margin, on the same GPU output the passing assertion uses. If this ever lands inside TOL the
/// gates above are worthless.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn post_form_fails_the_gate() {
    let _gpu = gpu_guard();
    for shape in [Shape::Dense, Shape::Moe { zero_routed: true }] {
        let h = Harness::new(shape);
        let ids = tokens(8, 0x9C1A_9057);
        let got = h.model.forward(&h.engine, &ids).expect("GPU prefill");
        let label = shape.label();
        check(
            &format!("{label} mutation control (pre)"),
            &got,
            &h.reference_logits(&ids),
        );

        let rel = relative(&got, &h.reference_logits_as(&post_form(&h.plan), &ids));
        println!("mutation[{label}] post-for-pre: relative maxdiff {rel:.3e} (tol {TOL:.1e})");
        assert!(
            rel >= MUTATION_FLOOR,
            "[{label}] post-for-pre substitution is only {rel:.3e} away — this gate does not bind"
        );
    }
}

/// THE BUG THIS LANE FIXES — plain `silu(gate)*up`, which is what every glm5_next FFN ran while
/// the clamp accessors were step35-only. Must fail by a wide margin against the same GPU output.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn plain_form_fails_the_gate() {
    let _gpu = gpu_guard();
    for shape in [Shape::Dense, Shape::Moe { zero_routed: true }] {
        let h = Harness::new(shape);
        let ids = tokens(8, 0x9C1A_9057);
        let got = h.model.forward(&h.engine, &ids).expect("GPU prefill");
        let label = shape.label();
        let rel = relative(&got, &h.reference_logits_as(&plain_form(&h.plan), &ids));
        println!("mutation[{label}] plain-for-pre: relative maxdiff {rel:.3e} (tol {TOL:.1e})");
        assert!(
            rel >= MUTATION_FLOOR,
            "[{label}] the unclamped form is only {rel:.3e} away — this gate does not bind"
        );
    }
}
