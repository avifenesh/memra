//! Reference-oracle acceptance gate for glm5_next's GROUPED MoE PREFILL arm
//! (`MEMRA_MOE_GROUPED_PREFILL`): token-sort by expert, one grouped NVFP4 tensor-core GEMM per
//! projection per layer-chunk, sigmoid `noaux_tc` routing, PRE-clamped SwiGLU, per-expert
//! `weight_scale_2` macro fold.
//!
//! Truth is `memra_reference::execute`, the unfused f32 executor, on the SAME fixture family the
//! fused-epilogue gate qualified for this arch (`glm5_moe_epilogue_gpu.rs`: real NVFP4 expert
//! banks in memra's `block_nvfp4` layout, a live `<stem>.scale` macro plane, a live shared
//! expert, and the clamp made to bind at LIMIT=0.75 with MLP_NORM_GAIN=2.0, the (limit, gain)
//! pair that gate chose by measurement). The fixture builders below are that gate's, kept
//! textually in step so the two gates measure one fixture.
//!
//! WHAT THE BAR IS, AND WHY IT IS NOT BYTE IDENTITY. The grouped GEMM class is measured
//! non-bit-stable run to run (`run_tensor_parallel_routes_nvfp4_prime_grouped`'s
//! MEMRA_MOE_DETERM note in `hybrid_forward.rs`), so the honest acceptance is:
//!   * REFERENCE BAND: the epilogue gate's tolerance class (TOL 1e-2 against the measured
//!     4.822e-3 sequential-control floor on this fixture);
//!   * ROUTING EXACTNESS: the selected experts and routing weights are BIT-identical to the
//!     sequential arm BY CONSTRUCTION (the arm makes the same `moe_router_logits` +
//!     `moe_route_sigmoid_cfg` invocation), and `routing_is_identical_between_arms` measures it
//!     through the MEMRA_MOE_TRACE / MEMRA_MOE_WEIGHT_TRACE files rather than asserting it from
//!     provenance. Only the GEMM accumulation order may move.
//!   * CLASS SEAMS: every chunk width states which dispatch class it rides, and the engagement
//!     counter (incremented at the arm's own call site,
//!     `memra_engine::moe_grouped_prefill_dispatches`, LAW:wiring-assertions-match-prose)
//!     asserts it per width. The width set {1, 2, 15, 16, 17, 64, 4096} deliberately crosses
//!     the t=16 knee: `MOE_DEV_MAX_T` is 16 (decode widths 2..=16 ride the per-token program;
//!     grouped/pairs prefill classes start at 17) AND `Engine::matmul`'s GEMM_M_THRESHOLD is 16
//!     (the shexp trio crosses from the matvec class onto the GEMM class at t>=16, the batch
//!     lane's numeric knee). So t=16 is the knee row: per-token routed experts with GEMM-class
//!     shexp; t=17 is the first grouped row.
//!
//! THE RED ARMS, and why each is a wrong answer someone would actually ship:
//!   * `shared-expert-dropped` / `softmax-for-sigmoid` / `post-for-pre-clamp` / `plain-swiglu`
//!     are reference-plan mutations, measured against the SAME GPU output the passing assertion
//!     uses (GATE:pin-against-truth). The shared expert is a separate branch the grouped arm
//!     must still sum in; the router half is the predicate this arm relaxes; the clamp forms
//!     are the family's known plausible-but-wrong substitutions.
//!   * `macro-plane-flattened`: minted into the FIXTURE bytes (`FixtureMutation::MacroDropped`),
//!     because no `ModelPlan` knob can express a checkpoint artifact. This is the arm that
//!     proves the grouped path's `scale_rows` gate/up fold and the down-into-scatter-weight
//!     fold are load-bearing.
//!   * `token-sort permutation` and `scatter off-by-one`: engine-side breakages that CANNOT be
//!     expressed against the reference: they were built deliberately, run against this gate,
//!     banked RED, and reverted (the epilogue lane's wrong-fusion protocol; receipts in
//!     `research/glm53-flash-bringup-20260827/moe-grouped-prefill-receipts/`).
//!
//! FAIL-CLOSED IS EXERCISED, NOT ASSUMED: `slru_placement_fails_closed` runs the arm ON with no
//! resident slab and requires ZERO grouped dispatches with a green sequential fallback: the
//! rollback seam and the placement predicate are both measured.
//!
//! SCOPE, or what this gate does NOT prove. A 2-layer fixture, not the 190.7 GB artifact. It
//! proves the grouped arm is the same PROGRAM as the reference for this arch's router, clamp,
//! macro plane, token-sort and scatter, at widths crossing every dispatch seam. It proves
//! nothing about throughput (the flip needs the interleaved x5 box A/B with the sampled twin,
//! per the FLAGS row) and nothing about any other sigmoid-router arch (no generic-model claims:
//! the arm itself is keyed to `cfg.glm5`).
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

/// GLM-5.3-Flash's real routing shape, shrunk only in width (the epilogue gate's constants).
const EXPERTS: usize = 8;
const TOP_K: usize = 3;
const SCALING: f32 = 2.5;
const NORM: bool = true;
const MOE_FF: usize = 64;

/// See `glm5_moe_epilogue_gpu.rs` for the measured derivation of this (limit, gain) pair and of
/// TOL/MUTATION_FLOOR: LIMIT far below the shipped 10.0 so the PRE/POST forms disagree by 0.23
/// per element instead of 4.5e-4, gain 2.0 so the sequential control floor sits at 4.822e-3.
const LIMIT: f32 = 0.75;
const MLP_NORM_GAIN: f32 = 2.0;
const ROUTER_BIAS_GAIN: f32 = 12.0;

/// Scale-relative bound: the epilogue gate's tolerance class, calibrated there from the
/// measured unfused control (worst row 4.822e-3, the fixture's q8_1-activation floor). The
/// grouped arm is a DIFFERENT numeric class (f16-mirror grouped GEMM), so its distance may sit
/// above the sequential arm's floor but must stay inside the same class bound. Calibrate
/// downward, never upward: if a run drifts above this, find out why rather than raising it.
const TOL: f32 = 1e-2;
const MUTATION_FLOOR: f32 = 3e-2;

/// The dispatch seam under gate: `MOE_DEV_MAX_T` in `hybrid_forward.rs`: per-token classes
/// serve t <= 16, grouped/pairs prefill classes start at 17. Also `Engine::matmul`'s
/// GEMM_M_THRESHOLD, the shexp trio's matvec->GEMM numeric knee.
const KNEE_T: usize = 16;

/// Chunk widths for the per-width gates, deliberately bracketing the knee. The 4096 row (the
/// real chunk cap, `PRIME_CHUNK_MAX_TOKENS`) runs as its own test: the reference executor is
/// CPU f32 and that row dominates the suite's wall clock.
const WIDTHS: [usize; 6] = [1, 2, 15, 16, 17, 64];
const CHUNK_CAP: usize = 4096;

/// Which class a width rides, and how many grouped dispatches the fixture's ONE MoE layer must
/// record for it. Stated here once so every test asserts the same seam map.
fn expected_class(t: usize) -> (&'static str, u64) {
    if t > KNEE_T {
        ("grouped f16 GEMM (routed) + GEMM-class shexp", 1)
    } else if t == KNEE_T {
        ("per-token routed loop + GEMM-class shexp (the knee row)", 0)
    } else {
        ("per-token routed loop + matvec-class shexp", 0)
    }
}

/// Per-expert `weight_scale_2` macro scales: the epilogue gate's plane (both bands clear of
/// 1.0 so flattening is a 1.25x-2x error on EVERY expert).
fn macro_scales() -> Vec<f32> {
    (0..EXPERTS)
        .map(|e| {
            if e < EXPERTS / 2 {
                0.5 + 0.1 * e as f32
            } else {
                1.2 + 0.1 * (e - EXPERTS / 2) as f32
            }
        })
        .collect()
}

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

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

/// The epilogue gate's mini glm5_next config: real `config.json` through the real
/// `HfConfig`/`ModelConfig` path, compiled by the real glm5_next model pack.
fn mini_config_json() -> String {
    format!(
        r#"{{
      "model_type": "glm5_next_text",
      "num_hidden_layers": 2,
      "num_nextn_predict_layers": 0,
      "hidden_size": {HIDDEN},
      "intermediate_size": 64,
      "vocab_size": 32,
      "max_position_embeddings": 8192,
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
// The wrong programs. Only the REFERENCE reads the plan mutations; the GPU always runs the
// real plan. The two engine-side breakages this gate additionally banked (token-sort
// permutation, scatter off-by-one) cannot be expressed here; see the receipts directory.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mutation {
    PostClamp,
    PlainSwiglu,
    NoSharedExpert,
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

/// The macro-plane mutation: a checkpoint artifact, minted into the bytes the ENGINE loads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FixtureMutation {
    None,
    MacroDropped,
}

// ---------------------------------------------------------------------------------------------
// Fixture (the epilogue gate's, verbatim in construction).
// ---------------------------------------------------------------------------------------------

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

fn engine_fixture(plan: &ModelPlan) -> BTreeMap<TensorId, ReferenceTensor> {
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

fn check_all(label: &str, got: &[Row], want: &[Vec<f32>]) {
    assert_eq!(got.len(), want.len(), "{label}: row count mismatch");
    let mut worst = 0.0f32;
    let mut worst_row = String::new();
    for (row, w) in got.iter().zip(want) {
        assert!(
            row.logits.iter().all(|v| v.is_finite()),
            "{label} {}: GPU output has non-finite values",
            row.name
        );
        let rel = relative(&row.logits, w);
        println!(
            "{label} {}: relative maxdiff {rel:.3e} (tol {TOL:.1e})",
            row.name
        );
        if rel > worst {
            worst = rel;
            worst_row = row.name.clone();
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
// Host-only gates. No CUDA. These carry the reference-side RED evidence.
// ---------------------------------------------------------------------------------------------

/// The plan the engine will group must be the plan this gate thinks it is grouping, and the
/// seam constants must be the engine's. `expected_class` is half the gate: if the knee moves,
/// this fails on a GPU-less machine.
#[test]
#[allow(clippy::assertions_on_constants)] // allow: const pins; fail loudly if the seam map drifts
fn the_plan_declares_the_program_under_gate() {
    let config = mini_config();
    let plan = mini_plan(&config);

    let MlpPlan::Moe(moe) = &plan.layers[1].mlp else {
        panic!("layer 1 must be the routed-MoE branch");
    };
    assert_eq!(
        moe.activation,
        ActivationPlan::SwiGluPreClamped { limit: LIMIT },
        "the grouped arm is qualified for the PRE form only (its dispatch refuses POST)"
    );
    assert_eq!(
        moe.router,
        RouterPlan::Sigmoid {
            normalize_selected: NORM,
            scaling_factor: SCALING,
            selection_bias: true,
        },
        "the grouped arm must route the noaux_tc recipe through the host oracle"
    );
    assert!(
        moe.shared.is_some(),
        "the fixture must carry a shared expert, or `shared-expert-dropped` cannot turn red"
    );
    assert_eq!(moe.expert_count as usize, EXPERTS);
    assert_eq!(moe.experts_per_token as usize, TOP_K);
    assert_eq!(
        config.sigmoid_router(),
        Some((SCALING, NORM)),
        "the dispatch predicate keys off sigmoid_router()"
    );
    assert!(
        config.glm5.is_some(),
        "the arm is keyed to cfg.glm5 (no generic-model support claims); a fixture that stops \
         parsing as glm5_next gates nothing"
    );
    assert!(
        config.swiglu_clamped_at(1),
        "the clamp must be live on the routed layer"
    );

    // The seam map: t <= 16 per-token, t >= 17 grouped, with 16 the knee row. KNEE_T mirrors
    // MOE_DEV_MAX_T and GEMM_M_THRESHOLD; the per-width dispatch assertions below enforce the
    // engine side of the same map.
    assert_eq!(KNEE_T, 16);
    for (t, want) in [(1, 0u64), (2, 0), (15, 0), (16, 0), (17, 1), (64, 1)] {
        assert_eq!(expected_class(t).1, want, "seam map drifted at t={t}");
    }
    assert_eq!(expected_class(CHUNK_CAP).1, 1);
}

/// NON-VACUITY + reference-side RED evidence: each wrong program must disagree with the vendor
/// one by a wide margin on THIS fixture, measured on the reference alone, at a width the
/// GROUPED arm actually serves (t=17, past the knee; the epilogue gate measured t=8, since a
/// mutation that collapsed toward truth only at grouped widths would leave this gate vacuous).
#[test]
fn the_wrong_programs_actually_bind_at_grouped_widths() {
    let config = mini_config();
    let plan = mini_plan(&config);
    let engine_weights = engine_fixture(&plan);
    let weights = reference_fixture(&plan, &engine_weights);
    let ids = tokens(KNEE_T + 1, 0x_67_F0_11);
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
        println!(
            "binding[{label}] at t={}: {rel:.3e} (floor {MUTATION_FLOOR:.1e})",
            ids.len()
        );
        assert!(
            rel >= MUTATION_FLOOR,
            "[{label}] the wrong program is only {rel:.3e} from the right one at a grouped \
             width; this mutation does not bind there"
        );
    }
}

/// NON-VACUITY (macro plane): all scales clear of 1.0, or MacroDropped is a partial no-op.
#[test]
fn the_macro_plane_is_live() {
    let macros = macro_scales();
    assert_eq!(macros.len(), EXPERTS);
    assert!(
        macros.iter().all(|&m| (m - 1.0).abs() > 0.05),
        "every macro scale must sit clear of 1.0, or MacroDropped is a no-op for some experts"
    );
    println!("macro plane: {macros:?}");
}

// ---------------------------------------------------------------------------------------------
// GPU gates.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// The shipped per-token sequential loop (`MEMRA_MOE_GROUPED_PREFILL=0`).
    Sequential,
    /// The grouped prefill arm under gate (`=1`).
    Grouped,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Arm::Sequential => "sequential per-token loop",
            Arm::Grouped => "grouped MoE prefill",
        }
    }
}

/// The grouped arm is SLAB-ONLY by design (an SLRU cannot hold a chunk's expert working set on
/// the serving recipe, and it must fail closed there). Both placements are gated: Slab is the
/// serving config, Slru is the fail-closed proof.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Placement {
    Slab,
    Slru,
}

struct Row {
    name: String,
    logits: Vec<f32>,
    /// Grouped-arm dispatches this row alone produced (before/after counter delta).
    dispatches: u64,
    /// The class this width was asserted to ride.
    class: &'static str,
}

struct Run {
    rows: Vec<Row>,
    dispatches: u64,
}

/// Run the width sweep + the prime/decode workload under one arm. A FRESH Engine and model load
/// per arm (residency is a load-time decision, and the arms must not share state).
/// `widths` lets the chunk-cap test reuse this with its single wide row.
fn run_arm_at(
    arm: Arm,
    mutation: FixtureMutation,
    placement: Placement,
    widths: &[usize],
    with_prime: bool,
) -> Run {
    force_true_f32();
    // SAFETY: every caller holds `gpu_guard()`, which is the only thing in this binary touching
    // these vars, and no other thread runs engine code.
    unsafe {
        match placement {
            Placement::Slab => {
                std::env::set_var("MEMRA_MOE_CACHE", "0");
                std::env::set_var("MEMRA_MOE_RESIDENT", "1");
                std::env::set_var("MEMRA_MOE_SLOTS", "0");
            }
            Placement::Slru => {
                std::env::set_var("MEMRA_MOE_CACHE", "1");
                std::env::set_var("MEMRA_MOE_RESIDENT", "0");
                std::env::set_var("MEMRA_MOE_SLOTS", "12");
            }
        }
        // The fused epilogue is a DIFFERENT flag with its own gate; pin it off so this gate
        // measures exactly one seam.
        std::env::set_var("MEMRA_MOE_FUSED_EPI", "0");
        match arm {
            Arm::Sequential => std::env::set_var("MEMRA_MOE_GROUPED_PREFILL", "0"),
            Arm::Grouped => std::env::set_var("MEMRA_MOE_GROUPED_PREFILL", "1"),
        }
    }

    let config = mini_config();
    let plan = mini_plan(&config);
    let engine_weights = engine_fixture(&plan);
    let source = fixture_source(&config, &plan, &engine_weights, mutation);
    let engine = Engine::new(0).expect("CUDA engine on device 0");
    let model = HybridModel::load_from_source_without_mtp(&engine, &source)
        .expect("mini glm5_next model loads from the contract");

    println!(
        "arm[{} / {placement:?}]: MEMRA_MOE_GROUPED_PREFILL={} fixture={mutation:?}",
        arm.label(),
        match arm {
            Arm::Sequential => "0",
            Arm::Grouped => "1",
        }
    );

    let start = memra_engine::moe_grouped_prefill_dispatches();
    let mut rows: Vec<Row> = Vec::new();
    for &n in widths {
        let before = memra_engine::moe_grouped_prefill_dispatches();
        let ids = tokens(n, 0x_67_F0_11 ^ n as u64);
        let got = model.forward(&engine, &ids).expect("GPU routed prefill");
        let (class, _) = expected_class(n);
        rows.push(Row {
            name: format!("prefill T={n}"),
            logits: got,
            dispatches: memra_engine::moe_grouped_prefill_dispatches() - before,
            class,
        });
    }

    if with_prime {
        // The REAL serving path: prime_cache (chunked prefill) then per-token decode. Prompt 20
        // sits past the knee, so the grouped arm must engage exactly once (one chunk, one MoE
        // layer); every decode step must stay on the per-token class.
        let prompt = 20usize;
        let steps = 4usize;
        #[allow(clippy::unusual_byte_groupings)]
        // allow: mnemonic grouping of a pinned seed/magic constant
        let ids = tokens(prompt + steps, 0x_67_F0_11_DEC0);
        let mut cache = memra_engine::cache::Cache::new_planned(&engine, &model.cfg, &plan, 64)
            .expect("cache for the mini glm5_next model");
        let before = memra_engine::moe_grouped_prefill_dispatches();
        let (primed, _seed, _hiddens) = model
            .prime_cache(&engine, &ids[..prompt], &mut cache, 0)
            .expect("GPU routed prime");
        rows.push(Row {
            name: format!("prime T={prompt} last row"),
            logits: primed,
            dispatches: memra_engine::moe_grouped_prefill_dispatches() - before,
            class: "prime chunk (grouped at T>16)",
        });
        for step in 0..steps {
            let before = memra_engine::moe_grouped_prefill_dispatches();
            let got = model
                .decode_step(&engine, ids[prompt + step], &mut cache)
                .expect("GPU routed decode step");
            rows.push(Row {
                name: format!("decode step {step}"),
                logits: got,
                dispatches: memra_engine::moe_grouped_prefill_dispatches() - before,
                class: "decode t=1 (never grouped)",
            });
        }
    }

    Run {
        dispatches: memra_engine::moe_grouped_prefill_dispatches() - start,
        rows,
    }
}

/// Reference logits for `run_arm_at`'s workload, row for row.
fn reference_rows(
    plan: &ModelPlan,
    weights: &BTreeMap<TensorId, ReferenceTensor>,
    widths: &[usize],
    with_prime: bool,
) -> Vec<Vec<f32>> {
    let vocab = VOCAB as usize;
    let mut rows = Vec::new();
    for &n in widths {
        let ids = tokens(n, 0x_67_F0_11 ^ n as u64);
        rows.push(
            memra_reference::execute(plan, weights, &ids)
                .expect("reference execute")
                .logits,
        );
    }
    if with_prime {
        let prompt = 20usize;
        let steps = 4usize;
        #[allow(clippy::unusual_byte_groupings)]
        // allow: mnemonic grouping of a pinned seed/magic constant
        let ids = tokens(prompt + steps, 0x_67_F0_11_DEC0);
        let full = memra_reference::execute(plan, weights, &ids)
            .expect("reference execute")
            .logits;
        rows.push(full[(prompt - 1) * vocab..prompt * vocab].to_vec());
        for step in 0..steps {
            let row = prompt + step;
            rows.push(full[row * vocab..(row + 1) * vocab].to_vec());
        }
    }
    rows
}

/// Assert the per-width engagement map: exactly `expected_class(t).1` grouped dispatches per
/// prefill row when the arm is ON (slab placement), exactly 1 for the prime chunk, 0 for every
/// decode step, and 0 everywhere for the sequential arm or the SLRU placement.
fn check_engagement(label: &str, run: &Run, arm: Arm, placement: Placement) {
    for row in &run.rows {
        let want = match (arm, placement) {
            (Arm::Sequential, _) | (_, Placement::Slru) => 0,
            (Arm::Grouped, Placement::Slab) => {
                if let Some(t) = row
                    .name
                    .strip_prefix("prefill T=")
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    expected_class(t).1
                } else if row.name.starts_with("prime T=") {
                    1
                } else {
                    0 // decode steps
                }
            }
        };
        println!(
            "{label} {}: grouped dispatches {} (want {want}) class `{}`",
            row.name, row.dispatches, row.class
        );
        assert_eq!(
            row.dispatches, want,
            "{label} {}: the engagement map says {want} grouped dispatches for this row, \
             measured {}; a dispatch predicate moved (class `{}`)",
            row.name, row.dispatches, row.class
        );
    }
}

/// GATE A: the grouped arm on the serving placement (resident slabs) against the reference
/// oracle, across every width in the seam map plus the real prime/decode path, with the
/// engagement map asserted row by row.
#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn grouped_prefill_matches_the_reference() {
    let _gpu = gpu_guard();
    let config = mini_config();
    let plan = mini_plan(&config);
    let want = reference_rows(
        &plan,
        &reference_fixture(&plan, &engine_fixture(&plan)),
        &WIDTHS,
        true,
    );

    let run = run_arm_at(
        Arm::Grouped,
        FixtureMutation::None,
        Placement::Slab,
        &WIDTHS,
        true,
    );
    check_all("grouped", &run.rows, &want);
    check_engagement("grouped", &run, Arm::Grouped, Placement::Slab);
    assert!(
        run.dispatches > 0,
        "MEMRA_MOE_GROUPED_PREFILL=1 produced {} grouped dispatches; the arm never fired and \
         this gate measured the sequential loop",
        run.dispatches
    );
    println!("grouped-prefill dispatches: {}", run.dispatches);
}

/// GATE B: the sequential control on the same placement. Proves the fixture and tolerance are
/// honest independently of the grouped arm, and that the rollback seam rolls back (zero
/// dispatches with the flag off).
#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn sequential_control_matches_the_reference() {
    let _gpu = gpu_guard();
    let config = mini_config();
    let plan = mini_plan(&config);
    let want = reference_rows(
        &plan,
        &reference_fixture(&plan, &engine_fixture(&plan)),
        &WIDTHS,
        true,
    );

    let run = run_arm_at(
        Arm::Sequential,
        FixtureMutation::None,
        Placement::Slab,
        &WIDTHS,
        true,
    );
    check_all("sequential", &run.rows, &want);
    check_engagement("sequential", &run, Arm::Sequential, Placement::Slab);
}

/// GATE C: the two arms against each other, and the CLASS assertion made bitwise where it can
/// be: at t <= 16 the flag-ON run refuses the arm and runs the SAME sequential program, so
/// those rows must be BIT-identical between arms; at t > 16 the grouped rows are a different
/// numeric class (f16-mirror grouped GEMM) and carry the band bar, with the exact-bit
/// disagreement count reported rather than asserted (the grouped GEMM is measured
/// non-bit-stable, so a bit-identity claim would be dishonest in either direction).
#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn the_two_arms_agree_and_the_knee_is_bitwise() {
    let _gpu = gpu_guard();
    let sequential = run_arm_at(
        Arm::Sequential,
        FixtureMutation::None,
        Placement::Slab,
        &WIDTHS,
        true,
    );
    let grouped = run_arm_at(
        Arm::Grouped,
        FixtureMutation::None,
        Placement::Slab,
        &WIDTHS,
        true,
    );
    assert!(grouped.dispatches > 0, "the grouped arm never fired");
    assert_eq!(
        sequential.dispatches, 0,
        "the sequential arm took the grouped path"
    );
    assert_eq!(sequential.rows.len(), grouped.rows.len());
    for (a, b) in sequential.rows.iter().zip(&grouped.rows) {
        let bits = a
            .logits
            .iter()
            .zip(&b.logits)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let rel = relative(&b.logits, &a.logits);
        println!(
            "arms {}: relative maxdiff {rel:.3e} (tol {TOL:.1e}), {bits}/{} elements differ \
             in bits, grouped dispatches {}",
            a.name,
            a.logits.len(),
            b.dispatches
        );
        if b.dispatches == 0 {
            assert_eq!(
                bits, 0,
                "arms {}: this row rides the per-token class in BOTH arms (0 grouped \
                 dispatches) and must be bit-identical; {bits} bits differ; the flag is \
                 leaking into a program it does not dispatch",
                a.name
            );
        } else {
            assert!(
                rel <= TOL,
                "arms {}: grouped vs sequential relative maxdiff {rel:.3e} (tol {TOL:.1e})",
                a.name
            );
        }
    }
}

/// GATE D: the 4096-token chunk cap (`PRIME_CHUNK_MAX_TOKENS`), the width the product actually
/// primes at. Its own test because the CPU reference at t=4096 dominates the suite's wall
/// clock. Reference band + engagement (exactly one grouped dispatch for the one MoE layer) +
/// the sequential twin for attribution.
#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn grouped_prefill_matches_the_reference_at_the_chunk_cap() {
    let _gpu = gpu_guard();
    let config = mini_config();
    let plan = mini_plan(&config);
    let widths = [CHUNK_CAP];
    let want = reference_rows(
        &plan,
        &reference_fixture(&plan, &engine_fixture(&plan)),
        &widths,
        false,
    );

    let grouped = run_arm_at(
        Arm::Grouped,
        FixtureMutation::None,
        Placement::Slab,
        &widths,
        false,
    );
    check_all("grouped/4096", &grouped.rows, &want);
    check_engagement("grouped/4096", &grouped, Arm::Grouped, Placement::Slab);

    let sequential = run_arm_at(
        Arm::Sequential,
        FixtureMutation::None,
        Placement::Slab,
        &widths,
        false,
    );
    check_all("sequential/4096", &sequential.rows, &want);
    assert_eq!(sequential.dispatches, 0);
}

/// GATE E: FAIL-CLOSED on the SLRU placement. The arm is slab-only by design; with no resident
/// slab it must refuse (zero dispatches) and the sequential fallback must still be green. This
/// is the rollback seam and the placement predicate, measured rather than assumed.
#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn slru_placement_fails_closed() {
    let _gpu = gpu_guard();
    let config = mini_config();
    let plan = mini_plan(&config);
    let want = reference_rows(
        &plan,
        &reference_fixture(&plan, &engine_fixture(&plan)),
        &WIDTHS,
        false,
    );

    let run = run_arm_at(
        Arm::Grouped,
        FixtureMutation::None,
        Placement::Slru,
        &WIDTHS,
        false,
    );
    check_all("grouped/slru-fail-closed", &run.rows, &want);
    assert_eq!(
        run.dispatches, 0,
        "the grouped arm dispatched {} times with NO resident slab; the placement predicate \
         is not the one the FLAGS row documents, and the arm is reading pointers from \
         somewhere untested",
        run.dispatches
    );
}

/// ROUTING EXACTNESS, measured rather than only argued from construction. Both arms run the same
/// workload with MEMRA_MOE_TRACE + MEMRA_MOE_WEIGHT_TRACE; the trace files must be
/// BYTE-IDENTICAL. Expert ids are exact integers, so their identity is bit-level; weights ride
/// the trace's `{:.9}` formatting (distinct f32s in the sigmoid-weight range differ well above
/// 1e-9; the by-construction argument (one `moe_route_sigmoid_cfg` invocation shared by both
/// arms) closes the remainder). A routing difference larger than formatting also cannot hide:
/// `softmax-for-sigmoid` binds at >= 3e-2 on this fixture's logits.
#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn routing_is_identical_between_arms() {
    let _gpu = gpu_guard();
    let dir = std::env::temp_dir().join(format!("memra-gpf-routes-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("trace dir");
    let capture = |arm: Arm, tag: &str| -> (Vec<u8>, Vec<u8>) {
        let routes = dir.join(format!("routes-{tag}.txt"));
        let weights = dir.join(format!("weights-{tag}.txt"));
        // SAFETY: gpu_guard is held; nothing else in this binary touches these vars.
        unsafe {
            std::env::set_var("MEMRA_MOE_TRACE", &routes);
            std::env::set_var("MEMRA_MOE_WEIGHT_TRACE", &weights);
        }
        let run = run_arm_at(arm, FixtureMutation::None, Placement::Slab, &WIDTHS, false);
        // SAFETY: as above.
        unsafe {
            std::env::remove_var("MEMRA_MOE_TRACE");
            std::env::remove_var("MEMRA_MOE_WEIGHT_TRACE");
        }
        if arm == Arm::Grouped {
            assert!(
                run.dispatches > 0,
                "the grouped arm never fired under tracing; this comparison would be \
                 sequential-vs-sequential and prove nothing about the grouped arm's routing"
            );
        }
        (
            std::fs::read(&routes).expect("routes trace written"),
            std::fs::read(&weights).expect("weights trace written"),
        )
    };
    let (seq_r, seq_w) = capture(Arm::Sequential, "seq");
    let (grp_r, grp_w) = capture(Arm::Grouped, "grp");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!seq_r.is_empty() && !seq_w.is_empty(), "empty trace files");
    assert_eq!(
        seq_r, grp_r,
        "selected experts differ between the sequential and grouped arms; routing exactness \
         is broken, not merely the accumulation order"
    );
    assert_eq!(
        seq_w, grp_w,
        "routing weights differ between the sequential and grouped arms at trace precision"
    );
    println!(
        "routing traces byte-identical: {} route bytes, {} weight bytes",
        seq_r.len(),
        seq_w.len()
    );
}

/// MUTATION CHECK: each wrong program must fail GATE A by a wide margin on the SAME grouped
/// GPU output the passing assertion uses. `shared-expert-dropped` and `softmax-for-sigmoid`
/// are two of this lane's required red arms.
#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn wrong_programs_fail_the_gate() {
    let _gpu = gpu_guard();
    let config = mini_config();
    let plan = mini_plan(&config);
    let weights = reference_fixture(&plan, &engine_fixture(&plan));

    let run = run_arm_at(
        Arm::Grouped,
        FixtureMutation::None,
        Placement::Slab,
        &WIDTHS,
        false,
    );
    assert!(
        run.dispatches > 0,
        "the grouped arm never fired; these mutations would be measured against the \
         sequential loop"
    );
    check_all(
        "mutation control",
        &run.rows,
        &reference_rows(&plan, &weights, &WIDTHS, false),
    );

    for mutation in [
        Mutation::PostClamp,
        Mutation::PlainSwiglu,
        Mutation::NoSharedExpert,
        Mutation::SoftmaxRouter,
    ] {
        let wrong = reference_rows(&mutation.apply(&plan), &weights, &WIDTHS, false);
        let label = mutation.label();
        // Only rows the GROUPED arm produced can indict the grouped arm; per-token rows would
        // dilute `closest` with distances the epilogue gate already owns.
        let grouped_rows: Vec<(&Row, &Vec<f32>)> = run
            .rows
            .iter()
            .zip(&wrong)
            .filter(|(row, _)| row.dispatches > 0)
            .collect();
        assert!(!grouped_rows.is_empty(), "no grouped rows to measure");
        let closest = grouped_rows
            .iter()
            .map(|(row, expect)| relative(&row.logits, expect))
            .fold(f32::INFINITY, f32::min);
        println!(
            "mutation[{label}]: closest grouped row {closest:.3e} (tol {TOL:.1e}, floor \
             {MUTATION_FLOOR:.1e})"
        );
        assert!(
            closest >= MUTATION_FLOOR,
            "[{label}] is only {closest:.3e} from the grouped arm's output on its closest \
             grouped row; this gate does not bind"
        );
    }
}

/// MUTATION CHECK (macro plane): the required `macro-plane-flattened` red arm. The gate bank's
/// whole `weight_scale_2` plane is flattened in the bytes the ENGINE loads; the reference still
/// carries it, so the grouped rows must break. This is what proves the grouped arm's
/// `scale_rows` gate/up fold is load-bearing (the down fold is covered jointly: it rides the
/// same macro plane through the scatter weight).
#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn flattened_macro_plane_fails_the_gate() {
    let _gpu = gpu_guard();
    let config = mini_config();
    let plan = mini_plan(&config);
    let want = reference_rows(
        &plan,
        &reference_fixture(&plan, &engine_fixture(&plan)),
        &WIDTHS,
        false,
    );

    let run = run_arm_at(
        Arm::Grouped,
        FixtureMutation::MacroDropped,
        Placement::Slab,
        &WIDTHS,
        false,
    );
    assert!(
        run.dispatches > 0,
        "the grouped arm never fired; the macro mutation would be measured against the \
         sequential loop"
    );
    let worst = run
        .rows
        .iter()
        .zip(&want)
        .filter(|(row, _)| row.dispatches > 0)
        .map(|(row, expect)| relative(&row.logits, expect))
        .fold(0.0f32, f32::max);
    println!(
        "mutation[macro-plane-flattened]: worst grouped row {worst:.3e} (tol {TOL:.1e}, floor \
         {MUTATION_FLOOR:.1e})"
    );
    assert!(
        worst >= MUTATION_FLOOR,
        "flattening the gate macro plane moved the grouped output by only {worst:.3e}; the \
         macro fold is not under gate"
    );
}
