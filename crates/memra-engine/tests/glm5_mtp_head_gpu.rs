//! GPU-vs-reference gate for the glm5_next **NextN/MTP draft head** — the license-clean native
//! speculative draft (MLA + own k-pool indexer + MoE, serial residual, trunk-lm_head projection).
//!
//! WHY THIS GATE EXISTS. The GLM-5.3-Flash artifact carries the full MTP layer (census: the
//! appended layer owns enorm/hnorm/eh_proj/shared_head.norm, a 12th indexer tensor set, and a
//! full MoE), the plan compiles it (`mtp_blocks[0]`, serial residual), and the contract binds it
//! — but until this lane NOTHING executed it: the four `nextn.*` glue names had no glm5_next
//! ggml->HF mapping row (the embedded-MTP loader silently loaded no head), the reference's
//! `execute_mtp` refused every hc plan (it was handed the raw `[tokens, streams*hidden]` stream
//! stack), and the engine's `mtp_head_forward_dev` had `Mixer::Mla => unimplemented`. This file
//! gates the repaired chain end to end at fixture scale.
//!
//! TRUTH is `memra_reference::execute`'s MTP output: `fused[i] = eh_proj([enorm(embed(ids[i]));
//! hnorm(collapsed_trunk_hidden[i])])` through the block, its private output norm, and the trunk
//! lm_head. The engine walk `mtp_head_forward_mla_cached(depth 0, ids[i], hiddens[i], pos i)`
//! must reproduce row `i` — including the k-pool indexer's sparse regime, which this fixture is
//! sized to reach on the draft plane (48 draft rows over a 16-raw-token budget).
//!
//! GATES:
//!   1. `the_plan_declares_the_mtp_block_and_the_reference_executes_it` (host) — plan wiring +
//!      the reference collapse fix (an hc plan's `execute` must return MTP logits, not the
//!      "HyperConnections MTP fusion" refusal).
//!   2. `gpu_mtp_draft_logits_match_the_reference` — teacher-forced walk over every position,
//!      engine draft logits vs the reference's, worst row printed.
//!   3. `gpu_mtp_eh_proj_transpose_mutation_fails_the_gate` — RED: the fusion projection served
//!      transposed (same bytes, permuted) must blow past the tolerance, or the gate is a
//!      decoration.
//!   4. `gpu_mtp_h_seed_off_by_one_row_fails_the_gate` — RED: seeding step i with the trunk
//!      hidden of row i-1 must fail, or the gate cannot see the h_seed contract at all.
//!   5. `gpu_mtp_head_load_is_flag_gated_default_off` — MEMRA_GLM5_MTP unset loads NO head
//!      (prod byte-identical by default); `=1` loads the MLA+MoE head.
//!   6. `gpu_mtp_position_discipline_refuses_gaps` — a draft position that skips the plane
//!      length is an Err naming the contract, never a silent wrong-horizon attend.
//!
//! Rig law (exactness only, never timing):
//!   NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
//!     cargo test -p memra-engine --test glm5_mtp_head_gpu -- --ignored

use cudarc::driver::CudaSlice;
use memra_engine::Engine;
use memra_engine::hybrid::{Ffn, HybridModel, Mixer};
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::{
    AttentionPlan, MlaAttentionPlan, MlpPlan, ModelPlan, ResidualTopology, SparseIndexPlan,
};
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, LayerTensor, MtpTensor, OutputHead, TensorContract,
    TensorId, TensorMatch,
};
use memra_reference::{ReferenceTensor, ReferenceWeights, deterministic_fixture};
use std::borrow::Cow;
use std::collections::BTreeMap;

const HIDDEN: usize = 128;
const VOCAB: u32 = 32;
/// Prompt length: comfortably past `PRIME_MIN_T` (16) and past the k-pool raw budget
/// (`index_topk` 16 with kpool 4), so the DRAFT plane's indexer runs in the sparse regime for
/// the late rows rather than degenerating into full selection.
const CTX: usize = 48;

/// End-to-end tolerance against the reference. Unlike the all-f32 trunk k-pool gate (1e-5),
/// this chain crosses the NVFP4 expert matmul: the WEIGHT bytes are identical on both sides
/// (the reference reads the roundtrip of the exact bytes the engine decodes), so the residual
/// is the expert kernel's own arithmetic (quantized-activation dp4a vs the reference's plain
/// f32 accumulation).
///
/// MEASURED on this fixture (5090, TF32 off, NVIDIA_TF32_OVERRIDE=0, 2026-08-30), worst
/// per-row relative maxdiff over all 48 draft rows:
///
/// | arm | worst row |
/// |---|---|
/// | green (gate 2) | 1.144e-3 |
/// | eh_proj TRANSPOSED (gate 3) | 4.165e0 |
/// | h_seed off by one row (gate 4) | 6.723e-1 |
///
/// 5e-3 sits ~4x above the measured green and two orders BELOW the weakest mutation. That
/// separation is the gate.
const TOL: f32 = 5e-3;
/// What a REAL mutation must exceed: two orders above TOL, one-seventh of the WEAKEST
/// measured mutation, so a red arm can neither pass on accumulated noise nor fail on it.
const RED: f32 = 1e-1;

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

/// The load flag under test. Set/unset ONLY inside `gpu_guard` — every test in this binary that
/// loads a model holds the guard, so the process-global env cannot race.
fn set_mtp_flag(on: bool) {
    // SAFETY: serialized behind gpu_guard by every caller.
    unsafe {
        if on {
            std::env::set_var("MEMRA_GLM5_MTP", "1");
        } else {
            std::env::remove_var("MEMRA_GLM5_MTP");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The mini model: the k-pool gate's config with ONE MoE trunk layer and ONE NextN block, parsed
// through the real config path so the plan under test is the pack's, not an imitation.
// ---------------------------------------------------------------------------------------------

fn mini_config_json() -> String {
    format!(
        r#"{{
      "model_type": "glm5_next_text",
      "num_hidden_layers": 2,
      "num_nextn_predict_layers": 1,
      "hidden_size": {HIDDEN},
      "intermediate_size": 64,
      "vocab_size": {VOCAB},
      "max_position_embeddings": 512,
      "rms_norm_eps": 1e-05,
      "hidden_act": "silu",
      "swiglu_limit": 10.0,
      "tie_word_embeddings": true,
      "hc_mult": 4,
      "hc_eps": 1e-06,
      "hc_sinkhorn_iters": 20,
      "mhc": true,
      "layer_types": ["deepseek_sparse_attention", "deepseek_sparse_attention"],
      "mlp_layer_types": ["dense", "dense"],
      "first_k_dense_replace": 2,
      "indexer_types": ["full", "full"],
      "linear_attn_config": {{
        "num_heads": 1,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [],
        "full_attn_layers": [0, 1]
      }},
      "num_attention_heads": 2,
      "num_key_value_heads": 2,
      "q_lora_rank": 16,
      "kv_lora_rank": 16,
      "qk_head_dim": 16,
      "qk_nope_head_dim": 16,
      "qk_rope_head_dim": 0,
      "v_head_dim": 16,
      "mla_use_nope": true,
      "index_n_heads": 2,
      "index_head_dim": 8,
      "index_topk": 16,
      "index_kpool": 4,
      "index_kpool_always_select_tail": true,
      "index_kpool_compress": true,
      "indexer_rope_interleave": true,
      "index_share_for_mtp_iteration": true,
      "n_routed_experts": 4,
      "num_experts_per_tok": 2,
      "moe_intermediate_size": 64,
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

fn mini_config() -> ModelConfig {
    ModelConfig::from_hf(&HfConfig::parse(&mini_config_json()))
}

fn mini_plan(config: &ModelConfig) -> ModelPlan {
    memra_gguf::model_packs::for_config(config)
        .expect("glm5_next model pack matches the mini config")
        .compile_plan(config)
        .expect("mini glm5_next plan compiles")
}

/// Deterministic non-trivial values so an identity-weight norm cannot mask a wrong operand.
fn varied(len: usize, seed: u64, spread: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed)
                .rotate_left(17) as f64
                / u64::MAX as f64;
            1.0 + spread * (x as f32 - 0.5)
        })
        .collect()
}

/// The fixture weights, strengthened where the generic fixture is weak for THIS gate: the
/// generic MTP glue norms are all-ones (an RMSNorm with weight 1 cannot catch a swapped norm
/// operand), and no private output norm exists (the real artifact ships `shared_head.norm`,
/// so the prefer-private-then-model arm must be the one under test on BOTH sides).
fn fixture_weights(plan: &ModelPlan) -> ReferenceWeights {
    let mut weights = deterministic_fixture(plan).expect("fixture").weights;
    weights.insert(
        TensorId::Mtp {
            depth: 0,
            tensor: MtpTensor::EmbeddingNorm,
        },
        ReferenceTensor::new(vec![HIDDEN], varied(HIDDEN, 0xE0_12, 0.8)).unwrap(),
    );
    weights.insert(
        TensorId::Mtp {
            depth: 0,
            tensor: MtpTensor::HiddenNorm,
        },
        ReferenceTensor::new(vec![HIDDEN], varied(HIDDEN, 0x40_77, 0.8)).unwrap(),
    );
    weights.insert(
        TensorId::Mtp {
            depth: 0,
            tensor: MtpTensor::OutputNorm,
        },
        ReferenceTensor::new(vec![HIDDEN], varied(HIDDEN, 0x5EAD, 0.8)).unwrap(),
    );
    weights
}

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
    ggml_type: GgmlType,
}

/// Serves the reference fixture's own f32 numbers under the contract's ggml names, so the
/// reference and the GPU read ONE set of weights (`glm5_kpool_indexer_gpu`'s pattern).
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

/// `transpose_eh_proj` is the red-arm instrument: serve the fusion projection's bytes
/// TRANSPOSED under the same declared shape (a pure data permutation — a shape change would
/// refuse at load and prove nothing about the gate).
///
/// The MTP block's routed-expert banks are served **NVFP4** — the quant class the real
/// artifact ships for exactly these tensors, and the only class the engine's expert loader
/// accepts here (F32 banks refuse at load). To keep ONE set of numbers on both sides, the
/// banks in `weights` are REPLACED with their NVFP4 roundtrip (encode -> `dequant_gguf_row`),
/// so the reference computes on the values the engine's bytes decode to; the remaining
/// engine-vs-reference difference is the expert MATMUL's own arithmetic, which is what the
/// measured tolerance covers.
fn fixture_source(
    config: &ModelConfig,
    plan: &ModelPlan,
    weights: &mut ReferenceWeights,
    transpose_eh_proj: bool,
) -> FixtureSource {
    let contract = TensorContract::for_plan(
        plan,
        CheckpointDialect::Gguf,
        ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        },
    )
    .expect("contract for the mini glm5_next plan");
    let eh_proj_id = TensorId::Mtp {
        depth: 0,
        tensor: MtpTensor::FusionProjection,
    };
    let is_expert_bank = |id: &TensorId| {
        matches!(
            id,
            TensorId::Layer {
                tensor: LayerTensor::MoeExpertGateBank
                    | LayerTensor::MoeExpertUpBank
                    | LayerTensor::MoeExpertDownBank,
                ..
            }
        )
    };
    let mut tensors = BTreeMap::new();
    for req in contract.requirements.iter() {
        if !req.required && !weights.contains_key(&req.id) {
            continue;
        }
        let tensor = weights
            .get(&req.id)
            .unwrap_or_else(|| panic!("reference fixture is missing {:?}", req.id));
        let logical_shape = tensor.shape.clone();
        let elements: usize = req.shape.iter().map(|&d| d as usize).product();
        assert_eq!(
            elements,
            tensor.data.len(),
            "fixture {:?} has {} elements, contract requires {elements}",
            req.id,
            tensor.data.len()
        );
        let data: Vec<f32> = if transpose_eh_proj && req.id == eh_proj_id {
            // [rows, cols] row-major -> the transpose's row-major bytes under the SAME shape.
            let (rows, cols) = (req.shape[0] as usize, req.shape[1] as usize);
            let mut out = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    out[c * rows + r] = tensor.data[r * cols + c];
                }
            }
            out
        } else {
            tensor.data.clone()
        };
        let (bytes, ggml_type) = if is_expert_bank(&req.id) {
            let enc = memra_gguf::nvfp4_repack::f32_to_nvfp4(&data);
            // Roundtrip the reference to the values these bytes decode to. Every row width in
            // this fixture is a multiple of 64, so 64-element sub-blocks never straddle rows
            // and decoding block-by-block equals decoding row-by-row.
            let mut roundtrip = Vec::with_capacity(data.len());
            for block in enc.chunks_exact(36) {
                roundtrip.extend(memra_gguf::nvfp4_repack::dequant_gguf_row(block, 64));
            }
            assert_eq!(roundtrip.len(), data.len());
            weights.insert(
                req.id.clone(),
                ReferenceTensor::new(logical_shape, roundtrip).unwrap(),
            );
            (enc, GgmlType::NVFP4)
        } else {
            (
                data.iter().flat_map(|v| v.to_le_bytes()).collect(),
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

fn tokens(n: usize, seed: u64) -> Vec<u32> {
    (0..n)
        .map(|i| {
            ((i as u64)
                .wrapping_mul(2862933555777941757)
                .wrapping_add(seed)
                .rotate_left(23)
                % VOCAB as u64) as u32
        })
        .collect()
}

fn relative(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
    got.iter()
        .zip(want)
        .fold(0.0f32, |a, (g, w)| a.max((g - w).abs()))
        / scale
}

// ---------------------------------------------------------------------------------------------
// Gate 1 (host): plan wiring + the reference collapse fix.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_plan_declares_the_mtp_block_and_the_reference_executes_it() {
    let config = mini_config();
    let plan = mini_plan(&config);
    assert_eq!(plan.mtp_blocks.len(), 1);
    let block = &plan.mtp_blocks[0];
    assert_eq!(block.layer.index, 2, "the MTP block is the appended layer");
    assert!(
        matches!(
            block.layer.attention,
            AttentionPlan::Mla(MlaAttentionPlan::LatentKv {
                sparse_index: SparseIndexPlan::Own { kpool: Some(_), .. },
                ..
            })
        ),
        "the glm5_next MTP block is MLA with its OWN k-pool indexer"
    );
    assert!(matches!(block.layer.mlp, MlpPlan::Moe(_)));
    assert!(
        matches!(block.layer.residual, ResidualTopology::Serial),
        "the NextN layer carries no hc_* tensors — serial residual outside the stream stack"
    );
    assert!(
        matches!(
            plan.layers[0].residual,
            ResidualTopology::HyperConnections { .. }
        ),
        "the TRUNK must be hc, or this gate is not covering the collapse seam at all"
    );

    // The reference collapse fix: execute() on an hc plan with an MTP block must produce MTP
    // logits (pre-fix it returned the "HyperConnections MTP fusion" refusal for every hc plan).
    let weights = fixture_weights(&plan);
    let ids = tokens(CTX, 0x6_17E5_7A11);
    let output = memra_reference::execute(&plan, &weights, &ids)
        .expect("reference execute on the hc plan with an MTP block");
    assert_eq!(output.mtp.len(), 1, "one MTP depth executed");
    let mtp = &output.mtp[0];
    assert_eq!(mtp.logits.len(), CTX * VOCAB as usize);
    assert!(
        mtp.logits.iter().all(|v| v.is_finite()),
        "MTP logits must be finite"
    );
    // Non-vacuity: the draft head is a different function from the trunk head.
    let last = (CTX - 1) * VOCAB as usize;
    let rel = relative(&mtp.logits[last..], &output.logits[last..]);
    assert!(
        rel > RED,
        "draft and trunk logits agree to {rel:.3e} on the last row — the fixture degenerated \
         and this gate would pass a head that just re-projects the trunk hidden"
    );
}

// ---------------------------------------------------------------------------------------------
// GPU harness
// ---------------------------------------------------------------------------------------------

struct Harness {
    engine: Engine,
    model: HybridModel,
    plan: ModelPlan,
    weights: ReferenceWeights,
}

impl Harness {
    /// Loads WITH the MTP head (`MEMRA_GLM5_MTP=1`). Callers hold `gpu_guard`.
    fn new(transpose_eh_proj: bool) -> Self {
        force_true_f32();
        set_mtp_flag(true);
        let config = mini_config();
        let plan = mini_plan(&config);
        let mut weights = fixture_weights(&plan);
        let source = fixture_source(&config, &plan, &mut weights, transpose_eh_proj);
        let engine = Engine::new(0).expect("CUDA engine on device 0");
        let model = HybridModel::load_from_source(&engine, &source)
            .expect("mini glm5_next model loads from the contract");
        Self {
            engine,
            model,
            plan,
            weights,
        }
    }

    /// Trunk prime + per-position collapsed pre-output_norm hiddens (MTP-PLAN §A h_seeds).
    fn primed(&self, ids: &[u32]) -> (memra_engine::cache::Cache, CudaSlice<f32>) {
        let mut cache = memra_engine::cache::Cache::new_planned(
            &self.engine,
            &self.model.cfg,
            &self.plan,
            CTX + 8,
        )
        .expect("cache for the mini glm5_next model");
        let mtp_il = self.plan.mtp_blocks[0].layer.index as usize;
        let plane = cache.latent[mtp_il]
            .as_ref()
            .expect("the cache allocates the MTP block's latent plane");
        assert!(
            plane.index_rows.is_some(),
            "the MTP plane must carry the indexer's packed state"
        );
        let (_logits, _seed, hiddens) = self
            .model
            .prime_cache(&self.engine, ids, &mut cache, 0)
            .expect("hc prime");
        assert_eq!(
            cache.latent[mtp_il].as_ref().unwrap().len,
            0,
            "trunk prime must not touch the MTP block's plane"
        );
        (cache, hiddens)
    }

    fn seed_row(&self, hiddens: &CudaSlice<f32>, row: usize) -> CudaSlice<f32> {
        let e = &self.engine;
        let stack = e.view(hiddens, CTX * HIDDEN);
        let view = stack.slice(row * HIDDEN..(row + 1) * HIDDEN);
        let mut seed = e.uninit(HIDDEN).expect("seed row");
        e.copy_view_into(&mut seed, 0, &view, HIDDEN)
            .expect("seed copy");
        seed
    }

    fn reference_mtp_logits(&self, ids: &[u32]) -> Vec<f32> {
        let output =
            memra_reference::execute(&self.plan, &self.weights, ids).expect("reference execute");
        assert_eq!(output.mtp.len(), 1);
        output.mtp.into_iter().next().unwrap().logits
    }

    /// The teacher-forced walk: step i drafts from (ids[i], trunk_hidden[row_for(i)]) at plane
    /// position i. `row_for` is identity on the green arm; the red arm shifts it.
    fn walk(&self, ids: &[u32], row_for: impl Fn(usize) -> usize) -> Vec<f32> {
        let (mut cache, hiddens) = self.primed(ids);
        let vocab = VOCAB as usize;
        let mut out = vec![0.0f32; CTX * vocab];
        for i in 0..CTX {
            let seed = self.seed_row(&hiddens, row_for(i));
            let (logits, _carrier) = self
                .model
                .mtp_head_forward_mla_cached(&self.engine, 0, ids[i], &seed, &mut cache, i)
                .expect("MTP draft step");
            let host = self.engine.dtoh(&logits).expect("logits readback");
            out[i * vocab..(i + 1) * vocab].copy_from_slice(&host);
        }
        out
    }
}

/// Worst per-row relative maxdiff over the walk.
fn worst_row(got: &[f32], want: &[f32]) -> (usize, f32) {
    let vocab = VOCAB as usize;
    let mut worst = (0usize, 0.0f32);
    for i in 0..CTX {
        let rel = relative(
            &got[i * vocab..(i + 1) * vocab],
            &want[i * vocab..(i + 1) * vocab],
        );
        if rel > worst.1 {
            worst = (i, rel);
        }
    }
    worst
}

// ---------------------------------------------------------------------------------------------
// Gate 2 — GREEN: the draft head matches the reference at every position.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_mtp_draft_logits_match_the_reference() {
    let _gpu = gpu_guard();
    let h = Harness::new(false);
    let mtp = h
        .model
        .mtp
        .as_ref()
        .expect("MEMRA_GLM5_MTP=1 loads the head");
    assert!(matches!(mtp.mixer, Mixer::Mla(_)), "glm5 MTP mixer is MLA");
    assert!(matches!(mtp.ffn, Ffn::Moe(_)), "glm5 MTP FFN is MoE");
    assert!(
        mtp.shared_head_norm.is_some(),
        "the private shared_head.norm must load (prefer-private arm under test)"
    );
    assert!(
        mtp.shared_head_head.is_none(),
        "glm5_next ships no private MTP head — the trunk lm_head fallback is the contract"
    );

    let ids = tokens(CTX, 0x6_17E5_7A11);
    let want = h.reference_mtp_logits(&ids);
    let got = h.walk(&ids, |i| i);
    let (row, rel) = worst_row(&got, &want);
    println!("worst of {CTX} draft rows: {rel:.3e} at row {row}");
    assert!(
        rel <= TOL,
        "draft row {row} relative maxdiff {rel:.3e} exceeds {TOL:.1e}"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 3 — RED: transposed eh_proj must fail.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_mtp_eh_proj_transpose_mutation_fails_the_gate() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let ids = tokens(CTX, 0x6_17E5_7A11);
    let want = h.reference_mtp_logits(&ids);
    let got = h.walk(&ids, |i| i);
    let (row, rel) = worst_row(&got, &want);
    println!("eh_proj-transposed worst row: {rel:.3e} at row {row}");
    assert!(
        rel > RED,
        "a TRANSPOSED fusion projection stayed within {rel:.3e} of the reference — the gate \
         cannot see the eh_proj operand and gate 2 is a decoration"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 4 — RED: h_seed off by one row must fail.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_mtp_h_seed_off_by_one_row_fails_the_gate() {
    let _gpu = gpu_guard();
    let h = Harness::new(false);
    let ids = tokens(CTX, 0x6_17E5_7A11);
    let want = h.reference_mtp_logits(&ids);
    let got = h.walk(&ids, |i| i.saturating_sub(1));
    let (row, rel) = worst_row(&got, &want);
    println!("h_seed-off-by-one worst row: {rel:.3e} at row {row}");
    assert!(
        rel > RED,
        "seeding every step with the PREVIOUS row's trunk hidden stayed within {rel:.3e} — \
         the gate cannot see the h_seed pairing contract"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 5 — the flag: default OFF loads no head; ON loads the MLA+MoE head.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_mtp_head_load_is_flag_gated_default_off() {
    let _gpu = gpu_guard();
    force_true_f32();
    let config = mini_config();
    let plan = mini_plan(&config);
    let mut weights = fixture_weights(&plan);
    let source = fixture_source(&config, &plan, &mut weights, false);
    let engine = Engine::new(0).expect("CUDA engine on device 0");

    set_mtp_flag(false);
    let off = HybridModel::load_from_source(&engine, &source)
        .expect("default load must stay green with the head skipped");
    assert!(
        off.mtp.is_none(),
        "MEMRA_GLM5_MTP unset must load NO glm5_next MTP head — prod stays byte-identical \
         by default (a loaded head costs a trunk-layer of VRAM with no consumer yet)"
    );

    set_mtp_flag(true);
    let on = HybridModel::load_from_source(&engine, &source).expect("flagged load");
    let mtp = on.mtp.as_ref().expect("MEMRA_GLM5_MTP=1 loads the head");
    assert!(matches!(mtp.mixer, Mixer::Mla(_)));
    assert!(matches!(mtp.ffn, Ffn::Moe(_)));
}

// ---------------------------------------------------------------------------------------------
// Gate 6 — position discipline: a gap between mtp_pos and the plane length is a named Err.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_mtp_position_discipline_refuses_gaps() {
    let _gpu = gpu_guard();
    let h = Harness::new(false);
    let ids = tokens(CTX, 0x6_17E5_7A11);
    let (mut cache, hiddens) = h.primed(&ids);
    let seed = h.seed_row(&hiddens, 0);
    let err = h
        .model
        .mtp_head_forward_mla_cached(&h.engine, 0, ids[0], &seed, &mut cache, 5)
        .expect_err("a draft position past the plane length must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("latent plane"),
        "the refusal must name the plane-length contract, got: {msg}"
    );
}
