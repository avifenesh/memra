//! CORRECTNESS GATE for the chunked mHC prime (`hyper_prime_ranges`).
//!
//! The capacity gate next door (`glm5_prime_capacity.rs`) says the prime must be SPLIT into
//! calls, or GLM-5.3-Flash's 1,048,576-token context needs 1.7 TB of transients per call. This
//! gate says the split does not change the answer.
//!
//! TWO ARMS, AND THE TRUTH ARM IS THE ONE THAT MATTERS. Comparing the chunked prime only against
//! the monolithic prime is comparing SIBLINGS: two arms sharing one corrupted input both pass
//! (the Q8RP postmortem, LAW:pin-against-truth). So the primary assertion is against
//! `memra_reference::execute` — the portable trunk executor that is the arithmetic contract for
//! this residual topology — and the sibling comparison is kept only to isolate THE SPLIT as the
//! variable.
//!
//! THE BAR IS A CALIBRATED BAND, NOT BYTE IDENTITY, AND THAT IS A MEASURED FINDING.
//! `MEMRA_PRIME_CHUNK` is documented a pure memory knob on the SERIAL trunk, held there by the
//! `chunkinv` gate at byte identity. It cannot be held to that on the mHC trunk, and the reason
//! is not the split:
//!   * the arms diverge at ROW 0, at every chunk size. Row 0 cannot be reached by any
//!     cross-token state, so this is not the KDA conv ring, not the recurrent state, not the
//!     latent plane, and not the indexer's incremental pool keys;
//!   * isolated directly on the rig: `Engine::linear` — cuBLASLt f32, the `mixes` GEMM in
//!     `hyper::pre` — is NOT m-invariant. m=32 against m=200 moves 9601/12288 output bits at
//!     worst 3.815e-6, the SAME worst the chunked prime reports, while m=128 and m=199 against
//!     m=200 are bit-identical. cuBLASLt reselects its algorithm somewhere between, and the
//!     algorithm's reduction order is the whole difference.
//!     `hyper.rs`'s own header already concedes this: that GEMM is "a serving trunk, not a byte-parity
//!     oracle". So the split EXPOSES a near-tie that was always there; it does not create one. This
//!     is the same shape as the GDN off-grid near-tie class the `MEMRA_PRIME_CHUNK` FLAGS row already
//!     documents, and it is written into that row.
//!     Receipts: `research/glm53-flash-bringup-20260827/1m-context-20260828/02-diag-chunk-divergence.txt`.
//!
//! WHAT THE BAND WOULD STILL CATCH. The serial trunk's real chunk-invariance incident
//! (`research/chunk-invariance-20260805/VERDICT.md`) had a per-row maxdiff of exactly 0.0 before
//! the first boundary and O(1) right after — 1.813e0 in the step35 case. That is five orders
//! above this band, and it diverges AT the boundary rather than at row 0, so the two signatures
//! are not confusable. One near-tie consequence is named rather than hidden: the k-pool selection
//! sorts on ReLU'd scores where exact-0.0 ties are ORDINARY (see `cu/mla_attn.cu`), so a last-ulp
//! move can flip which zero-scoring pool enters the budget. The reference arm is the instrument
//! that covers that; the sibling band alone would not.
//!
//! THE FIXTURE CARRIES BOTH MIXERS AND THE INDEXER. One KDA layer and one DSA layer
//! (MLA + k-pool indexer), which is glm5_next's own alternation, because the three things a
//! split could plausibly break live in three different places: the KDA conv ring and recurrent
//! state carried across the boundary; the MLA latent plane written by an earlier call and read
//! by a later one; and the DSA indexer's resident pool keys with their `index_pools_ready`
//! incremental build.
//!
//! ARMS ARE ONE ENV READ APART on the SAME binary and the SAME loaded weights:
//! `MEMRA_PRIME_CHUNK=0` is the monolithic arm (and the shipped rollback seam),
//! `MEMRA_PRIME_CHUNK=<small>` is the chunked candidate. `prime_chunk_tokens` reads the variable
//! per call, which is what lets one process run both.
//!
//! GPU-gated (`#[ignore]`); rig law = exactness only, never timing. Run under
//! `flock /tmp/memra-5090.lock` with `NVIDIA_TF32_OVERRIDE=0` and `-- --ignored`.

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_engine::hybrid_forward::{hyper_prime_call_rows, hyper_prime_ranges};
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::ModelPlan;
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, OutputHead, TensorContract, TensorId, TensorMatch,
};
use memra_reference::{ReferenceTensor, deterministic_fixture};
use std::borrow::Cow;
use std::collections::BTreeMap;

const HIDDEN: usize = 128;
const VOCAB: u32 = 32;
const MAX_CTX: usize = 1024;

/// Layer 0 KDA, layer 1 DSA (MLA + k-pool indexer) — glm5_next's own alternation, at the
/// smallest widths the kernels are instantiated for. `head_dim` 128 is forced:
/// `memra_kda_scan_s128` is instantiated for that width only. The indexer is deliberately TINY
/// (`index_kpool` 4, `index_topk` 8 -> select_k 2) so the top-k budget does not cover every
/// pool: a selection that is not actually a selection would make this gate vacuous.
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

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
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
            ggml_type: GgmlType::F32,
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
        let bytes: Vec<u8> = tensor.data.iter().flat_map(|v| v.to_le_bytes()).collect();
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
                },
            );
        }
    }
    FixtureSource {
        config: config.clone(),
        tensors,
    }
}

/// GPU tests serialize on one device, AND on the process-wide `MEMRA_PRIME_CHUNK` these arms
/// flip. Both reasons need the same lock.
fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call has been made in this process yet, and call_once serializes
            // every test thread behind this write.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
    });
}

/// SAFETY: every caller holds `gpu_guard`, so no other test thread is reading the environment.
fn set_prime_chunk(value: &str) {
    unsafe { std::env::set_var("MEMRA_PRIME_CHUNK", value) };
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
        let fixture = deterministic_fixture(&plan).expect("deterministic glm5_next fixture");
        let source = fixture_source(&config, &plan, &fixture.weights);
        let engine = Engine::new(0).expect("CUDA engine on device 0");
        let model = HybridModel::load_from_source_without_mtp(&engine, &source)
            .expect("mini glm5_next model loads from the contract");
        Self {
            engine,
            model,
            plan,
            weights: fixture.weights,
        }
    }

    fn reference_logits(&self, ids: &[u32]) -> Vec<f32> {
        memra_reference::execute(&self.plan, &self.weights, ids)
            .expect("reference execute")
            .logits
    }

    /// Prime `ids` in ONE `prime_cache` call under the given chunk setting and return the
    /// last-row logits plus the whole pre-output_norm hidden stack, both on the host.
    fn prime(&self, ids: &[u32], chunk: &str) -> (Vec<f32>, Vec<f32>) {
        set_prime_chunk(chunk);
        let mut cache = memra_engine::cache::Cache::new_planned(
            &self.engine,
            &self.model.cfg,
            &self.plan,
            MAX_CTX,
        )
        .expect("cache for the mini glm5_next model");
        let (logits, _seed, hiddens) = self
            .model
            .prime_cache(&self.engine, ids, &mut cache, 0)
            .unwrap_or_else(|e| panic!("prime at MEMRA_PRIME_CHUNK={chunk}: {e}"));
        let stack = self.engine.dtoh(&hiddens).expect("hidden stack to host");
        (logits, stack)
    }
}

/// Scale-relative bound, the SAME constant `hyper_connections_gpu.rs` calibrated next door for
/// this trunk. Both arms below use it, so the split is measured against the same bar the
/// residual topology itself is held to.
///
/// CALIBRATED, not guessed: the worst measured chunked-vs-monolithic divergence on this fixture
/// is 3.815e-6 absolute on unit-scale activations, so 2e-5 carries about 5x headroom while
/// staying five orders BELOW the 1.813e0 signature of the serial trunk's real chunk-invariance
/// defect. It is also ~35x below what losing `NVIDIA_TF32_OVERRIDE=0` costs, so a rig that lost
/// it fails here instead of passing under a widened bar. Calibrate downward, never upward.
const TOL: f32 = 2e-5;

fn relative(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "compared slices differ in length");
    let worst = got
        .iter()
        .zip(want)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale = want.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
    worst / scale
}

fn check(name: &str, got: &[f32], want: &[f32]) {
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{name}: output has non-finite values"
    );
    let rel = relative(got, want);
    assert!(
        rel <= TOL,
        "{name}: relative maxdiff {rel:.3e} (tol {TOL:.1e})"
    );
}

/// The plan really is the two-mixer hyper-connections one. No CUDA — if this fails, every
/// assertion below is measuring something other than what it claims.
#[test]
fn the_mini_plan_carries_both_a_kda_and_a_dsa_indexer_layer() {
    use memra_gguf::model_plan::{
        AttentionPlan, MlaAttentionPlan, ResidualTopology, SparseIndexPlan,
    };
    let config = mini_config();
    let plan = mini_plan(&config);
    assert_eq!(plan.layers.len(), 2);
    assert!(
        matches!(plan.layers[0].attention, AttentionPlan::KimiDeltaNet(_)),
        "layer 0 should be the KDA mixer, got {:?}",
        plan.layers[0].attention
    );
    let sparse = match &plan.layers[1].attention {
        AttentionPlan::Mla(MlaAttentionPlan::LatentKv { sparse_index, .. }) => sparse_index,
        other => panic!("layer 1 should be the MLA/DSA mixer, got {other:?}"),
    };
    assert!(
        !matches!(sparse, SparseIndexPlan::None),
        "layer 1 carries no k-pool indexer: {sparse:?}"
    );
    for layer in &plan.layers {
        assert!(
            matches!(layer.residual, ResidualTopology::HyperConnections { .. }),
            "layer {} is not under the mHC residual",
            layer.index
        );
    }
}

/// The schedule this gate exercises must really be a SPLIT — otherwise both arms run the same
/// code and the byte comparison below is vacuous. No CUDA.
#[test]
fn the_gate_prompt_really_is_split_into_several_prime_calls() {
    // Holds the same lock the GPU arms do: this test WRITES the process-wide MEMRA_PRIME_CHUNK.
    let _gpu = gpu_guard();
    let n_layers = mini_config().n_layer as usize;
    set_prime_chunk("32");
    let ranges = hyper_prime_ranges(200, n_layers, false);
    assert!(
        ranges.len() >= 4,
        "the chunked arm must make several prime calls, got {ranges:?}"
    );
    // 32 + the tail-merge term: `fixed_prime_chunk_ranges` folds a remainder shorter than
    // PRIME_MIN_T into the call before it rather than emitting a call too short to batch.
    assert!(hyper_prime_call_rows(200, n_layers, false) <= 32 + 16);
    set_prime_chunk("0");
    assert_eq!(
        hyper_prime_ranges(200, n_layers, false),
        vec![(0, 200)],
        "MEMRA_PRIME_CHUNK=0 must restore the monolithic walk — it is the rollback seam"
    );
}

/// TRUTH ARM. A prompt primed in MANY calls must reproduce the reference executor's logits.
/// This is the assertion that catches a real state-carry defect, and it does not care what the
/// monolithic prime does — it is anchored OUTSIDE the feature under test.
///
/// Several prompt lengths and several chunk sizes, including lengths that are NOT a multiple of
/// the chunk (the last call of any real prompt is a short one) and chunks that do not divide the
/// indexer's `index_kpool`, so a boundary that split a pool would show up.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn a_chunked_mhc_prime_matches_the_reference_executor() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let vocab = VOCAB as usize;
    for &prompt in &[64usize, 200, 501] {
        let ids = tokens(prompt, 0x1_0000_0C11 ^ prompt as u64);
        let want = h.reference_logits(&ids);
        let last = &want[(prompt - 1) * vocab..prompt * vocab];
        for &chunk in &["0", "16", "32", "37", "128"] {
            let (got, _stack) = h.prime(&ids, chunk);
            check(
                &format!("T={prompt} chunk={chunk} vs reference"),
                &got,
                last,
            );
        }
    }
    set_prime_chunk("0");
}

/// TRUTH ARM, decode continuation. A chunked prime must leave a cache a DECODE can continue
/// from. The prime's own outputs are covered above; this covers what the prime LEFT BEHIND —
/// the KDA conv ring and recurrent state, the MLA latent plane, and the indexer's resident pool
/// keys with their `index_pools_ready` count.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn a_chunked_prime_then_decode_matches_the_reference_executor() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let vocab = VOCAB as usize;
    let prompt = 200usize;
    let steps = 6usize;
    let ids = tokens(prompt + steps, 0xC0FFEE);
    let want = h.reference_logits(&ids);

    for &chunk in &["0", "16", "37"] {
        set_prime_chunk(chunk);
        let mut cache =
            memra_engine::cache::Cache::new_planned(&h.engine, &h.model.cfg, &h.plan, MAX_CTX)
                .expect("cache for the mini glm5_next model");
        h.model
            .prime_cache(&h.engine, &ids[..prompt], &mut cache, 0)
            .unwrap_or_else(|e| panic!("prime at MEMRA_PRIME_CHUNK={chunk}: {e}"));
        for s in 0..steps {
            let row = prompt + s;
            let got = h
                .model
                .decode_step(&h.engine, ids[row], &mut cache)
                .expect("decode step");
            check(
                &format!("chunk={chunk} decode step {s} vs reference"),
                &got,
                &want[row * vocab..(row + 1) * vocab],
            );
        }
    }
    set_prime_chunk("0");
}

/// SIBLING ARM. Chunked against monolithic, so THE SPLIT is the only variable — the reference
/// arm above cannot tell a split defect from a pre-existing one. The bar is the near-tie band,
/// not byte identity, for the reason the file header measures: the mHC `mixes` GEMM is cuBLASLt
/// f32 and its algorithm selection is m-dependent, so the two arms run different reduction
/// orders. Every row of the hidden stack is compared, not just the last, because a defect that
/// only touched an interior chunk would not reach the last-row logits.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn a_chunked_mhc_prime_stays_inside_the_near_tie_band_of_the_monolithic_prime() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    for &prompt in &[64usize, 200, 501] {
        let ids = tokens(prompt, 0x1_0000_0C11 ^ prompt as u64);
        let (want_logits, want_stack) = h.prime(&ids, "0");
        for &chunk in &["16", "32", "37", "128"] {
            let (got_logits, got_stack) = h.prime(&ids, chunk);
            check(
                &format!("T={prompt} chunk={chunk} logits vs monolithic"),
                &got_logits,
                &want_logits,
            );
            check(
                &format!("T={prompt} chunk={chunk} hidden stack vs monolithic"),
                &got_stack,
                &want_stack,
            );
        }
    }
    set_prime_chunk("0");
}

/// NEGATIVE CONTROL — the comparison the gates above rest on has TEETH.
///
/// A parity assertion between two runs of the same program is worth nothing until something is
/// shown to break it: an all-zero buffer, a length-zero slice or a comparison against itself
/// would all pass silently. This primes two prompts differing in ONE token and asserts the
/// comparison sees it, under the CHUNKED arm so the teeth are shown on the path the gates
/// measure.
///
/// THE PERTURBED TOKEN IS THE LAST ONE, and that is not cosmetic. The first version of this
/// control perturbed the MIDDLE of the prompt and FAILED: with KDA's per-channel decay floor of
/// -5.0 and a fixture-scale DSA budget of two pools, a token 100 positions back genuinely has no
/// reach to the last row. That is correct model behaviour, not a defect, and it is exactly the
/// kind of silent-diagnostic trap a control exists to expose — so both halves are asserted: the
/// last-row logits move when the LAST token moves, and the hidden stack moves at the perturbed
/// row itself, which holds by construction at any position (a different token is a different
/// embedding).
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn the_parity_comparison_has_teeth() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let prompt = 200usize;

    // Half 1: the LAST token moves, so the last-row logits must move.
    let a = tokens(prompt, 0x7EE7);
    let mut b = a.clone();
    b[prompt - 1] = (a[prompt - 1] + 1) % VOCAB;
    assert_ne!(a, b, "the two prompts must actually differ");
    let (logits_a, _) = h.prime(&a, "32");
    let (logits_b, _) = h.prime(&b, "32");
    let rel = relative(&logits_b, &logits_a);
    assert!(
        rel > TOL,
        "moving the LAST prompt token moved the logits by only {rel:.3e}, inside the {TOL:.1e} \
         band the gates pass at — the comparison cannot distinguish two different programs"
    );

    // Half 2: a MIDDLE token moves, so the hidden stack must move AT that row. Decay-proof:
    // it holds by construction whatever the mixers do downstream.
    let mut c = a.clone();
    let mid = prompt / 2;
    c[mid] = (a[mid] + 1) % VOCAB;
    let (_, stack_a) = h.prime(&a, "32");
    let (_, stack_c) = h.prime(&c, "32");
    let row_rel = relative(
        &stack_c[mid * HIDDEN..(mid + 1) * HIDDEN],
        &stack_a[mid * HIDDEN..(mid + 1) * HIDDEN],
    );
    assert!(
        row_rel > TOL,
        "changing the token at row {mid} moved that row's hidden state by only {row_rel:.3e} — \
         the hidden-stack comparison is vacuous"
    );
    set_prime_chunk("0");
}
