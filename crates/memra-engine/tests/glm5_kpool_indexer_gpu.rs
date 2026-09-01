//! GPU-vs-reference gate for glm5_next's DSA **k-pool indexer** — the sparse arm of the MLA path.
//!
//! WHY THIS GATE EXISTS. The MLA CUDA forward attended DENSELY over every cached position.
//! glm5_next's 11 MLA layers (+1 MTP) each run an indexer that selects at most `index_topk` = 2048
//! positions per query. Below that budget the indexer selects everything and the two are the same
//! function — which is exactly why `mla_gpu_forward` and `mla_fixture_forward` passed. ABOVE it
//! they are different functions, and 1,048,576 native context is this model's entire product
//! claim. A gate that cannot reach the sparse regime proves nothing about the model we sell.
//!
//! THE FIXTURE IS SIZED TO REACH THAT REGIME. `index_topk` 16 with `index_kpool` 4 gives a budget
//! of FOUR pools; at 64 tokens the cache holds SIXTEEN complete pools, so twelve of them are
//! rejected per query. A 2048-token budget can never be exercised in a micro fixture, so the
//! budget is what shrinks — not the property. `the_fixture_reaches_the_sparse_regime` asserts the
//! bite on the reference alone, with no CUDA, so a fixture that drifted back into full selection
//! fails on a GPU-less machine rather than turning every gate below into a tautology.
//!
//! TRUTH is `memra_reference::kpool_allowed_tokens`, itself a transcription of
//! `Glm5NextTextIndexer.forward` (research/glm53-flash-bringup-20260827/modular_glm5_next-ref.py).
//! Scope, and it is load-bearing: single sequence, unpadded, pooling from cache row 0. The engine
//! plane carries no per-token validity channel for the same reason.
//!
//! GATES:
//!   1. `the_plan_declares_the_kpool_indexer_and_its_state_plane` (host) — plan + contract wiring.
//!   2. `the_fixture_reaches_the_sparse_regime` (host) — non-vacuity, measured.
//!   3. `the_rival_programs_disagree_with_the_oracle` (host) — the mutations are real mutations.
//!   4. `gpu_kpool_selection_matches_the_reference` — SAME index sets, four context lengths, two
//!      of them above the budget.
//!   5. `gpu_kpool_attention_matches_the_reference_and_differs_from_dense` — end to end at a
//!      length where sparse and dense differ, against both.
//!   6. `gpu_kpool_mutations_change_the_selection_and_the_output` — the three rivals, with numbers.
//!   7. `gpu_a_missing_indexer_tensor_refuses_the_load_by_name` — no silent dense fallback.
//!   8. `gpu_kpool_prime_then_decode_matches_the_reference` — the cached arm, inside the sparse
//!      regime, where the indexer state has to survive across steps.
//!   9. `gpu_kpool_ties_break_on_the_lowest_pool_index` — the tie-break, deterministically.
//!      MEASURED to bite: flipping the kernel's membership test to `p >= tp` fails BOTH this gate
//!      and gate 4 (2026-08-28).
//!  10. `gpu_kpool_radix_selection_is_byte_identical_to_the_reference_kernel` — the SHIPPED radix
//!      selection against the `select_k`-rounds definition of the order, at the shipped budget
//!      (select_k 512) and thousands of pools, on tie-heavy distributions the micro fixture
//!      cannot produce. Gates 4 and 9 pin the reference kernel to the Rust oracle; this pins the
//!      fast kernel to the reference kernel, so the chain reaches serving scale.
//!  11. `gpu_kpool_resident_pool_keys_match_a_full_rebuild` — the resident pool-key plane: a
//!      decode step selecting from keys built by an earlier call must match the oracle's
//!      from-scratch rebuild.
//!  12. `gpu_kpool_scoring_is_byte_identical_to_the_reference_kernel` — the SHIPPED tiled scorer
//!      against the retained block-per-(query, pool) definition of the arithmetic, compared as
//!      u32 BITS across both tile-dispatch boundaries, ragged tiles, the micro fixture's own
//!      `d`, a `d` that overflows the tile's shared memory, and three causal horizons. Same
//!      arrangement as gate 10: the fast kernel is pinned to the reference kernel, which gates 4
//!      and 9 pin to the Rust oracle.
//!
//!  13. `gpu_kpool_tail_ring_wraps_and_matches_the_flat_plane` — the TAIL RING: selection parity
//!      across FOUR wraps of a 16-row ring at 64 tokens, the ring's index sets against the flat
//!      plane's, and a ring too small for the chunk refusing instead of selecting against rows it
//!      is about to overwrite.
//!  14. `gpu_kpool_tail_ring_mutations_change_the_pool_keys` — the ring's modulus is load-bearing:
//!      the incremental ring build is BIT-identical to the flat build, and both an off-by-one-pool
//!      reader ring and an off-by-one append slot move pool keys, with counts.
//!
//! Rig law (exactness only, never timing):
//!   NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
//!     cargo test -p memra-engine --test glm5_kpool_indexer_gpu -- --ignored

use cudarc::driver::CudaSlice;
use memra_engine::Engine;
use memra_engine::hybrid::{HybridModel, Mixer, MlaIndexer};
use memra_engine::hybrid_forward::IndexerPlanes;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::{
    AttentionPlan, KpoolPlan, MlaAttentionPlan, ModelPlan, SparseIndexPlan, StatePlan,
};
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, LayerTensor, OutputHead, TensorContract, TensorId,
    TensorMatch,
};
use memra_reference::{ReferenceWeights, deterministic_fixture};
use std::borrow::Cow;
use std::collections::BTreeMap;

const HIDDEN: usize = 128;
const VOCAB: u32 = 32;
const Q_LORA: usize = 16;
const INDEX_HEADS: usize = 2;
const INDEX_HEAD_DIM: usize = 8;
/// RAW-TOKEN budget. `select_k = INDEX_TOPK / KPOOL = 4` pools survive per query.
const INDEX_TOPK: usize = 16;
const KPOOL: usize = 4;

/// Context lengths the gates run. 8 and 16 sit AT or below the raw budget (selection is the
/// identity there — the regime every prior MLA gate lived in); 40 and 64 sit above it, where
/// 10 of 10 and 12 of 16 pools are rejected per late query.
const LENGTHS: [usize; 4] = [8, 16, 40, 64];
/// The length the end-to-end and mutation gates run at.
const CTX: usize = 64;

/// End-to-end tolerance against the reference. The engine runs the ABSORBED MLA form with f16
/// activation converts inside `matmul`; the reference runs the expanded form in plain f32 with a
/// different reduction order, so parity is a maxdiff bound, never bit-identity.
///
/// MEASURED on this fixture (5090, TF32 off, NVIDIA_TF32_OVERRIDE=0, 2026-08-28), relative
/// maxdiff of the GPU logits against `memra_reference::execute`:
///
/// | T  | indexed  | DENSE twin |
/// |----|----------|------------|
/// | 8  | 6.843e-7 | 6.843e-7   |  budget covers every pool: selection is the identity
/// | 16 | 6.843e-7 | 6.843e-7   |
/// | 40 | 6.843e-7 | 1.170e-1   |  budget bites: 264 (query, position) pairs rejected
/// | 64 | 6.843e-7 | 1.411e-1   |  1104 rejected
///
/// 1e-5 sits ~15x above the measured value and four orders BELOW the dense twin at the lengths
/// where the two are different functions. That separation is the gate, and the flat 6.843e-7
/// across all four lengths is what says the indexed arm is not accumulating error with context.
const TOL: f32 = 1e-5;

/// GPU tests serialize on one device.
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

// ---------------------------------------------------------------------------------------------
// The mini model: a real config.json through the real parser and the real glm5_next pack, so the
// plan under test is the one the pack compiles rather than a hand-built imitation.
// ---------------------------------------------------------------------------------------------

fn mini_config_json(kpool_compress: bool) -> String {
    format!(
        r#"{{
      "model_type": "glm5_next_text",
      "num_hidden_layers": 2,
      "num_nextn_predict_layers": 0,
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
      "q_lora_rank": {Q_LORA},
      "kv_lora_rank": 16,
      "qk_head_dim": 16,
      "qk_nope_head_dim": 16,
      "qk_rope_head_dim": 0,
      "v_head_dim": 16,
      "mla_use_nope": true,
      "index_n_heads": {INDEX_HEADS},
      "index_head_dim": {INDEX_HEAD_DIM},
      "index_topk": {INDEX_TOPK},
      "index_kpool": {KPOOL},
      "index_kpool_always_select_tail": true,
      "index_kpool_compress": {kpool_compress},
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

fn mini_config(kpool_compress: bool) -> ModelConfig {
    ModelConfig::from_hf(&HfConfig::parse(&mini_config_json(kpool_compress)))
}

fn mini_plan(config: &ModelConfig) -> ModelPlan {
    memra_gguf::model_packs::for_config(config)
        .expect("glm5_next model pack matches the mini config")
        .compile_plan(config)
        .expect("mini glm5_next plan compiles")
}

/// `tensor_contract::layer_id` is crate-private; the id shape is part of the contract, so the
/// test spells it rather than widening that surface for a fixture lookup.
fn layer_id(index: u32, tensor: LayerTensor) -> TensorId {
    TensorId::Layer { index, tensor }
}

fn kpool_plan() -> KpoolPlan {
    KpoolPlan {
        pool: KPOOL as u32,
        always_select_tail: true,
    }
}

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
    ggml_type: GgmlType,
}

/// Serves the reference fixture's own f32 numbers under the contract's ggml names, so the
/// reference and the GPU read ONE set of weights. Must answer `config()`:
/// `HybridModel::load_from_source_without_mtp` compiles the plan from it.
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

/// `skip` drops a tensor the contract requires — the loud-refusal gate's instrument.
fn fixture_source(
    config: &ModelConfig,
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    skip: Option<&str>,
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
        let names = match req.match_mode {
            TensorMatch::OneOf => &req.names[..1],
            TensorMatch::All => req.names.as_slice(),
        };
        for name in names {
            if Some(name.as_str()) == skip {
                continue;
            }
            tensors.insert(
                name.clone(),
                OwnedTensor {
                    bytes: tensor.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    ne: req.shape.clone(),
                    ggml_type: GgmlType::F32,
                },
            );
        }
    }
    FixtureSource {
        config: config.clone(),
        tensors,
    }
}

/// The layer-0 attention plan, as the pack compiled it.
fn layer0_attention(plan: &ModelPlan) -> &AttentionPlan {
    &plan.layers[0].attention
}

fn layer0_sparse_index(plan: &ModelPlan) -> &SparseIndexPlan {
    match layer0_attention(plan) {
        AttentionPlan::Mla(MlaAttentionPlan::LatentKv { sparse_index, .. }) => sparse_index,
        other => panic!("layer 0 must be an MLA LatentKv layer, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Host-side inputs and the oracle
// ---------------------------------------------------------------------------------------------

fn noise(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * scale
        })
        .collect()
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

fn oracle(weights: &ReferenceWeights, x: &[f32], q_resid: &[f32], t: usize) -> Vec<Vec<usize>> {
    memra_reference::kpool_allowed_tokens(
        0,
        INDEX_HEADS,
        INDEX_HEAD_DIM,
        INDEX_TOPK,
        &kpool_plan(),
        weights,
        x,
        q_resid,
        t,
        HIDDEN,
        Q_LORA,
    )
    .expect("reference k-pool selection")
}

/// The rival selection programs. Each is the oracle with ONE step replaced by a plausible wrong
/// one; `Rival::None` reproduces the oracle exactly, which is what keeps this harness from being
/// wrong in a way that flatters the engine (`the_rival_harness_reproduces_the_oracle`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Rival {
    None,
    /// The incomplete tail is never appended — the reference's `index_kpool_always_select_tail`.
    NoTail,
    /// The learned per-channel softmax collapse is skipped: a pool's key is its LAST token's key,
    /// un-collapsed. Every other step is untouched.
    UncollapsedKeys,
    /// Pool validity ignores causality: every complete pool in the cache is a candidate,
    /// including pools that end after the query.
    NoCausality,
}

impl Rival {
    fn label(self) -> &'static str {
        match self {
            Rival::None => "oracle (control)",
            Rival::NoTail => "drop always-select-tail",
            Rival::UncollapsedKeys => "un-collapsed per-token keys",
            Rival::NoCausality => "no causal pool validity",
        }
    }
}

/// The oracle's program with one step swapped. Deliberately a SEPARATE implementation rather than
/// a knob on `memra_reference::kpool_allowed_tokens`: a mutation that lives in the oracle is a
/// mutation the oracle could be wrong about.
fn rival_allowed(
    rival: Rival,
    weights: &ReferenceWeights,
    x: &[f32],
    q_resid: &[f32],
    t: usize,
) -> Vec<Vec<usize>> {
    let w = |tensor: LayerTensor| -> &[f32] {
        weights
            .get(&layer_id(0, tensor))
            .unwrap_or_else(|| panic!("fixture is missing {tensor:?}"))
            .data
            .as_slice()
    };
    let linear = |input: &[f32], weight: &[f32], rows: usize, inn: usize, out: usize| {
        let mut o = vec![0.0f32; rows * out];
        for r in 0..rows {
            for c in 0..out {
                let mut acc = 0.0f32;
                for k in 0..inn {
                    acc += input[r * inn + k] * weight[c * inn + k];
                }
                o[r * out + c] = acc;
            }
        }
        o
    };
    let d = INDEX_HEAD_DIM;
    let q = linear(
        q_resid,
        w(LayerTensor::SparseQuery),
        t,
        Q_LORA,
        INDEX_HEADS * d,
    );
    let mut key = linear(x, w(LayerTensor::SparseKey), t, HIDDEN, d);
    let (kw, kb) = (
        w(LayerTensor::SparseKeyNorm),
        w(LayerTensor::SparseKeyNormBias),
    );
    for row in 0..t {
        let slice = &mut key[row * d..(row + 1) * d];
        let mean = slice.iter().sum::<f32>() / d as f32;
        let var = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for (i, v) in slice.iter_mut().enumerate() {
            *v = (*v - mean) * inv * kw[i] + kb[i];
        }
    }
    let gate = linear(x, w(LayerTensor::SparseCompressorGate), t, HIDDEN, d);
    let ape = w(LayerTensor::SparseCompressorPosition);
    let pools = t / KPOOL;
    let mut pool_keys = vec![0.0f32; pools * d];
    for p in 0..pools {
        for c in 0..d {
            if rival == Rival::UncollapsedKeys {
                // MUTATION: no learned collapse — the pool is represented by its last token.
                pool_keys[p * d + c] = key[(p * KPOOL + KPOOL - 1) * d + c];
                continue;
            }
            let mut logits: Vec<f32> = (0..KPOOL)
                .map(|s| gate[(p * KPOOL + s) * d + c] + ape[s * d + c])
                .collect();
            let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for l in logits.iter_mut() {
                *l = (*l - m).exp();
                sum += *l;
            }
            let mut acc = 0.0;
            for (s, l) in logits.iter().enumerate() {
                acc += (l / sum) * key[(p * KPOOL + s) * d + c];
            }
            pool_keys[p * d + c] = acc;
        }
    }
    let mut hw = linear(x, w(LayerTensor::SparseProjection), t, HIDDEN, INDEX_HEADS);
    let head_scale = (INDEX_HEADS as f32).powf(-0.5);
    for v in &mut hw {
        *v *= head_scale;
    }
    let qk_scale = (d as f32).powf(-0.5);
    let select_k = (INDEX_TOPK / KPOOL).min(pools);
    (0..t)
        .map(|token| {
            let visible_pools = if rival == Rival::NoCausality {
                // MUTATION: every complete pool is a candidate, visible or not.
                pools
            } else {
                ((token + 1) / KPOOL).min(pools)
            };
            let mut scored: Vec<(usize, f32)> = (0..visible_pools)
                .map(|p| {
                    let mut score = 0.0f32;
                    for head in 0..INDEX_HEADS {
                        let mut dot = 0.0f32;
                        for dim in 0..d {
                            dot +=
                                q[(token * INDEX_HEADS + head) * d + dim] * pool_keys[p * d + dim];
                        }
                        score += (dot * qk_scale).max(0.0) * hw[token * INDEX_HEADS + head];
                    }
                    (p, score)
                })
                .collect();
            scored.sort_by(|l, r| {
                r.1.partial_cmp(&l.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(l.0.cmp(&r.0))
            });
            let mut selected: Vec<usize> = Vec::new();
            for &(p, _) in scored.iter().take(select_k) {
                selected.extend(p * KPOOL..(p + 1) * KPOOL);
            }
            if rival != Rival::NoTail {
                let visible = token + 1;
                let tail = visible % KPOOL;
                selected.extend(visible - tail..visible);
            }
            selected.sort_unstable();
            selected.dedup();
            selected
        })
        .collect()
}

/// The engine's `-1`-padded index rows, read back as ascending sets.
fn device_sets(idx: &[i32], t: usize, width: usize) -> Vec<Vec<usize>> {
    (0..t)
        .map(|i| {
            let mut row: Vec<usize> = idx[i * width..(i + 1) * width]
                .iter()
                .filter(|&&v| v >= 0)
                .map(|&v| v as usize)
                .collect();
            row.sort_unstable();
            row.dedup();
            row
        })
        .collect()
}

fn differing_queries(a: &[Vec<usize>], b: &[Vec<usize>]) -> usize {
    a.iter().zip(b).filter(|(l, r)| l != r).count()
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared slices differ in length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn relative(got: &[f32], want: &[f32]) -> f32 {
    maxdiff(got, want) / want.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6)
}

// ---------------------------------------------------------------------------------------------
// Host-only gates. No CUDA — these fail on a GPU-less machine when the wiring or the fixture
// stops being what the GPU gates assume.
// ---------------------------------------------------------------------------------------------

/// GATE 1 — WIRING. The pack must compile the k-pool indexer AND the state plane that carries it,
/// and the contract must name every indexer tensor. A plan that declares the indexer but no
/// `index_width` allocates no plane, and the forward refuses rather than attending densely.
#[test]
fn the_plan_declares_the_kpool_indexer_and_its_state_plane() {
    let config = mini_config(true);
    let plan = mini_plan(&config);
    assert_eq!(
        layer0_sparse_index(&plan),
        &SparseIndexPlan::Own {
            heads: INDEX_HEADS as u32,
            head_dim: INDEX_HEAD_DIM as u32,
            top_k: INDEX_TOPK as u32,
            kpool: Some(kpool_plan()),
        },
        "the glm5_next pack must compile a k-pool indexer for an MLA layer"
    );
    assert_eq!(
        plan.layers[0].state,
        StatePlan::LatentKvCache {
            width: 16,
            index_width: (2 * INDEX_HEAD_DIM) as u32,
        },
        "the latent state plan must carry the indexer's packed [k | gate] row width"
    );
    // With the collapse switched off the plan must fall back to the per-token variant, which is
    // what the dense twin in gate 5 rides.
    let dense = mini_plan(&mini_config(false));
    assert!(
        matches!(
            layer0_sparse_index(&dense),
            SparseIndexPlan::Own { kpool: None, .. }
        ),
        "index_kpool_compress=false must not compile a k-pool indexer"
    );
    assert_eq!(
        dense.layers[0].state,
        StatePlan::LatentKvCache {
            width: 16,
            index_width: 0,
        },
        "a layer with no k-pool indexer must not allocate an indexer plane"
    );

    let contract = TensorContract::for_plan(
        &plan,
        CheckpointDialect::Gguf,
        ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        },
    )
    .expect("contract");
    for tensor in [
        LayerTensor::SparseQuery,
        LayerTensor::SparseKey,
        LayerTensor::SparseKeyNorm,
        LayerTensor::SparseKeyNormBias,
        LayerTensor::SparseProjection,
        LayerTensor::SparseCompressorGate,
        LayerTensor::SparseCompressorPosition,
    ] {
        let id = layer_id(0, tensor);
        assert!(
            contract.requirements.iter().any(|r| r.id == id),
            "the GGUF contract must require {tensor:?} on a k-pool indexer layer"
        );
    }
}

/// GATE 2 — NON-VACUITY, MEASURED. If the fixture's budget covered every visible pool, every gate
/// below would compare dense attention against dense attention and pass forever. This asserts the
/// selection actually REJECTS candidates, on the reference alone.
#[test]
fn the_fixture_reaches_the_sparse_regime() {
    let plan = mini_plan(&mini_config(true));
    let weights = deterministic_fixture(&plan).expect("fixture").weights;
    let x = noise(CTX * HIDDEN, 0x51A7, 0.6);
    let q_resid = noise(CTX * Q_LORA, 0x9F13, 0.6);
    let allowed = oracle(&weights, &x, &q_resid, CTX);

    let budget = INDEX_TOPK + KPOOL - 1;
    let mut bitten = 0;
    let mut tightest = usize::MAX;
    for (token, sources) in allowed.iter().enumerate() {
        assert!(
            sources.iter().all(|&s| s <= token),
            "query {token} selected a source past itself"
        );
        assert!(
            sources.len() <= budget,
            "query {token} selected {} sources, budget is {budget}",
            sources.len()
        );
        if sources.len() < token + 1 {
            bitten += 1;
            tightest = tightest.min(token + 1 - sources.len());
        }
    }
    println!(
        "sparse regime: {bitten}/{CTX} queries attend a PROPER subset of their causal prefix; \
         query {} keeps {} of {CTX} positions (budget {budget})",
        CTX - 1,
        allowed[CTX - 1].len()
    );
    assert!(
        bitten >= CTX / 2,
        "only {bitten} of {CTX} queries are budget-limited — the fixture does not reach the \
         sparse regime and every gate below is vacuous"
    );
    assert!(
        allowed[CTX - 1].len() < CTX,
        "the last query still attends everything"
    );
    let _ = tightest;
}

/// GATE 3a — the rival harness must reproduce the oracle when nothing is mutated. Without this,
/// a rival that disagrees for its own reasons would "prove" every mutation.
#[test]
fn the_rival_harness_reproduces_the_oracle() {
    let plan = mini_plan(&mini_config(true));
    let weights = deterministic_fixture(&plan).expect("fixture").weights;
    let x = noise(CTX * HIDDEN, 0x51A7, 0.6);
    let q_resid = noise(CTX * Q_LORA, 0x9F13, 0.6);
    assert_eq!(
        rival_allowed(Rival::None, &weights, &x, &q_resid, CTX),
        oracle(&weights, &x, &q_resid, CTX),
        "the mutation harness disagrees with the oracle before any mutation"
    );
}

/// GATE 3b — each rival is a REAL mutation: it changes the selected sets on this fixture. A
/// mutation that agrees with the oracle here cannot fail the GPU gate, and saying so is the
/// difference between a mutation check and a decoration.
#[test]
fn the_rival_programs_disagree_with_the_oracle() {
    let plan = mini_plan(&mini_config(true));
    let weights = deterministic_fixture(&plan).expect("fixture").weights;
    let x = noise(CTX * HIDDEN, 0x51A7, 0.6);
    let q_resid = noise(CTX * Q_LORA, 0x9F13, 0.6);
    let truth = oracle(&weights, &x, &q_resid, CTX);
    for rival in [Rival::NoTail, Rival::UncollapsedKeys, Rival::NoCausality] {
        let got = rival_allowed(rival, &weights, &x, &q_resid, CTX);
        let n = differing_queries(&truth, &got);
        println!("mutation `{}`: {n}/{CTX} queries differ", rival.label());
        assert!(
            n > 0,
            "mutation `{}` selects the same sets as the oracle — it cannot fail a parity gate",
            rival.label()
        );
    }
}

// ---------------------------------------------------------------------------------------------
// GPU gates
// ---------------------------------------------------------------------------------------------

struct Harness {
    engine: Engine,
    model: HybridModel,
    plan: ModelPlan,
    weights: ReferenceWeights,
}

impl Harness {
    fn new(kpool_compress: bool) -> Self {
        force_true_f32();
        let config = mini_config(kpool_compress);
        let plan = mini_plan(&config);
        let weights = deterministic_fixture(&plan).expect("fixture").weights;
        // The dense twin must read the SAME numbers, so its source is built from the k-pool
        // plan's fixture; its own contract simply asks for fewer of them.
        let kpool_plan_for_weights = mini_plan(&mini_config(true));
        let weights = if kpool_compress {
            weights
        } else {
            deterministic_fixture(&kpool_plan_for_weights)
                .expect("fixture")
                .weights
        };
        let source = fixture_source(&config, &plan, &weights, None);
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

    fn indexer(&self) -> &MlaIndexer {
        match &self.model.layers[0].mixer {
            Mixer::Mla(mla) => mla
                .index
                .as_ref()
                .expect("layer 0 loaded its k-pool indexer"),
            _ => panic!("layer 0 must be Mixer::Mla"),
        }
    }

    /// Run the engine's selection on explicit inputs — the same entry point the MLA forward calls.
    fn select(&self, x: &[f32], q_resid: &[f32], t: usize) -> (Vec<i32>, usize) {
        let e = &self.engine;
        let indexer = self.indexer();
        let h = e.htod(x).expect("upload x");
        let qr = e.htod(q_resid).expect("upload q_resid");
        let mut plane = e
            .uninit(t * indexer.geom.state_width())
            .expect("indexer plane");
        let mut pool_keys = None;
        let mut ready = 0usize;
        let planes = IndexerPlanes {
            state: &mut plane,
            pool_keys: &mut pool_keys,
            ready: &mut ready,
            state_ring_rows: 0,
            capacity_tokens: t,
        };
        let (idx, width) = HybridModel::mla_kpool_indices(e, indexer, &h, &qr, planes, t, 0)
            .expect("k-pool selection");
        (self.dtoh_i32(&idx), width)
    }

    /// The same entry point, run TWICE over a plane that survives between the calls: the first
    /// call primes `prime` tokens, the second appends one. The second call's pool keys for every
    /// pool the first finished are RESIDENT, never rebuilt. Returns the decode step's selection.
    fn select_resident(&self, x: &[f32], q_resid: &[f32], prime: usize) -> (Vec<i32>, usize) {
        let e = &self.engine;
        let indexer = self.indexer();
        let t = prime + 1;
        let mut plane = e
            .uninit(t * indexer.geom.state_width())
            .expect("indexer plane");
        let mut pool_keys = None;
        let mut ready = 0usize;
        let hidden = x.len() / t;
        let q_lora = q_resid.len() / t;
        let h_head = e.htod(&x[..prime * hidden]).expect("upload x head");
        let qr_head = e
            .htod(&q_resid[..prime * q_lora])
            .expect("upload q_resid head");
        HybridModel::mla_kpool_indices(
            e,
            indexer,
            &h_head,
            &qr_head,
            IndexerPlanes {
                state: &mut plane,
                pool_keys: &mut pool_keys,
                ready: &mut ready,
                state_ring_rows: 0,
                capacity_tokens: t,
            },
            prime,
            0,
        )
        .expect("prime selection");
        assert_eq!(
            ready,
            prime / indexer.geom.pool,
            "the prime must leave every complete pool resident"
        );
        let h_tail = e.htod(&x[prime * hidden..]).expect("upload x tail");
        let qr_tail = e
            .htod(&q_resid[prime * q_lora..])
            .expect("upload q_resid tail");
        let (idx, width) = HybridModel::mla_kpool_indices(
            e,
            indexer,
            &h_tail,
            &qr_tail,
            IndexerPlanes {
                state: &mut plane,
                pool_keys: &mut pool_keys,
                ready: &mut ready,
                state_ring_rows: 0,
                capacity_tokens: t,
            },
            1,
            prime,
        )
        .expect("resident-plane decode selection");
        (self.dtoh_i32(&idx), width)
    }

    /// CHUNKED prime over a plane the caller sizes: `ring_rows` 0 is the flat `t`-row plane every
    /// other gate uses, `ring_rows > 0` is a TAIL RING of that many PHYSICAL rows. Returns every
    /// chunk's selection concatenated, so the whole context is compared, not just the last query.
    ///
    /// This is the shape the cached arm runs in production — `mla_attn_cached` calls
    /// `mla_kpool_indices` once per prime chunk and once per decode step against one surviving
    /// plane — with the chunk shrunk until a micro fixture can lap a micro ring.
    fn select_chunked(
        &self,
        x: &[f32],
        q_resid: &[f32],
        t: usize,
        chunk: usize,
        ring_rows: usize,
    ) -> Vec<Vec<usize>> {
        let e = &self.engine;
        let indexer = self.indexer();
        let plane_rows = if ring_rows > 0 { ring_rows } else { t };
        let mut plane = e
            .zeros(plane_rows * indexer.geom.state_width())
            .expect("indexer plane");
        let mut pool_keys = None;
        let mut ready = 0usize;
        let mut out = Vec::with_capacity(t);
        let mut slot = 0usize;
        while slot < t {
            let n = chunk.min(t - slot);
            let h = e
                .htod(&x[slot * HIDDEN..(slot + n) * HIDDEN])
                .expect("upload x chunk");
            let qr = e
                .htod(&q_resid[slot * Q_LORA..(slot + n) * Q_LORA])
                .expect("upload q_resid chunk");
            let (idx, width) = HybridModel::mla_kpool_indices(
                e,
                indexer,
                &h,
                &qr,
                IndexerPlanes {
                    state: &mut plane,
                    pool_keys: &mut pool_keys,
                    ready: &mut ready,
                    state_ring_rows: ring_rows,
                    capacity_tokens: t,
                },
                n,
                slot,
            )
            .unwrap_or_else(|e| panic!("chunk at slot {slot} (t={n}, ring {ring_rows}): {e}"));
            out.extend(device_sets(&self.dtoh_i32(&idx), n, width));
            slot += n;
        }
        out
    }

    /// The same walk, but stopping at the FIRST refusal — the lapped-ring arm.
    fn select_chunked_result(
        &self,
        x: &[f32],
        q_resid: &[f32],
        t: usize,
        chunk: usize,
        ring_rows: usize,
    ) -> Result<(), String> {
        let e = &self.engine;
        let indexer = self.indexer();
        let mut plane = e
            .zeros(ring_rows * indexer.geom.state_width())
            .expect("indexer plane");
        let mut pool_keys = None;
        let mut ready = 0usize;
        let mut slot = 0usize;
        while slot < t {
            let n = chunk.min(t - slot);
            let h = e
                .htod(&x[slot * HIDDEN..(slot + n) * HIDDEN])
                .expect("upload x chunk");
            let qr = e
                .htod(&q_resid[slot * Q_LORA..(slot + n) * Q_LORA])
                .expect("upload q_resid chunk");
            HybridModel::mla_kpool_indices(
                e,
                indexer,
                &h,
                &qr,
                IndexerPlanes {
                    state: &mut plane,
                    pool_keys: &mut pool_keys,
                    ready: &mut ready,
                    state_ring_rows: ring_rows,
                    capacity_tokens: t,
                },
                n,
                slot,
            )
            .map_err(|e| e.to_string())?;
            slot += n;
        }
        Ok(())
    }

    fn dtoh_i32(&self, d: &CudaSlice<i32>) -> Vec<i32> {
        self.engine
            .stream()
            .clone_dtoh(d)
            .expect("index list readback")
    }

    fn logits(&self, ids: &[u32]) -> Vec<f32> {
        self.model.forward(&self.engine, ids).expect("gpu prefill")
    }

    fn reference_logits(&self, ids: &[u32]) -> Vec<f32> {
        memra_reference::execute(&self.plan, &self.weights, ids)
            .expect("reference execute")
            .logits
    }
}

/// GATE 4 — SELECTION PARITY. The device's selected index sets must be the oracle's, at four
/// context lengths INCLUDING two above the raw budget where the selection actually rejects.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_kpool_selection_matches_the_reference() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    for &t in &LENGTHS {
        let x = noise(t * HIDDEN, 0x51A7, 0.6);
        let q_resid = noise(t * Q_LORA, 0x9F13, 0.6);
        let (idx, width) = h.select(&x, &q_resid, t);
        let got = device_sets(&idx, t, width);
        let want = oracle(&h.weights, &x, &q_resid, t);
        let pools = t / KPOOL;
        let select_k = (INDEX_TOPK / KPOOL).min(pools);
        let rejected: usize = want
            .iter()
            .enumerate()
            .map(|(token, s)| (token + 1) - s.len())
            .sum();
        println!(
            "T={t}: {pools} pools, budget {select_k}, index width {width}, \
             {rejected} (query, position) pairs rejected across the batch"
        );
        for (token, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g, w,
                "T={t} query {token}: device selected {g:?}, oracle selected {w:?}"
            );
        }
    }
}

/// GATE 5 — END TO END, at a length where sparse and dense DIFFER. Three logit sets from one set
/// of weights: the reference (truth), the engine with the indexer, and the engine WITHOUT it. The
/// gate is only meaningful if the third disagrees, which is asserted rather than assumed.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_kpool_attention_matches_the_reference_and_differs_from_dense() {
    let _gpu = gpu_guard();
    let sparse = Harness::new(true);
    let dense = Harness::new(false);
    let ids = tokens(CTX, 0xC7E1);

    // Reported at EVERY length, not just the sparse one: below the budget the indexed and dense
    // arms must be the SAME number (selection is the identity there), and above it they must
    // separate. Both halves are the receipt that this gate reaches the regime it claims.
    for &t in &LENGTHS {
        let short = tokens(t, 0xC7E1);
        let truth = sparse.reference_logits(&short);
        println!(
            "T={t}: indexed vs reference {:.3e}   dense vs reference {:.3e}",
            relative(&sparse.logits(&short), &truth),
            relative(&dense.logits(&short), &truth)
        );
    }
    let truth = sparse.reference_logits(&ids);
    let got = sparse.logits(&ids);
    let dense_logits = dense.logits(&ids);

    assert!(
        got.iter().all(|v| v.is_finite()),
        "indexed MLA produced non-finite logits"
    );
    let rel = relative(&got, &truth);
    let dense_rel = relative(&dense_logits, &truth);
    println!(
        "T={CTX}: indexed vs reference {rel:.3e} (tol {TOL:.1e}); DENSE vs reference {dense_rel:.3e}"
    );
    assert!(
        dense_rel > 10.0 * TOL,
        "dense attention answers the same as the indexed path ({dense_rel:.3e}) — this fixture \
         never leaves the full-selection regime and the gate proves nothing"
    );
    assert!(
        rel <= TOL,
        "indexed MLA vs reference relative maxdiff {rel:.3e} exceeds {TOL:.1e}"
    );
}

/// GATE 6 — MUTATIONS, WITH NUMBERS. Each rival program is fed through the SAME gathered-attention
/// kernel the passing path uses, so the reported difference is what that mutation would have done
/// to the model's output — not just to a list of integers.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_kpool_mutations_change_the_selection_and_the_output() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let e = &h.engine;
    let t = CTX;
    let x = noise(t * HIDDEN, 0x51A7, 0.6);
    let q_resid = noise(t * Q_LORA, 0x9F13, 0.6);

    let (idx, width) = h.select(&x, &q_resid, t);
    let truth_sets = oracle(&h.weights, &x, &q_resid, t);
    assert_eq!(
        device_sets(&idx, t, width),
        truth_sets,
        "the device selection must match the oracle before mutations mean anything"
    );

    // A latent plane and queries in rank space, so the gathered kernel runs the real body. The
    // numbers are arbitrary: what is under test is which cache rows each query reaches.
    let (n_head, kv_rank, d_rope) = (2usize, 16usize, 0usize);
    let cache = e.htod(&noise(t * kv_rank, 0x2B41, 1.0)).expect("cache");
    let q_lat = e
        .htod(&noise(t * n_head * kv_rank, 0x77C5, 1.0))
        .expect("q_lat");
    let q_pe = e.uninit(1).expect("q_pe placeholder");
    let scale = 1.0 / (16.0f32).sqrt();

    let attend = |list: &[i32], slots: usize| -> Vec<f32> {
        let idx_d = e.htod_i32(list).expect("upload index list");
        let mut o = e.uninit(t * n_head * kv_rank).expect("o_lat");
        e.mla_attn_gathered(
            &q_lat, &q_pe, &cache, &idx_d, &mut o, n_head, kv_rank, d_rope, t, slots, scale,
        )
        .expect("gathered attention");
        e.stream().clone_dtoh(&o).expect("o readback")
    };
    let truth_out = attend(&idx, width);

    for rival in [Rival::NoTail, Rival::UncollapsedKeys, Rival::NoCausality] {
        let sets = rival_allowed(rival, &h.weights, &x, &q_resid, t);
        let differing = differing_queries(&truth_sets, &sets);
        let positions: usize = truth_sets
            .iter()
            .zip(&sets)
            .map(|(a, b)| {
                let (mut i, mut j, mut diff) = (0, 0, 0);
                while i < a.len() && j < b.len() {
                    match a[i].cmp(&b[j]) {
                        std::cmp::Ordering::Equal => {
                            i += 1;
                            j += 1;
                        }
                        std::cmp::Ordering::Less => {
                            diff += 1;
                            i += 1;
                        }
                        std::cmp::Ordering::Greater => {
                            diff += 1;
                            j += 1;
                        }
                    }
                }
                diff + (a.len() - i) + (b.len() - j)
            })
            .sum();
        // A rival's list padded to the same width, so the same kernel launch shape runs.
        let mut list = vec![-1i32; t * width];
        for (token, sources) in sets.iter().enumerate() {
            assert!(
                sources.len() <= width,
                "mutation `{}` selected {} sources, wider than the index list ({width})",
                rival.label(),
                sources.len()
            );
            for (slot, &s) in sources.iter().enumerate() {
                list[token * width + slot] = s as i32;
            }
        }
        // Rows the mutation leaves with NO candidate divide by a zero softmax denominator — the
        // reference refuses that program outright — so the output number is measured over the
        // rows that survive, and the count of emptied rows is reported beside it rather than
        // hidden behind an infinity.
        let emptied = sets.iter().filter(|s| s.is_empty()).count();
        let mut live_got = Vec::new();
        let mut live_want = Vec::new();
        let mutated_out = attend(&list, width);
        let row = n_head * kv_rank;
        for (token, sources) in sets.iter().enumerate() {
            if sources.is_empty() {
                continue;
            }
            live_got.extend_from_slice(&mutated_out[token * row..(token + 1) * row]);
            live_want.extend_from_slice(&truth_out[token * row..(token + 1) * row]);
        }
        let out_rel = relative(&live_got, &live_want);
        println!(
            "mutation `{}`: {differing}/{t} queries differ, {positions} position mismatches, \
             {emptied} queries left with no candidate, attention output relative maxdiff \
             {out_rel:.3e} over the {} rows that keep one",
            rival.label(),
            t - emptied
        );
        assert_ne!(
            sets,
            truth_sets,
            "mutation `{}` did not change the selection",
            rival.label()
        );
        assert!(
            out_rel > TOL,
            "mutation `{}` moved the attention output by only {out_rel:.3e} — below the gate's \
             own tolerance, so the gate could not fail on it",
            rival.label()
        );
    }
}

/// GATE 7 — LOUD REFUSAL. A layer whose plan declares the k-pool indexer and whose checkpoint is
/// missing one of its tensors must FAIL THE LOAD, naming the tensor. Falling back to dense here
/// is the exact failure this whole lane exists to prevent: fluent, plausible, and wrong past
/// `index_topk` tokens.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_a_missing_indexer_tensor_refuses_the_load_by_name() {
    let _gpu = gpu_guard();
    force_true_f32();
    let config = mini_config(true);
    let plan = mini_plan(&config);
    let weights = deterministic_fixture(&plan).expect("fixture").weights;
    let engine = Engine::new(0).expect("CUDA engine on device 0");
    for missing in [
        "blk.0.indexer.kpool_ape.weight",
        "blk.0.indexer.kpool_gate.weight",
        "blk.0.indexer.attn_q_b.weight",
        "blk.0.indexer.k_norm.bias",
    ] {
        let source = fixture_source(&config, &plan, &weights, Some(missing));
        let error = HybridModel::load_from_source_without_mtp(&engine, &source)
            .err()
            .unwrap_or_else(|| {
                panic!("loading without `{missing}` must fail, not fall back to dense attention")
            })
            .to_string();
        assert!(
            error.contains(missing),
            "the refusal must name the missing tensor; got: {error}"
        );
    }
}

/// GATE 8 — THE DECODE SEAM. Prefill allocates the indexer's state plane for one call; the
/// cached arm carries it in `LatentKvLayer::index_rows` across steps, and every decode step
/// re-derives its pool keys from the WHOLE plane. That is a different code path from the
/// stateless forward, and it is the one that runs in production.
///
/// The prompt is 40 tokens and the run extends to 64 — every decode step scores 10 to 16 pools
/// against a 4-pool budget, so the seam is exercised inside the sparse regime, not below it.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_kpool_prime_then_decode_matches_the_reference() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let prompt = 40usize; // `prime_cache` asserts T >= PRIME_MIN_T
    let steps = CTX - prompt;
    let ids = tokens(CTX, 0x8_0A17_DEC0);
    let vocab = VOCAB as usize;
    let want = h.reference_logits(&ids);

    let mut cache =
        memra_engine::cache::Cache::new_planned(&h.engine, &h.model.cfg, &h.plan, CTX + 8)
            .expect("cache for the mini glm5_next model");
    for (il, plane) in cache.latent.iter().enumerate() {
        let plane = plane
            .as_ref()
            .unwrap_or_else(|| panic!("layer {il} must carry a latent plane"));
        assert_eq!(
            plane.index_width,
            2 * INDEX_HEAD_DIM,
            "layer {il}: the cache must allocate the indexer's packed [k | gate] plane"
        );
        assert!(
            plane.index_rows.is_some(),
            "layer {il}: index_width is set but no plane was allocated"
        );
    }

    let (primed, _seed, _hiddens) = h
        .model
        .prime_cache(&h.engine, &ids[..prompt], &mut cache, 0)
        .expect("indexed prime");
    let rel = relative(&primed, &want[(prompt - 1) * vocab..prompt * vocab]);
    println!("prime last row (T={prompt}): {rel:.3e}");
    assert!(
        rel <= TOL,
        "indexed prime last row {rel:.3e} exceeds {TOL:.1e}"
    );

    let mut worst = 0.0f32;
    for step in 0..steps {
        let row = prompt + step;
        let got = h
            .model
            .decode_step(&h.engine, ids[row], &mut cache)
            .expect("indexed decode step");
        let rel = relative(&got, &want[row * vocab..(row + 1) * vocab]);
        worst = worst.max(rel);
        assert!(
            rel <= TOL,
            "indexed decode step {step} (position {row}) {rel:.3e} exceeds {TOL:.1e}"
        );
    }
    println!("worst of {steps} decode steps: {worst:.3e}");
}

/// GATE 9 — THE TIE-BREAK, PINNED DETERMINISTICALLY. ReLU zeroes every head whose query-pool dot
/// is non-positive, so a pool scoring EXACTLY 0.0 is ordinary, not exotic; gate 4 only covers the
/// tie-break stochastically (through whichever ties this fixture happens to produce). The oracle
/// orders candidates score-descending then pool-index-ASCENDING, so an all-zero score row must
/// select pools `0..select_k` and nothing else. A max-first-wins or last-wins reduction passes
/// every other gate here and fails this one.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_kpool_ties_break_on_the_lowest_pool_index() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let (n_pools, select_k, queries) = (16usize, 4usize, 3usize);
    let width = select_k * KPOOL + KPOOL - 1;
    // Every visible pool ties at 0.0. `first_pos` puts each query at the end of the cache, so all
    // 16 pools are causally visible and only the tie-break decides.
    let score = e
        .htod(&vec![0.0f32; queries * n_pools])
        .expect("upload scores");
    let mut idx = e.uninit_i32(queries * width).expect("index list");
    e.mla_kpool_select(
        &score,
        &mut idx,
        queries,
        n_pools,
        KPOOL,
        select_k,
        width,
        n_pools * KPOOL - queries,
        true,
    )
    .expect("selection on an all-tied score row");
    let got = device_sets(
        &e.stream().clone_dtoh(&idx).expect("readback"),
        queries,
        width,
    );
    for (query, sources) in got.iter().enumerate() {
        let pools: Vec<usize> = sources
            .iter()
            .filter(|&&s| s < select_k * KPOOL)
            .map(|&s| s / KPOOL)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(
            pools,
            (0..select_k).collect::<Vec<_>>(),
            "query {query}: an all-tied row must select the LOWEST-indexed pools, got {sources:?}"
        );
    }
    println!(
        "all-tied row selects pools {:?}",
        (0..select_k).collect::<Vec<_>>()
    );
}

/// GATE 10 — THE RADIX SELECTION IS THE REFERENCE SELECTION, AT SERVING SCALE.
///
/// WHY THIS GATE EXISTS. Gates 4/8/9 above pin the SELECTION ORDER, but the fixture that pins it
/// tops out at 16 pools with a budget of 4 — one radix descent bin, per-thread chunks of one
/// element. The shipped shape is 262,144 pools with a budget of 512, where the radix kernel's
/// multi-pass descent, its early exit on a singleton bin, its rank bookkeeping across passes and
/// its two degenerate arms (`n_fin == 0`, `n_fin < select_k`) all run for the first time. None of
/// that is reachable from the micro fixture, so it is gated HERE, against the kernel the order is
/// DEFINED by: `memra_mla_kpool_select_ref_kernel`, the `select_k`-rounds original, which gates 4
/// and 9 hold to the Rust oracle.
///
/// The comparison is on the RAW index buffers, not on sets: byte identity, including the -1 pad
/// and the emit order, at 4096 and 8192 pools with the shipped budget of 512.
///
/// THE DISTRIBUTIONS ARE THE POINT. A uniform random row resolves in one or two passes and proves
/// almost nothing about the tie-break. These rows are chosen so the threshold lands INSIDE a run
/// of exactly-equal scores, which is the regime ReLU scoring actually produces:
///   * `Relu` — the real shape: `max(dot, 0) * w` summed over heads, so a majority of pools score
///     EXACTLY 0.0 and the budget boundary falls inside that run. Negative scores occur (the head
///     weights are not sign-constrained), which the key mapping has to order correctly.
///   * `Levels` — every score quantized to one of five values, so every bin is a huge tie run.
///   * `AllZero` — one value, the whole row: pure tie-break, 4096 candidates deep.
///   * `Masked` — only `select_k / 3` pools finite, the rest `-INFINITY`: fewer candidates than
///     the budget, the arm where the ref kernel exhausts its rounds early.
///   * `AllMasked` — nothing visible at all: the selection must emit only the tail.
/// Every query in a launch draws a DIFFERENT distribution, so one launch exercises all five and
/// the per-block independence at the same time.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_kpool_radix_selection_is_byte_identical_to_the_reference_kernel() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");

    /// Score-row shapes, in the order the queries of one launch draw them.
    #[derive(Clone, Copy, Debug)]
    enum Shape {
        Relu,
        Levels,
        AllZero,
        Masked,
        Uniform,
        AllMasked,
    }
    const SHAPES: [Shape; 6] = [
        Shape::Relu,
        Shape::Levels,
        Shape::AllZero,
        Shape::Masked,
        Shape::Uniform,
        Shape::AllMasked,
    ];

    fn row(shape: Shape, n_pools: usize, select_k: usize, seed: u64) -> Vec<f32> {
        let raw = noise(n_pools * 4, seed, 2.0);
        (0..n_pools)
            .map(|p| match shape {
                // relu(dot) * w summed over 4 heads: mostly exact 0.0, some negative.
                Shape::Relu => (0..4)
                    .map(|h| raw[p * 4 + h].max(0.0) * (raw[(p * 4 + h + 1) % raw.len()] - 0.25))
                    .sum(),
                Shape::Levels => ((raw[p * 4] * 2.5).floor() * 0.5).clamp(-1.0, 1.0),
                Shape::AllZero => 0.0,
                Shape::Masked => {
                    if p < select_k / 3 {
                        raw[p * 4]
                    } else {
                        f32::NEG_INFINITY
                    }
                }
                Shape::Uniform => raw[p * 4],
                Shape::AllMasked => f32::NEG_INFINITY,
            })
            .collect()
    }

    // The shipped budget. `pool` 4 and `select_k` 512 are glm5_next's `index_kpool` and
    // `index_topk / index_kpool`; only `n_pools` shrinks to keep the gate on this rig.
    const POOL: usize = 4;
    const SELECT_K: usize = 512;
    for &n_pools in &[4096usize, 8192] {
        let queries = SHAPES.len();
        let width = SELECT_K * POOL + POOL - 1;
        let mut score = Vec::with_capacity(queries * n_pools);
        for (q, &shape) in SHAPES.iter().enumerate() {
            score.extend(row(
                shape,
                n_pools,
                SELECT_K,
                0x5EED ^ (q as u64 * 0x9E37) ^ n_pools as u64,
            ));
        }
        let finite: Vec<usize> = (0..queries)
            .map(|q| {
                score[q * n_pools..(q + 1) * n_pools]
                    .iter()
                    .filter(|v| v.is_finite())
                    .count()
            })
            .collect();
        let ties: Vec<usize> = (0..queries)
            .map(|q| {
                let mut r: Vec<f32> = score[q * n_pools..(q + 1) * n_pools]
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .collect();
                r.sort_by(|a, b| b.partial_cmp(a).expect("finite"));
                match r.get(SELECT_K.min(r.len()).saturating_sub(1)) {
                    Some(&t) => r.iter().filter(|&&v| v == t).count(),
                    None => 0,
                }
            })
            .collect();

        let score_d = e.htod(&score).expect("upload scores");
        // `first_pos` puts the queries at the end of the cache so every pool is visible and the
        // tail is non-empty for some queries and empty for others.
        let first_pos = n_pools * POOL - queries;
        let mut fast = e.uninit_i32(queries * width).expect("radix index list");
        let mut refr = e.uninit_i32(queries * width).expect("ref index list");
        e.mla_kpool_select(
            &score_d, &mut fast, queries, n_pools, POOL, SELECT_K, width, first_pos, true,
        )
        .expect("radix selection");
        e.mla_kpool_select_ref(
            &score_d, &mut refr, queries, n_pools, POOL, SELECT_K, width, first_pos, true,
        )
        .expect("reference selection");
        let fast = e.stream().clone_dtoh(&fast).expect("radix readback");
        let refr = e.stream().clone_dtoh(&refr).expect("ref readback");

        for (q, &shape) in SHAPES.iter().enumerate() {
            let (lo, hi) = (q * width, (q + 1) * width);
            println!(
                "n_pools={n_pools} query {q} {shape:?}: {} finite candidates, {} pools tied AT the \
                 budget boundary, {} selected rows",
                finite[q],
                ties[q],
                fast[lo..hi].iter().filter(|&&v| v >= 0).count()
            );
            assert_eq!(
                &fast[lo..hi],
                &refr[lo..hi],
                "n_pools={n_pools} query {q} ({shape:?}): the radix selection differs from the \
                 reference kernel's — the 64-bit order key does not reproduce the oracle's \
                 (score descending, pool index ascending) total order"
            );
        }
        // Non-vacuity: at least one row must have the budget boundary sitting inside a tie run
        // deeper than one, or this gate tested distinct scores and proved nothing about ties.
        assert!(
            ties.iter().any(|&t| t > 1),
            "n_pools={n_pools}: no distribution put the budget boundary inside a tie run — this \
             gate no longer exercises the tie-break"
        );
    }
}

/// GATE 11 — THE RESIDENT POOL-KEY PLANE IS NOT A DIFFERENT PROGRAM.
///
/// The indexer no longer rebuilds every pool key per call: keys for pools completed by earlier
/// calls stay resident and only the new ones are built. That is only sound because a pool's key
/// depends on nothing but its own `pool` append-only state rows and the constant `kpool_ape`.
/// This gate spends the claim: prime `CTX - 1` tokens through a plane, decode one more through
/// the SAME plane, and require the decode step's selection to be the oracle's — which is computed
/// from scratch over all `CTX` tokens, i.e. from fully rebuilt keys.
///
/// `select_resident` also asserts the residency actually happened (`ready == prime / pool` after
/// the prime), so a regression that silently rebuilt everything would fail rather than pass by
/// doing more work.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_kpool_resident_pool_keys_match_a_full_rebuild() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let t = CTX;
    let x = noise(t * HIDDEN, 0x51A7, 0.6);
    let q_resid = noise(t * Q_LORA, 0x9F13, 0.6);
    let want = oracle(&h.weights, &x, &q_resid, t);

    let (idx, width) = h.select_resident(&x, &q_resid, t - 1);
    let got = device_sets(&idx, 1, width);
    println!(
        "primed {} tokens then decoded 1 over a resident pool-key plane: {} selected rows, \
         oracle {} — pools resident across the step: {}",
        t - 1,
        got[0].len(),
        want[t - 1].len(),
        (t - 1) / KPOOL
    );
    assert_eq!(
        got[0],
        want[t - 1],
        "the decode step selected {:?} from resident pool keys; a full rebuild selects {:?}",
        got[0],
        want[t - 1]
    );
}

/// GATE 12 — THE SHIPPED SCORING KERNEL IS NOT A DIFFERENT PROGRAM.
///
/// Scoring was the indexer's dominant stage at every shape on the ladder and catastrophic at
/// prefill (1294.5 ms per MLA layer for one 512-token chunk at 1M context, x12 layers = 15.5 s
/// per chunk). The shipped kernel is now a register-tiled fused GEMM+head-reduce; the old
/// block-per-(query, pool) kernel is retained as `mla_kpool_score_ref`, exactly as the radix
/// selection retained its `select_k`-rounds definition one gate up.
///
/// WHY BIT-IDENTITY IS THE BAR AND A TOLERANCE IS NOT. Selection sorts these scores with a
/// tie-break, and ReLU makes exact 0.0 ties ORDINARY rather than rare. A last-ulp difference
/// either side of zero moves a pool in or out of the budget, so a scorer that changed a score
/// bit would be a different selection program. The tiled kernel therefore reproduces the
/// reference's SIX-STEP rounding sequence — `fma.rn` chain over `c` ascending from +0.0f, then
/// `mul.rn(qk_scale)`, `max(_, +0.0f)`, `mul.rn(head_scale)`, `mul.rn`, `add.rn` over `h`
/// ascending from +0.0f — spelled with explicit intrinsics so no contraction decision can fork
/// it. This gate spends that claim as u32 BITS, not float equality: `-0.0` and `+0.0` compare
/// equal as floats and are exactly the divergence the epilogue's order decides.
///
/// SHAPES. `t_q` crosses both tile-dispatch boundaries (1 and 3 -> the decode tile, 8 and 100 ->
/// the 32-row tile, 128 and 512 -> the 128-row tile, 512 being the shipped prefill chunk);
/// `n_pools` is deliberately not a multiple of the 64-pool tile; `d` covers the micro fixture's
/// 8, the shipped 128, and a 256 that overflows the tile's shared-memory budget and must take
/// the reference fallback inside the launcher. `first_pos` puts the causal horizon at the front
/// (whole tiles invisible, the block early-out), the middle (the horizon INSIDE a tile) and the
/// end (everything visible).
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_kpool_scoring_is_byte_identical_to_the_reference_kernel() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    const POOL: usize = 4;

    /// `(t_q, n_pools, d, heads)`. The comment on each is what it is there to break.
    const CASES: [(usize, usize, usize, usize); 9] = [
        (1, 4097, 128, 32),   // decode at shipped geometry, ragged pool count
        (3, 1024, 128, 32),   // decode tile, several queries
        (8, 300, 128, 32),    // the 32-row tile's lower boundary
        (100, 4097, 128, 32), // 32-row tile, ragged in both axes
        (128, 1000, 128, 32), // the 128-row tile's lower boundary
        (512, 4097, 128, 32), // the SHIPPED prefill chunk
        (64, 17, 8, 2),       // the micro fixture's own geometry: d 8, heads 2, tiny pool count
        (5, 3, 8, 2),         // fewer pools than one tile row of threads
        (33, 129, 256, 3),    // d over the tile's smem budget -> reference fallback, odd heads
    ];

    let (mut zeros, mut finite_nonzero, mut masked) = (0usize, 0usize, 0usize);
    for (t_q, n_pools, d, heads) in CASES {
        // Every third (query, head) row is exactly zero, so its dot is an exact +0.0 and its
        // head partial is a SIGNED zero — the value whose sign the rounding order decides.
        let mut q = noise(
            t_q * heads * d,
            0x9F13 ^ (t_q as u64) << 8 ^ n_pools as u64,
            1.0,
        );
        for t in 0..t_q {
            for h in 0..heads {
                if (t + h) % 3 == 0 {
                    for c in 0..d {
                        q[(t * heads + h) * d + c] = 0.0;
                    }
                }
            }
        }
        let pool_keys = noise(n_pools * d, 0x51A7 ^ n_pools as u64, 1.0);
        // Head weights straddling zero, so a zero ReLU meets a NEGATIVE weight and the partial
        // is -0.0 rather than +0.0 before it reaches the accumulator.
        let hw: Vec<f32> = noise(t_q * heads, 0x2B41 ^ t_q as u64, 2.0)
            .iter()
            .map(|v| v - 0.4)
            .collect();
        let qk_scale = (d as f32).powf(-0.5);
        let head_scale = (heads as f32).powf(-0.5);

        let q_d = e.htod(&q).expect("upload q");
        let keys_d = e.htod(&pool_keys).expect("upload pool keys");
        let hw_d = e.htod(&hw).expect("upload head weights");
        let mut fast = e.uninit(t_q * n_pools).expect("fast score plane");
        let mut refr = e.uninit(t_q * n_pools).expect("reference score plane");

        let rows = n_pools * POOL;
        let horizons: Vec<usize> = [Some(0), Some(rows / 2), rows.checked_sub(t_q)]
            .into_iter()
            .flatten()
            .collect();
        for first_pos in horizons {
            e.mla_kpool_score(
                &q_d, &keys_d, &hw_d, &mut fast, t_q, heads, d, n_pools, POOL, first_pos, qk_scale,
                head_scale,
            )
            .expect("tiled scoring");
            e.mla_kpool_score_ref(
                &q_d, &keys_d, &hw_d, &mut refr, t_q, heads, d, n_pools, POOL, first_pos, qk_scale,
                head_scale,
            )
            .expect("reference scoring");
            let fast_h = e.stream().clone_dtoh(&fast).expect("fast readback");
            let refr_h = e.stream().clone_dtoh(&refr).expect("reference readback");

            zeros += refr_h.iter().filter(|v| **v == 0.0).count();
            finite_nonzero += refr_h
                .iter()
                .filter(|v| v.is_finite() && **v != 0.0)
                .count();
            masked += refr_h.iter().filter(|v| **v == f32::NEG_INFINITY).count();

            let bad = (0..fast_h.len()).find(|&i| fast_h[i].to_bits() != refr_h[i].to_bits());
            if let Some(i) = bad {
                let differing = (0..fast_h.len())
                    .filter(|&j| fast_h[j].to_bits() != refr_h[j].to_bits())
                    .count();
                panic!(
                    "t_q={t_q} n_pools={n_pools} d={d} heads={heads} first_pos={first_pos}: the \
                     tiled scoring kernel is not the reference kernel. {differing} of {} scores \
                     differ; first at (query {}, pool {}): tiled {:e} (0x{:08x}) vs reference \
                     {:e} (0x{:08x}). A changed score bit is a changed SELECTION, not a faster \
                     one — see the rounding-sequence contract in cu/mla_attn.cu.",
                    fast_h.len(),
                    i / n_pools,
                    i % n_pools,
                    fast_h[i],
                    fast_h[i].to_bits(),
                    refr_h[i],
                    refr_h[i].to_bits(),
                );
            }
        }
        println!(
            "t_q={t_q} n_pools={n_pools} d={d} heads={heads}: {} scores x {} horizons, \
             bit-identical",
            t_q * n_pools,
            3
        );
    }
    // Non-vacuity: the comparison must have seen all three score classes, or it proved nothing
    // about ties (exact zeros), about the arithmetic (finite non-zeros), or about the causal
    // horizon (-inf marks).
    println!(
        "across every case: {zeros} exact zeros, {finite_nonzero} finite non-zero, {masked} masked"
    );
    assert!(
        zeros > 0 && finite_nonzero > 0 && masked > 0,
        "the shape matrix produced {zeros} exact zeros, {finite_nonzero} finite non-zero scores \
         and {masked} -inf marks — a class with zero members means this gate no longer covers it"
    );
}

// ---------------------------------------------------------------------------------------------
// The TAIL RING
// ---------------------------------------------------------------------------------------------

/// PHYSICAL rows of the ring the two gates below run. A multiple of `KPOOL` (4), so the effective
/// ring equals it; `RING_ROWS + 2` exercises the round-DOWN the engine applies because the state
/// plan does not carry `pool` and the allocator therefore cannot book a pool-aligned budget.
const RING_ROWS: usize = 16;
/// Chunk width for the ring gates. Deliberately NOT a multiple of `KPOOL`: with a pool-aligned
/// chunk every call starts with `pools_ready * pool == slot` and the liveness bound degenerates to
/// `ring >= t`, which would never exercise the `pool - 1` term the ring is actually sized against.
const RING_CHUNK: usize = 7;

/// GATE 13 — THE TAIL RING WRAPS, AND WRAPPING CHANGES NOTHING.
///
/// The indexer state plane is read EXACTLY ONCE per row — by the pool-key build of the pool that
/// row belongs to — so every row under `pools_ready * pool` is dead and the plane can be a ring of
/// `R` rows instead of `max_ctx`. That deletes 11.94 GiB at 1M over glm5_next's 12 MLA layers.
/// The claim is that it costs NOTHING numerically, which is only true if the wrap is exercised:
/// a ring never driven past `R` is a flat plane with an unused modulo.
///
/// NON-VACUITY IS ASSERTED, NOT ASSUMED, in the shape of gate 2. `RING_ROWS` 16 against `CTX` 64
/// means the ring is overwritten FOUR times, and the gate refuses to pass if `CTX <= R`.
///
/// THREE THINGS, in one fixture:
///   1. SELECTION PARITY across the wrap — every one of the 64 queries, chunk by chunk, against
///      `memra_reference::kpool_allowed_tokens`, including the queries whose pools were collapsed
///      from rows that have since been overwritten twice.
///   2. RING == FLAT, compared as index SETS, not a tolerance. Same rows, same kernel, same
///      chunking; only the addresses differ. This is the direct receipt for "zero numeric cost".
///   3. A LAPPED ring REFUSES. `R` 8 with chunk 7 leaves `pools_ready * pool` three rows behind
///      `slot`, so the second call would collapse a pool over rows it is about to overwrite —
///      the liveness inequality catches it and the call fails loudly instead of selecting against
///      garbage.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_kpool_tail_ring_wraps_and_matches_the_flat_plane() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let t = CTX;
    let x = noise(t * HIDDEN, 0x51A7, 0.6);
    let q_resid = noise(t * Q_LORA, 0x9F13, 0.6);
    let want = oracle(&h.weights, &x, &q_resid, t);

    for rows in [RING_ROWS, RING_ROWS + 2] {
        let effective = rows / KPOOL * KPOOL;
        assert!(
            t > effective,
            "a ring of {effective} rows is never lapped by {t} tokens — this gate \
             would prove nothing about wraparound"
        );
        assert!(
            effective >= KPOOL - 1 + RING_CHUNK,
            "ring {effective} is below the liveness bound pool-1+t = {} — the gate \
             would be asserting on a refusal, not on parity",
            KPOOL - 1 + RING_CHUNK
        );
        let got = h.select_chunked(&x, &q_resid, t, RING_CHUNK, rows);
        println!(
            "ring {rows} physical rows (effective {effective}, {} tokens in chunks \
             of {RING_CHUNK}): the plane is overwritten {} times; {} pools collapse \
             from rows that were later reused",
            t,
            t / effective,
            t / KPOOL - effective / KPOOL
        );
        assert_eq!(got.len(), t, "every chunk's queries must be compared");
        for (token, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g,
                w,
                "ring {rows}, query {token} (ring row {}): device selected {g:?}, \
                 oracle selected {w:?}",
                token % effective
            );
        }
    }

    // 2. RING == FLAT, exactly.
    let flat = h.select_chunked(&x, &q_resid, t, RING_CHUNK, 0);
    let ring = h.select_chunked(&x, &q_resid, t, RING_CHUNK, RING_ROWS);
    let differ = differing_queries(&flat, &ring);
    println!(
        "ring {RING_ROWS} vs flat {t}: {differ}/{t} queries differ ({}x less state, same \
         selection)",
        t / RING_ROWS
    );
    assert_eq!(
        differ, 0,
        "the ring selected different rows than the flat plane on {differ} of {t} queries — the \
         ring is not a pure addressing change"
    );

    // 3. A LAPPED ring REFUSES rather than selecting against overwritten rows.
    let lapped = RING_ROWS / 2; // 8: fine while slot % pool <= 1, lapped at slot 7
    let err = h
        .select_chunked_result(&x, &q_resid, t, RING_CHUNK, lapped)
        .expect_err("a ring too small for the chunk must refuse, not select against dead rows");
    println!("ring {lapped} rows, chunk {RING_CHUNK}: refused with `{err}`");
    assert!(
        err.contains("tail ring lapped"),
        "the refusal must name the lapped ring; got `{err}`"
    );
}

/// GATE 14 — THE RING'S ADDRESSING IS LOAD-BEARING, WITH NUMBERS.
///
/// Gate 13 proves the ring agrees with the oracle. This proves the agreement is not an accident
/// of a fixture too small to notice — it mutates the mod itself and counts what changes, in the
/// shape of gate 6. Both mutations stay in bounds and are deterministic (the plane is zeroed), so
/// the numbers are the gate, not a crash.
///
///   * READER RING one POOL smaller than the WRITER's (12 vs 16): the classic off-by-one in a
///     modulus. Every pool whose rows do not happen to land on the same residue collapses over
///     the wrong rows.
///   * APPEND SLOT off by one: the writer stores each row one slot late, so every pool reads a
///     window shifted by one token.
///
/// Truth is the FLAT plane's own pool keys, compared as f32 bits: the ring must be BIT-identical
/// to the flat build, because it is the same kernel over the same values in the same order.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_kpool_tail_ring_mutations_change_the_pool_keys() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let e = &h.engine;
    let indexer = h.indexer();
    let ig = indexer.geom;
    let d = ig.head_dim;
    let sw = ig.state_width();
    let t = CTX;
    let n_pools = t / KPOOL;
    let ape = indexer.kpool_ape.float_data();
    let state = noise(t * sw, 0x2B7D, 0.9);

    // TRUTH: the flat plane, one shot, absolute addressing.
    let flat = e.htod(&state).expect("upload flat plane");
    let mut truth = e.uninit(n_pools * d).expect("truth keys");
    e.mla_kpool_pool_keys(&flat, ape, &mut truth, 0, n_pools, KPOOL, d, 0)
        .expect("flat pool keys");
    let truth: Vec<f32> = e.stream().clone_dtoh(&truth).expect("truth readback");

    // The ring, built INCREMENTALLY in chunks exactly as the cached arm does.
    let build = |ring_write: usize, ring_read: usize, slot_bias: usize| -> Vec<f32> {
        let mut plane = e.zeros(RING_ROWS * sw).expect("ring plane");
        let mut keys = e.zeros(n_pools * d).expect("ring keys");
        let mut ready = 0usize;
        let mut slot = 0usize;
        while slot < t {
            let n = RING_CHUNK.min(t - slot);
            // The packed row is [k_norm | gate], each `d` wide, so the two halves go in
            // de-interleaved exactly as `mla_kpool_indices` hands them over.
            let a: Vec<f32> = (0..n)
                .flat_map(|r| state[(slot + r) * sw..(slot + r) * sw + d].iter().copied())
                .collect();
            let b: Vec<f32> = (0..n)
                .flat_map(|r| {
                    state[(slot + r) * sw + d..(slot + r + 1) * sw]
                        .iter()
                        .copied()
                })
                .collect();
            let ad = e.htod(&a).expect("upload k half");
            let bd = e.htod(&b).expect("upload gate half");
            e.mla_index_append(&mut plane, &ad, &bd, slot + slot_bias, n, d, d, ring_write)
                .expect("ring append");
            let np = (slot + n) / KPOOL;
            e.mla_kpool_pool_keys(&plane, ape, &mut keys, ready, np, KPOOL, d, ring_read)
                .expect("ring pool keys");
            ready = np;
            slot += n;
        }
        e.stream().clone_dtoh(&keys).expect("ring keys readback")
    };

    let differing_pools = |got: &[f32]| -> usize {
        (0..n_pools)
            .filter(|p| (0..d).any(|c| got[p * d + c].to_bits() != truth[p * d + c].to_bits()))
            .count()
    };

    let control = build(RING_ROWS, RING_ROWS, 0);
    println!(
        "ring {RING_ROWS} vs flat {t}: {}/{n_pools} pool keys differ in BITS",
        differing_pools(&control)
    );
    assert_eq!(
        differing_pools(&control),
        0,
        "the ring build is not bit-identical to the flat build — maxdiff {:.3e}",
        maxdiff(&control, &truth)
    );

    for (label, got) in [
        (
            "reader ring 12 vs writer ring 16 (off-by-one-pool in the mod)",
            build(RING_ROWS, RING_ROWS - KPOOL, 0),
        ),
        ("append slot off by one", build(RING_ROWS, RING_ROWS, 1)),
    ] {
        let n = differing_pools(&got);
        println!(
            "mutation `{label}`: {n}/{n_pools} pool keys differ, maxdiff {:.3e}",
            maxdiff(&got, &truth)
        );
        assert!(
            n > 0,
            "mutation `{label}` produced the same pool keys as the correct ring — this gate \
             cannot catch a broken modulus"
        );
    }
}
