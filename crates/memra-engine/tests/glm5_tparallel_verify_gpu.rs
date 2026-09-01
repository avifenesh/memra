//! glm5_next T-PARALLEL VERIFY + ROLLBACK gate (lane/glm5-tparallel-verify).
//!
//! Holds down the two claims the spec loop stands on, on the same mini-glm5_next hc fixture
//! family as `glm5-hyper-batch-gate` / `glm5-hyper-ppn-gate` (4 mHC streams, mean collapse,
//! KDA + DSA(MLA+kpool) alternating, dense L0 + sigmoid noaux_tc MoE) plus ONE NextN block:
//!
//! 1. THE WALK: `glm5_verify_rows` scores t sequential tokens in one forward whose row r is
//!    BIT-IDENTICAL (full logits) to the plain `decode_step` that consumes the same token at
//!    the same position — the batched-decode kernel classes over ONE cache with causal
//!    row->row state chaining.
//! 2. THE ROLLBACK: accept-j-then-continue is BYTE-IDENTICAL to the sequential decode path
//!    that never drafted, for EVERY j in 0..=K. Red-proven with a stale-KDA-state mutation
//!    (rollback that forgets the state-column restore) and a pool-key-finalized-past-j
//!    mutation (rollback that forgets `truncate_index_pool_keys` — the kpool residency
//!    tripwire must fire by name).
//! 3. END TO END: `generate_spec_glm5` (draft -> verify -> accept -> rollback -> re-seed)
//!    produces a greedy tape byte-identical to plain decode at K=1..7, including forced
//!    full-accept and forced-rejection rounds; red-proven by disabling rollback.
//!
//! The compare is FULL-LOGIT bit identity wherever logits are compared (the M3 lesson: a
//! greedy tape can match while every logit differs); the e2e arm compares token tapes
//! because that IS the served contract, on top of walk rows already pinned bit-exact.
//!
//! Rig law (exactness only, never timing):
//!   NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
//!     cargo test -p memra-engine --test glm5_tparallel_verify_gpu -- --ignored --test-threads=1

use memra_engine::Engine;
use memra_engine::forward::argmax;
use memra_engine::glm_spec::Glm5SpecKnobs;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::ModelPlan;
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
/// Prompt length: past the k-pool raw budget (index_topk 8, kpool 4) so the trunk indexer
/// runs in the SPARSE regime for the verify rows — pool-key state is live, not decorative.
const PROMPT: usize = 24;
/// Drafts per round in the rollback arms (t = K+1 = 8 verify rows, inside the cap of 15).
const K: usize = 7;

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call yet in this process; call_once serializes test threads.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
    });
}

/// MTP load flag, mutated ONLY under `gpu_guard` (every loading test holds it).
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

/// FR-Spec trim flag, mutated ONLY under `gpu_guard`. Set before a load, cleared right
/// after it, so no other test in this process inherits a trim.
fn set_trim_flag(path: Option<&str>) {
    // SAFETY: serialized behind gpu_guard by every caller.
    unsafe {
        match path {
            Some(p) => std::env::set_var("MEMRA_FRSPEC_TRIM", p),
            None => std::env::remove_var("MEMRA_FRSPEC_TRIM"),
        }
    }
}

/// Verify-batch walk arm (lane/glm5-verify-batch), mutated ONLY under `gpu_guard`. The
/// walk reads `MEMRA_GLM5_VERIFY_BATCH` PER CALL, so both arms run in one process:
/// `Some(false)` = the per-row walk (flag `0`), `Some(true)`/`None` = the batched default.
fn set_verify_batch_flag(arm: Option<bool>) {
    // SAFETY: serialized behind gpu_guard by every caller.
    unsafe {
        match arm {
            Some(false) => std::env::set_var("MEMRA_GLM5_VERIFY_BATCH", "0"),
            Some(true) => std::env::set_var("MEMRA_GLM5_VERIFY_BATCH", "1"),
            None => std::env::remove_var("MEMRA_GLM5_VERIFY_BATCH"),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Fixture: the hyper-batch-gate mini config + ONE NextN block, through the real pack/contract.
// ---------------------------------------------------------------------------------------------

fn mini_config_json() -> String {
    r#"{
      "model_type": "glm5_next_text",
      "num_hidden_layers": 4,
      "num_nextn_predict_layers": 1,
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
      "layer_types": ["linear_attention", "deepseek_sparse_attention",
                      "linear_attention", "deepseek_sparse_attention"],
      "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
      "first_k_dense_replace": 1,
      "indexer_types": ["full", "full", "full", "full"],
      "linear_attn_config": {
        "num_heads": 1,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [0, 2],
        "full_attn_layers": [1, 3]
      },
      "num_attention_heads": 2,
      "num_key_value_heads": 2,
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

/// Deterministic non-trivial values (an all-ones norm cannot catch a swapped operand).
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

fn fixture_weights(plan: &ModelPlan) -> ReferenceWeights {
    let mut weights = deterministic_fixture(plan)
        .expect("deterministic glm5 hc+mtp fixture")
        .weights;
    // The generic fixture's MTP glue norms are all-ones; strengthen them (mtp-head gate's move).
    for (tensor, seed) in [
        (MtpTensor::EmbeddingNorm, 0xE0_12u64),
        (MtpTensor::HiddenNorm, 0x40_77),
        (MtpTensor::OutputNorm, 0x5EAD),
    ] {
        weights.insert(
            TensorId::Mtp { depth: 0, tensor },
            ReferenceTensor::new(vec![HIDDEN], varied(HIDDEN, seed, 0.8)).unwrap(),
        );
    }
    weights
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

/// F32 weights + NVFP4 expert banks with a LIVE per-expert `weight_scale_2` macro plane
/// (the loader refuses float banks) — engine-vs-engine identity needs one loadable set of
/// numbers, not a reference roundtrip. NVFP4 + macros since lane/glm5-vrest: it is the
/// SERVING expert class (Q8_0 is not `q8_expert_supported`, so the old banks rode the f32
/// staging path and the batched verify-rows MoE arm could never engage here — the walk
/// gates would have been vacuous for it). `in_f % 64 == 0` holds on every projection
/// (gate/up in = 128, down in = 64).
fn fixture_source(config: &ModelConfig, plan: &ModelPlan) -> FixtureSource {
    let weights = fixture_weights(plan);
    let contract = TensorContract::for_plan(
        plan,
        CheckpointDialect::Gguf,
        ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        },
    )
    .expect("contract for the mini glm5_next hc+mtp plan");
    let mut tensors = BTreeMap::new();
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
            "shape mismatch for {:?}",
            req.id
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
    // LIVE `weight_scale_2` macro plane per routed bank (`<stem>.scale`) — every value off
    // 1.0 in two bands (the epilogue-gate construction: a ramp through 1.0 would make one
    // expert's fold a no-op), so a dropped macro fold anywhere in the walk moves bits.
    let n_expert = config
        .moe
        .as_ref()
        .expect("mini fixture carries an MoE block")
        .expert_count as usize;
    let macros: Vec<f32> = (0..n_expert)
        .map(|e| {
            if e < n_expert / 2 {
                0.5 + 0.1 * e as f32
            } else {
                1.2 + 0.1 * (e - n_expert / 2) as f32
            }
        })
        .collect();
    assert!(!bank_stems.is_empty(), "no routed expert banks collected");
    for stem in &bank_stems {
        tensors.insert(
            format!("{stem}.scale"),
            OwnedTensor {
                bytes: macros.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ne: vec![n_expert as u64],
                ggml_type: GgmlType::F32,
            },
        );
    }
    FixtureSource {
        config: config.clone(),
        tensors,
    }
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
}

impl Harness {
    /// Callers hold `gpu_guard`. `with_mtp` loads the NextN draft head (e2e arms).
    fn new(with_mtp: bool) -> Self {
        set_trim_flag(None);
        Self::load(with_mtp)
    }

    /// Load WITH the MTP head AND an FR-Spec ranks artifact (`MEMRA_FRSPEC_TRIM=<path>`).
    /// The flag is cleared again right after the load. Callers hold `gpu_guard`.
    fn new_trimmed(ranks_path: &str) -> Self {
        set_trim_flag(Some(ranks_path));
        let h = Self::load(true);
        set_trim_flag(None);
        h
    }

    fn load(with_mtp: bool) -> Self {
        force_true_f32();
        set_mtp_flag(with_mtp);
        let config = mini_config();
        let plan = mini_plan(&config);
        let source = fixture_source(&config, &plan);
        let engine = Engine::new(0).expect("CUDA engine on device 0");
        let model = if with_mtp {
            HybridModel::load_from_source(&engine, &source).expect("mini glm5 loads (with MTP)")
        } else {
            HybridModel::load_from_source_without_mtp(&engine, &source)
                .expect("mini glm5 loads (trunk only)")
        };
        assert!(
            model.hyper.is_some(),
            "the fixture must load as a HyperConnections trunk"
        );
        Self {
            engine,
            model,
            plan,
        }
    }

    fn fresh_primed(
        &self,
        prompt: &[u32],
        max_ctx: usize,
    ) -> (memra_engine::cache::Cache, Vec<f32>) {
        let mut cache = memra_engine::cache::Cache::new_planned(
            &self.engine,
            &self.model.cfg,
            &self.plan,
            max_ctx,
        )
        .expect("cache for the mini glm5 model");
        let (logits, _seed, _hiddens) = self
            .model
            .prime_cache(&self.engine, prompt, &mut cache, 0)
            .expect("hc prime");
        (cache, logits)
    }
}

fn bit_diffs(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

// ---------------------------------------------------------------------------------------------
// Gate 1 — THE WALK: verify rows == plain decode rows, full logits, bit for bit.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_verify_rows_match_plain_decode_rows_bitwise() {
    let _gpu = gpu_guard();
    let h = Harness::new(false);
    let prompt = tokens(PROMPT, 0xA11CE);
    let vt = tokens(K + 1, 0xBEEF);
    let max_ctx = PROMPT + K + 16;

    // Plain arm: sequential decode_step over the same tokens.
    let (mut cache_ref, _l) = h.fresh_primed(&prompt, max_ctx);
    let mut plain_rows: Vec<Vec<f32>> = Vec::with_capacity(vt.len());
    for &tok in &vt {
        plain_rows.push(
            h.model
                .decode_step(&h.engine, tok, &mut cache_ref)
                .expect("plain decode step"),
        );
    }

    // Verify arm: one t=K+1 walk.
    let (mut cache, _l) = h.fresh_primed(&prompt, max_ctx);
    let (vlogits, _collapsed, _ckpt) = h
        .model
        .glm5_verify_rows(&h.engine, &vt, &mut cache)
        .expect("verify rows walk");
    let host = h.engine.dtoh(&vlogits).expect("verify logits readback");
    let nv = plain_rows[0].len();
    let mut bad = 0usize;
    for (r, plain) in plain_rows.iter().enumerate() {
        let diffs = bit_diffs(&host[r * nv..(r + 1) * nv], plain);
        if diffs > 0 {
            bad += 1;
            println!("row {r}: {diffs}/{nv} logits differ");
        }
    }
    assert_eq!(
        bad, 0,
        "verify walk rows must be BIT-IDENTICAL to the plain decode chain"
    );
    println!(
        "gate 1 PASS: {} verify rows bit-identical to plain decode (n_vocab={nv})",
        vt.len()
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 1b — VERIFY-BATCH FLAG A/B (lane/glm5-verify-batch): both walk arms bit-identical to
// the plain chain, each arm PROVEN to have run via the ckpt stash kind (the wiring anchor),
// and accept-j rollback byte-identical under BOTH arms.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_verify_batch_flag_ab_bitwise() {
    let _gpu = gpu_guard();
    let h = Harness::new(false);
    let prompt = tokens(PROMPT, 0xA11CE);
    let vt = tokens(K + 1, 0xBEEF);
    let cc = tokens(12, 0xC0FFEE);
    let max_ctx = PROMPT + K + cc.len() + 16;

    // Plain reference rows.
    let (mut cache_ref, _l) = h.fresh_primed(&prompt, max_ctx);
    let mut plain_rows: Vec<Vec<f32>> = Vec::with_capacity(vt.len());
    for &tok in &vt {
        plain_rows.push(h.model.decode_step(&h.engine, tok, &mut cache_ref).unwrap());
    }
    let nv = plain_rows[0].len();

    for (arm, batched) in [(Some(true), true), (Some(false), false)] {
        set_verify_batch_flag(arm);
        let (mut cache, _l) = h.fresh_primed(&prompt, max_ctx);
        let vrows0 = memra_engine::moe_vrows_dispatches();
        let (vlogits, _coll, ckpt) = h
            .model
            .glm5_verify_rows(&h.engine, &vt, &mut cache)
            .expect("verify rows walk");
        // Wiring anchor: the arm we set actually ran (kda rows-stash vs per-row cols).
        let (rows_stash, col_stash) = ckpt.kda_stash_kinds();
        // Wiring anchor 2 (lane/glm5-vrest): the batched arm's MoE took the pairs-shaped
        // rows program on this fixture's NVFP4+macro banks (>0), and the per-row arm did
        // NOT (==0) — anchored on the dispatch counter, never on liveness.
        let vrows_delta = memra_engine::moe_vrows_dispatches() - vrows0;
        if batched {
            assert!(
                rows_stash > 0 && col_stash == 0,
                "batched arm did not fill the rows stash (rows={rows_stash}, cols={col_stash})"
            );
            assert!(
                vrows_delta > 0,
                "batched arm ran zero verify-rows MoE dispatches — the pairs arm is \
                 predicate-denied on this fixture and the walk gates are vacuous for it"
            );
        } else {
            assert!(
                col_stash > 0 && rows_stash == 0,
                "per-row arm did not fill the column stash (rows={rows_stash}, cols={col_stash})"
            );
            assert_eq!(
                vrows_delta, 0,
                "per-row arm took the batched MoE program — the =0 seam does not restore \
                 the per-(token,expert) class"
            );
        }
        let host = h.engine.dtoh(&vlogits).expect("verify logits");
        let mut bad = 0usize;
        for (r, plain) in plain_rows.iter().enumerate() {
            bad += usize::from(bit_diffs(&host[r * nv..(r + 1) * nv], plain) > 0);
        }
        assert_eq!(
            bad, 0,
            "arm batched={batched}: verify rows diverged from the plain chain"
        );

        // Accept-j-then-continue under this arm, three keeps (full sweep is gate 2).
        for j in [0usize, 2, K] {
            let keep = j + 1;
            let (mut cr, _l) = h.fresh_primed(&prompt, max_ctx);
            for &tok in &vt[..keep] {
                let _ = h.model.decode_step(&h.engine, tok, &mut cr).unwrap();
            }
            let mut ref_rows: Vec<Vec<f32>> = Vec::with_capacity(cc.len());
            for &tok in &cc {
                ref_rows.push(h.model.decode_step(&h.engine, tok, &mut cr).unwrap());
            }
            let (mut cache, _l) = h.fresh_primed(&prompt, max_ctx);
            let (_vl, _co, ckpt) = h
                .model
                .glm5_verify_rows(&h.engine, &vt, &mut cache)
                .expect("walk");
            h.model
                .glm5_verify_rollback(&h.engine, &mut cache, &ckpt, keep)
                .expect("rollback");
            let mut bad = 0usize;
            for (step, &tok) in cc.iter().enumerate() {
                let got = h.model.decode_step(&h.engine, tok, &mut cache).unwrap();
                bad += usize::from(bit_diffs(&got, &ref_rows[step]) > 0);
            }
            assert_eq!(
                bad, 0,
                "arm batched={batched} j={j}: accept-then-continue diverged"
            );
        }
        println!("gate 1b PASS batched={batched}: rows + rollback bit-identical");
    }
    set_verify_batch_flag(None);
}

// ---------------------------------------------------------------------------------------------
// Gate 1c — RED (batched arm): a corrupted conv ring must make the walk rows diverge from
// plain — proves the walk compare can see the conv-window state the prefill arm reads.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_batched_walk_sees_corrupted_conv_ring() {
    let _gpu = gpu_guard();
    let h = Harness::new(false);
    let prompt = tokens(PROMPT, 0xA11CE);
    let vt = tokens(K + 1, 0xBEEF);
    let max_ctx = PROMPT + K + 16;

    let (mut cache_ref, _l) = h.fresh_primed(&prompt, max_ctx);
    let mut plain_rows: Vec<Vec<f32>> = Vec::with_capacity(vt.len());
    for &tok in &vt {
        plain_rows.push(h.model.decode_step(&h.engine, tok, &mut cache_ref).unwrap());
    }
    let nv = plain_rows[0].len();

    set_verify_batch_flag(Some(true));
    let (mut cache, _l) = h.fresh_primed(&prompt, max_ctx);
    // Corrupt the first KDA layer's conv ring (every slot bumped — the prefill conv arm
    // reads the ring as the left pad for the early rows).
    let mut mutated = false;
    for il in 0..h.model.layers.len() {
        if let Some(rl) = cache.recur[il].as_mut() {
            let n = rl.conv_state.len();
            let bumped: Vec<f32> = h
                .engine
                .dtoh(&rl.conv_state)
                .unwrap()
                .iter()
                .map(|v| v + 1.0)
                .collect();
            h.engine
                .copy_into(&mut rl.conv_state, 0, &h.engine.htod(&bumped).unwrap(), n)
                .unwrap();
            mutated = true;
            break;
        }
    }
    assert!(mutated, "fixture has no KDA layer to corrupt");
    let (vlogits, _co, _ck) = h
        .model
        .glm5_verify_rows(&h.engine, &vt, &mut cache)
        .expect("walk over the corrupted ring");
    set_verify_batch_flag(None);
    let host = h.engine.dtoh(&vlogits).unwrap();
    let mut diffs = 0usize;
    for (r, plain) in plain_rows.iter().enumerate() {
        diffs += bit_diffs(&host[r * nv..(r + 1) * nv], plain);
    }
    assert!(
        diffs > 0,
        "a corrupted conv ring produced bit-identical verify rows — the batched walk gate \
         cannot see the conv window"
    );
    println!("gate 1c RED bites: {diffs} differing logits over the corrupted ring");
}

// ---------------------------------------------------------------------------------------------
// Gate 2 — THE ROLLBACK (green): accept-j-then-continue == never-drafted, for every j 0..=K.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_accept_j_then_continue_is_byte_identical_for_every_j() {
    let _gpu = gpu_guard();
    let h = Harness::new(false);
    let prompt = tokens(PROMPT, 0xA11CE);
    let vt = tokens(K + 1, 0xBEEF);
    // Continuation stream DIFFERENT from vt: the re-appended rows after a partial accept
    // must carry different bytes than the rejected drafts, or a rollback that keeps stale
    // rows could pass by coincidence.
    let cc = tokens(12, 0xC0FFEE);
    let max_ctx = PROMPT + K + cc.len() + 16;

    for j in 0..=K {
        let keep = j + 1;
        // Reference: never drafted — prime, consume vt[0..=j], then the continuation.
        let (mut cache_ref, _l) = h.fresh_primed(&prompt, max_ctx);
        for &tok in &vt[..keep] {
            let _ = h
                .model
                .decode_step(&h.engine, tok, &mut cache_ref)
                .expect("reference step");
        }
        let mut ref_rows: Vec<Vec<f32>> = Vec::with_capacity(cc.len());
        for &tok in &cc {
            ref_rows.push(
                h.model
                    .decode_step(&h.engine, tok, &mut cache_ref)
                    .expect("reference continue"),
            );
        }

        // Spec arm: drafted K, accepted j, rolled back, then the same continuation.
        let (mut cache, _l) = h.fresh_primed(&prompt, max_ctx);
        let pos0 = cache.pos;
        let (_vl, _coll, ckpt) = h
            .model
            .glm5_verify_rows(&h.engine, &vt, &mut cache)
            .expect("verify rows walk");
        h.model
            .glm5_verify_rollback(&h.engine, &mut cache, &ckpt, keep)
            .expect("rollback");
        assert_eq!(
            cache.pos,
            pos0 + keep,
            "rollback must land pos at snap+keep"
        );
        let mut bad = 0usize;
        for (step, &tok) in cc.iter().enumerate() {
            let got = h
                .model
                .decode_step(&h.engine, tok, &mut cache)
                .expect("spec-arm continue");
            let diffs = bit_diffs(&got, &ref_rows[step]);
            if diffs > 0 {
                bad += 1;
                println!(
                    "j={j} continue step {step}: {diffs}/{} logits differ",
                    got.len()
                );
            }
        }
        assert_eq!(
            bad, 0,
            "accept-{j}-then-continue must be BYTE-IDENTICAL to the never-drafted chain"
        );
        println!(
            "gate 2 PASS j={j}: {} continue steps bit-identical",
            cc.len()
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Gate 3 — RED: a rollback that forgets the KDA state-column restore must bite.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_stale_kda_state_mutation_bites() {
    let _gpu = gpu_guard();
    let h = Harness::new(false);
    let prompt = tokens(PROMPT, 0xA11CE);
    let vt = tokens(K + 1, 0xBEEF);
    let cc = tokens(12, 0xC0FFEE);
    let j = 2usize; // partial accept: the stale state is 5 rows ahead of the accepted one
    let keep = j + 1;
    let max_ctx = PROMPT + K + cc.len() + 16;

    // Reference (never drafted).
    let (mut cache_ref, _l) = h.fresh_primed(&prompt, max_ctx);
    for &tok in &vt[..keep] {
        let _ = h.model.decode_step(&h.engine, tok, &mut cache_ref).unwrap();
    }
    let mut ref_rows: Vec<Vec<f32>> = Vec::with_capacity(cc.len());
    for &tok in &cc {
        ref_rows.push(h.model.decode_step(&h.engine, tok, &mut cache_ref).unwrap());
    }

    // Mutated arm: correct rollback, then REINSTATE the post-row-K KDA state (the exact
    // hole a rollback that skips the column restore leaves behind).
    let (mut cache, _l) = h.fresh_primed(&prompt, max_ctx);
    let (_vl, _coll, ckpt) = h
        .model
        .glm5_verify_rows(&h.engine, &vt, &mut cache)
        .expect("verify rows walk");
    let mut stale: Vec<
        Option<(
            cudarc::driver::CudaSlice<f32>,
            cudarc::driver::CudaSlice<f32>,
        )>,
    > = Vec::new();
    for il in 0..h.model.layers.len() {
        stale.push(match cache.recur[il].as_ref() {
            Some(rl) => Some((
                h.engine.clone_dtod(&rl.conv_state).unwrap(),
                h.engine.clone_dtod(&rl.ssm_state).unwrap(),
            )),
            None => None,
        });
    }
    h.model
        .glm5_verify_rollback(&h.engine, &mut cache, &ckpt, keep)
        .expect("rollback");
    let mut mutated = 0usize;
    for (il, s) in stale.into_iter().enumerate() {
        if let (Some(rl), Some((conv, ssm))) = (cache.recur[il].as_mut(), s) {
            h.engine
                .copy_into(&mut rl.conv_state, 0, &conv, conv.len())
                .unwrap();
            h.engine
                .copy_into(&mut rl.ssm_state, 0, &ssm, ssm.len())
                .unwrap();
            mutated += 1;
        }
    }
    assert!(
        mutated > 0,
        "the mutation must touch at least one KDA layer"
    );

    let mut diffs_total = 0usize;
    for (step, &tok) in cc.iter().enumerate() {
        let got = h.model.decode_step(&h.engine, tok, &mut cache).unwrap();
        diffs_total += bit_diffs(&got, &ref_rows[step]);
    }
    assert!(
        diffs_total > 0,
        "stale post-row-K KDA state produced a byte-identical continuation — the gate \
         cannot see the KDA rollback seam and gate 2 is a decoration"
    );
    println!("gate 3 RED bites: {diffs_total} differing logits across the continuation");
}

// ---------------------------------------------------------------------------------------------
// Gate 4 — RED: a rollback that forgets the pool-key clamp must trip the residency tripwire.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_pool_keys_finalized_past_j_mutation_bites() {
    let _gpu = gpu_guard();
    let h = Harness::new(false);
    let prompt = tokens(PROMPT, 0xA11CE);
    let vt = tokens(K + 1, 0xBEEF);
    // keep=1: len falls 24+8 -> 25, complete pools 8 -> 6 — the clamp MUST move.
    let keep = 1usize;
    let max_ctx = PROMPT + K + 16;

    let (mut cache, _l) = h.fresh_primed(&prompt, max_ctx);
    let (_vl, _coll, ckpt) = h
        .model
        .glm5_verify_rows(&h.engine, &vt, &mut cache)
        .expect("verify rows walk");
    // Record the finalized-past-j pool counts BEFORE the (correct) rollback clamps them.
    let pre: Vec<Option<usize>> = cache
        .latent
        .iter()
        .take(h.model.layers.len())
        .map(|p| p.as_ref().map(|p| p.index_pools_ready))
        .collect();
    h.model
        .glm5_verify_rollback(&h.engine, &mut cache, &ckpt, keep)
        .expect("rollback");
    // THE MUTATION: reinstate the un-clamped counts (a rollback that forgot
    // truncate_index_pool_keys).
    let mut mutated = 0usize;
    #[allow(clippy::needless_range_loop)]
    // allow: il indexes BOTH cache.latent and the pre snapshot; zipping would hide the pairing
    for il in 0..h.model.layers.len() {
        if let (Some(plane), Some(ready)) = (cache.latent[il].as_mut(), pre[il])
            && ready > plane.index_pools_ready
        {
            plane.index_pools_ready = ready;
            mutated += 1;
        }
    }
    assert!(
        mutated > 0,
        "the walk+rollback did not move index_pools_ready on any trunk MLA layer — the \
         fixture never exercised pool finality and this red arm is vacuous"
    );

    let err = h
        .model
        .decode_step(&h.engine, tokens(1, 0xD00D)[0], &mut cache)
        .expect_err("continuing over keys finalized past j must fail LOUDLY");
    let msg = err.to_string();
    assert!(
        msg.contains("index_pools_ready"),
        "the failure must be the kpool residency tripwire, got: {msg}"
    );
    println!("gate 4 RED bites: {msg}");
}

// ---------------------------------------------------------------------------------------------
// Gate 5 — END TO END: spec-vs-plain greedy tape byte identity, K=1..7, incl. forced rounds.
// ---------------------------------------------------------------------------------------------

fn plain_tape(h: &Harness, prompt: &[u32], max_new: usize) -> Vec<u32> {
    let (mut cache, logits) = h.fresh_primed(prompt, prompt.len() + max_new + 16);
    let mut tape = Vec::with_capacity(max_new);
    tape.push(argmax(&logits) as u32);
    while tape.len() < max_new {
        let ll = h
            .model
            .decode_step(&h.engine, *tape.last().unwrap(), &mut cache)
            .expect("plain decode step");
        tape.push(argmax(&ll) as u32);
    }
    tape
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_spec_vs_plain_greedy_tape_byte_identity_k1_to_7() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let tape = plain_tape(&h, &prompt, max_new);

    // Natural drafter (the fixture head's own greedy drafts — mixed accepts/rejections).
    for k in 1..=K {
        let (out, drafted, accepted) = h
            .model
            .generate_spec_glm5(&h.engine, &prompt, max_new, k)
            .expect("spec generation");
        assert_eq!(
            out, tape,
            "K={k}: spec tape diverged from plain greedy (drafted {drafted}, accepted {accepted})"
        );
        println!(
            "gate 5 PASS K={k} natural drafter: tape identical, {accepted}/{drafted} accepted"
        );
    }

    // Forced FULL-ACCEPT rounds: the override feeds the plain tape itself as drafts, so
    // every round takes the keep==t (no-restore) path and every state plane still lands
    // byte-identical.
    for k in [3usize, K] {
        let tape_for_override = tape.clone();
        let mut over = move |round: usize, ki: usize, _greedy: u32| -> u32 {
            // With every draft forced correct, round r starts with 1 + r*(k+1) committed.
            let cursor = 1 + round * (k + 1);
            let pos = cursor + ki;
            if pos < tape_for_override.len() {
                tape_for_override[pos]
            } else {
                // Past the compare horizon: any token — the tape is truncated to max_new.
                0
            }
        };
        let (out, drafted, accepted) = h
            .model
            .generate_spec_glm5_gated(
                &h.engine,
                &prompt,
                max_new,
                k,
                Glm5SpecKnobs {
                    draft_override: Some(&mut over),
                    ..Default::default()
                },
            )
            .expect("forced-accept spec generation");
        assert_eq!(
            out, tape,
            "K={k}: forced-accept tape diverged from plain greedy"
        );
        assert!(
            accepted * 2 >= drafted,
            "K={k}: forced-correct drafts were mostly rejected ({accepted}/{drafted}) — the \
             override plumbing is broken and the full-accept path went unexercised"
        );
        println!("gate 5 PASS K={k} forced-accept: tape identical, {accepted}/{drafted} accepted");
    }

    // Forced-REJECTION positions cycling j across rounds: rounds commit j+1 tokens with
    // j = round % k, exercising every partial-keep rollback inside the served loop.
    {
        let k = K;
        let tape_for_override = tape.clone();
        let committed_before = move |round: usize| -> usize {
            let mut c = 1usize;
            for r in 0..round {
                c += (r % k) + 1;
            }
            c
        };
        let mut over = move |round: usize, ki: usize, _greedy: u32| -> u32 {
            let j_target = round % k;
            let cursor = committed_before(round);
            let pos = cursor + ki;
            let correct = if pos < tape_for_override.len() {
                tape_for_override[pos]
            } else {
                0
            };
            if ki < j_target {
                correct
            } else {
                (correct + 1) % VOCAB // guaranteed rejection at position j_target
            }
        };
        let (out, drafted, accepted) = h
            .model
            .generate_spec_glm5_gated(
                &h.engine,
                &prompt,
                max_new,
                k,
                Glm5SpecKnobs {
                    draft_override: Some(&mut over),
                    ..Default::default()
                },
            )
            .expect("forced-rejection spec generation");
        assert_eq!(
            out, tape,
            "K={k}: forced-rejection tape diverged from plain greedy (corrupted drafts must \
             yield IDENTICAL output to plain decode)"
        );
        println!(
            "gate 5 PASS K={k} forced-rejection sweep: tape identical, {accepted}/{drafted} accepted"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Gate 6 — RED: disabling rollback must break the end-to-end identity.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_disabled_rollback_breaks_the_tape() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let tape = plain_tape(&h, &prompt, max_new);

    // Force a rejection at draft 0 of every round: with rollback disabled the trunk keeps
    // post-row-K state on a j=0 accept, so the tape must diverge (or the kpool residency
    // tripwire fires — either way, NOT byte-identical-and-green).
    let mut over = |_round: usize, ki: usize, greedy: u32| -> u32 {
        if ki == 0 {
            (greedy + 1) % VOCAB
        } else {
            greedy
        }
    };
    let result = h.model.generate_spec_glm5_gated(
        &h.engine,
        &prompt,
        max_new,
        K,
        Glm5SpecKnobs {
            draft_override: Some(&mut over),
            disable_rollback: true,
            ..Default::default()
        },
    );
    match result {
        Ok((out, drafted, accepted)) => {
            assert_ne!(
                out, tape,
                "rollback disabled + forced rejections still produced the plain tape — the \
                 end-to-end gate cannot see the rollback at all ({accepted}/{drafted})"
            );
            println!("gate 6 RED bites: tape diverged with rollback disabled");
        }
        Err(err) => {
            println!("gate 6 RED bites: loop failed loudly with rollback disabled: {err}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Gate 7 — FR-SPEC TRIM (the house spec recipe, the q38 way): the draft head projects over
// gathered top-N rows and drafts remap through d2t to true vocab BEFORE verify; the verify
// walk stays full-vocab and untouched.
//
// Instrument: a FIXED-POINT-FREE full-vocab permutation ranks list (rotation by 1). The
// trimmed head then holds the SAME 32 rows permuted, so `logits_trim[r] ==
// logits_full[d2t[r]]` byte for byte and the REMAPPED draft sequence must be IDENTICAL to
// the untrimmed arm's — an unwired remap cannot pass (every position would differ), which
// is what makes this a wiring assertion on the invocation rather than the prose. Red arm:
// `skip_d2t_remap` drafts rank ids as vocab ids (the q38 0/248 skipped-remap defect, which
// no exactness gate catches because the tape stays correct) — the gate asserts the drafted
// sequence DIVERGES while the output tape STAYS byte-identical to plain decode. Partial
// (top-16) trim: a genuinely restricted draft vocabulary may change which tokens get
// drafted; the tape must still be byte-identical, and accept-j-then-continue must still
// hold with the trim loaded (the trim can never reach the verify/rollback planes).
//
// Acceptance-at-k on this fixture is a plumbing receipt only (untrained weights): the real
// acceptance A/B runs on the real artifact with owner-minted SXC ranks per traffic class
// (LAW:real-prompts-for-spec) — the lane's named input dependency.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_frspec_trim_equivalence_partial_and_skipped_remap_red() {
    let _gpu = gpu_guard();
    // Ranks artifacts in a scratch dir, removed before the test returns (tmp hygiene).
    let dir = std::env::temp_dir().join(format!("glm5-tv-ranks-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("ranks scratch dir");
    let rot: Vec<u32> = (0..VOCAB).map(|r| (r + 1) % VOCAB).collect();
    let join = |ids: &[u32]| {
        ids.iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    };
    let full_path = dir.join("rot-full.txt");
    std::fs::write(&full_path, join(&rot)).expect("full ranks file");
    let part_path = dir.join("rot-top16.txt");
    std::fs::write(&part_path, join(&rot[..16])).expect("partial ranks file");

    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;

    // One recorded run: the identity override taps every REMAPPED draft the loop produces.
    let run_recorded = |h: &Harness, skip_remap: bool| -> (Vec<u32>, usize, usize, Vec<u32>) {
        let mut recorded: Vec<u32> = Vec::new();
        let mut over = |_round: usize, _ki: usize, d: u32| -> u32 {
            recorded.push(d);
            d
        };
        let (out, drafted, accepted) = h
            .model
            .generate_spec_glm5_gated(
                &h.engine,
                &prompt,
                max_new,
                K,
                Glm5SpecKnobs {
                    draft_override: Some(&mut over),
                    skip_d2t_remap: skip_remap,
                    ..Default::default()
                },
            )
            .expect("spec generation");
        (out, drafted, accepted, recorded)
    };

    // ---- arm A: untrimmed reference (tape + the recorded draft sequence) ----
    let ha = Harness::new(true);
    let tape = plain_tape(&ha, &prompt, max_new);
    let (out_a, drafted_a, accepted_a, drafts_a) = run_recorded(&ha, false);
    assert_eq!(out_a, tape, "untrimmed spec tape must match plain greedy");
    assert!(
        ha.model.mtp.as_ref().unwrap().d2t.is_none(),
        "arm A must be untrimmed or the equivalence below is vacuous"
    );
    drop(ha);

    // ---- arm B: full-vocab permutation trim — remapped drafts IDENTICAL to arm A ----
    let hb = Harness::new_trimmed(full_path.to_str().unwrap());
    {
        let head = hb.model.mtp.as_ref().expect("MTP head loads");
        assert_eq!(
            head.d2t.as_deref(),
            Some(rot.as_slice()),
            "the loader must carry the ranks list as d2t"
        );
        assert!(
            head.shared_head_head.is_some(),
            "the trim must gather rows into the draft head"
        );
    }
    let (out_b, drafted_b, accepted_b, drafts_b) = run_recorded(&hb, false);
    assert_eq!(out_b, tape, "trimmed spec tape must match plain greedy");
    assert_eq!(
        drafts_b, drafts_a,
        "a full-vocab permutation trim holds the SAME rows permuted, so the remapped \
         draft sequence must be IDENTICAL to the untrimmed arm's — the d2t remap is not \
         wired to the draft argmax"
    );
    assert_eq!(
        (drafted_b, accepted_b),
        (drafted_a, accepted_a),
        "identical drafts must produce identical acceptance"
    );
    println!(
        "gate 7 PASS full-perm trim: {} drafts identical to untrimmed, tape identical, \
         acceptance {accepted_b}/{drafted_b}",
        drafts_b.len()
    );

    // ---- arm B accept-j spot check: the trim can never reach the verify/rollback planes ----
    {
        let vt = tokens(K + 1, 0xBEEF);
        let cc = tokens(12, 0xC0FFEE);
        let keep = 3usize; // j = 2
        let max_ctx = PROMPT + K + cc.len() + 16;
        let (mut cache_ref, _l) = hb.fresh_primed(&prompt, max_ctx);
        for &tok in &vt[..keep] {
            let _ = hb
                .model
                .decode_step(&hb.engine, tok, &mut cache_ref)
                .unwrap();
        }
        let mut ref_rows: Vec<Vec<f32>> = Vec::with_capacity(cc.len());
        for &tok in &cc {
            ref_rows.push(
                hb.model
                    .decode_step(&hb.engine, tok, &mut cache_ref)
                    .unwrap(),
            );
        }
        let (mut cache, _l) = hb.fresh_primed(&prompt, max_ctx);
        let (_vl, _coll, ckpt) = hb
            .model
            .glm5_verify_rows(&hb.engine, &vt, &mut cache)
            .expect("verify rows walk (trim loaded)");
        hb.model
            .glm5_verify_rollback(&hb.engine, &mut cache, &ckpt, keep)
            .expect("rollback (trim loaded)");
        let mut bad = 0usize;
        for (step, &tok) in cc.iter().enumerate() {
            let got = hb.model.decode_step(&hb.engine, tok, &mut cache).unwrap();
            bad += usize::from(bit_diffs(&got, &ref_rows[step]) > 0);
        }
        assert_eq!(
            bad, 0,
            "accept-j-then-continue must stay byte-identical with the trim loaded"
        );
        println!("gate 7 PASS trim-on accept-j spot check (j=2): 12 continue steps bit-identical");
    }

    // ---- arm B RED: remap skipped — rank ids drafted as vocab ids ----
    let (out_red, drafted_red, accepted_red, drafts_red) = run_recorded(&hb, true);
    assert_eq!(
        out_red, tape,
        "even rank-ids-drafted-as-vocab-ids must verify to the plain tape — the verify \
         walk arbitrates and a wrong draft can only be rejected, never served"
    );
    assert_ne!(
        drafts_red, drafts_a,
        "skipping the d2t remap produced the SAME draft sequence as the remapped arm — \
         the fixed-point-free permutation makes that impossible if the seam is wired, so \
         the gate cannot see the q38 skipped-remap defect"
    );
    let diverged = drafts_red
        .iter()
        .zip(&drafts_a)
        .filter(|(x, y)| x != y)
        .count();
    println!(
        "gate 7 RED bites: {diverged}/{} drafts diverge with the remap skipped \
         (acceptance {accepted_red}/{drafted_red} vs remapped {accepted_a}/{drafted_a}); \
         tape stayed identical — exactly the silent defect class this arm makes loud",
        drafts_a.len()
    );
    drop(hb);

    // ---- arm C: PARTIAL trim (top 16 of 32): restricted draft vocab, tape unchanged ----
    let hc = Harness::new_trimmed(part_path.to_str().unwrap());
    assert_eq!(
        hc.model.mtp.as_ref().unwrap().d2t.as_ref().unwrap().len(),
        16,
        "partial ranks list must trim the head to 16 rows"
    );
    let (out_c, drafted_c, accepted_c, _drafts_c) = run_recorded(&hc, false);
    assert_eq!(
        out_c, tape,
        "a restricted draft vocabulary may change WHICH tokens get drafted, never how \
         they verify — the tape must stay byte-identical to plain decode"
    );
    println!(
        "gate 7 PASS partial trim (16/32 rows): tape identical, acceptance \
         {accepted_c}/{drafted_c} (fixture plumbing receipt; the real acceptance-at-k A/B \
         is the real-artifact + SXC-ranks cell)"
    );

    std::fs::remove_dir_all(&dir).ok();
}
