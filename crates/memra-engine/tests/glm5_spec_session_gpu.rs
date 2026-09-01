//! glm5_next SERVED-PATH spec session gate (lane/glm5-spec-routing, 2026-08-30).
//!
//! The tparallel gate (`glm5_tparallel_verify_gpu.rs`) pins the ROUND machinery
//! (walk/rollback/e2e one-shot). This file pins the SERVING SHAPE on the same fixture
//! family: `glm5_spec_session_new` + `glm5_spec_session_burst` — the EXACT invocations the
//! worker's `step_glm5_spec` makes — driven in worker-sized bursts with state carried
//! ACROSS burst boundaries:
//!
//! 1. Served-burst greedy byte identity: the concatenated burst outputs are byte-identical
//!    to plain decode at K=1..7 (natural drafter) and under forced-rejection rounds
//!    cycling every partial-accept j — the partial-accept continuation crosses burst
//!    boundaries by construction (burst target 3 < K+1).
//! 2. Session-state consistency at every burst boundary: `pos() == committed.len()`,
//!    `committed` == prompt + served tape minus the live anchor, emission counts sum
//!    exactly (the anchor emitted once) — the worker's one-event-per-public-id receipt
//!    rests on these.
//! 3. Sampled twin (vendor-default shape is the product; greedy is the instrument):
//!    pinned-seed determinism, burst-split invariance (Philox counters persist on the
//!    session — the session-continuity law), seed sensitivity, and greedy-arm agreement
//!    of the plain tape.
//! 4. EOS mid-burst: the session finishes, later bursts are empty.
//! 5. RED: rollback disabled + forced rejections through the SESSION burst diverges (or
//!    fails loudly) — the tparallel red arm re-proven on the served path.
//! 6. RECEIPT LOG GATE (subprocess-captured stderr, red-proven): MEMRA_GLM5_SPEC=1 boots
//!    the `[glm5-spec] serve route ARMED` line (TRIMMED variant under MEMRA_FRSPEC_TRIM;
//!    fail-closed warn without the MTP head); flag OFF = ZERO `[glm5-spec]` lines.
//!
//! Rig law (exactness only, never timing):
//!   NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
//!     cargo test -p memra-engine --test glm5_spec_session_gpu -- --ignored --test-threads=1

use memra_engine::Engine;
use memra_engine::forward::argmax;
use memra_engine::glm_spec::{Glm5SpecKnobs, Glm5SpecSession};
use memra_engine::hybrid::HybridModel;
use memra_engine::spec::SpecSampling;
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

/// F32 weights + Q8_0 expert banks (the loader refuses float banks) — engine-vs-engine
/// identity needs one loadable set of numbers, not a reference roundtrip.
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
    /// Callers hold `gpu_guard`. `with_mtp` loads the NextN draft head.
    fn new(with_mtp: bool) -> Self {
        // No trim may leak in from the environment: this file's arms are all untrimmed.
        // SAFETY: serialized behind gpu_guard by every caller.
        unsafe { std::env::remove_var("MEMRA_FRSPEC_TRIM") };
        Self::load(with_mtp)
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

/// Drive a session in worker-sized bursts (the step_glm5_spec shape) until `total` tokens
/// are out or the session finishes, asserting the burst-boundary invariants the worker's
/// emission receipt rests on. Returns (tape, drafted, accepted, bursts).
fn drive_bursts(
    h: &Harness,
    sess: &mut Glm5SpecSession,
    prompt: &[u32],
    k: usize,
    total: usize,
    burst_target: usize,
    eos: &[u32],
) -> (Vec<u32>, usize, usize, usize) {
    let mut tape: Vec<u32> = Vec::new();
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    let mut bursts = 0usize;
    while tape.len() < total && !sess.finished() {
        let room = (total - tape.len()).min(burst_target);
        let (burst, d, a) = h
            .model
            .glm5_spec_session_burst(&h.engine, sess, room, k, eos)
            .expect("glm5 spec session burst");
        if burst.is_empty() {
            break;
        }
        bursts += 1;
        drafted += d;
        accepted += a;
        tape.extend(burst);
        // BURST-BOUNDARY INVARIANTS (gate 2): the cache rows are exactly the committed
        // tokens, and committed == prompt + tape minus the LIVE anchor (the last emitted
        // token, not yet consumed by the trunk).
        assert_eq!(
            sess.pos(),
            sess.committed.len(),
            "cache rows != committed tokens at a burst boundary"
        );
        let mut expect: Vec<u32> = prompt.to_vec();
        expect.extend_from_slice(&tape[..tape.len() - 1]);
        assert_eq!(
            sess.committed, expect,
            "committed must be prompt + served tape minus the live anchor"
        );
    }
    (tape, drafted, accepted, bursts)
}

// ---------------------------------------------------------------------------------------------
// Gates 1+2 — served-burst greedy byte identity + session-state consistency across bursts.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_served_bursts_greedy_tape_matches_plain_decode_k1_to_7() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let tape = plain_tape(&h, &prompt, max_new);

    for k in 1..=K {
        // Burst target 3: every round's partial accept crosses a burst boundary for k >= 3
        // — the worker's MEMRA_SPEC_BURST cadence in miniature.
        let mut sess = h
            .model
            .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + max_new + k + 8, None)
            .expect("glm5 spec session");
        let (out, drafted, accepted, bursts) =
            drive_bursts(&h, &mut sess, &prompt, k, max_new, 3, &[]);
        assert_eq!(
            &out[..max_new],
            &tape[..],
            "K={k}: served-burst tape diverged from plain greedy \
             ({accepted}/{drafted} over {bursts} bursts)"
        );
        assert!(
            bursts >= max_new / (k + 2),
            "K={k}: the drive never actually split into bursts ({bursts})"
        );
        println!(
            "gate 1 PASS K={k}: served bursts byte-identical over {bursts} bursts, \
             {accepted}/{drafted} accepted"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Gate 3 — forced-rejection rounds THROUGH the served burst API: every partial-accept j,
// with the continuation crossing burst boundaries.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_served_bursts_forced_rejection_partial_accepts_stay_byte_identical() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let tape = plain_tape(&h, &prompt, max_new);
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
    let mut knobs = Glm5SpecKnobs {
        draft_override: Some(&mut over),
        ..Default::default()
    };
    let mut sess = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + max_new + k + 8, None)
        .expect("glm5 spec session");
    let mut out: Vec<u32> = Vec::new();
    let mut bursts = 0usize;
    while out.len() < max_new && !sess.finished() {
        let room = (max_new - out.len()).min(2); // tiny bursts: every j continues across a boundary
        let (burst, _d, _a) = h
            .model
            .glm5_spec_session_burst_gated(&h.engine, &mut sess, room, k, &[], &mut knobs)
            .expect("forced-rejection served burst");
        if burst.is_empty() {
            break;
        }
        bursts += 1;
        out.extend(burst);
        assert_eq!(sess.pos(), sess.committed.len());
    }
    assert_eq!(
        &out[..max_new],
        &tape[..],
        "forced-rejection served bursts diverged from plain greedy (corrupted drafts must \
         yield IDENTICAL output)"
    );
    println!("gate 3 PASS: forced-rejection j-sweep byte-identical over {bursts} bursts");
}

// ---------------------------------------------------------------------------------------------
// Gate 4 — SAMPLED twin: pinned-seed determinism + burst-split invariance (the
// session-continuity Philox law) + seed sensitivity.
// ---------------------------------------------------------------------------------------------

fn sampled_cfg(seed: u64) -> SpecSampling {
    SpecSampling {
        temp: 0.9,
        seed,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        penalty_last_n: 0,
        penalty_repeat: 1.0,
        penalty_freq: 0.0,
        penalty_present: 0.0,
    }
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_sampled_twin_is_deterministic_and_burst_split_invariant() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 24usize;
    let k = 3usize;
    let ctx = prompt.len() + max_new + k + 8;

    let run = |seed: u64, burst_target: usize| -> Vec<u32> {
        let mut sess = h
            .model
            .glm5_spec_session_new(&h.engine, &prompt, ctx, Some(sampled_cfg(seed)))
            .expect("sampled glm5 spec session");
        let (tape, _d, _a, _b) =
            drive_bursts(&h, &mut sess, &prompt, k, max_new, burst_target, &[]);
        tape[..max_new.min(tape.len())].to_vec()
    };

    let a = run(42, 3);
    let b = run(42, 3);
    assert_eq!(
        a, b,
        "same seed, same burst split: the sampled tape must be reproducible"
    );
    let c = run(42, max_new);
    assert_eq!(
        a, c,
        "burst-split invariance: Philox counters persist ON THE SESSION, so 3-token bursts \
         and one whole-budget burst must draw the identical stream"
    );
    let d = run(43, 3);
    assert_ne!(
        a, d,
        "a different seed must change the sampled tape (a constant tape would mean the \
         sampling seam is dead and the arm is secretly greedy)"
    );
    println!("gate 4 PASS: sampled twin deterministic, burst-split invariant, seed-sensitive");
}

// ---------------------------------------------------------------------------------------------
// Gate 5 — EOS mid-burst: the session finishes; later bursts are empty.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_eos_finishes_the_session_and_later_bursts_are_empty() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let tape = plain_tape(&h, &prompt, max_new);
    // Declare the greedy tape's 7th token as EOS: the session must stop at that round.
    let eos = [tape[6]];
    let first_eos = tape.iter().position(|t| eos.contains(t)).unwrap();

    let k = 3usize;
    let mut sess = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + max_new + k + 8, None)
        .expect("glm5 spec session");
    let mut out: Vec<u32> = Vec::new();
    while out.len() < max_new && !sess.finished() {
        let (burst, _d, _a) = h
            .model
            .glm5_spec_session_burst(&h.engine, &mut sess, 3, k, &eos)
            .expect("burst");
        if burst.is_empty() {
            break;
        }
        out.extend(burst);
    }
    assert!(sess.finished(), "EOS must finish the session");
    let cut = out
        .iter()
        .position(|t| eos.contains(t))
        .expect("EOS emitted");
    assert_eq!(
        &out[..=cut],
        &tape[..=first_eos],
        "the public prefix through EOS must match plain greedy"
    );
    assert!(
        out.len() <= first_eos + k + 1,
        "post-EOS overshoot must stay within the final round (k+1 rows)"
    );
    let (again, d2, a2) = h
        .model
        .glm5_spec_session_burst(&h.engine, &mut sess, 8, k, &eos)
        .expect("post-EOS burst");
    assert!(
        again.is_empty() && d2 == 0 && a2 == 0,
        "a finished session must emit nothing"
    );
    println!("gate 5 PASS: EOS at pos {first_eos} finished the session; post-EOS burst empty");
}

// ---------------------------------------------------------------------------------------------
// Gate 6 — RED: rollback disabled + forced rejections THROUGH THE SESSION BURST must not
// stay byte-identical-and-green (the tparallel red arm, re-proven on the served path).
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_served_burst_with_rollback_disabled_bites() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let tape = plain_tape(&h, &prompt, max_new);

    let mut over = |_round: usize, ki: usize, greedy: u32| -> u32 {
        if ki == 0 {
            (greedy + 1) % VOCAB
        } else {
            greedy
        }
    };
    let mut knobs = Glm5SpecKnobs {
        draft_override: Some(&mut over),
        disable_rollback: true,
        ..Default::default()
    };
    let mut sess = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + max_new + K + 8, None)
        .expect("glm5 spec session");
    let mut out: Vec<u32> = Vec::new();
    let mut failed: Option<String> = None;
    while out.len() < max_new && !sess.finished() {
        match h
            .model
            .glm5_spec_session_burst_gated(&h.engine, &mut sess, 3, K, &[], &mut knobs)
        {
            Ok((burst, _d, _a)) => {
                if burst.is_empty() {
                    break;
                }
                out.extend(burst);
            }
            Err(err) => {
                failed = Some(err.to_string());
                break;
            }
        }
    }
    match failed {
        Some(err) => println!("gate 6 RED bites: served burst failed loudly: {err}"),
        None => {
            assert_ne!(
                &out[..max_new.min(out.len())],
                &tape[..max_new.min(out.len())],
                "rollback disabled + forced rejections still produced the plain tape through \
                 the served burst — the red arm went blind"
            );
            println!("gate 6 RED bites: served tape diverged with rollback disabled");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Gate 7 — RECEIPT LOG GATE (captured stderr, red-proven): the boot lines a deploy gate
// greps. Runs the receipt helper below in a CHILD PROCESS per arm so each arm reads its
// own MEMRA_GLM5_SPEC / MEMRA_GLM5_MTP environment fresh (both are OnceLock'd per process)
// and its stderr is a real captured log, not an in-process assumption.
// ---------------------------------------------------------------------------------------------

/// Child body for gate 7: load the fixture per the ambient env, then run one served burst
/// when both flags arm. Asserts nothing itself — the parent asserts on the captured log.
#[test]
#[ignore = "receipt-gate child body; spawned by gpu_receipt_log_lines_red_and_green"]
fn helper_emit_glm5_receipts() {
    let _gpu = gpu_guard();
    force_true_f32();
    let with_mtp = std::env::var("MEMRA_GLM5_MTP").as_deref() == Ok("1");
    let config = mini_config();
    let plan = mini_plan(&config);
    let source = fixture_source(&config, &plan);
    let engine = Engine::new(0).expect("CUDA engine on device 0");
    let model = if with_mtp {
        HybridModel::load_from_source(&engine, &source).expect("mini glm5 loads (with MTP)")
    } else {
        HybridModel::load_from_source_without_mtp(&engine, &source).expect("mini glm5 loads")
    };
    if with_mtp && memra_engine::glm_spec::glm5_spec_on() {
        let prompt = tokens(PROMPT, 0xA11CE);
        let mut sess = model
            .glm5_spec_session_new(&engine, &prompt, prompt.len() + 40, None)
            .expect("glm5 spec session");
        let (burst, d, a) = model
            .glm5_spec_session_burst(&engine, &mut sess, 8, 3, &[])
            .expect("burst");
        eprintln!("[helper] burst={} drafted={d} accepted={a}", burst.len());
    }
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_receipt_log_lines_red_and_green() {
    let _gpu = gpu_guard();
    let run_child =
        |glm5_spec: Option<&str>, glm5_mtp: Option<&str>, trim: Option<&str>| -> String {
            let exe = std::env::current_exe().expect("test binary path");
            let mut cmd = std::process::Command::new(exe);
            cmd.args([
                "helper_emit_glm5_receipts",
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ]);
            cmd.env_remove("MEMRA_GLM5_SPEC");
            cmd.env_remove("MEMRA_GLM5_MTP");
            cmd.env_remove("MEMRA_FRSPEC_TRIM");
            cmd.env("NVIDIA_TF32_OVERRIDE", "0");
            if let Some(v) = glm5_spec {
                cmd.env("MEMRA_GLM5_SPEC", v);
            }
            if let Some(v) = glm5_mtp {
                cmd.env("MEMRA_GLM5_MTP", v);
            }
            if let Some(v) = trim {
                cmd.env("MEMRA_FRSPEC_TRIM", v);
            }
            let out = cmd.output().expect("spawn receipt child");
            assert!(
                out.status.success(),
                "receipt child failed (spec={glm5_spec:?} mtp={glm5_mtp:?} trim={trim:?}):\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stderr).into_owned()
        };

    // GREEN: both flags — the ARMED boot line, and the engine burst actually ran.
    let log = run_child(Some("1"), Some("1"), None);
    assert!(
        log.contains("[glm5-spec] serve route ARMED: MTP head loaded; draft head FULL"),
        "armed boot receipt missing from the captured log:\n{log}"
    );
    assert!(
        log.contains("[helper] burst="),
        "armed child never burst:\n{log}"
    );

    // GREEN, TRIMMED: an FR-Spec ranks artifact (fixed-point-free rotation over the full
    // 32-token vocab, the tparallel gate-7 instrument) must flip the boot line to the
    // TRIMMED variant — the line a trim-armed deploy gate greps.
    let ranks_path = std::env::temp_dir().join(format!(
        "glm5-spec-receipt-ranks-{}.txt",
        std::process::id()
    ));
    let ranks: String = (0..VOCAB)
        .map(|r| format!("{}\n", (r + 1) % VOCAB))
        .collect();
    std::fs::write(&ranks_path, ranks).expect("write ranks fixture");
    let log = run_child(Some("1"), Some("1"), Some(ranks_path.to_str().unwrap()));
    std::fs::remove_file(&ranks_path).ok(); // tmp hygiene: the task that made it deletes it
    assert!(
        log.contains(&format!(
            "[glm5-spec] serve route ARMED: MTP head loaded; draft head TRIMMED to {VOCAB} rows"
        )),
        "trimmed boot receipt missing from the captured log:\n{log}"
    );

    // FAIL-CLOSED WARN: flag on, head not loaded — the loud misconfiguration line.
    let log = run_child(Some("1"), None, None);
    assert!(
        log.contains("[glm5-spec] MEMRA_GLM5_SPEC=1 but no MTP head loaded"),
        "fail-closed warn missing:\n{log}"
    );

    // RED ARM (the receipt gate's whole point): flag OFF must show ZERO [glm5-spec] lines
    // — a deploy gate grepping the log can trust absence, with the head loaded or not.
    for mtp in [None, Some("1")] {
        let log = run_child(None, mtp, None);
        assert!(
            !log.contains("[glm5-spec]"),
            "MEMRA_GLM5_SPEC off (mtp={mtp:?}) must print no [glm5-spec] line:\n{log}"
        );
    }
    println!(
        "gate 7 PASS: boot receipts green (full + trimmed) + fail-closed warn + flag-off red \
         (zero lines)"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 8 — CONFIDENCE GATE (lane/glm5-loop-port, port 2): MEMRA_SPEC_PMIN/PMIN0-class chain
// truncation moves DRAFT COUNTS, never the tape. Decisive arms: p_min = 1.1 can never be
// cleared (a softmax confidence is <= 1.0), so with PMIN0 EVERY round drafts ZERO (each
// round degenerates to a plain step) and without PMIN0 exactly the slot-0 survivor drafts —
// neither can pass by accident, and both must leave the greedy tape byte-identical to plain
// decode. The env pair latches OnceLock-once per process, so the arms drive through the
// knobs override (`pmin_override`), the gate instrument built for exactly this.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_confidence_gate_truncates_drafts_never_the_tape() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let k = 3usize;
    let ctx = prompt.len() + max_new + k + 8;
    let tape = plain_tape(&h, &prompt, max_new);

    let drive =
        |pmin: Option<(f32, bool)>, sampling: Option<SpecSampling>| -> (Vec<u32>, usize, usize) {
            let mut sess = h
                .model
                .glm5_spec_session_new(&h.engine, &prompt, ctx, sampling)
                .expect("glm5 spec session");
            let mut knobs = Glm5SpecKnobs {
                pmin_override: pmin,
                ..Default::default()
            };
            let mut out: Vec<u32> = Vec::new();
            let (mut drafted, mut accepted) = (0usize, 0usize);
            while out.len() < max_new && !sess.finished() {
                let room = (max_new - out.len()).min(3);
                let (burst, d, a) = h
                    .model
                    .glm5_spec_session_burst_gated(&h.engine, &mut sess, room, k, &[], &mut knobs)
                    .expect("gated burst");
                if burst.is_empty() {
                    break;
                }
                drafted += d;
                accepted += a;
                out.extend(burst);
            }
            (out, drafted, accepted)
        };

    // Gate-off reference: byte-identical, drafting engaged.
    let (out_off, drafted_off, _) = drive(None, None);
    assert_eq!(&out_off[..max_new], &tape[..], "gate-off arm diverged");
    assert!(
        drafted_off > 0,
        "gate-off arm drafted nothing — fixture defect"
    );

    // PMIN0 zero-draft arm: every round is an m=1 plain step; the tape must not move.
    let (out, drafted, accepted) = drive(Some((1.1, true)), None);
    assert_eq!(
        &out[..max_new],
        &tape[..],
        "PMIN0 zero-draft rounds must stay byte-identical (each round IS a plain step)"
    );
    assert_eq!(
        (drafted, accepted),
        (0, 0),
        "p_min=1.1 + PMIN0 must truncate EVERY chain to zero drafts"
    );

    // Slot-0 survivor arm (the spec.rs break semantics: j==0 survives without PMIN0).
    let (out, drafted, _) = drive(Some((1.1, false)), None);
    assert_eq!(&out[..max_new], &tape[..], "slot-0 survivor arm diverged");
    assert!(
        drafted > 0 && drafted < drafted_off,
        "without PMIN0 exactly slot 0 drafts per round: got {drafted} vs gate-off {drafted_off}"
    );

    // SAMPLED zero-draft twin: every round takes the shared full-accept bonus draw
    // (`glm5_sampled_bonus`) through the session's Philox stream — pinned seed must
    // reproduce, and the session must not stall.
    let (sa, da, _) = drive(Some((1.1, true)), Some(sampled_cfg(42)));
    let (sb, db, _) = drive(Some((1.1, true)), Some(sampled_cfg(42)));
    assert_eq!(sa, sb, "sampled zero-draft rounds must be deterministic");
    assert_eq!(
        (da, db),
        (0, 0),
        "sampled arms must also draft zero at p_min=1.1"
    );
    assert!(
        sa.len() >= max_new,
        "sampled zero-draft session stalled at {} of {max_new}",
        sa.len()
    );

    println!(
        "gate 8 PASS: confidence gate truncates drafts (0 with PMIN0, {drafted} slot-0 \
         survivors vs {drafted_off} gate-off), tape byte-identical on every arm, sampled \
         zero-draft twin deterministic"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 9 — DEMOTION HANDOFF (lane/glm5-loop-port, map #8): a mid-stream demote hands
// (cache, next_pred) whose continuation on the PLAIN batched program is byte-identical to
// the never-demoted plain tape. The live anchor is the carried-pending shape: the flush
// commits it through one plain decode step and next_pred is that step's argmax — so the
// handed-over stream must splice seamlessly: emitted-so-far + next_pred + plain chain ==
// the plain tape, token for token. Sampled sessions must refuse.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_demote_handoff_continues_byte_identical() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 24usize;
    let k = 3usize;
    let ctx = prompt.len() + max_new + k + 8;
    let tape = plain_tape(&h, &prompt, max_new);

    // Drive a spec session part-way (worker-sized bursts), then demote.
    let mut sess = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, ctx, None)
        .expect("glm5 spec session");
    let (head, _d, _a, bursts) = drive_bursts(&h, &mut sess, &prompt, k, 8, 3, &[]);
    assert!(
        bursts >= 2,
        "the demote must land mid-stream, not at turn 1"
    );
    assert_eq!(
        &head[..],
        &tape[..head.len()],
        "pre-demote spec tape diverged — nothing downstream is attributable"
    );
    assert!(sess.demote_eligible(), "greedy session must be eligible");

    let emitted = head.len();
    let (mut cache, next) = h
        .model
        .glm5_spec_into_demoted(&h.engine, sess)
        .expect("demotion handoff");
    // The flush committed the live anchor: the cache now holds exactly the emitted stream.
    assert_eq!(
        cache.pos,
        prompt.len() + emitted,
        "handed-over cache rows != prompt + emitted tokens (the flush contract)"
    );
    // next_pred is the NEXT plain token (device_next semantics: emitted then fed).
    assert_eq!(
        next, tape[emitted],
        "handoff next_pred != the plain tape's next token"
    );
    // Continue on the plain program from the handed-over cache.
    let mut out = head.clone();
    out.push(next);
    let mut cur = next;
    while out.len() < max_new {
        let ll = h
            .model
            .decode_step(&h.engine, cur, &mut cache)
            .expect("post-demote plain decode step");
        cur = argmax(&ll) as u32;
        out.push(cur);
    }
    assert_eq!(
        out, tape,
        "post-demote plain continuation diverged from the never-demoted tape"
    );

    // Sampled sessions refuse the handoff (session-owned Philox vs the worker sampler).
    let mut sampled = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, ctx, Some(sampled_cfg(42)))
        .expect("sampled session");
    let _ = drive_bursts(&h, &mut sampled, &prompt, k, 4, 3, &[]);
    assert!(!sampled.demote_eligible(), "sampled must not be eligible");
    let err = match h.model.glm5_spec_into_demoted(&h.engine, sampled) {
        Ok(_) => panic!("sampled demote must refuse loudly"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("sampled"),
        "the refusal must name the sampled exclusion, got: {err}"
    );

    println!(
        "gate 9 PASS: demote handoff spliced byte-identically at token {emitted} \
         ({bursts} bursts before), sampled refused by name"
    );
}

// ---------------------------------------------------------------------------------------------
// Door W (lane/glm5-matvec, MEMRA_GLM5_VERIFY_WS) — the verify-walk workspace: served-burst
// byte identity ON vs OFF, the SCRATCH_ALLOC_CALLS receipt, and the pool-hit engagement
// anchor. The compare instrument's bite is proven by gate 5's rollback-disabled red in this
// file (same tape compare, same fixture family); the door's failure class (aliased or
// short-written reuse) lands exactly there — as a tape divergence.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_verify_ws_bursts_byte_identical_with_alloc_receipt() {
    let _gpu = gpu_guard();
    let h = Harness::new(true);
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 24usize;
    let k = 4usize; // t = K+1 = 5 verify rows: the multi-row walk every round
    let ctx = prompt.len() + max_new + k + 8;
    let alloc_calls =
        || memra_engine::SCRATCH_ALLOC_CALLS.load(std::sync::atomic::Ordering::Relaxed);

    // OFF arm — pinned =0 (the flag is DEFAULT ON since 2026-08-31; unset now means ON,
    // and an unset arm would silently run the workspace twice — the MLA-TC flip lesson).
    // SAFETY: serialized behind gpu_guard.
    unsafe { std::env::set_var("MEMRA_GLM5_VERIFY_WS", "0") };
    let hits0 = memra_engine::verify_ws_hits();
    let a0 = alloc_calls();
    let mut off_sess = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, ctx, None)
        .expect("off-arm session");
    let (off_tape, off_drafted, ..) = drive_bursts(&h, &mut off_sess, &prompt, k, max_new, 5, &[]);
    let allocs_off = alloc_calls() - a0;
    assert_eq!(
        memra_engine::verify_ws_hits(),
        hits0,
        "flag-off arm drew from the verify workspace pool"
    );
    assert!(off_drafted > 0, "off arm drafted nothing — fixture defect");

    // ON arm — same prompt, fresh session, the workspace armed.
    // SAFETY: serialized behind gpu_guard.
    unsafe { std::env::set_var("MEMRA_GLM5_VERIFY_WS", "1") };
    let a1 = alloc_calls();
    let mut on_sess = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, ctx, None)
        .expect("on-arm session");
    let (on_tape, ..) = drive_bursts(&h, &mut on_sess, &prompt, k, max_new, 5, &[]);
    let allocs_on = alloc_calls() - a1;
    let hits = memra_engine::verify_ws_hits() - hits0;
    // SAFETY: serialized behind gpu_guard.
    unsafe { std::env::remove_var("MEMRA_GLM5_VERIFY_WS") };

    assert_eq!(
        on_tape, off_tape,
        "verify-ws arm diverged from the shipped program's tape"
    );
    assert!(
        hits > 0,
        "verify-ws arm never hit the pool — the door is not wired to the walk"
    );
    assert!(
        allocs_on < allocs_off,
        "verify-ws arm did not reduce SCRATCH_ALLOC_CALLS ({allocs_on} vs {allocs_off})"
    );
    println!(
        "door W PASS: {} tokens byte-identical; alloc calls {} -> {} (-{:.1}%), {} pool hits",
        off_tape.len(),
        allocs_off,
        allocs_on,
        100.0 * (allocs_off - allocs_on) as f64 / allocs_off as f64,
        hits
    );
}
