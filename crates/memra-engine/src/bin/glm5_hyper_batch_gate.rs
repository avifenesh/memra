//! glm5_next mHC BATCHED-DECODE gate: for B sessions decoding concurrently through
//! `decode_step_batch` (the `[B, streams, n_embd]` hyper walk), each session's logits must
//! be BIT-IDENTICAL to that session decoding ALONE through the serial hyper walk
//! (`decode_step` -> `decode_step_hyper`), at every step, on the same fixture family the
//! ppn gate uses.
//!
//! WHY THIS GATE EXISTS. Every batched entry point in decode_batch.rs refused the
//! HyperConnections residual, so GLM-5.3-Flash served SINGLE-STREAM ONLY at any
//! `MEMRA_MAX_SESSIONS`. Lifting that refusal is what this gate holds down, and the failure
//! mode it is built around is CROSS-SESSION CONTAMINATION: a batch walk that routes one
//! session's stream-state row or cache slot into another session's math corrupts customer
//! outputs silently and fluently. The banked red arms break exactly that seam (a swapped
//! h-row, a wrong cache slot); weight corruption is NOT a useful mutation here (it breaks
//! both arms equally).
//!
//! THE COMPARE IS FULL-LOGIT, NOT GREEDY-TAPE. The PP lane's M3 mutation proved the tape
//! can match while 32/32 logits differ; the tape is printed as a separate, weaker receipt.
//!
//! WHAT THE COMPARISON PROVES, AND HOW THE TRUTH CHAIN CLOSES (GATE:pin-against-truth):
//! batched-vs-serial is ARM-EQUALITY. The chain closes by COMPOSITION and every half must
//! be cited together:
//!
//! * `tests/hyper_connections_gpu.rs` anchors the serial hc walk to `memra_reference`
//!   (a host executor sharing no GPU code) on this fixture family;
//! * `glm5-hyper-ppn-gate` anchors the SPLIT serial walk to the unsplit one;
//! * this gate anchors the BATCHED walk to the serial one, per session, bit for bit.
//!
//! ARMS:
//!
//! 1. STAGGERED BATCH — B sessions with DIFFERENT token streams advanced to DIFFERENT
//!    depths (distinct positions, distinct KDA recurrent states, distinct kpool index
//!    plane fills), then N concurrent `decode_step_batch` ticks. Row b of every tick is
//!    bit-compared against session b's isolated serial tape.
//! 2. B=1 CLASS PIN — one session through `decode_step_batch` at B=1 vs its serial walk:
//!    the batched body must be ONE numeric class at every live width (the step35/Q35
//!    class-crossing law), so width-1 must not take a different program.
//! 3. DEVICE-SAMPLE GREEDY — the same staggered batch through `decode_step_batch_sampled`
//!    with greedy DevSamp rows: returned logits rows stay bit-identical and every device
//!    token equals the reference argmax (the serving epilogue's device-argmax contract).
//!
//! NOT COVERED, stated rather than implied: the lean logits park and grammar masks ride
//! `decode_batch_epilogue`, which is the SAME tail every proven batched arm serves — this
//! gate pins the hyper TRUNK, not the shared epilogue. The dual-wave and pending entries
//! REFUSE the hyper trunk by name (decode_batch.rs) and are not exercised here. Fused MoE
//! epilogue (`MEMRA_MOE_FUSED_EPI`) stays at its default (OFF) in this gate's arms.
//!
//! Knobs: `stages` (arg 4) opens the ppN door FOR THE WHOLE PROCESS — reference and batched
//! arms BOTH run split, isolating the batching axis; the split-vs-unsplit axis is
//! `glm5-hyper-ppn-gate`'s job. Other MEMRA_PP_* knobs pass through and are printed.
//!
//! Rig law: exactness only, never a timing number. Run under `flock /tmp/memra-5090.lock`
//! with `NVIDIA_TF32_OVERRIDE=0`.
//!
//! usage: glm5-hyper-batch-gate [B=3] [P=5] [N=8] [stages=1]
use memra_engine::Engine;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::{ModelPlan, StatePlan};
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
const LAYERS: usize = 4;

/// glm5_next's real shape, shrunk only in width — IDENTICAL to `glm5_hyper_ppn_gate`'s
/// fixture so the two gates' receipts compose over one artifact family: 4 mHC streams,
/// mean collapse, sigmoid noaux_tc router, PRE-clamped SwiGLU, KDA + DSA(MLA+kpool)
/// alternating, dense layer 0 then sparse.
fn mini_config_json() -> String {
    r#"{
      "model_type": "glm5_next_text",
      "num_hidden_layers": 4,
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

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
    ggml_type: GgmlType,
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
    .expect("contract for the mini glm5_next hc plan");
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

/// Deterministic token stream, seeded per session so no two sessions share a prompt.
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

fn tape_hash(tape: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in tape {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

struct ArmCheck {
    name: String,
    bad_steps: usize,
    checked_steps: usize,
    /// Device-token mismatches (arm 3 only). Tracked SEPARATELY from `bad_steps` so the
    /// verdict cannot print "38/24 comparisons mismatched" — the ppn gate's mutation runs
    /// caught exactly this class of self-arithmetic bug, and this gate's own M1 run
    /// re-caught it here before the field existed.
    token_bad: usize,
    tape_bad: bool,
    tape_checked: bool,
    first: Option<(usize, usize, f32, f32)>, // (step, idx, ref, got)
}

impl ArmCheck {
    fn new(name: impl Into<String>) -> Self {
        ArmCheck {
            name: name.into(),
            bad_steps: 0,
            checked_steps: 0,
            token_bad: 0,
            tape_bad: false,
            tape_checked: false,
            first: None,
        }
    }
    fn check(&mut self, step: usize, phase: &str, got: &[f32], r: &[f32]) {
        self.checked_steps += 1;
        assert_eq!(
            got.len(),
            r.len(),
            "[{}] step {step} ({phase}): logit row length {} != reference {}",
            self.name,
            got.len(),
            r.len()
        );
        let diffs = got
            .iter()
            .zip(r.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        if diffs > 0 {
            self.bad_steps += 1;
            let (idx, (a, b)) = got
                .iter()
                .zip(r.iter())
                .enumerate()
                .find(|(_, (a, b))| a.to_bits() != b.to_bits())
                .map(|(i, (a, b))| (i, (*b, *a)))
                .unwrap();
            if self.first.is_none() {
                self.first = Some((step, idx, a, b));
            }
            if self.bad_steps <= 5 {
                println!(
                    "[{}] MISMATCH step {step} ({phase}): {diffs}/{} logits differ, first \
                     @[{idx}] ref={a:?} batch={b:?}",
                    self.name,
                    r.len()
                );
            }
        }
    }
    fn check_tape(&mut self, got: &[u32], want: &[u32]) {
        self.tape_checked = true;
        let hg = tape_hash(got);
        let hw = tape_hash(want);
        if got == want {
            println!(
                "[{}] greedy tape MATCH: {} tokens, fnv1a={hg:#018x}",
                self.name,
                got.len()
            );
        } else {
            self.tape_bad = true;
            let at = got
                .iter()
                .zip(want)
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| got.len().min(want.len()));
            println!(
                "[{}] greedy tape DIVERGED at index {at}: ref={hw:#018x} batch={hg:#018x}",
                self.name
            );
        }
    }
}

/// One session's fixture shape: its token stream and its prefix depth.
struct SessionPlan {
    ids: Vec<u32>,
    prefix: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let b_n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let p: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let n: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let stages: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    assert!(
        (2..=64).contains(&b_n),
        "B={b_n}: the staggered arm needs 2..=64 sessions (B=1 has its own class-pin arm). \
         Widths past `hyper_batch_cap()` (15, the shexp PRIME_MIN_T knee) are EXPECTED to \
         stop on the engine's named refusal — that run is the over-cap receipt, and forcing \
         the cap up is the knee probe (a banked temporary edit, like the mutations)."
    );

    // cuBLASLt f32 rides TF32 on Blackwell by default — wrong for an exactness gate. Must
    // precede the first Engine::new in the process.
    if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
        // SAFETY: single-threaded, before any CUDA call in this process.
        unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
    }
    // The gate exercises the opt-in arm DELIBERATELY (the engine body fail-closes without
    // it). Set before the OnceLock's first read.
    // SAFETY: single-threaded, before any engine exists.
    unsafe { std::env::set_var("MEMRA_HYPER_BATCH", "1") };
    // DOOR-H ALIAS HYGIENE (lane/glm5-extract2). This gate's composed arms are driven from
    // the SHELL with `MEMRA_GLM5_HTOD_DIET=1` (the matrix runners' `compose-*-doors-EDH`
    // cells), and they assert BIT-IDENTITY — which passes whether the door armed or not. So a
    // leaked `MEMRA_HTOD_DIET=0` in the runner's environment would DISAGREE with the alias,
    // fall the door closed, and make those ON arms silently vacuous. The alias the caller set
    // is left alone; only the general name is cleared, so it cannot outvote them.
    // SAFETY: single-threaded, before any engine or runtime exists.
    unsafe { std::env::remove_var("MEMRA_HTOD_DIET") };
    if stages > 1 {
        // Door on for the WHOLE process: reference AND batched arms both run split, so the
        // only axis under test is batching. Split-vs-unsplit is glm5-hyper-ppn-gate's job.
        // SAFETY: single-threaded, before any engine or runtime exists.
        unsafe { std::env::set_var("MEMRA_PP_STAGES", stages.to_string()) };
    }

    let knobs = format!(
        "B={b_n} P={p}(staggered +bi) N={n} stages={stages} streams={} shard={}",
        if memra_engine::pp::pp2_streams_off() {
            "OFF(same-stream seam)"
        } else {
            "per-stage"
        },
        if memra_engine::pp::pp_shard_off() {
            "OFF(bring-up placement)"
        } else {
            "per-stage"
        },
    );
    println!("glm5-hyper-batch-gate config: {knobs}");

    let config = ModelConfig::from_hf(&HfConfig::parse(&mini_config_json()));
    let plan = memra_gguf::model_packs::for_config(&config)
        .expect("glm5_next model pack matches the mini config")
        .compile_plan(&config)
        .expect("mini glm5_next plan compiles");
    assert_eq!(plan.layers.len(), LAYERS);
    assert_eq!(plan.hidden_size as usize, HIDDEN);
    let fixture = deterministic_fixture(&plan).expect("deterministic glm5_next hc fixture");
    let source = fixture_source(&config, &plan, &fixture.weights);

    let e = Engine::new(0)?;
    let m = HybridModel::load_from_source_without_mtp(&e, &source)?;

    // ---- NON-VACUITY ----
    let topology = m.hyper.as_ref().expect(
        "the fixture must load as a HyperConnections trunk — otherwise this gate measures \
         the generic batched body that decode-batch-gate already covers",
    );
    println!(
        "hc topology: streams={} collapse={:?} sinkhorn_iters={}",
        topology.streams, topology.collapse, topology.sinkhorn_iterations
    );
    // Both per-layer state classes must be present, or a cache-slot mutation could pass on
    // the class the batch never routes.
    let mut has_recur = false;
    let mut has_latent = false;
    for layer in &plan.layers {
        match layer.state {
            StatePlan::Recurrent { .. } => has_recur = true,
            StatePlan::LatentKvCache { .. } => has_latent = true,
            _ => {}
        }
    }
    assert!(
        has_recur && has_latent,
        "the fixture must carry BOTH a Recurrent (KDA) and a LatentKvCache (MLA+kpool) \
         layer; a single-class fixture cannot see a per-class routing bug"
    );

    // Sessions: distinct streams, distinct depths (prefix = P + bi), so recurrent states,
    // latent lengths and kpool plane fills all differ across rows.
    let sessions: Vec<SessionPlan> = (0..b_n)
        .map(|bi| {
            let prefix = p + bi;
            SessionPlan {
                ids: tokens(prefix + n, 0xBA7C_4ED0 + bi as u64),
                prefix,
            }
        })
        .collect();
    let max_ctx = p + b_n + n + 8;

    let new_cache = |e: &Engine| -> Result<memra_engine::cache::Cache, Box<dyn std::error::Error>> {
        if stages > 1 {
            memra_engine::pp::new_cache_planned(e, &m.cfg, &plan, max_ctx)
        } else {
            memra_engine::cache::Cache::new_planned(e, &m.cfg, &plan, max_ctx)
        }
    };

    // ================= reference: each session ALONE through the serial hyper walk =========
    eprintln!("[phase] reference: per-session serial decode (isolated tapes)");
    let mut ref_logits: Vec<Vec<Vec<f32>>> = Vec::with_capacity(b_n); // [bi][step][vocab]
    let mut ref_tapes: Vec<Vec<u32>> = Vec::with_capacity(b_n);
    for s in &sessions {
        let mut cache = new_cache(&e)?;
        let mut logits_steps = Vec::with_capacity(s.ids.len());
        let mut tape = Vec::with_capacity(s.ids.len());
        for &tok in &s.ids {
            let ll = m.decode_step(&e, tok, &mut cache)?;
            tape.push(argmax(&ll) as u32);
            logits_steps.push(ll);
        }
        ref_logits.push(logits_steps);
        ref_tapes.push(tape);
    }
    let n_vocab = ref_logits[0][0].len();

    // ================= arm 1: STAGGERED BATCH =================
    eprintln!("[phase] arm 1: staggered batch — {b_n} concurrent sessions, {n} ticks");
    let mut batch_arm = ArmCheck::new("staggered-batch");
    {
        let mut caches: Vec<memra_engine::cache::Cache> = Vec::with_capacity(b_n);
        for (bi, s) in sessions.iter().enumerate() {
            let mut cache = new_cache(&e)?;
            // Advance the prefix on the SERIAL walk (the served prime/decode path), checking
            // the prefix logits too: a drifted prefix would poison every batched compare.
            for (step, &tok) in s.ids[..s.prefix].iter().enumerate() {
                let ll = m.decode_step(&e, tok, &mut cache)?;
                batch_arm.check(step, &format!("s{bi} prefix"), &ll, &ref_logits[bi][step]);
            }
            caches.push(cache);
        }
        // Distinct positions really are distinct (the shape this gate exists to pin).
        let depths: std::collections::BTreeSet<usize> = caches.iter().map(|c| c.pos).collect();
        assert_eq!(
            depths.len(),
            b_n,
            "sessions must sit at pairwise-distinct positions; got {depths:?}"
        );
        let mut tapes: Vec<Vec<u32>> = vec![Vec::with_capacity(n); b_n];
        for k in 0..n {
            let toks: Vec<u32> = sessions.iter().map(|s| s.ids[s.prefix + k]).collect();
            let mut cache_refs: Vec<&mut memra_engine::cache::Cache> = caches.iter_mut().collect();
            let rows = m.decode_step_batch(&e, &toks, &mut cache_refs)?;
            for (bi, row) in rows.iter().enumerate() {
                let step = sessions[bi].prefix + k;
                tapes[bi].push(argmax(row) as u32);
                batch_arm.check(
                    step,
                    &format!("s{bi} batched tick {k}"),
                    row,
                    &ref_logits[bi][step],
                );
            }
        }
        for (bi, s) in sessions.iter().enumerate() {
            batch_arm.check_tape(&tapes[bi], &ref_tapes[bi][s.prefix..s.prefix + n]);
        }
    }

    // ================= arm 2: B=1 CLASS PIN =================
    eprintln!("[phase] arm 2: B=1 through the batched body (one numeric class per width law)");
    let mut b1_arm = ArmCheck::new("b1-class-pin");
    {
        let s = &sessions[0];
        let mut cache = new_cache(&e)?;
        let mut tape = Vec::with_capacity(s.ids.len());
        for (step, &tok) in s.ids.iter().enumerate() {
            let mut cache_refs: Vec<&mut memra_engine::cache::Cache> = vec![&mut cache];
            let rows = m.decode_step_batch(&e, &[tok], &mut cache_refs)?;
            tape.push(argmax(&rows[0]) as u32);
            b1_arm.check(step, "B=1 batched", &rows[0], &ref_logits[0][step]);
        }
        b1_arm.check_tape(&tape, &ref_tapes[0]);
    }

    // ================= arm 3: DEVICE-SAMPLE GREEDY =================
    // The serving tick's shape: every row requests a greedy device sample. Rows must stay
    // bit-identical AND every device token must equal the reference argmax.
    eprintln!("[phase] arm 3: device-sample greedy batch (the serving epilogue contract)");
    let mut samp_arm = ArmCheck::new("devsample-greedy");
    {
        let mut caches: Vec<memra_engine::cache::Cache> = Vec::with_capacity(b_n);
        for s in &sessions {
            let mut cache = new_cache(&e)?;
            for &tok in &s.ids[..s.prefix] {
                let _ = m.decode_step(&e, tok, &mut cache)?;
            }
            caches.push(cache);
        }
        let mut token_bad = 0usize;
        for k in 0..n {
            let toks: Vec<u32> = sessions.iter().map(|s| s.ids[s.prefix + k]).collect();
            let samp: Vec<Option<memra_engine::decode_batch::DevSamp>> = (0..b_n)
                .map(|bi| {
                    Some(memra_engine::decode_batch::DevSamp::new(
                        0.0,
                        0,
                        (k * b_n + bi) as u32,
                        0,
                        1.0,
                        0.0,
                    ))
                })
                .collect();
            let mut cache_refs: Vec<&mut memra_engine::cache::Cache> = caches.iter_mut().collect();
            let (rows, next) = m.decode_step_batch_sampled(&e, &toks, &mut cache_refs, &samp)?;
            for bi in 0..b_n {
                let step = sessions[bi].prefix + k;
                samp_arm.check(
                    step,
                    &format!("s{bi} sampled tick {k}"),
                    &rows[bi],
                    &ref_logits[bi][step],
                );
                let want = ref_tapes[bi][step];
                match next[bi] {
                    Some(got) if got == want => {}
                    got => {
                        token_bad += 1;
                        println!(
                            "[devsample-greedy] TOKEN MISMATCH s{bi} tick {k}: device {got:?} \
                             != reference argmax {want}"
                        );
                    }
                }
            }
        }
        samp_arm.token_bad = token_bad;
    }

    // ================= verdicts =================
    let mut fail = false;
    for arm in [&batch_arm, &b1_arm, &samp_arm] {
        assert!(
            arm.checked_steps > 0,
            "[{}] compared ZERO steps — a vacuous arm never prints PASS",
            arm.name
        );
        if arm.bad_steps == 0 && arm.token_bad == 0 && !arm.tape_bad {
            println!(
                "glm5-hyper-batch gate PASS [{}]: {} comparisons BIT-IDENTICAL vs the \
                 isolated serial hc walk (n_vocab={n_vocab}; {knobs})",
                arm.name, arm.checked_steps
            );
        } else {
            let detail = match arm.first {
                Some((s, i, a, b)) => format!(
                    "{}/{} comparisons mismatched (first @ step {s} idx {i}: ref={a:?} \
                     batch={b:?})",
                    arm.bad_steps, arm.checked_steps
                ),
                None => format!(
                    "{}/{} comparisons mismatched",
                    arm.bad_steps, arm.checked_steps
                ),
            };
            let tokens = if arm.token_bad > 0 {
                format!("; {} device tokens != reference argmax", arm.token_bad)
            } else {
                String::new()
            };
            let tape = match (arm.tape_checked, arm.tape_bad) {
                (false, _) => "",
                (true, true) => "; greedy tape DIVERGED",
                (true, false) => {
                    "; greedy tape MATCHED even so — the logit compare is the load-bearing bar"
                }
            };
            println!(
                "glm5-hyper-batch gate FAIL [{}]: {detail}{tokens}{tape} ({knobs})",
                arm.name
            );
            fail = true;
        }
    }
    if fail {
        std::process::exit(1);
    }
    Ok(())
}
