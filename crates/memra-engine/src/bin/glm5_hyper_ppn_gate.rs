//! glm5_next mHC ppN gate: the N-stage pipeline-split walk over a HYPER-CONNECTION trunk
//! (`ResidualTopology::HyperConnections`) must produce BIT-IDENTICAL logits to the unsplit
//! walk at every step, for prime and for decode, at every N/knob combination.
//!
//! WHY A SECOND ppN GATE. `ppn-gate` drives `decode_step` over a real GGUF checkpoint, and
//! every model it can open runs the SERIAL residual. glm5_next (GLM-5.3-Flash) does not:
//! its trunk carries a 4-stream mHC residual, its walks are `forward_hyper` /
//! `prime_cache_hyper` / `decode_step_hyper`, and those three walks REFUSED the pp door
//! outright ("the sharded stage handoff is unwired for this residual topology"). Lifting
//! that refusal is what this gate exists to hold down. It is fixture-driven rather than
//! checkpoint-driven for two reasons: the artifact is 190.7 GB and has never been on a
//! one-card rig, and the fixture is the only way this runs in a place where a stage handoff
//! can be deliberately broken and the red arm banked.
//!
//! WHAT THE COMPARISON PROVES, AND HOW THE TRUTH CHAIN CLOSES. Split-vs-unsplit is
//! ARM-EQUALITY, not truth (GATE:pin-against-truth): both arms run the same kernels over the
//! same weights, so an error in the hc arithmetic itself cancels. The chain closes by
//! COMPOSITION, and both halves must be run to claim anything:
//!
//! * `tests/hyper_connections_gpu.rs` anchors the UNSPLIT hc walk to `memra_reference`
//!   (a host executor that shares no code with the GPU path) on the same fixture family;
//! * this gate anchors the SPLIT walk to the unsplit one, bit for bit.
//!
//! Neither is sufficient alone. State both when citing this gate.
//!
//! THE SPLIT WALK IS NOT A SCHEDULING CHANGE, which is why the bar is bit-identity and not a
//! tolerance. Every boundary handoff is an exact copy of the `[streams, n_embd]` stream state
//! (decode) or `[t, streams, n_embd]` (prime); every layer runs the same kernels on the same
//! bytes; only the stream, the context and (under `MEMRA_PP_DEVICES`) the device change. Any
//! differing bit in any of the `n_vocab` f32 logits is a seam bug.
//!
//! ARMS, one placement per invocation (`PpNRt` freezes its stage/device map at first build,
//! so knob matrices are driven by re-invoking this binary, never by looping inside it):
//!
//! 1. DECODE SERIAL — door-off `decode_step` reference tape, replayed into a fresh
//!    `pp::new_cache_planned` cache with the door on. Full-logit bit compare per step,
//!    plus the greedy token tape compared as its own receipt line.
//! 2. PRIME TWIN — `prime_cache` over the prompt in ONE call (this model has no batched
//!    prime; `prime_cache_hyper` is the monolithic walk), then decode continuation on the
//!    cache it left behind. Both halves bit-compared. This is the arm that sees a prime
//!    whose per-stage KDA conv/recurrent state or MLA latent+kpool planes landed on the
//!    wrong device: the decode continuation reads exactly that state.
//! 3. PREFILL TWIN — `forward` (all rows) and `forward_last`, the stateless walk, split vs
//!    unsplit. It shares the layer-range helpers with the other two but has its own trunk
//!    entry and its own `last_only` head branch, so without this arm `forward_hyper_ppn`
//!    would be shipped uncovered.
//! 4. PIPELINED — SKIPPED for hc by construction. `decode_step_h_ppn_deferred` calls
//!    `refuse_hyper` (decode.rs), so the deferred-readback arm is not wired for this
//!    residual topology and this gate prints a NOTE rather than pretending to cover it.
//!
//!
//! NON-VACUITY IS ENFORCED, NOT ASSUMED (the wiring-assertions-match-prose law). Before any
//! comparison the gate asserts: the loaded model really declares HyperConnections; the door
//! really opened with the requested stage count; every stage owns at least one layer; and the
//! fence really SEPARATES the two per-layer state classes glm5_next carries — a `Recurrent`
//! (KDA conv ring + delta-rule state) layer and a `LatentKvCache` (MLA rows + kpool indexer
//! plane) layer land on DIFFERENT stages. A fence that kept every stateful layer on one stage
//! would pass while proving nothing about `pp::new_cache`'s per-stage placement contract.
//!
//! MUTATION-BOUND. See `research/glm53-flash-bringup-20260827/ppn-hyper-gate/` for the banked
//! red arm: the stage handoff is broken two ways (a dropped TX, and an off-by-one layer range
//! that skips a layer) and this gate is required to go red on each. Weight corruption is NOT
//! the right mutation class here — it breaks both arms equally and the comparison stays green.
//!
//! SCOPE — what this does NOT prove. F32 fixture weights, tiny widths, no quantized expert
//! class (that is `glm5_moe_residency_gpu`), no throughput claim of any kind, and no evidence
//! about the 190.7 GB artifact's placement. Cross-device arms (`MEMRA_PP_DEVICES=0,1`) and
//! weight-sharding arms need a multi-card box; on a one-card rig the same-device multi-stream
//! arms still exercise the whole split walk, the boundary transport and the per-stage caches.
//!
//! Knobs PASS THROUGH from the environment and are printed in the verdict:
//!   MEMRA_PP_DEVICES=d0,..,dN-1  stage->device placement
//!   MEMRA_PP_SPLITS=c1,..,cN-1   explicit cuts (default even split)
//!   MEMRA_PP_SHARD=0             rollback to bring-up placement (weights all-primary)
//!   MEMRA_PP_STREAMS=0           the same-stream seam
//!   MEMRA_PP_OVERLAP=1           double-buffered boundary slots
//!
//! Rig law: exactness only, never a timing number. Run under `flock /tmp/memra-5090.lock`
//! with `NVIDIA_TF32_OVERRIDE=0`.
//!
//! usage: glm5-hyper-ppn-gate [stages=2] [P=6] [N=8]
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

/// KDA `head_dim` is 128 because `memra_kda_scan_s128` is instantiated for that width only,
/// and the hidden size must carry it.
const HIDDEN: usize = 128;
const VOCAB: u32 = 32;
/// Four trunk layers, alternating the two mixer classes and the two MLP classes, so a
/// 2-stage even split lands one of each on both sides and a 4-stage split gives every layer
/// its own stage. This is what makes the state-class separation assertion satisfiable.
const LAYERS: usize = 4;

/// glm5_next's real shape, shrunk only in width: 4 mHC streams, mean collapse, sigmoid
/// noaux_tc router with `routed_scaling_factor` 2.5, PRE-clamped SwiGLU, kpool indexer with
/// the always-select tail. Layer classes alternate KDA / DSA(MLA+kpool) and dense / sparse.
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

/// The three stacked MoE expert slabs are the only tensors the engine refuses to load as F32
/// (`HostExps` rejects an F32 slab), so they ride Q8_0 — the encoding `micro_gguf` fixtures use.
/// This gate compares two GPU arms over the SAME bytes, so the encoding costs it no parity: it
/// only has to be an encoding the loader accepts.
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

/// Serves the reference fixture's numbers under the contract's ggml names. `config()` is
/// load-bearing: `HybridModel::load_from_source` compiles the plan from it, so the real
/// glm5_next pack decides the topology rather than a hand-built plan.
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

/// Deterministic token stream. Same generator shape as the hc reference gates.
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

/// FNV-1a over the greedy tape, printed so a receipt carries a single comparable token.
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
    name: &'static str,
    bad_steps: usize,
    checked_steps: usize,
    /// Tracked SEPARATELY from `bad_steps` so the verdict cannot print "15/14 comparisons
    /// mismatched". The mutation runs found that: a tape divergence was being counted as a
    /// logit comparison. A gate whose own arithmetic is wrong is a gate nobody reads.
    tape_bad: bool,
    /// Whether this arm has a greedy tape at all (the stateless prefill arm does not).
    tape_checked: bool,
    first: Option<(usize, usize, f32, f32)>, // (step, idx, ref, got)
}

impl ArmCheck {
    fn new(name: &'static str) -> Self {
        ArmCheck {
            name,
            bad_steps: 0,
            checked_steps: 0,
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
                     @[{idx}] ref={a:?} pp={b:?}",
                    self.name,
                    r.len()
                );
            }
        }
    }
    /// The greedy tape is a separate receipt line, not a restatement of the logit compare:
    /// it is the thing a serving cell would actually observe, and it prints a single hash.
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
                "[{}] greedy tape DIVERGED at index {at}: ref={hw:#018x} pp={hg:#018x}",
                self.name
            );
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stages: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let p: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let n: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    // cuBLASLt f32 compute rides TF32 on Blackwell by default — right for serving, wrong for
    // an exactness gate. The driver reads this at CUDA init, so it must be set before the
    // first Engine::new in the process.
    if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
        // SAFETY: single-threaded, before any CUDA call in this process.
        unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
    }
    // The gate owns the door and OPENS IT BEFORE LOAD (weight sharding is a load-time
    // decision). Every other knob passes through from the caller.
    // SAFETY: single-threaded, before any engine or runtime exists.
    unsafe { std::env::set_var("MEMRA_PP_STAGES", stages.to_string()) };

    let devices_env = std::env::var("MEMRA_PP_DEVICES").unwrap_or_default();
    let primary_dev: usize = devices_env
        .split(',')
        .next()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let knobs = format!(
        "stages={stages} streams={} overlap={} devices={} splits={} shard={}",
        if memra_engine::pp::pp2_streams_off() {
            "OFF(same-stream seam)"
        } else {
            "per-stage"
        },
        if memra_engine::pp::pp2_overlap() {
            "1(double-buffered)"
        } else {
            "0"
        },
        if devices_env.is_empty() {
            "default(primary)"
        } else {
            &devices_env
        },
        std::env::var("MEMRA_PP_SPLITS").unwrap_or_else(|_| "default(even)".into()),
        if memra_engine::pp::pp_shard_off() {
            "OFF(bring-up placement)"
        } else {
            "per-stage"
        },
    );
    println!("glm5-hyper-ppn-gate config: {knobs}");

    let config = ModelConfig::from_hf(&HfConfig::parse(&mini_config_json()));
    let plan = memra_gguf::model_packs::for_config(&config)
        .expect("glm5_next model pack matches the mini config")
        .compile_plan(&config)
        .expect("mini glm5_next plan compiles");
    assert_eq!(
        plan.layers.len(),
        LAYERS,
        "the fixture plan must carry {LAYERS} trunk layers"
    );
    assert_eq!(plan.hidden_size as usize, HIDDEN);
    let fixture = deterministic_fixture(&plan).expect("deterministic glm5_next hc fixture");
    let source = fixture_source(&config, &plan, &fixture.weights);

    let e = Engine::new(primary_dev)?;
    let m = HybridModel::load_from_source_without_mtp(&e, &source)?;

    // ---- NON-VACUITY: this really is the hc trunk, really split, really across state classes ----
    let topology = m.hyper.as_ref().expect(
        "the fixture must load as a HyperConnections trunk — otherwise this gate is \
                 measuring the generic ppN arm that `ppn-gate` already covers",
    );
    println!(
        "hc topology: streams={} collapse={:?} sinkhorn_iters={}",
        topology.streams, topology.collapse, topology.sinkhorn_iterations
    );
    let n_layers = m.layers.len();
    let fence = memra_engine::pp::pp_cuts(n_layers).unwrap_or_else(|| {
        panic!("ppn door failed to open (n_layers={n_layers}, stages={stages})")
    });
    assert_eq!(
        fence.len() - 1,
        stages,
        "fence {fence:?} != stages {stages}"
    );
    for w in fence.windows(2) {
        assert!(
            w[1] > w[0],
            "fence {fence:?} leaves an empty stage; a stage with no layers proves nothing"
        );
    }
    println!("stage fence: {fence:?} over {n_layers} layers");

    // The audit that item 3 of the roadmap asks for, expressed as an assertion rather than a
    // paragraph: glm5_next carries TWO per-layer state classes, and `pp::new_cache` places
    // each on its owning stage's device. If the fence kept them together, a cache placed
    // entirely on one stage would still pass, and the placement contract would be untested.
    let mut recur_stages: Vec<usize> = Vec::new();
    let mut latent_stages: Vec<usize> = Vec::new();
    for layer in &plan.layers {
        let s = memra_engine::pp::stage_of(&fence, layer.index as usize);
        match layer.state {
            StatePlan::Recurrent { .. } => recur_stages.push(s),
            StatePlan::LatentKvCache { .. } => latent_stages.push(s),
            _ => {}
        }
    }
    println!(
        "state placement: Recurrent(KDA) layers on stages {recur_stages:?}, \
         LatentKvCache(MLA+kpool) layers on stages {latent_stages:?}"
    );
    assert!(
        !recur_stages.is_empty() && !latent_stages.is_empty(),
        "the fixture must carry BOTH a Recurrent and a LatentKvCache layer; got \
         recur={recur_stages:?} latent={latent_stages:?}"
    );
    let spread = recur_stages
        .iter()
        .chain(latent_stages.iter())
        .collect::<std::collections::BTreeSet<_>>();
    assert!(
        spread.len() >= 2,
        "fence {fence:?} puts every stateful layer on one stage ({spread:?}); this arm cannot \
         see a per-stage cache placement bug — choose a split that separates them"
    );

    let ids = tokens(p + n, 0x0915_5EED);
    let max_ctx = p + n + 8;

    // PHASE MARKERS. A gate that dies mid-run must say WHICH walk died: the door-OFF
    // reference and the door-ON arms fail for completely different reasons, and telling them
    // apart is the difference between "the engine is broken" and "this harness cannot run this
    // knob". Earned in the 2026-08-28 cross-device run, where MEMRA_PP_HOST_BOUNCE=1 died with
    // a bare `split_latent` CUDA error and the phase was not recoverable from the output.
    eprintln!("[phase] reference A: door OFF, step-by-step decode over the whole sequence");
    // ================= reference: door OFF =================
    // SAFETY: single-threaded; no PP runtime has been built yet in this process.
    unsafe { std::env::remove_var("MEMRA_PP_STAGES") };
    assert!(
        memra_engine::pp::pp_cuts(n_layers).is_none(),
        "the reference arm must run with the door SHUT"
    );

    // Reference A: step-by-step decode over the whole sequence (the ppn-gate shape).
    let mut ref_step_logits: Vec<Vec<f32>> = Vec::with_capacity(p + n);
    let mut ref_step_tape: Vec<u32> = Vec::with_capacity(p + n);
    {
        let mut cache = memra_engine::cache::Cache::new_planned(&e, &m.cfg, &plan, max_ctx)?;
        for &tok in &ids {
            let ll = m.decode_step(&e, tok, &mut cache)?;
            ref_step_tape.push(argmax(&ll) as u32);
            ref_step_logits.push(ll);
        }
    }
    let n_vocab = ref_step_logits[0].len();

    eprintln!("[phase] reference C: door OFF, stateless prefill (forward / forward_last)");
    // Reference C: the stateless prefill walk, all rows and the last-row head branch.
    let ref_forward = m.forward(&e, &ids)?;
    let ref_forward_last = m.forward_last(&e, &ids)?;

    eprintln!("[phase] reference B: door OFF, prime + decode continuation");
    // Reference B: prime the prompt in ONE call, then decode the continuation — the serving
    // shape, and the one that exercises the prime walk's per-stage state.
    let mut ref_cont_logits: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut ref_cont_tape: Vec<u32> = Vec::with_capacity(n);
    let ref_prime_logits: Vec<f32>;
    {
        let mut cache = memra_engine::cache::Cache::new_planned(&e, &m.cfg, &plan, max_ctx)?;
        let (primed, _seed, _hiddens) = m.prime_cache(&e, &ids[..p], &mut cache, 0)?;
        ref_prime_logits = primed;
        for &tok in &ids[p..] {
            let ll = m.decode_step(&e, tok, &mut cache)?;
            ref_cont_tape.push(argmax(&ll) as u32);
            ref_cont_logits.push(ll);
        }
    }

    // ================= arm 1 (DECODE SERIAL): door ON =================
    // SAFETY: single-threaded.
    unsafe { std::env::set_var("MEMRA_PP_STAGES", stages.to_string()) };
    eprintln!("[phase] arm 1: door ON, decode-serial split walk");
    let mut decode_arm = ArmCheck::new("decode-serial");
    {
        let mut cache = memra_engine::pp::new_cache_planned(&e, &m.cfg, &plan, max_ctx)?;
        let mut tape: Vec<u32> = Vec::with_capacity(p + n);
        for (step, &tok) in ids.iter().enumerate() {
            let ll = m.decode_step(&e, tok, &mut cache)?;
            tape.push(argmax(&ll) as u32);
            decode_arm.check(step, "decode", &ll, &ref_step_logits[step]);
        }
        decode_arm.check_tape(&tape, &ref_step_tape);
    }

    // ================= arm 2 (PRIME TWIN): door ON =================
    eprintln!("[phase] arm 2: door ON, prime-twin split walk");
    let mut prime_arm = ArmCheck::new("prime-twin");
    {
        let mut cache = memra_engine::pp::new_cache_planned(&e, &m.cfg, &plan, max_ctx)?;
        let (primed, _seed, _hiddens) = m.prime_cache(&e, &ids[..p], &mut cache, 0)?;
        prime_arm.check(0, "prime last row", &primed, &ref_prime_logits);
        let mut tape: Vec<u32> = Vec::with_capacity(n);
        for (k, &tok) in ids[p..].iter().enumerate() {
            let ll = m.decode_step(&e, tok, &mut cache)?;
            tape.push(argmax(&ll) as u32);
            prime_arm.check(k + 1, "decode after prime", &ll, &ref_cont_logits[k]);
        }
        prime_arm.check_tape(&tape, &ref_cont_tape);
    }

    // ================= arm 3 (PREFILL TWIN): door ON =================
    // `forward`/`forward_last` route through `forward_hyper_ppn`, which shares the layer-range
    // helpers with the other two walks but has its own trunk entry and its own `last_only` head
    // branch. Without this arm that code would ship uncovered.
    eprintln!("[phase] arm 3: door ON, prefill-twin split walk");
    let mut prefill_arm = ArmCheck::new("prefill-twin");
    {
        let got = m.forward(&e, &ids)?;
        prefill_arm.check(0, "forward (all rows)", &got, &ref_forward);
        let got_last = m.forward_last(&e, &ids)?;
        prefill_arm.check(1, "forward_last", &got_last, &ref_forward_last);
    }

    // ---- arm 4 (PIPELINED): refused for hc by construction, never silently skipped ----
    println!(
        "glm5-hyper-ppn-gate NOTE: pipelined arm skipped — `decode_step_h_ppn_deferred` calls \
         refuse_hyper(), so deferred readback is not wired for the mHC residual. This gate \
         covers the serial split walk only; do not cite it for the pipelined arm."
    );

    // ================= verdicts =================
    let mut fail = false;
    for arm in [&decode_arm, &prime_arm, &prefill_arm] {
        assert!(
            arm.checked_steps > 0,
            "[{}] compared ZERO steps — a vacuous arm never prints PASS",
            arm.name
        );
        if arm.bad_steps == 0 && !arm.tape_bad {
            println!(
                "glm5-hyper-ppn gate PASS [{}]: {} comparisons BIT-IDENTICAL vs the unsplit hc \
                 walk (n_vocab={n_vocab}, P={p} N={n}, fence={fence:?}; {knobs})",
                arm.name, arm.checked_steps
            );
        } else {
            let detail = match arm.first {
                Some((s, i, a, b)) => format!(
                    "{}/{} comparisons mismatched (first @ step {s} idx {i}: ref={a:?} \
                     pp={b:?})",
                    arm.bad_steps, arm.checked_steps
                ),
                None => format!(
                    "{}/{} comparisons mismatched",
                    arm.bad_steps, arm.checked_steps
                ),
            };
            let tape = match (arm.tape_checked, arm.tape_bad) {
                (false, _) => "; this arm carries no greedy tape (stateless walk)",
                (true, true) => "; greedy tape DIVERGED",
                // Mutations M2/M3 both produced this combination: the logit compare is the
                // load-bearing bar and the tape is a weaker second signal, never a substitute.
                (true, false) => {
                    "; greedy tape MATCHED even so — the logit compare is the \
                                  load-bearing bar"
                }
            };
            println!(
                "glm5-hyper-ppn gate FAIL [{}]: {detail}{tape} (fence={fence:?}; {knobs})",
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
