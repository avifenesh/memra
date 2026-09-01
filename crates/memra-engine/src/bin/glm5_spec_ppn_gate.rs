//! glm5_next SPEC-UNDER-ppN gate (lane/glm5-ppn-verify): the T-parallel verify walk, the
//! per-plane rollback and the MTP draft chain must hold their byte-identity contracts when
//! the trunk runs as an N-stage pipeline split — the tparallel battery, re-fought under the
//! door `glm5_verify_rows` used to refuse by name.
//!
//! WHY A THIRD ppN GATE. `glm5-hyper-ppn-gate` pins the PLAIN hc walks (decode/prime/
//! prefill) split-vs-unsplit; `glm5_tparallel_verify_gpu` pins the SPEC machinery
//! single-device. Neither covers the composition: the verify walk chains K+1 rows through
//! ONE cache whose state planes are stage-owned under the split, the rollback must restore
//! every stage's layers coherently, and the MTP chain consumes last-stage h_seeds. This
//! gate holds all three down, red-proven, one placement per invocation (`PpNRt` freezes its
//! stage/device map at first build — knob matrices re-invoke the binary).
//!
//! THE TRUTH CHAIN, stated as in the hyper gate: split-vs-unsplit alone is arm-equality
//! (GATE:pin-against-truth). It closes by COMPOSITION:
//!   * `tests/hyper_connections_gpu.rs` anchors the unsplit hc walk to `memra_reference`;
//!   * `tests/glm5_tparallel_verify_gpu.rs` anchors the single-device spec loop to the
//!     unsplit plain walk (and `glm5_mtp_head_gpu` anchors the draft head to reference);
//!   * `glm5-hyper-ppn-gate` anchors the split plain walk to the unsplit one;
//!   * THIS gate anchors the split spec loop to the door-OFF references bit for bit, and
//!     re-pins plain ppN decode against the same references in the same process (arm W0),
//!     so "verify rows == plain ppN decode" holds by the same-reference transitivity.
//!
//! ARMS:
//!   P0. PRIME DETERMINISM (quiet regime) — R repeated split primes of one prompt, each
//!       bit-identical to the door-OFF prime.
//!   P1. PRIME DETERMINISM AFTER THE SPEC ARMS — the SAME census once the stage streams
//!       carry rollback/teardown tails. lane/glm5-accrace: P1 is the detector for a
//!       stage-stream exit-publication hole (P0's regime is measurably too quiet), and
//!       both catch it at its source instead of many rounds downstream.
//!   W0. plain ppN decode re-pin — door-ON `decode_step` rows vs the door-OFF chain.
//!   W1. WALK — door-ON `glm5_verify_rows` (the ppN twin) rows vs the door-OFF plain
//!       chain, full logits, bit for bit.
//!   A.  ROLLBACK — accept-j-then-continue under the split vs the door-OFF never-drafted
//!       chain, every j in 0..=K.
//!   E.  END TO END — `generate_spec_glm5` under the split: greedy tapes K=1..7 (natural
//!       drafter) + forced full-accept + a forced-rejection j-sweep, each byte-identical
//!       to the door-OFF plain tape.
//!   R1. RED stale-KDA — reinstating post-row-K KDA state after a partial-accept rollback
//!       (through the OWNING stage engines) must diverge the continuation.
//!   R2. RED pool-keys — reinstating un-clamped `index_pools_ready` must trip the kpool
//!       residency tripwire by name on the next door-ON step.
//!   R3. RED rollback-disabled — forced rejections with the rollback skipped must break
//!       the tape or fail loudly, never byte-identical-and-green.
//!
//! NON-VACUITY IS ENFORCED (the wiring-assertions-match-prose law): hc topology loaded,
//! door really open at the requested stage count, no empty stage, and the fence SEPARATES
//! the two per-layer state classes (Recurrent vs LatentKvCache) across stages — otherwise
//! the per-stage rollback contract goes untested.
//!
//! SCOPE — what this does NOT prove: F32/Q8_0 fixture classes only, no throughput claim,
//! no evidence about the real artifact's placement. On a one-card rig the same-device
//! multi-stream arms exercise the whole split walk, the boundary transport, the per-stage
//! streams and the rollback seams; CROSS-DEVICE arms (`MEMRA_PP_DEVICES=0,1[,2]`) need a
//! multi-card box and are the named final gate of the lane.
//!
//! Knobs pass through: MEMRA_PP_DEVICES / MEMRA_PP_SPLITS / MEMRA_PP_SHARD=0 /
//! MEMRA_PP_STREAMS=0 / MEMRA_PP_OVERLAP.
//!
//! Rig law: exactness only, never a timing number. Run under `flock /tmp/memra-5090.lock`
//! with `NVIDIA_TF32_OVERRIDE=0`.
//!
//! usage: glm5-spec-ppn-gate [stages=2] [P=24] [N=20] [accept-probe=0] [prime-reps=8]
use memra_engine::Engine;
use memra_engine::forward::argmax;
use memra_engine::glm_spec::Glm5SpecKnobs;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::{ModelPlan, StatePlan};
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
/// Drafts per round in the walk/rollback arms (t = K+1 = 8 rows, inside the cap of 15).
const K: usize = 7;

// ---------------------------------------------------------------------------------------------
// Fixture: the tparallel gate's mini glm5_next (hc trunk + ONE NextN block), verbatim.
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

/// NVFP4 expert banks with a LIVE per-expert macro plane since lane/glm5-vrest (the same
/// flip as `glm5_tparallel_verify_gpu`'s fixture): Q8_0 is not `q8_expert_supported`, so
/// the batched verify-rows MoE arm could never engage under the split and the ppN arms
/// were vacuous for it. `in_f % 64 == 0` holds (gate/up in = 128, down in = 64).
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
    // LIVE `weight_scale_2` macro plane per routed bank, every value off 1.0 in two bands
    // (the epilogue-gate construction) so a dropped macro fold anywhere in the split walk
    // moves bits.
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

fn bit_diffs(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// Fresh stage-aware cache + prime. Door state decides the allocation path inside
/// `pp::new_cache_planned` (door shut = plain `Cache::new_planned`, byte-identical).
fn fresh_primed(
    e: &Engine,
    m: &HybridModel,
    plan: &ModelPlan,
    prompt: &[u32],
    max_ctx: usize,
) -> (memra_engine::cache::Cache, Vec<f32>) {
    let mut cache = memra_engine::pp::new_cache_planned(e, &m.cfg, plan, max_ctx)
        .expect("cache for the mini glm5 model");
    let (logits, _seed, _hiddens) = m.prime_cache(e, prompt, &mut cache, 0).expect("hc prime");
    (cache, logits)
}

fn plain_tape(
    e: &Engine,
    m: &HybridModel,
    plan: &ModelPlan,
    prompt: &[u32],
    max_new: usize,
) -> Vec<u32> {
    let (mut cache, logits) = fresh_primed(e, m, plan, prompt, prompt.len() + max_new + 16);
    let mut tape = Vec::with_capacity(max_new);
    tape.push(argmax(&logits) as u32);
    while tape.len() < max_new {
        let ll = m
            .decode_step(e, *tape.last().unwrap(), &mut cache)
            .expect("plain decode step");
        tape.push(argmax(&ll) as u32);
    }
    tape
}

/// First index where two tapes differ, with both tokens — the bisect anchor a diverging
/// e2e arm owes its reader (lane/glm5-accrace follow-up 6). `None` when they are identical.
fn first_diff(out: &[u32], tape: &[u32]) -> Option<(usize, Option<u32>, Option<u32>)> {
    let n = out.len().max(tape.len());
    (0..n)
        .find(|&i| out.get(i) != tape.get(i))
        .map(|i| (i, out.get(i).copied(), tape.get(i).copied()))
}

/// The detail string of a tape-identity arm, REPORTING THE COMPARISON IT RAN.
///
/// WHY THIS EXISTS: every e2e arm used to hand `arm()` a STATIC "tape identical" string,
/// which `Verdicts::arm` then printed under the word FAIL — a failing line that describes
/// a passing one (`gate FAIL [E forced-rejection sweep K=7]: ... tape identical (13/42)`,
/// banked in `research/glm53-flash-bringup-20260827/dedup-20260831/receipts/ppn-gate/
/// flake-20260831/`). A FAIL that reads like a PASS is how a real red gets waved past, so
/// the message is now derived from `out` vs `tape` rather than asserted alongside it.
fn tape_verdict(what: &str, out: &[u32], tape: &[u32], accepted: usize, drafted: usize) -> String {
    match first_diff(out, tape) {
        None => format!("{what}, tape identical to door-OFF plain greedy ({accepted}/{drafted})"),
        Some((i, got, want)) => format!(
            "{what}, TAPE DIVERGED from door-OFF plain greedy at index {i} \
             (got {got:?}, want {want:?}; lens {}/{}) ({accepted}/{drafted})",
            out.len(),
            tape.len()
        ),
    }
}

struct Verdicts {
    fails: usize,
}

impl Verdicts {
    fn arm(&mut self, name: &str, ok: bool, detail: &str) {
        if ok {
            println!("glm5-spec-ppn gate PASS [{name}]: {detail}");
        } else {
            println!("glm5-spec-ppn gate FAIL [{name}]: {detail}");
            self.fails += 1;
        }
    }
}

#[allow(clippy::too_many_lines)]
// allow: the gate is one linear battery; splitting it would scatter the arms' shared
// references and the door choreography across functions that run exactly once
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stages: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    // Prompt length 24 (default): past the k-pool raw budget (index_topk 8, kpool 4) so
    // the trunk indexer runs SPARSE and pool-key finality is live, not decorative.
    let p: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let n: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    // Arg 4 = the accrace ACCEPT PROBE (a positional gate instrument, deliberately NOT an
    // env flag: it is diagnosis plumbing for one arm, not a runtime surface). `1` traces
    // every forced-rejection round's device-vs-host accept row and per-row logit hashes to
    // stderr; see `Glm5SpecKnobs::accept_probe`.
    let probe: bool = std::env::args().nth(4).as_deref() == Some("1");
    // Arg 5 = repetitions of the P0 prime-determinism arm (default 8). It is the SENSITIVE
    // instrument of lane/glm5-accrace: one gate run samples the ppN prime R times, so the
    // arm is ~R times likelier to catch a stage-stream publication hole than any single
    // end-to-end tape.
    let prime_reps: usize = std::env::args()
        .nth(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    // TF32 must be off before the first CUDA call (exactness gate).
    if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
        // SAFETY: single-threaded, before any CUDA call in this process.
        unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
    }
    // The gate owns the door and OPENS IT BEFORE LOAD (weight sharding is a load-time
    // decision under MEMRA_PP_DEVICES); the MTP head must load for the draft arms.
    // SAFETY: single-threaded, before any engine or runtime exists.
    unsafe {
        std::env::set_var("MEMRA_PP_STAGES", stages.to_string());
        std::env::set_var("MEMRA_GLM5_MTP", "1");
    }
    // DOOR-H ALIAS HYGIENE (lane/glm5-extract2). This gate's composed arms are driven from
    // the SHELL with `MEMRA_GLM5_HTOD_DIET=1` (the matrix runners' `compose-*-doors-EDH`
    // cells), and they assert BIT-IDENTITY — which passes whether the door armed or not. So a
    // leaked `MEMRA_HTOD_DIET=0` in the runner's environment would DISAGREE with the alias,
    // fall the door closed, and make those ON arms silently vacuous. The alias the caller set
    // is left alone; only the general name is cleared, so it cannot outvote them.
    // SAFETY: single-threaded, before any engine or runtime exists.
    unsafe { std::env::remove_var("MEMRA_HTOD_DIET") };

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
    println!("glm5-spec-ppn-gate config: {knobs}");

    let config = ModelConfig::from_hf(&HfConfig::parse(&mini_config_json()));
    let plan = memra_gguf::model_packs::for_config(&config)
        .expect("glm5_next model pack matches the mini config")
        .compile_plan(&config)
        .expect("mini glm5_next plan compiles");
    let e = Engine::new(primary_dev)?;
    let m = HybridModel::load_from_source(&e, &fixture_source(&config, &plan))?;

    // ---- NON-VACUITY: hc trunk, MTP head, door open, stages non-empty, states split ----
    let topology = m
        .hyper
        .as_ref()
        .expect("the fixture must load as a HyperConnections trunk");
    assert!(
        m.mtp.is_some(),
        "the NextN draft head must load (MEMRA_GLM5_MTP=1)"
    );
    println!(
        "hc topology: streams={} collapse={:?}",
        topology.streams, topology.collapse
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
        assert!(w[1] > w[0], "fence {fence:?} leaves an empty stage");
    }
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
        "stage fence: {fence:?}; Recurrent(KDA) on stages {recur_stages:?}, \
         LatentKvCache(MLA+kpool) on stages {latent_stages:?}"
    );
    let spread = recur_stages
        .iter()
        .chain(latent_stages.iter())
        .collect::<std::collections::BTreeSet<_>>();
    assert!(
        spread.len() >= 2,
        "fence {fence:?} puts every stateful layer on one stage — the per-stage rollback \
         contract would go untested; choose a split that separates them"
    );

    let prompt = tokens(p, 0xA11CE);
    let vt = tokens(K + 1, 0xBEEF);
    // Continuation stream DIFFERENT from vt (a rollback keeping stale rows must not pass
    // by coincidence).
    let cc = tokens(12, 0xC0FFEE);
    let max_ctx = p + K + cc.len() + 16;

    // ================= references: door OFF =================
    // Under a sharded cross-device placement the door-off walks peer-read remote weights —
    // byte-exact (the hyper-ppn gate's banked correction), slow, and fine for a reference.
    eprintln!("[phase] references: door OFF");
    // SAFETY: single-threaded; no PP runtime has been built yet in this process.
    unsafe { std::env::remove_var("MEMRA_PP_STAGES") };
    assert!(
        memra_engine::pp::pp_cuts(n_layers).is_none(),
        "the reference phase must run with the door SHUT"
    );

    // R-walk: prime + plain decode chain over vt. `ref_prime` is the door-OFF prime's own
    // logits — the truth the P0 prime-determinism arm holds the split prime to.
    let mut ref_rows: Vec<Vec<f32>> = Vec::with_capacity(vt.len());
    let ref_prime;
    {
        let (mut cache, l) = fresh_primed(&e, &m, &plan, &prompt, max_ctx);
        ref_prime = l;
        for &tok in &vt {
            ref_rows.push(m.decode_step(&e, tok, &mut cache)?);
        }
    }
    let n_vocab = ref_rows[0].len();

    // R-acceptj: never-drafted continuations for every j.
    let mut ref_cont: Vec<Vec<Vec<f32>>> = Vec::with_capacity(K + 1);
    for j in 0..=K {
        let keep = j + 1;
        let (mut cache, _l) = fresh_primed(&e, &m, &plan, &prompt, max_ctx);
        for &tok in &vt[..keep] {
            let _ = m.decode_step(&e, tok, &mut cache)?;
        }
        let mut rows = Vec::with_capacity(cc.len());
        for &tok in &cc {
            rows.push(m.decode_step(&e, tok, &mut cache)?);
        }
        ref_cont.push(rows);
    }

    // R-tape: the plain greedy tape the e2e arms must reproduce.
    let tape = plain_tape(&e, &m, &plan, &prompt, n);

    // ================= arms: door ON =================
    // SAFETY: single-threaded.
    unsafe { std::env::set_var("MEMRA_PP_STAGES", stages.to_string()) };
    let mut v = Verdicts { fails: 0 };

    // ---- arm W0: plain ppN decode re-pin (the walk-identity bar's other arm) ----
    eprintln!("[phase] arm W0: door ON, plain ppN decode re-pin");
    {
        let (mut cache, _l) = fresh_primed(&e, &m, &plan, &prompt, max_ctx);
        let mut bad = 0usize;
        for (r, &tok) in vt.iter().enumerate() {
            let got = m.decode_step(&e, tok, &mut cache)?;
            let diffs = bit_diffs(&got, &ref_rows[r]);
            if diffs > 0 {
                bad += 1;
                println!("[W0] row {r}: {diffs}/{n_vocab} logits differ");
            }
        }
        v.arm(
            "W0 plain-ppn-decode",
            bad == 0,
            &format!(
                "{} rows bit-identical vs door-OFF (n_vocab={n_vocab})",
                vt.len()
            ),
        );
    }

    // ---- arm P0: ppN PRIME DETERMINISM (lane/glm5-accrace) ----
    // THE ARM THAT WAS MISSING. The acceptance-race defect was a stage-stream publication
    // hole whose FIRST observable effect was that the split PRIME of a fixed prompt stopped
    // being a function of its inputs: repeated primes in ONE process returned three
    // different logit rows, all with the same argmax, and the drift only became visible
    // many rounds later as one silently lost spec acceptance. Every arm in this gate
    // compares a door-ON walk against a door-OFF walk ONCE, so none of them could see it.
    //
    // TWO PASSES, AND THE SECOND ONE IS THE TEETH — measured, not assumed. `P0` here runs
    // with nothing yet in flight on the stage streams, which is the QUIET regime: in the
    // hunt the FIRST prime of a process was canonical in 14/14 reps even while later ones
    // drifted, and a first pass of this arm went 0/96 deviations on a tree with the defect
    // present. `P1` below re-runs the same census AFTER arm E, where the spec sessions have
    // left rollback and teardown tails queued on the stage streams — that is the regime the
    // race lives in. Keep both: P0 is the cheap always-on canary, P1 is the detector.
    eprintln!("[phase] arm P0: door ON, prime determinism x{prime_reps} (quiet regime)");
    {
        let mut deviated = 0usize;
        let mut worst = 0usize;
        for r in 0..prime_reps {
            let (_c, l) = fresh_primed(&e, &m, &plan, &prompt, max_ctx);
            let d = bit_diffs(&l, &ref_prime);
            if d > 0 {
                deviated += 1;
                worst = worst.max(d);
                println!(
                    "[P0] prime rep {r}: {d}/{} logits differ from the door-OFF prime",
                    ref_prime.len()
                );
            }
        }
        v.arm(
            "P0 prime-determinism",
            deviated == 0,
            &format!(
                "{prime_reps} split primes of one prompt vs the door-OFF prime: \
                 {deviated} deviated (worst {worst}/{} logits)",
                ref_prime.len()
            ),
        );
    }

    // ---- arm W1: THE WALK under the split ----
    eprintln!("[phase] arm W1: door ON, verify walk (ppN twin)");
    {
        let (mut cache, _l) = fresh_primed(&e, &m, &plan, &prompt, max_ctx);
        let (vlogits, _collapsed, _ckpt) = m.glm5_verify_rows(&e, &vt, &mut cache)?;
        let host = e.dtoh(&vlogits)?;
        let mut bad = 0usize;
        for (r, plain) in ref_rows.iter().enumerate() {
            let diffs = bit_diffs(&host[r * n_vocab..(r + 1) * n_vocab], plain);
            if diffs > 0 {
                bad += 1;
                println!("[W1] row {r}: {diffs}/{n_vocab} logits differ");
            }
        }
        v.arm(
            "W1 verify-walk",
            bad == 0,
            &format!(
                "{} verify rows bit-identical to plain decode under the split (and to \
                 plain ppN decode via W0)",
                vt.len()
            ),
        );
    }

    // ---- arm A: accept-j-then-continue under the split, every j ----
    eprintln!("[phase] arm A: door ON, accept-j rollback battery");
    #[allow(clippy::needless_range_loop)]
    // allow: j is the accept depth (drives keep AND indexes its own reference chain); an
    // enumerate over ref_cont would hide that pairing
    for j in 0..=K {
        let keep = j + 1;
        let (mut cache, _l) = fresh_primed(&e, &m, &plan, &prompt, max_ctx);
        let pos0 = cache.pos;
        let (_vl, _coll, ckpt) = m.glm5_verify_rows(&e, &vt, &mut cache)?;
        m.glm5_verify_rollback(&e, &mut cache, &ckpt, keep)?;
        assert_eq!(
            cache.pos,
            pos0 + keep,
            "rollback must land pos at snap+keep"
        );
        let mut bad = 0usize;
        for (step, &tok) in cc.iter().enumerate() {
            let got = m.decode_step(&e, tok, &mut cache)?;
            let diffs = bit_diffs(&got, &ref_cont[j][step]);
            if diffs > 0 {
                bad += 1;
                println!("[A] j={j} continue step {step}: {diffs}/{n_vocab} logits differ");
            }
        }
        v.arm(
            &format!("A accept-j={j}"),
            bad == 0,
            &format!("{} continue steps bit-identical", cc.len()),
        );
    }

    // ---- arm E: end-to-end tapes under the split ----
    eprintln!("[phase] arm E: door ON, e2e spec-vs-plain tapes");
    for k in 1..=K {
        let (out, drafted, accepted) = m.generate_spec_glm5(&e, &prompt, n, k)?;
        v.arm(
            &format!("E natural K={k}"),
            out == tape,
            &tape_verdict("natural drafter", &out, &tape, accepted, drafted),
        );
    }
    for k in [3usize, K] {
        let tape_for_override = tape.clone();
        let mut over = move |round: usize, ki: usize, _greedy: u32| -> u32 {
            let cursor = 1 + round * (k + 1);
            let pos = cursor + ki;
            if pos < tape_for_override.len() {
                tape_for_override[pos]
            } else {
                0
            }
        };
        let (out, drafted, accepted) = m.generate_spec_glm5_gated(
            &e,
            &prompt,
            n,
            k,
            Glm5SpecKnobs {
                draft_override: Some(&mut over),
                ..Default::default()
            },
        )?;
        let plumbing_live = accepted * 2 >= drafted;
        v.arm(
            &format!("E forced-accept K={k}"),
            out == tape && plumbing_live,
            &format!(
                "{} (full-accept path {})",
                tape_verdict("forced full accept", &out, &tape, accepted, drafted),
                if plumbing_live {
                    "exercised"
                } else {
                    "NOT exercised — accepted*2 < drafted"
                }
            ),
        );
    }
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
                (correct + 1) % VOCAB
            }
        };
        let (out, drafted, accepted) = m.generate_spec_glm5_gated(
            &e,
            &prompt,
            n,
            k,
            Glm5SpecKnobs {
                draft_override: Some(&mut over),
                accept_probe: probe,
                ..Default::default()
            },
        )?;
        // The detail string REPORTS THE COMPARISON IT RAN (lane/glm5-accrace): the old
        // format said "tape identical" on the FAIL line too, so a real red read like a
        // green one — and a FAIL that describes a PASS is how a red gets waved past. On a
        // divergence it names the first differing index and both tokens, which is the
        // anchor a bisect needs.
        v.arm(
            &format!("E forced-rejection sweep K={k}"),
            out == tape,
            &tape_verdict(
                "every partial-keep rollback cycled",
                &out,
                &tape,
                accepted,
                drafted,
            ),
        );
    }

    // ---- arm P1: PRIME DETERMINISM AFTER THE SPEC ARMS (lane/glm5-accrace) ----
    // The same census as P0, in the regime that actually carries the defect: arm E's ten
    // spec sessions have just run, so every stage stream holds the tail of a per-stage
    // rollback and a torn-down stage-owned cache. That is where the exit-publication hole
    // let a caller allocation land under queued stage work, and it is why this arm — not P0
    // and not any tape — is the detector. Under the lane's loaded protocol the pre-fix tree
    // put roughly one prime in three off the door-OFF value here while every argmax held.
    eprintln!("[phase] arm P1: door ON, prime determinism x{prime_reps} AFTER the spec arms");
    {
        let mut deviated = 0usize;
        let mut worst = 0usize;
        for r in 0..prime_reps {
            let (_c, l) = fresh_primed(&e, &m, &plan, &prompt, max_ctx);
            let d = bit_diffs(&l, &ref_prime);
            if d > 0 {
                deviated += 1;
                worst = worst.max(d);
                println!(
                    "[P1] prime rep {r}: {d}/{} logits differ from the door-OFF prime",
                    ref_prime.len()
                );
            }
        }
        v.arm(
            "P1 prime-determinism-post-spec",
            deviated == 0,
            &format!(
                "{prime_reps} split primes after the spec arms vs the door-OFF prime: \
                 {deviated} deviated (worst {worst}/{} logits)",
                ref_prime.len()
            ),
        );
    }

    // ---- arm R1: RED stale-KDA under the split ----
    // The mutation clones/reinstates state through the OWNING stage engines inside stage
    // scopes, so the writes are stream-ordered exactly like the engine's own rollback.
    eprintln!("[phase] arm R1: RED stale-KDA state");
    {
        let j = 2usize;
        let keep = j + 1;
        let (mut cache, _l) = fresh_primed(&e, &m, &plan, &prompt, max_ctx);
        let (_vl, _coll, ckpt) = m.glm5_verify_rows(&e, &vt, &mut cache)?;
        let streams_off = memra_engine::pp::pp2_streams_off();
        let rt = if streams_off {
            None
        } else {
            Some(memra_engine::pp::PpNRt::get(&e)?)
        };
        #[allow(clippy::type_complexity)]
        // allow: one-shot closure contract; naming it would hide the stage-scope shape
        let with_stage_engine =
            |il: usize,
             f: &mut dyn FnMut(&Engine) -> Result<(), Box<dyn std::error::Error>>|
             -> Result<(), Box<dyn std::error::Error>> {
                match rt {
                    Some(rt) => {
                        let s = memra_engine::pp::stage_of(&fence, il);
                        let _g = rt.enter(s);
                        f(rt.engine(s, &e))
                    }
                    None => f(&e),
                }
            };
        // Clone the post-row-K resident state (walk drained by the walk's own contract).
        let mut stale: Vec<
            Option<(
                cudarc::driver::CudaSlice<f32>,
                cudarc::driver::CudaSlice<f32>,
            )>,
        > = Vec::new();
        for il in 0..n_layers {
            let mut cloned = None;
            if cache.recur[il].is_some() {
                with_stage_engine(il, &mut |es| {
                    let rl = cache.recur[il].as_ref().unwrap();
                    cloned = Some((
                        es.clone_dtod(&rl.conv_state)?,
                        es.clone_dtod(&rl.ssm_state)?,
                    ));
                    Ok(())
                })?;
            }
            stale.push(cloned);
        }
        m.glm5_verify_rollback(&e, &mut cache, &ckpt, keep)?;
        // THE MUTATION: reinstate post-row-K state (a rollback that forgot the columns).
        let mut mutated = 0usize;
        for (il, s) in stale.into_iter().enumerate() {
            if let Some((conv, ssm)) = s {
                with_stage_engine(il, &mut |es| {
                    let rl = cache.recur[il].as_mut().unwrap();
                    es.copy_into(&mut rl.conv_state, 0, &conv, conv.len())?;
                    es.copy_into(&mut rl.ssm_state, 0, &ssm, ssm.len())?;
                    Ok(())
                })?;
                mutated += 1;
            }
        }
        assert!(
            mutated > 0,
            "the mutation must touch at least one KDA layer"
        );
        let mut diffs_total = 0usize;
        for (step, &tok) in cc.iter().enumerate() {
            let got = m.decode_step(&e, tok, &mut cache)?;
            diffs_total += bit_diffs(&got, &ref_cont[j][step]);
        }
        v.arm(
            "R1 stale-KDA RED",
            diffs_total > 0,
            &format!("bites — {diffs_total} differing logits across the continuation"),
        );
    }

    // ---- arm R2: RED pool-keys finalized past j ----
    eprintln!("[phase] arm R2: RED pool-key clamp");
    {
        let keep = 1usize; // maximal clamp movement (the tparallel gate's choice)
        let (mut cache, _l) = fresh_primed(&e, &m, &plan, &prompt, max_ctx);
        let (_vl, _coll, ckpt) = m.glm5_verify_rows(&e, &vt, &mut cache)?;
        let pre: Vec<Option<usize>> = cache
            .latent
            .iter()
            .take(n_layers)
            .map(|p| p.as_ref().map(|p| p.index_pools_ready))
            .collect();
        m.glm5_verify_rollback(&e, &mut cache, &ckpt, keep)?;
        let mut mutated = 0usize;
        #[allow(clippy::needless_range_loop)]
        // allow: il indexes BOTH cache.latent and the pre snapshot; zipping would hide the pairing
        for il in 0..n_layers {
            if let (Some(plane), Some(ready)) = (cache.latent[il].as_mut(), pre[il])
                && ready > plane.index_pools_ready
            {
                plane.index_pools_ready = ready;
                mutated += 1;
            }
        }
        assert!(
            mutated > 0,
            "the walk+rollback did not move index_pools_ready — this red arm is vacuous"
        );
        match m.decode_step(&e, tokens(1, 0xD00D)[0], &mut cache) {
            Err(err) => {
                let msg = err.to_string();
                v.arm(
                    "R2 pool-key RED",
                    msg.contains("index_pools_ready"),
                    &format!("bites by name — {msg}"),
                );
            }
            Ok(_) => v.arm(
                "R2 pool-key RED",
                false,
                "continuing over keys finalized past j did NOT fail — the tripwire is dead \
                 under the split",
            ),
        }
    }

    // ---- arm R3: RED rollback disabled ----
    eprintln!("[phase] arm R3: RED rollback disabled");
    {
        let mut over = |_round: usize, ki: usize, greedy: u32| -> u32 {
            if ki == 0 {
                (greedy + 1) % VOCAB
            } else {
                greedy
            }
        };
        match m.generate_spec_glm5_gated(
            &e,
            &prompt,
            n,
            K,
            Glm5SpecKnobs {
                draft_override: Some(&mut over),
                disable_rollback: true,
                ..Default::default()
            },
        ) {
            Ok((out, drafted, accepted)) => {
                let detail = match first_diff(&out, &tape) {
                    Some((i, got, want)) => format!(
                        "bites — tape diverged with rollback disabled at index {i} \
                         (got {got:?}, want {want:?}) ({accepted}/{drafted})"
                    ),
                    None => format!(
                        "does NOT bite — the tape stayed BYTE-IDENTICAL with rollback \
                         disabled, so the red arm proves nothing ({accepted}/{drafted})"
                    ),
                };
                v.arm("R3 rollback-disabled RED", out != tape, &detail);
            }
            Err(err) => v.arm(
                "R3 rollback-disabled RED",
                true,
                &format!("bites — loop failed loudly: {err}"),
            ),
        }
    }

    println!("==========================================================");
    if v.fails == 0 {
        println!("glm5-spec-ppn gate: ALL ARMS PASS (P={p} N={n} K={K}, fence={fence:?}; {knobs})");
        Ok(())
    } else {
        println!(
            "glm5-spec-ppn gate: {} ARM(S) FAILED (fence={fence:?}; {knobs})",
            v.fails
        );
        std::process::exit(1);
    }
}
