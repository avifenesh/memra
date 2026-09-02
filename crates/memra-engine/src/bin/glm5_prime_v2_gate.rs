//! `glm5-prime-v2-gate`: the acceptance gate for `MEMRA_B200_PRIME_V2`, the B200 mHC PRIME
//! schedule door (lane/b200-prefill-roofline-20260902).
//!
//! WHY THE DOOR EXISTS, in one paragraph so this file stands alone. `prime_chunk_tokens`
//! applies the `PRIME_PIPE_MICROBATCHES = 8` geometry whenever a PP-2 stage fence exists, so a
//! 4096-token glm5 prime splits into EIGHT calls. That geometry belongs to
//! `prime_cache_pp2_pipelined`, the SERIAL trunk's microbatched PP-2 prime, which the mHC walk
//! has never called: glm5 pays the split and collects none of the overlap. Measured on the 2x
//! B200 pair 2026-09-02, the split costs about 2x per token in the grouped MoE prefill
//! (7.3-8.3 us per token per MoE layer at the ~900-token microchunks the 4k rung produced,
//! against 4.1 us at ~4000-token chunks on the same binary and the same boot), and the MoE is
//! 5.09 s of that rung's 5.5 s TTFT. Arm 1 removes the split; arm 2 gives the walk the overlap
//! the geometry was always meant to buy.
//!
//! TWO ARMS, TWO DIFFERENT BARS, DELIBERATELY SEPARATED. Collapsing them into one tolerance is
//! how a scheduling bug hides inside a numeric band.
//!
//! * ARM 1 — SCHEDULE (`hyper_prime_ranges`). Door OFF (the microbatch split) vs door ON (one
//!   natural chunk) over the same prompt. The bar is the CALIBRATED NEAR-TIE BAND
//!   `tests/glm5_chunked_prime_gpu.rs` already holds the chunked prime to (relative maxdiff
//!   <= 2e-5) plus argmax equality, NOT bit identity — and no chunk-size change on this trunk
//!   ever was bit-identical: `Engine::linear`, the cuBLASLt f32 mixes GEMM in `hyper::pre`, is
//!   not m-invariant, and the arms diverge at ROW 0 where no cross-token state can reach.
//!   `hyper_prime_ranges`' own header documents that near-tie. The band sits five orders below
//!   the 1.813e0 signature of a real chunk-invariance defect, and ~35x below what losing
//!   `NVIDIA_TF32_OVERRIDE=0` costs, so a rig that lost it fails here rather than passing under
//!   a widened bar.
//! * ARM 2 — PIPELINE (`prime_cache_hyper_pp2_pipelined`). Both arms run the door ON with an
//!   EXPLICIT `MEMRA_PRIME_CHUNK`, so the two walks see byte-for-byte the same ranges and the
//!   only axis under test is whether stage 0 of chunk k+1 is issued from a second host thread
//!   while stage 1 of chunk k runs. That is a scheduling change and nothing else — same kernels,
//!   same operand bytes, same stage engines, an exact copy at the boundary — so the bar is BIT
//!   IDENTITY on the full logit row and on the whole hidden stack. Any differing bit is a seam
//!   bug.
//!
//! NON-VACUITY IS ENFORCED, NOT ASSUMED (the wiring-assertions-match-prose law, and the
//! checks-need-a-red-arm law). Each arm asserts that the thing it claims to test actually
//! changed:
//!
//! * arm 1 FAILS if the door did not change the chunk count. Two identical schedules compared
//!   bit-for-bit would pass while proving nothing, and that is exactly the shape this lane's
//!   roofline caught in the shipped code (a microbatch geometry armed for a walk that never
//!   used it, invisible from outside);
//! * arm 2 FAILS if `HYPER_PRIME_PIPELINED_CHUNKS` did not advance by the number of chunks the
//!   schedule says it should. Bit identity between two walks that BOTH ran serially is vacuous,
//!   and the pipelined body declines by name on several shapes (overlay present, hc tap sink
//!   armed, `MEMRA_PRIME_PIPE=0`) — a decline would otherwise read as a pass;
//! * both arms assert the trunk really is a HyperConnections plan carrying BOTH per-layer state
//!   classes (a `Recurrent` KDA layer and a `LatentKvCache` MLA+kpool layer) on DIFFERENT
//!   stages. A fence that kept every stateful layer on one stage would pass while proving
//!   nothing about the per-stage cache split the pipelined body depends on.
//!
//! WHAT THE COMPARISON PROVES, AND HOW THE TRUTH CHAIN CLOSES (GATE:pin-against-truth). Both
//! arms are ARM-EQUALITY, not truth: an error in the hc arithmetic itself cancels. The chain
//! closes only by composition, and all three halves must be cited together:
//!
//! * `tests/hyper_connections_gpu.rs` anchors the unsplit hc walk to `memra_reference`
//!   (a host executor sharing no GPU code) on this fixture family;
//! * `glm5-hyper-ppn-gate` anchors the SPLIT walk to the unsplit one, bit for bit;
//! * this gate anchors the RESCHEDULED and the PIPELINED walks to the walk those two pinned.
//!
//! RED ARM. `MEMRA_PRIME_V2_GATE_RED=schedule` forces both arm-1 walks onto the SAME schedule:
//! the door then changes nothing and the non-vacuity assertion must fire. `=pipe` runs arm 2
//! with `MEMRA_PRIME_PIPE=0` on both sides, so no chunk is pipelined and arm 2's counter
//! assertion must fire. A runner banks both exit codes; a gate whose red arm has never been run
//! is a PASS with no caller.
//!
//! SCOPE — what this does NOT prove. F32 fixture weights at HIDDEN=128, no quantized expert
//! class (that is `glm5_moe_residency_gpu` and `glm5_moe_grouped_prefill_gpu`), no throughput
//! claim of any kind, and no evidence about the 190.7 GB artifact. Cross-DEVICE overlap — the
//! entire point of arm 2 on the pair — needs a multi-card box (`MEMRA_PP_DEVICES=0,1`); on a
//! one-card rig the same-device two-stream arm still exercises the whole pipelined scheduler,
//! the two host threads, the per-stage cache split and the boundary transport, which is the
//! correctness half. The SPEED half is a box A/B and is not this gate's business.
//!
//! Rig law: exactness only, never a timing number. The per-phase walls are printed ONLY under
//! `MEMRA_PRIME_PROF=1`, are sync-bounded (each mark drains the stage stream, so absolute time
//! inflates), and are ATTRIBUTION on a 128-wide fixture — never a performance claim.
//!
//! usage: glm5-prime-v2-gate [T1=4096] [T2=8192] [CHUNK2=4096]
//!   run under `flock /tmp/memra-5090.lock` with `NVIDIA_TF32_OVERRIDE=0`.
use memra_engine::Engine;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::hybrid_forward::{HYPER_PRIME_PIPELINED_CHUNKS, hyper_prime_ranges};
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
use std::sync::atomic::Ordering;

const HIDDEN: usize = 128;
const VOCAB: u32 = 32;
const LAYERS: usize = 4;

/// The calibrated chunked-prime near-tie band, quoted from `tests/glm5_chunked_prime_gpu.rs`
/// rather than re-derived: the worst measured chunked-vs-monolithic divergence on this fixture
/// family is 3.815e-6 on unit-scale activations, so 2e-5 carries about 5x headroom while
/// staying five orders BELOW the 1.813e0 signature of a real chunk-invariance defect.
/// Calibrate downward, never upward.
const TOL: f32 = 2e-5;

fn mini_config_json() -> String {
    r#"{
      "model_type": "glm5_next_text",
      "num_hidden_layers": 4,
      "num_nextn_predict_layers": 0,
      "hidden_size": 128,
      "intermediate_size": 64,
      "vocab_size": 32,
      "max_position_embeddings": 16384,
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

/// Worst absolute difference relative to the reference's own scale — the same measure
/// `glm5_chunked_prime_gpu` uses, so the two receipts compose on one number.
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

/// Count of differing bits, so a BIT arm reports what actually moved rather than a tolerance.
fn bit_mismatches(got: &[f32], want: &[f32]) -> usize {
    assert_eq!(got.len(), want.len(), "compared slices differ in length");
    got.iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

/// One prime through the mHC walk on a FRESH cache, with the door state the caller has already
/// set. Returns the last-row logits and the whole hidden stack: the stack is what sees a defect
/// that the last row averages away, and a chunk-boundary bug lives in the middle rows.
fn prime_once(
    e: &Engine,
    m: &HybridModel,
    plan: &ModelPlan,
    ids: &[u32],
    max_ctx: usize,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let mut cache = memra_engine::pp::new_cache_planned(e, &m.cfg, plan, max_ctx)?;
    let started = std::time::Instant::now();
    let (logits, _seed, hiddens) = m.prime_cache(e, ids, &mut cache, 0)?;
    let stack = e.dtoh(&hiddens)?;
    if std::env::var("MEMRA_PRIME_PROF").as_deref() == Ok("1") {
        // Attribution only, on a 128-wide fixture, with sync-bounded phase marks upstream.
        // NOT a rig timing number and never quotable as one (rig-gpu-exactness-only).
        eprintln!(
            "[prime-v2-gate] attribution-only wall {:.1} ms for t={} (128-wide FIXTURE; \
             not a performance claim)",
            started.elapsed().as_secs_f64() * 1e3,
            ids.len()
        );
    }
    assert_eq!(
        cache.pos,
        ids.len(),
        "the prime must leave the cache at the prompt length"
    );
    Ok((logits, stack))
}

/// SAFETY contract for every env mutation in this file: single-threaded, and either before any
/// engine exists or between two completed primes with no other thread running. The doors are
/// read PER CALL by design, which is what makes both arms runnable in one process.
fn set_env(key: &str, value: &str) {
    unsafe { std::env::set_var(key, value) };
}
fn clear_env(key: &str) {
    unsafe { std::env::remove_var(key) };
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t1: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let t2: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192);
    let chunk2: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let red = std::env::var("MEMRA_PRIME_V2_GATE_RED").unwrap_or_default();

    // cuBLASLt f32 rides TF32 on Blackwell by default — wrong for an exactness gate. Must
    // precede the first Engine::new in the process.
    if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
        set_env("NVIDIA_TF32_OVERRIDE", "0");
    }
    // The door's arms only EXIST under a PP-2 fence: `hyper_prime_ranges` inherits the
    // microbatch geometry from `prime_pp2_auto_geometry`, and the pipelined body needs two
    // stages to overlap. Set for the whole process, before any runtime is built.
    set_env("MEMRA_PP_STAGES", "2");
    clear_env("MEMRA_B200_PRIME_V2");
    clear_env("MEMRA_PRIME_CHUNK");
    clear_env("MEMRA_PRIME_PIPE");

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

    // ---- NON-VACUITY: the trunk, the fence, and the two state classes ----
    let topology = m.hyper.as_ref().expect(
        "the fixture must load as a HyperConnections trunk — otherwise this gate measures a \
         schedule the door does not steer",
    );
    let fence = memra_engine::pp::pp_cuts(LAYERS)
        .expect("MEMRA_PP_STAGES=2 must open a two-stage fence over the fixture trunk");
    assert_eq!(
        fence.len(),
        3,
        "this gate's arms are PP-2 arms: fence {fence:?}"
    );
    let (mut recur_stage, mut latent_stage) = (None, None);
    for layer in &plan.layers {
        let il = layer.index as usize;
        let stage = usize::from(il >= fence[1]);
        match layer.state {
            StatePlan::Recurrent { .. } => recur_stage = Some(stage),
            StatePlan::LatentKvCache { .. } => latent_stage = Some(stage),
            _ => {}
        }
    }
    assert!(
        recur_stage.is_some() && latent_stage.is_some(),
        "the fixture must carry BOTH a Recurrent (KDA) and a LatentKvCache (MLA+kpool) layer"
    );
    assert_ne!(
        recur_stage, latent_stage,
        "the fence must SEPARATE the two per-layer state classes (recur stage {recur_stage:?}, \
         latent stage {latent_stage:?}); a fence that keeps both on one stage would pass while \
         proving nothing about the per-stage cache split the pipelined body depends on"
    );
    println!(
        "glm5-prime-v2-gate config: T1={t1} T2={t2} CHUNK2={chunk2} fence={fence:?} \
         streams={} collapse={:?} sinkhorn={} red={}",
        topology.streams,
        topology.collapse,
        topology.sinkhorn_iterations,
        if red.is_empty() { "none" } else { red.as_str() },
    );

    let mut failures: Vec<String> = Vec::new();

    // ================= ARM 1: the schedule (near-tie band + argmax) =================
    {
        let ids = tokens(t1, 0x5EED_0001);
        let max_ctx = t1 + 16;

        clear_env("MEMRA_B200_PRIME_V2");
        let shipped_ranges = hyper_prime_ranges(t1, LAYERS, m.gdn_prime_grid_on());
        let (ref_logits, ref_stack) = prime_once(&e, &m, &plan, &ids, max_ctx)?;

        // RED `schedule`: leave the door shut for the "on" walk too, so the schedules match and
        // the non-vacuity assertion below MUST fire.
        if red != "schedule" {
            set_env("MEMRA_B200_PRIME_V2", "1");
        }
        let door_ranges = hyper_prime_ranges(t1, LAYERS, m.gdn_prime_grid_on());
        let (got_logits, got_stack) = prime_once(&e, &m, &plan, &ids, max_ctx)?;
        clear_env("MEMRA_B200_PRIME_V2");

        println!(
            "arm 1 SCHEDULE  t={t1}: shipped {} chunks {:?} -> door {} chunks {:?}",
            shipped_ranges.len(),
            shipped_ranges
                .iter()
                .map(|&(s, x)| x - s)
                .collect::<Vec<_>>(),
            door_ranges.len(),
            door_ranges.iter().map(|&(s, x)| x - s).collect::<Vec<_>>(),
        );
        if shipped_ranges.len() == door_ranges.len() {
            failures.push(format!(
                "arm 1 VACUOUS: the door left the chunk count at {} — comparing two identical \
                 schedules proves nothing (this is the exact shape the lane's roofline caught: \
                 an armed geometry that changed nothing and was invisible from outside)",
                door_ranges.len()
            ));
        }
        let rel_logits = relative(&got_logits, &ref_logits);
        let rel_stack = relative(&got_stack, &ref_stack);
        let arg_ref = argmax(&ref_logits);
        let arg_got = argmax(&got_logits);
        println!(
            "arm 1 SCHEDULE  logits rel {rel_logits:.3e} stack rel {rel_stack:.3e} \
             (band {TOL:.1e}) argmax {arg_ref} vs {arg_got}"
        );
        if !got_logits.iter().all(|v| v.is_finite()) || !got_stack.iter().all(|v| v.is_finite()) {
            failures.push("arm 1: the door walk produced non-finite values".to_string());
        }
        if rel_logits > TOL || rel_stack > TOL {
            failures.push(format!(
                "arm 1: outside the near-tie band — logits {rel_logits:.3e}, stack \
                 {rel_stack:.3e}, band {TOL:.1e}"
            ));
        }
        if arg_ref != arg_got {
            failures.push(format!("arm 1: argmax moved {arg_ref} -> {arg_got}"));
        }
    }

    // ================= ARM 2: the pipeline (BIT identity) =================
    {
        let ids = tokens(t2, 0x5EED_0002);
        let max_ctx = t2 + 16;
        // Both walks take the door AND an explicit chunk, so the ranges are byte-identical and
        // the ONLY axis under test is the two-host-thread overlap.
        set_env("MEMRA_B200_PRIME_V2", "1");
        set_env("MEMRA_PRIME_CHUNK", &chunk2.to_string());
        let ranges = hyper_prime_ranges(t2, LAYERS, m.gdn_prime_grid_on());
        assert!(
            ranges.len() >= 2,
            "arm 2 needs at least two chunks to overlap: T2={t2} CHUNK2={chunk2} gave \
             {} range(s). Raise T2 or lower CHUNK2.",
            ranges.len()
        );

        set_env("MEMRA_PRIME_PIPE", "0");
        let serial_before = HYPER_PRIME_PIPELINED_CHUNKS.load(Ordering::Relaxed);
        let (ref_logits, ref_stack) = prime_once(&e, &m, &plan, &ids, max_ctx)?;
        let serial_chunks = HYPER_PRIME_PIPELINED_CHUNKS.load(Ordering::Relaxed) - serial_before;

        // RED `pipe`: keep the pipeline shut for the "on" walk too — nothing is pipelined and
        // the counter assertion below MUST fire.
        if red != "pipe" {
            clear_env("MEMRA_PRIME_PIPE");
        }
        let pipe_before = HYPER_PRIME_PIPELINED_CHUNKS.load(Ordering::Relaxed);
        let (got_logits, got_stack) = prime_once(&e, &m, &plan, &ids, max_ctx)?;
        let pipe_chunks = HYPER_PRIME_PIPELINED_CHUNKS.load(Ordering::Relaxed) - pipe_before;
        clear_env("MEMRA_PRIME_PIPE");
        clear_env("MEMRA_PRIME_CHUNK");
        clear_env("MEMRA_B200_PRIME_V2");

        let bad_logits = bit_mismatches(&got_logits, &ref_logits);
        let bad_stack = bit_mismatches(&got_stack, &ref_stack);
        println!(
            "arm 2 PIPELINE  t={t2} chunks={} ({:?}): serial pipelined-count {serial_chunks}, \
             door pipelined-count {pipe_chunks}; bit mismatches logits {bad_logits}/{} stack \
             {bad_stack}/{}",
            ranges.len(),
            ranges.iter().map(|&(s, x)| x - s).collect::<Vec<_>>(),
            got_logits.len(),
            got_stack.len(),
        );
        if serial_chunks != 0 {
            failures.push(format!(
                "arm 2 CONTROL BROKEN: MEMRA_PRIME_PIPE=0 still pipelined {serial_chunks} \
                 chunk(s) — the reference arm is not the serial walk it claims to be"
            ));
        }
        if pipe_chunks != ranges.len() as u64 {
            failures.push(format!(
                "arm 2 VACUOUS: the pipelined body ran {pipe_chunks} chunk(s), schedule says \
                 {} — bit identity between two SERIAL walks proves nothing, and the body \
                 declines by name on several shapes (overlay, armed hc tap, MEMRA_PRIME_PIPE=0); \
                 check stderr for a [hyper-prime-pipe] DECLINED line",
                ranges.len()
            ));
        }
        if bad_logits != 0 || bad_stack != 0 {
            let rel_l = relative(&got_logits, &ref_logits);
            let rel_s = relative(&got_stack, &ref_stack);
            failures.push(format!(
                "arm 2: the pipelined walk is NOT bit-identical — {bad_logits} logit bits and \
                 {bad_stack} stack bits differ (relative {rel_l:.3e} / {rel_s:.3e}). The \
                 schedule is unchanged between these two walks, so this is a seam bug, not a \
                 numeric class"
            ));
        }
    }

    if failures.is_empty() {
        println!("glm5-prime-v2-gate: PASS (arm 1 near-tie band + argmax, arm 2 bit-identical)");
        Ok(())
    } else {
        for f in &failures {
            eprintln!("glm5-prime-v2-gate FAIL: {f}");
        }
        Err(format!("glm5-prime-v2-gate: {} failure(s)", failures.len()).into())
    }
}
