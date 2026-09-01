//! glm5_next TP-2 gate (`MEMRA_GLM5_TP`, lane/glm5-tp2): the head-sharded / EP-2 execution
//! program against the door-OFF plain walk, per layer-class arm and at the model level, on
//! the mini glm5_next hc fixture — BYTE identity at decode widths, a calibrated band at
//! prime widths (the two-regime bar below).
//!
//! THE TWO-REGIME BAR, measured on this gate's own first runs and matching the house's
//! MEMRA_PRIME_CHUNK precedent:
//!
//!   * DECODE (t=1, the product surface of the TP-2 lane): BYTE IDENTITY. The TP program is
//!     column-parallel-over-gather — every arithmetic op runs the SAME per-row kernel over
//!     the SAME values as the plain walk; only data movement differs; the ONE cross-rank
//!     arithmetic site (the MoE slot-ordered axpy combine) reproduces the plain fmaf chain
//!     operation for operation. Any differing bit at t=1 is a seam bug.
//!   * PRIME (t>=2): a DOCUMENTED NEAR-TIE CLASS, band + tape bar. Batched GEMM widths
//!     (cuBLASLt f32/f16, MMQ tiles) select shape-dependent K-reduction splits, so a
//!     128-row shard and its 256-row full tensor legally differ by ulps for the same
//!     logical row — the same class as `Engine::linear`'s documented m-dependence
//!     (FLAGS.md MEMRA_PRIME_CHUNK; measured here before the bar was set: 1-ulp logit
//!     diffs, tape identical). Bar: max relative diff <= 2e-5 (the chunked-prime band)
//!     AND tape identity AND repetition byte-identity; reds must exceed the band by
//!     orders or break the tape.
//!
//! THE TRUTH CHAIN (GATE:pin-against-truth): split-vs-plain alone is arm-equality. It closes
//! by COMPOSITION: `tests/hyper_connections_gpu.rs` anchors the UNSPLIT hc walk to
//! `memra_reference` on this fixture family; THIS gate anchors the TP-2 walk to the unsplit
//! one, bit for bit. State both when citing this gate.
//!
//! ARMS (one process; the TP door is per-model-load, so plain and TP models coexist):
//!   A. PLAIN reference — door OFF: prime P, then teacher-forced decode of N steps, full
//!      logits + greedy tape banked.
//!   B. TP-ALL — `all@0,1` (same-device dual-context emulation on the one-card rig): the
//!      whole trunk sharded (KDA + MLA mixers, EP on every sparse layer). Bit compare per
//!      step + tape. TWO REPETITIONS (fresh cache each) — both must match (the hy3 dense-gate
//!      repetition discipline).
//!   C. TP-KDA-ONLY — the KDA layers alone (mixed-owner layout; MLA layers stay plain).
//!   D. TP-MLA-ONLY — the MLA layers alone.
//!   E. STATELESS-POISON — `forward` on the TP model must REFUSE by name (the plain-path
//!      choke points hold).
//!   F. SPEC-CO-REFUSAL — `glm5_spec_session_new` on the TP model must refuse by name.
//!   G. PP-COMPOSITION-REFUSAL — `MEMRA_PP_STAGES=2` + the TP door must refuse AT LOAD.
//!   M. MEASURED-MAP SKEW — `MEMRA_GLM5_EP_MAP` with a deliberately skewed placement
//!      (all-but-one expert on rank 0; the rank-1 singleton chosen from the probe walk's
//!      ROUTED union so the arm cannot be vacuous): decode BYTE-identical to plain, prime
//!      in band — the placement-independence-by-construction proof
//!      (LAW:coactivation-expert-placement's engine leg).
//!   H1-H5. MAP FAIL-CLOSED — missing file / wrong assignment length / missing layer
//!      row / wrong entry_rank all refuse at load BY NAME; the map with the TP door
//!      COLD refuses at load (the silent-even-split trap).
//!   T. TRACE-TAP IDENTITY — `MEMRA_MOE_WEIGHT_TRACE` (the fleet's co-activation
//!      measurement tap) armed on the PLAIN walk: both regimes byte-identical to the
//!      banked references, and the trace rows are COUNTED against the walk arithmetic
//!      (token events per MoE layer == primes + decodes, exactly).
//!   R1. RED swap-wo — each rank holds the other rank's wo out rows: MUST diverge.
//!   R2. RED swap-ep-gateup — root EP slab's gate/up swapped: MUST diverge.
//!   R3. RED skip-peer-combine — peer-owned expert slots dropped: MUST diverge (this is also
//!      the non-vacuity proof that the peer rank contributes real work).
//!   R4. RED corrupt-ep-map — the armed map's rank-0 local-slot table reversed after slab
//!      packing (owner table and slab bytes disagree — a corrupted map row): MUST diverge.
//!
//! NON-VACUITY IS ENFORCED: the TP model's layers must actually carry shards (tp sidecars +
//! EP arms counted against the spec), the fixture must carry BOTH mixer classes and a
//! routed-expert layer, and the reds must bite.
//!
//! SCOPE — what this does NOT prove: F32/Q8_0 fixture classes only (the real artifact's
//! BF16-resident/NVFP4 classes ride the same shard mechanics but carry their own box gate);
//! host-canonical transport only (real peer transport is qualified by the pro6000 batteries
//! on the box card class); B=1 only (batched TP is refused by construction); no throughput
//! claim of any kind.
//!
//! Rig law: exactness only. Run under `flock /tmp/memra-5090.lock` with
//! `NVIDIA_TF32_OVERRIDE=0`.
//!
//! usage: glm5-tp-gate [P=16] [N=12] [TRACE_OUT (preserve arm T's fixture weight-trace)]

use memra_engine::Engine;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::ModelPlan;
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

/// The hyper/spec ppN gates' mini glm5_next, with the ONE change this gate needs: TWO KDA
/// heads and TWO MLA heads, so a 2-rank head shard is non-trivial on both mixer classes
/// (heads_per_rank = 1). KDA head_dim stays 128 (`memra_kda_scan_s128` is width-pinned);
/// the indexer keeps ONE head (it is REPLICATED, never sharded, per the shard map).
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
        "num_heads": 2,
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

/// The big KDA projections ride Q8_0 in this fixture — DELIBERATELY. As F32 they take
/// `Engine::linear` (cuBLAS f32 GEMV/GEMM), whose K-reduction split is SHAPE-DEPENDENT:
/// a 128-row shard and the 256-row full tensor produce 1-ulp-class differences for the
/// same logical row (measured by this gate's first run; the same class as the documented
/// `Engine::linear` m-dependence, FLAGS.md MEMRA_PRIME_CHUNK). Q8_0 rides the MMVQ/MMQ
/// custom kernels, whose per-row K order is row-count-independent — and it is also the
/// REAL serving class for f_b/g_b (loader law) while wq/wk/wv/wo serve BF16-resident on
/// another custom per-row matvec. The F32-class near-tie is documented in the lane doc;
/// the real-artifact classes are re-gated on the box.
fn is_kda_q8_class(id: &TensorId) -> bool {
    matches!(
        id,
        TensorId::Layer {
            tensor: LayerTensor::KdaQuery
                | LayerTensor::KdaKey
                | LayerTensor::KdaValue
                | LayerTensor::KdaForgetUp
                | LayerTensor::KdaGateUp
                | LayerTensor::KdaBeta
                | LayerTensor::KdaOutput,
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
        let (bytes, ggml_type) = if is_expert_bank(&req.id) || is_kda_q8_class(&req.id) {
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

/// Prime `ids[..p]`, then teacher-forced decode over `ids[p..]`: full logits + argmax tape.
type WalkOut = (Vec<Vec<f32>>, Vec<u32>);

fn walk(
    e: &Engine,
    m: &HybridModel,
    plan: &ModelPlan,
    ids: &[u32],
    p: usize,
    max_ctx: usize,
) -> Result<WalkOut, Box<dyn std::error::Error>> {
    let mut cache = memra_engine::cache::Cache::new_planned(e, &m.cfg, plan, max_ctx)?;
    let (prime_logits, _seed, _hiddens) = m.prime_cache(e, &ids[..p], &mut cache, 0)?;
    let mut step_logits: Vec<Vec<f32>> = Vec::with_capacity(ids.len() - p + 1);
    let mut tape: Vec<u32> = Vec::with_capacity(ids.len() - p + 1);
    tape.push(argmax(&prime_logits) as u32);
    step_logits.push(prime_logits);
    for &tok in &ids[p..] {
        let ll = m.decode_step(e, tok, &mut cache)?;
        tape.push(argmax(&ll) as u32);
        step_logits.push(ll);
    }
    Ok((step_logits, tape))
}

fn bit_equal(a: &[Vec<f32>], b: &[Vec<f32>]) -> Option<(usize, usize)> {
    for (step, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(ra.len(), rb.len(), "logit rows disagree in length");
        if let Some(idx) = ra
            .iter()
            .zip(rb.iter())
            .position(|(x, y)| x.to_bits() != y.to_bits())
        {
            return Some((step, idx));
        }
    }
    None
}

/// Max relative difference across all steps/logits (rel = |a-b| / max(|a|, 1e-6)).
fn max_rel_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let mut worst = 0f32;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (x, y) in ra.iter().zip(rb.iter()) {
            let rel = (x - y).abs() / x.abs().max(1e-6);
            if rel > worst {
                worst = rel;
            }
        }
    }
    worst
}

/// The prime-regime near-tie band, CALIBRATED ON THIS GATE'S OWN MEASUREMENTS (the
/// cert-lines law: budgets are measured calibration rows, never borrowed or loosened
/// until green). Measured green-arm class on the fixture: 2.8e-5..4.9e-5 max relative
/// (multiple sharded projections x layers compound the per-GEMM ulp class; the
/// chunked-prime lane's own band was 2e-5 for its smaller single-site class). Band =
/// 10x margin over the measured worst; the reds land at 1.4e2..2.7e2 — SIX orders above
/// the band — which is what proves the band distinguishes rather than absorbs.
const PRIME_BAND: f32 = 2e-4;
/// A red must land at least this far out (orders above the band) or break the tape.
const RED_FLOOR: f32 = 1e-3;

/// SAFETY wrapper: this binary is single-threaded and every env change happens between
/// model loads (no engine call in flight).
fn set_env(k: &str, v: &str) {
    unsafe { std::env::set_var(k, v) };
}
fn rm_env(k: &str) {
    unsafe { std::env::remove_var(k) };
}

struct Verdict {
    name: String,
    pass: bool,
    detail: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    // Optional 3rd arg: preserve arm T's fixture weight-trace at this path (a labeled
    // FIXTURE-traffic input for the shared placement mint tool; the scratch dir is
    // still removed either way).
    let trace_out: Option<String> = std::env::args().nth(3);

    if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
        set_env("NVIDIA_TF32_OVERRIDE", "0");
    }
    // A leaked door from the caller would poison the PLAIN reference arm.
    rm_env("MEMRA_GLM5_TP");
    rm_env("MEMRA_GLM5_TP_GATE_RED");
    rm_env("MEMRA_PP_STAGES");
    // EP dispatch-diet doors PINNED =0 for every banked arm (the moe-loc §4.5 lesson: an
    // A/B arm that leaves a variable UNSET inherits any future default flip; a pin does
    // not). The ON twins run after the reds, and the flat counter across the banked arms
    // is asserted there.
    set_env("MEMRA_GLM5_EP_DIET", "0");
    set_env("MEMRA_GLM5_EP_GROUPED_PRIME", "0");
    // A leaked route tap would trace every arm's walks before arm T banks its tap-cold
    // references (and would break run-to-run receipt comparability).
    rm_env("MEMRA_MOE_TRACE");
    rm_env("MEMRA_MOE_WEIGHT_TRACE");
    rm_env("MEMRA_GLM5_EP_MAP");
    set_env("MEMRA_GLM5_TP_GATE_SAME_DEV", "1");

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
    let max_ctx = p + n + 8;
    let mut verdicts: Vec<Verdict> = Vec::new();

    // ---- SEED SEARCH: the EP identity arms are VACUOUS unless the token stream routes at
    // least one PEER-owned expert (measured: the first candidate seed routed only experts
    // 0..1 across the whole walk — R3 could not bite). Probe candidate seeds on a TP-all
    // model and take the first whose walk dispatches peer slots; the count is asserted
    // again per arm below.
    let (ids, sel_trace_probe) = {
        set_env("MEMRA_GLM5_TP", "all@0,1");
        let m_probe = HybridModel::load_from_source_without_mtp(&e, &source)?;
        rm_env("MEMRA_GLM5_TP");
        let mut chosen = None;
        for seed in [
            0x0915_5EEDu64,
            0xDEAD_BEEF,
            0x00C0_FFEE,
            0x1234_5678,
            0xA5A5_A5A5,
            0x0BAD_F00D,
        ] {
            let cand = tokens(p + n, seed);
            let before = memra_engine::glm5_tp::glm5_ep_peer_slot_dispatches();
            let _ = walk(&e, &m_probe, &plan, &cand, p, max_ctx)?;
            let dispatched = memra_engine::glm5_tp::glm5_ep_peer_slot_dispatches() - before;
            println!("[seed-search] seed={seed:#010x}: peer_slot_dispatches={dispatched}");
            if dispatched > 0 {
                chosen = Some(cand);
                break;
            }
        }
        let ids: Vec<u32> = chosen.ok_or(
            "no candidate seed routes a peer-owned expert; the EP arms would be vacuous — \
             extend the candidate list or the fixture",
        )?;
        // MoESD capture over both regimes of the chosen ids: the per-layer ROUTED unions
        // pick arm M's rank-1 singleton (a routed expert, so the skew arm cannot be
        // vacuous) and arm M's map rows below.
        memra_engine::moesd::begin_capture()?;
        let _ = walk(&e, &m_probe, &plan, &ids, 1, max_ctx)?;
        let _ = walk(&e, &m_probe, &plan, &ids, p, max_ctx)?;
        let unions = memra_engine::moesd::finish_capture()?;
        (ids, unions)
    };

    // ================= A. PLAIN references (door OFF), both regimes =================
    // DECODE regime: P=1 (a 1-token prime is the t=1 program), every step is t=1.
    // PRIME regime: P=p as invoked (t=P batched prime), the near-tie class.
    eprintln!("[phase] arm A: PLAIN references, door OFF (decode P=1 and prime P={p})");
    let m_plain = HybridModel::load_from_source_without_mtp(&e, &source)?;
    assert!(
        m_plain.hyper.is_some(),
        "the fixture must load as a HyperConnections trunk"
    );
    let (ref_dec_logits, ref_dec_tape) = walk(&e, &m_plain, &plan, &ids, 1, max_ctx)?;
    let (ref_pri_logits, ref_pri_tape) = walk(&e, &m_plain, &plan, &ids, p, max_ctx)?;
    println!(
        "[plain] banked decode-regime {} steps + prime-regime {} steps x {} logits",
        ref_dec_logits.len(),
        ref_pri_logits.len(),
        ref_dec_logits[0].len(),
    );

    // A TP arm loader + walk + compare, reused by every green/red arm below. `ep_map`
    // arms MEMRA_GLM5_EP_MAP for the load (the measured-placement door).
    let tp_arm = |name: &str,
                  spec: &str,
                  red: Option<&str>,
                  ep_map: Option<&str>,
                  expect_diverge: bool,
                  verdicts: &mut Vec<Verdict>|
     -> Result<(), Box<dyn std::error::Error>> {
        eprintln!("[phase] arm {name}: MEMRA_GLM5_TP={spec} red={red:?} ep_map={ep_map:?}");
        set_env("MEMRA_GLM5_TP", spec);
        match red {
            Some(r) => set_env("MEMRA_GLM5_TP_GATE_RED", r),
            None => rm_env("MEMRA_GLM5_TP_GATE_RED"),
        }
        match ep_map {
            Some(p) => set_env("MEMRA_GLM5_EP_MAP", p),
            None => rm_env("MEMRA_GLM5_EP_MAP"),
        }
        let m_tp = HybridModel::load_from_source_without_mtp(&e, &source)?;
        rm_env("MEMRA_GLM5_TP");
        rm_env("MEMRA_GLM5_EP_MAP");
        // NOTE: the RED knob stays SET through the walks — `skip-peer-combine` acts at
        // RUNTIME (the EP combine reads it per slot), unlike the two load-time shard
        // mutations. It is removed at the end of this arm.

        // Non-vacuity: the spec'd layers really carry shards/EP arms.
        let mut sharded = 0usize;
        let mut ep_armed = 0usize;
        for layer in &m_tp.layers {
            match &layer.mixer {
                memra_engine::hybrid::Mixer::Kda(la) if la.tp.is_some() => sharded += 1,
                memra_engine::hybrid::Mixer::Mla(la) if la.tp.is_some() => sharded += 1,
                _ => {}
            }
            if let memra_engine::hybrid::Ffn::Moe(m) = &layer.ffn
                && m.glm5_ep.is_some()
            {
                ep_armed += 1;
            }
        }
        let expect_layers = if spec == "all@0,1" {
            LAYERS
        } else {
            spec.split(';').count()
        };
        assert_eq!(
            sharded, expect_layers,
            "[{name}] {sharded} sharded mixers != {expect_layers} spec'd layers — vacuous arm"
        );
        println!("[{name}] sharded mixers={sharded} ep_armed={ep_armed}");
        let peer_slots_before = memra_engine::glm5_tp::glm5_ep_peer_slot_dispatches();

        // ---- DECODE regime (P=1): byte identity, two repetitions ----
        let (dec1, dec_tape1) = walk(&e, &m_tp, &plan, &ids, 1, max_ctx)?;
        let (dec2, dec_tape2) = walk(&e, &m_tp, &plan, &ids, 1, max_ctx)?;
        if let Some((s, i)) = bit_equal(&dec1, &dec2) {
            verdicts.push(Verdict {
                name: format!("{name} decode self-consistency"),
                pass: false,
                detail: format!("two repetitions diverge at step {s} logit {i}"),
            });
        } else {
            assert_eq!(dec_tape1, dec_tape2);
            verdicts.push(Verdict {
                name: format!("{name} decode self-consistency"),
                pass: true,
                detail: "two repetitions bit-identical".into(),
            });
        }
        let dec_diff = bit_equal(&ref_dec_logits, &dec1);
        let dec_tape_ok = ref_dec_tape == dec_tape1;
        let dec_rel = max_rel_diff(&ref_dec_logits, &dec1);
        let (pass, detail) = if expect_diverge {
            // The decode-side red report is informational: the BINDING red verdict is the
            // prime-regime arm below, which exercises every shard surface (a t=1 token
            // stream may legitimately never route a peer-owned expert on a 4-expert
            // fixture). A decode-side bite is still reported when it happens.
            (
                true,
                format!(
                    "red decode-side report: max_rel={dec_rel:.3e} tape_match={dec_tape_ok} \
                     (binding verdict = prime-regime arm)"
                ),
            )
        } else {
            match dec_diff {
                None if dec_tape_ok => (
                    true,
                    format!(
                        "DECODE BYTE-IDENTICAL to plain: {} t=1 steps x {} logits + tape",
                        ref_dec_logits.len(),
                        ref_dec_logits[0].len()
                    ),
                ),
                None => (
                    false,
                    "logits identical but tape differs (harness bug)".into(),
                ),
                Some((st, i)) => {
                    let (a, b) = (ref_dec_logits[st][i], dec1[st][i]);
                    (
                        false,
                        format!(
                            "DECODE MISMATCH at step {st} logit {i}: plain={a:?} tp={b:?} \
                             (max_rel={dec_rel:.3e})"
                        ),
                    )
                }
            }
        };
        verdicts.push(Verdict {
            name: format!("{name} decode-identity"),
            pass,
            detail,
        });

        // ---- PRIME regime (P=p, t>=2): band + tape (byte identity is a bonus receipt).
        // Red arms run it too: a red whose mutated surface the decode tape never touches
        // (e.g. peer-owned experts unrouted by the t=1 token stream) must still bite here.
        let (pri, pri_tape) = walk(&e, &m_tp, &plan, &ids, p, max_ctx)?;
        let pri_rel = max_rel_diff(&ref_pri_logits, &pri);
        let pri_bytes = bit_equal(&ref_pri_logits, &pri).is_none();
        let pri_tape_ok = ref_pri_tape == pri_tape;
        if expect_diverge {
            let loud = pri_rel > RED_FLOOR || !pri_tape_ok;
            verdicts.push(Verdict {
                name: format!("{name} prime-regime"),
                pass: loud,
                detail: if loud {
                    format!(
                        "RED bites LOUD in prime: max_rel={pri_rel:.3e} tape_match={pri_tape_ok}"
                    )
                } else {
                    format!(
                        "RED FAILED TO BITE in prime: max_rel={pri_rel:.3e} \
                         tape_match={pri_tape_ok}"
                    )
                },
            });
        } else {
            let pass = pri_rel <= PRIME_BAND && pri_tape_ok;
            verdicts.push(Verdict {
                name: format!("{name} prime-band"),
                pass,
                detail: format!(
                    "prime P={p} regime: max_rel={pri_rel:.3e} (band {PRIME_BAND:.0e}) \
                     tape_match={pri_tape_ok} byte_identical={pri_bytes} — t>=2 batched \
                     GEMM widths are the documented shard-shape near-tie class"
                ),
            });
        }
        rm_env("MEMRA_GLM5_TP_GATE_RED");
        // EP non-vacuity, evaluated over BOTH regimes' walks: at least one PEER-owned
        // slot must have dispatched, or the arm's EP identity claim is vacuous.
        if ep_armed > 0 {
            let peer_slots =
                memra_engine::glm5_tp::glm5_ep_peer_slot_dispatches() - peer_slots_before;
            assert!(
                peer_slots > 0,
                "[{name}] EP armed on {ep_armed} layer(s) but ZERO peer-owned slots were \
                 dispatched — the EP identity claim would be vacuous"
            );
            println!("[{name}] EP peer-slot dispatches: {peer_slots}");
        }
        Ok(())
    };

    // ================= B/C/D. GREEN arms =================
    tp_arm("B tp-all", "all@0,1", None, None, false, &mut verdicts)?;
    tp_arm(
        "C tp-kda-only",
        "0@0,1;2@0,1",
        None,
        None,
        false,
        &mut verdicts,
    )?;
    tp_arm(
        "C0 tp-layer0-only",
        "0@0,1",
        None,
        None,
        false,
        &mut verdicts,
    )?;
    tp_arm(
        "C2 tp-layer2-only",
        "2@0,1",
        None,
        None,
        false,
        &mut verdicts,
    )?;
    tp_arm(
        "D tp-mla-only",
        "1@0,1;3@0,1",
        None,
        None,
        false,
        &mut verdicts,
    )?;

    // ================= E/F. poison + co-refusal on a live TP model =================
    eprintln!("[phase] arm E/F: plain-path poison + spec co-refusal");
    set_env("MEMRA_GLM5_TP", "all@0,1");
    let m_tp = HybridModel::load_from_source_without_mtp(&e, &source)?;
    // E: the stateless forward must refuse a sharded layer by name.
    let fwd = m_plain.forward(&e, &ids); // plain model: sanity that the walk itself works
    assert!(fwd.is_ok(), "plain forward must work: {:?}", fwd.err());
    match m_tp.forward(&e, &ids) {
        Err(err) if err.to_string().contains("glm5-TP-sharded") => verdicts.push(Verdict {
            name: "E stateless-poison".into(),
            pass: true,
            detail: format!("forward refused by name: {err}"),
        }),
        Err(err) => verdicts.push(Verdict {
            name: "E stateless-poison".into(),
            pass: false,
            detail: format!("refused with the WRONG error: {err}"),
        }),
        Ok(_) => verdicts.push(Verdict {
            name: "E stateless-poison".into(),
            pass: false,
            detail: "stateless forward ran on a sharded model".into(),
        }),
    }
    // F: spec sessions co-refuse while the door is armed (the armed check reads the env).
    match m_tp.glm5_spec_session_new(&e, &ids[..p.min(8)], max_ctx, None) {
        Err(err) if err.to_string().contains("co-refused") => verdicts.push(Verdict {
            name: "F spec-co-refusal".into(),
            pass: true,
            detail: format!("session refused by name: {err}"),
        }),
        Err(err) => verdicts.push(Verdict {
            name: "F spec-co-refusal".into(),
            pass: false,
            detail: format!("refused with the WRONG error: {err}"),
        }),
        Ok(_) => verdicts.push(Verdict {
            name: "F spec-co-refusal".into(),
            pass: false,
            detail: "spec session created on a TP-armed model".into(),
        }),
    }
    rm_env("MEMRA_GLM5_TP");
    drop(m_tp);

    // ================= G. PP composition refusal at load =================
    eprintln!("[phase] arm G: MEMRA_PP_STAGES=2 + TP door must refuse at load");
    set_env("MEMRA_GLM5_TP", "all@0,1");
    set_env("MEMRA_PP_STAGES", "2");
    match HybridModel::load_from_source_without_mtp(&e, &source) {
        Err(err) if err.to_string().contains("MEMRA_PP_STAGES") => verdicts.push(Verdict {
            name: "G pp-composition-refusal".into(),
            pass: true,
            detail: format!("load refused by name: {err}"),
        }),
        Err(err) => verdicts.push(Verdict {
            name: "G pp-composition-refusal".into(),
            pass: false,
            detail: format!("refused with the WRONG error: {err}"),
        }),
        Ok(_) => verdicts.push(Verdict {
            name: "G pp-composition-refusal".into(),
            pass: false,
            detail: "TP + PP>1 loaded without refusing".into(),
        }),
    }
    rm_env("MEMRA_PP_STAGES");
    rm_env("MEMRA_GLM5_TP");

    // ================= M / H. Measured-placement map arms =================
    // Scratch dir for the gate's map artifacts; removed at the end (tmp hygiene law).
    let map_dir = std::env::temp_dir().join(format!("glm5-tp-gate-epmap-{}", std::process::id()));
    std::fs::create_dir_all(&map_dir)?;
    let write_map = |name: &str, text: &str| -> Result<String, Box<dyn std::error::Error>> {
        let path = map_dir.join(name);
        std::fs::write(&path, text)?;
        Ok(path.to_string_lossy().into_owned())
    };

    // The deliberately skewed map (frozen memra-ep-map-v1 JSON, emitted through the
    // engine reader's own render): per MoE layer, ALL BUT ONE expert on rank 0; the
    // rank-1 singleton is a PROVABLY ROUTED expert (the probe walks' union, highest id)
    // so the skew arm's EP non-vacuity assertion has teeth by construction.
    let moe_layers: [usize; 3] = [1, 2, 3];
    let skew_map_text = {
        let mut layers = std::collections::BTreeMap::new();
        for &il in &moe_layers {
            let u = sel_trace_probe
                .iter()
                .find(|l| l.id as usize == il)
                .ok_or(format!(
                    "probe capture carries no routed union for layer {il}"
                ))?;
            let singleton = *u.experts.last().ok_or("empty routed union")? as usize;
            println!(
                "[map-skew] layer {il}: routed_union={:?} rank1_singleton={singleton}",
                u.experts
            );
            let owners: Vec<u8> = (0..4).map(|ex| u8::from(ex == singleton)).collect();
            layers.insert(il, owners);
        }
        memra_engine::ep_map::EpMap {
            n_experts: 4,
            ranks: 2,
            entry_rank: 0,
            layers,
        }
        .render()
    };
    let skew_map = write_map("skew.map", &skew_map_text)?;
    tp_arm(
        "M map-skew",
        "all@0,1",
        None,
        Some(&skew_map),
        false,
        &mut verdicts,
    )?;

    // H1: missing map file refuses by name.
    let h_refusal = |name: &str,
                     map_path: &str,
                     must_contain: &str,
                     tp_on: bool,
                     verdicts: &mut Vec<Verdict>| {
        eprintln!("[phase] arm {name}: MEMRA_GLM5_EP_MAP={map_path} tp={tp_on}");
        if tp_on {
            set_env("MEMRA_GLM5_TP", "all@0,1");
        } else {
            rm_env("MEMRA_GLM5_TP");
        }
        set_env("MEMRA_GLM5_EP_MAP", map_path);
        let res = HybridModel::load_from_source_without_mtp(&e, &source);
        rm_env("MEMRA_GLM5_EP_MAP");
        rm_env("MEMRA_GLM5_TP");
        match res {
            Err(err)
                if err.to_string().contains("MEMRA_GLM5_EP_MAP")
                    && err.to_string().contains(must_contain) =>
            {
                verdicts.push(Verdict {
                    name: name.into(),
                    pass: true,
                    detail: format!("load refused by name: {err}"),
                })
            }
            Err(err) => verdicts.push(Verdict {
                name: name.into(),
                pass: false,
                detail: format!("refused with the WRONG error: {err}"),
            }),
            Ok(_) => verdicts.push(Verdict {
                name: name.into(),
                pass: false,
                detail: "loaded without refusing".into(),
            }),
        }
    };
    let missing_path = map_dir
        .join("does-not-exist.map")
        .to_string_lossy()
        .into_owned();
    h_refusal(
        "H1 map-missing-file",
        &missing_path,
        "cannot read",
        true,
        &mut verdicts,
    );
    let malformed = write_map(
        "malformed.map",
        "{\"format\": \"memra-ep-map-v1\", \"ranks\": 2, \"entry_rank\": 0, \
         \"expert_count\": 4, \"layers\": [\
         {\"layer\": 1, \"assignment\": [0, 1]}, \
         {\"layer\": 2, \"assignment\": [0, 0, 1, 1]}, \
         {\"layer\": 3, \"assignment\": [0, 0, 1, 1]}]}",
    )?;
    h_refusal(
        "H2 map-wrong-expert-count",
        &malformed,
        "assignment",
        true,
        &mut verdicts,
    );
    let missing_layer = write_map(
        "missing-layer.map",
        "{\"format\": \"memra-ep-map-v1\", \"ranks\": 2, \"entry_rank\": 0, \
         \"expert_count\": 4, \"layers\": [{\"layer\": 1, \"assignment\": [0, 0, 1, 1]}]}",
    )?;
    h_refusal(
        "H3 map-missing-layer-row",
        &missing_layer,
        "no map row",
        true,
        &mut verdicts,
    );
    // H5: a map minted with the wrong first-hop card (entry_rank != 0) refuses — the
    // glm5 TP-2 entry is root by construction (router + combine + shared expert).
    let wrong_entry = write_map(
        "wrong-entry.map",
        "{\"format\": \"memra-ep-map-v1\", \"ranks\": 2, \"entry_rank\": 1, \
         \"expert_count\": 4, \"layers\": [\
         {\"layer\": 1, \"assignment\": [0, 0, 1, 1]}, \
         {\"layer\": 2, \"assignment\": [0, 0, 1, 1]}, \
         {\"layer\": 3, \"assignment\": [0, 0, 1, 1]}]}",
    )?;
    h_refusal(
        "H5 map-wrong-entry-rank",
        &wrong_entry,
        "entry_rank",
        true,
        &mut verdicts,
    );
    // H4: the map with the TP door COLD must refuse at load (silent-even-split trap).
    {
        eprintln!("[phase] arm H4: MEMRA_GLM5_EP_MAP set, MEMRA_GLM5_TP off");
        rm_env("MEMRA_GLM5_TP");
        set_env("MEMRA_GLM5_EP_MAP", &skew_map);
        let res = HybridModel::load_from_source_without_mtp(&e, &source);
        rm_env("MEMRA_GLM5_EP_MAP");
        match res {
            Err(err) if err.to_string().contains("MEMRA_GLM5_TP is off") => {
                verdicts.push(Verdict {
                    name: "H4 map-without-tp".into(),
                    pass: true,
                    detail: format!("load refused by name: {err}"),
                })
            }
            Err(err) => verdicts.push(Verdict {
                name: "H4 map-without-tp".into(),
                pass: false,
                detail: format!("refused with the WRONG error: {err}"),
            }),
            Ok(_) => verdicts.push(Verdict {
                name: "H4 map-without-tp".into(),
                pass: false,
                detail: "a placement map loaded with the TP door cold (silent even split)".into(),
            }),
        }
    }

    // ================= R1-R4. RED arms =================
    tp_arm(
        "R1 red-swap-wo",
        "all@0,1",
        Some("swap-wo"),
        None,
        true,
        &mut verdicts,
    )?;
    tp_arm(
        "R2 red-swap-ep-gateup",
        "all@0,1",
        Some("swap-ep-gateup"),
        None,
        true,
        &mut verdicts,
    )?;
    tp_arm(
        "R3 red-skip-peer-combine",
        "all@0,1",
        Some("skip-peer-combine"),
        None,
        true,
        &mut verdicts,
    )?;
    // R4: the corrupted-map red rides the SKEWED map (rank 0 owns three experts there, so
    // reversing its local-slot table misindexes real weights on the dominant rank).
    tp_arm(
        "R4 red-corrupt-ep-map",
        "all@0,1",
        Some("corrupt-ep-map"),
        Some(&skew_map),
        true,
        &mut verdicts,
    )?;

    // ================= DIET arms (lane/glm5-ep-diet, MEMRA_GLM5_EP_DIET) =================
    // The doors were PINNED =0 for every banked arm above; a flat dispatch counter across
    // that whole battery is itself the OFF-arm receipt, asserted before the ON twins run.
    {
        use memra_engine::glm5_tp as gtp;
        assert_eq!(
            gtp::glm5_ep_diet_dispatches(),
            0,
            "the EP diet dispatched during the pinned-=0 banked arms — the pin is not holding"
        );
        assert_eq!(
            gtp::glm5_ep_grouped_prime_dispatches(),
            0,
            "the EP grouped prime dispatched during the pinned-=0 banked arms"
        );

        // B2: the whole trunk with the diet ON — the transport/combine restructure must hold
        // the SAME bars as v1 (decode BYTE identity, prime band), with engagement proven.
        set_env("MEMRA_GLM5_EP_DIET", "1");
        let (d0, b0, r0) = (
            gtp::glm5_ep_diet_dispatches(),
            gtp::glm5_ep_diet_bulk_returns(),
            gtp::glm5_ep_diet_peer_roundtrips_avoided(),
        );
        tp_arm(
            "B2 tp-all-diet",
            "all@0,1",
            None,
            None,
            false,
            &mut verdicts,
        )?;
        let (dd, db, dr) = (
            gtp::glm5_ep_diet_dispatches() - d0,
            gtp::glm5_ep_diet_bulk_returns() - b0,
            gtp::glm5_ep_diet_peer_roundtrips_avoided() - r0,
        );
        verdicts.push(Verdict {
            name: "B2 diet-engagement".into(),
            pass: dd > 0 && db > 0 && dr > 0,
            detail: format!(
                "diet layer-calls={dd} bulk peer returns={db} per-slot round-trips \
                 avoided={dr} (all must be >0 or the identity claim above is vacuous)"
            ),
        });

        // B3: grouped prime armed ON TOP of the diet. The fixture bank is Q8_0 — not
        // f16g-eligible — so the arm must FALL CLOSED (counter pinned 0) while the walk
        // stays band-green vs plain at a t>MOE_DEV_MAX_T prime that actually KEYS the arm
        // (P=16 primes at t=16, below the key, so this uses P=24 explicitly).
        set_env("MEMRA_GLM5_EP_GROUPED_PRIME", "1");
        {
            eprintln!("[phase] arm B3: diet + grouped prime (must fall closed on Q8_0), P=24");
            let p24 = 24usize.min(ids.len().saturating_sub(2));
            let ctx24 = ids.len() + 8;
            let (ref24, ref24_tape) = walk(&e, &m_plain, &plan, &ids, p24, ctx24)?;
            set_env("MEMRA_GLM5_TP", "all@0,1");
            let m_tp = HybridModel::load_from_source_without_mtp(&e, &source)?;
            rm_env("MEMRA_GLM5_TP");
            let gp0 = gtp::glm5_ep_grouped_prime_dispatches();
            let (got, got_tape) = walk(&e, &m_tp, &plan, &ids, p24, ctx24)?;
            let gp_delta = gtp::glm5_ep_grouped_prime_dispatches() - gp0;
            let rel = max_rel_diff(&ref24, &got);
            let pass = gp_delta == 0 && rel <= PRIME_BAND && got_tape == ref24_tape;
            verdicts.push(Verdict {
                name: "B3 grouped-prime-fall-closed".into(),
                pass,
                detail: format!(
                    "P={p24} keys the arm (announce fires), Q8_0 bank falls closed: \
                     grouped-prime dispatches={gp_delta} (must be 0), max_rel={rel:.3e} \
                     (band {PRIME_BAND:.0e}) tape_match={}",
                    got_tape == ref24_tape
                ),
            });
        }
        set_env("MEMRA_GLM5_EP_GROUPED_PRIME", "0");

        // M2: the deliberately skewed measured map UNDER the diet — the diet is
        // placement-agnostic by construction (it consumes owner_of/local_of; the map moves
        // bytes, the diet changes how they move). Same bars as arm M.
        tp_arm(
            "M2 map-skew-diet",
            "all@0,1",
            None,
            Some(&skew_map),
            false,
            &mut verdicts,
        )?;

        // Reds THROUGH the dieted walk: the load-time wrong-expert-weights red and the
        // RUNTIME skip-peer-combine red (which now acts at the compact-staging site and
        // must still reach the combine loudly).
        tp_arm(
            "R2D red-swap-ep-gateup-diet",
            "all@0,1",
            Some("swap-ep-gateup"),
            None,
            true,
            &mut verdicts,
        )?;
        tp_arm(
            "R3D red-skip-peer-combine-diet",
            "all@0,1",
            Some("skip-peer-combine"),
            None,
            true,
            &mut verdicts,
        )?;
        set_env("MEMRA_GLM5_EP_DIET", "0");
    }

    // ================= T. Trace-tap identity (MEMRA_MOE_WEIGHT_TRACE) =================
    // The FLEET tap (`trace_moe_routes`), armed on the served glm5 walk — the
    // co-activation measurement input of LAW:coactivation-expert-placement. LAST
    // deliberately, so every preceding arm ran tap-cold (receipts comparable run to
    // run). Line format: `<layer> <t> <expert:weight,...>` with t*k pairs.
    {
        eprintln!(
            "[phase] arm T: MEMRA_MOE_TRACE + MEMRA_MOE_WEIGHT_TRACE identity + row count \
             on the plain walk"
        );
        let id_trace_path = map_dir.join("id-trace.txt");
        let trace_path = map_dir.join("weight-trace.txt");
        let trace_str = trace_path.to_string_lossy().into_owned();
        // Both sibling taps armed together: the id lines are the shared placement
        // tool's co-occurrence input, the weight lines its hotness signal — arm T's
        // identity claim covers the exact pair the box trace cells will arm.
        set_env("MEMRA_MOE_TRACE", &id_trace_path.to_string_lossy());
        set_env("MEMRA_MOE_WEIGHT_TRACE", &trace_str);
        let (t_dec, t_dec_tape) = walk(&e, &m_plain, &plan, &ids, 1, max_ctx)?;
        let (t_pri, t_pri_tape) = walk(&e, &m_plain, &plan, &ids, p, max_ctx)?;
        rm_env("MEMRA_MOE_TRACE");
        rm_env("MEMRA_MOE_WEIGHT_TRACE");
        let dec_ok = bit_equal(&ref_dec_logits, &t_dec).is_none() && t_dec_tape == ref_dec_tape;
        let pri_ok = bit_equal(&ref_pri_logits, &t_pri).is_none() && t_pri_tape == ref_pri_tape;
        verdicts.push(Verdict {
            name: "T trace-identity".into(),
            pass: dec_ok && pri_ok,
            detail: format!(
                "trace ON vs banked plain: decode byte-identical={dec_ok} prime \
                 byte-identical={pri_ok} (same walk, same program — any differing bit is \
                 a tap side effect)"
            ),
        });
        // Rows counted: token events per MoE layer across the two walks must equal
        // 2*(p+n) EXACTLY (each walk feeds p+n tokens through every MoE layer's
        // router; the count is chunking-invariant because rows carry t). Every row is
        // shape-checked: t*k id:weight pairs, ids inside the 4-expert bank.
        let text = std::fs::read_to_string(&trace_path)?;
        let mut events_by_layer: BTreeMap<usize, usize> = BTreeMap::new();
        let mut parse_err: Option<String> = None;
        let mut file_rows = 0usize;
        'rows: for line in text.lines().filter(|l| !l.trim().is_empty()) {
            file_rows += 1;
            let mut f = line.split_whitespace();
            let (Some(il), Some(t), Some(pairs), None) = (f.next(), f.next(), f.next(), f.next())
            else {
                parse_err = Some(format!("row is not `<layer> <t> <pairs>`: {line}"));
                break;
            };
            let (Ok(il), Ok(t)) = (il.parse::<usize>(), t.parse::<usize>()) else {
                parse_err = Some(format!("non-integer layer/t: {line}"));
                break;
            };
            let mut n_pairs = 0usize;
            for pair in pairs.split(',') {
                let Some((id, w)) = pair.split_once(':') else {
                    parse_err = Some(format!("pair without `:`: {line}"));
                    break 'rows;
                };
                let ok_id = id.parse::<usize>().map(|v| v < 4).unwrap_or(false);
                if !ok_id || w.parse::<f32>().is_err() {
                    parse_err = Some(format!("bad id/weight pair {pair:?}: {line}"));
                    break 'rows;
                }
                n_pairs += 1;
            }
            if t == 0 || !n_pairs.is_multiple_of(t) {
                parse_err = Some(format!("{n_pairs} pairs do not split across t={t}: {line}"));
                break;
            }
            *events_by_layer.entry(il).or_insert(0) += t;
        }
        let expected = 2 * (p + n);
        let per_layer_ok = moe_layers
            .iter()
            .all(|il| events_by_layer.get(il).copied().unwrap_or(0) == expected);
        let extra_layers = events_by_layer.keys().any(|il| !moe_layers.contains(il));
        verdicts.push(Verdict {
            name: "T trace-rows-counted".into(),
            pass: parse_err.is_none() && per_layer_ok && !extra_layers && file_rows > 0,
            detail: match parse_err {
                Some(e) => format!("a trace row failed the shape check: {e}"),
                None => format!(
                    "events per MoE layer {events_by_layer:?} (expected {expected} on layers \
                     {moe_layers:?}), file rows={file_rows}"
                ),
            },
        });
    }
    if let Some(out) = &trace_out {
        // Both formats preserved: `<out>` = the weight trace, `<out>.ids` = the id
        // trace (the shared placement tool's --trace input).
        std::fs::copy(map_dir.join("weight-trace.txt"), out)?;
        std::fs::copy(map_dir.join("id-trace.txt"), format!("{out}.ids"))?;
        println!("[trace-out] arm T fixture traces preserved at {out} (+.ids)");
    }
    std::fs::remove_dir_all(&map_dir)?;

    // ================= verdict =================
    println!("==========================================================");
    let mut fails = 0usize;
    for v in &verdicts {
        println!(
            "glm5-tp-gate {}: [{}] {}",
            if v.pass { "PASS" } else { "FAIL" },
            v.name,
            v.detail
        );
        if !v.pass {
            fails += 1;
        }
    }
    println!("==========================================================");
    if fails == 0 {
        println!(
            "glm5-tp-gate: ALL ARMS PASS (P={p} N={n}, fixture kda_heads=2 mla_heads=2 \
             experts=4 top-2, same-device dual-context emulation, host-canonical transport)"
        );
        Ok(())
    } else {
        Err(format!("glm5-tp-gate: {fails} ARM(S) FAILED").into())
    }
}
