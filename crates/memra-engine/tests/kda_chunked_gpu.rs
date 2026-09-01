//! Gate for the CHUNKED KDA prefill scan (`MEMRA_KDA_CHUNKED`, cu/kda.cu `memra_kda_chunk_*`).
//!
//! Truth anchors, stated honestly:
//! - Full-mixer gates anchor on `memra_reference::kimi_delta_net_layer` (the same pinned
//!   reference the sequential fixture gate uses), with the chunked scan ENGAGED end to end.
//! - Scan-level gates anchor on `memra_kda_scan_s128` (itself reference-gated in
//!   kda_fixture_gpu.rs), which lets them compare the two forms on identical stage-5 inputs
//!   including a NONZERO carried-in state — the case the full-mixer zero-state gates miss.
//!
//! NUMERIC CLASS: the chunked form is NOT bit-identical to the sequential scan — the WY form
//! replaces per-token rank-1 updates with a forward substitution plus chunk-wide reductions,
//! a different FP accumulation order (the GDN A4 precedent states the same). The bar is the
//! scale-relative band `maxdiff <= TOL * scale`, the kda_fixture_gpu.rs shape and constant.
//! The one BIT-identity claim the chunked form does make is split-invariance at multiples of
//! the chunk size (grids realign and K4's smem state round-trips through f32 global exactly),
//! and that claim is gated as bit-identity below.
//!
//! RED ARMS: three mutants that MUST exceed the band, each emulating a named boundary bug —
//! state not carried across a chunk boundary, gate cumulative product off by one, decay
//! applied twice at the boundary. A red arm passing means the band CATCHES that bug class.
//!
//! Tests serialize on one mutex and set their env under it: the chunked seam reads its env
//! PER CALL by design (the kernel-check precedent), so arms can alternate in one process, and
//! the lock keeps one test's arm from leaking into another's.

use memra_engine::Engine;
use memra_engine::kda::{KDA_HEAD_DIM, KdaAttnLayer, kda_attn, kda_attn_decode, kda_attn_prime};
use memra_gguf::GgmlType;
use memra_gguf::config::ModelConfig;
use memra_gguf::model_plan::{
    ActivationPlan, AttentionPlan, DenseMlpPlan, DraftSourcePlan, KimiDeltaNetPlan, LayerPlan,
    MlpPlan, ModelPlan, NormKind, NormPlan, ResidualTopology, StatePlan, WeightTransform,
};
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, OutputHead, TensorContract, TensorId, TensorMatch,
};
use memra_reference::{ReferenceTensor, deterministic_fixture, kimi_delta_net_layer};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Mutex;

const HIDDEN: usize = 256;
const HEADS: u32 = 2;
const HEAD_DIM: u32 = 128;
const CONV_KERNEL: u32 = 4;
const GATE_LOWER_BOUND: f32 = -5.0;
const EPS: f32 = 1e-5;
/// Same calibrated scale-relative bar as kda_fixture_gpu.rs (5090, TF32 off, 2026-08-27:
/// worst sequential-vs-reference comparison 1.3e-6 relative; 5e-5 keeps ~40x reduction-order
/// headroom while staying an order of magnitude below the ~7e-4 TF32-on class, so an
/// accidental TF32 return fails instead of hiding). The chunked form's measured relative
/// error on these gates is recorded in the lane receipts next to this bar.
const TOL: f32 = 5e-5;
/// Chunk size the gates assume when they construct boundary-crossing lengths. Kept equal to
/// the shipped `kda_chunk_size()` default; the tests do not set MEMRA_KDA_CHUNK.
const C: usize = 64;

const QKV: usize = (HEADS * HEAD_DIM) as usize;
const CONV_WIDTH: usize = 3 * QKV;
const PAD: usize = CONV_KERNEL as usize - 1;
const STATE_WIDTH: usize = (HEADS * HEAD_DIM * HEAD_DIM) as usize;

/// Serializes the tests AND guards the per-call env arms. Poisoning is ignored on purpose:
/// a failed test must not mask the remaining gates behind a poisoned lock.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn lock_and_arm_chunked() -> std::sync::MutexGuard<'static, ()> {
    let guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    force_true_f32();
    // SAFETY: process-global env, mutated only under GPU_LOCK, and the seam reads it per call.
    unsafe {
        std::env::set_var("MEMRA_KDA_CHUNKED", "1");
        std::env::set_var("MEMRA_KDA_CHUNK_MIN_T", "2");
    }
    guard
}

fn kda_plan() -> KimiDeltaNetPlan {
    KimiDeltaNetPlan {
        num_heads: HEADS,
        head_dim: HEAD_DIM,
        conv_kernel: CONV_KERNEL,
        gate_lower_bound: GATE_LOWER_BOUND,
    }
}

/// One KDA layer with inert scaffolding — verbatim the kda_fixture_gpu.rs plan, so the two
/// gates read one contract.
fn one_kda_layer_plan() -> ModelPlan {
    let norm = NormPlan {
        kind: NormKind::Rms,
        epsilon: EPS,
        weight_transform: WeightTransform::Identity,
    };
    ModelPlan {
        arch: memra_gguf::config::Arch::Glm5Next,
        hidden_size: HIDDEN as u32,
        vocab_size: 32,
        context_length: 512,
        embedding_scale: 1.0,
        vision: None,
        multimodal: None,
        layers: vec![LayerPlan {
            index: 0,
            pre_attention_norm: norm,
            attention: AttentionPlan::KimiDeltaNet(kda_plan()),
            pre_mlp_norm: norm,
            mlp: MlpPlan::Dense(DenseMlpPlan {
                intermediate_size: 32,
                activation: ActivationPlan::Silu,
            }),
            residual: ResidualTopology::Serial,
            state: StatePlan::Recurrent {
                conv_width: CONV_WIDTH as u32,
                conv_kernel: CONV_KERNEL,
                state_width: STATE_WIDTH as u32,
            },
        }],
        output_norm: norm,
        logits: Vec::new(),
        mtp_blocks: Vec::new(),
        drafter: None,
        draft_source: DraftSourcePlan::Embedded,
        sampling_defaults: None,
        partition_boundaries: Vec::new(),
    }
}

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
}

struct FixtureSource {
    tensors: BTreeMap<String, OwnedTensor>,
}

impl TensorSource for FixtureSource {
    fn config(&self) -> ModelConfig {
        unreachable!("the KDA fixture source is tensor-only; nothing in the load path reads config")
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
    .expect("contract for the one-KDA-layer plan");
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
    FixtureSource { tensors }
}

fn hidden_states(tokens: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..tokens * HIDDEN)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.6
        })
        .collect()
}

/// Deterministic uniform in [lo, hi).
fn uniform(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            lo + ((s >> 33) as f32 / (1u64 << 31) as f32) * (hi - lo)
        })
        .collect()
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared slices differ in length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn scale_of(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6)
}

/// Band check that PRINTS its measurement — the printed rel values are the banked calibration
/// rows for this numeric class.
fn check(name: &str, got: &[f32], want: &[f32]) {
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{name}: output has non-finite values"
    );
    let scale = scale_of(want);
    let md = maxdiff(got, want);
    println!(
        "[kda-chunk gate] {name}: maxdiff {md:.3e} scale {scale:.3e} rel {:.3e}",
        md / scale
    );
    assert!(
        md <= TOL * scale,
        "{name}: maxdiff {md:.3e} (scale {scale:.3e}, rel {:.3e}, tol {TOL:.1e})",
        md / scale
    );
}

/// The red-arm twin of `check`: the mutant must EXCEED the band, or the gate cannot see the
/// bug class it exists for.
fn check_red(name: &str, got: &[f32], want: &[f32]) {
    let scale = scale_of(want);
    let md = maxdiff(got, want);
    println!(
        "[kda-chunk RED] {name}: maxdiff {md:.3e} scale {scale:.3e} rel {:.3e}",
        md / scale
    );
    assert!(
        md > TOL * scale,
        "{name}: mutant stayed INSIDE the band (maxdiff {md:.3e}, scale {scale:.3e}, tol \
         {TOL:.1e}) — the gate cannot detect this bug class at this tolerance"
    );
}

/// TF32 off before the first CUDA call — verbatim the kda_fixture_gpu.rs rationale.
fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call has happened and no Engine exists yet; call_once
            // serializes every test thread behind this write.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
    });
}

struct Harness {
    engine: Engine,
    layer: KdaAttnLayer,
    weights: BTreeMap<TensorId, ReferenceTensor>,
    plan: KimiDeltaNetPlan,
}

impl Harness {
    fn new() -> Self {
        let model_plan = one_kda_layer_plan();
        let fixture = deterministic_fixture(&model_plan).expect("deterministic KDA fixture");
        let source = fixture_source(&model_plan, &fixture.weights);
        let engine = Engine::new(0).expect("CUDA engine on device 0");
        let plan = kda_plan();
        let layer =
            KdaAttnLayer::load(&engine, &source, 0, &plan).expect("KDA mixer loads from contract");
        Self {
            engine,
            layer,
            weights: fixture.weights,
            plan,
        }
    }

    fn reference(&self, x: &[f32], tokens: usize) -> Vec<f32> {
        kimi_delta_net_layer(0, &self.plan, EPS, &self.weights, x, tokens, HIDDEN)
            .expect("reference KDA layer")
            .0
    }
}

/// Scan-level inputs in the stage-5 contract's ranges: q/k L2-normalized rows, |v| <= 1,
/// g in [-g_mag, 0) (raw log gates are ALWAYS non-positive — the chunked kernels' no-overflow
/// property rests on it), beta in (0, 1), and a NONZERO carried-in state.
struct ScanInputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    state: Vec<f32>,
}

fn scan_inputs(t: usize, seed: u64, g_mag: f32) -> ScanInputs {
    let h = HEADS as usize;
    let d = KDA_HEAD_DIM;
    let qkv = h * d;
    let l2 = |mut v: Vec<f32>| {
        for row in v.chunks_mut(d) {
            let n = (row.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
            for x in row {
                *x /= n;
            }
        }
        v
    };
    ScanInputs {
        q: l2(uniform(t * qkv, seed ^ 0x51, -1.0, 1.0)),
        k: l2(uniform(t * qkv, seed ^ 0x52, -1.0, 1.0)),
        v: uniform(t * qkv, seed ^ 0x53, -1.0, 1.0),
        g: uniform(t * qkv, seed ^ 0x54, -g_mag, 0.0),
        beta: uniform(t * h, seed ^ 0x55, 0.1, 0.9),
        state: uniform(h * d * d, seed ^ 0x56, -0.2, 0.2),
    }
}

/// Run one scan form on `inp` (rows [t0, t0+t)) from `state_in`, returning (o, state_out).
#[allow(clippy::type_complexity)]
fn run_scan(
    e: &Engine,
    inp: &ScanInputs,
    t0: usize,
    t: usize,
    state_in: &[f32],
    chunked: bool,
) -> (Vec<f32>, Vec<f32>) {
    let h = HEADS as usize;
    let d = KDA_HEAD_DIM;
    let qkv = h * d;
    let scale = 1.0 / (d as f32).sqrt();
    let q = e.htod(&inp.q[t0 * qkv..(t0 + t) * qkv]).unwrap();
    let k = e.htod(&inp.k[t0 * qkv..(t0 + t) * qkv]).unwrap();
    let v = e.htod(&inp.v[t0 * qkv..(t0 + t) * qkv]).unwrap();
    let g = e.htod(&inp.g[t0 * qkv..(t0 + t) * qkv]).unwrap();
    let beta = e.htod(&inp.beta[t0 * h..(t0 + t) * h]).unwrap();
    let s_in = e.htod(state_in).unwrap();
    let mut s_out = e.zeros(h * d * d).unwrap();
    let mut o = e.zeros(t * qkv).unwrap();
    if chunked {
        e.kda_scan_chunked(
            &q, &k, &v, &g, &beta, &s_in, &mut s_out, &mut o, h, t, scale, C,
        )
        .expect("chunked KDA scan");
    } else {
        e.kda_scan(
            &q, &k, &v, &g, &beta, &s_in, &mut s_out, &mut o, h, t, scale,
        )
        .expect("sequential KDA scan");
    }
    (e.dtoh(&o).unwrap(), e.dtoh(&s_out).unwrap())
}

/// GATE 1 — full mixer vs the pinned reference with the chunked scan ENGAGED, at widths that
/// cross every chunk-boundary shape: below one chunk (63), exactly one (64), one plus a
/// remainder (65), exactly two (128), two plus a remainder (130), exactly three (192).
#[test]
fn chunked_prefill_matches_reference_across_chunk_boundaries() {
    let _g = lock_and_arm_chunked();
    let h = Harness::new();
    for &tokens in &[63usize, 64, 65, 128, 130, 192] {
        let x = hidden_states(tokens, 0xC4C4 ^ tokens as u64);
        let want = h.reference(&x, tokens);
        let x_d = h.engine.htod(&x).unwrap();
        let got_d =
            kda_attn(&h.engine, &h.layer, &x_d, tokens, EPS).expect("GPU chunked KDA prefill");
        let got = h.engine.dtoh(&got_d).unwrap();
        check(&format!("chunked prefill T={tokens}"), &got, &want);
    }
}

/// GATE 2 — chunked vs sequential on identical stage-5 inputs with a NONZERO carried-in
/// state (the case zero-state mixer gates cannot see), crossing the same boundary shapes.
/// Both the output and the carried-out state are banded.
#[test]
fn chunked_scan_matches_sequential_scan_with_carried_state() {
    let _g = lock_and_arm_chunked();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    for &t in &[63usize, 64, 65, 128, 145, 192] {
        let inp = scan_inputs(t, 0x5CA0 ^ t as u64, 0.5);
        let (o_seq, s_seq) = run_scan(&e, &inp, 0, t, &inp.state, false);
        let (o_chk, s_chk) = run_scan(&e, &inp, 0, t, &inp.state, true);
        check(&format!("scan out T={t}"), &o_chk, &o_seq);
        check(&format!("scan state T={t}"), &s_chk, &s_seq);
    }
}

/// GATE 3 — split-invariance BIT-identity: one chunked call over 192 tokens equals two
/// chunked calls split at a multiple of C (64+128 and 128+64), byte for byte in both the
/// outputs and the final state. This is the one exactness claim the chunked form makes, and
/// it is what lets the prime-chunk schedule (4096-token multiples of C) stay bit-stable.
#[test]
fn chunked_split_at_chunk_multiples_is_bit_identical() {
    let _g = lock_and_arm_chunked();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let t = 192usize;
    let inp = scan_inputs(t, 0xB17, 0.5);
    let (o_one, s_one) = run_scan(&e, &inp, 0, t, &inp.state, true);
    for &split in &[64usize, 128] {
        let (o_a, s_mid) = run_scan(&e, &inp, 0, split, &inp.state, true);
        let (o_b, s_end) = run_scan(&e, &inp, split, t - split, &s_mid, true);
        let mut o_two = o_a;
        o_two.extend_from_slice(&o_b);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        assert_eq!(
            bits(&o_two),
            bits(&o_one),
            "split {split}/{}: chunked outputs are not bit-identical across a C-multiple split",
            t - split
        );
        assert_eq!(
            bits(&s_end),
            bits(&s_one),
            "split {split}/{}: chunked final state is not bit-identical across a C-multiple split",
            t - split
        );
        println!(
            "[kda-chunk gate] split {split}/{}: BIT-identical (out + state)",
            t - split
        );
    }
}

/// GATE 4 — decode byte-identity: with the flag ON, a decode step (t=1, Decode conv arm)
/// runs the identical sequential kernel and produces BYTE-identical output and state to the
/// flag-OFF step from the same starting state. Proves the seam cannot touch decode.
#[test]
fn decode_is_byte_identical_with_the_flag_on() {
    let _g = lock_and_arm_chunked();
    let h = Harness::new();
    let t0 = 65usize;
    let steps = 3usize;
    let total = t0 + steps;
    let x = hidden_states(total, 0xDECBE);

    // One stateful chunked prime builds the shared starting state.
    let mut ring0 = h.engine.zeros(CONV_WIDTH * PAD).unwrap();
    let state0 = h.engine.zeros(STATE_WIDTH).unwrap();
    let mut state0_out = h.engine.zeros(STATE_WIDTH).unwrap();
    let x0 = h.engine.htod(&x[..t0 * HIDDEN]).unwrap();
    kda_attn_prime(
        &h.engine,
        &h.layer,
        &x0,
        t0,
        EPS,
        &mut ring0,
        &state0,
        &mut state0_out,
    )
    .expect("chunked stateful prime");
    let ring_h = h.engine.dtoh(&ring0).unwrap();
    let state_h = h.engine.dtoh(&state0_out).unwrap();

    // Decode the remaining tokens under BOTH env arms from clones of that state.
    let mut arms: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
    for flag in ["1", "0"] {
        // SAFETY: under GPU_LOCK; the seam reads the env per call.
        unsafe { std::env::set_var("MEMRA_KDA_CHUNKED", flag) };
        let mut ring = h.engine.htod(&ring_h).unwrap();
        let mut state = h.engine.htod(&state_h).unwrap();
        let mut state_alt = h.engine.zeros(STATE_WIDTH).unwrap();
        let mut out_all = Vec::new();
        for step in 0..steps {
            let row = t0 + step;
            let xs = h.engine.htod(&x[row * HIDDEN..(row + 1) * HIDDEN]).unwrap();
            let out = kda_attn_decode(
                &h.engine,
                &h.layer,
                &xs,
                EPS,
                &mut ring,
                &state,
                &mut state_alt,
            )
            .expect("decode step");
            std::mem::swap(&mut state, &mut state_alt);
            out_all.extend_from_slice(&h.engine.dtoh(&out).unwrap());
        }
        arms.push((
            out_all,
            h.engine.dtoh(&state).unwrap(),
            h.engine.dtoh(&ring).unwrap(),
        ));
    }
    unsafe { std::env::set_var("MEMRA_KDA_CHUNKED", "1") };
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    assert_eq!(
        bits(&arms[0].0),
        bits(&arms[1].0),
        "decode outputs differ between flag arms — the chunked seam leaked into decode"
    );
    assert_eq!(
        bits(&arms[0].1),
        bits(&arms[1].1),
        "decode state differs between flag arms"
    );
    assert_eq!(
        bits(&arms[0].2),
        bits(&arms[1].2),
        "decode conv ring differs between flag arms"
    );
    println!("[kda-chunk gate] decode {steps} steps after prime {t0}: BYTE-identical both arms");
}

/// GATE 5 — stateful two-call prime plus decode continuation vs a single full-sequence
/// reference recompute, chunked engaged throughout: the conv-ring carry, the recurrent-state
/// carry, and the prefill-to-decode handoff under the chunked scan, banded on the reference.
#[test]
fn chunked_stateful_prime_and_decode_match_full_recompute() {
    let _g = lock_and_arm_chunked();
    let h = Harness::new();
    for &(t0, t1, steps) in &[(64usize, 66usize, 3usize), (65, 63, 2), (128, 64, 2)] {
        let total = t0 + t1 + steps;
        let x = hidden_states(total, 0x57A7E ^ total as u64);
        let want = h.reference(&x, total);

        let mut ring = h.engine.zeros(CONV_WIDTH * PAD).unwrap();
        let mut state = h.engine.zeros(STATE_WIDTH).unwrap();
        let mut state_alt = h.engine.zeros(STATE_WIDTH).unwrap();
        let mut got = Vec::with_capacity(total * HIDDEN);
        let mut start = 0usize;
        for len in [t0, t1] {
            let xs = h
                .engine
                .htod(&x[start * HIDDEN..(start + len) * HIDDEN])
                .unwrap();
            let out = kda_attn_prime(
                &h.engine,
                &h.layer,
                &xs,
                len,
                EPS,
                &mut ring,
                &state,
                &mut state_alt,
            )
            .expect("chunked stateful prime");
            std::mem::swap(&mut state, &mut state_alt);
            got.extend_from_slice(&h.engine.dtoh(&out).unwrap());
            start += len;
        }
        for step in 0..steps {
            let row = t0 + t1 + step;
            let xs = h.engine.htod(&x[row * HIDDEN..(row + 1) * HIDDEN]).unwrap();
            let out = kda_attn_decode(
                &h.engine,
                &h.layer,
                &xs,
                EPS,
                &mut ring,
                &state,
                &mut state_alt,
            )
            .expect("decode step after chunked prime");
            std::mem::swap(&mut state, &mut state_alt);
            got.extend_from_slice(&h.engine.dtoh(&out).unwrap());
        }
        check(
            &format!("chunked prime {t0}+{t1} then decode {steps}"),
            &got,
            &want,
        );
    }
}

/// RED ARM 1 — state not carried across a chunk boundary. Mutant: the second window starts
/// from a ZEROED state instead of the carried one. Must exceed the band, or a dropped K4
/// carry would ship invisibly. The gates use g magnitudes small enough that the first
/// window's state still matters at the second window's outputs (a heavy-decay fixture would
/// make this arm insensitive by construction).
#[test]
fn red_arm_state_not_carried_across_chunk_boundary() {
    let _g = lock_and_arm_chunked();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let t = 128usize;
    let inp = scan_inputs(t, 0x2ED1, 0.02);
    let (o_truth, s_truth) = run_scan(&e, &inp, 0, t, &inp.state, false);
    let (_o_a, _s_mid) = run_scan(&e, &inp, 0, 64, &inp.state, true); // the mutant DROPS this carry
    let zero_state = vec![0.0f32; STATE_WIDTH];
    let (o_b, s_end) = run_scan(&e, &inp, 64, 64, &zero_state, true);
    check_red(
        "state-drop mutant (out, window 2)",
        &o_b,
        &o_truth[64 * QKV..],
    );
    check_red("state-drop mutant (state)", &s_end, &s_truth);
}

/// RED ARM 2 — gate cumulative product off by one. Mutant: the chunked form consumes an
/// EXCLUSIVE per-chunk cumsum (each token's gate shifted one position later within its
/// chunk), which is exactly the K1 off-by-one a wrong inclusive/exclusive convention
/// produces. Must exceed the band.
#[test]
fn red_arm_gate_cumulative_product_off_by_one() {
    let _g = lock_and_arm_chunked();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let t = 128usize;
    let inp = scan_inputs(t, 0x0FF1, 0.5);
    let (o_truth, s_truth) = run_scan(&e, &inp, 0, t, &inp.state, false);
    let mut mutant = scan_inputs(t, 0x0FF1, 0.5);
    // Exclusive-cumsum emulation: within every C-token chunk, g'[0] = 0 and g'[j] = g[j-1].
    for chunk_start in (0..t).step_by(C) {
        let end = (chunk_start + C).min(t);
        for j in (chunk_start + 1..end).rev() {
            for d in 0..QKV {
                mutant.g[j * QKV + d] = inp.g[(j - 1) * QKV + d];
            }
        }
        for d in 0..QKV {
            mutant.g[chunk_start * QKV + d] = 0.0;
        }
    }
    let (o_bad, s_bad) = run_scan(&e, &mutant, 0, t, &inp.state, true);
    check_red("gcum off-by-one mutant (out)", &o_bad, &o_truth);
    check_red("gcum off-by-one mutant (state)", &s_bad, &s_truth);
}

/// RED ARM 3 — decay applied twice at the boundary. Mutant: the state handed to the second
/// window is decayed ONCE MORE by the first window's end-of-chunk per-channel gate (the K4
/// double-application shape). Must exceed the band.
#[test]
fn red_arm_decay_applied_twice_at_boundary() {
    let _g = lock_and_arm_chunked();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let t = 128usize;
    let inp = scan_inputs(t, 0xDB1D, 0.02);
    let (o_truth, s_truth) = run_scan(&e, &inp, 0, t, &inp.state, false);
    let (_o_a, s_mid) = run_scan(&e, &inp, 0, 64, &inp.state, true);
    // Per-channel Gcum of window 1's last token: G[h*D+i] = sum over its 64 tokens of g.
    let d = KDA_HEAD_DIM;
    let mut gtot = vec![0.0f32; QKV];
    for j in 0..64 {
        for (ch, tot) in gtot.iter_mut().enumerate() {
            *tot += inp.g[j * QKV + ch];
        }
    }
    // State layout is the transposed M[col][i] at (h*D + col)*D + i; decay runs over i.
    let mut s_double = s_mid.clone();
    for h_i in 0..HEADS as usize {
        for col in 0..d {
            for i in 0..d {
                s_double[(h_i * d + col) * d + i] *= gtot[h_i * d + i].exp();
            }
        }
    }
    let (o_b, s_end) = run_scan(&e, &inp, 64, 64, &s_double, true);
    check_red(
        "double-decay mutant (out, window 2)",
        &o_b,
        &o_truth[64 * QKV..],
    );
    check_red("double-decay mutant (state)", &s_end, &s_truth);
}
