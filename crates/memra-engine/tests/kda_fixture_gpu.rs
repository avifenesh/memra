//! GPU-vs-reference gate for the glm5_next KDA (Kimi Delta Attention) mixer.
//!
//! Truth is `memra_reference::kimi_delta_net_layer` — the same `kimi_delta_net` the portable
//! trunk executor dispatches, pinned by
//! `kimi_delta_net_matches_hand_derived_three_token_recurrence`. The candidate is
//! `memra_engine::kda`, loaded through the real contract names from a fixture `TensorSource`, so
//! the loader (tensor names, shapes, the conv-weight fusion) is under gate too — not just the
//! kernels.
//!
//! MIXER-SCOPED BY DESIGN. The gate runs ONE KDA layer over hidden states it supplies directly,
//! never a whole glm5_next model: that family's residual topology (Sinkhorn hyper-connections)
//! and its MLA/DSA layers are separate surfaces owned elsewhere, and a KDA parity claim must not
//! be able to pass or fail on them.
//!
//! head_dim is 128 here because 128 is what glm5_next ships and the only width
//! `memra_kda_scan_s128` is instantiated for — gating a different instantiation would prove
//! nothing about the serving geometry.

use memra_engine::Engine;
use memra_engine::kda::{KdaAttnLayer, kda_attn, kda_attn_decode, kda_attn_prime};
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

const HIDDEN: usize = 256;
const HEADS: u32 = 2;
const HEAD_DIM: u32 = 128;
const CONV_KERNEL: u32 = 4;
const GATE_LOWER_BOUND: f32 = -5.0;
/// The layer norm epsilon the gated output norm uses. Distinct from the KDA L2 norm's fixed
/// 1e-6, which lives inside the engine and the reference alike.
const EPS: f32 = 1e-5;
/// Scale-relative bound. The reference sums on the host in declaration order while the kernels
/// use warp-tree reductions and cuBLAS GEMMs, so bit-identity is not the bar; the mla_gpu fixture
/// uses the same `maxdiff <= tol * scale` shape.
///
/// CALIBRATED, not guessed (5090, TF32 off, 2026-08-27): the worst of the 18 comparisons these
/// three gates make is 1.3e-6 relative, so 5e-5 carries ~40x headroom for reduction order while
/// staying an order of magnitude BELOW the ~7e-4 that TF32-on costs — an accidental return to
/// TF32 compute fails this bar instead of hiding under it (the dflash2 parity lesson).
const TOL: f32 = 5e-5;

const QKV: usize = (HEADS * HEAD_DIM) as usize;
const CONV_WIDTH: usize = 3 * QKV;
const PAD: usize = CONV_KERNEL as usize - 1;
const STATE_WIDTH: usize = (HEADS * HEAD_DIM * HEAD_DIM) as usize;

fn kda_plan() -> KimiDeltaNetPlan {
    KimiDeltaNetPlan {
        num_heads: HEADS,
        head_dim: HEAD_DIM,
        conv_kernel: CONV_KERNEL,
        gate_lower_bound: GATE_LOWER_BOUND,
    }
}

/// One KDA layer with a dense MLP and a serial residual — the smallest plan that makes
/// `deterministic_fixture` emit this layer's KDA tensors and `TensorContract::for_plan` name
/// them. The MLP and residual are inert scaffolding; nothing downstream of the mixer runs.
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
            ple: None,
            sparse_overlay: None,
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
        exit_mixer: None,
        draft_source: DraftSourcePlan::Embedded,
        sampling_defaults: None,
        partition_boundaries: Vec::new(),
    }
}

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
}

/// Serves the reference fixture's own numbers under the contract's ggml names, so the reference
/// and the GPU read ONE set of weights. `config()` is unreachable: `GpuTensor::load_from_source`
/// never consults it, and fabricating a ModelConfig to satisfy a trait method nobody calls would
/// be a second, unpinned source of truth.
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

/// Deterministic hidden states. Same numbers reach the reference (host slice) and the GPU
/// (uploaded verbatim), so any difference is the mixer's, never the input's.
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

fn check(name: &str, got: &[f32], want: &[f32]) {
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{name}: GPU output has non-finite values"
    );
    let scale = scale_of(want);
    let md = maxdiff(got, want);
    assert!(
        md <= TOL * scale,
        "{name}: GPU vs reference maxdiff {md:.3e} (scale {scale:.3e}, rel {:.3e}, tol {TOL:.1e})",
        md / scale
    );
}

/// cuBLASLt f32 compute rides TF32 (19-bit mantissa) by default on Blackwell. That is the right
/// setting for SERVING and the wrong one for a parity gate: TF32-on costs ~7e-4 relative through
/// the projections alone, so any bar that passed would be wide enough to hide a semantic wiring
/// bug — the lesson banked in crates/memra-engine/src/bin/dflash2_parity.rs, which refuses to run
/// without NVIDIA_TF32_OVERRIDE=0. The driver reads that variable at CUDA init, so it has to be
/// set before the first `Engine::new` in this process; `call_once` blocks every other test thread
/// until it is, and no CUDA call happens before this returns. If it ever failed to take effect,
/// TOL below fails loudly rather than passing on a widened bar.
fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: this process has made no CUDA call and handed out no Engine yet, and
            // call_once serializes every test thread behind this write.
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
        force_true_f32();
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

    /// Reference output for `x` over `tokens`, from a zero state.
    fn reference(&self, x: &[f32], tokens: usize) -> Vec<f32> {
        kimi_delta_net_layer(0, &self.plan, EPS, &self.weights, x, tokens, HIDDEN)
            .expect("reference KDA layer")
            .0
    }
}

/// GATE 1 — prefill parity from a zero state, at lengths that straddle the 64-token chunk size.
/// The lengths are load-bearing for the chunked-prefill twin this increment defers: 63/64/65/130
/// are exactly the boundaries a chunked scan can get wrong, and they are already armed here.
#[test]
fn kda_prefill_matches_reference_across_chunk_boundaries() {
    let h = Harness::new();
    for &tokens in &[1usize, 7, 63, 64, 65, 130] {
        let x = hidden_states(tokens, 0xA11CE ^ tokens as u64);
        let want = h.reference(&x, tokens);
        let x_d = h.engine.htod(&x).unwrap();
        let got_d = kda_attn(&h.engine, &h.layer, &x_d, tokens, EPS).expect("GPU KDA prefill");
        let got = h.engine.dtoh(&got_d).unwrap();
        check(&format!("prefill T={tokens}"), &got, &want);
    }
}

/// GATE 2 — decode after prefill. Prefill `t0` tokens statefully, then step the remaining tokens
/// one at a time through the decode arm; the concatenation must equal a single full-sequence
/// reference recompute. This is where a wrong conv ring or a dropped state carry shows up: both
/// are invisible to gate 1, which always starts from zero.
#[test]
fn kda_decode_after_prefill_matches_full_recompute() {
    let h = Harness::new();
    for &(t0, steps) in &[(1usize, 4usize), (7, 3), (63, 5), (64, 4), (65, 3)] {
        let total = t0 + steps;
        let x = hidden_states(total, 0xDEC0DE ^ total as u64);
        let want = h.reference(&x, total);

        let mut ring = h.engine.zeros(CONV_WIDTH * PAD).unwrap();
        let mut state = h.engine.zeros(STATE_WIDTH).unwrap();
        let mut state_alt = h.engine.zeros(STATE_WIDTH).unwrap();

        let x0 = h.engine.htod(&x[..t0 * HIDDEN]).unwrap();
        let out0 = kda_attn_prime(
            &h.engine,
            &h.layer,
            &x0,
            t0,
            EPS,
            &mut ring,
            &state,
            &mut state_alt,
        )
        .expect("GPU KDA stateful prefill");
        std::mem::swap(&mut state, &mut state_alt);
        let mut got = h.engine.dtoh(&out0).unwrap();

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
            .expect("GPU KDA decode step");
            std::mem::swap(&mut state, &mut state_alt);
            got.extend_from_slice(&h.engine.dtoh(&out).unwrap());
        }
        check(&format!("prime {t0} + decode {steps}"), &got, &want);
    }
}

/// GATE 3 — chunk-boundary invariance. The same sequence primed in two stateful chunks must
/// equal the single-shot reference, for splits that straddle 64. The sequential scan makes this
/// uniform by construction; the gate exists so the chunked twin cannot land without it, and so a
/// conv-ring roll that is wrong only for `T < kernel-1` (the 3-token second chunk) is caught.
#[test]
fn kda_two_chunk_prime_matches_single_shot() {
    let h = Harness::new();
    let total = 130usize;
    let x = hidden_states(total, 0xC0FFEE);
    let want = h.reference(&x, total);

    for &split in &[1usize, 2, 3, 63, 64, 65, 127] {
        let mut ring = h.engine.zeros(CONV_WIDTH * PAD).unwrap();
        let mut state = h.engine.zeros(STATE_WIDTH).unwrap();
        let mut state_alt = h.engine.zeros(STATE_WIDTH).unwrap();
        let mut got = Vec::with_capacity(total * HIDDEN);
        let mut start = 0usize;
        for len in [split, total - split] {
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
            .expect("GPU KDA chunked prime");
            std::mem::swap(&mut state, &mut state_alt);
            got.extend_from_slice(&h.engine.dtoh(&out).unwrap());
            start += len;
        }
        check(
            &format!("two-chunk prime split {split}/{}", total - split),
            &got,
            &want,
        );
    }
}
