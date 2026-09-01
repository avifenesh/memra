//! Acceptance gate for the glm5_next BF16-RESIDENT trunk riding the `MEMRA_PP_BF16`
//! tensor-core prefill door (`f16_ffi::bf16_tc_gemm`, cuBLASLt `CUDA_R_16BF` TN on the raw
//! checkpoint bytes): the L2 lever of the prefill-gap plan
//! (`research/glm53-flash-bringup-20260827/prefill-gap-20260829/PREFILL-GAP.md`).
//!
//! WHAT IS UNDER GATE. On the real GLM-5.3-Flash NVFP4 artifact the precision split keeps
//! every KDA projection in checkpoint BF16 (`modules_to_not_convert`, the keeplist fix in
//! BRINGUP.md), so with `MEMRA_BF16_MMV=1` the big four per layer (`kda_q/k/v/out`, 33.5M
//! elements each) load `GpuTensor::FloatBf16` while the low-rank pairs and `b_proj` (< 2M)
//! stay `Float` f32: a MIXED-residency layer. This gate builds exactly that split on the
//! `kda_fixture_gpu` fixture family and proves, against `memra_reference::kimi_delta_net_layer`:
//!
//!   * REFERENCE BAND at t in {16, 64, 4096}: the mixer output with the door ENGAGED stays in
//!     the bf16-activation-cast numeric class vs the reference fed the SAME weight values
//!     (the FloatBf16 bytes' exact f32 expansion: value-identical weights, so the band
//!     measures ONLY the door's numeric config: the f32->bf16 activation cast + the
//!     tensor-core accumulate order). Weight bytes are the checkpoint's own; the "strictly
//!     closer to the checkpoint" argument of the MEMRA_PP_BF16 FLAGS row holds by construction.
//!   * ENGAGEMENT at the invocation: `f16_ffi::bf16_tc_dispatches()` deltas, not log greps
//!     (LAW:wiring-assertions-match-prose): 4 accepted tensor-core GEMMs per mixer call
//!     (q, k, v, out), zero below the m=16 knee.
//!   * DECODE IS BYTE-UNTOUCHED, measured not assumed: at m in {1, 2, 15} with the flag ON,
//!     `Engine::matmul` output is BIT-identical to the expansion program (`bf16_to_f32` +
//!     f32 cuBLASLt `linear`) reconstructed call-for-call, and the dispatch counter stays 0.
//!     The red twin proves the comparator has teeth: the door FORCED at m=15 produces bytes
//!     that FAIL the same identity check, so a predicate leak (m>=16 loosened) cannot pass
//!     silently.
//!
//! THE RED ARMS, each a wrong answer someone would actually ship:
//!   * `transposed weight bytes`: the cuBLASLt TN form consumes the checkpoint's row-major
//!     [out, in]; a transposed operand is the classic layout swap and must exceed the
//!     mutation floor (in-fixture, square 256x256 so the byte count cannot save you).
//!   * `raw f32 bits fed as bf16` (the dropped activation cast): the exact byte stream a
//!     skipped `f32->bf16` convert would hand the GEMM, reconstructed losslessly through the
//!     public entry (each u16 of the f32 buffer's first half lifted to the f32 whose RNE
//!     bf16 cast is that u16), must exceed the mutation floor.
//!   * `door forced into decode` (above): byte-identity red.
//!   * Engine-side predicate mutations that cannot be expressed through public entries
//!     (dropping the `m >= 16` guard in `linear_bf16_chunked_inner`, skipping the convert
//!     inside `cu/f16_prefill.cu`) follow the deliberate-patch protocol: built, run RED
//!     against THIS gate, banked in
//!     `research/glm53-flash-bringup-20260827/tc-trunk-prefill-20260829/`, reverted.
//!
//! SCOPE, or what this gate does NOT prove. A 2-head 256-hidden fixture, not the 190.7 GB
//! artifact; the mHC hyper walk (raw `CudaSlice<f32>` sites, structurally unreachable by the
//! door), the MoE router (< 2M, exact f32), and the MLA 3-D f32 planes are other gates'
//! surfaces. It proves nothing about throughput: the flip needs the interleaved x5 box A/B
//! with TTFD, the sampled vendor-default twin, the max_tokens=1 argmax gate on the three real
//! prompts, and the 8-draw census on any flip, per the FLAGS row.
//!
//! GPU-gated (`#[ignore]`); rig law = exactness only, never timing. Run under
//! `flock /tmp/memra-5090.lock` with `NVIDIA_TF32_OVERRIDE=0` and `-- --ignored --test-threads=1`.

use memra_engine::Engine;
use memra_engine::f16_ffi::bf16_tc_dispatches;
use memra_engine::kda::{KdaAttnLayer, kda_attn};
use memra_engine::model::GpuTensor;
use memra_gguf::GgmlType;
use memra_gguf::config::ModelConfig;
use memra_gguf::model_plan::{
    ActivationPlan, AttentionPlan, DenseMlpPlan, DraftSourcePlan, KimiDeltaNetPlan, LayerPlan,
    MlpPlan, ModelPlan, NormKind, NormPlan, ResidualTopology, StatePlan, WeightTransform,
};
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, LayerTensor, OutputHead, TensorContract, TensorId,
    TensorMatch,
};
use memra_reference::{ReferenceTensor, deterministic_fixture, kimi_delta_net_layer};
use std::borrow::Cow;
use std::collections::BTreeMap;

const HIDDEN: usize = 256;
const HEADS: u32 = 2;
const HEAD_DIM: u32 = 128;
const CONV_KERNEL: u32 = 4;
const GATE_LOWER_BOUND: f32 = -5.0;
const EPS: f32 = 1e-5;
const QKV: usize = (HEADS * HEAD_DIM) as usize;
const CONV_WIDTH: usize = 3 * QKV;
const STATE_WIDTH: usize = (HEADS * HEAD_DIM * HEAD_DIM) as usize;

/// The task's t set: the m=16 GEMM knee, a mid prefill, and the real chunk cap
/// (`PRIME_CHUNK_MAX_TOKENS`).
const PREFILL_WIDTHS: [usize; 3] = [16, 64, 4096];
/// Below the knee: decode (1), the smallest verify tier (2), and the last pre-knee width (15).
const DECODE_WIDTHS: [usize; 3] = [1, 2, 15];

/// Scale-relative band for the DOOR's numeric class vs the value-identical reference.
/// Weight values are shared exactly (bf16 bytes expanded), so the only sources of distance are
/// the f32->bf16 activation cast (8-bit significand, worst rel ~2^-9 per element) and the
/// tensor-core f32-accumulate order, run through the full KDA chain (conv+SiLU, L2 norm,
/// sigmoid gates, delta-rule scan, gated RMSNorm, out-projection).
///
/// CALIBRATED, not guessed (5090, TF32 off, NVIDIA_TF32_OVERRIDE=0, 2026-08-29, banked in
/// `research/glm53-flash-bringup-20260827/tc-trunk-prefill-20260829/gate-green.log`): the
/// measured green rows are mixer T=16/64/4096 rel 3.526e-3 / 3.796e-3 / 3.653e-3 and wq GEMM
/// m=16/64/4096 rel 2.501e-3 / 1.701e-3 / 1.887e-3: worst 3.796e-3, the bf16 activation-cast
/// class (~2^-9 per element) amplified by the KDA chain. 8e-3 sits 2.1x above the measured
/// worst (the kda_quant_operand headroom protocol) and 3.75x below the mutation floor.
/// Calibrate downward, never upward: a drift above this bar is a finding, not a tolerance
/// problem.
const TOL_BF16: f32 = 8e-3;
/// Red-arm floor: every mutation must land above this. Measured red rows (same run):
/// transposed bytes rel 1.708e0, dropped activation cast rel 7.984e37 (bf16-garbage exponents
/// explode). The floor sits 57x below the tightest red and 3.75x above the green bar.
const MUTATION_FLOOR: f32 = 3e-2;

fn kda_plan() -> KimiDeltaNetPlan {
    KimiDeltaNetPlan {
        num_heads: HEADS,
        head_dim: HEAD_DIM,
        conv_kernel: CONV_KERNEL,
        gate_lower_bound: GATE_LOWER_BOUND,
    }
}

/// One KDA layer with a dense MLP and a serial residual: `kda_fixture_gpu`'s plan, kept
/// textually in step so the two gates measure one fixture family.
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
        context_length: 8192,
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

/// Deterministic hidden states: `kda_fixture_gpu`'s generator, same seed discipline.
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

fn check_band(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{name}: GPU output has non-finite values"
    );
    let scale = scale_of(want);
    let md = maxdiff(got, want);
    eprintln!(
        "[bf16-tc-gate] {name}: maxdiff {md:.3e} scale {scale:.3e} rel {:.3e} (tol {tol:.1e})",
        md / scale
    );
    assert!(
        md <= tol * scale,
        "{name}: maxdiff {md:.3e} (scale {scale:.3e}, rel {:.3e}) exceeds tol {tol:.1e}",
        md / scale
    );
}

fn check_red(name: &str, got: &[f32], want: &[f32]) {
    let scale = scale_of(want);
    let md = maxdiff(got, want);
    eprintln!(
        "[bf16-tc-gate] RED {name}: maxdiff {md:.3e} scale {scale:.3e} rel {:.3e} (floor {MUTATION_FLOOR:.1e})",
        md / scale
    );
    assert!(
        md > MUTATION_FLOOR * scale,
        "{name}: mutation maxdiff {md:.3e} (rel {:.3e}) did NOT exceed the {MUTATION_FLOOR:.1e} \
         floor: the gate would not catch this breakage",
        md / scale
    );
}

/// f32 -> bf16 with round-to-nearest-even: the same rounding `__float2bfloat16` applies, so
/// the fixture's resident bytes are the class the real loader keeps (raw checkpoint bf16).
fn f32_to_bf16_rne(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7FFF).wrapping_add(lsb)) >> 16) as u16
}

fn bf16_to_f32_exact(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// The four TensorIds whose real-artifact twins are >= 2M elements and load `FloatBf16` under
/// `MEMRA_BF16_MMV=1`. Everything else on the layer (low-rank pairs, b_proj, conv, norms)
/// stays f32: the mixed-residency split this gate exists to measure.
fn bf16_resident_ids() -> [TensorId; 4] {
    [
        TensorId::Layer {
            index: 0,
            tensor: LayerTensor::KdaQuery,
        },
        TensorId::Layer {
            index: 0,
            tensor: LayerTensor::KdaKey,
        },
        TensorId::Layer {
            index: 0,
            tensor: LayerTensor::KdaValue,
        },
        TensorId::Layer {
            index: 0,
            tensor: LayerTensor::KdaOutput,
        },
    ]
}

/// TF32 off before the first CUDA call (the kda_fixture_gpu contract), AND the door open:
/// `MEMRA_PP_BF16` is a OnceLock read, so it must be set before anything consults it.
fn force_env() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call has happened and no Engine exists yet; call_once
            // serializes every test thread behind this write.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
        // SAFETY: same single-threaded init window as above.
        unsafe { std::env::set_var("MEMRA_PP_BF16", "1") };
    });
}

struct Harness {
    engine: Engine,
    /// Mixed-residency layer: wq/wk/wv/wo are `FloatBf16` (raw bf16 bytes), the rest f32.
    layer: KdaAttnLayer,
    /// Reference weights with the four resident tensors' VALUES bf16-rounded (the exact f32
    /// expansion of the resident bytes), so reference and GPU share weight values and the
    /// band measures only the door's numeric config.
    weights: BTreeMap<TensorId, ReferenceTensor>,
    plan: KimiDeltaNetPlan,
}

impl Harness {
    fn new() -> Self {
        force_env();
        let model_plan = one_kda_layer_plan();
        let fixture = deterministic_fixture(&model_plan).expect("deterministic KDA fixture");
        let mut weights = fixture.weights.clone();
        // Round the four resident tensors' values through bf16 (RNE), exactly once. The GPU
        // gets these values as raw bf16 bytes; the reference gets their exact f32 expansion.
        for id in bf16_resident_ids() {
            let t = weights
                .get_mut(&id)
                .unwrap_or_else(|| panic!("fixture is missing {id:?}"));
            for v in &mut t.data {
                *v = bf16_to_f32_exact(f32_to_bf16_rne(*v));
            }
        }
        let source = fixture_source(&model_plan, &weights);
        let engine = Engine::new(0).expect("CUDA engine on device 0");
        let plan = kda_plan();
        let mut layer =
            KdaAttnLayer::load(&engine, &source, 0, &plan).expect("KDA mixer loads from contract");
        // Swap the big four to bf16 residency (GpuTensor::FloatBf16, raw checkpoint-class
        // bytes): the state MEMRA_BF16_MMV=1 puts the real artifact's >=2M BF16 tensors in.
        let ids = bf16_resident_ids();
        for (field, id) in [
            (&mut layer.wq, &ids[0]),
            (&mut layer.wk, &ids[1]),
            (&mut layer.wv, &ids[2]),
            (&mut layer.wo, &ids[3]),
        ] {
            *field = bf16_resident(&engine, &weights[id], field.ne().to_vec());
        }
        Self {
            engine,
            layer,
            weights,
            plan,
        }
    }

    fn reference(&self, x: &[f32], tokens: usize) -> Vec<f32> {
        kimi_delta_net_layer(0, &self.plan, EPS, &self.weights, x, tokens, HIDDEN)
            .expect("reference KDA layer")
            .0
    }
}

/// Build a `FloatBf16` from a (bf16-value-rounded) reference tensor. The rounding is
/// idempotent, so encoding the rounded values back to bf16 recovers the exact resident bytes.
fn bf16_resident(e: &Engine, t: &ReferenceTensor, ne: Vec<u64>) -> GpuTensor {
    let bytes: Vec<u8> = t
        .data
        .iter()
        .flat_map(|&v| f32_to_bf16_rne(v).to_le_bytes())
        .collect();
    GpuTensor::FloatBf16 {
        data: e.htod_bytes(&bytes).expect("bf16 resident upload"),
        ne,
    }
}

/// Host row-major GEMM: y[m, out] = x[m, in] @ w[out, in]^T: the single-projection truth.
fn host_linear(x: &[f32], w: &[f32], m: usize, in_f: usize, out_f: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * out_f];
    for mi in 0..m {
        for o in 0..out_f {
            let mut acc = 0.0f64;
            for i in 0..in_f {
                acc += (x[mi * in_f + i] as f64) * (w[o * in_f + i] as f64);
            }
            y[mi * out_f + o] = acc as f32;
        }
    }
    y
}

/// GATE 1: the mixed-residency mixer matches the value-identical reference at every task
/// width, and the door's own counter proves 4 tensor-core dispatches per call (q, k, v, out;
/// the f32 members must not count).
#[test]
#[ignore]
fn bf16_resident_mixer_matches_reference_band_with_door_engaged() {
    let h = Harness::new();
    for &tokens in &PREFILL_WIDTHS {
        let x = hidden_states(tokens, 0xB16C ^ tokens as u64);
        let want = h.reference(&x, tokens);
        let x_d = h.engine.htod(&x).unwrap();
        let before = bf16_tc_dispatches();
        let got_d = kda_attn(&h.engine, &h.layer, &x_d, tokens, EPS).expect("GPU KDA prefill");
        let engaged = bf16_tc_dispatches() - before;
        let got = h.engine.dtoh(&got_d).unwrap();
        check_band(&format!("mixer T={tokens}"), &got, &want, TOL_BF16);
        assert_eq!(
            engaged, 4,
            "T={tokens}: expected exactly 4 accepted bf16 tensor-core GEMMs (q, k, v, out), \
             counted {engaged}: a decline fell back to the f32 dequant GEMM (see the \
             [bf16-tc] DECLINED line) or the door leaked onto an f32 member"
        );
    }
}

/// GATE 2: single-projection band: one door GEMM vs the host f64-accumulated truth on the
/// same bf16 weight values. Isolates the per-GEMM class (activation cast + TC accumulate)
/// from the mixer chain, and is the `want` the red arms below must fail against.
#[test]
#[ignore]
fn single_projection_door_gemm_stays_in_class() {
    let h = Harness::new();
    let wq = &h.weights[&bf16_resident_ids()[0]];
    for &m in &PREFILL_WIDTHS {
        let x = hidden_states(m, 0x51D ^ m as u64);
        let want = host_linear(&x, &wq.data, m, HIDDEN, QKV);
        let x_d = h.engine.htod(&x).unwrap();
        let before = bf16_tc_dispatches();
        let y_d = h.engine.matmul(&h.layer.wq, &x_d, m).expect("door GEMM");
        assert_eq!(
            bf16_tc_dispatches() - before,
            1,
            "m={m}: the wq projection did not ride the tensor-core door"
        );
        let y = h.engine.dtoh(&y_d).unwrap();
        check_band(&format!("wq GEMM m={m}"), &y[..m * QKV], &want, TOL_BF16);
    }
}

/// GATE 3: decode byte-identity, measured. Below the m=16 knee the flag-ON matmul must be
/// BIT-identical to the expansion program (`bf16_to_f32` + f32 `linear`), reconstructed here
/// call-for-call, and the door counter must not move. This is the decode-exact contract's
/// survival proof: with the door on, m < 16 runs byte-for-byte the code it ran before.
#[test]
#[ignore]
fn decode_widths_are_byte_identical_to_the_expansion_program() {
    let h = Harness::new();
    let GpuTensor::FloatBf16 { data, .. } = &h.layer.wq else {
        panic!("harness wq must be FloatBf16");
    };
    for &m in &DECODE_WIDTHS {
        let x = hidden_states(m, 0xDEC0 ^ m as u64);
        let x_d = h.engine.htod(&x).unwrap();
        let before = bf16_tc_dispatches();
        let y_door_on = h
            .engine
            .matmul(&h.layer.wq, &x_d, m)
            .expect("decode matmul");
        assert_eq!(
            bf16_tc_dispatches(),
            before,
            "m={m}: the tensor-core door engaged BELOW the m=16 knee: decode is no longer \
             byte-untouched"
        );
        // The expansion program, exactly as linear_bf16_chunked_inner runs it for this
        // (unchunked) shape: exact bf16->f32 expansion of the resident bytes, then the same
        // f32 cuBLASLt linear.
        let wf32 = h
            .engine
            .bf16_to_f32(&data.slice(0..HIDDEN * QKV * 2), HIDDEN * QKV)
            .expect("bf16 expansion");
        let y_expansion = h
            .engine
            .linear(&x_d, &wf32, m, HIDDEN, QKV)
            .expect("expansion linear");
        let a = h.engine.dtoh(&y_door_on).unwrap();
        let b = h.engine.dtoh(&y_expansion).unwrap();
        let bits_equal = a[..m * QKV]
            .iter()
            .zip(&b[..m * QKV])
            .all(|(p, q)| p.to_bits() == q.to_bits());
        assert!(
            bits_equal,
            "m={m}: flag-ON decode output is not byte-identical to the expansion program"
        );
    }
}

/// GATE 3-RED: the byte-identity comparator has teeth: the door FORCED at m=15 (the widest
/// decode-tier width) produces bytes that FAIL the identity check, so a predicate leak
/// (`m >= 16` loosened toward decode) cannot pass gate 3 silently. If cuBLASLt declines every
/// sub-knee width on this fixture the arm fails loudly rather than passing vacuously.
#[test]
#[ignore]
fn door_forced_into_decode_fails_byte_identity() {
    let h = Harness::new();
    let GpuTensor::FloatBf16 { data, .. } = &h.layer.wq else {
        panic!("harness wq must be FloatBf16");
    };
    let mut proved = false;
    for &m in &[15usize, 2, 1] {
        let x = hidden_states(m, 0x1EAC ^ m as u64);
        let x_d = h.engine.htod(&x).unwrap();
        let Some(y_tc) = h
            .engine
            .bf16_tc_gemm(data, &x_d, m, HIDDEN, QKV)
            .expect("forced door GEMM")
        else {
            eprintln!("[bf16-tc-gate] forced door DECLINED at m={m}; trying the next width");
            continue;
        };
        let wf32 = h
            .engine
            .bf16_to_f32(&data.slice(0..HIDDEN * QKV * 2), HIDDEN * QKV)
            .expect("bf16 expansion");
        let y_expansion = h
            .engine
            .linear(&x_d, &wf32, m, HIDDEN, QKV)
            .expect("expansion linear");
        let a = h.engine.dtoh(&y_tc).unwrap();
        let b = h.engine.dtoh(&y_expansion).unwrap();
        let bits_equal = a[..m * QKV]
            .iter()
            .zip(&b[..m * QKV])
            .all(|(p, q)| p.to_bits() == q.to_bits());
        assert!(
            !bits_equal,
            "m={m}: the forced tensor-core door produced bytes identical to the expansion \
             program: gate 3 could not detect a decode leak at this width"
        );
        proved = true;
        break;
    }
    assert!(
        proved,
        "cuBLASLt declined the forced door at every sub-knee width; the decode-leak red arm \
         proved nothing on this fixture: widen the fixture shapes"
    );
}

/// RED ARM: transposed weight bytes (the layout swap). Square 256x256, so only the layout
/// is wrong, never the byte count.
#[test]
#[ignore]
fn transposed_weight_bytes_go_red() {
    let h = Harness::new();
    let wq = &h.weights[&bf16_resident_ids()[0]];
    assert_eq!(
        HIDDEN, QKV,
        "the transpose mutation needs the square fixture"
    );
    let mut wt = vec![0.0f32; wq.data.len()];
    for o in 0..QKV {
        for i in 0..HIDDEN {
            wt[i * QKV + o] = wq.data[o * HIDDEN + i];
        }
    }
    let mutated = bf16_resident(
        &h.engine,
        &ReferenceTensor {
            shape: wq.shape.clone(),
            data: wt,
        },
        h.layer.wq.ne().to_vec(),
    );
    let m = 64usize;
    let x = hidden_states(m, 0x7A05);
    let want = host_linear(&x, &wq.data, m, HIDDEN, QKV);
    let x_d = h.engine.htod(&x).unwrap();
    let before = bf16_tc_dispatches();
    let y_d = h.engine.matmul(&mutated, &x_d, m).expect("mutated GEMM");
    assert_eq!(
        bf16_tc_dispatches() - before,
        1,
        "the transposed operand must still ride the door (the mutation is the layout, not \
         the dispatch)"
    );
    let y = h.engine.dtoh(&y_d).unwrap();
    check_red("transposed weight bytes", &y[..m * QKV], &want);
}

/// RED ARM: the dropped activation cast: raw f32 bits fed as bf16. A skipped f32->bf16
/// convert hands the GEMM the f32 buffer's bytes read as u16 pairs; that exact stream is
/// reconstructed losslessly through the public entry (each target u16 lifted to the f32
/// whose RNE bf16 cast is itself), so the arm measures what the gate would see if
/// `cu/f16_prefill.cu` ever lost its convert.
#[test]
#[ignore]
fn raw_f32_bits_fed_as_bf16_go_red() {
    let h = Harness::new();
    let wq = &h.weights[&bf16_resident_ids()[0]];
    let GpuTensor::FloatBf16 { data, .. } = &h.layer.wq else {
        panic!("harness wq must be FloatBf16");
    };
    let m = 64usize;
    let x = hidden_states(m, 0xCA57);
    let want = host_linear(&x, &wq.data, m, HIDDEN, QKV);
    // The byte stream a dropped cast would consume: the f32 buffer's first m*k u16 words.
    let raw: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_mut: Vec<f32> = raw
        .chunks_exact(2)
        .take(m * HIDDEN)
        .map(|c| bf16_to_f32_exact(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let x_mut_d = h.engine.htod(&x_mut).unwrap();
    let y_d = h
        .engine
        .bf16_tc_gemm(data, &x_mut_d, m, HIDDEN, QKV)
        .expect("mutated-activation GEMM")
        .expect("cuBLASLt accepted this shape in gate 2; a decline here is a gate defect");
    let y = h.engine.dtoh(&y_d).unwrap();
    check_red("raw f32 bits fed as bf16", &y[..m * QKV], &want);
}
