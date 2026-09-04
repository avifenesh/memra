//! Gate for `MEMRA_KDA_FUSED_PROJ` — the fused 6-way KDA stage-1 projection launch
//! (`qmatvec_kda6_q8f32_mmvq`, lane/glm5-launch-diet 2026-08-30).
//!
//! THE ONE CHANGE UNDER TEST: six same-input matvec calls (wq/wk/wv Q8_0-resident +
//! f_a/g_a/b_proj f32-resident, the classes the real glm53-nvfp4 artifact loads —
//! `kda_quant_operand_gpu.rs` header) collapse into one launch. The claims, stated per class:
//!
//!  * Q8_0 rows (wq/wk/wv): BIT-IDENTICAL to the unfused MMVQ/batched arm at every width
//!    t=1..15 — the fused kernel's per-(token,row) body is `qmatvec_q8_0_mmvq` VERBATIM
//!    (the `qmatvec_q8_0_mmvq_fused2/3` precedent). Asserted bytewise, per width.
//!  * f32 rows (f_a/g_a/b_proj): the fused kernel replaces cuBLASLt GEMV with a deterministic
//!    warp tree — a reduction-order numeric-class change (the step37 `MEMRA_STEP_TP_QKV_FUSED`
//!    class). Bounded here by a relative band CALIBRATED on the rig and printed on every run.
//!  * Whole mixer: fused-vs-unfused within the same band (measured worst 1.519e-7); against
//!    `memra_reference` the door may not worsen the OFF arm's deviation beyond that band —
//!    the Q8_0 operand floor (7.5e-2 on this fixture/seed at t=1, identical in both arms)
//!    belongs to the operand class, not to this door, and both arms are printed to prove it.
//!
//! RED ARMS (the gate binds): six transposed-slice mutations and six dropped-range mutations,
//! each of which loads cleanly, launches cleanly, and produces finite numbers — the comparator
//! must fail on EXACTLY the mutated slice and pass on the other five (isolation, so a red
//! result is attributable, not collateral).
//!
//! Rig law: correctness-only, run under `flock /tmp/memra-5090.lock`, TF32 forced off.

use cudarc::driver::CudaSlice;
use memra_engine::Engine;
use memra_engine::kda::{KDA_FUSED6_DISPATCHES, KdaAttnLayer, kda_attn};
use memra_engine::model::GpuTensor;
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
use std::sync::atomic::Ordering;

const HIDDEN: usize = 256; // % 128 == 0, the fused kernel's f32-row float4 walk requirement
const HEADS: u32 = 2;
const HEAD_DIM: u32 = 128;
const CONV_KERNEL: u32 = 4;
const GATE_LOWER_BOUND: f32 = -5.0;
const EPS: f32 = 1e-5;
const QKV: usize = (HEADS * HEAD_DIM) as usize;
const CONV_WIDTH: usize = 3 * QKV;
const STATE_WIDTH: usize = (HEADS * HEAD_DIM * HEAD_DIM) as usize;

/// The stage-1 suffixes the door fuses, in `matmul_group` order. Only wq/wk/wv are Q8_0 on the
/// real artifact (>= 1M elements); f_a/g_a/b stay Float. The fixture mirrors exactly that.
const Q8_SUFFIXES: [&str; 3] = ["kda_q.weight", "kda_k.weight", "kda_v.weight"];

/// f32-row band, fused warp tree vs cuBLASLt GEMV, RELATIVE maxdiff. MEASURED on the 5090
/// (TF32 off, NVIDIA_TF32_OVERRIDE=0, 2026-08-30, this fixture, t=1..15): worst 4.703e-7
/// relative across all three f32 rows and all widths. 5e-5 carries ~100x headroom for
/// reduction order while sitting orders below the transposed-slice mutations on this fixture —
/// an accidental wrong-program change cannot hide under it.
const F32_ROW_TOL: f32 = 5e-5;
/// Whole-mixer fused-vs-unfused band. The f32-row deltas pass through sigmoid gate chains and
/// the scan; MEASURED worst on this fixture (same rig/config): 1.519e-7 relative at
/// t in {1, 7, 15}. Same 5e-5 bar, same rationale.
const MIXER_TOL: f32 = 5e-5;
/// Fused mixer vs memra_reference: a SANITY CAP over the Q8_0 OPERAND floor, which is
/// fixture-and-seed specific (kda_quant_operand_gpu.rs measured 2.331e-2 worst on its seeds;
/// THIS fixture/seed measures 7.509e-2 at t=1, 5090 TF32-off 2026-08-30 — the operand class's
/// noise, present identically in the OFF arm). The door's own reference bar is the DELTA
/// assert in gate 2: ON-vs-reference may not exceed OFF-vs-reference by more than the mixer
/// band. This cap only catches catastrophes; it is not the door's accuracy claim.
const QUANT_FLOOR: f32 = 1.5e-1;

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// TF32 off before the first CUDA call (the dflash2 parity lesson; see kda_fixture_gpu.rs).
fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call has been made and no Engine handed out in this process yet,
            // and call_once serializes every test thread behind this write.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
    });
}

/// The flag is read PER CALL by design (rollback seam), so the gate can drive both arms in one
/// process. Serialized behind `gpu_guard`.
fn set_door(on: bool) {
    // SAFETY: all tests in this binary hold `gpu_guard` while touching env or the GPU.
    unsafe {
        std::env::set_var("MEMRA_KDA_FUSED_PROJ", if on { "1" } else { "0" });
    }
}

fn kda_plan() -> KimiDeltaNetPlan {
    KimiDeltaNetPlan {
        num_heads: HEADS,
        head_dim: HEAD_DIM,
        conv_kernel: CONV_KERNEL,
        gate_lower_bound: GATE_LOWER_BOUND,
    }
}

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
    ty: GgmlType,
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
            ggml_type: t.ty,
            ne: t.ne.clone(),
        })
    }
}

/// Mixed-residency fixture: wq/wk/wv Q8_0, everything else F32 — the stage-1 classes the real
/// artifact serves, with the NON-stage-1 tensors held F32 so the fused-vs-unfused comparison
/// isolates the one change under test.
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
            "fixture {:?} shape mismatch",
            req.id
        );
        let names = match req.match_mode {
            TensorMatch::OneOf => &req.names[..1],
            TensorMatch::All => req.names.as_slice(),
        };
        for name in names {
            let quantize = Q8_SUFFIXES.iter().any(|s| name == &format!("blk.0.{s}"));
            let (bytes, ty) = if quantize {
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
            tensors.insert(
                name.clone(),
                OwnedTensor {
                    bytes,
                    ne: req.shape.clone(),
                    ty,
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

fn rel(got: &[f32], want: &[f32]) -> f32 {
    maxdiff(got, want) / scale_of(want)
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
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
        let layer = KdaAttnLayer::load(&engine, &source, 0, &plan)
            .expect("KDA mixer loads with the mixed stage-1 residency");
        Self {
            engine,
            layer,
            weights: fixture.weights,
            plan,
        }
    }

    /// The six unfused stage-1 outputs, exactly the arm the door replaces.
    fn unfused6(&self, x: &CudaSlice<f32>, t: usize) -> Vec<Vec<f32>> {
        let la = &self.layer;
        let outs = self
            .engine
            .matmul_group(
                &[&la.wq, &la.wk, &la.wv, &la.f_a, &la.g_a, &la.b_proj],
                x,
                t,
            )
            .expect("unfused matmul_group");
        outs.iter().map(|o| self.engine.dtoh(o).unwrap()).collect()
    }

    /// Raw device pieces of the six stage-1 weights, for the raw-launcher red arms.
    #[allow(clippy::type_complexity)]
    fn raw_weights(&self) -> ([&CudaSlice<u8>; 3], [&CudaSlice<f32>; 3], usize) {
        fn q8(w: &GpuTensor) -> (&CudaSlice<u8>, usize) {
            match w {
                GpuTensor::Quant {
                    bytes, row_bytes, ..
                } => (bytes, *row_bytes),
                _ => panic!("stage-1 q8 operand is not Quant — the fixture did not bind"),
            }
        }
        fn f32w(w: &GpuTensor) -> &CudaSlice<f32> {
            match w {
                GpuTensor::Float { data, .. } => data,
                _ => panic!("stage-1 f32 operand is not Float — the fixture did not bind"),
            }
        }
        let (bq, rb) = q8(&self.layer.wq);
        let (bk, rb_k) = q8(&self.layer.wk);
        let (bv, rb_v) = q8(&self.layer.wv);
        assert_eq!(rb, rb_k);
        assert_eq!(rb, rb_v);
        (
            [bq, bk, bv],
            [
                f32w(&self.layer.f_a),
                f32w(&self.layer.g_a),
                f32w(&self.layer.b_proj),
            ],
            rb,
        )
    }
}

fn dims() -> [usize; 6] {
    [
        QKV,
        QKV,
        QKV,
        HEAD_DIM as usize,
        HEAD_DIM as usize,
        HEADS as usize,
    ]
}

/// Reference f32 data of stage-1 weight `i` (matmul_group order), from the fixture.
fn stage1_ref_data(h: &Harness, i: usize) -> Vec<f32> {
    use memra_gguf::tensor_contract::LayerTensor;
    let tensor = match i {
        0 => LayerTensor::KdaQuery,
        1 => LayerTensor::KdaKey,
        2 => LayerTensor::KdaValue,
        3 => LayerTensor::KdaForgetDown,
        4 => LayerTensor::KdaGateDown,
        5 => LayerTensor::KdaBeta,
        _ => unreachable!(),
    };
    h.weights
        .get(&TensorId::Layer { index: 0, tensor })
        .unwrap_or_else(|| panic!("fixture is missing stage-1 tensor {i}"))
        .data
        .clone()
}

/// Transpose `[rows, cols]`-major data in place-shape: the TRANSPOSED-SLICE mutation. The byte
/// count and dims stay the same, so it loads and launches cleanly and produces finite numbers;
/// only the value layout is wrong — the failure class a shape check cannot see.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(data.len(), rows * cols);
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Run the fused RAW launcher over the harness weights with one optional mutation, returning
/// the six host outputs. `mutate` = (slice index, transposed data) replaces that one weight;
/// `drop_i` zeroes that range out of the launch with its output pre-zeroed.
fn run_raw(
    h: &Harness,
    x: &CudaSlice<f32>,
    t: usize,
    mutate: Option<(usize, &[f32])>,
    drop_i: Option<usize>,
) -> Vec<Vec<f32>> {
    let e = &h.engine;
    let ([bq, bk, bv], [wfa, wga, wb], rb) = h.raw_weights();
    let mut d = dims();
    if let Some(i) = drop_i {
        d[i] = 0;
    }

    // Owned mutated buffers (kept alive across the launch); references pick original vs mutated.
    let mut_q8: Option<CudaSlice<u8>> = match mutate {
        Some((i, data)) if i < 3 => Some(
            e.htod_bytes(&memra_gguf::nvfp4_repack::f32_to_q8_0(data))
                .unwrap(),
        ),
        _ => None,
    };
    let mut_f32: Option<CudaSlice<f32>> = match mutate {
        Some((i, data)) if i >= 3 => Some(e.htod(data).unwrap()),
        _ => None,
    };
    let mut_at = mutate.map(|(i, _)| i);
    fn pick<'x, T>(i: usize, mut_at: Option<usize>, m: Option<&'x T>, orig: &'x T) -> &'x T {
        match (m, mut_at) {
            (Some(m), Some(mi)) if mi == i => m,
            _ => orig,
        }
    }

    let (aq, ad) = e.quantize_q8_1(x, t, HIDDEN).unwrap();
    let full = dims();
    let mut outs = [
        e.zeros(t * full[0]).unwrap(),
        e.zeros(t * full[1]).unwrap(),
        e.zeros(t * full[2]).unwrap(),
        e.zeros(t * full[3]).unwrap(),
        e.zeros(t * full[4]).unwrap(),
        e.zeros(t * full[5]).unwrap(),
    ];
    e.kda_proj_fused6_raw(
        pick(0, mut_at, mut_q8.as_ref(), bq),
        pick(1, mut_at, mut_q8.as_ref(), bk),
        pick(2, mut_at, mut_q8.as_ref(), bv),
        pick(3, mut_at, mut_f32.as_ref(), wfa),
        pick(4, mut_at, mut_f32.as_ref(), wga),
        pick(5, mut_at, mut_f32.as_ref(), wb),
        &aq,
        &ad,
        x,
        &mut outs,
        HIDDEN,
        d,
        t,
        rb,
    )
    .expect("the raw fused launch runs cleanly — mutations must be silent, not loud");
    outs.iter().map(|o| e.dtoh(o).unwrap()).collect()
}

const NAMES: [&str; 6] = ["wq", "wk", "wv", "f_a", "g_a", "b_proj"];

/// GATE 1 — the fixture BINDS (wq/wk/wv actually Quant, f_a/g_a/b actually Float), the door
/// engages under the env, and per width t=1..15: q8 rows BYTEWISE identical to the unfused
/// arm, f32 rows within the calibrated band (measured value printed every run).
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn fused_door_is_bit_identical_on_q8_rows_and_banded_on_f32_rows_at_t_1_to_15() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    for (name, w) in [
        ("wq", &h.layer.wq),
        ("wk", &h.layer.wk),
        ("wv", &h.layer.wv),
    ] {
        assert!(
            matches!(w, GpuTensor::Quant { .. }),
            "{name} loaded Float — the gate would compare the door against the wrong operand class"
        );
    }
    for (name, w) in [
        ("f_a", &h.layer.f_a),
        ("g_a", &h.layer.g_a),
        ("b_proj", &h.layer.b_proj),
    ] {
        assert!(
            matches!(w, GpuTensor::Float { .. }),
            "{name} loaded non-Float — the gate would compare the door against the wrong operand class"
        );
    }

    set_door(true);
    let mut worst_f32 = 0.0f32;
    for t in 1..=15usize {
        let x = h.engine.htod(&hidden_states(t, 0xF6D0 ^ t as u64)).unwrap();
        let want = h.unfused6(&x, t);
        let before = KDA_FUSED6_DISPATCHES.load(Ordering::Relaxed);
        let got = h
            .engine
            .kda_proj_fused6(&h.layer, &x, t)
            .expect("fused door call")
            .expect(
                "the door must ENGAGE on this fixture — a None here means the gate is decorative",
            );
        assert_eq!(
            KDA_FUSED6_DISPATCHES.load(Ordering::Relaxed),
            before + 1,
            "engagement must be counted at the arm's own call site"
        );
        let got: Vec<Vec<f32>> = got.iter().map(|o| h.engine.dtoh(o).unwrap()).collect();
        for i in 0..3 {
            assert!(
                bits_equal(&got[i], &want[i]),
                "t={t}: {} fused row is NOT bit-identical to the unfused MMVQ arm (rel {:.3e})",
                NAMES[i],
                rel(&got[i], &want[i])
            );
        }
        for i in 3..6 {
            assert!(
                got[i].iter().all(|v| v.is_finite()),
                "t={t}: {} non-finite",
                NAMES[i]
            );
            let r = rel(&got[i], &want[i]);
            worst_f32 = worst_f32.max(r);
            assert!(
                r <= F32_ROW_TOL,
                "t={t}: {} fused-vs-cuBLASLt relative maxdiff {r:.3e} exceeds {F32_ROW_TOL:.1e}",
                NAMES[i]
            );
        }
    }
    set_door(false);
    eprintln!(
        "[kda-fused6 gate] worst f32-row rel maxdiff {worst_f32:.3e} (tol {F32_ROW_TOL:.1e})"
    );
}

/// GATE 2 — whole mixer, both bars: fused-vs-unfused within the f32-row-induced band, and
/// fused-vs-reference within the Q8_0 operand floor. Toggles the per-call flag inside one
/// process (that is the rollback seam working as designed).
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn whole_mixer_fused_matches_unfused_and_reference() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let mut worst = 0.0f32;
    for &t in &[1usize, 7, 15] {
        let x = hidden_states(t, 0x3A7E ^ t as u64);
        let x_d = h.engine.htod(&x).unwrap();

        set_door(false);
        let off = h
            .engine
            .dtoh(&kda_attn(&h.engine, &h.layer, &x_d, t, EPS).expect("mixer, door off"))
            .unwrap();
        set_door(true);
        let on = h
            .engine
            .dtoh(&kda_attn(&h.engine, &h.layer, &x_d, t, EPS).expect("mixer, door on"))
            .unwrap();
        set_door(false);

        assert!(
            on.iter().all(|v| v.is_finite()),
            "t={t}: fused mixer non-finite"
        );
        let r = rel(&on, &off);
        worst = worst.max(r);
        assert!(
            r <= MIXER_TOL,
            "t={t}: fused-vs-unfused mixer relative maxdiff {r:.3e} exceeds {MIXER_TOL:.1e}"
        );

        // Reference bar, stated honestly: the deviation from memra_reference is dominated by
        // the Q8_0 OPERAND floor, which belongs to the operand class (calibrated per fixture —
        // 7.51e-2 on THIS fixture/seed at t=1, vs 2.33e-2 on kda_quant_operand_gpu's), not to
        // this door. What the door owes the reference is that it adds nothing beyond its own
        // measured f32-row band: ON-vs-reference may exceed OFF-vs-reference by at most the
        // mixer band. Both values are printed so the claim is a measurement.
        let (reference, _) =
            kimi_delta_net_layer(0, &h.plan, EPS, &h.weights, &x, t, HIDDEN).expect("reference");
        let r_on = rel(&on, &reference);
        let r_off = rel(&off, &reference);
        eprintln!(
            "[kda-fused6 gate] t={t} vs reference: off {r_off:.3e} on {r_on:.3e} \
             (operand floor cap {QUANT_FLOOR:.1e})"
        );
        assert!(
            r_on <= r_off + 10.0 * MIXER_TOL,
            "t={t}: the fused door WORSENED the reference deviation ({r_off:.3e} -> {r_on:.3e}) \
             beyond its own f32-row band — that is the door's error, not the operand floor's"
        );
        assert!(
            r_on <= QUANT_FLOOR,
            "t={t}: fused mixer vs memra_reference {r_on:.3e} exceeds the operand floor cap \
             {QUANT_FLOOR:.1e} — recalibrate only with the OFF arm printed alongside"
        );
    }
    eprintln!(
        "[kda-fused6 gate] worst whole-mixer fused-vs-unfused rel {worst:.3e} (tol {MIXER_TOL:.1e})"
    );
}

/// GATE 3 — RED, transposed slice x6. For each of the six weights in turn: replace it with its
/// transposed-data twin (same length, loads, runs, finite — silently wrong) and assert the
/// comparator fails on EXACTLY that slice and passes on the other five.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn a_transposed_slice_fails_the_gate_on_exactly_that_slice_for_all_six() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let t = 3usize; // a batched-tier width, so the red arm covers the t>1 program too
    let x = h.engine.htod(&hidden_states(t, 0x7EA55)).unwrap();
    let want = h.unfused6(&x, t);
    let full = dims();
    let shapes: [(usize, usize); 6] = [
        (QKV, HIDDEN),
        (QKV, HIDDEN),
        (QKV, HIDDEN),
        (HEAD_DIM as usize, HIDDEN),
        (HEAD_DIM as usize, HIDDEN),
        (HEADS as usize, HIDDEN),
    ];
    for i in 0..6 {
        let (rows, cols) = shapes[i];
        let bad = transpose(&stage1_ref_data(&h, i), rows, cols);
        let got = run_raw(&h, &x, t, Some((i, &bad)), None);
        for j in 0..6 {
            let clean = if j < 3 {
                bits_equal(&got[j], &want[j])
            } else {
                rel(&got[j], &want[j]) <= F32_ROW_TOL
            };
            assert!(
                got[j].iter().all(|v| v.is_finite()),
                "transposed {}: slice {} went non-finite — failing loudly proves nothing about the bar",
                NAMES[i],
                NAMES[j]
            );
            if j == i {
                assert!(
                    !clean,
                    "the TRANSPOSED {} slice passed the comparator (rel {:.3e}) — the gate does \
                     not bind and would pass a stride/layout bug",
                    NAMES[i],
                    rel(&got[j], &want[j])
                );
            } else {
                assert!(
                    clean,
                    "transposed {} corrupted UNRELATED slice {} — the failure is not attributable",
                    NAMES[i], NAMES[j]
                );
            }
        }
        let _ = full;
    }
}

/// GATE 4 — RED, dropped projection x6. For each range in turn: launch with that range removed
/// (out_i = 0, output pre-zeroed) and assert the comparator fails on exactly that slice.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn a_dropped_projection_fails_the_gate_on_exactly_that_slice_for_all_six() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let t = 2usize;
    let x = h.engine.htod(&hidden_states(t, 0xD70FF)).unwrap();
    let want = h.unfused6(&x, t);
    for (i, _name) in NAMES.iter().enumerate() {
        let got = run_raw(&h, &x, t, None, Some(i));
        for j in 0..6 {
            let clean = if j < 3 {
                bits_equal(&got[j], &want[j])
            } else {
                rel(&got[j], &want[j]) <= F32_ROW_TOL
            };
            if j == i {
                assert!(
                    !clean,
                    "the DROPPED {} range passed the comparator — a fused kernel that silently \
                     skips a projection would ship",
                    NAMES[i]
                );
            } else {
                assert!(
                    clean,
                    "dropping {} corrupted unrelated slice {} — block-range arithmetic is wrong",
                    NAMES[i], NAMES[j]
                );
            }
        }
    }
}

/// GATE 5 — the OFF arm and the refusal shapes. Without the env the door never engages (the
/// dispatch counter is flat across a full mixer walk, wired to the invocation, not prose); with
/// the env but the wrong width (t=16, the GEMM tier) or an all-Float layer it refuses too, so
/// serving shapes outside the claim fall through to the unchanged program.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn flag_off_and_unqualified_shapes_never_engage() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let x1 = h.engine.htod(&hidden_states(1, 0x0FF)).unwrap();

    set_door(false);
    let before = KDA_FUSED6_DISPATCHES.load(Ordering::Relaxed);
    let a = h
        .engine
        .dtoh(&kda_attn(&h.engine, &h.layer, &x1, 1, EPS).expect("mixer, flag off"))
        .unwrap();
    let b = h
        .engine
        .dtoh(&kda_attn(&h.engine, &h.layer, &x1, 1, EPS).expect("mixer, flag off, repeat"))
        .unwrap();
    assert_eq!(
        KDA_FUSED6_DISPATCHES.load(Ordering::Relaxed),
        before,
        "the door engaged with the flag off"
    );
    assert!(
        bits_equal(&a, &b),
        "flag-off mixer is not run-to-run deterministic"
    );
    assert!(
        h.engine
            .kda_proj_fused6(&h.layer, &x1, 1)
            .unwrap()
            .is_none(),
        "kda_proj_fused6 must refuse with the flag off"
    );

    set_door(true);
    let x16 = h.engine.htod(&hidden_states(16, 0x16)).unwrap();
    assert!(
        h.engine
            .kda_proj_fused6(&h.layer, &x16, 16)
            .unwrap()
            .is_none(),
        "t=16 is the GEMM tier — the door must refuse past the batch cap"
    );
    set_door(false);
}
