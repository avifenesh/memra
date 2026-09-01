//! Gate for the BF16 operand arm of `MEMRA_KDA_FUSED_PROJ` — `qmatvec_kda6_bf16f32`
//! (lane/glm5-decode-diet lever 3, 2026-08-31). The serving-recipe twin of
//! `kda_fused_proj_gpu.rs`: on the adopted glm5 arm (MEMRA_BF16_MMV=1) the loader admits
//! wq/wk/wv to raw bf16 residency (`admit=bf16_mmv`), where the q8 arm refuses by design and
//! the unfused stage-1 group is 3x `matvec_bf16_f32acc_x4_rows` + 3x cuBLASLt f32 GEMV.
//!
//! Claims, per class (the q8 gate's structure, rebased on the bf16 residency):
//!  * bf16 rows (wq/wk/wv): BIT-IDENTICAL to the unfused arm at every t=1..15 — the fused
//!    kernel's per-row body is `matvec_bf16_f32acc_x4_rows` VERBATIM at the same blockDim.
//!  * f32 rows (f_a/g_a/b_proj): the same deterministic warp tree the q8 arm gates
//!    (cuBLASLt replacement, a reduction-order numeric class) — banded, measured, printed.
//!  * Whole mixer: fused-vs-unfused within the band; vs `memra_reference` the ON arm may not
//!    exceed the OFF arm's deviation beyond the band (the bf16 OPERAND floor — the f32
//!    reference weights were rounded to bf16 for BOTH arms — belongs to the residency class,
//!    not to this door).
//!
//! RED ARMS: six transposed-slice mutations and six dropped-range mutations through the raw
//! launcher, each caught on exactly its own slice.
//!
//! OWN TEST BINARY, deliberately: `Engine::bf16_mmv_on` latches in a OnceLock at first read,
//! and the q8 gate's fixture must load WITHOUT the bf16 residency door. This binary sets
//! MEMRA_BF16_MMV=1 before the first CUDA call; the q8 binary never does.
//!
//! Rig law: correctness-only, run under `flock /tmp/memra-5090.lock`, TF32 forced off,
//! `-- --ignored --test-threads=1`.

use cudarc::driver::CudaSlice;
use memra_engine::Engine;
use memra_engine::kda::{KDA_FUSED6_BF16_DISPATCHES, KdaAttnLayer, kda_attn};
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
use std::sync::atomic::Ordering;

/// At least 2,000,000 elements per stage-1 matrix: the `bf16_mmv` residency threshold the
/// loader enforces (model.rs) — the whole point of this fixture is that wq/wk/wv load as
/// `GpuTensor::FloatBf16`, which needs `in_f * out_f` of 2M or more. The deterministic
/// fixture caps hidden at 256, so the width comes from the head count instead — 64 heads x
/// 128 IS the real glm5_next geometry: 256 x 8192 = 2,097,152.
const HIDDEN: usize = 256;
const HEADS: u32 = 64;
const HEAD_DIM: u32 = 128;
const CONV_KERNEL: u32 = 4;
const GATE_LOWER_BOUND: f32 = -5.0;
const EPS: f32 = 1e-5;
const QKV: usize = (HEADS * HEAD_DIM) as usize;
const CONV_WIDTH: usize = 3 * QKV;
const STATE_WIDTH: usize = (HEADS * HEAD_DIM * HEAD_DIM) as usize;

const BF16_SUFFIXES: [&str; 3] = ["kda_q.weight", "kda_k.weight", "kda_v.weight"];

/// f32-row band, fused warp tree vs cuBLASLt GEMV — the SAME numeric class the q8 gate
/// measured at worst 4.703e-7 relative (bar 5e-5, ~100x headroom). Printed every run.
const F32_ROW_TOL: f32 = 5e-5;
/// Whole-mixer fused-vs-unfused band, and the cap on how far the ON arm may sit above the
/// OFF arm against the f32 reference (the q8 gate's delta-assert form).
const MIXER_TOL: f32 = 5e-5;

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// TF32 off AND the bf16 residency door on, both before the first CUDA call / first
/// `bf16_mmv_on` read (it latches in a OnceLock — this is why the bf16 arm owns a binary).
fn force_env() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: no CUDA call has been made and no Engine handed out in this process yet,
        // and call_once serializes every test thread behind these writes.
        unsafe {
            if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
                std::env::set_var("NVIDIA_TF32_OVERRIDE", "0");
            }
            std::env::set_var("MEMRA_BF16_MMV", "1");
        }
    });
}

/// The door flag is read PER CALL by design (rollback seam). Serialized behind `gpu_guard`.
fn set_door(on: bool) {
    // SAFETY: all tests in this binary hold `gpu_guard` while touching env or the GPU.
    unsafe {
        if on {
            std::env::set_var("MEMRA_KDA_FUSED_PROJ", "1");
        } else {
            std::env::remove_var("MEMRA_KDA_FUSED_PROJ");
        }
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

/// Round-to-nearest-even f32 -> bf16 bytes. Both arms of every comparison read THESE bytes;
/// the f32 originals stay in the reference weights, which is exactly the bf16 operand floor
/// the mixer delta-assert accounts for.
fn f32_to_bf16_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for v in data {
        let b = v.to_bits();
        let rounded = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
        out.extend_from_slice(&(((rounded >> 16) & 0xFFFF) as u16).to_le_bytes());
    }
    out
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

/// Serving-residency fixture: wq/wk/wv BF16 (>= 2M elements each, so the loader's bf16_mmv
/// arm admits them), everything else F32.
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
        let names = match req.match_mode {
            TensorMatch::OneOf => &req.names[..1],
            TensorMatch::All => req.names.as_slice(),
        };
        for name in names {
            let to_bf16 = BF16_SUFFIXES.iter().any(|s| name == &format!("blk.0.{s}"));
            let (bytes, ty) = if to_bf16 {
                (f32_to_bf16_bytes(&tensor.data), GgmlType::BF16)
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
        force_env();
        let model_plan = one_kda_layer_plan();
        let fixture = deterministic_fixture(&model_plan).expect("deterministic KDA fixture");
        let source = fixture_source(&model_plan, &fixture.weights);
        let engine = Engine::new(0).expect("CUDA engine on device 0");
        let plan = kda_plan();
        let layer = KdaAttnLayer::load(&engine, &source, 0, &plan)
            .expect("KDA mixer loads with the bf16 stage-1 residency");
        Self {
            engine,
            layer,
            weights: fixture.weights,
            plan,
        }
    }

    /// The six unfused stage-1 outputs — on this residency: bf16 matvec rows + cuBLASLt f32.
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

    #[allow(clippy::type_complexity)]
    fn raw_weights(&self) -> ([&CudaSlice<u8>; 3], [&CudaSlice<f32>; 3]) {
        fn bf(w: &GpuTensor) -> &CudaSlice<u8> {
            match w {
                GpuTensor::FloatBf16 { data, .. } => data,
                _ => panic!(
                    "stage-1 bf16 operand is not FloatBf16 — the loader's bf16_mmv admission \
                     did not bind and this gate is measuring the wrong residency"
                ),
            }
        }
        fn f32w(w: &GpuTensor) -> &CudaSlice<f32> {
            match w {
                GpuTensor::Float { data, .. } => data,
                _ => panic!("stage-1 f32 operand is not Float — the fixture did not bind"),
            }
        }
        (
            [bf(&self.layer.wq), bf(&self.layer.wk), bf(&self.layer.wv)],
            [
                f32w(&self.layer.f_a),
                f32w(&self.layer.g_a),
                f32w(&self.layer.b_proj),
            ],
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

/// Run the fused BF16 RAW launcher with one optional mutation (transposed-slice) or dropped
/// range (dims[i] = 0, output pre-zeroed).
fn run_raw(
    h: &Harness,
    x: &CudaSlice<f32>,
    t: usize,
    mutate: Option<(usize, &[f32])>,
    drop_i: Option<usize>,
) -> Vec<Vec<f32>> {
    let e = &h.engine;
    let ([bq, bk, bv], [wfa, wga, wb]) = h.raw_weights();
    let mut d = dims();
    if let Some(i) = drop_i {
        d[i] = 0;
    }
    let mut_bf: Option<CudaSlice<u8>> = match mutate {
        Some((i, data)) if i < 3 => Some(e.htod_bytes(&f32_to_bf16_bytes(data)).unwrap()),
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
    let full = dims();
    let mut outs = [
        e.zeros(t * full[0]).unwrap(),
        e.zeros(t * full[1]).unwrap(),
        e.zeros(t * full[2]).unwrap(),
        e.zeros(t * full[3]).unwrap(),
        e.zeros(t * full[4]).unwrap(),
        e.zeros(t * full[5]).unwrap(),
    ];
    e.kda_proj_fused6_bf16_raw(
        pick(0, mut_at, mut_bf.as_ref(), bq),
        pick(1, mut_at, mut_bf.as_ref(), bk),
        pick(2, mut_at, mut_bf.as_ref(), bv),
        pick(3, mut_at, mut_f32.as_ref(), wfa),
        pick(4, mut_at, mut_f32.as_ref(), wga),
        pick(5, mut_at, mut_f32.as_ref(), wb),
        x,
        &mut outs,
        HIDDEN,
        d,
        t,
    )
    .expect("the raw fused bf16 launch runs cleanly — mutations must be silent, not loud");
    outs.iter().map(|o| e.dtoh(o).unwrap()).collect()
}

const NAMES: [&str; 6] = ["wq", "wk", "wv", "f_a", "g_a", "b_proj"];

/// GATE 1 — the fixture BINDS the serving residency (wq/wk/wv FloatBf16, f32 trio Float),
/// the door engages, and per width t=1..15: bf16 rows BYTEWISE identical to the unfused arm,
/// f32 rows within the calibrated band (measured value printed).
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn bf16_door_is_bit_identical_on_bf16_rows_and_banded_on_f32_rows_at_t_1_to_15() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    for (name, w, want_bf16) in [
        ("wq", &h.layer.wq, true),
        ("wk", &h.layer.wk, true),
        ("wv", &h.layer.wv, true),
        ("f_a", &h.layer.f_a, false),
        ("g_a", &h.layer.g_a, false),
        ("b_proj", &h.layer.b_proj, false),
    ] {
        match (want_bf16, w) {
            (true, GpuTensor::FloatBf16 { .. }) | (false, GpuTensor::Float { .. }) => {}
            _ => panic!(
                "fixture residency did not bind for {name} — the loader's bf16_mmv admission \
                 (>= 2M elements, BF16 source) must produce FloatBf16 for the stage-1 trio"
            ),
        }
    }

    let before = KDA_FUSED6_BF16_DISPATCHES.load(Ordering::Relaxed);
    let mut worst_f32 = 0.0f32;
    for t in 1..=15usize {
        let x = h.engine.htod(&hidden_states(t, 0xB16 ^ t as u64)).unwrap();
        set_door(false);
        let unfused = h.unfused6(&x, t);
        set_door(true);
        let fused = h
            .engine
            .kda_proj_fused6(&h.layer, &x, t)
            .expect("door call")
            .expect("the door must engage on the bf16 serving residency");
        set_door(false);
        let fused: Vec<Vec<f32>> = fused.iter().map(|o| h.engine.dtoh(o).unwrap()).collect();
        for i in 0..3 {
            assert!(
                bits_equal(&fused[i], &unfused[i]),
                "t={t} {}: bf16 rows are not bit-identical to matvec_bf16_f32acc_x4_rows",
                NAMES[i]
            );
        }
        for i in 3..6 {
            let r = rel(&fused[i], &unfused[i]);
            worst_f32 = worst_f32.max(r);
            assert!(
                r <= F32_ROW_TOL,
                "t={t} {}: f32 row relative maxdiff {r:.3e} exceeds {F32_ROW_TOL:.1e}",
                NAMES[i]
            );
        }
    }
    let after = KDA_FUSED6_BF16_DISPATCHES.load(Ordering::Relaxed);
    assert!(
        after >= before + 15,
        "the bf16 arm never engaged (counter {before} -> {after})"
    );
    println!(
        "[kda-fused6 bf16 receipt] bf16 rows bitwise at t=1..15; f32-row worst relative \
         {worst_f32:.3e} (bar {F32_ROW_TOL:.1e}); dispatches {before} -> {after}"
    );
}

/// GATE 2 — whole mixer: fused-vs-unfused within the band, and against `memra_reference` the
/// ON arm may not exceed the OFF arm beyond the band (the bf16 operand floor is in BOTH arms).
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn whole_mixer_matches_across_the_door_and_does_not_worsen_vs_reference() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    for &t in &[1usize, 7, 15] {
        let xs = hidden_states(t, 0x00B1_6A57 ^ t as u64);
        let x = h.engine.htod(&xs).unwrap();
        set_door(false);
        let off = h
            .engine
            .dtoh(&kda_attn(&h.engine, &h.layer, &x, t, EPS).expect("unfused mixer"))
            .unwrap();
        set_door(true);
        let on = h
            .engine
            .dtoh(&kda_attn(&h.engine, &h.layer, &x, t, EPS).expect("fused mixer"))
            .unwrap();
        set_door(false);
        let r = rel(&on, &off);
        assert!(
            r <= MIXER_TOL,
            "t={t}: whole-mixer fused-vs-unfused relative maxdiff {r:.3e} exceeds {MIXER_TOL:.1e}"
        );

        let (reference, _) =
            kimi_delta_net_layer(0, &h.plan, EPS, &h.weights, &xs, t, HIDDEN).expect("reference");
        let r_off = rel(&off, &reference);
        let r_on = rel(&on, &reference);
        assert!(
            r_on <= r_off + MIXER_TOL,
            "t={t}: the ON arm sits {r_on:.3e} from the reference vs OFF {r_off:.3e} — the \
             door worsened the mixer beyond its own band"
        );
        println!(
            "[kda-fused6 bf16 receipt] mixer t={t}: fused-vs-unfused {r:.3e}; vs reference \
             OFF {r_off:.3e} / ON {r_on:.3e} (bf16 operand floor, both arms)"
        );
    }
}

/// RED ARMS — transposed-slice x6: each weight replaced by its transposed-data twin (loads,
/// runs, finite, silently wrong); the comparator must fail on EXACTLY that slice and pass on
/// the other five.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn transposed_slice_mutations_are_caught_on_exactly_their_own_slice() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let t = 3usize;
    let x = h.engine.htod(&hidden_states(t, 0x7A05)).unwrap();
    let clean = run_raw(&h, &x, t, None, None);
    let d = dims();
    for i in 0..6 {
        let reference = stage1_ref_data(&h, i);
        let mutated = transpose(&reference, d[i], HIDDEN);
        let got = run_raw(&h, &x, t, Some((i, &mutated)), None);
        for (j, name) in NAMES.iter().enumerate() {
            let differs = !bits_equal(&got[j], &clean[j]);
            assert!(
                got[j].iter().all(|v| v.is_finite()),
                "mutation {i}: output {name} went non-finite — reds must be silent"
            );
            if j == i {
                assert!(
                    differs,
                    "transposing {} did not move its own output — the gate does not bind to \
                     this slice",
                    NAMES[i]
                );
            } else {
                assert!(
                    !differs,
                    "transposing {} moved output {name} — mutation is not isolated",
                    NAMES[i]
                );
            }
        }
    }
}

/// RED ARMS — dropped-range x6: each range removed from the launch (dims[i]=0, output
/// zero-filled); the comparator must fail on exactly that slice.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn dropped_range_mutations_are_caught_on_exactly_their_own_slice() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let t = 2usize;
    let x = h.engine.htod(&hidden_states(t, 0xD20B)).unwrap();
    let clean = run_raw(&h, &x, t, None, None);
    for (i, dropped) in NAMES.iter().enumerate() {
        let got = run_raw(&h, &x, t, None, Some(i));
        for (j, name) in NAMES.iter().enumerate() {
            let differs = !bits_equal(&got[j], &clean[j]);
            if j == i {
                assert!(
                    differs,
                    "dropping range {dropped} left its output unchanged — the drop did not \
                     reach the launch"
                );
            } else {
                assert!(
                    !differs,
                    "dropping range {dropped} moved output {name} — ranges are not independent"
                );
            }
        }
    }
}

/// FLAG-OFF — the door never engages without the env; the mixer walk leaves the counter flat.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn the_door_never_engages_with_the_flag_off() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    set_door(false);
    let before = KDA_FUSED6_BF16_DISPATCHES.load(Ordering::Relaxed);
    let x = h.engine.htod(&hidden_states(1, 0x0FF)).unwrap();
    let door = h
        .engine
        .kda_proj_fused6(&h.layer, &x, 1)
        .expect("door call");
    assert!(
        door.is_none(),
        "the door engaged without MEMRA_KDA_FUSED_PROJ"
    );
    let _ = kda_attn(&h.engine, &h.layer, &x, 1, EPS).expect("mixer walk");
    assert_eq!(
        before,
        KDA_FUSED6_BF16_DISPATCHES.load(Ordering::Relaxed),
        "flag-off walk advanced the bf16 dispatch counter"
    );
    // t=16 is the GEMM tier — outside the door's claim; it must refuse even with the flag on.
    set_door(true);
    let x16 = h.engine.htod(&hidden_states(16, 0x1616)).unwrap();
    let door = h
        .engine
        .kda_proj_fused6(&h.layer, &x16, 16)
        .expect("door call");
    set_door(false);
    assert!(door.is_none(), "the door engaged at t=16 (the GEMM tier)");
}
