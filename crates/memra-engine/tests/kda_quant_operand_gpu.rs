//! QUANTIZED-OPERAND gate for the glm5_next KDA mixer, plus the loader rank guard that keeps a
//! quantized 3-D MLA operand from being read with a silently wrong stride.
//!
//! WHY THIS EXISTS (glm53-flash lane, 2026-08-28). `kda_fixture_gpu` proves the KDA mixer against
//! `memra_reference` with every weight F32-resident. That is NOT the arithmetic class our own
//! artifact runs. The NVFP4 mint keeps every KDA projection out of quantization (`ignore` list,
//! mint-receipts/hf_quant_config.json), but the loader's BF16 law re-encodes any kept BF16 2-D
//! weight of >= 1M elements to Q8_0 (`source.rs`, "LOADER LAW, 2026-07-08"), and the engine reads
//! only `modules_to_not_convert` — a key the modelopt mint does not write. So on the real
//! checkpoint `kda_q/k/v/out` and `kda_f_b/g_b` arrive `GpuTensor::Quant`, and `kda_f_a/g_a/b`
//! (below 1M) arrive Float: a MIXED-residency layer that no gate covered.
//!
//! The bar is the SAME path's Float result, not the reference: this gate isolates the one change
//! under test (operand residency) from the kernel-vs-reference question `kda_fixture_gpu` already
//! owns. Anything else it caught would be that gate's failure, not this one's.

use memra_engine::Engine;
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
    CheckpointDialect, ContractOptions, OutputHead, TensorContract, TensorId, TensorMatch,
};
use memra_reference::{ReferenceTensor, deterministic_fixture};
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

/// The ggml suffixes that are 2-D matmul weights on a KDA layer — the class the loader's BF16
/// law re-encodes on the real checkpoint. Everything else the mixer loads (the fused conv1d
/// planes, `kda_a_log`, `kda_dt.bias`, `kda_o_norm.weight`) is read through `float_data()` and
/// MUST stay F32; the conv1d weights are 3-D in HF (`[qkv, 1, kernel]`) and the rest are 1-D, so
/// the >= 1M / 2-D gates in `source.rs` skip them on the real artifact for the same reason they
/// are skipped here.
const QUANTIZED_SUFFIXES: [&str; 9] = [
    "kda_q.weight",
    "kda_k.weight",
    "kda_v.weight",
    "kda_f_a.weight",
    "kda_f_b.weight",
    "kda_g_a.weight",
    "kda_g_b.weight",
    "kda_b.weight",
    "kda_out.weight",
];

/// QUANTIZATION-FLOOR TOLERANCE. MEASURED on this fixture (5090, TF32 off, NVIDIA_TF32_OVERRIDE=0,
/// 2026-08-28), not guessed and not inherited from another gate:
///
/// | arm                              | relative maxdiff vs the Float twin |
/// |----------------------------------|------------------------------------|
/// | Q8_0, T=1                        | 2.331e-2  (worst)                  |
/// | Q8_0, T=7 / 16 / 65 / 130        | 1.602e-2 / 1.247e-2 / 1.664e-2 / 9.428e-3 |
/// | MUTATION: stride too long x HEADS| 1.310e-1                           |
/// | MUTATION: rows rotated by one    | 2.500e0                            |
///
/// Q8_0 stores a per-32 fp16 scale plus int8 codes, so each weight carries up to ~1/254 of its
/// block's peak as error; the KDA chain then runs that through two conv+SiLU stages, an L2 norm,
/// a sigmoid gate pair, the delta-rule recurrence and a gated RMSNorm before the output
/// projection, which is why ~2e-2 and not ~4e-3 is the floor here.
///
/// 5e-2 sits 2.1x above the measured worst legitimate value and 2.6x below the tightest mutation.
/// That band is only ~5.6x wide, and saying so is part of the receipt: on a 256-hidden 2-head
/// fixture with random weights the Q8_0 floor is not far below a real stride bug. A wider fixture
/// separates them further; a NARROWER bar than this would start failing on reduction-order luck.
///
/// This is a FLOOR, not an accuracy claim. It says the Q8_0 operand class is arithmetically sane
/// on this path — not that Q8_0 is accurate enough to serve glm5_next. That question needs a
/// checkpoint-parity cell on the real artifact and is named in the lane's remaining-gaps list.
const QUANT_TOL: f32 = 5e-2;

/// GPU tests serialize on one device.
fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// cuBLASLt f32 compute rides TF32 on Blackwell by default — right for serving, wrong for a
/// parity gate. The driver reads this at CUDA init, so it must be set before the first
/// `Engine::new` in the process; `call_once` serializes every test thread behind the write.
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

fn kda_plan() -> KimiDeltaNetPlan {
    KimiDeltaNetPlan {
        num_heads: HEADS,
        head_dim: HEAD_DIM,
        conv_kernel: CONV_KERNEL,
        gate_lower_bound: GATE_LOWER_BOUND,
    }
}

/// One KDA layer with an inert dense MLP and a serial residual — the smallest plan that makes
/// `deterministic_fixture` emit this layer's KDA tensors and `TensorContract::for_plan` name them.
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

/// How a projection's bytes are produced. `Float` is the control arm and `Q8_0` the arm under
/// test; the other two are the mutations gate 3 uses, both of which LOAD cleanly, stay in bounds
/// and produce finite numbers:
///   * `Q8_0StrideTooLong` — buffer padded to `HEADS` times its true size, so the loader derives a
///     row stride `HEADS` times too large. The 3-D `ne[1]` mis-derivation, on a 2-D operand.
///   * `Q8_0RowRotated` — identical length, codes and alignment; rows rotated by one. Pure
///     assignment error, used to measure the bar's resolution.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Float,
    Q8_0,
    Q8_0StrideTooLong,
    Q8_0RowRotated,
}

fn fixture_source(
    plan: &ModelPlan,
    weights: &BTreeMap<TensorId, ReferenceTensor>,
    arm: Arm,
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
        let names = match req.match_mode {
            TensorMatch::OneOf => &req.names[..1],
            TensorMatch::All => req.names.as_slice(),
        };
        for name in names {
            let quantize = arm != Arm::Float
                && QUANTIZED_SUFFIXES
                    .iter()
                    .any(|s| name == &format!("blk.0.{s}"));
            let (bytes, ty) = if quantize {
                let in_f = req.shape[0] as usize;
                let out_f = req.shape[1] as usize;
                assert_eq!(
                    in_f % 32,
                    0,
                    "{name}: Q8_0 needs in_features % 32 == 0, got {in_f}"
                );
                let mut bytes = memra_gguf::nvfp4_repack::f32_to_q8_0(&tensor.data);
                if arm == Arm::Q8_0StrideTooLong {
                    // MUTATION A, modelled on the 3-D bug exactly: make the loader derive a row
                    // stride that is TOO LONG by the head count. `row_bytes = bytes.len() / out_f`
                    // is whatever the buffer length says, so padding the buffer to `HEADS` times
                    // its true size makes every output row start `HEADS` rows further in — which
                    // is precisely what `out_f = ne[1]` would have done to `attn_k_b` (gate 4
                    // measures 272 vs a true 68 at 4 heads).
                    //
                    // The padding is a repeat of the tensor's OWN encoded blocks, so every byte
                    // the kernel reads is a valid Q8_0 block of a real weight and every read stays
                    // in bounds: the output is finite and plausible, wrong only in which weights
                    // each row used. (Shortening the buffer instead was measured first — it reads
                    // past the allocation and yields non-finite values, which is a different and
                    // less interesting failure.)
                    let row_bytes = in_f / 32 * 34;
                    assert_eq!(bytes.len(), out_f * row_bytes);
                    let target = out_f * row_bytes * HEADS as usize;
                    let src = bytes.clone();
                    while bytes.len() < target {
                        let take = (target - bytes.len()).min(src.len());
                        bytes.extend_from_slice(&src[..take]);
                    }
                }
                if arm == Arm::Q8_0RowRotated {
                    // SUBTLER MUTATION: same length, same codes, same alignment — the rows are
                    // rotated by one. Nothing about the buffer is malformed; only the row->output
                    // assignment is wrong. This is the resolution test for the bar: if a pure
                    // permutation of correct weights lands inside the quantization floor, the
                    // floor is too wide to mean anything.
                    let row_bytes = in_f / 32 * 34;
                    bytes.rotate_left(row_bytes);
                }
                (bytes, GgmlType::Q8_0)
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

/// Relative maxdiff of `got` against the Float twin `want`.
fn rel(got: &[f32], want: &[f32]) -> f32 {
    maxdiff(got, want) / scale_of(want)
}

struct Harness {
    engine: Engine,
    float: KdaAttnLayer,
    quant: KdaAttnLayer,
}

impl Harness {
    fn new() -> Self {
        force_true_f32();
        let model_plan = one_kda_layer_plan();
        let fixture = deterministic_fixture(&model_plan).expect("deterministic KDA fixture");
        let engine = Engine::new(0).expect("CUDA engine on device 0");
        let plan = kda_plan();
        let float = KdaAttnLayer::load(
            &engine,
            &fixture_source(&model_plan, &fixture.weights, Arm::Float),
            0,
            &plan,
        )
        .expect("KDA mixer loads with F32 operands");
        let quant = KdaAttnLayer::load(
            &engine,
            &fixture_source(&model_plan, &fixture.weights, Arm::Q8_0),
            0,
            &plan,
        )
        .expect("KDA mixer loads with Q8_0 operands");
        Self {
            engine,
            float,
            quant,
        }
    }

    fn run(&self, layer: &KdaAttnLayer, x: &[f32], tokens: usize) -> Vec<f32> {
        let x_d = self.engine.htod(x).unwrap();
        let out = kda_attn(&self.engine, layer, &x_d, tokens, EPS).expect("GPU KDA prefill");
        self.engine.dtoh(&out).unwrap()
    }
}

/// GATE 1 — the gate BINDS. A comparison of two layers is worth nothing if both arms are the same
/// residency: the Q8_0 arm would agree to the last bit and the gate would be decorative. Assert
/// the arithmetic class is actually different before comparing anything, and assert the
/// float-side control is Float, so a loader change in either direction fails here by name.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn kda_quant_arm_is_actually_quantized() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let pairs: [(&str, &GpuTensor, &GpuTensor); 9] = [
        ("kda_q", &h.float.wq, &h.quant.wq),
        ("kda_k", &h.float.wk, &h.quant.wk),
        ("kda_v", &h.float.wv, &h.quant.wv),
        ("kda_f_a", &h.float.f_a, &h.quant.f_a),
        ("kda_f_b", &h.float.f_b, &h.quant.f_b),
        ("kda_g_a", &h.float.g_a, &h.quant.g_a),
        ("kda_g_b", &h.float.g_b, &h.quant.g_b),
        ("kda_b", &h.float.b_proj, &h.quant.b_proj),
        ("kda_out", &h.float.wo, &h.quant.wo),
    ];
    for (name, f, q) in pairs {
        assert!(
            matches!(f, GpuTensor::Float { .. }),
            "{name}: the control arm must be Float, got a quantized tensor"
        );
        assert!(
            matches!(q, GpuTensor::Quant { .. }),
            "{name}: the arm under test loaded Float — the Q8_0 fixture bytes did not reach the \
             quantized loader path, so this whole gate would compare a layer with itself"
        );
    }
}

/// GATE 2 — the Q8_0 operand class produces the same answer as the Float twin, within the
/// quantization floor, on BOTH matmul dispatch classes. `matmul_group` and `matmul` route by row
/// count: T=1 and T=7 take the MMVQ matvec path, T=16 and above take the GEMM path. A quantized
/// arm that is wired correctly for one and not the other passes a single-length gate.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn kda_q8_0_operands_match_the_float_twin_within_the_quantization_floor() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for &tokens in &[1usize, 7, 16, 65, 130] {
        #[allow(clippy::unusual_byte_groupings)]
        // allow: mnemonic grouping of a pinned seed/magic constant
        let x = hidden_states(tokens, 0x9_11A_57 ^ tokens as u64);
        let want = h.run(&h.float, &x, tokens);
        let got = h.run(&h.quant, &x, tokens);
        assert!(
            got.iter().all(|v| v.is_finite()),
            "T={tokens}: the quantized arm produced non-finite values"
        );
        let r = rel(&got, &want);
        eprintln!("[kda-quant] T={tokens} rel {r:.3e}");
        if r > worst {
            worst = r;
            worst_at = tokens;
        }
        assert!(
            r <= QUANT_TOL,
            "T={tokens}: Q8_0 vs Float relative maxdiff {r:.3e} exceeds the quantization floor \
             {QUANT_TOL:.1e}"
        );
    }
    eprintln!(
        "[kda-quant] worst relative maxdiff {worst:.3e} at T={worst_at} (tol {QUANT_TOL:.1e})"
    );
}

/// GATE 3 — MUTATION CHECK. Gate 2 only means something if a WRONG quantized operand fails it.
/// Two mutations, both of which LOAD without complaint and produce finite, plausible-looking
/// numbers — the failure class a shape check or a liveness check cannot see:
///
///  * `StrideTooLong` — the buffer is padded to `HEADS` times its true size, so the loader's
///    `row_bytes = bytes.len() / out_f` comes out `HEADS` times too large and every output row
///    starts that many rows further in. This is the 3-D mis-derivation transplanted onto a 2-D
///    operand the KDA path actually loads: gate 4 measures the same factor on the real MLA shape
///    (272 derived vs 68 true, at 4 heads). Every byte read is a real Q8_0 block of this tensor,
///    so nothing overruns and nothing is non-finite.
///  * `RowRotated` — same length, same codes, same alignment, rows rotated by one. Nothing is
///    malformed; only the row->output assignment is wrong. This is the RESOLUTION test: a bar
///    that a pure permutation of correct weights slips under is not measuring anything.
///
/// Both must land outside `QUANT_TOL`, and the margins are printed so the bar's strength is a
/// measurement rather than an assertion.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn a_mis_strided_quantized_operand_fails_the_gate() {
    let _gpu = gpu_guard();
    force_true_f32();
    let model_plan = one_kda_layer_plan();
    let fixture = deterministic_fixture(&model_plan).expect("deterministic KDA fixture");
    let engine = Engine::new(0).expect("CUDA engine on device 0");
    let plan = kda_plan();
    let load = |arm: Arm| {
        KdaAttnLayer::load(
            &engine,
            &fixture_source(&model_plan, &fixture.weights, arm),
            0,
            &plan,
        )
        .expect("the mutated fixture still LOADS — that is the point of the mutation")
    };
    let float = load(Arm::Float);

    let tokens = 16usize;
    let x = hidden_states(tokens, 0xBAD_5721DE);
    let x_d = engine.htod(&x).unwrap();
    let want = engine
        .dtoh(&kda_attn(&engine, &float, &x_d, tokens, EPS).expect("float arm"))
        .unwrap();

    for (label, arm) in [
        ("stride-too-long", Arm::Q8_0StrideTooLong),
        ("row-rotated", Arm::Q8_0RowRotated),
    ] {
        let bad = load(arm);
        let got = engine
            .dtoh(&kda_attn(&engine, &bad, &x_d, tokens, EPS).expect("mutated arm"))
            .unwrap();
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{label}: the mutation produced non-finite values — it is failing loudly for the \
             wrong reason, which proves nothing about the bar"
        );
        let r = rel(&got, &want);
        eprintln!(
            "[kda-quant mutation] {label} relative maxdiff {r:.3e} vs tol {QUANT_TOL:.1e} \
             ({:.3e}x the bar)",
            r / QUANT_TOL
        );
        assert!(
            r > QUANT_TOL,
            "the {label} Q8_0 operand produced {r:.3e} relative error, INSIDE the \
             {QUANT_TOL:.1e} bar — gate 2 does not bind and would pass a corrupted weight"
        );
    }
}

/// GATE 4 — CENSUS-LEVEL GUARD, the one that keeps the MLA gap from reopening silently.
///
/// Every quantized resident layout in this engine is a MATRIX: `GpuTensor::load_from_source`
/// derives `row_bytes = bytes.len() / ne[1]`, and on a 3-D tensor `ne[1]` is the MIDDLE axis. The
/// MLA conversion-split operands are exactly that shape — `attn_k_b` ne [nope, kv_rank, head],
/// `attn_v_b` ne [kv_rank, v, head] — so a checkpoint that ships `kv_b_proj` quantized (ours
/// keeps it BF16; the vendor FP8 artifact does too; a third-party NVFP4 mint need not) would have
/// been read with a stride off by the HEAD COUNT.
///
/// The numbers this test pins, for a [64 nope, 32 rank, 4 heads] Q8_0 operand:
///   true row (64 in-features)          = 64/32 * 34 = 68 bytes
///   rows                               = 32 * 4     = 128
///   total                              = 128 * 68   = 8704 bytes
///   what `ne[1]` would have derived    = 8704 / 32  = 272 bytes = 4x the true row = the head count
/// Four heads, four times the stride — the error scales with `ne[2]`, so it is invisible at
/// heads == 1 and catastrophic on the real 64-head artifact. The loader must REFUSE by name.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn a_quantized_3d_operand_is_refused_by_name_not_mis_strided() {
    let _gpu = gpu_guard();
    force_true_f32();
    let engine = Engine::new(0).expect("CUDA engine on device 0");

    let (nope, rank, heads) = (64usize, 32usize, 4usize);
    let elements = nope * rank * heads;
    let data = vec![0.25f32; elements];
    let bytes = memra_gguf::nvfp4_repack::f32_to_q8_0(&data);

    // The arithmetic the guard exists to prevent, computed here so the test fails if the layout
    // constants ever move underneath the prose above.
    let true_row_bytes = nope / 32 * 34;
    let rows = rank * heads;
    assert_eq!(true_row_bytes, 68);
    assert_eq!(bytes.len(), rows * true_row_bytes);
    assert_eq!(bytes.len(), 8704);
    let mis_derived = bytes.len() / rank; // what `out_f = ne[1]` would have produced
    assert_eq!(mis_derived, 272);
    assert_eq!(
        mis_derived,
        true_row_bytes * heads,
        "the mis-derivation is exactly a factor of the head count"
    );

    struct One {
        name: String,
        bytes: Vec<u8>,
        ne: Vec<u64>,
    }
    impl TensorSource for One {
        fn config(&self) -> ModelConfig {
            unreachable!("rank-guard fixture is tensor-only")
        }
        fn find(&self, name: &str) -> Option<TensorView<'_>> {
            (name == self.name).then(|| TensorView {
                bytes: Cow::Borrowed(&self.bytes),
                ggml_type: GgmlType::Q8_0,
                ne: self.ne.clone(),
            })
        }
    }
    let name = "blk.0.attn_k_b.weight";
    let src = One {
        name: name.into(),
        bytes,
        ne: vec![nope as u64, rank as u64, heads as u64],
    };

    let err = GpuTensor::load_from_source(&engine, &src, name)
        .err()
        .expect("a quantized 3-D tensor must be refused, not loaded with a derived stride");
    let msg = err.to_string();
    assert!(
        msg.contains(name),
        "the refusal must NAME the tensor; got: {msg}"
    );
    assert!(
        msg.contains("3-D") && msg.contains("2-D"),
        "the refusal must say what the constraint is; got: {msg}"
    );

    // And the 1-D case, which used to index out of bounds with no name attached at all.
    let src_1d = One {
        name: name.into(),
        bytes: memra_gguf::nvfp4_repack::f32_to_q8_0(&vec![0.25f32; 64]),
        ne: vec![64],
    };
    let err = GpuTensor::load_from_source(&engine, &src_1d, name)
        .err()
        .expect("a quantized 1-D tensor must be refused, not panic on ne[1]");
    assert!(err.to_string().contains(name));
}
