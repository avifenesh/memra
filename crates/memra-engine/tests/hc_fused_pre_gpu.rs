//! Gate for `MEMRA_HC_FUSED_PRE` — the fused mHC site pre-chain launch
//! (`memra_dsv4_hc_pre_fused`, lane/glm5-decode-diet 2026-08-31).
//!
//! THE ONE CHANGE UNDER TEST: the three per-site kernels of `hyper::pre_finish`
//! (`memra_dsv4_rowsq_scale` + `memra_dsv4_hc_sinkhorn_m` + `memra_dsv4_hc_collapse`) run as
//! ONE launch. The claim is BIT identity, not a band: every body in the fused kernel is the
//! unfused kernel's verbatim (rowsq at the same pinned blockDim=128; Sinkhorn on shared
//! operands; collapse per-element expression), so all four outputs — pre, post, comb, y —
//! must compare `to_bits`-equal at every shape, and the whole-model decode walk must be
//! byte-identical ON vs OFF. The kernel's Sinkhorn stationarity exit fires only on bitwise
//! fixed points, so it is covered by the same equality assert (the fused arm may run FEWER
//! iterations and must still produce the SAME bits as the unfused 20).
//!
//! RED ARMS (the gate binds, per output): the three gate scales route to disjoint outputs
//! (scale[0] -> pre -> y, scale[1] -> post, scale[2] -> comb), so flipping one must fail the
//! comparator on exactly its own slice and pass on the others — isolation, so a red result is
//! attributable, not collateral.
//!
//! Rig law: correctness-only, run under `flock /tmp/memra-5090.lock`, TF32 forced off,
//! `-- --ignored --test-threads=1`.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::hyper::HC_FUSED_PRE_DISPATCHES;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::ModelPlan;
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, OutputHead, TensorContract, TensorId, TensorMatch,
};
use memra_reference::{ReferenceTensor, deterministic_fixture};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::os::raw::c_void;
use std::sync::atomic::Ordering;

const ITERS: i32 = 20; // glm5_next's hc_sinkhorn_iters
const EPS: f32 = 1e-6; // glm5_next's hc_eps

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
        if on {
            std::env::set_var("MEMRA_HC_FUSED_PRE", "1");
        } else {
            std::env::remove_var("MEMRA_HC_FUSED_PRE");
        }
    }
}

/// Deterministic pseudo-random f32 in [-1, 1) — fixture values, not statistics.
fn randf(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as u32 as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect()
}

fn dp(s: &CudaSlice<f32>, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> *const f32 {
    s.device_ptr(stream).0 as *const f32
}
fn dpm(s: &mut CudaSlice<f32>, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> *mut f32 {
    s.device_ptr_mut(stream).0 as *mut f32
}

struct PreChainOut {
    pre: Vec<f32>,
    post: Vec<f32>,
    comb: Vec<f32>,
    y: Vec<f32>,
}

struct Operands {
    x: Vec<f32>,
    mixes: Vec<f32>,
    scale: Vec<f32>,
    base: Vec<f32>,
}

fn operands(t: usize, hc: usize, d: usize, seed: u64) -> Operands {
    let rows = (2 + hc) * hc;
    Operands {
        x: randf(t * hc * d, seed ^ 0x0011),
        mixes: randf(t * rows, seed ^ 0x0022),
        scale: randf(3, seed ^ 0x0033),
        base: randf(rows, seed ^ 0x0044),
    }
}

/// The unfused chain, exactly as `hyper::pre_finish` runs it: rowsq_scale (mutates a copy of
/// mixes in place), sinkhorn_m, collapse.
fn run_unfused(e: &Engine, ops: &Operands, t: usize, hc: usize, d: usize) -> PreChainOut {
    let rows = (2 + hc) * hc;
    let w = hc * d;
    let stream = e.stream();
    let x = e.htod(&ops.x).unwrap();
    let mut mixes = e.htod(&ops.mixes).unwrap();
    let scale = e.htod(&ops.scale).unwrap();
    let base = e.htod(&ops.base).unwrap();
    let mut pre = e.uninit(t * hc).unwrap();
    let mut post = e.uninit(t * hc).unwrap();
    let mut comb = e.uninit(t * hc * hc).unwrap();
    let mut y = e.uninit(t * d).unwrap();
    unsafe {
        assert_eq!(
            memra_engine::dsv4_ffi::memra_dsv4_rowsq_scale(
                dp(&x, &stream),
                dpm(&mut mixes, &stream),
                t as i32,
                w as i32,
                rows as i32,
                EPS,
                stream.cu_stream() as *mut c_void,
            ),
            0,
            "unfused rowsq_scale"
        );
        assert_eq!(
            memra_engine::dsv4_ffi::memra_dsv4_hc_sinkhorn_m(
                dp(&mixes, &stream),
                dp(&scale, &stream),
                dp(&base, &stream),
                dpm(&mut pre, &stream),
                dpm(&mut post, &stream),
                dpm(&mut comb, &stream),
                t as i32,
                hc as i32,
                ITERS,
                EPS,
                stream.cu_stream() as *mut c_void,
            ),
            0,
            "unfused sinkhorn_m"
        );
        assert_eq!(
            memra_engine::dsv4_ffi::memra_dsv4_hc_collapse(
                dp(&x, &stream),
                dp(&pre, &stream),
                dpm(&mut y, &stream),
                t as i32,
                hc as i32,
                d as i32,
                stream.cu_stream() as *mut c_void,
            ),
            0,
            "unfused collapse"
        );
    }
    PreChainOut {
        pre: e.dtoh(&pre).unwrap(),
        post: e.dtoh(&post).unwrap(),
        comb: e.dtoh(&comb).unwrap(),
        y: e.dtoh(&y).unwrap(),
    }
}

/// The fused launch. `niters_out`, when requested, receives the per-token executed Sinkhorn
/// iteration counts (the stationarity-exit receipt).
fn run_fused(
    e: &Engine,
    ops: &Operands,
    t: usize,
    hc: usize,
    d: usize,
    niters_out: Option<&mut Vec<i32>>,
) -> PreChainOut {
    let stream = e.stream();
    let x = e.htod(&ops.x).unwrap();
    let mixes = e.htod(&ops.mixes).unwrap();
    let scale = e.htod(&ops.scale).unwrap();
    let base = e.htod(&ops.base).unwrap();
    let mut pre = e.uninit(t * hc).unwrap();
    let mut post = e.uninit(t * hc).unwrap();
    let mut comb = e.uninit(t * hc * hc).unwrap();
    let mut y = e.uninit(t * d).unwrap();
    let mut niters_d = e.uninit_i32(t).unwrap();
    let want_niters = niters_out.is_some();
    unsafe {
        use cudarc::driver::DevicePtrMut as _;
        let np = if want_niters {
            niters_d.device_ptr_mut(&stream).0 as *mut i32
        } else {
            std::ptr::null_mut()
        };
        assert_eq!(
            memra_engine::dsv4_ffi::memra_dsv4_hc_pre_fused(
                dp(&x, &stream),
                dp(&mixes, &stream),
                dp(&scale, &stream),
                dp(&base, &stream),
                dpm(&mut pre, &stream),
                dpm(&mut post, &stream),
                dpm(&mut comb, &stream),
                dpm(&mut y, &stream),
                t as i32,
                hc as i32,
                d as i32,
                ITERS,
                EPS,
                np,
                stream.cu_stream() as *mut c_void,
            ),
            0,
            "fused pre-chain"
        );
    }
    if let Some(out) = niters_out {
        *out = e.dtoh_i32(&niters_d).unwrap();
    }
    PreChainOut {
        pre: e.dtoh(&pre).unwrap(),
        post: e.dtoh(&post).unwrap(),
        comb: e.dtoh(&comb).unwrap(),
        y: e.dtoh(&y).unwrap(),
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn assert_bits_eq(name: &str, got: &[f32], want: &[f32], shape: &str) {
    assert_eq!(got.len(), want.len(), "{name} {shape}: length");
    let diffs = got
        .iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        diffs,
        0,
        "{name} {shape}: {diffs}/{} elements differ in bits (the fused body is not the \
         unfused body verbatim)",
        got.len()
    );
}

fn assert_bits_ne(name: &str, got: &[f32], want: &[f32], shape: &str) {
    assert!(
        bits(got) != bits(want),
        "{name} {shape}: outputs are bit-identical but the mutation should have moved them — \
         the comparator is not reading this output"
    );
}

/// GATE 1 — per-site bit identity across the shape family: decode (t=1), batched decode
/// widths, a verify-walk width, a prefill width; glm5_next's hc=4 at both a mini and the
/// real hidden width, plus hc=2 (shape generality inside the static-cap guard).
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn fused_prechain_is_bit_identical_to_the_unfused_chain() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    for &(hc, d) in &[(4usize, 128usize), (4, 4096), (2, 640)] {
        for &t in &[1usize, 2, 7, 15, 64] {
            let ops = operands(t, hc, d, 0x51_C0DE ^ ((t * 31 + hc * 7 + d) as u64));
            let unfused = run_unfused(&e, &ops, t, hc, d);
            let fused = run_fused(&e, &ops, t, hc, d, None);
            let shape = format!("t={t} hc={hc} d={d}");
            assert!(
                fused.y.iter().all(|v| v.is_finite()),
                "{shape}: fused y has non-finite values"
            );
            assert_bits_eq("pre", &fused.pre, &unfused.pre, &shape);
            assert_bits_eq("post", &fused.post, &unfused.post, &shape);
            assert_bits_eq("comb", &fused.comb, &unfused.comb, &shape);
            assert_bits_eq("y", &fused.y, &unfused.y, &shape);
        }
    }
}

/// GATE 2 — the stationarity exit's receipt: the executed iteration counts are reported, lie
/// in [1, ITERS], and the outputs remain bit-identical to the FULL 20-iteration unfused chain
/// (which is the whole point: the exit fires only where the remaining iterations are bitwise
/// identity). Printed for the lane receipt.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn stationarity_exit_is_bit_invisible_and_its_count_is_reported() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let (hc, d, t) = (4usize, 4096usize, 64usize);
    let ops = operands(t, hc, d, 0xEC0_517);
    let unfused = run_unfused(&e, &ops, t, hc, d);
    let mut niters = Vec::new();
    let fused = run_fused(&e, &ops, t, hc, d, Some(&mut niters));
    let shape = format!("t={t} hc={hc} d={d}");
    assert_bits_eq("pre", &fused.pre, &unfused.pre, &shape);
    assert_bits_eq("post", &fused.post, &unfused.post, &shape);
    assert_bits_eq("comb", &fused.comb, &unfused.comb, &shape);
    assert_bits_eq("y", &fused.y, &unfused.y, &shape);
    assert_eq!(niters.len(), t);
    assert!(
        niters.iter().all(|&n| (1..=ITERS).contains(&n)),
        "executed iteration counts out of range: {niters:?}"
    );
    let (min, max) = (
        niters.iter().min().copied().unwrap(),
        niters.iter().max().copied().unwrap(),
    );
    let mean = niters.iter().map(|&n| n as f64).sum::<f64>() / t as f64;
    println!(
        "[hc-fused-pre receipt] sinkhorn stationarity exit over {t} tokens (iters cap {ITERS}): \
         min={min} mean={mean:.2} max={max}"
    );
}

/// RED ARMS — the three gate scales route to disjoint outputs, so a one-scale mutation must
/// fail the comparator on exactly its own slice and leave the others bit-identical:
/// scale[0] gates `pre` (and through it `y`); scale[1] gates only `post`; scale[2] gates only
/// `comb`. A fourth arm flips one x element, which moves the shared rowsq rescale and must
/// move every output.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn scale_mutations_are_caught_on_exactly_their_own_output() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let (hc, d, t) = (4usize, 512usize, 3usize);
    let ops = operands(t, hc, d, 0x8ED_A87);
    let clean = run_unfused(&e, &ops, t, hc, d);

    for (which, name) in [(0usize, "pre/y"), (1, "post"), (2, "comb")] {
        let mut mutated = Operands {
            x: ops.x.clone(),
            mixes: ops.mixes.clone(),
            scale: ops.scale.clone(),
            base: ops.base.clone(),
        };
        mutated.scale[which] += 0.25;
        let fused = run_fused(&e, &mutated, t, hc, d, None);
        let shape = format!("scale[{which}] ({name}) t={t} hc={hc} d={d}");
        match which {
            0 => {
                assert_bits_ne("pre", &fused.pre, &clean.pre, &shape);
                assert_bits_ne("y", &fused.y, &clean.y, &shape);
                assert_bits_eq("post", &fused.post, &clean.post, &shape);
                assert_bits_eq("comb", &fused.comb, &clean.comb, &shape);
            }
            1 => {
                assert_bits_ne("post", &fused.post, &clean.post, &shape);
                assert_bits_eq("pre", &fused.pre, &clean.pre, &shape);
                assert_bits_eq("comb", &fused.comb, &clean.comb, &shape);
                assert_bits_eq("y", &fused.y, &clean.y, &shape);
            }
            _ => {
                assert_bits_ne("comb", &fused.comb, &clean.comb, &shape);
                assert_bits_eq("pre", &fused.pre, &clean.pre, &shape);
                assert_bits_eq("post", &fused.post, &clean.post, &shape);
                assert_bits_eq("y", &fused.y, &clean.y, &shape);
            }
        }
    }

    // x mutation: the rowsq rescale is a function of the whole slab, so one flipped element
    // must move pre/post/comb (rescaled mixes) and y (collapse reads x directly).
    let mut xmut = Operands {
        x: ops.x.clone(),
        mixes: ops.mixes.clone(),
        scale: ops.scale.clone(),
        base: ops.base.clone(),
    };
    xmut.x[5] += 0.5;
    let fused = run_fused(&e, &xmut, t, hc, d, None);
    let shape = format!("x[5] flip t={t} hc={hc} d={d}");
    assert_bits_ne("pre", &fused.pre, &clean.pre, &shape);
    assert_bits_ne("post", &fused.post, &clean.post, &shape);
    assert_bits_ne("comb", &fused.comb, &clean.comb, &shape);
    assert_bits_ne("y", &fused.y, &clean.y, &shape);
}

// ---------------------------------------------------------------------------
// Whole-model arm: the mini glm5_next hyper-connections fixture from
// hyper_connections_gpu.rs, decoded 24 steps ON vs OFF. The door is inside
// `hyper::pre_finish`, so a byte-identical walk here pins the wiring, the
// engagement counter pins that the ON arm actually took the fused launch.
// ---------------------------------------------------------------------------

const VOCAB: u32 = 32;

fn mini_config_json() -> String {
    r#"{
      "model_type": "glm5_next_text",
      "num_hidden_layers": 2,
      "num_nextn_predict_layers": 0,
      "hidden_size": 128,
      "intermediate_size": 64,
      "vocab_size": 32,
      "max_position_embeddings": 512,
      "rms_norm_eps": 1e-05,
      "hidden_act": "silu",
      "swiglu_limit": 1e30,
      "tie_word_embeddings": true,
      "hc_mult": 4,
      "hc_eps": 1e-06,
      "hc_sinkhorn_iters": 20,
      "mhc": true,
      "layer_types": ["linear_attention", "linear_attention"],
      "mlp_layer_types": ["dense", "dense"],
      "first_k_dense_replace": 2,
      "indexer_types": ["full", "full"],
      "linear_attn_config": {
        "num_heads": 1,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [0, 1],
        "full_attn_layers": []
      },
      "num_attention_heads": 1,
      "num_key_value_heads": 1,
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
      "moe_intermediate_size": 32,
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

fn mini_config() -> ModelConfig {
    ModelConfig::from_hf(&HfConfig::parse(&mini_config_json()))
}

fn mini_plan(config: &ModelConfig) -> ModelPlan {
    memra_gguf::model_packs::for_config(config)
        .expect("glm5_next model pack matches the mini config")
        .compile_plan(config)
        .expect("mini glm5_next plan compiles")
}

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
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
            ggml_type: GgmlType::F32,
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
    .expect("contract for the mini hyper-connections plan");
    let mut tensors = BTreeMap::new();
    for req in contract
        .requirements
        .iter()
        .filter(|r| r.required || weights.contains_key(&r.id))
    {
        let tensor = weights
            .get(&req.id)
            .unwrap_or_else(|| panic!("reference fixture is missing {:?}", req.id));
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

/// GATE 3 — the standing decode-identity gate: 24 greedy-shaped decode steps through the whole
/// mini hyper-connections model, flag OFF then flag ON, per-step logits compared `to_bits`.
/// The ON arm must show engagement (counter advanced) and the OFF arm must show none — a gate
/// that never reached the fused launch would otherwise pass vacuously (wiring-assertions
/// lesson).
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn decode_24_steps_are_byte_identical_on_vs_off_and_the_door_engages() {
    let _gpu = gpu_guard();
    force_true_f32();
    set_door(false);
    let config = mini_config();
    let plan = mini_plan(&config);
    let fixture = deterministic_fixture(&plan).expect("deterministic hc fixture");
    let source = fixture_source(&config, &plan, &fixture.weights);
    let engine = Engine::new(0).expect("CUDA engine on device 0");
    let model = memra_engine::hybrid::HybridModel::load_from_source_without_mtp(&engine, &source)
        .expect("mini hyper-connections model loads");

    let prompt = 6usize;
    let steps = 24usize;
    let ids = tokens(prompt + steps, 0x00D1_E724);

    let run = |on: bool| -> Vec<Vec<u32>> {
        set_door(on);
        let mut cache = memra_engine::cache::Cache::new_planned(&engine, &model.cfg, &plan, 64)
            .expect("cache for the mini hc model");
        let (_primed, _seed, _hiddens) = model
            .prime_cache(&engine, &ids[..prompt], &mut cache, 0)
            .expect("GPU hc prime");
        let mut out = Vec::with_capacity(steps);
        for step in 0..steps {
            let logits = model
                .decode_step(&engine, ids[prompt + step], &mut cache)
                .expect("GPU hc decode step");
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "step {step} (door {on}): non-finite logits"
            );
            out.push(bits(&logits));
        }
        set_door(false);
        out
    };

    let before_off = HC_FUSED_PRE_DISPATCHES.load(Ordering::Relaxed);
    let off = run(false);
    let after_off = HC_FUSED_PRE_DISPATCHES.load(Ordering::Relaxed);
    assert_eq!(
        before_off, after_off,
        "the OFF arm must never take the fused launch (flag-off engagement)"
    );

    let on = run(true);
    let after_on = HC_FUSED_PRE_DISPATCHES.load(Ordering::Relaxed);
    assert!(
        after_on > after_off,
        "the ON arm never engaged the fused pre-chain — the gate would be vacuous \
         (counter {after_off} -> {after_on})"
    );

    for (step, (a, b)) in off.iter().zip(&on).enumerate() {
        assert_eq!(
            a, b,
            "decode step {step}: ON and OFF logits differ in bits — the fused pre-chain is \
             not byte-identical through the whole model"
        );
    }
    println!(
        "[hc-fused-pre receipt] 24-step decode byte identity ON==OFF; engagement counter \
         {after_off} -> {after_on} (ON arm), flat in OFF arm"
    );
}
