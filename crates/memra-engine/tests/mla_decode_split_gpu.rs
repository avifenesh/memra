//! Gate for `MEMRA_MLA_DECODE_SPLIT` — the MLA absorb/decompress decode-split twins
//! (`memra_mla_absorb_q_split_f32` / `memra_mla_decompress_v_split_f32`,
//! lane/glm5-decode-diet lever 4, 2026-08-31).
//!
//! THE ONE CHANGE UNDER TEST: launch geometry. The unsplit launchers put `t_q * n_head`
//! blocks on the grid (64 at t=1 on the glm5 geometry — single-digit occupancy on the
//! serving card class); the twins split each block's OUTPUT RANGE across `split` blocks.
//! Every output element keeps the same one-thread serial ascending-index dot, so the claim
//! is BIT identity — for EVERY split value, including ones that do not divide the output
//! width (the tail guard) — plus a whole-model 24-step decode byte-identity twin on the
//! DSA-indexed MLA mini fixture (the kpool gate's config), engagement counted.
//!
//! Rig law: correctness-only, run under `flock /tmp/memra-5090.lock`, TF32 forced off,
//! `-- --ignored --test-threads=1`.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::mla_ffi::MLA_DECODE_SPLIT_DISPATCHES;
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

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

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

/// The flag is read PER CALL by design (rollback seam). Serialized behind `gpu_guard`.
///
/// THE OFF ARM SETS `=0` AND MUST NOT UNSET, since the 2026-09-03 default flip. It used
/// `remove_var`, which was correct while the door defaulted OFF and became a VACUITY BUG the
/// moment the default moved: unset now resolves to ON, so the "off" arm would have run the
/// split path and this gate would have compared split against split and passed, while proving
/// nothing about the bit identity it exists to assert. Setting `=0` explicitly is the arm the
/// door's own rollback seam names, and it stays correct under either default.
fn set_door(on: bool) {
    // SAFETY: all tests in this binary hold `gpu_guard` while touching env or the GPU.
    unsafe {
        std::env::set_var("MEMRA_MLA_DECODE_SPLIT", if on { "1" } else { "0" });
    }
}

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

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn dp(s: &CudaSlice<f32>, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> *const f32 {
    s.device_ptr(stream).0 as *const f32
}
fn dpm(s: &mut CudaSlice<f32>, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> *mut f32 {
    s.device_ptr_mut(stream).0 as *mut f32
}

#[allow(clippy::too_many_arguments)]
fn absorb(
    e: &Engine,
    q_nope: &[f32],
    wk_b: &[f32],
    t: usize,
    nh: usize,
    dn: usize,
    r: usize,
    split: Option<i32>,
) -> Vec<f32> {
    let stream = e.stream();
    let q = e.htod(q_nope).unwrap();
    let w = e.htod(wk_b).unwrap();
    let mut out = e.uninit(t * nh * r).unwrap();
    let rc = unsafe {
        match split {
            None => memra_engine::mla_ffi::memra_mla_absorb_q_f32(
                dp(&q, &stream),
                dp(&w, &stream),
                dpm(&mut out, &stream),
                t as i32,
                nh as i32,
                dn as i32,
                r as i32,
                stream.cu_stream() as *mut c_void,
            ),
            Some(s) => memra_engine::mla_ffi::memra_mla_absorb_q_split_f32(
                dp(&q, &stream),
                dp(&w, &stream),
                dpm(&mut out, &stream),
                t as i32,
                nh as i32,
                dn as i32,
                r as i32,
                s,
                stream.cu_stream() as *mut c_void,
            ),
        }
    };
    assert_eq!(rc, 0, "absorb launch (split {split:?})");
    e.dtoh(&out).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn decompress(
    e: &Engine,
    o_lat: &[f32],
    wv_b: &[f32],
    t: usize,
    nh: usize,
    dv: usize,
    r: usize,
    split: Option<i32>,
) -> Vec<f32> {
    let stream = e.stream();
    let o = e.htod(o_lat).unwrap();
    let w = e.htod(wv_b).unwrap();
    let mut out = e.uninit(t * nh * dv).unwrap();
    let rc = unsafe {
        match split {
            None => memra_engine::mla_ffi::memra_mla_decompress_v_f32(
                dp(&o, &stream),
                dp(&w, &stream),
                dpm(&mut out, &stream),
                t as i32,
                nh as i32,
                dv as i32,
                r as i32,
                stream.cu_stream() as *mut c_void,
            ),
            Some(s) => memra_engine::mla_ffi::memra_mla_decompress_v_split_f32(
                dp(&o, &stream),
                dp(&w, &stream),
                dpm(&mut out, &stream),
                t as i32,
                nh as i32,
                dv as i32,
                r as i32,
                s,
                stream.cu_stream() as *mut c_void,
            ),
        }
    };
    assert_eq!(rc, 0, "decompress launch (split {split:?})");
    e.dtoh(&out).unwrap()
}

/// GATE 1 — kernel-level bit identity, both twins, across the glm5 serving geometry and a
/// mini geometry, at decode/verify widths, for split values that do and do NOT divide the
/// output width (the tail guard), including split == out_dim (one output per block).
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn split_twins_are_bit_identical_to_the_unsplit_kernels_at_every_split() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    // (n_head, d_nope, kv_rank, d_v): glm5_next serving geometry + the mini fixture's.
    for &(nh, dn, r, dv) in &[(64usize, 128usize, 512usize, 128usize), (2, 16, 16, 16)] {
        for &t in &[1usize, 2, 8] {
            let q = randf(t * nh * dn, 0xAB50 ^ (t * 31 + nh) as u64);
            let wk = randf(nh * r * dn, 0xAB51 ^ nh as u64);
            let ol = randf(t * nh * r, 0xAB52 ^ (t * 7 + nh) as u64);
            let wv = randf(nh * dv * r, 0xAB53 ^ nh as u64);
            let a0 = absorb(&e, &q, &wk, t, nh, dn, r, None);
            let d0 = decompress(&e, &ol, &wv, t, nh, dv, r, None);
            for &s in &[2i32, 3, 5, 16] {
                if (s as usize) <= r {
                    let a1 = absorb(&e, &q, &wk, t, nh, dn, r, Some(s));
                    assert!(
                        bits_equal(&a0, &a1),
                        "absorb nh={nh} r={r} t={t} split={s}: bits differ from unsplit"
                    );
                }
                if (s as usize) <= dv {
                    let d1 = decompress(&e, &ol, &wv, t, nh, dv, r, Some(s));
                    assert!(
                        bits_equal(&d0, &d1),
                        "decompress nh={nh} dv={dv} t={t} split={s}: bits differ from unsplit"
                    );
                }
            }
            // split == out_dim: one output element per block, the extreme tail shape.
            let a1 = absorb(&e, &q, &wk, t, nh, dn, r, Some(r as i32));
            assert!(bits_equal(&a0, &a1), "absorb split=out_dim differs");
            let d1 = decompress(&e, &ol, &wv, t, nh, dv, r, Some(dv as i32));
            assert!(bits_equal(&d0, &d1), "decompress split=out_dim differs");
        }
    }
    println!(
        "[mla-decode-split receipt] bit identity split-vs-unsplit at splits {{2,3,5,16,out}} \
         x t in {{1,2,8}} on (64h,128,512,128) and (2h,16,16,16)"
    );
}

/// RED — the comparator binds: a transposed weight (loads, runs, finite, silently wrong) and
/// a one-element input flip must both move the split twin's output.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn mutations_move_the_split_output() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let (nh, dn, r, t) = (4usize, 32usize, 64usize, 2usize);
    let q = randf(t * nh * dn, 0x8ED1);
    let wk = randf(nh * r * dn, 0x8ED2);
    let clean = absorb(&e, &q, &wk, t, nh, dn, r, Some(3));

    // Transposed weight slab per head: same bytes, wrong layout.
    let mut wk_t = vec![0.0f32; wk.len()];
    for h in 0..nh {
        let s = &wk[h * r * dn..(h + 1) * r * dn];
        let d = &mut wk_t[h * r * dn..(h + 1) * r * dn];
        for l in 0..r {
            for p in 0..dn {
                d[p * r + l] = s[l * dn + p];
            }
        }
    }
    let got = absorb(&e, &q, &wk_t, t, nh, dn, r, Some(3));
    assert!(
        got.iter().all(|v| v.is_finite()),
        "transposed-weight red went non-finite — reds must be silent"
    );
    assert!(
        !bits_equal(&clean, &got),
        "transposing wk_b did not move the split output — the comparator does not bind"
    );

    let mut q2 = q.clone();
    q2[7] += 0.5;
    let got = absorb(&e, &q2, &wk, t, nh, dn, r, Some(3));
    assert!(
        !bits_equal(&clean, &got),
        "an input flip did not move the split output"
    );
}

// ---------------------------------------------------------------------------
// Whole-model arm: the DSA-indexed MLA mini fixture (glm5_kpool_indexer_gpu's
// config — two deepseek_sparse_attention layers), decoded 24 steps ON vs OFF.
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
      "swiglu_limit": 10.0,
      "tie_word_embeddings": true,
      "hc_mult": 4,
      "hc_eps": 1e-06,
      "hc_sinkhorn_iters": 20,
      "mhc": true,
      "layer_types": ["deepseek_sparse_attention", "deepseek_sparse_attention"],
      "mlp_layer_types": ["dense", "dense"],
      "first_k_dense_replace": 2,
      "indexer_types": ["full", "full"],
      "linear_attn_config": {
        "num_heads": 1,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [],
        "full_attn_layers": [0, 1]
      },
      "num_attention_heads": 2,
      "num_key_value_heads": 2,
      "q_lora_rank": 16,
      "kv_lora_rank": 64,
      "qk_head_dim": 16,
      "qk_nope_head_dim": 16,
      "qk_rope_head_dim": 0,
      "v_head_dim": 64,
      "mla_use_nope": true,
      "index_n_heads": 2,
      "index_head_dim": 8,
      "index_topk": 16,
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
    .expect("contract for the mini MLA plan");
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

/// GATE 2 — the standing decode-identity gate: 24 decode steps through the DSA-indexed MLA
/// mini model, flag OFF then ON, per-step logits compared `to_bits`; the ON arm must engage
/// (the fixture's kv_lora_rank/v_head_dim are 64 so the policy's 32-outputs-per-block floor
/// still admits a split at 2 blocks) and the OFF arm must not.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn decode_24_steps_are_byte_identical_split_on_vs_off_and_the_door_engages() {
    let _gpu = gpu_guard();
    force_true_f32();
    set_door(false);
    let config = mini_config();
    let plan = mini_plan(&config);
    let fixture = deterministic_fixture(&plan).expect("deterministic MLA fixture");
    let source = fixture_source(&config, &plan, &fixture.weights);
    let engine = Engine::new(0).expect("CUDA engine on device 0");
    let model = memra_engine::hybrid::HybridModel::load_from_source_without_mtp(&engine, &source)
        .expect("mini MLA model loads");

    let prompt = 6usize;
    let steps = 24usize;
    let ids = tokens(prompt + steps, 0x005E_ED24);

    let run = |on: bool| -> Vec<Vec<u32>> {
        set_door(on);
        let mut cache = memra_engine::cache::Cache::new_planned(&engine, &model.cfg, &plan, 64)
            .expect("cache for the mini MLA model");
        let (_primed, _seed, _hiddens) = model
            .prime_cache(&engine, &ids[..prompt], &mut cache, 0)
            .expect("GPU MLA prime");
        let mut out = Vec::with_capacity(steps);
        for step in 0..steps {
            let logits = model
                .decode_step(&engine, ids[prompt + step], &mut cache)
                .expect("GPU MLA decode step");
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "step {step} (door {on}): non-finite logits"
            );
            out.push(logits.iter().map(|v| v.to_bits()).collect());
        }
        set_door(false);
        out
    };

    let c0 = MLA_DECODE_SPLIT_DISPATCHES.load(Ordering::Relaxed);
    let off = run(false);
    let c1 = MLA_DECODE_SPLIT_DISPATCHES.load(Ordering::Relaxed);
    assert_eq!(c0, c1, "the OFF arm must never take the split launch");

    let on = run(true);
    let c2 = MLA_DECODE_SPLIT_DISPATCHES.load(Ordering::Relaxed);
    assert!(
        c2 > c1,
        "the ON arm never engaged the split door (counter {c1} -> {c2}) — the gate would be \
         vacuous"
    );
    for (step, (a, b)) in off.iter().zip(&on).enumerate() {
        assert_eq!(
            a, b,
            "decode step {step}: split-ON and split-OFF logits differ in bits"
        );
    }
    println!(
        "[mla-decode-split receipt] 24-step decode byte identity ON==OFF; engagement counter \
         {c1} -> {c2} (ON arm), flat in OFF arm"
    );
}
