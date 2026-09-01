//! Gate for `MEMRA_HC_DECODE_WS` — the persistent hc-glue decode workspace
//! (lane/glm5-decode-diet lever 2, 2026-08-31).
//!
//! THE ONE CHANGE UNDER TEST: the T=1 hc decode walk lands its glue transients (mixes,
//! Sinkhorn gates, comb, collapse y, both norm scratches, the per-site post output) in one
//! per-engine `HyperDecodeWs` instead of fresh allocations every layer. The kernels, their
//! order and their operand bytes are unchanged, so the claim is BYTE identity of the decode
//! logits ON vs OFF — plus a counted receipt that the allocator-call class the launch-diet
//! census measured (2,358 `cuMemAllocAsync+Free`/token) actually shrinks (`SCRATCH_ALLOC_CALLS`
//! delta per 24 steps, printed both arms — the launch-econ instrument's host twin).
//!
//! COMPOSITION: lever 1 (`MEMRA_HC_FUSED_PRE`) shares `pre_finish_into` with this walk; the
//! compose arm runs both doors ON against the both-OFF baseline, byte-compared, with both
//! engagement counters advancing (wiring-assertions lesson: engagement is asserted, never
//! inferred from a green diff).
//!
//! Rig law: correctness-only, run under `flock /tmp/memra-5090.lock`, TF32 forced off,
//! `-- --ignored --test-threads=1`.

use memra_engine::hybrid_forward::HC_DECODE_WS_DISPATCHES;
use memra_engine::hyper::HC_FUSED_PRE_DISPATCHES;
use memra_engine::{Engine, SCRATCH_ALLOC_CALLS};
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
use std::sync::atomic::Ordering;

const VOCAB: u32 = 32;

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

/// Both flags are read PER CALL by design (rollback seam), so the gate drives all arms in one
/// process. Serialized behind `gpu_guard`.
fn set_flag(name: &str, on: bool) {
    // SAFETY: all tests in this binary hold `gpu_guard` while touching env or the GPU.
    unsafe {
        if on {
            std::env::set_var(name, "1");
        } else {
            std::env::remove_var(name);
        }
    }
}

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

struct Harness {
    engine: Engine,
    model: memra_engine::hybrid::HybridModel,
    plan: ModelPlan,
}

impl Harness {
    fn new() -> Self {
        force_true_f32();
        let config = mini_config();
        let plan = mini_plan(&config);
        let fixture = deterministic_fixture(&plan).expect("deterministic hc fixture");
        let source = fixture_source(&config, &plan, &fixture.weights);
        let engine = Engine::new(0).expect("CUDA engine on device 0");
        let model =
            memra_engine::hybrid::HybridModel::load_from_source_without_mtp(&engine, &source)
                .expect("mini hyper-connections model loads");
        Self {
            engine,
            model,
            plan,
        }
    }

    /// Prime + `steps` decode steps; returns per-step logits bits and the SCRATCH_ALLOC_CALLS
    /// delta across ONLY the decode loop (the workspace's claim is a decode claim).
    fn decode_bits(&self, ids: &[u32], prompt: usize, steps: usize) -> (Vec<Vec<u32>>, u64) {
        let mut cache =
            memra_engine::cache::Cache::new_planned(&self.engine, &self.model.cfg, &self.plan, 64)
                .expect("cache for the mini hc model");
        let (_primed, _seed, _hiddens) = self
            .model
            .prime_cache(&self.engine, &ids[..prompt], &mut cache, 0)
            .expect("GPU hc prime");
        let alloc0 = SCRATCH_ALLOC_CALLS.load(Ordering::Relaxed);
        let mut out = Vec::with_capacity(steps);
        for step in 0..steps {
            let logits = self
                .model
                .decode_step(&self.engine, ids[prompt + step], &mut cache)
                .expect("GPU hc decode step");
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "step {step}: non-finite logits"
            );
            out.push(logits.iter().map(|v| v.to_bits()).collect());
        }
        let allocs = SCRATCH_ALLOC_CALLS.load(Ordering::Relaxed) - alloc0;
        (out, allocs)
    }
}

/// GATE 1 — the standing decode-identity gate for the workspace door: 24 decode steps, flag
/// OFF then ON, per-step logits compared `to_bits`; the ON arm must engage (counter) and its
/// decode-loop allocator-call count must be STRICTLY LOWER than the OFF arm's (the census
/// class this lever exists to shrink). Both counts print — the lane receipt.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn decode_24_steps_are_byte_identical_ws_on_vs_off_and_alloc_calls_drop() {
    let _gpu = gpu_guard();
    set_flag("MEMRA_HC_DECODE_WS", false);
    set_flag("MEMRA_HC_FUSED_PRE", false);
    let h = Harness::new();
    let (prompt, steps) = (6usize, 24usize);
    let ids = tokens(prompt + steps, 0x00D1_E7B5);

    let ws0 = HC_DECODE_WS_DISPATCHES.load(Ordering::Relaxed);
    let (off, off_allocs) = h.decode_bits(&ids, prompt, steps);
    assert_eq!(
        ws0,
        HC_DECODE_WS_DISPATCHES.load(Ordering::Relaxed),
        "the OFF arm must never take the workspace walk"
    );

    set_flag("MEMRA_HC_DECODE_WS", true);
    let (on, on_allocs) = h.decode_bits(&ids, prompt, steps);
    set_flag("MEMRA_HC_DECODE_WS", false);
    let ws1 = HC_DECODE_WS_DISPATCHES.load(Ordering::Relaxed);
    assert!(
        ws1 > ws0,
        "the ON arm never engaged the workspace walk (counter {ws0} -> {ws1}) — the gate \
         would be vacuous"
    );

    for (step, (a, b)) in off.iter().zip(&on).enumerate() {
        assert_eq!(
            a, b,
            "decode step {step}: WS-ON and WS-OFF logits differ in bits — the workspace walk \
             is not the allocating walk's program"
        );
    }
    assert!(
        on_allocs < off_allocs,
        "the workspace arm did not reduce allocator calls (ON {on_allocs} vs OFF {off_allocs} \
         over {steps} steps) — the lever is wired but not landing"
    );
    println!(
        "[hc-decode-ws receipt] 24-step decode byte identity ON==OFF; scratch-alloc calls \
         over {steps} steps: OFF {off_allocs} ({:.1}/step) -> ON {on_allocs} ({:.1}/step), \
         -{} calls ({:.1}%)",
        off_allocs as f64 / steps as f64,
        on_allocs as f64 / steps as f64,
        off_allocs - on_allocs,
        100.0 * (off_allocs - on_allocs) as f64 / off_allocs as f64
    );
}

/// GATE 2 — composition: lever 1 (fused pre-chain) + lever 2 (workspace) both ON against the
/// both-OFF baseline, byte-compared over 24 steps, with BOTH engagement counters advancing.
/// The two doors share `pre_finish_into`, so this is the arm that would catch a bad merge of
/// the two code paths.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn ws_composes_with_the_fused_prechain_bitwise() {
    let _gpu = gpu_guard();
    set_flag("MEMRA_HC_DECODE_WS", false);
    set_flag("MEMRA_HC_FUSED_PRE", false);
    let h = Harness::new();
    let (prompt, steps) = (5usize, 24usize);
    let ids = tokens(prompt + steps, 0x00C0_4B05);

    let (off, _) = h.decode_bits(&ids, prompt, steps);

    let ws0 = HC_DECODE_WS_DISPATCHES.load(Ordering::Relaxed);
    let fp0 = HC_FUSED_PRE_DISPATCHES.load(Ordering::Relaxed);
    set_flag("MEMRA_HC_DECODE_WS", true);
    set_flag("MEMRA_HC_FUSED_PRE", true);
    let (on, _) = h.decode_bits(&ids, prompt, steps);
    set_flag("MEMRA_HC_DECODE_WS", false);
    set_flag("MEMRA_HC_FUSED_PRE", false);
    assert!(
        HC_DECODE_WS_DISPATCHES.load(Ordering::Relaxed) > ws0,
        "compose arm: workspace walk never engaged"
    );
    assert!(
        HC_FUSED_PRE_DISPATCHES.load(Ordering::Relaxed) > fp0,
        "compose arm: fused pre-chain never engaged"
    );

    for (step, (a, b)) in off.iter().zip(&on).enumerate() {
        assert_eq!(
            a, b,
            "decode step {step}: WS+FUSED and both-OFF logits differ in bits"
        );
    }
    println!("[hc-decode-ws receipt] 24-step compose arm (WS+FUSED vs both-OFF) byte-identical");
}
