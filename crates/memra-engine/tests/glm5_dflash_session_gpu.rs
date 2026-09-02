//! glm5_next DFLASH2 DRAFT SOURCE gate (lane/glm5-dflash-draft-src, 2026-08-30).
//!
//! `glm5_spec_session_gpu.rs` pins the SERVED shape with the native MTP draft source; this
//! file pins the SAME served shape with the ALTERNATE source — the DFlash2 block-diffusion
//! drafter (`MEMRA_GLM5_DFLASH`) — on the same mini glm5 hc fixture plus a mini DFlash2
//! drafter checkpoint written through the REAL loader (`DflashDraft::load`: config census,
//! safetensors, precision seam). The trunk loads WITHOUT the MTP head throughout (the q38
//! pattern this lane ships: the drafter needs the target's embed/lm_head only).
//!
//! 1. Served-burst greedy byte identity vs plain decode at K=1..7 — THE gate: a draft
//!    source can only move acceptance, never output.
//! 2. Forced-rejection j-sweep through the served burst, byte-identical.
//! 3. RED, tap-shift (`MEMRA_GLM5_DFLASH_GATE_RED=tap-shift`): deliberately wrong feature
//!    input (every tap layer +1 — the probe-measured contract violated) must CHANGE the
//!    draft stream (features are live) while the output tape STAYS byte-identical and
//!    acceptance does not improve. The acceptance-COLLAPSE magnitude is a real-artifact
//!    number (probe band: 0.73 acc@1) and lands in the box three-way window — the mini
//!    fixture's random drafter accepts near chance on both arms, so the rig arm pins the
//!    mechanism, not the magnitude.
//! 4. RED, rollback disabled + forced rejections through the dflash session: diverges or
//!    fails loudly (the tparallel red arm re-proven on this source).
//! 5. Sampled twin: pinned-seed determinism, burst-split invariance (selector draws and
//!    accept uniforms ride the session's Philox counters), seed sensitivity.
//! 6. EOS mid-burst finishes the session; later bursts are empty.
//! 7. DRAFT-SOURCE SELECTION MATRIX (subprocess-captured boot logs, red-proven):
//!    dflash2 armed (head NOT loaded / head ALSO loaded), native-mtp, fail-closed warn,
//!    and flag-off = zero `[glm5-spec]` lines. The dflash-vs-mtp VRAM-at-ready delta is
//!    printed for the lane doc (mini scale; box numbers come with the three-way window).
//!
//! Rig law (exactness only, never timing):
//!   NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
//!     cargo test -p memra-engine --test glm5_dflash_session_gpu -- --ignored --test-threads=1

use memra_engine::Engine;
use memra_engine::forward::argmax;
use memra_engine::glm_spec::{Glm5SpecKnobs, Glm5SpecSession};
use memra_engine::hybrid::HybridModel;
use memra_engine::spec::SpecSampling;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::ModelPlan;
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, LayerTensor, MtpTensor, OutputHead, TensorContract,
    TensorId, TensorMatch,
};
use memra_reference::{ReferenceTensor, ReferenceWeights, deterministic_fixture};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const HIDDEN: usize = 128;
const VOCAB: u32 = 32;
/// Past the k-pool raw budget (index_topk 8, kpool 4) — the trunk indexer runs sparse.
const PROMPT: usize = 24;
/// Drafter block: 8 = anchor + 7 drafts (the pinned artifact's shape, kept exactly).
const BLOCK: usize = 8;
const K: usize = BLOCK - 1;

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call yet in this process; call_once serializes test threads.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Trunk fixture: the glm5_spec_session_gpu mini config, through the real pack/contract.
// ---------------------------------------------------------------------------------------------

fn mini_config_json() -> String {
    r#"{
      "model_type": "glm5_next_text",
      "num_hidden_layers": 4,
      "num_nextn_predict_layers": 1,
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
        "num_heads": 1,
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

fn mini_config() -> ModelConfig {
    ModelConfig::from_hf(&HfConfig::parse(&mini_config_json()))
}

fn mini_plan(config: &ModelConfig) -> ModelPlan {
    memra_gguf::model_packs::for_config(config)
        .expect("glm5_next model pack matches the mini config")
        .compile_plan(config)
        .expect("mini glm5_next plan compiles")
}

/// Deterministic non-trivial values (an all-ones norm cannot catch a swapped operand).
fn varied(len: usize, seed: u64, spread: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed)
                .rotate_left(17) as f64
                / u64::MAX as f64;
            1.0 + spread * (x as f32 - 0.5)
        })
        .collect()
}

/// Zero-centered deterministic values for drafter matrices.
fn centered(len: usize, seed: u64, spread: f32) -> Vec<f32> {
    varied(len, seed, spread)
        .into_iter()
        .map(|v| v - 1.0)
        .collect()
}

fn fixture_weights(plan: &ModelPlan) -> ReferenceWeights {
    let mut weights = deterministic_fixture(plan)
        .expect("deterministic glm5 hc+mtp fixture")
        .weights;
    for (tensor, seed) in [
        (MtpTensor::EmbeddingNorm, 0xE0_12u64),
        (MtpTensor::HiddenNorm, 0x40_77),
        (MtpTensor::OutputNorm, 0x5EAD),
    ] {
        weights.insert(
            TensorId::Mtp { depth: 0, tensor },
            ReferenceTensor::new(vec![HIDDEN], varied(HIDDEN, seed, 0.8)).unwrap(),
        );
    }
    weights
}

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
    ggml_type: GgmlType,
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

fn fixture_source(config: &ModelConfig, plan: &ModelPlan) -> FixtureSource {
    let weights = fixture_weights(plan);
    let contract = TensorContract::for_plan(
        plan,
        CheckpointDialect::Gguf,
        ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        },
    )
    .expect("contract for the mini glm5_next hc+mtp plan");
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
            "shape mismatch for {:?}",
            req.id
        );
        let (bytes, ggml_type) = if is_expert_bank(&req.id) {
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

// ---------------------------------------------------------------------------------------------
// Drafter fixture: a mini DFlash2 checkpoint DIR through the REAL loader (config census +
// safetensors + tensor census). Geometry scaled to the trunk: hidden 128 (== n_embd, the
// loader's own bound), taps [0,1,2] of the 4-layer trunk (shift-red [1,2,3] stays in range),
// block 8, mask id 4 (< vocab 32), selector rank 8 top_k 4, conv group 16 (16 | 128).
// ---------------------------------------------------------------------------------------------

fn f32_to_bf16_bytes(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    // round-to-nearest-even on the dropped half (fixture-grade; loader widens exactly).
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (((rounded >> 16) & 0xFFFF) as u16).to_le_bytes()
}

fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut header = String::from("{");
    let mut data: Vec<u8> = Vec::new();
    for (i, (name, shape, vals)) in tensors.iter().enumerate() {
        let elements: usize = shape.iter().product();
        assert_eq!(elements, vals.len(), "drafter fixture shape for {name}");
        let start = data.len();
        for v in vals {
            data.extend_from_slice(&f32_to_bf16_bytes(*v));
        }
        let end = data.len();
        if i > 0 {
            header.push(',');
        }
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{start},{end}]}}",
            dims.join(",")
        ));
    }
    header.push('}');
    let hb = header.as_bytes();
    let mut out = Vec::with_capacity(8 + hb.len() + data.len());
    out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
    out.extend_from_slice(hb);
    out.extend_from_slice(&data);
    std::fs::write(path, out).expect("write drafter safetensors");
}

const DRAFT_LAYERS: usize = 2;
const DRAFT_NH: usize = 2;
const DRAFT_NKV: usize = 1;
const DRAFT_HD: usize = 32;
const DRAFT_FF: usize = 64;
const DRAFT_RANK: usize = 8;
const DRAFT_TOPK: usize = 4;
const DRAFT_GROUP: usize = 16;
const DRAFT_CONV_K: usize = 2;

/// Write the mini drafter checkpoint dir; returns its path. One dir per process id under
/// the temp dir; the caller that made it deletes it (tmp hygiene).
fn write_mini_drafter(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("glm5-dflash-mini-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("drafter fixture dir");
    let config = format!(
        r#"{{
  "architectures": ["DFlash2DraftModel"],
  "hidden_size": {HIDDEN},
  "num_attention_heads": {DRAFT_NH},
  "num_key_value_heads": {DRAFT_NKV},
  "head_dim": {DRAFT_HD},
  "intermediate_size": {DRAFT_FF},
  "num_hidden_layers": {DRAFT_LAYERS},
  "rms_norm_eps": 1e-06,
  "sliding_window": 2048,
  "is_causal": false,
  "layer_types": ["sliding_attention", "sliding_attention"],
  "rope_parameters": {{"rope_type": "default", "rope_theta": 1000000.0}},
  "dflash_config": {{
    "block_size": {BLOCK},
    "mask_token_id": 4,
    "target_layer_ids": [0, 1, 2],
    "selector_rank": {DRAFT_RANK},
    "selector_top_k": {DRAFT_TOPK},
    "conv_kernel_size": {DRAFT_CONV_K},
    "conv_group_size": {DRAFT_GROUP}
  }}
}}"#
    );
    std::fs::write(dir.join("config.json"), config).expect("drafter config.json");

    let h = HIDDEN;
    let groups = h / DRAFT_GROUP;
    let mut seed = 0x0DF1_A500_u64;
    let mut next = |len: usize, spread: f32, norm: bool| -> Vec<f32> {
        seed = seed.wrapping_add(0x9E37_79B9);
        if norm {
            varied(len, seed, spread)
        } else {
            centered(len, seed, spread)
        }
    };
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    for i in 0..DRAFT_LAYERS {
        let p = |s: &str| format!("layers.{i}.{s}");
        // safetensors shape order: [out_features, in_features]
        tensors.push((
            p("self_attn.q_proj.weight"),
            vec![DRAFT_NH * DRAFT_HD, h],
            next(DRAFT_NH * DRAFT_HD * h, 0.3, false),
        ));
        tensors.push((
            p("self_attn.k_proj.weight"),
            vec![DRAFT_NKV * DRAFT_HD, h],
            next(DRAFT_NKV * DRAFT_HD * h, 0.3, false),
        ));
        tensors.push((
            p("self_attn.v_proj.weight"),
            vec![DRAFT_NKV * DRAFT_HD, h],
            next(DRAFT_NKV * DRAFT_HD * h, 0.3, false),
        ));
        tensors.push((
            p("self_attn.o_proj.weight"),
            vec![h, DRAFT_NH * DRAFT_HD],
            next(h * DRAFT_NH * DRAFT_HD, 0.3, false),
        ));
        tensors.push((
            p("self_attn.q_norm.weight"),
            vec![DRAFT_HD],
            next(DRAFT_HD, 0.4, true),
        ));
        tensors.push((
            p("self_attn.k_norm.weight"),
            vec![DRAFT_HD],
            next(DRAFT_HD, 0.4, true),
        ));
        tensors.push((p("input_layernorm.weight"), vec![h], next(h, 0.4, true)));
        tensors.push((
            p("post_attention_layernorm.weight"),
            vec![h],
            next(h, 0.4, true),
        ));
        tensors.push((
            p("mlp.gate_proj.weight"),
            vec![DRAFT_FF, h],
            next(DRAFT_FF * h, 0.3, false),
        ));
        tensors.push((
            p("mlp.up_proj.weight"),
            vec![DRAFT_FF, h],
            next(DRAFT_FF * h, 0.3, false),
        ));
        tensors.push((
            p("mlp.down_proj.weight"),
            vec![h, DRAFT_FF],
            next(h * DRAFT_FF, 0.3, false),
        ));
        for conv in ["attention_conv", "mlp_conv"] {
            tensors.push((
                p(&format!("{conv}.base_kernel")),
                vec![2, DRAFT_CONV_K, h],
                next(2 * DRAFT_CONV_K * h, 0.4, false),
            ));
            tensors.push((
                p(&format!("{conv}.kernel_projection.weight")),
                vec![2 * DRAFT_CONV_K * groups, h],
                next(2 * DRAFT_CONV_K * groups * h, 0.3, false),
            ));
        }
    }
    let n_taps = 3usize;
    tensors.push((
        "fc.weight".into(),
        vec![h, n_taps * h],
        next(h * n_taps * h, 0.3, false),
    ));
    tensors.push(("hidden_norm.weight".into(), vec![h], next(h, 0.4, true)));
    tensors.push(("norm.weight".into(), vec![h], next(h, 0.4, true)));
    tensors.push((
        "candidate_selector.hidden_projection.weight".into(),
        vec![DRAFT_RANK, h],
        next(DRAFT_RANK * h, 0.3, false),
    ));
    tensors.push((
        "candidate_selector.predecessor_codebook".into(),
        vec![VOCAB as usize, DRAFT_RANK],
        next(VOCAB as usize * DRAFT_RANK, 1.6, false),
    ));
    tensors.push((
        "candidate_selector.successor_codebook".into(),
        vec![VOCAB as usize, DRAFT_RANK],
        next(VOCAB as usize * DRAFT_RANK, 1.6, false),
    ));
    write_safetensors(&dir.join("model.safetensors"), &tensors);
    dir
}

// ---------------------------------------------------------------------------------------------
// Harness: trunk WITHOUT the MTP head + the drafter armed via MEMRA_GLM5_DFLASH — the q38
// VRAM pattern this lane ships. Env mutations serialized behind gpu_guard by every caller.
// ---------------------------------------------------------------------------------------------

struct Harness {
    engine: Engine,
    model: HybridModel,
    plan: ModelPlan,
    drafter_dir: PathBuf,
}

impl Drop for Harness {
    fn drop(&mut self) {
        // tmp hygiene: the task that made the fixture deletes it.
        std::fs::remove_dir_all(&self.drafter_dir).ok();
        // SAFETY: serialized behind gpu_guard by every caller.
        unsafe { std::env::remove_var("MEMRA_GLM5_DFLASH") };
    }
}

impl Harness {
    fn new(tag: &str) -> Self {
        force_true_f32();
        let drafter_dir = write_mini_drafter(tag);
        // SAFETY: serialized behind gpu_guard by every caller.
        unsafe {
            std::env::remove_var("MEMRA_FRSPEC_TRIM");
            std::env::remove_var("MEMRA_GLM5_MTP"); // the head is NOT loaded on this route
            std::env::remove_var("MEMRA_GLM5_DFLASH_GATE_RED");
            std::env::set_var("MEMRA_GLM5_DFLASH", &drafter_dir);
        }
        let config = mini_config();
        let plan = mini_plan(&config);
        let source = fixture_source(&config, &plan);
        let engine = Engine::new(0).expect("CUDA engine on device 0");
        let model = HybridModel::load_from_source(&engine, &source)
            .expect("mini glm5 loads with the DFlash2 drafter");
        assert!(model.hyper.is_some(), "hc trunk expected");
        assert!(
            model.mtp.is_none(),
            "the q38 pattern: the native MTP head must NOT load on the dflash route"
        );
        assert!(
            model.glm5_dflash.is_some(),
            "MEMRA_GLM5_DFLASH must attach the drafter"
        );
        Self {
            engine,
            model,
            plan,
            drafter_dir,
        }
    }

    fn fresh_primed(
        &self,
        prompt: &[u32],
        max_ctx: usize,
    ) -> (memra_engine::cache::Cache, Vec<f32>) {
        let mut cache = memra_engine::cache::Cache::new_planned(
            &self.engine,
            &self.model.cfg,
            &self.plan,
            max_ctx,
        )
        .expect("cache for the mini glm5 model");
        let (logits, _seed, _hiddens) = self
            .model
            .prime_cache(&self.engine, prompt, &mut cache, 0)
            .expect("hc prime");
        (cache, logits)
    }
}

fn plain_tape(h: &Harness, prompt: &[u32], max_new: usize) -> Vec<u32> {
    let (mut cache, logits) = h.fresh_primed(prompt, prompt.len() + max_new + 16);
    let mut tape = Vec::with_capacity(max_new);
    tape.push(argmax(&logits) as u32);
    while tape.len() < max_new {
        let ll = h
            .model
            .decode_step(&h.engine, *tape.last().unwrap(), &mut cache)
            .expect("plain decode step");
        tape.push(argmax(&ll) as u32);
    }
    tape
}

/// Worker-shaped burst drive with the burst-boundary invariants (the session_gpu shape).
fn drive_bursts(
    h: &Harness,
    sess: &mut Glm5SpecSession,
    prompt: &[u32],
    k: usize,
    total: usize,
    burst_target: usize,
    eos: &[u32],
) -> (Vec<u32>, usize, usize, usize) {
    let mut tape: Vec<u32> = Vec::new();
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    let mut bursts = 0usize;
    while tape.len() < total && !sess.finished() {
        let room = (total - tape.len()).min(burst_target);
        let (burst, d, a) = h
            .model
            .glm5_spec_session_burst(&h.engine, sess, room, k, eos)
            .expect("glm5 dflash spec session burst");
        if burst.is_empty() {
            break;
        }
        bursts += 1;
        drafted += d;
        accepted += a;
        tape.extend(burst);
        assert_eq!(
            sess.pos(),
            sess.committed.len(),
            "cache rows != committed tokens at a burst boundary"
        );
        let mut expect: Vec<u32> = prompt.to_vec();
        expect.extend_from_slice(&tape[..tape.len() - 1]);
        assert_eq!(
            sess.committed, expect,
            "committed must be prompt + served tape minus the live anchor"
        );
    }
    (tape, drafted, accepted, bursts)
}

// ---------------------------------------------------------------------------------------------
// Gate 1 — served-burst greedy byte identity, K=1..7, MTP head not loaded.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dflash_served_bursts_greedy_tape_matches_plain_decode_k1_to_7() {
    let _gpu = gpu_guard();
    let h = Harness::new("g1");
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let tape = plain_tape(&h, &prompt, max_new);

    for k in 1..=K {
        let mut sess = h
            .model
            .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + max_new + k + 8, None)
            .expect("glm5 dflash spec session");
        let (out, drafted, accepted, bursts) =
            drive_bursts(&h, &mut sess, &prompt, k, max_new, 3, &[]);
        assert_eq!(
            &out[..max_new],
            &tape[..],
            "K={k}: dflash served-burst tape diverged from plain greedy \
             ({accepted}/{drafted} over {bursts} bursts) — the draft source may only move \
             acceptance, never output"
        );
        assert!(
            bursts >= max_new / (k + 2),
            "K={k}: the drive never actually split into bursts ({bursts})"
        );
        println!(
            "gate 1 PASS K={k}: dflash served bursts byte-identical over {bursts} bursts, \
             {accepted}/{drafted} accepted"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Gate 2 — forced-rejection j-sweep through the served burst (every partial-accept j).
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dflash_forced_rejection_partial_accepts_stay_byte_identical() {
    let _gpu = gpu_guard();
    let h = Harness::new("g2");
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let k = K;
    // The reference tape covers the FINAL round's overshoot too (a round may commit up to
    // k+1 tokens past the budget): the schedule-exact teeth below need every override to
    // see the true continuation, or the last round's deep accepts silently miss (found by
    // the teeth themselves: 14/15 with a 20-token tape).
    let tape = plain_tape(&h, &prompt, max_new + k + 2);

    let tape_for_override = tape.clone();
    let committed_before = move |round: usize| -> usize {
        let mut c = 1usize;
        for r in 0..round {
            c += (r % k) + 1;
        }
        c
    };
    let mut over = move |round: usize, ki: usize, _drafted: u32| -> u32 {
        let j_target = round % k;
        let cursor = committed_before(round);
        let pos = cursor + ki;
        let correct = if pos < tape_for_override.len() {
            tape_for_override[pos]
        } else {
            0
        };
        if ki < j_target {
            correct
        } else {
            (correct + 1) % VOCAB
        }
    };
    let mut knobs = Glm5SpecKnobs {
        draft_override: Some(&mut over),
        ..Default::default()
    };
    let mut sess = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + max_new + k + 8, None)
        .expect("glm5 dflash spec session");
    let mut out: Vec<u32> = Vec::new();
    let mut bursts = 0usize;
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    while out.len() < max_new && !sess.finished() {
        let room = (max_new - out.len()).min(2);
        let (burst, d, a) = h
            .model
            .glm5_spec_session_burst_gated(&h.engine, &mut sess, room, k, &[], &mut knobs)
            .expect("forced-rejection served burst");
        if burst.is_empty() {
            break;
        }
        bursts += 1;
        drafted += d;
        accepted += a;
        out.extend(burst);
        assert_eq!(sess.pos(), sess.committed.len());
    }
    assert_eq!(
        &out[..max_new],
        &tape[..max_new],
        "forced-rejection dflash bursts diverged from plain greedy"
    );
    // NON-VACUITY TEETH (wiring-assertions law: byte identity alone would also pass with
    // every accept silently dead — bonus tokens carry the tape). The override's schedule
    // is deterministic: round r accepts exactly r % k drafts, so the counters must match
    // the replayed schedule token for token — this is the proof the dflash draft->verify
    // position mapping ACCEPTS correct drafts, not merely rejects everything gracefully.
    let (mut exp_drafted, mut exp_accepted, mut committed, mut round) = (0usize, 0, 0, 0);
    while committed < max_new {
        let j = round % k;
        exp_drafted += k;
        exp_accepted += j;
        committed += j + 1;
        round += 1;
    }
    assert_eq!(
        (drafted, accepted),
        (exp_drafted, exp_accepted),
        "the j-sweep must accept exactly j drafts per round (a vacuous sweep means the \
         dflash accept mapping is dead)"
    );
    assert!(accepted > 0, "the sweep never exercised a real accept");
    println!(
        "gate 2 PASS: forced-rejection j-sweep byte-identical over {bursts} bursts, \
         schedule-exact {accepted}/{drafted} accepted"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 3 — RED, tap-shift: wrong feature input changes the DRAFT STREAM (features are
// live) and never improves acceptance, while the OUTPUT TAPE stays byte-identical. The
// acceptance-collapse MAGNITUDE (probe band 0.73 acc@1) is a real-artifact number and
// lands in the box three-way window; the mini fixture pins the mechanism.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dflash_tap_shift_red_arm_moves_drafts_never_the_tape() {
    let _gpu = gpu_guard();
    let h = Harness::new("g3");
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let tape = plain_tape(&h, &prompt, max_new);
    let k = K;

    // A recording override: returns every draft unchanged, banking the stream.
    let run = |red: bool| -> (Vec<u32>, Vec<u32>, usize, usize) {
        // SAFETY: serialized behind gpu_guard (held by this test).
        unsafe {
            if red {
                std::env::set_var("MEMRA_GLM5_DFLASH_GATE_RED", "tap-shift");
            } else {
                std::env::remove_var("MEMRA_GLM5_DFLASH_GATE_RED");
            }
        }
        let mut drafts: Vec<u32> = Vec::new();
        let mut out: Vec<u32> = Vec::new();
        let mut drafted = 0usize;
        let mut accepted = 0usize;
        {
            // Scope: `knobs` borrows `drafts` through the recorder; the block returns it.
            let mut rec = |_round: usize, _ki: usize, d: u32| -> u32 {
                drafts.push(d);
                d
            };
            let mut knobs = Glm5SpecKnobs {
                draft_override: Some(&mut rec),
                ..Default::default()
            };
            let mut sess = h
                .model
                .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + max_new + k + 8, None)
                .expect("glm5 dflash spec session");
            while out.len() < max_new && !sess.finished() {
                let (burst, d, a) = h
                    .model
                    .glm5_spec_session_burst_gated(&h.engine, &mut sess, 3, k, &[], &mut knobs)
                    .expect("burst");
                if burst.is_empty() {
                    break;
                }
                drafted += d;
                accepted += a;
                out.extend(burst);
            }
        }
        // SAFETY: as above.
        unsafe { std::env::remove_var("MEMRA_GLM5_DFLASH_GATE_RED") };
        (out, drafts, drafted, accepted)
    };

    let (out_green, drafts_green, d_g, a_g) = run(false);
    let (out_red, drafts_red, d_r, a_r) = run(true);
    assert_eq!(
        &out_green[..max_new],
        &tape[..],
        "green arm tape must match plain greedy"
    );
    assert_eq!(
        &out_red[..max_new],
        &tape[..],
        "RED ARM TAPE DIVERGED: wrong drafter features must be invisible in the output — \
         only acceptance may move (the exactness seam is verify, not the drafter)"
    );
    assert_ne!(
        drafts_green, drafts_red,
        "tap-shift did not change the draft stream — the feature seam is DEAD (the drafter \
         is not consuming the tapped trunk features)"
    );
    let acc_g = a_g as f64 / d_g.max(1) as f64;
    let acc_r = a_r as f64 / d_r.max(1) as f64;
    assert!(
        acc_r <= acc_g + 1e-9,
        "wrong features IMPROVED acceptance ({acc_r:.3} > {acc_g:.3}) — the tap layers are \
         mislabeled"
    );
    println!(
        "gate 3 PASS: tape byte-identical both arms; drafts diverged; acceptance green \
         {a_g}/{d_g}={acc_g:.3} vs red {a_r}/{d_r}={acc_r:.3} (collapse magnitude lands on \
         the box with the real artifact — probe band 0.73 acc@1)"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 4 — RED: rollback disabled + forced rejections through the dflash session.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dflash_rollback_disabled_bites() {
    let _gpu = gpu_guard();
    let h = Harness::new("g4");
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let tape = plain_tape(&h, &prompt, max_new);

    let mut over = |_round: usize, ki: usize, drafted: u32| -> u32 {
        if ki == 0 {
            (drafted + 1) % VOCAB
        } else {
            drafted
        }
    };
    let mut knobs = Glm5SpecKnobs {
        draft_override: Some(&mut over),
        disable_rollback: true,
        ..Default::default()
    };
    let mut sess = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + max_new + K + 8, None)
        .expect("glm5 dflash spec session");
    let mut out: Vec<u32> = Vec::new();
    let mut failed: Option<String> = None;
    while out.len() < max_new && !sess.finished() {
        match h
            .model
            .glm5_spec_session_burst_gated(&h.engine, &mut sess, 3, K, &[], &mut knobs)
        {
            Ok((burst, _d, _a)) => {
                if burst.is_empty() {
                    break;
                }
                out.extend(burst);
            }
            Err(err) => {
                failed = Some(err.to_string());
                break;
            }
        }
    }
    match failed {
        Some(err) => println!("gate 4 RED bites: dflash burst failed loudly: {err}"),
        None => {
            assert_ne!(
                &out[..max_new.min(out.len())],
                &tape[..max_new.min(out.len())],
                "rollback disabled + forced rejections still produced the plain tape — the \
                 red arm went blind on the dflash source"
            );
            println!("gate 4 RED bites: dflash tape diverged with rollback disabled");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Gate 5 — sampled twin: determinism, burst-split invariance, seed sensitivity.
// ---------------------------------------------------------------------------------------------

fn sampled_cfg(seed: u64) -> SpecSampling {
    SpecSampling {
        temp: 0.9,
        seed,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        penalty_last_n: 0,
        penalty_repeat: 1.0,
        penalty_freq: 0.0,
        penalty_present: 0.0,
    }
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dflash_sampled_twin_is_deterministic_and_burst_split_invariant() {
    let _gpu = gpu_guard();
    let h = Harness::new("g5");
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 24usize;
    let k = 3usize;
    let ctx = prompt.len() + max_new + k + 8;

    let run = |seed: u64, burst_target: usize| -> Vec<u32> {
        let mut sess = h
            .model
            .glm5_spec_session_new(&h.engine, &prompt, ctx, Some(sampled_cfg(seed)))
            .expect("sampled glm5 dflash spec session");
        let (tape, _d, _a, _b) =
            drive_bursts(&h, &mut sess, &prompt, k, max_new, burst_target, &[]);
        tape[..max_new.min(tape.len())].to_vec()
    };

    let a = run(42, 3);
    let b = run(42, 3);
    assert_eq!(a, b, "same seed, same burst split: reproducible");
    let c = run(42, max_new);
    assert_eq!(
        a, c,
        "burst-split invariance: the selector's draws and accept uniforms ride the \
         session's Philox counters, so the split must not change the stream"
    );
    let d = run(43, 3);
    assert_ne!(a, d, "a different seed must change the sampled tape");
    println!("gate 5 PASS: dflash sampled twin deterministic, split-invariant, seed-sensitive");
}

// ---------------------------------------------------------------------------------------------
// Gate 6 — EOS mid-burst finishes the session; later bursts are empty.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dflash_eos_finishes_the_session() {
    let _gpu = gpu_guard();
    let h = Harness::new("g6");
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let tape = plain_tape(&h, &prompt, max_new);
    let eos = [tape[6]];
    let first_eos = tape.iter().position(|t| eos.contains(t)).unwrap();

    let k = 3usize;
    let mut sess = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + max_new + k + 8, None)
        .expect("glm5 dflash spec session");
    let mut out: Vec<u32> = Vec::new();
    while out.len() < max_new && !sess.finished() {
        let (burst, _d, _a) = h
            .model
            .glm5_spec_session_burst(&h.engine, &mut sess, 3, k, &eos)
            .expect("burst");
        if burst.is_empty() {
            break;
        }
        out.extend(burst);
    }
    assert!(sess.finished(), "EOS must finish the session");
    let cut = out
        .iter()
        .position(|t| eos.contains(t))
        .expect("EOS emitted");
    assert_eq!(
        &out[..=cut],
        &tape[..=first_eos],
        "the public prefix through EOS must match plain greedy"
    );
    let (again, d2, a2) = h
        .model
        .glm5_spec_session_burst(&h.engine, &mut sess, 8, k, &eos)
        .expect("post-EOS burst");
    assert!(
        again.is_empty() && d2 == 0 && a2 == 0,
        "a finished session must emit nothing"
    );
    println!("gate 6 PASS: EOS at pos {first_eos} finished the dflash session");
}

// ---------------------------------------------------------------------------------------------
// Gate 7 — K bound: the drafter blocks 8 tokens; K=8 must refuse LOUDLY at the burst.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dflash_k_past_the_block_refuses_loudly() {
    let _gpu = gpu_guard();
    let h = Harness::new("g7");
    let prompt = tokens(PROMPT, 0xA11CE);
    let mut sess = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + 40, None)
        .expect("glm5 dflash spec session");
    let err = h
        .model
        .glm5_spec_session_burst(&h.engine, &mut sess, 8, BLOCK, &[])
        .expect_err("K == block_size must refuse");
    assert!(
        err.to_string().contains("DFlash2 drafter's block"),
        "refusal must name the drafter block bound, got: {err}"
    );
    println!("gate 7 PASS: K={BLOCK} refused loudly ({err})");
}

// ---------------------------------------------------------------------------------------------
// Gate 8 — DRAFT-SOURCE SELECTION MATRIX (subprocess-captured boot logs, red-proven) +
// the VRAM-at-ready print for the lane doc.
// ---------------------------------------------------------------------------------------------

/// Child body: load per ambient env, print VRAM at ready, run one burst when armed.
#[test]
#[ignore = "receipt-gate child body; spawned by gpu_draft_source_selection_matrix"]
fn helper_emit_dflash_receipts() {
    let _gpu = gpu_guard();
    force_true_f32();
    let with_mtp = std::env::var("MEMRA_GLM5_MTP").as_deref() == Ok("1");
    let config = mini_config();
    let plan = mini_plan(&config);
    let source = fixture_source(&config, &plan);
    let engine = Engine::new(0).expect("CUDA engine on device 0");
    let model = HybridModel::load_from_source(&engine, &source).expect("mini glm5 loads per env");
    let (free, total) = engine.ctx().mem_get_info().expect("mem_get_info");
    eprintln!(
        "[helper] vram-used-at-ready-mib={} mtp={} dflash={}",
        (total - free) >> 20,
        with_mtp,
        model.glm5_dflash.is_some()
    );
    if memra_engine::glm_spec::glm5_spec_on()
        && (model.mtp.is_some() || model.glm5_dflash.is_some())
    {
        let prompt = tokens(PROMPT, 0xA11CE);
        let mut sess = model
            .glm5_spec_session_new(&engine, &prompt, prompt.len() + 40, None)
            .expect("glm5 spec session");
        let (burst, d, a) = model
            .glm5_spec_session_burst(&engine, &mut sess, 8, 3, &[])
            .expect("burst");
        eprintln!("[helper] burst={} drafted={d} accepted={a}", burst.len());
    }
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_draft_source_selection_matrix() {
    let _gpu = gpu_guard();
    let drafter_dir = write_mini_drafter("matrix");
    let run_child =
        |glm5_spec: Option<&str>, glm5_mtp: Option<&str>, dflash: Option<&Path>| -> String {
            let exe = std::env::current_exe().expect("test binary path");
            let mut cmd = std::process::Command::new(exe);
            cmd.args([
                "helper_emit_dflash_receipts",
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ]);
            cmd.env_remove("MEMRA_GLM5_SPEC");
            cmd.env_remove("MEMRA_GLM5_MTP");
            cmd.env_remove("MEMRA_GLM5_DFLASH");
            cmd.env_remove("MEMRA_GLM5_DFLASH_GATE_RED");
            cmd.env_remove("MEMRA_FRSPEC_TRIM");
            cmd.env("NVIDIA_TF32_OVERRIDE", "0");
            if let Some(v) = glm5_spec {
                cmd.env("MEMRA_GLM5_SPEC", v);
            }
            if let Some(v) = glm5_mtp {
                cmd.env("MEMRA_GLM5_MTP", v);
            }
            if let Some(dir) = dflash {
                cmd.env("MEMRA_GLM5_DFLASH", dir);
            }
            let out = cmd.output().expect("spawn receipt child");
            assert!(
                out.status.success(),
                "receipt child failed (spec={glm5_spec:?} mtp={glm5_mtp:?} dflash={}):\n{}",
                dflash.is_some(),
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stderr).into_owned()
        };
    let vram_mib = |log: &str| -> u64 {
        log.lines()
            .find_map(|l| l.strip_prefix("[helper] vram-used-at-ready-mib="))
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|v| v.parse().ok())
            .expect("vram line")
    };

    // ARM A: dflash2 source, MTP head NOT loaded — the q38 pattern; source line + burst.
    let log = run_child(Some("1"), None, Some(&drafter_dir));
    assert!(
        log.contains("draft source = dflash2 @ "),
        "dflash2 selection receipt missing:\n{log}"
    );
    assert!(
        log.contains("native MTP head NOT loaded"),
        "the head-not-loaded note is the VRAM receipt:\n{log}"
    );
    assert!(
        log.contains("[helper] burst="),
        "dflash child never burst:\n{log}"
    );
    let vram_dflash = vram_mib(&log);

    // ARM B: both loaded — dflash2 wins by selection, stated.
    let log = run_child(Some("1"), Some("1"), Some(&drafter_dir));
    assert!(
        log.contains("draft source = dflash2 @ ") && log.contains("ALSO loaded"),
        "both-armed selection must state dflash2 wins:\n{log}"
    );

    // ARM C: native only — the existing receipts plus the source line.
    let log = run_child(Some("1"), Some("1"), None);
    assert!(
        log.contains("[glm5-spec] draft source = native-mtp"),
        "native-mtp selection receipt missing:\n{log}"
    );
    let vram_mtp = vram_mib(&log);

    // ARM D: neither — fail-closed warn, no route.
    let log = run_child(Some("1"), None, None);
    assert!(
        log.contains("[glm5-spec] MEMRA_GLM5_SPEC=1 but no MTP head loaded"),
        "fail-closed warn missing:\n{log}"
    );

    // RED: MEMRA_GLM5_SPEC off must print ZERO [glm5-spec] lines, drafter loaded or not.
    for dfl in [None, Some(drafter_dir.as_path())] {
        let log = run_child(None, None, dfl);
        assert!(
            !log.contains("[glm5-spec]"),
            "MEMRA_GLM5_SPEC off (dflash={}) must print no [glm5-spec] line:\n{log}",
            dfl.is_some()
        );
    }
    std::fs::remove_dir_all(&drafter_dir).ok(); // tmp hygiene
    println!(
        "gate 8 PASS: selection matrix green+red; VRAM-at-ready mini-fixture: dflash-boot \
         {vram_dflash} MiB vs mtp-boot {vram_mtp} MiB (delta {} MiB — mini scale; the box \
         three-way window banks the real-artifact delta)",
        vram_dflash as i64 - vram_mtp as i64
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 10 — CONFIDENCE GATE, DFlash2 arm (lane/glm5-loop-port, port 2): tau-slot truncation
// over the selector's recorded q moves DRAFT COUNTS, never the tape. Decisive arms as in the
// native gate: p_min = 1.1 is never cleared (q is a softmax mass <= 1.0), so PMIN0 forces
// zero-draft rounds (plain steps) and !PMIN0 forces the slot-0 survivor. The greedy walk's
// q is its new T=1 recorded confidence (`dflash2_walk_greedy_q`); the sampled walk's is the
// `q_chosen` it always recorded.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dflash_confidence_gate_truncates_drafts_never_the_tape() {
    let _gpu = gpu_guard();
    let h = Harness::new("g10");
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let k = 3usize;
    let ctx = prompt.len() + max_new + k + 8;
    let tape = plain_tape(&h, &prompt, max_new);

    let drive =
        |pmin: Option<(f32, bool)>, sampling: Option<SpecSampling>| -> (Vec<u32>, usize, usize) {
            let mut sess = h
                .model
                .glm5_spec_session_new(&h.engine, &prompt, ctx, sampling)
                .expect("glm5 dflash spec session");
            let mut knobs = Glm5SpecKnobs {
                pmin_override: pmin,
                ..Default::default()
            };
            let mut out: Vec<u32> = Vec::new();
            let (mut drafted, mut accepted) = (0usize, 0usize);
            while out.len() < max_new && !sess.finished() {
                let room = (max_new - out.len()).min(3);
                let (burst, d, a) = h
                    .model
                    .glm5_spec_session_burst_gated(&h.engine, &mut sess, room, k, &[], &mut knobs)
                    .expect("gated burst");
                if burst.is_empty() {
                    break;
                }
                drafted += d;
                accepted += a;
                out.extend(burst);
            }
            (out, drafted, accepted)
        };

    let (out_off, drafted_off, _) = drive(None, None);
    assert_eq!(&out_off[..max_new], &tape[..], "gate-off arm diverged");
    assert!(
        drafted_off > 0,
        "gate-off arm drafted nothing — fixture defect"
    );

    let (out, drafted, accepted) = drive(Some((1.1, true)), None);
    assert_eq!(
        &out[..max_new],
        &tape[..],
        "PMIN0 zero-draft rounds must stay byte-identical (each round IS a plain step)"
    );
    assert_eq!(
        (drafted, accepted),
        (0, 0),
        "p_min=1.1 + PMIN0 must truncate EVERY proposal to zero drafts"
    );

    let (out, drafted, _) = drive(Some((1.1, false)), None);
    assert_eq!(&out[..max_new], &tape[..], "slot-0 survivor arm diverged");
    assert!(
        drafted > 0 && drafted < drafted_off,
        "without PMIN0 exactly slot 0 rides per round: got {drafted} vs gate-off {drafted_off}"
    );

    // Sampled zero-draft twin over the Selector q side: the shared bonus draw must be
    // deterministic on a pinned seed and the session must not stall.
    let (sa, da, _) = drive(Some((1.1, true)), Some(sampled_cfg(42)));
    let (sb, db, _) = drive(Some((1.1, true)), Some(sampled_cfg(42)));
    assert_eq!(sa, sb, "sampled zero-draft rounds must be deterministic");
    assert_eq!(
        (da, db),
        (0, 0),
        "sampled arms must also draft zero at p_min=1.1"
    );
    assert!(
        sa.len() >= max_new,
        "sampled zero-draft session stalled at {} of {max_new}",
        sa.len()
    );

    println!(
        "gate 10 PASS: dflash tau-slot gate truncates drafts (0 with PMIN0, {drafted} \
         slot-0 survivors vs {drafted_off} gate-off), tape byte-identical on every arm"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 11 — RESTORED SESSION (lane/glm5-prefix-latent2, 2026-09-01): a prefix-hit re-arm
// (`glm5_spec_session_from_restored`: boundary trunk cache + drafter KV from the published
// tail + suffix primed through the continuation program with origin-anchored taps) must
// produce the PLAIN continuation's bytes, and its drafter must see a byte-equivalent
// context to a cold session's — a tap-origin or tail-cut bug moves acceptance, never
// output (verify arbitrates), so the acceptance-equality arm is the sensitive half.
//
// SCOPE NOTE: the boundary cache here is direct-primed over the prefix; entry-plane
// restore fidelity into a fresh cache is gate 16/18 of glm5_kpool_indexer_gpu plus the
// parent lane's box C1 (byte-identical with engagement, cross-box). This gate composes on
// that proven boundary and pins the NEW session mechanics.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_restored_session_bytes_match_plain_decode_and_cold_acceptance() {
    let _gpu = gpu_guard();
    let h = Harness::new("g11");
    let prompt = tokens(PROMPT, 0xA11CE);
    let split = PROMPT - BLOCK; // suffix = one drafter block, prefix past the kpool budget
    let (prefix, suffix) = prompt.split_at(split);
    let max_new = 20usize;
    let k = 3usize;
    let ctx = prompt.len() + max_new + K + 8;

    // The byte reference: plain decode over the full prompt.
    let tape_plain = plain_tape(&h, &prompt, max_new);

    // The acceptance reference: a COLD spec session over the full prompt.
    let mut cold = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, ctx, None)
        .expect("cold spec session");
    let (tape_cold, drafted_cold, accepted_cold, _) =
        drive_bursts(&h, &mut cold, &prompt, k, max_new, 7, &[]);
    assert_eq!(
        tape_cold,
        tape_plain[..tape_cold.len()],
        "cold spec tape must match plain decode (gate 1's bar, re-anchored here)"
    );

    // DONOR: a cold spec session over the PREFIX; its first burst ingests the prefix's
    // feature rows into the drafter KV, then the tail is exported AT THE BOUNDARY —
    // exactly the publication the worker's drain performs (`export_draft_tail(end)`).
    let mut donor = h
        .model
        .glm5_spec_session_new(&h.engine, prefix, ctx, None)
        .expect("donor spec session over the prefix");
    let (_donor_burst, donor_drafted, _, _) = drive_bursts(&h, &mut donor, prefix, k, 4, 4, &[]);
    assert!(donor_drafted > 0, "the donor must have drafted (kv filled)");
    let tail = donor
        .export_draft_tail(&h.engine, prefix.len())
        .expect("drafter tail export at the boundary");
    let dr = h.model.glm5_dflash.as_ref().expect("drafter attached");
    let dkv = memra_engine::dflash::DflashKv::from_tail(&h.engine, &dr.draft.cfg, ctx, &tail)
        .expect("drafter KV rebuilt from the tail");
    drop(donor);

    // The restored trunk cache: primed to exactly the boundary (scope note above).
    let (boundary_cache, boundary_logits) = h.fresh_primed(prefix, ctx);

    // RED: shape refusals fire before any prime: empty suffix with the full-cover arm
    // DISARMED (memra#74: `MEMRA_GLM5_SPEC_FULLCOVER` unset is the shipped default and must
    // keep the pre-lane refusal verbatim), and pos/fed disagreement.
    {
        assert!(
            std::env::var("MEMRA_GLM5_SPEC_FULLCOVER").is_err(),
            "gate 11 pins the DISARMED posture; run gate 13 for the armed one"
        );
        let (c2, _) = h.fresh_primed(prefix, ctx);
        let dkv_red =
            memra_engine::dflash::DflashKv::from_tail(&h.engine, &dr.draft.cfg, ctx, &tail)
                .expect("red-arm dkv");
        assert!(
            h.model
                .glm5_spec_session_from_restored(
                    &h.engine,
                    c2,
                    prefix,
                    &[],
                    &boundary_logits,
                    dkv_red,
                    ctx,
                    None,
                )
                .is_err(),
            "RED: an empty suffix must refuse while MEMRA_GLM5_SPEC_FULLCOVER is unset"
        );
        let (c3, _) = h.fresh_primed(prefix, ctx);
        let dkv_red2 =
            memra_engine::dflash::DflashKv::from_tail(&h.engine, &dr.draft.cfg, ctx, &tail)
                .expect("red-arm dkv 2");
        assert!(
            h.model
                .glm5_spec_session_from_restored(
                    &h.engine,
                    c3,
                    &prefix[..prefix.len() - 1],
                    suffix,
                    &boundary_logits,
                    dkv_red2,
                    ctx,
                    None,
                )
                .is_err(),
            "RED: cache.pos != restored prefix must refuse"
        );
    }

    // The restored session: suffix primes onto the boundary cache, drafter re-arms from
    // the tail, and the served bytes must be the plain continuation's.
    let mut restored = h
        .model
        .glm5_spec_session_from_restored(
            &h.engine,
            boundary_cache,
            prefix,
            suffix,
            &boundary_logits,
            dkv,
            ctx,
            None,
        )
        .expect("restored spec session");
    let (tape, drafted, accepted, _bursts) =
        drive_bursts(&h, &mut restored, &prompt, k, max_new, 7, &[]);
    assert!(drafted > 0, "the restored session must actually draft");
    assert_eq!(
        tape,
        tape_plain[..tape.len()],
        "restored spec tape must be BYTE-IDENTICAL to plain decode"
    );
    assert_eq!(
        tape.len(),
        tape_cold.len(),
        "restored and cold sessions must serve the same tape length"
    );
    // The sensitive half: a byte-equivalent drafter context (prefix features via the tail,
    // suffix features via the origin-anchored taps) drafts the same greedy rounds — any
    // tap-origin shift or wrong tail cut moves these counts while the tape stays green.
    assert_eq!(
        (drafted, accepted),
        (drafted_cold, accepted_cold),
        "restored-session drafter context must be byte-equivalent to the cold session's"
    );
    println!(
        "gate 11 PASS: restored session == plain bytes over {} tokens, acceptance {} / {} \
         identical to cold",
        tape.len(),
        accepted,
        drafted,
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 12 — C1b ATTRIBUTION (memra#34): the restored-then-continue program is bit-identical to
// the SPLIT cold prime at this scale, so a serving-battery delta against the MONOLITHIC cold
// prime ATTRIBUTES to the documented chunked-prime numeric class (`hyper_prime_ranges`' "NOT
// BIT-STABLE ACROSS CHUNK SIZES" note: cuBLASLt reselects by shape, worst 3.815e-6 on the mixes
// GEMM) ONCE the battery's split-twin oracle confirms at prod scale — the prod half lives in
// the battery (a `MEMRA_PRIME_CHUNK=<boundary>` cold boot matching the restored bytes), not
// here. SCOPE: a scale-dependent restore defect would not reproduce at PROMPT=24/BLOCK=8. Born from the 2026-09-01 slot-B C1b red: a strict-prefix spec
// restore (269 of 448) flipped '\n' vs '\n\n' at verify-round token 1 while full-cover restores
// stayed byte-exact — the anchor token itself matched, so the boundary state was right and the
// suspect is the differently-shaped prime over the suffix rows.
//
// Arms, all greedy:
//   A (mono):  one prime call over the whole prompt, then plain decode — the battery's cold ref.
//   B (split): prefix prime, then suffix prime as a CONTINUATION on the same cache, then plain
//              decode. No cache entry, no restore machinery: the pure chunking control.
//   C (spec-restored): gate 11's construction (donor session over the prefix, drafter tail
//              export at the boundary, fresh boundary cache) driven through worker-shaped
//              bursts — the path the serving battery exercised.
//
// THE INVARIANT (hard assert): C's tape == B's tape, byte for byte, and B's continuation logits
// carry NO drift the restore could hide behind. A vs B is REPORTED with bit counts and max
// delta, and held to the chunked-prime band rather than asserted equal — bit equality across
// chunk shapes is the documented non-goal.
#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_restored_continuation_is_bit_identical_to_the_split_prime_cold_twin() {
    let _gpu = gpu_guard();
    let h = Harness::new("g12");
    let prompt = tokens(PROMPT, 0xC1B7);
    let split = PROMPT - BLOCK;
    let (prefix, suffix) = prompt.split_at(split);
    let max_new = 20usize;
    let k = 3usize;
    let ctx = prompt.len() + max_new + K + 8;

    // ARM A — monolithic cold prime + plain decode (the battery's cold reference shape).
    let (mut cache_mono, logits_mono) = h.fresh_primed(&prompt, ctx);
    let mut tape_mono = Vec::with_capacity(max_new);
    tape_mono.push(argmax(&logits_mono) as u32);
    while tape_mono.len() < max_new {
        let ll = h
            .model
            .decode_step(&h.engine, *tape_mono.last().unwrap(), &mut cache_mono)
            .expect("mono decode step");
        tape_mono.push(argmax(&ll) as u32);
    }

    // ARM B — split cold prime (prefix, then suffix on the same cache) + plain decode.
    let (mut cache_split, _boundary_logits) = h.fresh_primed(prefix, ctx);
    let (logits_split, _seed, _hiddens) = h
        .model
        .prime_cache(&h.engine, suffix, &mut cache_split, 0)
        .expect("suffix continuation prime");
    let mut tape_split = Vec::with_capacity(max_new);
    tape_split.push(argmax(&logits_split) as u32);
    while tape_split.len() < max_new {
        let ll = h
            .model
            .decode_step(&h.engine, *tape_split.last().unwrap(), &mut cache_split)
            .expect("split decode step");
        tape_split.push(argmax(&ll) as u32);
    }

    // ATTRIBUTION REPORT — A vs B logits at the anchor position: the chunking class, measured.
    assert_eq!(logits_mono.len(), logits_split.len(), "logit widths match");
    assert!(
        logits_mono
            .iter()
            .chain(logits_split.iter())
            .all(|v| v.is_finite()),
        "non-finite anchor logits — the band below cannot see NaN (f32::max drops it)"
    );
    let mut diff_bits = 0usize;
    let mut max_delta = 0f32;
    for (a, b) in logits_mono.iter().zip(logits_split.iter()) {
        if a.to_bits() != b.to_bits() {
            diff_bits += 1;
            max_delta = max_delta.max((a - b).abs());
        }
    }
    // The chunked-prime band (prime_chunk_ranges doc: worst 3.815e-6 at prod scale). A toy
    // trunk may read exactly zero; a REAL drift class lands well under this bar; a restore
    // defect masquerading as chunking would not.
    assert!(
        max_delta <= 1e-3,
        "mono-vs-split anchor logits moved {max_delta:.3e} — beyond the chunked-prime class"
    );

    // ARM C — the spec restore over the same boundary (gate 11's construction, worker-shaped).
    let mut donor = h
        .model
        .glm5_spec_session_new(&h.engine, prefix, ctx, None)
        .expect("donor spec session over the prefix");
    let (_burst, donor_drafted, _, _) = drive_bursts(&h, &mut donor, prefix, k, 4, 4, &[]);
    assert!(donor_drafted > 0, "the donor must have drafted (kv filled)");
    let tail = donor
        .export_draft_tail(&h.engine, prefix.len())
        .expect("drafter tail export at the boundary");
    let dr = h.model.glm5_dflash.as_ref().expect("drafter attached");
    let dkv = memra_engine::dflash::DflashKv::from_tail(&h.engine, &dr.draft.cfg, ctx, &tail)
        .expect("drafter KV rebuilt from the tail");
    drop(donor);
    let (boundary_cache, bl) = h.fresh_primed(prefix, ctx);
    let mut restored = h
        .model
        .glm5_spec_session_from_restored(
            &h.engine,
            boundary_cache,
            prefix,
            suffix,
            &bl,
            dkv,
            ctx,
            None,
        )
        .expect("restored spec session");
    let (tape_spec, drafted, _accepted, _bursts) =
        drive_bursts(&h, &mut restored, &prompt, k, max_new, 7, &[]);
    assert!(drafted > 0, "the restored session must actually draft");

    // THE INVARIANT: the restored continuation serves the SPLIT twin's bytes exactly. If this
    // ever reds, the divergence is a restoration defect and memra#34 reopens as engine work;
    // while it holds, a battery C1b red against a MONOLITHIC cold ref attributes to the
    // documented chunking class and the battery's oracle must use the split twin.
    assert_eq!(
        tape_spec,
        tape_split[..tape_spec.len()],
        "restored spec tape must be BYTE-IDENTICAL to the split-prime cold twin"
    );
    println!(
        "gate 12 PASS: restored == split twin over {} tokens; mono-vs-split anchor logits: \
         {} of {} values differ, max |delta| {:.3e}; mono tape == split tape: {}",
        tape_spec.len(),
        diff_bits,
        logits_mono.len(),
        max_delta,
        tape_mono == tape_split,
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 13, FULL-COVER RESTORE (memra#74, lane/glm5-fullcover-spec-route 2026-09-02): a prefix
// hit that covers the WHOLE prompt leaves no suffix to prime. The parent lane refused that
// shape and served it PLAIN, which on the live glm5 box cost the customer half the decode
// speed on every repeated prompt (30.8 vs 69.7 tok/s decode, same minute and vantage,
// 2026-09-02). `MEMRA_GLM5_SPEC_FULLCOVER=1` admits it: trunk cache at the boundary, drafter
// ctx KV rebuilt from the published tail, NO prime, NO pending tap rows, and the anchor drawn
// from the ENTRY's boundary logits by the cold burst's own rule.
//
// THE BAR, same as gate 11: the served tape must be byte-identical to plain decode, and the
// drafter must see a byte-equivalent context to a cold session's, so acceptance must match
// the cold session's exactly. Acceptance is the sensitive half: a wrong tail cut or a missing
// ingest moves those counts while the tape stays green.
//
// This is the gate the deploy lane runs before the flag may flip on a serving box.
// ---------------------------------------------------------------------------------------------

/// Sets `MEMRA_GLM5_SPEC_FULLCOVER=1` for the life of the value and removes it on drop, so a
/// panicking gate cannot leave the arm armed for gate 11's disarmed-posture assertion. Safe
/// because every gate in this binary holds `gpu_guard()` while it runs.
struct FullCoverArm;

impl FullCoverArm {
    fn arm() -> Self {
        // SAFETY: the caller holds gpu_guard(), which serializes every test in this binary.
        unsafe { std::env::set_var("MEMRA_GLM5_SPEC_FULLCOVER", "1") };
        Self
    }
}

impl Drop for FullCoverArm {
    fn drop(&mut self) {
        // SAFETY: as above, still under gpu_guard().
        unsafe { std::env::remove_var("MEMRA_GLM5_SPEC_FULLCOVER") };
    }
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_full_cover_restore_bytes_match_plain_decode_and_cold_acceptance() {
    let _gpu = gpu_guard();
    let h = Harness::new("g13");
    let prompt = tokens(PROMPT, 0xF00C);
    let max_new = 20usize;
    let k = 3usize;
    let ctx = prompt.len() + max_new + K + 8;

    // The byte reference: plain decode over the whole prompt.
    let tape_plain = plain_tape(&h, &prompt, max_new);

    // The acceptance reference: a COLD spec session over the same prompt.
    let mut cold = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, ctx, None)
        .expect("cold spec session");
    let (tape_cold, drafted_cold, accepted_cold, _) =
        drive_bursts(&h, &mut cold, &prompt, k, max_new, 7, &[]);
    assert_eq!(
        tape_cold,
        tape_plain[..tape_cold.len()],
        "cold spec tape must match plain decode"
    );

    // DONOR: a cold spec session over the WHOLE prompt whose drafter tail is exported AT the
    // prompt boundary: the publication a full-cover entry carries.
    let mut donor = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, ctx, None)
        .expect("donor spec session over the whole prompt");
    let (_donor_burst, donor_drafted, _, _) = drive_bursts(&h, &mut donor, &prompt, k, 4, 4, &[]);
    assert!(donor_drafted > 0, "the donor must have drafted (kv filled)");
    let tail = donor
        .export_draft_tail(&h.engine, prompt.len())
        .expect("drafter tail export at the prompt boundary");
    let dr = h.model.glm5_dflash.as_ref().expect("drafter attached");
    drop(donor);

    // The restored trunk cache + the entry's boundary logits, both at pos == prompt.len().
    let (boundary_cache, boundary_logits) = h.fresh_primed(&prompt, ctx);
    let dkv = memra_engine::dflash::DflashKv::from_tail(&h.engine, &dr.draft.cfg, ctx, &tail)
        .expect("drafter KV rebuilt from the tail");

    let _arm = FullCoverArm::arm();

    // RED: armed, but with NO boundary logits there is no anchor row: refuse, never guess.
    {
        let (c2, _) = h.fresh_primed(&prompt, ctx);
        let dkv_red =
            memra_engine::dflash::DflashKv::from_tail(&h.engine, &dr.draft.cfg, ctx, &tail)
                .expect("red-arm dkv");
        assert!(
            h.model
                .glm5_spec_session_from_restored(
                    &h.engine,
                    c2,
                    &prompt,
                    &[],
                    &[],
                    dkv_red,
                    ctx,
                    None,
                )
                .is_err(),
            "RED: a full-cover restore without the entry's boundary logits must refuse"
        );
    }

    let mut restored = h
        .model
        .glm5_spec_session_from_restored(
            &h.engine,
            boundary_cache,
            &prompt,
            &[],
            &boundary_logits,
            dkv,
            ctx,
            None,
        )
        .expect("full-cover restored spec session");
    let (tape, drafted, accepted, _bursts) =
        drive_bursts(&h, &mut restored, &prompt, k, max_new, 7, &[]);
    assert!(
        drafted > 0,
        "the full-cover restored session must actually draft"
    );
    assert_eq!(
        tape,
        tape_plain[..tape.len()],
        "full-cover restored spec tape must be BYTE-IDENTICAL to plain decode"
    );
    assert_eq!(
        tape.len(),
        tape_cold.len(),
        "full-cover restored and cold sessions must serve the same tape length"
    );
    assert_eq!(
        (drafted, accepted),
        (drafted_cold, accepted_cold),
        "full-cover restored drafter context must be byte-equivalent to the cold session's"
    );
    // SAMPLED TWIN (review round 1, finding 5). Prod serves vendor-default sampled, and the
    // anchor draw is the one thing structurally new on this arm: it comes off the ENTRY's
    // stored boundary row through a FRESH Philox stream at sctr=0 rather than off a prime
    // this session ran. The bar is the restore's own law (`spec_restore_refusal`'s doc): the
    // restored session's seed rule is bit-identical to the cold spec path's, so at one seed
    // the two tapes must be the same bytes. Greedy alone would never see a Philox slip.
    let seed = 0x5EEDu64;
    let mut cold_sampled = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, ctx, Some(sampled_cfg(seed)))
        .expect("cold sampled spec session");
    let (tape_cold_sampled, _d, _a, _b) =
        drive_bursts(&h, &mut cold_sampled, &prompt, k, max_new, 7, &[]);
    let (cache_sampled, logits_sampled) = h.fresh_primed(&prompt, ctx);
    let dkv_sampled =
        memra_engine::dflash::DflashKv::from_tail(&h.engine, &dr.draft.cfg, ctx, &tail)
            .expect("sampled-arm drafter KV rebuilt from the tail");
    let mut restored_sampled = h
        .model
        .glm5_spec_session_from_restored(
            &h.engine,
            cache_sampled,
            &prompt,
            &[],
            &logits_sampled,
            dkv_sampled,
            ctx,
            Some(sampled_cfg(seed)),
        )
        .expect("full-cover restored sampled spec session");
    let (tape_restored_sampled, drafted_s, _a_s, _b_s) =
        drive_bursts(&h, &mut restored_sampled, &prompt, k, max_new, 7, &[]);
    assert!(
        drafted_s > 0,
        "the sampled full-cover restored session must actually draft"
    );
    assert_eq!(
        tape_restored_sampled, tape_cold_sampled,
        "at one seed the full-cover restored sampled tape must equal the COLD sampled \
         tape byte for byte: the anchor is drawn from the entry's boundary row at Philox \
         counter 0, exactly as the cold session draws from its prime's row"
    );

    println!(
        "gate 13 PASS: full-cover restore == plain bytes over {} tokens, acceptance {} / {} \
         identical to cold; sampled twin == cold sampled over {} tokens at seed {seed:#x}",
        tape.len(),
        accepted,
        drafted,
        tape_restored_sampled.len(),
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 14 — ROUND-CADENCE COMMIT HOOK (lane/b200-spec-ttft-20260902, the engine half of
// `MEMRA_SPEC_FIRST_TOKEN_EAGER`): `glm5_spec_session_burst_streamed` hands the hook
// disjoint in-order slices whose concatenation IS the returned burst, the first slice is
// the prime's anchor ALONE (the first token is available before any round runs), and the
// tokens are byte-identical to the un-hooked twin on a fresh session — greedy AND sampled
// (pinned seed). The hook only moves WHEN the caller learns a token, never WHICH.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dflash_streamed_burst_slices_concat_to_the_unhooked_burst() {
    let _gpu = gpu_guard();
    let h = Harness::new("g14");
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let k = 3usize;
    let tape = plain_tape(&h, &prompt, max_new);
    for (arm, sampling) in [
        ("greedy", None),
        ("sampled", Some(sampled_cfg(0x5EED_0B20))),
    ] {
        // Un-hooked twin: one burst of `max_new` on a fresh session.
        let mut plain_sess = h
            .model
            .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + max_new + k + 8, sampling)
            .expect("glm5 dflash spec session (un-hooked twin)");
        let (plain_burst, pd, pa) = h
            .model
            .glm5_spec_session_burst(&h.engine, &mut plain_sess, max_new, k, &[])
            .expect("un-hooked burst");
        // Hooked arm: same request, every committed slice recorded as it lands.
        let mut sess = h
            .model
            .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + max_new + k + 8, sampling)
            .expect("glm5 dflash spec session (streamed)");
        let mut slices: Vec<Vec<u32>> = Vec::new();
        let (burst, d, a) = h
            .model
            .glm5_spec_session_burst_streamed(&h.engine, &mut sess, max_new, k, &[], &mut |s| {
                slices.push(s.to_vec())
            })
            .expect("streamed burst");
        let concat: Vec<u32> = slices.iter().flatten().copied().collect();
        assert_eq!(
            concat, burst,
            "{arm}: the hook's slices must concatenate to the returned burst"
        );
        assert_eq!(
            slices.first().map(|s| s.as_slice()),
            Some(&burst[..1]),
            "{arm}: the first slice must be the prime's anchor alone"
        );
        assert!(
            slices.iter().all(|s| !s.is_empty()),
            "{arm}: the hook never sees an empty slice"
        );
        assert_eq!(
            slices.len(),
            1 + sess.rounds,
            "{arm}: one slice per round plus the anchor slice"
        );
        assert_eq!(
            (burst.clone(), d, a),
            (plain_burst.clone(), pd, pa),
            "{arm}: the streamed burst must be byte-identical to the un-hooked twin \
             (tokens AND counters)"
        );
        if arm == "greedy" {
            assert_eq!(
                &burst[..max_new],
                &tape[..],
                "greedy streamed burst diverged from plain decode"
            );
        }
        assert_eq!(sess.pos(), sess.committed.len());
        println!(
            "gate 14 PASS ({arm}): {} slices over {} rounds concatenate to the {}-token burst, \
             byte-identical to the un-hooked twin ({a}/{d} accepted)",
            slices.len(),
            sess.rounds,
            burst.len()
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Gate 15 — CHUNKED DRAFTER PRIME (lane/spec-route-depth-20260902, `MEMRA_GLM5_DRAFT_PRIME_V2`):
// the drafter's ctx KV after a chunked prime (device-staged chunk sinks, pinned bounce, async
// upload, per-chunk fc + k/v ingest at the chunk width) is BIT-IDENTICAL to the eager arm's
// (whole-prompt host sink, round-1 ingest) when both see the prompt as ONE chunk, and the served
// tape is byte-identical to plain decode on BOTH arms at every chunking. The multi-chunk arm
// (a forced small `MEMRA_PRIME_CHUNK`) REPORTS the KV max-abs-diff and the acceptance delta
// rather than asserting them: a GEMM whose per-row bits depend on M moves drafts only, never
// output (verify arbitrates), and the box decides whether the delta is one it accepts.
// ---------------------------------------------------------------------------------------------

/// Sets one env var for the life of the value and removes it on drop (the `FullCoverArm`
/// shape). Safe because every gate in this binary holds `gpu_guard()` while it runs.
struct EnvArm(&'static str);

impl EnvArm {
    fn set(name: &'static str, value: &str) -> Self {
        // SAFETY: the caller holds gpu_guard(), which serializes every test in this binary.
        unsafe { std::env::set_var(name, value) };
        Self(name)
    }
}

impl Drop for EnvArm {
    fn drop(&mut self) {
        // SAFETY: as above, still under gpu_guard().
        unsafe { std::env::remove_var(self.0) };
    }
}

/// One gate-15 arm's outputs: (served tape, drafted, accepted, K planes, V planes).
type Gate15Arm = (Vec<u32>, usize, usize, Vec<Vec<f32>>, Vec<Vec<f32>>);

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dflash_chunked_drafter_prime_kv_matches_eager_ingest() {
    let _gpu = gpu_guard();
    let h = Harness::new("g15");
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let k = 3usize;
    let ctx = prompt.len() + max_new + k + 8;
    let tape = plain_tape(&h, &prompt, max_new);
    let cfg = &h
        .model
        .glm5_dflash
        .as_ref()
        .expect("drafter attached")
        .draft
        .cfg;
    let row_floats = cfg.n_kv * cfg.head_dim;
    // Both arms need the drafter KV filled before it is read: the eager arm ingests in
    // round 1, so ONE burst of one round is driven on each session before the KV is read.
    let kv_after_one_round = |sess: &mut Glm5SpecSession| -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let (_burst, _d, _a) = h
            .model
            .glm5_spec_session_burst(&h.engine, sess, 1, k, &[])
            .expect("one round");
        assert!(
            sess.draft_kv_len().expect("dflash session") >= prompt.len(),
            "the drafter KV must cover the prompt after round 1"
        );
        sess.draft_kv_rows_host(&h.engine, prompt.len(), row_floats)
            .expect("dflash session exports its KV rows")
    };
    // ---- arm A: one-chunk prompt (the default schedule), eager vs chunked: bit-identical ----
    let (k_eager, v_eager) = {
        // SAFETY: under gpu_guard.
        unsafe { std::env::remove_var("MEMRA_GLM5_DRAFT_PRIME_V2") };
        let mut sess = h
            .model
            .glm5_spec_session_new(&h.engine, &prompt, ctx, None)
            .expect("eager session");
        kv_after_one_round(&mut sess)
    };
    let (k_v2, v_v2) = {
        let _v2 = EnvArm::set("MEMRA_GLM5_DRAFT_PRIME_V2", "1");
        let mut sess = h
            .model
            .glm5_spec_session_new(&h.engine, &prompt, ctx, None)
            .expect("chunked-prime session");
        assert_eq!(
            sess.draft_kv_len(),
            Some(prompt.len()),
            "the chunked arm must have ingested the whole prompt at creation"
        );
        kv_after_one_round(&mut sess)
    };
    let bits = |planes: &[Vec<f32>]| -> Vec<Vec<u32>> {
        planes
            .iter()
            .map(|p| p.iter().map(|f| f.to_bits()).collect())
            .collect()
    };
    assert_eq!(
        bits(&k_v2),
        bits(&k_eager),
        "one-chunk prompt: the chunked drafter prime's K planes must be bit-identical to \
         the eager ingest's (same GEMM, same M, only the data movement changed)"
    );
    assert_eq!(
        bits(&v_v2),
        bits(&v_eager),
        "one-chunk prompt: V planes must be bit-identical"
    );
    // ---- arm B: forced multi-chunk prime (MEMRA_PRIME_CHUNK=16 over a 24-token prompt): the
    // trunk is chunked identically on both arms; the drafter ingest runs at t=16+8 on the
    // chunked arm vs t=24 on the eager arm. Tape identity is the bar; the KV diff and the
    // acceptance delta are REPORTED.
    let _chunk = EnvArm::set("MEMRA_PRIME_CHUNK", "16");
    let run = |v2: bool| -> Gate15Arm {
        let arm = v2.then(|| EnvArm::set("MEMRA_GLM5_DRAFT_PRIME_V2", "1"));
        let mut sess = h
            .model
            .glm5_spec_session_new(&h.engine, &prompt, ctx, None)
            .expect("session");
        drop(arm);
        let (kk, vv) = kv_after_one_round(&mut sess);
        let (mut out, mut d, mut a) = (Vec::new(), 0usize, 0usize);
        while out.len() < max_new && !sess.finished() {
            let (b, bd, ba) = h
                .model
                .glm5_spec_session_burst(&h.engine, &mut sess, 4, k, &[])
                .expect("burst");
            if b.is_empty() {
                break;
            }
            out.extend(b);
            d += bd;
            a += ba;
        }
        (out, d, a, kk, vv)
    };
    let (out_e, d_e, a_e, k_e2, v_e2) = run(false);
    let (out_v, d_v, a_v, k_v3, v_v3) = run(true);
    assert_eq!(
        &out_e[..max_new],
        &tape[..],
        "eager arm, chunked trunk: tape == plain"
    );
    assert_eq!(&out_v[..max_new], &tape[..], "chunked arm: tape == plain");
    let maxdiff = |a: &[Vec<f32>], b: &[Vec<f32>]| -> f32 {
        a.iter()
            .zip(b)
            .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
            .fold(0f32, f32::max)
    };
    let (kd, vd) = (maxdiff(&k_v3, &k_e2), maxdiff(&v_v3, &v_e2));
    println!(
        "gate 15 PASS: one-chunk KV bit-identical; multi-chunk (PRIME_CHUNK=16) tape == plain \
         on both arms; KV max-abs-diff k={kd:e} v={vd:e}; acceptance eager {a_e}/{d_e} vs \
         chunked {a_v}/{d_v}"
    );
}
