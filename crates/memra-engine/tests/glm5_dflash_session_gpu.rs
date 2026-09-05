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
//!    Since lane/frspec-dflash2-20260902 the matrix also covers the RANK-TRIMMED arms: the
//!    slab boot receipt (`draft head RANK-TRIMMED n_ranks=N src=<sha16>`, `FULL target
//!    vocab` gone), the per-session engagement line, the MtpHead-preferred shape when both
//!    are loaded, and the three boot REFUSALS (id >= head rows, duplicate id, non-numeric
//!    line) that a wrong-model ranks file must trip.
//! 8. RANK-TRIMMED DRAFT HEAD (gate 13, lane/frspec-dflash2-20260902): under the SAME
//!    `MEMRA_FRSPEC_TRIM=<ranks.txt>` contract the box uses, the DFlash2 round drafts over
//!    an `[n_ranks x d]` slab. Pinned: the greedy tape is byte-identical trimmed vs untrimmed
//!    vs plain at K=1..7 while acceptance is free to move; every drafted id lies inside the
//!    ranks set and the untrimmed census proves the set BINDS (it excludes ids the untrimmed
//!    drafter actually drafted); the slab rows are bit-identical to the head rows they were
//!    gathered from; the `rank_trimmed_rounds` counters equal `rounds`; and the RED
//!    remap-skipped arm (rank ids drafted as token ids) still leaves the tape untouched
//!    while the drafted sequence moves, the q38 silent defect made loud on this route.
//!
//! Rig law (exactness only, never timing):
//!   NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
//!     cargo test -p memra-engine --test glm5_dflash_session_gpu -- --ignored --test-threads=1

use memra_engine::Engine;
use memra_engine::forward::argmax;
use memra_engine::glm_spec::{Glm5SpecKnobs, Glm5SpecSession};
use memra_engine::hybrid::HybridModel;
use memra_engine::model::GpuTensor;
use memra_engine::spec::{PEN_WINDOW_MAX, SpecSampling};
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::ModelPlan;
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, LayerTensor, MtpTensor, OutputHead, TensorContract,
    TensorId, TensorMatch,
};
use memra_reference::{ReferenceTensor, ReferenceWeights, deterministic_fixture};
use memra_sampling::{Sampler, SamplerConfig};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

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

/// A ranks `.txt` fixture (one id per line, rank order), the box's `MEMRA_FRSPEC_TRIM`
/// artifact shape, verbatim. The caller that made it deletes it (tmp hygiene).
fn write_ranks_fixture(tag: &str, ranks: &[u32]) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "glm5-dflash-ranks-{}-{tag}.txt",
        std::process::id()
    ));
    let text: String = ranks.iter().map(|t| format!("{t}\n")).collect();
    std::fs::write(&path, text).expect("write ranks fixture");
    path
}

/// First 16 hex of sha256 over a file, what the loader prints as `src=<sha16>`.
fn sha16_of(path: &Path) -> String {
    let bytes = std::fs::read(path).expect("read ranks fixture");
    Sha256::digest(&bytes)
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect()
}

struct Harness {
    engine: Engine,
    model: HybridModel,
    plan: ModelPlan,
    drafter_dir: PathBuf,
    /// The ranks fixture this harness loaded under MEMRA_FRSPEC_TRIM (deleted on drop).
    ranks_path: Option<PathBuf>,
}

impl Drop for Harness {
    fn drop(&mut self) {
        // tmp hygiene: the task that made the fixture deletes it.
        std::fs::remove_dir_all(&self.drafter_dir).ok();
        if let Some(p) = self.ranks_path.as_ref() {
            std::fs::remove_file(p).ok();
        }
        // SAFETY: serialized behind gpu_guard by every caller.
        unsafe {
            std::env::remove_var("MEMRA_GLM5_DFLASH");
            std::env::remove_var("MEMRA_FRSPEC_TRIM");
        }
    }
}

impl Harness {
    fn new(tag: &str) -> Self {
        Self::build(tag, None)
    }

    /// The RANK-TRIMMED twin (lane/frspec-dflash2-20260902): `ranks` is written as a `.txt`
    /// fixture and loaded under `MEMRA_FRSPEC_TRIM`, the box env contract, no other flag.
    fn with_trim(tag: &str, ranks: &[u32]) -> Self {
        let path = write_ranks_fixture(tag, ranks);
        Self::build(tag, Some(path))
    }

    fn build(tag: &str, ranks_path: Option<PathBuf>) -> Self {
        force_true_f32();
        let drafter_dir = write_mini_drafter(tag);
        // SAFETY: serialized behind gpu_guard by every caller.
        unsafe {
            match ranks_path.as_ref() {
                Some(p) => std::env::set_var("MEMRA_FRSPEC_TRIM", p),
                None => std::env::remove_var("MEMRA_FRSPEC_TRIM"),
            }
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
        assert_eq!(
            model.dflash_trim.is_some(),
            ranks_path.is_some(),
            "the draft-head slab loads iff MEMRA_FRSPEC_TRIM names a ranks file"
        );
        Self {
            engine,
            model,
            plan,
            drafter_dir,
            ranks_path,
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

/// `drive_bursts` through the gate-instrument entry (`glm5_spec_session_burst_gated`) so a
/// census / red-arm knob rides every round; same burst-boundary invariants.
fn drive_gated(
    h: &Harness,
    sess: &mut Glm5SpecSession,
    prompt: &[u32],
    k: usize,
    total: usize,
    burst_target: usize,
    knobs: &mut Glm5SpecKnobs<'_>,
) -> (Vec<u32>, usize, usize, usize) {
    let mut tape: Vec<u32> = Vec::new();
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    let mut bursts = 0usize;
    while tape.len() < total && !sess.finished() {
        let room = (total - tape.len()).min(burst_target);
        let (burst, d, a) = h
            .model
            .glm5_spec_session_burst_gated(&h.engine, sess, room, k, &[], knobs)
            .expect("glm5 dflash gated spec session burst");
        if burst.is_empty() {
            break;
        }
        bursts += 1;
        drafted += d;
        accepted += a;
        tape.extend(burst);
        assert_eq!(sess.pos(), sess.committed.len());
        let mut expect: Vec<u32> = prompt.to_vec();
        expect.extend_from_slice(&tape[..tape.len() - 1]);
        assert_eq!(sess.committed, expect);
    }
    (tape, drafted, accepted, bursts)
}

/// A drafted-id CENSUS knob: records every draft the round produced (post-remap, pre-verify)
/// and returns it unchanged, so the tape stays the natural one. Shared handle so the census
/// outlives the knobs borrow.
fn census_knob(census: &Rc<RefCell<Vec<u32>>>) -> impl FnMut(usize, usize, u32) -> u32 + use<> {
    let c = Rc::clone(census);
    move |_round: usize, _ki: usize, d: u32| -> u32 {
        c.borrow_mut().push(d);
        d
    }
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
        eprintln!(
            "[helper] trim_rounds={} rounds={}",
            sess.rank_trimmed_rounds, sess.rounds
        );
    }
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_draft_source_selection_matrix() {
    let _gpu = gpu_guard();
    let drafter_dir = write_mini_drafter("matrix");
    // `(exit_ok, stderr)`: the refusal arms below assert on a FAILED boot by name.
    let run_child = |glm5_spec: Option<&str>,
                     glm5_mtp: Option<&str>,
                     dflash: Option<&Path>,
                     trim: Option<&Path>|
     -> (bool, String) {
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
        if let Some(path) = trim {
            cmd.env("MEMRA_FRSPEC_TRIM", path);
        }
        let out = cmd.output().expect("spawn receipt child");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let run_ok = |glm5_spec: Option<&str>,
                  glm5_mtp: Option<&str>,
                  dflash: Option<&Path>,
                  trim: Option<&Path>|
     -> String {
        let (ok, log) = run_child(glm5_spec, glm5_mtp, dflash, trim);
        assert!(
            ok,
            "receipt child failed (spec={glm5_spec:?} mtp={glm5_mtp:?} dflash={} trim={}):\n{log}",
            dflash.is_some(),
            trim.is_some()
        );
        log
    };
    let vram_mib = |log: &str| -> u64 {
        log.lines()
            .find_map(|l| l.strip_prefix("[helper] vram-used-at-ready-mib="))
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|v| v.parse().ok())
            .expect("vram line")
    };

    // ARM A: dflash2 source, MTP head NOT loaded — the q38 pattern; source line + burst.
    let log = run_ok(Some("1"), None, Some(&drafter_dir), None);
    assert!(
        log.contains("draft source = dflash2 @ "),
        "dflash2 selection receipt missing:\n{log}"
    );
    assert!(
        log.contains("native MTP head NOT loaded"),
        "the head-not-loaded note is the VRAM receipt:\n{log}"
    );
    assert!(
        log.contains("draft head FULL target vocab"),
        "no ranks file = the full-head note stays:\n{log}"
    );
    assert!(
        !log.contains("RANK-TRIMMED"),
        "no ranks file must never print a trim receipt:\n{log}"
    );
    assert!(
        log.contains("[helper] burst="),
        "dflash child never burst:\n{log}"
    );
    let vram_dflash = vram_mib(&log);

    // ARM B: both loaded — dflash2 wins by selection, stated.
    let log = run_ok(Some("1"), Some("1"), Some(&drafter_dir), None);
    assert!(
        log.contains("draft source = dflash2 @ ") && log.contains("ALSO loaded"),
        "both-armed selection must state dflash2 wins:\n{log}"
    );

    // ARM C: native only — the existing receipts plus the source line.
    let log = run_ok(Some("1"), Some("1"), None, None);
    assert!(
        log.contains("[glm5-spec] draft source = native-mtp"),
        "native-mtp selection receipt missing:\n{log}"
    );
    let vram_mtp = vram_mib(&log);

    // ARM D: neither — fail-closed warn, no route.
    let log = run_ok(Some("1"), None, None, None);
    assert!(
        log.contains("[glm5-spec] MEMRA_GLM5_SPEC=1 but no MTP head loaded"),
        "fail-closed warn missing:\n{log}"
    );

    // ---- RANK-TRIMMED arms (lane/frspec-dflash2-20260902) ----
    // A partial, reversed ranking: 30 of the 32 ids (ids 4 and 9 excluded), rank r != id.
    let ranks: Vec<u32> = (0..VOCAB).rev().filter(|t| *t != 4 && *t != 9).collect();
    let n_ranks = ranks.len();
    let ranks_path = write_ranks_fixture("matrix-ok", &ranks);
    let sha16 = sha16_of(&ranks_path);

    // ARM E: dflash2 + ranks, MTP NOT loaded, the box shape. The slab boots, the ARMED line
    // names n_ranks + the file sha16 (`FULL target vocab` gone), the session engagement line
    // prints, the round bursts.
    let log = run_ok(Some("1"), None, Some(&drafter_dir), Some(&ranks_path));
    let armed = format!("draft head RANK-TRIMMED n_ranks={n_ranks} src={sha16}");
    assert!(
        log.contains("[glm5-spec] serve route ARMED: draft source = dflash2 @ ")
            && log.contains(&armed),
        "ARMED line must carry `{armed}`:\n{log}"
    );
    assert!(
        !log.contains("FULL target vocab"),
        "a loaded slab must retire the full-head note:\n{log}"
    );
    assert!(
        log.contains(&format!(
            "[frspec-trim] glm5 DFlash2 draft-head slab: {n_ranks} rows of {VOCAB} gathered \
             from main output.weight"
        )) && log.contains(&format!("src={sha16}")),
        "slab build receipt missing:\n{log}"
    );
    assert!(
        log.contains(&format!(
            "[glm5-spec] draft head RANK-TRIMMED n_ranks={n_ranks} src={sha16}"
        )),
        "per-session engagement line missing:\n{log}"
    );
    assert!(
        log.contains("[helper] burst=") && log.contains("[helper] trim_rounds="),
        "trimmed dflash child never burst / never counted:\n{log}"
    );
    let trim_rounds: usize = log
        .lines()
        .find_map(|l| l.strip_prefix("[helper] trim_rounds="))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .expect("trim_rounds line");
    assert!(
        trim_rounds > 0,
        "the counter must count trimmed rounds:\n{log}"
    );

    // ARM F: both loaded + ranks, the MtpHead self-trim (target-head rows) is preferred,
    // the RANK-TRIMMED note still names the file, and NO second slab is built.
    let log = run_ok(Some("1"), Some("1"), Some(&drafter_dir), Some(&ranks_path));
    assert!(
        log.contains(&armed) && log.contains("ALSO loaded"),
        "MtpHead-preferred shape must still print the RANK-TRIMMED note:\n{log}"
    );
    assert!(
        log.contains("[frspec-trim] self-trimmed head:")
            && !log.contains("glm5 DFlash2 draft-head slab"),
        "with a target-head-trimmed MtpHead loaded no second slab may be built:\n{log}"
    );

    // ARM G (RED, owner order): a ranks file whose max id >= the head's rows is a
    // wrong-model file, the boot REFUSES by name, quoting the sha16; nothing serves.
    let mut oob = ranks.clone();
    oob[3] = VOCAB;
    let oob_path = write_ranks_fixture("matrix-oob", &oob);
    let oob_sha = sha16_of(&oob_path);
    let (ok, log) = run_child(Some("1"), None, Some(&drafter_dir), Some(&oob_path));
    assert!(
        !ok && log.contains(&format!("token id {VOCAB} >= head rows {VOCAB}"))
            && log.contains(&format!("sha16={oob_sha}")),
        "an out-of-vocab ranks file must refuse the boot by name (ok={ok}):\n{log}"
    );
    assert!(
        !log.contains("RANK-TRIMMED") && !log.contains("[helper] burst="),
        "a refused ranks file must never reach a receipt or a burst:\n{log}"
    );

    // ARM G2 (RED, the revuto finding on the re-land): the BOTH-LOADED shape consumes the
    // same env var through the MtpHead self-trim arm; it must refuse the same wrong-model
    // file by name, never boot a shorter list or abort in the gather's assert.
    let (ok, log) = run_child(Some("1"), Some("1"), Some(&drafter_dir), Some(&oob_path));
    assert!(
        !ok && log.contains(&format!("token id {VOCAB} >= head rows {VOCAB}"))
            && log.contains(&format!("sha16={oob_sha}")),
        "the both-loaded shape must refuse an out-of-vocab ranks file by name (ok={ok}):\n{log}"
    );
    assert!(
        !log.contains("RANK-TRIMMED") && !log.contains("[helper] burst="),
        "a refused ranks file must never reach a receipt or a burst (both loaded):\n{log}"
    );

    // ARM H (RED): a duplicated id refuses.
    let mut dup = ranks.clone();
    dup[5] = dup[6];
    let dup_path = write_ranks_fixture("matrix-dup", &dup);
    let (ok, log) = run_child(Some("1"), None, Some(&drafter_dir), Some(&dup_path));
    assert!(
        !ok && log.contains(&format!("token id {} appears more than once", dup[6])),
        "a duplicated id must refuse the boot (ok={ok}):\n{log}"
    );

    // ARM I (RED): a non-numeric line refuses (the lenient parse would have dropped it and
    // booted a SHORTER trim silently, the defect class the strict parser closes).
    let bad_path = std::env::temp_dir().join(format!(
        "glm5-dflash-ranks-{}-matrix-bad.txt",
        std::process::id()
    ));
    std::fs::write(&bad_path, "31\n30\nid\n29\n").expect("write bad ranks fixture");
    let (ok, log) = run_child(Some("1"), None, Some(&drafter_dir), Some(&bad_path));
    assert!(
        !ok && log.contains("line 3 is not a token id"),
        "a non-numeric ranks line must refuse the boot (ok={ok}):\n{log}"
    );

    // RED: MEMRA_GLM5_SPEC off must print ZERO [glm5-spec] lines, drafter/ranks loaded or not.
    for (dfl, trim) in [
        (None, None),
        (Some(drafter_dir.as_path()), None),
        (Some(drafter_dir.as_path()), Some(ranks_path.as_path())),
    ] {
        let log = run_ok(None, None, dfl, trim);
        assert!(
            !log.contains("[glm5-spec]"),
            "MEMRA_GLM5_SPEC off (dflash={} trim={}) must print no [glm5-spec] line:\n{log}",
            dfl.is_some(),
            trim.is_some()
        );
    }
    // tmp hygiene: the task that made the fixtures deletes them.
    for p in [&ranks_path, &oob_path, &dup_path, &bad_path] {
        std::fs::remove_file(p).ok();
    }
    std::fs::remove_dir_all(&drafter_dir).ok();
    println!(
        "gate 8 PASS: selection matrix green+red incl. the RANK-TRIMMED arms (slab receipt, \
         MtpHead-preferred, 3 boot refusals on both shapes); VRAM-at-ready mini-fixture: dflash-boot \
         {vram_dflash} MiB vs mtp-boot {vram_mtp} MiB (delta {} MiB — mini scale; the box \
         three-way window banks the real-artifact delta)",
        vram_dflash as i64 - vram_mtp as i64
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 13, RANK-TRIMMED DRAFT HEAD (lane/frspec-dflash2-20260902): under the box's exact
// `MEMRA_FRSPEC_TRIM=<ranks.txt>` contract the DFlash2 round drafts over the `[n_ranks x d]`
// slab. The tape is byte-identical trimmed vs untrimmed vs plain (verify is full-vocab and
// untouched, a draft source can only move acceptance); every drafted id is inside the ranks
// set and the untrimmed census proves the set BINDS; the slab rows are bit-identical to the
// head rows they were gathered from; the counters equal `rounds`; and the remap-skipped RED
// arm moves the drafted sequence while the tape stays put.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dflash_rank_trimmed_head_moves_acceptance_never_the_tape() {
    let _gpu = gpu_guard();
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 20usize;
    let census_ks = [3usize, K];

    // ARM 0: untrimmed, the plain tape and the drafted-id census per K.
    let (tape, untrimmed) = {
        let h = Harness::new("g11u");
        let tape = plain_tape(&h, &prompt, max_new);
        let mut per_k: Vec<(usize, Vec<u32>, usize, usize)> = Vec::new();
        for &k in &census_ks {
            let census = Rc::new(RefCell::new(Vec::<u32>::new()));
            let mut record = census_knob(&census);
            let mut knobs = Glm5SpecKnobs {
                draft_override: Some(&mut record),
                ..Default::default()
            };
            let mut sess = h
                .model
                .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + max_new + k + 8, None)
                .expect("untrimmed glm5 dflash spec session");
            let (out, drafted, accepted, _) =
                drive_gated(&h, &mut sess, &prompt, k, max_new, 3, &mut knobs);
            assert_eq!(&out[..max_new], &tape[..], "K={k}: untrimmed tape != plain");
            assert_eq!(
                sess.rank_trimmed_rounds, 0,
                "K={k}: no trim loaded, the counter must stay 0"
            );
            let ids = census.borrow().clone();
            assert_eq!(ids.len(), drafted, "K={k}: the census must see every draft");
            per_k.push((k, ids, drafted, accepted));
        }
        (tape, per_k)
    };

    // The ranks set EXCLUDES the two ids the untrimmed drafter drafted most, so the trim
    // binds (an unbinding subset would pass every identity check vacuously); REVERSED order
    // so rank r != token id almost everywhere (the remap is live, not the identity).
    let mut freq = BTreeMap::<u32, usize>::new();
    for (_, ids, _, _) in &untrimmed {
        for &t in ids {
            *freq.entry(t).or_default() += 1;
        }
    }
    assert!(
        freq.len() >= 3,
        "degenerate fixture: only {} distinct drafted ids",
        freq.len()
    );
    let mut by_freq: Vec<(u32, usize)> = freq.iter().map(|(&t, &n)| (t, n)).collect();
    by_freq.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let excluded: BTreeSet<u32> = by_freq.iter().take(2).map(|(t, _)| *t).collect();
    let ranks: Vec<u32> = (0..VOCAB).rev().filter(|t| !excluded.contains(t)).collect();
    let n_ranks = ranks.len();
    assert!(n_ranks >= DRAFT_TOPK && n_ranks < VOCAB as usize);
    let ranks_set: BTreeSet<u32> = ranks.iter().copied().collect();

    // ARM 1: trimmed twin under the box contract.
    let h = Harness::with_trim("g11t", &ranks);
    let slab = h.model.dflash_trim.as_ref().expect("the draft-head slab");
    assert_eq!(
        slab.d2t, ranks,
        "slab d2t must be the ranks file, in rank order"
    );
    let sha16 = sha16_of(h.ranks_path.as_ref().unwrap());
    assert_eq!(slab.src_sha16, sha16);
    assert_eq!(h.model.frspec_src_sha16.as_deref(), Some(sha16.as_str()));
    assert!(
        h.model
            .glm5_dflash_trim()
            .is_some_and(|(_, d2t)| d2t == ranks.as_slice()),
        "the round's trim resolution must select the slab"
    );

    // SLAB BYTE IDENTITY: slab row r == head row d2t[r], bit for bit, read back from the
    // DEVICE tensors the round actually multiplies through.
    let (full, slab_rows) = match (&h.model.output, &slab.head) {
        (GpuTensor::Float { data: f, ne: fne }, GpuTensor::Float { data: sl, ne: sne }) => {
            assert_eq!(fne, &vec![HIDDEN as u64, VOCAB as u64]);
            assert_eq!(sne, &vec![HIDDEN as u64, n_ranks as u64]);
            (
                h.engine.dtoh(f).expect("head dtoh"),
                h.engine.dtoh(sl).expect("slab dtoh"),
            )
        }
        _ => panic!("the mini fixture's head is F32 on both sides"),
    };
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let head_row = |t: u32| bits(&full[t as usize * HIDDEN..(t as usize + 1) * HIDDEN]);
    for (r, &t) in ranks.iter().enumerate() {
        assert_eq!(
            bits(&slab_rows[r * HIDDEN..(r + 1) * HIDDEN]),
            head_row(t),
            "slab row {r} != head row {t}"
        );
    }
    // Non-vacuity: head rows are distinct, so a permuted gather could not have passed.
    assert_ne!(head_row(ranks[0]), head_row(ranks[1]));
    assert_ne!(bits(&slab_rows[..HIDDEN]), head_row(ranks[1]));

    // Tape identity at every K, census inside the ranks set, counters.
    let global_before = memra_engine::glm_spec::glm5_rank_trimmed_draft_rounds();
    let mut trimmed_census: Vec<(usize, Vec<u32>)> = Vec::new();
    for k in 1..=K {
        let census = Rc::new(RefCell::new(Vec::<u32>::new()));
        let mut record = census_knob(&census);
        let mut knobs = Glm5SpecKnobs {
            draft_override: Some(&mut record),
            ..Default::default()
        };
        let mut sess = h
            .model
            .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + max_new + k + 8, None)
            .expect("trimmed glm5 dflash spec session");
        let (out, drafted, accepted, bursts) =
            drive_gated(&h, &mut sess, &prompt, k, max_new, 3, &mut knobs);
        assert_eq!(
            &out[..max_new],
            &tape[..],
            "K={k}: RANK-TRIMMED tape diverged from plain greedy ({accepted}/{drafted} over \
             {bursts} bursts), the slab may only move acceptance, never output"
        );
        let ids = census.borrow().clone();
        assert_eq!(ids.len(), drafted, "K={k}: the census must see every draft");
        assert!(
            ids.iter().all(|t| ranks_set.contains(t)),
            "K={k}: a drafted id lies outside the ranks set: {ids:?}"
        );
        assert!(sess.rounds > 0);
        assert_eq!(
            sess.rank_trimmed_rounds, sess.rounds,
            "K={k}: every round drafted through the slab must be counted"
        );
        if let Some((_, uids, ud, ua)) = untrimmed.iter().find(|(uk, ..)| *uk == k) {
            assert!(
                uids.iter().any(|t| excluded.contains(t)),
                "K={k}: the untrimmed census never drafted an excluded id, the ranks set \
                 does not bind and the identity above is vacuous"
            );
            println!(
                "gate 13 K={k}: tape identical; acceptance trimmed {accepted}/{drafted} vs \
                 untrimmed {ua}/{ud} (free to move; n_ranks={n_ranks}, excluded {excluded:?})"
            );
        }
        trimmed_census.push((k, ids));
    }
    assert!(
        memra_engine::glm_spec::glm5_rank_trimmed_draft_rounds() > global_before,
        "the process-wide counter must move"
    );

    // RED ARM: remap skipped, rank ids drafted AS token ids (the q38 silent defect). The
    // verify walk is full-vocab so the tape stays byte-identical; the drafted sequence must
    // MOVE (the remap is live) and the counter still counts the slab.
    {
        let k = K;
        let census = Rc::new(RefCell::new(Vec::<u32>::new()));
        let mut record = census_knob(&census);
        let mut knobs = Glm5SpecKnobs {
            draft_override: Some(&mut record),
            skip_d2t_remap: true,
            ..Default::default()
        };
        let mut sess = h
            .model
            .glm5_spec_session_new(&h.engine, &prompt, prompt.len() + max_new + k + 8, None)
            .expect("red-arm glm5 dflash spec session");
        let (out, drafted, accepted, _) =
            drive_gated(&h, &mut sess, &prompt, k, max_new, 3, &mut knobs);
        assert_eq!(
            &out[..max_new],
            &tape[..],
            "remap-skipped red arm: the tape must STILL be plain (verify arbitrates)"
        );
        let ids = census.borrow().clone();
        let (_, remapped) = trimmed_census
            .iter()
            .find(|(tk, _)| *tk == k)
            .expect("K census");
        assert_ne!(
            &ids, remapped,
            "skipping the d2t remap must change WHICH ids get drafted (the remap is live)"
        );
        assert_eq!(sess.rank_trimmed_rounds, sess.rounds);
        println!(
            "gate 13 RED PASS: remap skipped -> tape identical, drafted sequence moved \
             ({accepted}/{drafted} accepted)"
        );
    }
    println!("gate 13 PASS: RANK-TRIMMED slab n_ranks={n_ranks} src={sha16}");
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
        unsafe { std::env::set_var(self.0, "0") };
    }
}

/// One gate-15 arm's outputs: (served tape, drafted, accepted, K planes, V planes).
type Gate15Arm = (Vec<u32>, usize, usize, Vec<Vec<f32>>, Vec<Vec<f32>>);

// ---------------------------------------------------------------------------------------------
// Gate 16 — DEVICE-RESIDENT DRAFTER PRIME (lane/spec-route-depth-20260902,
// `MEMRA_GLM5_DRAFT_TAPS_DEVICE`): the trunk prime stays the one whole-prompt call, the tap
// planes never leave the device (chunk ring on the writing stage, 2D-copy interleave on the
// head device, fc + k/v ingest at the range width inside the prime's own range loop), and the
// drafter KV is BIT-IDENTICAL to the eager (host-tap) arm's when both ingest the prompt as
// one range; the served GREEDY tape is byte-identical between host taps and device taps and
// to plain decode on both arms at every chunking: the exactness receipt of the default flip
// to device taps (boot D, 2026-09-03). The
// forced multi-range arm (`MEMRA_PRIME_CHUNK=16`, two ranges of 16 + 8 against the eager
// arm's one 24-row ingest) REPORTS the KV max-abs-diff and the acceptance delta (a GEMM
// whose per-row bits depend on M moves drafts only; verify arbitrates).
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dflash_device_resident_drafter_prime_kv_matches_eager_ingest() {
    let _gpu = gpu_guard();
    let h = Harness::new("g16");
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
    let bits = |planes: &[Vec<f32>]| -> Vec<Vec<u32>> {
        planes
            .iter()
            .map(|p| p.iter().map(|f| f.to_bits()).collect())
            .collect()
    };
    let maxdiff = |a: &[Vec<f32>], b: &[Vec<f32>]| -> f32 {
        a.iter()
            .zip(b)
            .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
            .fold(0f32, f32::max)
    };
    // One session on the named arm: KV rows right after creation (both arms ingest the
    // whole prompt at creation), then the served tape over 4-token bursts.
    let run = |device: bool| -> Gate15Arm {
        // Both arms are set explicitly: device taps are default OFF (the flip is blocked on
        // the arm-2 range-hook panic and the ring cross-stream race), `=1` arms them.
        let arm = EnvArm::set(
            "MEMRA_GLM5_DRAFT_TAPS_DEVICE",
            if device { "1" } else { "0" },
        );
        let mut sess = h
            .model
            .glm5_spec_session_new(&h.engine, &prompt, ctx, None)
            .expect("session");
        drop(arm);
        assert_eq!(
            sess.draft_kv_len(),
            Some(prompt.len()),
            "device={device}: the drafter KV must cover the prompt at creation"
        );
        let (kk, vv) = sess
            .draft_kv_rows_host(&h.engine, prompt.len(), row_floats)
            .expect("dflash session exports its KV rows");
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
    // ---- arm A: the default schedule (one range): bit-identical KV, identical tape ----
    let (out_e, d_e, a_e, k_e, v_e) = run(false);
    let (out_d, d_d, a_d, k_d, v_d) = run(true);
    assert_eq!(
        bits(&k_d),
        bits(&k_e),
        "one range: device-resident K planes != eager"
    );
    assert_eq!(
        bits(&v_d),
        bits(&v_e),
        "one range: device-resident V planes != eager"
    );
    assert_eq!(&out_e[..max_new], &tape[..], "eager arm: tape == plain");
    assert_eq!(&out_d[..max_new], &tape[..], "device arm: tape == plain");
    assert_eq!(
        out_d, out_e,
        "GREEDY TAPE IDENTITY, host taps vs device taps (the default flip's exactness \
         receipt): the served tapes must be byte-identical"
    );
    assert_eq!(
        (d_d, a_d),
        (d_e, a_e),
        "one range: identical KV must give identical acceptance"
    );
    // ---- arm B: forced two-range prime: the trunk is chunked identically on both arms;
    // the device arm ingests at 16 + 8, the eager arm at 24. Tape is the bar; the KV diff
    // and the acceptance delta are reported.
    let _chunk = EnvArm::set("MEMRA_PRIME_CHUNK", "16");
    let (out_e2, d_e2, a_e2, k_e2, v_e2) = run(false);
    let (out_d2, d_d2, a_d2, k_d2, v_d2) = run(true);
    assert_eq!(
        &out_e2[..max_new],
        &tape[..],
        "eager arm, chunked trunk: tape == plain"
    );
    assert_eq!(
        &out_d2[..max_new],
        &tape[..],
        "device arm, chunked trunk: tape == plain"
    );
    assert_eq!(
        out_d2, out_e2,
        "two ranges: greedy tape identity host taps vs device taps"
    );
    println!(
        "gate 16 PASS: one-range KV bit-identical ({a_e}/{d_e} accepted both arms); two-range \
         (PRIME_CHUNK=16) tape == plain on both arms, KV max-abs-diff k={:e} v={:e}, \
         acceptance eager {a_e2}/{d_e2} vs device {a_d2}/{d_d2}",
        maxdiff(&k_d2, &k_e2),
        maxdiff(&v_d2, &v_e2)
    );
}

// ---------------------------------------------------------------------------------------------
// Gates 17-20 — THE SPEC EXCLUSIONS LANE (lane/spec-exclusions-20260902): the two doors that
// admit request classes the route used to serve plain. Every gate here is exactness only.
//
// 17. PENALTY ARM, greedy: a penalized GREEDY request's served tape is byte-identical to the
//     plain penalized sampler's (the host `Sampler`: prompt accepted into the history, then
//     penalize-and-argmax per token), K=1..7 across burst boundaries; the penalties visibly
//     engage (the tape differs from the unpenalized plain tape, so the gate is not vacuous);
//     and with the door dark the session REFUSES (the pre-lane posture, verbatim).
// 18. DEVICE PENALTIES == HOST SAMPLER, bit for bit: `penalize_logits_rows_inc` (the round's
//     per-row evolving window) and `penalize_logits` (the anchor) produce the host
//     `Sampler::penalized_logits` bytes over random rows and histories with repeats, for
//     every coefficient class and for a window that slides. This is the property gate 17's
//     identity rests on; a 1-ulp drift is an argmax flip on a near-tie.
// 19. PENALTY ARM, sampled: pinned-seed determinism, burst-split invariance, seed
//     sensitivity — and the penalized tape differs from the unpenalized same-seed tape.
//     Token-for-token identity against the PLAIN sampled route is impossible by
//     construction (host SplitMix64 vs the session's device Philox stream); the sampled
//     claim is distributional and is carried by gate 18 + the rejection walk's exactness.
// 20. WARM, cold drafter: a restored session whose drafter re-arms EMPTY at the restored
//     boundary (`DflashKv::new_cold_at`, no tail on the entry) serves the plain tape byte
//     for byte and actually drafts; a tail exported from it is floor-bearing and imports
//     with the floor; the re-restored continuation is byte-identical to plain decode again;
//     and under `MEMRA_DFLASH2_SDPA_CLIP=0` the cold drafter refuses by name.
//
// Rig law (exactness only, never timing):
//   NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
//     cargo test -p memra-engine --test glm5_dflash_session_gpu -- --ignored --test-threads=1
// ---------------------------------------------------------------------------------------------

/// Sets `MEMRA_SPEC_PENALTY=1` for the life of the value and removes it on drop (the
/// `FullCoverArm` pattern). Safe because every gate in this binary holds `gpu_guard()`.
struct PenaltyArm;

impl PenaltyArm {
    fn arm() -> Self {
        // SAFETY: the caller holds gpu_guard(), which serializes every test in this binary.
        unsafe { std::env::set_var("MEMRA_SPEC_PENALTY", "1") };
        Self
    }
}

impl Drop for PenaltyArm {
    fn drop(&mut self) {
        // SAFETY: as above, still under gpu_guard().
        unsafe { std::env::remove_var("MEMRA_SPEC_PENALTY") };
    }
}

/// A strong, mixed penalty on the 32-token fixture vocabulary — every coefficient live, the
/// serve API's window (`PEN_WINDOW_MAX`). Strong on purpose: with a 32-id vocab the plain
/// tape repeats within a few tokens, so penalties MUST move it or the identity is vacuous.
fn penalty_cfg(temperature: f32, seed: u64) -> SamplerConfig {
    SamplerConfig {
        temperature,
        penalty_last_n: PEN_WINDOW_MAX,
        penalty_repeat: 1.3,
        penalty_freq: 0.35,
        penalty_present: 0.2,
        seed,
        ..SamplerConfig::default()
    }
}

/// The worker's `glm5_spec_sampling_for`, field for field: the session seam that carries a
/// greedy request's penalties in (temp 0) as well as a sampled config.
fn spec_sampling_of(cfg: &SamplerConfig) -> SpecSampling {
    SpecSampling {
        temp: cfg.temperature,
        seed: cfg.seed,
        top_k: cfg.top_k as i32,
        top_p: cfg.top_p,
        min_p: cfg.min_p,
        penalty_last_n: cfg.penalty_last_n,
        penalty_repeat: cfg.penalty_repeat,
        penalty_freq: cfg.penalty_freq,
        penalty_present: cfg.penalty_present,
    }
}

/// THE PLAIN PENALIZED TAPE: tokenwise decode driven by the host `Sampler` exactly as the
/// plain worker route drives it — the prompt is `accept`ed into the penalty history, each
/// step's logits go through `sample` (penalize over the window, then argmax at temperature
/// 0), and the chosen token is accepted before the next step.
fn plain_tape_sampler(
    h: &Harness,
    prompt: &[u32],
    max_new: usize,
    cfg: &SamplerConfig,
) -> Vec<u32> {
    let (mut cache, logits) = h.fresh_primed(prompt, prompt.len() + max_new + 16);
    let mut sampler = Sampler::new(cfg.clone());
    for &t in prompt {
        sampler.accept(t);
    }
    let mut tape = Vec::with_capacity(max_new);
    let first = sampler.sample(&logits);
    sampler.accept(first);
    tape.push(first);
    while tape.len() < max_new {
        let ll = h
            .model
            .decode_step(&h.engine, *tape.last().unwrap(), &mut cache)
            .expect("plain decode step");
        let t = sampler.sample(&ll);
        sampler.accept(t);
        tape.push(t);
    }
    tape
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_penalized_greedy_spec_tape_matches_the_plain_penalized_sampler() {
    let _gpu = gpu_guard();
    let h = Harness::new("g15");
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 24usize;
    let cfg = penalty_cfg(0.0, 0);
    let tape_pen = plain_tape_sampler(&h, &prompt, max_new, &cfg);
    let tape_raw = plain_tape(&h, &prompt, max_new);
    assert_ne!(
        tape_pen, tape_raw,
        "the penalties must visibly move the plain tape at this scale, or the identity \
         below proves nothing"
    );
    let ctx = prompt.len() + max_new + K + 8;

    // RED: the door dark is the shipped default and must keep the pre-lane refusal.
    assert!(
        std::env::var("MEMRA_SPEC_PENALTY").is_err(),
        "gate 17 pins the DARK posture first; the arm is thrown below"
    );
    let dark = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, ctx, Some(spec_sampling_of(&cfg)));
    assert!(
        dark.is_err(),
        "RED: a penalized session must refuse while MEMRA_SPEC_PENALTY is unset"
    );
    let why = dark.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(
        why.contains("MEMRA_SPEC_PENALTY"),
        "the refusal must name the door: {why}"
    );

    let _arm = PenaltyArm::arm();
    for k in 1..=K {
        let mut sess = h
            .model
            .glm5_spec_session_new(
                &h.engine,
                &prompt,
                prompt.len() + max_new + k + 8,
                Some(spec_sampling_of(&cfg)),
            )
            .expect("penalized greedy glm5 dflash spec session");
        let (out, drafted, accepted, bursts) =
            drive_bursts(&h, &mut sess, &prompt, k, max_new, 3, &[]);
        assert_eq!(
            &out[..max_new],
            &tape_pen[..],
            "K={k}: penalized greedy spec tape diverged from the plain penalized sampler \
             ({accepted}/{drafted} over {bursts} bursts)"
        );
        assert!(
            drafted > 0,
            "K={k}: the penalized session must actually draft"
        );
        println!(
            "gate 17 PASS K={k}: penalized greedy spec == plain penalized sampler over \
             {bursts} bursts, {accepted}/{drafted} accepted"
        );
    }
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_device_penalties_are_bit_identical_to_the_host_sampler() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let n = 1000usize; // the logits row width (ids >= n in the history are inert)
    let nrow = 5usize; // anchor row + 4 drafts
    let mut state = 0x5EEDu64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    // Rows with positive AND negative logits (the repeat rule branches on the sign).
    let rows: Vec<f32> = (0..nrow * n)
        .map(|_| ((next() % 20_001) as f32 - 10_000.0) / 997.0)
        .collect();
    // A long history from a SMALL alphabet so counts run past 1 (ids stay inside the row: the
    // host sampler is only ever handed in-vocabulary ids and debug-asserts otherwise; the
    // kernel's own out-of-row no-op is exercised by sample_check's sparse arm).
    let hist0: Vec<u32> = (0..200).map(|_| (next() % 40) as u32).collect();
    let drafts: Vec<u32> = (0..nrow - 1).map(|_| (next() % 40) as u32).collect();
    // (last_n, rep, freq, present): each coefficient alone, all together, and a window that
    // SLIDES (64 over a 200-token history: every row drops one oldest entry).
    let classes: [(usize, f32, f32, f32); 6] = [
        (PEN_WINDOW_MAX, 1.3, 0.0, 0.0),
        (PEN_WINDOW_MAX, 1.0, 0.35, 0.0),
        (PEN_WINDOW_MAX, 1.0, 0.0, 0.2),
        (PEN_WINDOW_MAX, 1.3, 0.35, 0.2),
        (PEN_WINDOW_MAX, 0.8, -0.1, -0.05),
        (64, 1.3, 0.35, 0.2),
    ];
    for (last_n, rep, freq, present) in classes {
        let cfg = SamplerConfig {
            penalty_last_n: last_n,
            penalty_repeat: rep,
            penalty_freq: freq,
            penalty_present: present,
            ..SamplerConfig::default()
        };
        // HOST: row r's window is hist0 ++ drafts[..r], the plain sampler's own history.
        let mut expect: Vec<f32> = Vec::with_capacity(nrow * n);
        for r in 0..nrow {
            let mut s = Sampler::new(cfg.clone());
            for &t in hist0.iter().chain(drafts[..r].iter()) {
                s.accept(t);
            }
            expect.extend(s.penalized_logits(&rows[r * n..(r + 1) * n]));
        }
        // DEVICE, the round's kernel: pen_win = the session window pre-trimmed to `win`,
        // hist = pen_win ++ drafts, row r penalizes over the last min(win, n_win + r).
        let win = last_n.min(PEN_WINDOW_MAX);
        let w0 = hist0.len().saturating_sub(win);
        let mut hist: Vec<u32> = hist0[w0..].to_vec();
        let n_win = hist.len();
        hist.extend_from_slice(&drafts);
        let hd = e.htod_u32_v(&hist).expect("hist");
        let mut buf = e.htod(&rows).expect("rows");
        e.penalize_logits_rows_inc(&mut buf, &hd, n_win, rep, freq, present, n, nrow, win)
            .expect("penalize_logits_rows_inc");
        let got = e.dtoh(&buf).expect("dtoh");
        let bad = got
            .iter()
            .zip(&expect)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            bad,
            0,
            "rows_inc (last_n={last_n} rep={rep} freq={freq} present={present}): {bad} of \
             {} logits differ from the host sampler's bytes",
            got.len()
        );
        let touched = got
            .iter()
            .zip(&rows)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(touched > 0, "the class must actually penalize something");
        // DEVICE, the anchor's kernel: one row over the trimmed window.
        let hd0 = e.htod_u32_v(&hist[..n_win]).expect("hist0");
        let mut col = e.htod(&rows[..n]).expect("row 0");
        e.penalize_logits(&mut col, &hd0, n_win, rep, freq, present, n)
            .expect("penalize_logits");
        let got0 = e.dtoh(&col).expect("dtoh");
        let bad0 = got0
            .iter()
            .zip(&expect[..n])
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            bad0, 0,
            "penalize_logits (last_n={last_n} rep={rep} freq={freq} present={present}): \
             {bad0} of {n} logits differ from the host sampler's bytes"
        );
        println!(
            "gate 18 PASS last_n={last_n} rep={rep} freq={freq} present={present}: \
             {touched} penalized logits over {nrow} rows, all bit-identical to the host"
        );
    }
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_penalized_sampled_twin_is_deterministic_split_invariant_and_engaged() {
    let _gpu = gpu_guard();
    let h = Harness::new("g17");
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 24usize;
    let k = 3usize;
    let ctx = prompt.len() + max_new + k + 8;
    let _arm = PenaltyArm::arm();

    let run = |cfg: SamplerConfig, burst_target: usize| -> Vec<u32> {
        let mut sess = h
            .model
            .glm5_spec_session_new(&h.engine, &prompt, ctx, Some(spec_sampling_of(&cfg)))
            .expect("penalized sampled glm5 dflash spec session");
        let (tape, d, _a, _b) = drive_bursts(&h, &mut sess, &prompt, k, max_new, burst_target, &[]);
        assert!(d > 0, "the sampled penalized session must draft");
        tape[..max_new.min(tape.len())].to_vec()
    };

    let a = run(penalty_cfg(0.9, 42), 3);
    let b = run(penalty_cfg(0.9, 42), 3);
    assert_eq!(a, b, "same seed, same burst split: reproducible");
    let c = run(penalty_cfg(0.9, 42), max_new);
    assert_eq!(
        a, c,
        "burst-split invariance: the accept uniforms and every draw ride the session's \
         Philox counters, so the split must not change the stream"
    );
    let d = run(penalty_cfg(0.9, 43), 3);
    assert_ne!(a, d, "a different seed must change the sampled tape");
    // ENGAGEMENT: the same seed WITHOUT penalties draws from a different target.
    let unpen = run(
        SamplerConfig {
            temperature: 0.9,
            seed: 42,
            ..SamplerConfig::default()
        },
        3,
    );
    assert_ne!(
        a, unpen,
        "the penalties must move the sampled target (same seed, same counters, different p)"
    );
    println!(
        "gate 19 PASS: penalized sampled twin deterministic, split-invariant, seed-sensitive, \
         and distinct from the unpenalized same-seed tape"
    );
}

/// Sets `MEMRA_DFLASH2_SDPA_CLIP=0` (the clip rollback seam) for the life of the value.
struct ClipOff;

impl ClipOff {
    fn arm() -> Self {
        // SAFETY: the caller holds gpu_guard().
        unsafe { std::env::set_var("MEMRA_DFLASH2_SDPA_CLIP", "0") };
        Self
    }
}

impl Drop for ClipOff {
    fn drop(&mut self) {
        // SAFETY: as above.
        unsafe { std::env::remove_var("MEMRA_DFLASH2_SDPA_CLIP") };
    }
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_cold_drafter_restore_bytes_match_plain_decode_and_republishes_a_floor_tail() {
    let _gpu = gpu_guard();
    let h = Harness::new("g18");
    let prompt = tokens(PROMPT, 0xA11CE);
    let split = PROMPT - BLOCK;
    let (prefix, suffix) = prompt.split_at(split);
    let max_new = 40usize;
    let k = 3usize;
    let ctx = prompt.len() + max_new + K + 8;
    let dr = h.model.glm5_dflash.as_ref().expect("drafter attached");

    // The byte reference: plain decode over the whole prompt, long enough for two legs.
    let tape_plain = plain_tape(&h, &prompt, max_new);

    // RED: the cold drafter needs the clipped round attention (the legacy full-scan kernel
    // has no context-floor arm and would score the empty rows).
    {
        let _clip_off = ClipOff::arm();
        assert!(
            memra_engine::dflash::DflashKv::new_cold_at(&h.engine, &dr.draft.cfg, ctx, split)
                .is_err(),
            "RED: a cold drafter must refuse under MEMRA_DFLASH2_SDPA_CLIP=0"
        );
    }

    // LEG 1: the restored trunk at the prefix boundary + a COLD drafter (no tail anywhere).
    let (boundary_cache, boundary_logits) = h.fresh_primed(prefix, ctx);
    let dkv = memra_engine::dflash::DflashKv::new_cold_at(&h.engine, &dr.draft.cfg, ctx, split)
        .expect("cold drafter at the restored boundary");
    assert_eq!(
        dkv.len, split,
        "the cold drafter sits at the restored boundary"
    );
    assert_eq!(dkv.floor(), split, "and owns no rows below it");
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
        .expect("restored spec session with a cold drafter");
    let n1 = 12usize;
    let (tape1, drafted1, accepted1, _) = drive_bursts(&h, &mut restored, &prompt, k, n1, 5, &[]);
    assert!(drafted1 > 0, "the cold-drafter session must actually draft");
    assert_eq!(
        tape1,
        tape_plain[..tape1.len()],
        "cold-drafter restored tape must be BYTE-IDENTICAL to plain decode (the drafter \
         can only move acceptance)"
    );

    // REPUBLISH: the tail exported from the cold-drafter session is FLOOR-BEARING — it
    // starts at or above the boundary the drafter was born at, never below. Exported at the
    // PROMPT boundary (the publication a prefix entry carries, gate 13's shape): the drafter
    // KV covers it after round 1 ingested the suffix taps (the ingest runs at the START of a
    // round, so `kv.len` trails `pos()` by the last round's rows until the next round).
    let upto = prompt.len();
    let tail = restored
        .export_draft_tail(&h.engine, upto)
        .expect("tail export from the cold-drafter session");
    assert_eq!(tail.floor, split, "the tail carries its exporter's floor");
    assert!(
        tail.base >= split && tail.base + tail.rows == upto,
        "the tail covers [{}, {upto}) and nothing below the floor {split}",
        tail.base
    );
    assert!(
        tail.rows < dr.draft.cfg.sliding_window,
        "at this scale the tail is SHORT (window {}): the floor is what admits it",
        dr.draft.cfg.sliding_window
    );
    drop(restored);

    // LEG 2: re-restore from the floor-bearing tail at `upto` (the whole prompt) with the
    // first 8 plain tokens as the suffix; the continuation must be plain decode's
    // continuation from there, byte for byte.
    let j0 = 0usize; // index into tape_plain of the first token past `upto`
    let committed2: Vec<u32> = prompt.to_vec();
    let suffix2 = &tape_plain[j0..j0 + BLOCK];
    let dkv2 = memra_engine::dflash::DflashKv::from_tail(&h.engine, &dr.draft.cfg, ctx, &tail)
        .expect("a floor-bearing tail imports");
    assert_eq!(
        dkv2.floor(),
        tail.base,
        "the import inherits the floor at the tail's base"
    );
    assert_eq!(dkv2.len, upto);
    let (cache2, logits2) = h.fresh_primed(&committed2, ctx);
    let prompt2: Vec<u32> = committed2
        .iter()
        .copied()
        .chain(suffix2.iter().copied())
        .collect();
    let mut restored2 = h
        .model
        .glm5_spec_session_from_restored(
            &h.engine,
            cache2,
            &committed2,
            suffix2,
            &logits2,
            dkv2,
            ctx,
            None,
        )
        .expect("session restored from the floor-bearing tail");
    let n2 = 12usize;
    let (tape2, drafted2, accepted2, _) = drive_bursts(&h, &mut restored2, &prompt2, k, n2, 5, &[]);
    assert!(drafted2 > 0, "the re-restored session must draft");
    let expect2 = &tape_plain[j0 + BLOCK..j0 + BLOCK + tape2.len()];
    assert_eq!(
        tape2, expect2,
        "the continuation from the floor-bearing tail must be plain decode's continuation"
    );
    println!(
        "gate 20 PASS: cold-drafter restore == plain bytes ({accepted1}/{drafted1} accepted), \
         floor tail [{}, {upto}) re-imports with floor {} and continues byte-identical \
         ({accepted2}/{drafted2} accepted)",
        tail.base, tail.base
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 21 — PENALIZED SESSIONS REFUSE DEMOTION (revuto finding on the lane/spec-exclusions
// re-land, 2026-09-03): `glm5_spec_into_demoted`'s flush is one plain `decode_step` +
// `argmax(&logits)` on the boundary row — no penalty pass. Demoting a penalized greedy
// session (`sampling: None, pen: Some`) would silently emit that one token unpenalized,
// exactly the failure class `glm5_penalty_admit`'s refusal exists to prevent and the reason
// dspark keeps penalized-greedy off its own spec route structurally. `demote_eligible` must
// read `pen` as well as `sampling`, and the handoff must refuse loudly, the same shape gate 9
// (glm5_spec_session_gpu.rs) already pins for sampled sessions.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_penalized_greedy_session_refuses_demotion() {
    let _gpu = gpu_guard();
    let h = Harness::new("g19");
    let prompt = tokens(PROMPT, 0xA11CE);
    let max_new = 12usize;
    let cfg = penalty_cfg(0.0, 0);
    let ctx = prompt.len() + max_new + K + 8;

    let _arm = PenaltyArm::arm();
    let mut sess = h
        .model
        .glm5_spec_session_new(&h.engine, &prompt, ctx, Some(spec_sampling_of(&cfg)))
        .expect("penalized greedy glm5 dflash spec session");
    let _ = drive_bursts(&h, &mut sess, &prompt, K, max_new, 3, &[]);

    assert!(
        !sess.demote_eligible(),
        "a penalized greedy session must NOT be demotion-eligible: the flush's plain \
         argmax carries no penalty pass and would silently drop the request's penalties"
    );

    let err = match h.model.glm5_spec_into_demoted(&h.engine, sess) {
        Ok(_) => panic!(
            "penalized greedy demote must refuse loudly, not silently emit an unpenalized \
             token"
        ),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("sampled") || err.to_string().to_lowercase().contains("penal"),
        "the refusal should name why a penalized session stays on spec, got: {err}"
    );

    println!(
        "gate 21 PASS: penalized greedy session refuses demotion by name (demote_eligible() \
         reads pen as well as sampling)"
    );
}
