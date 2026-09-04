//! MOE-SLRU-PLAN §B.3 / §D.2 for the safetensors **NVFP4** expert class — the glm5_next
//! (GLM-5.3-Flash) placement gate.
//!
//! WHY THIS GATE EXISTS. The minted GLM-5.3-Flash NVFP4 artifact is 190.7 GB against 192 GB of
//! VRAM on the 2x96GB serving box, and 92.3% of it (175.9 GB) is routed-expert mass
//! (`research/glm53-flash-bringup-20260827/PLACEMENT-RECEIPT.md`). It cannot be fully resident
//! beside a 1,048,576-token KV plane, so glm5_next MUST serve through `moe_cache`'s SLRU expert
//! residency tier. That tier's whole licence to exist is the §B.3 property:
//!
//!   a cache HIT and a cache MISS feed `qmatvec_view` the SAME block bytes — the only
//!   difference is whether the `memcpy_htod` ran.
//!
//! If that holds, serving a hot subset is byte-for-byte the same program as staging every routed
//! expert every token, and residency is a pure memory-placement decision with no numeric cost.
//! `moe_cache.rs` has always CLAIMED this, and there IS a §D.2 gate for it — but read what it
//! covers. `src/bin/kernel_check.rs:8262` (`d2-cache-bit-identity`) stages ONE expert of a real
//! 35B GGUF into a scratch and into a slot and compares one `qmatvec_view`. Its dtype match arm
//! accepts `IQ3_S | IQ4_XS | Q6_K | Q8_0` and `cells.skip(...)`s everything else, so **NVFP4 has
//! always fallen out of it** — and it needs a multi-GB checkpoint at a hardcoded path, so it does
//! not run in CI. The property was therefore pinned for GGUF k-quant blocks, at one block, out of
//! band. It was never pinned for the NVFP4 class, which is the class glm5_next actually is, and
//! which carries two things a k-quant block does not:
//!
//!   * a **repacked** block — modelopt `weight` (U8 packed e2m1) + per-16 `weight_scale`
//!     (F8_E4M3) are fused by `nvfp4_repack::repack_modelopt_to_gguf` into ONE contiguous
//!     `row_bytes = in_f / 64 * 36` block. That block is what a slot holds.
//!   * a per-expert **macro scale** (`weight_scale_2`) that is NOT in the block. It rides
//!     `HostExps::macros` and is folded post-matmul. Dropping it is a ~3e4x error that is fluent
//!     and invisible (measured garbage, 2026-07-16).
//!
//! So the question this gate answers is narrow and load-bearing: does residency stay bit-exact
//! when the staged bytes are a repacked NVFP4 block AND part of the expert's value lives outside
//! the cached bytes entirely?
//!
//! THE COMPARISON IS UNUSUALLY CLEAN ON THIS ARCH, and that is deliberate. glm5_next is denied
//! every fused/device-dispatch MoE arm by construction — `moe_ffn_pairs`, `moe_ffn_dev` and the
//! grouped-decode pair all require `cfg.sigmoid_router().is_none()` (glm5_next is a sigmoid
//! `noaux_tc` router) and `!cfg.swiglu_clamped_at(il)` (glm5_next's PRE-clamped SwiGLU has no
//! fused twin), and the macro-carrying banks are denied again by `no_exp_macros`. Both arms
//! therefore run the SAME per-expert sequential `qmatvec_view` loop and differ in exactly one
//! thing: whether the block was already resident. This is provenance-only — the strongest form
//! of the §B.3 claim, with no dispatch-class difference hiding inside it. Equality here is
//! asserted BIT-EXACT (`f32::to_bits`), not within a tolerance.
//!
//! THE TWO ARMS ARE THE PRODUCT QUESTION, not a flag A/B: `FullyResident` (every routed expert
//! in a device-resident slab — what a small model gets and what glm5_next cannot have) versus
//! `SlruResidency` (host-resident banks, bounded GPU hot set — what glm5_next must serve on).
//!
//! NON-VACUITY IS ENFORCED, NOT ASSUMED (the wiring-assertions-match-prose law), and it EARNED
//! its keep: the first run of this gate failed on it. Three ways:
//!   * `moe_cache_stats()` must report hits > 0 AND misses > 0 on the SLRU arm.
//!   * `MEMRA_MOE_SLOTS` is pinned BELOW the layer's live block count, so evictions are forced
//!     and misses RECUR rather than happening once during warm-up. The gate asserts that too.
//!   * the fully-resident arm must report `moe_cache_stats() == None` — proof the arms differ.
//!
//! AND THE PIN THAT MAKES THE SLRU ARM REAL IS ITSELF A FINDING. `MEMRA_MOE_RESIDENT=0` is
//! required on that arm because the resident planner is RESIDENT-IF-FITS by default: on the
//! first run it slabbed this fixture's ~0.00 GB expert set, no SLRU was ever built, and the arms
//! were secretly the same program. That is not a fixture quirk — it is the same decision the
//! planner will make on the real 2x96GB box, where per-stage expert mass is compared against
//! free VRAM that has NOT yet had a 1,048,576-token KV plane taken out of it. The placement
//! receipt carries the recommended pins for that reason.
//!
//! AND THE GATE IS MUTATION-BOUND. `the_residency_gate_actually_binds` corrupts the model two
//! ways and requires the comparison to break each time: one byte flipped inside a cached NVFP4
//! block, and one per-expert macro scale perturbed. The macro mutation is the one that matters —
//! it is the only part of an expert's value that residency does NOT carry, so a gate that could
//! not see it would be blind to exactly the failure this class is prone to.
//!
//! FIXTURE SHAPE. Real glm5_next routing constants (`routed_scaling_factor` 2.5,
//! `norm_topk_prob`, sigmoid `noaux_tc`, 1 shared expert, PRE-clamped SwiGLU at limit 10.0),
//! shrunk only in width. `moe_intermediate_size` is 64 rather than the router gate's 32 because
//! NVFP4 requires `in_f % 64 == 0` on every projection and `down`'s `in_f` IS that dimension.
//!
//! SCOPE — what this gate does NOT prove. It runs a 2-layer fixture, not the 190.7 GB artifact
//! (which has never been loaded; it is not on this rig). It proves the residency PROGRAM is
//! exact for the NVFP4 macro-carrying class; it proves nothing about the hot-expert mass, the
//! real checkpoint's name resolution, or any throughput figure.
//!
//! GPU-gated (`#[ignore]`); rig law = exactness only, never timing. Run under
//! `flock /tmp/memra-5090.lock` with `NVIDIA_TF32_OVERRIDE=0` and `-- --ignored`.

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::{MlpPlan, ModelPlan};
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, LayerTensor, OutputHead, TensorContract, TensorId,
    TensorMatch,
};
use memra_reference::{ReferenceTensor, deterministic_fixture};
use std::borrow::Cow;
use std::collections::BTreeMap;

const VOCAB: u32 = 32;

/// GLM-5.3-Flash's real routing shape, shrunk only in width.
const EXPERTS: usize = 8;
const TOP_K: usize = 3;
const SCALING: f32 = 2.5;
const NORM: bool = true;

/// `moe_intermediate_size`. NVFP4 requires `in_f % 64 == 0` on EVERY projection, and `down`'s
/// `in_f` is this dimension — 32 (the router gate's value) would make the bank unrepackable.
const MOE_FF: usize = 64;

/// Live routed blocks in the MoE layer: `EXPERTS * {gate, up, down}`.
const LIVE_BLOCKS: usize = EXPERTS * 3;

/// SLRU slots for the cached arm. Pinned BELOW `LIVE_BLOCKS` so the cache cannot hold the layer
/// and evictions recur — the construction that keeps `misses > 0` true after warm-up instead of
/// only during it. `MoeCache::new` floors the slot count at 8, so 8 is the tightest legal pin.
const SLOTS: usize = 8;

/// Per-expert `weight_scale_2` macro scales. Deliberately NOT all 1.0: `stacked_macros` returns
/// `None` for an all-ones vector, which would drop the macro plane entirely and leave the macro
/// mutation below unable to bind. Deliberately near 1.0 so the fixture stays numerically sane.
fn macro_scales() -> Vec<f32> {
    (0..EXPERTS).map(|e| 0.85 + 0.03 * e as f32).collect()
}

/// GPU tests serialize on one device, and these arms additionally mutate PROCESS-GLOBAL env vars
/// (`MEMRA_MOE_CACHE`, `MEMRA_MOE_SLOTS`) that the engine reads at load time and per forward.
fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// cuBLASLt f32 compute rides TF32 on Blackwell by default — right for serving, wrong for an
/// exactness gate. The driver reads this at CUDA init, so it must be set before the first
/// `Engine::new` in the process.
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

/// A glm5_next trunk with one dense layer and one routed-MoE layer, through the real
/// `HfConfig`/`ModelConfig` path and the real glm5_next model pack — `HybridModel` compiles the
/// plan from `src.config()`, so a hand-built `ModelPlan` could not reach it.
///
/// `head_dim` is 128 because that is the only width `memra_kda_scan_s128` is instantiated for.
/// The MLA/DSA fields are required by the glm5_next config parser and are inert here: no layer in
/// `layer_types` selects them.
fn mini_config_json() -> String {
    format!(
        r#"{{
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
      "layer_types": ["linear_attention", "linear_attention"],
      "mlp_layer_types": ["dense", "sparse"],
      "first_k_dense_replace": 1,
      "indexer_types": ["full", "full"],
      "linear_attn_config": {{
        "num_heads": 1,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [0, 1],
        "full_attn_layers": []
      }},
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
      "n_routed_experts": {EXPERTS},
      "num_experts_per_tok": {TOP_K},
      "moe_intermediate_size": {MOE_FF},
      "n_shared_experts": 1,
      "scoring_func": "sigmoid",
      "topk_method": "noaux_tc",
      "routed_scaling_factor": {SCALING},
      "norm_topk_prob": {NORM},
      "n_group": 1,
      "topk_group": 1,
      "head_dim": 0,
      "attention_bias": false,
      "moe_router_dtype": "float32",
      "dtype": "bfloat16"
    }}"#
    )
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

/// The three stacked routed-expert slabs — the tensors that become NVFP4 blocks in SLRU slots.
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

/// What to break. Each must make the residency comparison fail, or the gate is decorative.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mutation {
    /// Baseline: the honest model.
    None,
    /// One byte flipped INSIDE a cached NVFP4 block — the bytes residency actually carries.
    ExpertBlockByte,
    /// One per-expert `weight_scale_2` macro perturbed — the part of an expert's value that
    /// lives OUTSIDE the cached bytes. The failure mode this class is prone to.
    MacroScale,
}

impl Mutation {
    fn label(self) -> &'static str {
        match self {
            Mutation::None => "none (baseline)",
            Mutation::ExpertBlockByte => "one byte inside a cached NVFP4 block",
            Mutation::MacroScale => "one per-expert weight_scale_2 macro",
        }
    }
}

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
    ggml_type: GgmlType,
}

/// Serves the fixture under the contract's ggml names. Must answer `config()`:
/// `HybridModel::load_from_source*` compiles the plan from it.
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

/// Build the source. Routed-expert banks are minted as REAL NVFP4 blocks in memra's internal
/// `block_nvfp4` layout (`f32_to_nvfp4`, 36 B per 64 elements) — the same layout
/// `repack_modelopt_to_gguf` produces from a modelopt checkpoint, so the slots hold exactly what
/// they would hold for the real artifact. Each bank additionally gets its `<stem>.scale` sibling
/// (F32, one per expert), which is how `HostExps::stacked_macros` finds the macro plane.
fn fixture_source(
    config: &ModelConfig,
    plan: &ModelPlan,
    weights: &BTreeMap<TensorId, ReferenceTensor>,
    mutation: Mutation,
) -> FixtureSource {
    let contract = TensorContract::for_plan(
        plan,
        CheckpointDialect::Gguf,
        ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        },
    )
    .expect("contract for the mini glm5_next plan");

    let mut tensors: BTreeMap<String, OwnedTensor> = BTreeMap::new();
    let mut bank_stems: Vec<String> = Vec::new();

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

        let expert_bank = is_expert_bank(&req.id);
        let (mut bytes, ggml_type) = if expert_bank {
            (
                memra_gguf::nvfp4_repack::f32_to_nvfp4(&tensor.data),
                GgmlType::NVFP4,
            )
        } else {
            (
                tensor.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                GgmlType::F32,
            )
        };

        // Corrupt one byte of ONE bank, deep enough inside to land in a code nibble rather than
        // a sub-block scale — either would bind, but a code byte is the honest "the bytes the
        // cache moved are wrong" mutation.
        if expert_bank
            && mutation == Mutation::ExpertBlockByte
            && matches!(
                req.id,
                TensorId::Layer {
                    tensor: LayerTensor::MoeExpertGateBank,
                    ..
                }
            )
        {
            let victim = bytes.len() / 2 + 7;
            bytes[victim] ^= 0xFF;
        }

        let names = match req.match_mode {
            TensorMatch::OneOf => &req.names[..1],
            TensorMatch::All => req.names.as_slice(),
        };
        for name in names {
            if expert_bank && let Some(stem) = name.strip_suffix(".weight") {
                bank_stems.push(stem.to_string());
            }
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

    assert_eq!(
        bank_stems.len(),
        3,
        "expected exactly gate/up/down routed-expert banks, got {bank_stems:?}"
    );

    // The macro plane. `HostExps::stacked_macros` reads `<stem>.scale` as an F32 vector of
    // n_expert values and folds it post-matmul; it returns None if every value is 1.0.
    for stem in &bank_stems {
        let mut macros = macro_scales();
        if mutation == Mutation::MacroScale && stem.ends_with("ffn_gate_exps") {
            macros[EXPERTS / 2] *= 1.5;
        }
        tensors.insert(
            format!("{stem}.scale"),
            OwnedTensor {
                bytes: macros.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ne: vec![EXPERTS as u64],
                ggml_type: GgmlType::F32,
            },
        );
    }

    FixtureSource {
        config: config.clone(),
        tensors,
    }
}

fn residency_fixture(plan: &ModelPlan) -> BTreeMap<TensorId, ReferenceTensor> {
    deterministic_fixture(plan)
        .expect("deterministic glm5_next fixture")
        .weights
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

/// The two placements under comparison. This IS the product question for glm5_next: the 190.7 GB
/// artifact cannot be fully resident beside a 1M KV plane, so serving it means running
/// `SlruResidency` — and that must be the same program as `FullyResident`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Regime {
    /// What a small model gets today and what a big one cannot have: every routed expert in a
    /// device-resident slab (`dev_exps`), no cache, no staging.
    FullyResident,
    /// What glm5_next must serve on: host-resident expert banks with a bounded GPU hot set.
    ///
    /// `MEMRA_MOE_RESIDENT=0` is REQUIRED here and is not a detail. The resident planner is
    /// RESIDENT-IF-FITS by default (`docs/FLAGS.md` `MEMRA_MOE_RESIDENT`), and this fixture's
    /// experts are ~0.00 GB against a 23 GB card — so without the pin the planner slabs them,
    /// the SLRU is never built, and this arm silently becomes a copy of `FullyResident`. The
    /// non-vacuity assertions below exist because that is exactly what happened on the first
    /// run of this gate. It is also the trap waiting on the real 2x96GB box, where per-stage
    /// expert mass evaluates as "fits" against free VRAM that has not yet had 1M of KV taken
    /// out of it — see the placement receipt's recommended pins.
    SlruResidency,
}

impl Regime {
    fn label(self) -> &'static str {
        match self {
            Regime::FullyResident => "fully-resident slabs",
            Regime::SlruResidency => "SLRU hot-set residency",
        }
    }
}

/// One arm's whole observable behaviour: every logits row it produced, and what the cache did.
struct Arm {
    rows: Vec<(String, Vec<f32>)>,
    stats: Option<(u64, u64, u64, usize)>,
}

/// Run the full workload under one placement regime. A FRESH `Engine` and a FRESH model load per
/// arm are mandatory, not hygiene: the resident-vs-SLRU decision and `MEMRA_MOE_PINNED`
/// (pinned-vs-paged host expert buffers) are both taken at LOAD time, and the SLRU itself is
/// per-Engine. Reusing either would silently compare an arm against itself.
fn run_arm(regime: Regime, mutation: Mutation) -> Arm {
    force_true_f32();
    // SAFETY: every caller holds `gpu_guard()`, which is the only thing in this binary that
    // touches these vars, and no other thread is running engine code.
    unsafe {
        match regime {
            Regime::FullyResident => {
                std::env::set_var("MEMRA_MOE_CACHE", "0");
                std::env::set_var("MEMRA_MOE_RESIDENT", "1");
                std::env::set_var("MEMRA_MOE_SLOTS", "0");
            }
            Regime::SlruResidency => {
                std::env::set_var("MEMRA_MOE_CACHE", "1");
                std::env::set_var("MEMRA_MOE_RESIDENT", "0");
                std::env::set_var("MEMRA_MOE_SLOTS", SLOTS.to_string());
            }
        }
        // MEMRA_MOE_GROUPED_PREFILL went DEFAULT ON on 2026-08-29. It is slab-only and fails
        // closed on the SLRU placement, so with it live the two regimes here would run
        // DIFFERENT dispatch classes at t > 16 (grouped f16 GEMM vs the staged loop) and the
        // §B.3 bit-identity this gate exists to pin would be comparing programs, not
        // placements. The gate measures placement provenance in ISOLATION; pin the grouped
        // seam off in both regimes (`glm5_moe_grouped_prefill_gpu.rs` owns the grouped arm's
        // own placement gates, including the SLRU fail-closed one).
        std::env::set_var("MEMRA_MOE_GROUPED_PREFILL", "0");
    }

    let config = mini_config();
    let plan = mini_plan(&config);
    let weights = residency_fixture(&plan);
    let source = fixture_source(&config, &plan, &weights, mutation);
    let engine = Engine::new(0).expect("CUDA engine on device 0");
    let model = HybridModel::load_from_source_without_mtp(&engine, &source)
        .expect("mini glm5_next model loads from the contract");

    let mut rows: Vec<(String, Vec<f32>)> = Vec::new();

    // Prefill widths: several dispatch tiers, all of which must agree across the arms.
    for &n in &[1usize, 3, 8, 65] {
        let ids = tokens(n, 0x6E5_1DE0 ^ n as u64);
        let got = model.forward(&engine, &ids).expect("GPU routed prefill");
        rows.push((format!("prefill T={n}"), got));
    }

    // Prime + decode. Decode (t=1) is the arm that certainly rides the per-expert sequential
    // staged loop, so this is what guarantees the cache is exercised rather than bypassed.
    let prompt = 6usize;
    let steps = 6usize;
    let ids = tokens(prompt + steps, 0x6E5_1DE0_DEC0);
    let mut cache = memra_engine::cache::Cache::new_planned(&engine, &model.cfg, &plan, 64)
        .expect("cache for the mini glm5_next model");
    let (primed, _seed, _hiddens) = model
        .prime_cache(&engine, &ids[..prompt], &mut cache, 0)
        .expect("GPU routed prime");
    rows.push(("prime last row".to_string(), primed));
    for step in 0..steps {
        let got = model
            .decode_step(&engine, ids[prompt + step], &mut cache)
            .expect("GPU routed decode step");
        rows.push((format!("decode step {step}"), got));
    }

    let stats = engine.moe_cache_stats();
    Arm { rows, stats }
}

/// Bit-exact, not tolerant: §B.3 claims the SAME bytes reach the same kernel, so the only honest
/// bar is identical IEEE-754 bit patterns.
fn first_bit_difference(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    assert_eq!(a.len(), b.len(), "compared rows differ in length");
    a.iter()
        .zip(b)
        .enumerate()
        .find(|(_, (x, y))| x.to_bits() != y.to_bits())
        .map(|(i, (x, y))| (i, *x, *y))
}

fn compare_arms(resident: &Arm, cached: &Arm) -> usize {
    assert_eq!(
        resident.rows.len(),
        cached.rows.len(),
        "arms produced different row counts"
    );
    let mut differing = 0;
    for ((ln, lv), (rn, rv)) in resident.rows.iter().zip(cached.rows.iter()) {
        assert_eq!(ln, rn, "arms ran different workloads");
        if first_bit_difference(lv, rv).is_some() {
            differing += 1;
        }
    }
    differing
}

/// GATE — the §B.3 property for the NVFP4 macro-carrying expert class.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn the_residency_path_is_bit_identical_to_stage_every_token() {
    let _gpu = gpu_guard();

    let resident = run_arm(Regime::FullyResident, Mutation::None);
    let cached = run_arm(Regime::SlruResidency, Mutation::None);

    // NON-VACUITY 1: the fully-resident arm must genuinely have no cache.
    assert!(
        resident.stats.is_none(),
        "the {} arm still built an SLRU ({:?}) — the arms are not distinct",
        Regime::FullyResident.label(),
        resident.stats
    );

    // NON-VACUITY 2: the SLRU arm must have actually hit AND missed. A cache that was never
    // built (the resident planner slabbed the experts instead), or one that never evicted,
    // would make the equality below trivially true. The first run of this gate failed exactly
    // here: RESIDENT-IF-FITS slabbed a 0.00 GB expert set and no SLRU was ever constructed.
    let (hits, misses, staged, slots) = cached.stats.expect(
        "the SLRU arm never built a cache — the resident planner slabbed the experts \
         (MEMRA_MOE_RESIDENT=0 is what forces the SLRU path) and the gate would be vacuous",
    );
    println!(
        "[residency] slots={slots} hits={hits} misses={misses} staged={staged}B \
         (live routed blocks = {LIVE_BLOCKS}, slots pinned to {SLOTS} to force eviction)"
    );
    assert!(
        hits > 0,
        "SLRU recorded no hits — residency was never exercised"
    );
    assert!(
        misses > 0,
        "SLRU recorded no misses — nothing was ever staged, so hit-vs-miss is untested"
    );
    assert!(
        slots < LIVE_BLOCKS,
        "slots ({slots}) >= live blocks ({LIVE_BLOCKS}): the layer fits, so no eviction ever \
         happens and the gate degenerates into a warm-up test"
    );

    // THE PROPERTY.
    for ((name, want), (_, got)) in resident.rows.iter().zip(cached.rows.iter()) {
        if let Some((i, w, g)) = first_bit_difference(want, got) {
            panic!(
                "{name}: {} is NOT bit-identical to {} — element {i} \
                 {w:e} (0x{:08x}) vs {g:e} (0x{:08x})",
                Regime::SlruResidency.label(),
                Regime::FullyResident.label(),
                w.to_bits(),
                g.to_bits()
            );
        }
    }
    println!(
        "[residency] {} rows bit-identical: {} vs {}",
        resident.rows.len(),
        Regime::FullyResident.label(),
        Regime::SlruResidency.label()
    );
}

/// MUTATION CHECK — the gate above must be able to SEE a broken model. Each mutation is applied
/// to the cached arm only; the resident arm stays honest, so a mutation that residency cannot
/// carry correctly shows up as a row mismatch.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn the_residency_gate_actually_binds() {
    let _gpu = gpu_guard();

    let resident = run_arm(Regime::FullyResident, Mutation::None);
    let total_rows = resident.rows.len();

    let mutations = [Mutation::ExpertBlockByte, Mutation::MacroScale];
    let mut caught = 0usize;
    for m in mutations {
        let broken = run_arm(Regime::SlruResidency, m);
        // The mutated arm must still be a REAL SLRU arm, or this degenerates into
        // resident-vs-resident and stops testing what it claims to.
        assert!(
            broken.stats.is_some_and(|(h, mi, _, _)| h > 0 && mi > 0),
            "mutation {m:?}: the SLRU arm did not hit and miss ({:?}) — the mutation check \
             would not be exercising residency at all",
            broken.stats
        );
        let differing = compare_arms(&resident, &broken);
        println!(
            "[mutation] {:<40} -> {differing}/{total_rows} rows differ",
            m.label()
        );
        if differing > 0 {
            caught += 1;
        } else {
            println!("[mutation] NOT CAUGHT: {}", m.label());
        }
    }
    println!("[mutation] caught {caught}/{} mutants", mutations.len());
    assert_eq!(
        caught,
        mutations.len(),
        "a mutated model still compared equal — the residency gate does not bind"
    );
}

/// CPU-only companion: the fixture must actually be the shape the GPU gates assume, and it must
/// carry a LIVE macro plane. Runs without CUDA so a fixture that drifted (all-ones macros, a
/// non-NVFP4 bank, slots that no longer force eviction) fails on any machine rather than
/// silently turning the GPU gates into tautologies.
#[test]
#[allow(clippy::assertions_on_constants)] // allow: const pins; fail the suite loudly if the fixture constants drift out of the gated window
fn the_fixture_is_the_nvfp4_macro_carrying_shape() {
    let config = mini_config();
    let plan = mini_plan(&config);
    let weights = residency_fixture(&plan);
    let source = fixture_source(&config, &plan, &weights, Mutation::None);

    assert_eq!(
        config.sigmoid_router(),
        Some((SCALING, NORM)),
        "the fixture must route with glm5_next's sigmoid noaux_tc program — every dispatch \
         predicate that keeps glm5_next out of the fused arms keys off this accessor"
    );

    let moe_layer = plan
        .layers
        .iter()
        .find(|l| matches!(l.mlp, MlpPlan::Moe(_)))
        .expect("the mini plan must carry a routed-MoE layer");
    assert!(
        config.swiglu_clamped_at(moe_layer.index),
        "the MoE layer must report a live SwiGLU clamp — that is what denies glm5_next the \
         fused grouped-decode arm and forces the per-expert staged loop this gate compares"
    );

    let mut banks = 0usize;
    for (name, t) in &source.tensors {
        if !name.ends_with("_exps.weight") {
            continue;
        }
        banks += 1;
        assert_eq!(
            t.ggml_type,
            GgmlType::NVFP4,
            "{name} must be served as NVFP4 blocks, not a k-quant stand-in"
        );
        let (in_f, out_f, n_expert) = (t.ne[0] as usize, t.ne[1] as usize, t.ne[2] as usize);
        assert_eq!(n_expert, EXPERTS, "{name} expert count");
        assert_eq!(
            in_f % 64,
            0,
            "{name}: NVFP4 needs in_f % 64 == 0, got {in_f}"
        );
        // The block layout a slot holds: row_bytes = in_f / 64 * 36.
        let row_bytes = in_f / 64 * 36;
        assert_eq!(
            t.bytes.len(),
            n_expert * out_f * row_bytes,
            "{name}: repacked size != n_expert * out_f * (in_f/64*36)"
        );

        let scale = source
            .tensors
            .get(&name.replace(".weight", ".scale"))
            .unwrap_or_else(|| panic!("{name} has no .scale macro sibling"));
        let macros: Vec<f32> = scale
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(macros.len(), EXPERTS, "{name}.scale length");
        assert!(
            macros.iter().any(|&m| m != 1.0),
            "{name}.scale is all ones — stacked_macros would return None and the macro plane \
             would not exist, leaving the macro mutation unable to bind"
        );
    }
    assert_eq!(banks, 3, "expected gate/up/down routed-expert banks");

    assert!(
        SLOTS < LIVE_BLOCKS,
        "SLOTS ({SLOTS}) must stay below the {LIVE_BLOCKS} live routed blocks or evictions \
         never happen"
    );

    // The mutations must actually change the bytes the model loads, or the GPU mutation check
    // cannot bind for a reason that has nothing to do with residency.
    for m in [Mutation::ExpertBlockByte, Mutation::MacroScale] {
        let broken = fixture_source(&config, &plan, &weights, m);
        let changed = source
            .tensors
            .iter()
            .filter(|(name, t)| {
                broken
                    .tensors
                    .get(*name)
                    .is_some_and(|b| b.bytes != t.bytes)
            })
            .count();
        assert_eq!(
            changed, 1,
            "mutation {:?} should perturb exactly one tensor, perturbed {changed}",
            m
        );
    }
}
