//! memra#1225: a gemma4_moe model's logits must not depend on where its experts were placed at
//! load.
//!
//! WHY. With the routed experts resident (the fits-VRAM slab), `gemma4_moe` runs the q8 decode
//! and verify kernels (t < 16) and the pairs arm (t >= 16, the int8-MMA expert GEMM on sm_120a)
//! over the slab's pointer row, and the layer tail emits the expert input pre-quantized. Without
//! a slab, before memra#1225, every MoE call ran a host-staged loop instead: one synchronous H2D
//! per (token, expert, projection) and the f32 `qmatvec_f32` chain, another numeric program for
//! prefill AND decode. On the RTX 5090, gemma-4-26B-A4B served a different greedy text from the
//! same request depending only on how much memory was free at load. The fix runs the resident
//! kernels on both placements: the SLRU placement reads the same expert bytes from cache slots
//! through the same `[3 * n_expert]` pointer table (`MoeSlotCache::acquire_pinned`, memra#1217).
//!
//! WHAT THIS GATE HOLDS, BIT-EXACT (`f32::to_bits`). A two-layer gemma4_moe model (softmax
//! router with the per-expert output scale, GELU experts, Q4_0 routed experts with the real
//! model's ragged down width of 704, 128 experts, top 8) runs the same workload in these
//! placements:
//!   * `Resident`: the slab, the reference.
//!   * `SlruRoomy`: an auto-sized cache; the one-shot prewarm makes every layer fully cache
//!     resident, so every call reads the cache's own full-layer row.
//!   * `SlruMid`: room for one layer's 384 routed blocks, not both layers'; the layer the
//!     prewarm skipped admits each call's active set pinned (133 experts fit, so it never
//!     splits).
//!   * `SlruTight`: exactly the floor, the 360 blocks a verify-width call can need
//!     (`min(n_expert, 15 * top_k)` experts), below one layer: small-t calls run pinned, and a
//!     prefill that routes to more than 120 experts runs in contiguous CSR runs of experts. (This
//!     fixture's widest prefills reach 118 to 121 of the 128 experts, so the floor is the only
//!     slot count that splits one; the arm asserts that it did.)
//!
//! Every logits row (prefill at several widths, a prime and decode steps, and two interleaved
//! sessions) must equal the resident arm's.
//!
//! BELOW THE FLOOR. A cache that cannot pin one verify-width call's experts keeps the staged
//! program for EVERY call of that boot (`SlruBelow`, 300 slots): the choice is made per boot,
//! never per call, so one request can never cross programs. That arm asserts no SLRU mode ran.
//!
//! THE q8 SUB-ARM. Run the same binary with `MEMRA_GEMMA_MOE_MMA=0` and the pairs arm takes its
//! decode-once dp4a program instead of the int8 MMA; it splits into runs the same way.
//!
//! NON-VACUITY. Each SLRU arm asserts the modes it exists for actually ran
//! (`memra_engine::moe_slru_pairs_calls`), and that its cache staged (and, except the roomy arm,
//! whose full-layer rows count no hit, hit). Before memra#1225 every SLRU arm differs at every
//! row, 20 of 20; that red run is in the PR's receipts.
//!
//! GPU-gated (`#[ignore]`); exactness only, never timing. Run under the 5090 lease with
//! `NVIDIA_TF32_OVERRIDE=0` and `-- --ignored --test-threads=1`.

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::ModelPlan;
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, LayerTensor, OutputHead, TensorContract, TensorId,
    TensorMatch,
};
use memra_reference::{ReferenceTensor, deterministic_fixture};
use std::borrow::Cow;
use std::collections::BTreeMap;

/// Ids enough that the widest prefills reach 118 to 125 of the 128 experts (32 ids: about 100).
const VOCAB: u32 = 512;
const MAX_CTX: usize = 1024;
/// Every gemma4 q8_1 row kernel covers a row in 128-column quads, and the int8-MMA expert GEMM
/// needs `n_embd % 256 == 0` (gemma-4-26B-A4B: 2816).
const WIDTH: usize = 256;
const EXPERTS: usize = 128;
const TOP_K: usize = 8;
/// gemma-4-26B-A4B's `moe_intermediate_size`: the down projection's ragged k (704 is not a
/// multiple of 256), whose MMA k-walk reads past the last row of an expert.
const MOE_FF: usize = 704;
/// Routed blocks per MoE layer: `EXPERTS * {gate, up, down}`.
const LAYER_BLOCKS: usize = EXPERTS * 3;
/// The blocks one call below the prefill width can need at once: `min(n_expert, 15 * top_k)`
/// experts. A cache below this keeps the staged program for the whole boot.
const SMALL_T_BLOCKS: usize = 3 * if EXPERTS < 15 * TOP_K {
    EXPERTS
} else {
    15 * TOP_K
};
const MID_SLOTS: usize = 400;
const TIGHT_SLOTS: usize = SMALL_T_BLOCKS;
const BELOW_SLOTS: usize = 300;

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// cuBLASLt f32 rides TF32 on Blackwell unless this is set before CUDA init.
fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call has been made yet; call_once serializes the write.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
    });
}

fn mini_config() -> ModelConfig {
    ModelConfig::from_hf(&HfConfig::parse(&format!(
        r#"{{"model_type":"gemma4","text_config":{{"model_type":"gemma4_text",
        "num_hidden_layers":2,"hidden_size":{WIDTH},"num_attention_heads":2,
        "num_key_value_heads":1,"num_global_key_value_heads":1,"head_dim":256,
        "global_head_dim":512,"intermediate_size":{WIDTH},"moe_intermediate_size":{MOE_FF},
        "num_experts":{EXPERTS},"top_k_experts":{TOP_K},"vocab_size":{VOCAB},
        "max_position_embeddings":{MAX_CTX},"rms_norm_eps":0.000001,
        "sliding_window":64,"final_logit_softcapping":30,"hidden_activation":"gelu_pytorch_tanh",
        "layer_types":["sliding_attention","full_attention"],
        "rope_parameters":{{"full_attention":{{"rope_theta":1000000,
        "partial_rotary_factor":0.25}},"sliding_attention":{{"rope_theta":10000}}}}}}}}"#
    )))
}

fn mini_plan(config: &ModelConfig) -> ModelPlan {
    let pack = memra_gguf::model_packs::for_config(config).expect("a pack matches the mini config");
    assert_eq!(
        pack.family, "gemma4_moe",
        "the mini config is the MoE gemma"
    );
    pack.compile_plan(config)
        .expect("mini gemma4_moe plan compiles")
}

/// See `gemma4_chunked_prime_gpu.rs`: the GGUF contract needs `rope_freqs.weight` for a
/// `PartialRotary` rope, which `deterministic_fixture` only synthesizes for checkpoint factors.
fn add_partial_rotary_factors(plan: &ModelPlan, weights: &mut BTreeMap<TensorId, ReferenceTensor>) {
    use memra_gguf::model_plan::{AttentionPlan, RopeFactors};
    if weights.contains_key(&TensorId::RopeFactors) {
        return;
    }
    let mut table: Option<Vec<f32>> = None;
    for layer in &plan.layers {
        let rope = match &layer.attention {
            AttentionPlan::Full(a) | AttentionPlan::SlidingWindow { attention: a, .. } => &a.rope,
            _ => continue,
        };
        if let RopeFactors::PartialRotary { factor } = rope.factors {
            let width = rope.dimensions as usize / 2;
            let keep = (width as f32 * factor.clamp(0.0, 1.0)).round() as usize;
            let t: Vec<f32> = (0..width)
                .map(|i| if i < keep { 1.0 } else { 1.0e30 })
                .collect();
            match &table {
                Some(prev) if prev.len() >= t.len() => {}
                _ => table = Some(t),
            }
        }
    }
    if let Some(t) = table {
        let n = t.len();
        weights.insert(
            TensorId::RopeFactors,
            ReferenceTensor::new(vec![n], t).expect("rope factors tensor"),
        );
    }
}

/// ggml's reference Q4_0 encoder: per 32 values, `d = max / -8` from the signed value of largest
/// magnitude, `q = min(15, trunc(x / d + 8.5))`, value j in the low nibble of byte j and value
/// j + 16 in the high nibble.
fn f32_to_q4_0(data: &[f32]) -> Vec<u8> {
    assert_eq!(data.len() % 32, 0, "Q4_0 encodes whole 32-value blocks");
    let mut out = Vec::with_capacity(data.len() / 32 * 18);
    for blk in data.chunks_exact(32) {
        let max = blk
            .iter()
            .fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&memra_gguf::nvfp4_repack::f32_to_f16_bits(d).to_le_bytes());
        let q = |v: f32| ((v * id + 8.5) as i32).clamp(0, 15) as u8;
        for j in 0..16 {
            out.push(q(blk[j]) | (q(blk[j + 16]) << 4));
        }
    }
    out
}

/// Uniform values in [-0.5, 0.5) from a splitmix64 stream seeded by the tensor id.
fn spread_router(n: usize, id: TensorId) -> Vec<f32> {
    let mut s: u64 = 0x1225_5EED
        ^ match id {
            TensorId::Layer { index, .. } => index as u64,
            _ => 0,
        };
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

fn is_expert_bank(id: &TensorId) -> bool {
    matches!(
        id,
        TensorId::Layer {
            tensor: LayerTensor::MoeExpertGateUpBank
                | LayerTensor::MoeExpertGateBank
                | LayerTensor::MoeExpertUpBank
                | LayerTensor::MoeExpertDownBank,
            ..
        }
    )
}

/// Layer projections the gemma4 q8_1 arms consume as quantized weights; the router reads its
/// weight as f32, and norms, scales and the tied embedding stay F32.
fn is_dense_projection(id: &TensorId, shape: &[u64]) -> bool {
    matches!(id, TensorId::Layer { tensor, .. } if *tensor != LayerTensor::MoeRouter)
        && shape.len() == 2
        && shape[0].is_multiple_of(32)
}

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
    ty: GgmlType,
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
            ggml_type: t.ty,
            ne: t.ne.clone(),
        })
    }
    fn tensor_census(&self) -> Result<memra_gguf::source::TensorCensus, String> {
        Ok(memra_gguf::source::census_from_views(
            self.tensors
                .iter()
                .map(|(name, t)| (name.as_str(), t.ty, t.ne.as_slice(), t.bytes.len() as u64)),
        ))
    }
}

/// The deterministic reference fixture under the GGUF contract's names: routed-expert banks as
/// Q4_0 (gemma-4-26B-A4B's QAT class), other layer projections Q8_0, the rest F32.
fn fixture_source(config: &ModelConfig, plan: &ModelPlan) -> FixtureSource {
    let mut weights = deterministic_fixture(plan)
        .expect("deterministic gemma4_moe fixture")
        .weights;
    add_partial_rotary_factors(plan, &mut weights);
    let contract = TensorContract::for_plan(
        plan,
        CheckpointDialect::Gguf,
        ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        },
    )
    .expect("contract for the mini gemma4_moe plan");
    let mut tensors = BTreeMap::new();
    let mut banks = 0;
    for req in contract
        .requirements
        .iter()
        .filter(|r| r.required || weights.contains_key(&r.id))
    {
        let tensor = weights
            .get(&req.id)
            .unwrap_or_else(|| panic!("reference fixture is missing {:?}", req.id));
        let elements: usize = req.shape.iter().map(|&d| d as usize).product();
        assert_eq!(elements, tensor.data.len(), "fixture {:?} shape", req.id);
        let (bytes, ty) = if matches!(
            req.id,
            TensorId::Layer {
                tensor: LayerTensor::MoeRouter,
                ..
            }
        ) {
            // A spread router: the deterministic fixture's router, with the 32-id vocabulary
            // the first version used, sent a 600-token prefill to about 100 of the 128 experts,
            // below the tight arm's 120 pinnable, so no call split into runs. Pseudo-random
            // weights and a 512-id vocabulary reach 118 to 125.
            let data = spread_router(elements, req.id.clone());
            (
                data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                GgmlType::F32,
            )
        } else if is_expert_bank(&req.id) {
            banks += 1;
            (f32_to_q4_0(&tensor.data), GgmlType::Q4_0)
        } else if is_dense_projection(&req.id, &req.shape) {
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
                    ty,
                },
            );
        }
    }
    assert_eq!(
        banks,
        2 * 2,
        "two MoE layers x the fused gate/up bank and the down bank"
    );
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Placement {
    Resident,
    SlruRoomy,
    SlruMid,
    SlruTight,
    SlruBelow,
}

struct Arm {
    rows: Vec<(String, Vec<f32>)>,
    stats: Option<(u64, u64, u64, usize)>,
    /// `moe_slru_pairs_calls()` deltas over the arm: (layer row, pinned, chunked).
    modes: (u64, u64, u64),
}

fn run_arm(placement: Placement) -> Arm {
    force_true_f32();
    // SAFETY: callers hold `gpu_guard()`; nothing else in this binary touches these vars and no
    // other thread runs engine code.
    unsafe {
        let (resident, slots) = match placement {
            Placement::Resident => ("1", None),
            Placement::SlruRoomy => ("0", None),
            Placement::SlruMid => ("0", Some(MID_SLOTS)),
            Placement::SlruTight => ("0", Some(TIGHT_SLOTS)),
            Placement::SlruBelow => ("0", Some(BELOW_SLOTS)),
        };
        std::env::set_var("MEMRA_MOE_CACHE", "1");
        std::env::set_var("MEMRA_MOE_RESIDENT", resident);
        match slots {
            Some(n) => std::env::set_var("MEMRA_MOE_SLOTS", n.to_string()),
            // Empty is the auto size (`parse_moe_slots`), pinned rather than unset.
            None => std::env::set_var("MEMRA_MOE_SLOTS", ""),
        }
    }
    let modes0 = memra_engine::moe_slru_pairs_calls();
    let config = mini_config();
    let plan = mini_plan(&config);
    let source = fixture_source(&config, &plan);
    let engine = Engine::new(0).expect("CUDA engine on device 0");
    let model = HybridModel::load_from_source_without_mtp(&engine, &source)
        .expect("mini gemma4_moe model loads from the contract");

    let mut rows = Vec::new();
    // Widths across the dispatch tiers: decode (1), verify rows (2..15), and the pairs arm
    // (>= 16). 600 tokens put more than 32 rows in most expert groups.
    for &n in &[1usize, 8, 15, 16, 40, 130, 600, 1000] {
        let ids = tokens(n, 0x1225_0000 ^ n as u64);
        let got = model.forward(&engine, &ids).expect("GPU forward");
        rows.push((format!("forward T={n}"), got));
    }

    // Two sessions on one engine and one cache: A primes and decodes, B's prime lands between
    // A's decode steps, then both decode alternately.
    let new_cache = || {
        memra_engine::cache::Cache::new_planned(&engine, &model.cfg, &plan, MAX_CTX)
            .expect("cache for the mini gemma4_moe model")
    };
    let ids_a = tokens(40 + 6, 0x1225_00A0);
    let ids_b = tokens(97 + 6, 0x1225_00B0);
    let (mut cache_a, mut cache_b) = (new_cache(), new_cache());
    let (primed_a, _, _) = model
        .prime_cache(&engine, &ids_a[..40], &mut cache_a, 0)
        .expect("prime A");
    rows.push(("A prime last row".into(), primed_a));
    for step in 0..6 {
        if step == 2 {
            let (primed_b, _, _) = model
                .prime_cache(&engine, &ids_b[..97], &mut cache_b, 0)
                .expect("prime B");
            rows.push((
                "B prime last row (between A's decode steps)".into(),
                primed_b,
            ));
        }
        let got = model
            .decode_step(&engine, ids_a[40 + step], &mut cache_a)
            .expect("A decode");
        rows.push((format!("A decode {step}"), got));
        if step >= 2 {
            let got = model
                .decode_step(&engine, ids_b[97 + step - 2], &mut cache_b)
                .expect("B decode");
            rows.push((format!("B decode {}", step - 2), got));
        }
    }

    let modes1 = memra_engine::moe_slru_pairs_calls();
    Arm {
        rows,
        stats: engine.moe_cache_stats(),
        modes: (
            modes1.0 - modes0.0,
            modes1.1 - modes0.1,
            modes1.2 - modes0.2,
        ),
    }
}

fn first_bit_difference(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    assert_eq!(a.len(), b.len(), "compared rows differ in length");
    a.iter()
        .zip(b)
        .enumerate()
        .find(|(_, (x, y))| x.to_bits() != y.to_bits())
        .map(|(i, (x, y))| (i, *x, *y))
}

/// Every row of `got` must be bit-identical to `want`; returns the names of the rows that are not.
fn differing_rows(want: &Arm, got: &Arm) -> Vec<String> {
    assert_eq!(
        want.rows.len(),
        got.rows.len(),
        "arms produced different row counts"
    );
    let mut out = Vec::new();
    for ((wn, wv), (gn, gv)) in want.rows.iter().zip(&got.rows) {
        assert_eq!(wn, gn, "arms ran different workloads");
        if let Some((i, w, g)) = first_bit_difference(wv, gv) {
            println!(
                "  {wn}: element {i} {w:e} (0x{:08x}) vs {g:e} (0x{:08x})",
                w.to_bits(),
                g.to_bits()
            );
            out.push(wn.clone());
        }
    }
    out
}

#[test]
#[ignore = "needs a CUDA device; run under the 5090 lease"]
fn gemma_slru_placement_is_bit_identical_to_resident() {
    let _gpu = gpu_guard();
    let resident = run_arm(Placement::Resident);
    assert!(
        resident.stats.is_none(),
        "the resident arm built an SLRU ({:?}): the arms are not distinct",
        resident.stats
    );
    assert_eq!(
        resident.modes,
        (0, 0, 0),
        "the resident arm ran the SLRU arm"
    );
    println!(
        "[gemma-placement] pairs program: {}",
        if std::env::var("MEMRA_GEMMA_MOE_MMA").as_deref() == Ok("0") {
            "decode-once dp4a (MEMRA_GEMMA_MOE_MMA=0)"
        } else {
            "int8 MMA"
        }
    );

    let mut failures = Vec::new();
    for placement in [
        Placement::SlruRoomy,
        Placement::SlruMid,
        Placement::SlruTight,
        Placement::SlruBelow,
    ] {
        let arm = run_arm(placement);
        let (row, pinned, chunked) = arm.modes;
        let stats = arm.stats;
        println!(
            "[gemma-placement] {placement:?}: cache {stats:?} (hits, misses, staged B, slots) \
             SLRU calls row={row} pinned={pinned} chunked={chunked}"
        );
        let differ = differing_rows(&resident, &arm);
        println!(
            "[gemma-placement] {placement:?}: {}/{} rows differ from the resident arm",
            differ.len(),
            arm.rows.len()
        );
        if placement == Placement::SlruBelow {
            // Below the floor the whole boot keeps the staged program: no SLRU mode may run, so
            // no request can cross programs. Its rows are the staged program's, not compared.
            if (row, pinned, chunked) != (0, 0, 0) {
                failures.push(format!(
                    "{placement:?}: a cache below {SMALL_T_BLOCKS} slots ran SLRU modes \
                     row={row} pinned={pinned} chunked={chunked}; the boot must keep one program"
                ));
            }
            continue;
        }
        let Some((hits, misses, staged, slots)) = stats else {
            failures.push(format!(
                "{placement:?}: no SLRU was built; the gate is vacuous"
            ));
            continue;
        };
        // The roomy arm's prewarm stages every block before the first call and every call reads
        // the full-layer row, which counts no hit; the other arms admit through dispatch.
        if !(staged > 0 && (hits > 0 || placement == Placement::SlruRoomy)) {
            failures.push(format!(
                "{placement:?}: the cache was not exercised (hits={hits} misses={misses} \
                 staged={staged})"
            ));
        }
        let mode_ok = match placement {
            Placement::SlruRoomy => row > 0 && slots >= 2 * LAYER_BLOCKS,
            Placement::SlruMid => {
                row > 0
                    && pinned > 0
                    && chunked == 0
                    && (LAYER_BLOCKS..2 * LAYER_BLOCKS).contains(&slots)
            }
            Placement::SlruTight => {
                row == 0
                    && pinned > 0
                    && chunked > 0
                    && (SMALL_T_BLOCKS..LAYER_BLOCKS).contains(&slots)
            }
            Placement::Resident | Placement::SlruBelow => unreachable!(),
        };
        if !mode_ok {
            failures.push(format!(
                "{placement:?}: SLRU modes row={row} pinned={pinned} chunked={chunked} with \
                 {slots} slots are not the ones this arm exists for (Roomy row, Mid row and \
                 pinned, Tight pinned and chunked)"
            ));
        }
        if !differ.is_empty() {
            failures.push(format!(
                "{placement:?}: {} rows differ from the resident arm: {differ:?}",
                differ.len()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "SLRU placements are not bit-identical to the resident placement: {failures:?}"
    );
}

/// CPU companion: the fixture is the shape the GPU gate assumes, so drift fails anywhere.
#[test]
#[allow(clippy::assertions_on_constants)] // allow: const pins; fail loudly if the gate's window drifts
fn the_fixture_is_the_gemma_q4_0_moe_shape() {
    let config = mini_config();
    let plan = mini_plan(&config);
    let source = fixture_source(&config, &plan);
    let moe = config.moe.as_ref().expect("the mini config is MoE");
    assert_eq!(moe.expert_count as usize, EXPERTS);
    assert_eq!(moe.expert_used_count as usize, TOP_K);
    assert_eq!(moe.expert_ff_length as usize, MOE_FF);
    assert!(
        !MOE_FF.is_multiple_of(256),
        "the down projection's k must be ragged, like gemma-4-26B-A4B's 704"
    );
    assert_eq!(
        source
            .tensors
            .values()
            .filter(|t| t.ty == GgmlType::Q4_0)
            .count(),
        4,
        "routed banks must be Q4_0 (the gemma-4-26B-A4B QAT expert class)"
    );
    assert!(BELOW_SLOTS < SMALL_T_BLOCKS && SMALL_T_BLOCKS <= TIGHT_SLOTS);
    assert!(TIGHT_SLOTS < LAYER_BLOCKS);
    assert!(LAYER_BLOCKS <= MID_SLOTS && MID_SLOTS < 2 * LAYER_BLOCKS);
}
