//! memra#1217: a softmax-routed MoE model's prefill must not depend on where its experts were
//! placed at load.
//!
//! WHY. With the routed experts resident (the fits-VRAM slab), prefill (t > 16) runs the pairs
//! arm, and on sm_120a that is the grouped f16 lane. Without a slab the SLRU cache serves the
//! experts, and before memra#1217 prefill fell through to the per-token decode program. Same
//! request, two numeric programs: on the RTX 5090 Ornith-1.5-35B-A3B's prefill logits moved by
//! up to 0.56 between placements and a greedy served text split after 95 characters, depending
//! only on how much memory was free when the model loaded. The fix runs the same pairs program
//! on the SLRU placement, reading the same bytes from cache slots (`moe_pairs_y_down_slru`).
//!
//! WHAT THIS GATE HOLDS, BIT-EXACT (`f32::to_bits`). A two-layer qwen35_moe model (softmax
//! router, shared expert, NVFP4 routed experts, the Ornith expert class) runs the same workload
//! in five placements:
//!   * `Resident`: the slab, the reference.
//!   * `SlruRoomy`: an auto-sized cache; the one-shot prewarm makes every layer fully cache
//!     resident, so prefill reads the cache's own full-layer row (the idle-5090 case).
//!   * `SlruMid`: 120 slots, room for one layer's 96 routed blocks but not both layers'; the
//!     layer the prewarm skipped admits its whole active set pinned.
//!   * `SlruTight`: 72 slots against 96 routed blocks per layer; a layer is never fully
//!     resident, a narrow prefill's active set is admitted pinned, a wide one is chunked.
//!   * `SlruFloor`: the 8-slot floor; every prefill runs as many chunks of two experts.
//!
//! Every logits row (prefill at seven widths, a prime and decode steps, and two interleaved
//! sessions) must equal the resident arm's. The 1,100-token prefill gives every expert group
//! more than 64 rows, so the sk visitor's 128-row tile form runs pinned and chunked too. Decode
//! rows were already placement-independent; they stay in the comparison so the fix cannot trade
//! one class for another.
//!
//! THE q8 SUB-ARM. Run the same binary with `MEMRA_MOE_F16G=0` and the pairs arm takes its q8
//! program instead of the grouped f16 lane. That program cannot be chunked, so the tight and
//! floor arms fall back to the per-token program (byte-identical to pairs-q8 for NVFP4), and the
//! mid arm runs q8 on pinned slots. The expectations below follow the env.
//!
//! NON-VACUITY. Each SLRU arm asserts the mode it exists for actually ran
//! (`memra_engine::moe_slru_pairs_calls`), that its cache staged and hit (the tight arms also
//! missed; the roomy arm stages by prewarm), and the tight arms that their slot count is below a
//! layer's routed blocks. On main (before the fix) every SLRU arm differs at every row after the
//! first prefill wider than 16 tokens, 16 of 18 rows; that red run is in the PR's receipts.
//!
//! TWO SESSIONS. One session's chunked prime runs between another session's decode steps on the
//! same engine and cache. The pin only lives inside one layer call's admission, under the cache
//! lock, so neither session can block or corrupt the other; both sessions' rows must equal the
//! resident arm running the same interleaving.
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
use memra_reference::deterministic_fixture;
use std::borrow::Cow;
use std::collections::BTreeMap;

const VOCAB: u32 = 32;
const EXPERTS: usize = 32;
const TOP_K: usize = 2;
/// `moe_intermediate_size`: NVFP4 needs `in_f % 64 == 0`, and `down`'s `in_f` is this.
const MOE_FF: usize = 64;
/// Routed blocks per MoE layer: `EXPERTS * {gate, up, down}`.
const LAYER_BLOCKS: usize = EXPERTS * 3;
/// At least one layer's routed blocks, below two layers'.
const MID_SLOTS: usize = 120;
/// Below `LAYER_BLOCKS`, above a narrow prefill's active set.
const TIGHT_SLOTS: usize = 72;
/// `MoeSlotCache` floors the slot count at 8.
const FLOOR_SLOTS: usize = 8;

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
        r#"{{"model_type":"qwen3_5_moe","num_hidden_layers":2,
        "num_nextn_predict_layers":0,"hidden_size":256,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":128,
        "intermediate_size":64,"vocab_size":{VOCAB},"max_position_embeddings":2048,
        "rms_norm_eps":0.000001,"full_attention_interval":2,
        "linear_conv_kernel_dim":4,"linear_key_head_dim":128,
        "linear_value_head_dim":128,"linear_num_key_heads":1,
        "linear_num_value_heads":2,"num_experts":{EXPERTS},"num_experts_per_tok":{TOP_K},
        "moe_intermediate_size":{MOE_FF},"shared_expert_intermediate_size":64}}"#
    )))
}

fn mini_plan(config: &ModelConfig) -> ModelPlan {
    memra_gguf::model_packs::for_config(config)
        .expect("qwen35_moe pack matches the mini config")
        .compile_plan(config)
        .expect("mini qwen35_moe plan compiles")
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
    fn tensor_census(&self) -> Result<memra_gguf::source::TensorCensus, String> {
        Ok(memra_gguf::source::census_from_views(
            self.tensors.iter().map(|(name, t)| {
                (
                    name.as_str(),
                    t.ggml_type,
                    t.ne.as_slice(),
                    t.bytes.len() as u64,
                )
            }),
        ))
    }
}

/// The deterministic reference fixture under the GGUF contract's names: routed-expert banks as
/// real NVFP4 blocks (`f32_to_nvfp4`, the GGUF NVFP4 layout Ornith's banks use, no macro plane),
/// every other tensor F32.
fn fixture_source(config: &ModelConfig, plan: &ModelPlan) -> FixtureSource {
    let weights = deterministic_fixture(plan)
        .expect("deterministic qwen35_moe fixture")
        .weights;
    let contract = TensorContract::for_plan(
        plan,
        CheckpointDialect::Gguf,
        ContractOptions {
            output_head: OutputHead::Separate,
        },
    )
    .expect("contract for the mini qwen35_moe plan");
    let mut tensors = BTreeMap::new();
    let mut banks = 0;
    for req in contract
        .requirements
        .iter()
        .filter(|r| r.required || weights.contains_key(&r.id))
    {
        let tensor = weights
            .get(&req.id)
            .or_else(|| {
                (req.id == TensorId::OutputProjection)
                    .then(|| weights.get(&TensorId::TokenEmbedding))
                    .flatten()
            })
            .unwrap_or_else(|| panic!("reference fixture is missing {:?}", req.id));
        let elements: usize = req.shape.iter().map(|&d| d as usize).product();
        assert_eq!(elements, tensor.data.len(), "fixture {:?} shape", req.id);
        let expert_bank = is_expert_bank(&req.id);
        let (bytes, ggml_type) = if expert_bank {
            banks += 1;
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
    assert_eq!(
        banks,
        3 * 2,
        "two MoE layers x gate/up/down routed-expert banks"
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
    SlruFloor,
}

/// The pairs arm runs the grouped f16 lane unless the run pins `MEMRA_MOE_F16G=0` (q8).
fn f16g_lane() -> bool {
    std::env::var("MEMRA_MOE_F16G").as_deref() != Ok("0")
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
            Placement::SlruFloor => ("0", Some(FLOOR_SLOTS)),
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
        .expect("mini qwen35_moe model loads from the contract");

    let mut rows = Vec::new();
    // Prefill widths across the dispatch tiers: per-token (<= 16) and pairs (> 16).
    for &n in &[1usize, 8, 17, 20, 65, 130, 1100] {
        let ids = tokens(n, 0x1217_0000 ^ n as u64);
        let got = model.forward(&engine, &ids).expect("GPU prefill");
        rows.push((format!("prefill T={n}"), got));
    }

    // Two sessions on one engine and one cache: A primes and decodes, B's prime (chunked on the
    // tight arms) lands between A's decode steps, then both decode alternately.
    let new_cache = || {
        memra_engine::cache::Cache::new_planned(&engine, &model.cfg, &plan, 160)
            .expect("cache for the mini qwen35_moe model")
    };
    let ids_a = tokens(40 + 6, 0x1217_00A0);
    let ids_b = tokens(97 + 6, 0x1217_00B0);
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
fn slru_prefill_is_bit_identical_to_resident_prefill() {
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
        "the resident arm ran the SLRU pairs arm"
    );

    let mut failures = Vec::new();
    let f16g = f16g_lane();
    println!(
        "[placement] pairs program: {}",
        if f16g {
            "grouped f16 lane"
        } else {
            "q8 (MEMRA_MOE_F16G=0)"
        }
    );
    for placement in [
        Placement::SlruRoomy,
        Placement::SlruMid,
        Placement::SlruTight,
        Placement::SlruFloor,
    ] {
        let arm = run_arm(placement);
        let (hits, misses, staged, slots) = arm
            .stats
            .unwrap_or_else(|| panic!("{placement:?}: no SLRU was built; the gate is vacuous"));
        let (row, pinned, chunked) = arm.modes;
        println!(
            "[placement] {placement:?}: slots={slots} hits={hits} misses={misses} \
             staged={staged}B pairs calls row={row} pinned={pinned} chunked={chunked}"
        );
        // The roomy arm's one-shot prewarm stages every block before the first dispatch, so it
        // hits and never misses; the tight arms must miss on every wide prefill.
        assert!(
            staged > 0 && hits > 0 && (misses > 0 || placement == Placement::SlruRoomy),
            "{placement:?}: the cache was not exercised (hits={hits} misses={misses} staged={staged})"
        );
        // Non-vacuity of the mode each arm exists for. Collected, not asserted on the spot, so a
        // run on a build without the SLRU pairs arm (the red run) still reports its rows.
        let mode_ok = match (placement, f16g) {
            (Placement::SlruRoomy, _) => row > 0 && slots >= LAYER_BLOCKS,
            (Placement::SlruMid, _) => {
                pinned > 0 && chunked == 0 && (LAYER_BLOCKS..2 * LAYER_BLOCKS).contains(&slots)
            }
            (Placement::SlruTight, true) => {
                pinned > 0 && chunked > 0 && row == 0 && slots < LAYER_BLOCKS
            }
            (Placement::SlruFloor, true) => {
                chunked > 0 && pinned == 0 && row == 0 && slots < LAYER_BLOCKS
            }
            // q8 cannot be chunked: a cache below one layer keeps the per-token program.
            (Placement::SlruTight | Placement::SlruFloor, false) => {
                row == 0 && pinned == 0 && chunked == 0 && slots < LAYER_BLOCKS
            }
            (Placement::Resident, _) => unreachable!(),
        };
        if !mode_ok {
            failures.push(format!(
                "{placement:?}: SLRU pairs modes row={row} pinned={pinned} chunked={chunked} \
                 with {slots} slots are not the ones this arm exists for (f16 lane: Roomy row, \
                 Mid pinned, Tight pinned and chunked, Floor chunked only; q8: Roomy row, Mid \
                 pinned, Tight and Floor none)"
            ));
        }
        let differ = differing_rows(&resident, &arm);
        println!(
            "[placement] {placement:?}: {}/{} rows differ from the resident arm",
            differ.len(),
            arm.rows.len()
        );
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
fn the_fixture_is_the_softmax_nvfp4_pairs_shape() {
    let config = mini_config();
    let plan = mini_plan(&config);
    let source = fixture_source(&config, &plan);
    assert!(
        config.sigmoid_router().is_none(),
        "the pairs arm is the softmax-router class"
    );
    let moe = config.moe.as_ref().expect("the mini config is MoE");
    assert_eq!(moe.expert_count as usize, EXPERTS);
    assert_eq!(moe.expert_used_count as usize, TOP_K);
    assert!(
        source
            .tensors
            .values()
            .filter(|t| t.ggml_type == GgmlType::NVFP4)
            .count()
            == 6,
        "routed banks must be NVFP4 (the f16g-admitted Ornith expert class)"
    );
    assert!(
        source.tensors.keys().all(|n| !n.ends_with(".scale")),
        "no macro plane: macro-carrying banks are denied the pairs arm"
    );
    assert!(FLOOR_SLOTS < TIGHT_SLOTS && TIGHT_SLOTS < LAYER_BLOCKS);
    assert!(LAYER_BLOCKS <= MID_SLOTS && MID_SLOTS < 2 * LAYER_BLOCKS);
    // A 17-token prefill routes at most 34 pairs: at most 34 active experts, but the tight arm's
    // pinned mode needs the active set's blocks <= TIGHT_SLOTS, which the routing decides; the
    // GPU gate asserts the mode ran rather than assuming it.
}
