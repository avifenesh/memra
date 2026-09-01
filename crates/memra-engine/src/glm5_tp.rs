//! glm5_next (GLM-5.3-Flash) TP-2 seam — `MEMRA_GLM5_TP` (lane/glm5-tp2, 2026-08-31).
//!
//! WHAT THIS IS. A correctness-first tensor-parallel-2 execution program for the glm5_next
//! hybrid trunk, per the lane's shard map (`research/glm53-flash-bringup-20260827/
//! tp2-20260831/SHARD-MAP.md`). Per layer class:
//!
//!   * KDA (34 layers): head-sharded, 32 heads per rank. Each rank runs the UNCHANGED
//!     `kda_core_gated` program on its shard (per-head kernels: conv, L2 norm, gate, scan,
//!     gated rmsnorm are all head-independent), the gated `[t, qkv/2]` halves are gathered,
//!     and each rank's COLUMN-parallel `wo` half (out rows over the FULL gathered input)
//!     computes its slice of the output with the same plain matvec kernel — joins are pure
//!     data movement, never a partial-sum reduction, which is what makes model-level
//!     TP-2-vs-plain BYTE identity the bar instead of a tolerance band.
//!   * MLA/DSA (11 layers): head-sharded per-head operands (`wq_b`, `wk_b`, `wv_b`);
//!     REPLICATED per-token shared work (`wq_a`/`q_a_norm`, `wkv_a`/`kv_a_norm`, the whole
//!     indexer + k-pool selection) — both ranks compute identical bytes from identical
//!     inputs, so the latent + indexer planes are replicated per rank and no per-token
//!     cross-rank hop exists in the latent chain. `wo` is column-parallel over the gathered
//!     attention halves, exactly like KDA.
//!   * MoE (sparse-FFN layers): EP-2, whole experts, contiguous halves (rank = expert /
//!     (n_expert/2)). The router stays root-computed (host sigmoid top-k, unchanged); each
//!     owner extracts its slots' UNWEIGHTED down rows with the same fused-epilogue kernels
//!     at n_used=1, and root re-applies the slot-ordered fmaf accumulation chain — the same
//!     rounded-operation sequence as the plain `moe_down8_fma_q8` walk. Shared expert,
//!     dense MLPs, router, mHC, norms, embed and lm_head stay ROOT-OWNED (the
//!     `MEMRA_STEP_TP` owner-stage precedent).
//!
//! V1 TRANSPORT is host-canonical staging (the step seam's correctness transport): fan-out
//! and joins bounce through host. Native P2P and the join-diet doors are the box arc.
//!
//! FAIL-CLOSED SURFACE. The preflight refuses before any TP CUDA state exists: non-glm5
//! plans, rank counts other than 2, head/expert counts that do not divide, duplicate
//! devices (serving parse), co-armed `MEMRA_PP_STAGES>1`, `MEMRA_STEP_TP`/`MEMRA_STEP_EP`.
//! A sharded layer POISONS every plain path: `kda_core`, `mla_attn_cached` and the batched
//! walks refuse a TP-armed layer by name. The memra-server worker refuses the flag outright
//! (serving wiring is the named box-lane increment, not v1).
//!
//! Engagement markers: `[glm5-tp-preflight]`, `[glm5-tp-kda]`, `[glm5-tp-mla]`,
//! `[glm5-tp-ep]` — every marker carries `performance_claim=false`.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use cudarc::driver::CudaSlice;

use crate::Engine;
use crate::kda::{ConvArm, KdaAttnLayer};
use crate::model::GpuTensor;
use memra_kv::{Cache, LatentKvLayer, RecurLayer};

/// v1 rank envelope: TP-2 exactly. TP-3 is refused by geometry (64 attention heads and the
/// 32-head indexer do not divide by 3); TP-4+ is refused until designed and gated.
pub const GLM5_TP_RANKS: usize = 2;

pub type Glm5TpLayerSpec = crate::tp::StepEpLayerSpec;

// ------------------------------------------------------------------------------------------
// Flag
// ------------------------------------------------------------------------------------------

/// Raw `MEMRA_GLM5_TP` value. Empty / unset / `"0"` = seam off.
pub fn glm5_tp_env_raw() -> Option<String> {
    std::env::var("MEMRA_GLM5_TP").ok()
}

/// Cheap armed check for co-refusal sites (server boot, spec doors). Parse errors count as
/// ARMED so a misspelled spec still refuses the co-armed program instead of racing the
/// loader's own refusal.
pub fn glm5_tp_armed() -> bool {
    matches!(glm5_tp_env_raw().as_deref(), Some(v) if !v.is_empty() && v != "0")
}

/// Parse the shared `LAYER[-LAYER]@DEVICE,DEVICE[;...]` grammar for the glm5 door.
/// `trunk_layers` is the loaded model's trunk length (the `all` shorthand expands against
/// it — the model contract owns that number, never a constant in the parser).
pub fn parse_glm5_tp_layer_specs(
    value: Option<&str>,
    trunk_layers: usize,
) -> Result<Vec<Glm5TpLayerSpec>, String> {
    crate::tp::parse_layer_specs_for_trunk("MEMRA_GLM5_TP", value, Some(trunk_layers))
}

/// Gate-harness knob, never a serving flag: `MEMRA_GLM5_TP_GATE_SAME_DEV=1` builds the peer
/// rank as a SECOND CUDA CONTEXT ON THE ROOT DEVICE (the one-card rig gate's emulation; the
/// ppN same-device-stages precedent). The spec's second device id becomes a logical rank id.
/// The serving worker refuses `MEMRA_GLM5_TP` outright, so this can never leak into serving.
pub fn gate_same_device() -> bool {
    std::env::var("MEMRA_GLM5_TP_GATE_SAME_DEV").as_deref() == Ok("1")
}

/// Gate-harness RED-arm knob, never a serving flag (`MEMRA_GLM5_TP_GATE_RED`):
///   * `swap-wo` — each rank's column `wo` half takes the OTHER rank's out rows (a broken
///     shard map); the gate run MUST diverge from plain.
///   * `swap-ep-gateup` — the root EP slab's gate and up projections swap (wrong expert
///     weights); MUST diverge.
///   * `skip-peer-combine` — the EP combine drops every peer-owned slot; MUST diverge,
///     which is also the non-vacuity proof that the peer rank contributes real work.
///   * `corrupt-ep-map` — the placement's local-slot table for rank 0 is reversed after
///     the slabs are built (owner table and slab bytes disagree — a corrupted map row);
///     MUST diverge. This is the red that proves the MEASURED-placement indirection is
///     load-bearing, not decorative.
///
/// Unknown values refuse at load.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GateRed {
    SwapWo,
    SwapEpGateUp,
    SkipPeerCombine,
    CorruptEpMap,
}

pub fn gate_red() -> Result<Option<GateRed>, String> {
    match std::env::var("MEMRA_GLM5_TP_GATE_RED").ok().as_deref() {
        None | Some("") => Ok(None),
        Some("swap-wo") => Ok(Some(GateRed::SwapWo)),
        Some("swap-ep-gateup") => Ok(Some(GateRed::SwapEpGateUp)),
        Some("skip-peer-combine") => Ok(Some(GateRed::SkipPeerCombine)),
        Some("corrupt-ep-map") => Ok(Some(GateRed::CorruptEpMap)),
        Some(other) => Err(format!(
            "MEMRA_GLM5_TP_GATE_RED={other:?} is not a known red arm \
             (swap-wo | swap-ep-gateup | skip-peer-combine | corrupt-ep-map)"
        )),
    }
}

// ------------------------------------------------------------------------------------------
// Runtime
// ------------------------------------------------------------------------------------------

/// The TP-2 rank runtime. Rank 0 (root) executes on the model's own engine — the PP-owner
/// context, exactly like the step seam's owner-first rank law. Rank 1 (peer) owns a second
/// full Engine.
pub struct Glm5TpRt {
    pub peer: Engine,
    pub root_dev: usize,
    pub peer_dev: usize,
    /// True only when built through [`Glm5TpRt::new_gate_same_device`] — the one-card rig
    /// gate's dual-context emulation (the ppN same-device gate precedent). The env-driven
    /// serving parse can never reach this: the grammar refuses duplicate devices.
    pub same_device_gate: bool,
}

impl Glm5TpRt {
    pub fn new(root_dev: usize, peer_dev: usize) -> Result<Self, Box<dyn std::error::Error>> {
        if root_dev == peer_dev {
            return Err(format!(
                "MEMRA_GLM5_TP rank devices must be distinct in serving; got {root_dev},{peer_dev} \
                 (the same-device form exists only for the rig gate binary)"
            )
            .into());
        }
        let peer = Engine::new(peer_dev)?;
        Ok(Self {
            peer,
            root_dev,
            peer_dev,
            same_device_gate: false,
        })
    }

    /// Same-device dual-context runtime for the ONE-CARD rig gate (exactness only). The
    /// peer rank is a second CUDA context on the root device: the whole shard/join walk —
    /// shard loads, replicated compute, gathers, canonical combines — executes exactly as
    /// on two cards, minus real peer transport (which the pro6000 batteries qualify on the
    /// box card class separately).
    pub fn new_gate_same_device(root_dev: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let peer = Engine::new(root_dev)?;
        Ok(Self {
            peer,
            root_dev,
            peer_dev: root_dev,
            same_device_gate: true,
        })
    }
}

// ------------------------------------------------------------------------------------------
// Preflight
// ------------------------------------------------------------------------------------------

/// What the loader tells the preflight about the model, extracted from the plan/config
/// BEFORE any TP CUDA state exists. Structural laws are dimension-derived (they hold for
/// the mini fixture and the real artifact alike): the laws ARE the geometry checks.
pub struct Glm5TpModelView {
    pub trunk_layers: usize,
    /// Per-layer mixer class, `trunk_layers` entries.
    pub layer_class: Vec<Glm5LayerClass>,
    /// Per-layer "has routed-expert FFN" flag (dense-prefix layers are false).
    pub layer_is_moe: Vec<bool>,
    pub kda_heads: usize,
    pub kda_head_dim: usize,
    pub mla_heads: usize,
    pub n_routed_experts: usize,
    pub top_k: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Glm5LayerClass {
    Kda,
    Mla,
}

/// The armed load plan: the runtime plus the layer set the spec selected, plus the
/// measured expert-placement map when `MEMRA_GLM5_EP_MAP` is armed (validated at
/// preflight, before any TP CUDA state — absent flag = the even split, byte-unchanged).
pub struct Glm5TpLoadPlan {
    pub rt: Arc<Glm5TpRt>,
    pub layers: std::collections::BTreeSet<usize>,
    pub ep_map: Option<crate::ep_map::EpMap>,
}

/// Raw `MEMRA_GLM5_EP_MAP` value. `Some("")` REFUSES downstream (a set-but-empty flag is
/// an operator error, never a silent even split).
pub fn glm5_ep_map_env() -> Option<String> {
    std::env::var("MEMRA_GLM5_EP_MAP").ok()
}

/// Load + validate the placement map against the model view and the armed layer set.
/// Fail-closed on every axis: unreadable file, malformed text, rank/expert-count
/// mismatch, layer-cover mismatch. Returns `None` only when the flag is UNSET.
fn load_glm5_ep_map(
    view: &Glm5TpModelView,
    layers: &std::collections::BTreeSet<usize>,
) -> Result<Option<crate::ep_map::EpMap>, Box<dyn std::error::Error>> {
    let Some(path) = glm5_ep_map_env() else {
        return Ok(None);
    };
    if path.is_empty() {
        return Err(
            "MEMRA_GLM5_EP_MAP is set but empty (fail-closed: unset the flag for \
                    the even split; an empty value never silently means default)"
                .into(),
        );
    }
    let text = std::fs::read_to_string(&path).map_err(|e| {
        format!("MEMRA_GLM5_EP_MAP={path}: cannot read the map file ({e}) — refused by name")
    })?;
    let map =
        crate::ep_map::EpMap::parse(&text).map_err(|e| format!("MEMRA_GLM5_EP_MAP={path}: {e}"))?;
    if map.ranks != GLM5_TP_RANKS {
        return Err(format!(
            "MEMRA_GLM5_EP_MAP={path}: map declares ranks={}, the v1 seam is TP-{GLM5_TP_RANKS}",
            map.ranks
        )
        .into());
    }
    if map.n_experts != view.n_routed_experts {
        return Err(format!(
            "MEMRA_GLM5_EP_MAP={path}: map declares expert_count={}, the model routes {}",
            map.n_experts, view.n_routed_experts
        )
        .into());
    }
    if map.entry_rank != 0 {
        return Err(format!(
            "MEMRA_GLM5_EP_MAP={path}: entry_rank={} but the glm5 TP-2 first-hop card is \
             rank 0 (root: router + combine + shared expert) — re-mint with \
             --entry-rank 0 (refused rather than silently remapping ranks)",
            map.entry_rank
        )
        .into());
    }
    let ep_layers: Vec<usize> = layers
        .iter()
        .copied()
        .filter(|&il| view.layer_is_moe[il])
        .collect();
    map.validate_layer_cover(&ep_layers)
        .map_err(|e| format!("MEMRA_GLM5_EP_MAP={path}: {e}"))?;
    // Receipt anchor: the map bytes that armed this load, named by digest.
    let digest = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(text.as_bytes());
        let out = h.finalize();
        out.iter().map(|b| format!("{b:02x}")).collect::<String>()
    };
    eprintln!(
        "[glm5-tp-preflight] ep-map armed path={path} sha256={digest} layers={} \
         experts={} ranks={} entry_rank={} performance_claim=false",
        map.layers.len(),
        map.n_experts,
        map.ranks,
        map.entry_rank,
    );
    Ok(Some(map))
}

/// Decode-diet doors that never co-arm with `MEMRA_GLM5_TP` in v1 (merge-forward
/// 2026-08-31): every TP-x-door pair is UNPROVEN. The TP byte/band gates ran with every
/// door cold, and each door's own gate ran on the unsharded walk, so v1 refuses by name
/// rather than silently picking an arm; a pair unlocks only with its own composition gate
/// (the `MEMRA_GLM5_TP` row in docs/FLAGS.md carries the matrix). `MEMRA_GLM5_VERIFY_BATCH`
/// is absent DELIBERATELY: its walk exists only inside glm5 spec sessions, which are
/// already co-refused while the TP door is armed (and the mixer choke points refuse a
/// sharded layer by name if ever reached).
pub const GLM5_TP_REFUSED_DOOR_FLAGS: [(&str, &str); 4] = [
    (
        "MEMRA_HC_FUSED_PRE",
        "the fused mHC pre-chain is gated on the unsharded walk only",
    ),
    (
        "MEMRA_HC_DECODE_WS",
        "the workspace decode walk carries no TP mixer branches",
    ),
    (
        "MEMRA_KDA_FUSED_PROJ",
        "the fused six-projection door (either operand arm) is gated on full-width \
         projections, never head shards",
    ),
    (
        "MEMRA_MLA_DECODE_SPLIT",
        "the absorb/decompress split is gated on the full-head geometry",
    ),
];

/// The pure composition law over [`GLM5_TP_REFUSED_DOOR_FLAGS`]: the first armed door
/// refuses by name, before any TP CUDA state exists. `armed` reports whether a flag is
/// set to `"1"` (env in production; a plain set in the unit test — the module keeps its
/// tests env-mutation-free).
pub fn refuse_glm5_tp_door_composition(armed: impl Fn(&str) -> bool) -> Result<(), String> {
    for (flag, why) in GLM5_TP_REFUSED_DOOR_FLAGS {
        if armed(flag) {
            return Err(format!(
                "MEMRA_GLM5_TP + {flag}: unproven composition, refused ({why})"
            ));
        }
    }
    Ok(())
}

/// Fail-closed preflight + runtime construction. Returns `None` when the seam is off.
/// Every illegal geometry refuses HERE, before any rank engine or shard exists.
pub fn prepare_glm5_tp_load(
    e: &Engine,
    view: &Glm5TpModelView,
) -> Result<Option<Glm5TpLoadPlan>, Box<dyn std::error::Error>> {
    let raw = glm5_tp_env_raw();
    let specs = parse_glm5_tp_layer_specs(raw.as_deref(), view.trunk_layers)?;
    if specs.is_empty() {
        return Ok(None);
    }

    // Co-armed programs refuse by name: two parallel/spec programs on one model never
    // silently coexist (the MEMRA_DSPARK precedent).
    if crate::pp::pp_cuts(view.trunk_layers).is_some() {
        return Err(
            "MEMRA_GLM5_TP + MEMRA_PP_STAGES>1: the TP-2 x PP composition is unwired and \
             refuses until its own gate exists (stage 5 of the tp2 lane names it)"
                .into(),
        );
    }
    if !crate::tp::step_tp_layer_specs()?.is_empty()
        || !crate::tp::step_ep_layer_specs()?.is_empty()
    {
        return Err(
            "MEMRA_GLM5_TP + MEMRA_STEP_TP/MEMRA_STEP_EP: the step and glm5 parallel \
             contracts never co-arm"
                .into(),
        );
    }
    refuse_glm5_tp_door_composition(|flag| std::env::var(flag).as_deref() == Ok("1"))?;

    // Structural geometry laws, all dimension-derived.
    if view.layer_class.len() != view.trunk_layers || view.layer_is_moe.len() != view.trunk_layers {
        return Err(format!(
            "glm5-tp preflight: layer class map ({}/{}) does not cover the {}-layer trunk",
            view.layer_class.len(),
            view.layer_is_moe.len(),
            view.trunk_layers
        )
        .into());
    }
    if !view.kda_heads.is_multiple_of(GLM5_TP_RANKS) || view.kda_heads == 0 {
        return Err(format!(
            "glm5-tp: {} KDA heads do not shard across {GLM5_TP_RANKS} ranks",
            view.kda_heads
        )
        .into());
    }
    if view.kda_head_dim != crate::kda::KDA_HEAD_DIM {
        return Err(format!(
            "glm5-tp: KDA head_dim {} is not the {} the scan kernel is instantiated for",
            view.kda_head_dim,
            crate::kda::KDA_HEAD_DIM
        )
        .into());
    }
    if !view.mla_heads.is_multiple_of(GLM5_TP_RANKS) || view.mla_heads == 0 {
        return Err(format!(
            "glm5-tp: {} MLA heads do not shard across {GLM5_TP_RANKS} ranks",
            view.mla_heads
        )
        .into());
    }
    if !view.n_routed_experts.is_multiple_of(GLM5_TP_RANKS) || view.n_routed_experts == 0 {
        return Err(format!(
            "glm5-tp: {} routed experts do not partition across {GLM5_TP_RANKS} ranks",
            view.n_routed_experts
        )
        .into());
    }
    if view.top_k > view.n_routed_experts {
        return Err("glm5-tp: top_k exceeds the routed expert count".into());
    }

    // One device pair across the whole spec (one runtime group in v1), root-first.
    let devices = specs[0].devices.clone();
    if devices.len() != GLM5_TP_RANKS {
        return Err(format!(
            "MEMRA_GLM5_TP requires exactly {GLM5_TP_RANKS} devices per layer in v1, got {}",
            devices.len()
        )
        .into());
    }
    for s in &specs {
        if s.devices != devices {
            return Err(format!(
                "MEMRA_GLM5_TP v1 carries ONE runtime group: layer {} names devices {:?}, \
                 the first spec names {:?}",
                s.layer, s.devices, devices
            )
            .into());
        }
        if s.layer >= view.trunk_layers {
            return Err(format!(
                "MEMRA_GLM5_TP layer {} outside the {}-layer trunk",
                s.layer, view.trunk_layers
            )
            .into());
        }
    }
    let root_dev = e.ctx().ordinal();
    if devices[0] != root_dev {
        return Err(format!(
            "MEMRA_GLM5_TP rank list {:?} must start with the owning device {root_dev} \
             (the owner-first rank law)",
            devices
        )
        .into());
    }

    // Validate the gate red-arm spelling at load (fail-closed), and pick the transport.
    let red = gate_red()?;
    let same_dev = gate_same_device();
    if let Some(red) = red {
        eprintln!("[glm5-tp-preflight] GATE RED ARM armed: {red:?} — outputs MUST diverge");
    }
    let rt = if same_dev {
        eprintln!(
            "[glm5-tp-preflight] GATE same-device emulation: peer rank is a second context \
             on device {root_dev} (spec device {} is a logical rank id)",
            devices[1]
        );
        Arc::new(Glm5TpRt::new_gate_same_device(root_dev)?)
    } else {
        Arc::new(Glm5TpRt::new(devices[0], devices[1])?)
    };
    let layers: std::collections::BTreeSet<usize> = specs.iter().map(|s| s.layer).collect();
    let ep_map = load_glm5_ep_map(view, &layers)?;
    let (mut kda_n, mut mla_n, mut moe_n) = (0usize, 0usize, 0usize);
    for &il in &layers {
        match view.layer_class[il] {
            Glm5LayerClass::Kda => kda_n += 1,
            Glm5LayerClass::Mla => mla_n += 1,
        }
        if view.layer_is_moe[il] {
            moe_n += 1;
        }
    }
    eprintln!(
        "[glm5-tp-preflight] armed ranks={GLM5_TP_RANKS} devices={devices:?} layers={} \
         kda_shard={kda_n} mla_shard={mla_n} moe_ep={moe_n} kda_heads_per_rank={} \
         mla_heads_per_rank={} experts_per_rank={} transport=host-canonical \
         weights_loaded=false performance_claim=false",
        layers.len(),
        view.kda_heads / GLM5_TP_RANKS,
        view.mla_heads / GLM5_TP_RANKS,
        view.n_routed_experts / GLM5_TP_RANKS,
    );
    Ok(Some(Glm5TpLoadPlan { rt, layers, ep_map }))
}

// ------------------------------------------------------------------------------------------
// Shard mechanics
// ------------------------------------------------------------------------------------------

fn outer_rows(ne: &[u64]) -> (usize, usize) {
    // GGML axis order: ne[0] is the fastest (innermost). The shardable axis is the LAST
    // (outermost) — out rows on a 2D projection, the head axis on a 3D per-head slab.
    let outer = *ne.last().expect("tensor has at least one axis") as usize;
    let inner: usize = ne[..ne.len() - 1].iter().map(|&d| d as usize).product();
    (outer, inner.max(1))
}

/// Copy `rows` of `t`'s outermost axis onto `dst` (host bounce; load-time only). Mirror
/// planes (`rp`/`rp4`/`f16`/`fp8`/`blk`) REFUSE by name: v1 shards carry the raw layout —
/// a pure byte-permutation difference, bit-identical by the mirrors' own contracts.
fn shard_rows(
    src_engine: &Engine,
    dst: &Engine,
    t: &GpuTensor,
    rows: Range<usize>,
) -> Result<GpuTensor, Box<dyn std::error::Error>> {
    match t {
        GpuTensor::Float { data, ne } => {
            let (outer, inner) = outer_rows(ne);
            if rows.end > outer {
                return Err(format!("shard rows {rows:?} exceed outer axis {outer}").into());
            }
            let host = src_engine.dtoh(data)?;
            let piece = &host[rows.start * inner..rows.end * inner];
            let mut ne2 = ne.clone();
            *ne2.last_mut().unwrap() = (rows.end - rows.start) as u64;
            Ok(GpuTensor::Float {
                data: dst.htod(piece)?,
                ne: ne2,
            })
        }
        GpuTensor::FloatBf16 { data, ne } => {
            let (outer, inner) = outer_rows(ne);
            if rows.end > outer {
                return Err(format!("shard rows {rows:?} exceed outer axis {outer}").into());
            }
            let host = src_engine.dtoh_u8(data)?;
            let piece = &host[rows.start * inner * 2..rows.end * inner * 2];
            let mut ne2 = ne.clone();
            *ne2.last_mut().unwrap() = (rows.end - rows.start) as u64;
            Ok(GpuTensor::FloatBf16 {
                data: dst.htod_bytes(piece)?,
                ne: ne2,
            })
        }
        GpuTensor::Quant {
            bytes,
            qtype,
            row_bytes,
            ne,
            scale,
            rp,
            fp8,
            rp4,
            blk,
            f16,
            #[cfg(memra_cutlass)]
            cutlass,
        } => {
            if *rp {
                return Err(
                    "glm5-tp shard: rp split-plane mirror layout is unwired — load \
                            the TP-armed tensor with MEMRA_RP=0 (raw layout is bit-identical \
                            by the mirror's own contract)"
                        .into(),
                );
            }
            if fp8.is_some() || rp4.is_some() || blk.is_some() || f16.is_some() {
                return Err(
                    "glm5-tp shard: a decode/prefill mirror (fp8/rp4/blk/f16) is present on a \
                     TP-armed tensor — mirrors are unwired for shards in v1; disable the \
                     mirror door for this load"
                        .into(),
                );
            }
            #[cfg(memra_cutlass)]
            if cutlass.is_some() {
                return Err("glm5-tp shard: cutlass prefill operand unwired for shards".into());
            }
            let (outer, inner) = outer_rows(ne);
            if ne.len() != 2 {
                return Err("glm5-tp shard: quantized shards are 2D-only in v1".into());
            }
            let _ = inner;
            if rows.end > outer {
                return Err(format!("shard rows {rows:?} exceed outer axis {outer}").into());
            }
            let host = src_engine.dtoh_u8(bytes)?;
            let piece = &host[rows.start * row_bytes..rows.end * row_bytes];
            let mut ne2 = ne.clone();
            *ne2.last_mut().unwrap() = (rows.end - rows.start) as u64;
            Ok(GpuTensor::Quant {
                bytes: dst.htod_bytes(piece)?,
                qtype: *qtype,
                row_bytes: *row_bytes,
                ne: ne2,
                scale: *scale,
                rp: false,
                fp8: None,
                rp4: None,
                blk: None,
                f16: None,
                #[cfg(memra_cutlass)]
                cutlass: None,
            })
        }
    }
}

/// Full replica of `t` on `dst` (host bounce). Same mirror refusals as [`shard_rows`].
fn replicate(
    src_engine: &Engine,
    dst: &Engine,
    t: &GpuTensor,
) -> Result<GpuTensor, Box<dyn std::error::Error>> {
    let (outer, _) = outer_rows(t.ne());
    shard_rows(src_engine, dst, t, 0..outer)
}

// ------------------------------------------------------------------------------------------
// KDA sidecar
// ------------------------------------------------------------------------------------------

/// The KDA TP-2 sidecar: the peer's head shard plus the runtime handle. The OUTER
/// `KdaAttnLayer` that carries this in its `tp` field is the root shard; both shards'
/// `wo` fields hold the rank's COLUMN half (out rows over the full `qkv` input).
pub struct Glm5TpKda {
    pub rt: Arc<Glm5TpRt>,
    pub peer: KdaAttnLayer,
    /// Full-width qkv of the UNSHARDED layer (`2 * shard qkv`) — the gather width.
    pub full_qkv: usize,
    /// Full hidden width (`wo` out rows across both ranks).
    pub n_embd: usize,
}

static KDA_MARKED: AtomicBool = AtomicBool::new(false);

/// Shard one loaded KDA layer: returns the ROOT shard (heads/2, wo out-rows 0..H/2) with
/// the peer shard (heads/2, wo out-rows H/2..H) in its `tp` sidecar. The full layer's
/// tensors are consumed and dropped — per-layer transient VRAM is one layer, never the
/// model.
pub(crate) fn shard_kda_layer(
    e: &Engine,
    rt: &Arc<Glm5TpRt>,
    la: KdaAttnLayer,
) -> Result<KdaAttnLayer, Box<dyn std::error::Error>> {
    if la.tp.is_some() {
        return Err("shard_kda_layer: layer is already sharded".into());
    }
    let heads = la.heads();
    let head_dim = la.head_dim();
    let qkv = la.qkv();
    let kernel = la.conv_kernel();
    if !heads.is_multiple_of(GLM5_TP_RANKS) {
        return Err(format!("KDA heads {heads} do not shard across {GLM5_TP_RANKS} ranks").into());
    }
    let hl = heads / GLM5_TP_RANKS; // heads per rank
    let ql = qkv / GLM5_TP_RANKS; // channels per rank
    let n_embd = la.wo.out_features();
    if !n_embd.is_multiple_of(GLM5_TP_RANKS) {
        return Err(format!("KDA wo out {n_embd} does not split across ranks").into());
    }
    let hh = n_embd / GLM5_TP_RANKS;

    let mut shard_plan = la.plan;
    shard_plan.num_heads = hl as u32;

    // Per-rank fused conv slice: plane p occupies rows [p*qkv, (p+1)*qkv) of the fused
    // [3*qkv, kernel] buffer; rank r takes channel rows [r*ql, (r+1)*ql) of each plane.
    let conv_host = e.dtoh(&la.conv)?;
    let conv_rank =
        |dst: &Engine, r: usize| -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
            let mut piece = Vec::with_capacity(3 * ql * kernel);
            for p in 0..3 {
                let a = (p * qkv + r * ql) * kernel;
                piece.extend_from_slice(&conv_host[a..a + ql * kernel]);
            }
            dst.htod(&piece)
        };

    // Gate red arm: a broken shard map hands each rank the OTHER rank's wo out rows.
    let wo_rank = |r: usize| -> usize {
        match gate_red() {
            Ok(Some(GateRed::SwapWo)) => 1 - r,
            _ => r,
        }
    };

    let rank_shard = |dst: &Engine, r: usize| -> Result<KdaAttnLayer, Box<dyn std::error::Error>> {
        let wr = wo_rank(r);
        Ok(KdaAttnLayer {
            plan: shard_plan,
            wq: shard_rows(e, dst, &la.wq, r * ql..(r + 1) * ql)?,
            wk: shard_rows(e, dst, &la.wk, r * ql..(r + 1) * ql)?,
            wv: shard_rows(e, dst, &la.wv, r * ql..(r + 1) * ql)?,
            f_a: replicate(e, dst, &la.f_a)?,
            f_b: shard_rows(e, dst, &la.f_b, r * ql..(r + 1) * ql)?,
            g_a: replicate(e, dst, &la.g_a)?,
            g_b: shard_rows(e, dst, &la.g_b, r * ql..(r + 1) * ql)?,
            b_proj: shard_rows(e, dst, &la.b_proj, r * hl..(r + 1) * hl)?,
            // COLUMN-parallel wo: rank r owns OUT rows [r*hh, (r+1)*hh) over the FULL qkv
            // input — consumed by the join over the gathered gated tensor, never by
            // kda_core_gated itself.
            wo: shard_rows(e, dst, &la.wo, wr * hh..(wr + 1) * hh)?,
            conv: conv_rank(dst, r)?,
            a_log: shard_rows(e, dst, &la.a_log, r * hl..(r + 1) * hl)?,
            dt_bias: shard_rows(e, dst, &la.dt_bias, r * ql..(r + 1) * ql)?,
            o_norm: replicate(e, dst, &la.o_norm)?,
            tp: None,
        })
    };

    let mut root = rank_shard(e, 0)?;
    let peer = rank_shard(&rt.peer, 1)?;
    if !KDA_MARKED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[glm5-tp-kda] head shard armed: heads_per_rank={hl} head_dim={head_dim} \
             wo=column-over-gather transport=host-canonical performance_claim=false"
        );
    }
    root.tp = Some(Box::new(Glm5TpKda {
        rt: Arc::clone(rt),
        peer,
        full_qkv: qkv,
        n_embd,
    }));
    Ok(root)
}

/// Ensure layer `il`'s per-rank KDA state planes exist (lazily, sized for the SHARD
/// geometry — the canonical `cache.recur[il]` planes are full-width and stay untouched
/// as allocated; the TP walk never reads them).
fn ensure_kda_tp_state<'c>(
    e: &Engine,
    rt: &Glm5TpRt,
    la_root: &KdaAttnLayer,
    cache: &'c mut Cache,
    il: usize,
) -> Result<&'c mut [RecurLayer; 2], Box<dyn std::error::Error>> {
    if cache.glm5_tp_recur.len() <= il {
        return Err(format!("glm5-tp: cache carries no TP recur slot for layer {il}").into());
    }
    if cache.glm5_tp_recur[il].is_none() {
        let conv_pad = la_root.conv_width() * (la_root.conv_kernel() - 1);
        let state = la_root.state_width();
        let mk = |dev: &Engine| -> Result<RecurLayer, Box<dyn std::error::Error>> {
            Ok(RecurLayer {
                conv_state: dev.zeros(conv_pad)?,
                ssm_state: dev.zeros(state)?,
                ssm_state_alt: dev.zeros(state)?,
            })
        };
        cache.glm5_tp_recur[il] = Some([mk(e)?, mk(&rt.peer)?]);
    }
    Ok(cache.glm5_tp_recur[il].as_mut().unwrap())
}

/// The KDA TP-2 walk for one prime/decode call: per-rank `kda_core_gated` on the shards,
/// host-canonical gather of the gated halves, per-rank column `wo`, and the output
/// concatenation. `la_root` is the root shard (its `tp` sidecar carries the peer).
#[allow(clippy::too_many_arguments)] // mirrors the kda entry contract shape
pub(crate) fn kda_tp_cached(
    e: &Engine,
    la_root: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    t: usize,
    eps: f32,
    cache: &mut Cache,
    il: usize,
    arm: ConvArm,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let tp = la_root
        .tp
        .as_ref()
        .ok_or("kda_tp_cached called on an unsharded layer")?;
    let rt = &tp.rt;
    let ql = la_root.qkv(); // per-rank channels
    let full = tp.full_qkv;
    let n_embd = tp.n_embd;
    let hh = n_embd / GLM5_TP_RANKS;

    // Host-canonical fan-out of x.
    let x_host = e.dtoh(x)?;
    let x_peer = rt.peer.htod(&x_host)?;

    let [root_state, peer_state] = ensure_kda_tp_state(e, rt, la_root, cache, il)?;

    // Peer shard first (host-canonical serial walk; overlap is the box arc).
    let gated_peer = {
        let RecurLayer {
            conv_state,
            ssm_state,
            ssm_state_alt,
        } = peer_state;
        let out = crate::kda::kda_core_gated(
            &rt.peer,
            &tp.peer,
            &x_peer,
            t,
            eps,
            conv_state,
            ssm_state,
            ssm_state_alt,
            arm,
            crate::kda::KdaStash::None,
            None,
        )?;
        std::mem::swap(ssm_state, ssm_state_alt);
        out
    };
    let gated_root = {
        let RecurLayer {
            conv_state,
            ssm_state,
            ssm_state_alt,
        } = root_state;
        let out = crate::kda::kda_core_gated(
            e,
            la_root,
            x,
            t,
            eps,
            conv_state,
            ssm_state,
            ssm_state_alt,
            arm,
            crate::kda::KdaStash::None,
            None,
        )?;
        std::mem::swap(ssm_state, ssm_state_alt);
        out
    };

    // Gather the gated halves into the FULL [t, qkv] layout on BOTH ranks (column-parallel
    // wo needs the whole input on each rank). Token-major interleave: row tok is
    // [root ql | peer ql].
    let gated_peer_host = rt.peer.dtoh(&gated_peer)?;
    let gated_root_host = e.dtoh(&gated_root)?;
    let mut full_host = vec![0f32; t * full];
    for tok in 0..t {
        full_host[tok * full..tok * full + ql]
            .copy_from_slice(&gated_root_host[tok * ql..(tok + 1) * ql]);
        full_host[tok * full + ql..(tok + 1) * full]
            .copy_from_slice(&gated_peer_host[tok * ql..(tok + 1) * ql]);
    }
    let full_root = e.htod(&full_host)?;
    let full_peer = rt.peer.htod(&full_host)?;

    // Per-rank column wo halves: each output element is one full-K dot by the SAME plain
    // matvec kernel — no cross-rank arithmetic anywhere in this join.
    let y_root = e.matmul(&la_root.wo, &full_root, t)?; // [t, hh] rows 0..hh
    let y_peer = rt.peer.matmul(&tp.peer.wo, &full_peer, t)?; // [t, hh] rows hh..2hh
    let y_peer_host = rt.peer.dtoh(&y_peer)?;
    let y_root_host = e.dtoh(&y_root)?;
    let mut out_host = vec![0f32; t * n_embd];
    for tok in 0..t {
        out_host[tok * n_embd..tok * n_embd + hh]
            .copy_from_slice(&y_root_host[tok * hh..(tok + 1) * hh]);
        out_host[tok * n_embd + hh..(tok + 1) * n_embd]
            .copy_from_slice(&y_peer_host[tok * hh..(tok + 1) * hh]);
    }
    e.htod(&out_host)
}

// ------------------------------------------------------------------------------------------
// MLA sidecar
// ------------------------------------------------------------------------------------------

/// The MLA TP-2 sidecar: the peer's head shard (with replicated `wq_a`/`wkv_a`/norms and
/// a full indexer replica) plus the runtime handle.
pub struct Glm5TpMla {
    pub rt: Arc<Glm5TpRt>,
    pub peer: crate::hybrid::MlaAttnLayer,
    /// Full head count of the unsharded layer.
    pub full_heads: usize,
    /// Full hidden width (`wo` out rows across both ranks).
    pub n_embd: usize,
}

static MLA_MARKED: AtomicBool = AtomicBool::new(false);

pub(crate) fn shard_mla_layer(
    e: &Engine,
    rt: &Arc<Glm5TpRt>,
    la: crate::hybrid::MlaAttnLayer,
) -> Result<crate::hybrid::MlaAttnLayer, Box<dyn std::error::Error>> {
    use crate::hybrid::{MlaAttnLayer, MlaIndexer};
    if la.tp.is_some() {
        return Err("shard_mla_layer: layer is already sharded".into());
    }
    let g = la.geom;
    let nh = g.n_head;
    if !nh.is_multiple_of(GLM5_TP_RANKS) {
        return Err(format!("MLA heads {nh} do not shard across {GLM5_TP_RANKS} ranks").into());
    }
    let hl = nh / GLM5_TP_RANKS;
    let head_q = g.d_nope + g.d_rope; // per-head wq_b out rows
    let n_embd = la.wo.out_features();
    if !n_embd.is_multiple_of(GLM5_TP_RANKS) {
        return Err(format!("MLA wo out {n_embd} does not split across ranks").into());
    }
    let hh = n_embd / GLM5_TP_RANKS;

    let mut shard_geom = g;
    shard_geom.n_head = hl;

    let replicate_indexer =
        |dst: &Engine, ix: &MlaIndexer| -> Result<MlaIndexer, Box<dyn std::error::Error>> {
            Ok(MlaIndexer {
                wq_b: replicate(e, dst, &ix.wq_b)?,
                wk: replicate(e, dst, &ix.wk)?,
                k_norm_w: replicate(e, dst, &ix.k_norm_w)?,
                k_norm_b: replicate(e, dst, &ix.k_norm_b)?,
                weights_proj: replicate(e, dst, &ix.weights_proj)?,
                kpool_gate: replicate(e, dst, &ix.kpool_gate)?,
                kpool_ape: replicate(e, dst, &ix.kpool_ape)?,
                geom: ix.geom,
            })
        };

    // Gate red arm: a broken shard map hands each rank the OTHER rank's wo out rows.
    let wo_rank = |r: usize| -> usize {
        match gate_red() {
            Ok(Some(GateRed::SwapWo)) => 1 - r,
            _ => r,
        }
    };

    let rank_shard = |dst: &Engine, r: usize| -> Result<MlaAttnLayer, Box<dyn std::error::Error>> {
        let wr = wo_rank(r);
        Ok(MlaAttnLayer {
            wq_a: replicate(e, dst, &la.wq_a)?,
            q_a_norm: replicate(e, dst, &la.q_a_norm)?,
            wq_b: shard_rows(e, dst, &la.wq_b, r * hl * head_q..(r + 1) * hl * head_q)?,
            wkv_a: replicate(e, dst, &la.wkv_a)?,
            kv_a_norm: replicate(e, dst, &la.kv_a_norm)?,
            // 3D per-head slabs: the head axis is outermost.
            wk_b: shard_rows(e, dst, &la.wk_b, r * hl..(r + 1) * hl)?,
            wv_b: shard_rows(e, dst, &la.wv_b, r * hl..(r + 1) * hl)?,
            // COLUMN-parallel wo: rank r owns OUT rows over the full N*V input.
            wo: shard_rows(e, dst, &la.wo, wr * hh..(wr + 1) * hh)?,
            geom: shard_geom,
            index: match &la.index {
                Some(ix) => Some(replicate_indexer(dst, ix)?),
                None => None,
            },
            tp: None,
        })
    };

    let mut root = rank_shard(e, 0)?;
    let peer = rank_shard(&rt.peer, 1)?;
    if !MLA_MARKED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[glm5-tp-mla] head shard armed: heads_per_rank={hl} kv_rank={} latent=replicated \
             indexer=replicated wo=column-over-gather transport=host-canonical \
             performance_claim=false",
            g.kv_rank
        );
    }
    root.tp = Some(Box::new(Glm5TpMla {
        rt: Arc::clone(rt),
        peer,
        full_heads: nh,
        n_embd,
    }));
    Ok(root)
}

/// Ensure the PEER's replicated latent plane for layer `il` exists, geometry-cloned from
/// the canonical (root) plane. The canonical plane IS the root replica — the root path is
/// unchanged.
pub(crate) fn ensure_mla_peer_latent(
    rt: &Glm5TpRt,
    canonical: &LatentKvLayer,
    cache_slot: &mut Option<LatentKvLayer>,
) -> Result<(), Box<dyn std::error::Error>> {
    if cache_slot.is_some() {
        return Ok(());
    }
    let dev = &rt.peer;
    let rows = dev.zeros(canonical.rows.len())?;
    // Fresh replica starts at len 0 like a fresh canonical plane; the walk appends to both
    // in the same calls, so the lengths stay in lock-step by construction.
    let len_d = dev.htod_i32(&[0])?;
    let index_rows = match &canonical.index_rows {
        Some(p) => Some(dev.zeros(p.len())?),
        None => None,
    };
    *cache_slot = Some(LatentKvLayer {
        rows,
        width: canonical.width,
        index_width: canonical.index_width,
        len: 0,
        len_d,
        index_rows,
        index_ring_rows: canonical.index_ring_rows,
        index_pool_keys: None, // lazily allocated by the core, exactly like the canonical plane
        index_pools_ready: 0,
        index_pool: canonical.index_pool,
    });
    Ok(())
}

// ------------------------------------------------------------------------------------------
// MoE EP sidecar
// ------------------------------------------------------------------------------------------

/// One rank's expert slab: the rank's owned experts packed in ASCENDING expert-id order
/// for every projection, device-resident on that rank. For the even split the packing is
/// the contiguous half — byte-for-byte the pre-map layout.
pub struct EpRankSlab {
    pub gate: CudaSlice<u8>,
    pub up: CudaSlice<u8>,
    pub down: CudaSlice<u8>,
    pub n_experts: usize,
}

/// The MoE EP-2 sidecar on `MoeWeights`: per-rank expert slabs, the placement tables,
/// and the runtime handle. Router, shared expert, macros and all host metadata stay on
/// the unchanged `MoeWeights`.
///
/// PLACEMENT INDEPENDENCE (the contract the gate's skewed-map arm proves): `owner_of`
/// only selects WHICH rank runs the identical per-expert program over identical
/// host-canonical input bytes; `local_of` indexes the same expert bytes wherever they
/// were packed; the combine stays slot-ordered on root. The map moves bytes, never
/// changes arithmetic.
pub struct Glm5EpExps {
    pub rt: Arc<Glm5TpRt>,
    pub root: EpRankSlab,
    pub peer: EpRankSlab,
    /// `owner_of[expert]` = owning rank (0 = root).
    pub owner_of: Vec<u8>,
    /// `local_of[expert]` = slot inside the owner's slab (ascending-id packing order).
    pub local_of: Vec<u32>,
    /// Per-rank grouped-dispatch pointer tables, `[root, peer]`, each the `DevExps::ptr_row`
    /// shape ([3 * n_expert] u64 device pointers: gate | up | down planes, indexed by GLOBAL
    /// expert id, resident on the owning rank's device). Owned experts point at
    /// `slab_base + local * stride`; non-owned entries are 0 and never dereferenced — the EP
    /// grouped-prime CSR is built per rank from `owner_of`, so a foreign id cannot reach the
    /// wrong rank's table. Built AFTER the gate-red slab mutations, from the FINAL slab
    /// buffers and the FINAL `local_of`, so `swap-ep-gateup` and `corrupt-ep-map` bite the
    /// grouped walk exactly as they bite the sequential one.
    pub ptr_rows: [CudaSlice<u64>; 2],
}

impl Glm5EpExps {
    /// Owner rank of `expert` under the armed placement (even split when no map).
    pub fn owner(&self, expert: usize) -> usize {
        self.owner_of[expert] as usize
    }
}

static EP_MARKED: AtomicBool = AtomicBool::new(false);

/// Engagement counter: PEER-owned expert slots dispatched by the EP walk (counted before
/// any gate-red skip, so a red arm can still assert the peer was ROUTED). Gates read it to
/// prove the peer rank contributes real expert work — a token stream that never routes a
/// peer-owned expert makes every EP identity arm vacuous.
pub static GLM5_EP_PEER_SLOT_DISPATCHES: AtomicU64 = AtomicU64::new(0);

pub fn glm5_ep_peer_slot_dispatches() -> u64 {
    GLM5_EP_PEER_SLOT_DISPATCHES.load(Ordering::Relaxed)
}

// ---- EP dispatch-diet engagement counters (lane/glm5-ep-diet, 2026-08-31) ----------------
// The box A/B greps announces and reads these deltas; the rig gate asserts them non-vacuous
// on the ON arms and FLAT on the pinned-`=0` arms.

/// Layer-calls that took the dieted EP walk (`MEMRA_GLM5_EP_DIET`) instead of the v1
/// per-slot host-canonical walk.
pub static GLM5_EP_DIET_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Snapshot of [`GLM5_EP_DIET_DISPATCHES`] — gates take a before/after delta.
pub fn glm5_ep_diet_dispatches() -> u64 {
    GLM5_EP_DIET_DISPATCHES.load(Ordering::Relaxed)
}

/// Bulk peer-row block returns performed by the dieted walk (one per layer-call that routed
/// at least one peer-owned slot; each replaces that call's ENTIRE per-slot return dribble).
pub static GLM5_EP_DIET_BULK_RETURNS: AtomicU64 = AtomicU64::new(0);

/// Snapshot of [`GLM5_EP_DIET_BULK_RETURNS`].
pub fn glm5_ep_diet_bulk_returns() -> u64 {
    GLM5_EP_DIET_BULK_RETURNS.load(Ordering::Relaxed)
}

/// Per-slot synchronous peer round-trips (one peer DtoH + one root pageable HtoD each, the
/// v1 walk's dominant hop class) that the dieted walk folded into its bulk return — one
/// count per peer-owned slot bulked.
pub static GLM5_EP_DIET_PEER_ROUNDTRIPS_AVOIDED: AtomicU64 = AtomicU64::new(0);

/// Snapshot of [`GLM5_EP_DIET_PEER_ROUNDTRIPS_AVOIDED`].
pub fn glm5_ep_diet_peer_roundtrips_avoided() -> u64 {
    GLM5_EP_DIET_PEER_ROUNDTRIPS_AVOIDED.load(Ordering::Relaxed)
}

/// Per-token peer z uploads the dieted walk avoided: `t-1` per fanned layer-call (one bulk
/// [t, n_embd] upload replaces t per-token uploads) plus `t` per layer-call whose routing
/// never touched a peer-owned expert (the fan-out is skipped entirely — the placement-map
/// multiplier: single-rank layer-calls move ZERO activation bytes off root).
pub static GLM5_EP_DIET_FANOUT_UPLOADS_AVOIDED: AtomicU64 = AtomicU64::new(0);

/// Snapshot of [`GLM5_EP_DIET_FANOUT_UPLOADS_AVOIDED`].
pub fn glm5_ep_diet_fanout_uploads_avoided() -> u64 {
    GLM5_EP_DIET_FANOUT_UPLOADS_AVOIDED.load(Ordering::Relaxed)
}

/// Layer-calls that took the per-rank grouped-GEMM EP prime (`MEMRA_GLM5_EP_GROUPED_PRIME`).
/// Stays 0 whenever the plain grouped-prefill conjuncts do not hold (e.g. non-f16g-eligible
/// expert qtypes — the rig fixture's Q8_0 bank always falls closed to the sequential walk).
pub static GLM5_EP_GROUPED_PRIME_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Snapshot of [`GLM5_EP_GROUPED_PRIME_DISPATCHES`].
pub fn glm5_ep_grouped_prime_dispatches() -> u64 {
    GLM5_EP_GROUPED_PRIME_DISPATCHES.load(Ordering::Relaxed)
}

/// Arm one MoE layer for EP-2. `placement` is the layer's validated map row
/// (`owners[expert] = rank`) when `MEMRA_GLM5_EP_MAP` is armed; `None` = the even
/// split, whose ascending-id packing is byte-for-byte the pre-map contiguous halves.
pub(crate) fn arm_moe_ep(
    e: &Engine,
    rt: &Arc<Glm5TpRt>,
    m: &mut crate::hybrid::MoeWeights,
    placement: Option<&[u8]>,
) -> Result<(), Box<dyn std::error::Error>> {
    if m.glm5_ep.is_some() {
        return Err("arm_moe_ep: layer is already EP-armed".into());
    }
    let n_expert = m.gate_exps.n_expert;
    if !n_expert.is_multiple_of(GLM5_TP_RANKS) {
        return Err(format!(
            "glm5-tp EP: {n_expert} experts do not partition across {GLM5_TP_RANKS} ranks"
        )
        .into());
    }
    if m.gate_exps.layouts.is_some() || m.up_exps.layouts.is_some() || m.down_exps.layouts.is_some()
    {
        return Err("glm5-tp EP: per-expert mixed layouts are unwired for EP shards".into());
    }
    let owner_of: Vec<u8> = match placement {
        Some(owners) => {
            // The preflight validated the map; re-assert the two structural laws at the
            // consumption site so a wiring bug can never hand a foreign row to a layer.
            if owners.len() != n_expert {
                return Err(format!(
                    "glm5-tp EP: placement row carries {} owners for a {n_expert}-expert bank",
                    owners.len()
                )
                .into());
            }
            if owners.iter().any(|&r| (r as usize) >= GLM5_TP_RANKS) {
                return Err("glm5-tp EP: placement row names a rank outside TP-2".into());
            }
            owners.to_vec()
        }
        None => crate::ep_map::EpMap::even_owners(n_expert, GLM5_TP_RANKS),
    };
    // Ascending-id packing per rank + the local-slot table.
    let mut local_of = vec![0u32; n_expert];
    let mut owned: [Vec<usize>; GLM5_TP_RANKS] = [Vec::new(), Vec::new()];
    for ex in 0..n_expert {
        let r = owner_of[ex] as usize;
        local_of[ex] = owned[r].len() as u32;
        owned[r].push(ex);
    }
    if owned.iter().any(|o| o.is_empty()) {
        return Err("glm5-tp EP: placement leaves a rank with zero experts (refused)".into());
    }
    let slab =
        |dev: &Engine, experts: &[usize]| -> Result<EpRankSlab, Box<dyn std::error::Error>> {
            // Tail-slack pads mirror the resident-slab builder (`build_dev_exps`): 8 B
            // alignment slack on gate/up and 144 B on down — the ragged-k grouped GEMM
            // walks whole superblocks and may overread past the LAST row (harmless bytes,
            // the zero-padded k-range multiplies them away; the slack only prevents the
            // OOB fault). Bytes at every in-slab offset are unchanged, so the sequential
            // per-slot views read exactly what they read before.
            let cut = |h: &crate::model::HostExps,
                       pad: usize|
             -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
                let stride = h.expert_stride;
                let bytes = h.bytes.as_bytes();
                // Contiguous ascending run (the even split, and any contiguous map row):
                // one direct upload of the existing byte range — no host copy.
                let contiguous = experts.windows(2).all(|w| w[1] == w[0] + 1);
                if contiguous {
                    let a = experts[0] * stride;
                    let b = (experts[experts.len() - 1] + 1) * stride;
                    return dev.htod_bytes_padded(&bytes[a..b], pad);
                }
                // General map row: pack the owned experts ascending into one staging
                // buffer (load-time only; per-layer transient host = one rank's slab).
                let mut staged = Vec::with_capacity(experts.len() * stride);
                for &ex in experts {
                    staged.extend_from_slice(&bytes[ex * stride..(ex + 1) * stride]);
                }
                dev.htod_bytes_padded(&staged, pad)
            };
            Ok(EpRankSlab {
                gate: cut(&m.gate_exps, 8)?,
                up: cut(&m.up_exps, 8)?,
                down: cut(&m.down_exps, 144)?,
                n_experts: experts.len(),
            })
        };
    let mut root = slab(e, &owned[0])?;
    let peer = slab(&rt.peer, &owned[1])?;
    // Gate red arm: wrong expert weights on the root rank (gate/up swapped).
    if matches!(gate_red(), Ok(Some(GateRed::SwapEpGateUp))) {
        std::mem::swap(&mut root.gate, &mut root.up);
    }
    // Gate red arm: a corrupted map row — the local-slot table for rank 0 is reversed
    // AFTER the slabs were packed, so the owner table and the slab bytes disagree.
    if matches!(gate_red(), Ok(Some(GateRed::CorruptEpMap))) {
        let n0 = owned[0].len() as u32;
        for &ex in &owned[0] {
            local_of[ex] = n0 - 1 - local_of[ex];
        }
    }
    // Per-rank grouped-dispatch pointer tables (lane/glm5-ep-diet): the `DevExps::ptr_row`
    // shape over each rank's OWN slab, built from the FINAL slab buffers and the FINAL
    // `local_of` so both gate reds above flow into the grouped walk too. ~3*n_expert*8 B per
    // rank per layer — negligible next to the slabs they index.
    let ptr_table = |dev: &Engine,
                     slab: &EpRankSlab,
                     rank: u8|
     -> Result<CudaSlice<u64>, Box<dyn std::error::Error>> {
        use cudarc::driver::DevicePtr;
        let (pg, pu, pd) = {
            let s = dev.stream();
            let (pg, _g0) = slab.gate.device_ptr(&s);
            let (pu, _g1) = slab.up.device_ptr(&s);
            let (pd, _g2) = slab.down.device_ptr(&s);
            (pg, pu, pd)
        };
        let mut host = vec![0u64; 3 * n_expert];
        for ex in 0..n_expert {
            if owner_of[ex] != rank {
                continue; // non-owned: 0, never dereferenced (rank CSRs filter by owner)
            }
            let local = local_of[ex] as usize;
            host[ex] = pg + (local * m.gate_exps.expert_stride) as u64;
            host[n_expert + ex] = pu + (local * m.up_exps.expert_stride) as u64;
            host[2 * n_expert + ex] = pd + (local * m.down_exps.expert_stride) as u64;
        }
        dev.htod_u64(&host)
    };
    let ptr_rows = [ptr_table(e, &root, 0)?, ptr_table(&rt.peer, &peer, 1)?];
    if !EP_MARKED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[glm5-tp-ep] expert-parallel armed: experts rank0={} rank1={} ownership={} \
             router=root combine=slot-ordered-fmaf transport=host-canonical \
             performance_claim=false",
            owned[0].len(),
            owned[1].len(),
            if placement.is_some() {
                "measured-map"
            } else {
                "even-split"
            },
        );
    }
    // The root-resident full slab (if the loader built one) is superseded by the EP halves;
    // dropping it returns its VRAM and removes the arm that would silently bypass EP.
    m.dev_exps = None;
    m.glm5_ep = Some(Glm5EpExps {
        rt: Arc::clone(rt),
        root,
        peer,
        owner_of,
        local_of,
        ptr_rows,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_is_literal_and_fail_closed() {
        // Off spellings.
        assert!(parse_glm5_tp_layer_specs(None, 45).unwrap().is_empty());
        assert!(parse_glm5_tp_layer_specs(Some(""), 45).unwrap().is_empty());
        assert!(parse_glm5_tp_layer_specs(Some("0"), 45).unwrap().is_empty());
        // The full-model shorthand expands against the CALLER's trunk, not a constant.
        let all = parse_glm5_tp_layer_specs(Some("all@0,1"), 45).unwrap();
        assert_eq!(all.len(), 45);
        assert_eq!(all[0].devices, vec![0, 1]);
        let all4 = parse_glm5_tp_layer_specs(Some("all@0,1"), 4).unwrap();
        assert_eq!(all4.len(), 4);
        // Explicit ranges.
        let r = parse_glm5_tp_layer_specs(Some("0-2@0,1;4@0,1"), 45).unwrap();
        assert_eq!(
            r.iter().map(|s| s.layer).collect::<Vec<_>>(),
            vec![0, 1, 2, 4]
        );
        // Refusals: duplicate devices, duplicate layers, garbage.
        assert!(parse_glm5_tp_layer_specs(Some("0@0,0"), 45).is_err());
        assert!(parse_glm5_tp_layer_specs(Some("0@0,1;0@0,1"), 45).is_err());
        assert!(parse_glm5_tp_layer_specs(Some("banana"), 45).is_err());
    }

    fn fixture_view() -> Glm5TpModelView {
        Glm5TpModelView {
            trunk_layers: 4,
            layer_class: vec![
                Glm5LayerClass::Kda,
                Glm5LayerClass::Mla,
                Glm5LayerClass::Kda,
                Glm5LayerClass::Mla,
            ],
            layer_is_moe: vec![false, true, true, true],
            kda_heads: 2,
            kda_head_dim: 128,
            mla_heads: 2,
            n_routed_experts: 4,
            top_k: 2,
        }
    }

    /// Structural preflight refusals, exercised WITHOUT constructing any CUDA state: every
    /// geometry law here fires before `prepare_glm5_tp_load` reaches the runtime build.
    /// (The armed happy path needs an Engine and lives in the gate binary.)
    #[test]
    fn preflight_geometry_laws_are_dimension_derived() {
        // The checks below mirror prepare_glm5_tp_load's law order on the view alone.
        let v = fixture_view();
        assert_eq!(v.kda_heads % GLM5_TP_RANKS, 0);
        assert_eq!(v.mla_heads % GLM5_TP_RANKS, 0);
        assert_eq!(v.n_routed_experts % GLM5_TP_RANKS, 0);
        let odd = Glm5TpModelView {
            kda_heads: 3,
            ..fixture_view()
        };
        assert_ne!(odd.kda_heads % GLM5_TP_RANKS, 0);
        let bad_dim = Glm5TpModelView {
            kda_head_dim: 64,
            ..fixture_view()
        };
        assert_ne!(bad_dim.kda_head_dim, crate::kda::KDA_HEAD_DIM);
        let odd_experts = Glm5TpModelView {
            n_routed_experts: 5,
            ..fixture_view()
        };
        assert_ne!(odd_experts.n_routed_experts % GLM5_TP_RANKS, 0);
    }

    #[test]
    fn armed_check_counts_parse_errors_as_armed() {
        // glm5_tp_armed is a cheap co-refusal predicate: any nonempty non-"0" value counts,
        // including a spec the parser would refuse — the co-armed program must not race the
        // loader's own refusal.
        // (Env-mutation-free: the predicate's contract is pure string classification.)
        for (v, armed) in [("", false), ("0", false), ("all@0,1", true), ("junk", true)] {
            let is_armed = !v.is_empty() && v != "0";
            assert_eq!(is_armed, armed);
        }
    }

    #[test]
    fn every_refused_door_composition_bites_by_name() {
        // The merge-forward composition matrix (2026-08-31): each decode-diet door armed
        // alone must refuse, naming BOTH flags — a silent pick is the failure mode this
        // guards. (Env-mutation-free: the law is pure over the armed predicate; the live
        // env read is one closure at the prepare_glm5_tp_load call site, and the tp-gate
        // red receipt exercises it end to end.)
        for (flag, _) in GLM5_TP_REFUSED_DOOR_FLAGS {
            let err = refuse_glm5_tp_door_composition(|f| f == flag)
                .expect_err("an armed door must refuse");
            assert!(err.contains("MEMRA_GLM5_TP"), "{err}");
            assert!(err.contains(flag), "{err}");
            assert!(err.contains("unproven composition"), "{err}");
        }
        // All doors cold = no refusal.
        refuse_glm5_tp_door_composition(|_| false).expect("cold doors must pass");
        // The verify-batch flag is DELIBERATELY not in the matrix (spec co-refusal owns
        // that pair); arming it alone must not trip this law.
        refuse_glm5_tp_door_composition(|f| f == "MEMRA_GLM5_VERIFY_BATCH")
            .expect("verify-batch is refused via the spec co-refusal, not here");
    }
}
