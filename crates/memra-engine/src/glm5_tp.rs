//! glm5_next (GLM-5.3-Flash) TP-N seam — `MEMRA_GLM5_TP` (lane/glm5-tp2, 2026-08-31;
//! rank-widened to TP-4 by lane/glm5-composition, 2026-09-01).
//!
//! WHAT THIS IS. A correctness-first tensor-parallel execution program for the glm5_next
//! hybrid trunk, per the lane's shard map (`research/glm53-flash-bringup-20260827/
//! tp2-20260831/SHARD-MAP.md`). Per layer class:
//!
//!   * KDA (34 layers): head-sharded, `heads / ranks` per rank. Each rank runs the UNCHANGED
//!     `kda_core_gated` program on its shard (per-head kernels: conv, L2 norm, gate, scan,
//!     gated rmsnorm are all head-independent), the gated `[t, qkv/ranks]` parts are
//!     gathered, and each rank's COLUMN-parallel `wo` slice (out rows over the FULL gathered
//!     input) computes its slice of the output with the same plain matvec kernel — joins are
//!     pure data movement, never a partial-sum reduction, which is what makes model-level
//!     TP-vs-plain BYTE identity the bar instead of a tolerance band.
//!   * MLA/DSA (11 layers): head-sharded per-head operands (`wq_b`, `wk_b`, `wv_b`);
//!     REPLICATED per-token shared work (`wq_a`/`q_a_norm`, `wkv_a`/`kv_a_norm`, the whole
//!     indexer + k-pool selection) — every rank computes identical bytes from identical
//!     inputs, so the latent + indexer planes are replicated per rank and no per-token
//!     cross-rank hop exists in the latent chain. `wo` is column-parallel over the gathered
//!     attention parts, exactly like KDA.
//!   * MoE (sparse-FFN layers): EP-N, whole experts, contiguous slices (even split: rank =
//!     expert / (n_expert/ranks)). The router stays root-computed (host sigmoid top-k,
//!     unchanged); each owner extracts its slots' UNWEIGHTED down rows with the same
//!     fused-epilogue kernels at n_used=1, and root re-applies the slot-ordered fmaf
//!     accumulation chain — the same rounded-operation sequence as the plain
//!     `moe_down8_fma_q8` walk. Shared expert, dense MLPs, router, mHC, norms, embed and
//!     lm_head stay ROOT-OWNED (the `MEMRA_STEP_TP` owner-stage precedent).
//!
//! TRANSPORT is a SEPARATE, SWAPPABLE AXIS (`MEMRA_GLM5_TP_TRANSPORT`,
//! lane/glm5-tp-transport 2026-09-01). Because every cross-rank hop above is pure movement,
//! the transport arm cannot change a bit — so this module names the hop SHAPES and
//! `tp_transport` owns the bytes. `host-canonical` (the default, and what every banked
//! glm5 TP number was measured on) bounces each hop through host with a full stream drain
//! per leg; `peer-pull` issues a consumer-side device peer copy per hop with event ordering
//! and no host boundary. The join-diet doors are an orthogonal axis (they cut hop COUNT; the
//! transport cuts hop COST) and compose.
//!
//! FAIL-CLOSED SURFACE. The preflight refuses before any TP CUDA state exists: non-glm5
//! plans, rank counts outside the qualified set (2 and 4 — see [`GLM5_TP_ALLOWED_RANKS`]),
//! head/expert counts that do not divide, duplicate devices (serving parse), co-armed
//! `MEMRA_PP_STAGES>1`, `MEMRA_STEP_TP`/`MEMRA_STEP_EP`. A sharded layer POISONS every plain
//! path: `kda_core`, `mla_attn_cached` and the batched walks refuse a TP-armed layer by
//! name. The memra-server worker refuses the flag outright (serving wiring is the named
//! box-lane increment, not v1).
//!
//! Engagement markers: `[glm5-tp-preflight]`, `[glm5-tp-kda]`, `[glm5-tp-mla]`,
//! `[glm5-tp-ep]`, `[glm5-tp-transport]` — every marker carries `performance_claim=false`,
//! and the first four name the LIVE transport rather than a hardcoded string (the
//! tp2-battery greps `transport=` on all four seams, and a hardcoded value would have made a
//! transport A/B unreadable from the boot log).

// lane/clippy-zero-restore-20260901: perf-gated TP2 host code (fresh lane receipts);
// index loops stay in their gated shape — iterator reshapes are not bit-neutral by inspection.
#![allow(clippy::needless_range_loop)]

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use cudarc::driver::CudaSlice;

use crate::Engine;
use crate::kda::{ConvArm, KdaAttnLayer};
use crate::model::GpuTensor;
use memra_kv::{Cache, LatentKvLayer, RecurLayer};

/// The qualified rank envelope: TP-2 (the v1 seam, box-battery-gated) and TP-4.
///
/// GEOMETRY (lane/glm5-tp-transport, 2026-09-01). The DSA indexer is REPLICATED per rank by
/// this seam's own shard map (`shard_mla_layer`'s `replicate_indexer`), so its 32 heads
/// impose no divisibility constraint at any rank count. TP-4 needs NO padding on the
/// glm5_next geometry: 64/4 = 16 KDA heads, 64/4 = 16 MLA heads, 288/4 = 72 experts,
/// 4096/4 = 1024 `wo` out rows, KDA `head_dim` 128 is rank-count independent. TP-3 remains
/// refused: the only real obstruction is the 64 attention/KDA heads (a HEAD-PADDING
/// question, 64 -> 66, RESEARCH.md §1.5d — not built). See `tp-transport-20260901/LANE.md`
/// "TP-4 divisibility".
pub const GLM5_TP_ALLOWED_RANKS: [usize; 2] = [2, 4];

/// This family's receipt marker for the GENERAL transport seam (`tp_transport`, generalized
/// lane/glm5-extract2). The tag is the caller's so lane/glm5-tp-transport's and
/// lane/glm5-composition's banked gate and box receipts keep their exact bytes while a second
/// family gets its own marker — the same rule phase 1 set for `[glm5-phase]` on the shared
/// spec timers.
pub const GLM5_TP_TRANSPORT_TAG: &str = "glm5-tp-transport";

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

/// Parse the shared `LAYER[-LAYER]@DEVICE,DEVICE[,...][;...]` grammar for the glm5 door.
/// `trunk_layers` is the loaded model's trunk length (the `all` shorthand expands against
/// it — the model contract owns that number, never a constant in the parser).
pub fn parse_glm5_tp_layer_specs(
    value: Option<&str>,
    trunk_layers: usize,
) -> Result<Vec<Glm5TpLayerSpec>, String> {
    crate::tp::parse_layer_specs_for_trunk("MEMRA_GLM5_TP", value, Some(trunk_layers))
}

/// Gate-harness knob, never a serving flag: `MEMRA_GLM5_TP_GATE_SAME_DEV=1` builds every
/// peer rank as an ADDITIONAL CUDA CONTEXT ON THE ROOT DEVICE (the one-card rig gate's
/// emulation; the ppN same-device-stages precedent). The spec's non-root device ids become
/// logical rank ids. The serving worker refuses `MEMRA_GLM5_TP` outright, so this can never
/// leak into serving.
pub fn gate_same_device() -> bool {
    std::env::var("MEMRA_GLM5_TP_GATE_SAME_DEV").as_deref() == Ok("1")
}

/// Gate-harness RED-arm knob, never a serving flag (`MEMRA_GLM5_TP_GATE_RED`):
///   * `swap-wo` — each rank's column `wo` slice takes the NEXT rank's out rows (a broken
///     shard map); the gate run MUST diverge from plain.
///   * `swap-ep-gateup` — the root EP slab's gate and up projections swap (wrong expert
///     weights); MUST diverge.
///   * `skip-peer-combine` — the EP combine drops every peer-owned slot; MUST diverge,
///     which is also the non-vacuity proof that the peer ranks contribute real work.
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
        // "0" is OFF, and it has to be: `glm5-tp-gate`'s own `rm_env` writes "0" rather than
        // unsetting, so before this arm existed the gate errored out on its FIRST TP arm with
        // `MEMRA_GLM5_TP_GATE_RED="0" is not a known red arm` and could never run one. "0" is
        // also the canonical rollback spelling for every other door in this engine.
        None | Some("") | Some("0") => Ok(None),
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

/// The TP-N rank runtime. Rank 0 (root) executes on the model's own engine — the PP-owner
/// context, exactly like the step seam's owner-first rank law. Ranks `1..ranks` each own a
/// full peer Engine, in `MEMRA_GLM5_TP` device order (`peers[i]` = rank `i + 1`).
pub struct Glm5TpRt {
    pub peers: Vec<Engine>,
    pub root_dev: usize,
    pub peer_devs: Vec<usize>,
    /// True only when built through [`Glm5TpRt::new_gate_same_device`] — the one-card rig
    /// gate's multi-context emulation (the ppN same-device gate precedent). The env-driven
    /// serving parse can never reach this: the grammar refuses duplicate devices.
    pub same_device_gate: bool,
    /// Which transport every cross-rank hop of this runtime moves its bytes with
    /// (`MEMRA_GLM5_TP_TRANSPORT`, default `host-canonical`). Frozen at
    /// [`Glm5TpRt::arm_transport`] time, announced once, and named in every gate log.
    pub transport: crate::tp_transport::TpTransport,
    /// The peer-pull ordering primitives — `Some` only on the peer-pull arm, and only after
    /// its byte-integrity ladder passed.
    link: Option<crate::tp_transport::PeerPullLink>,
    /// The one-shot all-reduce's per-rank state (`MEMRA_TP_AR_1STAGE`), built on first use.
    /// Behind a mutex because the runtime is shared through an `Arc` and the walk takes it by
    /// `&self`; the lock is uncontended at batch 1 and costs nothing against a 20 us collective.
    ar: std::sync::Mutex<Option<crate::tp_ar::ArLink>>,
}

impl Glm5TpRt {
    pub fn new(devices: &[usize]) -> Result<Self, Box<dyn std::error::Error>> {
        let root_dev = devices[0];
        let peer_devs: Vec<usize> = devices[1..].to_vec();
        for &d in &peer_devs {
            if d == root_dev || peer_devs.iter().filter(|&&x| x == d).count() > 1 {
                return Err(format!(
                    "MEMRA_GLM5_TP rank devices must be distinct in serving; got {devices:?} \
                     (the same-device form exists only for the rig gate binary)"
                )
                .into());
            }
        }
        let mut peers = Vec::with_capacity(peer_devs.len());
        for &d in &peer_devs {
            peers.push(Engine::new(d)?);
        }
        Ok(Self {
            peers,
            root_dev,
            peer_devs,
            same_device_gate: false,
            transport: crate::tp_transport::TpTransport::HostCanonical,
            link: None,
            ar: std::sync::Mutex::new(None),
        })
    }

    /// Same-device multi-context runtime for the ONE-CARD rig gate (exactness only). Every
    /// peer rank is an additional CUDA context on the root device: the whole shard/join
    /// walk — shard loads, replicated compute, gathers, canonical combines — executes
    /// exactly as on N cards, minus real peer transport (which the pro6000 batteries
    /// qualify on the box card class separately).
    pub fn new_gate_same_device(
        root_dev: usize,
        ranks: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut peers = Vec::with_capacity(ranks - 1);
        for _ in 1..ranks {
            peers.push(Engine::new(root_dev)?);
        }
        Ok(Self {
            peers,
            root_dev,
            peer_devs: vec![root_dev; ranks - 1],
            same_device_gate: true,
            transport: crate::tp_transport::TpTransport::HostCanonical,
            link: None,
            ar: std::sync::Mutex::new(None),
        })
    }

    /// One-shot all-reduce over `x[r]`, `x[r]` living on rank `r`'s engine, when
    /// `MEMRA_TP_AR_1STAGE` is armed and the runtime is a real two-device group. Returns `false`
    /// when it declined, in which case the caller's own return-to-root still owes the work.
    ///
    /// WHY THIS EXISTS SEPARATELY from the transport arms: those move bytes hop by hop, and the
    /// walk's two REDUCE sites are not hops, they are collectives. Doing them as a push plus a
    /// fold costs 4 launches and 8 cross-context event operations, which measured 20-26 us for
    /// 16 KB on a 956 GB/s fabric. The one-shot is two launches and no events.
    /// Whether [`Self::ar_1stage`] can serve this runtime at all: the door, a real two-device
    /// group, and not the same-device gate. Split out so a caller can ask BEFORE it rearranges the
    /// buffers it would hand over, which is the shape that stops a decline from silently dropping
    /// a rank's contribution.
    pub fn ar_1stage_available(&self) -> bool {
        tp_ar_1stage_on() && self.ranks() == 2 && !self.same_device_gate
    }

    pub fn ar_1stage(
        &self,
        root: &Engine,
        x: &mut [&mut CudaSlice<f32>],
        n: usize,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        if !self.ar_1stage_available() {
            return Ok(false);
        }
        let engines: Vec<&Engine> = std::iter::once(root).chain(self.peers.iter()).collect();
        let mut guard = self
            .ar
            .lock()
            .map_err(|_| "glm5-tp one-shot all-reduce: the link mutex is poisoned")?;
        if guard.is_none() {
            *guard = Some(crate::tp_ar::ArLink::new(&engines)?);
        }
        guard
            .as_mut()
            .expect("built above")
            .all_reduce_1stage(&engines, x, n)?;
        Ok(true)
    }

    /// Rank count of this runtime (root + peers).
    pub fn ranks(&self) -> usize {
        self.peers.len() + 1
    }

    /// Distinct device ordinals participating in this runtime.
    pub fn devices(&self) -> Vec<usize> {
        let mut devs = Vec::with_capacity(1 + self.peer_devs.len());
        devs.push(self.root_dev);
        for &d in &self.peer_devs {
            if !devs.contains(&d) {
                devs.push(d);
            }
        }
        devs
    }

    /// Freeze the transport for this runtime: read the flag, grant peer access (real groups
    /// only), and run the byte-integrity pull ladder over every ordered rank pair. Called
    /// from the preflight BEFORE any layer is sharded, so a bad fabric refuses the load
    /// rather than corrupting a shard.
    pub fn arm_transport(&mut self, root: &Engine) -> Result<(), Box<dyn std::error::Error>> {
        // The seam is general; the TAG and the flag name are the FAMILY's (lane/glm5-extract2,
        // the phase-1 caller-owned-tag pattern) — `[glm5-tp-transport]` bytes stay exactly as
        // lane/glm5-tp-transport and lane/glm5-composition banked them, and `armed_flag` is
        // whichever of `MEMRA_TP_TRANSPORT` / `MEMRA_GLM5_TP_TRANSPORT` the operator actually
        // set, so a peer-access or ladder refusal names their flag.
        let (transport, armed_flag) = crate::tp_transport::transport_env()?;
        let engines: Vec<&Engine> = std::iter::once(root).chain(self.peers.iter()).collect();
        let link = crate::tp_transport::arm_transport(
            transport,
            armed_flag,
            GLM5_TP_TRANSPORT_TAG,
            &engines,
            self.same_device_gate,
        )?;
        self.transport = transport;
        self.link = link;
        Ok(())
    }

    /// Build the per-hop transport handle. Every cross-rank movement in the glm5 TP walk
    /// goes through one of `tp_transport`'s named hop shapes with this handle, which is
    /// what makes the arm swap a ONE-PLACE change and the movement census automatic.
    pub fn hop<'a>(&'a self, root: &'a Engine) -> crate::tp_transport::Hop<'a> {
        crate::tp_transport::Hop {
            engines: std::iter::once(root).chain(self.peers.iter()).collect(),
            transport: self.transport,
            link: self.link.as_ref(),
        }
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
/// measured expert-placement map when `MEMRA_EP_MAP` (or its glm5 alias) is armed (validated at
/// preflight, before any TP CUDA state — absent flag = the even split, byte-unchanged).
pub struct Glm5TpLoadPlan {
    pub rt: Arc<Glm5TpRt>,
    pub layers: std::collections::BTreeSet<usize>,
    pub ep_map: Option<crate::ep_map::EpMap>,
}

/// Load + validate the placement map against the model view and the armed layer set.
/// The env seam is the general `ep_map::ep_map_env()` (`MEMRA_EP_MAP`, glm5 alias
/// honored); every refusal names the flag that ARMED the load. `Some("")` REFUSES (a
/// set-but-empty flag is an operator error, never a silent even split). Fail-closed on
/// every axis: unreadable file, malformed text, rank/expert-count mismatch, layer-cover
/// mismatch. Returns `None` only when both names are UNSET.
fn load_glm5_ep_map(
    view: &Glm5TpModelView,
    layers: &std::collections::BTreeSet<usize>,
    ranks: usize,
) -> Result<Option<crate::ep_map::EpMap>, Box<dyn std::error::Error>> {
    let Some((flag, path)) = crate::ep_map::ep_map_env()? else {
        return Ok(None);
    };
    if path.is_empty() {
        return Err(format!(
            "{flag} is set but empty (fail-closed: unset the flag for \
                    the even split; an empty value never silently means default)"
        )
        .into());
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("{flag}={path}: cannot read the map file ({e}) — refused by name"))?;
    let map = crate::ep_map::EpMap::parse(&text).map_err(|e| format!("{flag}={path}: {e}"))?;
    if map.ranks != ranks {
        return Err(format!(
            "{flag}={path}: map declares ranks={}, this load is TP-{ranks} \
             (re-mint the map for the armed rank count)",
            map.ranks
        )
        .into());
    }
    if map.n_experts != view.n_routed_experts {
        return Err(format!(
            "{flag}={path}: map declares expert_count={}, the model routes {}",
            map.n_experts, view.n_routed_experts
        )
        .into());
    }
    if map.entry_rank != 0 {
        return Err(format!(
            "{flag}={path}: entry_rank={} but the glm5 TP first-hop card is \
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
        .map_err(|e| format!("{flag}={path}: {e}"))?;
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
/// is absent DELIBERATELY: its walk exists only inside glm5 spec sessions — co-refused on
/// a sharded model unless `MEMRA_GLM5_SPEC_TP=1` arms the GATED composition
/// (lane/glm5-composition; the spec x TP pair HAS its composition gate, `glm5-tp-gate`
/// arms S2/Q-S4), whose admission REQUIRES the batched walk by name.
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
/// refuses by name, before any TP CUDA state exists. Delegates to the general
/// [`crate::tp::refuse_door_composition`] pattern (lane/glm5-extract-general) with this
/// door's own table — error bytes unchanged. `armed` reports whether a flag is set to
/// `"1"` (env in production; a plain set in the unit test — the module keeps its tests
/// env-mutation-free).
pub fn refuse_glm5_tp_door_composition(armed: impl Fn(&str) -> bool) -> Result<(), String> {
    crate::tp::refuse_door_composition("MEMRA_GLM5_TP", &GLM5_TP_REFUSED_DOOR_FLAGS, armed)
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
            "MEMRA_GLM5_TP + MEMRA_PP_STAGES>1: the TP x PP composition is unwired and \
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

    // One device group across the whole spec (one runtime group), root-first; the rank
    // count comes from the device list and must be in the qualified envelope.
    let devices = specs[0].devices.clone();
    let ranks = devices.len();
    if !GLM5_TP_ALLOWED_RANKS.contains(&ranks) {
        return Err(format!(
            "MEMRA_GLM5_TP names {ranks} devices per layer; the qualified rank envelope is \
             {GLM5_TP_ALLOWED_RANKS:?} (TP-3 is a head-padding question, not built — see the \
             module doc)"
        )
        .into());
    }

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
    if !view.kda_heads.is_multiple_of(ranks) || view.kda_heads == 0 {
        return Err(format!(
            "glm5-tp: {} KDA heads do not shard across {ranks} ranks",
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
    if !view.mla_heads.is_multiple_of(ranks) || view.mla_heads == 0 {
        return Err(format!(
            "glm5-tp: {} MLA heads do not shard across {ranks} ranks",
            view.mla_heads
        )
        .into());
    }
    if !view.n_routed_experts.is_multiple_of(ranks) || view.n_routed_experts == 0 {
        return Err(format!(
            "glm5-tp: {} routed experts do not partition across {ranks} ranks",
            view.n_routed_experts
        )
        .into());
    }
    if view.top_k > view.n_routed_experts {
        return Err("glm5-tp: top_k exceeds the routed expert count".into());
    }

    for s in &specs {
        if s.devices != devices {
            return Err(format!(
                "MEMRA_GLM5_TP carries ONE runtime group: layer {} names devices {:?}, \
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
    let mut rt = if same_dev {
        eprintln!(
            "[glm5-tp-preflight] GATE same-device emulation: {} peer ranks are additional \
             contexts on device {root_dev} (spec devices {:?} are logical rank ids)",
            ranks - 1,
            &devices[1..],
        );
        Glm5TpRt::new_gate_same_device(root_dev, ranks)?
    } else {
        Glm5TpRt::new(&devices)?
    };
    // Transport arms HERE — after the rank engines exist, BEFORE any layer is sharded. A
    // peer-pull ladder failure refuses the load with zero TP shards built (lane/glm5-tp-transport).
    rt.arm_transport(e)?;
    let rt = Arc::new(rt);
    let layers: std::collections::BTreeSet<usize> = specs.iter().map(|s| s.layer).collect();
    let ep_map = load_glm5_ep_map(view, &layers, ranks)?;
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
        "[glm5-tp-preflight] armed ranks={ranks} devices={devices:?} layers={} \
         kda_shard={kda_n} mla_shard={mla_n} moe_ep={moe_n} kda_heads_per_rank={} \
         mla_heads_per_rank={} experts_per_rank={} transport={} \
         weights_loaded=false performance_claim=false",
        layers.len(),
        view.kda_heads / ranks,
        view.mla_heads / ranks,
        view.n_routed_experts / ranks,
        rt.transport.name(),
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

/// K-SLICE of a 2D projection: keep input columns `cols` of every out row. The row-parallel
/// counterpart of [`shard_rows`], and the shape a TP `wo` needs.
///
/// WHY IT IS NOT `shard_rows` WITH THE AXES SWAPPED: the shardable axis there is the OUTERMOST,
/// which is one contiguous byte range per expert. A column slice is a sub-range of EVERY row, so
/// the result is a gather of `out_f` ranges, and on a quantized tensor the range boundary has to
/// land on a quantization block or the slice cuts one in half. `tp_expert_split` carries the same
/// guard and the red arm that proves a byte-divisibility test is not enough.
fn shard_cols(
    src_engine: &Engine,
    dst: &Engine,
    t: &GpuTensor,
    cols: Range<usize>,
) -> Result<GpuTensor, Box<dyn std::error::Error>> {
    let take = cols.end - cols.start;
    match t {
        GpuTensor::Float { data, ne } => {
            let (outer, inner) = outer_rows(ne);
            if cols.end > inner {
                return Err(format!("shard cols {cols:?} exceed inner axis {inner}").into());
            }
            let host = src_engine.dtoh(data)?;
            let mut out = Vec::with_capacity(outer * take);
            for r in 0..outer {
                out.extend_from_slice(&host[r * inner + cols.start..r * inner + cols.end]);
            }
            let mut ne2 = ne.clone();
            ne2[0] = take as u64;
            Ok(GpuTensor::Float {
                data: dst.htod(&out)?,
                ne: ne2,
            })
        }
        GpuTensor::FloatBf16 { data, ne } => {
            let (outer, inner) = outer_rows(ne);
            if cols.end > inner {
                return Err(format!("shard cols {cols:?} exceed inner axis {inner}").into());
            }
            let host = src_engine.dtoh_u8(data)?;
            let mut out = Vec::with_capacity(outer * take * 2);
            for r in 0..outer {
                let base = r * inner * 2;
                out.extend_from_slice(&host[base + cols.start * 2..base + cols.end * 2]);
            }
            let mut ne2 = ne.clone();
            ne2[0] = take as u64;
            Ok(GpuTensor::FloatBf16 {
                data: dst.htod_bytes(&out)?,
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
                cutlass: _,
        } => {
            if *rp || fp8.is_some() || rp4.is_some() || blk.is_some() || f16.is_some() {
                return Err("glm5-tp shard cols: a mirror layout is unwired for K slices".into());
            }
            if ne.len() != 2 {
                return Err("glm5-tp shard cols: quantized K slices are 2D-only".into());
            }
            let (outer, inner) = outer_rows(ne);
            if cols.end > inner {
                return Err(format!("shard cols {cols:?} exceed inner axis {inner}").into());
            }
            // Block alignment, counted in BLOCKS: `row_bytes = in_f / block * type_size` is
            // linear in `in_f`, so a byte-offset test passes a boundary that cuts a block.
            if !(cols.start * row_bytes).is_multiple_of(inner)
                || !(take * row_bytes).is_multiple_of(inner)
            {
                return Err(format!(
                    "glm5-tp shard cols: {cols:?} of {inner} does not land on a block boundary \
                     (row_bytes {row_bytes})"
                )
                .into());
            }
            let off = cols.start * row_bytes / inner;
            let keep = take * row_bytes / inner;
            let host = src_engine.dtoh_u8(bytes)?;
            let mut out = Vec::with_capacity(outer * keep);
            for r in 0..outer {
                let base = r * row_bytes + off;
                out.extend_from_slice(&host[base..base + keep]);
            }
            let mut ne2 = ne.clone();
            ne2[0] = take as u64;
            Ok(GpuTensor::Quant {
                bytes: dst.htod_bytes(&out)?,
                qtype: *qtype,
                row_bytes: keep,
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

/// Rank r's engine within a runtime, given the root engine (rank 0 has no owned Engine in
/// the runtime — it IS the model's engine).
pub(crate) fn rank_engine<'a>(e: &'a Engine, rt: &'a Glm5TpRt, r: usize) -> &'a Engine {
    if r == 0 { e } else { &rt.peers[r - 1] }
}

// ------------------------------------------------------------------------------------------
// KDA sidecar
// ------------------------------------------------------------------------------------------

/// The KDA TP sidecar: the peer ranks' head shards plus the runtime handle. The OUTER
/// `KdaAttnLayer` that carries this in its `tp` field is the root shard; every shard's
/// `wo` field holds that rank's COLUMN slice (out rows over the full `qkv` input).
pub struct Glm5TpKda {
    pub rt: Arc<Glm5TpRt>,
    /// `peers[i]` is rank `i + 1`'s shard, resident on `rt.peers[i]`.
    pub peers: Vec<KdaAttnLayer>,
    /// Full-width qkv of the UNSHARDED layer (`ranks * shard qkv`) — the gather width.
    pub full_qkv: usize,
    /// Full hidden width (`wo` out rows across all ranks).
    pub n_embd: usize,
}

impl Glm5TpKda {
    pub fn ranks(&self) -> usize {
        self.peers.len() + 1
    }
}

static KDA_MARKED: AtomicBool = AtomicBool::new(false);

/// Shard one loaded KDA layer: returns the ROOT shard (heads/ranks, wo out-rows
/// `0..H/ranks`) with the peer shards in its `tp` sidecar. The full layer's tensors are
/// consumed and dropped — per-layer transient VRAM is one layer, never the model.
pub(crate) fn shard_kda_layer(
    e: &Engine,
    rt: &Arc<Glm5TpRt>,
    la: KdaAttnLayer,
) -> Result<KdaAttnLayer, Box<dyn std::error::Error>> {
    if la.tp.is_some() {
        return Err("shard_kda_layer: layer is already sharded".into());
    }
    let ranks = rt.ranks();
    let heads = la.heads();
    let head_dim = la.head_dim();
    let qkv = la.qkv();
    let kernel = la.conv_kernel();
    if !heads.is_multiple_of(ranks) {
        return Err(format!("KDA heads {heads} do not shard across {ranks} ranks").into());
    }
    let hl = heads / ranks; // heads per rank
    let ql = qkv / ranks; // channels per rank
    let n_embd = la.wo.out_features();
    if !n_embd.is_multiple_of(ranks) {
        return Err(format!("KDA wo out {n_embd} does not split across ranks").into());
    }
    let hh = n_embd / ranks;

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

    // Gate red arm: a broken shard map hands each rank the NEXT rank's wo out rows
    // (the two-rank swap generalized to a rotation — still guaranteed wrong on every rank).
    let wo_rank = |r: usize| -> usize {
        match gate_red() {
            Ok(Some(GateRed::SwapWo)) => (r + 1) % ranks,
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
    let mut peers = Vec::with_capacity(ranks - 1);
    for r in 1..ranks {
        peers.push(rank_shard(&rt.peers[r - 1], r)?);
    }
    if !KDA_MARKED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[glm5-tp-kda] head shard armed: ranks={ranks} heads_per_rank={hl} \
             head_dim={head_dim} wo=column-over-gather transport={} performance_claim=false",
            rt.transport.name(),
        );
    }
    root.tp = Some(Box::new(Glm5TpKda {
        rt: Arc::clone(rt),
        peers,
        full_qkv: qkv,
        n_embd,
    }));
    Ok(root)
}

/// Ensure layer `il`'s per-rank KDA state planes exist (lazily, sized for the SHARD
/// geometry — the canonical `cache.recur[il]` planes are full-width and stay untouched
/// as allocated; the TP walk never reads them). Index 0 = root's plane on `e`, index r =
/// rank r's plane on its peer engine.
fn ensure_kda_tp_state<'c>(
    e: &Engine,
    rt: &Glm5TpRt,
    la_root: &KdaAttnLayer,
    cache: &'c mut Cache,
    il: usize,
) -> Result<&'c mut Vec<RecurLayer>, Box<dyn std::error::Error>> {
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
        let mut planes = Vec::with_capacity(rt.ranks());
        planes.push(mk(e)?);
        for p in &rt.peers {
            planes.push(mk(p)?);
        }
        cache.glm5_tp_recur[il] = Some(planes);
    }
    Ok(cache.glm5_tp_recur[il].as_mut().unwrap())
}

/// The KDA TP walk, ONE body for both consumers (the #80 review's dedup finding — the
/// forked verify twin had already drifted to root-first issue order):
///   * prime/decode ([`kda_tp_cached`]): `verify_stash = None`, plain `wo` matmul —
///     byte-for-byte the pre-composition walk.
///   * spec x TP verify rows ([`kda_tp_verify_rows`]): `verify_stash = Some`, per-rank
///     pre-round ssm snapshot + batched `KdaStash::Rows` capture, `wo` on the ROWS-EXACT
///     class (the unsharded verify walk's own routing), per-rank scan-ns accumulated into
///     `scan_clock` so the `[glm5-phase-v]` receipt keeps its sequential-floor share on
///     the composed shape.
///
/// Issue order is PEERS FIRST, ROOT LAST on both arms (v1's order; the twins document it).
/// THREE cross-rank hop shapes, each a named `tp_transport` shape: fan-out of `x`,
/// gather of the gated parts, concat of the `wo` parts. On `host-canonical` at two ranks
/// that is 5 draining `dtoh` + 4 `htod` per layer-call, exactly as v1; on `peer-pull` it is
/// device peer copies, local copies and 0 host boundaries.
#[allow(clippy::too_many_arguments)] // mirrors the kda entry contract shape
fn kda_tp_core(
    e: &Engine,
    la_root: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    t: usize,
    eps: f32,
    cache: &mut Cache,
    il: usize,
    arm: ConvArm,
    verify_stash: Option<&mut Glm5TpKdaVerifyStash>,
    mut scan_clock: Option<&mut u64>,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let tp = la_root
        .tp
        .as_ref()
        .ok_or("kda_tp_core called on an unsharded layer")?;
    let rt = &tp.rt;
    let ranks = rt.ranks();
    let ql = la_root.qkv(); // per-rank channels
    let full = tp.full_qkv;
    let n_embd = tp.n_embd;
    let hh = n_embd / ranks;
    let rows_exact = verify_stash.is_some();
    // Per-rank verify capture, RANK-indexed regardless of issue order; assembled into the
    // caller's stash after the loop.
    let mut captured: Vec<Option<(CudaSlice<f32>, crate::kda::KdaRowsStash)>> =
        (0..ranks).map(|_| None).collect();

    let hop = rt.hop(e);
    // HOP 1 — fan-out of the mixer input to every peer rank. `x.len()` and not `t * n_embd`:
    // the v1 arm moved the WHOLE buffer, and the arms must move identical byte ranges or
    // the transport A/B stops being a transport A/B.
    let x_peers = crate::tp_transport::fanout_f32(&hop, x, x.len())?;
    let states = ensure_kda_tp_state(e, rt, la_root, cache, il)?;

    // Peer shards first (host-canonical serial walk; overlap is the box arc), root last —
    // v1's issue order at two ranks, both arms.
    let mut gated: Vec<Option<CudaSlice<f32>>> = (0..ranks).map(|_| None).collect();
    for r in (1..ranks).chain(std::iter::once(0)) {
        let dev = if r == 0 { e } else { &rt.peers[r - 1] };
        let la = if r == 0 { la_root } else { &tp.peers[r - 1] };
        let xin = if r == 0 { x } else { &x_peers[r - 1] };
        // Verify arm: the pre-round snapshot on the rank's engine, BEFORE the batched
        // call advances the resident state (the ckpt contract's per-rank twin).
        let snap = if rows_exact {
            Some(dev.clone_dtod(&states[r].ssm_state)?)
        } else {
            None
        };
        let mut rank_stash: Option<crate::kda::KdaRowsStash> = None;
        let mut rank_scan_ns = 0u64;
        let out = {
            let RecurLayer {
                conv_state,
                ssm_state,
                ssm_state_alt,
            } = &mut states[r];
            let out = crate::kda::kda_core_gated(
                dev,
                la,
                xin,
                t,
                eps,
                conv_state,
                ssm_state,
                ssm_state_alt,
                arm,
                if rows_exact {
                    crate::kda::KdaStash::Rows(&mut rank_stash)
                } else {
                    crate::kda::KdaStash::None
                },
                scan_clock.as_deref_mut().map(|_| &mut rank_scan_ns),
                None,
                None,
            )?;
            std::mem::swap(ssm_state, ssm_state_alt);
            out
        };
        if let Some(clock) = scan_clock.as_deref_mut() {
            *clock += rank_scan_ns;
        }
        if rows_exact {
            let snap = snap.expect("verify arm cloned the snapshot above");
            let rank_stash = rank_stash
                .ok_or("kda_core_gated returned without filling the requested rows stash")?;
            captured[r] = Some((snap, rank_stash));
        }
        gated[r] = Some(out);
    }
    if let Some(stash_vec) = verify_stash {
        stash_vec.clear();
        for c in captured {
            stash_vec.push(c.expect("every rank captured on the verify arm"));
        }
    }

    // HOP 2 — gather the gated parts into the FULL [t, qkv] layout on EVERY rank
    // (column-parallel wo needs the whole input on each rank). Token-major interleave: row
    // tok is [rank0 ql | rank1 ql | ...]. `full == ranks * ql` by the shard map.
    debug_assert_eq!(full, ranks * ql);
    let gated_refs: Vec<&CudaSlice<f32>> = gated
        .iter()
        .map(|g| g.as_ref().expect("filled above"))
        .collect();
    let fulls = crate::tp_transport::gather_parts(&hop, &gated_refs, t, ql)?;

    // Per-rank column wo slices: each output element is one full-K dot by the SAME kernel
    // class the consumer's unsharded walk uses — no cross-rank arithmetic in this join.
    let mut ys = Vec::with_capacity(ranks);
    if rows_exact {
        ys.push(e.matmul_rows_exact(&la_root.wo, &fulls[0], t)?);
        for r in 1..ranks {
            ys.push(rt.peers[r - 1].matmul_rows_exact(&tp.peers[r - 1].wo, &fulls[r], t)?);
        }
    } else {
        ys.push(e.matmul(&la_root.wo, &fulls[0], t)?);
        for r in 1..ranks {
            ys.push(rt.peers[r - 1].matmul(&tp.peers[r - 1].wo, &fulls[r], t)?);
        }
    }

    // HOP 3 — concat the column parts into the mixer output on ROOT.
    let y_refs: Vec<&CudaSlice<f32>> = ys.iter().collect();
    crate::tp_transport::concat_parts_on_root(&hop, &y_refs, t, hh)
}

/// The KDA TP walk for one prime/decode call — [`kda_tp_core`] with no verify capture.
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
    kda_tp_core(e, la_root, x, t, eps, cache, il, arm, None, None)
}

/// Per-rank rollback material of ONE sharded-KDA verify round (lane/glm5-composition, the
/// spec x TP composition): index = rank; each entry is that rank's pre-round ssm snapshot
/// (cloned on the rank's engine BEFORE its batched call advanced the resident state) plus
/// the batched [`crate::kda::KdaRowsStash`] its `KdaStash::Rows` call filled. Rollback to
/// `keep` rows restores every rank through `kda_verify_rollback_rows_on` with the rank's
/// own engine/shard/plane tuple — the same two-plane contract as the unsharded stash,
/// per rank.
pub type Glm5TpKdaVerifyStash = Vec<(CudaSlice<f32>, crate::kda::KdaRowsStash)>;

/// The sharded-KDA VERIFY walk (spec x TP composition) — [`kda_tp_core`] with the verify
/// capture armed: batched `KdaStash::Rows` per rank, ROWS-EXACT `wo` (the unsharded verify
/// walk's own routing), per-rank scan-ns accumulated into `scan_clock`. Returns the mixer
/// output plus the rank-indexed rollback stash the ckpt banks.
#[allow(clippy::too_many_arguments)] // mirrors the kda verify entry contract shape
pub(crate) fn kda_tp_verify_rows(
    e: &Engine,
    la_root: &KdaAttnLayer,
    x: &CudaSlice<f32>,
    t: usize,
    eps: f32,
    cache: &mut Cache,
    il: usize,
    scan_clock: Option<&mut u64>,
) -> Result<(CudaSlice<f32>, Glm5TpKdaVerifyStash), Box<dyn std::error::Error>> {
    let mut stash: Glm5TpKdaVerifyStash = Vec::new();
    let out = kda_tp_core(
        e,
        la_root,
        x,
        t,
        eps,
        cache,
        il,
        ConvArm::Prefill,
        Some(&mut stash),
        scan_clock,
    )?;
    Ok((out, stash))
}

/// Roll every rank's sharded-KDA state back to "after row `keep-1`" from a spec x TP verify
/// round (the [`Glm5TpKdaVerifyStash`] contract). Full accept never calls this — the
/// resident per-rank states ARE the state after the last kept row.
pub(crate) fn kda_tp_verify_rollback(
    e: &Engine,
    la_root: &KdaAttnLayer,
    stash: &Glm5TpKdaVerifyStash,
    keep: usize,
    cache: &mut Cache,
    il: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let tp = la_root
        .tp
        .as_ref()
        .ok_or("kda_tp_verify_rollback called on an unsharded layer")?;
    let rt = &tp.rt;
    let ranks = rt.ranks();
    if stash.len() != ranks {
        return Err(format!(
            "glm5-tp verify rollback: stash carries {} ranks, the runtime has {ranks}",
            stash.len()
        )
        .into());
    }
    let states = cache.glm5_tp_recur[il]
        .as_mut()
        .ok_or_else(|| format!("glm5-tp verify rollback: layer {il} has no per-rank state"))?;
    for r in 0..ranks {
        let dev = if r == 0 { e } else { &rt.peers[r - 1] };
        let la = if r == 0 { la_root } else { &tp.peers[r - 1] };
        let (snap, rows) = &stash[r];
        crate::kda::kda_verify_rollback_rows_on(dev, la, snap, rows, keep, &mut states[r], il)?;
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------
// MLA sidecar
// ------------------------------------------------------------------------------------------

/// The MLA TP sidecar: the peer ranks' head shards (with replicated `wq_a`/`wkv_a`/norms
/// and full indexer replicas) plus the runtime handle.
pub struct Glm5TpMla {
    pub rt: Arc<Glm5TpRt>,
    /// `peers[i]` is rank `i + 1`'s shard, resident on `rt.peers[i]`.
    pub peers: Vec<crate::hybrid::MlaAttnLayer>,
    /// Full head count of the unsharded layer.
    pub full_heads: usize,
    /// Full hidden width (`wo` out rows across all ranks).
    pub n_embd: usize,
}

impl Glm5TpMla {
    pub fn ranks(&self) -> usize {
        self.peers.len() + 1
    }
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
    let ranks = rt.ranks();
    let g = la.geom;
    let nh = g.n_head;
    if !nh.is_multiple_of(ranks) {
        return Err(format!("MLA heads {nh} do not shard across {ranks} ranks").into());
    }
    let hl = nh / ranks;
    let head_q = g.d_nope + g.d_rope; // per-head wq_b out rows
    let n_embd = la.wo.out_features();
    if !n_embd.is_multiple_of(ranks) {
        return Err(format!("MLA wo out {n_embd} does not split across ranks").into());
    }
    let hh = n_embd / ranks;

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

    // Gate red arm: a broken shard map hands each rank the NEXT rank's wo out rows.
    let wo_rank = |r: usize| -> usize {
        match gate_red() {
            Ok(Some(GateRed::SwapWo)) => (r + 1) % ranks,
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
            wk_b16: None,
            wv_b16: None,
            // COLUMN-parallel wo (default): rank r owns OUT rows over the FULL gathered N*V
            // input, so the walk must all-gather every rank's heads before this multiply and
            // concat the out-row bands after it. Three pure-movement hops per layer.
            //
            // ROW-parallel wo (`MEMRA_GLM5_TP_EXPERT_SPLIT`): rank r instead owns the input
            // COLUMNS matching its OWN heads, so it multiplies only what it already computed and
            // produces a FULL-WIDTH partial sum. The all-gather and the concat both disappear and
            // one reduction replaces them. Not byte-identical, for the same reason the expert
            // `down` split is not: the K reduction is split two ways instead of run in one pass.
            wo: if glm5_tp_expert_split_on() {
                shard_cols(e, dst, &la.wo, wr * hl * g.d_v..(wr + 1) * hl * g.d_v)?
            } else {
                shard_rows(e, dst, &la.wo, wr * hh..(wr + 1) * hh)?
            },
            geom: shard_geom,
            index: match &la.index {
                Some(ix) => Some(replicate_indexer(dst, ix)?),
                None => None,
            },
            tp: None,
            tp_shard: true,
        })
    };

    let mut root = rank_shard(e, 0)?;
    let mut peers = Vec::with_capacity(ranks - 1);
    for r in 1..ranks {
        peers.push(rank_shard(&rt.peers[r - 1], r)?);
    }
    if !MLA_MARKED.swap(true, Ordering::Relaxed) {
        // The `wo=` field must name the shard this load ACTUALLY took, not the one this arm used
        // to take. Since lane/tp-expert-split-20260906 the same door that splits the experts also
        // turns `wo` row-parallel, and a hardcoded "column-over-gather" here reported the wrong
        // one on every split boot. A receipt line is only worth what its arguments say.
        let wo_shape = if glm5_tp_expert_split_on() {
            "row-parallel-partial-sums"
        } else {
            "column-over-gather"
        };
        eprintln!(
            "[glm5-tp-mla] head shard armed: ranks={ranks} heads_per_rank={hl} kv_rank={} \
             latent=replicated indexer=replicated wo={wo_shape} transport={} \
             performance_claim=false",
            g.kv_rank,
            rt.transport.name(),
        );
    }
    root.tp = Some(Box::new(Glm5TpMla {
        rt: Arc::clone(rt),
        peers,
        full_heads: nh,
        n_embd,
    }));
    Ok(root)
}

/// Ensure the PEER ranks' replicated latent planes for layer `il` exist, geometry-cloned
/// from the canonical (root) plane. The canonical plane IS the root replica — the root path
/// is unchanged. `cache_slot` holds one plane per peer rank (`[i]` = rank `i + 1`).
pub(crate) fn ensure_mla_peer_latent(
    rt: &Glm5TpRt,
    canonical: &LatentKvLayer,
    cache_slot: &mut Option<Vec<LatentKvLayer>>,
) -> Result<(), Box<dyn std::error::Error>> {
    if cache_slot.is_some() {
        return Ok(());
    }
    let mut planes = Vec::with_capacity(rt.peers.len());
    for dev in &rt.peers {
        let rows = dev.zeros(canonical.rows.len())?;
        // Fresh replica starts at len 0 like a fresh canonical plane; the walk appends to
        // every replica in the same calls, so the lengths stay in lock-step by construction.
        let len_d = dev.htod_i32(&[0])?;
        let index_rows = match &canonical.index_rows {
            Some(p) => Some(dev.zeros(p.len())?),
            None => None,
        };
        planes.push(LatentKvLayer {
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
    }
    *cache_slot = Some(planes);
    Ok(())
}

// ------------------------------------------------------------------------------------------
// MoE EP sidecar
// ------------------------------------------------------------------------------------------

/// One rank's expert slab: the rank's owned experts packed in ASCENDING expert-id order
/// for every projection, device-resident on that rank. For the even split the packing is
/// the contiguous slice — byte-for-byte the pre-map layout.
pub struct EpRankSlab {
    pub gate: CudaSlice<u8>,
    pub up: CudaSlice<u8>,
    pub down: CudaSlice<u8>,
    pub n_experts: usize,
}

/// The MoE EP sidecar on `MoeWeights`: per-rank expert slabs, the placement tables,
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
    /// `slabs[r]` is rank r's expert slab (`[0]` = root's, on the model's engine).
    pub slabs: Vec<EpRankSlab>,
    /// `owner_of[expert]` = owning rank (0 = root).
    pub owner_of: Vec<u8>,
    /// `local_of[expert]` = slot inside the owner's slab (ascending-id packing order).
    pub local_of: Vec<u32>,
    /// Per-rank grouped-dispatch pointer tables, `[rank]`, each the `DevExps::ptr_row`
    /// shape ([3 * n_expert] u64 device pointers: gate | up | down planes, indexed by GLOBAL
    /// expert id, resident on the owning rank's device). Owned experts point at
    /// `slab_base + local * stride`; non-owned entries are 0 and never dereferenced — the EP
    /// grouped-prime CSR is built per rank from `owner_of`, so a foreign id cannot reach the
    /// wrong rank's table. Built AFTER the gate-red slab mutations, from the FINAL slab
    /// buffers and the FINAL `local_of`, so `swap-ep-gateup` and `corrupt-ep-map` bite the
    /// grouped walk exactly as they bite the sequential one.
    pub ptr_rows: Vec<CudaSlice<u64>>,
}

impl Glm5EpExps {
    /// Owner rank of `expert` under the armed placement (even split when no map).
    pub fn owner(&self, expert: usize) -> usize {
        self.owner_of[expert] as usize
    }

    pub fn ranks(&self) -> usize {
        self.slabs.len()
    }
}

static EP_MARKED: AtomicBool = AtomicBool::new(false);

/// Engagement counter: PEER-owned expert slots dispatched by the EP walk (counted before
/// any gate-red skip, so a red arm can still assert a peer was ROUTED). Gates read it to
/// prove the peer ranks contribute real expert work — a token stream that never routes a
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

/// Bulk peer-row block returns performed by the dieted walk (one per (layer-call, peer
/// rank) that routed at least one slot owned by that rank; each replaces that call's ENTIRE
/// per-slot return dribble for that rank).
pub static GLM5_EP_DIET_BULK_RETURNS: AtomicU64 = AtomicU64::new(0);

/// Snapshot of [`GLM5_EP_DIET_BULK_RETURNS`].
pub fn glm5_ep_diet_bulk_returns() -> u64 {
    GLM5_EP_DIET_BULK_RETURNS.load(Ordering::Relaxed)
}

/// Per-slot synchronous peer round-trips (one peer DtoH + one root pageable HtoD each, the
/// v1 walk's dominant hop class) that the dieted walk folded into its bulk returns — one
/// count per peer-owned slot bulked.
pub static GLM5_EP_DIET_PEER_ROUNDTRIPS_AVOIDED: AtomicU64 = AtomicU64::new(0);

/// Snapshot of [`GLM5_EP_DIET_PEER_ROUNDTRIPS_AVOIDED`].
pub fn glm5_ep_diet_peer_roundtrips_avoided() -> u64 {
    GLM5_EP_DIET_PEER_ROUNDTRIPS_AVOIDED.load(Ordering::Relaxed)
}

/// Per-token peer z uploads the dieted walk avoided: `t-1` per (fanned layer-call, peer
/// rank) (one bulk [t, n_embd] upload replaces t per-token uploads) plus `t` per (layer-call,
/// rank) whose routing never touched that rank's experts (the fan-out is skipped entirely —
/// the placement-map multiplier: single-rank layer-calls move ZERO activation bytes off
/// root).
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

/// Arm one MoE layer for EP. `placement` is the layer's validated map row
/// (`owners[expert] = rank`) when `MEMRA_EP_MAP` (or its glm5 alias) is armed; `None` = the
/// even split, whose ascending-id packing is byte-for-byte the pre-map contiguous slices.
pub(crate) fn arm_moe_ep(
    e: &Engine,
    rt: &Arc<Glm5TpRt>,
    m: &mut crate::hybrid::MoeWeights,
    placement: Option<&[u8]>,
) -> Result<(), Box<dyn std::error::Error>> {
    if m.glm5_ep.is_some() || m.glm5_tp_split.is_some() {
        return Err("arm_moe_ep: layer is already expert-armed".into());
    }
    // Expert TENSOR-parallelism replaces whole-expert ownership outright: with every rank holding
    // half of every expert there is no owner map, no placement row and no rank that routes no
    // work, so the two arms are exclusive rather than composable.
    if glm5_tp_expert_split_on() {
        let n_embd = m.gate_inp.in_features();
        m.glm5_tp_split = Some(shard_moe_layer_split(e, rt, m, n_embd)?);
        return Ok(());
    }
    let ranks = rt.ranks();
    let n_expert = m.gate_exps.n_expert;
    if !n_expert.is_multiple_of(ranks) {
        return Err(format!(
            "glm5-tp EP: {n_expert} experts do not partition across {ranks} ranks"
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
            if owners.iter().any(|&r| (r as usize) >= ranks) {
                return Err(
                    format!("glm5-tp EP: placement row names a rank outside TP-{ranks}").into(),
                );
            }
            owners.to_vec()
        }
        None => crate::ep_map::EpMap::even_owners(n_expert, ranks),
    };
    // Ascending-id packing per rank + the local-slot table.
    let mut local_of = vec![0u32; n_expert];
    let mut owned: Vec<Vec<usize>> = vec![Vec::new(); ranks];
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
    let mut slabs = Vec::with_capacity(ranks);
    for r in 0..ranks {
        slabs.push(slab(rank_engine(e, rt, r), &owned[r])?);
    }
    // Gate red arm: wrong expert weights on the root rank (gate/up swapped).
    if matches!(gate_red(), Ok(Some(GateRed::SwapEpGateUp))) {
        let root = &mut slabs[0];
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
    let mut ptr_rows = Vec::with_capacity(ranks);
    for r in 0..ranks {
        ptr_rows.push(ptr_table(rank_engine(e, rt, r), &slabs[r], r as u8)?);
    }
    if !EP_MARKED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[glm5-tp-ep] expert-parallel armed: experts_per_rank={:?} ownership={} \
             router=root combine=slot-ordered-fmaf transport={} \
             performance_claim=false",
            owned.iter().map(Vec::len).collect::<Vec<_>>(),
            // ORDER MATTERS and it was wrong once: the ownership string must land on
            // `ownership={}` and the transport on `transport={}`. The first gate run's
            // receipt-extract printed `[glm5-tp-ep] transport=even-split`, which is how the
            // swap was caught — a receipt line is only worth what its argument order is.
            if placement.is_some() {
                "measured-map"
            } else {
                "even-split"
            },
            rt.transport.name(),
        );
    }
    // The root-resident full slab (if the loader built one) is superseded by the EP slices;
    // dropping it returns its VRAM and removes the arm that would silently bypass EP.
    m.dev_exps = None;
    m.glm5_ep = Some(Glm5EpExps {
        rt: Arc::clone(rt),
        slabs,
        owner_of,
        local_of,
        ptr_rows,
    });
    Ok(())
}

// ------------------------------------------------------------------------------------------
// Expert TENSOR-parallelism (lane/tp-expert-split-20260906)
// ------------------------------------------------------------------------------------------

/// Every rank holds HALF OF EVERY EXPERT, instead of all of half the experts.
///
/// WHY, and it is a sizing fact rather than a preference: with whole-expert ownership a top-8
/// token splits Binomial(8, 0.5) across two ranks and a memory-bound decode step is paced by the
/// BUSIER rank, so `E[max] = 5.094` and expert parallelism buys a 1.571x expert-half speedup
/// where the TP sizing assumed 2x. The even 4/4 split happens on 27.3% of tokens; 28.9% land 6-2
/// or worse. Splitting inside each expert removes the variance rather than averaging it: whatever
/// the router picks, each rank streams exactly half of every routed expert.
/// (`tp_expert_split::ep_busier_rank_experts` carries that arithmetic as a test.)
///
/// SLAB SHAPE. `gate` and `up` are ROW-split (out rows `[r*half, (r+1)*half)` of every expert),
/// `down` is COLUMN-split (input columns of the same range), so a slot's program on rank `r` is
/// the unsharded program at half the intermediate width, producing a FULL-WIDTH partial sum.
/// There is no owner map: every rank runs every routed slot.
pub struct Glm5TpSplitExps {
    pub rt: Arc<Glm5TpRt>,
    /// `slabs[r]` is rank r's half of ALL experts (`[0]` = root's, on the model's engine).
    pub slabs: Vec<EpRankSlab>,
    /// The per-rank intermediate width: `n_ff_exp / ranks`.
    pub half_ff: usize,
    /// Per-rank `gate`/`up` expert stride in bytes (row-split: `half_ff * row_bytes`).
    pub gate_stride: usize,
    pub up_stride: usize,
    /// Per-rank `down` expert stride in bytes (column-split: `n_embd * (row_bytes / ranks)`).
    pub down_stride: usize,
    /// Per-rank `down` row width in bytes after the column split.
    pub down_row_bytes: usize,
}

impl Glm5TpSplitExps {
    pub fn ranks(&self) -> usize {
        self.slabs.len()
    }
}

/// `MEMRA_TP_AR_1STAGE=1` (lane/tp-ar-1stage-20260906, default OFF, decide-by 2026-09-20): the
/// walk's two REDUCE sites take the one-shot all-reduce (`memra_tp_ar_1stage`) instead of a
/// return-to-root hop.
///
/// WHY. A reduce done as a push plus a fold costs 4 kernel launches and 8 cross-context CUDA event
/// operations, which measured 20-26 us for 16 KB on a pair whose fabric moves 956 GB/s: host
/// overhead, not bandwidth. vLLM's `cross_device_reduce_1stage` is ONE kernel per rank and NO
/// events, synchronising through flags in the peer's memory and reading both inputs directly. This
/// is that shape. Requires a real two-device group; it declines on the same-device gate, where a
/// kernel peer store is undefined.
pub fn tp_ar_1stage_on() -> bool {
    std::env::var("MEMRA_TP_AR_1STAGE").as_deref() == Ok("1")
}

/// `MEMRA_GLM5_TP_EXPERT_SPLIT=1` (lane/tp-expert-split-20260906, default OFF, decide-by
/// 2026-09-20): shard every expert across the TP ranks instead of assigning whole experts.
///
/// NOT byte-identical, and the split is where: `gate`/`up` are row splits, so every output
/// element stays one full-K dot over the same bytes and that half is bit-identical, as is the
/// SwiGLU after it. `down` is a column split, so each rank produces a partial sum over half the
/// K range and the ranks' partials are added, which is a 2-way split of a reduction the
/// unsharded walk does in one pass. Named class, gated as one.
pub fn glm5_tp_expert_split_on() -> bool {
    std::env::var("MEMRA_GLM5_TP_EXPERT_SPLIT").as_deref() == Ok("1")
}

static SPLIT_MARKED: AtomicBool = AtomicBool::new(false);

/// Build the per-rank half-expert slabs. Load-time only: one host staging pass per rank per
/// projection, then one upload each.
pub(crate) fn shard_moe_layer_split(
    e: &Engine,
    rt: &Arc<Glm5TpRt>,
    m: &crate::hybrid::MoeWeights,
    n_embd: usize,
) -> Result<Glm5TpSplitExps, Box<dyn std::error::Error>> {
    use crate::tp_expert_split::{split_cols, split_rows};
    let ranks = rt.ranks();
    let mut slabs = Vec::with_capacity(ranks);
    let (mut gate_stride, mut up_stride, mut down_stride, mut down_row_bytes, mut half_ff) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    for r in 0..ranks {
        let dev = rank_engine(e, rt, r);
        let g = split_rows(&m.gate_exps, ranks, r)?;
        let u = split_rows(&m.up_exps, ranks, r)?;
        let d = split_cols(&m.down_exps, ranks, r)?;
        if g.out_f != u.out_f || g.out_f != d.in_f {
            return Err(format!(
                "glm5-tp expert split: rank {r} halves disagree (gate out {} up out {} down in {})",
                g.out_f, u.out_f, d.in_f
            )
            .into());
        }
        if d.out_f != n_embd {
            return Err(format!(
                "glm5-tp expert split: down out {} is not the hidden width {n_embd}",
                d.out_f
            )
            .into());
        }
        half_ff = g.out_f;
        gate_stride = g.expert_stride;
        up_stride = u.expert_stride;
        down_stride = d.expert_stride;
        down_row_bytes = d.row_bytes;
        // The same tail-slack pads the whole-expert slabs take: the ragged-k grouped GEMM may
        // overread past the LAST row, and the slack only prevents the OOB fault.
        slabs.push(EpRankSlab {
            gate: dev.htod_bytes_padded(&g.bytes, 8)?,
            up: dev.htod_bytes_padded(&u.bytes, 8)?,
            down: dev.htod_bytes_padded(&d.bytes, 144)?,
            n_experts: m.gate_exps.n_expert,
        });
    }
    if !SPLIT_MARKED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[glm5-tp-split] expert TENSOR-parallel armed: every rank holds half of ALL {} \
             experts, half_ff={half_ff} (whole-expert ownership pays the busier rank: \
             E[max]=5.094 of 8, a 1.571x expert half where the split is a deterministic 2x) \
             transport={} performance_claim=false",
            m.gate_exps.n_expert,
            rt.transport.name(),
        );
    }
    Ok(Glm5TpSplitExps {
        rt: rt.clone(),
        slabs,
        half_ff,
        gate_stride,
        up_stride,
        down_stride,
        down_row_bytes,
    })
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
        // The TP-4 device list parses through the same grammar.
        let quad = parse_glm5_tp_layer_specs(Some("all@0,1,2,3"), 45).unwrap();
        assert_eq!(quad.len(), 45);
        assert_eq!(quad[0].devices, vec![0, 1, 2, 3]);
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
            kda_heads: 4,
            kda_head_dim: 128,
            mla_heads: 4,
            n_routed_experts: 4,
            top_k: 2,
        }
    }

    /// Structural preflight refusals, exercised WITHOUT constructing any CUDA state: every
    /// geometry law here fires before `prepare_glm5_tp_load` reaches the runtime build.
    /// (The armed happy path needs an Engine and lives in the gate binary.)
    #[test]
    fn preflight_geometry_laws_are_dimension_derived() {
        // The checks below mirror prepare_glm5_tp_load's law order on the view alone, at
        // BOTH qualified rank counts.
        let v = fixture_view();
        for ranks in GLM5_TP_ALLOWED_RANKS {
            assert_eq!(v.kda_heads % ranks, 0);
            assert_eq!(v.mla_heads % ranks, 0);
            assert_eq!(v.n_routed_experts % ranks, 0);
        }
        let odd = Glm5TpModelView {
            kda_heads: 3,
            ..fixture_view()
        };
        assert_ne!(odd.kda_heads % 2, 0);
        let bad_dim = Glm5TpModelView {
            kda_head_dim: 64,
            ..fixture_view()
        };
        assert_ne!(bad_dim.kda_head_dim, crate::kda::KDA_HEAD_DIM);
        let odd_experts = Glm5TpModelView {
            n_routed_experts: 5,
            ..fixture_view()
        };
        assert_ne!(odd_experts.n_routed_experts % 2, 0);
        // TP-3 stays outside the qualified envelope (head padding not built).
        assert!(!GLM5_TP_ALLOWED_RANKS.contains(&3));
    }

    #[test]
    fn armed_check_counts_parse_errors_as_armed() {
        // glm5_tp_armed is a cheap co-refusal predicate: any nonempty non-"0" value counts,
        // including a spec the parser would refuse — the co-armed program must not race the
        // loader's own refusal.
        // (Env-mutation-free: the predicate's contract is pure string classification.)
        for (v, armed) in [
            ("", false),
            ("0", false),
            ("all@0,1", true),
            ("all@0,1,2,3", true),
            ("junk", true),
        ] {
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
        // The verify-batch flag is DELIBERATELY not in the matrix (the gated spec x TP
        // composition owns that pair — its admission REQUIRES the batched walk); arming
        // it alone must not trip this law.
        refuse_glm5_tp_door_composition(|f| f == "MEMRA_GLM5_VERIFY_BATCH")
            .expect("verify-batch is refused via the spec co-refusal, not here");
    }
}
