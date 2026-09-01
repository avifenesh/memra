//! M2 pipeline-parallel N-stage runtime (generalizes the M1 2-stage seam).
//!
//! Door: `MEMRA_PP_STAGES=N` (default OFF — unset/0/1 = no behavior change anywhere).
//! Stage map: N stages over the trunk layers with N-1 cuts. `MEMRA_PP_SPLITS=c1,..,cN-1`
//! sets the cuts explicitly (strictly increasing, in (0, n_layers)); `MEMRA_PP_SPLIT=<i>`
//! is the N=2 back-compat spelling; default = even split (cut s = s*n_layers/N).
//! Placement: `MEMRA_PP_DEVICES=d0,..,dN-1` maps stage s to device ds (default: all on
//! the primary engine's device).
//!
//! M1 history (increments 1-2, merged + hardened on the 8x box 2026-08-02): seam + gate
//! single-device; then real transport — per-stage streams/events, device placement,
//! peer-copy boundary (M0: cudaMemcpyPeerAsync beats NCCL 2.8x at PP activation sizes),
//! per-context PDL module caches, default-mempool peer grants. All five r3 gates PASS
//! bit-identical (receipts ~/receipts/m1-pp2/ on <bench-instance>).
//!
//! M2 increment 1 (this file): N-STAGE GENERALIZATION — `Pp2Rt` becomes `PpNRt`:
//!   - `stages`: Vec of per-stage execution homes (device, context, stream, remote Engine);
//!   - `boundaries`: N-1 boundary runtimes, each with TWO persistent double-buffered slots
//!     (ev_tx/ev_rx per slot) and its own overlap step counter; transport is selected PER
//!     BOUNDARY (dtod same-device / cudaMemcpyPeerAsync cross-device by default; opt-in
//!     `MEMRA_PP_HOST_BOUNCE=1` uses pinned D2H + H2D instead);
//!   - the default peer transport grants peer + default-mempool access between EVERY distinct
//!     pair of devices in use. Host bounce skips serving-time grants; its boot diagnostics
//!     transiently enable peer + pool access, then revoke the pool grants and disable peer access
//!     before proceeding. Sharded weights plus stage-local auxiliary buffers ensure that no peer
//!     read can bypass the bounced boundary.
//!
//! M2 increment 2 (weight sharding): the loader uploads each stage's layer range THROUGH
//! that stage's engine (`layer_engine`), so weights land on the device that runs them —
//! the bring-up peer-read placement dies. `output_norm` + lm head load through the LAST
//! stage's engine; the embed table stays host-side with stage 0. Split-plane/f16 decode
//! mirrors are built per layer through the owning stage's engine too (the rp4 mirrors ARE
//! the decode weights on the q8 path — leaving them on dev0 would fake the kill).
//! Rollback seam: `MEMRA_PP_SHARD=0` = M1 bring-up placement (all weights on primary,
//! remote stages peer-read).
//!
//! M2 increment 3 (deferred readback — the pipelining seed): `PendingLogits` — the eager
//! decode arm can END a step without the logits D2H (`decode_step_h_ppn_deferred`): the
//! logits stay device-resident with a completion event; `wait()` drains them through a
//! DEDICATED readback stream (waits the event, copies, syncs) so tokens t+1.. keep
//! enqueuing on the stage streams while token t drains. Per-token math is fully
//! event-ordered (same slots, same ev_tx/ev_rx chain) — scheduling changes, math does
//! not; the pipelined replay arm of `ppn-gate` proves bit-identity per step.
//!
//! Ownership across a boundary (unchanged from M1):
//!   - hidden state [n_embd] f32 is the ONLY tensor that crosses;
//!   - KV/linear-attn cache entries are per-layer: stage s exclusively owns cache state
//!     for its layer range (and, under MEMRA_PP_DEVICES, allocates it on its device);
//!   - position/rope state is the scalar `cache.pos` snapshot taken once per step; every stage
//!     uploads its own position buffer on its own stream (no cross-device position pointer);
//!   - the embed table lives with stage 0, output_norm + lm head with the last stage.
//!
//! THE MULTI-STREAM LAW (why this is safe with cudarc event tracking disabled): all
//! cross-stage bytes flow through the persistent boundary slots, ordered by ev_tx/ev_rx;
//! per-stage scratch is allocated AND freed on that stage's stream (stream-ordered); the
//! async mem pool runs with opportunistic reuse OFF + internal dependencies ON
//! (memra-runtime), so a block freed on stream A and reused on stream B carries a
//! driver-inserted dependency. Weights are load-time state no stage stream can precede,
//! and the step's terminal logits readback (sync D2H, or PendingLogits' event-ordered
//! readback stream) drains the last stage, whose TX-wait chain transitively drains all.
//!
//! Scope: plain eager decode only (generic arm N-stage; gemma4 arm 2-stage). NOT wired:
//! batch/dc/graph/spec loops and the gemma4-E4B eager arm.
//!
//! CORRECTION (pp2-hardening 2026-08-06): this header used to add "(`warn_unwired_once`
//! fires)" to that list, which was wrong. `warn_unwired_once` has exactly two call sites
//! and BOTH are gemma4-specific (decode.rs, hybrid_forward.rs) — the batch/dc/graph/spec
//! loops never warned. Worse, the batched loop did not merely run unsplit: it walked the
//! whole trunk on the primary stream and, under a sharded cross-device placement,
//! peer-read every remote stage's weights each step — 28x slower at B=1 with all three
//! `decode-batch-gate` gates PASSING (peer reads are byte-exact, so only perf broke).
//! `decode_step_batch` now FAILS CLOSED in that regime via `pp_sharded_cross_device()`
//! (`MEMRA_PP_ALLOW_UNSPLIT_BATCH=1` = measurement override). "Unwired" for dc/graph/spec
//! still means "runs unsplit, silently" — audit each before trusting it on a pair.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream};

use crate::Engine;

/// Returns the stage fence iff the ppN door is open: `MEMRA_PP_STAGES=N` (N >= 2) with a
/// valid cut list. The fence has N+1 entries: `[0, c1, .., cN-1, n_layers]`; stage s runs
/// layers `[fence[s], fence[s+1])`. Reads the environment on every call (gates toggle the
/// door in-process); the cost is a few getenv per decode step, eager-loop noise.
pub fn pp_cuts(n_layers: usize) -> Option<Vec<usize>> {
    let n_st: usize = match std::env::var("MEMRA_PP_STAGES") {
        Ok(v) if v.is_empty() || v == "0" || v == "1" => return None,
        Ok(v) => match v.parse::<usize>() {
            Ok(n) => n,
            Err(_) => {
                warn_bad_once(&format!("MEMRA_PP_STAGES={v} unparseable; door stays OFF"));
                return None;
            }
        },
        Err(_) => return None,
    };
    if n_st < 2 || n_st > n_layers {
        warn_bad_once(&format!(
            "MEMRA_PP_STAGES={n_st} outside [2, n_layers={n_layers}]; door stays OFF"
        ));
        return None;
    }
    let mut fence = Vec::with_capacity(n_st + 1);
    fence.push(0usize);
    if let Ok(s) = std::env::var("MEMRA_PP_SPLITS") {
        let parts: Result<Vec<usize>, _> =
            s.split(',').map(|p| p.trim().parse::<usize>()).collect();
        match parts {
            Ok(cuts) if cuts.len() == n_st - 1 => fence.extend(cuts),
            _ => {
                warn_bad_once(&format!(
                    "MEMRA_PP_SPLITS={s} invalid (want {} comma-separated cuts); door stays OFF",
                    n_st - 1
                ));
                return None;
            }
        }
    } else if let Ok(v) = std::env::var("MEMRA_PP_SPLIT") {
        // N=2 back-compat spelling. With N>2 a single split is ambiguous — fail the door
        // loudly rather than guess (a silent even-split would fake a gate config).
        if n_st != 2 {
            warn_bad_once(&format!(
                "MEMRA_PP_SPLIT={v} set with MEMRA_PP_STAGES={n_st}; use MEMRA_PP_SPLITS \
                 for N>2 — door stays OFF"
            ));
            return None;
        }
        match v.parse::<usize>() {
            Ok(c) => fence.push(c),
            Err(_) => {
                warn_bad_once(&format!("MEMRA_PP_SPLIT={v} unparseable; door stays OFF"));
                return None;
            }
        }
    } else {
        for s in 1..n_st {
            fence.push(s * n_layers / n_st);
        }
    }
    fence.push(n_layers);
    for w in fence.windows(2) {
        if w[0] >= w[1] {
            warn_bad_once(&format!(
                "pp stage fence {fence:?} not strictly increasing over [0, {n_layers}]; \
                 door stays OFF"
            ));
            return None;
        }
    }
    Some(fence)
}

/// N=2 back-compat view of the door (the gemma4 arm and `pp2-gate` are 2-stage): `Some(cut)`
/// iff the door is open with EXACTLY two stages.
pub fn pp2_split(n_layers: usize) -> Option<usize> {
    pp_cuts(n_layers).filter(|f| f.len() == 3).map(|f| f[1])
}

/// The stage that owns layer `il` under `fence` (see `pp_cuts`).
pub fn stage_of(fence: &[usize], il: usize) -> usize {
    debug_assert!(fence.len() >= 2);
    match fence[1..fence.len() - 1].binary_search(&il) {
        // fence[1..][k] == il means il is the FIRST layer of stage k+1
        Ok(k) => k + 1,
        Err(k) => k,
    }
}

/// MEMRA_PP_STREAMS=0: rollback to the increment-1 same-stream seam (boundary = two plain
/// dtod copies on the ambient compute stream, no per-stage streams/events/devices).
pub fn pp2_streams_off() -> bool {
    matches!(std::env::var("MEMRA_PP_STREAMS").as_deref(), Ok("0"))
}

/// True iff the ppN door would put TWO OR MORE stage streams on ONE device (devices
/// unset = all stages on the primary; or an explicit placement with a repeated device).
/// The deferred-readback (pipelined) arm is REFUSED in this regime: the 2026-08-02 x20
/// soak record — singledev pipelined 13/20 PASS default, 7 failures each diverging at a
/// different step (timing-race signature); MEMRA_PDL=0 went 20/20 on one soak but a
/// second same-config soak on the auto-gated build failed 2/20 (n2) and battery-4 failed
/// n4 — so PDL narrows the window without closing it, and the true root cause (same
/// Engine kernels concurrent on two streams of one device) is NOT fixed by any flag yet.
/// Cross-device pipelined (one stage stream per device) is 23/23 clean post-fix. Refuse
/// loudly rather than return silently-wrong logits. Env-only read (callable pre-runtime).
pub fn pp_multi_stream_same_device() -> bool {
    let stages_open = std::env::var("MEMRA_PP_STAGES")
        .map(|v| v.parse::<usize>().map(|n| n >= 2).unwrap_or(false))
        .unwrap_or(false);
    let devices = std::env::var("MEMRA_PP_DEVICES")
        .ok()
        .filter(|v| !v.is_empty());
    if (!stages_open && devices.is_none()) || pp2_streams_off() {
        return false;
    }
    match devices {
        None => true, // door open, no placement: every stage stream lands on the primary
        Some(s) => {
            let mut v: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
            let n = v.len();
            v.sort_unstable();
            v.dedup();
            v.len() < n // repeated device = shared-device streams
        }
    }
}

/// True iff the ppN door is open AND the placement spans 2+ DISTINCT devices AND the
/// per-stage sharded loader is on — i.e. some layers' weights live on a device other than
/// the primary. Any path that walks the WHOLE trunk on one stream in this regime reads
/// those weights over PCIe every step. Env-only read (callable pre-runtime).
///
/// Measured cost of doing that (pp2-hardening 2026-08-06, 2x RTX PRO 6000, PCIe Gen5 x16
/// P2P, decode-batch-bench q9, N=5 interleaved, `research/pp2-hardening-20260806`):
/// **B=1 7.4 vs 208.9 tok/s (28x), B=4 29.8 vs 491.3 (16.5x), B=8 47.4 vs 657.0 (13.9x)**.
/// The same sweep with `MEMRA_PP_SHARD=0` (weights all home) returns 178.5/491.1/656.6 —
/// identical to the single-device door-open arm — so the entire cliff is the peer read,
/// not the door and not the placement plumbing. Exactness is NOT the issue: peer reads
/// return identical bytes and every `decode-batch-gate` gate PASSED on this config, which
/// is precisely why it needs a refusal rather than a gate.
pub fn pp_sharded_cross_device() -> bool {
    let stages_open = std::env::var("MEMRA_PP_STAGES")
        .map(|v| v.parse::<usize>().map(|n| n >= 2).unwrap_or(false))
        .unwrap_or(false);
    // MEMRA_PP_STREAMS=0 (2026-08-06, pp2-batch): the same-stream rollback seam ALSO turns
    // the sharded loader off — `layer_engine` returns the primary engine whenever
    // `pp2_streams_off()`, and `new_cache` skips `Cache::new_ppn` on the same condition. So
    // in that regime every weight and every cache is home on the primary and an unsplit walk
    // peer-reads NOTHING. Without this term the guard refused that config too: a spurious
    // refusal of a placement that is sound and full-speed. Found wiring the batched pp arm.
    if !stages_open || pp_shard_off() || pp2_streams_off() {
        return false;
    }
    match pp2_devices_env() {
        None => false, // no placement: every stage is the primary device, nothing remote
        Some(s) => {
            let mut v: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
            v.sort_unstable();
            v.dedup();
            v.len() >= 2
        }
    }
}

/// The shared fail-closed guard for EVERY decode path that has no pp stage split.
/// Returns `Err` iff `pp_sharded_cross_device()` — i.e. the caller would walk the whole
/// trunk on one stream while some layers' weights live on another device, peer-reading
/// them every step. `path` names the refusing function so the operator knows which loop
/// they hit; `alt` names the working alternative for that loop.
///
/// One helper rather than four copies because the audit found FOUR paths with the same
/// hole (`decode_step_batch`, `decode_step_dc`, the graph capture that wraps dc, and
/// `decode_step_t*` verify), and a per-path copy is how one gets missed on the next
/// addition. Override: `MEMRA_PP_ALLOW_UNSPLIT_BATCH=1` (one door for all of them —
/// they are the same measurement question).
pub fn refuse_unsplit_if_remote(path: &str, alt: &str) -> Result<(), Box<dyn std::error::Error>> {
    if pp_host_bounce_active() {
        return Err(format!(
            "{path}: refused with MEMRA_PP_HOST_BOUNCE=1 on sharded cross-device PP — \
             this unsplit path peer-reads remote weights, while host bounce covers only \
             explicit stage-boundary transfers. Use {alt}; the \
             MEMRA_PP_ALLOW_UNSPLIT_BATCH override is unavailable on a broken-peer host."
        )
        .into());
    }
    if pp_sharded_cross_device()
        && std::env::var("MEMRA_PP_ALLOW_UNSPLIT_BATCH").as_deref() != Ok("1")
    {
        return Err(format!(
            "{path}: refused with the ppN door open across 2+ devices — this path has no pp \
             stage split, so it would walk ALL layers on one stream and peer-read every \
             remote stage's weights each step (measured 28x slower at B=1, 13.9x at B=8 on \
             a PRO 6000 pair over PCIe Gen5 x16 P2P; research/pp2-hardening-20260806). \
             Exactness is unaffected — peer reads return identical bytes and the exactness \
             gates PASS on this config — which is exactly why it must refuse instead of \
             being caught by a gate. Fixes, in order: {alt}; or MEMRA_PP_SHARD=0 (all \
             weights home on the primary — full speed, forfeits the capacity PP-2 exists \
             for); or close the pp door. MEMRA_PP_ALLOW_UNSPLIT_BATCH=1 overrides for \
             measurement."
        )
        .into());
    }
    Ok(())
}

/// MEMRA_BATCH_PP=0: rollback/A-B seam for the BATCHED stage split (pp2-batch 2026-08-06).
/// Default ON — with the ppN door open the batched decode step takes its own stage split
/// (`decode_step_batch_ppn`) exactly as the eager step does. Setting 0 sends the batched
/// path back through the unsplit body, which under a sharded cross-device placement is
/// then caught by `refuse_unsplit_if_remote` (the 28x peer-read regime) rather than run
/// silently. Exists so the bit-identity gate can A/B split vs unsplit IN ONE PROCESS
/// against the same loaded weights — read per step, never memoized, for that reason.
pub fn batch_pp_on() -> bool {
    std::env::var("MEMRA_BATCH_PP").as_deref() != Ok("0")
}

/// MEMRA_DUAL_PP three-state mode for the dual-active PP-2 batched decode path.
/// Default ON (owner flip 2026-08-11) after the box1 PRO-pair re-gate: correctness
/// bit-identity B=1..5, servestress no-thrash, 10-boot soak 929/929 golden matches with
/// 0 slot collisions across 9123 pairs (research/dualpp2-20260811/RESULTS-regate.md), plus
/// the dualpp1 c>=8 interleaved perf floor (+20.753% minimum,
/// research/dualpp1-20260811/RESULTS.md).
///
/// The three states carry different failure semantics on purpose:
/// - `Off` (`MEMRA_DUAL_PP=0`): the serial rollback seam. Overlap also follows OFF unless
///   `MEMRA_PP_OVERLAP` is set explicitly, so one flag restores the exact pre-flip naked path.
/// - `Forced` (`MEMRA_DUAL_PP=1`): the pre-flip explicit request. A placement that cannot
///   run dual (single-slot boundary, host bounce, non-PP-2 fence) REFUSES with the binding
///   quoted reason before any token or cache advance — the gate negative cells pin this.
/// - `Auto` (unset): the flipped default. Dual runs where the re-gate validated it
///   (PP-2 fence, double-slot, peer transport, B>=2) and silently degrades to the serial
///   PP-N walker everywhere else — naked PP-3 serving and the MEMRA_PP_HOST_BOUNCE=1
///   broken-peer escape hatch must keep decoding, not refuse.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DualPpMode {
    Off,
    Forced,
    Auto,
}

/// Pure resolution for MEMRA_DUAL_PP, split from the env read so the flip regression tests
/// cannot race parallel test threads on process env.
pub fn dual_pp_mode_resolve(v: Option<&str>) -> DualPpMode {
    match v {
        Some("0") => DualPpMode::Off,
        Some("1") => DualPpMode::Forced,
        _ => DualPpMode::Auto,
    }
}

pub fn dual_pp_mode() -> DualPpMode {
    dual_pp_mode_resolve(std::env::var("MEMRA_DUAL_PP").ok().as_deref())
}

/// True when the dual-active door is open (Forced or Auto). Read per step so the
/// model-level gate can replay serial and waved arms against one loaded checkpoint.
pub fn dual_pp_on() -> bool {
    dual_pp_mode() != DualPpMode::Off
}

/// Engine-entry routing for the dual-active path, kept pure for the flip regression
/// tests. `Forced` routes every B>=2 PP-2 call into `decode_step_batch_dual` even when
/// the placement cannot run it, so the binding refusals stay reachable and loud.
/// `Auto` routes only the exact re-gated regime and leaves everything else on the serial
/// PP-N walker. `dual_pp_eligibility` remains behind this as defense in depth.
pub fn dual_pp_route(
    mode: DualPpMode,
    batch: usize,
    stages: usize,
    double_slot: bool,
    host_bounce: bool,
) -> bool {
    if batch < 2 {
        return false;
    }
    match mode {
        DualPpMode::Off => false,
        DualPpMode::Forced => true,
        DualPpMode::Auto => stages == 2 && double_slot && !host_bounce,
    }
}

/// Binding-amendment refusal text. The negative gate quotes this exact line and requires the
/// decode call to return before producing a token or advancing a cache.
pub const DUAL_PP_SINGLE_SLOT_REFUSAL: &str = "decode_step_batch_dual: refused: PP boundary is single-slot; set MEMRA_PP_OVERLAP=1 so both alternating boundary slots are prepared before dual-active decode";
pub const DUAL_PP_HOST_BOUNCE_REFUSAL: &str = "decode_step_batch_dual: refused: MEMRA_PP_HOST_BOUNCE=1 is unvalidated for dual-active decode; disable MEMRA_DUAL_PP or use peer transport";

/// Pure schedule policy shared by the runtime and kernel-check manifest cells. A single row
/// has no second wave and must stay on the serial PP-N walker.
pub fn dual_pp_wave_mid(batch: usize) -> Option<usize> {
    (batch >= 2).then_some((batch + 1) / 2)
}

/// Fail-closed eligibility check kept pure so the negative manifest cell cannot accidentally
/// initialize CUDA state. Slot preparation itself remains `PpNRt::prepare_overlap_slots`.
pub fn dual_pp_eligibility(
    stages: usize,
    double_slot: bool,
    host_bounce: bool,
) -> Result<(), &'static str> {
    if stages != 2 {
        return Err(
            "decode_step_batch_dual: refused: dual-active decode requires exactly two PP stages",
        );
    }
    if !double_slot {
        return Err(DUAL_PP_SINGLE_SLOT_REFUSAL);
    }
    if host_bounce {
        return Err(DUAL_PP_HOST_BOUNCE_REFUSAL);
    }
    Ok(())
}

/// Liveness is counted only while the two host-driven decode layer walkers are both active.
/// Enqueue order is not proof for Step: its router readback synchronizes the issuing thread.
static DUAL_PP_OVERLAPS: AtomicUsize = AtomicUsize::new(0);
static DUAL_PP_ACTIVE_STAGES: AtomicUsize = AtomicUsize::new(0);
static DUAL_PP_STAGE_NS: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static DUAL_PP_STAGE_SAMPLES: [AtomicUsize; 4] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];
static DUAL_PP_TIMING_DROPPED: AtomicUsize = AtomicUsize::new(0);
static DUAL_PP_SLOT_PAIRS: AtomicUsize = AtomicUsize::new(0);
static DUAL_PP_SLOT_USES: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];
static DUAL_PP_SLOT_COLLISIONS: AtomicUsize = AtomicUsize::new(0);

pub const DUAL_PP_STAGE_NAMES: [&str; 4] = [
    "wave_a_stage0",
    "wave_a_stage1",
    "wave_b_stage0",
    "wave_b_stage1",
];

pub fn dual_pp_overlaps() -> usize {
    DUAL_PP_OVERLAPS.load(Ordering::Relaxed)
}

/// Record the two boundary slots selected for one dual-active wave pair. A same-slot pair is
/// rejected by the caller before wave B can consume a residual; the collision counter makes that
/// fail-closed path observable to the detached soak instead of relying only on log scanning.
pub(crate) fn record_dual_pp_slot_pair(slot_a: usize, slot_b: usize) -> bool {
    debug_assert!(slot_a < DUAL_PP_SLOT_USES.len());
    debug_assert!(slot_b < DUAL_PP_SLOT_USES.len());
    if slot_a == slot_b {
        DUAL_PP_SLOT_COLLISIONS.fetch_add(1, Ordering::Relaxed);
        return false;
    }
    DUAL_PP_SLOT_USES[slot_a].fetch_add(1, Ordering::Relaxed);
    DUAL_PP_SLOT_USES[slot_b].fetch_add(1, Ordering::Relaxed);
    DUAL_PP_SLOT_PAIRS.fetch_add(1, Ordering::Relaxed);
    true
}

/// `(completed wave pairs, [slot 0 uses, slot 1 uses], rejected same-slot pairs)`.
pub fn dual_pp_slot_snapshot() -> (usize, [usize; 2], usize) {
    (
        DUAL_PP_SLOT_PAIRS.load(Ordering::Relaxed),
        std::array::from_fn(|i| DUAL_PP_SLOT_USES[i].load(Ordering::Relaxed)),
        DUAL_PP_SLOT_COLLISIONS.load(Ordering::Relaxed),
    )
}

/// CUDA-event timing is a diagnostic-only process door. The scored N=5 block runs without
/// it; the companion box1 diagnostic process enables it and exports cumulative per-wave
/// stage spans through `/metrics`.
pub fn dual_pp_timing_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_DUAL_PP_TIMING").as_deref() == Ok("1"))
}

pub(crate) fn record_dual_pp_stage_ms(stage: usize, ms: f32) {
    assert!(
        stage < DUAL_PP_STAGE_NS.len(),
        "dual PP timing stage out of range"
    );
    let ns = (f64::from(ms) * 1_000_000.0).round() as u64;
    DUAL_PP_STAGE_NS[stage].fetch_add(ns, Ordering::Relaxed);
    DUAL_PP_STAGE_SAMPLES[stage].fetch_add(1, Ordering::Relaxed);
}

/// Timing is diagnostic only: a CUDA event that is not ready (or otherwise fails) must not
/// change decode control flow. Count and warn once, then leave the scored-path result intact.
pub(crate) fn record_dual_pp_timing_drop(context: &str, err: &dyn std::fmt::Display) {
    let previous = DUAL_PP_TIMING_DROPPED.fetch_add(1, Ordering::Relaxed);
    if previous == 0 {
        eprintln!(
            "[dual-pp] WARN: skipped diagnostic timing sample at {context}: {err}; decode continues"
        );
    }
}

pub(crate) fn record_dual_pp_stage_result<E: std::fmt::Display>(
    stage: usize,
    elapsed: Result<f32, E>,
) {
    match elapsed {
        Ok(ms) => record_dual_pp_stage_ms(stage, ms),
        Err(err) => record_dual_pp_timing_drop(DUAL_PP_STAGE_NAMES[stage], &err),
    }
}

pub fn dual_pp_timing_dropped() -> usize {
    DUAL_PP_TIMING_DROPPED.load(Ordering::Relaxed)
}

/// `(total_nanoseconds, samples)` for wave-A stage0/stage1 then wave-B stage0/stage1.
pub fn dual_pp_timing_snapshot() -> ([u64; 4], [usize; 4]) {
    (
        std::array::from_fn(|i| DUAL_PP_STAGE_NS[i].load(Ordering::Relaxed)),
        std::array::from_fn(|i| DUAL_PP_STAGE_SAMPLES[i].load(Ordering::Relaxed)),
    )
}

pub(crate) struct DualPpStageGuard;

pub(crate) fn enter_dual_pp_stage() -> DualPpStageGuard {
    let active = DUAL_PP_ACTIVE_STAGES.fetch_add(1, Ordering::AcqRel);
    if active > 0 {
        DUAL_PP_OVERLAPS.fetch_add(1, Ordering::Relaxed);
    }
    DualPpStageGuard
}

impl Drop for DualPpStageGuard {
    fn drop(&mut self) {
        let active = DUAL_PP_ACTIVE_STAGES.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(active > 0, "dual PP active-stage counter underflow");
    }
}

/// MEMRA_PRIME_PP=0: rollback/A-B seam for the PRIME (chunked prefill) stage split
/// (lane/pp-leverb 2026-08-08). Default ON — with the ppN door open the chunked prime takes
/// its own per-stage range walk exactly as the eager/batched/verify steps do. Setting 0 sends
/// prime back through the unsplit whole-trunk walk. NOTE: unlike batch/dc/graph/spec, prime
/// keeps NO `refuse_unsplit_if_remote` — its unsplit walk over a sharded placement is the
/// measured 22% amortized peer-read tax (research/pp-prefill-20260807 anatomy: m=4096
/// amortizes the weight reads), not the decode 28x cliff, and the unsplit walk IS the
/// split-vs-unsplit gate's reference arm (`prime-split-gate`), so it must stay callable.
/// Read per call, never memoized (the gate A/Bs both arms in one process).
pub fn prime_pp_on() -> bool {
    std::env::var("MEMRA_PRIME_PP").as_deref() != Ok("0")
}

/// MEMRA_PRIME_PIPE=0: rollback/A-B seam for the PP-2 PRIME CHUNK PIPELINE
/// (lane/cx-pipeline-prime 2026-08-08). Default ON when the prime stage split is live;
/// setting 0 keeps the serial per-chunk stage walk. Read per prime call so the exactness
/// gate can replay both schedules against one loaded model.
pub fn prime_pipe_on() -> bool {
    std::env::var("MEMRA_PRIME_PIPE").as_deref() != Ok("0")
}

/// SPLIT-LIVENESS COUNTER for the prime stage split: bumped ONCE per prime chunk that
/// actually executed the per-stage walk. The `prime-split-gate` requires this to ADVANCE
/// during its split arm — bit-identity of two identical UNSPLIT walks is vacuous, so a gate
/// that only compared bits would go green while the walker doesn't exist. With the counter,
/// the gate is RED until the walker lands (the tickinv35 pattern: the gate exists and fails
/// before the mechanism does). Relaxed ordering: single-threaded host issue, count-only.
pub static PRIME_SPLIT_CHUNKS: AtomicUsize = AtomicUsize::new(0);

/// Read the split-liveness counter (gate-side).
pub fn prime_split_chunks() -> usize {
    PRIME_SPLIT_CHUNKS.load(Ordering::Relaxed)
}

/// PIPELINE-LIVENESS COUNTER: bumped only when a second PP-2 prime stage enters its layer
/// walker while the other stage's walker is still active. Step's per-layer router readback
/// synchronizes the host, so enqueue order alone is not liveness: a single host thread can
/// call stage 0(N+1) before the stage-1 epilogue and still serialize all trunk computation.
pub static PRIME_PIPE_OVERLAPS: AtomicUsize = AtomicUsize::new(0);

/// Read the prime-pipeline overlap counter (gate-side).
pub fn prime_pipe_overlaps() -> usize {
    PRIME_PIPE_OVERLAPS.load(Ordering::Relaxed)
}

static PRIME_PIPE_ACTIVE_STAGES: AtomicUsize = AtomicUsize::new(0);

pub(crate) struct PrimePipeStageGuard;

/// Mark one host-driven stage walker active. With PP-2, a transition 1 -> 2 proves the
/// two device walkers overlap in wall time; exactly one transition is counted per pair.
pub(crate) fn enter_prime_pipe_stage() -> PrimePipeStageGuard {
    let active = PRIME_PIPE_ACTIVE_STAGES.fetch_add(1, Ordering::AcqRel);
    if active > 0 {
        PRIME_PIPE_OVERLAPS.fetch_add(1, Ordering::Relaxed);
    }
    PrimePipeStageGuard
}

impl Drop for PrimePipeStageGuard {
    fn drop(&mut self) {
        let active = PRIME_PIPE_ACTIVE_STAGES.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(active > 0, "prime pipeline active-stage counter underflow");
    }
}

/// Step35 cross-request prime liveness counters (lane/cx-prime-batch, 2026-08-08).
/// The exactness gate requires BOTH to advance: a successful step35 batch alone is not
/// sufficient under PP-N if it walked the whole sharded trunk on one stream.
pub static STEP35_PRIME_BATCHES: AtomicUsize = AtomicUsize::new(0);
pub static STEP35_PRIME_BATCH_SPLITS: AtomicUsize = AtomicUsize::new(0);

pub fn step35_prime_batches() -> usize {
    STEP35_PRIME_BATCHES.load(Ordering::Relaxed)
}

pub fn step35_prime_batch_splits() -> usize {
    STEP35_PRIME_BATCH_SPLITS.load(Ordering::Relaxed)
}

/// MEMRA_SPEC_PP=0: rollback/A-B seam for the SPEC VERIFY stage split (pp2-spec 2026-08-06).
/// Default ON — with the ppN door open the verify forward (`decode_step_t_core_ppn`) takes its
/// own stage split exactly as the eager and batched steps do. Setting 0 sends verify back through
/// the unsplit trunk walk, which under a sharded cross-device placement is then caught by
/// `refuse_unsplit_if_remote` (the 28x peer-read regime) rather than running silently. Exists so
/// the bit-identity gate can A/B split vs unsplit IN ONE PROCESS against the same loaded weights
/// — read per verify call, never memoized, for that reason.
pub fn spec_pp_on() -> bool {
    std::env::var("MEMRA_SPEC_PP").as_deref() != Ok("0")
}

/// MEMRA_PP_OVERLAP: alternate the double-buffered boundary slots per step (the
/// pipelining seed). Scheduling structure only, never math. Read per step so gates can
/// A/B in-process.
///
/// Unset follows the dual-PP mode (owner flip 2026-08-11): `Auto` resolves ON — the naked
/// serve path is the box1 re-gate's dual arm (MEMRA_DUAL_PP=1 MEMRA_PP_OVERLAP=1,
/// 929/929 golden, 0/9123 slot collisions). `Off` resolves OFF so MEMRA_DUAL_PP=0 alone
/// restores the exact pre-flip serial naked path. `Forced` resolves OFF so the binding
/// single-slot refusal of the explicit pre-flip request stays reachable — the
/// decode-batch-gate negative cell pins and asserts precisely that combination.
pub fn pp2_overlap() -> bool {
    pp2_overlap_resolve(
        std::env::var("MEMRA_PP_OVERLAP").ok().as_deref(),
        dual_pp_mode(),
    )
}

/// Pure resolution for MEMRA_PP_OVERLAP, split from the env read for the flip
/// regression tests.
pub fn pp2_overlap_resolve(v: Option<&str>, mode: DualPpMode) -> bool {
    match v {
        Some("1") => true,
        Some(_) => false,
        None => mode == DualPpMode::Auto,
    }
}

/// Broken-peer escape hatch: stage-boundary activations travel through page-locked host
/// memory instead of `cudaMemcpyPeerAsync`. Default OFF; captured when `PpNRt` is built.
pub fn pp_host_bounce_on() -> bool {
    matches!(std::env::var("MEMRA_PP_HOST_BOUNCE").as_deref(), Ok("1"))
}

/// True when host bounce is the live transport for a sharded cross-device placement.
/// Callers use this to close paths that still peer-read non-boundary state.
pub fn pp_host_bounce_active() -> bool {
    (pp_host_bounce_on() || PEER_RUNTIME_HOST_BOUNCE.load(Ordering::Acquire))
        && pp_sharded_cross_device()
}

/// M2 increment 2 rollback seam: MEMRA_PP_SHARD=0 = the M1 bring-up placement (all
/// weights upload through the primary engine; remote stages peer-read). Default ON —
/// under MEMRA_PP_DEVICES each stage's layer range uploads through its own engine.
pub fn pp_shard_off() -> bool {
    matches!(std::env::var("MEMRA_PP_SHARD").as_deref(), Ok("0"))
}

/// Raw `MEMRA_PP_DEVICES` (parsed/validated at PpNRt build — a bad string must fail the
/// decode step loudly, never silently fall back to same-device and fake a gate PASS).
fn pp2_devices_env() -> Option<String> {
    std::env::var("MEMRA_PP_DEVICES")
        .ok()
        .filter(|v| !v.is_empty())
}

static WARNED_BAD: AtomicBool = AtomicBool::new(false);
fn warn_bad_once(msg: &str) {
    if !WARNED_BAD.swap(true, Ordering::Relaxed) {
        eprintln!("[pp] {msg}");
    }
}

static WARNED_UNWIRED: AtomicBool = AtomicBool::new(false);
/// One-time notice when the door is set but the executing path has no pp arm
/// (M2 wires the generic eager decode at any N and the gemma4 eager arm at N=2).
pub fn warn_unwired_once(path: &str) {
    let open = std::env::var("MEMRA_PP_STAGES")
        .map(|v| !v.is_empty() && v != "0" && v != "1")
        .unwrap_or(false);
    if open && !WARNED_UNWIRED.swap(true, Ordering::Relaxed) {
        eprintln!("[pp] MEMRA_PP_STAGES set but `{path}` has no pp arm at this N; running unsplit");
    }
}

// ======================================================================================
//  PpNRt: the M2 transport runtime (per-stage streams, per-boundary events + slots)
// ======================================================================================

/// One pipeline stage's execution home: device, context, launch stream, and (for a stage
/// remote to the primary engine's device) a dedicated Engine in that device's primary
/// context (CUmodules are per-context).
pub struct StageRt {
    pub dev: usize,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: Arc<cudarc::cublaslt::CudaBlasLT>,
    /// `Some` only when `dev` differs from the primary engine's device.
    engine: Option<Engine>,
}

/// One boundary slot: a persistent RX-side buffer + its TX/RX completion events.
/// PERSISTENT because the buffer is written by the TX stage's stream and read by the RX
/// stage's: a per-step alloc/free would enqueue the free on ONE stream while the other
/// might still be reading (the cross-stream free hazard) — a never-freed slot cannot race.
struct BoundarySlot {
    buf: Mutex<Option<CudaSlice<f32>>>,
    /// Recorded on the TX stage's stream after the TX copy; RX waits on it. Created in
    /// the TX stage's context (cuEventRecord requires event ctx == stream ctx).
    ev_tx: CudaEvent,
    /// Recorded on the RX stage's stream after the RX copy; the NEXT TX into this slot
    /// waits on it (write-after-read guard). Created in the RX stage's context. Waiting
    /// on a never-recorded event is a defined no-op, so step 0 needs no special case.
    ev_rx: CudaEvent,
}

/// Boundary b sits between stage b (TX) and stage b+1 (RX). Two slots, alternating per
/// step under MEMRA_PP_OVERLAP=1 (each boundary counts its own steps — a decode step
/// crosses every boundary exactly once, so the counters stay in lockstep).
struct BoundaryRt {
    slots: [BoundarySlot; 2],
    step: AtomicUsize,
    /// true iff stage b and stage b+1 live on different devices (peer transport).
    cross: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundaryTransport {
    Local,
    Peer,
    HostBounce,
}

#[derive(Clone, Copy)]
struct BoundaryPath {
    boundary: usize,
    src_stage: usize,
    dst_stage: usize,
    transport: BoundaryTransport,
}

fn boundary_transport(cross: bool, host_bounce: bool) -> BoundaryTransport {
    match (cross, host_bounce) {
        (false, _) => BoundaryTransport::Local,
        (true, false) => BoundaryTransport::Peer,
        (true, true) => BoundaryTransport::HostBounce,
    }
}

const PEER_PROBE_FIXED_BYTES: usize = 16 * 1024;
const PEER_PROBE_TOKEN_WIDTHS: [usize; 4] = [1, 8, 16, crate::cache::PRIME_CHUNK_MAX_TOKENS];

/// Native cross-device boundary copies between low-frequency runtime integrity probes.
/// Fixed rather than operator-tunable: this is a safety gate, not a performance experiment.
pub const PEER_RUNTIME_PROBE_INTERVAL_COPIES: u64 = 8 * 1024;
/// One complete runtime width rotation. The maximum-chunk rung runs once per cycle.
pub const PEER_RUNTIME_PROBE_CYCLE_COPIES: u64 =
    PEER_RUNTIME_PROBE_INTERVAL_COPIES * PEER_PROBE_TOKEN_WIDTHS.len() as u64;
/// Consecutive runnable probe intervals that may be blocked by live speculative UVA state before
/// integrity coverage becomes explicitly degraded. Four intervals are one full width rotation.
pub const PEER_RUNTIME_PROBE_DEFERRAL_BOUND_INTERVALS: u64 = PEER_PROBE_TOKEN_WIDTHS.len() as u64;
/// Maximum measured owner-thread wall cost that may remain on an interactive scheduler boundary.
const PEER_RUNTIME_PROBE_BUDGET_NS: u64 = 5_000_000;

pub const PEER_PROBE_REQUIRED_REFUSAL: &str = "PP bring-up refused: MEMRA_PEER_PROBE=0 cannot authorize native peer transport for a \
     sharded cross-device placement while MEMRA_PP_HOST_BOUNCE!=1; leave MEMRA_PEER_PROBE \
     enabled or set MEMRA_PP_HOST_BOUNCE=1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerProbeStartupPolicy {
    Allowed,
    BypassedWithHostBounce,
}

/// Pure startup policy so unit tests and kernel-check pin the entire refusal matrix without
/// mutating process-global environment variables.
pub fn peer_probe_startup_policy(
    probe_on: bool,
    sharded_cross_device: bool,
    host_bounce: bool,
) -> Result<PeerProbeStartupPolicy, &'static str> {
    match (probe_on, sharded_cross_device, host_bounce) {
        (false, true, false) => Err(PEER_PROBE_REQUIRED_REFUSAL),
        (false, true, true) => Ok(PeerProbeStartupPolicy::BypassedWithHostBounce),
        _ => Ok(PeerProbeStartupPolicy::Allowed),
    }
}

static PEER_PROBE_BYPASSED: AtomicU64 = AtomicU64::new(0);
static PEER_BOUNDARY_COPIES: AtomicU64 = AtomicU64::new(0);
static PEER_RUNTIME_PROBES: AtomicU64 = AtomicU64::new(0);
static PEER_RUNTIME_PROBE_FAILURES: AtomicU64 = AtomicU64::new(0);
static PEER_RUNTIME_PROBE_DEFERRED: AtomicU64 = AtomicU64::new(0);
static PEER_RUNTIME_PROBE_INTEGRITY_DEGRADED: AtomicBool = AtomicBool::new(false);
static PEER_RUNTIME_PROBE_FAILED: AtomicBool = AtomicBool::new(false);
static PEER_RUNTIME_HOST_BOUNCE: AtomicBool = AtomicBool::new(false);
static PEER_RUNTIME_NEXT_PROBE_COPY: [AtomicU64; PEER_PROBE_TOKEN_WIDTHS.len()] = [
    AtomicU64::new(PEER_RUNTIME_PROBE_INTERVAL_COPIES),
    AtomicU64::new(2 * PEER_RUNTIME_PROBE_INTERVAL_COPIES),
    AtomicU64::new(3 * PEER_RUNTIME_PROBE_INTERVAL_COPIES),
    AtomicU64::new(4 * PEER_RUNTIME_PROBE_INTERVAL_COPIES),
];
static PEER_RUNTIME_PROBE_MAX_COST_NS: [AtomicU64; PEER_PROBE_TOKEN_WIDTHS.len()] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PeerProbeMetrics {
    pub bypassed: u64,
    pub boundary_copies: u64,
    pub runtime_probes: u64,
    pub runtime_failures: u64,
    pub deferred_total: u64,
    pub integrity_degraded: bool,
    pub degraded_to_host_bounce: bool,
}

pub fn peer_probe_metrics() -> PeerProbeMetrics {
    PeerProbeMetrics {
        bypassed: PEER_PROBE_BYPASSED.load(Ordering::Relaxed),
        boundary_copies: PEER_BOUNDARY_COPIES.load(Ordering::Relaxed),
        runtime_probes: PEER_RUNTIME_PROBES.load(Ordering::Relaxed),
        runtime_failures: PEER_RUNTIME_PROBE_FAILURES.load(Ordering::Relaxed),
        deferred_total: PEER_RUNTIME_PROBE_DEFERRED.load(Ordering::Relaxed),
        integrity_degraded: PEER_RUNTIME_PROBE_INTEGRITY_DEGRADED.load(Ordering::Acquire),
        degraded_to_host_bounce: PEER_RUNTIME_HOST_BOUNCE.load(Ordering::Acquire),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimePeerProbeStatus {
    NotRun,
    Deferred,
    Passed,
    DegradedToHostBounce,
}

impl RuntimePeerProbeStatus {
    pub fn ran(self) -> bool {
        matches!(self, Self::Passed | Self::DegradedToHostBounce)
    }
}

fn publish_runtime_peer_probe_deferral(
    deferred_total: &AtomicU64,
    integrity_degraded: &AtomicBool,
    intervals: u64,
    bound_reached: bool,
) {
    deferred_total.fetch_add(intervals, Ordering::Relaxed);
    if bound_reached {
        integrity_degraded.store(true, Ordering::Release);
    }
}

/// Publish newly observed copy-count intervals where a runnable peer probe was blocked by live
/// speculative UVA state. The worker coalesces scheduler polls before calling this function.
pub fn record_runtime_peer_probe_deferral(intervals: u64, bound_reached: bool) {
    publish_runtime_peer_probe_deferral(
        &PEER_RUNTIME_PROBE_DEFERRED,
        &PEER_RUNTIME_PROBE_INTEGRITY_DEGRADED,
        intervals,
        bound_reached,
    );
}

/// A completed native probe or validated transport failover restores an explicit integrity state.
pub fn clear_runtime_peer_probe_integrity_degraded() {
    PEER_RUNTIME_PROBE_INTEGRITY_DEGRADED.store(false, Ordering::Release);
}

fn runtime_peer_probe_idle_only(width_index: usize, measured_cost_ns: u64) -> bool {
    width_index + 1 == PEER_PROBE_TOKEN_WIDTHS.len()
        || measured_cost_ns > PEER_RUNTIME_PROBE_BUDGET_NS
}

/// Pick the oldest runnable per-width deadline. Idle-only overdue work is skipped rather than
/// blocking later cheap deadlines, so the small integrity ladder keeps its copy-count cadence.
fn runtime_peer_probe_candidate(
    copies: u64,
    next_probe_copy: [u64; PEER_PROBE_TOKEN_WIDTHS.len()],
    measured_cost_ns: [u64; PEER_PROBE_TOKEN_WIDTHS.len()],
    scheduler_idle: bool,
) -> Option<(usize, usize)> {
    let mut selected: Option<(usize, u64)> = None;
    for width_index in 0..PEER_PROBE_TOKEN_WIDTHS.len() {
        let due = next_probe_copy[width_index];
        if copies < due
            || (!scheduler_idle
                && runtime_peer_probe_idle_only(width_index, measured_cost_ns[width_index]))
        {
            continue;
        }
        if selected.is_none_or(|(_, selected_due)| due < selected_due) {
            selected = Some((width_index, due));
        }
    }
    selected.map(|(width_index, _)| (width_index, PEER_PROBE_TOKEN_WIDTHS[width_index]))
}

/// Advance a late per-width deadline to the first future cycle. Missed idle opportunities
/// collapse into one probe instead of producing an owner-thread catch-up burst.
fn runtime_peer_probe_next_copy(due: u64, copies: u64) -> u64 {
    let cycles = copies.saturating_sub(due) / PEER_RUNTIME_PROBE_CYCLE_COPIES + 1;
    due.saturating_add(PEER_RUNTIME_PROBE_CYCLE_COPIES.saturating_mul(cycles))
}

/// Fail closed before arming the fallback, then publish host bounce only after its staging check
/// succeeds. The two atomics are parameters so unit tests never mutate process-global state.
fn latch_runtime_host_bounce<E>(
    native_failed: &AtomicBool,
    degraded_to_host_bounce: &AtomicBool,
    arm_and_validate: impl FnOnce() -> Result<(), E>,
) -> Result<(), E> {
    native_failed.store(true, Ordering::Release);
    arm_and_validate()?;
    degraded_to_host_bounce.store(true, Ordering::Release);
    Ok(())
}

fn peer_probe_on() -> bool {
    std::env::var("MEMRA_PEER_PROBE").as_deref() != Ok("0")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PeerProbeDecision {
    Clean,
    ProceedWithHostBounce { mismatches: usize },
}

fn peer_probe_mismatch_count(expected: &[u8], readback: &[u8]) -> usize {
    expected
        .iter()
        .zip(readback)
        .filter(|(a, b)| a != b)
        .count()
        + expected.len().abs_diff(readback.len())
}

fn peer_probe_decision(
    expected: &[u8],
    readback: &[u8],
    host_bounce: bool,
) -> Result<PeerProbeDecision, String> {
    let mismatches = peer_probe_mismatch_count(expected, readback);
    if mismatches == 0 {
        Ok(PeerProbeDecision::Clean)
    } else if host_bounce {
        Ok(PeerProbeDecision::ProceedWithHostBounce { mismatches })
    } else {
        Err(format!("{mismatches} mismatched byte(s)"))
    }
}

fn peer_probe_pattern(bytes: usize, boundary: usize, src_dev: usize, dst_dev: usize) -> Vec<u8> {
    let mut state = 0xD1B5_4A32_D192_ED03u64
        ^ (bytes as u64).rotate_left(7)
        ^ (boundary as u64).rotate_left(19)
        ^ (src_dev as u64).rotate_left(31)
        ^ (dst_dev as u64).rotate_left(43);
    (0..bytes)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

fn peer_probe_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % std::mem::size_of::<f32>(), 0);
    bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_bits(u32::from_ne_bytes(chunk.try_into().unwrap())))
        .collect()
}

fn peer_probe_f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_ne_bytes())
        .collect()
}

/// A legacy `cuMemAlloc` buffer used only by the boot probe. Unlike memra's normal
/// stream-ordered allocations, it becomes peer-visible through `cuCtxEnablePeerAccess`
/// without requiring the default-pool grants that deliberately happen after the probe.
struct PeerProbeBuffer {
    ctx: Arc<CudaContext>,
    ptr: cudarc::driver::sys::CUdeviceptr,
}

impl PeerProbeBuffer {
    fn new(ctx: &Arc<CudaContext>, bytes: usize) -> Result<Self, Box<dyn std::error::Error>> {
        ctx.bind_to_thread()?;
        let ptr = unsafe { cudarc::driver::result::malloc_sync(bytes)? };
        Ok(Self {
            ctx: ctx.clone(),
            ptr,
        })
    }
}

impl Drop for PeerProbeBuffer {
    fn drop(&mut self) {
        if self.ctx.bind_to_thread().is_ok() {
            let _ = unsafe { cudarc::driver::result::free_sync(self.ptr) };
        }
    }
}

fn peer_probe_copy(
    src: &StageRt,
    dst: &StageRt,
    expected: &[u8],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let bytes = expected.len();
    let src_buf = PeerProbeBuffer::new(&src.ctx, bytes)?;
    unsafe {
        cudarc::driver::result::memcpy_htod_sync(src_buf.ptr, expected)?;
    }

    let dst_buf = PeerProbeBuffer::new(&dst.ctx, bytes)?;
    let poison: Vec<u8> = expected.iter().map(|b| !b).collect();
    unsafe {
        cudarc::driver::result::memcpy_htod_sync(dst_buf.ptr, &poison)?;
    }

    src.ctx.bind_to_thread()?;
    unsafe {
        cudarc::driver::result::memcpy_peer_async(
            dst.ctx.cu_ctx(),
            dst_buf.ptr,
            src.ctx.cu_ctx(),
            src_buf.ptr,
            bytes,
            src.stream.cu_stream(),
        )?;
    }
    src.stream.synchronize()?;

    dst.ctx.bind_to_thread()?;
    let mut readback = vec![0u8; bytes];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(&mut readback, dst_buf.ptr)?;
    }
    Ok(readback)
}

fn run_peer_probe_pass(
    stages: &[StageRt],
    peer_capable: &[(usize, usize)],
    host_bounce: bool,
    label: &str,
    bytes: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if bytes == 0 {
        return Err(format!("PP peer byte-integrity probe {label} size is zero").into());
    }
    let started = std::time::Instant::now();
    let mut copies = 0usize;
    let mut skipped = 0usize;
    let mut total_mismatches = 0usize;

    for boundary in 0..stages.len() - 1 {
        if stages[boundary].dev == stages[boundary + 1].dev {
            continue;
        }
        for (src_idx, dst_idx) in [(boundary, boundary + 1), (boundary + 1, boundary)] {
            let src = &stages[src_idx];
            let dst = &stages[dst_idx];
            if !peer_capable.contains(&(src.dev, dst.dev)) {
                if host_bounce {
                    skipped += 1;
                    eprintln!(
                        "[pp] peer byte-integrity probe SKIP: boundary={boundary} \
                         dev{}->dev{} label={label} bytes={bytes} (peer capability unavailable; \
                         MEMRA_PP_HOST_BOUNCE=1 remains fail-safe)",
                        src.dev, dst.dev,
                    );
                    continue;
                }
                return Err(format!(
                    "PP peer byte-integrity probe cannot run boundary={boundary} \
                     dev{}->dev{}: peer access was not enabled",
                    src.dev, dst.dev,
                )
                .into());
            }

            let expected = peer_probe_pattern(bytes, boundary, src.dev, dst.dev);
            let readback = match peer_probe_copy(src, dst, &expected) {
                Ok(readback) => readback,
                Err(err) if host_bounce => {
                    skipped += 1;
                    eprintln!(
                        "[pp] peer byte-integrity probe ERROR: boundary={boundary} \
                         dev{}->dev{} label={label} bytes={bytes}: {err}; \
                         MEMRA_PP_HOST_BOUNCE=1, proceeding on the host-staged path",
                        src.dev, dst.dev,
                    );
                    continue;
                }
                Err(err) => {
                    return Err(format!(
                        "PP peer byte-integrity probe FAILED: boundary={boundary} \
                         dev{}->dev{} label={label} bytes={bytes}: {err}; refusing native P2P \
                         (set MEMRA_PP_HOST_BOUNCE=1 to use the host-staged path; \
                         MEMRA_PEER_PROBE=0 cannot authorize sharded native peer transport)",
                        src.dev, dst.dev,
                    )
                    .into());
                }
            };
            copies += 1;
            match peer_probe_decision(&expected, &readback, host_bounce) {
                Ok(PeerProbeDecision::Clean) => {}
                Ok(PeerProbeDecision::ProceedWithHostBounce { mismatches }) => {
                    total_mismatches += mismatches;
                    eprintln!(
                        "[pp] peer byte-integrity probe CORRUPTION: boundary={boundary} \
                         dev{}->dev{} label={label} bytes={bytes} mismatches={mismatches}; \
                         MEMRA_PP_HOST_BOUNCE=1, proceeding on the host-staged path",
                        src.dev, dst.dev,
                    );
                }
                Err(mismatch) => {
                    return Err(format!(
                        "PP peer byte-integrity probe FAILED: boundary={boundary} \
                         dev{}->dev{} label={label} bytes={bytes}: {mismatch}; refusing native \
                         P2P (set MEMRA_PP_HOST_BOUNCE=1 to use the host-staged path; \
                         MEMRA_PEER_PROBE=0 cannot authorize sharded native peer transport)",
                        src.dev, dst.dev,
                    )
                    .into());
                }
            }
        }
    }

    let status = if total_mismatches > 0 {
        "BOUNCE"
    } else if skipped > 0 && copies > 0 {
        "PARTIAL"
    } else if skipped > 0 {
        "SKIP"
    } else {
        "PASS"
    };
    eprintln!(
        "[pp] peer byte-integrity probe {}: label={label} bytes={bytes} copies={copies} \
         skipped={skipped} mismatches={total_mismatches} elapsed_ms={:.3}",
        status,
        started.elapsed().as_secs_f64() * 1e3,
    );
    Ok(())
}

fn host_bounce_capacity(n_embd: usize) -> Result<(usize, usize), String> {
    if n_embd == 0 {
        return Err("MEMRA_PP_HOST_BOUNCE needs non-zero model n_embd".into());
    }
    let elems = n_embd
        .checked_mul(crate::cache::PRIME_CHUNK_MAX_TOKENS)
        .ok_or_else(|| format!("host-bounce element count overflows for n_embd={n_embd}"))?;
    let bytes = elems
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("host-bounce byte count overflows for n_embd={n_embd}"))?;
    Ok((elems, bytes))
}

/// One bidirectional-DMA staging allocation. `CU_MEMHOSTALLOC_PORTABLE` matters here: the
/// D2H producer and H2D consumer are in distinct CUDA primary contexts. Cacheable memory is
/// intentional (rather than cudarc's write-combined pinned slice) because this allocation is
/// the destination of D2H as well as the source of H2D.
struct PinnedHostBounce {
    ptr: *mut f32,
    len: usize,
}

unsafe impl Send for PinnedHostBounce {}
unsafe impl Sync for PinnedHostBounce {}

impl PinnedHostBounce {
    fn new(len: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("host-bounce pinned allocation size overflow")?;
        let ptr = unsafe {
            cudarc::driver::result::malloc_host(
                bytes,
                cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE,
            )?
        } as *mut f32;
        if ptr.is_null() {
            return Err("cuMemHostAlloc returned a null host-bounce pointer".into());
        }
        Ok(Self { ptr, len })
    }

    fn prefix(&self, n: usize) -> &[f32] {
        assert!(
            n <= self.len,
            "host-bounce source {n} > capacity {}",
            self.len
        );
        unsafe { std::slice::from_raw_parts(self.ptr, n) }
    }

    fn prefix_mut(&mut self, n: usize) -> &mut [f32] {
        assert!(
            n <= self.len,
            "host-bounce destination {n} > capacity {}",
            self.len
        );
        unsafe { std::slice::from_raw_parts_mut(self.ptr, n) }
    }
}

impl Drop for PinnedHostBounce {
    fn drop(&mut self) {
        let _ = unsafe { cudarc::driver::result::free_host(self.ptr.cast()) };
    }
}

struct HostBounceRt {
    n_embd: usize,
    capacity: usize,
    slots: Vec<Option<[Mutex<PinnedHostBounce>; 2]>>,
}

impl HostBounceRt {
    fn new(n_embd: usize, boundaries: &[BoundaryRt]) -> Result<Self, Box<dyn std::error::Error>> {
        let (capacity, _) = host_bounce_capacity(n_embd)?;
        let mut slots = Vec::with_capacity(boundaries.len());
        for boundary in boundaries {
            slots.push(if boundary.cross {
                Some([
                    Mutex::new(PinnedHostBounce::new(capacity)?),
                    Mutex::new(PinnedHostBounce::new(capacity)?),
                ])
            } else {
                None
            });
        }
        Ok(Self {
            n_embd,
            capacity,
            slots,
        })
    }

    fn slot(
        &self,
        boundary: usize,
        slot: usize,
    ) -> Result<&Mutex<PinnedHostBounce>, Box<dyn std::error::Error>> {
        self.slots
            .get(boundary)
            .and_then(Option::as_ref)
            .and_then(|slots| slots.get(slot))
            .ok_or_else(|| format!("host-bounce slot {boundary}:{slot} is not initialized").into())
    }
}

pub struct PpNRt {
    stages: Vec<StageRt>,
    boundaries: Vec<BoundaryRt>,
    /// true iff ANY boundary crosses devices.
    cross_any: bool,
    /// Startup selection captured at runtime construction. A runtime probe failure may promote
    /// the process-wide one-way host-bounce latch without mutating this value.
    host_bounce: bool,
    /// Boot-time peer validation is default-on; `MEMRA_PEER_PROBE=0` is diagnostics-only.
    peer_probe: bool,
    /// Directed device pairs for which `cuDeviceCanAccessPeer` succeeded.
    peer_capable: Vec<(usize, usize)>,
    /// Sticky one-time model-width probe result. The value is the one-row geometry byte count.
    peer_probe_geometry: OnceLock<Result<usize, String>>,
    /// Lazily allocated after the authoritative model width is known at cache creation.
    bounce: OnceLock<Result<HostBounceRt, String>>,
    /// Dedicated readback stream in the LAST stage's context (deferred logits D2H —
    /// waiting there instead of on the compute stream keeps later tokens enqueuable).
    readback: Arc<CudaStream>,
}

/// M1 name kept alive for external callers (`pp-transport-smoke`, receipts, docs).
pub type Pp2Rt = PpNRt;

static RTN: OnceLock<Result<PpNRt, String>> = OnceLock::new();

impl PpNRt {
    /// The process-wide transport runtime, built on first use against the primary engine.
    /// The stage count + device map freeze at first build (one config per process — gates
    /// run one placement per invocation). Build errors are sticky and loud.
    pub fn get(e: &Engine) -> Result<&'static PpNRt, Box<dyn std::error::Error>> {
        RTN.get_or_init(|| Self::build(e).map_err(|err| err.to_string()))
            .as_ref()
            .map_err(|s| -> Box<dyn std::error::Error> { s.clone().into() })
    }

    fn build(e: &Engine) -> Result<PpNRt, Box<dyn std::error::Error>> {
        let primary_dev = e.ctx().ordinal();
        // Stage count: MEMRA_PP_DEVICES length wins when set (it IS the placement);
        // else MEMRA_PP_STAGES; else 2 (the M1 default — pp-transport-smoke runs doorless).
        let devices: Vec<usize> =
            match pp2_devices_env() {
                Some(s) => {
                    let parts: Result<Vec<usize>, _> =
                        s.split(',').map(|p| p.trim().parse::<usize>()).collect();
                    match parts {
                        Ok(v) if v.len() >= 2 => v,
                        _ => return Err(format!(
                            "MEMRA_PP_DEVICES={s} unparseable (want <d0>,..,<dN-1> e.g. 0,1,2,3)"
                        )
                        .into()),
                    }
                }
                None => {
                    let n_st = std::env::var("MEMRA_PP_STAGES")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|&n| n >= 2)
                        .unwrap_or(2);
                    vec![primary_dev; n_st]
                }
            };
        if let Ok(v) = std::env::var("MEMRA_PP_STAGES") {
            if let Ok(n) = v.parse::<usize>() {
                if n >= 2 && n != devices.len() {
                    return Err(format!(
                        "MEMRA_PP_DEVICES lists {} devices but MEMRA_PP_STAGES={n} — \
                         refusing an ambiguous placement",
                        devices.len()
                    )
                    .into());
                }
            }
        }
        let n_st = devices.len();
        let cross_any = devices.iter().any(|&d| d != devices[0]);
        let host_bounce = pp_host_bounce_on();
        let peer_probe = peer_probe_on();
        let sharded_cross_device = cross_any && !pp_shard_off();
        if host_bounce && cross_any {
            if pp_shard_off() {
                return Err(
                    "MEMRA_PP_HOST_BOUNCE=1 refuses MEMRA_PP_SHARD=0: the boundary can bounce, \
                     but remote stages would still peer-read primary-device weights"
                        .into(),
                );
            }
            if devices.last().copied() != Some(primary_dev) {
                return Err(format!(
                    "MEMRA_PP_HOST_BOUNCE=1 requires the primary engine on the last/head stage \
                     (primary dev{primary_dev}, placement {devices:?}); otherwise returned \
                     logits/hidden state remain peer reads"
                )
                .into());
            }
        }
        let peer_probe_policy =
            peer_probe_startup_policy(peer_probe, sharded_cross_device, host_bounce)?;
        if peer_probe_policy == PeerProbeStartupPolicy::BypassedWithHostBounce {
            PEER_PROBE_BYPASSED.fetch_add(1, Ordering::Relaxed);
            eprintln!(
                "[pp] SECURITY RED: peer_probe_bypassed: MEMRA_PEER_PROBE=0 on a sharded \
                 cross-device placement; MEMRA_PP_HOST_BOUNCE=1 is the only enabled transport"
            );
        }

        // Validate every placement ordinal in both transports. Native peer transport requires
        // access both ways. Host bounce remains usable without it, but records any capable pairs
        // so the byte probe can still diagnose a lying peer path before selecting the fallback.
        let mut used: Vec<usize> = devices.clone();
        used.push(primary_dev);
        used.sort_unstable();
        used.dedup();
        let mut peer_capable = Vec::new();
        if used.len() > 1 {
            let n = cudarc::driver::result::device::get_count()? as usize;
            for &d in &used {
                if d >= n {
                    return Err(format!(
                        "MEMRA_PP_DEVICES={devices:?} but only {n} CUDA device(s) present"
                    )
                    .into());
                }
            }
            if !host_bounce || peer_probe {
                for &a in &used {
                    for &b in &used {
                        if a == b {
                            continue;
                        }
                        let da = cudarc::driver::result::device::get(a as i32)?;
                        let db = cudarc::driver::result::device::get(b as i32)?;
                        let mut can: i32 = 0;
                        let capability = unsafe {
                            cudarc::driver::sys::cuDeviceCanAccessPeer(&mut can, da, db).result()
                        };
                        if let Err(err) = capability {
                            if host_bounce {
                                eprintln!(
                                    "[pp] peer byte-integrity probe capability query failed for \
                                     dev{a}->dev{b}: {err}; MEMRA_PP_HOST_BOUNCE=1 remains active"
                                );
                                continue;
                            }
                            return Err(err.into());
                        }
                        if can == 0 {
                            if !host_bounce {
                                return Err(format!(
                                    "device {a} cannot peer-access device {b} \
                                     (cuDeviceCanAccessPeer=0); ppN cross-device needs P2P — \
                                     refusing a silently-staged path"
                                )
                                .into());
                            }
                        } else {
                            peer_capable.push((a, b));
                        }
                    }
                }
            }
        }

        // PER-STAGE ENGINE ISOLATION (2026-08-02 singledev pipelined find): Engine owns
        // lazily-grown SHARED scratch pools (fa_part_pool, fa_vf16_scratch, argmax
        // partials, ...) that are stable-pointer by design — safe on one stream, a data
        // race the moment two stage streams run concurrently through the SAME Engine
        // (deferred readback, >=2 tokens in flight: token t+1's stage-0 fa memsets the
        // partials while token t's stage-s fa still reads them — the nondeterministic
        // all-logits divergence; cross-device arms were immune because remote stages
        // already got their own Engine). Every stage s>0 gets its OWN Engine even on the
        // primary device: same CUcontext (primary retain), so the per-context CUmodule
        // cache makes it cheap; scratch pools are per-Engine, so stages never share.
        // Stage 0 keeps the primary engine (single-threaded host issue: the only
        // concurrent user of `e` during a pp walk is stage 0 itself).
        let mk_stage = |dev: usize, s: usize| -> Result<StageRt, Box<dyn std::error::Error>> {
            if dev == primary_dev && s == 0 {
                let ctx = e.ctx().clone();
                let stream = ctx.new_stream()?;
                let blas = Arc::new(cudarc::cublaslt::CudaBlasLT::new(stream.clone())?);
                Ok(StageRt {
                    dev,
                    ctx,
                    stream,
                    blas,
                    engine: None,
                })
            } else {
                let eng = Engine::new(dev)?;
                let ctx = eng.ctx().clone();
                let stream = ctx.new_stream()?;
                let blas = Arc::new(cudarc::cublaslt::CudaBlasLT::new(stream.clone())?);
                Ok(StageRt {
                    dev,
                    ctx,
                    stream,
                    blas,
                    engine: Some(eng),
                })
            }
        };
        let mut stages = Vec::with_capacity(n_st);
        for (s, &d) in devices.iter().enumerate() {
            stages.push(mk_stage(d, s)?);
        }

        if cross_any
            && !peer_probe
            && peer_probe_policy != PeerProbeStartupPolicy::BypassedWithHostBounce
        {
            eprintln!(
                "[pp] WARNING: MEMRA_PEER_PROBE=0 skips the boot-time peer byte-integrity \
                 gate; diagnostics escape hatch active"
            );
        }

        if used.len() > 1 {
            if !host_bounce {
                // A context per distinct device (first stage that lives there; the primary's
                // context for the primary device).
                let ctx_of = |d: usize| -> &Arc<CudaContext> {
                    if d == primary_dev {
                        e.ctx()
                    } else {
                        &stages.iter().find(|s| s.dev == d).unwrap().ctx
                    }
                };
                // Enable peer access BOTH ways for every distinct pair (idempotent;
                // ALREADY_ENABLED is success).
                for &a in &used {
                    for &b in &used {
                        if a == b {
                            continue;
                        }
                        ctx_of(a).bind_to_thread()?;
                        let rc = unsafe {
                            cudarc::driver::sys::cuCtxEnablePeerAccess(ctx_of(b).cu_ctx(), 0)
                        };
                        use cudarc::driver::sys::cudaError_enum as E;
                        if rc != E::CUDA_SUCCESS && rc != E::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
                        {
                            return Err(format!(
                                "cuCtxEnablePeerAccess(dev{a} -> dev{b}) failed: {rc:?}"
                            )
                            .into());
                        }
                    }
                }
                // The fixed-size byte gate runs immediately after peer enable and before pool
                // grants. Legacy allocations make it exercise the exact `cuMemcpyPeerAsync` API
                // without depending on the pool setup that follows.
                if peer_probe && cross_any {
                    let probe = run_peer_probe_pass(
                        &stages,
                        &peer_capable,
                        host_bounce,
                        "fixed-16KiB",
                        PEER_PROBE_FIXED_BYTES,
                    );
                    e.ctx().bind_to_thread()?;
                    probe?;
                }
                // MEM-POOL access grant (8x box 2026-08-02, M1 cross-device fix #2):
                // cuCtxEnablePeerAccess does NOT map STREAM-ORDERED POOL allocations, and every
                // engine buffer/weight goes through the device default pool (cuMemAllocAsync via
                // cudarc; memra-runtime configures that pool). A stage kernel dereferencing
                // another device's weights — or a boundary peer TX writing the RX slot — needs
                // cuMemPoolSetAccess on the OWNING device's default pool for the ACCESSING
                // device; without it the first remote dereference is CUDA_ERROR_ILLEGAL_ADDRESS
                // (reported at the next API call in the poisoned context). Grant all pairs.
                for &owner in &used {
                    for &accessor in &used {
                        if owner == accessor {
                            continue;
                        }
                        let dev = cudarc::driver::result::device::get(owner as i32)?;
                        let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
                        unsafe {
                            cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, dev)
                                .result()?;
                        }
                        let desc = cudarc::driver::sys::CUmemAccessDesc {
                        location: cudarc::driver::sys::CUmemLocation {
                            type_: cudarc::driver::sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                            id: accessor as i32,
                        },
                        flags: cudarc::driver::sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                    };
                        let rc = unsafe { cudarc::driver::sys::cuMemPoolSetAccess(pool, &desc, 1) };
                        if rc != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                            return Err(format!(
                            "cuMemPoolSetAccess(dev{owner} pool -> dev{accessor}) failed: {rc:?}"
                        )
                        .into());
                        }
                    }
                }
                // MEM-POOL access grant (8x box 2026-08-02, cross-device fix #2):
                // cuCtxEnablePeerAccess does NOT map STREAM-ORDERED POOL allocations, and every
                // engine buffer/weight goes through the device default pool (cuMemAllocAsync via
                // cudarc; memra-runtime configures that pool). A stage-1 kernel dereferencing
                // dev0 weights — or the stage-0 peer TX writing dev1's RX slot — needs
                // cuMemPoolSetAccess on the OWNING device's default pool for the ACCESSING
                // device; without it the first remote dereference is CUDA_ERROR_ILLEGAL_ADDRESS
                // (reported at the next API call in the poisoned context). Grant both ways.
                for (owner, accessor) in [
                    (stages[0].dev, stages[1].dev),
                    (stages[1].dev, stages[0].dev),
                ] {
                    let dev = cudarc::driver::result::device::get(owner as i32)?;
                    let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
                    unsafe {
                        cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, dev).result()?;
                    }
                    let desc = cudarc::driver::sys::CUmemAccessDesc {
                    location: cudarc::driver::sys::CUmemLocation {
                        type_: cudarc::driver::sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                        id: accessor as i32,
                    },
                    flags: cudarc::driver::sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                };
                    let rc = unsafe { cudarc::driver::sys::cuMemPoolSetAccess(pool, &desc, 1) };
                    if rc != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(format!(
                            "cuMemPoolSetAccess(dev{owner} pool -> dev{accessor}) failed: {rc:?}"
                        )
                        .into());
                    }
                }
                // restore the primary context for the caller's subsequent work
                e.ctx().bind_to_thread()?;
                eprintln!(
                    "[pp] cross-device transport: {} (cudaMemcpyPeerAsync per cross boundary; \
                 peer + default-pool access granted all pairs over {used:?}; weight home: {})",
                    devices
                        .iter()
                        .enumerate()
                        .map(|(s, d)| format!("stage{s}=dev{d}"))
                        .collect::<Vec<_>>()
                        .join(" "),
                    if pp_shard_off() {
                        format!("dev{primary_dev} (MEMRA_PP_SHARD=0 bring-up placement)")
                    } else {
                        "per-stage (sharded loader)".to_string()
                    }
                );
            } else {
                e.ctx().bind_to_thread()?;
                eprintln!(
                    "[pp] cross-device transport: {} (HOST-STAGED pinned D2H -> H2D per cross \
                     boundary; MEMRA_PP_HOST_BOUNCE=1; peer-pool grants bypassed; \
                     diagnostic peer access is removed before host-staged serving; \
                     weight home: per-stage (sharded loader))",
                    devices
                        .iter()
                        .enumerate()
                        .map(|(s, d)| format!("stage{s}=dev{d}"))
                        .collect::<Vec<_>>()
                        .join(" "),
                );
            }
        }

        let mk_slot =
            |tx: &StageRt, rx: &StageRt| -> Result<BoundarySlot, Box<dyn std::error::Error>> {
                Ok(BoundarySlot {
                    buf: Mutex::new(None),
                    ev_tx: tx.ctx.new_event(None)?,
                    ev_rx: rx.ctx.new_event(None)?,
                })
            };
        let mut boundaries = Vec::with_capacity(n_st - 1);
        for b in 0..n_st - 1 {
            let (tx, rx) = (&stages[b], &stages[b + 1]);
            boundaries.push(BoundaryRt {
                slots: [mk_slot(tx, rx)?, mk_slot(tx, rx)?],
                step: AtomicUsize::new(0),
                cross: tx.dev != rx.dev,
            });
        }
        let readback = stages[n_st - 1].ctx.new_stream()?;
        let rt = PpNRt {
            stages,
            boundaries,
            cross_any,
            host_bounce,
            peer_probe,
            peer_capable,
            peer_probe_geometry: OnceLock::new(),
            bounce: OnceLock::new(),
            readback,
        };
        if rt.peer_probe && rt.cross_any && rt.host_bounce {
            rt.run_host_bounce_legacy_probe(e)?;
        }
        Ok(rt)
    }

    pub fn n_stages(&self) -> usize {
        self.stages.len()
    }

    /// True iff any boundary crosses devices.
    pub fn cross_device(&self) -> bool {
        self.cross_any
    }

    fn host_bounce_active(&self) -> bool {
        self.host_bounce || PEER_RUNTIME_HOST_BOUNCE.load(Ordering::Acquire)
    }

    fn context_for_dev<'a>(
        &'a self,
        e: &'a Engine,
        dev: usize,
    ) -> Result<&'a Arc<CudaContext>, Box<dyn std::error::Error>> {
        if dev == e.ctx().ordinal() {
            return Ok(e.ctx());
        }
        self.stages
            .iter()
            .find(|stage| stage.dev == dev)
            .map(|stage| &stage.ctx)
            .ok_or_else(|| format!("PP peer probe has no CUDA context for dev{dev}").into())
    }

    fn enable_probe_peer_access(
        &self,
        e: &Engine,
        pairs: &[(usize, usize)],
    ) -> Result<Vec<(usize, usize)>, Box<dyn std::error::Error>> {
        let mut enabled = Vec::new();
        for &(src_dev, dst_dev) in pairs {
            let enable = (|| -> Result<(), Box<dyn std::error::Error>> {
                let src_ctx = self.context_for_dev(e, src_dev)?;
                let dst_ctx = self.context_for_dev(e, dst_dev)?;
                src_ctx.bind_to_thread()?;
                let rc = unsafe { cudarc::driver::sys::cuCtxEnablePeerAccess(dst_ctx.cu_ctx(), 0) };
                use cudarc::driver::sys::cudaError_enum as E;
                if rc == E::CUDA_SUCCESS || rc == E::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
                    Ok(())
                } else {
                    Err(format!("{rc:?}").into())
                }
            })();
            if let Err(err) = enable {
                eprintln!(
                    "[pp] peer byte-integrity probe could not enable \
                     dev{src_dev}->dev{dst_dev}: {err}; MEMRA_PP_HOST_BOUNCE=1 remains active"
                );
            } else {
                enabled.push((src_dev, dst_dev));
            }
        }
        Ok(enabled)
    }

    fn disable_probe_peer_access(
        &self,
        e: &Engine,
        pairs: &[(usize, usize)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut failures = Vec::new();
        for &(src_dev, dst_dev) in pairs {
            let disable = (|| -> Result<(), Box<dyn std::error::Error>> {
                let src_ctx = self.context_for_dev(e, src_dev)?;
                let dst_ctx = self.context_for_dev(e, dst_dev)?;
                src_ctx.bind_to_thread()?;
                let rc = unsafe { cudarc::driver::sys::cuCtxDisablePeerAccess(dst_ctx.cu_ctx()) };
                use cudarc::driver::sys::cudaError_enum as E;
                if rc == E::CUDA_SUCCESS || rc == E::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED {
                    Ok(())
                } else {
                    Err(format!("{rc:?}").into())
                }
            })();
            if let Err(err) = disable {
                failures.push(format!("dev{src_dev}->dev{dst_dev}: {err}"));
            }
        }
        e.ctx().bind_to_thread()?;
        if failures.is_empty() {
            eprintln!(
                "[pp] peer byte-integrity probe teardown: disabled {} diagnostic pair(s); \
                 host-bounce serving has no probe-enabled peer access",
                pairs.len(),
            );
            Ok(())
        } else {
            Err(format!(
                "PP peer probe could not disable diagnostic peer access ({}); \
                 refusing host-bounce serving",
                failures.join(", "),
            )
            .into())
        }
    }

    fn grant_probe_pool_access(
        &self,
        e: &Engine,
        pairs: &[(usize, usize)],
    ) -> Result<Vec<(usize, usize)>, Box<dyn std::error::Error>> {
        let mut granted = Vec::new();
        for &(src_dev, dst_dev) in pairs {
            let grant = (|| -> Result<(), Box<dyn std::error::Error>> {
                self.context_for_dev(e, dst_dev)?.bind_to_thread()?;
                let dev = cudarc::driver::result::device::get(dst_dev as i32)?;
                let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
                unsafe {
                    cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, dev).result()?;
                }
                let desc = cudarc::driver::sys::CUmemAccessDesc {
                    location: cudarc::driver::sys::CUmemLocation {
                        type_: cudarc::driver::sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                        id: src_dev as i32,
                    },
                    flags:
                        cudarc::driver::sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                };
                let rc = unsafe { cudarc::driver::sys::cuMemPoolSetAccess(pool, &desc, 1) };
                if rc == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    Ok(())
                } else {
                    Err(format!("{rc:?}").into())
                }
            })();
            if let Err(err) = grant {
                eprintln!(
                    "[pp] production-slot probe could not grant dev{src_dev} access to \
                     dev{dst_dev}'s default pool: {err}; MEMRA_PP_HOST_BOUNCE=1 remains active"
                );
            } else {
                granted.push((src_dev, dst_dev));
            }
        }
        Ok(granted)
    }

    fn revoke_probe_pool_access(
        &self,
        e: &Engine,
        pairs: &[(usize, usize)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut failures = Vec::new();
        for &(src_dev, dst_dev) in pairs {
            let revoke = (|| -> Result<(), Box<dyn std::error::Error>> {
                self.context_for_dev(e, dst_dev)?.bind_to_thread()?;
                let dev = cudarc::driver::result::device::get(dst_dev as i32)?;
                let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
                unsafe {
                    cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, dev).result()?;
                }
                let desc = cudarc::driver::sys::CUmemAccessDesc {
                    location: cudarc::driver::sys::CUmemLocation {
                        type_: cudarc::driver::sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                        id: src_dev as i32,
                    },
                    flags: cudarc::driver::sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_NONE,
                };
                let rc = unsafe { cudarc::driver::sys::cuMemPoolSetAccess(pool, &desc, 1) };
                if rc == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    Ok(())
                } else {
                    Err(format!("{rc:?}").into())
                }
            })();
            if let Err(err) = revoke {
                failures.push(format!("dev{src_dev}->dev{dst_dev}: {err}"));
            }
        }
        e.ctx().bind_to_thread()?;
        if failures.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "PP peer probe could not revoke diagnostic pool access ({}); \
                 refusing host-bounce serving",
                failures.join(", "),
            )
            .into())
        }
    }

    fn run_host_bounce_legacy_probe(&self, e: &Engine) -> Result<(), Box<dyn std::error::Error>> {
        let enabled = self.enable_probe_peer_access(e, &self.peer_capable)?;
        let probe = run_peer_probe_pass(
            &self.stages,
            &enabled,
            true,
            "fixed-16KiB-legacy-preflight",
            PEER_PROBE_FIXED_BYTES,
        );
        let disable = self.disable_probe_peer_access(e, &enabled);
        disable?;
        probe
    }

    fn new_peer_probe_boundary(
        &self,
        src_stage: usize,
        dst_stage: usize,
    ) -> Result<BoundaryRt, Box<dyn std::error::Error>> {
        let tx = &self.stages[src_stage];
        let rx = &self.stages[dst_stage];
        let mk_slot = || -> Result<BoundarySlot, Box<dyn std::error::Error>> {
            Ok(BoundarySlot {
                buf: Mutex::new(None),
                ev_tx: tx.ctx.new_event(None)?,
                ev_rx: rx.ctx.new_event(None)?,
            })
        };
        Ok(BoundaryRt {
            slots: [mk_slot()?, mk_slot()?],
            step: AtomicUsize::new(0),
            cross: tx.dev != rx.dev,
        })
    }

    fn production_probe_readback(
        &self,
        path: BoundaryPath,
        boundary: &BoundaryRt,
        expected: &[u8],
        n: usize,
        slot_idx: usize,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        debug_assert_eq!(expected.len(), n * std::mem::size_of::<f32>());
        let host = peer_probe_bytes_to_f32(expected);
        let poison_bytes: Vec<u8> = expected.iter().map(|byte| !byte).collect();
        let poison = peer_probe_bytes_to_f32(&poison_bytes);
        let src = &self.stages[path.src_stage];
        let dst = &self.stages[path.dst_stage];

        // Pre-poison the exact stream-ordered BoundarySlot allocation so a missing or partial
        // peer write cannot accidentally agree where the deterministic source contains zeroes.
        dst.ctx.bind_to_thread()?;
        let poison_buf = dst.stream.clone_htod(&poison)?;
        dst.stream.synchronize()?;
        let replaced = boundary.slots[slot_idx]
            .buf
            .lock()
            .unwrap()
            .replace(poison_buf);
        drop(replaced);
        dst.stream.synchronize()?;

        src.ctx.bind_to_thread()?;
        let x = src.stream.clone_htod(&host)?;
        self.tx_slot_path(path, boundary, &x, n, slot_idx)?;

        dst.ctx.bind_to_thread()?;
        let work = self.rx_slot_path(path, boundary, slot_idx, n)?;
        let back = dst.stream.clone_dtoh(&work)?;
        dst.stream.synchronize()?;
        Ok(peer_probe_f32_to_bytes(&back))
    }

    fn clear_peer_probe_boundary(
        &self,
        boundary: &BoundaryRt,
        src_stage: usize,
        dst_stage: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.stages[dst_stage].ctx.bind_to_thread()?;
        for slot in &boundary.slots {
            let buffer = slot.buf.lock().unwrap().take();
            drop(buffer);
        }
        self.stages[src_stage].stream.synchronize()?;
        self.stages[dst_stage].stream.synchronize()?;
        Ok(())
    }

    fn run_production_peer_probe(
        &self,
        enabled_pairs: &[(usize, usize)],
        host_bounce: bool,
        n_embd: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let started = std::time::Instant::now();
        let mut copies = 0usize;
        let mut skipped = 0usize;
        let mut total_mismatches = 0usize;
        let mut largest_clean_payload = 0usize;

        for boundary_idx in 0..self.stages.len() - 1 {
            if self.stages[boundary_idx].dev == self.stages[boundary_idx + 1].dev {
                continue;
            }
            for (src_stage, dst_stage) in [
                (boundary_idx, boundary_idx + 1),
                (boundary_idx + 1, boundary_idx),
            ] {
                let src_dev = self.stages[src_stage].dev;
                let dst_dev = self.stages[dst_stage].dev;
                if !enabled_pairs.contains(&(src_dev, dst_dev)) {
                    if host_bounce {
                        skipped += PEER_PROBE_TOKEN_WIDTHS.len();
                        eprintln!(
                            "[pp] production-slot peer probe SKIP: boundary={boundary_idx} \
                             dev{src_dev}->dev{dst_dev} widths_tokens={:?} \
                             (peer or pool access unavailable; MEMRA_PP_HOST_BOUNCE=1 remains \
                             fail-safe)",
                            PEER_PROBE_TOKEN_WIDTHS,
                        );
                        continue;
                    }
                    return Err(format!(
                        "PP production-slot peer probe cannot run boundary={boundary_idx} \
                         dev{src_dev}->dev{dst_dev}: peer/pool access is not enabled"
                    )
                    .into());
                }

                let probe_boundary = self.new_peer_probe_boundary(src_stage, dst_stage)?;
                let path = BoundaryPath {
                    boundary: boundary_idx,
                    src_stage,
                    dst_stage,
                    transport: BoundaryTransport::Peer,
                };
                let mut direction_copies = 0usize;
                let mut direction_skipped = 0usize;
                let mut direction_mismatches = 0usize;
                let mut direction_largest_clean = 0usize;
                let mut failure = None;

                for (width_idx, tokens) in PEER_PROBE_TOKEN_WIDTHS.into_iter().enumerate() {
                    let n = n_embd.checked_mul(tokens).ok_or_else(|| {
                        format!(
                            "PP production-slot probe element count overflows for \
                             n_embd={n_embd} tokens={tokens}"
                        )
                    })?;
                    let bytes = n.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
                        format!(
                            "PP production-slot probe byte count overflows for \
                             n_embd={n_embd} tokens={tokens}"
                        )
                    })?;
                    let expected = peer_probe_pattern(bytes, boundary_idx, src_dev, dst_dev);
                    let readback = match self.production_probe_readback(
                        path,
                        &probe_boundary,
                        &expected,
                        n,
                        width_idx % 2,
                    ) {
                        Ok(readback) => readback,
                        Err(err) if host_bounce => {
                            skipped += 1;
                            direction_skipped += 1;
                            eprintln!(
                                "[pp] production-slot peer probe ERROR: \
                                 boundary={boundary_idx} dev{src_dev}->dev{dst_dev} \
                                 tokens={tokens} bytes={bytes}: {err}; \
                                 MEMRA_PP_HOST_BOUNCE=1, proceeding on the host-staged path"
                            );
                            continue;
                        }
                        Err(err) => {
                            failure = Some(format!(
                                "PP production-slot peer probe FAILED: \
                                 boundary={boundary_idx} dev{src_dev}->dev{dst_dev} \
                                 tokens={tokens} bytes={bytes}: {err}; refusing native P2P \
                                 (set MEMRA_PP_HOST_BOUNCE=1 to use the host-staged path; \
                                 MEMRA_PEER_PROBE=0 cannot authorize sharded native peer \
                                 transport)"
                            ));
                            break;
                        }
                    };
                    copies += 1;
                    direction_copies += 1;
                    let mismatches = peer_probe_mismatch_count(&expected, &readback);
                    if mismatches == 0 {
                        largest_clean_payload = largest_clean_payload.max(bytes);
                        direction_largest_clean = direction_largest_clean.max(bytes);
                    } else if host_bounce {
                        total_mismatches += mismatches;
                        direction_mismatches += mismatches;
                        eprintln!(
                            "[pp] production-slot peer probe CORRUPTION: \
                             boundary={boundary_idx} dev{src_dev}->dev{dst_dev} tokens={tokens} \
                             bytes={bytes} mismatches={mismatches}; MEMRA_PP_HOST_BOUNCE=1, \
                             proceeding on the host-staged path"
                        );
                    } else {
                        failure = Some(format!(
                            "PP production-slot peer probe FAILED: boundary={boundary_idx} \
                             dev{src_dev}->dev{dst_dev} tokens={tokens} bytes={bytes}: \
                             {mismatches} mismatched byte(s); refusing native P2P \
                             (set MEMRA_PP_HOST_BOUNCE=1 to use the host-staged path; \
                             MEMRA_PEER_PROBE=0 cannot authorize sharded native peer transport)"
                        ));
                        break;
                    }
                }

                self.clear_peer_probe_boundary(&probe_boundary, src_stage, dst_stage)?;
                if let Some(err) = failure {
                    return Err(err.into());
                }
                eprintln!(
                    "[pp] production-slot peer probe direction: boundary={boundary_idx} \
                     dev{src_dev}->dev{dst_dev} copies={direction_copies} \
                     skipped={direction_skipped} mismatches={direction_mismatches} \
                     largest_clean_payload_bytes={direction_largest_clean}"
                );
            }
        }

        let status = if total_mismatches > 0 {
            "BOUNCE"
        } else if skipped > 0 && copies > 0 {
            "PARTIAL"
        } else if skipped > 0 {
            "SKIP"
        } else {
            "PASS"
        };
        eprintln!(
            "[pp] production-slot peer probe {status}: widths_tokens={:?} copies={copies} \
             skipped={skipped} mismatches={total_mismatches} \
             largest_clean_payload_bytes={largest_clean_payload} elapsed_ms={:.3}",
            PEER_PROBE_TOKEN_WIDTHS,
            started.elapsed().as_secs_f64() * 1e3,
        );
        Ok(())
    }

    fn run_host_bounce_production_probe(
        &self,
        e: &Engine,
        n_embd: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let enabled = self.enable_probe_peer_access(e, &self.peer_capable)?;
        let granted = self.grant_probe_pool_access(e, &enabled)?;
        let probe = self.run_production_peer_probe(&granted, true, n_embd);
        // Teardown always runs, but the probe verdict wins: a CORRUPTION verdict (probe is
        // Err) must never be masked by a teardown failure. `revoke?; disable?; probe`
        // short-circuited teardown errors BEFORE probe was inspected, discarding the byte-
        // integrity signal on any teardown hiccup (hermes 9d6ae8d3). Surface teardown errors
        // only when the probe itself succeeded.
        let revoke = self.revoke_probe_pool_access(e, &granted);
        let disable = self.disable_probe_peer_access(e, &enabled);
        probe?;
        revoke?;
        disable?;
        Ok(())
    }

    fn init_peer_probe_geometry(
        &self,
        e: &Engine,
        n_embd: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.peer_probe || !self.cross_any {
            return Ok(());
        }
        let bytes = n_embd
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("PP boundary-slot byte count overflows for n_embd={n_embd}"))?;
        let result = self.peer_probe_geometry.get_or_init(|| {
            let probe = if self.host_bounce_active() {
                self.run_host_bounce_production_probe(e, n_embd)
            } else {
                self.run_production_peer_probe(&self.peer_capable, false, n_embd)
            };
            let restore = e.ctx().bind_to_thread();
            match (probe, restore) {
                (Ok(()), Ok(())) => Ok(bytes),
                (Err(err), _) => Err(err.to_string()),
                (_, Err(err)) => Err(err.to_string()),
            }
        });
        let probed = result
            .as_ref()
            .map_err(|err| -> Box<dyn std::error::Error> { err.clone().into() })?;
        if *probed != bytes {
            return Err(format!(
                "peer probe initialized for boundary-slot bytes={probed} but model requests \
                 bytes={bytes}; one PP runtime supports one model geometry per process"
            )
            .into());
        }
        Ok(())
    }

    fn init_host_bounce_staging(
        &self,
        e: &Engine,
        n_embd: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.cross_any {
            return Ok(());
        }
        e.ctx().bind_to_thread()?;
        let result = self.bounce.get_or_init(|| {
            HostBounceRt::new(n_embd, &self.boundaries)
                .map(|rt| {
                    let bytes = rt.capacity * std::mem::size_of::<f32>();
                    eprintln!(
                        "[pp] host-bounce staging ready: n_embd={n_embd} max_tokens={} \
                         slot_bytes={bytes} slots_per_cross_boundary=2",
                        crate::cache::PRIME_CHUNK_MAX_TOKENS,
                    );
                    rt
                })
                .map_err(|err| err.to_string())
        });
        let bounce = result
            .as_ref()
            .map_err(|err| -> Box<dyn std::error::Error> { err.clone().into() })?;
        if bounce.n_embd != n_embd {
            return Err(format!(
                "host-bounce runtime initialized for n_embd={} but model requests n_embd={n_embd}; \
                 one PP runtime supports one model geometry per process",
                bounce.n_embd,
            )
            .into());
        }
        Ok(())
    }

    /// Exercise the newly armed staging through the real D2H/event/H2D boundary path before the
    /// live transport latch can observe it. One row per cross boundary is enough to validate the
    /// pinned capacity, event ordering, contexts, and byte continuity without touching peer DMA.
    fn validate_host_bounce_staging(
        &self,
        e: &Engine,
        n_embd: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = n_embd
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("host-bounce validation byte count overflows for n_embd={n_embd}")
            })?;
        for boundary_idx in 0..self.stages.len() - 1 {
            if !self.boundaries[boundary_idx].cross {
                continue;
            }
            let src_stage = boundary_idx;
            let dst_stage = boundary_idx + 1;
            let probe_boundary = self.new_peer_probe_boundary(src_stage, dst_stage)?;
            let path = BoundaryPath {
                boundary: boundary_idx,
                src_stage,
                dst_stage,
                transport: BoundaryTransport::HostBounce,
            };
            let expected = peer_probe_pattern(
                bytes,
                boundary_idx,
                self.stages[src_stage].dev,
                self.stages[dst_stage].dev,
            );
            let readback =
                self.production_probe_readback(path, &probe_boundary, &expected, n_embd, 0);
            let clear = self.clear_peer_probe_boundary(&probe_boundary, src_stage, dst_stage);
            let readback = readback?;
            clear?;
            let mismatches = peer_probe_mismatch_count(&expected, &readback);
            if mismatches > 0 {
                return Err(format!(
                    "runtime host-bounce staging validation FAILED: boundary={boundary_idx} \
                     bytes={bytes} mismatches={mismatches}"
                )
                .into());
            }
        }
        e.ctx().bind_to_thread()?;
        eprintln!(
            "[pp] runtime host-bounce staging validation PASS: row_bytes={bytes} \
             cross_boundaries={}",
            self.boundaries
                .iter()
                .filter(|boundary| boundary.cross)
                .count(),
        );
        Ok(())
    }

    fn arm_runtime_host_bounce(
        &self,
        e: &Engine,
        row_bytes: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if row_bytes == 0 || row_bytes % std::mem::size_of::<f32>() != 0 {
            return Err(format!(
                "runtime host-bounce cannot recover n_embd from row_bytes={row_bytes}"
            )
            .into());
        }
        let n_embd = row_bytes / std::mem::size_of::<f32>();
        self.init_host_bounce_staging(e, n_embd)?;
        self.validate_host_bounce_staging(e, n_embd)
    }

    /// Finish boot-time transport setup from the authoritative model width. This runs the
    /// production `BoundarySlot` ladder at 1/8/16/`PRIME_CHUNK_MAX_TOKENS` `[n_embd] f32` rows
    /// once, then allocates host-bounce slots when selected. The loader calls it before uploading
    /// the first model weight; `new_cache` repeats the call as an idempotent guard before the first
    /// forward.
    pub fn init_boundary_transport(
        &self,
        e: &Engine,
        n_embd: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if PEER_RUNTIME_PROBE_FAILED.load(Ordering::Acquire)
            && !PEER_RUNTIME_HOST_BOUNCE.load(Ordering::Acquire)
        {
            return Err(
                "PP runtime peer byte-integrity probe previously failed; refusing native P2P \
                 reuse because runtime host-bounce staging could not be armed"
                    .into(),
            );
        }
        self.init_peer_probe_geometry(e, n_embd)?;
        if !self.host_bounce_active() || !self.cross_any {
            return Ok(());
        }
        self.init_host_bounce_staging(e, n_embd)
    }

    /// Run one due peer re-probe at a scheduler boundary on the CUDA owner thread. Each width has
    /// an independent copy-count deadline: an idle-only rung can remain pending while later cheap
    /// rungs keep running. The probe synchronizes the stage streams it exercises; no background
    /// thread touches CUDA.
    fn service_runtime_peer_probe(
        &self,
        e: &Engine,
        scheduler_idle: bool,
        probe_allowed: bool,
    ) -> Result<RuntimePeerProbeStatus, Box<dyn std::error::Error>> {
        if !self.peer_probe || !self.cross_any || self.host_bounce_active() {
            return Ok(RuntimePeerProbeStatus::NotRun);
        }
        if PEER_RUNTIME_PROBE_FAILED.load(Ordering::Acquire) {
            return Err(
                "PP runtime peer byte-integrity probe previously failed; native P2P is latched off"
                    .into(),
            );
        }
        let row_bytes = match self.peer_probe_geometry.get() {
            Some(Ok(bytes)) => *bytes,
            _ => return Ok(RuntimePeerProbeStatus::NotRun),
        };

        let copies = PEER_BOUNDARY_COPIES.load(Ordering::Relaxed);
        let (width_index, tokens) = loop {
            let next_probe_copy = std::array::from_fn(|width_index| {
                PEER_RUNTIME_NEXT_PROBE_COPY[width_index].load(Ordering::Relaxed)
            });
            let measured_cost_ns = std::array::from_fn(|width_index| {
                PEER_RUNTIME_PROBE_MAX_COST_NS[width_index].load(Ordering::Relaxed)
            });
            let Some(candidate) = runtime_peer_probe_candidate(
                copies,
                next_probe_copy,
                measured_cost_ns,
                scheduler_idle,
            ) else {
                return Ok(RuntimePeerProbeStatus::NotRun);
            };
            // A mismatch immediately revokes native peer access before validated host bounce is
            // published. Live speculative sessions still dereference token/position state through
            // UVA outside the bounced boundary, so the worker may defer a runnable cheap rung until
            // those sessions retire. Do not consume its deadline or completed-probe counter.
            if !probe_allowed {
                return Ok(RuntimePeerProbeStatus::Deferred);
            }
            let due = next_probe_copy[candidate.0];
            let next = runtime_peer_probe_next_copy(due, copies);
            if PEER_RUNTIME_NEXT_PROBE_COPY[candidate.0]
                .compare_exchange(due, next, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break candidate;
            }
        };
        let probe_index = PEER_RUNTIME_PROBES.fetch_add(1, Ordering::Relaxed);
        let probe_bytes = row_bytes.checked_mul(tokens);
        let scheduler_class = if scheduler_idle { "idle" } else { "busy" };
        let label = format!("runtime-{scheduler_class}-{tokens}tok");
        let started = std::time::Instant::now();
        let probe = match probe_bytes {
            Some(bytes) => {
                run_peer_probe_pass(&self.stages, &self.peer_capable, false, &label, bytes)
            }
            None => Err(format!(
                "PP runtime peer probe byte count overflows for row_bytes={row_bytes} \
                 tokens={tokens}"
            )
            .into()),
        };
        let restore = e.ctx().bind_to_thread();
        let elapsed_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let previous_max =
            PEER_RUNTIME_PROBE_MAX_COST_NS[width_index].fetch_max(elapsed_ns, Ordering::Relaxed);
        let verdict = match (probe, restore) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(err), _) => Err(err.to_string()),
            (_, Err(err)) => Err(err.to_string()),
        };
        if let Err(err) = verdict {
            PEER_RUNTIME_PROBE_FAILURES.fetch_add(1, Ordering::Relaxed);
            let arm = latch_runtime_host_bounce(
                &PEER_RUNTIME_PROBE_FAILED,
                &PEER_RUNTIME_HOST_BOUNCE,
                || {
                    self.arm_runtime_host_bounce(e, row_bytes)
                        .map_err(|arm_err| arm_err.to_string())
                },
            );
            if let Err(arm_err) = arm {
                let message = format!(
                    "PP runtime peer byte-integrity re-probe FAILED after \
                     boundary_copies={copies} rung={}/{} tokens={tokens}: {err}; native P2P is \
                     latched off and host-bounce staging could not be armed: {arm_err}",
                    width_index + 1,
                    PEER_PROBE_TOKEN_WIDTHS.len(),
                );
                eprintln!("[pp] SECURITY RED: {message}; worker must stop");
                return Err(message.into());
            }
            eprintln!(
                "[pp] SECURITY RED: PP runtime peer byte-integrity re-probe FAILED after \
                 boundary_copies={copies} rung={}/{} tokens={tokens}: {err}; native P2P is \
                 latched off and the live transport DEGRADED to validated host bounce for the \
                 remainder of this process",
                width_index + 1,
                PEER_PROBE_TOKEN_WIDTHS.len(),
            );
            return Ok(RuntimePeerProbeStatus::DegradedToHostBounce);
        }
        if width_index + 1 != PEER_PROBE_TOKEN_WIDTHS.len()
            && previous_max <= PEER_RUNTIME_PROBE_BUDGET_NS
            && elapsed_ns > PEER_RUNTIME_PROBE_BUDGET_NS
        {
            eprintln!(
                "[pp] runtime peer re-probe rung exceeded the {:.3}ms owner-thread budget: \
                 tokens={tokens} measured_ms={:.3}; future runs are idle-only",
                PEER_RUNTIME_PROBE_BUDGET_NS as f64 / 1e6,
                elapsed_ns as f64 / 1e6,
            );
        }
        eprintln!(
            "[pp] runtime peer byte-integrity re-probe PASS: \
             boundary_copies={copies} interval_copies={PEER_RUNTIME_PROBE_INTERVAL_COPIES} \
             rung={}/{} tokens={tokens} bytes={} probe_index={probe_index} elapsed_ms={:.3} \
             scheduler_idle={scheduler_idle}",
            width_index + 1,
            PEER_PROBE_TOKEN_WIDTHS.len(),
            probe_bytes.unwrap(),
            elapsed_ns as f64 / 1e6,
        );
        Ok(RuntimePeerProbeStatus::Passed)
    }

    fn bounce_rt(&self) -> Result<&HostBounceRt, Box<dyn std::error::Error>> {
        self.bounce
            .get()
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                "MEMRA_PP_HOST_BOUNCE=1 staging was not initialized from model geometry".into()
            })?
            .as_ref()
            .map_err(|err| -> Box<dyn std::error::Error> { err.clone().into() })
    }

    /// The engine a stage's subgraph must run through: the primary engine when the stage
    /// lives on the primary device, else the stage's own (remote-context) engine.
    pub fn engine<'a>(&'a self, s: usize, primary: &'a Engine) -> &'a Engine {
        self.stages[s].engine.as_ref().unwrap_or(primary)
    }

    /// Bind this OS thread to stage `s`'s CUDA context before issuing work there.
    pub fn bind_stage(&self, s: usize) -> Result<(), Box<dyn std::error::Error>> {
        self.stages[s].ctx.bind_to_thread()?;
        Ok(())
    }

    /// Enter stage `s`: until the guard drops, every engine op on this thread launches on
    /// the stage's stream (memra_runtime ambient-stream override).
    pub fn enter(&self, s: usize) -> memra_runtime::StreamOverride {
        memra_runtime::push_stream_override(
            self.stages[s].stream.clone(),
            self.stages[s].blas.clone(),
        )
    }

    /// Allocate/grow BOTH slots for a boundary before pipelined issue starts. `tx()` can
    /// grow a slot lazily, but first-use ordering requires synchronizing the RX stream
    /// after that allocation. If slot 1 first grows after stage 1 of chunk N has already
    /// been queued, that sync drains chunk N and erases the only overlap in a two-chunk
    /// prime. Prewarming both slots pays the same one-time sync before either stage starts.
    pub fn prepare_overlap_slots(
        &self,
        b: usize,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bd = &self.boundaries[b];
        let s_rx = &self.stages[b + 1].stream;
        let mut grew = false;
        for sl in &bd.slots {
            let mut guard = sl.buf.lock().unwrap();
            if guard.as_ref().map(|bf| bf.len() < n).unwrap_or(true) {
                *guard = Some(s_rx.alloc_zeros::<f32>(n)?);
                grew = true;
            }
        }
        if grew {
            s_rx.synchronize()?;
        }
        Ok(())
    }

    /// Boundary TX at boundary `b` (call within the stage-`b` scope; `x` = the
    /// materialized [n] residual): wait for the slot's previous RX (write-after-read
    /// guard), copy `x` into the slot's persistent buffer via the boundary's transport on
    /// stage-b's stream (the owning-stream/publication law), record ev_tx. Returns the
    /// slot index for the paired rx().
    ///
    /// `n` is the PAYLOAD ELEMENT COUNT, not a fixed model constant: the eager arm passes
    /// `n_embd` (one row), the batched arm passes `b_n * n_embd` (B stacked rows, the
    /// [B, n_embd] boundary). The slot buffer is GROW-ONLY and the transport moves exactly
    /// the first `n` elements — batched serving changes B every tick (chunk fill), and a
    /// realloc-on-every-size-change would host-sync the RX stream per width change (see the
    /// SLOT FIRST-USE ORDERING note below for why each allocation needs that sync). Growing
    /// to the high-water mark makes the syncs O(distinct widths) instead of O(width changes).
    pub fn tx(
        &self,
        b: usize,
        x: &CudaSlice<f32>,
        n: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        assert_eq!(x.len(), n, "pp tx: residual length mismatch");
        let bd = &self.boundaries[b];
        let slot_idx = if pp2_overlap() {
            bd.step.fetch_add(1, Ordering::Relaxed) % 2
        } else {
            0
        };
        self.tx_slot(b, x, n, slot_idx)
    }

    /// Pipelined boundary TX: always alternate the shared double-buffer slots, independent
    /// of the decode-side `MEMRA_PP_OVERLAP` experiment flag. The boundary-local atomic
    /// keeps concurrent callers on one slot sequence rather than each restarting at A.
    pub fn tx_pipelined(
        &self,
        b: usize,
        x: &CudaSlice<f32>,
        n: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        assert_eq!(x.len(), n, "pp tx: residual length mismatch");
        let slot_idx = self.boundaries[b].step.fetch_add(1, Ordering::Relaxed) % 2;
        self.tx_slot(b, x, n, slot_idx)
    }

    fn tx_slot(
        &self,
        b: usize,
        x: &CudaSlice<f32>,
        n: usize,
        slot_idx: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let bd = &self.boundaries[b];
        let path = BoundaryPath {
            boundary: b,
            src_stage: b,
            dst_stage: b + 1,
            transport: boundary_transport(bd.cross, self.host_bounce_active()),
        };
        let copied_slot = self.tx_slot_path(path, bd, x, n, slot_idx)?;
        if path.transport == BoundaryTransport::Peer {
            PEER_BOUNDARY_COPIES.fetch_add(1, Ordering::Relaxed);
        }
        Ok(copied_slot)
    }

    fn tx_slot_path(
        &self,
        path: BoundaryPath,
        bd: &BoundaryRt,
        x: &CudaSlice<f32>,
        n: usize,
        slot_idx: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        debug_assert!(slot_idx < 2);
        let sl = &bd.slots[slot_idx];
        let s_tx = &self.stages[path.src_stage].stream;
        s_tx.wait(&sl.ev_rx)?;
        let mut guard = sl.buf.lock().unwrap();
        if guard.as_ref().map(|bf| bf.len() < n).unwrap_or(true) {
            // allocated on the RX stage's stream: the buffer lives on the RX device.
            let s_rx = &self.stages[path.dst_stage].stream;
            *guard = Some(s_rx.alloc_zeros::<f32>(n)?);
            // SLOT FIRST-USE ORDERING (2026-08-02 pipelined-gate find): the lazy alloc's
            // pool-alloc + memset enqueue on the RX stream; the TX copy below issues on
            // the TX stream, and on a slot's FIRST use ev_rx has never been recorded —
            // nothing orders them. With >=2 tokens in flight the RX stream is still busy
            // with the previous token, the memset lands AFTER the TX copy, and the
            // boundary residual is zeroed (window=1 passed, window>=2 failed at the
            // slot-1 first-use step; -overlap arms passed because the synchronous serial
            // arm pre-warmed both slots). Host-sync the RX stream once per slot
            // allocation — at most 2*(N-1) one-time syncs per process, all during prime.
            s_rx.synchronize()?;
        }
        let buf = guard.as_mut().unwrap();
        match path.transport {
            BoundaryTransport::Local => s_tx.memcpy_dtod(x, buf)?,
            BoundaryTransport::HostBounce => {
                debug_assert_eq!(path.src_stage, path.boundary);
                debug_assert_eq!(path.dst_stage, path.boundary + 1);
                let bounce = self.bounce_rt()?;
                if n > bounce.capacity {
                    return Err(format!(
                        "pp host-bounce payload {n} exceeds geometry-sized capacity {} \
                         (n_embd={}, max prime tokens={})",
                        bounce.capacity,
                        bounce.n_embd,
                        crate::cache::PRIME_CHUNK_MAX_TOKENS,
                    )
                    .into());
                }
                let mut host = bounce.slot(path.boundary, slot_idx)?.lock().unwrap();
                // D2H is issued on the producing stage's stream. ev_tx below publishes the
                // completed host bytes to the receiving stream; the exact prefix avoids moving
                // a full 64 MiB slot for a one-row decode, and no peer pointer is formed here.
                s_tx.memcpy_dtoh(x, host.prefix_mut(n))?;
            }
            BoundaryTransport::Peer => {
                // cudaMemcpyPeerAsync (M0: 2.8x NCCL at PP activation sizes), issued on the
                // publishing TX stream with explicit src/dst contexts.
                use cudarc::driver::{DevicePtr, DevicePtrMut};
                let (sp, _g0) = x.device_ptr(s_tx);
                let (dp, _g1) = buf.device_ptr_mut(s_tx);
                self.stages[path.src_stage].ctx.bind_to_thread()?;
                unsafe {
                    cudarc::driver::result::memcpy_peer_async(
                        self.stages[path.dst_stage].ctx.cu_ctx(),
                        dp,
                        self.stages[path.src_stage].ctx.cu_ctx(),
                        sp,
                        n * std::mem::size_of::<f32>(),
                        s_tx.cu_stream(),
                    )?;
                }
            }
        }
        sl.ev_tx.record(s_tx)?;
        Ok(slot_idx)
    }

    /// Boundary RX at boundary `b` (call within the stage-`b+1` scope): wait on the slot's
    /// ev_tx, copy the boundary buffer into a fresh working buffer (dtod on the RX stream —
    /// local on the RX device in both transports), record ev_rx. The returned buffer is
    /// RX-stage-owned: allocated, consumed, and eventually freed on that stage's stream.
    pub fn rx(
        &self,
        b: usize,
        slot_idx: usize,
        n: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let bd = &self.boundaries[b];
        let path = BoundaryPath {
            boundary: b,
            src_stage: b,
            dst_stage: b + 1,
            transport: boundary_transport(bd.cross, self.host_bounce_active()),
        };
        self.rx_slot_path(path, bd, slot_idx, n)
    }

    fn rx_slot_path(
        &self,
        path: BoundaryPath,
        bd: &BoundaryRt,
        slot_idx: usize,
        n: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let sl = &bd.slots[slot_idx];
        let s_rx = &self.stages[path.dst_stage].stream;
        s_rx.wait(&sl.ev_tx)?;
        let mut guard = sl.buf.lock().unwrap();
        let buf = guard.as_mut().expect("pp rx before tx");
        assert!(
            buf.len() >= n,
            "pp rx: slot holds {} < requested {n}",
            buf.len()
        );
        if path.transport == BoundaryTransport::HostBounce {
            debug_assert_eq!(path.src_stage, path.boundary);
            debug_assert_eq!(path.dst_stage, path.boundary + 1);
            let bounce = self.bounce_rt()?;
            let host = bounce.slot(path.boundary, slot_idx)?.lock().unwrap();
            let mut dst = buf.slice_mut(0..n);
            // The destination stream already waits ev_tx, so this H2D cannot observe the
            // staging slot before the source stream's D2H completes.
            s_rx.memcpy_htod(host.prefix(n), &mut dst)?;
        }
        // uninit working buffer (fully overwritten by the copy), allocated explicitly on
        // the stage stream so rx() is correct even outside an enter() scope.
        let mut work = unsafe { s_rx.alloc::<f32>(n)? };
        // Slice the slot to the payload: the buffer is grow-only (see tx), so at a narrower
        // width it is LONGER than `work` and cudarc's memcpy_dtod (dst.len() >= src.len())
        // would assert. The paired tx wrote exactly these first n elements.
        s_rx.memcpy_dtod(&buf.slice(0..n), &mut work)?;
        sl.ev_rx.record(s_rx)?;
        Ok(work)
    }

    /// PUBLISH a DEVICE-RESIDENT result off the last stage to the caller's stream
    /// (lane/pp2-spec 2026-08-06).
    ///
    /// Every ppN body before this one returned HOST values — `decode_step_h_ppn` and
    /// `decode_step_batch_ppn` both `dtoh` inside the last-stage scope, and a dtoh on the
    /// producing stream is self-ordering. The verify trunk is the FIRST ppN body whose
    /// contract is device-resident output (`decode_step_t_h_emb_dev` exists precisely so the
    /// accept walk argmaxes on-device instead of moving T x n_vocab f32 per round), and
    /// device slices carry no stream affinity: the caller resumes on the PRIMARY stream and
    /// dereferences buffers whose producing kernels are still queued on the last stage's
    /// stream. Nothing orders them.
    ///
    /// Why this only ever failed on ONE device: with stages on separate devices the caller's
    /// first touch is a cross-device copy that the driver orders against the source context,
    /// and the readback path syncs. Two streams on the SAME device genuinely overlap, so the
    /// primary stream reads a buffer whose matmul has not run — nondeterministic garbage
    /// (measured: NaN, 3155.677, and 2.87e-5 where the reference had -2.0048926), and it
    /// poisons the NEXT arm in the same process because the corrupted KV persists. This is
    /// the same class as the SLOT FIRST-USE ORDERING find above, one level up: there the
    /// unordered pair was alloc-memset vs TX copy, here it is stage-N compute vs the
    /// caller's consumer.
    ///
    /// Fix = the boundary law applied to the exit: record an event on the producing stage
    /// stream, make the caller's stream wait on it. Event-wait, not a device sync, so the
    /// stage streams keep running for the deferred-readback arm. Call INSIDE the last-stage
    /// scope, after the last enqueue, with the caller's (pre-`enter`) stream.
    pub fn publish_to(
        &self,
        s: usize,
        dst: &Arc<CudaStream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let st = &self.stages[s];
        // Same stream (STREAMS=0 rollback, or a caller already on the stage stream): the
        // stream orders itself; recording+waiting would be a no-op with a stray event.
        if Arc::ptr_eq(&st.stream, dst) {
            return Ok(());
        }
        let ev = st.ctx.new_event(None)?;
        ev.record(&st.stream)?;
        dst.wait(&ev)?;
        Ok(())
    }

    /// REVERSE PUBLICATION (#87 root cause, lane/pp2spec-crash 2026-08-07): order every
    /// STAGE stream behind the CALLER's stream — the mirror of `publish_to`.
    ///
    /// `publish_to` orders caller READS behind stage COMPUTE. Nothing ordered the other
    /// direction: buffers ALLOCATED on a stage stream (the verify's returned logits/hidden,
    /// the VerifyCkpt stashes) are CONSUMED by kernels the caller enqueues on the PRIMARY
    /// stream, and when they drop, cudarc enqueues `free_async` on the ALLOCATING (stage)
    /// stream. With event tracking elided (the decode-path default) the drop carries no
    /// read-guard, so the pool can hand the block to the NEXT stage-stream allocation and
    /// its writes overwrite memory the queued primary-stream consumer has not read yet.
    /// Measured (research/pp2spec-crash-20260807): the spec round-seed read 13/4096 NaN =
    /// the uninitialized-bits signature (P(NaN|random u32) ~ 1/256), clean by host re-read
    /// time — a read-before-write race, fatal via the argmax-sentinel -> embed_gather MMU
    /// fault, and gated on c>=2 because a backed-up primary stream widens the window.
    ///
    /// Fix law: before a ppN body enqueues NEW stage-stream work (allocations that may
    /// reuse freed blocks), every stage stream waits the caller's stream at its current
    /// point. All primary consumers of the previous round's stage-allocated buffers are
    /// enqueued by then (single host thread), so reuse-writes land strictly after them.
    /// Call at ppN-body ENTRY with the pre-`enter` caller stream. Door-shut configs never
    /// build a PpNRt, so single-card behavior is untouched.
    pub fn fence_stages_behind(
        &self,
        src: &Arc<CudaStream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ev = src.context().new_event(None)?;
        ev.record(src)?;
        for st in &self.stages {
            if Arc::ptr_eq(&st.stream, src) {
                continue;
            }
            st.stream.wait(&ev)?;
        }
        Ok(())
    }

    /// Deferred readback: record a fresh completion event on the LAST stage's stream
    /// (call after the step's logits matmul has been enqueued there).
    pub fn record_done(&self) -> Result<CudaEvent, Box<dyn std::error::Error>> {
        let last = &self.stages[self.stages.len() - 1];
        let ev = last.ctx.new_event(None)?;
        ev.record(&last.stream)?;
        Ok(ev)
    }

    /// The dedicated readback stream (last stage's context).
    pub fn readback_stream(&self) -> &Arc<CudaStream> {
        &self.readback
    }
}

/// Service a due runtime peer probe without constructing a PP runtime on door-shut placements.
/// Must be called by the CUDA owner thread at a scheduling boundary.
pub fn service_runtime_peer_probe(
    e: &Engine,
    scheduler_idle: bool,
    probe_allowed: bool,
) -> Result<RuntimePeerProbeStatus, Box<dyn std::error::Error>> {
    let Some(rt) = RTN.get() else {
        return Ok(RuntimePeerProbeStatus::NotRun);
    };
    let rt = rt
        .as_ref()
        .map_err(|err| -> Box<dyn std::error::Error> { err.clone().into() })?;
    rt.service_runtime_peer_probe(e, scheduler_idle, probe_allowed)
}

/// M2 increment 3: a step's logits, still device-resident on the LAST stage. `wait()`
/// orders the readback stream behind the step's completion event, copies, and syncs —
/// tokens enqueued after this step keep running on the stage streams while the caller
/// drains token t. Dropping without waiting is safe (buffers free stream-ordered).
pub struct PendingLogits {
    logits: CudaSlice<f32>,
    ev: CudaEvent,
    rb: Arc<CudaStream>,
}

impl PendingLogits {
    pub fn new(logits: CudaSlice<f32>, ev: CudaEvent, rb: Arc<CudaStream>) -> Self {
        PendingLogits { logits, ev, rb }
    }

    /// Blocks until this step's logits are computed, returns them host-side. Only this
    /// step's work is waited on (event-ordered) — NOT later tokens already enqueued on
    /// the stage streams.
    pub fn wait(self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.rb.wait(&self.ev)?;
        let host = self.rb.clone_dtoh(&self.logits)?;
        self.rb.synchronize()?;
        // logits drop AFTER the sync: the D2H has fully completed, so the stream-ordered
        // free on the compute stream cannot race the copy.
        Ok(host)
    }
}

/// Bring up the PP transport while model geometry is known but before model weights upload.
/// Door-shut and placement-free loads remain untouched.
pub fn init_model_transport(
    e: &Engine,
    cfg: &memra_gguf::config::ModelConfig,
    n_trunk: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if pp2_streams_off() || pp2_devices_env().is_none() || pp_cuts(n_trunk).is_none() {
        return Ok(());
    }
    PpNRt::get(e)?.init_boundary_transport(e, cfg.n_embd as usize)
}

/// Stage-owned cache allocation door: when the ppN door is open AND `MEMRA_PP_DEVICES`
/// is set (placement plumbing), each layer's cache is allocated by its OWNING stage's
/// engine — on one device this is byte-for-byte today's allocation (gated); cross-device
/// it puts each stage's KV on that stage's HBM. Door shut or devices unset: plain
/// `Cache::new` (zero behavior change). Trailing MTP/NextN layers (beyond the trunk)
/// map to the LAST stage.
pub fn new_cache(
    e: &Engine,
    cfg: &memra_gguf::config::ModelConfig,
    max_ctx: usize,
) -> Result<crate::cache::Cache, Box<dyn std::error::Error>> {
    new_cache_inner(e, cfg, None, max_ctx)
}

pub fn new_cache_planned(
    e: &Engine,
    cfg: &memra_gguf::config::ModelConfig,
    plan: &memra_gguf::model_plan::ModelPlan,
    max_ctx: usize,
) -> Result<crate::cache::Cache, Box<dyn std::error::Error>> {
    new_cache_inner(e, cfg, Some(plan), max_ctx)
}

fn new_cache_inner(
    e: &Engine,
    cfg: &memra_gguf::config::ModelConfig,
    plan: Option<&memra_gguf::model_plan::ModelPlan>,
    max_ctx: usize,
) -> Result<crate::cache::Cache, Box<dyn std::error::Error>> {
    let n_trunk = (cfg.n_layer - cfg.nextn_predict_layers) as usize;
    if let Some(fence) = pp_cuts(n_trunk) {
        if pp2_devices_env().is_some() && !pp2_streams_off() {
            let rt = PpNRt::get(e)?;
            rt.init_boundary_transport(e, cfg.n_embd as usize)?;
            let n_st = fence.len() - 1;
            assert_eq!(
                rt.n_stages(),
                n_st,
                "PpNRt stage count {} != fence stages {n_st}",
                rt.n_stages()
            );
            // #87 REVERSE PUBLICATION at ADMISSION (lane/pp2spec-crash): this is the one
            // stage-stream allocation site OUTSIDE the ppN step bodies — a NEW session's
            // KV alloc_zeros enqueue on the STAGE streams, and their pool blocks can be
            // reuse of buffers freed from ANOTHER session's in-flight verify whose
            // primary-stream reads are still queued (the c=2 residual: exactly one trap
            // per admission collision, round 0, after the step-body fences landed).
            // Order the stage streams behind the caller before the memsets can clobber.
            // Anatomy: `PpNRt::fence_stages_behind`.
            rt.fence_stages_behind(&e.stream())?;
            let devs: Vec<&dyn memra_kv::KvDev> = (0..n_st)
                .map(|s| rt.engine(s, e) as &dyn memra_kv::KvDev)
                .collect();
            let cache = match plan {
                Some(plan) => {
                    crate::cache::Cache::new_ppn_planned(&devs, &fence, cfg, plan, max_ctx)?
                }
                None => crate::cache::Cache::new_ppn(&devs, &fence, cfg, max_ctx)?,
            };
            sync_stages_after_load(e, n_trunk)?;
            return Ok(cache);
        }
        if !pp2_streams_off() {
            // CACHE BIRTH BARRIER (2026-08-02 pipelined-arm residual race): with the door
            // open but no device placement, Cache::new's alloc_zeros memsets enqueue on
            // the PRIMARY worker stream while the first KV appends / recurrent-state
            // reads run on the per-stage streams — no event orders them, and under
            // deferred readback the stage streams are hot immediately (a memset tail
            // can zero an already-appended KV row; intermittent, ~1-in-3 gate FAIL).
            // One context-sync per cache creation kills the class.
            let cache = match plan {
                Some(plan) => crate::cache::Cache::new_planned(e, cfg, plan, max_ctx)?,
                None => crate::cache::Cache::new(e, cfg, max_ctx)?,
            };
            sync_stages_after_load(e, n_trunk)?;
            return Ok(cache);
        }
    }
    match plan {
        Some(plan) => crate::cache::Cache::new_planned(e, cfg, plan, max_ctx),
        None => crate::cache::Cache::new(e, cfg, max_ctx),
    }
}

/// M2 increment 2 LOAD BARRIER: weight uploads and decode-mirror builds enqueue on the
/// loading engines' WORKER streams; the first consumer launches on a DIFFERENT stream
/// with no load->decode event — the door-off reference walk on the primary worker
/// stream (sharded load: remote builds still in flight), or a fresh per-stage stream.
/// The 2026-08-02 gate finds (n2-dev01 step-0 168k-logit graze; split5 ref=0.0 head —
/// a half-built rp4 mirror — poisoning step-0 KV and every later step): one
/// context-wide synchronize per stage at load end kills the class. No-op when the door
/// is shut at load (single-stream load+decode is ordered by the stream itself).
pub fn sync_stages_after_load(
    e: &Engine,
    n_trunk: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if pp2_streams_off() || pp_cuts(n_trunk).is_none() {
        return Ok(());
    }
    let rt = PpNRt::get(e)?;
    for s in 0..rt.n_stages() {
        rt.stages[s].ctx.bind_to_thread()?;
        unsafe {
            cudarc::driver::sys::cuCtxSynchronize().result()?;
        }
    }
    e.ctx().bind_to_thread()?;
    unsafe {
        cudarc::driver::sys::cuCtxSynchronize().result()?;
    }
    Ok(())
}

/// M2 increment 2 (weight sharding): the engine that should UPLOAD layer `il`'s weights
/// (and build its decode mirrors) — the owning stage's engine when the door is open with
/// device placement and sharding not rolled back; else the primary. `il >= n_trunk`
/// (MTP/NextN blocks) maps to the last stage. The head (output_norm + lm head) belongs
/// to the last trunk layer's stage — call with `il = n_trunk - 1`.
pub fn layer_engine<'a>(
    e: &'a Engine,
    n_trunk: usize,
    il: usize,
) -> Result<&'a Engine, Box<dyn std::error::Error>> {
    if pp_shard_off() || pp2_devices_env().is_none() || pp2_streams_off() {
        return Ok(e);
    }
    let Some(fence) = pp_cuts(n_trunk) else {
        return Ok(e);
    };
    let rt = PpNRt::get(e)?;
    let s = stage_of(&fence, il.min(n_trunk - 1));
    Ok(rt.engine(s, e))
}

/// Restore a cache checkpoint through each layer's owning engine.
///
/// `source = None` is an in-place rewind: the target already owns the append-only KV bytes and
/// only its lengths plus recurrent state move back to the snapshot. `Some(source)` restores into
/// a freshly allocated larger cache: checkpoint-valid KV rows are copied from the parked cache,
/// rank-local TP sidecars are rebuilt through their model-owned runtimes, and recurrent state
/// always comes from the checkpoint's owned device copies.
///
/// This cannot use `Cache::rollback(e, ...)` under cross-device PP: a single primary engine is
/// not the owner of every stage's cache buffers. The rare rewind/grow boundary synchronizes open
/// PP contexts before publishing the restored cache to the next request.
pub fn restore_cache_checkpoint(
    e: &Engine,
    model: &crate::hybrid::HybridModel,
    source: Option<&crate::cache::Cache>,
    target: &mut crate::cache::Cache,
    snap: &crate::cache::CacheSnapshot,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = &model.cfg;
    let n = target.kv.len();
    if target.recur.len() != n
        || target.tp_kv.len() != n
        || snap.kv_len.len() != n
        || snap.tp_kv_len.len() != n
        || snap.conv.len() != n
        || snap.ssm.len() != n
        || source.is_some_and(|s| s.kv.len() != n || s.recur.len() != n || s.tp_kv.len() != n)
    {
        return Err("checkpoint cache layer-count mismatch".into());
    }
    if snap.pos > target.max_ctx {
        return Err(format!(
            "checkpoint pos {} exceeds target capacity {}",
            snap.pos, target.max_ctx,
        )
        .into());
    }

    let n_trunk = (cfg.n_layer - cfg.nextn_predict_layers) as usize;
    for il in 0..n {
        let owner = layer_engine(e, n_trunk, il)?;
        let src_kv = source.map(|s| &s.kv[il]);
        match (src_kv, target.kv[il].as_mut(), snap.kv_len[il]) {
            (Some(Some(src)), Some(dst), Some(len)) => {
                if len > src.len || len > target.max_ctx {
                    return Err(format!(
                        "checkpoint layer {il} len {len} exceeds source {} or target {}",
                        src.len, target.max_ctx,
                    )
                    .into());
                }
                if src.kv_dim_k != dst.kv_dim_k
                    || src.kv_dim_v != dst.kv_dim_v
                    || src.k_tok_bytes != dst.k_tok_bytes
                    || src.v_tok_bytes != dst.v_tok_bytes
                {
                    return Err(format!("checkpoint KV layout mismatch at layer {il}").into());
                }
                let kb = len * src.k_tok_bytes;
                let vb = len * src.v_tok_bytes;
                if kb > 0 {
                    owner.copy_u8_into(&mut dst.k, 0, &src.k, kb)?;
                }
                if vb > 0 {
                    owner.copy_u8_into(&mut dst.v, 0, &src.v, vb)?;
                }
                dst.len = len;
                owner.set_i32_one(&mut dst.len_d, len as i32)?;
            }
            (None, Some(dst), Some(len)) => {
                if len > dst.len || len > target.max_ctx {
                    return Err(format!(
                        "checkpoint layer {il} len {len} exceeds live {} or target {}",
                        dst.len, target.max_ctx,
                    )
                    .into());
                }
                dst.len = len;
                owner.set_i32_one(&mut dst.len_d, len as i32)?;
            }
            (Some(None), None, None) | (None, None, None) => {}
            _ => return Err(format!("checkpoint KV kind mismatch at layer {il}").into()),
        }

        match (source, snap.tp_kv_len[il]) {
            (None, Some(len)) => target.tp_kv[il]
                .as_mut()
                .ok_or_else(|| format!("checkpoint TP KV target is absent at layer {il}"))?
                .rewind_to(len)?,
            (None, None) => {
                if target.tp_kv[il].is_some() {
                    return Err(format!("checkpoint TP KV kind mismatch at layer {il}").into());
                }
            }
            (Some(src_cache), Some(len)) => {
                let src = src_cache.tp_kv[il]
                    .as_ref()
                    .ok_or_else(|| format!("checkpoint TP KV source is absent at layer {il}"))?;
                if target.tp_kv[il].is_some() {
                    return Err(
                        format!("checkpoint TP KV grow target is not fresh at layer {il}").into(),
                    );
                }
                let runtime = model.step_tp_runtime_for_layer(il).ok_or_else(|| {
                    format!("checkpoint TP KV layer {il} has no distributed runtime")
                })?;
                let grown = runtime.grow_tp_kv_cache(src, target.max_ctx, len)?;
                target.tp_kv[il] = Some(grown);
            }
            (Some(src_cache), None) => {
                if src_cache.tp_kv[il].is_some() || target.tp_kv[il].is_some() {
                    return Err(format!("checkpoint TP KV kind mismatch at layer {il}").into());
                }
            }
        }

        match (target.recur[il].as_mut(), &snap.conv[il], &snap.ssm[il]) {
            (Some(dst), Some(conv), Some(ssm)) => {
                if conv.len() != dst.conv_state.len() || ssm.len() != dst.ssm_state.len() {
                    return Err(
                        format!("checkpoint recurrent layout mismatch at layer {il}").into(),
                    );
                }
                owner.copy_into(&mut dst.conv_state, 0, conv, conv.len())?;
                owner.copy_into(&mut dst.ssm_state, 0, ssm, ssm.len())?;
            }
            (None, None, None) => {}
            _ => {
                return Err(format!("checkpoint recurrent kind mismatch at layer {il}").into());
            }
        }
    }
    target.pos = snap.pos;

    // Open PP uses per-stage streams/contexts; publish every restored plane before the caller
    // starts the next prime. Door-shut single-stream restores remain naturally ordered.
    sync_stages_after_load(e, n_trunk)?;
    if source.is_some() {
        // A grown cache replaces and drops the source immediately after this returns. Bound the
        // D2D copies first so an async-pool free cannot recycle a source plane prematurely.
        e.stream().synchronize()?;
    }
    Ok(())
}

#[cfg(test)]
mod host_bounce_tests {
    use super::{
        BoundaryTransport, DUAL_PP_HOST_BOUNCE_REFUSAL, DUAL_PP_SINGLE_SLOT_REFUSAL,
        PEER_PROBE_FIXED_BYTES, PEER_PROBE_REQUIRED_REFUSAL, PEER_PROBE_TOKEN_WIDTHS,
        PEER_RUNTIME_PROBE_BUDGET_NS, PEER_RUNTIME_PROBE_CYCLE_COPIES,
        PEER_RUNTIME_PROBE_DEFERRAL_BOUND_INTERVALS, PEER_RUNTIME_PROBE_INTERVAL_COPIES,
        PeerProbeDecision, PeerProbeStartupPolicy, boundary_transport, dual_pp_eligibility,
        dual_pp_timing_dropped, dual_pp_timing_snapshot, dual_pp_wave_mid, host_bounce_capacity,
        latch_runtime_host_bounce, peer_probe_bytes_to_f32, peer_probe_decision,
        peer_probe_f32_to_bytes, peer_probe_mismatch_count, peer_probe_pattern,
        peer_probe_startup_policy, publish_runtime_peer_probe_deferral,
        record_dual_pp_stage_result, runtime_peer_probe_candidate, runtime_peer_probe_next_copy,
    };

    // ---- 2026-08-11 default-flip safety regression (owner-ordered) ----------------------
    // All pure-resolution tests: no env mutation (parallel test threads share process env).

    #[test]
    fn flip_default_is_dual_auto_with_explicit_off_and_forced_seams() {
        use super::{DualPpMode, dual_pp_mode_resolve};
        assert_eq!(dual_pp_mode_resolve(None), DualPpMode::Auto);
        assert_eq!(dual_pp_mode_resolve(Some("0")), DualPpMode::Off);
        assert_eq!(dual_pp_mode_resolve(Some("1")), DualPpMode::Forced);
        // Any other value is not a silent third state: treat as the default.
        assert_eq!(dual_pp_mode_resolve(Some("2")), DualPpMode::Auto);
        assert_eq!(dual_pp_mode_resolve(Some("")), DualPpMode::Auto);
    }

    #[test]
    fn flip_overlap_follows_mode_and_one_flag_restores_preflip_serial() {
        use super::{DualPpMode, pp2_overlap_resolve};
        // Naked default = the re-gated dual arm: overlap ON.
        assert!(pp2_overlap_resolve(None, DualPpMode::Auto));
        // MEMRA_DUAL_PP=0 ALONE restores the exact pre-flip naked path (single-slot serial).
        assert!(!pp2_overlap_resolve(None, DualPpMode::Off));
        // The explicit pre-flip request keeps its binding single-slot refusal reachable.
        assert!(!pp2_overlap_resolve(None, DualPpMode::Forced));
        // Explicit values always win over the mode.
        for mode in [DualPpMode::Off, DualPpMode::Forced, DualPpMode::Auto] {
            assert!(pp2_overlap_resolve(Some("1"), mode));
            assert!(!pp2_overlap_resolve(Some("0"), mode));
        }
    }

    #[test]
    fn flip_auto_routes_only_the_regated_regime_and_degrades_serially_elsewhere() {
        use super::{DualPpMode, dual_pp_route};
        // The exact box1 re-gate regime: PP-2, double-slot, peer transport, B>=2.
        assert!(dual_pp_route(DualPpMode::Auto, 2, 2, true, false));
        assert!(dual_pp_route(DualPpMode::Auto, 17, 2, true, false));
        // Outside it, Auto must DEGRADE (serial PP-N walker), never refuse:
        assert!(!dual_pp_route(DualPpMode::Auto, 1, 2, true, false)); // no second wave
        assert!(!dual_pp_route(DualPpMode::Auto, 2, 3, true, false)); // naked PP-3 keeps serving
        assert!(!dual_pp_route(DualPpMode::Auto, 2, 2, false, false)); // single-slot boundary
        assert!(!dual_pp_route(DualPpMode::Auto, 2, 2, true, true)); // host-bounce escape hatch
        // Forced routes every B>=2 call into the dual body so the binding refusals fire loud.
        assert!(dual_pp_route(DualPpMode::Forced, 2, 3, false, true));
        assert!(!dual_pp_route(DualPpMode::Forced, 1, 2, true, false));
        // Off is the rollback seam: never dual.
        assert!(!dual_pp_route(DualPpMode::Off, 8, 2, true, false));
    }

    #[test]
    fn dual_pp_split_is_honest_at_one_and_ceil_first_afterward() {
        assert_eq!(dual_pp_wave_mid(1), None);
        assert_eq!(dual_pp_wave_mid(2), Some(1));
        assert_eq!(dual_pp_wave_mid(3), Some(2));
        assert_eq!(dual_pp_wave_mid(8), Some(4));
        assert_eq!(dual_pp_wave_mid(16), Some(8));
        assert_eq!(dual_pp_wave_mid(31), Some(16));
        assert_eq!(dual_pp_wave_mid(32), Some(16));
    }

    #[test]
    fn dual_pp_refuses_single_slot_and_non_pp2_shapes() {
        assert_eq!(
            dual_pp_eligibility(2, false, false),
            Err(DUAL_PP_SINGLE_SLOT_REFUSAL)
        );
        assert!(dual_pp_eligibility(2, true, false).is_ok());
        assert!(dual_pp_eligibility(3, true, false).is_err());
    }

    #[test]
    fn dual_pp_refuses_unvalidated_host_bounce_transport() {
        assert_eq!(
            dual_pp_eligibility(2, true, true),
            Err(DUAL_PP_HOST_BOUNCE_REFUSAL),
        );
    }

    #[test]
    fn dual_pp_timing_error_is_counted_without_recording_a_sample() {
        let dropped_before = dual_pp_timing_dropped();
        let (_, samples_before) = dual_pp_timing_snapshot();
        record_dual_pp_stage_result(0, Err::<f32, _>("CUDA_ERROR_NOT_READY"));
        let (_, samples_after) = dual_pp_timing_snapshot();
        assert_eq!(samples_after[0], samples_before[0]);
        assert!(dual_pp_timing_dropped() >= dropped_before + 1);
    }

    #[test]
    fn corrupted_peer_readback_fails_closed_unless_host_bounce_is_selected() {
        assert_eq!(
            PEER_PROBE_TOKEN_WIDTHS,
            [1, 8, 16, crate::cache::PRIME_CHUNK_MAX_TOKENS],
        );
        let largest_payload_bytes = PEER_PROBE_TOKEN_WIDTHS[3] * 4096 * std::mem::size_of::<f32>();
        assert_eq!(largest_payload_bytes, 64 * 1024 * 1024);
        assert!(largest_payload_bytes >= 1024 * 1024);
        let expected = peer_probe_pattern(PEER_PROBE_FIXED_BYTES, 2, 0, 1);
        assert_eq!(
            peer_probe_f32_to_bytes(&peer_probe_bytes_to_f32(&expected)),
            expected,
        );
        let mut corrupted = expected.clone();
        for offset in [0, 8_191, PEER_PROBE_FIXED_BYTES - 1] {
            corrupted[offset] ^= 0x5a;
        }

        assert_eq!(peer_probe_mismatch_count(&expected, &corrupted), 3);
        assert_eq!(
            peer_probe_decision(&expected, &corrupted, false),
            Err("3 mismatched byte(s)".to_string()),
        );
        assert_eq!(
            peer_probe_decision(&expected, &corrupted, true),
            Ok(PeerProbeDecision::ProceedWithHostBounce { mismatches: 3 }),
        );
    }

    #[test]
    fn probe_off_refusal_matrix_is_fail_closed_only_for_sharded_native_peer() {
        for probe_on in [false, true] {
            for sharded in [false, true] {
                for host_bounce in [false, true] {
                    let got = peer_probe_startup_policy(probe_on, sharded, host_bounce);
                    let expected = match (probe_on, sharded, host_bounce) {
                        (false, true, false) => Err(PEER_PROBE_REQUIRED_REFUSAL),
                        (false, true, true) => Ok(PeerProbeStartupPolicy::BypassedWithHostBounce),
                        _ => Ok(PeerProbeStartupPolicy::Allowed),
                    };
                    assert_eq!(
                        got, expected,
                        "probe_on={probe_on} sharded={sharded} host_bounce={host_bounce}",
                    );
                }
            }
        }
        assert!(PEER_PROBE_REQUIRED_REFUSAL.contains("MEMRA_PEER_PROBE=0"));
        assert!(PEER_PROBE_REQUIRED_REFUSAL.contains("MEMRA_PP_HOST_BOUNCE!=1"));
    }

    #[test]
    fn runtime_reprobe_keeps_cheap_deadlines_live_while_expensive_work_waits_for_idle() {
        let every = PEER_RUNTIME_PROBE_INTERVAL_COPIES;
        assert_eq!(PEER_RUNTIME_PROBE_CYCLE_COPIES, 4 * every);
        let mut next = [every, 2 * every, 3 * every, 4 * every];
        let measured_ns = [1_000_000, 2_000_000, 3_000_000, 0];

        assert_eq!(
            runtime_peer_probe_candidate(every - 1, next, measured_ns, false),
            None,
        );
        assert_eq!(
            runtime_peer_probe_candidate(every, next, measured_ns, false),
            Some((0, 1)),
        );

        // Pretend the three cheap deadlines completed. The maximum rung is due but must not run
        // on the interactive boundary.
        next[..3].copy_from_slice(&[5 * every, 6 * every, 7 * every]);
        assert_eq!(
            runtime_peer_probe_candidate(4 * every, next, measured_ns, false),
            None,
        );
        // Once the next cheap deadline arrives, it remains runnable even though the older max
        // deadline is still pending.
        assert_eq!(
            runtime_peer_probe_candidate(5 * every, next, measured_ns, false),
            Some((0, 1)),
        );
        // An idle boundary drains the oldest pending rung first.
        assert_eq!(
            runtime_peer_probe_candidate(5 * every, next, measured_ns, true),
            Some((3, crate::cache::PRIME_CHUNK_MAX_TOKENS)),
        );
    }

    #[test]
    fn runtime_reprobe_moves_any_measured_over_budget_rung_to_idle_only() {
        let every = PEER_RUNTIME_PROBE_INTERVAL_COPIES;
        let next = [u64::MAX, every, u64::MAX, u64::MAX];
        let mut measured_ns = [0; PEER_PROBE_TOKEN_WIDTHS.len()];
        measured_ns[1] = PEER_RUNTIME_PROBE_BUDGET_NS + 1;
        assert_eq!(
            runtime_peer_probe_candidate(every, next, measured_ns, false),
            None
        );
        assert_eq!(
            runtime_peer_probe_candidate(every, next, measured_ns, true),
            Some((1, 8)),
        );
    }

    #[test]
    fn late_runtime_reprobe_advances_once_instead_of_bursting_catchup() {
        let every = PEER_RUNTIME_PROBE_INTERVAL_COPIES;
        let due = every;
        assert_eq!(runtime_peer_probe_next_copy(due, due), due + 4 * every);
        assert_eq!(runtime_peer_probe_next_copy(due, 20 * every), 21 * every);
    }

    #[test]
    fn runtime_reprobe_deferral_metric_counts_intervals_and_publishes_bound_state() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

        assert_eq!(
            PEER_RUNTIME_PROBE_DEFERRAL_BOUND_INTERVALS * PEER_RUNTIME_PROBE_INTERVAL_COPIES,
            PEER_RUNTIME_PROBE_CYCLE_COPIES,
        );
        let deferred = AtomicU64::new(0);
        let degraded = AtomicBool::new(false);
        publish_runtime_peer_probe_deferral(&deferred, &degraded, 1, false);
        assert_eq!(deferred.load(Ordering::Relaxed), 1);
        assert!(!degraded.load(Ordering::Acquire));

        publish_runtime_peer_probe_deferral(
            &deferred,
            &degraded,
            PEER_RUNTIME_PROBE_DEFERRAL_BOUND_INTERVALS - 1,
            true,
        );
        assert_eq!(
            deferred.load(Ordering::Relaxed),
            PEER_RUNTIME_PROBE_DEFERRAL_BOUND_INTERVALS,
        );
        assert!(degraded.load(Ordering::Acquire));
    }

    #[test]
    fn runtime_probe_failure_latches_native_before_publishing_validated_bounce() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let failed = AtomicBool::new(false);
        let degraded = AtomicBool::new(false);
        let armed = latch_runtime_host_bounce(&failed, &degraded, || Ok::<_, String>(()));
        assert!(armed.is_ok());
        assert!(failed.load(Ordering::Acquire));
        assert!(degraded.load(Ordering::Acquire));

        let failed = AtomicBool::new(false);
        let degraded = AtomicBool::new(false);
        let refused = latch_runtime_host_bounce(&failed, &degraded, || {
            Err::<(), _>("injected staging mismatch".to_string())
        });
        assert_eq!(refused, Err("injected staging mismatch".to_string()));
        assert!(failed.load(Ordering::Acquire));
        assert!(!degraded.load(Ordering::Acquire));
    }

    #[test]
    fn transport_selection_keeps_peer_default_and_bounces_only_cross_device() {
        assert_eq!(boundary_transport(false, false), BoundaryTransport::Local);
        assert_eq!(boundary_transport(false, true), BoundaryTransport::Local);
        assert_eq!(boundary_transport(true, false), BoundaryTransport::Peer);
        assert_eq!(
            boundary_transport(true, true),
            BoundaryTransport::HostBounce
        );
    }

    #[test]
    fn step37_geometry_sizes_each_slot_from_the_prime_cap() {
        let (elems, bytes) = host_bounce_capacity(4096).expect("valid Step-3.7 geometry");
        assert_eq!(elems, 4096 * crate::cache::PRIME_CHUNK_MAX_TOKENS);
        assert_eq!(bytes, 64 * 1024 * 1024);
    }

    #[test]
    fn host_bounce_capacity_rejects_invalid_or_overflowing_geometry() {
        assert!(host_bounce_capacity(0).is_err());
        assert!(host_bounce_capacity(usize::MAX).is_err());
    }
}
