//! Tensor-parallel correctness runtime.
//!
//! This module is deliberately narrower than the serving runtime. It executes real rank-local
//! E4M3 projections on distinct CUDA devices. Deterministic host-staged collectives remain the
//! default exactness reference; an opt-in native-P2P path must reproduce the same canonical
//! checkpoint-block program before it can advance. Neither path is product-throughput evidence.

use crate::Engine;
use crate::mmq_ffi::{DeviceExpertCsr, ExpertCsr, Fp8GroupedWorkspace};
use crate::parallel::{PRODUCT_MAX_CARDS, STEP37_TRUNK_LAYERS};
use cudarc::driver::{CudaEvent, CudaSlice, DevicePtr, DeviceSlice, LaunchConfig, PushKernelArg};
use std::ops::Range;

/// Previous gate output per (rank, t), so the determ probe can report the SHAPE of a divergence
/// (dense-ULP vs sparse-huge) and not merely that a checksum moved. Probe-only state.
#[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
static DETERM_PREV: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<(usize, usize), Vec<f32>>>,
> = std::sync::OnceLock::new();

const FP8_BLOCK: usize = 128;
const NATIVE_P2P_PROBE_WORDS: &[usize] = &[4096, 16_384, 262_144, 16_777_216];
const STEP_GROUPED_FP8_EXPERTS: usize = 288;
const STEP_GROUPED_FP8_TOP_K: usize = 8;
const STEP_GROUPED_FP8_WIDTH: usize = 1280;

fn validate_step_expert_activation_limit(limit: Option<f32>) -> Result<(), String> {
    if let Some(limit) = limit
        && (!limit.is_finite() || limit <= 0.0)
    {
        return Err(format!(
            "Step routed-expert activation limit must be positive and finite, got {limit}"
        ));
    }
    Ok(())
}

/// Host-canonical Step routed-expert SwiGLU operation.
///
/// Step's final routed layers clamp the linear arm symmetrically and the SiLU arm only above.
/// Keeping this scalar order explicit also defines the device-host-exact CUDA gate.
/// Raw stream-ordered device copy for capture-safe cross-context seams (cudarc's slice-use
/// tracking creates capture-illegal dependencies there). Pointers must be pre-cached with
/// their owners' streams; bytes flow identically to the tracked copy.
/// MEMRA_OPROJ_DIRECT=1 (o-proj direct join, default OFF until gated): peer ranks write
/// their fused O partial OVER P2P into a root-resident buffer (UVA kernel stores), and the
/// model engine adds the two partials itself — the root stream leaves the join entirely
/// (no peer pull copy, no root add, no second event hop, no final 16KB ownership copy).
/// Reduction order and kernel programs are unchanged, so the row is BIT-IDENTICAL.
/// MEMRA_MOE_DIRECT=1 (moe direct join, default OFF until gated): the o-proj direct-join
/// recipe on the expert combine — peer ranks' accumulators live root-side (the axpy twin
/// register-accumulates and stores ONCE, so the P2P cost is a single 16KB store pass), and
/// the model engine adds the two shard rows itself. Operand order matches root's add:
/// BIT-IDENTICAL.
/// MEMRA_ROUTES_PRESTAGE=1 (default OFF until gated): stage the shared layer input to
/// every rank and quantize it BEFORE the router runs — neither depends on the selection,
/// so the rank streams' pull+quantize overlaps dev0's router gemv+topk instead of chaining
/// behind it (the router->quantize and axpy->add gap edges). Same copies, same quantize
/// kernel, same operands: BIT-IDENTICAL.
pub(crate) fn routes_prestage_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_ROUTES_PRESTAGE").as_deref() == Ok("1"))
}

/// MEMRA_FENCE_MEMOPS=1 (default OFF until gated): the moe direct join's two event
/// fences become cuStreamWriteValue32/cuStreamWaitValue32 doorbells — hardware stream
/// memops with lower signal->wake latency than cross-device cuStreamWaitEvent. Ordering:
/// PCIe posted writes from one device arrive in order, so rank1's accumulator stores are
/// visible before its flag write lands; e's GEQ wait then covers them. Falls back to
/// events when the device rejects stream memops. Scheduling-only: BIT-IDENTICAL values.
/// MEMRA_LEN_MIRROR_LAZY=1 (default OFF until gated): skip redundant per-layer 4B len
/// htods — the local device mirror is unread in TP decode, and under FUSE_ROPE_APPEND the
/// fused append's atomicInc owns the rank counters. Every one of those tiny copies is a
/// compute->copy engine turnaround in the middle of the layer stream.
/// MEMRA_RANK0_MERGE=1 (default OFF until gated): same-device rank0 rides e's stream via
/// the runtime redirect — see decode_step_h.
/// MEMRA_OPROJ_TAIL=1 (default OFF until gated): the o-proj direct-join add is DEFERRED —
/// the finish arm keeps its waits, stores the two partial pointers here, and the residual
/// add_rms_norm consumer composes mixed = a0+a1 in-register (join_add_rms_norm, verbatim
/// program: BIT-IDENTICAL). The returned `mixed` buffer is UNWRITTEN in this mode; its
/// only live consumer is the residual_norm_ffn seam, which takes the handoff.
pub(crate) fn oproj_tail_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_OPROJ_TAIL").as_deref() == Ok("1"))
}
thread_local! {
    static OPROJ_TAIL_PENDING: std::cell::Cell<Option<(u64, u64)>> =
        const { std::cell::Cell::new(None) };
}
thread_local! {
    /// The deferral is legal ONLY under callers whose walk flows into
    /// residual_norm_ffn (decode_step_h / decode_step_chain arm this) — the verify
    /// prefill reaches the same finish and would consume unwritten `mixed` otherwise
    /// (M2-MISMATCH receipt: prefill argmax corrupted while decode stayed exact).
    static OPROJ_TAIL_ELIGIBLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
/// RAII eligibility scope for the o-proj tail deferral.
pub(crate) struct OprojTailScope(());
pub(crate) fn oproj_tail_scope() -> OprojTailScope {
    OPROJ_TAIL_ELIGIBLE.with(|c| c.set(true));
    OprojTailScope(())
}
impl Drop for OprojTailScope {
    fn drop(&mut self) {
        OPROJ_TAIL_ELIGIBLE.with(|c| c.set(false));
        // A leftover un-consumed handoff must never leak across calls.
        OPROJ_TAIL_PENDING.with(|c| c.set(None));
    }
}
thread_local! {
    /// T-COLUMN verify select: the verify driver sets the column before each per-column
    /// attention call; decode_v2_input_qkv takes it (once) and selects from the slabs.
    static VERIFY_TCOL: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}
pub(crate) fn set_verify_tcol(c: Option<usize>) {
    VERIFY_TCOL.with(|x| x.set(c));
}
pub(crate) fn take_verify_tcol() -> Option<usize> {
    VERIFY_TCOL.with(|x| x.take())
}

/// MEMRA_TCOL_OPROJ=1 (spec verify): defer each column's o_proj out of the per-column
/// walk — the finish seam stashes the column's `gated` rows instead of running the
/// per-column finish choreography (rank events, P2P join, engine handoff), and one
/// weight-amortized b4_tcol per rank + one elementwise join produce every column's
/// `mixed` afterwards. Bit-exact per column: the tcol kernel is the t=1 b4 program per
/// column, and the slab join adds the same operand values elementwise.
pub(crate) fn tcol_oproj_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_TCOL_OPROJ").as_deref() == Ok("1"))
}
thread_local! {
    /// The verify driver arms the column before each per-column attention call; the
    /// finish seam takes it (once). Stashed=true reports the defer actually happened
    /// (the seam falls back to the normal finish when the config is ineligible).
    static TCOL_OPROJ_DEFER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static TCOL_OPROJ_STASHED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
pub(crate) fn set_tcol_oproj_defer(c: Option<usize>) {
    TCOL_OPROJ_DEFER.with(|x| x.set(c));
}
pub(crate) fn take_tcol_oproj_defer() -> Option<usize> {
    TCOL_OPROJ_DEFER.with(|x| x.take())
}
pub(crate) fn set_tcol_oproj_stashed() {
    TCOL_OPROJ_STASHED.with(|x| x.set(true));
}
pub(crate) fn take_tcol_oproj_stashed() -> bool {
    TCOL_OPROJ_STASHED.with(|x| x.replace(false))
}

pub(crate) fn oproj_tail_eligible() -> bool {
    OPROJ_TAIL_ELIGIBLE.with(|c| c.get())
}
pub(crate) fn take_oproj_tail() -> Option<(u64, u64)> {
    OPROJ_TAIL_PENDING.with(|c| c.take())
}
pub(crate) fn set_oproj_tail(v: (u64, u64)) {
    OPROJ_TAIL_PENDING.with(|c| c.set(Some(v)));
}

pub(crate) fn rank0_merge_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_RANK0_MERGE").as_deref() == Ok("1"))
}

pub(crate) fn len_mirror_lazy_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_LEN_MIRROR_LAZY").as_deref() == Ok("1"))
}

pub(crate) fn fence_memops_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_FENCE_MEMOPS").as_deref() == Ok("1"))
}

pub(crate) fn moe_direct_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_MOE_DIRECT").as_deref() == Ok("1"))
}

/// MEMRA_SEL_MIRROR=1: the per-rank routed-selection pull runs as ONE `moe_sel_w_mirror`
/// launch instead of two 32-byte D2D copies, and when every consuming rank shares e's device
/// the intermediate e-context staging pair is skipped entirely (the caller's sel/route_w rows
/// are process-persistent, so the ranks read them directly). Bit-identical: same bytes, one
/// fewer hop. Refused under the graph door, whose captured copies need the fixed staging
/// addresses. Default OFF until receipted.
/// MEMRA_SPEC_FA2=1 (the DSpark verify lesson): the verify walk defers each column's
/// ATTENTION CORE — the dcw arm appends the column's K/V and stashes its post-rope q and
/// gate rows, then ONE fa_decode_vec_q_v3_dcw_rows per rank attends every stashed row
/// (each row runs its own t=1 dcw program, bit-identical per row), the per-row combine
/// writes the gated rows, and the o_proj join runs on the TCOL slabs.
/// ROW-TABLE RESTAGE (`MEMRA_ROWS_TAB_RESTAGE`, DEFAULT ON since this lane).
///
/// ON: `decode_v2_rope_fa_rows` builds the 6-word-per-row pointer table from the caller's
/// freshly-read live cache pointers and stages it into a persistent per-rank slab before
/// every launch. OFF (`=0`): the retired process-lifetime `rows_tabs` memo, keyed by a hash
/// of (k pointer, base pointer, layer, t) that could not see the V or LEN pointers the
/// entry also carried, and that nothing invalidated when a session's KV cache was dropped.
///
/// Default ON because the OFF arm is a proven use-after-free, not a slower correct path:
/// on step37-flash with MEMRA_FUSE_ROPE_APPEND=1 it made speculative decoding unservable
/// (whole non-finite verify rows, then CUDA_ERROR_ILLEGAL_ADDRESS). ON is value-neutral on
/// every fresh lookup by construction: identical bytes reach the same kernels. Rollback
/// seam: `MEMRA_ROWS_TAB_RESTAGE=0`.
/// The 6-word-per-row launch table `{k, v, len, base, ctr, back}` the fused rope/append/fa
/// kernels dereference. Pure so it can be tested: the words come from the caller's live
/// per-row `[k, v, len, base]` pointers, `ctr` is this rank's counter slab (one shared cell
/// for same-session rows, one cell per row otherwise) and `back` is the same-session causal
/// step-back `t-1-r` (0 across sessions, where each row owns its own len).
pub(crate) fn rows_tab_host(
    parts_rank: &[[u64; 4]],
    ctr_base: u64,
    same_session: bool,
    t: usize,
) -> Vec<u64> {
    let mut host = Vec::with_capacity(t * 6);
    for (r, parts) in parts_rank.iter().enumerate().take(t) {
        host.extend_from_slice(&[
            parts[0],
            parts[1],
            parts[2],
            parts[3],
            if same_session {
                ctr_base
            } else {
                ctr_base + (r as u64) * 4
            },
            if same_session {
                (t - 1 - r) as u64
            } else {
                0u64
            },
        ]);
    }
    host
}

/// The RETIRED memo key, kept ONLY so a test can assert what it cannot see. Both historical
/// call sites hashed a SUBSET of the pointers the table carries; this reproduces the verify
/// site's formula verbatim.
#[cfg(test)]
pub(crate) fn retired_rows_tab_key(kp: u64, bp: u64, il: usize, t: usize) -> u64 {
    kp.rotate_left(17)
        .wrapping_add(bp)
        .wrapping_add((il as u64) << 32)
        .wrapping_add(t as u64)
        .wrapping_add(1 << 63)
}

pub(crate) fn rows_tab_restage_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_ROWS_TAB_RESTAGE").as_deref() != Ok("0"))
}

/// STALE-HIT RECEIPT (`MEMRA_ROWS_TAB_STALE_SCAN`, DEFAULT OFF, diagnostic only).
///
/// Keeps a HOST shadow of the last table staged under each retired memo key and prints one
/// line whenever the key repeats with different contents, naming the words that moved. It
/// costs a host hash lookup and a small clone per rank per layer per verify round, so it is
/// off in serving. `[rows-tab] engaged=` on the counter proves the path executes at all,
/// which is what separates "the memo was innocent" from "the memo never ran".
/// Rollback seam: unset it (or `=0`).
pub(crate) fn rows_tab_stale_scan() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_ROWS_TAB_STALE_SCAN").as_deref() == Ok("1"))
}

pub(crate) static ROWS_TAB_ENGAGED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static ROWS_TAB_STALE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(crate) fn spec_fa2_on() -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    crate::step37_door(&ENV, "MEMRA_SPEC_FA2")
}
thread_local! {
    /// The verify driver arms the column before each per-column attention call; the dcw
    /// arm takes it (once) and stashes q/gate instead of running fa+finish.
    static SPEC_FA2_DEFER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static SPEC_FA2_STASHED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
pub(crate) fn set_spec_fa2_defer(c: Option<usize>) {
    SPEC_FA2_DEFER.with(|x| x.set(c));
}
pub(crate) fn take_spec_fa2_defer() -> Option<usize> {
    SPEC_FA2_DEFER.with(|x| x.take())
}
pub(crate) fn set_spec_fa2_stashed() {
    SPEC_FA2_STASHED.with(|x| x.set(true));
}
pub(crate) fn take_spec_fa2_stashed() -> bool {
    SPEC_FA2_STASHED.with(|x| x.replace(false))
}

pub(crate) fn sel_mirror_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SEL_MIRROR").as_deref() == Ok("1"))
}

pub(crate) fn oproj_direct_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_OPROJ_DIRECT").as_deref() == Ok("1"))
}

// ─── Slot-major NVFP4 expert-bank programs: THREE independent doors ───────────────────────────
//
// These restore, under separate flags, the three programs that the 2026-08-29 removal
// (`fd0a175ab`) deleted behind ONE env var (`MEMRA_NVFP4_BANK_V2`). That coupling is why the
// incident's bisect could not name a mechanism: toggling one var moved the bank layout, the
// gate+up fusion (which auto-armed on the same predicate with no door of its own) and the fused
// down+combine (which hard-refused without the layout) all at once, so the priced -21.5% wall /
// -23.7% decode (research/perf-chain-20260831 cell 1) was an unattributable bundle.
//
// The corruption they were removed for was NOT any of them: it was a defaulted `in_f = 0`
// argument at two `kq_fetch` call sites in the PREFILL grouped-GEMM tail
// (research/step37-bankv3-20260901/DIAGNOSIS.md), fixed compiler-enforced at `1b18a61e8` and
// gated device-side by the `nvfp4-bank-oracle` bin. Each door below is strict `0`/`1` and
// admitted separately so its contribution is a number; BANK_SM and SEL_DOWN8 default ON since
// 2026-09-01 (one coupled decision, PR #76 battery), SEL_GU and the sub-doors default OFF.
//
// LAYOUT IS A PROPERTY OF THE BANK. `bank_slot_major_on()` is read ONCE, at bank BUILD, and
// recorded on the resident bank (`ResidentNvfp4{Column,Row}BankRank::slot_major`). Every reader
// branches on that stored field, never on the env door. The removed implementation read
// `nvfp4_bank_v2_on()` at each reader site instead, which is the same class of hole as the
// defaulted `in_f`: a piece of layout geometry that a caller can fail to supply or can supply
// inconsistently with the bytes actually resident.

/// Read a DEFAULT-ON door strictly, and report the SOURCE of the answer rather than only the
/// answer. `0` disables (the rollback seam), `1` re-states the default, unset takes the default.
///
/// Two properties this buys, both learned the hard way in this lane's own archaeology:
///
/// * **A typo cannot silently disarm a rollback seam.** The default-OFF doors here parse as
///   `== Ok("1")`, which is safe when the default is OFF (a typo reads as the default) and
///   DANGEROUS when the default is ON: `MEMRA_NVFP4_BANK_SM=false` under a `!= Ok("0")` rule
///   would keep the program armed while the operator believed it was rolled back. So an
///   unrecognized value is reported as such and the default is kept, loudly.
/// * **The engagement receipt can name the source.** `default-on` and `MEMRA_..=1` are
///   different facts about the same boot: one says the flip is doing the work, the other says
///   a recipe is. A pricing or post-deploy receipt that cannot tell them apart cannot prove a
///   DEFAULT was measured (TRAP:corrupt-arm-inflates-its-own-perf-price's sibling: an arm that
///   cannot name what armed it is not an arm).
fn door_default_on(name: &'static str) -> (bool, &'static str) {
    let raw = std::env::var(name).ok();
    door_default_on_value(name, raw.as_deref())
}

/// The parse, separated from the environment so it can be TESTED. `std::env` is process-global
/// state and these doors are `OnceLock`-cached, so an env-var test would be both racy under
/// `cargo test`'s thread pool and unrepeatable within one process — i.e. exactly the kind of
/// gate that passes because it never really ran.
fn door_default_on_value(name: &str, value: Option<&str>) -> (bool, &'static str) {
    match value {
        Some("0") => (false, "env=0 (rollback seam)"),
        Some("1") => (true, "env=1"),
        None => (true, "default-on"),
        Some(_) => {
            eprintln!(
                "[nvfp4-door] WARN {name} has an unrecognized value; only `0` and `1` are \
                 accepted and the DEFAULT-ON answer is kept. To roll back, set {name}=0."
            );
            (true, "default-on (unrecognized value ignored)")
        }
    }
}

/// MEMRA_NVFP4_BANK_SM (PROGRAM 1, **default ON since 2026-09-01**): build the step TP
/// contiguous NVFP4 expert banks (gate/up/down) in the SLOT-MAJOR row layout — slot g's 16 qs
/// bytes contiguous at `g*16` (one coalesced 512B warp wave) and the two UE4M3 scale bytes at
/// `nslots*16 + g*2` — and dispatch the `_sel_v2` decode readers over them. Pure byte
/// permutation, so BIT-IDENTICAL per row; the claim is gated by `nvfp4-bank-oracle`
/// (device-side, prefill GEMM included) and by end-to-end greedy byte identity, never by a
/// comment.
///
/// **WHY A BIT-IDENTICAL, MEASURABLY-FREE PROGRAM DEFAULTS ON.** On its own this layout earns
/// nothing: x5 interleaved, 105.35 vs 106.78 decode tok/s, per-boot range `[104.66, 107.95]`
/// overlapping the OFF arm's `[105.11, 107.09]`. It defaults ON for exactly one reason —
/// `MEMRA_NVFP4_SEL_DOWN8` (PROGRAM 3), the one program that DOES separate (+5.48% decode), is
/// gated at its call site on `shard.slot_major`, so with this door off PROGRAM 3's default-ON
/// is a SILENT NO-OP: `down8=false door=true`, no refusal, no warning, and the win simply does
/// not happen. The deployable unit is the two together, which makes this one coupled default
/// decision and not two independent ones. Receipts:
/// `research/step37-bankv3-20260901/RESULTS.md` (the down8 default-ON qualification battery).
///
/// ROLLBACK SEAM: `MEMRA_NVFP4_BANK_SM=0`, which also disarms PROGRAM 3 by construction.
pub(crate) fn bank_slot_major_on() -> bool {
    bank_slot_major_source().0
}

/// `bank_slot_major_on()` plus the SOURCE of the answer, for the engagement receipt.
pub(crate) fn bank_slot_major_source() -> (bool, &'static str) {
    static ON: std::sync::OnceLock<(bool, &'static str)> = std::sync::OnceLock::new();
    *ON.get_or_init(|| door_default_on("MEMRA_NVFP4_BANK_SM"))
}

/// MEMRA_NVFP4_SEL_DOWN8 (PROGRAM 3, default ON since 2026-09-01; `=0` is the rollback seam): fuse the routed DOWN sweep with the
/// route-weight combine into one launch (`qmatvec_nvfp4_dp4a_sel_v2_down8`, the q8 `down8 w8`
/// occupancy arm ported to the NVFP4 banks) — one warp per routed slot instead of one warp per
/// (row, slot), and the `n_sel x out_f` partial-buffer round trip disappears. Bit-identical
/// (same dot program, same reduce tree, same slot-ordered combine chain). Also subordinate to
/// PROGRAM 1: the caller arms it only when the down shard reports `slot_major`, and only on the
/// device-routed arm at `nsb <= 32` (the fit-block class the reduce identity is argued at).
/// Rides LAST per the lane mandate: it is priced only on green gates for the layers beneath it.
///
/// **DEFAULT ON since 2026-09-01**, and it is the reason PROGRAM 1 defaults ON too. This is the
/// only one of the three restored programs that separates from noise: +5.48% decode / +5.09%
/// wall, x5 interleaved vendor-default sampled, per-boot range `[112.59, 114.82]` with NO
/// overlap against either the OFF arm or the arm directly beneath it, re-qualified at deploy
/// grade in `research/step37-bankv3-20260901/RESULTS.md`.
///
/// ELIGIBILITY IS NARROWER THAN THE DEFAULT, and the engagement receipt below prints every
/// condition: the arm needs `device_routed`, `shard.slot_major` (i.e. PROGRAM 1) and
/// `nsb <= 32`. On any other geometry or route the default is INERT, which is correct-by-
/// refusal and NOT a regression — but it does mean "default ON" and "engaged" are two facts,
/// and only the `[nvfp4-sweep]` line settles the second.
///
/// ROLLBACK SEAM: `MEMRA_NVFP4_SEL_DOWN8=0` (or `MEMRA_NVFP4_BANK_SM=0`, which disarms it by
/// construction).
pub(crate) fn sel_down8_on() -> bool {
    sel_down8_source().0
}

/// `sel_down8_on()` plus the SOURCE of the answer, for the engagement receipt.
pub(crate) fn sel_down8_source() -> (bool, &'static str) {
    static ON: std::sync::OnceLock<(bool, &'static str)> = std::sync::OnceLock::new();
    *ON.get_or_init(|| door_default_on("MEMRA_NVFP4_SEL_DOWN8"))
}

pub(crate) fn raw_copy_bytes(
    dst: u64,
    src: u64,
    bytes: usize,
    engine: &Engine,
) -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::sys;
    let r = unsafe {
        sys::cuMemcpyAsync(
            dst as sys::CUdeviceptr,
            src as sys::CUdeviceptr,
            bytes,
            engine.stream().cu_stream() as sys::CUstream,
        )
    };
    if r == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        // MEMRA_RAW_COPY_TRACE=1: a raw D2D failure carries no call site by itself, and
        // every slab-width bug in the t-row family surfaces here. Operands + backtrace.
        if std::env::var("MEMRA_RAW_COPY_TRACE").as_deref() == Ok("1") {
            eprintln!(
                "[raw-copy-fail] dst={dst:#x} src={src:#x} bytes={bytes} {r:?}\n{}",
                std::backtrace::Backtrace::force_capture()
            );
        }
        Err(format!("raw_copy_bytes: {r:?} bytes={bytes} dst={dst:#x} src={src:#x}").into())
    }
}

pub fn step_expert_activation_host(gate: f32, up: f32, limit: Option<f32>) -> f32 {
    let silu = gate / (1.0 + (-gate).exp());
    match limit {
        Some(limit) => silu.min(limit) * up.clamp(-limit, limit),
        None => silu * up,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpertOwnerRoutes {
    rank: usize,
    selected: Vec<usize>,
    token_rows: Vec<usize>,
    global_pairs: Vec<usize>,
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn partition_expert_owner_routes(
    expert_count: usize,
    ranks: usize,
    tokens: usize,
    experts_per_token: usize,
    selected: &[usize],
) -> Result<Vec<ExpertOwnerRoutes>, String> {
    if expert_count == 0
        || ranks == 0
        || tokens == 0
        || experts_per_token == 0
        || expert_count % ranks != 0
    {
        return Err(format!(
            "invalid expert-owner route geometry experts={expert_count} ranks={ranks} \
             tokens={tokens} experts_per_token={experts_per_token}"
        ));
    }
    let pairs = tokens
        .checked_mul(experts_per_token)
        .ok_or("expert-owner route count overflow")?;
    if selected.len() != pairs {
        return Err(format!(
            "expert-owner routes {} != {tokens}x{experts_per_token} ({pairs})",
            selected.len()
        ));
    }
    let per_rank = expert_count / ranks;
    let mut owners = (0..ranks)
        .map(|rank| ExpertOwnerRoutes {
            rank,
            selected: Vec::new(),
            token_rows: Vec::new(),
            global_pairs: Vec::new(),
        })
        .collect::<Vec<_>>();
    for (pair, &expert) in selected.iter().enumerate() {
        if expert >= expert_count {
            return Err(format!(
                "expert-owner route {pair} selects expert {expert} outside 0..{expert_count}"
            ));
        }
        let rank = expert / per_rank;
        owners[rank].selected.push(expert - rank * per_rank);
        owners[rank].token_rows.push(pair / experts_per_token);
        owners[rank].global_pairs.push(pair);
    }
    Ok(owners)
}

fn validate_step_grouped_owner_routes(
    expert_count: usize,
    tokens: usize,
    selected: &[usize],
) -> Result<usize, String> {
    if expert_count != STEP_GROUPED_FP8_EXPERTS || tokens == 0 {
        return Err(format!(
            "official Step owner-grouped FP8 requires {} experts and nonzero tokens, got \
             experts={expert_count} tokens={tokens}",
            STEP_GROUPED_FP8_EXPERTS
        ));
    }
    let pairs = tokens
        .checked_mul(STEP_GROUPED_FP8_TOP_K)
        .ok_or("official Step owner-grouped FP8 route count overflow")?;
    if selected.len() != pairs {
        return Err(format!(
            "official Step owner-grouped FP8 routes {} != {tokens}x{} ({pairs})",
            selected.len(),
            STEP_GROUPED_FP8_TOP_K,
        ));
    }
    for (token, routes) in selected.chunks_exact(STEP_GROUPED_FP8_TOP_K).enumerate() {
        let mut unique = routes.to_vec();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != STEP_GROUPED_FP8_TOP_K {
            return Err(format!(
                "official Step owner-grouped FP8 token {token} routes are not top-8 unique: \
                 {routes:?}"
            ));
        }
    }
    Ok(pairs)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WeightedRouteCombineShape {
    pairs: usize,
    max_pairs: usize,
}

fn validate_weighted_route_combine(
    width: usize,
    experts_per_token: usize,
    max_tokens: usize,
    tokens: usize,
    owner_global_pairs: &[&[usize]],
    route_weights: &[f32],
) -> Result<WeightedRouteCombineShape, String> {
    if width == 0
        || experts_per_token == 0
        || max_tokens == 0
        || tokens == 0
        || tokens > max_tokens
        || width > i32::MAX as usize
        || experts_per_token > i32::MAX as usize
        || tokens > i32::MAX as usize
    {
        return Err(format!(
            "invalid weighted route combine geometry width={width} experts_per_token=\
             {experts_per_token} tokens={tokens}/{max_tokens}"
        ));
    }
    let pairs = tokens
        .checked_mul(experts_per_token)
        .ok_or("weighted route combine pair count overflow")?;
    let max_pairs = max_tokens
        .checked_mul(experts_per_token)
        .ok_or("weighted route combine capacity overflow")?;
    if route_weights.len() != pairs || !route_weights.iter().all(|weight| weight.is_finite()) {
        return Err(format!(
            "weighted route combine weights {} != pairs {pairs} or contain a non-finite value",
            route_weights.len()
        ));
    }
    let mut seen = vec![false; pairs];
    let mut observed = 0usize;
    for pairs_for_owner in owner_global_pairs {
        observed = observed
            .checked_add(pairs_for_owner.len())
            .ok_or("weighted route combine observed pair count overflow")?;
        for &pair in *pairs_for_owner {
            if pair >= pairs || std::mem::replace(&mut seen[pair], true) {
                return Err(format!(
                    "weighted route combine pair {pair} is outside 0..{pairs} or duplicated"
                ));
            }
        }
    }
    if observed != pairs || seen.iter().any(|present| !present) {
        return Err(format!(
            "weighted route combine owner schedules cover {observed} of {pairs} canonical pairs"
        ));
    }
    Ok(WeightedRouteCombineShape { pairs, max_pairs })
}

fn cache_rank_rows(
    rows: &[u8],
    tokens: usize,
    local_token_bytes: usize,
    ranks: usize,
    rank: usize,
) -> Result<Vec<u8>, String> {
    if ranks == 0 || rank >= ranks {
        return Err(format!(
            "TP cache rank {rank} is outside a {ranks}-rank layout"
        ));
    }
    let global_token_bytes = local_token_bytes
        .checked_mul(ranks)
        .ok_or("TP cache global token-byte overflow")?;
    let expected = tokens
        .checked_mul(global_token_bytes)
        .ok_or("TP cache row-byte overflow")?;
    if rows.len() != expected {
        return Err(format!(
            "TP cache rows contain {} bytes, expected {tokens}x{global_token_bytes}={expected}",
            rows.len()
        ));
    }
    let mut shard = Vec::with_capacity(tokens * local_token_bytes);
    for token in 0..tokens {
        let start = token * global_token_bytes + rank * local_token_bytes;
        shard.extend_from_slice(&rows[start..start + local_token_bytes]);
    }
    Ok(shard)
}

fn parse_step_tp_native_p2p(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_STEP_TP_NATIVE_P2P={value:?} is invalid; expected 0 or 1"
        )),
    }
}

pub fn step_tp_native_p2p_enabled() -> Result<bool, String> {
    parse_step_tp_native_p2p(std::env::var("MEMRA_STEP_TP_NATIVE_P2P").ok().as_deref())
}

fn parse_step_tp_bulk_p2p(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_STEP_TP_BULK_P2P={value:?} is invalid; expected 0 or 1"
        )),
    }
}

pub fn step_tp_bulk_p2p_enabled() -> Result<bool, String> {
    parse_step_tp_bulk_p2p(std::env::var("MEMRA_STEP_TP_BULK_P2P").ok().as_deref())
}

fn parse_step_ep_device_arithmetic(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_STEP_EP_DEVICE_ARITHMETIC={value:?} is invalid; expected 0 or 1"
        )),
    }
}

fn parse_step_nvfp4_dev_routes(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_STEP_NVFP4_DEV_ROUTES={value:?} is invalid; expected 0 or 1"
        )),
    }
}

/// Opt-in door for the device-resident NVFP4 TP routed-expert decode program. Default OFF; the
/// host-canonical program remains the oracle until the device path carries its own gates.
pub fn step_nvfp4_dev_routes_enabled() -> Result<bool, String> {
    parse_step_nvfp4_dev_routes(std::env::var("MEMRA_STEP_NVFP4_DEV_ROUTES").ok().as_deref())
}

pub fn step_ep_device_arithmetic_enabled() -> Result<bool, String> {
    parse_step_ep_device_arithmetic(
        std::env::var("MEMRA_STEP_EP_DEVICE_ARITHMETIC")
            .ok()
            .as_deref(),
    )
}

fn parse_step_tp_f32_mirror(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_STEP_TP_F32_MIRROR={value:?} is invalid; expected 0 or 1"
        )),
    }
}

pub fn step_tp_f32_mirror_enabled() -> Result<bool, String> {
    parse_step_tp_f32_mirror(std::env::var("MEMRA_STEP_TP_F32_MIRROR").ok().as_deref())
}

fn parse_step_tp_decode_v2(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_STEP_TP_DECODE_V2={value:?} is invalid; expected 0 or 1"
        )),
    }
}

/// The v2 rank-local Step decode-attention driver: persistent workspaces, evented cross-stream
/// ordering, and a root-device O reduction — same kernels, values, and canonical reduction order
/// as the v1 driver (it requires the F32 mirror so no per-call weight expansion exists on either
/// side of the comparison).
pub fn step_tp_decode_v2_enabled() -> Result<bool, String> {
    parse_step_tp_decode_v2(std::env::var("MEMRA_STEP_TP_DECODE_V2").ok().as_deref())
}

fn parse_step_tp_qkv_fused(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_STEP_TP_QKV_FUSED={value:?} is invalid; expected 0 or 1"
        )),
    }
}

fn parse_step_tp_dev_router(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_STEP_TP_DEV_ROUTER={value:?} is invalid; expected 0 or 1"
        )),
    }
}

/// Device-side sigmoid top-k routing for the TP device-IO expert program: the per-layer host
/// logits readback (the last per-layer host sync) disappears. Selection tie-breaking may
/// differ from the host router — NUMERIC-CLASS door, run-gen argmax gate + boot battery.
pub fn step_tp_dev_router_enabled() -> Result<bool, String> {
    parse_step_tp_dev_router(std::env::var("MEMRA_STEP_TP_DEV_ROUTER").ok().as_deref())
}

fn parse_step_tp_graph(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_STEP_TP_GRAPH={value:?} is invalid; expected 0 or 1"
        )),
    }
}

fn parse_step_tp_dcw(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_STEP_TP_DCW={value:?} is invalid; expected 0 or 1"
        )),
    }
}

/// Device-counter attention path (graph increment A run EAGERLY): append at len_d - base_d,
/// inc_i32, fa over the counter-derived window — with bucket = the effective t_kv this is
/// bit-identical to the host-row + kvmod path (the one-partition law), and it is the exact
/// child content the capture wraps. Rebase tokens and sub-vec-floor contexts fall back.
pub fn step_tp_dcw_enabled() -> Result<bool, String> {
    parse_step_tp_dcw(std::env::var("MEMRA_STEP_TP_DCW").ok().as_deref())
}

/// CUDA-graph door for the shape-stable TP segments (first increment: the device-routed
/// expert program — per-layer multi-device parents built from per-rank children, launched on
/// the model engine's stream; zero per-token node updates). Mechanism proven by
/// tp_graph_probe. VALUE-IDENTICAL: the graphs replay exactly the eager kernel/copy sequence.
pub fn step_tp_graph_enabled() -> Result<bool, String> {
    parse_step_tp_graph(std::env::var("MEMRA_STEP_TP_GRAPH").ok().as_deref())
}

/// GRAPH-LAUNCH HEADROOM GUARD for the routed-prejoin graph door (see
/// `spec::GRAPH_LAUNCH_MIN_FREE`): checked on the launching engine only when the door
/// is armed (short-circuit after `step_tp_graph_enabled`), noting once per process with
/// the sweep's grep-stable `graph replay suspended:` key.
fn step_tp_graph_headroom_ok(e: &Engine) -> bool {
    let ok = crate::spec::graph_launch_headroom_ok(e);
    if !ok {
        static NOTED: std::sync::Once = std::sync::Once::new();
        NOTED.call_once(|| crate::spec::graph_replay_suspended_note("step-tp-routes"));
    }
    ok
}

/// Fused single-launch QKV projection inside the v2 decode driver — a NUMERIC-CLASS door
/// (per-row deterministic tree reduce instead of the chunked cuBLASLt program), default OFF,
/// gated by the run-gen argmax gate + boot battery like MEMRA_STEP_NVFP4_DEV_ROUTES.
pub fn step_tp_qkv_fused_enabled() -> Result<bool, String> {
    parse_step_tp_qkv_fused(std::env::var("MEMRA_STEP_TP_QKV_FUSED").ok().as_deref())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepEpLayerSpec {
    pub layer: usize,
    pub devices: Vec<usize>,
}

pub type StepTpLayerSpec = StepEpLayerSpec;

/// ModelPlan-driven whole-model parallel policy. `auto` removes per-layer family recipes; the
/// loader derives its scope from dense/MoE operations and selects a registered numeric backend
/// from the artifact tensor/activation contract.
fn parse_auto_parallel_devices(
    mode: Option<&str>,
    raw_devices: Option<&str>,
) -> Result<Option<Vec<usize>>, String> {
    let mode = match mode {
        None | Some("") | Some("0") | Some("off") => return Ok(None),
        Some("auto") => "auto",
        Some(value) => {
            return Err(format!(
                "MEMRA_PARALLEL={value:?} is invalid; expected off or auto"
            ));
        }
    };
    let raw = raw_devices.ok_or_else(|| {
        format!("{mode} parallel placement requires MEMRA_PARALLEL_DEVICES=DEVICE,DEVICE[...]")
    })?;
    let devices =
        raw.split(',')
            .map(|device| {
                device.trim().parse::<usize>().map_err(|_| {
                    format!("MEMRA_PARALLEL_DEVICES entry {device:?} is not an integer")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
    if !(2..=crate::parallel::AUTO_PARALLEL_MAX_CARDS).contains(&devices.len()) {
        return Err(format!(
            "MEMRA_PARALLEL=auto requires 2..={} devices, got {}",
            crate::parallel::AUTO_PARALLEL_MAX_CARDS,
            devices.len()
        ));
    }
    let mut unique = devices.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != devices.len() {
        return Err(format!(
            "MEMRA_PARALLEL_DEVICES must be distinct, got {devices:?}"
        ));
    }
    Ok(Some(devices))
}

pub fn auto_parallel_devices() -> Result<Option<Vec<usize>>, String> {
    parse_auto_parallel_devices(
        std::env::var("MEMRA_PARALLEL").ok().as_deref(),
        std::env::var("MEMRA_PARALLEL_DEVICES").ok().as_deref(),
    )
}

fn parse_parallel_ep_device_router(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_PARALLEL_EP_DEVICE_ROUTER={value:?} is invalid; expected 0 or 1"
        )),
    }
}

pub fn parallel_ep_device_router_enabled() -> Result<bool, String> {
    parse_parallel_ep_device_router(
        std::env::var("MEMRA_PARALLEL_EP_DEVICE_ROUTER")
            .ok()
            .as_deref(),
    )
}

fn parse_parallel_ep_pair_down(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_PARALLEL_EP_PAIR_DOWN={value:?} is invalid; expected 0 or 1"
        )),
    }
}

pub fn parallel_ep_pair_down_enabled() -> Result<bool, String> {
    parse_parallel_ep_pair_down(std::env::var("MEMRA_PARALLEL_EP_PAIR_DOWN").ok().as_deref())
}

fn parse_parallel_ep_q8_act(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "MEMRA_PARALLEL_EP_Q8_ACT={value:?} is invalid; expected 0 or 1"
        )),
    }
}

pub fn parallel_ep_q8_act_enabled() -> Result<bool, String> {
    parse_parallel_ep_q8_act(std::env::var("MEMRA_PARALLEL_EP_Q8_ACT").ok().as_deref())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ParallelEpQ8Scope {
    All,
    GateUp,
    Down,
}

impl ParallelEpQ8Scope {
    fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::GateUp => "gate-up",
            Self::Down => "down",
        }
    }
}

fn parse_parallel_ep_q8_scope(value: Option<&str>) -> Result<Option<ParallelEpQ8Scope>, String> {
    match value {
        None | Some("") => Ok(None),
        Some("all") => Ok(Some(ParallelEpQ8Scope::All)),
        Some("gate-up") => Ok(Some(ParallelEpQ8Scope::GateUp)),
        Some("down") => Ok(Some(ParallelEpQ8Scope::Down)),
        Some(value) => Err(format!(
            "MEMRA_PARALLEL_EP_Q8_SCOPE={value:?} is invalid; expected all, gate-up, or down"
        )),
    }
}

pub(crate) fn parallel_ep_q8_scope() -> Result<Option<ParallelEpQ8Scope>, String> {
    parse_parallel_ep_q8_scope(std::env::var("MEMRA_PARALLEL_EP_Q8_SCOPE").ok().as_deref())
}

/// The paired-CTA Q8 gate/up program: one CTA computes the same output row for gate and up and
/// shares the q8_1 activation loads. Always on where Q8 gate/up arithmetic runs.
pub(crate) fn parallel_ep_q8_gu_paired_enabled(
    q8_active: bool,
    scope: Option<ParallelEpQ8Scope>,
) -> bool {
    q8_active && scope != Some(ParallelEpQ8Scope::Down)
}

fn parse_step_layer_specs(
    flag: &str,
    value: Option<&str>,
    allow_full_model: bool,
) -> Result<Vec<StepEpLayerSpec>, String> {
    let trunk = allow_full_model.then_some(STEP37_TRUNK_LAYERS);
    parse_layer_specs_for_trunk(flag, value, trunk)
}

/// The pure composition-refusal law behind every parallel door's UNPROVEN-pair matrix
/// (hoisted from the glm5 TP door, lane/glm5-extract-general): the first armed flag in
/// `table` refuses by name, BEFORE any parallel CUDA state exists. Each family owns its
/// own TABLE of `(flag, why)` rows — the reasons are gate receipts, part of the law; a
/// pair unlocks only with its own composition gate (the primary flag's FLAGS.md row
/// carries the matrix). `armed` reports whether a flag is set to `"1"` (env in
/// production; a plain set in unit tests — the pattern keeps tests env-mutation-free).
pub(crate) fn refuse_door_composition(
    primary: &str,
    table: &[(&str, &str)],
    armed: impl Fn(&str) -> bool,
) -> Result<(), String> {
    for (flag, why) in table {
        if armed(flag) {
            return Err(format!(
                "{primary} + {flag}: unproven composition, refused ({why})"
            ));
        }
    }
    Ok(())
}

/// The shared `LAYER[-LAYER]@DEVICE,DEVICE[;...]` grammar behind every per-layer parallel
/// door. `full_model_trunk` enables the `all` shorthand and names the trunk it expands to —
/// the caller's model contract owns that constant, never this parser (the step door passes
/// `STEP37_TRUNK_LAYERS`; the glm5 door passes its own trunk length at load time).
pub(crate) fn parse_layer_specs_for_trunk(
    flag: &str,
    value: Option<&str>,
    full_model_trunk: Option<usize>,
) -> Result<Vec<StepEpLayerSpec>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_empty() || value == "0" {
        return Ok(Vec::new());
    }

    let mut specs = Vec::new();
    for item in value.split(';') {
        let (layers, devices) = item.split_once('@').ok_or_else(|| {
            let layers = if full_model_trunk.is_some() {
                "LAYER[-LAYER] or all"
            } else {
                "LAYER[-LAYER]"
            };
            format!("{flag} must be {layers}@DEVICE,DEVICE[;...]")
        })?;
        let (first, last) = if layers == "all" {
            let Some(trunk) = full_model_trunk else {
                return Err(format!(
                    "{flag} does not support the full-model shorthand; assign routed layers \
                     explicitly"
                ));
            };
            (0, trunk - 1)
        } else {
            match layers.split_once('-') {
                Some((first, last)) => {
                    let first = first
                        .parse::<usize>()
                        .map_err(|_| format!("{flag} layer {first:?} is not an integer"))?;
                    let last = last
                        .parse::<usize>()
                        .map_err(|_| format!("{flag} layer {last:?} is not an integer"))?;
                    if first > last {
                        return Err(format!("{flag} layer range {first}-{last} is reversed"));
                    }
                    if last - first + 1 > 128 {
                        return Err(format!(
                            "{flag} layer range {first}-{last} exceeds the 128-layer parser cap"
                        ));
                    }
                    (first, last)
                }
                None => {
                    let layer = layers
                        .parse::<usize>()
                        .map_err(|_| format!("{flag} layer {layers:?} is not an integer"))?;
                    (layer, layer)
                }
            }
        };
        let devices = devices
            .split(',')
            .map(|device| {
                device
                    .parse::<usize>()
                    .map_err(|_| format!("{flag} device {device:?} is not an integer"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !(2..=8).contains(&devices.len()) {
            return Err(format!(
                "{flag} requires 2..=8 devices, got {}",
                devices.len()
            ));
        }
        let mut unique = devices.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != devices.len() {
            return Err(format!("{flag} devices must be distinct, got {devices:?}"));
        }
        for layer in first..=last {
            if specs
                .iter()
                .any(|existing: &StepEpLayerSpec| existing.layer == layer)
            {
                return Err(format!("{flag} assigns layer {layer} more than once"));
            }
            specs.push(StepEpLayerSpec {
                layer,
                devices: devices.clone(),
            });
        }
    }
    Ok(specs)
}

pub fn parse_step_ep_layer_specs(value: Option<&str>) -> Result<Vec<StepEpLayerSpec>, String> {
    parse_step_layer_specs("MEMRA_STEP_EP", value, false)
}

pub fn step_ep_layer_specs() -> Result<Vec<StepEpLayerSpec>, String> {
    parse_step_ep_layer_specs(std::env::var("MEMRA_STEP_EP").ok().as_deref())
}

pub fn parse_step_tp_layer_specs(value: Option<&str>) -> Result<Vec<StepTpLayerSpec>, String> {
    parse_step_layer_specs("MEMRA_STEP_TP", value, true)
}

pub fn step_tp_layer_specs() -> Result<Vec<StepTpLayerSpec>, String> {
    parse_step_tp_layer_specs(std::env::var("MEMRA_STEP_TP").ok().as_deref())
}

#[derive(Clone, Copy)]
pub struct E4m3BlockMatrix<'a> {
    pub codes: &'a [u8],
    pub scales: &'a [f32],
    pub out_features: usize,
    pub in_features: usize,
}

impl E4m3BlockMatrix<'_> {
    fn validate(&self) -> Result<(), String> {
        let code_count = self
            .out_features
            .checked_mul(self.in_features)
            .ok_or_else(|| "E4M3 matrix size overflow".to_string())?;
        if self.codes.len() != code_count {
            return Err(format!(
                "E4M3 code count {} != {}x{} ({code_count})",
                self.codes.len(),
                self.out_features,
                self.in_features,
            ));
        }
        let scale_count =
            self.out_features.div_ceil(FP8_BLOCK) * self.in_features.div_ceil(FP8_BLOCK);
        if self.scales.len() != scale_count {
            return Err(format!(
                "E4M3 scale count {} != {scale_count} for {}x{}",
                self.scales.len(),
                self.out_features,
                self.in_features,
            ));
        }
        if !self
            .scales
            .iter()
            .all(|scale| scale.is_finite() && *scale > 0.0)
        {
            return Err("E4M3 scale grid contains a non-finite or non-positive value".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct E4m3ExpertBank<'a> {
    pub codes: &'a [u8],
    pub scales: &'a [f32],
    pub expert_count: usize,
    pub out_features: usize,
    pub in_features: usize,
}

impl E4m3ExpertBank<'_> {
    fn validate(&self) -> Result<(), String> {
        if self.expert_count == 0 {
            return Err("E4M3 expert bank is empty".to_string());
        }
        let code_stride = self
            .out_features
            .checked_mul(self.in_features)
            .ok_or_else(|| "E4M3 expert code stride overflow".to_string())?;
        let code_count = self
            .expert_count
            .checked_mul(code_stride)
            .ok_or_else(|| "E4M3 expert code count overflow".to_string())?;
        if self.codes.len() != code_count {
            return Err(format!(
                "E4M3 expert code count {} != {}x{} ({code_count})",
                self.codes.len(),
                self.expert_count,
                code_stride,
            ));
        }
        let scale_stride =
            self.out_features.div_ceil(FP8_BLOCK) * self.in_features.div_ceil(FP8_BLOCK);
        let scale_count = self
            .expert_count
            .checked_mul(scale_stride)
            .ok_or_else(|| "E4M3 expert scale count overflow".to_string())?;
        if self.scales.len() != scale_count {
            return Err(format!(
                "E4M3 expert scale count {} != {}x{} ({scale_count})",
                self.scales.len(),
                self.expert_count,
                scale_stride,
            ));
        }
        if !self
            .scales
            .iter()
            .all(|scale| scale.is_finite() && *scale > 0.0)
        {
            return Err(
                "E4M3 expert scale grid contains a non-finite or non-positive value".to_string(),
            );
        }
        Ok(())
    }

    pub fn expert(&self, expert: usize) -> Result<E4m3BlockMatrix<'_>, String> {
        if expert >= self.expert_count {
            return Err(format!("expert {expert} outside 0..{}", self.expert_count));
        }
        let code_stride = self.out_features * self.in_features;
        let scale_stride =
            self.out_features.div_ceil(FP8_BLOCK) * self.in_features.div_ceil(FP8_BLOCK);
        Ok(E4m3BlockMatrix {
            codes: &self.codes[expert * code_stride..(expert + 1) * code_stride],
            scales: &self.scales[expert * scale_stride..(expert + 1) * scale_stride],
            out_features: self.out_features,
            in_features: self.in_features,
        })
    }
}

pub struct ColumnParallelResult {
    pub gathered: Vec<f32>,
    pub rank_outputs: Vec<Vec<f32>>,
}

pub struct RowParallelResult {
    pub reduced: Vec<f32>,
    pub rank_partials: Vec<Vec<f32>>,
}

#[derive(Clone, Copy)]
pub struct Bf16Matrix<'a> {
    pub bytes: &'a [u8],
    pub out_features: usize,
    pub in_features: usize,
}

impl Bf16Matrix<'_> {
    pub fn validate(&self) -> Result<(), String> {
        if self.out_features == 0 || self.in_features == 0 {
            return Err("BF16 matrix dimensions must be nonzero".into());
        }
        let expected = self
            .out_features
            .checked_mul(self.in_features)
            .and_then(|values| values.checked_mul(2))
            .ok_or("BF16 matrix byte count overflow")?;
        if self.bytes.len() != expected {
            return Err(format!(
                "BF16 matrix bytes {} != {}x{}x2 ({expected})",
                self.bytes.len(),
                self.out_features,
                self.in_features,
            ));
        }
        Ok(())
    }
}

struct ResidentE4m3Rank {
    codes: CudaSlice<u8>,
    scales: CudaSlice<f32>,
    out_features: usize,
    in_features: usize,
}

enum ResidentBf16Weight {
    Bf16(CudaSlice<u8>),
    F32(CudaSlice<f32>),
}

impl ResidentBf16Weight {
    fn ordinal(&self) -> usize {
        match self {
            Self::Bf16(bytes) => bytes.ordinal(),
            Self::F32(values) => values.ordinal(),
        }
    }
}

struct ResidentBf16Rank {
    weight: ResidentBf16Weight,
    out_features: usize,
    in_features: usize,
    /// q8_0 mirror built at load under MEMRA_STEP_TP_W8 (numeric-class door; the bf16 slab
    /// stays resident because every prefill/verify path is qualified against it).
    q8: Option<CudaSlice<u8>>,
}

pub struct ResidentColumnParallel {
    ranks: Vec<ResidentE4m3Rank>,
    out_features: usize,
    in_features: usize,
}

pub struct ResidentRowParallel {
    ranks: Vec<ResidentE4m3Rank>,
    out_features: usize,
    in_features: usize,
}

pub struct ResidentBf16ColumnParallel {
    ranks: Vec<ResidentBf16Rank>,
    out_features: usize,
    in_features: usize,
    canonical_chunk_rows: Option<usize>,
}

pub struct ResidentBf16RowParallel {
    ranks: Vec<ResidentBf16Rank>,
    out_features: usize,
    in_features: usize,
}

pub struct ResidentStepBf16RowParallel {
    ranks: Vec<Vec<ResidentBf16Rank>>,
    out_features: usize,
    in_features: usize,
    canonical_chunk_cols: usize,
}

/// Root-owned BF16 sigmoid router with persistent F32 weight, bias, and active mask.
pub struct ResidentSigmoidTopKRouter {
    weight: CudaSlice<f32>,
    correction_bias: CudaSlice<f32>,
    active: CudaSlice<u8>,
    root_device: usize,
    input_width: usize,
    expert_count: usize,
    experts_per_token: usize,
    active_count: usize,
    scaling_factor: f32,
    route_norm: bool,
}

pub struct SigmoidTopKHostOutput {
    pub logits: Vec<f32>,
    pub selected: Vec<u32>,
    pub weights: Vec<f32>,
}

/// Full BF16 SwiGLU weights replicated independently on every runtime rank.
pub struct ResidentReplicatedBf16SwiGlu {
    gate: Vec<ResidentBf16Rank>,
    up: Vec<ResidentBf16Rank>,
    down: Vec<ResidentBf16Rank>,
    input_width: usize,
    intermediate_width: usize,
}

/// One token-major F32 batch replicated across a native-P2P rank group.
///
/// Every allocation is owned by its matching rank CUDA context. This is the generic handoff
/// substrate between independently sharded operators; it carries no model or topology claim.
pub struct ResidentReplicatedDeviceRows {
    ranks: Vec<CudaSlice<f32>>,
    tokens: usize,
    width: usize,
}

impl ResidentReplicatedDeviceRows {
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn ranks(&self) -> usize {
        self.ranks.len()
    }
}

/// Canonical MoE output order: routed plus shared, then add the layer residual.
pub fn moe_residual_host(
    residual: &[f32],
    routed: &[f32],
    shared: &[f32],
) -> Result<Vec<f32>, String> {
    if residual.len() != routed.len() || residual.len() != shared.len() {
        return Err(format!(
            "MoE residual lengths residual={} routed={} shared={}",
            residual.len(),
            routed.len(),
            shared.len()
        ));
    }
    let ffn = routed
        .iter()
        .zip(shared)
        .map(|(&routed, &shared)| routed + shared)
        .collect::<Vec<_>>();
    Ok(residual
        .iter()
        .zip(ffn)
        .map(|(&residual, ffn)| residual + ffn)
        .collect())
}

pub use memra_kv::{
    KvRingAppend, ResidentTpKvCache, ResidentTpKvCacheRank, TpKvAppendPlan, TpKvTransaction,
};

/// Persistent TP2/TP4/TP8 routed-expert reference.
///
/// Rank-local checkpoint shards are uploaded once and remain tied to their owning CUDA context.
/// Activations and deterministic host-staged collectives remain per invocation. This is the
/// correctness substrate for serving TP/EP, not product-throughput evidence.
pub struct ResidentTpExpert {
    gate: ResidentColumnParallel,
    up: ResidentColumnParallel,
    down: ResidentRowParallel,
    input_width: usize,
    expert_width: usize,
}

struct ResidentE4m3ExpertBankRank {
    codes: CudaSlice<u8>,
    scales: CudaSlice<f32>,
    expert_range: Range<usize>,
    out_features: usize,
    in_features: usize,
    code_stride: usize,
    scale_stride: usize,
    /// TP row banks are packed by native 128-wide K block so reduction can replay the
    /// checkpoint's global block order exactly. Other banks remain row-major.
    k_blocks: Option<usize>,
}

struct PackedE4m3ExpertBankRank {
    codes: Vec<u8>,
    scales: Vec<f32>,
    expert_range: Range<usize>,
    out_features: usize,
    in_features: usize,
    code_stride: usize,
    scale_stride: usize,
    k_blocks: Option<usize>,
}

struct ResidentEpRank {
    gate: ResidentE4m3ExpertBankRank,
    up: ResidentE4m3ExpertBankRank,
    down: ResidentE4m3ExpertBankRank,
}

/// Persistent expert-parallel reference.
///
/// Every routed expert has exactly one owner rank. Shared experts are deliberately absent from
/// this object because Step replicates them per rank. Routes execute on the owner CUDA context.
/// The default oracle stages through host memory; the native path peer-dispatches inputs and
/// peer-returns owner outputs while preserving host-canonical activation and accumulation.
pub struct ResidentExpertParallel {
    ranks: Vec<ResidentEpRank>,
    expert_count: usize,
    input_width: usize,
    expert_width: usize,
}

/// Projection-level output from the opt-in official Step grouped-FP8 gate.
///
/// Rows remain pair-major. Routing, weighted combine, and production integration are deliberately
/// outside this gate-only adapter.
pub struct StepGroupedFp8ProjectionOutput {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

/// Prepared official Step grouped-FP8 projection gate.
///
/// The complete tensor banks, both CSR schedules, input, activation buffer, and three projection
/// workspaces are uploaded or allocated once. Repeated execution performs no device allocation.
pub struct PreparedStepGroupedFp8Gate {
    device: usize,
    gate: ResidentE4m3ExpertBankRank,
    up: ResidentE4m3ExpertBankRank,
    down: ResidentE4m3ExpertBankRank,
    input: CudaSlice<f32>,
    route_csr: DeviceExpertCsr,
    down_csr: DeviceExpertCsr,
    gate_workspace: Fp8GroupedWorkspace,
    up_workspace: Fp8GroupedWorkspace,
    down_workspace: Fp8GroupedWorkspace,
    activation: CudaSlice<f32>,
    activation_limit: Option<f32>,
    tokens: usize,
    pairs: usize,
}

impl PreparedStepGroupedFp8Gate {
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn pairs(&self) -> usize {
        self.pairs
    }
}

struct PreparedStepGroupedExpertOwner {
    rank: usize,
    global_pairs: Vec<usize>,
    route_csr: DeviceExpertCsr,
    down_csr: DeviceExpertCsr,
    gate_workspace: Fp8GroupedWorkspace,
    up_workspace: Fp8GroupedWorkspace,
    down_workspace: Fp8GroupedWorkspace,
    activation: CudaSlice<f32>,
}

struct StepGroupedExpertOwnerSchedule {
    global_pairs: Vec<usize>,
    route_csr: ExpertCsr,
    down_csr: ExpertCsr,
}

/// Prepared official Step expert-owner grouped-FP8 projection gate.
///
/// Route partitioning, owner-local CSR uploads, input dispatch, activation buffers, and grouped
/// workspaces are persistent. Projection rows are scattered back to canonical pair order only
/// after every owner has completed its rank-local program.
pub struct PreparedStepGroupedExpertParallelGate {
    rank_inputs: Vec<CudaSlice<f32>>,
    owners: Vec<PreparedStepGroupedExpertOwner>,
    activation_limit: Option<f32>,
    tokens: usize,
    pairs: usize,
    max_tokens: usize,
    max_pairs: usize,
    input_width: usize,
    expert_width: usize,
    generation: u64,
    executed_generation: Option<u64>,
    ready: bool,
}

impl PreparedStepGroupedExpertParallelGate {
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn pairs(&self) -> usize {
        self.pairs
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn input_width(&self) -> usize {
        self.input_width
    }

    pub fn expert_width(&self) -> usize {
        self.expert_width
    }

    pub fn set_activation_limit(&mut self, limit: Option<f32>) -> Result<(), String> {
        validate_step_expert_activation_limit(limit)?;
        self.activation_limit = limit;
        self.executed_generation = None;
        Ok(())
    }

    pub fn active_owners(&self) -> usize {
        self.owners
            .iter()
            .filter(|owner| !owner.global_pairs.is_empty())
            .count()
    }

    pub fn owner_pair_counts(&self) -> Vec<usize> {
        self.owners
            .iter()
            .map(|owner| owner.global_pairs.len())
            .collect()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}

struct PreparedPeerWeightedRouteOwner {
    token_rows: CudaSlice<i32>,
    slots: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    active_pairs: usize,
}

/// Persistent root-side weighted combine for peer-owned canonical route rows.
///
/// Owner metadata, one reusable peer staging buffer, the canonical slot bank, weight bank, and
/// output are allocated once. Refreshes update metadata prefixes; execution peer-copies active
/// rows, scatters them by canonical token/slot, and reduces in the requested numeric order.
pub struct PreparedPeerWeightedRouteCombine {
    root_device: usize,
    owners: Vec<PreparedPeerWeightedRouteOwner>,
    peer_staging: CudaSlice<f32>,
    slots: CudaSlice<f32>,
    weights: CudaSlice<f32>,
    output: CudaSlice<f32>,
    peer_devices: Vec<usize>,
    peer_outputs: Vec<CudaSlice<f32>>,
    width: usize,
    experts_per_token: usize,
    max_tokens: usize,
    max_pairs: usize,
    tokens: usize,
    pairs: usize,
    projection_generation: u64,
    output_generation: Option<u64>,
    broadcast_generation: Option<u64>,
    ready: bool,
}

impl PreparedPeerWeightedRouteCombine {
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn pairs(&self) -> usize {
        self.pairs
    }

    pub fn owner_pair_counts(&self) -> Vec<usize> {
        self.owners.iter().map(|owner| owner.active_pairs).collect()
    }

    pub fn distributed_ranks(&self) -> usize {
        1 + self.peer_outputs.len()
    }
}

struct ResidentTpExpertBank {
    gate: Vec<ResidentE4m3ExpertBankRank>,
    up: Vec<ResidentE4m3ExpertBankRank>,
    down: Vec<ResidentE4m3ExpertBankRank>,
    expert_count: usize,
    input_width: usize,
    expert_width: usize,
}

/// Persistent tensor-parallel expert bank.
///
/// Every rank owns a checkpoint-aligned output-row shard of every gate/up projection and an
/// input-column shard of every down projection. Activations cross deterministic host-staged
/// collectives on hosts where native peer copies are unavailable or corrupt.
pub struct ResidentTensorParallel {
    bank: ResidentTpExpertBank,
}

/// Multi-context TP correctness runtime. Each rank owns an independent `Engine` and CUDA context.
///
/// Host bounce is the default oracle. Native P2P is opt-in and preserves the oracle's global
/// checkpoint-block reduction order; it remains a correctness path until serving gates and
/// repeated performance evidence qualify it.
pub struct TpE4m3HostBounce {
    devices: Vec<usize>,
    ranks: Vec<Engine>,
    native_p2p: bool,
    ep_device_arithmetic: bool,
    bulk_p2p: bool,
    /// v2 decode-attention workspace (MEMRA_STEP_TP_DECODE_V2). One per runtime, shared by
    /// every TP attention layer — the buffer shapes are geometry-constant across the trunk.
    decode_v2: std::sync::Mutex<Vec<StepTpDecodeV2Ws>>,
}

pub struct TpKvVerifiedLayer<'a> {
    pub cache: &'a mut ResidentTpKvCache,
    pub start: usize,
    pub logical_len: usize,
    pub source_k_raw: u64,
    pub source_v_raw: u64,
    pub source_k_tok_bytes: usize,
    pub source_v_tok_bytes: usize,
}

/// Persistent two-rank all-reduce for one replicated f32 row.
///
/// Each rank pushes its local partial directly into a peer-resident staging row, records one
/// reusable event, waits for the peer's corresponding event, then adds `(rank0, rank1)` in that
/// same operand order on both devices. Two staging rows per direction are enough because join
/// `j + 1` is ordered after each rank's add at join `j`, so overwriting parity `j` cannot race its
/// peer consumer. The collective allocates nothing and performs no host synchronization per call.
pub struct Tp2ReplicatedRowJoin {
    width: usize,
    parity: usize,
    stage0: [CudaSlice<f32>; 2],
    stage1: [CudaSlice<f32>; 2],
    stage0_raw: [u64; 2],
    stage1_raw: [u64; 2],
    event0: [CudaEvent; 2],
    event1: [CudaEvent; 2],
}

fn validate_tp2_replicated_row_join(
    ranks: usize,
    native_p2p: bool,
    width: usize,
) -> Result<(), String> {
    if ranks != 2 {
        return Err(format!(
            "replicated-row join requires exactly two ranks, got {ranks}"
        ));
    }
    if !native_p2p {
        return Err("replicated-row join requires native P2P".into());
    }
    if width == 0 || width > i32::MAX as usize {
        return Err(format!(
            "replicated-row join width must be in 1..={}, got {width}",
            i32::MAX
        ));
    }
    Ok(())
}

fn launch_tp2_peer_push(
    engine: &Engine,
    source: &CudaSlice<f32>,
    destination: u64,
    width: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let function = engine.func("q4e_push_f32");
    let config = LaunchConfig::for_num_elems(width as u32);
    let width = width as i64;
    let stream = engine.gpu.stream();
    let mut launch = stream.launch_builder(&function);
    launch.arg(source).arg(&destination).arg(&width);
    unsafe {
        launch.launch(config)?;
    }
    Ok(())
}

/// Persistent workspace of the v2 rank-local decode-attention driver.
///
/// Buffers live in their producing rank's CUDA context, are never freed, and events are
/// re-recorded per call — the pp.rs `BoundarySlot` discipline — so the per-token path has no
/// cuMemAlloc, no cross-stream free, and no host round-trip. Every buffer is fully overwritten
/// before its consumers run in the same call; nothing carries state between tokens.
/// Per-rank attn_gate row shards for the fused QKV+gate kernel, in the weight class the
/// fused kernels read (F32 mirror or raw checkpoint bf16).
pub enum StepTpGateShards<'a> {
    F32(&'a [crate::CudaSlice<f32>]),
    Bf16(&'a [crate::CudaSlice<u8>]),
}

pub struct StepTpDecodeV2Ws {
    /// T-COLUMN verify slabs (spec MTP): per-rank [t, local_dim] projections computed by
    /// the weight-amortized qkvg_tcol kernel; the col-select door copies one column into
    /// the single-row buffers and everything downstream runs the unmodified t=1 program.
    pub(crate) tcol_q: Vec<CudaSlice<f32>>,
    pub(crate) tcol_k: Vec<CudaSlice<f32>>,
    pub(crate) tcol_v: Vec<CudaSlice<f32>>,
    pub(crate) tcol_g: Vec<CudaSlice<f32>>,
    pub(crate) tcol_in: Vec<CudaSlice<f32>>,
    pub(crate) tcol_cap: usize,
    /// MEMRA_STEP_TP_W8 activation scratch: per-rank q8_1 quantized attention input
    /// ([in_f] i8 + one f32 scale pair per 32). Persistent because the alternative is an
    /// allocation per rank per layer per token.
    w8_aq: Vec<CudaSlice<i8>>,
    w8_ad: Vec<CudaSlice<f32>>,
    w8_in: usize,
    /// o_proj-side twin of the same scratch (its activation is the gated attention output,
    /// a different vector from the QKV input, so it needs its own buffers).
    w8o_aq: Vec<CudaSlice<i8>>,
    w8o_ad: Vec<CudaSlice<f32>>,
    w8o_in: usize,
    /// VERIFY-WALK q8_1 activation scratch, t columns wide (the decode scratch above is one
    /// row). Two sets because the QKV input and the gated attention output are different
    /// vectors of different widths.
    w8t_aq: Vec<CudaSlice<i8>>,
    w8t_ad: Vec<CudaSlice<f32>>,
    w8t_in: usize,
    w8t_oaq: Vec<CudaSlice<i8>>,
    w8t_oad: Vec<CudaSlice<f32>>,
    w8t_oin: usize,
    w8t_cap: usize,
    /// MEMRA_TCOL_OPROJ slabs: per-rank stashed `gated` rows ([8, local_q_dim]), per-rank
    /// b4_tcol partials ([8, o_out]), a root-side peer pull of rank1's partial slab, and
    /// the root-side joined `mixed` slab. Armed lazily by the first stash.
    /// MEMRA_SPEC_FA2 slabs: per-rank stashed post-rope q rows ([2, local_q_dim]), gate
    /// rows ([2, heads/ranks]) and the two gated outputs the per-row combine writes
    /// ([2, local_q_dim]). Armed lazily by the first stash.
    pub(crate) fa2_q: Vec<CudaSlice<f32>>,
    pub(crate) fa2_gate: Vec<CudaSlice<f32>>,
    pub(crate) fa2_gated: Vec<CudaSlice<f32>>,
    pub(crate) fa2_cap: usize,
    /// T-ROW rope/append twin scratch: per-rank roped-k rows ([8, local_kv]), per-row
    /// last-block counters ([8]) and the per-tick position slab ([8]). Armed with the
    /// fa2 slabs.
    rope_k_t: Vec<CudaSlice<f32>>,
    rope_ctr_t: Vec<CudaSlice<u32>>,
    rope_pos_t: Vec<CudaSlice<i32>>,
    /// Per-rank combined 6-word row tables, keyed by the caller's (layer, session-set,
    /// base-arming) signature. LEGACY: only the `MEMRA_ROWS_TAB_RESTAGE=0` rollback arm
    /// reads this. See `rows_tab_t` for why the key cannot be made safe.
    rows_tabs: Vec<std::collections::HashMap<u64, CudaSlice<u64>>>,
    /// Per-rank PERSISTENT 6-word row-table slab ([32, 6] u64), RESTAGED from the live
    /// distributed cache before every launch. Replaces the `rows_tabs` memo, whose key was
    /// a hash of (k pointer, base pointer, layer, t) while the table it returned also
    /// carried the V and LEN pointers: a session whose K buffer address was recycled hit
    /// another session's table and the append kernel wrote its K/V through the FREED
    /// pointers the entry still held. Same defect and same cure as the row-table twin in
    /// `step35_verify_fa_rows_join` (8c8397e0b2, Hermes `11339f5cd3c132a3`), which this
    /// path was left out of. One 32-word htod per rank per layer replaces the map lookup;
    /// no allocation, and the staging is stream-ordered exactly like `rope_pos_t`.
    rows_tab_t: Vec<CudaSlice<u64>>,
    /// HOST shadow of the last table staged under each retired memo key, used ONLY by
    /// `MEMRA_ROWS_TAB_STALE_SCAN=1` to prove that the retired key would have handed a live
    /// launch another allocation's pointers. Never read by a kernel.
    rows_tab_shadow: Vec<std::collections::HashMap<u64, Vec<u64>>>,
    tcol_gated: Vec<CudaSlice<f32>>,
    tcol_opart: Vec<CudaSlice<f32>>,
    tcol_opeer: Option<CudaSlice<f32>>,
    tcol_omix: Option<CudaSlice<f32>>,
    tcol_ocap: usize,
    // rank-context buffers, indexed by rank (pub(crate): the v2 driver in hybrid_forward
    // feeds them to the KV transaction and attention kernels between the two v2 phases)
    pub(crate) q_raw: Vec<CudaSlice<f32>>,
    pub(crate) k_raw: Vec<CudaSlice<f32>>,
    pub(crate) v_raw: Vec<CudaSlice<f32>>,
    pub(crate) q: Vec<CudaSlice<f32>>,
    pub(crate) k: Vec<CudaSlice<f32>>,
    pub(crate) pos: Vec<CudaSlice<i32>>,
    /// FUSION #1 last-block counters (one per rank; atomicInc auto-resets per launch).
    pub(crate) fuse_ctr: Vec<CudaSlice<u32>>,
    pub(crate) gate: Vec<CudaSlice<f32>>,
    pub(crate) attn_out: Vec<CudaSlice<f32>>,
    pub(crate) gated: Vec<CudaSlice<f32>>,
    /// [rank][block] O partials, each `o_out` wide, in the owning rank's context.
    o_partials: Vec<Vec<CudaSlice<f32>>>,
    /// Stable workspace pointers for the rank-done-fenced raw P2P gather. Safe
    /// `memcpy_dtod` creates a fresh source event for every cross-context copy; the v2
    /// driver already records one persistent `ev_rank` after all three source families.
    raw_o_partials: Vec<Vec<u64>>,
    raw_k: Vec<u64>,
    raw_v_raw: Vec<u64>,
    /// Recorded on each rank's stream after its per-call work; root waits before peer reads.
    ev_rank: Vec<CudaEvent>,
    // root-context buffers
    peer_partial: CudaSlice<f32>,
    reduce_a: CudaSlice<f32>,
    reduce_b: CudaSlice<f32>,
    /// Never written; the canonical zero start of the v1 add chain.
    zeros: CudaSlice<f32>,
    pub(crate) k_shadow: CudaSlice<f32>,
    pub(crate) v_shadow: CudaSlice<f32>,
    ev_refresh: CudaEvent,
    ev_oproj: CudaEvent,
    // model-engine (e) context
    gate_e: CudaSlice<f32>,
    /// Per-token stages (e-ctx, fixed addresses): one eager e-stream copy each per layer; the
    /// rank flows raw-copy FROM them, which is exactly the shape graph capture needs.
    pub(crate) h_stage: Option<CudaSlice<f32>>,
    pub(crate) pos_stage: Option<CudaSlice<i32>>,
    /// Workspace-owned per-rank attention input rows (the stage flow copies into THESE, not
    /// the per-layer decode_input buffers — the workspace is shared across layers, so every
    /// captured/raw address it uses must be layer-invariant).
    attn_in: Vec<CudaSlice<f32>>,
    /// Cached raw pointers of the stage-flow operands (set when the stages arm).
    raw_h_stage: u64,
    raw_pos_stage: u64,
    raw_attn_in: Vec<u64>,
    raw_pos: Vec<u64>,
    raw_o_partial1: u64,
    raw_peer_partial: u64,
    raw_k1: u64,
    raw_v1: u64,
    raw_k_shadow: u64,
    raw_v_shadow: u64,
    /// Token-graph e-context mirrors (armed by the orchestrator): the root section
    /// raw-copies the reduced attention output and the shadow rows here so the e-glue
    /// children read same-context memory (cross-context kernel args are capture-illegal).
    raw_mixed_stage_e: u64,
    raw_reduce_a: u64,
    raw_shadow_stage_e: (u64, u64),
    ev_entry: CudaEvent,
    e_device: usize,
    // geometry pins
    local_q_dim: usize,
    local_kv_dim: usize,
    heads: usize,
    pub(crate) o_out: usize,
    o_block_cols: usize,
    blocks_per_rank: usize,
}

impl TpE4m3HostBounce {
    pub fn new(devices: &[usize]) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_inner(devices, false, false, false, false)
    }

    pub fn new_native_p2p(devices: &[usize]) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_inner(devices, false, true, false, false)
    }

    pub fn new_native_p2p_device_arithmetic(
        devices: &[usize],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_inner(devices, false, true, true, false)
    }

    pub(crate) fn new_configured(
        devices: &[usize],
        native_p2p: bool,
        ep_device_arithmetic: bool,
        bulk_p2p: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_inner(devices, false, native_p2p, ep_device_arithmetic, bulk_p2p)
    }

    /// Single-rank execution of the canonical checkpoint-block TP program.
    ///
    /// This is an oracle for distributed exactness, not a serving topology. It lets gates compare
    /// TP=1 and TP>1 with the same packing, kernel launches, and deterministic reduction order.
    pub fn new_single_rank_oracle(device: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_inner(&[device], true, false, false, false)
    }

    fn new_inner(
        devices: &[usize],
        allow_single_rank: bool,
        native_p2p: bool,
        ep_device_arithmetic: bool,
        bulk_p2p: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if ep_device_arithmetic && !native_p2p {
            return Err("device-resident EP arithmetic requires native P2P".into());
        }
        if bulk_p2p && !native_p2p {
            return Err("bulk TP transport requires native P2P".into());
        }
        let minimum = if allow_single_rank { 1 } else { 2 };
        if !(minimum..=8).contains(&devices.len()) {
            return Err(format!(
                "TP reference requires {minimum}..=8 devices, got {}",
                devices.len()
            )
            .into());
        }
        let mut unique = devices.to_vec();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != devices.len() {
            return Err(format!("TP devices must be distinct, got {devices:?}").into());
        }
        let ranks = devices
            .iter()
            .map(|&device| Engine::new(device))
            .collect::<Result<Vec<_>, _>>()?;
        if native_p2p {
            configure_native_p2p(&ranks, devices)?;
        }
        if allow_single_rank {
            eprintln!(
                "[tp] canonical oracle transport=local device={} performance_claim=false",
                devices[0]
            );
        } else if native_p2p {
            if ep_device_arithmetic {
                eprintln!(
                    "[tp] correctness transport=native-p2p devices={devices:?} \
                     native_p2p=true activation=device-host-exact \
                     accumulation=device-host-exact output=root-readback \
                     bulk_p2p={bulk_p2p} performance_claim=false"
                );
            } else {
                eprintln!(
                    "[tp] correctness transport=native-p2p devices={devices:?} \
                     native_p2p=true activation=host-canonical bulk_p2p={bulk_p2p} \
                     performance_claim=false"
                );
            }
        } else {
            eprintln!(
                "[tp] correctness transport=host-bounce devices={devices:?} \
                 native_p2p=false performance_claim=false"
            );
        }
        Ok(Self {
            devices: devices.to_vec(),
            ranks,
            native_p2p,
            ep_device_arithmetic,
            bulk_p2p,
            decode_v2: std::sync::Mutex::new(Vec::new()),
        })
    }

    pub fn devices(&self) -> &[usize] {
        &self.devices
    }

    pub fn native_p2p(&self) -> bool {
        self.native_p2p
    }

    pub fn bulk_p2p(&self) -> bool {
        self.bulk_p2p
    }

    pub fn expert_activation_label(&self) -> &'static str {
        if self.ep_device_arithmetic {
            "device-host-exact"
        } else {
            "host-canonical"
        }
    }

    pub fn expert_accumulation_label(&self) -> &'static str {
        self.expert_activation_label()
    }

    pub fn expert_output_label(&self) -> &'static str {
        if self.ep_device_arithmetic {
            "root-readback"
        } else {
            "host-accumulated"
        }
    }

    pub fn transport_label(&self) -> &'static str {
        if self.devices.len() == 1 {
            "local"
        } else if self.native_p2p {
            "native-p2p"
        } else {
            "host-bounce"
        }
    }

    pub fn device_names(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        self.ranks
            .iter()
            .map(|rank| rank.ctx().name().map_err(Into::into))
            .collect()
    }

    /// Correctness-gate access to the engine that owns one TP rank.
    ///
    /// Model execution should prefer collective methods on this runtime. This accessor exists so
    /// focused gates can prove that the rank-local projection outputs remain device-resident
    /// through the next ownership boundary before that boundary is wired into serving.
    pub fn rank_engine(&self, rank: usize) -> Option<&Engine> {
        self.ranks.get(rank)
    }

    /// Allocate the persistent ping-pong staging and event state for a two-rank replicated-row
    /// all-reduce. The caller owns one instance per concurrently live collective sequence.
    pub fn prepare_tp2_replicated_row_join(
        &self,
        width: usize,
    ) -> Result<Tp2ReplicatedRowJoin, Box<dyn std::error::Error>> {
        validate_tp2_replicated_row_join(self.ranks.len(), self.native_p2p, width)?;
        let rank0 = &self.ranks[0];
        let rank1 = &self.ranks[1];

        let (stage0, stage0_raw, event0) = {
            let _main = rank0.gpu.enter_main()?;
            let stage = [rank0.zeros(width)?, rank0.zeros(width)?];
            let stream = rank0.gpu.stream();
            let raw = [
                stage[0].device_ptr(&stream).0,
                stage[1].device_ptr(&stream).0,
            ];
            let events = [rank0.ctx().new_event(None)?, rank0.ctx().new_event(None)?];
            (stage, raw, events)
        };
        let (stage1, stage1_raw, event1) = {
            let _main = rank1.gpu.enter_main()?;
            let stage = [rank1.zeros(width)?, rank1.zeros(width)?];
            let stream = rank1.gpu.stream();
            let raw = [
                stage[0].device_ptr(&stream).0,
                stage[1].device_ptr(&stream).0,
            ];
            let events = [rank1.ctx().new_event(None)?, rank1.ctx().new_event(None)?];
            (stage, raw, events)
        };
        Ok(Tp2ReplicatedRowJoin {
            width,
            parity: 0,
            stage0,
            stage1,
            stage0_raw,
            stage1_raw,
            event0,
            event1,
        })
    }

    /// Sum one rank-local partial from each of two ranks and publish the same canonical
    /// `(rank0 + rank1)` row on both devices. The method only enqueues work; subsequent work on
    /// each rank's stream consumes its corresponding output without a host fence.
    pub fn tp2_replicated_row_join(
        &self,
        join: &mut Tp2ReplicatedRowJoin,
        partial0: &CudaSlice<f32>,
        partial1: &CudaSlice<f32>,
        output0: &mut CudaSlice<f32>,
        output1: &mut CudaSlice<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        validate_tp2_replicated_row_join(self.ranks.len(), self.native_p2p, join.width)?;
        let rank0 = &self.ranks[0];
        let rank1 = &self.ranks[1];
        let width = join.width;
        if partial0.len() < width
            || partial1.len() < width
            || output0.len() < width
            || output1.len() < width
            || partial0.ordinal() != rank0.ctx().ordinal()
            || output0.ordinal() != rank0.ctx().ordinal()
            || partial1.ordinal() != rank1.ctx().ordinal()
            || output1.ordinal() != rank1.ctx().ordinal()
        {
            return Err("replicated-row join buffer geometry or ownership mismatch".into());
        }

        let parity = join.parity;
        {
            let _main = rank0.gpu.enter_main()?;
            launch_tp2_peer_push(rank0, partial0, join.stage1_raw[parity], width)?;
            join.event0[parity].record(&rank0.gpu.stream())?;
        }
        {
            let _main = rank1.gpu.enter_main()?;
            launch_tp2_peer_push(rank1, partial1, join.stage0_raw[parity], width)?;
            join.event1[parity].record(&rank1.gpu.stream())?;
        }
        {
            let _main = rank0.gpu.enter_main()?;
            rank0.gpu.stream().wait(&join.event1[parity])?;
            rank0.add(partial0, &join.stage0[parity], output0, width)?;
        }
        {
            let _main = rank1.gpu.enter_main()?;
            rank1.gpu.stream().wait(&join.event0[parity])?;
            rank1.add(&join.stage1[parity], partial1, output1, width)?;
        }
        join.parity ^= 1;
        Ok(())
    }

    pub fn allocate_tp_kv_cache(
        &self,
        kv_dim_k: usize,
        kv_dim_v: usize,
        capacity: usize,
    ) -> Result<ResidentTpKvCache, Box<dyn std::error::Error>> {
        self.allocate_tp_kv_cache_inner(kv_dim_k, kv_dim_v, capacity, None)
    }

    pub fn allocate_tp_swa_kv_cache(
        &self,
        kv_dim_k: usize,
        kv_dim_v: usize,
        capacity: usize,
        window: usize,
    ) -> Result<ResidentTpKvCache, Box<dyn std::error::Error>> {
        if window == 0 {
            return Err("TP SWA KV window must be nonzero".into());
        }
        self.allocate_tp_kv_cache_inner(kv_dim_k, kv_dim_v, capacity, Some(window))
    }

    fn allocate_tp_kv_cache_inner(
        &self,
        kv_dim_k: usize,
        kv_dim_v: usize,
        capacity: usize,
        window: Option<usize>,
    ) -> Result<ResidentTpKvCache, Box<dyn std::error::Error>> {
        if capacity == 0 || capacity > i32::MAX as usize {
            return Err(
                format!("TP KV capacity must be in 1..={}, got {capacity}", i32::MAX).into(),
            );
        }
        let tp = self.ranks.len();
        let shape = crate::cache::tp_kv_rank_allocation_shape(kv_dim_k, kv_dim_v, tp)?;
        let physical_rows = window
            .map(|window| crate::cache::swa_ring_rows(window, capacity))
            .unwrap_or(capacity);
        let k_plane_bytes = physical_rows
            .checked_mul(shape.k_token_bytes)
            .and_then(|bytes| bytes.checked_add(8))
            .ok_or("TP KV K plane-byte overflow")?;
        let v_plane_bytes = physical_rows
            .checked_mul(shape.v_token_bytes)
            .and_then(|bytes| bytes.checked_add(8))
            .ok_or("TP KV V plane-byte overflow")?;
        let mut ranks = Vec::with_capacity(tp);
        for engine in &self.ranks {
            let _main = engine.gpu.enter_main()?;
            ranks.push(ResidentTpKvCacheRank::new(
                engine.alloc_u8(k_plane_bytes)?,
                engine.alloc_u8(v_plane_bytes)?,
                engine.htod_i32(&[0])?,
            ));
        }
        Ok(match window {
            Some(window) => ResidentTpKvCache::new_swa(
                ranks,
                shape.kv_dim_k,
                shape.kv_dim_v,
                shape.k_token_bytes,
                shape.v_token_bytes,
                capacity,
                window,
            ),
            None => ResidentTpKvCache::new(
                ranks,
                shape.kv_dim_k,
                shape.kv_dim_v,
                shape.k_token_bytes,
                shape.v_token_bytes,
                capacity,
            ),
        })
    }

    pub fn grow_tp_kv_cache(
        &self,
        source: &ResidentTpKvCache,
        target_capacity: usize,
        rows: usize,
    ) -> Result<ResidentTpKvCache, Box<dyn std::error::Error>> {
        self.validate_tp_kv_cache(source)?;
        let plan = source.prepare_grow(target_capacity, rows)?;
        let ranks = self.ranks.len();
        let global_k = source
            .kv_dim_k()
            .checked_mul(ranks)
            .ok_or("TP KV grow global K dimension overflow")?;
        let global_v = source
            .kv_dim_v()
            .checked_mul(ranks)
            .ok_or("TP KV grow global V dimension overflow")?;
        let mut target = match source.ring_window() {
            Some(window) => {
                self.allocate_tp_swa_kv_cache(global_k, global_v, target_capacity, window)?
            }
            None => self.allocate_tp_kv_cache(global_k, global_v, target_capacity)?,
        };
        self.validate_tp_kv_cache(&target)?;

        for (rank, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            let src = source
                .rank(rank)
                .ok_or_else(|| format!("TP KV grow source has no rank {rank}"))?;
            let dst = target
                .rank_mut(rank)
                .ok_or_else(|| format!("TP KV grow target has no rank {rank}"))?;
            if plan.k_bytes() > 0 {
                engine.copy_u8_range_into(
                    dst.k_mut(),
                    0,
                    src.k(),
                    plan.source_row() * source.k_tok_bytes(),
                    plan.k_bytes(),
                )?;
            }
            if plan.v_bytes() > 0 {
                engine.copy_u8_range_into(
                    dst.v_mut(),
                    0,
                    src.v(),
                    plan.source_row() * source.v_tok_bytes(),
                    plan.v_bytes(),
                )?;
            }
        }
        self.set_tp_kv_len_mirrors(&mut target, plan.rows())?;

        // The caller publishes `target` and immediately drops `source`. Drain every rank's
        // stream so an async-pool free cannot recycle a source plane under an in-flight D2D copy.
        for engine in &self.ranks {
            let _main = engine.gpu.enter_main()?;
            engine.stream().synchronize()?;
        }
        let physical_copy_rows = plan.copy_rows();
        target.publish_grow(plan)?;
        eprintln!(
            "[step-tp-kv-grow] rows={} source_capacity={} target_capacity={} ranks={} \
             physical_copy_rows={} ring_window={:?} copy=rank-local-dtod \
             rank_streams_synchronized=true generation_preserved=true",
            rows,
            source.capacity(),
            target_capacity,
            ranks,
            physical_copy_rows,
            source.ring_window(),
        );
        Ok(target)
    }

    pub fn hydrate_tp_kv_cache(
        &self,
        cache: &mut ResidentTpKvCache,
        rows: usize,
        k_rows: &[u8],
        v_rows: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.hydrate_tp_kv_cache_from(cache, rows, 0, k_rows, v_rows)
    }

    pub fn hydrate_tp_kv_cache_from(
        &self,
        cache: &mut ResidentTpKvCache,
        logical_len: usize,
        resident_start: usize,
        k_rows: &[u8],
        v_rows: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.validate_tp_kv_cache(cache)?;
        if cache.committed_len() != 0 || cache.staged_len() != 0 {
            return Err(format!(
                "TP KV hydration requires an empty cache, got committed/staged={}/{}",
                cache.committed_len(),
                cache.staged_len()
            )
            .into());
        }
        if resident_start > logical_len || logical_len > cache.capacity() {
            return Err(format!(
                "TP KV hydration range [{resident_start},{logical_len}) exceeds capacity {}",
                cache.capacity(),
            )
            .into());
        }
        let rows = logical_len - resident_start;
        if rows > cache.physical_capacity() {
            return Err(format!(
                "TP KV hydration rows {rows} exceed physical capacity {}",
                cache.physical_capacity()
            )
            .into());
        }
        for rank in 0..self.ranks.len() {
            let k_rank =
                cache_rank_rows(k_rows, rows, cache.k_tok_bytes(), self.ranks.len(), rank)?;
            let v_rank =
                cache_rank_rows(v_rows, rows, cache.v_tok_bytes(), self.ranks.len(), rank)?;
            let engine = &self.ranks[rank];
            let _main = engine.gpu.enter_main()?;
            let rank_cache = cache
                .rank_mut(rank)
                .ok_or_else(|| format!("TP KV cache has no rank {rank}"))?;
            engine.htod_u8_into(rank_cache.k_mut(), 0, &k_rank)?;
            engine.htod_u8_into(rank_cache.v_mut(), 0, &v_rank)?;
        }
        cache.publish_hydration(logical_len, resident_start)?;
        Ok(())
    }

    /// Restore rows retained from a speculative verify walk into an already-live distributed
    /// cache. The verify oracle appends canonical full-width quantized K/V rows on the model
    /// device. Each destination rank pulls its own contiguous KV-head slice over native P2P; its
    /// length mirror is published on that same rank stream after every row copy.
    #[allow(clippy::too_many_arguments)]
    pub fn restore_tp_kv_rows_from_device(
        &self,
        cache: &mut ResidentTpKvCache,
        start: usize,
        logical_len: usize,
        source_k_raw: u64,
        source_v_raw: u64,
        source_k_tok_bytes: usize,
        source_v_tok_bytes: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.validate_tp_kv_cache(cache)?;
        if cache.committed_len() != cache.staged_len() {
            return Err(format!(
                "TP KV verify restore requires quiescent state, got committed/staged={}/{}",
                cache.committed_len(),
                cache.staged_len()
            )
            .into());
        }
        if start > logical_len || logical_len > cache.capacity() {
            return Err(format!(
                "TP KV verify restore range [{start},{logical_len}) exceeds capacity {}",
                cache.capacity()
            )
            .into());
        }
        let rows = logical_len - start;
        let physical = cache.physical_range(start, logical_len)?;
        if physical.len() != rows {
            return Err(format!(
                "TP KV verify restore range [{start},{logical_len}) is not physically contiguous"
            )
            .into());
        }
        let ranks = self.ranks.len();
        let k_tok_bytes = cache.k_tok_bytes();
        let v_tok_bytes = cache.v_tok_bytes();
        if source_k_tok_bytes != k_tok_bytes * ranks || source_v_tok_bytes != v_tok_bytes * ranks {
            return Err(format!(
                "TP KV verify source token bytes k={source_k_tok_bytes} v={source_v_tok_bytes} \
                 do not match distributed k={}x{ranks} v={}x{ranks}",
                k_tok_bytes, v_tok_bytes
            )
            .into());
        }
        for rank in 0..self.ranks.len() {
            let engine = &self.ranks[rank];
            let _main = engine.gpu.enter_main()?;
            use cudarc::driver::DevicePtr;
            let stream = engine.stream();
            let k_offset = physical
                .start
                .checked_mul(k_tok_bytes)
                .ok_or("TP KV verify K offset overflow")?;
            let v_offset = physical
                .start
                .checked_mul(v_tok_bytes)
                .ok_or("TP KV verify V offset overflow")?;
            let rank_cache = cache
                .rank_mut(rank)
                .ok_or_else(|| format!("TP KV cache has no rank {rank}"))?;
            let k_dst = {
                let (pointer, _guard) = rank_cache.k_mut().device_ptr(&stream);
                pointer
            };
            let v_dst = {
                let (pointer, _guard) = rank_cache.v_mut().device_ptr(&stream);
                pointer
            };
            for row in 0..rows {
                let k_src = source_k_raw + (row * source_k_tok_bytes + rank * k_tok_bytes) as u64;
                let v_src = source_v_raw + (row * source_v_tok_bytes + rank * v_tok_bytes) as u64;
                let k_out = k_dst + (k_offset + row * k_tok_bytes) as u64;
                let v_out = v_dst + (v_offset + row * v_tok_bytes) as u64;
                raw_copy_bytes(k_out, k_src, k_tok_bytes, engine)?;
                raw_copy_bytes(v_out, v_src, v_tok_bytes, engine)?;
            }
        }
        cache.rewind_to(logical_len)?;
        Ok(())
    }

    /// Batch the accepted verify rows of every uniform TP-attention layer into one kernel per
    /// rank. Returns `false` before enqueueing anything when the layer group is not uniform, so
    /// the caller can use the existing per-layer repair without splitting semantics.
    pub fn restore_tp_kv_layers_from_device(
        &self,
        layers: &mut [TpKvVerifiedLayer<'_>],
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let Some(first) = layers.first() else {
            return Ok(false);
        };
        if !self.native_p2p || first.start >= first.logical_len {
            return Ok(false);
        }
        let ranks = self.ranks.len();
        let rows = first.logical_len - first.start;
        let logical_len = first.logical_len;
        let k_row_bytes = first.cache.k_tok_bytes();
        let v_row_bytes = first.cache.v_tok_bytes();
        let k_src_stride = first.source_k_tok_bytes;
        let v_src_stride = first.source_v_tok_bytes;
        let Some(expected_k_stride) = k_row_bytes.checked_mul(ranks) else {
            return Ok(false);
        };
        let Some(expected_v_stride) = v_row_bytes.checked_mul(ranks) else {
            return Ok(false);
        };
        if k_src_stride != expected_k_stride || v_src_stride != expected_v_stride {
            return Ok(false);
        }

        for layer in layers.iter() {
            self.validate_tp_kv_cache(layer.cache)?;
            if layer.cache.committed_len() != layer.cache.staged_len()
                || layer.start > layer.logical_len
                || layer.logical_len != logical_len
                || layer.logical_len - layer.start != rows
                || layer.cache.k_tok_bytes() != k_row_bytes
                || layer.cache.v_tok_bytes() != v_row_bytes
                || layer.source_k_tok_bytes != k_src_stride
                || layer.source_v_tok_bytes != v_src_stride
            {
                return Ok(false);
            }
            let physical = layer.cache.physical_range(layer.start, layer.logical_len)?;
            if physical.len() != rows {
                return Ok(false);
            }
        }

        for rank in 0..ranks {
            let engine = &self.ranks[rank];
            let _main = engine.gpu.enter_main()?;
            let stream = engine.stream();
            let n = layers.len();
            let mut table = vec![0u64; 5 * n];
            for (index, layer) in layers.iter_mut().enumerate() {
                let physical = layer.cache.physical_range(layer.start, layer.logical_len)?;
                let rank_cache = layer
                    .cache
                    .rank_mut(rank)
                    .ok_or_else(|| format!("TP KV cache has no rank {rank}"))?;
                table[index] = layer.source_k_raw + (rank * k_row_bytes) as u64;
                table[n + index] = layer.source_v_raw + (rank * v_row_bytes) as u64;
                table[2 * n + index] = rank_cache.k_mut().device_ptr(&stream).0
                    + (physical.start * k_row_bytes) as u64;
                table[3 * n + index] = rank_cache.v_mut().device_ptr(&stream).0
                    + (physical.start * v_row_bytes) as u64;
                table[4 * n + index] = rank_cache.len_d_mut().device_ptr(&stream).0;
            }
            let table = engine.htod_u64(&table)?;
            engine.copy_batch_uniform_kv_u8_set_len(
                &table,
                n,
                rows,
                k_row_bytes,
                v_row_bytes,
                k_src_stride,
                v_src_stride,
                logical_len,
            )?;
        }
        let layer_count = layers.len();
        for layer in layers.iter_mut() {
            layer.cache.publish_device_rewind(logical_len)?;
        }
        static ANNOUNCED: std::sync::Once = std::sync::Once::new();
        ANNOUNCED.call_once(|| {
            eprintln!(
                "[tp-kv-verify-batch] engaged: layers={} ranks={ranks} rows={rows} \
                 k_row_bytes={k_row_bytes} v_row_bytes={v_row_bytes}",
                layer_count
            );
        });
        Ok(true)
    }

    pub fn append_tp_kv_transaction(
        &self,
        cache: &mut ResidentTpKvCache,
        transaction: TpKvTransaction,
        k_shards: &[CudaSlice<f32>],
        v_shards: &[CudaSlice<f32>],
        rows: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.append_tp_kv_transaction_inner(cache, transaction, k_shards, v_shards, rows, false)
    }

    /// `external_rank_appends`: the dcw path already wrote the rank rows (device-counter
    /// append) — run everything EXCEPT the per-rank quantize/append loop (plan validation,
    /// rebase arm — unreachable when the caller peeked — and the absolute len-mirror sets,
    /// which land the same value the in-stream inc produced).
    #[allow(clippy::too_many_arguments)]
    pub fn append_tp_kv_transaction_inner(
        &self,
        cache: &mut ResidentTpKvCache,
        transaction: TpKvTransaction,
        k_shards: &[CudaSlice<f32>],
        v_shards: &[CudaSlice<f32>],
        rows: usize,
        external_rank_appends: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.validate_tp_kv_cache(cache)?;
        let plan = cache.prepare_append(transaction, rows)?;
        let target = plan.target();
        let expected_k = rows
            .checked_mul(cache.kv_dim_k())
            .ok_or("TP KV K append size overflow")?;
        let expected_v = rows
            .checked_mul(cache.kv_dim_v())
            .ok_or("TP KV V append size overflow")?;
        // external_rank_appends passes no shards — the graph's dcw appends already wrote
        // the rank rows, so this call is bookkeeping-only and the shard slices are unused.
        if !external_rank_appends
            && (k_shards.len() != self.ranks.len() || v_shards.len() != self.ranks.len())
        {
            return Err(format!(
                "TP KV append shard counts k={} v={} != ranks {}",
                k_shards.len(),
                v_shards.len(),
                self.ranks.len()
            )
            .into());
        }
        let kv_dim_k = cache.kv_dim_k();
        let kv_dim_v = cache.kv_dim_v();
        let k_tok_bytes = cache.k_tok_bytes();
        let v_tok_bytes = cache.v_tok_bytes();
        if let Some(ring_base) = cache.ring_base() {
            let base_val = ring_base as i32;
            for rank in 0..self.ranks.len() {
                let engine = &self.ranks[rank];
                let _main = engine.gpu.enter_main()?;
                let rank_cache = cache
                    .rank_mut(rank)
                    .ok_or_else(|| format!("TP KV cache has no rank {rank}"))?;
                if rank_cache.base_d().is_none() {
                    rank_cache.arm_base_d(engine.htod_i32(&[base_val])?);
                }
            }
        }
        if let Some(KvRingAppend::Rebase {
            src_row,
            keep_rows,
            new_base,
            ..
        }) = plan.ring_append()
        {
            for rank in 0..self.ranks.len() {
                let engine = &self.ranks[rank];
                let _main = engine.gpu.enter_main()?;
                let rank_cache = cache
                    .rank_mut(rank)
                    .ok_or_else(|| format!("TP KV cache has no rank {rank}"))?;
                if keep_rows > 0 {
                    let k_len = keep_rows
                        .checked_mul(k_tok_bytes)
                        .ok_or("TP KV K rebase-byte overflow")?;
                    let v_len = keep_rows
                        .checked_mul(v_tok_bytes)
                        .ok_or("TP KV V rebase-byte overflow")?;
                    let mut k_tmp = engine.alloc_u8_uninit(k_len)?;
                    let mut v_tmp = engine.alloc_u8_uninit(v_len)?;
                    engine.copy_u8_range_into(
                        &mut k_tmp,
                        0,
                        rank_cache.k(),
                        src_row * k_tok_bytes,
                        k_len,
                    )?;
                    engine.copy_u8_range_into(
                        &mut v_tmp,
                        0,
                        rank_cache.v(),
                        src_row * v_tok_bytes,
                        v_len,
                    )?;
                    engine.copy_u8_into(rank_cache.k_mut(), 0, &k_tmp, k_len)?;
                    engine.copy_u8_into(rank_cache.v_mut(), 0, &v_tmp, v_len)?;
                }
                // dcw base mirror (graph increment A): physical row 0 now holds logical
                // row `new_base`; armed device mirrors track it (rebases are rare host
                // events, so a host set here is the whole maintenance cost).
                let value = new_base as i32;
                if let Some(base_d) = rank_cache.base_d_mut() {
                    engine.set_i32_one(base_d, value)?;
                } else {
                    rank_cache.arm_base_d(engine.htod_i32(&[value])?);
                }
            }
        }
        cache.publish_append_rebase(plan)?;
        let write_row = plan.write_row();
        for rank in 0..self.ranks.len() {
            if external_rank_appends {
                break;
            }
            let engine = &self.ranks[rank];
            let _main = engine.gpu.enter_main()?;
            if k_shards[rank].len() != expected_k
                || v_shards[rank].len() != expected_v
                || k_shards[rank].ordinal() != engine.ctx().ordinal()
                || v_shards[rank].ordinal() != engine.ctx().ordinal()
            {
                return Err(format!(
                    "TP KV rank {rank} shard geometry/device k={}/{} v={}/{} \
                     != expected {expected_k}/{expected_v} on device {}",
                    k_shards[rank].len(),
                    k_shards[rank].ordinal(),
                    v_shards[rank].len(),
                    v_shards[rank].ordinal(),
                    engine.ctx().ordinal(),
                )
                .into());
            }
            let rank_cache = cache
                .rank_mut(rank)
                .ok_or_else(|| format!("TP KV cache has no rank {rank}"))?;
            let (rank_k, rank_v) = rank_cache.planes_mut();
            engine.append_kv_quantized_rows(
                &k_shards[rank],
                &v_shards[rank],
                rank_k,
                rank_v,
                write_row,
                rows,
                kv_dim_k,
                kv_dim_v,
                k_tok_bytes,
                v_tok_bytes,
                false,
            )?;
        }
        if !external_rank_appends {
            // dcw appends advance the device counters with in-stream inc_i32; an absolute set
            // here would race the merged per-rank append (it reads len_d for its write row).
            self.set_tp_kv_len_mirrors(cache, target)?;
        }
        cache.publish_append_plan(plan)?;
        Ok(())
    }

    pub fn commit_tp_kv_transaction(
        &self,
        cache: &mut ResidentTpKvCache,
        transaction: TpKvTransaction,
        accepted_rows: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.validate_tp_kv_cache(cache)?;
        let target = cache.commit_target(transaction, accepted_rows)?;
        self.set_tp_kv_len_mirrors(cache, target)?;
        cache.publish_finalize(transaction, target)?;
        // This path derived the rank rows from the canonical rows, so a later restore from
        // the canonical cache is at worst redundant. See `rows_external`.
        cache.mark_rows_external(false);
        Ok(())
    }

    /// Commit for the external-appends (token graph) path: host bookkeeping only, NO absolute
    /// len-mirror sets. The graph's in-stream inc_i32 owns the device counters; a rank-stream
    /// set here has no ordering edge against the NEXT token's graph launch (graph children do
    /// not wait on the rank streams), so it can land AFTER that graph's inc and drag the
    /// counter backward mid-token.
    pub fn commit_tp_kv_transaction_external(
        &self,
        cache: &mut ResidentTpKvCache,
        transaction: TpKvTransaction,
        accepted_rows: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.validate_tp_kv_cache(cache)?;
        let target = cache.commit_target(transaction, accepted_rows)?;
        cache.publish_finalize(transaction, target)?;
        // The rank rows were written on-device by the caller (dcw / fa2 verify); the
        // canonical model-device cache only had its length advanced and holds stale content
        // for these rows. A verified-prefix restore must skip this layer (memra#128).
        cache.mark_rows_external(true);
        Ok(())
    }

    pub fn rollback_tp_kv_transaction(
        &self,
        cache: &mut ResidentTpKvCache,
        transaction: TpKvTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.validate_tp_kv_cache(cache)?;
        cache.validate_transaction(transaction)?;
        let target = transaction.base_len();
        self.set_tp_kv_len_mirrors(cache, target)?;
        cache.publish_finalize(transaction, target)?;
        Ok(())
    }

    pub fn tp_kv_device_lengths(
        &self,
        cache: &ResidentTpKvCache,
    ) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
        self.validate_tp_kv_cache(cache)?;
        let mut lengths = Vec::with_capacity(self.ranks.len());
        for (engine, rank_cache) in self.ranks.iter().zip(cache.ranks()) {
            let _main = engine.gpu.enter_main()?;
            lengths.push(engine.dtoh_i32_one(rank_cache.len_d())?);
        }
        Ok(lengths)
    }

    fn set_tp_kv_len_mirrors(
        &self,
        cache: &mut ResidentTpKvCache,
        len: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let len = i32::try_from(len).map_err(|_| "TP KV length exceeds i32 device mirror")?;
        for (engine, rank_cache) in self.ranks.iter().zip(cache.ranks_mut()) {
            let _main = engine.gpu.enter_main()?;
            engine.set_i32_one(rank_cache.len_d_mut(), len)?;
        }
        Ok(())
    }

    fn validate_tp_kv_cache(
        &self,
        cache: &ResidentTpKvCache,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if cache.ranks_len() != self.ranks.len() {
            return Err(format!(
                "TP KV cache ranks {} != runtime ranks {}",
                cache.ranks_len(),
                self.ranks.len()
            )
            .into());
        }
        let expected_k = cache
            .physical_capacity()
            .checked_mul(cache.k_tok_bytes())
            .and_then(|bytes| bytes.checked_add(8))
            .ok_or("TP KV K plane validation overflow")?;
        let expected_v = cache
            .physical_capacity()
            .checked_mul(cache.v_tok_bytes())
            .and_then(|bytes| bytes.checked_add(8))
            .ok_or("TP KV V plane validation overflow")?;
        for (rank, (engine, rank_cache)) in self.ranks.iter().zip(cache.ranks()).enumerate() {
            let device = engine.ctx().ordinal();
            if rank_cache.k().len() != expected_k
                || rank_cache.v().len() != expected_v
                || rank_cache.len_d().len() != 1
                || rank_cache.k().ordinal() != device
                || rank_cache.v().ordinal() != device
                || rank_cache.len_d().ordinal() != device
            {
                return Err(format!(
                    "TP KV rank {rank} residency does not match device {device} or plane geometry"
                )
                .into());
            }
        }
        Ok(())
    }

    pub fn full(
        &self,
        matrix: E4m3BlockMatrix<'_>,
        activations: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        matrix.validate()?;
        validate_activations(activations, tokens, matrix.in_features)?;
        run_rank(&self.ranks[0], matrix, activations, tokens)
    }

    /// Column-parallel projection. Weight output rows and their scale rows are partitioned across
    /// ranks. The input is host-broadcast, rank-local projections execute independently, and the
    /// output is host-gathered in rank order.
    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    pub fn column_parallel(
        &self,
        matrix: E4m3BlockMatrix<'_>,
        activations: &[f32],
        tokens: usize,
    ) -> Result<ColumnParallelResult, Box<dyn std::error::Error>> {
        matrix.validate()?;
        validate_activations(activations, tokens, matrix.in_features)?;
        let tp = self.ranks.len();
        if matrix.out_features % tp != 0 {
            return Err(format!(
                "column-parallel out_features {} is not divisible by TP={tp}",
                matrix.out_features
            )
            .into());
        }
        let local_out = matrix.out_features / tp;
        if !local_out.is_multiple_of(FP8_BLOCK) {
            return Err(format!(
                "column-parallel output shard {local_out} cuts through a {FP8_BLOCK}-row \
                 E4M3 scale block"
            )
            .into());
        }

        let mut gathered = vec![0.0f32; tokens * matrix.out_features];
        let mut rank_outputs = Vec::with_capacity(tp);
        for (rank_index, rank) in self.ranks.iter().enumerate() {
            let shard = column_shard(matrix, tp, rank_index)?;
            let output = run_rank(rank, shard, activations, tokens)?;
            let row_start = rank_index * local_out;
            for token in 0..tokens {
                gathered[token * matrix.out_features + row_start
                    ..token * matrix.out_features + row_start + local_out]
                    .copy_from_slice(&output[token * local_out..(token + 1) * local_out]);
            }
            rank_outputs.push(output);
        }
        Ok(ColumnParallelResult {
            gathered,
            rank_outputs,
        })
    }

    pub fn upload_column_parallel(
        &self,
        matrix: E4m3BlockMatrix<'_>,
    ) -> Result<ResidentColumnParallel, Box<dyn std::error::Error>> {
        matrix.validate()?;
        let tp = self.ranks.len();
        validate_column_shape(matrix, tp)?;
        let mut ranks = Vec::with_capacity(tp);
        for (rank_index, engine) in self.ranks.iter().enumerate() {
            ranks.push(upload_rank(engine, column_shard(matrix, tp, rank_index)?)?);
        }
        Ok(ResidentColumnParallel {
            ranks,
            out_features: matrix.out_features,
            in_features: matrix.in_features,
        })
    }

    pub fn column_parallel_resident(
        &self,
        matrix: &ResidentColumnParallel,
        activations: &[f32],
        tokens: usize,
    ) -> Result<ColumnParallelResult, Box<dyn std::error::Error>> {
        validate_resident_ranks(&self.ranks, &matrix.ranks)?;
        validate_activations(activations, tokens, matrix.in_features)?;
        let local_out = matrix.out_features / self.ranks.len();
        let mut gathered = vec![0.0f32; tokens * matrix.out_features];
        let mut rank_outputs = Vec::with_capacity(self.ranks.len());
        for (rank_index, (engine, shard)) in self.ranks.iter().zip(&matrix.ranks).enumerate() {
            let output = run_resident_rank(engine, shard, activations, tokens)?;
            let row_start = rank_index * local_out;
            for token in 0..tokens {
                gathered[token * matrix.out_features + row_start
                    ..token * matrix.out_features + row_start + local_out]
                    .copy_from_slice(&output[token * local_out..(token + 1) * local_out]);
            }
            rank_outputs.push(output);
        }
        Ok(ColumnParallelResult {
            gathered,
            rank_outputs,
        })
    }

    /// Row-parallel projection. Weight/input columns and their scale columns are partitioned
    /// across ranks. Rank-local partials return through host memory and are reduced in stable
    /// rank order.
    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    pub fn row_parallel(
        &self,
        matrix: E4m3BlockMatrix<'_>,
        activations: &[f32],
        tokens: usize,
    ) -> Result<RowParallelResult, Box<dyn std::error::Error>> {
        matrix.validate()?;
        validate_activations(activations, tokens, matrix.in_features)?;
        let tp = self.ranks.len();
        if matrix.in_features % tp != 0 {
            return Err(format!(
                "row-parallel in_features {} is not divisible by TP={tp}",
                matrix.in_features
            )
            .into());
        }
        let local_in = matrix.in_features / tp;
        if !local_in.is_multiple_of(FP8_BLOCK) {
            return Err(format!(
                "row-parallel input shard {local_in} cuts through a {FP8_BLOCK}-column \
                 E4M3 scale block"
            )
            .into());
        }

        let mut reduced = vec![0.0f32; tokens * matrix.out_features];
        let mut rank_partials = Vec::with_capacity(tp);
        for (rank_index, rank) in self.ranks.iter().enumerate() {
            let (codes, scales) = row_shard(matrix, tp, rank_index)?;
            let local_activations =
                activation_shard(activations, tokens, matrix.in_features, tp, rank_index);
            let shard = E4m3BlockMatrix {
                codes: &codes,
                scales: &scales,
                out_features: matrix.out_features,
                in_features: local_in,
            };
            let partial = run_rank(rank, shard, &local_activations, tokens)?;
            for (sum, value) in reduced.iter_mut().zip(&partial) {
                *sum += *value;
            }
            rank_partials.push(partial);
        }
        Ok(RowParallelResult {
            reduced,
            rank_partials,
        })
    }

    pub fn upload_row_parallel(
        &self,
        matrix: E4m3BlockMatrix<'_>,
    ) -> Result<ResidentRowParallel, Box<dyn std::error::Error>> {
        matrix.validate()?;
        let tp = self.ranks.len();
        validate_row_shape(matrix, tp)?;
        let local_in = matrix.in_features / tp;
        let mut ranks = Vec::with_capacity(tp);
        for (rank_index, engine) in self.ranks.iter().enumerate() {
            let (codes, scales) = row_shard(matrix, tp, rank_index)?;
            ranks.push(upload_rank(
                engine,
                E4m3BlockMatrix {
                    codes: &codes,
                    scales: &scales,
                    out_features: matrix.out_features,
                    in_features: local_in,
                },
            )?);
        }
        Ok(ResidentRowParallel {
            ranks,
            out_features: matrix.out_features,
            in_features: matrix.in_features,
        })
    }

    pub fn row_parallel_resident(
        &self,
        matrix: &ResidentRowParallel,
        activations: &[f32],
        tokens: usize,
    ) -> Result<RowParallelResult, Box<dyn std::error::Error>> {
        validate_resident_ranks(&self.ranks, &matrix.ranks)?;
        validate_activations(activations, tokens, matrix.in_features)?;
        let tp = self.ranks.len();
        let mut reduced = vec![0.0f32; tokens * matrix.out_features];
        let mut rank_partials = Vec::with_capacity(tp);
        for (rank_index, (engine, shard)) in self.ranks.iter().zip(&matrix.ranks).enumerate() {
            let local_activations =
                activation_shard(activations, tokens, matrix.in_features, tp, rank_index);
            let partial = run_resident_rank(engine, shard, &local_activations, tokens)?;
            for (sum, value) in reduced.iter_mut().zip(&partial) {
                *sum += *value;
            }
            rank_partials.push(partial);
        }
        Ok(RowParallelResult {
            reduced,
            rank_partials,
        })
    }

    pub fn upload_bf16_column_parallel(
        &self,
        matrix: Bf16Matrix<'_>,
    ) -> Result<ResidentBf16ColumnParallel, Box<dyn std::error::Error>> {
        self.upload_bf16_column_parallel_inner(matrix, None, false)
    }

    /// Step-3.7 column projection with one numerical program across TP1/TP2/TP4/TP8.
    pub fn upload_step_bf16_column_parallel(
        &self,
        matrix: Bf16Matrix<'_>,
    ) -> Result<ResidentBf16ColumnParallel, Box<dyn std::error::Error>> {
        self.upload_step_bf16_column_parallel_inner(matrix, false)
    }

    /// Load-time exact F32 expansion of a Step BF16 shard.
    ///
    /// The original BF16 allocation is released after the stream-ordered conversion. Decode then
    /// reuses the resident F32 values with the same topology-invariant output-row chunks.
    pub fn upload_step_bf16_column_parallel_f32_mirror(
        &self,
        matrix: Bf16Matrix<'_>,
    ) -> Result<ResidentBf16ColumnParallel, Box<dyn std::error::Error>> {
        self.upload_step_bf16_column_parallel_inner(matrix, true)
    }

    fn upload_step_bf16_column_parallel_inner(
        &self,
        matrix: Bf16Matrix<'_>,
        f32_mirror: bool,
    ) -> Result<ResidentBf16ColumnParallel, Box<dyn std::error::Error>> {
        let canonical_chunk_rows =
            step_bf16_canonical_chunk_rows(matrix.out_features, self.ranks.len())?;
        self.upload_bf16_column_parallel_inner(matrix, Some(canonical_chunk_rows), f32_mirror)
    }

    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    fn upload_bf16_column_parallel_inner(
        &self,
        matrix: Bf16Matrix<'_>,
        canonical_chunk_rows: Option<usize>,
        f32_mirror: bool,
    ) -> Result<ResidentBf16ColumnParallel, Box<dyn std::error::Error>> {
        matrix.validate()?;
        let tp = self.ranks.len();
        if matrix.out_features % tp != 0 {
            return Err(format!(
                "BF16 column-parallel out_features {} is not divisible by TP={tp}",
                matrix.out_features
            )
            .into());
        }
        let mut ranks = Vec::with_capacity(tp);
        for (rank, engine) in self.ranks.iter().enumerate() {
            ranks.push(upload_bf16_rank(
                engine,
                bf16_column_shard(matrix, tp, rank)?,
                f32_mirror,
            )?);
        }
        Ok(ResidentBf16ColumnParallel {
            ranks,
            out_features: matrix.out_features,
            in_features: matrix.in_features,
            canonical_chunk_rows,
        })
    }

    pub fn bf16_column_parallel_resident(
        &self,
        matrix: &ResidentBf16ColumnParallel,
        activations: &[f32],
        tokens: usize,
    ) -> Result<ColumnParallelResult, Box<dyn std::error::Error>> {
        validate_resident_bf16_ranks(&self.ranks, &matrix.ranks)?;
        validate_activations(activations, tokens, matrix.in_features)?;
        let local_out = matrix.out_features / self.ranks.len();
        let mut gathered = vec![0.0f32; tokens * matrix.out_features];
        let mut rank_outputs = Vec::with_capacity(self.ranks.len());
        for (rank, (engine, shard)) in self.ranks.iter().zip(&matrix.ranks).enumerate() {
            let output = run_resident_bf16_rank(
                engine,
                shard,
                activations,
                tokens,
                matrix.canonical_chunk_rows,
            )?;
            for token in 0..tokens {
                let src = &output[token * local_out..(token + 1) * local_out];
                let dst_start = token * matrix.out_features + rank * local_out;
                gathered[dst_start..dst_start + local_out].copy_from_slice(src);
            }
            rank_outputs.push(output);
        }
        Ok(ColumnParallelResult {
            gathered,
            rank_outputs,
        })
    }

    /// Native-P2P twin of [`Self::bf16_column_parallel_resident`].
    ///
    /// The host-canonical activation is uploaded once on rank zero and peer-broadcast to the
    /// remaining ranks. Rank-local outputs are peer-gathered in token-major order before one root
    /// readback. This removes per-rank host staging but deliberately still returns a host oracle;
    /// attention and KV ownership are separate milestones.
    pub fn bf16_column_parallel_resident_native(
        &self,
        matrix: &ResidentBf16ColumnParallel,
        activations: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let rank_outputs =
            self.bf16_column_parallel_resident_device_shards(matrix, activations, tokens)?;
        let local_out = matrix.out_features / self.ranks.len();
        self.gather_native_column_shards(&rank_outputs, tokens, local_out)
    }

    /// Does the serving engine live in the SAME CUDA context as this runtime's root rank?
    /// The device-resident input/output seams below hand raw device buffers across the
    /// Engine boundary, which is only addressable when both sides share the root device's
    /// primary context — the generic full-attention TP seam keys its residency dispatch on.
    pub fn root_shares_ctx(&self, e: &Engine) -> bool {
        self.ranks
            .first()
            .is_some_and(|root| root.ctx().cu_ctx() == e.ctx().cu_ctx())
    }

    /// Device-input twin of [`Self::bf16_column_parallel_resident_native`] (lane/
    /// hermes-perf-fixes, 2026-08-23 — the step QKV TP host-bounce finding). The activation
    /// arrives as a ROOT-DEVICE buffer (first `tokens * in_features` values) instead of a
    /// host slice, and the gathered output stays root-resident: no DtoH of the hidden state,
    /// no host q/k/v staging, no re-upload. BYTE-IDENTICAL to the host-canonical native arm
    /// by construction — the root input bytes are dtod-copied where the host arm htod'd the
    /// same bytes, and every kernel, peer copy, and gather order is shared.
    ///
    /// FENCES: caller must have synchronized the producer stream that wrote
    /// `root_activation` (the serving engine's — a DIFFERENT stream in the same context);
    /// this method synchronizes the root stream before returning so the caller's stream can
    /// consume the gathered output immediately.
    pub fn bf16_column_parallel_resident_native_device(
        &self,
        matrix: &ResidentBf16ColumnParallel,
        root_activation: &CudaSlice<f32>,
        tokens: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let rank_outputs = self.bf16_column_parallel_resident_device_shards_from_root(
            matrix,
            root_activation,
            tokens,
        )?;
        let local_out = matrix.out_features / self.ranks.len();
        let gathered = self.gather_native_column_shards_device(&rank_outputs, tokens, local_out)?;
        let root = &self.ranks[0];
        let _main = root.gpu.enter_main()?;
        root.stream().synchronize()?;
        Ok(gathered)
    }

    /// Root-device-input twin of [`Self::bf16_column_parallel_resident_device_shards`]:
    /// the canonical activation is already resident on the root device (len >=
    /// `tokens * in_features`; extra tail values beyond the active prefix are ignored,
    /// the reused-prime-slab contract of `active_matrix_values`).
    pub fn bf16_column_parallel_resident_device_shards_from_root(
        &self,
        matrix: &ResidentBf16ColumnParallel,
        root_activation: &CudaSlice<f32>,
        tokens: usize,
    ) -> Result<Vec<CudaSlice<f32>>, Box<dyn std::error::Error>> {
        if self.ranks.len() > 1 && !self.native_p2p {
            return Err("device-resident BF16 column parallelism requires native P2P ranks".into());
        }
        validate_resident_bf16_ranks(&self.ranks, &matrix.ranks)?;
        let values = tokens
            .checked_mul(matrix.in_features)
            .ok_or("device BF16 column activation size overflow")?;
        let root = &self.ranks[0];
        if tokens == 0
            || root_activation.len() < values
            || root_activation.ordinal() != root.ctx().ordinal()
        {
            return Err("device BF16 column root activation geometry mismatch".into());
        }

        let mut rank_inputs = Vec::with_capacity(self.ranks.len());
        let root_input = {
            let _main = root.gpu.enter_main()?;
            let mut root_input = root.uninit(values)?;
            root.stream()
                .memcpy_dtod(&root_activation.slice(0..values), &mut root_input)?;
            root_input
        };
        // PRODUCER FENCE (same discipline as the host-input twin): the peer broadcast
        // below reads this buffer from the OTHER ranks' streams while the root dtod may
        // still be in flight.
        {
            let _main = root.gpu.enter_main()?;
            root.stream().synchronize()?;
        }
        rank_inputs.push(root_input);
        for engine in &self.ranks[1..] {
            let peer_input = {
                let _main = engine.gpu.enter_main()?;
                let mut peer_input = engine.uninit(values)?;
                engine
                    .stream()
                    .memcpy_dtod(&rank_inputs[0], &mut peer_input)?;
                peer_input
            };
            rank_inputs.push(peer_input);
        }

        let mut rank_outputs = Vec::with_capacity(self.ranks.len());
        #[allow(clippy::needless_range_loop)]
        // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
        for rank in 0..self.ranks.len() {
            rank_outputs.push(run_resident_bf16_rank_device(
                &self.ranks[rank],
                &matrix.ranks[rank],
                &rank_inputs[rank],
                tokens,
                matrix.canonical_chunk_rows,
                self.bulk_p2p,
            )?);
        }
        Ok(rank_outputs)
    }

    /// Keep Step BF16 column outputs resident on their owning TP ranks.
    ///
    /// Rank zero receives the host-canonical activation once and peer-broadcasts it when TP>1.
    /// Unlike [`Self::bf16_column_parallel_resident_native`], this method performs no output
    /// gather or readback. It is the correctness substrate for rank-local norm, RoPE, attention,
    /// and cache ownership; callers must not treat its existence as serving qualification.
    pub fn bf16_column_parallel_resident_device_shards(
        &self,
        matrix: &ResidentBf16ColumnParallel,
        activations: &[f32],
        tokens: usize,
    ) -> Result<Vec<CudaSlice<f32>>, Box<dyn std::error::Error>> {
        if self.ranks.len() > 1 && !self.native_p2p {
            return Err("device-resident BF16 column parallelism requires native P2P ranks".into());
        }
        validate_resident_bf16_ranks(&self.ranks, &matrix.ranks)?;
        validate_activations(activations, tokens, matrix.in_features)?;

        let mut rank_inputs = Vec::with_capacity(self.ranks.len());
        let root_input = {
            let root = &self.ranks[0];
            let _main = root.gpu.enter_main()?;
            root.htod(activations)?
        };
        // PRODUCER FENCE (2026-08-20 flake fix): the peer broadcast below reads this buffer from
        // the OTHER ranks' streams, and clone_htod is asynchronous on the root stream. Without
        // this fence a peer copy can overtake the in-flight H2D and replicate stale bytes — the
        // measured ~30%-of-boots prefill/decode argmax flake. Same discipline as
        // `upload_replicated_device_rows`.
        {
            let root = &self.ranks[0];
            let _main = root.gpu.enter_main()?;
            root.stream().synchronize()?;
        }
        rank_inputs.push(root_input);
        for engine in &self.ranks[1..] {
            let peer_input = {
                let _main = engine.gpu.enter_main()?;
                let mut peer_input = engine.uninit(activations.len())?;
                engine
                    .stream()
                    .memcpy_dtod(&rank_inputs[0], &mut peer_input)?;
                peer_input
            };
            rank_inputs.push(peer_input);
        }

        let mut rank_outputs = Vec::with_capacity(self.ranks.len());
        #[allow(clippy::needless_range_loop)]
        // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
        for rank in 0..self.ranks.len() {
            rank_outputs.push(run_resident_bf16_rank_device(
                &self.ranks[rank],
                &matrix.ranks[rank],
                &rank_inputs[rank],
                tokens,
                matrix.canonical_chunk_rows,
                self.bulk_p2p,
            )?);
        }
        Ok(rank_outputs)
    }

    /// Allocate one fixed-shape replicated batch without initializing its contents.
    ///
    /// Callers must refresh every rank before passing the batch to an operator.
    pub fn allocate_replicated_device_rows(
        &self,
        tokens: usize,
        width: usize,
    ) -> Result<ResidentReplicatedDeviceRows, Box<dyn std::error::Error>> {
        if self.ranks.len() > 1 && !self.native_p2p {
            return Err("replicated device rows require native P2P ranks".into());
        }
        let values = tokens
            .checked_mul(width)
            .ok_or("replicated device row size overflow")?;
        let rank_lengths = vec![values; self.ranks.len()];
        replicated_device_row_values(tokens, width, self.ranks.len(), &rank_lengths)?;
        let mut ranks = Vec::with_capacity(self.ranks.len());
        for engine in &self.ranks {
            let _main = engine.gpu.enter_main()?;
            ranks.push(engine.uninit(values)?);
        }
        Ok(ResidentReplicatedDeviceRows {
            ranks,
            tokens,
            width,
        })
    }

    /// Replace a fixed-shape replicated batch from a root-device source.
    pub fn refresh_replicated_device_rows_from_root(
        &self,
        rows: &mut ResidentReplicatedDeviceRows,
        source: &CudaSlice<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.ranks.len() > 1 && !self.native_p2p {
            return Err("replicated device rows require native P2P ranks".into());
        }
        validate_replicated_device_rows(&self.ranks, rows)?;
        let root = self
            .ranks
            .first()
            .ok_or("replicated rows have no root rank")?;
        let values = replicated_device_row_source_values(
            rows.tokens,
            rows.width,
            source.len(),
            source.ordinal(),
            root.ctx().ordinal(),
        )?;
        let (root_rows, peer_rows) = rows
            .ranks
            .split_first_mut()
            .ok_or("replicated rows have no root allocation")?;
        {
            let _main = root.gpu.enter_main()?;
            let mut destination = root_rows.slice_mut(0..values);
            root.stream()
                .memcpy_dtod(&source.slice(0..values), &mut destination)?;
            root.stream().synchronize()?;
        }
        for (engine, peer_rows) in self.ranks.iter().skip(1).zip(peer_rows) {
            let _main = engine.gpu.enter_main()?;
            let mut destination = peer_rows.slice_mut(0..values);
            engine
                .stream()
                .memcpy_dtod(&root_rows.slice(0..values), &mut destination)?;
        }
        Ok(())
    }

    /// Upload one canonical batch on rank zero and replicate it over native P2P.
    pub fn upload_replicated_device_rows(
        &self,
        rows: &[f32],
        tokens: usize,
        width: usize,
    ) -> Result<ResidentReplicatedDeviceRows, Box<dyn std::error::Error>> {
        if self.ranks.len() > 1 && !self.native_p2p {
            return Err("replicated device rows require native P2P ranks".into());
        }
        validate_activations(rows, tokens, width)?;
        let root = self
            .ranks
            .first()
            .ok_or("replicated rows have no root rank")?;
        let root_rows = {
            let _main = root.gpu.enter_main()?;
            root.htod(rows)?
        };
        {
            let _main = root.gpu.enter_main()?;
            root.stream().synchronize()?;
        }
        let mut ranks = Vec::with_capacity(self.ranks.len());
        ranks.push(root_rows);
        for engine in self.ranks.iter().skip(1) {
            let _main = engine.gpu.enter_main()?;
            let mut peer_rows = engine.uninit(rows.len())?;
            engine.stream().memcpy_dtod(&ranks[0], &mut peer_rows)?;
            ranks.push(peer_rows);
        }
        Ok(ResidentReplicatedDeviceRows {
            ranks,
            tokens,
            width,
        })
    }

    /// Execute a column-parallel BF16 matrix directly from rank-local replicated inputs.
    pub fn bf16_column_parallel_resident_replicated_device_shards(
        &self,
        matrix: &ResidentBf16ColumnParallel,
        activations: &ResidentReplicatedDeviceRows,
    ) -> Result<Vec<CudaSlice<f32>>, Box<dyn std::error::Error>> {
        validate_resident_bf16_ranks(&self.ranks, &matrix.ranks)?;
        validate_replicated_device_rows(&self.ranks, activations)?;
        if activations.width != matrix.in_features {
            return Err(format!(
                "replicated BF16 column input width {} != matrix width {}",
                activations.width, matrix.in_features
            )
            .into());
        }
        let mut outputs = Vec::with_capacity(self.ranks.len());
        for rank in 0..self.ranks.len() {
            outputs.push(run_resident_bf16_rank_device(
                &self.ranks[rank],
                &matrix.ranks[rank],
                &activations.ranks[rank],
                activations.tokens,
                matrix.canonical_chunk_rows,
                self.bulk_p2p,
            )?);
        }
        Ok(outputs)
    }

    /// Upload a BF16 router once on rank zero and retain its exact F32 expansion.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_sigmoid_topk_router(
        &self,
        weight: Bf16Matrix<'_>,
        correction_bias: &[f32],
        active: Option<&[bool]>,
        experts_per_token: usize,
        scaling_factor: f32,
        route_norm: bool,
    ) -> Result<ResidentSigmoidTopKRouter, Box<dyn std::error::Error>> {
        weight.validate()?;
        if correction_bias.len() != weight.out_features
            || experts_per_token == 0
            || experts_per_token > weight.out_features
            || !correction_bias.iter().all(|value| value.is_finite())
            || !scaling_factor.is_finite()
            || scaling_factor <= 0.0
        {
            return Err(format!(
                "sigmoid router geometry weight={}x{} bias={} top_k={} scale={scaling_factor}",
                weight.out_features,
                weight.in_features,
                correction_bias.len(),
                experts_per_token,
            )
            .into());
        }
        let active_row = active
            .map(|mask| {
                if mask.len() != weight.out_features {
                    return Err(format!(
                        "sigmoid router active mask {} != experts {}",
                        mask.len(),
                        weight.out_features
                    ));
                }
                Ok(mask
                    .iter()
                    .map(|&enabled| u8::from(enabled))
                    .collect::<Vec<_>>())
            })
            .transpose()?
            .unwrap_or_else(|| vec![1; weight.out_features]);
        let active_count = active_row.iter().filter(|&&enabled| enabled != 0).count();
        crate::sigrouter_contract::validate_active_count(experts_per_token, active_count)?;

        let root = self
            .ranks
            .first()
            .ok_or("sigmoid router runtime has no root rank")?;
        let _main = root.gpu.enter_main()?;
        let bf16 = root.htod_bytes(weight.bytes)?;
        let weight_f32 = root.bf16_to_f32(
            &bf16.slice(0..bf16.len()),
            weight.out_features * weight.in_features,
        )?;
        Ok(ResidentSigmoidTopKRouter {
            weight: weight_f32,
            correction_bias: root.htod(correction_bias)?,
            active: root.htod_bytes(&active_row)?,
            root_device: root.ctx().ordinal(),
            input_width: weight.in_features,
            expert_count: weight.out_features,
            experts_per_token,
            active_count,
            scaling_factor,
            route_norm,
        })
    }

    /// Route rank-zero replicated rows and return the narrow host control result plus logits.
    ///
    /// The logits readback exists for independent oracle comparison. This method is a correctness
    /// surface; a serving scheduler may retain logits and selected routes on device.
    pub fn sigmoid_topk_replicated_device_rows_host(
        &self,
        router: &ResidentSigmoidTopKRouter,
        input: &ResidentReplicatedDeviceRows,
    ) -> Result<SigmoidTopKHostOutput, Box<dyn std::error::Error>> {
        validate_replicated_device_rows(&self.ranks, input)?;
        if input.width != router.input_width {
            return Err(format!(
                "sigmoid router input width {} != resident width {}",
                input.width, router.input_width
            )
            .into());
        }
        let root = self
            .ranks
            .first()
            .ok_or("sigmoid router runtime has no root rank")?;
        let _main = root.gpu.enter_main()?;
        if root.ctx().ordinal() != router.root_device
            || router.weight.ordinal() != router.root_device
            || router.correction_bias.ordinal() != router.root_device
            || router.active.ordinal() != router.root_device
        {
            return Err("sigmoid router root residency changed".into());
        }
        let logits = root.router_gemv(
            &router.weight,
            &input.ranks[0],
            router.input_width,
            router.expert_count,
            input.tokens,
        )?;
        let (selected, weights) = root.moe_router_sigmoid_topk_host(
            &logits,
            input.tokens,
            router.expert_count,
            router.experts_per_token,
            router.active_count,
            &router.correction_bias,
            &router.active,
            router.scaling_factor,
            router.route_norm,
        )?;
        Ok(SigmoidTopKHostOutput {
            logits: root.dtoh(&logits)?,
            selected,
            weights,
        })
    }

    /// Replicate a full BF16 SwiGLU bank on every rank.
    pub fn upload_replicated_bf16_swiglu(
        &self,
        gate: Bf16Matrix<'_>,
        up: Bf16Matrix<'_>,
        down: Bf16Matrix<'_>,
    ) -> Result<ResidentReplicatedBf16SwiGlu, Box<dyn std::error::Error>> {
        gate.validate()?;
        up.validate()?;
        down.validate()?;
        if gate.in_features != up.in_features
            || gate.out_features != up.out_features
            || down.in_features != gate.out_features
            || down.out_features != gate.in_features
        {
            return Err(format!(
                "replicated BF16 SwiGLU geometry gate={}x{} up={}x{} down={}x{}",
                gate.out_features,
                gate.in_features,
                up.out_features,
                up.in_features,
                down.out_features,
                down.in_features,
            )
            .into());
        }
        let mut gate_ranks = Vec::with_capacity(self.ranks.len());
        let mut up_ranks = Vec::with_capacity(self.ranks.len());
        let mut down_ranks = Vec::with_capacity(self.ranks.len());
        for engine in &self.ranks {
            gate_ranks.push(upload_bf16_rank(engine, gate, false)?);
            up_ranks.push(upload_bf16_rank(engine, up, false)?);
            down_ranks.push(upload_bf16_rank(engine, down, false)?);
        }
        Ok(ResidentReplicatedBf16SwiGlu {
            gate: gate_ranks,
            up: up_ranks,
            down: down_ranks,
            input_width: gate.in_features,
            intermediate_width: gate.out_features,
        })
    }

    /// Execute a fully replicated BF16 SwiGLU directly from replicated device rows.
    pub fn replicated_bf16_swiglu_resident_device(
        &self,
        mlp: &ResidentReplicatedBf16SwiGlu,
        input: &ResidentReplicatedDeviceRows,
        activation_limit: Option<f32>,
    ) -> Result<ResidentReplicatedDeviceRows, Box<dyn std::error::Error>> {
        validate_step_expert_activation_limit(activation_limit)?;
        validate_replicated_device_rows(&self.ranks, input)?;
        validate_resident_bf16_ranks(&self.ranks, &mlp.gate)?;
        validate_resident_bf16_ranks(&self.ranks, &mlp.up)?;
        validate_resident_bf16_ranks(&self.ranks, &mlp.down)?;
        if input.width != mlp.input_width
            || mlp.gate.len() != self.ranks.len()
            || mlp.up.len() != self.ranks.len()
            || mlp.down.len() != self.ranks.len()
        {
            return Err("replicated BF16 SwiGLU residency or input width changed".into());
        }

        let mut outputs = Vec::with_capacity(self.ranks.len());
        for rank in 0..self.ranks.len() {
            let engine = &self.ranks[rank];
            let gate = run_resident_bf16_rank_device(
                engine,
                &mlp.gate[rank],
                &input.ranks[rank],
                input.tokens,
                None,
                self.bulk_p2p,
            )?;
            let up = run_resident_bf16_rank_device(
                engine,
                &mlp.up[rank],
                &input.ranks[rank],
                input.tokens,
                None,
                self.bulk_p2p,
            )?;
            let _main = engine.gpu.enter_main()?;
            let values = input
                .tokens
                .checked_mul(mlp.intermediate_width)
                .ok_or("replicated BF16 SwiGLU activation size overflow")?;
            let mut activation = engine.uninit(values)?;
            if let Some(limit) = activation_limit {
                engine.silu_clamped_mul_host_expf(&gate, &up, limit, &mut activation, values)?;
            } else {
                engine.silu_mul_host_expf(&gate, &up, &mut activation, values)?;
            }
            outputs.push(run_resident_bf16_rank_device(
                engine,
                &mlp.down[rank],
                &activation,
                input.tokens,
                None,
                self.bulk_p2p,
            )?);
        }
        Ok(ResidentReplicatedDeviceRows {
            ranks: outputs,
            tokens: input.tokens,
            width: mlp.input_width,
        })
    }

    /// Apply the same RMS-norm row program independently on every replicated rank.
    pub fn rms_norm_replicated_device_rows(
        &self,
        input: &ResidentReplicatedDeviceRows,
        weight: &[f32],
        eps: f32,
    ) -> Result<ResidentReplicatedDeviceRows, Box<dyn std::error::Error>> {
        validate_replicated_device_rows(&self.ranks, input)?;
        if weight.len() != input.width || !eps.is_finite() || eps <= 0.0 {
            return Err(format!(
                "replicated RMS norm weight/eps {}/{} != width {}",
                weight.len(),
                eps,
                input.width
            )
            .into());
        }
        let mut ranks = Vec::with_capacity(self.ranks.len());
        for (rank, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            let weight = engine.htod(weight)?;
            let mut output = engine.uninit(input.tokens * input.width)?;
            engine.rms_norm(
                &input.ranks[rank],
                &weight,
                &mut output,
                input.width,
                input.tokens,
                eps,
            )?;
            ranks.push(output);
        }
        Ok(ResidentReplicatedDeviceRows {
            ranks,
            tokens: input.tokens,
            width: input.width,
        })
    }

    /// Add two replicated batches and RMS-normalize the exact residual on every rank.
    pub fn add_rms_norm_replicated_device_rows(
        &self,
        input: &ResidentReplicatedDeviceRows,
        update: &ResidentReplicatedDeviceRows,
        weight: &[f32],
        eps: f32,
    ) -> Result<
        (ResidentReplicatedDeviceRows, ResidentReplicatedDeviceRows),
        Box<dyn std::error::Error>,
    > {
        validate_replicated_device_rows(&self.ranks, input)?;
        validate_replicated_device_rows(&self.ranks, update)?;
        if input.tokens != update.tokens
            || input.width != update.width
            || weight.len() != input.width
            || !eps.is_finite()
            || eps <= 0.0
        {
            return Err(format!(
                "replicated add/RMS geometry input={}x{} update={}x{} weight={} eps={eps}",
                input.tokens,
                input.width,
                update.tokens,
                update.width,
                weight.len(),
            )
            .into());
        }
        let values = input.tokens * input.width;
        let mut residual_ranks = Vec::with_capacity(self.ranks.len());
        let mut normalized_ranks = Vec::with_capacity(self.ranks.len());
        for (rank, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            let weight = engine.htod(weight)?;
            let mut residual = engine.uninit(values)?;
            let mut normalized = engine.uninit(values)?;
            engine.add_rms_norm(
                &input.ranks[rank],
                &update.ranks[rank],
                &weight,
                &mut residual,
                &mut normalized,
                input.width,
                input.tokens,
                eps,
            )?;
            residual_ranks.push(residual);
            normalized_ranks.push(normalized);
        }
        Ok((
            ResidentReplicatedDeviceRows {
                ranks: residual_ranks,
                tokens: input.tokens,
                width: input.width,
            },
            ResidentReplicatedDeviceRows {
                ranks: normalized_ranks,
                tokens: input.tokens,
                width: input.width,
            },
        ))
    }

    pub fn collect_replicated_device_rows(
        &self,
        rows: &ResidentReplicatedDeviceRows,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        validate_replicated_device_rows(&self.ranks, rows)?;
        let mut outputs = Vec::with_capacity(self.ranks.len());
        for (rank, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            outputs.push(engine.dtoh(&rows.ranks[rank])?);
        }
        Ok(outputs)
    }

    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    pub fn upload_bf16_row_parallel(
        &self,
        matrix: Bf16Matrix<'_>,
    ) -> Result<ResidentBf16RowParallel, Box<dyn std::error::Error>> {
        matrix.validate()?;
        let tp = self.ranks.len();
        if matrix.in_features % tp != 0 {
            return Err(format!(
                "BF16 row-parallel in_features {} is not divisible by TP={tp}",
                matrix.in_features
            )
            .into());
        }
        let mut ranks = Vec::with_capacity(tp);
        for (rank, engine) in self.ranks.iter().enumerate() {
            let shard = bf16_row_shard(matrix, tp, rank)?;
            ranks.push(upload_bf16_rank(
                engine,
                Bf16Matrix {
                    bytes: &shard,
                    out_features: matrix.out_features,
                    in_features: matrix.in_features / tp,
                },
                false,
            )?);
        }
        Ok(ResidentBf16RowParallel {
            ranks,
            out_features: matrix.out_features,
            in_features: matrix.in_features,
        })
    }

    pub fn bf16_row_parallel_resident(
        &self,
        matrix: &ResidentBf16RowParallel,
        activations: &[f32],
        tokens: usize,
    ) -> Result<RowParallelResult, Box<dyn std::error::Error>> {
        validate_resident_bf16_ranks(&self.ranks, &matrix.ranks)?;
        validate_activations(activations, tokens, matrix.in_features)?;
        let tp = self.ranks.len();
        let mut reduced = vec![0.0f32; tokens * matrix.out_features];
        let mut rank_partials = Vec::with_capacity(tp);
        for (rank, (engine, shard)) in self.ranks.iter().zip(&matrix.ranks).enumerate() {
            let local_activations =
                activation_shard(activations, tokens, matrix.in_features, tp, rank);
            let partial = run_resident_bf16_rank(engine, shard, &local_activations, tokens, None)?;
            for (sum, value) in reduced.iter_mut().zip(&partial) {
                *sum += value;
            }
            rank_partials.push(partial);
        }
        Ok(RowParallelResult {
            reduced,
            rank_partials,
        })
    }

    /// Step-3.7 row projection split into the same eight global K blocks for TP1/TP2/TP4/TP8.
    pub fn upload_step_bf16_row_parallel(
        &self,
        matrix: Bf16Matrix<'_>,
    ) -> Result<ResidentStepBf16RowParallel, Box<dyn std::error::Error>> {
        self.upload_step_bf16_row_parallel_inner(matrix, false)
    }

    pub fn upload_step_bf16_row_parallel_f32_mirror(
        &self,
        matrix: Bf16Matrix<'_>,
    ) -> Result<ResidentStepBf16RowParallel, Box<dyn std::error::Error>> {
        self.upload_step_bf16_row_parallel_inner(matrix, true)
    }

    fn upload_step_bf16_row_parallel_inner(
        &self,
        matrix: Bf16Matrix<'_>,
        f32_mirror: bool,
    ) -> Result<ResidentStepBf16RowParallel, Box<dyn std::error::Error>> {
        matrix.validate()?;
        let tp = self.ranks.len();
        let canonical_chunk_cols = step_bf16_canonical_chunk_cols(matrix.in_features, tp)?;
        let local_in = matrix.in_features / tp;
        let blocks_per_rank = local_in / canonical_chunk_cols;
        let mut ranks = Vec::with_capacity(tp);
        for (rank, engine) in self.ranks.iter().enumerate() {
            let mut blocks = Vec::with_capacity(blocks_per_rank);
            for block in 0..blocks_per_rank {
                let global_block = rank * blocks_per_rank + block;
                let col_start = global_block * canonical_chunk_cols;
                let bytes = bf16_row_block(matrix, col_start, canonical_chunk_cols)?;
                blocks.push(upload_bf16_rank(
                    engine,
                    Bf16Matrix {
                        bytes: &bytes,
                        out_features: matrix.out_features,
                        in_features: canonical_chunk_cols,
                    },
                    f32_mirror,
                )?);
            }
            ranks.push(blocks);
        }
        Ok(ResidentStepBf16RowParallel {
            ranks,
            out_features: matrix.out_features,
            in_features: matrix.in_features,
            canonical_chunk_cols,
        })
    }

    /// Host-staged exactness twin of [`Self::step_bf16_row_parallel_resident_native`].
    ///
    /// Block inputs and partials cross host memory, but every partial is added on the root device
    /// in global checkpoint-column order. Native transport must reproduce this result bitwise.
    pub fn step_bf16_row_parallel_resident(
        &self,
        matrix: &ResidentStepBf16RowParallel,
        activations: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        validate_step_bf16_row_residency(&self.ranks, matrix)?;
        validate_activations(activations, tokens, matrix.in_features)?;
        let root = &self.ranks[0];
        let output_len = tokens
            .checked_mul(matrix.out_features)
            .ok_or("Step BF16 row output size overflow")?;
        let mut reduced = {
            let _main = root.gpu.enter_main()?;
            root.htod(&vec![0.0f32; output_len])?
        };
        let blocks_per_rank = PRODUCT_MAX_CARDS / self.ranks.len();
        for (rank, blocks) in matrix.ranks.iter().enumerate() {
            for (block, resident) in blocks.iter().enumerate() {
                let global_block = rank * blocks_per_rank + block;
                let input = activation_shard(
                    activations,
                    tokens,
                    matrix.in_features,
                    PRODUCT_MAX_CARDS,
                    global_block,
                );
                let partial =
                    run_resident_bf16_rank(&self.ranks[rank], resident, &input, tokens, None)?;
                let next = {
                    let _main = root.gpu.enter_main()?;
                    let partial = root.htod(&partial)?;
                    let mut next = root.uninit(output_len)?;
                    root.add(&reduced, &partial, &mut next, output_len)?;
                    next
                };
                reduced = next;
            }
        }
        let _main = root.gpu.enter_main()?;
        root.dtoh(&reduced)
    }

    /// Native-P2P Step row projection with canonical global K-block reduction.
    ///
    /// The full activation is uploaded once on the root. Each TP8-sized block is peer-scattered
    /// to its owning rank, its BF16 partial is peer-returned to the root, and root-device adds
    /// replay the same eight-block order as TP1 and the host-staged oracle.
    pub fn step_bf16_row_parallel_resident_native(
        &self,
        matrix: &ResidentStepBf16RowParallel,
        activations: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if self.ranks.len() > 1 && !self.native_p2p {
            return Err("native Step BF16 row parallelism requires P2P ranks".into());
        }
        validate_step_bf16_row_residency(&self.ranks, matrix)?;
        validate_activations(activations, tokens, matrix.in_features)?;
        let root = &self.ranks[0];
        let root_input = {
            let _main = root.gpu.enter_main()?;
            root.htod(activations)?
        };
        // PRODUCER FENCE (2026-08-20 flake fix): the non-bulk arm below peer-reads root_input
        // from the other ranks' streams while root's clone_htod may still be in flight.
        {
            let _main = root.gpu.enter_main()?;
            root.stream().synchronize()?;
        }
        let reduced = self.step_bf16_row_native_reduce_from_root(matrix, &root_input, tokens)?;
        let _main = root.gpu.enter_main()?;
        root.dtoh(&reduced)
    }

    /// Device-input twin of [`Self::step_bf16_row_parallel_resident_native`] (lane/
    /// hermes-perf-fixes, 2026-08-23): the full activation arrives as a ROOT-DEVICE buffer
    /// and the reduced output stays root-resident — no DtoH of the attention output, no
    /// host O staging, no re-upload. Byte-identical to the host-canonical arm by
    /// construction (same block scatter, kernels, and global TP8 reduction order; the root
    /// bytes are dtod-copied where the host arm htod'd the same bytes). Caller must have
    /// synchronized the producer stream; the root stream is synchronized before returning.
    pub fn step_bf16_row_parallel_resident_native_device(
        &self,
        matrix: &ResidentStepBf16RowParallel,
        root_activation: &CudaSlice<f32>,
        tokens: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if self.ranks.len() > 1 && !self.native_p2p {
            return Err("native Step BF16 row parallelism requires P2P ranks".into());
        }
        validate_step_bf16_row_residency(&self.ranks, matrix)?;
        let values = tokens
            .checked_mul(matrix.in_features)
            .ok_or("device Step BF16 row activation size overflow")?;
        let root = &self.ranks[0];
        if tokens == 0
            || root_activation.len() < values
            || root_activation.ordinal() != root.ctx().ordinal()
        {
            return Err("device Step BF16 row root activation geometry mismatch".into());
        }
        let root_input = {
            let _main = root.gpu.enter_main()?;
            let mut root_input = root.uninit(values)?;
            root.stream()
                .memcpy_dtod(&root_activation.slice(0..values), &mut root_input)?;
            root.stream().synchronize()?; // producer fence, as the host-input twin
            root_input
        };
        let reduced = self.step_bf16_row_native_reduce_from_root(matrix, &root_input, tokens)?;
        let _main = root.gpu.enter_main()?;
        root.stream().synchronize()?;
        Ok(reduced)
    }

    /// Shared core of the two native Step row arms above: block scatter + rank GEMMs +
    /// canonical global TP8-order root reduction, from a root-resident input, returning the
    /// root-resident reduced output. Extracted verbatim so the host and device twins cannot
    /// drift numerically.
    fn step_bf16_row_native_reduce_from_root(
        &self,
        matrix: &ResidentStepBf16RowParallel,
        root_input: &CudaSlice<f32>,
        tokens: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let root = &self.ranks[0];
        let output_len = tokens
            .checked_mul(matrix.out_features)
            .ok_or("native Step BF16 row output size overflow")?;
        let mut reduced = {
            let _main = root.gpu.enter_main()?;
            root.htod(&vec![0.0f32; output_len])?
        };
        let blocks_per_rank = PRODUCT_MAX_CARDS / self.ranks.len();
        let mut block_input_keepalive = Vec::with_capacity(PRODUCT_MAX_CARDS);
        let mut root_packed_keepalive = Vec::with_capacity(PRODUCT_MAX_CARDS);
        let mut remote_partial_keepalive = Vec::new();
        for (rank, blocks) in matrix.ranks.iter().enumerate() {
            for (block, resident) in blocks.iter().enumerate() {
                let global_block = rank * blocks_per_rank + block;
                let col_start = global_block * matrix.canonical_chunk_cols;
                let block_len = tokens
                    .checked_mul(matrix.canonical_chunk_cols)
                    .ok_or("native Step BF16 row block size overflow")?;
                let block_input = if self.bulk_p2p {
                    let root_packed = {
                        let _main = root.gpu.enter_main()?;
                        let mut root_packed = root.uninit(block_len)?;
                        root.copy_rows_strided(
                            root_input,
                            &mut root_packed,
                            matrix.canonical_chunk_cols,
                            tokens,
                            matrix.in_features,
                            col_start,
                        )?;
                        root_packed
                    };
                    if rank == 0 {
                        root_packed
                    } else {
                        // PRODUCER FENCE (2026-08-20 flake fix): the pack kernel runs on the
                        // root stream; this rank's peer read must not overtake it.
                        {
                            let _main = root.gpu.enter_main()?;
                            root.stream().synchronize()?;
                        }
                        let engine = &self.ranks[rank];
                        let _main = engine.gpu.enter_main()?;
                        let mut block_input = engine.uninit(block_len)?;
                        engine
                            .stream()
                            .memcpy_dtod(&root_packed, &mut block_input)?;
                        root_packed_keepalive.push(root_packed);
                        block_input
                    }
                } else {
                    let engine = &self.ranks[rank];
                    let _main = engine.gpu.enter_main()?;
                    let mut block_input = engine.uninit(block_len)?;
                    for token in 0..tokens {
                        let source_start = token * matrix.in_features + col_start;
                        let source = root_input
                            .slice(source_start..source_start + matrix.canonical_chunk_cols);
                        let destination_start = token * matrix.canonical_chunk_cols;
                        let mut destination = block_input.slice_mut(
                            destination_start..destination_start + matrix.canonical_chunk_cols,
                        );
                        engine.stream().memcpy_dtod(&source, &mut destination)?;
                    }
                    block_input
                };
                let partial = run_resident_bf16_rank_device(
                    &self.ranks[rank],
                    resident,
                    &block_input,
                    tokens,
                    None,
                    self.bulk_p2p,
                )?;
                block_input_keepalive.push(block_input);
                let root_partial = if rank == 0 {
                    partial
                } else {
                    // PRODUCER FENCE (2026-08-20 flake fix): the partial was produced by this
                    // rank's kernel on its own stream; root's peer read must not overtake it.
                    {
                        let engine = &self.ranks[rank];
                        let _main = engine.gpu.enter_main()?;
                        engine.stream().synchronize()?;
                    }
                    let _main = root.gpu.enter_main()?;
                    let mut peer_partial = root.uninit(output_len)?;
                    root.stream().memcpy_dtod(&partial, &mut peer_partial)?;
                    remote_partial_keepalive.push(partial);
                    peer_partial
                };
                let next = {
                    let _main = root.gpu.enter_main()?;
                    let mut next = root.uninit(output_len)?;
                    root.add(&reduced, &root_partial, &mut next, output_len)?;
                    next
                };
                reduced = next;
            }
        }
        {
            let _main = root.gpu.enter_main()?;
            root.stream().synchronize()?;
        }
        drop(remote_partial_keepalive);
        drop(root_packed_keepalive);
        drop(block_input_keepalive);
        Ok(reduced)
    }

    /// Reduce rank-local Step attention shards in canonical TP8 K-block order and keep the result
    /// on the root device.
    pub fn step_bf16_row_parallel_resident_root_device(
        &self,
        matrix: &ResidentStepBf16RowParallel,
        rank_activations: &[CudaSlice<f32>],
        tokens: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if self.ranks.len() > 1 && !self.native_p2p {
            return Err(
                "device-resident Step BF16 row parallelism requires native P2P ranks".into(),
            );
        }
        validate_step_bf16_row_residency(&self.ranks, matrix)?;
        let local_width = matrix.in_features / self.ranks.len();
        let shard_len = tokens
            .checked_mul(local_width)
            .ok_or("device Step BF16 row shard size overflow")?;
        if tokens == 0
            || rank_activations.len() != self.ranks.len()
            || rank_activations
                .iter()
                .zip(&self.ranks)
                .any(|(rows, engine)| {
                    rows.len() != shard_len || rows.ordinal() != engine.ctx().ordinal()
                })
        {
            return Err("device Step BF16 row activation shard geometry changed".into());
        }

        let blocks_per_rank = PRODUCT_MAX_CARDS / self.ranks.len();
        let mut block_inputs = Vec::with_capacity(self.ranks.len());
        let mut partials = Vec::with_capacity(self.ranks.len());
        for (rank, blocks) in matrix.ranks.iter().enumerate() {
            if blocks.len() != blocks_per_rank {
                return Err(format!(
                    "device Step BF16 row rank {rank} blocks {} != {blocks_per_rank}",
                    blocks.len()
                )
                .into());
            }
            let engine = &self.ranks[rank];
            let _main = engine.gpu.enter_main()?;
            let mut rank_inputs = Vec::with_capacity(blocks_per_rank);
            let mut rank_partials = Vec::with_capacity(blocks_per_rank);
            for (block, resident) in blocks.iter().enumerate() {
                let block_len = tokens
                    .checked_mul(matrix.canonical_chunk_cols)
                    .ok_or("device Step BF16 row block size overflow")?;
                let mut block_input = engine.uninit(block_len)?;
                let local_col_start = block * matrix.canonical_chunk_cols;
                if self.bulk_p2p {
                    engine.copy_rows_strided(
                        &rank_activations[rank],
                        &mut block_input,
                        matrix.canonical_chunk_cols,
                        tokens,
                        local_width,
                        local_col_start,
                    )?;
                } else {
                    for token in 0..tokens {
                        let source_start = token * local_width + local_col_start;
                        let source = rank_activations[rank]
                            .slice(source_start..source_start + matrix.canonical_chunk_cols);
                        let destination_start = token * matrix.canonical_chunk_cols;
                        let mut destination = block_input.slice_mut(
                            destination_start..destination_start + matrix.canonical_chunk_cols,
                        );
                        engine.stream().memcpy_dtod(&source, &mut destination)?;
                    }
                }
                let partial = run_resident_bf16_rank_device(
                    engine,
                    resident,
                    &block_input,
                    tokens,
                    None,
                    self.bulk_p2p,
                )?;
                rank_inputs.push(block_input);
                rank_partials.push(partial);
            }
            block_inputs.push(rank_inputs);
            partials.push(rank_partials);
        }
        for engine in self.ranks.iter().skip(1) {
            let _main = engine.gpu.enter_main()?;
            engine.stream().synchronize()?;
        }

        let output_len = tokens
            .checked_mul(matrix.out_features)
            .ok_or("device Step BF16 row output size overflow")?;
        let root = &self.ranks[0];
        let _main = root.gpu.enter_main()?;
        let mut reduced = root.htod(&vec![0.0f32; output_len])?;
        let mut remote_partials = Vec::new();
        for (rank, rank_partials) in partials.into_iter().enumerate() {
            for partial in rank_partials {
                let root_partial = if rank == 0 {
                    partial
                } else {
                    let mut peer_partial = root.uninit(output_len)?;
                    root.stream().memcpy_dtod(&partial, &mut peer_partial)?;
                    remote_partials.push(partial);
                    peer_partial
                };
                let mut next = root.uninit(output_len)?;
                root.add(&reduced, &root_partial, &mut next, output_len)?;
                reduced = next;
            }
        }
        root.stream().synchronize()?;
        drop(remote_partials);
        drop(block_inputs);
        Ok(reduced)
    }

    /// Reduce rank-local Step attention shards, then replicate the canonical root result.
    pub fn step_bf16_row_parallel_resident_replicated_device(
        &self,
        matrix: &ResidentStepBf16RowParallel,
        rank_activations: &[CudaSlice<f32>],
        tokens: usize,
    ) -> Result<ResidentReplicatedDeviceRows, Box<dyn std::error::Error>> {
        let reduced =
            self.step_bf16_row_parallel_resident_root_device(matrix, rank_activations, tokens)?;
        let output_len = tokens
            .checked_mul(matrix.out_features)
            .ok_or("device Step BF16 row output size overflow")?;
        let mut ranks = Vec::with_capacity(self.ranks.len());
        ranks.push(reduced);
        for engine in self.ranks.iter().skip(1) {
            let _main = engine.gpu.enter_main()?;
            let mut peer_output = engine.uninit(output_len)?;
            engine.stream().memcpy_dtod(&ranks[0], &mut peer_output)?;
            ranks.push(peer_output);
        }
        Ok(ResidentReplicatedDeviceRows {
            ranks,
            tokens,
            width: matrix.out_features,
        })
    }

    pub fn upload_expert(
        &self,
        gate: E4m3BlockMatrix<'_>,
        up: E4m3BlockMatrix<'_>,
        down: E4m3BlockMatrix<'_>,
    ) -> Result<ResidentTpExpert, Box<dyn std::error::Error>> {
        if gate.in_features != up.in_features || gate.out_features != up.out_features {
            return Err("TP expert gate/up dimensions differ".into());
        }
        if down.in_features != gate.out_features || down.out_features != gate.in_features {
            return Err(format!(
                "TP expert down {}x{} does not invert gate/up {}x{}",
                down.out_features, down.in_features, gate.out_features, gate.in_features
            )
            .into());
        }
        Ok(ResidentTpExpert {
            gate: self.upload_column_parallel(gate)?,
            up: self.upload_column_parallel(up)?,
            down: self.upload_row_parallel(down)?,
            input_width: gate.in_features,
            expert_width: gate.out_features,
        })
    }

    pub fn run_expert(
        &self,
        expert: &ResidentTpExpert,
        input: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        validate_activations(input, tokens, expert.input_width)?;
        let gate = self.column_parallel_resident(&expert.gate, input, tokens)?;
        let up = self.column_parallel_resident(&expert.up, input, tokens)?;
        let activated: Vec<f32> = gate
            .gathered
            .iter()
            .zip(&up.gathered)
            .map(|(&gate, &up)| gate / (1.0 + (-gate).exp()) * up)
            .collect();
        debug_assert_eq!(activated.len(), tokens * expert.expert_width);
        Ok(self
            .row_parallel_resident(&expert.down, &activated, tokens)?
            .reduced)
    }

    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    pub fn upload_expert_parallel(
        &self,
        gate: E4m3ExpertBank<'_>,
        up: E4m3ExpertBank<'_>,
        down: E4m3ExpertBank<'_>,
    ) -> Result<ResidentExpertParallel, Box<dyn std::error::Error>> {
        gate.validate()?;
        up.validate()?;
        down.validate()?;
        if gate.expert_count != up.expert_count || gate.expert_count != down.expert_count {
            return Err("EP gate/up/down expert counts differ".into());
        }
        if gate.in_features != up.in_features || gate.out_features != up.out_features {
            return Err("EP gate/up dimensions differ".into());
        }
        if down.in_features != gate.out_features || down.out_features != gate.in_features {
            return Err(format!(
                "EP down {}x{} does not invert gate/up {}x{}",
                down.out_features, down.in_features, gate.out_features, gate.in_features
            )
            .into());
        }
        if gate.expert_count % self.ranks.len() != 0 {
            return Err(format!(
                "EP expert count {} is not divisible by {} ranks",
                gate.expert_count,
                self.ranks.len()
            )
            .into());
        }

        let per_rank = gate.expert_count / self.ranks.len();
        let mut ranks = Vec::with_capacity(self.ranks.len());
        for (rank, engine) in self.ranks.iter().enumerate() {
            let expert_range = rank * per_rank..(rank + 1) * per_rank;
            ranks.push(ResidentEpRank {
                gate: upload_expert_bank_rank(engine, gate, expert_range.clone())?,
                up: upload_expert_bank_rank(engine, up, expert_range.clone())?,
                down: upload_expert_bank_rank(engine, down, expert_range)?,
            });
        }
        Ok(ResidentExpertParallel {
            ranks,
            expert_count: gate.expert_count,
            input_width: gate.in_features,
            expert_width: gate.out_features,
        })
    }

    /// Prepare the official Step gate-only grouped-FP8 projection oracle on rank zero.
    ///
    /// This intentionally does not alter the resident EP path. It owns a full rank-local tensor
    /// bank solely so the grouped projection can be compared with the existing per-route oracle
    /// without routing, transport, or combine changing underneath it.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_step_grouped_fp8_gate(
        &self,
        gate: E4m3ExpertBank<'_>,
        up: E4m3ExpertBank<'_>,
        down: E4m3ExpertBank<'_>,
        input: &[f32],
        tokens: usize,
        selected: &[usize],
        activation_limit: Option<f32>,
    ) -> Result<PreparedStepGroupedFp8Gate, Box<dyn std::error::Error>> {
        gate.validate()?;
        up.validate()?;
        down.validate()?;
        validate_step_expert_activation_limit(activation_limit)?;
        if gate.expert_count != STEP_GROUPED_FP8_EXPERTS
            || up.expert_count != STEP_GROUPED_FP8_EXPERTS
            || down.expert_count != STEP_GROUPED_FP8_EXPERTS
        {
            return Err(format!(
                "official Step grouped FP8 gate requires {STEP_GROUPED_FP8_EXPERTS} experts, \
                 got gate/up/down={}/{}/{}",
                gate.expert_count, up.expert_count, down.expert_count,
            )
            .into());
        }
        if gate.in_features != up.in_features
            || gate.out_features != STEP_GROUPED_FP8_WIDTH
            || up.out_features != STEP_GROUPED_FP8_WIDTH
            || down.in_features != STEP_GROUPED_FP8_WIDTH
            || down.out_features != gate.in_features
        {
            return Err(format!(
                "official Step grouped FP8 geometry gate={}x{} up={}x{} down={}x{}",
                gate.out_features,
                gate.in_features,
                up.out_features,
                up.in_features,
                down.out_features,
                down.in_features,
            )
            .into());
        }
        validate_activations(input, tokens, gate.in_features)?;
        let pairs = tokens
            .checked_mul(STEP_GROUPED_FP8_TOP_K)
            .ok_or("official Step grouped FP8 route count overflow")?;
        if selected.len() != pairs {
            return Err(format!(
                "official Step grouped FP8 routes {} != {tokens}x{STEP_GROUPED_FP8_TOP_K} \
                 ({pairs})",
                selected.len()
            )
            .into());
        }
        for (token, routes) in selected.chunks_exact(STEP_GROUPED_FP8_TOP_K).enumerate() {
            let mut unique = routes.to_vec();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != STEP_GROUPED_FP8_TOP_K {
                return Err(format!(
                    "official Step grouped FP8 token {token} routes are not top-8 unique: \
                     {routes:?}"
                )
                .into());
            }
        }

        let engine = self
            .ranks
            .first()
            .ok_or("official Step grouped FP8 gate has no rank-zero engine")?;
        let _main = engine.gpu.enter_main()?;
        let expert_range = 0..STEP_GROUPED_FP8_EXPERTS;
        let gate = upload_expert_bank_rank(engine, gate, expert_range.clone())?;
        let up = upload_expert_bank_rank(engine, up, expert_range.clone())?;
        let down = upload_expert_bank_rank(engine, down, expert_range)?;
        let input = engine.htod(input)?;
        let route_csr = ExpertCsr::from_token_routes(
            STEP_GROUPED_FP8_EXPERTS,
            tokens,
            STEP_GROUPED_FP8_TOP_K,
            selected,
        )?
        .upload(engine)?;
        let pair_rows = (0..pairs).collect::<Vec<_>>();
        let down_csr =
            ExpertCsr::from_pair_rows(STEP_GROUPED_FP8_EXPERTS, pairs, selected, &pair_rows)?
                .upload(engine)?;
        let gate_workspace =
            Fp8GroupedWorkspace::new(engine, gate.in_features, gate.out_features, tokens, pairs)?;
        let up_workspace =
            Fp8GroupedWorkspace::new(engine, up.in_features, up.out_features, tokens, pairs)?;
        let down_workspace =
            Fp8GroupedWorkspace::new(engine, down.in_features, down.out_features, pairs, pairs)?;
        let activation = engine.uninit(pairs * STEP_GROUPED_FP8_WIDTH)?;
        Ok(PreparedStepGroupedFp8Gate {
            device: engine.ctx().ordinal(),
            gate,
            up,
            down,
            input,
            route_csr,
            down_csr,
            gate_workspace,
            up_workspace,
            down_workspace,
            activation,
            activation_limit,
            tokens,
            pairs,
        })
    }

    /// Execute one prepared gate/up/activation/down projection sequence on rank zero.
    pub fn run_step_grouped_fp8_gate(
        &self,
        plan: &mut PreparedStepGroupedFp8Gate,
    ) -> Result<StepGroupedFp8ProjectionOutput, Box<dyn std::error::Error>> {
        let engine = self
            .ranks
            .first()
            .ok_or("official Step grouped FP8 gate has no rank-zero engine")?;
        if engine.ctx().ordinal() != plan.device {
            return Err(format!(
                "official Step grouped FP8 plan device {} != rank-zero device {}",
                plan.device,
                engine.ctx().ordinal()
            )
            .into());
        }
        let _main = engine.gpu.enter_main()?;

        plan.gate_workspace.quantize(engine, &plan.input)?;
        plan.gate_workspace.project(
            engine,
            &plan.gate.codes,
            &plan.gate.scales,
            &plan.route_csr,
            plan.gate.code_stride,
            plan.gate.scale_stride,
            1.0,
        )?;
        plan.up_workspace.quantize(engine, &plan.input)?;
        plan.up_workspace.project(
            engine,
            &plan.up.codes,
            &plan.up.scales,
            &plan.route_csr,
            plan.up.code_stride,
            plan.up.scale_stride,
            1.0,
        )?;
        if let Some(limit) = plan.activation_limit {
            engine.silu_clamped_mul_host_expf(
                plan.gate_workspace.output(),
                plan.up_workspace.output(),
                limit,
                &mut plan.activation,
                plan.pairs * STEP_GROUPED_FP8_WIDTH,
            )?;
        } else {
            engine.silu_mul_host_expf(
                plan.gate_workspace.output(),
                plan.up_workspace.output(),
                &mut plan.activation,
                plan.pairs * STEP_GROUPED_FP8_WIDTH,
            )?;
        }
        plan.down_workspace.quantize(engine, &plan.activation)?;
        plan.down_workspace.project(
            engine,
            &plan.down.codes,
            &plan.down.scales,
            &plan.down_csr,
            plan.down.code_stride,
            plan.down.scale_stride,
            1.0,
        )?;

        Ok(StepGroupedFp8ProjectionOutput {
            gate: engine.dtoh(plan.gate_workspace.output())?,
            up: engine.dtoh(plan.up_workspace.output())?,
            down: engine.dtoh(plan.down_workspace.output())?,
        })
    }

    pub fn prepare_step_grouped_expert_parallel_gate(
        &self,
        experts: &ResidentExpertParallel,
        input: &[f32],
        tokens: usize,
        selected: &[usize],
        activation_limit: Option<f32>,
    ) -> Result<PreparedStepGroupedExpertParallelGate, Box<dyn std::error::Error>> {
        self.prepare_step_grouped_expert_parallel_gate_with_capacity(
            experts,
            input,
            tokens,
            selected,
            activation_limit,
            tokens,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_step_grouped_expert_parallel_gate_with_capacity(
        &self,
        experts: &ResidentExpertParallel,
        input: &[f32],
        tokens: usize,
        selected: &[usize],
        activation_limit: Option<f32>,
        max_tokens: usize,
    ) -> Result<PreparedStepGroupedExpertParallelGate, Box<dyn std::error::Error>> {
        if !self.native_p2p || !self.ep_device_arithmetic {
            return Err(
                "Step owner-grouped FP8 requires native P2P and device-resident arithmetic".into(),
            );
        }
        validate_step_expert_activation_limit(activation_limit)?;
        validate_ep_residency(&self.ranks, experts)?;
        validate_activations(input, tokens, experts.input_width)?;
        if max_tokens < tokens || max_tokens > i32::MAX as usize {
            return Err(format!(
                "official Step owner-grouped FP8 tokens {tokens} exceed capacity {max_tokens}"
            )
            .into());
        }
        if experts.expert_count != STEP_GROUPED_FP8_EXPERTS
            || experts.expert_width != STEP_GROUPED_FP8_WIDTH
        {
            return Err(format!(
                "official Step owner-grouped FP8 requires {} experts at width {}, got {} at {}",
                STEP_GROUPED_FP8_EXPERTS,
                STEP_GROUPED_FP8_WIDTH,
                experts.expert_count,
                experts.expert_width,
            )
            .into());
        }
        validate_step_grouped_owner_routes(experts.expert_count, tokens, selected)?;
        let max_pairs = max_tokens
            .checked_mul(STEP_GROUPED_FP8_TOP_K)
            .ok_or("official Step owner-grouped FP8 capacity route count overflow")?;
        let input_capacity = max_tokens
            .checked_mul(experts.input_width)
            .ok_or("official Step owner-grouped FP8 input capacity overflow")?;

        let mut rank_inputs = Vec::with_capacity(self.ranks.len());
        for engine in &self.ranks {
            let _main = engine.gpu.enter_main()?;
            rank_inputs.push(engine.uninit(input_capacity)?);
        }

        let mut owners = Vec::with_capacity(self.ranks.len());
        for (owner_rank, rank) in experts.ranks.iter().enumerate() {
            if rank.gate.expert_range != rank.up.expert_range
                || rank.gate.expert_range != rank.down.expert_range
            {
                return Err(format!(
                    "owner-grouped FP8 rank {} gate/up/down expert ranges differ",
                    owner_rank
                )
                .into());
            }
            let local_experts = rank.gate.expert_range.len();
            let engine = &self.ranks[owner_rank];
            let _main = engine.gpu.enter_main()?;
            let route_csr =
                DeviceExpertCsr::with_capacity(engine, local_experts, max_tokens, max_pairs)?;
            let down_csr =
                DeviceExpertCsr::with_capacity(engine, local_experts, max_pairs, max_pairs)?;
            let gate_workspace = Fp8GroupedWorkspace::new(
                engine,
                experts.input_width,
                experts.expert_width,
                max_tokens,
                max_pairs,
            )?;
            let up_workspace = Fp8GroupedWorkspace::new(
                engine,
                experts.input_width,
                experts.expert_width,
                max_tokens,
                max_pairs,
            )?;
            let down_workspace = Fp8GroupedWorkspace::new(
                engine,
                experts.expert_width,
                experts.input_width,
                max_pairs,
                max_pairs,
            )?;
            let activation = engine.uninit(
                max_pairs
                    .checked_mul(experts.expert_width)
                    .ok_or("official Step owner-grouped FP8 activation capacity overflow")?,
            )?;
            owners.push(PreparedStepGroupedExpertOwner {
                rank: owner_rank,
                global_pairs: Vec::new(),
                route_csr,
                down_csr,
                gate_workspace,
                up_workspace,
                down_workspace,
                activation,
            });
        }

        let mut plan = PreparedStepGroupedExpertParallelGate {
            rank_inputs,
            owners,
            activation_limit,
            tokens: 0,
            pairs: 0,
            max_tokens,
            max_pairs,
            input_width: experts.input_width,
            expert_width: experts.expert_width,
            generation: 0,
            executed_generation: None,
            ready: false,
        };
        self.refresh_step_grouped_expert_parallel_gate(
            experts, &mut plan, input, tokens, selected,
        )?;
        Ok(plan)
    }

    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    fn prepare_step_grouped_expert_parallel_refresh(
        &self,
        experts: &ResidentExpertParallel,
        plan: &PreparedStepGroupedExpertParallelGate,
        tokens: usize,
        selected: &[usize],
    ) -> Result<(usize, u64, Vec<Option<StepGroupedExpertOwnerSchedule>>), Box<dyn std::error::Error>>
    {
        validate_ep_residency(&self.ranks, experts)?;
        if plan.rank_inputs.len() != self.ranks.len()
            || plan.owners.len() != self.ranks.len()
            || plan.input_width != experts.input_width
            || plan.expert_width != experts.expert_width
            || tokens > plan.max_tokens
        {
            return Err(format!(
                "Step owner-grouped FP8 refresh geometry changed ranks={}/{} owners={}/{} \
                 input={}/{} expert={}/{} tokens={}/{}",
                plan.rank_inputs.len(),
                self.ranks.len(),
                plan.owners.len(),
                self.ranks.len(),
                plan.input_width,
                experts.input_width,
                plan.expert_width,
                experts.expert_width,
                tokens,
                plan.max_tokens,
            )
            .into());
        }
        let pairs = validate_step_grouped_owner_routes(experts.expert_count, tokens, selected)?;
        if pairs > plan.max_pairs {
            return Err(format!(
                "Step owner-grouped FP8 route count {pairs} exceeds capacity {}",
                plan.max_pairs
            )
            .into());
        }
        let next_generation = plan
            .generation
            .checked_add(1)
            .ok_or("Step owner-grouped FP8 plan generation overflow")?;
        let owner_routes = partition_expert_owner_routes(
            experts.expert_count,
            self.ranks.len(),
            tokens,
            STEP_GROUPED_FP8_TOP_K,
            selected,
        )?;
        let mut schedules = Vec::with_capacity(self.ranks.len());
        for routes in owner_routes {
            if routes.selected.is_empty() {
                schedules.push(None);
                continue;
            }
            let local_experts = experts.ranks[routes.rank].gate.expert_range.len();
            let local_pairs = routes.selected.len();
            let route_csr = ExpertCsr::from_pair_rows(
                local_experts,
                tokens,
                &routes.selected,
                &routes.token_rows,
            )?;
            let down_rows = (0..local_pairs).collect::<Vec<_>>();
            let down_csr = ExpertCsr::from_pair_rows(
                local_experts,
                local_pairs,
                &routes.selected,
                &down_rows,
            )?;
            schedules.push(Some(StepGroupedExpertOwnerSchedule {
                global_pairs: routes.global_pairs,
                route_csr,
                down_csr,
            }));
        }
        Ok((pairs, next_generation, schedules))
    }

    fn commit_step_grouped_expert_parallel_refresh(
        &self,
        plan: &mut PreparedStepGroupedExpertParallelGate,
        tokens: usize,
        pairs: usize,
        next_generation: u64,
        schedules: Vec<Option<StepGroupedExpertOwnerSchedule>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (owner, schedule) in plan.owners.iter_mut().zip(schedules) {
            let engine = &self.ranks[owner.rank];
            let _main = engine.gpu.enter_main()?;
            if let Some(schedule) = schedule {
                owner.route_csr.refresh(engine, &schedule.route_csr)?;
                owner.down_csr.refresh(engine, &schedule.down_csr)?;
                owner.global_pairs = schedule.global_pairs;
            } else {
                owner.route_csr.clear();
                owner.down_csr.clear();
                owner.global_pairs.clear();
            }
        }
        plan.tokens = tokens;
        plan.pairs = pairs;
        plan.generation = next_generation;
        plan.ready = true;
        Ok(())
    }

    pub fn refresh_step_grouped_expert_parallel_gate(
        &self,
        experts: &ResidentExpertParallel,
        plan: &mut PreparedStepGroupedExpertParallelGate,
        input: &[f32],
        tokens: usize,
        selected: &[usize],
    ) -> Result<(), Box<dyn std::error::Error>> {
        validate_activations(input, tokens, experts.input_width)?;
        let (pairs, next_generation, schedules) =
            self.prepare_step_grouped_expert_parallel_refresh(experts, plan, tokens, selected)?;

        plan.ready = false;
        plan.executed_generation = None;
        {
            let root = &self.ranks[0];
            let _main = root.gpu.enter_main()?;
            let mut destination = plan.rank_inputs[0].slice_mut(0..input.len());
            root.stream().memcpy_htod(input, &mut destination)?;
            root.stream().synchronize()?;
        }
        let (root_inputs, peer_inputs) = plan.rank_inputs.split_at_mut(1);
        let root_input = &root_inputs[0];
        for (rank, peer_input) in peer_inputs.iter_mut().enumerate() {
            let engine = &self.ranks[rank + 1];
            let _main = engine.gpu.enter_main()?;
            let mut destination = peer_input.slice_mut(0..input.len());
            engine
                .stream()
                .memcpy_dtod(&root_input.slice(0..input.len()), &mut destination)?;
        }
        self.commit_step_grouped_expert_parallel_refresh(
            plan,
            tokens,
            pairs,
            next_generation,
            schedules,
        )
    }

    /// Refresh routes and inputs from an already-resident rank-zero activation.
    ///
    /// The caller must order the source producer before this call. The root copy is completed
    /// before peer dispatch, while CSR and workspace allocations retain their stable addresses.
    pub fn refresh_step_grouped_expert_parallel_gate_from_root_device(
        &self,
        experts: &ResidentExpertParallel,
        plan: &mut PreparedStepGroupedExpertParallelGate,
        input: &CudaSlice<f32>,
        tokens: usize,
        selected: &[usize],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let input_values = tokens
            .checked_mul(experts.input_width)
            .ok_or("Step owner-grouped FP8 input size overflow")?;
        let root = self
            .ranks
            .first()
            .ok_or("Step owner-grouped FP8 runtime has no root rank")?;
        if input.len() < input_values || input.ordinal() != root.ctx().ordinal() {
            return Err(format!(
                "Step owner-grouped FP8 root input len/device {}/{} does not cover {} values on \
                 device {}",
                input.len(),
                input.ordinal(),
                input_values,
                root.ctx().ordinal(),
            )
            .into());
        }
        let (pairs, next_generation, schedules) =
            self.prepare_step_grouped_expert_parallel_refresh(experts, plan, tokens, selected)?;

        plan.ready = false;
        plan.executed_generation = None;
        {
            let _main = root.gpu.enter_main()?;
            let mut destination = plan.rank_inputs[0].slice_mut(0..input_values);
            root.stream()
                .memcpy_dtod(&input.slice(0..input_values), &mut destination)?;
            root.stream().synchronize()?;
        }
        let (root_inputs, peer_inputs) = plan.rank_inputs.split_at_mut(1);
        let root_input = &root_inputs[0];
        for (rank, peer_input) in peer_inputs.iter_mut().enumerate() {
            let engine = &self.ranks[rank + 1];
            let _main = engine.gpu.enter_main()?;
            let mut destination = peer_input.slice_mut(0..input_values);
            engine
                .stream()
                .memcpy_dtod(&root_input.slice(0..input_values), &mut destination)?;
        }
        self.commit_step_grouped_expert_parallel_refresh(
            plan,
            tokens,
            pairs,
            next_generation,
            schedules,
        )
    }

    /// Replace a fixed route plan's rank inputs from an already replicated device batch.
    ///
    /// Route CSR remains unchanged. Advancing the generation invalidates every prior projection
    /// and combine result, so callers must refresh combine metadata before executing again.
    pub fn refresh_step_grouped_expert_parallel_inputs_from_replicated(
        &self,
        experts: &ResidentExpertParallel,
        plan: &mut PreparedStepGroupedExpertParallelGate,
        input: &ResidentReplicatedDeviceRows,
    ) -> Result<(), Box<dyn std::error::Error>> {
        validate_ep_residency(&self.ranks, experts)?;
        validate_replicated_device_rows(&self.ranks, input)?;
        if !plan.ready
            || input.tokens != plan.tokens
            || input.width != plan.input_width
            || input.tokens > plan.max_tokens
            || plan.rank_inputs.len() != self.ranks.len()
            || plan.owners.len() != self.ranks.len()
            || plan.input_width != experts.input_width
            || plan.expert_width != experts.expert_width
        {
            return Err("Step owner-grouped replicated input geometry changed".into());
        }
        let values = input
            .tokens
            .checked_mul(input.width)
            .ok_or("Step owner-grouped replicated input size overflow")?;
        let next_generation = plan
            .generation
            .checked_add(1)
            .ok_or("Step owner-grouped FP8 plan generation overflow")?;
        plan.ready = false;
        plan.executed_generation = None;
        for (rank, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            let mut destination = plan.rank_inputs[rank].slice_mut(0..values);
            engine
                .stream()
                .memcpy_dtod(&input.ranks[rank], &mut destination)?;
        }
        plan.generation = next_generation;
        plan.ready = true;
        Ok(())
    }

    pub fn execute_step_grouped_expert_parallel_gate(
        &self,
        experts: &ResidentExpertParallel,
        plan: &mut PreparedStepGroupedExpertParallelGate,
    ) -> Result<(), Box<dyn std::error::Error>> {
        validate_ep_residency(&self.ranks, experts)?;
        if !plan.ready
            || plan.rank_inputs.len() != self.ranks.len()
            || plan.owners.len() != self.ranks.len()
            || plan.input_width != experts.input_width
            || plan.expert_width != experts.expert_width
        {
            return Err("Step owner-grouped FP8 plan is not ready or its geometry changed".into());
        }
        plan.executed_generation = None;

        for owner in &mut plan.owners {
            if owner.global_pairs.is_empty() {
                continue;
            }
            let engine = &self.ranks[owner.rank];
            let bank = &experts.ranks[owner.rank];
            let _main = engine.gpu.enter_main()?;
            let local_pairs = owner.global_pairs.len();
            owner.gate_workspace.quantize_for_shape(
                engine,
                &plan.rank_inputs[owner.rank],
                plan.tokens,
                local_pairs,
            )?;
            owner.gate_workspace.project(
                engine,
                &bank.gate.codes,
                &bank.gate.scales,
                &owner.route_csr,
                bank.gate.code_stride,
                bank.gate.scale_stride,
                1.0,
            )?;
            owner.up_workspace.quantize_for_shape(
                engine,
                &plan.rank_inputs[owner.rank],
                plan.tokens,
                local_pairs,
            )?;
            owner.up_workspace.project(
                engine,
                &bank.up.codes,
                &bank.up.scales,
                &owner.route_csr,
                bank.up.code_stride,
                bank.up.scale_stride,
                1.0,
            )?;
        }
        for owner in &mut plan.owners {
            if owner.global_pairs.is_empty() {
                continue;
            }
            let engine = &self.ranks[owner.rank];
            let _main = engine.gpu.enter_main()?;
            let values = owner.global_pairs.len() * plan.expert_width;
            if let Some(limit) = plan.activation_limit {
                engine.silu_clamped_mul_host_expf(
                    owner.gate_workspace.output(),
                    owner.up_workspace.output(),
                    limit,
                    &mut owner.activation,
                    values,
                )?;
            } else {
                engine.silu_mul_host_expf(
                    owner.gate_workspace.output(),
                    owner.up_workspace.output(),
                    &mut owner.activation,
                    values,
                )?;
            }
        }
        for owner in &mut plan.owners {
            if owner.global_pairs.is_empty() {
                continue;
            }
            let engine = &self.ranks[owner.rank];
            let bank = &experts.ranks[owner.rank];
            let _main = engine.gpu.enter_main()?;
            let local_pairs = owner.global_pairs.len();
            owner.down_workspace.quantize_for_shape(
                engine,
                &owner.activation,
                local_pairs,
                local_pairs,
            )?;
            owner.down_workspace.project(
                engine,
                &bank.down.codes,
                &bank.down.scales,
                &owner.down_csr,
                bank.down.code_stride,
                bank.down.scale_stride,
                1.0,
            )?;
        }
        plan.executed_generation = Some(plan.generation);
        Ok(())
    }

    pub fn collect_step_grouped_expert_parallel_gate(
        &self,
        plan: &PreparedStepGroupedExpertParallelGate,
    ) -> Result<StepGroupedFp8ProjectionOutput, Box<dyn std::error::Error>> {
        if !plan.ready || plan.executed_generation != Some(plan.generation) {
            return Err("Step owner-grouped FP8 projection is stale or has not executed".into());
        }
        let mut gate = vec![0.0f32; plan.pairs * plan.expert_width];
        let mut up = vec![0.0f32; plan.pairs * plan.expert_width];
        let mut down = vec![0.0f32; plan.pairs * plan.input_width];
        for owner in &plan.owners {
            if owner.global_pairs.is_empty() {
                continue;
            }
            let engine = &self.ranks[owner.rank];
            let _main = engine.gpu.enter_main()?;
            let owner_gate = engine.dtoh_view(
                &owner
                    .gate_workspace
                    .output()
                    .slice(0..owner.gate_workspace.output_len()),
            )?;
            let owner_up = engine.dtoh_view(
                &owner
                    .up_workspace
                    .output()
                    .slice(0..owner.up_workspace.output_len()),
            )?;
            let owner_down = engine.dtoh_view(
                &owner
                    .down_workspace
                    .output()
                    .slice(0..owner.down_workspace.output_len()),
            )?;
            for (local_pair, &global_pair) in owner.global_pairs.iter().enumerate() {
                let local_expert = local_pair * plan.expert_width;
                let global_expert = global_pair * plan.expert_width;
                gate[global_expert..global_expert + plan.expert_width]
                    .copy_from_slice(&owner_gate[local_expert..local_expert + plan.expert_width]);
                up[global_expert..global_expert + plan.expert_width]
                    .copy_from_slice(&owner_up[local_expert..local_expert + plan.expert_width]);

                let local_hidden = local_pair * plan.input_width;
                let global_hidden = global_pair * plan.input_width;
                down[global_hidden..global_hidden + plan.input_width]
                    .copy_from_slice(&owner_down[local_hidden..local_hidden + plan.input_width]);
            }
        }
        Ok(StepGroupedFp8ProjectionOutput { gate, up, down })
    }

    pub fn run_step_grouped_expert_parallel_gate(
        &self,
        experts: &ResidentExpertParallel,
        plan: &mut PreparedStepGroupedExpertParallelGate,
    ) -> Result<StepGroupedFp8ProjectionOutput, Box<dyn std::error::Error>> {
        self.execute_step_grouped_expert_parallel_gate(experts, plan)?;
        self.collect_step_grouped_expert_parallel_gate(plan)
    }

    pub fn prepare_step_grouped_expert_parallel_combine(
        &self,
        plan: &PreparedStepGroupedExpertParallelGate,
        route_weights: &[f32],
    ) -> Result<PreparedPeerWeightedRouteCombine, Box<dyn std::error::Error>> {
        if !self.native_p2p || !self.ep_device_arithmetic || !plan.ready {
            return Err(
                "Step owner-grouped combine requires a ready native-P2P device plan".into(),
            );
        }
        let owner_pairs = plan
            .owners
            .iter()
            .map(|owner| owner.global_pairs.as_slice())
            .collect::<Vec<_>>();
        let shape = validate_weighted_route_combine(
            plan.input_width,
            STEP_GROUPED_FP8_TOP_K,
            plan.max_tokens,
            plan.tokens,
            &owner_pairs,
            route_weights,
        )?;
        if shape.max_pairs != plan.max_pairs {
            return Err(format!(
                "Step owner-grouped combine capacity {} != projection capacity {}",
                shape.max_pairs, plan.max_pairs
            )
            .into());
        }
        let root = self
            .ranks
            .first()
            .ok_or("Step owner-grouped combine has no root rank")?;
        let slot_values = shape
            .max_pairs
            .checked_mul(plan.input_width)
            .ok_or("Step owner-grouped combine slot capacity overflow")?;
        let output_values = plan
            .max_tokens
            .checked_mul(plan.input_width)
            .ok_or("Step owner-grouped combine output capacity overflow")?;
        let (root_device, owners, peer_staging, slots, weights, output) = {
            let _main = root.gpu.enter_main()?;
            let mut owners = Vec::with_capacity(plan.owners.len());
            for _ in &plan.owners {
                owners.push(PreparedPeerWeightedRouteOwner {
                    token_rows: root.htod_i32(&vec![0; shape.max_pairs])?,
                    slots: root.htod_i32(&vec![0; shape.max_pairs])?,
                    weights: root.htod(&vec![0.0; shape.max_pairs])?,
                    active_pairs: 0,
                });
            }
            (
                root.ctx().ordinal(),
                owners,
                root.uninit(slot_values)?,
                root.uninit(slot_values)?,
                root.uninit(shape.max_pairs)?,
                root.uninit(output_values)?,
            )
        };
        let mut peer_devices = Vec::with_capacity(self.ranks.len().saturating_sub(1));
        let mut peer_outputs = Vec::with_capacity(self.ranks.len().saturating_sub(1));
        for engine in self.ranks.iter().skip(1) {
            let _main = engine.gpu.enter_main()?;
            peer_devices.push(engine.ctx().ordinal());
            peer_outputs.push(engine.uninit(output_values)?);
        }
        let mut combine = PreparedPeerWeightedRouteCombine {
            root_device,
            owners,
            peer_staging,
            slots,
            weights,
            output,
            peer_devices,
            peer_outputs,
            width: plan.input_width,
            experts_per_token: STEP_GROUPED_FP8_TOP_K,
            max_tokens: plan.max_tokens,
            max_pairs: shape.max_pairs,
            tokens: 0,
            pairs: 0,
            projection_generation: 0,
            output_generation: None,
            broadcast_generation: None,
            ready: false,
        };
        self.refresh_step_grouped_expert_parallel_combine(plan, &mut combine, route_weights)?;
        Ok(combine)
    }

    pub fn refresh_step_grouped_expert_parallel_combine(
        &self,
        plan: &PreparedStepGroupedExpertParallelGate,
        combine: &mut PreparedPeerWeightedRouteCombine,
        route_weights: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let output_capacity = combine
            .max_tokens
            .checked_mul(combine.width)
            .ok_or("Step owner-grouped combine output capacity overflow")?;
        if !plan.ready
            || combine.owners.len() != plan.owners.len()
            || combine.peer_devices.len() + 1 != self.ranks.len()
            || combine.peer_outputs.len() + 1 != self.ranks.len()
            || combine.width != plan.input_width
            || combine.experts_per_token != STEP_GROUPED_FP8_TOP_K
            || combine.max_tokens != plan.max_tokens
            || combine.max_pairs != plan.max_pairs
            || combine.output.len() < output_capacity
            || combine
                .peer_outputs
                .iter()
                .any(|output| output.len() < output_capacity)
        {
            return Err("Step owner-grouped combine/projection geometry changed".into());
        }
        if self
            .ranks
            .iter()
            .skip(1)
            .zip(&combine.peer_devices)
            .any(|(engine, &device)| engine.ctx().ordinal() != device)
        {
            return Err("Step owner-grouped combine peer devices changed".into());
        }
        let owner_pairs = plan
            .owners
            .iter()
            .map(|owner| owner.global_pairs.as_slice())
            .collect::<Vec<_>>();
        let shape = validate_weighted_route_combine(
            combine.width,
            combine.experts_per_token,
            combine.max_tokens,
            plan.tokens,
            &owner_pairs,
            route_weights,
        )?;
        if shape.max_pairs != combine.max_pairs {
            return Err("Step owner-grouped combine capacity changed during refresh".into());
        }
        let metadata = owner_pairs
            .iter()
            .map(|pairs| {
                let token_rows = pairs
                    .iter()
                    .map(|&pair| (pair / combine.experts_per_token) as i32)
                    .collect::<Vec<_>>();
                let slots = pairs
                    .iter()
                    .map(|&pair| (pair % combine.experts_per_token) as i32)
                    .collect::<Vec<_>>();
                let weights = pairs
                    .iter()
                    .map(|&pair| route_weights[pair])
                    .collect::<Vec<_>>();
                (token_rows, slots, weights)
            })
            .collect::<Vec<_>>();

        combine.ready = false;
        combine.output_generation = None;
        combine.broadcast_generation = None;
        let root = self
            .ranks
            .first()
            .ok_or("Step owner-grouped combine has no root rank")?;
        let _main = root.gpu.enter_main()?;
        if root.ctx().ordinal() != combine.root_device {
            return Err(format!(
                "Step owner-grouped combine root device changed {} != {}",
                root.ctx().ordinal(),
                combine.root_device
            )
            .into());
        }
        for (owner, (token_rows, slots, weights)) in combine.owners.iter_mut().zip(metadata) {
            if token_rows.is_empty() {
                owner.active_pairs = 0;
                continue;
            }
            root.htod_i32_into(&mut owner.token_rows, &token_rows)?;
            root.htod_i32_into(&mut owner.slots, &slots)?;
            let mut weight_prefix = owner.weights.slice_mut(0..weights.len());
            root.stream().memcpy_htod(&weights, &mut weight_prefix)?;
            owner.active_pairs = token_rows.len();
        }
        combine.tokens = plan.tokens;
        combine.pairs = shape.pairs;
        combine.projection_generation = plan.generation;
        combine.ready = true;
        Ok(())
    }

    pub fn execute_step_grouped_expert_parallel_combine(
        &self,
        plan: &PreparedStepGroupedExpertParallelGate,
        combine: &mut PreparedPeerWeightedRouteCombine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !plan.ready
            || plan.executed_generation != Some(plan.generation)
            || !combine.ready
            || combine.tokens != plan.tokens
            || combine.pairs != plan.pairs
            || combine.width != plan.input_width
            || combine.owners.len() != plan.owners.len()
            || combine.projection_generation != plan.generation
        {
            return Err("Step owner-grouped combine is stale or its geometry changed".into());
        }
        combine.output_generation = None;
        combine.broadcast_generation = None;
        for owner in &plan.owners {
            if owner.rank == 0 || owner.global_pairs.is_empty() {
                continue;
            }
            let engine = &self.ranks[owner.rank];
            let _main = engine.gpu.enter_main()?;
            engine.stream().synchronize()?;
        }
        let root = self
            .ranks
            .first()
            .ok_or("Step owner-grouped combine has no root rank")?;
        let _main = root.gpu.enter_main()?;
        if root.ctx().ordinal() != combine.root_device {
            return Err("Step owner-grouped combine is not resident on the root device".into());
        }
        for (index, owner) in plan.owners.iter().enumerate() {
            let metadata = &combine.owners[index];
            if owner.global_pairs.len() != metadata.active_pairs {
                return Err(format!(
                    "Step owner-grouped combine owner {index} rows {} != metadata {}",
                    owner.global_pairs.len(),
                    metadata.active_pairs
                )
                .into());
            }
            if metadata.active_pairs == 0 {
                continue;
            }
            let values = metadata
                .active_pairs
                .checked_mul(combine.width)
                .ok_or("Step owner-grouped combine peer value count overflow")?;
            if owner.rank == 0 {
                root.scatter_slot(
                    owner.down_workspace.output(),
                    &metadata.token_rows,
                    &metadata.slots,
                    &metadata.weights,
                    &mut combine.slots,
                    &mut combine.weights,
                    combine.width,
                    combine.experts_per_token,
                    metadata.active_pairs,
                )?;
            } else {
                let source = owner.down_workspace.output().slice(0..values);
                let mut destination = combine.peer_staging.slice_mut(0..values);
                root.stream().memcpy_dtod(&source, &mut destination)?;
                root.scatter_slot(
                    &combine.peer_staging,
                    &metadata.token_rows,
                    &metadata.slots,
                    &metadata.weights,
                    &mut combine.slots,
                    &mut combine.weights,
                    combine.width,
                    combine.experts_per_token,
                    metadata.active_pairs,
                )?;
            }
        }
        root.reduce_slots_host(
            &combine.slots,
            &combine.weights,
            &mut combine.output,
            combine.width,
            combine.experts_per_token,
            combine.tokens,
        )?;
        combine.output_generation = Some(plan.generation);
        Ok(())
    }

    pub fn collect_step_grouped_expert_parallel_combine(
        &self,
        plan: &PreparedStepGroupedExpertParallelGate,
        combine: &PreparedPeerWeightedRouteCombine,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if !plan.ready
            || combine.output_generation != Some(plan.generation)
            || combine.projection_generation != plan.generation
        {
            return Err("Step owner-grouped combine output is stale or has not executed".into());
        }
        let root = self
            .ranks
            .first()
            .ok_or("Step owner-grouped combine has no root rank")?;
        let _main = root.gpu.enter_main()?;
        if root.ctx().ordinal() != combine.root_device {
            return Err("Step owner-grouped combine is not resident on the root device".into());
        }
        root.dtoh_view(&combine.output.slice(0..combine.tokens * combine.width))
    }

    /// Copy the active root combine result into a caller-owned engine on the same CUDA device.
    ///
    /// The persistent combine buffer remains reusable by the next route generation; the returned
    /// allocation follows the serving runtime's ordinary transient-output ownership.
    pub fn copy_step_grouped_expert_parallel_combine_root(
        &self,
        plan: &PreparedStepGroupedExpertParallelGate,
        combine: &PreparedPeerWeightedRouteCombine,
        destination: &Engine,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if !plan.ready
            || combine.output_generation != Some(plan.generation)
            || combine.projection_generation != plan.generation
        {
            return Err("Step owner-grouped combine output is stale or has not executed".into());
        }
        let root = self
            .ranks
            .first()
            .ok_or("Step owner-grouped combine has no root rank")?;
        if root.ctx().ordinal() != combine.root_device
            || destination.ctx().ordinal() != combine.root_device
        {
            return Err(format!(
                "Step owner-grouped combine root/destination devices {}/{} != {}",
                root.ctx().ordinal(),
                destination.ctx().ordinal(),
                combine.root_device,
            )
            .into());
        }
        let values = combine
            .tokens
            .checked_mul(combine.width)
            .ok_or("Step owner-grouped combine copy size overflow")?;
        {
            let _main = root.gpu.enter_main()?;
            root.stream().synchronize()?;
        }
        let _main = destination.gpu.enter_main()?;
        let mut output = destination.uninit(values)?;
        destination
            .stream()
            .memcpy_dtod(&combine.output.slice(0..values), &mut output)?;
        Ok(output)
    }

    pub fn broadcast_step_grouped_expert_parallel_combine(
        &self,
        plan: &PreparedStepGroupedExpertParallelGate,
        combine: &mut PreparedPeerWeightedRouteCombine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !plan.ready
            || combine.output_generation != Some(plan.generation)
            || combine.projection_generation != plan.generation
            || combine.peer_devices.len() + 1 != self.ranks.len()
            || combine.peer_outputs.len() + 1 != self.ranks.len()
        {
            return Err("Step owner-grouped combine output cannot be broadcast".into());
        }
        combine.broadcast_generation = None;
        let values = combine
            .tokens
            .checked_mul(combine.width)
            .ok_or("Step owner-grouped combine broadcast size overflow")?;
        {
            let root = self
                .ranks
                .first()
                .ok_or("Step owner-grouped combine has no root rank")?;
            let _main = root.gpu.enter_main()?;
            if root.ctx().ordinal() != combine.root_device {
                return Err("Step owner-grouped combine root device changed".into());
            }
            root.stream().synchronize()?;
        }
        let source = &combine.output;
        for (index, destination_buffer) in combine.peer_outputs.iter_mut().enumerate() {
            let engine = &self.ranks[index + 1];
            let _main = engine.gpu.enter_main()?;
            if engine.ctx().ordinal() != combine.peer_devices[index] {
                return Err(format!(
                    "Step owner-grouped combine peer {} device changed",
                    index + 1
                )
                .into());
            }
            let mut destination = destination_buffer.slice_mut(0..values);
            engine
                .stream()
                .memcpy_dtod(&source.slice(0..values), &mut destination)?;
        }
        combine.broadcast_generation = Some(plan.generation);
        Ok(())
    }

    pub fn collect_step_grouped_expert_parallel_broadcast(
        &self,
        plan: &PreparedStepGroupedExpertParallelGate,
        combine: &PreparedPeerWeightedRouteCombine,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        if !plan.ready
            || combine.output_generation != Some(plan.generation)
            || combine.broadcast_generation != Some(plan.generation)
            || combine.peer_outputs.len() + 1 != self.ranks.len()
        {
            return Err("Step owner-grouped combine broadcast is stale or incomplete".into());
        }
        let values = combine
            .tokens
            .checked_mul(combine.width)
            .ok_or("Step owner-grouped combine collection size overflow")?;
        let mut outputs = Vec::with_capacity(self.ranks.len());
        {
            let root = &self.ranks[0];
            let _main = root.gpu.enter_main()?;
            outputs.push(root.dtoh_view(&combine.output.slice(0..values))?);
        }
        for (index, output) in combine.peer_outputs.iter().enumerate() {
            let engine = &self.ranks[index + 1];
            let _main = engine.gpu.enter_main()?;
            outputs.push(engine.dtoh_view(&output.slice(0..values))?);
        }
        Ok(outputs)
    }

    /// Add routed and replicated shared-expert outputs, then add the attention residual.
    pub fn finish_step_grouped_expert_parallel_layer(
        &self,
        plan: &PreparedStepGroupedExpertParallelGate,
        combine: &PreparedPeerWeightedRouteCombine,
        shared: &ResidentReplicatedDeviceRows,
        residual: &ResidentReplicatedDeviceRows,
    ) -> Result<ResidentReplicatedDeviceRows, Box<dyn std::error::Error>> {
        validate_replicated_device_rows(&self.ranks, shared)?;
        validate_replicated_device_rows(&self.ranks, residual)?;
        if !plan.ready
            || plan.executed_generation != Some(plan.generation)
            || combine.output_generation != Some(plan.generation)
            || combine.broadcast_generation != Some(plan.generation)
            || combine.projection_generation != plan.generation
            || combine.peer_outputs.len() + 1 != self.ranks.len()
            || shared.tokens != combine.tokens
            || residual.tokens != combine.tokens
            || shared.width != combine.width
            || residual.width != combine.width
        {
            return Err("Step full-layer finish inputs are stale or their geometry changed".into());
        }
        let values = combine
            .tokens
            .checked_mul(combine.width)
            .ok_or("Step full-layer output size overflow")?;
        let mut ranks = Vec::with_capacity(self.ranks.len());
        for rank in 0..self.ranks.len() {
            let engine = &self.ranks[rank];
            let _main = engine.gpu.enter_main()?;
            let routed = if rank == 0 {
                &combine.output
            } else {
                &combine.peer_outputs[rank - 1]
            };
            let mut ffn = engine.uninit(values)?;
            engine.add(routed, &shared.ranks[rank], &mut ffn, values)?;
            let mut output = engine.uninit(values)?;
            engine.add(&residual.ranks[rank], &ffn, &mut output, values)?;
            ranks.push(output);
        }
        Ok(ResidentReplicatedDeviceRows {
            ranks,
            tokens: combine.tokens,
            width: combine.width,
        })
    }

    pub fn run_step_grouped_expert_parallel_combine(
        &self,
        plan: &PreparedStepGroupedExpertParallelGate,
        combine: &mut PreparedPeerWeightedRouteCombine,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.execute_step_grouped_expert_parallel_combine(plan, combine)?;
        self.collect_step_grouped_expert_parallel_combine(plan, combine)
    }

    pub fn upload_tensor_parallel(
        &self,
        gate: E4m3ExpertBank<'_>,
        up: E4m3ExpertBank<'_>,
        down: E4m3ExpertBank<'_>,
    ) -> Result<ResidentTensorParallel, Box<dyn std::error::Error>> {
        gate.validate()?;
        up.validate()?;
        down.validate()?;
        if gate.expert_count != up.expert_count || gate.expert_count != down.expert_count {
            return Err("TP gate/up/down expert counts differ".into());
        }
        if gate.in_features != up.in_features || gate.out_features != up.out_features {
            return Err("TP gate/up dimensions differ".into());
        }
        if down.in_features != gate.out_features || down.out_features != gate.in_features {
            return Err(format!(
                "TP down {}x{} does not invert gate/up {}x{}",
                down.out_features, down.in_features, gate.out_features, gate.in_features
            )
            .into());
        }
        let tp = self.ranks.len();
        validate_column_bank_shape(gate, tp)?;
        validate_column_bank_shape(up, tp)?;
        validate_row_bank_shape(down, tp)?;

        let mut gate_ranks = Vec::with_capacity(tp);
        let mut up_ranks = Vec::with_capacity(tp);
        let mut down_ranks = Vec::with_capacity(tp);
        for (rank, engine) in self.ranks.iter().enumerate() {
            gate_ranks.push(upload_column_bank_rank(engine, gate, tp, rank)?);
            up_ranks.push(upload_column_bank_rank(engine, up, tp, rank)?);
            down_ranks.push(upload_row_bank_rank(engine, down, tp, rank)?);
        }
        Ok(ResidentTensorParallel {
            bank: ResidentTpExpertBank {
                gate: gate_ranks,
                up: up_ranks,
                down: down_ranks,
                expert_count: gate.expert_count,
                input_width: gate.in_features,
                expert_width: gate.out_features,
            },
        })
    }

    pub fn run_tensor_parallel_routes(
        &self,
        experts: &ResidentTensorParallel,
        input: &[f32],
        tokens: usize,
        selected: &[usize],
        route_weights: &[f32],
        experts_per_token: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        validate_tp_bank_residency(&self.ranks, &experts.bank)?;
        validate_activations(input, tokens, experts.bank.input_width)?;
        let pairs = tokens
            .checked_mul(experts_per_token)
            .ok_or("TP route count overflow")?;
        if selected.len() != pairs || route_weights.len() != pairs {
            return Err(format!(
                "TP routes selected={} weights={} != tokens {tokens} x experts/token \
                 {experts_per_token} ({pairs})",
                selected.len(),
                route_weights.len(),
            )
            .into());
        }
        if !route_weights.iter().all(|weight| weight.is_finite()) {
            return Err("TP route weights contain a non-finite value".into());
        }

        let mut output = vec![0.0f32; tokens * experts.bank.input_width];
        for token in 0..tokens {
            let input_row =
                &input[token * experts.bank.input_width..(token + 1) * experts.bank.input_width];
            for slot in 0..experts_per_token {
                let pair = token * experts_per_token + slot;
                let expert = selected[pair];
                if expert >= experts.bank.expert_count {
                    return Err(format!(
                        "TP selected expert {expert} outside 0..{}",
                        experts.bank.expert_count
                    )
                    .into());
                }
                let down = if self.native_p2p {
                    self.run_tensor_parallel_expert_native(&experts.bank, expert, input_row)?
                } else {
                    let gate =
                        self.run_column_bank_expert(&experts.bank.gate, expert, input_row)?;
                    let up = self.run_column_bank_expert(&experts.bank.up, expert, input_row)?;
                    let activated: Vec<f32> = gate
                        .iter()
                        .zip(&up)
                        .map(|(&gate, &up)| gate / (1.0 + (-gate).exp()) * up)
                        .collect();
                    debug_assert_eq!(activated.len(), experts.bank.expert_width);
                    self.run_row_bank_expert(&experts.bank.down, expert, &activated)?
                };
                let weight = route_weights[pair];
                for (sum, value) in output
                    [token * experts.bank.input_width..(token + 1) * experts.bank.input_width]
                    .iter_mut()
                    .zip(down)
                {
                    *sum += weight * value;
                }
            }
        }
        Ok(output)
    }

    fn run_column_bank_expert(
        &self,
        ranks: &[ResidentE4m3ExpertBankRank],
        expert: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let local_out = ranks
            .first()
            .ok_or("TP column bank has no ranks")?
            .out_features;
        let mut gathered = vec![0.0f32; local_out * ranks.len()];
        for (rank, (engine, bank)) in self.ranks.iter().zip(ranks).enumerate() {
            let shard = run_resident_bank_expert(engine, bank, expert, input, 1)?;
            gathered[rank * local_out..(rank + 1) * local_out].copy_from_slice(&shard);
        }
        Ok(gathered)
    }

    fn run_row_bank_expert(
        &self,
        ranks: &[ResidentE4m3ExpertBankRank],
        expert: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let local_in = ranks.first().ok_or("TP row bank has no ranks")?.in_features;
        if input.len() != local_in * ranks.len() {
            return Err(format!(
                "TP row input {} != {} ranks x {local_in}",
                input.len(),
                ranks.len()
            )
            .into());
        }
        let out_features = ranks[0].out_features;
        let mut reduced = vec![0.0f32; out_features];
        for (rank, (engine, bank)) in self.ranks.iter().zip(ranks).enumerate() {
            let blocks = bank
                .k_blocks
                .ok_or("TP row bank is not packed in native K-block order")?;
            if blocks * FP8_BLOCK != local_in {
                return Err(format!(
                    "TP row bank has {blocks} blocks but local input width is {local_in}"
                )
                .into());
            }
            for block in 0..blocks {
                let global_start = rank * local_in + block * FP8_BLOCK;
                let partial = run_resident_bank_expert_block(
                    engine,
                    bank,
                    expert,
                    block,
                    &input[global_start..global_start + FP8_BLOCK],
                )?;
                for (sum, value) in reduced.iter_mut().zip(partial) {
                    *sum += value;
                }
            }
        }
        Ok(reduced)
    }

    fn run_tensor_parallel_expert_native(
        &self,
        bank: &ResidentTpExpertBank,
        expert: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if !self.native_p2p || self.ranks.len() < 2 {
            return Err("native TP expert execution requires at least two P2P ranks".into());
        }
        let local_out = bank
            .gate
            .first()
            .ok_or("native TP gate bank has no ranks")?
            .out_features;
        if local_out * self.ranks.len() != bank.expert_width {
            return Err(format!(
                "native TP gate shards {}x{local_out} != expert width {}",
                self.ranks.len(),
                bank.expert_width
            )
            .into());
        }

        // The caller's routed input is already host-canonical. Upload once on rank zero, then
        // broadcast over peer copies so no other rank receives a host-staged duplicate.
        let mut rank_inputs = Vec::with_capacity(self.ranks.len());
        let root_input = {
            let root = &self.ranks[0];
            let _main = root.gpu.enter_main()?;
            root.htod(input)?
        };
        rank_inputs.push(root_input);
        for engine in &self.ranks[1..] {
            let peer_input = {
                let _main = engine.gpu.enter_main()?;
                let mut peer_input = engine.uninit(input.len())?;
                engine
                    .stream()
                    .memcpy_dtod(&rank_inputs[0], &mut peer_input)?;
                peer_input
            };
            rank_inputs.push(peer_input);
        }

        let mut gate_shards = Vec::with_capacity(self.ranks.len());
        let mut up_shards = Vec::with_capacity(self.ranks.len());
        #[allow(clippy::needless_range_loop)]
        // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
        for rank in 0..self.ranks.len() {
            gate_shards.push(run_resident_bank_expert_device(
                &self.ranks[rank],
                &bank.gate[rank],
                expert,
                &rank_inputs[rank],
                1,
            )?);
            up_shards.push(run_resident_bank_expert_device(
                &self.ranks[rank],
                &bank.up[rank],
                expert,
                &rank_inputs[rank],
                1,
            )?);
        }

        // Preserve the established canonical activation program for the first native transport
        // milestone. The shards move to rank zero over P2P; only the scalar activation expression
        // executes on host. A later device-activation increment must earn its own exactness gate.
        let gate = self.gather_native_column_shards(&gate_shards, 1, local_out)?;
        let up = self.gather_native_column_shards(&up_shards, 1, local_out)?;
        let activated = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| gate / (1.0 + (-gate).exp()) * up)
            .collect::<Vec<_>>();
        debug_assert_eq!(activated.len(), bank.expert_width);

        let root_activated = {
            let root = &self.ranks[0];
            let _main = root.gpu.enter_main()?;
            root.htod(&activated)?
        };
        let mut rank_activated = Vec::with_capacity(self.ranks.len());
        for (rank, engine) in self.ranks.iter().enumerate() {
            let start = rank * local_out;
            let source = root_activated.slice(start..start + local_out);
            let local = {
                let _main = engine.gpu.enter_main()?;
                let mut local = engine.uninit(local_out)?;
                engine.stream().memcpy_dtod(&source, &mut local)?;
                local
            };
            rank_activated.push(local);
        }

        let out_features = bank
            .down
            .first()
            .ok_or("native TP down bank has no ranks")?
            .out_features;
        let mut reduced = {
            let root = &self.ranks[0];
            let _main = root.gpu.enter_main()?;
            root.htod(&vec![0.0f32; out_features])?
        };
        let mut remote_partial_keepalive = Vec::new();
        #[allow(clippy::needless_range_loop)]
        // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
        for rank in 0..self.ranks.len() {
            let down = &bank.down[rank];
            let blocks = down
                .k_blocks
                .ok_or("native TP row bank is not packed in checkpoint-block order")?;
            if blocks * FP8_BLOCK != local_out {
                return Err(format!(
                    "native TP rank {rank} has {blocks} blocks but local activation width is \
                     {local_out}"
                )
                .into());
            }
            for block in 0..blocks {
                let start = block * FP8_BLOCK;
                let input_block = rank_activated[rank].slice(start..start + FP8_BLOCK);
                let partial = run_resident_bank_expert_block_device(
                    &self.ranks[rank],
                    down,
                    expert,
                    block,
                    &input_block,
                )?;
                let root_partial = if rank == 0 {
                    partial
                } else {
                    let root = &self.ranks[0];
                    let _main = root.gpu.enter_main()?;
                    let mut peer_partial = root.uninit(out_features)?;
                    root.stream().memcpy_dtod(&partial, &mut peer_partial)?;
                    remote_partial_keepalive.push(partial);
                    peer_partial
                };
                let next = {
                    let root = &self.ranks[0];
                    let _main = root.gpu.enter_main()?;
                    let mut next = root.uninit(out_features)?;
                    root.add(&reduced, &root_partial, &mut next, out_features)?;
                    next
                };
                reduced = next;
            }
        }
        let output = {
            let root = &self.ranks[0];
            let _main = root.gpu.enter_main()?;
            root.dtoh(&reduced)?
        };
        drop(remote_partial_keepalive);
        Ok(output)
    }

    /// Gather token-major rank-local columns into one canonical root-device matrix.
    pub fn gather_native_column_shards_device(
        &self,
        shards: &[CudaSlice<f32>],
        tokens: usize,
        local_out: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let shard_len = tokens
            .checked_mul(local_out)
            .ok_or("native TP gather shard size overflow")?;
        if shards.len() != self.ranks.len() || shards.iter().any(|shard| shard.len() != shard_len) {
            return Err("native TP gather shard geometry mismatch".into());
        }
        // PRODUCER FENCE (2026-08-20 flake fix): the root stream peer-reads shards produced on
        // the other ranks' streams; without fencing those producers the copy can read a partial
        // kernel output.
        for engine in &self.ranks[1..] {
            let _main = engine.gpu.enter_main()?;
            engine.stream().synchronize()?;
        }
        let root = &self.ranks[0];
        let _main = root.gpu.enter_main()?;
        let global_out = shards
            .len()
            .checked_mul(local_out)
            .ok_or("native TP gather output width overflow")?;
        let gathered_len = tokens
            .checked_mul(global_out)
            .ok_or("native TP gather output size overflow")?;
        let mut gathered = root.uninit(gathered_len)?;
        if self.bulk_p2p {
            root.place_rows_strided(&shards[0], &mut gathered, local_out, tokens, global_out, 0)?;
            if shards.len() > 1 {
                let mut staging = root.uninit(shard_len)?;
                for (rank, shard) in shards.iter().enumerate().skip(1) {
                    root.stream().memcpy_dtod(shard, &mut staging)?;
                    root.place_rows_strided(
                        &staging,
                        &mut gathered,
                        local_out,
                        tokens,
                        global_out,
                        rank * local_out,
                    )?;
                }
            }
        } else {
            for token in 0..tokens {
                for (rank, shard) in shards.iter().enumerate() {
                    let source = shard.slice(token * local_out..(token + 1) * local_out);
                    let start = token * global_out + rank * local_out;
                    let mut destination = gathered.slice_mut(start..start + local_out);
                    root.stream().memcpy_dtod(&source, &mut destination)?;
                }
            }
        }
        Ok(gathered)
    }

    pub fn gather_native_column_shards(
        &self,
        shards: &[CudaSlice<f32>],
        tokens: usize,
        local_out: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let gathered = self.gather_native_column_shards_device(shards, tokens, local_out)?;
        let root = &self.ranks[0];
        let _main = root.gpu.enter_main()?;
        root.dtoh(&gathered)
    }

    pub(crate) fn decode_v2_workspace(&self) -> &std::sync::Mutex<Vec<StepTpDecodeV2Ws>> {
        &self.decode_v2
    }

    /// Build the v2 decode-attention workspace for this layer's geometry on first use, or
    /// return the index of the matching one. Attention geometry varies across the trunk
    /// (per-layer query-head counts), so workspaces are keyed by their geometry pins — a
    /// handful exist per model, never one per layer.
    ///
    /// Refuses non-F32-resident projections: the v2 driver's bit-exactness claim against v1
    /// holds per residency class, and only the mirror class has no per-call weight expansion
    /// to hide allocation churn behind.
    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    pub(crate) fn decode_v2_ensure(
        &self,
        e: &Engine,
        q_m: &ResidentBf16ColumnParallel,
        k_m: &ResidentBf16ColumnParallel,
        v_m: &ResidentBf16ColumnParallel,
        o_m: &ResidentStepBf16RowParallel,
        heads: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if self.ranks.len() > 1 && !self.native_p2p {
            return Err("step TP decode v2 requires native P2P ranks".into());
        }
        let ranks = self.ranks.len();
        // Residency contract: the canonical-chunk (non-fused) program needs the F32 mirror;
        // the fused-kernel door also reads raw checkpoint bf16 directly (halving the weight
        // traffic), so bf16 residency is accepted when that door is on.
        let fused_door = step_tp_qkv_fused_enabled()?;
        let arm_ok = |weight: &ResidentBf16Weight| match weight {
            ResidentBf16Weight::F32(_) => true,
            ResidentBf16Weight::Bf16(_) => fused_door,
        };
        for matrix in [q_m, k_m, v_m] {
            validate_resident_bf16_ranks(&self.ranks, &matrix.ranks)?;
            if matrix.out_features % ranks != 0 || matrix.in_features != q_m.in_features {
                return Err("step TP decode v2 QKV geometry mismatch".into());
            }
            for rank in &matrix.ranks {
                if !arm_ok(&rank.weight) {
                    return Err("step TP decode v2 requires MEMRA_STEP_TP_F32_MIRROR=1 or \
                                MEMRA_STEP_TP_QKV_FUSED=1 (bf16-resident fused kernels)"
                        .into());
                }
            }
        }
        validate_step_bf16_row_residency(&self.ranks, o_m)?;
        for blocks in &o_m.ranks {
            for block in blocks {
                if !arm_ok(&block.weight) {
                    return Err("step TP decode v2 requires MEMRA_STEP_TP_F32_MIRROR=1 or \
                                MEMRA_STEP_TP_QKV_FUSED=1 (bf16-resident fused kernels)"
                        .into());
                }
            }
        }
        if v_m.out_features != k_m.out_features
            || o_m.in_features != q_m.out_features
            || heads == 0
            || heads % ranks != 0
        {
            return Err("step TP decode v2 K/V/O geometry mismatch".into());
        }
        let local_q_dim = q_m.out_features / ranks;
        let local_kv_dim = k_m.out_features / ranks;
        let o_out = o_m.out_features;
        let o_block_cols = o_m.canonical_chunk_cols;
        let blocks_per_rank = o_m.ranks.first().map(Vec::len).unwrap_or(0);
        if blocks_per_rank == 0
            || o_m
                .ranks
                .iter()
                .any(|blocks| blocks.len() != blocks_per_rank)
            || blocks_per_rank * o_block_cols * ranks != o_m.in_features
        {
            return Err("step TP decode v2 O canonical block grid mismatch".into());
        }

        let mut guard = self
            .decode_v2
            .lock()
            .map_err(|_| "step TP decode v2 workspace lock is poisoned")?;
        if let Some(index) = guard.iter().position(|ws| {
            ws.local_q_dim == local_q_dim
                && ws.local_kv_dim == local_kv_dim
                && ws.heads == heads
                && ws.o_out == o_out
                && ws.o_block_cols == o_block_cols
                && ws.blocks_per_rank == blocks_per_rank
                && ws.e_device == e.ctx().ordinal()
                && ws.q.len() == ranks
        }) {
            return Ok(index);
        }

        let mut q_raw = Vec::with_capacity(ranks);
        let mut k_raw = Vec::with_capacity(ranks);
        let mut v_raw = Vec::with_capacity(ranks);
        let mut q = Vec::with_capacity(ranks);
        let mut k = Vec::with_capacity(ranks);
        let mut pos = Vec::with_capacity(ranks);
        let mut gate = Vec::with_capacity(ranks);
        let mut attn_out = Vec::with_capacity(ranks);
        let mut gated = Vec::with_capacity(ranks);
        let mut fuse_ctr = Vec::with_capacity(ranks);
        let mut o_partials = Vec::with_capacity(ranks);
        let mut ev_rank = Vec::with_capacity(ranks);
        let direct_join = oproj_direct_on();
        for (rank, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            q_raw.push(engine.uninit(local_q_dim)?);
            k_raw.push(engine.uninit(local_kv_dim)?);
            v_raw.push(engine.uninit(local_kv_dim)?);
            q.push(engine.uninit(local_q_dim)?);
            k.push(engine.uninit(local_kv_dim)?);
            pos.push(engine.htod_i32(&[0])?);
            fuse_ctr.push(engine.stream().clone_htod(&[0u32])?);
            gate.push(engine.uninit(heads / ranks)?);
            attn_out.push(engine.uninit(local_q_dim)?);
            gated.push(engine.uninit(local_q_dim)?);
            let mut rank_partials = Vec::with_capacity(blocks_per_rank);
            for _ in 0..blocks_per_rank {
                // Direct join: peer ranks' partials live on ROOT so the b4 kernel's
                // stores land there over P2P (UVA) and no pull copy is needed.
                if direct_join && rank != 0 {
                    let root = &self.ranks[0];
                    let _root_main = root.gpu.enter_main()?;
                    rank_partials.push(root.uninit(o_out)?);
                } else {
                    rank_partials.push(engine.uninit(o_out)?);
                }
            }
            o_partials.push(rank_partials);
            ev_rank.push(engine.ctx().new_event(None)?);
        }
        use cudarc::driver::DevicePtr;
        let mut raw_o_partials = Vec::with_capacity(ranks);
        let mut raw_k = Vec::with_capacity(ranks);
        let mut raw_v_raw = Vec::with_capacity(ranks);
        for rank in 0..ranks {
            let engine = &self.ranks[rank];
            {
                let _main = engine.gpu.enter_main()?;
                let stream = engine.stream();
                let (k_ptr, _k_guard) = k[rank].device_ptr(&stream);
                let (v_ptr, _v_guard) = v_raw[rank].device_ptr(&stream);
                raw_k.push(k_ptr);
                raw_v_raw.push(v_ptr);
            }
            let partial_engine = if direct_join && rank != 0 {
                &self.ranks[0]
            } else {
                engine
            };
            let _main = partial_engine.gpu.enter_main()?;
            let stream = partial_engine.stream();
            let mut rank_raw = Vec::with_capacity(blocks_per_rank);
            for partial in &o_partials[rank] {
                let (ptr, _guard) = partial.device_ptr(&stream);
                rank_raw.push(ptr);
            }
            raw_o_partials.push(rank_raw);
        }
        let root = &self.ranks[0];
        let (peer_partial, reduce_a, reduce_b, zeros, k_shadow, v_shadow, ev_refresh, ev_oproj) = {
            let _main = root.gpu.enter_main()?;
            (
                root.uninit(o_out)?,
                root.uninit(o_out)?,
                root.uninit(o_out)?,
                root.htod(&vec![0.0f32; o_out])?,
                root.uninit(ranks * local_kv_dim)?,
                root.uninit(ranks * local_kv_dim)?,
                root.ctx().new_event(None)?,
                root.ctx().new_event(None)?,
            )
        };
        let (raw_peer_partial, raw_k_shadow, raw_v_shadow) = {
            let _main = root.gpu.enter_main()?;
            let stream = root.stream();
            let (peer, _peer_guard) = peer_partial.device_ptr(&stream);
            let (k, _k_guard) = k_shadow.device_ptr(&stream);
            let (v, _v_guard) = v_shadow.device_ptr(&stream);
            (peer, k, v)
        };
        let (gate_e, ev_entry) = {
            let _main = e.gpu.enter_main()?;
            (e.uninit(heads)?, e.ctx().new_event(None)?)
        };
        let raw_attn_in = Vec::new();
        let raw_pos = Vec::new();
        guard.push(StepTpDecodeV2Ws {
            tcol_q: Vec::new(),
            tcol_k: Vec::new(),
            tcol_v: Vec::new(),
            tcol_g: Vec::new(),
            tcol_in: Vec::new(),
            tcol_cap: 0,
            w8_aq: Vec::new(),
            w8_ad: Vec::new(),
            w8_in: 0,
            w8o_aq: Vec::new(),
            w8o_ad: Vec::new(),
            w8o_in: 0,
            w8t_aq: Vec::new(),
            w8t_ad: Vec::new(),
            w8t_in: 0,
            w8t_oaq: Vec::new(),
            w8t_oad: Vec::new(),
            w8t_oin: 0,
            w8t_cap: 0,
            fa2_q: Vec::new(),
            fa2_gate: Vec::new(),
            fa2_gated: Vec::new(),
            fa2_cap: 0,
            rope_k_t: Vec::new(),
            rope_ctr_t: Vec::new(),
            rope_pos_t: Vec::new(),
            rows_tabs: Vec::new(),
            rows_tab_t: Vec::new(),
            rows_tab_shadow: Vec::new(),
            tcol_gated: Vec::new(),
            tcol_opart: Vec::new(),
            tcol_opeer: None,
            tcol_omix: None,
            tcol_ocap: 0,
            q_raw,
            k_raw,
            v_raw,
            q,
            k,
            pos,
            fuse_ctr,
            gate,
            attn_out,
            gated,
            o_partials,
            raw_o_partials,
            raw_k,
            raw_v_raw,
            ev_rank,
            peer_partial,
            reduce_a,
            reduce_b,
            zeros,
            k_shadow,
            v_shadow,
            ev_refresh,
            ev_oproj,
            gate_e,
            attn_in: Vec::new(),
            h_stage: None,
            pos_stage: None,
            raw_h_stage: 0,
            raw_pos_stage: 0,
            raw_attn_in,
            raw_pos,
            raw_o_partial1: 0,
            raw_peer_partial,
            raw_k1: 0,
            raw_v1: 0,
            raw_k_shadow,
            raw_v_shadow,
            raw_mixed_stage_e: 0,
            raw_reduce_a: 0,
            raw_shadow_stage_e: (0, 0),
            ev_entry,
            e_device: e.ctx().ordinal(),
            local_q_dim,
            local_kv_dim,
            heads,
            o_out,
            o_block_cols,
            blocks_per_rank,
        });
        eprintln!(
            "[step-tp-decode-v2] workspace ranks={ranks} local_q={local_q_dim} \
             local_kv={local_kv_dim} heads={heads} o_blocks={blocks_per_rank}x{o_block_cols} \
             residency=persistent ordering=evented performance_claim=false"
        );
        Ok(guard.len() - 1)
    }

    /// v2 phase 1: replicate the layer input, project QKV, norm, rope, and stage the gate —
    /// all into the persistent workspace, ordered by events instead of host syncs.
    ///
    /// The caller must have queued every producer of `h`, `pos_d`, and `gate_raw` on `e`'s
    /// stream BEFORE this call: `ev_entry` is recorded once here and every rank stream waits
    /// on it (the entry fence also guards workspace reuse across layers — any consumer of the
    /// previous layer's outputs was queued on `e`'s stream before this record).
    #[allow(clippy::too_many_arguments)]
    /// T-COLUMN verify precompute (spec MTP): stage T input rows to every rank and run the
    /// weight-amortized qkvg_tcol per rank into the ws slabs. Rope/norm/append stay per
    /// column in the unmodified t=1 program (defer_norm_rope contract). Bit-exact per
    /// column vs the t=1 kernel by construction.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_v2_input_qkv_tcol(
        &self,
        ws_index: usize,
        e: &Engine,
        h_t: &CudaSlice<f32>,
        t: usize,
        q_m: &ResidentBf16ColumnParallel,
        k_m: &ResidentBf16ColumnParallel,
        v_m: &ResidentBf16ColumnParallel,
        gate_shards: Option<StepTpGateShards<'_>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ranks = self.ranks.len();
        let mut guard = self
            .decode_v2
            .lock()
            .map_err(|_| "step TP decode v2 workspace lock is poisoned")?;
        let ws = guard
            .get_mut(ws_index)
            .ok_or("step TP decode v2 workspace index out of range")?;
        let in_f = q_m.in_features;
        if h_t.len() < t * in_f || t == 0 || t > 32 {
            return Err("decode_v2_input_qkv_tcol geometry".into());
        }
        // Lazily arm the slabs to capacity.
        if ws.tcol_cap < t || ws.tcol_q.len() != ranks {
            ws.tcol_q.clear();
            ws.tcol_k.clear();
            ws.tcol_v.clear();
            ws.tcol_g.clear();
            ws.tcol_in.clear();
            for engine in &self.ranks {
                let _m = engine.gpu.enter_main()?;
                ws.tcol_q.push(engine.uninit(32 * ws.local_q_dim)?);
                ws.tcol_k.push(engine.uninit(32 * ws.local_kv_dim)?);
                ws.tcol_v.push(engine.uninit(32 * ws.local_kv_dim)?);
                ws.tcol_g
                    .push(engine.uninit(32 * (ws.heads / ranks).max(1))?);
                ws.tcol_in.push(engine.uninit(32 * in_f)?);
            }
            ws.tcol_cap = 32;
        }
        // Stage the T input rows on e, fence, per-rank pull + tcol launch.
        use cudarc::driver::DevicePtr;
        let raw_src = {
            let _main = e.gpu.enter_main()?;
            let stream = e.stream();
            let (p, _g) = h_t.device_ptr(&stream);
            ws.ev_entry.record(&stream)?;
            p
        };
        for rank in 0..ranks {
            let engine = &self.ranks[rank];
            let _main = engine.gpu.enter_main()?;
            engine.stream().wait(&ws.ev_entry)?;
            let raw_dst = {
                let stream = engine.stream();
                let (p, _g) = ws.tcol_in[rank].device_ptr(&stream);
                p
            };
            raw_copy_bytes(raw_dst, raw_src, t * in_f * 4, engine)?;
            let out_g = match &gate_shards {
                Some(_) => ws.heads / ranks,
                None => 0,
            };
            match (
                &q_m.ranks[rank].weight,
                &k_m.ranks[rank].weight,
                &v_m.ranks[rank].weight,
            ) {
                (
                    ResidentBf16Weight::Bf16(wq),
                    ResidentBf16Weight::Bf16(wk),
                    ResidentBf16Weight::Bf16(wv),
                ) => {
                    let wg = match &gate_shards {
                        Some(StepTpGateShards::Bf16(shards)) => &shards[rank],
                        Some(StepTpGateShards::F32(_)) => {
                            return Err(
                                "tcol verify: gate shard class does not match bf16 QKV".into()
                            );
                        }
                        None => wq,
                    };
                    let StepTpDecodeV2Ws {
                        tcol_q,
                        tcol_k,
                        tcol_v,
                        tcol_g,
                        tcol_in,
                        local_q_dim,
                        local_kv_dim,
                        w8t_aq,
                        w8t_ad,
                        w8t_in,
                        w8t_cap,
                        ..
                    } = &mut *ws;
                    // MEMRA_TCOL_REFKERN=1 (bisect): fill the slabs via the t=1 kernel per
                    // column — separates driver bugs from tcol-kernel bugs.
                    static REFK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    let refk = *REFK
                        .get_or_init(|| std::env::var("MEMRA_TCOL_REFKERN").as_deref() == Ok("1"));
                    if refk {
                        let lq = *local_q_dim;
                        let lkv = *local_kv_dim;
                        let mut hrow = engine.uninit(in_f)?;
                        let mut qr = engine.uninit(lq)?;
                        let mut kr = engine.uninit(lkv)?;
                        let mut vr = engine.uninit(lkv)?;
                        let mut gr = engine.uninit(out_g.max(1))?;
                        for c in 0..t {
                            {
                                let mut dst = hrow.slice_mut(0..in_f);
                                engine.stream().memcpy_dtod(
                                    &tcol_in[rank].slice(c * in_f..(c + 1) * in_f),
                                    &mut dst,
                                )?;
                            }
                            engine.matvec_bf16_qkvg_into(
                                wq, wk, wv, wg, &hrow, &mut qr, &mut kr, &mut vr, &mut gr, in_f,
                                lq, lkv, out_g,
                            )?;
                            let stream = engine.stream();
                            {
                                let mut dst = tcol_q[rank].slice_mut(c * lq..(c + 1) * lq);
                                stream.memcpy_dtod(&qr.slice(0..lq), &mut dst)?;
                            }
                            {
                                let mut dst = tcol_k[rank].slice_mut(c * lkv..(c + 1) * lkv);
                                stream.memcpy_dtod(&kr.slice(0..lkv), &mut dst)?;
                            }
                            {
                                let mut dst = tcol_v[rank].slice_mut(c * lkv..(c + 1) * lkv);
                                stream.memcpy_dtod(&vr.slice(0..lkv), &mut dst)?;
                            }
                            if out_g > 0 {
                                let mut dst = tcol_g[rank].slice_mut(c * out_g..(c + 1) * out_g);
                                stream.memcpy_dtod(&gr.slice(0..out_g), &mut dst)?;
                            }
                        }
                    } else if crate::step_tp_w8_on()
                        && q_m.ranks[rank].q8.is_some()
                        && k_m.ranks[rank].q8.is_some()
                        && v_m.ranks[rank].q8.is_some()
                        && in_f.is_multiple_of(32)
                    {
                        // MEMRA_STEP_TP_W8 on the VERIFY walk. nsys put the bf16 tcol QKV at
                        // 12.3% of spec GPU time and the bf16 tcol o_proj at 24.8% — the door
                        // had only ever replaced the DECODE kernels, so 37% of the verify still
                        // streamed bf16 weights. One q8 launch over all t columns; the gate rows
                        // stay bf16 as on the decode side.
                        if *w8t_in != in_f || *w8t_cap < t || w8t_aq.len() != ranks {
                            w8t_aq.clear();
                            w8t_ad.clear();
                            for e_rank in &self.ranks {
                                let _m = e_rank.gpu.enter_main()?;
                                w8t_aq.push(e_rank.alloc_i8_uninit(32 * in_f)?);
                                w8t_ad.push(e_rank.alloc_uninit::<f32>(32 * (in_f / 32))?);
                            }
                            *w8t_in = in_f;
                            *w8t_cap = 32;
                        }
                        engine.quantize_q8_1_into(
                            &tcol_in[rank],
                            t,
                            in_f,
                            &mut w8t_aq[rank],
                            &mut w8t_ad[rank],
                        )?;
                        engine.qmatvec_q8_0_qkv_rp_t_into(
                            q_m.ranks[rank].q8.as_ref().unwrap(),
                            k_m.ranks[rank].q8.as_ref().unwrap(),
                            v_m.ranks[rank].q8.as_ref().unwrap(),
                            &w8t_aq[rank],
                            &w8t_ad[rank],
                            &mut tcol_q[rank],
                            &mut tcol_k[rank],
                            &mut tcol_v[rank],
                            in_f,
                            *local_q_dim,
                            *local_kv_dim,
                            t,
                        )?;
                        if out_g > 0 {
                            engine.matvec_bf16_rows_into(
                                wg,
                                &tcol_in[rank],
                                &mut tcol_g[rank],
                                in_f,
                                out_g,
                                t,
                            )?;
                        }
                    } else {
                        engine.matvec_bf16_qkvg_tcol_into(
                            wq,
                            wk,
                            wv,
                            wg,
                            &tcol_in[rank],
                            &mut tcol_q[rank],
                            &mut tcol_k[rank],
                            &mut tcol_v[rank],
                            &mut tcol_g[rank],
                            in_f,
                            *local_q_dim,
                            *local_kv_dim,
                            out_g,
                            t,
                        )?;
                    }
                }
                _ => return Err("tcol verify requires bf16-resident fused QKV".into()),
            }
        }
        Ok(())
    }

    /// MEMRA_TCOL_OPROJ eligibility: the defer replaces exactly the o_fused direct-join
    /// finish (bf16 b4 kernel, 2 ranks, 4 canonical blocks) with the shadow gathers
    /// skipped — so it requires the same doors that arm dictate that finish shape.
    pub(crate) fn decode_v2_oproj_tcol_eligible(
        &self,
        ws: &StepTpDecodeV2Ws,
        o_m: &ResidentStepBf16RowParallel,
    ) -> bool {
        self.ranks.len() == 2
            && ws.blocks_per_rank == 4
            && step_tp_qkv_fused_enabled().unwrap_or(false)
            && no_local_shadow_on()
            && o_m
                .ranks
                .iter()
                .flatten()
                .all(|block| matches!(block.weight, ResidentBf16Weight::Bf16(_)))
    }

    /// MEMRA_SPEC_FA2 stash: copy this column's per-rank post-rope q and gate rows into
    /// the fa2 slabs (rank-stream ordered behind the rope/append that produced them), and
    /// give `e` the same anti-dependency wait the skipped finish provided (next column's
    /// h/pos re-staging must not overtake this column's rank pulls).
    pub(crate) fn decode_v2_stash_fa2(
        &self,
        ws: &mut StepTpDecodeV2Ws,
        e: &Engine,
        col: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ranks = self.ranks.len();
        if col >= 32 {
            return Err("decode_v2_stash_fa2 column out of range".into());
        }
        let lq = ws.local_q_dim;
        let lg = (ws.heads / ranks).max(1);
        if ws.fa2_cap < 32 || ws.fa2_q.len() != ranks || ws.rows_tab_t.len() != ranks {
            ws.fa2_q.clear();
            ws.fa2_gate.clear();
            ws.fa2_gated.clear();
            ws.rope_k_t.clear();
            ws.rope_ctr_t.clear();
            ws.rope_pos_t.clear();
            ws.rows_tab_t.clear();
            for engine in &self.ranks {
                let _m = engine.gpu.enter_main()?;
                ws.fa2_q.push(engine.uninit(32 * lq)?);
                ws.fa2_gate.push(engine.uninit(32 * lg)?);
                ws.fa2_gated.push(engine.uninit(32 * lq)?);
                ws.rope_k_t.push(engine.uninit(32 * ws.local_kv_dim)?);
                ws.rope_ctr_t.push(engine.stream().clone_htod(&[0u32; 32])?);
                ws.rope_pos_t.push(engine.htod_i32(&[0i32; 32])?);
                ws.rows_tab_t
                    .push(engine.stream().clone_htod(&[0u64; 32 * 6])?);
            }
            ws.rows_tabs = (0..ranks).map(|_| Default::default()).collect();
            ws.fa2_cap = 32;
        }
        for rank in 0..ranks {
            let engine = &self.ranks[rank];
            let _main = engine.gpu.enter_main()?;
            {
                let mut dst = ws.fa2_q[rank].slice_mut(col * lq..(col + 1) * lq);
                engine
                    .stream()
                    .memcpy_dtod(&ws.q[rank].slice(0..lq), &mut dst)?;
            }
            {
                let mut dst = ws.fa2_gate[rank].slice_mut(col * lg..(col + 1) * lg);
                engine
                    .stream()
                    .memcpy_dtod(&ws.gate[rank].slice(0..lg), &mut dst)?;
            }
            ws.ev_rank[rank].record(&engine.stream())?;
        }
        {
            let _main = e.gpu.enter_main()?;
            for ev in ws.ev_rank.iter() {
                e.stream().wait(ev)?;
            }
        }
        Ok(())
    }

    /// FULL T-ROW ATTENTION PASS over per-row session tables (batched serving): reads
    /// the tcol raw-projection slabs, runs ONE rope/append rows launch + ONE fa rows
    /// launch + ONE combine per rank (gate straight from the tcol gate slab), then the
    /// o_proj tcol join — the whole per-row attention loop in 3 launches/rank/layer.
    /// Per-(row, head) programs are the t=1 kernels verbatim; each row appends to and
    /// attends its OWN session. `session_parts[rank][row]` = {k_plane, v_plane, len_ptr,
    /// base_ptr}; `tab_keys[rank]` keys the per-rank combined-table cache (caller folds
    /// layer + session-set + base-arming into it); `stage_pos` stages the position slab
    /// (positions are constant across layers within a tick — stage on the first layer).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_v2_rope_fa_rows(
        &self,
        ws_index: usize,
        e: &Engine,
        o_m: &ResidentStepBf16RowParallel,
        session_parts: &[Vec<[u64; 4]>],
        tab_keys: &[u64],
        positions: &[i32],
        stage_pos: bool,
        same_session: bool,
        q_norms: &[CudaSlice<f32>],
        k_norms: &[CudaSlice<f32>],
        rope_freqs: &[Option<&crate::CudaSlice<f32>>],
        t: usize,
        head_dim: usize,
        n_rot: usize,
        window: usize,
        max_ns: usize,
        scale: f32,
        k_tok_bytes: usize,
        v_tok_bytes: usize,
        eps: f32,
        rope_base: f32,
    ) -> Result<crate::CudaSlice<f32>, Box<dyn std::error::Error>> {
        use cudarc::driver::DevicePtr;
        let ranks = self.ranks.len();
        if session_parts.len() != ranks || tab_keys.len() != ranks || positions.len() < t {
            return Err("rope fa rows geometry".into());
        }
        {
            let mut guard = self
                .decode_v2
                .lock()
                .map_err(|_| "step TP decode v2 workspace lock is poisoned")?;
            let ws = guard
                .get_mut(ws_index)
                .ok_or("step TP decode v2 workspace index out of range")?;
            if ws.tcol_cap < t || ws.tcol_q.len() != ranks {
                return Err("rope fa rows without tcol slabs".into());
            }
            let lq = ws.local_q_dim;
            let lkv = ws.local_kv_dim;
            let lg = (ws.heads / ranks).max(1);
            let local_heads = (ws.heads / ranks).max(1);
            let local_kv_heads = (lkv / head_dim).max(1);
            // Arm the fa2/rope slabs (shared with the stash path).
            if ws.fa2_cap < 32 || ws.fa2_q.len() != ranks || ws.rows_tab_t.len() != ranks {
                ws.fa2_q.clear();
                ws.fa2_gate.clear();
                ws.fa2_gated.clear();
                ws.rope_k_t.clear();
                ws.rope_ctr_t.clear();
                ws.rope_pos_t.clear();
                ws.rows_tab_t.clear();
                for engine in &self.ranks {
                    let _m = engine.gpu.enter_main()?;
                    ws.fa2_q.push(engine.uninit(32 * lq)?);
                    ws.fa2_gate.push(engine.uninit(32 * lg)?);
                    ws.fa2_gated.push(engine.uninit(32 * lq)?);
                    ws.rope_k_t.push(engine.uninit(32 * lkv)?);
                    ws.rope_ctr_t.push(engine.stream().clone_htod(&[0u32; 32])?);
                    ws.rope_pos_t.push(engine.htod_i32(&[0i32; 32])?);
                    ws.rows_tab_t
                        .push(engine.stream().clone_htod(&[0u64; 32 * 6])?);
                }
                ws.rows_tabs = (0..ranks).map(|_| Default::default()).collect();
                ws.fa2_cap = 32;
            }
            if ws.tcol_ocap < t || ws.tcol_gated.len() != ranks {
                ws.tcol_gated.clear();
                ws.tcol_opart.clear();
                for engine in &self.ranks {
                    let _m = engine.gpu.enter_main()?;
                    ws.tcol_gated.push(engine.uninit(32 * lq)?);
                    ws.tcol_opart.push(engine.uninit(32 * ws.o_out)?);
                }
                let root = &self.ranks[0];
                let _m = root.gpu.enter_main()?;
                ws.tcol_opeer = Some(root.uninit(32 * ws.o_out)?);
                ws.tcol_omix = Some(root.uninit(32 * ws.o_out)?);
                ws.tcol_ocap = 32;
            }
            for rank in 0..ranks {
                let engine = &self.ranks[rank];
                let _main = engine.gpu.enter_main()?;
                if stage_pos {
                    let host: Vec<i32> = positions[..t].to_vec();
                    let mut view = ws.rope_pos_t[rank].slice_mut(0..t);
                    engine.stream().memcpy_htod(&host, &mut view)?;
                }
                // Combined 6-word table {k, v, len, base, ctr, back}; ctr = this rank's
                // per-row counter slab. Built from the pointers the CALLER just read off
                // the live distributed cache, and RESTAGED into a persistent slab before
                // every launch (MEMRA_ROWS_TAB_RESTAGE, default ON).
                //
                // The `rows_tabs` memo this replaces was keyed by a hash of
                // (k pointer, base pointer, layer, t) but the table it handed back ALSO
                // carried the V and LEN pointers, and nothing invalidated it when a
                // session's KV cache was dropped. A later session whose K buffer landed on
                // a recycled address therefore hit a dead entry, and
                // `qk_norm_rope_append_inc_dcw_rows` WROTE this session's K/V rows through
                // the freed V/len pointers it still held while `fa_decode_dcw_rows` read
                // them back: a whole non-finite row when the freed pages were re-mapped,
                // CUDA_ERROR_ILLEGAL_ADDRESS when they were not. The row-table twin in
                // `step35_verify_fa_rows_join` was cured of exactly this in 8c8397e0b2
                // ("a process-lifetime map cannot prove allocation generation", Hermes
                // `11339f5cd3c132a3`); this fused rope+append+fa path was left out of it,
                // and MEMRA_FUSE_ROPE_APPEND=1 makes it the arm that actually runs.
                let ctr_base = {
                    let s = engine.stream();
                    let (p, _g) = ws.rope_ctr_t[rank].device_ptr(&s);
                    p
                };
                let host = rows_tab_host(&session_parts[rank], ctr_base, same_session, t);
                // STALE-HIT RECEIPT (MEMRA_ROWS_TAB_STALE_SCAN=1, default OFF): replay the
                // retired key against the contents we are about to stage. `engaged` proves
                // this path executes at all; `STALE` proves the retired memo would have
                // handed a live launch another allocation's pointers, and names which word
                // moved. Diagnostic only: it never feeds a kernel.
                if rows_tab_stale_scan() {
                    if ws.rows_tab_shadow.len() != ranks {
                        ws.rows_tab_shadow = (0..ranks).map(|_| Default::default()).collect();
                    }
                    let n = ROWS_TAB_ENGAGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if let Some(prev) = ws.rows_tab_shadow[rank].get(&tab_keys[rank])
                        && prev != &host
                    {
                        let words = ["k", "v", "len", "base", "ctr", "back"];
                        let moved: Vec<String> = (0..host.len())
                            .filter(|&i| prev.get(i) != Some(&host[i]))
                            .map(|i| format!("{}[row{}]", words[i % 6], i / 6))
                            .collect();
                        let stale =
                            ROWS_TAB_STALE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        eprintln!(
                            "[rows-tab] STALE #{stale} lookup #{n} rank={rank} t={t} key={:#018x} moved={}: the retired memo would have launched this row on another allocation's pointers",
                            tab_keys[rank],
                            moved.join(",")
                        );
                    }
                    ws.rows_tab_shadow[rank].insert(tab_keys[rank], host.clone());
                }
                let legacy_memo = !rows_tab_restage_on();
                if legacy_memo && !ws.rows_tabs[rank].contains_key(&tab_keys[rank]) {
                    let tab = engine.stream().clone_htod(&host)?;
                    ws.rows_tabs[rank].insert(tab_keys[rank], tab);
                }
                if !legacy_memo {
                    let mut view = ws.rows_tab_t[rank].slice_mut(0..t * 6);
                    engine.stream().memcpy_htod(&host, &mut view)?;
                }
                let StepTpDecodeV2Ws {
                    tcol_q,
                    tcol_k,
                    tcol_v,
                    tcol_g,
                    fa2_q,
                    fa2_gated,
                    rope_k_t,
                    rope_pos_t,
                    rows_tabs,
                    rows_tab_t,
                    ..
                } = &mut *ws;
                let tab = if legacy_memo {
                    rows_tabs[rank]
                        .get(&tab_keys[rank])
                        .ok_or("rows tab memo lost its entry")?
                } else {
                    &rows_tab_t[rank]
                };
                engine.qk_norm_rope_append_inc_dcw_rows(
                    &tcol_q[rank],
                    &tcol_k[rank],
                    &tcol_v[rank],
                    &q_norms[rank],
                    &k_norms[rank],
                    &mut fa2_q[rank],
                    &mut rope_k_t[rank],
                    tab,
                    &rope_pos_t[rank],
                    same_session,
                    t,
                    lkv,
                    lkv,
                    k_tok_bytes,
                    v_tok_bytes,
                    head_dim,
                    n_rot,
                    local_heads,
                    local_kv_heads,
                    eps,
                    rope_base,
                    1.0,
                    rope_freqs[rank],
                )?;
                engine.fa_decode_dcw_rows(
                    &fa2_q[rank],
                    tab,
                    &mut fa2_gated[rank],
                    t,
                    head_dim,
                    local_heads,
                    local_kv_heads,
                    window,
                    max_ns,
                    scale,
                    k_tok_bytes,
                    v_tok_bytes,
                    &tcol_g[rank],
                )?;
                let StepTpDecodeV2Ws {
                    fa2_gated,
                    tcol_gated,
                    ..
                } = &mut *ws;
                let mut dst = tcol_gated[rank].slice_mut(0..t * lq);
                engine
                    .stream()
                    .memcpy_dtod(&fa2_gated[rank].slice(0..t * lq), &mut dst)?;
            }
        }
        self.decode_v2_oproj_tcol(ws_index, e, o_m, t)
    }

    /// T-ROW fa join over per-row session tables (the per-session distributed-KV
    /// primitive): after all t rows stashed q+gate (their appends landed in rank-stream
    /// order), ONE fa_decode_dcw_rows per rank walks every row's own ring with its own
    /// geometry — bit-identical per row to its per-row launch — then the o_proj tcol
    /// join lands the [t, o_out] `mixed` slab on `e`. `tabs[rank]` is the pre-staged
    /// device table on that rank.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_v2_fa_rows_join(
        &self,
        ws_index: usize,
        e: &Engine,
        o_m: &ResidentStepBf16RowParallel,
        tabs: &[&crate::CudaSlice<u64>],
        t: usize,
        head_dim: usize,
        window: usize,
        max_ns: usize,
        scale: f32,
        k_tok_bytes: usize,
        v_tok_bytes: usize,
    ) -> Result<crate::CudaSlice<f32>, Box<dyn std::error::Error>> {
        let ranks = self.ranks.len();
        if tabs.len() != ranks {
            return Err("fa rows join needs one table per rank".into());
        }
        {
            let mut guard = self
                .decode_v2
                .lock()
                .map_err(|_| "step TP decode v2 workspace lock is poisoned")?;
            let ws = guard
                .get_mut(ws_index)
                .ok_or("step TP decode v2 workspace index out of range")?;
            if ws.fa2_cap < t || ws.fa2_q.len() != ranks {
                return Err("fa rows join without stashed rows".into());
            }
            let lq = ws.local_q_dim;
            let local_heads = (ws.heads / ranks).max(1);
            let local_kv_heads = (ws.local_kv_dim / head_dim).max(1);
            if ws.tcol_ocap < t || ws.tcol_gated.len() != ranks {
                ws.tcol_gated.clear();
                ws.tcol_opart.clear();
                for engine in &self.ranks {
                    let _m = engine.gpu.enter_main()?;
                    ws.tcol_gated.push(engine.uninit(32 * lq)?);
                    ws.tcol_opart.push(engine.uninit(32 * ws.o_out)?);
                }
                let root = &self.ranks[0];
                let _m = root.gpu.enter_main()?;
                ws.tcol_opeer = Some(root.uninit(32 * ws.o_out)?);
                ws.tcol_omix = Some(root.uninit(32 * ws.o_out)?);
                ws.tcol_ocap = 32;
            }
            for rank in 0..ranks {
                let engine = &self.ranks[rank];
                let _main = engine.gpu.enter_main()?;
                {
                    let StepTpDecodeV2Ws {
                        fa2_q,
                        fa2_gate,
                        fa2_gated,
                        ..
                    } = &mut *ws;
                    engine.fa_decode_dcw_rows(
                        &fa2_q[rank],
                        tabs[rank],
                        &mut fa2_gated[rank],
                        t,
                        head_dim,
                        local_heads,
                        local_kv_heads,
                        window,
                        max_ns,
                        scale,
                        k_tok_bytes,
                        v_tok_bytes,
                        &fa2_gate[rank],
                    )?;
                }
                let StepTpDecodeV2Ws {
                    fa2_gated,
                    tcol_gated,
                    ..
                } = &mut *ws;
                let mut dst = tcol_gated[rank].slice_mut(0..t * lq);
                engine
                    .stream()
                    .memcpy_dtod(&fa2_gated[rank].slice(0..t * lq), &mut dst)?;
            }
        }
        self.decode_v2_oproj_tcol(ws_index, e, o_m, t)
    }

    /// MEMRA_TCOL_OPROJ stash: copy this column's per-rank `gated` rows into the o-tcol
    /// slabs (rank-stream ordered behind the attention kernels that produced them). The
    /// per-column finish choreography is skipped entirely; `decode_v2_oproj_tcol` joins
    /// every column afterwards.
    pub(crate) fn decode_v2_stash_gated(
        &self,
        ws: &mut StepTpDecodeV2Ws,
        e: &Engine,
        col: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ranks = self.ranks.len();
        // 32, not 8: the slabs below have been 32 rows since the slab-width fix, and the walk now
        // runs chunks up to t=32 (the w=16 arm died here on a guard three widths staler than its
        // own allocation, 2026-08-27).
        if col >= 32 {
            return Err("decode_v2_stash_gated column out of range".into());
        }
        let lq = ws.local_q_dim;
        if ws.tcol_ocap == 0 || ws.tcol_gated.len() != ranks {
            ws.tcol_gated.clear();
            ws.tcol_opart.clear();
            for engine in &self.ranks {
                let _m = engine.gpu.enter_main()?;
                ws.tcol_gated.push(engine.uninit(32 * lq)?);
                ws.tcol_opart.push(engine.uninit(32 * ws.o_out)?);
            }
            let root = &self.ranks[0];
            let _m = root.gpu.enter_main()?;
            ws.tcol_opeer = Some(root.uninit(32 * ws.o_out)?);
            ws.tcol_omix = Some(root.uninit(32 * ws.o_out)?);
            ws.tcol_ocap = 32;
        }
        for rank in 0..ranks {
            let engine = &self.ranks[rank];
            let _main = engine.gpu.enter_main()?;
            let mut dst = ws.tcol_gated[rank].slice_mut(col * lq..(col + 1) * lq);
            engine
                .stream()
                .memcpy_dtod(&ws.gated[rank].slice(0..lq), &mut dst)?;
            // The skipped finish's e-wait was ALSO the anti-dependency guard: it ordered
            // e's NEXT column's h/pos re-staging behind this column's rank-side raw pulls.
            // Record each rank here and make e wait — same protection, no o_proj work.
            ws.ev_rank[rank].record(&engine.stream())?;
        }
        {
            let _main = e.gpu.enter_main()?;
            for ev in ws.ev_rank.iter() {
                e.stream().wait(ev)?;
            }
        }
        Ok(())
    }

    /// MEMRA_TCOL_OPROJ join: one weight-amortized b4_tcol per rank over the stashed
    /// `gated` slabs (per-column FP order == the t=1 b4 kernel), one peer pull of rank1's
    /// partial slab, one elementwise slab add on the root (independent elements — each
    /// column's add is the exact direct-join `add(p0, p1)`), then the joined `mixed` slab
    /// lands on `e`. Returns [t, o_out] on the model engine.
    pub(crate) fn decode_v2_oproj_tcol(
        &self,
        ws_index: usize,
        e: &Engine,
        o_m: &ResidentStepBf16RowParallel,
        t: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let ranks = self.ranks.len();
        let mut guard = self
            .decode_v2
            .lock()
            .map_err(|_| "step TP decode v2 workspace lock is poisoned")?;
        let ws = guard
            .get_mut(ws_index)
            .ok_or("step TP decode v2 workspace index out of range")?;
        if ranks != 2 || ws.blocks_per_rank != 4 || t == 0 || t > 32 || ws.tcol_ocap < t {
            return Err("decode_v2_oproj_tcol geometry".into());
        }
        for rank in 0..ranks {
            let engine = &self.ranks[rank];
            let _main = engine.gpu.enter_main()?;
            let mut weights = Vec::with_capacity(4);
            for block in 0..4 {
                let ResidentBf16Weight::Bf16(weight) = &o_m.ranks[rank][block].weight else {
                    return Err("tcol o_proj requires bf16-resident O blocks".into());
                };
                weights.push(weight);
            }
            {
                let StepTpDecodeV2Ws {
                    tcol_gated,
                    tcol_opart,
                    local_q_dim,
                    o_block_cols,
                    o_out,
                    w8t_oaq,
                    w8t_oad,
                    w8t_oin,
                    w8t_cap,
                    ..
                } = &mut *ws;
                // MEMRA_TCOL_OPROJ_REF=1 (bisect): fill the partial slab via the t=1 b4
                // kernel per column — separates choreography bugs from tcol-kernel bugs.
                static REFK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let refk = *REFK
                    .get_or_init(|| std::env::var("MEMRA_TCOL_OPROJ_REF").as_deref() == Ok("1"));
                if refk {
                    let lq = *local_q_dim;
                    let mut xr = engine.uninit(lq)?;
                    let mut yr = engine.uninit(*o_out)?;
                    for c in 0..t {
                        {
                            let mut dst = xr.slice_mut(0..lq);
                            engine.stream().memcpy_dtod(
                                &tcol_gated[rank].slice(c * lq..(c + 1) * lq),
                                &mut dst,
                            )?;
                        }
                        engine.matvec_bf16_b4_into(
                            [weights[0], weights[1], weights[2], weights[3]],
                            &xr,
                            &mut yr,
                            *o_block_cols,
                            *o_out,
                        )?;
                        let mut dst = tcol_opart[rank].slice_mut(c * *o_out..(c + 1) * *o_out);
                        engine
                            .stream()
                            .memcpy_dtod(&yr.slice(0..*o_out), &mut dst)?;
                    }
                } else if crate::step_tp_w8_on()
                    && (0..4).all(|b| o_m.ranks[rank][b].q8.is_some())
                    && (4 * *o_block_cols) % 32 == 0
                {
                    // The verify walk's biggest single kernel: bf16 tcol o_proj was 24.8% of
                    // spec GPU time. Same planar q8_0 mirrors the decode arm uses, one launch
                    // over all t columns.
                    let in_f = 4 * *o_block_cols;
                    if *w8t_oin != in_f || *w8t_cap < t || w8t_oaq.len() != ranks {
                        w8t_oaq.clear();
                        w8t_oad.clear();
                        for e_rank in &self.ranks {
                            let _m = e_rank.gpu.enter_main()?;
                            w8t_oaq.push(e_rank.alloc_i8_uninit(32 * in_f)?);
                            w8t_oad.push(e_rank.alloc_uninit::<f32>(32 * (in_f / 32))?);
                        }
                        *w8t_oin = in_f;
                        *w8t_cap = (*w8t_cap).max(32);
                    }
                    engine.quantize_q8_1_into(
                        &tcol_gated[rank],
                        t,
                        in_f,
                        &mut w8t_oaq[rank],
                        &mut w8t_oad[rank],
                    )?;
                    engine.qmatvec_q8_0_b4_rp_t_into(
                        [
                            o_m.ranks[rank][0].q8.as_ref().unwrap(),
                            o_m.ranks[rank][1].q8.as_ref().unwrap(),
                            o_m.ranks[rank][2].q8.as_ref().unwrap(),
                            o_m.ranks[rank][3].q8.as_ref().unwrap(),
                        ],
                        &w8t_oaq[rank],
                        &w8t_oad[rank],
                        &mut tcol_opart[rank],
                        *o_block_cols,
                        *o_out,
                        t,
                    )?;
                } else {
                    engine.matvec_bf16_b4_tcol_into(
                        [weights[0], weights[1], weights[2], weights[3]],
                        &tcol_gated[rank],
                        &mut tcol_opart[rank],
                        *o_block_cols,
                        *o_out,
                        t,
                    )?;
                }
            }
            if rank != 0 {
                ws.ev_rank[rank].record(&engine.stream())?;
            }
        }
        let root = &self.ranks[0];
        {
            let _main = root.gpu.enter_main()?;
            for ev in ws.ev_rank.iter().skip(1) {
                root.stream().wait(ev)?;
            }
            {
                let StepTpDecodeV2Ws {
                    tcol_opart,
                    tcol_opeer,
                    tcol_omix,
                    o_out,
                    ..
                } = &mut *ws;
                let opeer = tcol_opeer.as_mut().ok_or("tcol o_proj slabs not armed")?;
                let omix = tcol_omix.as_mut().ok_or("tcol o_proj slabs not armed")?;
                {
                    let mut dst = opeer.slice_mut(0..t * *o_out);
                    root.stream()
                        .memcpy_dtod(&tcol_opart[1].slice(0..t * *o_out), &mut dst)?;
                }
                // Elementwise over the whole slab: per element identical to the per-column
                // direct-join add (independent lanes, same operand values).
                root.add(&tcol_opart[0], opeer, omix, t * *o_out)?;
            }
            ws.ev_oproj.record(&root.stream())?;
        }
        let _main = e.gpu.enter_main()?;
        e.stream().wait(&ws.ev_oproj)?;
        let mut out = e.uninit(t * ws.o_out)?;
        let omix = ws.tcol_omix.as_ref().ok_or("tcol o_proj slabs not armed")?;
        e.stream().memcpy_dtod(
            &omix.slice(0..t * ws.o_out),
            &mut out.slice_mut(0..t * ws.o_out),
        )?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub(crate) fn decode_v2_input_qkv(
        &self,
        ws: &mut StepTpDecodeV2Ws,
        e: &Engine,
        h: &CudaSlice<f32>,
        pos_d: &CudaSlice<i32>,
        gate_raw: Option<&CudaSlice<f32>>,
        gate_shards: Option<StepTpGateShards<'_>>,
        decode_input: &mut ResidentReplicatedDeviceRows,
        q_m: &ResidentBf16ColumnParallel,
        k_m: &ResidentBf16ColumnParallel,
        v_m: &ResidentBf16ColumnParallel,
        q_norm: &[CudaSlice<f32>],
        k_norm: &[CudaSlice<f32>],
        head_dim: usize,
        n_rot: usize,
        rope_base: f32,
        rope_freqs: &[Option<&CudaSlice<f32>>],
        rms_eps: f32,
        has_gate: bool,
        defer_norm_rope: bool,
        tcol_col: Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ranks = self.ranks.len();
        validate_replicated_device_rows(&self.ranks, decode_input)?;
        let gate_sources = usize::from(gate_raw.is_some()) + usize::from(gate_shards.is_some());
        if decode_input.tokens != 1
            || decode_input.width != q_m.in_features
            || pos_d.len() != 1
            || gate_raw.is_some_and(|gate| gate.len() != ws.heads)
            || (has_gate && gate_sources != 1)
            || (!has_gate && gate_sources != 0)
            || gate_shards.as_ref().is_some_and(|shards| match shards {
                StepTpGateShards::F32(shards) => shards.len() != ranks,
                StepTpGateShards::Bf16(shards) => shards.len() != ranks,
            })
            || q_norm.len() != ranks
            || k_norm.len() != ranks
            || rope_freqs.len() != ranks
            || e.ctx().ordinal() != ws.e_device
        {
            return Err("step TP decode v2 input geometry mismatch".into());
        }

        let qkv_fused = step_tp_qkv_fused_enabled()?;
        if gate_shards.is_some() && !qkv_fused {
            return Err("step TP decode v2 gate shards require MEMRA_STEP_TP_QKV_FUSED=1".into());
        }
        let values = decode_input.width;
        if h.len() != values {
            return Err(format!(
                "step TP decode v2 hidden width {} != replicated width {values}",
                h.len()
            )
            .into());
        }

        if qkv_fused {
            // STAGE-BASED flow (graph increment A): h and pos land in fixed e-context stages
            // (one e-stream copy each), the entry event covers them, and every rank raw-copies
            // from the stages on its own stream — exactly the shape graph capture wraps.
            if ws.h_stage.is_none() {
                use cudarc::driver::DevicePtr;
                let _main = e.gpu.enter_main()?;
                let h_stage = e.uninit(values)?;
                let pos_stage = e.htod_i32(&[0])?;
                {
                    let stream = e.stream();
                    let (hp, _g0) = h_stage.device_ptr(&stream);
                    let (pp, _g1) = pos_stage.device_ptr(&stream);
                    ws.raw_h_stage = hp;
                    ws.raw_pos_stage = pp;
                }
                ws.h_stage = Some(h_stage);
                ws.pos_stage = Some(pos_stage);
                for rank in 0..ranks {
                    use cudarc::driver::DevicePtr;
                    let engine = &self.ranks[rank];
                    let _rmain = engine.gpu.enter_main()?;
                    let attn_in = engine.uninit(values)?;
                    let (dp, pp) = {
                        let stream = engine.stream();
                        let (dp, _g2) = attn_in.device_ptr(&stream);
                        let (pp, _g3) = ws.pos[rank].device_ptr(&stream);
                        (dp, pp)
                    };
                    ws.raw_attn_in.push(dp);
                    ws.raw_pos.push(pp);
                    ws.attn_in.push(attn_in);
                }
                {
                    use cudarc::driver::DevicePtr;
                    let root = &self.ranks[0];
                    let _rmain = root.gpu.enter_main()?;
                    let stream = root.stream();
                    let (a, _g) = ws.peer_partial.device_ptr(&stream);
                    let (b, _g) = ws.k_shadow.device_ptr(&stream);
                    let (c, _g) = ws.v_shadow.device_ptr(&stream);
                    ws.raw_peer_partial = a;
                    ws.raw_k_shadow = b;
                    ws.raw_v_shadow = c;
                }
                {
                    use cudarc::driver::DevicePtr;
                    let rank1 = &self.ranks[1];
                    let _rmain = rank1.gpu.enter_main()?;
                    let stream = rank1.stream();
                    let (a, _g) = ws.o_partials[1][0].device_ptr(&stream);
                    let (b, _g) = ws.k[1].device_ptr(&stream);
                    let (c, _g) = ws.v_raw[1].device_ptr(&stream);
                    ws.raw_o_partial1 = a;
                    ws.raw_k1 = b;
                    ws.raw_v1 = c;
                }
            }
            {
                let _main = e.gpu.enter_main()?;
                {
                    // (Always staged: a tcol column below the dcw floor falls back to the
                    // normal fused arm, which reads h through this stage.)
                    let h_stage = ws.h_stage.as_mut().expect("stage armed above");
                    let mut dst = h_stage.slice_mut(0..values);
                    e.stream().memcpy_dtod(&h.slice(0..values), &mut dst)?;
                }
                {
                    let pos_stage = ws.pos_stage.as_mut().expect("stage armed above");
                    let mut dst = pos_stage.slice_mut(0..1);
                    e.stream().memcpy_dtod(&pos_d.slice(0..1), &mut dst)?;
                }
                ws.ev_entry.record(&e.stream())?;
            }
            for rank in 0..ranks {
                let engine = &self.ranks[rank];
                let _main = engine.gpu.enter_main()?;
                engine.stream().wait(&ws.ev_entry)?;
            }
        } else {
            // Evented replicate flow (the pre-stage shape, kept for the non-fused class).
            {
                let _main = e.gpu.enter_main()?;
                if let Some(gate_raw) = gate_raw {
                    let mut gate_dst = ws.gate_e.slice_mut(0..ws.heads);
                    e.stream()
                        .memcpy_dtod(&gate_raw.slice(0..ws.heads), &mut gate_dst)?;
                }
                ws.ev_entry.record(&e.stream())?;
            }
            {
                let root = &self.ranks[0];
                let _main = root.gpu.enter_main()?;
                root.stream().wait(&ws.ev_entry)?;
                let mut destination = decode_input.ranks[0].slice_mut(0..values);
                root.stream()
                    .memcpy_dtod(&h.slice(0..values), &mut destination)?;
                ws.ev_refresh.record(&root.stream())?;
            }
            for rank in 1..ranks {
                let engine = &self.ranks[rank];
                let _main = engine.gpu.enter_main()?;
                engine.stream().wait(&ws.ev_refresh)?;
                let (root_rows, peer_rows) = decode_input.ranks.split_at_mut(rank);
                let mut destination = peer_rows[0].slice_mut(0..values);
                engine
                    .stream()
                    .memcpy_dtod(&root_rows[0].slice(0..values), &mut destination)?;
            }
        }
        for rank in 0..ranks {
            self.decode_v2_input_qkv_rank(
                ws,
                pos_d,
                decode_input,
                q_m,
                k_m,
                v_m,
                q_norm,
                k_norm,
                head_dim,
                n_rot,
                rope_base,
                rope_freqs,
                rms_eps,
                gate_shards.as_ref(),
                has_gate,
                qkv_fused,
                defer_norm_rope,
                rank,
                tcol_col,
            )?;
        }
        Ok(())
    }

    /// One rank's slice of `decode_v2_input_qkv` (projection, norm+rope, gate staging) — the
    /// per-device issue unit the whole-token graph captures on that rank's stream.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_v2_input_qkv_rank(
        &self,
        ws: &mut StepTpDecodeV2Ws,
        pos_d: &CudaSlice<i32>,
        decode_input: &mut ResidentReplicatedDeviceRows,
        q_m: &ResidentBf16ColumnParallel,
        k_m: &ResidentBf16ColumnParallel,
        v_m: &ResidentBf16ColumnParallel,
        q_norm: &[CudaSlice<f32>],
        k_norm: &[CudaSlice<f32>],
        head_dim: usize,
        n_rot: usize,
        rope_base: f32,
        rope_freqs: &[Option<&CudaSlice<f32>>],
        rms_eps: f32,
        gate_shards: Option<&StepTpGateShards<'_>>,
        has_gate: bool,
        qkv_fused: bool,
        defer_norm_rope: bool,
        rank: usize,
        tcol_col: Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ranks = self.ranks.len();
        let local_heads = ws.local_q_dim / head_dim;
        let local_kv_heads = ws.local_kv_dim / head_dim;
        let engine = &self.ranks[rank];
        let _main = engine.gpu.enter_main()?;
        let ws_e_device = ws.e_device;
        // T-COLUMN SELECT (spec verify): the projections for this column were precomputed
        // by the weight-amortized tcol kernel — copy the column into the single-row buffers
        // (pure f32 moves, bit-exact) and skip the per-column matvec. Rope/norm/append run
        // below exactly as in the t=1 program.
        if qkv_fused && tcol_col.is_some() {
            #[allow(clippy::unnecessary_unwrap)]
            // allow: the Some-guard sits in a multi-clause regime gate; if-let would reshape the arm structure
            let c = tcol_col.expect("checked");
            if ws.tcol_cap == 0 || ws.tcol_q.len() != ranks {
                return Err("tcol select without precompute".into());
            }
            // The select skips the matvec but NOT the position: rope/append below still
            // read this rank's pos buffer, which only the (skipped) stage path fills for
            // peer-device ranks. Stage it here or rank1 ropes at the previous position.
            if engine.ctx().ordinal() != ws_e_device {
                raw_copy_bytes(ws.raw_pos[rank], ws.raw_pos_stage, 4, engine)?;
            }
            let StepTpDecodeV2Ws {
                tcol_q,
                tcol_k,
                tcol_v,
                tcol_g,
                q_raw,
                k_raw,
                v_raw,
                gate,
                local_q_dim,
                local_kv_dim,
                heads,
                ..
            } = &mut *ws;
            let lg = *heads / ranks;
            let stream = engine.stream();
            {
                let mut dst = q_raw[rank].slice_mut(0..*local_q_dim);
                stream.memcpy_dtod(
                    &tcol_q[rank].slice(c * *local_q_dim..(c + 1) * *local_q_dim),
                    &mut dst,
                )?;
            }
            {
                let mut dst = k_raw[rank].slice_mut(0..*local_kv_dim);
                stream.memcpy_dtod(
                    &tcol_k[rank].slice(c * *local_kv_dim..(c + 1) * *local_kv_dim),
                    &mut dst,
                )?;
            }
            {
                let mut dst = v_raw[rank].slice_mut(0..*local_kv_dim);
                stream.memcpy_dtod(
                    &tcol_v[rank].slice(c * *local_kv_dim..(c + 1) * *local_kv_dim),
                    &mut dst,
                )?;
            }
            if has_gate && lg > 0 {
                let mut dst = gate[rank].slice_mut(0..lg);
                stream.memcpy_dtod(&tcol_g[rank].slice(c * lg..(c + 1) * lg), &mut dst)?;
            }
            if !defer_norm_rope {
                // Below the dcw floor (or a non-defer shape) the col-select cannot apply:
                // fall through and recompute this column's QKV from the REAL h row — the
                // caller always passes it. The slab copies above are dead stores.
            } else {
                return Ok(());
            }
        }
        if qkv_fused {
            // Stage-based input: raw copies from the fixed e-context stages (capture-safe;
            // eager ordering comes from the caller's ev_entry wait on this stream). The rank
            // SHARING e's device reads the stages directly — same context (probed), ordering
            // identical (ev_entry / graph edge), bytes identical: the copies are pure waste.
            let same_dev = engine.ctx().ordinal() == ws.e_device;
            if !same_dev {
                raw_copy_bytes(
                    ws.raw_attn_in[rank],
                    ws.raw_h_stage,
                    q_m.in_features * 4,
                    engine,
                )?;
                raw_copy_bytes(ws.raw_pos[rank], ws.raw_pos_stage, 4, engine)?;
            }
            let StepTpDecodeV2Ws {
                q_raw,
                k_raw,
                v_raw,
                gate,
                gate_e,
                attn_in,
                h_stage,
                heads,
                local_q_dim,
                local_kv_dim,
                w8_aq,
                w8_ad,
                w8_in,
                ..
            } = &mut *ws;
            let input_ref: &CudaSlice<f32> = if same_dev {
                h_stage
                    .as_ref()
                    .ok_or("step TP decode v2 stage not armed")?
            } else {
                &attn_in[rank]
            };
            match (
                &q_m.ranks[rank].weight,
                &k_m.ranks[rank].weight,
                &v_m.ranks[rank].weight,
            ) {
                (
                    ResidentBf16Weight::F32(wq),
                    ResidentBf16Weight::F32(wk),
                    ResidentBf16Weight::F32(wv),
                ) => {
                    let (wg, out_g) = match &gate_shards {
                        Some(StepTpGateShards::F32(shards)) => (&shards[rank], *heads / ranks),
                        Some(StepTpGateShards::Bf16(_)) => {
                            return Err("step TP decode v2 gate shard class does not \
                                            match the F32 projections"
                                .into());
                        }
                        // out_g = 0: the kernel never reads wg; any resident buffer works.
                        None => (&*gate_e, 0),
                    };
                    engine.matvec_f32_qkv_into(
                        wq,
                        wk,
                        wv,
                        wg,
                        input_ref,
                        &mut q_raw[rank],
                        &mut k_raw[rank],
                        &mut v_raw[rank],
                        &mut gate[rank],
                        q_m.in_features,
                        *local_q_dim,
                        *local_kv_dim,
                        out_g,
                    )?;
                }
                (
                    ResidentBf16Weight::Bf16(wq),
                    ResidentBf16Weight::Bf16(wk),
                    ResidentBf16Weight::Bf16(wv),
                ) => {
                    let (wg, out_g) = match &gate_shards {
                        Some(StepTpGateShards::Bf16(shards)) => (&shards[rank], *heads / ranks),
                        Some(StepTpGateShards::F32(_)) => {
                            return Err("step TP decode v2 gate shard class does not \
                                            match the bf16 projections"
                                .into());
                        }
                        None => (wq, 0),
                    };
                    // MEMRA_STEP_TP_W8: q8_0 weights + q8_1 activation through mmvq instead of
                    // the fused bf16 qkvg. NUMERIC CLASS (int8 dp4a with per-32 scales, not a
                    // bf16 fma chain) — argmax-gated, never a bit-tape flip. Q, K and V each
                    // get their own launch because the fused kernel has no q8 twin; the gate
                    // rows stay bf16 (32 rows, ~0.3 MB, nothing to win and one less class to
                    // qualify). Measured motive: 23.0 us bf16 -> 14.0 us q8 at this shape.
                    let in_f = q_m.in_features;
                    let q8_ready = crate::step_tp_w8_on()
                        && q_m.ranks[rank].q8.is_some()
                        && k_m.ranks[rank].q8.is_some()
                        && v_m.ranks[rank].q8.is_some();
                    if q8_ready {
                        if *w8_in != in_f || w8_aq.len() != ranks {
                            w8_aq.clear();
                            w8_ad.clear();
                            for e_rank in &self.ranks {
                                let _m = e_rank.gpu.enter_main()?;
                                w8_aq.push(e_rank.alloc_uninit::<i8>(in_f)?);
                                w8_ad.push(e_rank.alloc_uninit::<f32>(in_f / 32)?);
                            }
                            *w8_in = in_f;
                        }
                        engine.quantize_q8_1_into(
                            input_ref,
                            1,
                            in_f,
                            &mut w8_aq[rank],
                            &mut w8_ad[rank],
                        )?;
                        // ONE launch over the stacked q/k/v rows. The three-call version
                        // measured 79.52 vs 80.72 tok/s — SLOWER than the bf16 fused kernel —
                        // because three launches plus the activation quantize cost more than
                        // the halved weight bytes save. Bit-identical to those three calls.
                        engine.qmatvec_q8_0_qkv_rp_into(
                            q_m.ranks[rank].q8.as_ref().unwrap(),
                            k_m.ranks[rank].q8.as_ref().unwrap(),
                            v_m.ranks[rank].q8.as_ref().unwrap(),
                            &w8_aq[rank],
                            &w8_ad[rank],
                            &mut q_raw[rank],
                            &mut k_raw[rank],
                            &mut v_raw[rank],
                            in_f,
                            *local_q_dim,
                            *local_kv_dim,
                        )?;
                        if out_g > 0 {
                            engine.matvec_bf16_into(wg, input_ref, &mut gate[rank], in_f, out_g)?;
                        }
                    } else {
                        engine.matvec_bf16_qkvg_into(
                            wq,
                            wk,
                            wv,
                            wg,
                            input_ref,
                            &mut q_raw[rank],
                            &mut k_raw[rank],
                            &mut v_raw[rank],
                            &mut gate[rank],
                            q_m.in_features,
                            *local_q_dim,
                            *local_kv_dim,
                            out_g,
                        )?;
                    }
                }
                _ => {
                    return Err("step TP decode v2 QKV projections mix residency classes".into());
                }
            }
        } else {
            for (matrix, local_out, raw) in [
                (q_m, ws.local_q_dim, &mut ws.q_raw),
                (k_m, ws.local_kv_dim, &mut ws.k_raw),
                (v_m, ws.local_kv_dim, &mut ws.v_raw),
            ] {
                let ResidentBf16Weight::F32(values_w) = &matrix.ranks[rank].weight else {
                    return Err("step TP decode v2 lost its F32 projection residency".into());
                };
                let chunk_rows = matrix.canonical_chunk_rows.unwrap_or(local_out);
                engine.linear_f32_resident_canonical_rows_t1_into(
                    &decode_input.ranks[rank],
                    values_w,
                    &mut raw[rank],
                    matrix.in_features,
                    local_out,
                    chunk_rows,
                )?;
            }
        }
        if qkv_fused && defer_norm_rope {
            // FUSION #1 defers norm+rope to the caller's fused rope+append+inc launch.
        } else if qkv_fused {
            // Fused norm+rope: one launch; the position comes from the rank-local staged
            // copy (raw-copied above from the fixed e-context pos stage — capture-safe).
            let StepTpDecodeV2Ws {
                q_raw,
                k_raw,
                q,
                k,
                pos,
                pos_stage,
                ..
            } = &mut *ws;
            let same_dev = engine.ctx().ordinal() == ws_e_device;
            let pos_ref: &CudaSlice<i32> = if same_dev {
                pos_stage
                    .as_ref()
                    .ok_or("step TP decode v2 pos stage not armed")?
            } else {
                &pos[rank]
            };
            engine.qk_norm_rope_into(
                &q_raw[rank],
                &k_raw[rank],
                &q_norm[rank],
                &k_norm[rank],
                &mut q[rank],
                &mut k[rank],
                pos_ref,
                head_dim,
                n_rot,
                local_heads,
                local_kv_heads,
                rms_eps,
                rope_base,
                1.0,
                rope_freqs[rank],
            )?;
        } else {
            engine.rms_norm(
                &ws.q_raw[rank],
                &q_norm[rank],
                &mut ws.q[rank],
                head_dim,
                local_heads,
                rms_eps,
            )?;
            engine.rms_norm(
                &ws.k_raw[rank],
                &k_norm[rank],
                &mut ws.k[rank],
                head_dim,
                local_kv_heads,
                rms_eps,
            )?;
            {
                let mut pos_dst = ws.pos[rank].slice_mut(0..1);
                engine
                    .stream()
                    .memcpy_dtod(&pos_d.slice(0..1), &mut pos_dst)?;
            }
            engine.rope_neox2(
                &mut ws.q[rank],
                &mut ws.k[rank],
                &ws.pos[rank],
                head_dim,
                n_rot,
                local_heads,
                local_kv_heads,
                1,
                rope_base,
                1.0,
                rope_freqs[rank],
            )?;
        }
        if has_gate && gate_shards.is_none() {
            let gate_start = rank * (ws.heads / ranks);
            let mut gate_dst = ws.gate[rank].slice_mut(0..ws.heads / ranks);
            engine.stream().memcpy_dtod(
                &ws.gate_e.slice(gate_start..gate_start + ws.heads / ranks),
                &mut gate_dst,
            )?;
        }
        Ok(())
    }

    /// One rank's O-partial slice of `decode_v2_finish` — the per-device issue unit the
    /// whole-token graph captures on that rank's stream (the rank-done event stays with the
    /// eager caller; graphs order via parent edges instead).
    pub(crate) fn decode_v2_finish_rank_partial(
        &self,
        ws: &mut StepTpDecodeV2Ws,
        o_m: &ResidentStepBf16RowParallel,
        o_fused: bool,
        rank: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let engine = &self.ranks[rank];
        let _main = engine.gpu.enter_main()?;
        if o_fused {
            let StepTpDecodeV2Ws {
                gated,
                o_partials,
                o_block_cols,
                o_out,
                w8o_aq,
                w8o_ad,
                w8o_in,
                ..
            } = &mut *ws;
            let all_f32 = o_m.ranks[rank]
                .iter()
                .all(|block| matches!(block.weight, ResidentBf16Weight::F32(_)));
            if all_f32 {
                let mut weights = Vec::with_capacity(4);
                for block in 0..4 {
                    let ResidentBf16Weight::F32(weight) = &o_m.ranks[rank][block].weight else {
                        unreachable!("all_f32 checked above");
                    };
                    weights.push(weight);
                }
                engine.matvec_f32_b4_into(
                    [weights[0], weights[1], weights[2], weights[3]],
                    &gated[rank],
                    &mut o_partials[rank][0],
                    *o_block_cols,
                    *o_out,
                )?;
            } else if crate::step_tp_w8_on() && (0..4).all(|b| o_m.ranks[rank][b].q8.is_some()) {
                // MEMRA_STEP_TP_W8, o_proj half: quantize the gated attention output once and
                // run all four HEAD_SPLIT blocks in one q8 launch. Measured motive: bf16 b4 is
                // 24.2 us/layer against 11.7 for the q8 shape — the largest decode line left
                // after the QKV arm banked +2.9%.
                let in_f = 4 * *o_block_cols;
                if *w8o_in != in_f || w8o_aq.len() != self.ranks.len() {
                    w8o_aq.clear();
                    w8o_ad.clear();
                    for e_rank in &self.ranks {
                        let _m = e_rank.gpu.enter_main()?;
                        w8o_aq.push(e_rank.alloc_uninit::<i8>(in_f)?);
                        w8o_ad.push(e_rank.alloc_uninit::<f32>(in_f / 32)?);
                    }
                    *w8o_in = in_f;
                }
                engine.quantize_q8_1_into(
                    &gated[rank],
                    1,
                    in_f,
                    &mut w8o_aq[rank],
                    &mut w8o_ad[rank],
                )?;
                engine.qmatvec_q8_0_b4_rp_into(
                    [
                        o_m.ranks[rank][0].q8.as_ref().unwrap(),
                        o_m.ranks[rank][1].q8.as_ref().unwrap(),
                        o_m.ranks[rank][2].q8.as_ref().unwrap(),
                        o_m.ranks[rank][3].q8.as_ref().unwrap(),
                    ],
                    &w8o_aq[rank],
                    &w8o_ad[rank],
                    &mut o_partials[rank][0],
                    *o_block_cols,
                    *o_out,
                )?;
            } else {
                let mut weights = Vec::with_capacity(4);
                for block in 0..4 {
                    let ResidentBf16Weight::Bf16(weight) = &o_m.ranks[rank][block].weight else {
                        return Err("step TP decode v2 O projections mix residency classes".into());
                    };
                    weights.push(weight);
                }
                engine.matvec_bf16_b4_into(
                    [weights[0], weights[1], weights[2], weights[3]],
                    &gated[rank],
                    &mut o_partials[rank][0],
                    *o_block_cols,
                    *o_out,
                )?;
            }
        } else {
            for block in 0..ws.blocks_per_rank {
                let x =
                    ws.gated[rank].slice(block * ws.o_block_cols..(block + 1) * ws.o_block_cols);
                let mut y = ws.o_partials[rank][block].slice_mut(0..ws.o_out);
                match &o_m.ranks[rank][block].weight {
                    ResidentBf16Weight::F32(weight) => {
                        let w = weight.slice(0..weight.len());
                        engine.linear_t1_into(&x, &w, &mut y, ws.o_block_cols, ws.o_out)?;
                    }
                    ResidentBf16Weight::Bf16(weight) => {
                        engine.matvec_bf16_views_into(
                            weight,
                            &x,
                            &mut y,
                            ws.o_block_cols,
                            ws.o_out,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// v2 phase 2: canonical-block O reduction on the root device plus the K/V shadow gathers,
    /// returning a fresh model-engine output ordered behind `ev_oproj` on `e`'s stream.
    ///
    /// The caller must have queued every rank's attention work (reading `ws.gated`, `ws.k`,
    /// `ws.v_raw`) on the rank streams before this call. Reduction order is identical to
    /// `step_bf16_row_parallel_resident_native`: zeros, then rank 0's blocks, then each peer
    /// rank's blocks, one `add` per block.
    pub(crate) fn decode_v2_finish(
        &self,
        ws: &mut StepTpDecodeV2Ws,
        e: &Engine,
        o_m: &ResidentStepBf16RowParallel,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let ranks = self.ranks.len();
        if e.ctx().ordinal() != ws.e_device {
            return Err("step TP decode v2 finish engine changed".into());
        }
        // MEMRA_STEP_TP_QKV_FUSED extends to the O path: one matvec_f32_b4 launch per rank
        // (in-order canonical block accumulation per element) and a single peer-copy + add on
        // the root, replacing 4 cuBLASLt launches per rank + the 4-copy/8-add chain. Same
        // numeric-class door and gate as the fused QKV projection.
        let o_fused = step_tp_qkv_fused_enabled()? && ws.blocks_per_rank == 4 && ranks == 2;

        // Per-rank O block partials on the owning rank's stream (serial after the attention
        // kernels the driver queued there), then the rank-done event for root's peer reads.
        for rank in 0..ranks {
            self.decode_v2_finish_rank_partial(ws, o_m, o_fused, rank)?;
            if rank == 0 {
                // root == rank0: its own stream order covers the partial; only peers need
                // the record/wait pair (host-op diet, matches the routes-arm skip).
                continue;
            }
            let engine = &self.ranks[rank];
            let _main = engine.gpu.enter_main()?;
            ws.ev_rank[rank].record(&engine.stream())?;
        }

        // Root reduce in canonical order + shadow gathers, all on the root stream.
        let root = &self.ranks[0];
        #[allow(unused_assignments)]
        let mut final_in_a = false;
        {
            let _main = root.gpu.enter_main()?;
            for ev in ws.ev_rank.iter().skip(1) {
                root.stream().wait(ev)?;
            }
            if o_fused && oproj_direct_on() && ranks == 2 && no_local_shadow_on() {
                // DIRECT JOIN: rank1's partial already sits in root memory (P2P kernel
                // stores; visibility guaranteed by the ev_rank[1] wait above), rank0's
                // partial is root-stream-ordered — record ONE event and let the model
                // engine do the single add itself, straight into its own output row.
                // Same operands, same add order as finish_root_fused: BIT-IDENTICAL.
                ws.ev_oproj.record(&root.stream())?;
                let _main = e.gpu.enter_main()?;
                e.stream().wait(&ws.ev_oproj)?;
                let mut output = e.uninit(ws.o_out)?;
                if oproj_tail_on() && oproj_tail_eligible() {
                    // M2: defer the add into the residual+norm consumer (waits stay HERE;
                    // only the arithmetic moves). `output` is returned unwritten.
                    use cudarc::driver::DevicePtr;
                    let stream = e.stream();
                    let (p0, _g0) = ws.o_partials[0][0].device_ptr(&stream);
                    let (p1, _g1) = ws.o_partials[1][0].device_ptr(&stream);
                    set_oproj_tail((p0, p1));
                    return Ok(output);
                }
                e.add(
                    &ws.o_partials[0][0],
                    &ws.o_partials[1][0],
                    &mut output,
                    ws.o_out,
                )?;
                return Ok(output);
            }
            if o_fused {
                self.decode_v2_finish_root_fused(ws)?;
                ws.ev_oproj.record(&root.stream())?;
                let _main = e.gpu.enter_main()?;
                e.stream().wait(&ws.ev_oproj)?;
                let mut output = e.uninit(ws.o_out)?;
                e.stream().memcpy_dtod(
                    &ws.reduce_a.slice(0..ws.o_out),
                    &mut output.slice_mut(0..ws.o_out),
                )?;
                return Ok(output);
            }
            let mut first = true;
            let mut current_is_a = false;
            for rank in 0..ranks {
                for block in 0..ws.blocks_per_rank {
                    let use_peer = rank != 0;
                    if use_peer {
                        raw_copy_bytes(
                            ws.raw_peer_partial,
                            ws.raw_o_partials[rank][block],
                            ws.o_out * std::mem::size_of::<f32>(),
                            root,
                        )?;
                    }
                    // add(prev, partial) -> the other reduce buffer, exactly one add per block
                    match (first, current_is_a, use_peer) {
                        (true, _, true) => {
                            root.add(&ws.zeros, &ws.peer_partial, &mut ws.reduce_a, ws.o_out)?
                        }
                        (true, _, false) => root.add(
                            &ws.zeros,
                            &ws.o_partials[0][block],
                            &mut ws.reduce_a,
                            ws.o_out,
                        )?,
                        (false, true, true) => {
                            root.add(&ws.reduce_a, &ws.peer_partial, &mut ws.reduce_b, ws.o_out)?
                        }
                        (false, true, false) => root.add(
                            &ws.reduce_a,
                            &ws.o_partials[0][block],
                            &mut ws.reduce_b,
                            ws.o_out,
                        )?,
                        (false, false, true) => {
                            root.add(&ws.reduce_b, &ws.peer_partial, &mut ws.reduce_a, ws.o_out)?
                        }
                        (false, false, false) => root.add(
                            &ws.reduce_b,
                            &ws.o_partials[0][block],
                            &mut ws.reduce_a,
                            ws.o_out,
                        )?,
                    }
                    current_is_a = first || !current_is_a;
                    first = false;
                }
            }
            final_in_a = current_is_a;

            if !no_local_shadow_on() {
                let bytes = ws.local_kv_dim * std::mem::size_of::<f32>();
                for rank in 0..ranks {
                    let offset = rank * bytes;
                    raw_copy_bytes(ws.raw_k_shadow + offset as u64, ws.raw_k[rank], bytes, root)?;
                    raw_copy_bytes(
                        ws.raw_v_shadow + offset as u64,
                        ws.raw_v_raw[rank],
                        bytes,
                        root,
                    )?;
                }
            }
            ws.ev_oproj.record(&root.stream())?;
        }

        // Model-engine output: e waits the root event, then copies the reduced row into a
        // fresh e-context buffer (same ownership contract as v1's `e.htod`). The same wait
        // orders the driver's shadow append (it reads ws.k_shadow/ws.v_shadow on e's stream).
        let _main = e.gpu.enter_main()?;
        e.stream().wait(&ws.ev_oproj)?;
        let mut output = e.uninit(ws.o_out)?;
        let source = if final_in_a {
            &ws.reduce_a
        } else {
            &ws.reduce_b
        };
        e.stream().memcpy_dtod(
            &source.slice(0..ws.o_out),
            &mut output.slice_mut(0..ws.o_out),
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub fn run_routed_experts(
        &self,
        experts: &ResidentExpertParallel,
        input: &[f32],
        tokens: usize,
        selected: &[usize],
        route_weights: &[f32],
        experts_per_token: usize,
        activation_limit: Option<f32>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        validate_step_expert_activation_limit(activation_limit)?;
        validate_ep_residency(&self.ranks, experts)?;
        validate_activations(input, tokens, experts.input_width)?;
        let pairs = tokens
            .checked_mul(experts_per_token)
            .ok_or("EP route count overflow")?;
        if selected.len() != pairs || route_weights.len() != pairs {
            return Err(format!(
                "EP routes selected={} weights={} != tokens {tokens} x experts/token \
                 {experts_per_token} ({pairs})",
                selected.len(),
                route_weights.len(),
            )
            .into());
        }
        if !route_weights.iter().all(|weight| weight.is_finite()) {
            return Err("EP route weights contain a non-finite value".into());
        }
        if self.native_p2p {
            return self.run_routed_experts_native(
                experts,
                input,
                tokens,
                selected,
                route_weights,
                experts_per_token,
                activation_limit,
            );
        }

        let mut output = vec![0.0f32; tokens * experts.input_width];
        let per_rank = experts.expert_count / experts.ranks.len();
        for token in 0..tokens {
            let input_row = &input[token * experts.input_width..(token + 1) * experts.input_width];
            for slot in 0..experts_per_token {
                let pair = token * experts_per_token + slot;
                let expert = selected[pair];
                if expert >= experts.expert_count {
                    return Err(format!(
                        "EP selected expert {expert} outside 0..{}",
                        experts.expert_count
                    )
                    .into());
                }
                let owner = expert / per_rank;
                let local_expert = expert - experts.ranks[owner].gate.expert_range.start;
                let rank = &experts.ranks[owner];
                let engine = &self.ranks[owner];
                let gate =
                    run_resident_bank_expert(engine, &rank.gate, local_expert, input_row, 1)?;
                let up = run_resident_bank_expert(engine, &rank.up, local_expert, input_row, 1)?;
                let activated: Vec<f32> = gate
                    .iter()
                    .zip(&up)
                    .map(|(&gate, &up)| step_expert_activation_host(gate, up, activation_limit))
                    .collect();
                debug_assert_eq!(activated.len(), experts.expert_width);
                let down =
                    run_resident_bank_expert(engine, &rank.down, local_expert, &activated, 1)?;
                let weight = route_weights[pair];
                for (sum, value) in output
                    [token * experts.input_width..(token + 1) * experts.input_width]
                    .iter_mut()
                    .zip(down)
                {
                    *sum += weight * value;
                }
            }
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    fn run_routed_experts_native(
        &self,
        experts: &ResidentExpertParallel,
        input: &[f32],
        tokens: usize,
        selected: &[usize],
        route_weights: &[f32],
        experts_per_token: usize,
        activation_limit: Option<f32>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if !self.native_p2p || self.ranks.len() < 2 {
            return Err("native EP execution requires at least two P2P ranks".into());
        }
        if self.ep_device_arithmetic {
            return self.run_routed_experts_native_device(
                experts,
                input,
                tokens,
                selected,
                route_weights,
                experts_per_token,
                activation_limit,
            );
        }
        let mut output = vec![0.0f32; tokens * experts.input_width];
        let per_rank = experts.expert_count / experts.ranks.len();
        for token in 0..tokens {
            let input_row = &input[token * experts.input_width..(token + 1) * experts.input_width];
            let mut rank_inputs = (0..self.ranks.len())
                .map(|_| None)
                .collect::<Vec<Option<CudaSlice<f32>>>>();
            rank_inputs[0] = Some({
                let root = &self.ranks[0];
                let _main = root.gpu.enter_main()?;
                root.htod(input_row)?
            });

            for slot in 0..experts_per_token {
                let pair = token * experts_per_token + slot;
                let expert = selected[pair];
                if expert >= experts.expert_count {
                    return Err(format!(
                        "EP selected expert {expert} outside 0..{}",
                        experts.expert_count
                    )
                    .into());
                }
                let owner = expert / per_rank;
                let local_expert = expert - experts.ranks[owner].gate.expert_range.start;
                if rank_inputs[owner].is_none() {
                    let peer_input = {
                        let root_input = rank_inputs[0]
                            .as_ref()
                            .ok_or("native EP lost its root input")?;
                        let engine = &self.ranks[owner];
                        let _main = engine.gpu.enter_main()?;
                        let mut peer_input = engine.uninit(experts.input_width)?;
                        engine.stream().memcpy_dtod(root_input, &mut peer_input)?;
                        peer_input
                    };
                    rank_inputs[owner] = Some(peer_input);
                }

                let rank = &experts.ranks[owner];
                let engine = &self.ranks[owner];
                let owner_input = rank_inputs[owner]
                    .as_ref()
                    .ok_or("native EP owner input is absent after dispatch")?;
                let gate = run_resident_bank_expert_device(
                    engine,
                    &rank.gate,
                    local_expert,
                    owner_input,
                    1,
                )?;
                let up = run_resident_bank_expert_device(
                    engine,
                    &rank.up,
                    local_expert,
                    owner_input,
                    1,
                )?;
                let (gate, up) = {
                    let _main = engine.gpu.enter_main()?;
                    (engine.dtoh(&gate)?, engine.dtoh(&up)?)
                };
                let activated = gate
                    .iter()
                    .zip(&up)
                    .map(|(&gate, &up)| step_expert_activation_host(gate, up, activation_limit))
                    .collect::<Vec<_>>();
                debug_assert_eq!(activated.len(), experts.expert_width);
                let activated = {
                    let _main = engine.gpu.enter_main()?;
                    engine.htod(&activated)?
                };
                let down = run_resident_bank_expert_device(
                    engine,
                    &rank.down,
                    local_expert,
                    &activated,
                    1,
                )?;
                let down = if owner == 0 {
                    let _main = engine.gpu.enter_main()?;
                    engine.dtoh(&down)?
                } else {
                    let root = &self.ranks[0];
                    let _main = root.gpu.enter_main()?;
                    let mut root_down = root.uninit(experts.input_width)?;
                    root.stream().memcpy_dtod(&down, &mut root_down)?;
                    root.dtoh(&root_down)?
                };
                let weight = route_weights[pair];
                for (sum, value) in output
                    [token * experts.input_width..(token + 1) * experts.input_width]
                    .iter_mut()
                    .zip(down)
                {
                    *sum += weight * value;
                }
            }
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    fn run_routed_experts_native_device(
        &self,
        experts: &ResidentExpertParallel,
        input: &[f32],
        tokens: usize,
        selected: &[usize],
        route_weights: &[f32],
        experts_per_token: usize,
        activation_limit: Option<f32>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if !self.native_p2p || !self.ep_device_arithmetic || self.ranks.len() < 2 {
            return Err(
                "device-resident EP arithmetic requires at least two native P2P ranks".into(),
            );
        }
        let mut output = Vec::with_capacity(tokens * experts.input_width);
        let per_rank = experts.expert_count / experts.ranks.len();
        let root = &self.ranks[0];
        for token in 0..tokens {
            let input_row = &input[token * experts.input_width..(token + 1) * experts.input_width];
            let mut rank_inputs = (0..self.ranks.len())
                .map(|_| None)
                .collect::<Vec<Option<CudaSlice<f32>>>>();
            rank_inputs[0] = Some({
                let _main = root.gpu.enter_main()?;
                root.htod(input_row)?
            });
            let mut root_output = {
                let _main = root.gpu.enter_main()?;
                root.zeros(experts.input_width)?
            };
            let mut remote_down_keepalive = Vec::new();

            for slot in 0..experts_per_token {
                let pair = token * experts_per_token + slot;
                let expert = selected[pair];
                if expert >= experts.expert_count {
                    return Err(format!(
                        "EP selected expert {expert} outside 0..{}",
                        experts.expert_count
                    )
                    .into());
                }
                let owner = expert / per_rank;
                let local_expert = expert - experts.ranks[owner].gate.expert_range.start;
                if rank_inputs[owner].is_none() {
                    let peer_input = {
                        let root_input = rank_inputs[0]
                            .as_ref()
                            .ok_or("native EP lost its root input")?;
                        let engine = &self.ranks[owner];
                        let _main = engine.gpu.enter_main()?;
                        let mut peer_input = engine.uninit(experts.input_width)?;
                        engine.stream().memcpy_dtod(root_input, &mut peer_input)?;
                        peer_input
                    };
                    rank_inputs[owner] = Some(peer_input);
                }

                let rank = &experts.ranks[owner];
                let engine = &self.ranks[owner];
                let owner_input = rank_inputs[owner]
                    .as_ref()
                    .ok_or("native EP owner input is absent after dispatch")?;
                let gate = run_resident_bank_expert_device(
                    engine,
                    &rank.gate,
                    local_expert,
                    owner_input,
                    1,
                )?;
                let up = run_resident_bank_expert_device(
                    engine,
                    &rank.up,
                    local_expert,
                    owner_input,
                    1,
                )?;
                let activated = {
                    let _main = engine.gpu.enter_main()?;
                    let mut activated = engine.uninit(experts.expert_width)?;
                    if let Some(limit) = activation_limit {
                        engine.silu_clamped_mul_host_expf(
                            &gate,
                            &up,
                            limit,
                            &mut activated,
                            experts.expert_width,
                        )?;
                    } else {
                        engine.silu_mul_host_expf(
                            &gate,
                            &up,
                            &mut activated,
                            experts.expert_width,
                        )?;
                    }
                    activated
                };
                let down = run_resident_bank_expert_device(
                    engine,
                    &rank.down,
                    local_expert,
                    &activated,
                    1,
                )?;
                let root_down = if owner == 0 {
                    down
                } else {
                    let _main = root.gpu.enter_main()?;
                    let mut root_down = root.uninit(experts.input_width)?;
                    root.stream().memcpy_dtod(&down, &mut root_down)?;
                    // The peer copy runs on the root stream. Keep its remote source alive until
                    // the final root readback synchronizes that stream; otherwise async free can
                    // recycle the owner's allocation while cuMemcpyPeerAsync is still reading it.
                    remote_down_keepalive.push(down);
                    root_down
                };
                let _main = root.gpu.enter_main()?;
                let mut destination = root_output.slice_mut(0..experts.input_width);
                root.axpy_host_into(
                    &root_down.slice(0..root_down.len()),
                    route_weights[pair],
                    &mut destination,
                    experts.input_width,
                )?;
            }

            let _main = root.gpu.enter_main()?;
            let root_output = root.dtoh(&root_output)?;
            drop(remote_down_keepalive);
            output.extend(root_output);
        }
        Ok(output)
    }
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn validate_column_shape(matrix: E4m3BlockMatrix<'_>, tp: usize) -> Result<(), String> {
    if matrix.out_features % tp != 0 {
        return Err(format!(
            "column-parallel out_features {} is not divisible by TP={tp}",
            matrix.out_features
        ));
    }
    let local_out = matrix.out_features / tp;
    if !local_out.is_multiple_of(FP8_BLOCK) {
        return Err(format!(
            "column-parallel output shard {local_out} cuts through a {FP8_BLOCK}-row \
             E4M3 scale block"
        ));
    }
    Ok(())
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn step_bf16_canonical_chunk_rows(out_features: usize, tp: usize) -> Result<usize, String> {
    if !matches!(tp, 1 | 2 | 4 | 8) {
        return Err(format!(
            "Step BF16 canonical projection requires TP1/TP2/TP4/TP8, got TP={tp}"
        ));
    }
    if out_features == 0 || !out_features.is_multiple_of(PRODUCT_MAX_CARDS) {
        return Err(format!(
            "Step BF16 output width {out_features} is not divisible by the TP8 product envelope"
        ));
    }
    let canonical_rows = out_features / PRODUCT_MAX_CARDS;
    let local_out = out_features / tp;
    if local_out % canonical_rows != 0 {
        return Err(format!(
            "Step BF16 TP={tp} output shard {local_out} is not divisible by canonical \
             {canonical_rows}-row chunks"
        ));
    }
    Ok(canonical_rows)
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn step_bf16_canonical_chunk_cols(in_features: usize, tp: usize) -> Result<usize, String> {
    if !matches!(tp, 1 | 2 | 4 | 8) {
        return Err(format!(
            "Step BF16 canonical row projection requires TP1/TP2/TP4/TP8, got TP={tp}"
        ));
    }
    if in_features == 0 || !in_features.is_multiple_of(PRODUCT_MAX_CARDS) {
        return Err(format!(
            "Step BF16 input width {in_features} is not divisible by the TP8 product envelope"
        ));
    }
    let canonical_cols = in_features / PRODUCT_MAX_CARDS;
    let local_in = in_features / tp;
    if local_in % canonical_cols != 0 {
        return Err(format!(
            "Step BF16 TP={tp} input shard {local_in} is not divisible by canonical \
             {canonical_cols}-column chunks"
        ));
    }
    Ok(canonical_cols)
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn validate_row_shape(matrix: E4m3BlockMatrix<'_>, tp: usize) -> Result<(), String> {
    if matrix.in_features % tp != 0 {
        return Err(format!(
            "row-parallel in_features {} is not divisible by TP={tp}",
            matrix.in_features
        ));
    }
    let local_in = matrix.in_features / tp;
    if !local_in.is_multiple_of(FP8_BLOCK) {
        return Err(format!(
            "row-parallel input shard {local_in} cuts through a {FP8_BLOCK}-column \
             E4M3 scale block"
        ));
    }
    Ok(())
}

fn upload_rank(
    engine: &Engine,
    matrix: E4m3BlockMatrix<'_>,
) -> Result<ResidentE4m3Rank, Box<dyn std::error::Error>> {
    let _main = engine.gpu.enter_main()?;
    matrix.validate()?;
    Ok(ResidentE4m3Rank {
        codes: engine.htod_bytes(matrix.codes)?,
        scales: engine.htod(matrix.scales)?,
        out_features: matrix.out_features,
        in_features: matrix.in_features,
    })
}

fn upload_bf16_rank(
    engine: &Engine,
    matrix: Bf16Matrix<'_>,
    f32_mirror: bool,
) -> Result<ResidentBf16Rank, Box<dyn std::error::Error>> {
    let _main = engine.gpu.enter_main()?;
    matrix.validate()?;
    let bytes = engine.htod_bytes(matrix.bytes)?;
    let weight = if f32_mirror {
        let values = matrix
            .out_features
            .checked_mul(matrix.in_features)
            .ok_or("resident BF16 mirror element count overflow")?;
        ResidentBf16Weight::F32(engine.bf16_to_f32(&bytes.slice(0..bytes.len()), values)?)
    } else {
        ResidentBf16Weight::Bf16(bytes)
    };
    // MEMRA_STEP_TP_W8: encode the q8_0 decode mirror once, here, while the bf16 bytes are
    // already resident. Rows whose in_features is not a multiple of 32 have no q8_0 form and
    // simply keep the bf16 program (the decode arm checks for the mirror, never assumes it).
    let q8 = if crate::step_tp_w8_on() && matrix.in_features.is_multiple_of(32) {
        if let ResidentBf16Weight::Bf16(bytes) = &weight {
            // Two steps, because the mmvq rp kernel does NOT read ggml-interleaved 34-byte
            // blocks: it reads a PLANAR mirror (all quants, then all half scales — the
            // q4_0/NVFP4 rp convention). The encoder writes the interleaved form and
            // `build_q8_rp4_raw` — the same kernel the GGUF loader uses — splits it into
            // planes. Skipping the split is what made the first W8 gate return zeros
            // (verify-prefill argmax=0, maxdiff=0.000e0).
            let row_bytes = Engine::q8_0_row_bytes(matrix.in_features);
            let mut interleaved = engine.alloc_u8_uninit(matrix.out_features * row_bytes)?;
            engine.encode_q8_0_from_bf16(
                bytes,
                &mut interleaved,
                matrix.in_features,
                matrix.out_features,
            )?;
            let mirror =
                engine.build_q8_rp4_raw(&interleaved, matrix.in_features, matrix.out_features)?;
            Some(mirror)
        } else {
            None
        }
    } else {
        None
    };
    Ok(ResidentBf16Rank {
        weight,
        out_features: matrix.out_features,
        in_features: matrix.in_features,
        q8,
    })
}

fn upload_expert_bank_rank(
    engine: &Engine,
    bank: E4m3ExpertBank<'_>,
    expert_range: Range<usize>,
) -> Result<ResidentE4m3ExpertBankRank, Box<dyn std::error::Error>> {
    let _main = engine.gpu.enter_main()?;
    bank.validate()?;
    if expert_range.start >= expert_range.end || expert_range.end > bank.expert_count {
        return Err(format!(
            "invalid EP expert range {expert_range:?} for {} experts",
            bank.expert_count
        )
        .into());
    }
    let code_stride = bank.out_features * bank.in_features;
    let scale_stride = bank.out_features.div_ceil(FP8_BLOCK) * bank.in_features.div_ceil(FP8_BLOCK);
    Ok(ResidentE4m3ExpertBankRank {
        codes: engine.htod_bytes(
            &bank.codes[expert_range.start * code_stride..expert_range.end * code_stride],
        )?,
        scales: engine.htod(
            &bank.scales[expert_range.start * scale_stride..expert_range.end * scale_stride],
        )?,
        expert_range,
        out_features: bank.out_features,
        in_features: bank.in_features,
        code_stride,
        scale_stride,
        k_blocks: None,
    })
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn validate_column_bank_shape(bank: E4m3ExpertBank<'_>, tp: usize) -> Result<(), String> {
    if bank.out_features % tp != 0 {
        return Err(format!(
            "TP expert output width {} is not divisible by TP={tp}",
            bank.out_features
        ));
    }
    let local_out = bank.out_features / tp;
    if !local_out.is_multiple_of(FP8_BLOCK) {
        return Err(format!(
            "TP expert output shard {local_out} cuts through a {FP8_BLOCK}-row E4M3 scale block"
        ));
    }
    Ok(())
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn validate_row_bank_shape(bank: E4m3ExpertBank<'_>, tp: usize) -> Result<(), String> {
    if bank.in_features % tp != 0 {
        return Err(format!(
            "TP expert input width {} is not divisible by TP={tp}",
            bank.in_features
        ));
    }
    let local_in = bank.in_features / tp;
    if !local_in.is_multiple_of(FP8_BLOCK) {
        return Err(format!(
            "TP expert input shard {local_in} cuts through a {FP8_BLOCK}-column E4M3 scale block"
        ));
    }
    Ok(())
}

fn upload_column_bank_rank(
    engine: &Engine,
    bank: E4m3ExpertBank<'_>,
    tp: usize,
    rank: usize,
) -> Result<ResidentE4m3ExpertBankRank, Box<dyn std::error::Error>> {
    let _main = engine.gpu.enter_main()?;
    let packed = pack_column_bank_rank(bank, tp, rank)?;
    Ok(ResidentE4m3ExpertBankRank {
        codes: engine.htod_bytes(&packed.codes)?,
        scales: engine.htod(&packed.scales)?,
        expert_range: packed.expert_range,
        out_features: packed.out_features,
        in_features: packed.in_features,
        code_stride: packed.code_stride,
        scale_stride: packed.scale_stride,
        k_blocks: packed.k_blocks,
    })
}

fn pack_column_bank_rank(
    bank: E4m3ExpertBank<'_>,
    tp: usize,
    rank: usize,
) -> Result<PackedE4m3ExpertBankRank, String> {
    bank.validate()?;
    validate_column_bank_shape(bank, tp)?;
    if rank >= tp {
        return Err(format!("TP rank {rank} outside 0..{tp}"));
    }
    let local_out = bank.out_features / tp;
    let full_code_stride = bank.out_features * bank.in_features;
    let local_code_stride = local_out * bank.in_features;
    let scale_cols = bank.in_features.div_ceil(FP8_BLOCK);
    let full_scale_stride = bank.out_features.div_ceil(FP8_BLOCK) * scale_cols;
    let local_scale_rows = local_out / FP8_BLOCK;
    let local_scale_stride = local_scale_rows * scale_cols;
    let mut codes = Vec::with_capacity(bank.expert_count * local_code_stride);
    let mut scales = Vec::with_capacity(bank.expert_count * local_scale_stride);
    let row_start = rank * local_out;
    let scale_row_start = rank * local_scale_rows;
    for expert in 0..bank.expert_count {
        let code_start = expert * full_code_stride + row_start * bank.in_features;
        codes.extend_from_slice(&bank.codes[code_start..code_start + local_code_stride]);
        let scale_start = expert * full_scale_stride + scale_row_start * scale_cols;
        scales.extend_from_slice(&bank.scales[scale_start..scale_start + local_scale_stride]);
    }
    Ok(PackedE4m3ExpertBankRank {
        codes,
        scales,
        expert_range: 0..bank.expert_count,
        out_features: local_out,
        in_features: bank.in_features,
        code_stride: local_code_stride,
        scale_stride: local_scale_stride,
        k_blocks: None,
    })
}

fn upload_row_bank_rank(
    engine: &Engine,
    bank: E4m3ExpertBank<'_>,
    tp: usize,
    rank: usize,
) -> Result<ResidentE4m3ExpertBankRank, Box<dyn std::error::Error>> {
    let _main = engine.gpu.enter_main()?;
    let packed = pack_row_bank_rank(bank, tp, rank)?;
    Ok(ResidentE4m3ExpertBankRank {
        codes: engine.htod_bytes(&packed.codes)?,
        scales: engine.htod(&packed.scales)?,
        expert_range: packed.expert_range,
        out_features: packed.out_features,
        in_features: packed.in_features,
        code_stride: packed.code_stride,
        scale_stride: packed.scale_stride,
        k_blocks: packed.k_blocks,
    })
}

fn pack_row_bank_rank(
    bank: E4m3ExpertBank<'_>,
    tp: usize,
    rank: usize,
) -> Result<PackedE4m3ExpertBankRank, String> {
    bank.validate()?;
    validate_row_bank_shape(bank, tp)?;
    if rank >= tp {
        return Err(format!("TP rank {rank} outside 0..{tp}"));
    }
    let local_in = bank.in_features / tp;
    let full_code_stride = bank.out_features * bank.in_features;
    let local_code_stride = bank.out_features * local_in;
    let full_scale_cols = bank.in_features.div_ceil(FP8_BLOCK);
    let local_scale_cols = local_in / FP8_BLOCK;
    let scale_rows = bank.out_features.div_ceil(FP8_BLOCK);
    let full_scale_stride = scale_rows * full_scale_cols;
    let local_scale_stride = scale_rows * local_scale_cols;
    let global_block_start = rank * local_scale_cols;
    let mut codes = Vec::with_capacity(bank.expert_count * local_code_stride);
    let mut scales = Vec::with_capacity(bank.expert_count * local_scale_stride);
    for expert in 0..bank.expert_count {
        let expert_code_start = expert * full_code_stride;
        let expert_scale_start = expert * full_scale_stride;
        for local_block in 0..local_scale_cols {
            let global_block = global_block_start + local_block;
            let column_start = global_block * FP8_BLOCK;
            for row in 0..bank.out_features {
                let start = expert_code_start + row * bank.in_features + column_start;
                codes.extend_from_slice(&bank.codes[start..start + FP8_BLOCK]);
            }
            for row in 0..scale_rows {
                scales.push(bank.scales[expert_scale_start + row * full_scale_cols + global_block]);
            }
        }
    }
    Ok(PackedE4m3ExpertBankRank {
        codes,
        scales,
        expert_range: 0..bank.expert_count,
        out_features: bank.out_features,
        in_features: local_in,
        code_stride: local_code_stride,
        scale_stride: local_scale_stride,
        k_blocks: Some(local_scale_cols),
    })
}

fn validate_resident_ranks(engines: &[Engine], ranks: &[ResidentE4m3Rank]) -> Result<(), String> {
    if engines.len() != ranks.len() {
        return Err(format!(
            "resident TP rank count {} != runtime rank count {}",
            ranks.len(),
            engines.len()
        ));
    }
    for (rank, (engine, matrix)) in engines.iter().zip(ranks).enumerate() {
        let device = engine.ctx().ordinal();
        if matrix.codes.ordinal() != device || matrix.scales.ordinal() != device {
            return Err(format!(
                "resident TP rank {rank} is not owned by runtime device {device}"
            ));
        }
    }
    Ok(())
}

fn validate_tp_bank_residency(
    engines: &[Engine],
    experts: &ResidentTpExpertBank,
) -> Result<(), String> {
    if engines.len() != experts.gate.len()
        || engines.len() != experts.up.len()
        || engines.len() != experts.down.len()
    {
        return Err(format!(
            "resident TP expert-bank rank counts gate={} up={} down={} != runtime {}",
            experts.gate.len(),
            experts.up.len(),
            experts.down.len(),
            engines.len()
        ));
    }
    for (rank, engine) in engines.iter().enumerate() {
        let device = engine.ctx().ordinal();
        for (projection, bank) in [
            ("gate", &experts.gate[rank]),
            ("up", &experts.up[rank]),
            ("down", &experts.down[rank]),
        ] {
            if bank.codes.ordinal() != device || bank.scales.ordinal() != device {
                return Err(format!(
                    "resident TP rank {rank} {projection} bank is not owned by runtime device \
                     {device}"
                ));
            }
        }
    }
    Ok(())
}

fn validate_ep_residency(
    engines: &[Engine],
    experts: &ResidentExpertParallel,
) -> Result<(), String> {
    if engines.len() != experts.ranks.len() {
        return Err(format!(
            "resident EP rank count {} != runtime rank count {}",
            experts.ranks.len(),
            engines.len()
        ));
    }
    for (rank, (engine, resident)) in engines.iter().zip(&experts.ranks).enumerate() {
        let device = engine.ctx().ordinal();
        for (projection, bank) in [
            ("gate", &resident.gate),
            ("up", &resident.up),
            ("down", &resident.down),
        ] {
            if bank.codes.ordinal() != device || bank.scales.ordinal() != device {
                return Err(format!(
                    "resident EP rank {rank} {projection} bank is not owned by runtime device \
                     {device}"
                ));
            }
        }
    }
    Ok(())
}

fn run_rank(
    engine: &Engine,
    matrix: E4m3BlockMatrix<'_>,
    activations: &[f32],
    tokens: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let _main = engine.gpu.enter_main()?;
    let codes = engine.htod_bytes(matrix.codes)?;
    let scales = engine.htod(matrix.scales)?;
    let activations = engine.htod(activations)?;
    let output = engine.qmatvec_mmq_fp8_blk(
        &codes,
        &scales,
        &activations,
        tokens,
        matrix.in_features,
        matrix.out_features,
    )?;
    engine.dtoh(&output)
}

fn run_resident_rank(
    engine: &Engine,
    matrix: &ResidentE4m3Rank,
    activations: &[f32],
    tokens: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let _main = engine.gpu.enter_main()?;
    let activations = engine.htod(activations)?;
    let output = engine.qmatvec_mmq_fp8_blk(
        &matrix.codes,
        &matrix.scales,
        &activations,
        tokens,
        matrix.in_features,
        matrix.out_features,
    )?;
    engine.dtoh(&output)
}

fn run_resident_bf16_rank(
    engine: &Engine,
    matrix: &ResidentBf16Rank,
    activations: &[f32],
    tokens: usize,
    canonical_chunk_rows: Option<usize>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let _main = engine.gpu.enter_main()?;
    let activations = engine.htod(activations)?;
    let output = run_resident_bf16_rank_device(
        engine,
        matrix,
        &activations,
        tokens,
        canonical_chunk_rows,
        false,
    )?;
    engine.dtoh(&output)
}

fn run_resident_bf16_rank_device(
    engine: &Engine,
    matrix: &ResidentBf16Rank,
    activations: &CudaSlice<f32>,
    tokens: usize,
    canonical_chunk_rows: Option<usize>,
    strided_chunk_output: bool,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let _main = engine.gpu.enter_main()?;
    if activations.ordinal() != engine.ctx().ordinal() {
        return Err(format!(
            "resident BF16 activation device {} != rank device {}",
            activations.ordinal(),
            engine.ctx().ordinal()
        )
        .into());
    }
    if activations.len() != tokens * matrix.in_features {
        return Err(format!(
            "resident BF16 activation count {} != {tokens}x{}",
            activations.len(),
            matrix.in_features
        )
        .into());
    }
    match (&matrix.weight, canonical_chunk_rows) {
        (ResidentBf16Weight::Bf16(bytes), Some(rows)) => engine
            .linear_bf16_resident_canonical_rows(
                activations,
                bytes,
                tokens,
                matrix.in_features,
                matrix.out_features,
                rows,
            ),
        (ResidentBf16Weight::Bf16(bytes), None) => engine.linear_bf16_resident(
            activations,
            bytes,
            tokens,
            matrix.in_features,
            matrix.out_features,
        ),
        (ResidentBf16Weight::F32(values), Some(rows)) if strided_chunk_output => engine
            .linear_f32_resident_canonical_rows_strided(
                activations,
                values,
                tokens,
                matrix.in_features,
                matrix.out_features,
                rows,
            ),
        (ResidentBf16Weight::F32(values), Some(rows)) => engine.linear_f32_resident_canonical_rows(
            activations,
            values,
            tokens,
            matrix.in_features,
            matrix.out_features,
            rows,
        ),
        (ResidentBf16Weight::F32(values), None) => engine.linear(
            activations,
            values,
            tokens,
            matrix.in_features,
            matrix.out_features,
        ),
    }
}

fn validate_resident_bf16_ranks(
    engines: &[Engine],
    ranks: &[ResidentBf16Rank],
) -> Result<(), String> {
    if engines.len() != ranks.len() {
        return Err(format!(
            "resident BF16 TP rank count {} != runtime rank count {}",
            ranks.len(),
            engines.len(),
        ));
    }
    for (rank, (engine, matrix)) in engines.iter().zip(ranks).enumerate() {
        let device = engine.ctx().ordinal();
        if matrix.weight.ordinal() != device {
            return Err(format!(
                "resident BF16 TP rank {rank} is not owned by runtime device {device}"
            ));
        }
    }
    Ok(())
}

fn validate_step_bf16_row_residency(
    engines: &[Engine],
    matrix: &ResidentStepBf16RowParallel,
) -> Result<(), String> {
    if engines.len() != matrix.ranks.len() {
        return Err(format!(
            "resident Step BF16 row rank count {} != runtime rank count {}",
            matrix.ranks.len(),
            engines.len(),
        ));
    }
    let canonical_cols = step_bf16_canonical_chunk_cols(matrix.in_features, engines.len())?;
    if matrix.canonical_chunk_cols != canonical_cols {
        return Err(format!(
            "resident Step BF16 row canonical columns {} != registered {canonical_cols}",
            matrix.canonical_chunk_cols
        ));
    }
    let blocks_per_rank = PRODUCT_MAX_CARDS / engines.len();
    for (rank, (engine, blocks)) in engines.iter().zip(&matrix.ranks).enumerate() {
        if blocks.len() != blocks_per_rank {
            return Err(format!(
                "resident Step BF16 row rank {rank} has {} blocks, expected {blocks_per_rank}",
                blocks.len()
            ));
        }
        let device = engine.ctx().ordinal();
        for (block, resident) in blocks.iter().enumerate() {
            if resident.weight.ordinal() != device
                || resident.in_features != canonical_cols
                || resident.out_features != matrix.out_features
            {
                return Err(format!(
                    "resident Step BF16 row rank {rank} block {block} has inconsistent \
                     device or geometry"
                ));
            }
        }
    }
    Ok(())
}

fn validate_replicated_device_rows(
    engines: &[Engine],
    rows: &ResidentReplicatedDeviceRows,
) -> Result<(), String> {
    let rank_lengths = rows
        .ranks
        .iter()
        .map(|rank_rows| rank_rows.len())
        .collect::<Vec<_>>();
    replicated_device_row_values(rows.tokens, rows.width, engines.len(), &rank_lengths)?;
    if rows
        .ranks
        .iter()
        .zip(engines)
        .any(|(rank_rows, engine)| rank_rows.ordinal() != engine.ctx().ordinal())
    {
        return Err("replicated device rows are owned by the wrong CUDA contexts".into());
    }
    Ok(())
}

fn replicated_device_row_values(
    tokens: usize,
    width: usize,
    expected_ranks: usize,
    rank_lengths: &[usize],
) -> Result<usize, String> {
    let values = tokens
        .checked_mul(width)
        .ok_or("replicated device row size overflow")?;
    if tokens == 0
        || width == 0
        || expected_ranks == 0
        || rank_lengths.len() != expected_ranks
        || rank_lengths.iter().any(|&rank_len| rank_len != values)
    {
        return Err(format!(
            "replicated device rows have inconsistent geometry tokens={} width={} ranks={}/{}",
            tokens,
            width,
            rank_lengths.len(),
            expected_ranks
        ));
    }
    Ok(values)
}

fn replicated_device_row_source_values(
    tokens: usize,
    width: usize,
    source_len: usize,
    source_device: usize,
    root_device: usize,
) -> Result<usize, String> {
    let values = tokens
        .checked_mul(width)
        .ok_or("replicated device row size overflow")?;
    if tokens == 0 || width == 0 || source_len != values || source_device != root_device {
        return Err(format!(
            "replicated device row source has inconsistent geometry/device \
             tokens={tokens} width={width} source={source_len}@{source_device} root={root_device}"
        ));
    }
    Ok(values)
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn bf16_column_shard(
    matrix: Bf16Matrix<'_>,
    tp: usize,
    rank: usize,
) -> Result<Bf16Matrix<'_>, String> {
    matrix.validate()?;
    if tp == 0 || rank >= tp || matrix.out_features % tp != 0 {
        return Err(format!(
            "invalid BF16 column shard out={} TP={tp} rank={rank}",
            matrix.out_features
        ));
    }
    let local_out = matrix.out_features / tp;
    let row_bytes = matrix.in_features * 2;
    let start = rank * local_out * row_bytes;
    Ok(Bf16Matrix {
        bytes: &matrix.bytes[start..start + local_out * row_bytes],
        out_features: local_out,
        in_features: matrix.in_features,
    })
}

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn bf16_row_shard(matrix: Bf16Matrix<'_>, tp: usize, rank: usize) -> Result<Vec<u8>, String> {
    matrix.validate()?;
    if tp == 0 || rank >= tp || matrix.in_features % tp != 0 {
        return Err(format!(
            "invalid BF16 row shard in={} TP={tp} rank={rank}",
            matrix.in_features
        ));
    }
    let local_in = matrix.in_features / tp;
    let mut bytes = Vec::with_capacity(matrix.out_features * local_in * 2);
    for row in 0..matrix.out_features {
        let start = (row * matrix.in_features + rank * local_in) * 2;
        bytes.extend_from_slice(&matrix.bytes[start..start + local_in * 2]);
    }
    Ok(bytes)
}

fn bf16_row_block(
    matrix: Bf16Matrix<'_>,
    col_start: usize,
    block_cols: usize,
) -> Result<Vec<u8>, String> {
    matrix.validate()?;
    let col_end = col_start
        .checked_add(block_cols)
        .ok_or("BF16 row block column overflow")?;
    if block_cols == 0 || col_end > matrix.in_features {
        return Err(format!(
            "invalid BF16 row block columns {col_start}..{col_end} for input width {}",
            matrix.in_features
        ));
    }
    let mut bytes = Vec::with_capacity(matrix.out_features * block_cols * 2);
    for row in 0..matrix.out_features {
        let start = (row * matrix.in_features + col_start) * 2;
        bytes.extend_from_slice(&matrix.bytes[start..start + block_cols * 2]);
    }
    Ok(bytes)
}

fn run_resident_bank_expert(
    engine: &Engine,
    bank: &ResidentE4m3ExpertBankRank,
    local_expert: usize,
    activations: &[f32],
    tokens: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let _main = engine.gpu.enter_main()?;
    if bank.k_blocks.is_some() {
        return Err("block-major TP row bank requires canonical block execution".into());
    }
    let local_count = bank.expert_range.end - bank.expert_range.start;
    if local_expert >= local_count {
        return Err(format!(
            "local EP expert {local_expert} outside 0..{local_count} for range {:?}",
            bank.expert_range
        )
        .into());
    }
    validate_activations(activations, tokens, bank.in_features)?;
    let activations = engine.htod(activations)?;
    let weight = bank
        .codes
        .slice(local_expert * bank.code_stride..(local_expert + 1) * bank.code_stride);
    let scales = bank
        .scales
        .slice(local_expert * bank.scale_stride..(local_expert + 1) * bank.scale_stride);
    let input = activations.slice(0..activations.len());
    let output = engine.qmatvec_mmq_fp8_blk_view(
        &weight,
        &scales,
        &input,
        tokens,
        bank.in_features,
        bank.out_features,
    )?;
    engine.dtoh(&output)
}

fn run_resident_bank_expert_block(
    engine: &Engine,
    bank: &ResidentE4m3ExpertBankRank,
    local_expert: usize,
    block: usize,
    activations: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let _main = engine.gpu.enter_main()?;
    let local_count = bank.expert_range.end - bank.expert_range.start;
    if local_expert >= local_count {
        return Err(format!(
            "local TP expert {local_expert} outside 0..{local_count} for range {:?}",
            bank.expert_range
        )
        .into());
    }
    let blocks = bank
        .k_blocks
        .ok_or("TP row bank is not packed in native K-block order")?;
    if block >= blocks {
        return Err(format!("TP row block {block} outside 0..{blocks}").into());
    }
    validate_activations(activations, 1, FP8_BLOCK)?;
    let block_code_stride = bank.out_features * FP8_BLOCK;
    let block_scale_stride = bank.out_features.div_ceil(FP8_BLOCK);
    if bank.in_features != blocks * FP8_BLOCK
        || bank.code_stride != blocks * block_code_stride
        || bank.scale_stride != blocks * block_scale_stride
    {
        return Err("TP row bank block-major geometry is inconsistent".into());
    }

    let expert_code_start = local_expert * bank.code_stride;
    let expert_scale_start = local_expert * bank.scale_stride;
    let weight = bank.codes.slice(
        expert_code_start + block * block_code_stride
            ..expert_code_start + (block + 1) * block_code_stride,
    );
    let scales = bank.scales.slice(
        expert_scale_start + block * block_scale_stride
            ..expert_scale_start + (block + 1) * block_scale_stride,
    );
    let activations = engine.htod(activations)?;
    let input = activations.slice(0..activations.len());
    let output = engine.qmatvec_mmq_fp8_blk_view(
        &weight,
        &scales,
        &input,
        1,
        FP8_BLOCK,
        bank.out_features,
    )?;
    engine.dtoh(&output)
}

fn run_resident_bank_expert_device(
    engine: &Engine,
    bank: &ResidentE4m3ExpertBankRank,
    local_expert: usize,
    activations: &CudaSlice<f32>,
    tokens: usize,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let _main = engine.gpu.enter_main()?;
    if bank.k_blocks.is_some() {
        return Err("block-major TP row bank requires canonical block execution".into());
    }
    let local_count = bank.expert_range.end - bank.expert_range.start;
    if local_expert >= local_count {
        return Err(format!(
            "local TP expert {local_expert} outside 0..{local_count} for range {:?}",
            bank.expert_range
        )
        .into());
    }
    let expected = tokens
        .checked_mul(bank.in_features)
        .ok_or("native TP activation size overflow")?;
    if activations.len() != expected || activations.ordinal() != engine.ctx().ordinal() {
        return Err(format!(
            "native TP activation len/device {}/{} != expected {expected}/{}",
            activations.len(),
            activations.ordinal(),
            engine.ctx().ordinal()
        )
        .into());
    }
    let weight = bank
        .codes
        .slice(local_expert * bank.code_stride..(local_expert + 1) * bank.code_stride);
    let scales = bank
        .scales
        .slice(local_expert * bank.scale_stride..(local_expert + 1) * bank.scale_stride);
    let input = activations.slice(0..activations.len());
    engine.qmatvec_mmq_fp8_blk_view(
        &weight,
        &scales,
        &input,
        tokens,
        bank.in_features,
        bank.out_features,
    )
}

fn run_resident_bank_expert_block_device(
    engine: &Engine,
    bank: &ResidentE4m3ExpertBankRank,
    local_expert: usize,
    block: usize,
    activations: &cudarc::driver::CudaView<'_, f32>,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let _main = engine.gpu.enter_main()?;
    let local_count = bank.expert_range.end - bank.expert_range.start;
    if local_expert >= local_count {
        return Err(format!(
            "local TP expert {local_expert} outside 0..{local_count} for range {:?}",
            bank.expert_range
        )
        .into());
    }
    let blocks = bank
        .k_blocks
        .ok_or("native TP row bank is not packed in checkpoint-block order")?;
    if block >= blocks {
        return Err(format!("native TP row block {block} outside 0..{blocks}").into());
    }
    let activation_device = activations.stream().context().ordinal();
    if activations.len() != FP8_BLOCK || activation_device != engine.ctx().ordinal() {
        return Err(format!(
            "native TP block activation len/device {}/{} != expected {FP8_BLOCK}/{}",
            activations.len(),
            activation_device,
            engine.ctx().ordinal()
        )
        .into());
    }
    let block_code_stride = bank.out_features * FP8_BLOCK;
    let block_scale_stride = bank.out_features.div_ceil(FP8_BLOCK);
    if bank.in_features != blocks * FP8_BLOCK
        || bank.code_stride != blocks * block_code_stride
        || bank.scale_stride != blocks * block_scale_stride
    {
        return Err("native TP row bank block-major geometry is inconsistent".into());
    }
    let expert_code_start = local_expert * bank.code_stride;
    let expert_scale_start = local_expert * bank.scale_stride;
    let weight = bank.codes.slice(
        expert_code_start + block * block_code_stride
            ..expert_code_start + (block + 1) * block_code_stride,
    );
    let scales = bank.scales.slice(
        expert_scale_start + block * block_scale_stride
            ..expert_scale_start + (block + 1) * block_scale_stride,
    );
    engine.qmatvec_mmq_fp8_blk_view(
        &weight,
        &scales,
        activations,
        1,
        FP8_BLOCK,
        bank.out_features,
    )
}

/// Grant `accessor` the right to reach `owner`'s memory — BOTH halves of the grant, which is
/// the part every caller gets wrong exactly once:
///
///   1. `cuCtxEnablePeerAccess`, which covers legacy `cuMemAlloc` allocations, and
///   2. `cuMemPoolSetAccess` on `owner`'s DEFAULT MEMORY POOL, because
///      `cuCtxEnablePeerAccess` does NOT map STREAM-ORDERED POOL allocations and every
///      normal memra buffer is one (the same note `pp.rs:1543`/`pp.rs:1578` carries).
///
/// Extracted from [`configure_native_p2p`] (which now calls it per ordered pair) so a seam
/// holding two `&Engine` rather than a `&[Engine]` — the glm5 TP-2 runtime — reuses the exact
/// grant sequence instead of growing a second, drifting copy of it. Directed: call it once
/// per direction. Refuses by name when `cuDeviceCanAccessPeer` says the pair has no path,
/// which is the only honest answer: this card class is NOT uniformly peer-connected. Some
/// 8-GPU host classes present PEER ISLANDS OF TWO — every cross-island cell of a peer-transfer
/// matrix reads `N/A` — so a TP group placed across an island boundary has no peer path at all
/// and must either stay inside one island or go through host memory. The per-host island map is
/// fleet data and lives in the private deployment repo, never here; the engine's job is to
/// refuse by name rather than to know which host it is on.
/// Made `pub` for `tp_ar`'s bench and gate, which must arm the same peer access the walk uses
/// rather than a second path that could differ from it.
pub fn grant_peer_access(
    accessor: &Engine,
    owner: &Engine,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (a_dev, o_dev) = (accessor.ctx().ordinal(), owner.ctx().ordinal());
    let mut can_access = 0;
    unsafe {
        cudarc::driver::sys::cuDeviceCanAccessPeer(
            &mut can_access,
            accessor.ctx().cu_device(),
            owner.ctx().cu_device(),
        )
        .result()?;
    }
    if can_access == 0 {
        return Err(
            format!("{label} requires P2P, but dev{a_dev} cannot access dev{o_dev}").into(),
        );
    }
    accessor.ctx().bind_to_thread()?;
    let rc = unsafe { cudarc::driver::sys::cuCtxEnablePeerAccess(owner.ctx().cu_ctx(), 0) };
    use cudarc::driver::sys::cudaError_enum as E;
    if rc != E::CUDA_SUCCESS && rc != E::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
        return Err(format!(
            "{label} cuCtxEnablePeerAccess(dev{a_dev} -> dev{o_dev}) failed: {rc:?}"
        )
        .into());
    }
    let device = cudarc::driver::result::device::get(o_dev as i32)?;
    let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
    unsafe {
        cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, device).result()?;
    }
    let desc = cudarc::driver::sys::CUmemAccessDesc {
        location: cudarc::driver::sys::CUmemLocation {
            type_: cudarc::driver::sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: a_dev as i32,
        },
        flags: cudarc::driver::sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };
    let rc = unsafe { cudarc::driver::sys::cuMemPoolSetAccess(pool, &desc, 1) };
    if rc != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
        return Err(format!(
            "{label} cuMemPoolSetAccess(dev{o_dev} pool -> dev{a_dev}) failed: {rc:?}"
        )
        .into());
    }
    Ok(())
}

fn configure_native_p2p(
    ranks: &[Engine],
    devices: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    if ranks.len() != devices.len() || ranks.len() < 2 {
        return Err("native TP P2P setup requires matching multi-rank devices".into());
    }
    for (rank, (&device, engine)) in devices.iter().zip(ranks).enumerate() {
        if engine.ctx().ordinal() != device {
            return Err(format!(
                "native TP rank {rank} context device {} != requested device {device}",
                engine.ctx().ordinal()
            )
            .into());
        }
    }

    for src in 0..ranks.len() {
        for dst in 0..ranks.len() {
            if src == dst {
                continue;
            }
            grant_peer_access(&ranks[src], &ranks[dst], "native TP")?;
        }
    }

    for src in 0..ranks.len() {
        for dst in 0..ranks.len() {
            if src == dst {
                continue;
            }
            for &words in NATIVE_P2P_PROBE_WORDS {
                let expected = (0..words)
                    .map(|index| {
                        (index as u32)
                            .wrapping_mul(0x9e37_79b9)
                            .wrapping_add(((src as u32) << 16) | dst as u32)
                    })
                    .collect::<Vec<_>>();
                let poison = expected.iter().map(|value| !value).collect::<Vec<_>>();
                let source = ranks[src].htod_u32_v(&expected)?;
                let mut destination = ranks[dst].htod_u32_v(&poison)?;
                ranks[dst].stream().memcpy_dtod(&source, &mut destination)?;
                let actual = ranks[dst].dtoh_u32(&destination)?;
                if actual != expected {
                    let mismatches = actual
                        .iter()
                        .zip(&expected)
                        .filter(|(actual, expected)| actual != expected)
                        .count();
                    return Err(format!(
                        "native TP peer probe dev{}->dev{} failed at {} bytes: \
                         {mismatches}/{} words differ",
                        devices[src],
                        devices[dst],
                        words * std::mem::size_of::<u32>(),
                        expected.len()
                    )
                    .into());
                }
            }
        }
    }
    ranks[0].ctx().bind_to_thread()?;
    eprintln!(
        "[tp] native peer byte-integrity probe PASS: devices={devices:?} \
         directions={} byte_ladder={:?} mismatches=0",
        ranks.len() * (ranks.len() - 1),
        NATIVE_P2P_PROBE_WORDS
            .iter()
            .map(|words| words * std::mem::size_of::<u32>())
            .collect::<Vec<_>>(),
    );
    Ok(())
}

fn validate_activations(
    activations: &[f32],
    tokens: usize,
    in_features: usize,
) -> Result<(), String> {
    let expected = tokens
        .checked_mul(in_features)
        .ok_or_else(|| "activation size overflow".to_string())?;
    if activations.len() != expected {
        return Err(format!(
            "activation count {} != {tokens}x{in_features} ({expected})",
            activations.len()
        ));
    }
    if !activations.iter().all(|value| value.is_finite()) {
        return Err("activations contain a non-finite value".to_string());
    }
    Ok(())
}

fn column_shard(
    matrix: E4m3BlockMatrix<'_>,
    tp: usize,
    rank: usize,
) -> Result<E4m3BlockMatrix<'_>, String> {
    let local_out = matrix.out_features / tp;
    let row_start = rank * local_out;
    let code_start = row_start * matrix.in_features;
    let code_end = code_start + local_out * matrix.in_features;
    let scale_cols = matrix.in_features.div_ceil(FP8_BLOCK);
    let local_scale_rows = local_out / FP8_BLOCK;
    let scale_start = rank * local_scale_rows * scale_cols;
    let scale_end = scale_start + local_scale_rows * scale_cols;
    Ok(E4m3BlockMatrix {
        codes: &matrix.codes[code_start..code_end],
        scales: &matrix.scales[scale_start..scale_end],
        out_features: local_out,
        in_features: matrix.in_features,
    })
}

fn row_shard(
    matrix: E4m3BlockMatrix<'_>,
    tp: usize,
    rank: usize,
) -> Result<(Vec<u8>, Vec<f32>), String> {
    let local_in = matrix.in_features / tp;
    let col_start = rank * local_in;
    let mut codes = Vec::with_capacity(matrix.out_features * local_in);
    for row in 0..matrix.out_features {
        let start = row * matrix.in_features + col_start;
        codes.extend_from_slice(&matrix.codes[start..start + local_in]);
    }

    let scale_rows = matrix.out_features.div_ceil(FP8_BLOCK);
    let scale_cols = matrix.in_features.div_ceil(FP8_BLOCK);
    let local_scale_cols = local_in / FP8_BLOCK;
    let scale_col_start = rank * local_scale_cols;
    let mut scales = Vec::with_capacity(scale_rows * local_scale_cols);
    for row in 0..scale_rows {
        let start = row * scale_cols + scale_col_start;
        scales.extend_from_slice(&matrix.scales[start..start + local_scale_cols]);
    }
    Ok((codes, scales))
}

fn activation_shard(
    activations: &[f32],
    tokens: usize,
    in_features: usize,
    tp: usize,
    rank: usize,
) -> Vec<f32> {
    let local_in = in_features / tp;
    let col_start = rank * local_in;
    let mut shard = Vec::with_capacity(tokens * local_in);
    for token in 0..tokens {
        let start = token * in_features + col_start;
        shard.extend_from_slice(&activations[start..start + local_in]);
    }
    shard
}

// ─── Step NVFP4 expert TP program (official Step-3.7-Flash-NVFP4 checkpoint class) ─────────────
//
// The routed experts of the NVFP4 checkpoint are modelopt-packed: e2m1 codes (2/byte), per-16
// UE4M3 sub-scales, and a per-EXPERT `weight_scale_2` f32 macro (~1e-5..1e-4, LOAD-BEARING).
// Rank compute repacks each shard host-side into memra block_nvfp4 rows (nibble reorder only —
// value-exact, see nvfp4_repack.rs) and runs the proven `qmatvec_nvfp4_fast` dp4a kernel; the
// activation q8_1 quantization uses per-32 blocks, and every shard cut here is 64-aligned, so a
// rank-local partial is bit-identical to the corresponding slice of the unsharded kernel.
//
// MACRO CANONICAL ORDER: the macro multiplies each assembled f32 output exactly ONCE — after the
// column gather (gate/up) and after the FULL row-parallel reduce (down), never per-partial.
// `(a + b) * m` and `a * m + b * m` differ in f32, so applying it per-rank would break the
// TP1-vs-TP2 bit gate. Every entry point below follows this order.
//
// TP2 shard legality is NVFP4-native: column parallelism splits whole output rows (scale rows
// ride along, nothing cuts), row parallelism splits input columns at 64-element superblock
// boundaries (16-element scale groups nest inside). The 128-block E4M3 constraint does not apply.

/// One expert's modelopt NVFP4 projection: packed codes + per-16 UE4M3 scale bytes + macro.
#[derive(Clone, Copy)]
pub struct Nvfp4BlockMatrix<'a> {
    pub codes: &'a [u8],  // [out_features, in_features/2] packed e2m1, row-major
    pub scales: &'a [u8], // [out_features, in_features/16] UE4M3 bytes, row-major
    pub macro_scale: f32, // per-expert weight_scale_2 dequant multiplier
    pub out_features: usize,
    pub in_features: usize,
}

impl Nvfp4BlockMatrix<'_> {
    pub fn validate(&self) -> Result<(), String> {
        if self.in_features == 0 || self.out_features == 0 {
            return Err("NVFP4 matrix has a zero dimension".to_string());
        }
        if !self.in_features.is_multiple_of(64) {
            return Err(format!(
                "NVFP4 in_features {} is not 64-aligned (memra block_nvfp4 superblock)",
                self.in_features
            ));
        }
        if self.codes.len() != self.out_features * self.in_features / 2 {
            return Err(format!(
                "NVFP4 code bytes {} != {}x{}/2",
                self.codes.len(),
                self.out_features,
                self.in_features
            ));
        }
        if self.scales.len() != self.out_features * self.in_features / 16 {
            return Err(format!(
                "NVFP4 scale bytes {} != {}x{}/16",
                self.scales.len(),
                self.out_features,
                self.in_features
            ));
        }
        if !self.macro_scale.is_finite() || self.macro_scale <= 0.0 {
            return Err(format!(
                "NVFP4 macro scale {} is not finite-positive",
                self.macro_scale
            ));
        }
        Ok(())
    }
}

/// Stacked modelopt NVFP4 expert bank (host view over the checkpoint bytes).
#[derive(Clone, Copy)]
pub struct Nvfp4ExpertBank<'a> {
    pub codes: &'a [u8],   // [expert_count, out_features, in_features/2]
    pub scales: &'a [u8],  // [expert_count, out_features, in_features/16]
    pub macros: &'a [f32], // [expert_count] weight_scale_2
    pub expert_count: usize,
    pub out_features: usize,
    pub in_features: usize,
}

impl Nvfp4ExpertBank<'_> {
    pub fn validate(&self) -> Result<(), String> {
        if self.expert_count == 0 {
            return Err("NVFP4 expert bank is empty".to_string());
        }
        if self.macros.len() != self.expert_count {
            return Err(format!(
                "NVFP4 bank macros {} != expert count {}",
                self.macros.len(),
                self.expert_count
            ));
        }
        self.expert(0).map(|_| ())
    }

    pub fn expert(&self, expert: usize) -> Result<Nvfp4BlockMatrix<'_>, String> {
        if expert >= self.expert_count {
            return Err(format!("expert {expert} outside 0..{}", self.expert_count));
        }
        let code_stride = self.out_features * self.in_features / 2;
        let scale_stride = self.out_features * self.in_features / 16;
        if self.codes.len() != self.expert_count * code_stride
            || self.scales.len() != self.expert_count * scale_stride
        {
            return Err("NVFP4 bank byte extents do not match the declared geometry".to_string());
        }
        let matrix = Nvfp4BlockMatrix {
            codes: &self.codes[expert * code_stride..(expert + 1) * code_stride],
            scales: &self.scales[expert * scale_stride..(expert + 1) * scale_stride],
            macro_scale: self.macros[expert],
            out_features: self.out_features,
            in_features: self.in_features,
        };
        matrix.validate()?;
        Ok(matrix)
    }
}

/// One rank's resident repacked NVFP4 shard: memra block_nvfp4 rows on device.
pub struct ResidentNvfp4Rank {
    blocks: crate::CudaSlice<u8>,
    macro_scale: f32,
    out_features: usize,
    in_features: usize,
    row_bytes: usize,
}

pub struct ResidentNvfp4ColumnParallel {
    ranks: Vec<ResidentNvfp4Rank>,
    pub out_features: usize,
    pub in_features: usize,
}

pub struct ResidentNvfp4RowParallel {
    ranks: Vec<ResidentNvfp4Rank>,
    pub out_features: usize,
    pub in_features: usize,
}

pub struct ResidentTpNvfp4Expert {
    gate: ResidentNvfp4ColumnParallel,
    up: ResidentNvfp4ColumnParallel,
    down: ResidentNvfp4RowParallel,
    pub input_width: usize,
    pub expert_width: usize,
}

/// One rank's resident NVFP4 expert bank shard: one repacked block buffer PER expert (per-expert
/// device allocations keep this increment off any new strided-kernel API; the strided twin is a
/// later perf rung, mirroring the FP8 bank's history).
pub struct ResidentNvfp4ColumnBankRank {
    /// Contiguous per-rank expert bank: `expert_count` repacked shards of `expert_bytes` each.
    /// Contiguity is what lets the device-routes program cover every selected expert with ONE
    /// launch (`qmatvec_nvfp4_dp4a_sel` indexes `sel[t] * expert_bytes`).
    bank: crate::CudaSlice<u8>,
    expert_bytes: usize,
    local_out: usize,
    in_features: usize,
    row_bytes: usize,
    /// TRUE when these bytes are the slot-major permutation (`nvfp4_matrix_v2_permute`) and the
    /// `_v2` readers must be used; FALSE when they are block_nvfp4 v1. Recorded at BUILD from
    /// `bank_slot_major_on()` and never re-derived: the layout travels with the pointer,
    /// so no reader can consult an env door that disagrees with the resident bytes. Feeding v1
    /// bytes to a `_v2` reader (or the reverse) is a garbage-output bug, and the 2026-08-29
    /// step37 incident was its neighbour — a piece of layout geometry a caller failed to supply.
    slot_major: bool,
}

impl ResidentNvfp4ColumnBankRank {
    /// THE host-canonical reader for this bank, selected from the layout the bank RECORDS. One
    /// place maps layout -> reader for the column banks; every oracle goes through it, so a new
    /// producer cannot leave a reader behind (the failure mode that put v1 bytes under a `_v2`
    /// reader, called out in the `run_tensor_parallel_routes_nvfp4_prime_grouped` receipt).
    fn host_canonical_expert(
        &self,
        engine: &Engine,
        expert: usize,
        activations: &crate::CudaSlice<f32>,
    ) -> Result<crate::CudaSlice<f32>, Box<dyn std::error::Error>> {
        let w = self.expert(expert);
        if self.slot_major {
            engine.qmatvec_nvfp4_fast_v2(
                &w,
                activations,
                1,
                self.in_features,
                self.local_out,
                self.row_bytes,
            )
        } else {
            engine.qmatvec_nvfp4_fast(
                &w,
                activations,
                1,
                self.in_features,
                self.local_out,
                self.row_bytes,
            )
        }
    }

    fn expert(&self, index: usize) -> cudarc::driver::CudaView<'_, u8> {
        self.bank
            .slice(index * self.expert_bytes..(index + 1) * self.expert_bytes)
    }
}

/// Canonical row-shard count for the NVFP4 down projection. The down reduction ALWAYS executes
/// as exactly this many input-column windows summed in shard order, at every world size: a
/// single full-width dot and a two-half-dots-plus-add differ in f32 parenthesization, so pinning
/// the shard grid (not the world size) is what makes the TP1-oracle-vs-TP2 bit gate meaningful.
/// This is the NVFP4 twin of the FP8 bank's canonical checkpoint-block reduction.
pub const NVFP4_CANONICAL_ROW_SHARDS: usize = 2;

pub struct ResidentNvfp4RowBankRank {
    /// Contiguous per-shard expert bank (see `ResidentNvfp4ColumnBankRank::bank`).
    bank: crate::CudaSlice<u8>,
    expert_bytes: usize,
    device_rank: usize, // index into the runtime's rank engines this canonical shard lives on
    out_features: usize,
    local_in: usize,
    row_bytes: usize,
    /// Slot-major layout marker — see `ResidentNvfp4ColumnBankRank::slot_major`.
    slot_major: bool,
}

impl ResidentNvfp4RowBankRank {
    /// THE host-canonical reader for this down shard — see
    /// `ResidentNvfp4ColumnBankRank::host_canonical_expert`.
    fn host_canonical_expert(
        &self,
        engine: &Engine,
        expert: usize,
        activations: &crate::CudaSlice<f32>,
    ) -> Result<crate::CudaSlice<f32>, Box<dyn std::error::Error>> {
        let w = self.expert(expert);
        if self.slot_major {
            engine.qmatvec_nvfp4_fast_v2(
                &w,
                activations,
                1,
                self.local_in,
                self.out_features,
                self.row_bytes,
            )
        } else {
            engine.qmatvec_nvfp4_fast(
                &w,
                activations,
                1,
                self.local_in,
                self.out_features,
                self.row_bytes,
            )
        }
    }

    fn expert(&self, index: usize) -> cudarc::driver::CudaView<'_, u8> {
        self.bank
            .slice(index * self.expert_bytes..(index + 1) * self.expert_bytes)
    }
}

impl ResidentNvfp4TensorParallel {
    pub(crate) fn device_workspace_handle(
        &self,
    ) -> &std::sync::Mutex<Option<Nvfp4DeviceRoutesWorkspace>> {
        &self.device_workspace
    }
}

pub struct ResidentNvfp4TensorParallel {
    gate: Vec<ResidentNvfp4ColumnBankRank>,
    up: Vec<ResidentNvfp4ColumnBankRank>,
    down: Vec<ResidentNvfp4RowBankRank>,
    macros_gate: Vec<f32>,
    macros_up: Vec<f32>,
    macros_down: Vec<f32>,
    /// Per-rank device copies of the gate/up macro-scales (E f32 each), indexed by the
    /// batched SwiGLU kernel via the selection array. Down macros stay host-side — they fold
    /// into the route-weight axpy scalar.
    macros_gate_dev: Vec<crate::CudaSlice<f32>>,
    macros_up_dev: Vec<crate::CudaSlice<f32>>,
    macros_down_dev: Vec<crate::CudaSlice<f32>>,
    pub expert_count: usize,
    pub input_width: usize,
    pub expert_width: usize,
    /// Lazily-built persistent decode workspace (device routes program). Interior mutability
    /// mirrors StepEpGroupedDecode: the forward holds the bank behind a shared reference.
    device_workspace: std::sync::Mutex<Option<Nvfp4DeviceRoutesWorkspace>>,
    /// Grouped-prime per-rank slot-major pointer tables (gate/up/down x n_expert), built once.
    /// The banks are resident and never move, so rebuilding + re-uploading 3*n_expert u64s per
    /// rank per LAYER was pure per-call host churn on the prime path.
    prime_tables: std::sync::Mutex<Vec<crate::CudaSlice<u64>>>,
}

/// Persistent per-call device buffers for the NVFP4 device routes program: one gate/up output,
/// one down partial, and one shard accumulator per rank, plus root combine staging. Reused every
/// (token, layer) call so the decode loop performs zero output allocations.
/// A stitched multi-device parent graph for one layer's device-routed expert program, plus
/// the children it was built from (retained: AddChildGraphNode clones, but the probe retains
/// conservatively) and the persistent e-context input staging its copies read.
struct RoutesGraph {
    exec: cudarc::driver::sys::CUgraphExec,
    parent: cudarc::driver::sys::CUgraph,
    _children: Vec<cudarc::driver::CudaGraph>,
}
// SAFETY: the raw handles are only used from the single decode thread; CUDA graph handles are
// context-agnostic process handles.
unsafe impl Send for RoutesGraph {}

impl Drop for RoutesGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::sys::cuGraphExecDestroy(self.exec);
            let _ = cudarc::driver::sys::cuGraphDestroy(self.parent);
        }
    }
}

impl Nvfp4DeviceRoutesWorkspace {
    pub(crate) fn in_stage_handle(&self) -> Option<&crate::CudaSlice<f32>> {
        self.in_stage_e.as_ref()
    }
    pub(crate) fn in_stage_mut(&mut self) -> Option<&mut crate::CudaSlice<f32>> {
        self.in_stage_e.as_mut()
    }
    #[allow(dead_code)] // allow: accessor twin of in_stage_mut; kept for the workspace API symmetry
    pub(crate) fn out_stage_mut(&mut self) -> Option<&mut crate::CudaSlice<f32>> {
        self.out_stage_e.as_mut()
    }
    /// Arm the e-context stages + router staging pair when absent (token-graph entry).
    pub(crate) fn arm_stages(
        &mut self,
        e: &Engine,
        width: usize,
        n_sel: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let _main = e.gpu.enter_main()?;
        if self.in_stage_e.is_none() {
            self.in_stage_e = Some(e.htod(&vec![0.0f32; width])?);
            self.out_stage_e = Some(e.htod(&vec![0.0f32; width])?);
        }
        if self.dev_route_e.is_none() {
            self.dev_route_e = Some((
                e.htod_i32(&vec![0i32; n_sel])?,
                e.htod(&vec![0.0f32; n_sel])?,
            ));
        }
        Ok(())
    }

    /// Split-borrow: the routes input (shared) + output (mut) stages together.
    pub(crate) fn in_and_out_stages_mut(
        &mut self,
    ) -> Option<(&crate::CudaSlice<f32>, &mut crate::CudaSlice<f32>)> {
        match (self.in_stage_e.as_ref(), self.out_stage_e.as_mut()) {
            (Some(input), Some(output)) => Some((input, output)),
            _ => None,
        }
    }
    pub(crate) fn dev_route_e_mut(
        &mut self,
    ) -> Option<(&mut crate::CudaSlice<i32>, &mut crate::CudaSlice<f32>)> {
        self.dev_route_e.as_mut().map(|(a, b)| (a, b))
    }
}

pub struct Nvfp4DeviceRoutesWorkspace {
    /// [n_sel, local_out] batched gate/up outputs and the SwiGLU q8_1 pair; [n_sel, width]
    /// down partials. Sized for `n_sel` selected experts per token (pinned at first call).
    gate_out: Vec<crate::CudaSlice<f32>>,
    up_out: Vec<crate::CudaSlice<f32>>,
    act_q: Vec<crate::CudaSlice<i8>>,
    act_d: Vec<crate::CudaSlice<f32>>,
    sel: Vec<crate::CudaSlice<i32>>,
    partial: Vec<crate::CudaSlice<f32>>,
    accumulator: Vec<crate::CudaSlice<f32>>,
    /// Per-rank folded combine weights (route_weight x down macro), one htod per call.
    combine_w: Vec<crate::CudaSlice<f32>>,
    /// Device-routed extension: per-rank raw route weights (the down-macro fold happens
    /// in-kernel via sel + macros_down_dev).
    route_w: Vec<crate::CudaSlice<f32>>,
    /// Persistent q8_1 pair of the shared layer input (one quantize per rank per call, no
    /// per-call allocation).
    in_q: Vec<crate::CudaSlice<i8>>,
    in_d: Vec<crate::CudaSlice<f32>>,
    /// e-context staging for the device router outputs (persistent — rank streams peer-read
    /// them, so the router's fresh outputs are copied here on e's stream first; the pp.rs
    /// never-free discipline).
    dev_route_e: Option<(crate::CudaSlice<i32>, crate::CudaSlice<f32>)>,
    /// Prestage door state: input pull + quantize already issued for this layer's call
    /// (nvfp4_routes_prestage), so the routed run skips them. Reset per call.
    prestaged: bool,
    /// Peer-router door state: rank1's sel/route_w were computed locally in prestage;
    /// the routed run skips rank1's sel pull. Reset per call.
    rank1_routed: bool,
    /// Doorbell fences (MEMRA_FENCE_MEMOPS): raw cuMemAlloc'd [rank1_flag, root_flag]
    /// u32 pair in ROOT memory (async-pool memory is memop-INELIGIBLE — receipted
    /// CUDA_ERROR_INVALID_VALUE) + the host-side monotonic ticket. 0 = unarmed.
    fence_flags_raw: u64,
    fence_ticket: u32,
    /// Prestage input fence, recorded on e after the input's producer.
    ev_input: Option<(CudaEvent, usize)>,
    /// Graph-door staging: persistent e-context input row + output row (fixed addresses the
    /// captured copies read/write), and the per-layer stitched parent.
    in_stage_e: Option<crate::CudaSlice<f32>>,
    out_stage_e: Option<crate::CudaSlice<f32>>,
    routes_graph: Option<RoutesGraph>,
    /// Token-graph raw pointer sets (armed once by routes_arm_raw).
    raw_dev_route_e: Option<(u64, u64)>,
    raw_combine: Option<(u64, u64, u64, u64)>,
    raw_input: Vec<u64>,
    raw_sel: Vec<u64>,
    raw_route_w: Vec<u64>,
    remote: crate::CudaSlice<f32>,
    combined: crate::CudaSlice<f32>,
    n_sel: usize,
    /// Device-IO extension (lazily built by `run_tensor_parallel_routes_nvfp4_device_io`):
    /// persistent per-rank input rows plus the evented ordering pair — the pp.rs
    /// BoundarySlot discipline, same as the v2 attention workspace.
    input: Vec<crate::CudaSlice<f32>>,
    ev_rank: Vec<CudaEvent>,
    ev_done: Option<CudaEvent>,
    ev_entry: Option<(CudaEvent, usize)>,
}

/// One rank's whole-expert NVFP4 residency (expert-parallel ownership).
struct ResidentNvfp4EpRank {
    gate: crate::CudaSlice<u8>,
    up: crate::CudaSlice<u8>,
    down: crate::CudaSlice<u8>,
    gate_expert_bytes: usize,
    down_expert_bytes: usize,
    macros_gate: crate::CudaSlice<f32>,
    macros_up: crate::CudaSlice<f32>,
    macros_down: crate::CudaSlice<f32>,
    expert_range: Range<usize>,
}

struct Nvfp4EpDeviceWorkspace {
    input: Vec<crate::CudaSlice<f32>>,
    input_bf16: Vec<crate::CudaSlice<u8>>,
    input_q8: Vec<crate::CudaSlice<i8>>,
    input_q8_scales: Vec<crate::CudaSlice<f32>>,
    sel: Vec<crate::CudaSlice<i32>>,
    token_rows: Vec<crate::CudaSlice<i32>>,
    global_pairs: Vec<crate::CudaSlice<i32>>,
    route_w: Vec<crate::CudaSlice<f32>>,
    gate_out: Vec<crate::CudaSlice<f32>>,
    up_out: Vec<crate::CudaSlice<f32>>,
    activation_bf16: Vec<crate::CudaSlice<u8>>,
    activation_q8: Vec<crate::CudaSlice<i8>>,
    activation_q8_scales: Vec<crate::CudaSlice<f32>>,
    slot_rows: crate::CudaSlice<f32>,
    slot_rows_raw: u64,
    route_weights: crate::CudaSlice<f32>,
    ev_entry: CudaEvent,
    ev_entry_device: usize,
    ev_rank: Vec<CudaEvent>,
    phase_events: Option<Nvfp4EpPhaseEvents>,
    capacity_tokens: usize,
    experts_per_token: usize,
}

struct Nvfp4EpPhaseEvents {
    head: Vec<CudaEvent>,
    copy_done: Vec<CudaEvent>,
    gate_up_done: Vec<CudaEvent>,
    activation_done: Vec<CudaEvent>,
    down_done: Vec<CudaEvent>,
}

pub(crate) const NVFP4_EP_DEVICE_BATCH_CAP: usize = 128;
pub(crate) const NVFP4_EP_DEVICE_ROUTER_BATCH_CAP: usize = 32;
pub(crate) const NVFP4_EP_Q8_BATCH_CAP: usize = 32;

fn nvfp4_ep_active_input_values(
    input_values: usize,
    tokens: usize,
    input_width: usize,
) -> Result<usize, String> {
    if !(1..=NVFP4_EP_DEVICE_BATCH_CAP).contains(&tokens) {
        return Err(format!(
            "W4A16 NVFP4 device EP batch {tokens} is outside 1..={NVFP4_EP_DEVICE_BATCH_CAP}"
        ));
    }
    let active_values = tokens
        .checked_mul(input_width)
        .ok_or("W4A16 NVFP4 device EP active input size overflows usize")?;
    if input_values < active_values {
        return Err(format!(
            "W4A16 NVFP4 device EP input {input_values} is smaller than active \
             tokens {tokens} x width {input_width} ({active_values})"
        ));
    }
    Ok(active_values)
}

pub struct ResidentNvfp4ExpertParallel {
    ranks: Vec<ResidentNvfp4EpRank>,
    macros_gate: Vec<f32>,
    macros_up: Vec<f32>,
    macros_down: Vec<f32>,
    pub expert_count: usize,
    pub input_width: usize,
    pub expert_width: usize,
    gate_row_bytes: usize,
    down_row_bytes: usize,
    device_workspace: std::sync::Mutex<Option<Nvfp4EpDeviceWorkspace>>,
}

fn nvfp4_repack_matrix(matrix: Nvfp4BlockMatrix<'_>) -> Vec<u8> {
    memra_gguf::nvfp4_repack::repack_modelopt_to_gguf(
        matrix.codes,
        matrix.scales,
        matrix.out_features,
        matrix.in_features,
    )
}

fn nvfp4_row_bytes(in_features: usize) -> usize {
    in_features / 64 * 36 // memra block_nvfp4: 64 elems -> 36 bytes (4 UE4M3 + 32 packed e2m1)
}

/// MEMRA_NO_LOCAL_SHADOW=1: skip the per-layer local-KV shadow gathers and appends in the
/// eager v2 decode (lengths still advance) — the graph door proved contents-stale local KV
/// is decode-identical (12/12). The local contents feed spec/MTP scratch only.
/// MEMRA_FUSE_ROPE_APPEND=1: fuse qk norms + rope + dcw KV append + len inc into one
/// launch per rank per layer (bit-identical; identity-gated). dcw path only.
pub(crate) fn fuse_rope_append_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_FUSE_ROPE_APPEND").as_deref() == Ok("1"))
}

pub(crate) fn no_local_shadow_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_NO_LOCAL_SHADOW").as_deref() == Ok("1"))
}

/// Permute one repacked block_nvfp4 matrix (out_features rows of `nvfp4_row_bytes(in_f)`)
/// into the slot-major row layout the EP2 kernels read: per row, slot g's 16 qs bytes at
/// g*16, then the two UE4M3 scale bytes per slot at nslots*16 + g*2. Row byte count
/// unchanged. This layout USED to be an env door (`MEMRA_NVFP4_BANK_V2`, removed 2026-08-29
/// after its ON arm changed generated text in serving, see
/// research/step37-bankv2-removal-20260829); it survives ONLY as the fixed layout of the
/// EP2 whole-expert banks, whose `*_ep` kernels read it unconditionally.
///
/// PUBLIC because it is the SINGLE SOURCE OF TRUTH for this byte map. Every reader — the
/// `*_ep` decode kernels, `kq_fetch<QT_NVFP4_V2>` in the grouped GEMM,
/// `dequant_nvfp4v2_f16_kernel` — is defined as "reads what this function writes", and the
/// `nvfp4-bank-oracle` bin is what proves it, on device, per kernel arm. Do not reimplement
/// the map anywhere: the two failures that appeared only on v2 readers were geometry-plumbing
/// bugs around a byte map that was itself correct in two separate places. The layout was
/// innocent; one live failure was the grouped-prefill sktail call site defaulting `in_f` to zero.
pub fn nvfp4_matrix_v2_permute(v1: &[u8], out_features: usize, in_features: usize) -> Vec<u8> {
    // The output row is n_slots*18 bytes; the stride every reader uses is
    // nvfp4_row_bytes(in_features) = (in_features/64)*36. Those are equal only when
    // in_features is a whole number of 64-element superblocks. At in_features % 64 == 32 the
    // permute would silently emit a LONGER row than the stride and every row after row 0
    // would be read at the wrong offset, so refuse instead of trusting the caller.
    assert_eq!(
        in_features % 64,
        0,
        "v2 permute needs whole 64-element superblocks, got in_features={in_features}"
    );
    let row_bytes = nvfp4_row_bytes(in_features);
    assert_eq!(v1.len(), out_features * row_bytes, "v2 permute geometry");
    let n_slots = in_features / 32;
    let mut out = Vec::with_capacity(v1.len());
    for row in 0..out_features {
        let r = &v1[row * row_bytes..(row + 1) * row_bytes];
        for g in 0..n_slots {
            let (sblk, h) = (g / 2, g % 2);
            let b = &r[sblk * 36..sblk * 36 + 36];
            out.extend_from_slice(&b[4 + 16 * h..4 + 16 * h + 16]);
        }
        for g in 0..n_slots {
            let (sblk, h) = (g / 2, g % 2);
            let b = &r[sblk * 36..sblk * 36 + 36];
            out.push(b[2 * h]);
            out.push(b[2 * h + 1]);
        }
    }
    out
}

/// Repack one expert shard for the contiguous banks. `slot_major` is true ONLY for the EP2
/// whole-expert banks, whose `*_ep` kernels read the slot-major permutation; the TP
/// column/row shard banks stay in the block_nvfp4 v1 layout every other kernel reads.
fn nvfp4_repack_bank_matrix(matrix: Nvfp4BlockMatrix<'_>, slot_major: bool) -> Vec<u8> {
    let (out_features, in_features) = (matrix.out_features, matrix.in_features);
    let v1 = nvfp4_repack_matrix(matrix);
    if slot_major {
        nvfp4_matrix_v2_permute(&v1, out_features, in_features)
    } else {
        v1
    }
}

/// Column shard: whole output rows per rank (codes and scales are row-major, so both slices are
/// contiguous borrows). The macro rides unchanged — it is applied post-gather by the caller.
#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn nvfp4_column_shard<'a>(
    matrix: Nvfp4BlockMatrix<'a>,
    tp: usize,
    rank: usize,
) -> Result<Nvfp4BlockMatrix<'a>, String> {
    if matrix.out_features % tp != 0 {
        return Err(format!(
            "NVFP4 column-parallel out_features {} is not divisible by TP={tp}",
            matrix.out_features
        ));
    }
    let local_out = matrix.out_features / tp;
    let code_row = matrix.in_features / 2;
    let scale_row = matrix.in_features / 16;
    Ok(Nvfp4BlockMatrix {
        codes: &matrix.codes[rank * local_out * code_row..(rank + 1) * local_out * code_row],
        scales: &matrix.scales[rank * local_out * scale_row..(rank + 1) * local_out * scale_row],
        macro_scale: matrix.macro_scale,
        out_features: local_out,
        in_features: matrix.in_features,
    })
}

/// Row shard: input-column windows per rank, 64-superblock aligned. Owned buffers: each output
/// row contributes one contiguous byte window, gathered across rows.
#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn nvfp4_row_shard(
    matrix: Nvfp4BlockMatrix<'_>,
    tp: usize,
    rank: usize,
) -> Result<(Vec<u8>, Vec<u8>, usize), String> {
    if matrix.in_features % tp != 0 {
        return Err(format!(
            "NVFP4 row-parallel in_features {} is not divisible by TP={tp}",
            matrix.in_features
        ));
    }
    let local_in = matrix.in_features / tp;
    if !local_in.is_multiple_of(64) {
        return Err(format!(
            "NVFP4 row-parallel input shard {local_in} cuts through a 64-element superblock"
        ));
    }
    let code_row = matrix.in_features / 2;
    let scale_row = matrix.in_features / 16;
    let local_code = local_in / 2;
    let local_scale = local_in / 16;
    let mut codes = Vec::with_capacity(matrix.out_features * local_code);
    let mut scales = Vec::with_capacity(matrix.out_features * local_scale);
    for row in 0..matrix.out_features {
        let code_start = row * code_row + rank * local_code;
        codes.extend_from_slice(&matrix.codes[code_start..code_start + local_code]);
        let scale_start = row * scale_row + rank * local_scale;
        scales.extend_from_slice(&matrix.scales[scale_start..scale_start + local_scale]);
    }
    Ok((codes, scales, local_in))
}

/// Rank compute leaf: repack modelopt -> block_nvfp4, upload, run the proven dp4a kernel. The
/// macro is NOT applied here — callers apply it once at the canonical post-gather/post-reduce
/// point (see the section header).
fn run_rank_nvfp4(
    engine: &Engine,
    matrix: Nvfp4BlockMatrix<'_>,
    activations: &[f32],
    tokens: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    matrix.validate()?;
    validate_activations(activations, tokens, matrix.in_features)?;
    let _main = engine.gpu.enter_main()?;
    let blocks = engine.htod_bytes(&nvfp4_repack_matrix(matrix))?;
    let activations = engine.htod(activations)?;
    let output = engine.qmatvec_nvfp4_fast(
        &blocks.slice(0..blocks.len()),
        &activations,
        tokens,
        matrix.in_features,
        matrix.out_features,
        nvfp4_row_bytes(matrix.in_features),
    )?;
    engine.dtoh(&output)
}

fn upload_rank_nvfp4(
    engine: &Engine,
    matrix: Nvfp4BlockMatrix<'_>,
) -> Result<ResidentNvfp4Rank, Box<dyn std::error::Error>> {
    matrix.validate()?;
    let _main = engine.gpu.enter_main()?;
    Ok(ResidentNvfp4Rank {
        blocks: engine.htod_bytes(&nvfp4_repack_matrix(matrix))?,
        macro_scale: matrix.macro_scale,
        out_features: matrix.out_features,
        in_features: matrix.in_features,
        row_bytes: nvfp4_row_bytes(matrix.in_features),
    })
}

fn run_resident_rank_nvfp4(
    engine: &Engine,
    rank: &ResidentNvfp4Rank,
    activations: &[f32],
    tokens: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    validate_activations(activations, tokens, rank.in_features)?;
    let _main = engine.gpu.enter_main()?;
    let activations = engine.htod(activations)?;
    let output = engine.qmatvec_nvfp4_fast(
        &rank.blocks.slice(0..rank.blocks.len()),
        &activations,
        tokens,
        rank.in_features,
        rank.out_features,
        rank.row_bytes,
    )?;
    engine.dtoh(&output)
}

fn apply_macro(values: &mut [f32], macro_scale: f32) {
    for value in values.iter_mut() {
        *value *= macro_scale;
    }
}

impl TpE4m3HostBounce {
    /// Unsharded NVFP4 projection on rank 0 (compatibility oracle). Macro applied post-kernel.
    pub fn full_nvfp4(
        &self,
        matrix: Nvfp4BlockMatrix<'_>,
        activations: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut output = run_rank_nvfp4(&self.ranks[0], matrix, activations, tokens)?;
        apply_macro(&mut output, matrix.macro_scale);
        Ok(output)
    }

    /// Column-parallel NVFP4 projection: output rows partition across ranks, host gather in rank
    /// order, macro applied ONCE post-gather.
    pub fn column_parallel_nvfp4(
        &self,
        matrix: Nvfp4BlockMatrix<'_>,
        activations: &[f32],
        tokens: usize,
    ) -> Result<ColumnParallelResult, Box<dyn std::error::Error>> {
        matrix.validate()?;
        validate_activations(activations, tokens, matrix.in_features)?;
        let tp = self.ranks.len();
        let local_out = matrix.out_features / tp;
        let mut gathered = vec![0.0f32; tokens * matrix.out_features];
        let mut rank_outputs = Vec::with_capacity(tp);
        for (rank_index, rank) in self.ranks.iter().enumerate() {
            let shard = nvfp4_column_shard(matrix, tp, rank_index)?;
            let output = run_rank_nvfp4(rank, shard, activations, tokens)?;
            let row_start = rank_index * local_out;
            for token in 0..tokens {
                gathered[token * matrix.out_features + row_start
                    ..token * matrix.out_features + row_start + local_out]
                    .copy_from_slice(&output[token * local_out..(token + 1) * local_out]);
            }
            rank_outputs.push(output);
        }
        apply_macro(&mut gathered, matrix.macro_scale);
        Ok(ColumnParallelResult {
            gathered,
            rank_outputs,
        })
    }

    /// Row-parallel NVFP4 projection: input columns partition at 64-superblock boundaries,
    /// rank-local partials reduce in stable rank order, macro applied ONCE post-reduce.
    pub fn row_parallel_nvfp4(
        &self,
        matrix: Nvfp4BlockMatrix<'_>,
        activations: &[f32],
        tokens: usize,
    ) -> Result<RowParallelResult, Box<dyn std::error::Error>> {
        matrix.validate()?;
        validate_activations(activations, tokens, matrix.in_features)?;
        let tp = self.ranks.len();
        let mut reduced = vec![0.0f32; tokens * matrix.out_features];
        let mut rank_partials = Vec::with_capacity(tp);
        for (rank_index, rank) in self.ranks.iter().enumerate() {
            let (codes, scales, local_in) = nvfp4_row_shard(matrix, tp, rank_index)?;
            let local_activations =
                activation_shard(activations, tokens, matrix.in_features, tp, rank_index);
            let shard = Nvfp4BlockMatrix {
                codes: &codes,
                scales: &scales,
                macro_scale: matrix.macro_scale,
                out_features: matrix.out_features,
                in_features: local_in,
            };
            let partial = run_rank_nvfp4(rank, shard, &local_activations, tokens)?;
            for (sum, value) in reduced.iter_mut().zip(&partial) {
                *sum += *value;
            }
            rank_partials.push(partial);
        }
        apply_macro(&mut reduced, matrix.macro_scale);
        Ok(RowParallelResult {
            reduced,
            rank_partials,
        })
    }

    pub fn upload_expert_nvfp4(
        &self,
        gate: Nvfp4BlockMatrix<'_>,
        up: Nvfp4BlockMatrix<'_>,
        down: Nvfp4BlockMatrix<'_>,
    ) -> Result<ResidentTpNvfp4Expert, Box<dyn std::error::Error>> {
        if gate.in_features != up.in_features || gate.out_features != up.out_features {
            return Err("NVFP4 TP expert gate/up dimensions differ".into());
        }
        if down.in_features != gate.out_features || down.out_features != gate.in_features {
            return Err(format!(
                "NVFP4 TP expert down {}x{} does not invert gate/up {}x{}",
                down.out_features, down.in_features, gate.out_features, gate.in_features
            )
            .into());
        }
        let tp = self.ranks.len();
        let mut gate_ranks = Vec::with_capacity(tp);
        let mut up_ranks = Vec::with_capacity(tp);
        let mut down_ranks = Vec::with_capacity(tp);
        for (rank_index, engine) in self.ranks.iter().enumerate() {
            gate_ranks.push(upload_rank_nvfp4(
                engine,
                nvfp4_column_shard(gate, tp, rank_index)?,
            )?);
            up_ranks.push(upload_rank_nvfp4(
                engine,
                nvfp4_column_shard(up, tp, rank_index)?,
            )?);
            let (codes, scales, local_in) = nvfp4_row_shard(down, tp, rank_index)?;
            down_ranks.push(upload_rank_nvfp4(
                engine,
                Nvfp4BlockMatrix {
                    codes: &codes,
                    scales: &scales,
                    macro_scale: down.macro_scale,
                    out_features: down.out_features,
                    in_features: local_in,
                },
            )?);
        }
        Ok(ResidentTpNvfp4Expert {
            gate: ResidentNvfp4ColumnParallel {
                ranks: gate_ranks,
                out_features: gate.out_features,
                in_features: gate.in_features,
            },
            up: ResidentNvfp4ColumnParallel {
                ranks: up_ranks,
                out_features: up.out_features,
                in_features: up.in_features,
            },
            down: ResidentNvfp4RowParallel {
                ranks: down_ranks,
                out_features: down.out_features,
                in_features: down.in_features,
            },
            input_width: gate.in_features,
            expert_width: gate.out_features,
        })
    }

    fn column_parallel_resident_nvfp4(
        &self,
        matrix: &ResidentNvfp4ColumnParallel,
        activations: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        validate_activations(activations, tokens, matrix.in_features)?;
        let local_out = matrix.out_features / self.ranks.len();
        let mut gathered = vec![0.0f32; tokens * matrix.out_features];
        let mut macro_scale = None;
        for (rank_index, (engine, shard)) in self.ranks.iter().zip(&matrix.ranks).enumerate() {
            let output = run_resident_rank_nvfp4(engine, shard, activations, tokens)?;
            let row_start = rank_index * local_out;
            for token in 0..tokens {
                gathered[token * matrix.out_features + row_start
                    ..token * matrix.out_features + row_start + local_out]
                    .copy_from_slice(&output[token * local_out..(token + 1) * local_out]);
            }
            macro_scale = Some(shard.macro_scale);
        }
        apply_macro(
            &mut gathered,
            macro_scale.ok_or("NVFP4 column-parallel matrix has no ranks")?,
        );
        Ok(gathered)
    }

    fn row_parallel_resident_nvfp4(
        &self,
        matrix: &ResidentNvfp4RowParallel,
        activations: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        validate_activations(activations, tokens, matrix.in_features)?;
        let tp = self.ranks.len();
        let local_in = matrix.in_features / tp;
        let mut reduced = vec![0.0f32; tokens * matrix.out_features];
        let mut macro_scale = None;
        for (rank_index, (engine, shard)) in self.ranks.iter().zip(&matrix.ranks).enumerate() {
            if shard.in_features != local_in {
                return Err(format!(
                    "NVFP4 resident row shard in_features {} != expected {local_in}",
                    shard.in_features
                )
                .into());
            }
            let local_activations =
                activation_shard(activations, tokens, matrix.in_features, tp, rank_index);
            let partial = run_resident_rank_nvfp4(engine, shard, &local_activations, tokens)?;
            for (sum, value) in reduced.iter_mut().zip(&partial) {
                *sum += *value;
            }
            macro_scale = Some(shard.macro_scale);
        }
        apply_macro(
            &mut reduced,
            macro_scale.ok_or("NVFP4 row-parallel matrix has no ranks")?,
        );
        Ok(reduced)
    }

    pub fn run_expert_nvfp4(
        &self,
        expert: &ResidentTpNvfp4Expert,
        input: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        validate_activations(input, tokens, expert.input_width)?;
        let gate = self.column_parallel_resident_nvfp4(&expert.gate, input, tokens)?;
        let up = self.column_parallel_resident_nvfp4(&expert.up, input, tokens)?;
        let activated: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| gate / (1.0 + (-gate).exp()) * up)
            .collect();
        debug_assert_eq!(activated.len(), tokens * expert.expert_width);
        self.row_parallel_resident_nvfp4(&expert.down, &activated, tokens)
    }

    /// Upload every expert's TP shards resident (one repacked block buffer per expert per rank).
    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    pub fn upload_tensor_parallel_nvfp4(
        &self,
        gate: Nvfp4ExpertBank<'_>,
        up: Nvfp4ExpertBank<'_>,
        down: Nvfp4ExpertBank<'_>,
    ) -> Result<ResidentNvfp4TensorParallel, Box<dyn std::error::Error>> {
        gate.validate()?;
        up.validate()?;
        down.validate()?;
        if gate.expert_count != up.expert_count || gate.expert_count != down.expert_count {
            return Err("NVFP4 TP gate/up/down expert counts differ".into());
        }
        if gate.in_features != up.in_features || gate.out_features != up.out_features {
            return Err("NVFP4 TP gate/up dimensions differ".into());
        }
        if down.in_features != gate.out_features || down.out_features != gate.in_features {
            return Err(format!(
                "NVFP4 TP down {}x{} does not invert gate/up {}x{}",
                down.out_features, down.in_features, gate.out_features, gate.in_features
            )
            .into());
        }
        let tp = self.ranks.len();
        if gate.out_features % tp != 0 {
            return Err(format!(
                "NVFP4 TP expert output width {} is not divisible by TP={tp}",
                gate.out_features
            )
            .into());
        }
        if !down.in_features.is_multiple_of(NVFP4_CANONICAL_ROW_SHARDS)
            || !(down.in_features / NVFP4_CANONICAL_ROW_SHARDS).is_multiple_of(64)
        {
            return Err(format!(
                "NVFP4 TP expert input width {} does not split into 64-aligned canonical \
                 shards ({NVFP4_CANONICAL_ROW_SHARDS})",
                down.in_features
            )
            .into());
        }
        if tp > NVFP4_CANONICAL_ROW_SHARDS {
            return Err(format!(
                "NVFP4 TP world {tp} exceeds the canonical row-shard grid \
                 ({NVFP4_CANONICAL_ROW_SHARDS})"
            )
            .into());
        }

        // LAYOUT DECISION, MADE ONCE PER BANK BUILD. TP shard banks are slot-major under
        // PROGRAM 1's door. Every reader below takes this from the bank it is reading, never
        // from `bank_slot_major_on()` again.
        let slot_major = bank_slot_major_on();
        // ENGAGEMENT RECEIPT, not a debug line. A pricing cell that proves only that the env var
        // is SET measures nothing: if the door fails to reach the code, the cell reports "the
        // program is worth 0%" when the truth is "the program never ran". That exact defect is
        // banked -- the MEMRA_BF16_MMV lane's first sweep grepped for engagement, got 0 in BOTH
        // arms, and the missing line was mistaken for a no-engagement result until an announce
        // was added. So the layout decision announces itself, WITH ITS SOURCE, so a receipt can
        // distinguish "armed by the door" from "armed because EP2" from "not armed".
        eprintln!(
            "[nvfp4-bank] layout={} source={} tp={tp} experts={} in_f={} out_f={}",
            if slot_major {
                "slot-major"
            } else {
                "block-nvfp4-v1"
            },
            // The source string distinguishes "armed by the 2026-09-01 DEFAULT" from "armed by
            // an explicit recipe" from "rolled back by the seam" from "armed because EP2". A
            // default flip whose receipt cannot say which of those happened cannot prove the
            // DEFAULT was what got measured.
            bank_slot_major_source().1,
            gate.expert_count,
            gate.in_features,
            gate.out_features
        );
        let mut gate_ranks = Vec::with_capacity(tp);
        let mut up_ranks = Vec::with_capacity(tp);
        let mut macros_gate_dev = Vec::with_capacity(tp);
        let mut macros_up_dev = Vec::with_capacity(tp);
        let mut macros_down_dev = Vec::with_capacity(tp);
        for (rank_index, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            // Contiguous per-rank banks: repack every expert shard into one host buffer, one
            // upload. Contiguity feeds the batched selected-experts launch; per-expert bytes
            // are unchanged (same repack).
            // EP2: this rank holds the FULL matrices of the experts it owns (id & 1 ==
            // rank_index), stacked at slot id >> 1 — same total bytes as the shard bank.
            let mut gate_host: Vec<u8> = Vec::new();
            let mut up_host: Vec<u8> = Vec::new();
            for expert in 0..gate.expert_count {
                let gate_shard = nvfp4_column_shard(gate.expert(expert)?, tp, rank_index)?;
                gate_host.extend_from_slice(&nvfp4_repack_bank_matrix(gate_shard, slot_major));
                let up_shard = nvfp4_column_shard(up.expert(expert)?, tp, rank_index)?;
                up_host.extend_from_slice(&nvfp4_repack_bank_matrix(up_shard, slot_major));
            }
            let bank_experts = gate.expert_count;
            let gate_expert_bytes = gate_host.len() / bank_experts.max(1);
            let up_expert_bytes = up_host.len() / bank_experts.max(1);
            let local_out = gate.out_features / tp;
            gate_ranks.push(ResidentNvfp4ColumnBankRank {
                bank: engine.htod_bytes(&gate_host)?,
                expert_bytes: gate_expert_bytes,
                local_out,
                in_features: gate.in_features,
                row_bytes: nvfp4_row_bytes(gate.in_features),
                slot_major,
            });
            up_ranks.push(ResidentNvfp4ColumnBankRank {
                bank: engine.htod_bytes(&up_host)?,
                expert_bytes: up_expert_bytes,
                local_out,
                in_features: up.in_features,
                row_bytes: nvfp4_row_bytes(up.in_features),
                slot_major,
            });
            macros_gate_dev.push(engine.htod(gate.macros)?);
            macros_up_dev.push(engine.htod(up.macros)?);
            macros_down_dev.push(engine.htod(down.macros)?);
        }
        // Down: canonical shard grid, NOT the world size (see NVFP4_CANONICAL_ROW_SHARDS).
        // Shard s lives on rank s % world, so TP1 holds both shards and TP2 one each, while the
        // execution and reduction order stay identical.
        let mut down_ranks = Vec::with_capacity(NVFP4_CANONICAL_ROW_SHARDS);
        for shard_index in 0..NVFP4_CANONICAL_ROW_SHARDS {
            let device_rank = shard_index % tp;
            let engine = &self.ranks[device_rank];
            let _main = engine.gpu.enter_main()?;
            let mut down_host: Vec<u8> = Vec::new();
            for expert in 0..down.expert_count {
                let down_matrix = down.expert(expert)?;
                let (codes, scales, local_in) =
                    nvfp4_row_shard(down_matrix, NVFP4_CANONICAL_ROW_SHARDS, shard_index)?;
                down_host.extend_from_slice(&nvfp4_repack_bank_matrix(
                    Nvfp4BlockMatrix {
                        codes: &codes,
                        scales: &scales,
                        macro_scale: down_matrix.macro_scale,
                        out_features: down_matrix.out_features,
                        in_features: local_in,
                    },
                    slot_major,
                ));
            }
            let bank_experts = down.expert_count;
            let down_expert_bytes = down_host.len() / bank_experts.max(1);
            let local_in = down.in_features / NVFP4_CANONICAL_ROW_SHARDS;
            down_ranks.push(ResidentNvfp4RowBankRank {
                bank: engine.htod_bytes(&down_host)?,
                expert_bytes: down_expert_bytes,
                device_rank,
                out_features: down.out_features,
                local_in,
                row_bytes: nvfp4_row_bytes(local_in),
                slot_major,
            });
        }
        Ok(ResidentNvfp4TensorParallel {
            gate: gate_ranks,
            up: up_ranks,
            down: down_ranks,
            macros_gate: gate.macros.to_vec(),
            macros_up: up.macros.to_vec(),
            macros_down: down.macros.to_vec(),
            macros_gate_dev,
            macros_up_dev,
            macros_down_dev,
            expert_count: gate.expert_count,
            input_width: gate.in_features,
            expert_width: gate.out_features,
            device_workspace: std::sync::Mutex::new(None),
            prime_tables: std::sync::Mutex::new(Vec::new()),
        })
    }

    fn run_column_bank_expert_nvfp4(
        &self,
        ranks: &[ResidentNvfp4ColumnBankRank],
        macros: &[f32],
        expert: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let local_out = ranks
            .first()
            .ok_or("NVFP4 TP column bank has no ranks")?
            .local_out;
        let mut gathered = vec![0.0f32; local_out * ranks.len()];
        for (rank_index, (engine, bank)) in self.ranks.iter().zip(ranks).enumerate() {
            let _main = engine.gpu.enter_main()?;
            let activations = engine.htod(input)?;
            let output = bank.host_canonical_expert(engine, expert, &activations)?;
            let output = engine.dtoh(&output)?;
            gathered[rank_index * local_out..(rank_index + 1) * local_out].copy_from_slice(&output);
        }
        apply_macro(&mut gathered, macros[expert]);
        Ok(gathered)
    }

    /// Canonical-shard row reduction: iterate the FIXED shard grid in shard order (each shard
    /// executes on its owning rank engine), so the reduction parenthesization is identical at
    /// every world size — that identity is what the TP1-oracle-vs-TP2 bit gate proves.
    fn run_row_bank_expert_nvfp4(
        &self,
        shards: &[ResidentNvfp4RowBankRank],
        macros: &[f32],
        expert: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let out_features = shards
            .first()
            .ok_or("NVFP4 TP row bank has no canonical shards")?
            .out_features;
        let in_features = shards.iter().map(|shard| shard.local_in).sum::<usize>();
        let mut reduced = vec![0.0f32; out_features];
        for (shard_index, shard) in shards.iter().enumerate() {
            let engine = self
                .ranks
                .get(shard.device_rank)
                .ok_or("NVFP4 canonical shard names a rank outside this runtime")?;
            let _main = engine.gpu.enter_main()?;
            let local_activations =
                activation_shard(input, 1, in_features, shards.len(), shard_index);
            let activations = engine.htod(&local_activations)?;
            let output = shard.host_canonical_expert(engine, expert, &activations)?;
            let partial = engine.dtoh(&output)?;
            for (sum, value) in reduced.iter_mut().zip(&partial) {
                *sum += *value;
            }
        }
        apply_macro(&mut reduced, macros[expert]);
        Ok(reduced)
    }

    /// Upload whole experts per owning rank (NVFP4 expert-parallel: the layout the clamped tail
    /// layers require — clamp semantics do not distribute across a tensor shard). Each owned
    /// expert keeps its full gate/up/down as one repacked block buffer on its owner.
    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    pub fn upload_expert_parallel_nvfp4(
        &self,
        gate: Nvfp4ExpertBank<'_>,
        up: Nvfp4ExpertBank<'_>,
        down: Nvfp4ExpertBank<'_>,
    ) -> Result<ResidentNvfp4ExpertParallel, Box<dyn std::error::Error>> {
        gate.validate()?;
        up.validate()?;
        down.validate()?;
        if gate.expert_count != up.expert_count || gate.expert_count != down.expert_count {
            return Err("NVFP4 EP gate/up/down expert counts differ".into());
        }
        if gate.in_features != up.in_features || gate.out_features != up.out_features {
            return Err("NVFP4 EP gate/up dimensions differ".into());
        }
        if down.in_features != gate.out_features || down.out_features != gate.in_features {
            return Err(format!(
                "NVFP4 EP down {}x{} does not invert gate/up {}x{}",
                down.out_features, down.in_features, gate.out_features, gate.in_features
            )
            .into());
        }
        let world = self.ranks.len();
        if gate.expert_count % world != 0 {
            return Err(format!(
                "NVFP4 EP expert count {} is not divisible by {world} ranks",
                gate.expert_count
            )
            .into());
        }
        let experts_per_rank = gate.expert_count / world;
        let mut ranks = Vec::with_capacity(world);
        for (rank_index, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            let expert_range = rank_index * experts_per_rank..(rank_index + 1) * experts_per_rank;
            let mut gate_host = Vec::new();
            let mut up_host = Vec::new();
            let mut down_host = Vec::new();
            for expert in expert_range.clone() {
                gate_host.extend_from_slice(&nvfp4_repack_matrix(gate.expert(expert)?));
                up_host.extend_from_slice(&nvfp4_repack_matrix(up.expert(expert)?));
                down_host.extend_from_slice(&nvfp4_repack_matrix(down.expert(expert)?));
            }
            let gate_expert_bytes = gate_host.len() / experts_per_rank;
            let up_expert_bytes = up_host.len() / experts_per_rank;
            if gate_expert_bytes != up_expert_bytes {
                return Err("NVFP4 EP gate/up packed expert bytes differ".into());
            }
            let down_expert_bytes = down_host.len() / experts_per_rank;
            ranks.push(ResidentNvfp4EpRank {
                gate: engine.htod_bytes(&gate_host)?,
                up: engine.htod_bytes(&up_host)?,
                down: engine.htod_bytes(&down_host)?,
                gate_expert_bytes,
                down_expert_bytes,
                macros_gate: engine.htod(&gate.macros[expert_range.clone()])?,
                macros_up: engine.htod(&up.macros[expert_range.clone()])?,
                macros_down: engine.htod(&down.macros[expert_range.clone()])?,
                expert_range,
            });
        }
        Ok(ResidentNvfp4ExpertParallel {
            ranks,
            macros_gate: gate.macros.to_vec(),
            macros_up: up.macros.to_vec(),
            macros_down: down.macros.to_vec(),
            expert_count: gate.expert_count,
            input_width: gate.in_features,
            expert_width: gate.out_features,
            gate_row_bytes: nvfp4_row_bytes(gate.in_features),
            down_row_bytes: nvfp4_row_bytes(down.in_features),
            device_workspace: std::sync::Mutex::new(None),
        })
    }

    /// Upload an already-normalized NVFP4 expert bank.
    ///
    /// `HostExps` is the physical-format boundary: stacked checkpoint tensors, gathered
    /// per-expert tensors, and manifest-backed overlays all become the same contiguous
    /// block_nvfp4 expert representation before the parallel backend sees them.
    pub fn upload_expert_parallel_nvfp4_normalized(
        &self,
        gate: &crate::model::HostExps,
        up: &crate::model::HostExps,
        down: &crate::model::HostExps,
    ) -> Result<ResidentNvfp4ExpertParallel, Box<dyn std::error::Error>> {
        for (label, bank) in [("gate", gate), ("up", up), ("down", down)] {
            if bank.qtype != crate::QT_NVFP4 || !bank.is_uniform_layout() {
                return Err(format!(
                    "NVFP4 EP normalized {label} bank requires one uniform NVFP4 layout, \
                     got qtype={} uniform={}",
                    bank.qtype,
                    bank.is_uniform_layout()
                )
                .into());
            }
            if bank.n_expert == 0
                || bank.expert_stride != bank.out_f * bank.row_bytes
                || (0..bank.n_expert)
                    .any(|expert| bank.expert_bytes(expert).len() != bank.expert_stride)
            {
                return Err(format!("NVFP4 EP normalized {label} bank geometry is invalid").into());
            }
        }
        if gate.n_expert != up.n_expert || gate.n_expert != down.n_expert {
            return Err("NVFP4 EP normalized gate/up/down expert counts differ".into());
        }
        if gate.in_f != up.in_f || gate.out_f != up.out_f {
            return Err("NVFP4 EP normalized gate/up dimensions differ".into());
        }
        if down.in_f != gate.out_f || down.out_f != gate.in_f {
            return Err(format!(
                "NVFP4 EP normalized down {}x{} does not invert gate/up {}x{}",
                down.out_f, down.in_f, gate.out_f, gate.in_f
            )
            .into());
        }
        let macros = |bank: &crate::model::HostExps| -> Result<Vec<f32>, String> {
            let values = bank
                .macros
                .clone()
                .unwrap_or_else(|| vec![1.0; bank.n_expert]);
            if values.len() != bank.n_expert
                || !values.iter().all(|value| value.is_finite() && *value > 0.0)
            {
                return Err("NVFP4 EP normalized macro row is not finite-positive".to_string());
            }
            Ok(values)
        };
        let macros_gate = macros(gate)?;
        let macros_up = macros(up)?;
        let macros_down = macros(down)?;
        let world = self.ranks.len();
        if !gate.n_expert.is_multiple_of(world) {
            return Err(format!(
                "NVFP4 EP normalized expert count {} is not divisible by {world} ranks",
                gate.n_expert
            )
            .into());
        }
        let experts_per_rank = gate.n_expert / world;
        let mut ranks = Vec::with_capacity(world);
        for (rank_index, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            let expert_range = rank_index * experts_per_rank..(rank_index + 1) * experts_per_rank;
            let mut gate_host = Vec::with_capacity(experts_per_rank * gate.expert_stride);
            let mut up_host = Vec::with_capacity(experts_per_rank * up.expert_stride);
            let mut down_host = Vec::with_capacity(experts_per_rank * down.expert_stride);
            for expert in expert_range.clone() {
                gate_host.extend_from_slice(gate.expert_bytes(expert));
                up_host.extend_from_slice(up.expert_bytes(expert));
                down_host.extend_from_slice(down.expert_bytes(expert));
            }
            ranks.push(ResidentNvfp4EpRank {
                gate: engine.htod_bytes(&gate_host)?,
                up: engine.htod_bytes(&up_host)?,
                down: engine.htod_bytes(&down_host)?,
                gate_expert_bytes: gate.expert_stride,
                down_expert_bytes: down.expert_stride,
                macros_gate: engine.htod(&macros_gate[expert_range.clone()])?,
                macros_up: engine.htod(&macros_up[expert_range.clone()])?,
                macros_down: engine.htod(&macros_down[expert_range.clone()])?,
                expert_range,
            });
        }
        Ok(ResidentNvfp4ExpertParallel {
            ranks,
            macros_gate,
            macros_up,
            macros_down,
            expert_count: gate.n_expert,
            input_width: gate.in_f,
            expert_width: gate.out_f,
            gate_row_bytes: gate.row_bytes,
            down_row_bytes: down.row_bytes,
            device_workspace: std::sync::Mutex::new(None),
        })
    }

    /// Routed NVFP4 expert-parallel program, host-canonical: every selected expert executes WHOLE
    /// on its owning rank (gate -> up -> clamped-or-plain SwiGLU on host -> down), each projection
    /// macro applied once post-kernel, route-weighted accumulate on the host in slot order. The
    /// activation uses `step_expert_activation_host`, so the clamped tail layers keep the official
    /// contract. Exactness-first; no throughput claim.
    #[allow(clippy::too_many_arguments)]
    pub fn run_routed_experts_nvfp4(
        &self,
        experts: &ResidentNvfp4ExpertParallel,
        input: &[f32],
        tokens: usize,
        selected: &[usize],
        route_weights: &[f32],
        experts_per_token: usize,
        activation_limit: Option<f32>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        validate_activations(input, tokens, experts.input_width)?;
        let pairs = tokens
            .checked_mul(experts_per_token)
            .ok_or("NVFP4 EP route count overflow")?;
        if selected.len() != pairs || route_weights.len() != pairs {
            return Err(format!(
                "NVFP4 EP routes selected={} weights={} != tokens {tokens} x experts/token \
                 {experts_per_token} ({pairs})",
                selected.len(),
                route_weights.len(),
            )
            .into());
        }
        if !route_weights.iter().all(|weight| weight.is_finite()) {
            return Err("NVFP4 EP route weights contain a non-finite value".into());
        }
        let experts_per_rank = experts.expert_count / experts.ranks.len();
        let mut output = vec![0.0f32; tokens * experts.input_width];
        for token in 0..tokens {
            let input_row = &input[token * experts.input_width..(token + 1) * experts.input_width];
            for slot in 0..experts_per_token {
                let pair = token * experts_per_token + slot;
                let expert = selected[pair];
                if expert >= experts.expert_count {
                    return Err(format!(
                        "NVFP4 EP selected expert {expert} outside 0..{}",
                        experts.expert_count
                    )
                    .into());
                }
                let owner = expert / experts_per_rank;
                let local = expert - owner * experts_per_rank;
                let rank = &experts.ranks[owner];
                let engine = &self.ranks[owner];
                let _main = engine.gpu.enter_main()?;
                let device_input = engine.htod(input_row)?;
                let gate_out = engine.qmatvec_nvfp4_fast(
                    &rank.gate.slice(
                        local * rank.gate_expert_bytes..(local + 1) * rank.gate_expert_bytes,
                    ),
                    &device_input,
                    1,
                    experts.input_width,
                    experts.expert_width,
                    experts.gate_row_bytes,
                )?;
                let up_out = engine.qmatvec_nvfp4_fast(
                    &rank.up.slice(
                        local * rank.gate_expert_bytes..(local + 1) * rank.gate_expert_bytes,
                    ),
                    &device_input,
                    1,
                    experts.input_width,
                    experts.expert_width,
                    experts.gate_row_bytes,
                )?;
                let mut gate_host = engine.dtoh(&gate_out)?;
                let mut up_host = engine.dtoh(&up_out)?;
                apply_macro(&mut gate_host, experts.macros_gate[expert]);
                apply_macro(&mut up_host, experts.macros_up[expert]);
                let activated: Vec<f32> = gate_host
                    .iter()
                    .zip(&up_host)
                    .map(|(&gate, &up)| step_expert_activation_host(gate, up, activation_limit))
                    .collect();
                let device_activated = engine.htod(&activated)?;
                let down_out = engine.qmatvec_nvfp4_fast(
                    &rank.down.slice(
                        local * rank.down_expert_bytes..(local + 1) * rank.down_expert_bytes,
                    ),
                    &device_activated,
                    1,
                    experts.expert_width,
                    experts.input_width,
                    experts.down_row_bytes,
                )?;
                let mut down_host = engine.dtoh(&down_out)?;
                apply_macro(&mut down_host, experts.macros_down[expert]);
                let weight = route_weights[pair];
                for (sum, value) in output
                    [token * experts.input_width..(token + 1) * experts.input_width]
                    .iter_mut()
                    .zip(down_host)
                {
                    *sum += weight * value;
                }
            }
        }
        Ok(output)
    }

    /// Device-resident W4A16 expert parallelism for one scheduler/prefill batch (1..=128 rows).
    ///
    /// The host router partitions token/slot pairs by contiguous expert owner. Each rank
    /// peer-reads the whole batch input once, rounds it to BF16, and executes its owner-local
    /// selected gate/up -> host-expf SwiGLU -> BF16 -> down program. Down rows scatter directly
    /// into canonical token-major pair positions in the model engine's peer-accessible pool at
    /// every batch width; the root reduces each token's slots in original order. Thus batching
    /// and owner assignment do not change route-reduction parenthesization.
    #[allow(clippy::too_many_arguments)]
    pub fn run_routed_experts_nvfp4_w4a16_device_io(
        &self,
        experts: &ResidentNvfp4ExpertParallel,
        e: &Engine,
        input_dev: &crate::CudaSlice<f32>,
        tokens: usize,
        selected: &[usize],
        route_weights: &[f32],
        experts_per_token: usize,
        activation_limit: Option<f32>,
    ) -> Result<crate::CudaSlice<f32>, Box<dyn std::error::Error>> {
        // Diagnostic attribution only: force the returned root event chain to completion so the
        // caller's shared-expert timer does not absorb routed-EP work. The normal path remains
        // fully asynchronous.
        static TIMING_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static TIMING_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let timing = std::env::var("MEMRA_STEP_TP_TIMING").as_deref() == Ok("1");
        let started = timing.then(std::time::Instant::now);
        if !self.native_p2p {
            return Err("W4A16 NVFP4 device EP requires native P2P".into());
        }
        if self.devices.first().copied() != Some(e.ctx().ordinal()) {
            return Err(format!(
                "W4A16 NVFP4 device EP root device {:?} != model engine device {}",
                self.devices.first(),
                e.ctx().ordinal()
            )
            .into());
        }
        // Prime/cache scratch buffers are grow-only: a 160-token host-oracle chunk can be
        // followed by a 44-token device-EP tail using the same 160-row allocation. Consume the
        // active prefix rather than requiring allocation length == active length.
        let active_input_values =
            nvfp4_ep_active_input_values(input_dev.len(), tokens, experts.input_width)?;
        let pairs = tokens
            .checked_mul(experts_per_token)
            .ok_or("W4A16 NVFP4 device EP route count overflow")?;
        if selected.len() != pairs || route_weights.len() != pairs {
            return Err(format!(
                "W4A16 NVFP4 device EP routes selected={} weights={} != tokens {tokens} x \
                 experts/token {experts_per_token} ({pairs})",
                selected.len(),
                route_weights.len(),
            )
            .into());
        }
        if !route_weights.iter().all(|weight| weight.is_finite()) {
            return Err("W4A16 NVFP4 device EP route weights contain a non-finite value".into());
        }
        let world = self.ranks.len();
        if world != experts.ranks.len() || !(2..=PRODUCT_MAX_CARDS).contains(&world) {
            return Err(format!(
                "W4A16 NVFP4 device EP runtime ranks {world} != bank ranks {}",
                experts.ranks.len()
            )
            .into());
        }
        let owner_routes = partition_expert_owner_routes(
            experts.expert_count,
            world,
            tokens,
            experts_per_token,
            selected,
        )?;

        let mut workspace_guard = experts
            .device_workspace
            .lock()
            .map_err(|_| "W4A16 NVFP4 device EP workspace lock is poisoned")?;
        if workspace_guard.is_none() {
            let capacity_tokens = NVFP4_EP_DEVICE_BATCH_CAP;
            let capacity_pairs = capacity_tokens * experts_per_token;
            let mut input = Vec::with_capacity(world);
            let mut input_bf16 = Vec::with_capacity(world);
            let mut input_q8 = Vec::with_capacity(world);
            let mut input_q8_scales = Vec::with_capacity(world);
            let mut sel = Vec::with_capacity(world);
            let mut token_rows = Vec::with_capacity(world);
            let mut global_pairs = Vec::with_capacity(world);
            let mut route_w = Vec::with_capacity(world);
            let mut gate_out = Vec::with_capacity(world);
            let mut up_out = Vec::with_capacity(world);
            let mut activation_bf16 = Vec::with_capacity(world);
            let mut activation_q8 = Vec::with_capacity(world);
            let mut activation_q8_scales = Vec::with_capacity(world);
            let mut ev_rank = Vec::with_capacity(world);
            for engine in &self.ranks {
                let _main = engine.gpu.enter_main()?;
                input.push(engine.uninit(capacity_tokens * experts.input_width)?);
                input_bf16.push(engine.alloc_u8_uninit(2 * capacity_tokens * experts.input_width)?);
                input_q8.push(engine.alloc_i8_uninit(capacity_tokens * experts.input_width)?);
                input_q8_scales
                    .push(engine.uninit(capacity_tokens * experts.input_width.div_ceil(32))?);
                sel.push(engine.htod_i32(&vec![0i32; capacity_pairs])?);
                token_rows.push(engine.htod_i32(&vec![0i32; capacity_pairs])?);
                global_pairs.push(engine.htod_i32(&vec![0i32; capacity_pairs])?);
                route_w.push(engine.htod(&vec![0.0f32; capacity_pairs])?);
                gate_out.push(engine.uninit(capacity_pairs * experts.expert_width)?);
                up_out.push(engine.uninit(capacity_pairs * experts.expert_width)?);
                activation_bf16
                    .push(engine.alloc_u8_uninit(2 * capacity_pairs * experts.expert_width)?);
                activation_q8.push(engine.alloc_i8_uninit(capacity_pairs * experts.expert_width)?);
                activation_q8_scales
                    .push(engine.uninit(capacity_pairs * experts.expert_width.div_ceil(32))?);
                ev_rank.push(engine.ctx().new_event(None)?);
            }
            let _main = e.gpu.enter_main()?;
            let slot_rows = e.uninit(capacity_pairs * experts.input_width)?;
            let slot_rows_raw = {
                use cudarc::driver::DevicePtr;
                let stream = e.stream();
                let (pointer, _guard) = slot_rows.device_ptr(&stream);
                pointer
            };
            *workspace_guard = Some(Nvfp4EpDeviceWorkspace {
                input,
                input_bf16,
                input_q8,
                input_q8_scales,
                sel,
                token_rows,
                global_pairs,
                route_w,
                gate_out,
                up_out,
                activation_bf16,
                activation_q8,
                activation_q8_scales,
                slot_rows,
                slot_rows_raw,
                route_weights: e.htod(&vec![0.0f32; capacity_pairs])?,
                ev_entry: e.ctx().new_event(None)?,
                ev_entry_device: e.ctx().ordinal(),
                ev_rank,
                phase_events: None,
                capacity_tokens,
                experts_per_token,
            });
        }
        let workspace = workspace_guard
            .as_mut()
            .expect("W4A16 NVFP4 device EP workspace initialized above");
        if workspace.experts_per_token != experts_per_token || tokens > workspace.capacity_tokens {
            return Err(format!(
                "W4A16 NVFP4 device EP workspace tokens={} experts/token={} cannot serve \
                 tokens={tokens} experts/token={experts_per_token}",
                workspace.capacity_tokens, workspace.experts_per_token,
            )
            .into());
        }
        if workspace.ev_entry_device != e.ctx().ordinal() {
            return Err("W4A16 NVFP4 device EP model engine changed".into());
        }

        {
            let _main = e.gpu.enter_main()?;
            let mut destination = workspace.route_weights.slice_mut(0..pairs);
            e.stream()
                .memcpy_htod(&route_weights[..pairs], &mut destination)?;
            workspace.ev_entry.record(&e.stream())?;
        }
        for (rank_index, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            engine.stream().wait(&workspace.ev_entry)?;
            {
                let mut destination = workspace.input[rank_index].slice_mut(0..active_input_values);
                engine
                    .stream()
                    .memcpy_dtod(&input_dev.slice(0..active_input_values), &mut destination)?;
            }
            engine.f32_to_bf16_into(
                &workspace.input[rank_index],
                &mut workspace.input_bf16[rank_index],
                tokens * experts.input_width,
            )?;
            let owner = &owner_routes[rank_index];
            debug_assert_eq!(owner.rank, rank_index);
            let local_count = owner.selected.len();
            if local_count > 0 {
                let local_selected = owner
                    .selected
                    .iter()
                    .map(|&expert| expert as i32)
                    .collect::<Vec<_>>();
                let local_token_rows = owner
                    .token_rows
                    .iter()
                    .map(|&token| token as i32)
                    .collect::<Vec<_>>();
                let local_global_pairs = owner
                    .global_pairs
                    .iter()
                    .map(|&pair| pair as i32)
                    .collect::<Vec<_>>();
                {
                    let mut destination = workspace.sel[rank_index].slice_mut(0..local_count);
                    engine
                        .stream()
                        .memcpy_htod(&local_selected, &mut destination)?;
                }
                {
                    let mut destination =
                        workspace.token_rows[rank_index].slice_mut(0..local_count);
                    engine
                        .stream()
                        .memcpy_htod(&local_token_rows, &mut destination)?;
                }
                {
                    let mut destination =
                        workspace.global_pairs[rank_index].slice_mut(0..local_count);
                    engine
                        .stream()
                        .memcpy_htod(&local_global_pairs, &mut destination)?;
                }
                let rank = &experts.ranks[rank_index];
                engine.qmatvec_nvfp4_bf16_sel_dual_rows_into(
                    &rank.gate,
                    &rank.up,
                    &workspace.sel[rank_index],
                    &workspace.token_rows[rank_index],
                    &workspace.input_bf16[rank_index],
                    &mut workspace.gate_out[rank_index],
                    &mut workspace.up_out[rank_index],
                    local_count,
                    experts.input_width,
                    experts.expert_width,
                    experts.gate_row_bytes,
                    rank.gate_expert_bytes,
                    tokens,
                )?;
                engine.silu_mul_scaled_host_expf_bf16_sel_into(
                    &workspace.gate_out[rank_index],
                    &workspace.up_out[rank_index],
                    &rank.macros_gate,
                    &rank.macros_up,
                    &workspace.sel[rank_index],
                    activation_limit,
                    &mut workspace.activation_bf16[rank_index],
                    experts.expert_width,
                    local_count,
                )?;
                engine.qmatvec_nvfp4_bf16_sel_down_rows_raw(
                    &rank.down,
                    &workspace.sel[rank_index],
                    &workspace.global_pairs[rank_index],
                    &workspace.activation_bf16[rank_index],
                    &rank.macros_down,
                    workspace.slot_rows_raw,
                    local_count,
                    experts.expert_width,
                    experts.input_width,
                    experts.down_row_bytes,
                    rank.down_expert_bytes,
                    pairs,
                )?;
            }
            workspace.ev_rank[rank_index].record(&engine.stream())?;
        }

        let output = {
            let _main = e.gpu.enter_main()?;
            for event in &workspace.ev_rank {
                e.stream().wait(event)?;
            }
            let mut output = e.uninit(tokens * experts.input_width)?;
            e.axpy_rows_seq_tokens_into(
                &workspace.slot_rows,
                &workspace.route_weights,
                &mut output,
                experts.input_width,
                experts_per_token,
                tokens,
            )?;
            output
        };
        if let Some(started) = started {
            use std::sync::atomic::Ordering;
            e.stream().synchronize()?;
            let elapsed = started.elapsed().as_nanos() as u64;
            let ns = TIMING_NS.fetch_add(elapsed, Ordering::Relaxed) + elapsed;
            let calls = TIMING_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
            if calls.is_multiple_of(430) {
                eprintln!(
                    "[nvfp4-ep-w4a16-timing] calls={calls} total_ms={:.1} avg_us={:.1}",
                    ns as f64 / 1.0e6,
                    ns as f64 / calls as f64 / 1.0e3,
                );
            }
        }
        Ok(output)
    }

    /// Fully device-routed W4A16 expert parallelism. Router ids/weights stay on the model GPU;
    /// each rank receives the fixed token/slot metadata, rejects non-owned experts in-kernel, and
    /// writes canonical token-major slot rows back to the root at every batch width. Preserving
    /// that one accumulation program is required by speculative verification: the former t=1
    /// owner-grouped FMA was a distinct numeric class and failed real HY3 MTP self-consistency.
    #[allow(clippy::too_many_arguments)]
    pub fn run_routed_experts_nvfp4_w4a16_device_routed(
        &self,
        experts: &ResidentNvfp4ExpertParallel,
        e: &Engine,
        input_dev: &crate::CudaSlice<f32>,
        selected_dev: &crate::CudaSlice<i32>,
        route_weights_dev: &crate::CudaSlice<f32>,
        tokens: usize,
        experts_per_token: usize,
        activation_limit: Option<f32>,
    ) -> Result<crate::CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.run_routed_experts_nvfp4_w4a16_device_routed_inner(
            experts,
            e,
            input_dev,
            selected_dev,
            route_weights_dev,
            tokens,
            experts_per_token,
            activation_limit,
            None,
        )
    }

    /// Automatic whole-expert EP with a PREJOIN hook. The hook runs after every rank's routed
    /// chain has been issued and before the root waits for rank completion, so independent
    /// root-device work can fill the peer drain without changing the routed accumulation order.
    #[allow(clippy::too_many_arguments)]
    pub fn run_routed_experts_nvfp4_w4a16_device_routed_prejoin(
        &self,
        experts: &ResidentNvfp4ExpertParallel,
        e: &Engine,
        input_dev: &crate::CudaSlice<f32>,
        selected_dev: &crate::CudaSlice<i32>,
        route_weights_dev: &crate::CudaSlice<f32>,
        tokens: usize,
        experts_per_token: usize,
        activation_limit: Option<f32>,
        mut pre_join: impl FnMut() -> Result<(), Box<dyn std::error::Error>>,
    ) -> Result<crate::CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.run_routed_experts_nvfp4_w4a16_device_routed_inner(
            experts,
            e,
            input_dev,
            selected_dev,
            route_weights_dev,
            tokens,
            experts_per_token,
            activation_limit,
            Some(&mut pre_join),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_routed_experts_nvfp4_w4a16_device_routed_inner(
        &self,
        experts: &ResidentNvfp4ExpertParallel,
        e: &Engine,
        input_dev: &crate::CudaSlice<f32>,
        selected_dev: &crate::CudaSlice<i32>,
        route_weights_dev: &crate::CudaSlice<f32>,
        tokens: usize,
        experts_per_token: usize,
        activation_limit: Option<f32>,
        mut pre_join: Option<&mut dyn FnMut() -> Result<(), Box<dyn std::error::Error>>>,
    ) -> Result<crate::CudaSlice<f32>, Box<dyn std::error::Error>> {
        if !self.native_p2p {
            return Err("W4A16 device-routed EP requires native P2P".into());
        }
        if self.devices.first().copied() != Some(e.ctx().ordinal()) {
            return Err(format!(
                "W4A16 device-routed EP root device {:?} != model engine device {}",
                self.devices.first(),
                e.ctx().ordinal()
            )
            .into());
        }
        let active_input_values =
            nvfp4_ep_active_input_values(input_dev.len(), tokens, experts.input_width)?;
        let pairs = tokens
            .checked_mul(experts_per_token)
            .ok_or("W4A16 device-routed EP route count overflow")?;
        if selected_dev.len() < pairs || route_weights_dev.len() < pairs {
            return Err(format!(
                "W4A16 device-routed EP metadata selected={} weights={} < pairs={pairs}",
                selected_dev.len(),
                route_weights_dev.len(),
            )
            .into());
        }
        let world = self.ranks.len();
        if world != experts.ranks.len() || !(2..=PRODUCT_MAX_CARDS).contains(&world) {
            return Err(format!(
                "W4A16 device-routed EP runtime ranks {world} != bank ranks {}",
                experts.ranks.len()
            )
            .into());
        }

        static TIMING_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static TIMING_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static ISSUE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static JOIN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static COPY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static GATE_UP_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static ACTIVATION_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static DOWN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static RANK_SPAN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let timing = std::env::var("MEMRA_STEP_TP_TIMING").as_deref() == Ok("1");
        let started = timing.then(std::time::Instant::now);
        let pair_down_enabled = parallel_ep_pair_down_enabled()?;

        let mut workspace_guard = experts
            .device_workspace
            .lock()
            .map_err(|_| "W4A16 device-routed EP workspace lock is poisoned")?;
        if workspace_guard.is_none() {
            let capacity_tokens = NVFP4_EP_DEVICE_BATCH_CAP;
            let capacity_pairs = capacity_tokens * experts_per_token;
            let mut input = Vec::with_capacity(world);
            let mut input_bf16 = Vec::with_capacity(world);
            let mut input_q8 = Vec::with_capacity(world);
            let mut input_q8_scales = Vec::with_capacity(world);
            let mut sel = Vec::with_capacity(world);
            let mut token_rows = Vec::with_capacity(world);
            let mut global_pairs = Vec::with_capacity(world);
            let mut route_w = Vec::with_capacity(world);
            let mut gate_out = Vec::with_capacity(world);
            let mut up_out = Vec::with_capacity(world);
            let mut activation_bf16 = Vec::with_capacity(world);
            let mut activation_q8 = Vec::with_capacity(world);
            let mut activation_q8_scales = Vec::with_capacity(world);
            let mut ev_rank = Vec::with_capacity(world);
            let mut phase_head = Vec::with_capacity(world);
            let mut phase_copy_done = Vec::with_capacity(world);
            let mut phase_gate_up_done = Vec::with_capacity(world);
            let mut phase_activation_done = Vec::with_capacity(world);
            let mut phase_down_done = Vec::with_capacity(world);
            for engine in &self.ranks {
                let _main = engine.gpu.enter_main()?;
                input.push(engine.uninit(capacity_tokens * experts.input_width)?);
                input_bf16.push(engine.alloc_u8_uninit(2 * capacity_tokens * experts.input_width)?);
                input_q8.push(engine.alloc_i8_uninit(capacity_tokens * experts.input_width)?);
                input_q8_scales
                    .push(engine.uninit(capacity_tokens * experts.input_width.div_ceil(32))?);
                sel.push(engine.htod_i32(&vec![0i32; capacity_pairs])?);
                token_rows.push(engine.htod_i32(&vec![0i32; capacity_pairs])?);
                global_pairs.push(engine.htod_i32(&vec![0i32; capacity_pairs])?);
                route_w.push(engine.htod(&vec![0.0f32; capacity_pairs])?);
                gate_out.push(engine.uninit(capacity_pairs * experts.expert_width)?);
                up_out.push(engine.uninit(capacity_pairs * experts.expert_width)?);
                activation_bf16
                    .push(engine.alloc_u8_uninit(2 * capacity_pairs * experts.expert_width)?);
                activation_q8.push(engine.alloc_i8_uninit(capacity_pairs * experts.expert_width)?);
                activation_q8_scales
                    .push(engine.uninit(capacity_pairs * experts.expert_width.div_ceil(32))?);
                ev_rank.push(engine.ctx().new_event(None)?);
                if timing {
                    phase_head.push(
                        engine.ctx().new_event(Some(
                            cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                        ))?,
                    );
                    phase_copy_done.push(
                        engine.ctx().new_event(Some(
                            cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                        ))?,
                    );
                    phase_gate_up_done.push(
                        engine.ctx().new_event(Some(
                            cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                        ))?,
                    );
                    phase_activation_done.push(
                        engine.ctx().new_event(Some(
                            cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                        ))?,
                    );
                    phase_down_done.push(
                        engine.ctx().new_event(Some(
                            cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                        ))?,
                    );
                }
            }
            let _main = e.gpu.enter_main()?;
            let slot_rows = e.uninit(capacity_pairs * experts.input_width)?;
            let slot_rows_raw = {
                use cudarc::driver::DevicePtr;
                let stream = e.stream();
                let (pointer, _guard) = slot_rows.device_ptr(&stream);
                pointer
            };
            *workspace_guard = Some(Nvfp4EpDeviceWorkspace {
                input,
                input_bf16,
                input_q8,
                input_q8_scales,
                sel,
                token_rows,
                global_pairs,
                route_w,
                gate_out,
                up_out,
                activation_bf16,
                activation_q8,
                activation_q8_scales,
                slot_rows,
                slot_rows_raw,
                route_weights: e.htod(&vec![0.0f32; capacity_pairs])?,
                ev_entry: e.ctx().new_event(None)?,
                ev_entry_device: e.ctx().ordinal(),
                ev_rank,
                phase_events: timing.then_some(Nvfp4EpPhaseEvents {
                    head: phase_head,
                    copy_done: phase_copy_done,
                    gate_up_done: phase_gate_up_done,
                    activation_done: phase_activation_done,
                    down_done: phase_down_done,
                }),
                capacity_tokens,
                experts_per_token,
            });
        }
        let workspace = workspace_guard
            .as_mut()
            .expect("W4A16 device-routed EP workspace initialized above");
        if workspace.experts_per_token != experts_per_token || tokens > workspace.capacity_tokens {
            return Err(format!(
                "W4A16 device-routed EP workspace tokens={} experts/token={} cannot serve \
                 tokens={tokens} experts/token={experts_per_token}",
                workspace.capacity_tokens, workspace.experts_per_token,
            )
            .into());
        }

        if tokens <= NVFP4_EP_Q8_BATCH_CAP && parallel_ep_q8_act_enabled()? {
            return self.run_routed_experts_nvfp4_w4a8_device_routed(
                experts,
                e,
                input_dev,
                selected_dev,
                route_weights_dev,
                workspace,
                tokens,
                experts_per_token,
                activation_limit,
                pre_join,
            );
        }

        {
            let _main = e.gpu.enter_main()?;
            e.memset_zeros_view(
                &mut workspace
                    .slot_rows
                    .slice_mut(0..pairs * experts.input_width),
            )?;
            workspace.ev_entry.record(&e.stream())?;
        }

        for (rank_index, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            if let Some(events) = workspace.phase_events.as_ref() {
                events.head[rank_index].record(&engine.stream())?;
            }
            engine.stream().wait(&workspace.ev_entry)?;
            let Nvfp4EpDeviceWorkspace {
                input_bf16,
                sel,
                route_w,
                ..
            } = &mut *workspace;
            engine.nvfp4_ep_stage_inputs(
                input_dev,
                selected_dev,
                route_weights_dev,
                &mut input_bf16[rank_index],
                &mut sel[rank_index],
                &mut route_w[rank_index],
                active_input_values,
                pairs,
                false,
            )?;
            if let Some(events) = workspace.phase_events.as_ref() {
                events.copy_done[rank_index].record(&engine.stream())?;
            }
            let rank = &experts.ranks[rank_index];
            let owner_start = rank.expert_range.start;
            let owner_end = rank.expert_range.end;
            engine.qmatvec_nvfp4_bf16_ep_dual_slots_into(
                &rank.gate,
                &rank.up,
                &workspace.sel[rank_index],
                &workspace.input_bf16[rank_index],
                &mut workspace.gate_out[rank_index],
                &mut workspace.up_out[rank_index],
                pairs,
                experts_per_token,
                experts.input_width,
                experts.expert_width,
                owner_start,
                owner_end,
                experts.gate_row_bytes,
                rank.gate_expert_bytes,
            )?;
            if let Some(events) = workspace.phase_events.as_ref() {
                events.gate_up_done[rank_index].record(&engine.stream())?;
            }
            engine.silu_mul_scaled_host_expf_bf16_ep_slots_into(
                &workspace.gate_out[rank_index],
                &workspace.up_out[rank_index],
                &rank.macros_gate,
                &rank.macros_up,
                &workspace.sel[rank_index],
                owner_start,
                owner_end,
                activation_limit,
                &mut workspace.activation_bf16[rank_index],
                experts.expert_width,
                pairs,
            )?;
            if let Some(events) = workspace.phase_events.as_ref() {
                events.activation_done[rank_index].record(&engine.stream())?;
            }
            if tokens > 1 && pair_down_enabled {
                engine.qmatvec_nvfp4_bf16_ep_down_pairs_raw(
                    &rank.down,
                    &workspace.sel[rank_index],
                    &workspace.activation_bf16[rank_index],
                    &rank.macros_down,
                    workspace.slot_rows_raw,
                    pairs,
                    experts.expert_width,
                    experts.input_width,
                    owner_start,
                    owner_end,
                    experts.down_row_bytes,
                    rank.down_expert_bytes,
                )?;
            } else {
                engine.qmatvec_nvfp4_bf16_ep_down_slots_raw(
                    &rank.down,
                    &workspace.sel[rank_index],
                    &workspace.activation_bf16[rank_index],
                    &rank.macros_down,
                    workspace.slot_rows_raw,
                    pairs,
                    experts.expert_width,
                    experts.input_width,
                    owner_start,
                    owner_end,
                    experts.down_row_bytes,
                    rank.down_expert_bytes,
                )?;
            }
            if let Some(events) = workspace.phase_events.as_ref() {
                events.down_done[rank_index].record(&engine.stream())?;
            }
            workspace.ev_rank[rank_index].record(&engine.stream())?;
        }

        if let Some(pre_join) = pre_join.as_mut() {
            pre_join()?;
        }
        let issue_ns_this = started
            .as_ref()
            .map(|started| started.elapsed().as_nanos() as u64);
        let join_started = timing.then(std::time::Instant::now);
        let output = {
            let _main = e.gpu.enter_main()?;
            for event in &workspace.ev_rank {
                e.stream().wait(event)?;
            }
            let mut output = e.uninit(tokens * experts.input_width)?;
            e.axpy_rows_seq_tokens_into(
                &workspace.slot_rows,
                route_weights_dev,
                &mut output,
                experts.input_width,
                experts_per_token,
                tokens,
            )?;
            output
        };

        if let Some(started) = started {
            use std::sync::atomic::Ordering;
            e.stream().synchronize()?;
            let elapsed = started.elapsed().as_nanos() as u64;
            let join_ns_this = join_started
                .expect("timing join starts with total timing")
                .elapsed()
                .as_nanos() as u64;
            let mut phase_max_ms = [0.0f32; 5];
            if let Some(events) = workspace.phase_events.as_ref() {
                for rank_index in 0..world {
                    let engine = &self.ranks[rank_index];
                    let _main = engine.gpu.enter_main()?;
                    phase_max_ms[0] = phase_max_ms[0]
                        .max(events.head[rank_index].elapsed_ms(&events.copy_done[rank_index])?);
                    phase_max_ms[1] = phase_max_ms[1].max(
                        events.copy_done[rank_index]
                            .elapsed_ms(&events.gate_up_done[rank_index])?,
                    );
                    phase_max_ms[2] = phase_max_ms[2].max(
                        events.gate_up_done[rank_index]
                            .elapsed_ms(&events.activation_done[rank_index])?,
                    );
                    phase_max_ms[3] = phase_max_ms[3].max(
                        events.activation_done[rank_index]
                            .elapsed_ms(&events.down_done[rank_index])?,
                    );
                    phase_max_ms[4] = phase_max_ms[4]
                        .max(events.head[rank_index].elapsed_ms(&events.down_done[rank_index])?);
                }
            }
            let phase_ns = phase_max_ms.map(|ms| (ms as f64 * 1.0e6) as u64);
            let ns = TIMING_NS.fetch_add(elapsed, Ordering::Relaxed) + elapsed;
            let issue_ns = ISSUE_NS.fetch_add(
                issue_ns_this.expect("timing issue starts with total timing"),
                Ordering::Relaxed,
            ) + issue_ns_this.expect("timing issue starts with total timing");
            let join_ns = JOIN_NS.fetch_add(join_ns_this, Ordering::Relaxed) + join_ns_this;
            let copy_ns = COPY_NS.fetch_add(phase_ns[0], Ordering::Relaxed) + phase_ns[0];
            let gate_up_ns = GATE_UP_NS.fetch_add(phase_ns[1], Ordering::Relaxed) + phase_ns[1];
            let activation_ns =
                ACTIVATION_NS.fetch_add(phase_ns[2], Ordering::Relaxed) + phase_ns[2];
            let down_ns = DOWN_NS.fetch_add(phase_ns[3], Ordering::Relaxed) + phase_ns[3];
            let rank_span_ns = RANK_SPAN_NS.fetch_add(phase_ns[4], Ordering::Relaxed) + phase_ns[4];
            let calls = TIMING_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
            if calls.is_multiple_of(430) {
                eprintln!(
                    "[nvfp4-ep-device-router-timing] calls={calls} total_ms={:.1} avg_us={:.1}",
                    ns as f64 / 1.0e6,
                    ns as f64 / calls as f64 / 1.0e3,
                );
                eprintln!(
                    "[nvfp4-ep-device-router-phases] calls={calls} issue_us={:.1} \
                     join_us={:.1} rank_span_us={:.1} copy_us={:.1} gate_up_us={:.1} \
                     activation_us={:.1} down_us={:.1}",
                    issue_ns as f64 / calls as f64 / 1.0e3,
                    join_ns as f64 / calls as f64 / 1.0e3,
                    rank_span_ns as f64 / calls as f64 / 1.0e3,
                    copy_ns as f64 / calls as f64 / 1.0e3,
                    gate_up_ns as f64 / calls as f64 / 1.0e3,
                    activation_ns as f64 / calls as f64 / 1.0e3,
                    down_ns as f64 / calls as f64 / 1.0e3,
                );
            }
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_routed_experts_nvfp4_w4a8_device_routed(
        &self,
        experts: &ResidentNvfp4ExpertParallel,
        e: &Engine,
        input_dev: &crate::CudaSlice<f32>,
        selected_dev: &crate::CudaSlice<i32>,
        route_weights_dev: &crate::CudaSlice<f32>,
        workspace: &mut Nvfp4EpDeviceWorkspace,
        tokens: usize,
        experts_per_token: usize,
        activation_limit: Option<f32>,
        mut pre_join: Option<&mut dyn FnMut() -> Result<(), Box<dyn std::error::Error>>>,
    ) -> Result<crate::CudaSlice<f32>, Box<dyn std::error::Error>> {
        static TIMING_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static TIMING_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static ISSUE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static JOIN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static COPY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static GATE_UP_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static ACTIVATION_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static DOWN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static RANK_SPAN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let timing = std::env::var("MEMRA_STEP_TP_TIMING").as_deref() == Ok("1");
        let started = timing.then(std::time::Instant::now);
        let pairs = tokens
            .checked_mul(experts_per_token)
            .ok_or("W4A8 device-routed EP route count overflow")?;
        let input_values = tokens
            .checked_mul(experts.input_width)
            .ok_or("W4A8 device-routed EP input size overflow")?;
        let scope = parallel_ep_q8_scope()?.unwrap_or(ParallelEpQ8Scope::All);
        let gate_up_paired = parallel_ep_q8_gu_paired_enabled(true, Some(scope));

        {
            let _main = e.gpu.enter_main()?;
            e.memset_zeros_view(
                &mut workspace
                    .slot_rows
                    .slice_mut(0..pairs * experts.input_width),
            )?;
            workspace.ev_entry.record(&e.stream())?;
        }
        for (rank_index, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            if let Some(events) = workspace.phase_events.as_ref() {
                events.head[rank_index].record(&engine.stream())?;
            }
            engine.stream().wait(&workspace.ev_entry)?;
            let rank = &experts.ranks[rank_index];
            let owner_start = rank.expert_range.start;
            let owner_end = rank.expert_range.end;
            match scope {
                ParallelEpQ8Scope::All | ParallelEpQ8Scope::GateUp => {
                    engine.quantize_q8_1_into(
                        input_dev,
                        tokens,
                        experts.input_width,
                        &mut workspace.input_q8[rank_index],
                        &mut workspace.input_q8_scales[rank_index],
                    )?;
                    engine.moe_sel_w_mirror(
                        selected_dev,
                        route_weights_dev,
                        &mut workspace.sel[rank_index],
                        &mut workspace.route_w[rank_index],
                        pairs,
                    )?;
                    if let Some(events) = workspace.phase_events.as_ref() {
                        events.copy_done[rank_index].record(&engine.stream())?;
                    }
                    if gate_up_paired {
                        engine.qmatvec_nvfp4_q8_ep_paired_slots_into(
                            &rank.gate,
                            &rank.up,
                            &workspace.sel[rank_index],
                            &workspace.input_q8[rank_index],
                            &workspace.input_q8_scales[rank_index],
                            &mut workspace.gate_out[rank_index],
                            &mut workspace.up_out[rank_index],
                            pairs,
                            experts_per_token,
                            experts.input_width,
                            experts.expert_width,
                            owner_start,
                            owner_end,
                            experts.gate_row_bytes,
                            rank.gate_expert_bytes,
                        )?;
                    } else {
                        engine.qmatvec_nvfp4_q8_ep_dual_slots_into(
                            &rank.gate,
                            &rank.up,
                            &workspace.sel[rank_index],
                            &workspace.input_q8[rank_index],
                            &workspace.input_q8_scales[rank_index],
                            &mut workspace.gate_out[rank_index],
                            &mut workspace.up_out[rank_index],
                            pairs,
                            experts_per_token,
                            experts.input_width,
                            experts.expert_width,
                            owner_start,
                            owner_end,
                            experts.gate_row_bytes,
                            rank.gate_expert_bytes,
                        )?;
                    }
                }
                ParallelEpQ8Scope::Down => {
                    engine.nvfp4_ep_stage_inputs(
                        input_dev,
                        selected_dev,
                        route_weights_dev,
                        &mut workspace.input_bf16[rank_index],
                        &mut workspace.sel[rank_index],
                        &mut workspace.route_w[rank_index],
                        input_values,
                        pairs,
                        false,
                    )?;
                    if let Some(events) = workspace.phase_events.as_ref() {
                        events.copy_done[rank_index].record(&engine.stream())?;
                    }
                    engine.qmatvec_nvfp4_bf16_ep_dual_slots_into(
                        &rank.gate,
                        &rank.up,
                        &workspace.sel[rank_index],
                        &workspace.input_bf16[rank_index],
                        &mut workspace.gate_out[rank_index],
                        &mut workspace.up_out[rank_index],
                        pairs,
                        experts_per_token,
                        experts.input_width,
                        experts.expert_width,
                        owner_start,
                        owner_end,
                        experts.gate_row_bytes,
                        rank.gate_expert_bytes,
                    )?;
                }
            }
            if let Some(events) = workspace.phase_events.as_ref() {
                events.gate_up_done[rank_index].record(&engine.stream())?;
            }
            match scope {
                ParallelEpQ8Scope::All | ParallelEpQ8Scope::Down => {
                    engine.silu_mul_scaled_host_expf_q8_ep_slots_into(
                        &workspace.gate_out[rank_index],
                        &workspace.up_out[rank_index],
                        &rank.macros_gate,
                        &rank.macros_up,
                        &workspace.sel[rank_index],
                        owner_start,
                        owner_end,
                        activation_limit,
                        &mut workspace.activation_q8[rank_index],
                        &mut workspace.activation_q8_scales[rank_index],
                        experts.expert_width,
                        pairs,
                    )?;
                    if let Some(events) = workspace.phase_events.as_ref() {
                        events.activation_done[rank_index].record(&engine.stream())?;
                    }
                    engine.qmatvec_nvfp4_q8_ep_down_slots_raw(
                        &rank.down,
                        &workspace.sel[rank_index],
                        &workspace.activation_q8[rank_index],
                        &workspace.activation_q8_scales[rank_index],
                        &rank.macros_down,
                        workspace.slot_rows_raw,
                        pairs,
                        experts.expert_width,
                        experts.input_width,
                        owner_start,
                        owner_end,
                        experts.down_row_bytes,
                        rank.down_expert_bytes,
                    )?;
                }
                ParallelEpQ8Scope::GateUp => {
                    engine.silu_mul_scaled_host_expf_bf16_ep_slots_into(
                        &workspace.gate_out[rank_index],
                        &workspace.up_out[rank_index],
                        &rank.macros_gate,
                        &rank.macros_up,
                        &workspace.sel[rank_index],
                        owner_start,
                        owner_end,
                        activation_limit,
                        &mut workspace.activation_bf16[rank_index],
                        experts.expert_width,
                        pairs,
                    )?;
                    if let Some(events) = workspace.phase_events.as_ref() {
                        events.activation_done[rank_index].record(&engine.stream())?;
                    }
                    engine.qmatvec_nvfp4_bf16_ep_down_slots_raw(
                        &rank.down,
                        &workspace.sel[rank_index],
                        &workspace.activation_bf16[rank_index],
                        &rank.macros_down,
                        workspace.slot_rows_raw,
                        pairs,
                        experts.expert_width,
                        experts.input_width,
                        owner_start,
                        owner_end,
                        experts.down_row_bytes,
                        rank.down_expert_bytes,
                    )?;
                }
            }
            if let Some(events) = workspace.phase_events.as_ref() {
                events.down_done[rank_index].record(&engine.stream())?;
            }
            workspace.ev_rank[rank_index].record(&engine.stream())?;
        }

        if let Some(pre_join) = pre_join.as_mut() {
            pre_join()?;
        }
        let issue_ns_this = started
            .as_ref()
            .map(|started| started.elapsed().as_nanos() as u64);
        let join_started = timing.then(std::time::Instant::now);
        let output = {
            let _main = e.gpu.enter_main()?;
            for event in &workspace.ev_rank {
                e.stream().wait(event)?;
            }
            let mut output = e.uninit(input_values)?;
            e.axpy_rows_seq_tokens_into(
                &workspace.slot_rows,
                route_weights_dev,
                &mut output,
                experts.input_width,
                experts_per_token,
                tokens,
            )?;
            output
        };
        if let Some(started) = started {
            use std::sync::atomic::Ordering;
            e.stream().synchronize()?;
            let elapsed = started.elapsed().as_nanos() as u64;
            let join_ns_this = join_started
                .expect("timing join starts with total timing")
                .elapsed()
                .as_nanos() as u64;
            let mut phase_max_ms = [0.0f32; 5];
            if let Some(events) = workspace.phase_events.as_ref() {
                for rank_index in 0..self.ranks.len() {
                    let engine = &self.ranks[rank_index];
                    let _main = engine.gpu.enter_main()?;
                    phase_max_ms[0] = phase_max_ms[0]
                        .max(events.head[rank_index].elapsed_ms(&events.copy_done[rank_index])?);
                    phase_max_ms[1] = phase_max_ms[1].max(
                        events.copy_done[rank_index]
                            .elapsed_ms(&events.gate_up_done[rank_index])?,
                    );
                    phase_max_ms[2] = phase_max_ms[2].max(
                        events.gate_up_done[rank_index]
                            .elapsed_ms(&events.activation_done[rank_index])?,
                    );
                    phase_max_ms[3] = phase_max_ms[3].max(
                        events.activation_done[rank_index]
                            .elapsed_ms(&events.down_done[rank_index])?,
                    );
                    phase_max_ms[4] = phase_max_ms[4]
                        .max(events.head[rank_index].elapsed_ms(&events.down_done[rank_index])?);
                }
            }
            let phase_ns = phase_max_ms.map(|ms| (ms as f64 * 1.0e6) as u64);
            let ns = TIMING_NS.fetch_add(elapsed, Ordering::Relaxed) + elapsed;
            let issue_ns = ISSUE_NS.fetch_add(
                issue_ns_this.expect("timing issue starts with total timing"),
                Ordering::Relaxed,
            ) + issue_ns_this.expect("timing issue starts with total timing");
            let join_ns = JOIN_NS.fetch_add(join_ns_this, Ordering::Relaxed) + join_ns_this;
            let copy_ns = COPY_NS.fetch_add(phase_ns[0], Ordering::Relaxed) + phase_ns[0];
            let gate_up_ns = GATE_UP_NS.fetch_add(phase_ns[1], Ordering::Relaxed) + phase_ns[1];
            let activation_ns =
                ACTIVATION_NS.fetch_add(phase_ns[2], Ordering::Relaxed) + phase_ns[2];
            let down_ns = DOWN_NS.fetch_add(phase_ns[3], Ordering::Relaxed) + phase_ns[3];
            let rank_span_ns = RANK_SPAN_NS.fetch_add(phase_ns[4], Ordering::Relaxed) + phase_ns[4];
            let calls = TIMING_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
            if calls.is_multiple_of(430) {
                eprintln!(
                    "[nvfp4-ep-q8-timing] calls={calls} total_ms={:.1} avg_us={:.1}",
                    ns as f64 / 1.0e6,
                    ns as f64 / calls as f64 / 1.0e3,
                );
                eprintln!(
                    "[nvfp4-ep-q8-phases] calls={calls} issue_us={:.1} join_us={:.1} \
                     rank_span_us={:.1} copy_us={:.1} gate_up_us={:.1} \
                     activation_us={:.1} down_us={:.1}",
                    issue_ns as f64 / calls as f64 / 1.0e3,
                    join_ns as f64 / calls as f64 / 1.0e3,
                    rank_span_ns as f64 / calls as f64 / 1.0e3,
                    copy_ns as f64 / calls as f64 / 1.0e3,
                    gate_up_ns as f64 / calls as f64 / 1.0e3,
                    activation_ns as f64 / calls as f64 / 1.0e3,
                    down_ns as f64 / calls as f64 / 1.0e3,
                );
            }
        }
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let (expert_input, post_activation, numeric_class) = match scope {
                ParallelEpQ8Scope::All => ("q8_1", "q8_1", "w4a8-internal"),
                ParallelEpQ8Scope::GateUp => ("q8_1", "bf16", "w4a8-gate-up-internal"),
                ParallelEpQ8Scope::Down => ("bf16", "q8_1", "w4a8-down-internal"),
            };
            eprintln!(
                "[parallel-ep-q8] devices={:?} tokens={tokens} scope={} \
                 expert_input={expert_input} post_activation={post_activation} \
                 gate_up_schedule={} \
                 external_boundary=bf16 numeric_class={numeric_class} \
                 host_expf=true accumulation=token-slot-order performance_claim=false",
                self.devices,
                scope.label(),
                if gate_up_paired {
                    "paired-cta"
                } else {
                    "separate-cta"
                },
            );
        }
        Ok(output)
    }

    /// Device-resident routed NVFP4 expert program (decode shape, t=1 rows). The geometry gift
    /// this exploits: gate/up column halves land on the SAME rank that owns the matching down
    /// canonical shard (act[rank r] is exactly down-shard r's input-column window), so the whole
    /// expert interior — gate, up, macro-scaled SwiGLU, down partial, route-weighted accumulate —
    /// runs rank-local with ZERO cross-rank transfer. Per (token, layer): one input upload per
    /// rank, one fenced peer copy of the remote accumulator, one root add, one readback.
    ///
    /// Numeric class: device silu (silu_mul_scaled) with gate/up macros folded as gs/us and the
    /// down macro folded into the accumulate scalar (weight * macro_down — exact, both are
    /// per-expert constants). This matches the owning-stage MoE dev-path semantics, NOT the
    /// host-canonical program bit-for-bit; gate it with argmax + relative bounds against the
    /// host-canonical oracle, and with repeat determinism against itself.
    /// Clamped layers refuse (they stay on the EP program).
    pub fn run_tensor_parallel_routes_nvfp4_device(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        input: &[f32],
        selected: &[usize],
        route_weights: &[f32],
        experts_per_token: usize,
        activation_limit: Option<f32>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        validate_activations(input, 1, experts.input_width)?;
        if selected.len() != experts_per_token || route_weights.len() != experts_per_token {
            return Err(format!(
                "NVFP4 device routes selected={} weights={} != experts/token {experts_per_token}",
                selected.len(),
                route_weights.len(),
            )
            .into());
        }
        if !route_weights.iter().all(|weight| weight.is_finite()) {
            return Err("NVFP4 device route weights contain a non-finite value".into());
        }
        let world = self.ranks.len();
        if world != NVFP4_CANONICAL_ROW_SHARDS {
            return Err(format!(
                "NVFP4 device routes require world == canonical shard grid \
                 ({NVFP4_CANONICAL_ROW_SHARDS}), got {world}"
            )
            .into());
        }
        let local_out = experts.expert_width / world;

        // MEMRA_STEP_TP_TIMING=1: cumulative wall-clock of this program, printed every 430 calls
        // (~one 43-layer decode step's worth) so a bench run decomposes expert-program time vs
        // everything else without Nsight.
        static TIMING_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static TIMING_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let timing = std::env::var("MEMRA_STEP_TP_TIMING").as_deref() == Ok("1");
        let started = timing.then(std::time::Instant::now);

        let n_sel = experts_per_token;
        let mut workspace_guard = experts
            .device_workspace
            .lock()
            .map_err(|_| "NVFP4 device routes workspace lock is poisoned")?;
        if workspace_guard.is_none() {
            let mut gate_out = Vec::with_capacity(world);
            let mut up_out = Vec::with_capacity(world);
            let mut act_q = Vec::with_capacity(world);
            let mut act_d = Vec::with_capacity(world);
            let mut sel = Vec::with_capacity(world);
            let mut partial = Vec::with_capacity(world);
            let mut accumulator = Vec::with_capacity(world);
            let mut combine_w = Vec::with_capacity(world);
            let mut route_w = Vec::with_capacity(world);
            let mut in_q = Vec::with_capacity(world);
            let mut in_d = Vec::with_capacity(world);
            let mut input = Vec::with_capacity(world);
            let mut ev_rank = Vec::with_capacity(world);
            let moe_direct = moe_direct_on();
            for (rank, engine) in self.ranks.iter().enumerate() {
                let _main = engine.gpu.enter_main()?;
                gate_out.push(engine.uninit(n_sel * local_out)?);
                up_out.push(engine.uninit(n_sel * local_out)?);
                act_q.push(engine.uninit_i8(n_sel * local_out)?);
                act_d.push(engine.uninit(n_sel * local_out / 32)?);
                sel.push(engine.htod_i32(&vec![0i32; n_sel])?);
                partial.push(engine.uninit(n_sel * experts.input_width)?);
                // Direct join: peer accumulators live on ROOT (single P2P store pass).
                if moe_direct && rank != 0 {
                    let root = &self.ranks[0];
                    let _root_main = root.gpu.enter_main()?;
                    accumulator.push(root.zeros(experts.input_width)?);
                } else {
                    accumulator.push(engine.zeros(experts.input_width)?);
                }
                combine_w.push(engine.htod(&vec![0.0f32; n_sel])?);
                route_w.push(engine.htod(&vec![0.0f32; n_sel])?);
                in_q.push(engine.uninit_i8(experts.input_width)?);
                in_d.push(engine.uninit(experts.input_width / 32)?);
                input.push(engine.uninit(experts.input_width)?);
                ev_rank.push(engine.ctx().new_event(None)?);
            }
            let root = &self.ranks[0];
            let _main = root.gpu.enter_main()?;
            *workspace_guard = Some(Nvfp4DeviceRoutesWorkspace {
                prestaged: false,
                rank1_routed: false,
                ev_input: None,
                fence_flags_raw: 0,
                fence_ticket: 0,
                gate_out,
                up_out,
                act_q,
                act_d,
                sel,
                partial,
                accumulator,
                combine_w,
                route_w,
                in_q,
                in_d,
                dev_route_e: None,
                in_stage_e: None,
                out_stage_e: None,
                routes_graph: None,
                raw_dev_route_e: None,
                raw_combine: None,
                raw_input: Vec::new(),
                raw_sel: Vec::new(),
                raw_route_w: Vec::new(),
                remote: root.uninit(experts.input_width)?,
                combined: root.uninit(experts.input_width)?,
                n_sel,
                input,
                ev_rank,
                ev_done: Some(root.ctx().new_event(None)?),
                ev_entry: None,
            });
        }
        let workspace = workspace_guard
            .as_mut()
            .expect("NVFP4 device routes workspace initialized above");
        if workspace.n_sel != n_sel {
            return Err(format!(
                "NVFP4 device routes experts/token changed: workspace {} != call {n_sel}",
                workspace.n_sel
            )
            .into());
        }
        for &expert in selected {
            if expert >= experts.expert_count {
                return Err(format!(
                    "NVFP4 device selected expert {expert} outside 0..{}",
                    experts.expert_count
                )
                .into());
            }
        }
        let sel_i32 = selected
            .iter()
            .map(|&expert| expert as i32)
            .collect::<Vec<_>>();

        // BATCHED program (2026-08-20): per rank, ONE launch per sweep (gate, up, SwiGLU,
        // down) covers every selected expert via the selection array and the contiguous bank —
        // the per-expert launch loop was pure host latency (~100 sequential launches/layer,
        // 291us wall for ~35us of arithmetic). Per (expert, row) the kernels are bit-identical
        // to the per-expert forms, and the route-weight axpy chain keeps its exact sequential
        // accumulation order — the program's values are unchanged.
        for (rank_index, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            let device_input = engine.htod(input)?;
            let Nvfp4DeviceRoutesWorkspace { in_q, in_d, .. } = &mut *workspace;
            engine.quantize_q8_1_into(
                &device_input,
                1,
                experts.input_width,
                &mut in_q[rank_index],
                &mut in_d[rank_index],
            )?;
            // device_input frees on this rank's stream after the quantize — same-stream order.
        }
        self.nvfp4_routes_batched_sweeps(
            experts,
            workspace,
            selected,
            route_weights,
            &sel_i32,
            local_out,
            n_sel,
            activation_limit,
            false,
        )?;

        // Combine: fence the remote shard's producer stream, peer-copy its accumulator to root,
        // reduce in canonical shard order, read back once.
        let root = &self.ranks[0];
        for engine in &self.ranks[1..] {
            let _main = engine.gpu.enter_main()?;
            engine.stream().synchronize()?;
        }
        let _main = root.gpu.enter_main()?;
        root.stream()
            .memcpy_dtod(&workspace.accumulator[1], &mut workspace.remote)?;
        root.add(
            &workspace.accumulator[0],
            &workspace.remote,
            &mut workspace.combined,
            experts.input_width,
        )?;
        let output = root.dtoh(&workspace.combined)?;
        if let Some(started) = started {
            use std::sync::atomic::Ordering;
            let ns = TIMING_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed)
                + started.elapsed().as_nanos() as u64;
            let calls = TIMING_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
            if calls.is_multiple_of(430) {
                eprintln!(
                    "[nvfp4-dev-routes-timing] calls={calls} total_ms={:.1} avg_us={:.1}",
                    ns as f64 / 1.0e6,
                    ns as f64 / calls as f64 / 1.0e3,
                );
            }
        }
        Ok(output)
    }

    /// The shared batched sweeps of the device routes program: per rank, upload the selection,
    /// reset the accumulator, run the gate/up/SwiGLU/down batched launches, then the
    /// route-weight axpy chain in exact sequential per-pair order. Every op queues on the
    /// owning rank's stream; callers own input acquisition and the combine.
    #[allow(clippy::too_many_arguments)]
    fn nvfp4_routes_batched_sweeps(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        workspace: &mut Nvfp4DeviceRoutesWorkspace,
        selected: &[usize],
        route_weights: &[f32],
        sel_i32: &[i32],
        local_out: usize,
        n_sel: usize,
        activation_limit: Option<f32>,
        device_routed: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for rank_index in 0..self.ranks.len() {
            self.nvfp4_routes_batched_sweeps_rank(
                experts,
                workspace,
                selected,
                route_weights,
                sel_i32,
                local_out,
                n_sel,
                activation_limit,
                device_routed,
                rank_index,
            )?;
        }
        Ok(())
    }

    /// One rank's sweeps (the per-rank body of `nvfp4_routes_batched_sweeps`) — separated so
    /// the graph door can capture each rank's segment on its own stream.
    #[allow(clippy::too_many_arguments)]
    fn nvfp4_routes_batched_sweeps_rank(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        workspace: &mut Nvfp4DeviceRoutesWorkspace,
        selected: &[usize],
        route_weights: &[f32],
        sel_i32: &[i32],
        local_out: usize,
        n_sel: usize,
        activation_limit: Option<f32>,
        device_routed: bool,
        rank_index: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        {
            let engine = &self.ranks[rank_index];
            let _main = engine.gpu.enter_main()?;
            if !device_routed {
                engine.htod_i32_into(&mut workspace.sel[rank_index], sel_i32)?;
                // Folded combine weights (route_weight x down macro) — one 40-byte upload
                // replaces the accumulator reset + n_sel sequential axpy launches below.
                let folded = (0..n_sel)
                    .map(|pair| route_weights[pair] * experts.macros_down[selected[pair]])
                    .collect::<Vec<_>>();
                let mut view = workspace.combine_w[rank_index].slice_mut(0..n_sel);
                engine.stream().memcpy_htod(&folded, &mut view)?;
            }
            let gate_bank = &experts.gate[rank_index];
            let up_bank = &experts.up[rank_index];
            let (aq, ad) = (&workspace.in_q[rank_index], &workspace.in_d[rank_index]);
            engine.qmatvec_nvfp4_sel_into(
                &gate_bank.bank,
                &workspace.sel[rank_index],
                aq,
                ad,
                &mut workspace.gate_out[rank_index],
                n_sel,
                gate_bank.in_features,
                gate_bank.local_out,
                gate_bank.row_bytes,
                gate_bank.expert_bytes,
                0,
                0,
                gate_bank.slot_major,
            )?;
            engine.qmatvec_nvfp4_sel_into(
                &up_bank.bank,
                &workspace.sel[rank_index],
                aq,
                ad,
                &mut workspace.up_out[rank_index],
                n_sel,
                up_bank.in_features,
                up_bank.local_out,
                up_bank.row_bytes,
                up_bank.expert_bytes,
                0,
                0,
                up_bank.slot_major,
            )?;
            // Fused macro-scaled SwiGLU that EMITS q8_1 directly — down consumes it with no
            // separate quantize launch. act[rank] IS down canonical shard `rank_index`'s
            // input-column window (the geometry gift; see the method doc).
            {
                let Nvfp4DeviceRoutesWorkspace {
                    gate_out,
                    up_out,
                    sel,
                    act_q,
                    act_d,
                    ..
                } = &mut *workspace;
                engine.silu_mul_scaled_q8_1_sel_into(
                    &gate_out[rank_index],
                    &up_out[rank_index],
                    &experts.macros_gate_dev[rank_index],
                    &experts.macros_up_dev[rank_index],
                    &sel[rank_index],
                    activation_limit,
                    &mut act_q[rank_index],
                    &mut act_d[rank_index],
                    local_out,
                    n_sel,
                )?;
            }
            let shard = &experts.down[rank_index];
            if shard.device_rank != rank_index || shard.local_in != local_out {
                return Err(
                    "NVFP4 device routes: down canonical shard placement drifted from \
                     the gate/up column split"
                        .into(),
                );
            }
            // PROGRAM 3 (`MEMRA_NVFP4_SEL_DOWN8`): the down sweep and the route-weight combine
            // in ONE launch, one warp per SLOT instead of one warp per (row, slot), and the
            // `n_sel x out_f` partial round trip gone. Device-routed only — the host-routed arm
            // folds the macro into `combine_w` instead of reading `md` on device — and
            // slot-major only, read off the shard. `nsb <= 32` is the fit-block class the reduce
            // identity is argued at. Its own door, priced LAST and only on green gates for the
            // programs beneath it (lane mandate, milestone 5).
            let down8 =
                device_routed && sel_down8_on() && shard.slot_major && (shard.local_in >> 5) <= 32;
            // ENGAGEMENT RECEIPT for PROGRAM 3, one line per distinct combo. `device_routed`
            // and `nsb <= 32` are printed because they are the two eligibility conditions that
            // can silently disqualify the arm on a geometry or a route the operator did not
            // expect -- exactly the case where a flat perf row would be misread as "no win".
            {
                static SEEN_D8: std::sync::Mutex<Vec<(bool, bool, bool, bool)>> =
                    std::sync::Mutex::new(Vec::new());
                let combo = (down8, sel_down8_on(), device_routed, shard.slot_major);
                let mut seen = SEEN_D8.lock().unwrap();
                if !seen.contains(&combo) {
                    seen.push(combo);
                    // `door_source` is what makes this line a DEFAULT-flip receipt rather than
                    // only an engagement receipt: `door=true door_source=default-on` is the
                    // flip doing the work, `env=1` is a recipe doing it, and
                    // `down8=false door=true` is the silent-no-op shape that PROGRAM 1's
                    // default exists to prevent.
                    eprintln!(
                        "[nvfp4-sweep] down8={} door={} door_source={} device_routed={} \
                         slot_major={} nsb={} in_class={} n_sel={n_sel}",
                        down8,
                        sel_down8_on(),
                        sel_down8_source().1,
                        device_routed,
                        shard.slot_major,
                        shard.local_in >> 5,
                        (shard.local_in >> 5) <= 32
                    );
                }
            }
            if down8 {
                let Nvfp4DeviceRoutesWorkspace {
                    sel,
                    act_q,
                    act_d,
                    route_w,
                    accumulator,
                    ..
                } = &mut *workspace;
                engine.qmatvec_nvfp4_sel_down8_into(
                    &shard.bank,
                    &sel[rank_index],
                    &act_q[rank_index],
                    &act_d[rank_index],
                    &route_w[rank_index],
                    &experts.macros_down_dev[rank_index],
                    &mut accumulator[rank_index],
                    n_sel,
                    shard.local_in,
                    shard.out_features,
                    shard.row_bytes,
                    shard.expert_bytes,
                    local_out,
                    local_out / 32,
                    shard.slot_major,
                )?;
            } else {
                let Nvfp4DeviceRoutesWorkspace {
                    sel,
                    act_q,
                    act_d,
                    partial,
                    ..
                } = &mut *workspace;
                engine.qmatvec_nvfp4_sel_into(
                    &shard.bank,
                    &sel[rank_index],
                    &act_q[rank_index],
                    &act_d[rank_index],
                    &mut partial[rank_index],
                    n_sel,
                    shard.local_in,
                    shard.out_features,
                    shard.row_bytes,
                    shard.expert_bytes,
                    local_out,
                    local_out / 32,
                    shard.slot_major,
                )?;
            }
            // Route-weight accumulation: axpy_rows_seq keeps the exact sequential per-pair
            // FP chain of the reset + n_sel axpy launches in ONE launch. Device-routed calls
            // fold the down macro in-kernel from the device selection. (down8 already produced
            // the accumulator inside the sweep.)
            if !down8 {
                let Nvfp4DeviceRoutesWorkspace {
                    partial,
                    combine_w,
                    route_w,
                    sel,
                    accumulator,
                    ..
                } = &mut *workspace;
                if device_routed {
                    engine.axpy_rows_seq_md_into(
                        &partial[rank_index],
                        &route_w[rank_index],
                        &experts.macros_down_dev[rank_index],
                        &sel[rank_index],
                        &mut accumulator[rank_index],
                        experts.input_width,
                        n_sel,
                    )?;
                } else {
                    engine.axpy_rows_seq_into(
                        &partial[rank_index],
                        &combine_w[rank_index],
                        &mut accumulator[rank_index],
                        experts.input_width,
                        n_sel,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Device-IO twin of `run_tensor_parallel_routes_nvfp4_device`: the layer input arrives as
    /// a device row on the model engine `e` and the combined output returns as a fresh
    /// `e`-context row — no host round-trip, no host stream sync. Ordering is evented (the v2
    /// attention discipline): `ev_entry` is recorded on `e`'s stream AFTER the caller queued
    /// the input's producer; each rank waits it before its peer read; the root reduce waits
    /// every rank's done event; `e` waits the root's done event before copying out. The
    /// program bytes are identical to the host-IO twin — dtoh/htod and dtod preserve f32 bits.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub fn run_tensor_parallel_routes_nvfp4_device_io(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        e: &Engine,
        input_dev: &crate::CudaSlice<f32>,
        selected: &[usize],
        route_weights: &[f32],
        experts_per_token: usize,
        activation_limit: Option<f32>,
    ) -> Result<crate::CudaSlice<f32>, Box<dyn std::error::Error>> {
        if input_dev.len() != experts.input_width {
            return Err(format!(
                "NVFP4 device-io routes input {} != width {}",
                input_dev.len(),
                experts.input_width
            )
            .into());
        }
        if selected.len() != experts_per_token || route_weights.len() != experts_per_token {
            return Err(format!(
                "NVFP4 device-io routes selected={} weights={} != experts/token {experts_per_token}",
                selected.len(),
                route_weights.len(),
            )
            .into());
        }
        if !route_weights.iter().all(|weight| weight.is_finite()) {
            return Err("NVFP4 device route weights contain a non-finite value".into());
        }
        let world = self.ranks.len();
        if world != NVFP4_CANONICAL_ROW_SHARDS {
            return Err(format!(
                "NVFP4 device routes require world == canonical shard grid \
                 ({NVFP4_CANONICAL_ROW_SHARDS}), got {world}"
            )
            .into());
        }
        let local_out = experts.expert_width / world;
        let n_sel = experts_per_token;

        static TIMING_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static TIMING_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let timing = std::env::var("MEMRA_STEP_TP_TIMING").as_deref() == Ok("1");
        let started = timing.then(std::time::Instant::now);

        let mut workspace_guard = experts
            .device_workspace
            .lock()
            .map_err(|_| "NVFP4 device routes workspace lock is poisoned")?;
        if workspace_guard.is_none() {
            drop(workspace_guard);
            // Build through the host-IO ensure path exactly once: run it with a zero input.
            // Cheaper than duplicating the init; the first real call overwrites everything.
            let zero = vec![0.0f32; experts.input_width];
            let zero_sel = vec![0usize; n_sel];
            let zero_w = vec![0.0f32; n_sel];
            let _ = self.run_tensor_parallel_routes_nvfp4_device(
                experts,
                &zero,
                &zero_sel,
                &zero_w,
                n_sel,
                activation_limit,
            )?;
            workspace_guard = experts
                .device_workspace
                .lock()
                .map_err(|_| "NVFP4 device routes workspace lock is poisoned")?;
        }
        let workspace = workspace_guard
            .as_mut()
            .expect("NVFP4 device routes workspace initialized above");
        if workspace.n_sel != n_sel {
            return Err(format!(
                "NVFP4 device routes experts/token changed: workspace {} != call {n_sel}",
                workspace.n_sel
            )
            .into());
        }
        for &expert in selected {
            if expert >= experts.expert_count {
                return Err(format!(
                    "NVFP4 device selected expert {expert} outside 0..{}",
                    experts.expert_count
                )
                .into());
            }
        }
        let sel_i32 = selected
            .iter()
            .map(|&expert| expert as i32)
            .collect::<Vec<_>>();

        // Entry fence: e's stream position covers the input's producer AND every consumer of
        // the previous layer's output (queued on e's stream before this call), guarding the
        // workspace reuse exactly like the v2 attention driver.
        if let Some((_, device)) = workspace.ev_entry.as_ref() {
            if *device != e.ctx().ordinal() {
                return Err("NVFP4 device-io routes engine changed".into());
            }
        } else {
            let _main = e.gpu.enter_main()?;
            workspace.ev_entry = Some((e.ctx().new_event(None)?, e.ctx().ordinal()));
        }
        {
            let _main = e.gpu.enter_main()?;
            let (ev_entry, _) = workspace.ev_entry.as_ref().expect("entry event set above");
            ev_entry.record(&e.stream())?;
        }
        for (rank_index, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            let (ev_entry, _) = workspace.ev_entry.as_ref().expect("entry event set above");
            engine.stream().wait(ev_entry)?;
            {
                let mut destination = workspace.input[rank_index].slice_mut(0..experts.input_width);
                engine
                    .stream()
                    .memcpy_dtod(&input_dev.slice(0..experts.input_width), &mut destination)?;
            }
            {
                let Nvfp4DeviceRoutesWorkspace {
                    input, in_q, in_d, ..
                } = &mut *workspace;
                engine.quantize_q8_1_into(
                    &input[rank_index],
                    1,
                    experts.input_width,
                    &mut in_q[rank_index],
                    &mut in_d[rank_index],
                )?;
            }
        }
        self.nvfp4_routes_batched_sweeps(
            experts,
            workspace,
            selected,
            route_weights,
            &sel_i32,
            local_out,
            n_sel,
            activation_limit,
            false,
        )?;

        // Evented combine: rank done events replace the host stream syncs, the reduce runs on
        // the root stream in canonical shard order, and e copies the combined row out behind
        // the root's done event.
        // rank0 == root: its own stream order already covers its sweep; only the PEER
        // ranks need the record/wait pair (host-op diet at the #1 eager seam, 2026-08-21).
        for (rank_index, engine) in self.ranks.iter().enumerate().skip(1) {
            let _main = engine.gpu.enter_main()?;
            workspace.ev_rank[rank_index].record(&engine.stream())?;
        }
        if moe_direct_on() && self.ranks.len() == 2 {
            // DIRECT JOIN: rank1's accumulator is root-resident (P2P single-store pass);
            // rank0's is root-stream-ordered. One root event + rank1's own event order
            // the model engine's single add — same operand order as root's add
            // (accumulator[0] + accumulator[1]): BIT-IDENTICAL. Output is a FRESH
            // e-context row (NOT an alias of ws state — the reverted zero-copy handoff's
            // hazard class does not apply).
            {
                let root = &self.ranks[0];
                let _main = root.gpu.enter_main()?;
                workspace
                    .ev_done
                    .as_ref()
                    .expect("device routes done event")
                    .record(&root.stream())?;
            }
            let _main = e.gpu.enter_main()?;
            e.stream().wait(
                workspace
                    .ev_done
                    .as_ref()
                    .expect("device routes done event"),
            )?;
            for ev in workspace.ev_rank.iter().skip(1) {
                e.stream().wait(ev)?;
            }
            let mut output = e.uninit(experts.input_width)?;
            e.add(
                &workspace.accumulator[0],
                &workspace.accumulator[1],
                &mut output,
                experts.input_width,
            )?;
            let output = output;
            if let Some(started) = started {
                use std::sync::atomic::Ordering;
                let ns = TIMING_NS
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed)
                    + started.elapsed().as_nanos() as u64;
                let calls = TIMING_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
                if calls.is_multiple_of(430) {
                    eprintln!(
                        "[nvfp4-dev-routes-direct-timing] calls={calls} total_ms={:.1} avg_us={:.1}",
                        ns as f64 / 1.0e6,
                        ns as f64 / calls as f64 / 1.0e3,
                    );
                }
            }
            return Ok(output);
        }
        {
            let root = &self.ranks[0];
            let _main = root.gpu.enter_main()?;
            for ev in workspace.ev_rank.iter().skip(1) {
                root.stream().wait(ev)?;
            }
            root.stream()
                .memcpy_dtod(&workspace.accumulator[1], &mut workspace.remote)?;
            {
                let Nvfp4DeviceRoutesWorkspace {
                    accumulator,
                    remote,
                    combined,
                    ..
                } = &mut *workspace;
                root.add(&accumulator[0], remote, combined, experts.input_width)?;
            }
            workspace
                .ev_done
                .as_ref()
                .expect("device routes done event")
                .record(&root.stream())?;
        }
        let output = {
            let _main = e.gpu.enter_main()?;
            e.stream().wait(
                workspace
                    .ev_done
                    .as_ref()
                    .expect("device routes done event"),
            )?;
            // (Zero-copy clone handoff REVERTED 2026-08-21: identity mismatch in the
            // routes-diet bisect. The alloc+copy stays until the hazard is understood.)
            let mut output = e.uninit(experts.input_width)?;
            e.stream().memcpy_dtod(
                &workspace.combined.slice(0..experts.input_width),
                &mut output.slice_mut(0..experts.input_width),
            )?;
            output
        };
        if let Some(started) = started {
            use std::sync::atomic::Ordering;
            let ns = TIMING_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed)
                + started.elapsed().as_nanos() as u64;
            let calls = TIMING_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
            if calls.is_multiple_of(430) {
                eprintln!(
                    "[nvfp4-dev-routes-io-timing] calls={calls} total_ms={:.1} avg_us={:.1}",
                    ns as f64 / 1.0e6,
                    ns as f64 / calls as f64 / 1.0e3,
                );
            }
        }
        Ok(output)
    }

    /// Device-routed twin of `run_tensor_parallel_routes_nvfp4_device_io`: the selection and
    /// route weights arrive as the device router's e-context outputs — the per-layer host
    /// logits readback disappears. The fresh router outputs are staged into persistent
    /// e-context buffers on e's stream (never-free discipline) before the entry event; each
    /// rank peer-reads them behind it. The down-macro fold happens in-kernel.
    #[allow(clippy::too_many_arguments)]
    /// Prestage the routed-expert input: pull the shared row to every rank and quantize it
    /// there, WITHOUT the selection — callable before the router so the rank chains overlap
    /// it. No-op (returns false) when the workspace is not built yet or the door is off;
    /// the routed run then does its own staging as before.
    pub fn nvfp4_routes_prestage(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        e: &Engine,
        input_dev: &crate::CudaSlice<f32>,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        self.nvfp4_routes_prestage_with(experts, e, input_dev, |_, _, _, _| Ok(false))
    }

    /// `nvfp4_routes_prestage` with a PEER-ROUTER hook: after rank1's input pull +
    /// quantize, the hook may compute rank1's route selection LOCALLY (replicated router —
    /// deterministic kernels on identical input bits produce identical sel/w, so the
    /// selection is bit-equal to the root's). Returns true when it wrote sel/route_w; the
    /// routed run then skips rank1's sel pull.
    pub fn nvfp4_routes_prestage_with(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        e: &Engine,
        input_dev: &crate::CudaSlice<f32>,
        rank1_router: impl FnOnce(
            &Engine,
            &crate::CudaSlice<f32>,
            &mut crate::CudaSlice<i32>,
            &mut crate::CudaSlice<f32>,
        ) -> Result<bool, Box<dyn std::error::Error>>,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        if !routes_prestage_on() || step_tp_graph_enabled()? {
            return Ok(false);
        }
        if input_dev.len() != experts.input_width {
            return Err("NVFP4 prestage input width mismatch".into());
        }
        let mut workspace_guard = experts
            .device_workspace
            .lock()
            .map_err(|_| "NVFP4 device routes workspace lock is poisoned")?;
        let Some(workspace) = workspace_guard.as_mut() else {
            return Ok(false);
        };
        if workspace.ev_input.is_none() {
            let _main = e.gpu.enter_main()?;
            workspace.ev_input = Some((e.ctx().new_event(None)?, e.ctx().ordinal()));
        } else if workspace.ev_input.as_ref().map(|(_, d)| *d) != Some(e.ctx().ordinal()) {
            return Err("NVFP4 prestage engine changed".into());
        }
        {
            let _main = e.gpu.enter_main()?;
            let (ev, _) = workspace.ev_input.as_ref().expect("armed above");
            ev.record(&e.stream())?;
        }
        for (rank_index, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            let (ev, _) = workspace.ev_input.as_ref().expect("armed above");
            engine.stream().wait(ev)?;
            {
                let mut destination = workspace.input[rank_index].slice_mut(0..experts.input_width);
                engine
                    .stream()
                    .memcpy_dtod(&input_dev.slice(0..experts.input_width), &mut destination)?;
            }
            {
                let Nvfp4DeviceRoutesWorkspace {
                    input, in_q, in_d, ..
                } = &mut *workspace;
                engine.quantize_q8_1_into(
                    &input[rank_index],
                    1,
                    experts.input_width,
                    &mut in_q[rank_index],
                    &mut in_d[rank_index],
                )?;
            }
        }
        if self.ranks.len() == 2 {
            let rank1 = &self.ranks[1];
            let _r1 = rank1.gpu.enter_main()?;
            let Nvfp4DeviceRoutesWorkspace {
                input,
                sel,
                route_w,
                ..
            } = &mut *workspace;
            let (in1, rest_sel) = (&input[1], &mut sel[1]);
            if rank1_router(rank1, in1, rest_sel, &mut route_w[1])? {
                workspace.rank1_routed = true;
            }
        }
        workspace.prestaged = true;
        Ok(true)
    }

    /// STEP TP2 GEMM PRIME (`MEMRA_STEP_GEMM_PRIME`, 2026-08-27, TTFT lane): one grouped
    /// f16 GEMM per projection over the RESIDENT NVFP4 banks for a prime chunk of `t` tokens.
    ///
    /// WHY: the t-row walk primes a 4,092-token prompt in 19.8 s at its widest (GEMV-bound) and
    /// the generic batch prime's decode-class MoE takes 240 s; the CUTLASS sizing rows put
    /// GEMM-class expert math at 170-270 TFLOP/s on this silicon, i.e. a sub-second cold prime.
    /// This reuses the grouped f16 lane end to end (`moe_f16g_act` -> `moe_f16_grouped`
    /// direct-from-NVFP4 -> silu pairs -> grouped down) once per RANK against that rank's bank
    /// half: gate/up are column-halves (silu runs on matching halves), down is the canonical
    /// row-shard pair producing partials joined in the pinned shard order, and the final
    /// weighted scatter runs a fixed slot-0..n_used-1 sum per token - no atomics anywhere.
    /// Per-expert NVFP4 macro scales land where they must: gate/up BEFORE silu (nonlinear),
    /// down folded into the scatter weight.
    ///
    /// NUMERIC CLASS: the f16-mirror grouped-prefill class other families already serve -
    /// admission is the prefill-KV acceptance gate plus the ship-shape tape, not byte identity.
    #[allow(clippy::too_many_arguments)]
    /// MEMRA_MOE_DETERM_STAGE=1: checksum a stage's device buffer so two back-to-back calls of the
    /// grouped routine can be compared STAGE BY STAGE. The routine's OUTPUT is nondeterministic above
    /// ~400 tokens on the direct lane (1.9e-7 / 99% of elements at t=4096) while its GEMM kernels are
    /// bit-exact in isolation, so the divergence enters somewhere between. The first stage whose
    /// checksum differs across the two calls is where.
    ///
    /// Sum-of-bits, not sum-of-floats: float addition would itself reorder and could mask exactly the
    /// class of difference being hunted.
    fn determ_stage_bytes(v: &[u8]) -> u64 {
        v.iter().fold(0u64, |a, b| {
            a.wrapping_mul(1_000_003).wrapping_add(*b as u64)
        })
    }

    /// Checksum an i32 index/offset buffer. The CSR, the active-expert ids and the group
    /// offsets are inputs the gate kernel dereferences just as much as the activations are;
    /// leaving them unchecksummed is what let "identical inputs, different output" stand on a
    /// SUBSET of the inputs for six rounds of this investigation.
    fn determ_stage_i32(v: &[i32]) -> u64 {
        v.iter().fold(0u64, |a, b| {
            a.wrapping_mul(1_000_003).wrapping_add(*b as u32 as u64)
        })
    }

    fn determ_stage_sum(v: &[f32]) -> u64 {
        v.iter().fold(0u64, |a, x| {
            a.wrapping_mul(1_000_003).wrapping_add(x.to_bits() as u64)
        })
    }

    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub fn run_tensor_parallel_routes_nvfp4_prime_grouped(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        e: &Engine,
        z_t: &crate::CudaSlice<f32>,
        t: usize,
        sel: &[i32],
        w: &[f32],
        n_used: usize,
        activation_limit: Option<f32>,
    ) -> Result<crate::CudaSlice<f32>, Box<dyn std::error::Error>> {
        let world = self.ranks.len();
        if world != NVFP4_CANONICAL_ROW_SHARDS {
            return Err("NVFP4 grouped prime requires the canonical 2-shard grid".into());
        }
        // The dequant must read the layout the bank was BUILT in (feeding slot-major bytes to
        // the v1 kernel was a garbage-output bug this line exists for). Taken from the BANK,
        // never from the environment: EP2 banks are always slot-major, TP shard banks are
        // slot-major only under PROGRAM 1 (`MEMRA_NVFP4_BANK_SM`). All three banks share one
        // decision at build (`nvfp4_repack_bank_matrix`), and the assert below refuses to run a
        // prime over banks that disagree instead of silently priming one of them wrong.
        //
        // THIS IS THE LINE THE 2026-08-29 CORRUPTION WENT THROUGH. `QT_NVFP4_V2` selects the
        // `kq_fetch` branch whose two prefetch callers omitted `in_f`; the codes stayed right
        // and the per-16 scale came from inside the packed-codes region, so the prime produced
        // fluent WRONG text. No v2 gate had ever run this GEMM. It is now covered device-side by
        // `nvfp4-bank-oracle` (both step37 layer geometries, all four tile forms) and end-to-end
        // by a prefill-heavy byte gate. Keep both: a decode-only byte gate proved nothing here.
        let slot_major = experts.gate.iter().all(|b| b.slot_major)
            && experts.up.iter().all(|b| b.slot_major)
            && experts.down.iter().all(|b| b.slot_major);
        let any_slot_major = experts.gate.iter().any(|b| b.slot_major)
            || experts.up.iter().any(|b| b.slot_major)
            || experts.down.iter().any(|b| b.slot_major);
        if any_slot_major != slot_major {
            return Err(
                "NVFP4 grouped prime: gate/up/down banks disagree on the row layout — \
                        one grouped GEMM cannot serve two byte maps"
                    .into(),
            );
        }
        let bank_qt = if slot_major {
            crate::QT_NVFP4_V2
        } else {
            crate::QT_NVFP4
        };
        let width = experts.input_width;
        let n_expert = experts.expert_count;
        let n_pairs = t * n_used;
        if sel.len() < n_pairs || w.len() < n_pairs || z_t.len() < t * width {
            return Err("NVFP4 grouped prime geometry".into());
        }
        // MEMRA_PRIME_PROF=1 sub-split of the grouped prime (2026-08-28). The [moe-prof] mark
        // around this whole call reads 90% of the MoE bucket, but the call is not just GEMMs:
        // it host-builds the CSR, allocates ~6 large device buffers per rank per layer (z_r is
        // 67 MB, act is 84 MB at t=4096), and does 5 H2D copies per rank. Tile form, occupancy,
        // padding, B double-buffering and register pressure have ALL come back null, which is
        // the signature of time that is not in the kernel. So measure HOST wall with no syncs
        // for the build and the issue, and let the join wait absorb the GPU time: host-bound and
        // GPU-bound then read differently instead of summing into one opaque number.
        let gprof = std::env::var("MEMRA_PRIME_PROF").as_deref() == Ok("1") && t >= 16;
        let g_t0 = std::time::Instant::now();
        // CSR: expert-major pair lists. Host-built - prime is chunk-granular, and the router
        // selections arrive host-side from the sigmoid router oracle.
        let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); n_expert];
        for (p, &s_id) in sel.iter().take(n_pairs).enumerate() {
            let s_id = s_id as usize;
            if s_id >= n_expert {
                return Err(format!("grouped prime selection {s_id} >= {n_expert}").into());
            }
            buckets[s_id].push(p as i32);
        }
        let mut ex_ids: Vec<i32> = Vec::new();
        let mut ex_off: Vec<i32> = vec![0];
        let mut ex_pairs: Vec<i32> = Vec::new();
        for (e_id, b) in buckets.iter().enumerate() {
            if !b.is_empty() {
                ex_ids.push(e_id as i32);
                ex_pairs.extend_from_slice(b);
                ex_off.push(ex_pairs.len() as i32);
            }
        }
        let n_active = ex_ids.len();
        if n_active == 0 {
            return e.zeros(t * width);
        }
        if n_active > 512 {
            return Err("grouped prime n_active > 512 (direct lane cap)".into());
        }
        let csr_tok: Vec<i32> = ex_pairs.iter().map(|&p| p / n_used as i32).collect();
        // pair-id -> CSR row: lets the fused tail read the partials in place, so the prime skips
        // a whole [n_pairs, width] permute (532 MB read + write per rank per layer at 4k).
        let mut inv = vec![0i32; n_pairs];
        for (row, &pair) in ex_pairs.iter().enumerate() {
            inv[pair as usize] = row as i32;
        }
        // Per-CSR-row gate/up macro scales (before silu); down macro folds into the scatter w.
        let mg: Vec<f32> = ex_pairs
            .iter()
            .map(|&p| experts.macros_gate[sel[p as usize] as usize])
            .collect();
        let mu: Vec<f32> = ex_pairs
            .iter()
            .map(|&p| experts.macros_up[sel[p as usize] as usize])
            .collect();
        let wd: Vec<f32> = (0..n_pairs)
            .map(|p| w[p] * experts.macros_down[sel[p] as usize])
            .collect();
        // Pointer tables: built on first use and kept on the bank. Resident banks never move,
        // so the old per-rank-per-LAYER rebuild+upload of 3*n_expert u64s was pure prime-path
        // host churn (45 layers x 2 ranks x 864 entries per prime).
        {
            let mut tabs = experts
                .prime_tables
                .lock()
                .map_err(|_| "grouped prime table cache is poisoned")?;
            if tabs.len() != world {
                tabs.clear();
                for rank in 0..world {
                    let engine = &self.ranks[rank];
                    let _main = engine.gpu.enter_main()?;
                    let (gb, ub, db) =
                        (&experts.gate[rank], &experts.up[rank], &experts.down[rank]);
                    let mut tab = vec![0u64; 3 * n_expert];
                    {
                        use cudarc::driver::DevicePtr;
                        let stream = engine.stream();
                        let (pg, _g0) = gb.bank.device_ptr(&stream);
                        let (pu, _g1) = ub.bank.device_ptr(&stream);
                        let (pd, _g2) = db.bank.device_ptr(&stream);
                        for ex in 0..n_expert {
                            tab[ex] = pg + (ex * gb.expert_bytes) as u64;
                            tab[n_expert + ex] = pu + (ex * ub.expert_bytes) as u64;
                            tab[2 * n_expert + ex] = pd + (ex * db.expert_bytes) as u64;
                        }
                    }
                    tabs.push(engine.htod_u64(&tab)?);
                }
            }
        }
        let g_csr = g_t0.elapsed().as_secs_f64() * 1e3;
        let g_t1 = std::time::Instant::now();
        // WHAT ARE THESE RANKS, ACTUALLY (2026-08-28)? The grouped MoE measures join ~ span_sum
        // (strictly serialized) at t=4096 while the same kernel hits 40 TFLOP/s standalone, and
        // one intervention based on cudarc's peer-copy event was refuted. Before proposing an
        // eleventh mechanism, verify the premise the whole question rests on: that the two ranks
        // are on DISTINCT devices, contexts and streams. If they share any of those, the
        // serialization needs no further explanation. One line per process.
        {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if gprof && !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                for rank in 0..world {
                    let e_r = &self.ranks[rank];
                    let _m = e_r.gpu.enter_main();
                    eprintln!(
                        "[rank-id] rank={rank} ordinal={} ctx={:?} stream={:?} root_ordinal={} \
                         root_stream={:?}",
                        e_r.ctx().ordinal(),
                        std::sync::Arc::as_ptr(e_r.ctx()),
                        e_r.stream().cu_stream(),
                        e.ctx().ordinal(),
                        e.stream().cu_stream(),
                    );
                }
            }
        }

        let mut partials: Vec<crate::CudaSlice<f32>> = Vec::with_capacity(world);
        let mut ev_rank: Vec<CudaEvent> = Vec::with_capacity(world);
        let mut ev_head: Vec<CudaEvent> = Vec::with_capacity(world);
        let mut ev_tail_prof: Vec<CudaEvent> = Vec::with_capacity(world);
        for rank in 0..world {
            let engine = &self.ranks[rank];
            let _main = engine.gpu.enter_main()?;
            if gprof {
                // CU_EVENT_DEFAULT, not None: cudarc's new_event(None) creates the event with
                // CU_EVENT_DISABLE_TIMING, and cuEventElapsedTime then returns INVALID_HANDLE.
                // That is what failed every span query for two build cycles — the ordering
                // events below correctly keep the default, since they are never timed.
                let h = engine
                    .ctx()
                    .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                h.record(&engine.stream())?;
                ev_head.push(h);
            }
            // The grouped-MoE FFI's raw launches follow the RUNTIME API's current device, not
            // the pushed driver context — bind it per rank or rank-1 calls die InvalidValue.
            engine.bind_runtime_device(engine.ctx().ordinal() as i32)?;
            let gb = &experts.gate[rank];
            let ub = &experts.up[rank];
            let db = &experts.down[rank];
            if db.device_rank != rank {
                return Err("grouped prime: down shard placement drifted".into());
            }
            let local_ff = gb.local_out;
            if ub.local_out != local_ff || db.local_in != local_ff || db.out_features != width {
                return Err("grouped prime: bank width mismatch".into());
            }
            // All of the rank's host-side staging lands before its first kernel, so the
            // launch chain below issues without host copies interleaved.
            let csr_tok_d = engine.htod_i32(&csr_tok)?;
            let exi_d = engine.htod_i32(&ex_ids)?;
            let exoff_d = engine.htod_i32(&ex_off)?;
            let mg_d = engine.htod(&mg)?;
            let mu_d = engine.htod(&mu)?;
            // Per-rank pointer table into the bank shards, slot-major like DevExps::ptr_row.
            let tabs_guard = experts
                .prime_tables
                .lock()
                .map_err(|_| "grouped prime table cache is poisoned")?;
            let tab_d = &tabs_guard[rank];
            let mut z_r = engine.uninit(t * width)?;
            {
                let mut dst = z_r.slice_mut(0..t * width);
                engine
                    .stream()
                    .memcpy_dtod(&z_t.slice(0..t * width), &mut dst)?;
            }
            let dstage = std::env::var("MEMRA_MOE_DETERM_STAGE").as_deref() == Ok("1") && t >= 16;
            let (z16, zs) = engine.moe_f16g_act(&z_r, Some(&csr_tok_d), width, n_pairs)?;
            if dstage {
                // z16 is the GEMM's actual DATA input and is a byte buffer; checksumming only
                // z_r and zs left "identical inputs" unestablished and produced a localization
                // that outran the measurement. Checksum it as bytes.
                let zr = engine.dtoh(&z_r)?;
                let zsv = engine.dtoh(&zs)?;
                let z16v = engine.dtoh_u8(&z16)?;
                eprintln!(
                    "[determ-stage] rank={rank} t={t} z_r={:016x} zs={:016x} z16={:016x}",
                    Self::determ_stage_sum(&zr),
                    Self::determ_stage_sum(&zsv),
                    Self::determ_stage_bytes(&z16v)
                );
            }
            if dstage {
                // INPUT CLOSURE. Everything the gate kernel dereferences, plus the launch
                // geometry that decides how it is summed, checksummed in ONE place. A kernel
                // proven bit-deterministic on live data, with no atomics, can only diverge if
                // (A) some byte it reads differs, (B) the launch differs, or (C) it reads
                // outside its declared inputs. This closes A and B; C is what compute-sanitizer
                // is for. Partial input sets are how the divergence kept retreating into the
                // part that was never measured.
                engine.stream().synchronize()?;
                let csr_v = engine.dtoh_i32(&csr_tok_d)?;
                let exi_v = engine.dtoh_i32(&exi_d)?;
                let exo_v = engine.dtoh_i32(&exoff_d)?;
                let mg_v = engine.dtoh(&mg_d)?;
                let mu_v = engine.dtoh(&mu_d)?;
                let tab_v = engine.dtoh_u64(tab_d)?;
                eprintln!(
                    "[determ-closure] rank={rank} t={t} csr_tok={:016x} exi={:016x} exoff={:016x}                      ex_off_host={:016x} mg={:016x} mu={:016x} tab={:016x} | n_active={n_active}                      n_pairs={n_pairs} width={width} local_ff={local_ff} n_expert={n_expert}                      qt={bank_qt} rb={}",
                    Self::determ_stage_i32(&csr_v),
                    Self::determ_stage_i32(&exi_v),
                    Self::determ_stage_i32(&exo_v),
                    Self::determ_stage_i32(&ex_off),
                    Self::determ_stage_sum(&mg_v),
                    Self::determ_stage_sum(&mu_v),
                    tab_v
                        .iter()
                        .fold(0u64, |a, b| a.wrapping_mul(1_000_003).wrapping_add(*b)),
                    gb.row_bytes
                );
                // The resident weight bank is the GEMM's OTHER operand and was never checked.
                // Opt-in because it is a ~424 MB dtoh per rank per layer.
                if std::env::var("MEMRA_MOE_DETERM_BANK").as_deref() == Ok("1") {
                    let bank_v = engine.dtoh_u8(&gb.bank)?;
                    eprintln!(
                        "[determ-closure] rank={rank} t={t} gate_bank={:016x} bytes={}",
                        Self::determ_stage_bytes(&bank_v),
                        bank_v.len()
                    );
                }
            }
            let mut g = engine.moe_f16_grouped(
                tab_d,
                0,
                n_expert,
                &exi_d,
                &ex_off,
                &exoff_d,
                &z16,
                &zs,
                width,
                local_ff,
                n_active,
                n_pairs,
                bank_qt,
                gb.row_bytes,
            )?;
            engine.scale_rows(&mut g, &mg_d, local_ff, n_pairs)?;
            let mut u = engine.moe_f16_grouped(
                tab_d,
                1,
                n_expert,
                &exi_d,
                &ex_off,
                &exoff_d,
                &z16,
                &zs,
                width,
                local_ff,
                n_active,
                n_pairs,
                bank_qt,
                ub.row_bytes,
            )?;
            engine.scale_rows(&mut u, &mu_d, local_ff, n_pairs)?;
            // step35 routed SwiGLU clamp (per-layer; live only on layers 43/44 for this
            // family): min(silu(g), lim) * clamp(u, +-lim). Dropping it was the second
            // correctness bug of the first engaged run.
            let act = match activation_limit.filter(|l| *l > 1e-6) {
                Some(lim) => {
                    let mut a = engine.uninit(n_pairs * local_ff)?;
                    engine.swiglu_clamped_mul_scaled(
                        &g,
                        &u,
                        1.0,
                        1.0,
                        lim,
                        &mut a,
                        n_pairs * local_ff,
                    )?;
                    a
                }
                None => engine.moe_pairs_silu_mul(&g, &u, n_pairs * local_ff)?,
            };
            if dstage {
                let gv = engine.dtoh(&g)?;
                let uv = engine.dtoh(&u)?;
                let av = engine.dtoh(&act)?;
                // A SUM tells you THAT gate differs; it does not tell you HOW. ULP-dense diffs
                // (nearly every element, ~1e-8) are an ordering/precision class; a handful of
                // huge ones are a corruption class. They need different hunts, so measure the
                // shape here instead of inferring it later.
                let key = (rank, t);
                let mut prev_map = DETERM_PREV
                    .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
                    .lock()
                    .map_err(|_| "determ prev map poisoned")?;
                let shape = match prev_map.get(&key) {
                    Some(prev) if prev.len() == gv.len() => {
                        let mut md = 0.0f32;
                        let mut n_diff = 0usize;
                        let mut n_big = 0usize;
                        for (a, b) in prev.iter().zip(gv.iter()) {
                            let d = (a - b).abs();
                            if d > 0.0 {
                                n_diff += 1;
                            }
                            if d > 1e-3 {
                                n_big += 1;
                            }
                            if d > md {
                                md = d;
                            }
                        }
                        format!(
                            " | vs_prev maxdiff={md:.3e} differing={n_diff}/{} big(>1e-3)={n_big}",
                            gv.len()
                        )
                    }
                    _ => String::new(),
                };
                prev_map.insert(key, gv.clone());
                drop(prev_map);
                eprintln!(
                    "[determ-stage] rank={rank} t={t} gate={:016x} up={:016x} silu={:016x}{shape}",
                    Self::determ_stage_sum(&gv),
                    Self::determ_stage_sum(&uv),
                    Self::determ_stage_sum(&av)
                );
            }
            let (a16, a_s) = engine.moe_f16g_act(&act, None, local_ff, n_pairs)?;
            let d_csr = engine.moe_f16_grouped(
                tab_d,
                2,
                n_expert,
                &exi_d,
                &ex_off,
                &exoff_d,
                &a16,
                &a_s,
                local_ff,
                width,
                n_active,
                n_pairs,
                bank_qt,
                db.row_bytes,
            )?;

            // No host sync: both ranks' chains must be in flight before anything waits.
            // The rank's tail event orders the root's cross-device pulls below.
            if dstage {
                engine.stream().synchronize()?;
                let a16v = engine.dtoh_u8(&a16)?;
                let dv = engine.dtoh(&d_csr)?;
                eprintln!(
                    "[determ-stage] rank={rank} t={t} a16={:016x} down_partial={:016x}",
                    Self::determ_stage_bytes(&a16v),
                    Self::determ_stage_sum(&dv)
                );
            }
            let ev = engine.ctx().new_event(None)?;
            ev.record(&engine.stream())?;
            if gprof {
                // Per-rank GPU SPAN (2026-08-28). Keep the tail event; the elapsed time is read
                // AFTER the join sync below. Reading it here returns NOT_READY (the work has only
                // been queued) and cudarc's elapsed_ms synchronizes, which serialized the very
                // ranks this is meant to test: host issue jumped 1.9 ms -> 34-47 ms per call and
                // the join wall fell to match. A probe that changes the schedule measures its own
                // perturbation.
                // CudaEvent is not Clone, so record a second tail event on the same stream —
                // adjacent to `ev`, so it carries the same completion timestamp for timing.
                let tp = engine
                    .ctx()
                    .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                tp.record(&engine.stream())?;
                ev_tail_prof.push(tp);
            }
            ev_rank.push(ev);
            partials.push(d_csr);
        }
        let _main = e.gpu.enter_main()?;
        e.bind_runtime_device(e.ctx().ordinal() as i32)?;
        // Host-only: every rank's chain is queued, nothing has been waited on yet.
        let g_issue = g_t1.elapsed().as_secs_f64() * 1e3;
        let g_t2 = std::time::Instant::now();
        for ev in &ev_rank {
            e.stream().wait(ev)?;
        }
        // Both partials land on the root (rank 1's crosses the link once), then ONE fused pass
        // does join + CSR permute + weight + scatter. Shard order stays pinned as (y0 + y1).
        let mut y0 = e.uninit(n_pairs * width)?;
        {
            let mut dst = y0.slice_mut(0..n_pairs * width);
            e.stream()
                .memcpy_dtod(&partials[0].slice(0..n_pairs * width), &mut dst)?;
        }
        let mut y1 = e.uninit(n_pairs * width)?;
        {
            let mut dst = y1.slice_mut(0..n_pairs * width);
            e.stream()
                .memcpy_dtod(&partials[1].slice(0..n_pairs * width), &mut dst)?;
        }
        let inv_d = e.htod_i32(&inv)?;
        let wd_d = e.htod(&wd)?;
        let mut out = e.uninit(t * width)?;
        e.moe_prime_join_scatter(&y0, &y1, &inv_d, &wd_d, &mut out, width, n_used, t)?;
        if gprof {
            let _ = e.stream().synchronize();
            let g_join = g_t2.elapsed().as_secs_f64() * 1e3;
            // Everything has completed, so both events of every pair are ready and elapsed_ms
            // cannot block. A negative entry means the query itself failed and the row must be
            // read as missing data, never as a zero-length span.
            // cuEventElapsedTime needs the events' OWN context current — computing it under the
            // root's pushed context returned an error for every pair, and the first version
            // swallowed that into -1.0 with no reason attached. Enter each rank's context, and
            // print the failure once so a dead probe can never again look like a zero-length span.
            let mut span_ms: Vec<f32> = Vec::with_capacity(world);
            for (rank, (h, tp)) in ev_head.iter().zip(ev_tail_prof.iter()).enumerate() {
                let guard = self.ranks[rank].gpu.enter_main();
                match guard.and_then(|_g| h.elapsed_ms(tp).map_err(|e| e.into())) {
                    Ok(v) => span_ms.push(v),
                    Err(err) => {
                        static SAID: std::sync::atomic::AtomicBool =
                            std::sync::atomic::AtomicBool::new(false);
                        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            eprintln!("[grp-prof] span query failed on rank {rank}: {err}");
                        }
                        span_ms.push(-1.0);
                    }
                }
            }
            eprintln!(
                "[grp-prof] t={t} n_active={n_active} csr={g_csr:.1}ms issue={g_issue:.1}ms \
                 join={g_join:.1}ms spans={span_ms:?} span_sum={:.1}ms span_max={:.1}ms",
                span_ms.iter().sum::<f32>(),
                span_ms.iter().cloned().fold(0.0f32, f32::max)
            );
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub fn run_tensor_parallel_routes_nvfp4_device_routed(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        e: &Engine,
        input_dev: &crate::CudaSlice<f32>,
        sel_d: &crate::CudaSlice<i32>,
        w_d: &crate::CudaSlice<f32>,
        experts_per_token: usize,
        activation_limit: Option<f32>,
    ) -> Result<crate::CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.run_tensor_parallel_routes_nvfp4_device_routed_prejoin(
            experts,
            e,
            input_dev,
            sel_d,
            w_d,
            experts_per_token,
            activation_limit,
            || Ok(()),
        )
    }

    /// `run_tensor_parallel_routes_nvfp4_device_routed` with a PREJOIN hook: `pre_join`
    /// runs on the host right before the join wait is enqueued on e's stream — work it
    /// issues there (e.g. the shexp overlap) executes WHILE the peer rank finishes its
    /// sweep, instead of after the join. Value-neutral by construction (the hook only
    /// reorders independent host issue).
    #[allow(clippy::too_many_arguments)]
    pub fn run_tensor_parallel_routes_nvfp4_device_routed_prejoin(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        e: &Engine,
        input_dev: &crate::CudaSlice<f32>,
        sel_d: &crate::CudaSlice<i32>,
        w_d: &crate::CudaSlice<f32>,
        experts_per_token: usize,
        activation_limit: Option<f32>,
        pre_join: impl FnOnce() -> Result<(), Box<dyn std::error::Error>>,
    ) -> Result<crate::CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.run_tensor_parallel_routes_nvfp4_device_routed_prejoin_add3(
            experts,
            e,
            input_dev,
            sel_d,
            w_d,
            experts_per_token,
            activation_limit,
            pre_join,
            None,
        )
    }

    /// The prejoin variant with MOE TAIL FUSION M1: when `post_add = Some((sh_raw,
    /// scale_raw))`, the direct-join arm folds the shexp apply into the join add
    /// (`dst = (acc0+acc1) + sh*scale[0]`, exact split-pair sequence) — the caller skips
    /// its apply launch. Raw UVA pointers so no lock is held across the call.
    #[allow(clippy::too_many_arguments)]
    pub fn run_tensor_parallel_routes_nvfp4_device_routed_prejoin_add3(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        e: &Engine,
        input_dev: &crate::CudaSlice<f32>,
        sel_d: &crate::CudaSlice<i32>,
        w_d: &crate::CudaSlice<f32>,
        experts_per_token: usize,
        activation_limit: Option<f32>,
        pre_join: impl FnOnce() -> Result<(), Box<dyn std::error::Error>>,
        post_add: Option<(u64, u64)>,
    ) -> Result<crate::CudaSlice<f32>, Box<dyn std::error::Error>> {
        if input_dev.len() != experts.input_width {
            return Err(format!(
                "NVFP4 device-routed input {} != width {}",
                input_dev.len(),
                experts.input_width
            )
            .into());
        }
        let n_sel = experts_per_token;
        if sel_d.len() < n_sel || w_d.len() < n_sel {
            return Err(format!(
                "NVFP4 device-routed routes sel={} w={} < experts/token {n_sel}",
                sel_d.len(),
                w_d.len()
            )
            .into());
        }
        let world = self.ranks.len();
        if world != NVFP4_CANONICAL_ROW_SHARDS {
            return Err(format!(
                "NVFP4 device routes require world == canonical shard grid \
                 ({NVFP4_CANONICAL_ROW_SHARDS}), got {world}"
            )
            .into());
        }
        let local_out = experts.expert_width / world;

        static TIMING_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static TIMING_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let timing = std::env::var("MEMRA_STEP_TP_TIMING").as_deref() == Ok("1");
        let started = timing.then(std::time::Instant::now);

        let mut workspace_guard = experts
            .device_workspace
            .lock()
            .map_err(|_| "NVFP4 device routes workspace lock is poisoned")?;
        if workspace_guard.is_none() {
            drop(workspace_guard);
            let zero = vec![0.0f32; experts.input_width];
            let zero_sel = vec![0usize; n_sel];
            let zero_w = vec![0.0f32; n_sel];
            let _ = self.run_tensor_parallel_routes_nvfp4_device(
                experts,
                &zero,
                &zero_sel,
                &zero_w,
                n_sel,
                activation_limit,
            )?;
            workspace_guard = experts
                .device_workspace
                .lock()
                .map_err(|_| "NVFP4 device routes workspace lock is poisoned")?;
        }
        let workspace = workspace_guard
            .as_mut()
            .expect("NVFP4 device routes workspace initialized above");
        if workspace.n_sel != n_sel {
            return Err(format!(
                "NVFP4 device routes experts/token changed: workspace {} != call {n_sel}",
                workspace.n_sel
            )
            .into());
        }

        // GRAPH DOOR (MEMRA_STEP_TP_GRAPH=1): the whole rank+root segment replays as one
        // stitched multi-device parent launched on e's stream — no events, no per-token node
        // updates (every address is persistent staging). VALUE-IDENTICAL to the eager path:
        // the children replay exactly the same kernel/copy sequence.
        //
        // GRAPH-LAUNCH HEADROOM GUARD (see spec::GRAPH_LAUNCH_MIN_FREE): below the
        // driver-free floor on the launching device this call falls through to the
        // eager routes path below — the exact body the graph captures, stateless per
        // call — instead of feeding cuGraphLaunch an exhausted card
        // (lane/graph-launch-guard-sweep-20260831).
        if step_tp_graph_enabled()? && step_tp_graph_headroom_ok(e) {
            if workspace.dev_route_e.is_none() {
                let _main = e.gpu.enter_main()?;
                workspace.dev_route_e = Some((
                    e.htod_i32(&vec![0i32; n_sel])?,
                    e.htod(&vec![0.0f32; n_sel])?,
                ));
            }
            if workspace.in_stage_e.is_none() {
                let _main = e.gpu.enter_main()?;
                workspace.in_stage_e = Some(e.htod(&vec![0.0f32; experts.input_width])?);
                workspace.out_stage_e = Some(e.htod(&vec![0.0f32; experts.input_width])?);
            }
            if workspace.routes_graph.is_none() {
                let graph = self.nvfp4_routes_build_graph(
                    experts,
                    workspace,
                    local_out,
                    n_sel,
                    activation_limit,
                )?;
                workspace.routes_graph = Some(graph);
                eprintln!(
                    "[step-tp-graph] routes segment captured: ranks={world} n_sel={n_sel} \
                     children=3 updates=none performance_claim=false"
                );
            }
            let output = {
                let _main = e.gpu.enter_main()?;
                {
                    let (sel_e, w_e) = workspace
                        .dev_route_e
                        .as_mut()
                        .expect("device route staging set above");
                    {
                        let mut dst = sel_e.slice_mut(0..n_sel);
                        e.stream().memcpy_dtod(&sel_d.slice(0..n_sel), &mut dst)?;
                    }
                    {
                        let mut dst = w_e.slice_mut(0..n_sel);
                        e.stream().memcpy_dtod(&w_d.slice(0..n_sel), &mut dst)?;
                    }
                }
                {
                    let in_stage = workspace
                        .in_stage_e
                        .as_mut()
                        .expect("graph staging set above");
                    let mut dst = in_stage.slice_mut(0..experts.input_width);
                    e.stream()
                        .memcpy_dtod(&input_dev.slice(0..experts.input_width), &mut dst)?;
                }
                unsafe {
                    let r = cudarc::driver::sys::cuGraphLaunch(
                        workspace
                            .routes_graph
                            .as_ref()
                            .expect("routes graph built above")
                            .exec,
                        e.stream().cu_stream() as cudarc::driver::sys::CUstream,
                    );
                    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        return Err(format!("routes graph launch: {r:?}").into());
                    }
                }
                let mut output = e.uninit(experts.input_width)?;
                {
                    let out_stage = workspace
                        .out_stage_e
                        .as_ref()
                        .expect("graph staging set above");
                    e.stream().memcpy_dtod(
                        &out_stage.slice(0..experts.input_width),
                        &mut output.slice_mut(0..experts.input_width),
                    )?;
                }
                output
            };
            if let Some(started) = started {
                use std::sync::atomic::Ordering;
                let ns = TIMING_NS
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed)
                    + started.elapsed().as_nanos() as u64;
                let calls = TIMING_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
                if calls.is_multiple_of(430) {
                    eprintln!(
                        "[nvfp4-dev-routed-timing] calls={calls} total_ms={:.1} avg_us={:.1}",
                        ns as f64 / 1.0e6,
                        ns as f64 / calls as f64 / 1.0e3,
                    );
                }
            }
            return Ok(output);
        }

        // Entry fence + router-output staging, all on e's stream: the fresh sel/w slices are
        // copied into the persistent e-context pair, then the event is recorded — the caller's
        // sel_d/w_d can free on e's stream with no cross-stream reader.
        if let Some((_, device)) = workspace.ev_entry.as_ref() {
            if *device != e.ctx().ordinal() {
                return Err("NVFP4 device-routed routes engine changed".into());
            }
        } else {
            let _main = e.gpu.enter_main()?;
            workspace.ev_entry = Some((e.ctx().new_event(None)?, e.ctx().ordinal()));
        }
        if workspace.dev_route_e.is_none() {
            let _main = e.gpu.enter_main()?;
            workspace.dev_route_e = Some((
                e.htod_i32(&vec![0i32; n_sel])?,
                e.htod(&vec![0.0f32; n_sel])?,
            ));
        }
        // MEMRA_SEL_MIRROR: the staging pair exists so the rank streams read a persistent
        // e-context address. The caller's sel_d/w_d ARE persistent (the process-static
        // selection rows), so when every consuming rank shares e's device the ranks can read
        // them directly and this hop disappears. The graph door keeps the staging (its
        // captured copies read the fixed addresses).
        let mirror = sel_mirror_on() && !step_tp_graph_enabled()?;
        let e_device = e.ctx().ordinal();
        // rank1_routed is consumed (taken) below; peek it here for the staging decision.
        let rank1_routed_peek = workspace.rank1_routed;
        let stage_needed = !mirror
            || self.ranks.iter().enumerate().any(|(rank_index, engine)| {
                !(rank1_routed_peek && rank_index == 1) && engine.ctx().ordinal() != e_device
            });
        {
            let _main = e.gpu.enter_main()?;
            if stage_needed {
                let (sel_e, w_e) = workspace
                    .dev_route_e
                    .as_mut()
                    .expect("device route staging set above");
                {
                    let mut dst = sel_e.slice_mut(0..n_sel);
                    e.stream().memcpy_dtod(&sel_d.slice(0..n_sel), &mut dst)?;
                }
                {
                    let mut dst = w_e.slice_mut(0..n_sel);
                    e.stream().memcpy_dtod(&w_d.slice(0..n_sel), &mut dst)?;
                }
            }
            let (ev_entry, _) = workspace.ev_entry.as_ref().expect("entry event set above");
            ev_entry.record(&e.stream())?;
        }
        // Prestage door: input pull + quantize were already issued on the rank streams
        // (before the router) — the rank stream order suffices, skip them here.
        let prestaged = std::mem::take(&mut workspace.prestaged);
        let rank1_routed = std::mem::take(&mut workspace.rank1_routed);
        for (rank_index, engine) in self.ranks.iter().enumerate() {
            let _main = engine.gpu.enter_main()?;
            let (ev_entry, _) = workspace.ev_entry.as_ref().expect("entry event set above");
            engine.stream().wait(ev_entry)?;
            if !prestaged {
                let mut destination = workspace.input[rank_index].slice_mut(0..experts.input_width);
                engine
                    .stream()
                    .memcpy_dtod(&input_dev.slice(0..experts.input_width), &mut destination)?;
            }
            if !(rank1_routed && rank_index == 1) {
                // ONE mirror launch instead of two 32-byte copy-engine dispatches; source is
                // the caller's persistent rows when this rank shares e's device (UVA, ordered
                // by ev_entry), else the staged e-context pair.
                let same_dev = engine.ctx().ordinal() == e_device;
                if mirror {
                    // Split the workspace borrow so the source (the staged pair, when this
                    // rank is off-device) and the destination rows coexist.
                    let Nvfp4DeviceRoutesWorkspace {
                        sel,
                        route_w,
                        dev_route_e,
                        ..
                    } = &mut *workspace;
                    let (src_sel, src_w): (&crate::CudaSlice<i32>, &crate::CudaSlice<f32>) =
                        if same_dev {
                            (sel_d, w_d)
                        } else {
                            let (sel_e, w_e) = dev_route_e
                                .as_ref()
                                .expect("device route staging set above");
                            (sel_e, w_e)
                        };
                    engine.moe_sel_w_mirror(
                        src_sel,
                        src_w,
                        &mut sel[rank_index],
                        &mut route_w[rank_index],
                        n_sel,
                    )?;
                } else {
                    let (sel_e, w_e) = workspace
                        .dev_route_e
                        .as_ref()
                        .expect("device route staging set above");
                    {
                        let mut dst = workspace.sel[rank_index].slice_mut(0..n_sel);
                        engine
                            .stream()
                            .memcpy_dtod(&sel_e.slice(0..n_sel), &mut dst)?;
                    }
                    {
                        let mut dst = workspace.route_w[rank_index].slice_mut(0..n_sel);
                        engine
                            .stream()
                            .memcpy_dtod(&w_e.slice(0..n_sel), &mut dst)?;
                    }
                }
            }
            if !prestaged {
                let Nvfp4DeviceRoutesWorkspace {
                    input, in_q, in_d, ..
                } = &mut *workspace;
                engine.quantize_q8_1_into(
                    &input[rank_index],
                    1,
                    experts.input_width,
                    &mut in_q[rank_index],
                    &mut in_d[rank_index],
                )?;
            }
        }
        self.nvfp4_routes_batched_sweeps(
            experts,
            workspace,
            &[],
            &[],
            &[],
            local_out,
            n_sel,
            activation_limit,
            true,
        )?;

        // rank0 == root: its own stream order already covers its sweep; only the PEER
        // ranks need the record/wait pair (host-op diet at the #1 eager seam, 2026-08-21).
        for (rank_index, engine) in self.ranks.iter().enumerate().skip(1) {
            let _main = engine.gpu.enter_main()?;
            workspace.ev_rank[rank_index].record(&engine.stream())?;
        }
        // Doorbell fences (MEMRA_FENCE_MEMOPS=1): rank1 + root ring their flags; e waits
        // the tickets instead of the two events. Arm lazily; 0-len = unsupported.
        let memops = fence_memops_on() && moe_direct_on() && self.ranks.len() == 2;
        let mut ticket = 0u32;
        if memops {
            use cudarc::driver::sys;
            if workspace.fence_flags_raw == 0 {
                let root = &self.ranks[0];
                let _main = root.gpu.enter_main()?;
                let mut ptr: sys::CUdeviceptr = 0;
                let r = unsafe { sys::cuMemAlloc_v2(&mut ptr, 8) };
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("fence flag alloc: {r:?}").into());
                }
                let r = unsafe { sys::cuMemsetD8_v2(ptr, 0, 8) };
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("fence flag memset: {r:?}").into());
                }
                workspace.fence_flags_raw = ptr as u64;
            }
            workspace.fence_ticket = workspace.fence_ticket.wrapping_add(1).max(1);
            ticket = workspace.fence_ticket;
            let base = workspace.fence_flags_raw;
            {
                let root = &self.ranks[0];
                let _main = root.gpu.enter_main()?;
                let r = unsafe {
                    sys::cuStreamWriteValue32_v2(
                        root.stream().cu_stream() as sys::CUstream,
                        (base + 4) as sys::CUdeviceptr,
                        ticket,
                        0,
                    )
                };
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("fence write root: {r:?}").into());
                }
            }
        }
        // PREJOIN hook: rank work is fully issued (dev1 running); independent e-stream
        // kernels queued here execute while the peer rank drains its sweep.
        pre_join()?;

        if moe_direct_on() && self.ranks.len() == 2 {
            // DIRECT JOIN: rank1's accumulator is root-resident (P2P single-store pass);
            // rank0's is root-stream-ordered. One root event + rank1's own event order
            // the model engine's single add — same operand order as root's add
            // (accumulator[0] + accumulator[1]): BIT-IDENTICAL. Output is a FRESH
            // e-context row (NOT an alias of ws state — the reverted zero-copy handoff's
            // hazard class does not apply).
            let _main = e.gpu.enter_main()?;
            if memops {
                use cudarc::driver::sys;
                let base = workspace.fence_flags_raw;
                let r = unsafe {
                    sys::cuStreamWaitValue32_v2(
                        e.stream().cu_stream() as sys::CUstream,
                        (base + 4) as sys::CUdeviceptr,
                        ticket,
                        sys::CUstreamWaitValue_flags::CU_STREAM_WAIT_VALUE_GEQ as u32,
                    )
                };
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("fence wait: {r:?}").into());
                }
                for ev in workspace.ev_rank.iter().skip(1) {
                    e.stream().wait(ev)?;
                }
            } else {
                {
                    let root = &self.ranks[0];
                    let _rmain = root.gpu.enter_main()?;
                    workspace
                        .ev_done
                        .as_ref()
                        .expect("device routes done event")
                        .record(&root.stream())?;
                }
                e.stream().wait(
                    workspace
                        .ev_done
                        .as_ref()
                        .expect("device routes done event"),
                )?;
                for ev in workspace.ev_rank.iter().skip(1) {
                    e.stream().wait(ev)?;
                }
            }
            let mut output = e.uninit(experts.input_width)?;
            if let Some((sh_raw, scale_raw)) = post_add {
                // MOE TAIL FUSION M1: fold the shexp apply into the join add —
                // dst = (acc0 + acc1) + sh*scale[0], the exact split-pair sequence.
                e.add3_raw(
                    &workspace.accumulator[0],
                    &workspace.accumulator[1],
                    sh_raw,
                    scale_raw,
                    &mut output,
                    experts.input_width,
                )?;
            } else {
                e.add(
                    &workspace.accumulator[0],
                    &workspace.accumulator[1],
                    &mut output,
                    experts.input_width,
                )?;
            }
            let output = output;
            if let Some(started) = started {
                use std::sync::atomic::Ordering;
                let ns = TIMING_NS
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed)
                    + started.elapsed().as_nanos() as u64;
                let calls = TIMING_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
                if calls.is_multiple_of(430) {
                    eprintln!(
                        "[nvfp4-dev-routes-direct-timing] calls={calls} total_ms={:.1} avg_us={:.1}",
                        ns as f64 / 1.0e6,
                        ns as f64 / calls as f64 / 1.0e3,
                    );
                }
            }
            return Ok(output);
        }
        {
            let root = &self.ranks[0];
            let _main = root.gpu.enter_main()?;
            for ev in workspace.ev_rank.iter().skip(1) {
                root.stream().wait(ev)?;
            }
            root.stream()
                .memcpy_dtod(&workspace.accumulator[1], &mut workspace.remote)?;
            {
                let Nvfp4DeviceRoutesWorkspace {
                    accumulator,
                    remote,
                    combined,
                    ..
                } = &mut *workspace;
                root.add(&accumulator[0], remote, combined, experts.input_width)?;
            }
            workspace
                .ev_done
                .as_ref()
                .expect("device routes done event")
                .record(&root.stream())?;
        }
        let output = {
            let _main = e.gpu.enter_main()?;
            e.stream().wait(
                workspace
                    .ev_done
                    .as_ref()
                    .expect("device routes done event"),
            )?;
            // (Zero-copy clone handoff REVERTED 2026-08-21: identity mismatch in the
            // routes-diet bisect. The alloc+copy stays until the hazard is understood.)
            let mut output = e.uninit(experts.input_width)?;
            e.stream().memcpy_dtod(
                &workspace.combined.slice(0..experts.input_width),
                &mut output.slice_mut(0..experts.input_width),
            )?;
            output
        };
        if let Some(started) = started {
            use std::sync::atomic::Ordering;
            let ns = TIMING_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed)
                + started.elapsed().as_nanos() as u64;
            let calls = TIMING_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
            if calls.is_multiple_of(430) {
                eprintln!(
                    "[nvfp4-dev-routed-timing] calls={calls} total_ms={:.1} avg_us={:.1}",
                    ns as f64 / 1.0e6,
                    ns as f64 / calls as f64 / 1.0e3,
                );
            }
        }
        Ok(output)
    }

    /// The fused finish's ROOT section (combine + shadow gathers), event-free: the eager
    /// caller wraps it with rank-event waits + the done record; the token graph captures it
    /// verbatim (parent edges provide the ordering).
    pub(crate) fn decode_v2_finish_root_fused(
        &self,
        ws: &mut StepTpDecodeV2Ws,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = &self.ranks[0];
        let _main = root.gpu.enter_main()?;
        if ws.raw_peer_partial != 0 {
            // Capture-safe raw seams (arming happened in the stage flow).
            raw_copy_bytes(ws.raw_peer_partial, ws.raw_o_partial1, ws.o_out * 4, root)?;
        } else {
            root.stream()
                .memcpy_dtod(&ws.o_partials[1][0], &mut ws.peer_partial)?;
        }
        {
            let StepTpDecodeV2Ws {
                o_partials,
                peer_partial,
                reduce_a,
                o_out,
                ..
            } = &mut *ws;
            root.add(&o_partials[0][0], peer_partial, reduce_a, *o_out)?;
        }
        let shadows = !no_local_shadow_on() || ws.raw_mixed_stage_e != 0;
        if shadows {
            // rank0's shadows are same-context (root) copies; rank1's cross-context reads go
            // raw when armed.
            let mut k_dst = ws.k_shadow.slice_mut(0..ws.local_kv_dim);
            root.stream().memcpy_dtod(&ws.k[0], &mut k_dst)?;
            let mut v_dst = ws.v_shadow.slice_mut(0..ws.local_kv_dim);
            root.stream().memcpy_dtod(&ws.v_raw[0], &mut v_dst)?;
        }
        if shadows && ws.raw_peer_partial != 0 {
            raw_copy_bytes(
                ws.raw_k_shadow + (ws.local_kv_dim * 4) as u64,
                ws.raw_k1,
                ws.local_kv_dim * 4,
                root,
            )?;
            raw_copy_bytes(
                ws.raw_v_shadow + (ws.local_kv_dim * 4) as u64,
                ws.raw_v1,
                ws.local_kv_dim * 4,
                root,
            )?;
        } else if shadows {
            let start = ws.local_kv_dim;
            let mut k_dst = ws.k_shadow.slice_mut(start..start + ws.local_kv_dim);
            root.stream().memcpy_dtod(&ws.k[1], &mut k_dst)?;
            let mut v_dst = ws.v_shadow.slice_mut(start..start + ws.local_kv_dim);
            root.stream().memcpy_dtod(&ws.v_raw[1], &mut v_dst)?;
        }
        if ws.raw_mixed_stage_e != 0 {
            // Token-graph mirrors: the e-glue children read same-context copies of the
            // root-produced rows.
            raw_copy_bytes(ws.raw_mixed_stage_e, ws.raw_reduce_a, ws.o_out * 4, root)?;
            let (k_stage, v_stage) = ws.raw_shadow_stage_e;
            raw_copy_bytes(k_stage, ws.raw_k_shadow, 2 * ws.local_kv_dim * 4, root)?;
            raw_copy_bytes(v_stage, ws.raw_v_shadow, 2 * ws.local_kv_dim * 4, root)?;
        }
        Ok(())
    }

    /// Arm the token-graph e-context mirrors (orchestrator-supplied fixed addresses) plus
    /// reduce_a's own pointer.
    pub(crate) fn decode_v2_arm_token_mirrors(
        &self,
        ws: &mut StepTpDecodeV2Ws,
        mixed_stage_e: u64,
        shadow_stage_e: (u64, u64),
    ) -> Result<(), Box<dyn std::error::Error>> {
        use cudarc::driver::DevicePtr;
        let root = &self.ranks[0];
        let _main = root.gpu.enter_main()?;
        let stream = root.stream();
        let (a, _g) = ws.reduce_a.device_ptr(&stream);
        ws.raw_reduce_a = a;
        ws.raw_mixed_stage_e = mixed_stage_e;
        ws.raw_shadow_stage_e = shadow_stage_e;
        Ok(())
    }

    /// Build one layer's stitched routes graph: per-rank children captured on their own
    /// streams (raw cuMemcpyAsync at every cross-context seam — cudarc's slice tracking is
    /// capture-illegal there), a root combine child, and a multi-device parent with
    /// {rank0, rank1} -> root dependency edges. Zero per-token updates: every address the
    /// nodes touch is persistent workspace/staging.
    fn nvfp4_routes_build_graph(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        workspace: &mut Nvfp4DeviceRoutesWorkspace,
        local_out: usize,
        n_sel: usize,
        activation_limit: Option<f32>,
    ) -> Result<RoutesGraph, Box<dyn std::error::Error>> {
        use cudarc::driver::DevicePtr;
        use cudarc::driver::sys;
        fn cu_try(r: sys::CUresult, what: &str) -> Result<(), Box<dyn std::error::Error>> {
            if r == sys::CUresult::CUDA_SUCCESS {
                Ok(())
            } else {
                Err(format!("{what}: {r:?}").into())
            }
        }
        let world = self.ranks.len();
        if world != 2 {
            return Err("routes graph door is built for the TP2 pair".into());
        }
        let width = experts.input_width;

        // Raw pointers cached before capture (each read with its owner's stream).
        let ptr_f32 = |buf: &crate::CudaSlice<f32>, engine: &Engine| -> u64 {
            let stream = engine.stream();
            let (ptr, _g) = buf.device_ptr(&stream);
            ptr
        };
        let ptr_i32 = |buf: &crate::CudaSlice<i32>, engine: &Engine| -> u64 {
            let stream = engine.stream();
            let (ptr, _g) = buf.device_ptr(&stream);
            ptr
        };
        let (sel_e, w_e) = workspace
            .dev_route_e
            .as_ref()
            .expect("device route staging set before graph build");
        let root_engine = &self.ranks[0];
        let p_in_stage = ptr_f32(
            workspace.in_stage_e.as_ref().expect("graph staging"),
            root_engine,
        );
        let p_out_stage = ptr_f32(
            workspace.out_stage_e.as_ref().expect("graph staging"),
            root_engine,
        );
        let p_sel_e = ptr_i32(sel_e, root_engine);
        let p_w_e = ptr_f32(w_e, root_engine);
        let p_input: Vec<u64> = (0..world)
            .map(|r| ptr_f32(&workspace.input[r], &self.ranks[r]))
            .collect();
        let p_sel: Vec<u64> = (0..world)
            .map(|r| ptr_i32(&workspace.sel[r], &self.ranks[r]))
            .collect();
        let p_route_w: Vec<u64> = (0..world)
            .map(|r| ptr_f32(&workspace.route_w[r], &self.ranks[r]))
            .collect();
        let p_acc1 = ptr_f32(&workspace.accumulator[1], &self.ranks[1]);
        let p_remote = ptr_f32(&workspace.remote, root_engine);
        let p_combined = ptr_f32(&workspace.combined, root_engine);

        let raw_copy = |dst: u64,
                        src: u64,
                        bytes: usize,
                        engine: &Engine|
         -> Result<(), Box<dyn std::error::Error>> {
            unsafe {
                cu_try(
                    sys::cuMemcpyAsync(
                        dst as sys::CUdeviceptr,
                        src as sys::CUdeviceptr,
                        bytes,
                        engine.stream().cu_stream() as sys::CUstream,
                    ),
                    "routes graph cuMemcpyAsync",
                )
            }
        };

        let mut children = Vec::with_capacity(3);
        for rank in 0..world {
            let engine = &self.ranks[rank];
            let _main = engine.gpu.enter_main()?;
            let (child, _retained) = engine.capture_graph_retained(|_| {
                raw_copy(p_input[rank], p_in_stage, width * 4, engine)?;
                raw_copy(p_sel[rank], p_sel_e, n_sel * 4, engine)?;
                raw_copy(p_route_w[rank], p_w_e, n_sel * 4, engine)?;
                {
                    let Nvfp4DeviceRoutesWorkspace {
                        input, in_q, in_d, ..
                    } = &mut *workspace;
                    engine.quantize_q8_1_into(
                        &input[rank],
                        1,
                        width,
                        &mut in_q[rank],
                        &mut in_d[rank],
                    )?;
                }
                self.nvfp4_routes_batched_sweeps_rank(
                    experts,
                    workspace,
                    &[],
                    &[],
                    &[],
                    local_out,
                    n_sel,
                    activation_limit,
                    true,
                    rank,
                )?;
                Ok(())
            })?;
            children.push(child);
        }
        {
            let root = &self.ranks[0];
            let _main = root.gpu.enter_main()?;
            let (child, _retained) = root.capture_graph_retained(|_| {
                raw_copy(p_remote, p_acc1, width * 4, root)?;
                {
                    let Nvfp4DeviceRoutesWorkspace {
                        accumulator,
                        remote,
                        combined,
                        ..
                    } = &mut *workspace;
                    root.add(&accumulator[0], remote, combined, width)?;
                }
                raw_copy(p_out_stage, p_combined, width * 4, root)?;
                Ok(())
            })?;
            children.push(child);
        }

        let mut parent: sys::CUgraph = std::ptr::null_mut();
        unsafe {
            cu_try(sys::cuGraphCreate(&mut parent, 0), "routes cuGraphCreate")?;
        }
        let mut n0: sys::CUgraphNode = std::ptr::null_mut();
        let mut n1: sys::CUgraphNode = std::ptr::null_mut();
        let mut n2: sys::CUgraphNode = std::ptr::null_mut();
        unsafe {
            cu_try(
                sys::cuGraphAddChildGraphNode(
                    &mut n0,
                    parent,
                    std::ptr::null(),
                    0,
                    children[0].cu_graph(),
                ),
                "routes child r0",
            )?;
            cu_try(
                sys::cuGraphAddChildGraphNode(
                    &mut n1,
                    parent,
                    std::ptr::null(),
                    0,
                    children[1].cu_graph(),
                ),
                "routes child r1",
            )?;
            let deps = [n0, n1];
            cu_try(
                sys::cuGraphAddChildGraphNode(
                    &mut n2,
                    parent,
                    deps.as_ptr(),
                    2,
                    children[2].cu_graph(),
                ),
                "routes child root",
            )?;
        }
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        unsafe {
            cu_try(
                sys::cuGraphInstantiateWithFlags(&mut exec, parent, 0),
                "routes instantiate",
            )?;
        }
        Ok(RoutesGraph {
            exec,
            parent,
            _children: children,
        })
    }

    /// One rank's routes section for the token graph (event-free): staged input copy (raw
    /// when the caller supplies the source pointer), quantize, and the batched sweeps.
    /// Eager device_routed wraps it with the entry-event wait.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn routes_rank_section(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        workspace: &mut Nvfp4DeviceRoutesWorkspace,
        raw_input_src: u64,
        local_out: usize,
        n_sel: usize,
        activation_limit: Option<f32>,
        rank_index: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let engine = &self.ranks[rank_index];
        {
            let _main = engine.gpu.enter_main()?;
            // sel/route_w land via raw copies from the e staging (fixed addresses).
            let (sel_e_ptr, w_e_ptr) = workspace
                .raw_dev_route_e
                .ok_or("routes rank section requires armed staging pointers")?;
            raw_copy_bytes(
                workspace.raw_input[rank_index],
                raw_input_src,
                experts.input_width * 4,
                engine,
            )?;
            raw_copy_bytes(workspace.raw_sel[rank_index], sel_e_ptr, n_sel * 4, engine)?;
            raw_copy_bytes(
                workspace.raw_route_w[rank_index],
                w_e_ptr,
                n_sel * 4,
                engine,
            )?;
            {
                let Nvfp4DeviceRoutesWorkspace {
                    input, in_q, in_d, ..
                } = &mut *workspace;
                engine.quantize_q8_1_into(
                    &input[rank_index],
                    1,
                    experts.input_width,
                    &mut in_q[rank_index],
                    &mut in_d[rank_index],
                )?;
            }
        }
        self.nvfp4_routes_batched_sweeps_rank(
            experts,
            workspace,
            &[],
            &[],
            &[],
            local_out,
            n_sel,
            activation_limit,
            true,
            rank_index,
        )
    }

    /// The routes ROOT combine section (event-free): peer accumulator read (raw), canonical
    /// add, combined row raw-copied into the fixed e-context out stage.
    pub(crate) fn routes_root_section(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        workspace: &mut Nvfp4DeviceRoutesWorkspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = &self.ranks[0];
        let _main = root.gpu.enter_main()?;
        let (acc1_ptr, remote_ptr, combined_ptr, out_stage_ptr) = workspace
            .raw_combine
            .ok_or("routes root section requires armed combine pointers")?;
        raw_copy_bytes(remote_ptr, acc1_ptr, experts.input_width * 4, root)?;
        {
            let Nvfp4DeviceRoutesWorkspace {
                accumulator,
                remote,
                combined,
                ..
            } = &mut *workspace;
            root.add(&accumulator[0], remote, combined, experts.input_width)?;
        }
        raw_copy_bytes(out_stage_ptr, combined_ptr, experts.input_width * 4, root)?;
        Ok(())
    }

    /// Arm the routes raw pointers (once): staging pair, per-rank input/sel/route_w, and the
    /// combine set. Requires dev_route_e + in/out stages already allocated.
    pub(crate) fn routes_arm_raw(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        workspace: &mut Nvfp4DeviceRoutesWorkspace,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use cudarc::driver::DevicePtr;
        if workspace.raw_dev_route_e.is_some() {
            return Ok(());
        }
        let _ = experts;
        let (sel_e, w_e) = workspace
            .dev_route_e
            .as_ref()
            .ok_or("routes staging not armed")?;
        let root = &self.ranks[0];
        {
            let _main = root.gpu.enter_main()?;
            let stream = root.stream();
            let (a, _g) = sel_e.device_ptr(&stream);
            let (b, _g) = w_e.device_ptr(&stream);
            workspace.raw_dev_route_e = Some((a, b));
            let (c, _g) = workspace.accumulator[1].device_ptr(&stream);
            let (d, _g) = workspace.remote.device_ptr(&stream);
            let (f, _g) = workspace.combined.device_ptr(&stream);
            let out_stage = workspace
                .out_stage_e
                .as_ref()
                .ok_or("routes out stage not armed")?;
            let (g_, _g) = out_stage.device_ptr(&stream);
            workspace.raw_combine = Some((c, d, f, g_));
        }
        for rank in 0..self.ranks.len() {
            let engine = &self.ranks[rank];
            let _main = engine.gpu.enter_main()?;
            let stream = engine.stream();
            let (a, _g) = workspace.input[rank].device_ptr(&stream);
            let (b, _g) = workspace.sel[rank].device_ptr(&stream);
            let (c, _g) = workspace.route_w[rank].device_ptr(&stream);
            workspace.raw_input.push(a);
            workspace.raw_sel.push(b);
            workspace.raw_route_w.push(c);
        }
        Ok(())
    }

    /// Routed NVFP4 expert program, host-canonical transport. Native/bulk P2P transport for the
    /// NVFP4 bank is a separate increment; this entry point is exactness-first and reports no
    /// throughput claim.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub fn run_tensor_parallel_routes_nvfp4(
        &self,
        experts: &ResidentNvfp4TensorParallel,
        input: &[f32],
        tokens: usize,
        selected: &[usize],
        route_weights: &[f32],
        experts_per_token: usize,
        activation_limit: Option<f32>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        validate_activations(input, tokens, experts.input_width)?;
        let pairs = tokens
            .checked_mul(experts_per_token)
            .ok_or("NVFP4 TP route count overflow")?;
        if selected.len() != pairs || route_weights.len() != pairs {
            return Err(format!(
                "NVFP4 TP routes selected={} weights={} != tokens {tokens} x experts/token \
                 {experts_per_token} ({pairs})",
                selected.len(),
                route_weights.len(),
            )
            .into());
        }
        if !route_weights.iter().all(|weight| weight.is_finite()) {
            return Err("NVFP4 TP route weights contain a non-finite value".into());
        }

        let mut output = vec![0.0f32; tokens * experts.input_width];
        for token in 0..tokens {
            let input_row = &input[token * experts.input_width..(token + 1) * experts.input_width];
            for slot in 0..experts_per_token {
                let pair = token * experts_per_token + slot;
                let expert = selected[pair];
                if expert >= experts.expert_count {
                    return Err(format!(
                        "NVFP4 TP selected expert {expert} outside 0..{}",
                        experts.expert_count
                    )
                    .into());
                }
                let gate = self.run_column_bank_expert_nvfp4(
                    &experts.gate,
                    &experts.macros_gate,
                    expert,
                    input_row,
                )?;
                let up = self.run_column_bank_expert_nvfp4(
                    &experts.up,
                    &experts.macros_up,
                    expert,
                    input_row,
                )?;
                let activated: Vec<f32> = gate
                    .iter()
                    .zip(&up)
                    .map(|(&gate, &up)| step_expert_activation_host(gate, up, activation_limit))
                    .collect();
                debug_assert_eq!(activated.len(), experts.expert_width);
                let down = self.run_row_bank_expert_nvfp4(
                    &experts.down,
                    &experts.macros_down,
                    expert,
                    &activated,
                )?;
                let weight = route_weights[pair];
                for (sum, value) in output
                    [token * experts.input_width..(token + 1) * experts.input_width]
                    .iter_mut()
                    .zip(down)
                {
                    *sum += weight * value;
                }
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod default_on_door_tests {
    use super::door_default_on_value;

    /// The DEFAULT-ON parse, pinned in every state — including the two that only matter because
    /// the default is ON.
    ///
    /// While these doors were default OFF the parse was `== Ok("1")` and its failure mode was
    /// benign: any typo read as the default, which was OFF, which was the safe program. Flipping
    /// the default INVERTS that. Under a naive `!= Ok("0")` rule, `MEMRA_NVFP4_BANK_SM=false`
    /// (or `=off`, or `=no`) would leave the program ARMED while the operator believed they had
    /// rolled it back — a rollback seam that silently does nothing, on the exact door whose
    /// predecessor shipped fluent wrong text. So the unrecognized-value case is a named,
    /// tested branch that keeps the default AND warns, rather than an accident of `!=`.
    #[test]
    fn the_default_on_door_parses_every_state_and_names_its_source() {
        // unset: the flip is what arms it, and the source string says so — this is the string a
        // default-flip receipt needs, because in the flip arms there is no env var to point at.
        assert_eq!(
            door_default_on_value("MEMRA_TEST_DOOR", None),
            (true, "default-on")
        );
        // explicit 1: armed by a RECIPE, not by the default. Different fact, different label.
        assert_eq!(
            door_default_on_value("MEMRA_TEST_DOOR", Some("1")),
            (true, "env=1")
        );
        // THE ROLLBACK SEAM. This is the assertion the flip's safety rests on.
        assert_eq!(
            door_default_on_value("MEMRA_TEST_DOOR", Some("0")),
            (false, "env=0 (rollback seam)")
        );
        // Unrecognized values keep the DEFAULT (ON) and are flagged as such, for every shape an
        // operator plausibly types when they mean "off". Every one of these MUST still read ON:
        // a parse that guessed "off" from `false` would be a second, undocumented seam, and a
        // parse that guessed "off" from `2` would make a typo a silent program change.
        for bad in [
            "false", "off", "no", "", " 0", "0 ", "00", "true", "2", "-1",
        ] {
            let (on, source) = door_default_on_value("MEMRA_TEST_DOOR", Some(bad));
            assert!(on, "value {bad:?} must NOT disarm a default-ON door");
            assert!(
                source.contains("default-on") && source.contains("unrecognized"),
                "value {bad:?} gave source {source:?}, which does not announce itself as an \
                 ignored value — a receipt reader would take it for a clean default"
            );
        }
    }
}

#[cfg(test)]
mod bank_v2_layout_tests {
    use super::{nvfp4_matrix_v2_permute, nvfp4_row_bytes};

    /// The slot-major permutation had NO test at all until 2026-08-29, while its (since
    /// removed) `MEMRA_NVFP4_BANK_V2` FLAGS row carried a bit-identity claim and the live
    /// serving env pinned it on. This pins the DOCUMENTED mapping so a reader can be checked
    /// against something: per row, slot g's 16 qs bytes land contiguously at `g*16`, and its
    /// two UE4M3 scale bytes at `nslots*16 + g*2`. Source layout is memra `block_nvfp4`:
    /// 36-byte superblocks of [4 scale bytes | 32 packed e2m1], two 32-value slots per
    /// superblock. The permutation's consumers are the slot-major TP shard banks
    /// (`MEMRA_NVFP4_BANK_SM`, `nvfp4_repack_bank_matrix(_, true)`) and the
    /// `qmatvec_nvfp4_fast_v2` oracle, which read this exact mapping.
    #[test]
    fn the_v2_bank_row_is_the_documented_slot_major_permutation() {
        // two rows, in_features 128 => 2 superblocks/row, 4 slots/row, 72 bytes/row.
        let (out_f, in_f) = (2usize, 128usize);
        let row_bytes = nvfp4_row_bytes(in_f);
        assert_eq!(row_bytes, 72);
        let v1: Vec<u8> = (0..out_f * row_bytes).map(|i| (i % 251) as u8).collect();
        let v2 = nvfp4_matrix_v2_permute(&v1, out_f, in_f);
        assert_eq!(v2.len(), v1.len(), "a permutation cannot change the size");
        let n_slots = in_f / 32;
        for row in 0..out_f {
            let src = &v1[row * row_bytes..(row + 1) * row_bytes];
            let dst = &v2[row * row_bytes..(row + 1) * row_bytes];
            for g in 0..n_slots {
                let (sblk, h) = (g / 2, g % 2);
                let sb = &src[sblk * 36..sblk * 36 + 36];
                assert_eq!(
                    &dst[g * 16..g * 16 + 16],
                    &sb[4 + 16 * h..4 + 16 * h + 16],
                    "row {row} slot {g} codes"
                );
                assert_eq!(
                    &dst[n_slots * 16 + g * 2..n_slots * 16 + g * 2 + 2],
                    &sb[2 * h..2 * h + 2],
                    "row {row} slot {g} scales"
                );
            }
            // and it moves bytes only: same multiset per row, rows never cross.
            let (mut a, mut b) = (src.to_vec(), dst.to_vec());
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "row {row} is not a byte permutation");
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn replicated_row_join_is_strictly_tp2_native_and_nonempty() {
        assert!(super::validate_tp2_replicated_row_join(2, true, 4096).is_ok());
        assert!(
            super::validate_tp2_replicated_row_join(1, true, 4096)
                .unwrap_err()
                .contains("exactly two ranks")
        );
        assert!(
            super::validate_tp2_replicated_row_join(4, true, 4096)
                .unwrap_err()
                .contains("exactly two ranks")
        );
        assert!(
            super::validate_tp2_replicated_row_join(2, false, 4096)
                .unwrap_err()
                .contains("native P2P")
        );
        assert!(super::validate_tp2_replicated_row_join(2, true, 0).is_err());
    }

    #[test]
    fn door_composition_refuses_first_armed_flag_by_name() {
        let table: [(&str, &str); 2] = [
            ("MEMRA_DOOR_A", "gated on the unsharded walk only"),
            ("MEMRA_DOOR_B", "no sharded branches"),
        ];
        // cold doors pass
        super::refuse_door_composition("MEMRA_X_TP", &table, |_| false).expect("cold doors pass");
        // an armed door refuses with the exact byte format the glm5 gate asserts on
        let err = super::refuse_door_composition("MEMRA_X_TP", &table, |f| f == "MEMRA_DOOR_B")
            .expect_err("armed door must refuse");
        assert_eq!(
            err,
            "MEMRA_X_TP + MEMRA_DOOR_B: unproven composition, refused (no sharded branches)"
        );
        // a flag outside the table never trips it
        super::refuse_door_composition("MEMRA_X_TP", &table, |f| f == "MEMRA_DOOR_C")
            .expect("foreign flags are not the matrix");
    }

    /// THE DEFECT, ASSERTED SO IT CANNOT COME BACK. The retired memo key hashed only the K
    /// pointer, the base pointer, the layer and t, while the table it returned ALSO carried
    /// the V and LEN pointers. Two different allocation generations that happen to share a K
    /// address therefore collide, and the entry the map hands back sends a live launch at
    /// another allocation's V and len. This test does not assert the key is fine; it asserts
    /// the key is BLIND, which is why `rows_tab_restage_on` exists and defaults ON.
    #[test]
    fn the_retired_rows_tab_key_cannot_see_the_v_and_len_pointers_it_hands_back() {
        let (kp, bp) = (0xdead_0000u64, 0u64);
        let live = [[kp, 0x00b1_0000u64, 0x00c1_0000u64, bp]];
        let recycled = [[kp, 0x00b2_0000u64, 0x00c2_0000u64, bp]];
        assert_eq!(
            super::retired_rows_tab_key(kp, bp, 20, 2),
            super::retired_rows_tab_key(kp, bp, 20, 2),
            "same layer and t must hash the same, or the test proves nothing"
        );
        let a = super::rows_tab_host(&live, 0x9000, true, 1);
        let b = super::rows_tab_host(&recycled, 0x9000, true, 1);
        assert_ne!(a, b, "the two generations write DIFFERENT tables");
        // ... yet one key covers both, which is exactly the use-after-free.
        assert_eq!(
            super::retired_rows_tab_key(live[0][0], live[0][3], 20, 1),
            super::retired_rows_tab_key(recycled[0][0], recycled[0][3], 20, 1),
            "the retired key collides across allocation generations"
        );
    }

    /// The restage must be VALUE-NEUTRAL: on a fresh lookup the memo and the restage produce
    /// identical bytes, which is what makes spec-on output byte-identical to spec-off.
    #[test]
    fn rows_tab_layout_is_the_same_bytes_the_memo_would_have_cached() {
        let parts = [
            [0x00a0u64, 0x00b0u64, 0x00c0u64, 0x00d0u64],
            [0x00a1u64, 0x00b1u64, 0x00c1u64, 0x00d1u64],
        ];
        let same = super::rows_tab_host(&parts, 0x7000, true, 2);
        assert_eq!(
            same,
            vec![
                0x00a0u64, 0x00b0u64, 0x00c0u64, 0x00d0u64, 0x7000,
                1, // row 0: back = t-1-r = 1
                0x00a1u64, 0x00b1u64, 0x00c1u64, 0x00d1u64, 0x7000, 0, // row 1: back = 0
            ],
            "same-session rows share one counter cell and step back t-1-r"
        );
        let cross = super::rows_tab_host(&parts, 0x7000, false, 2);
        assert_eq!(
            cross,
            vec![
                0x00a0u64, 0x00b0u64, 0x00c0u64, 0x00d0u64, 0x7000, 0, 0x00a1u64, 0x00b1u64,
                0x00c1u64, 0x00d1u64, 0x7004, 0,
            ],
            "cross-session rows get their own counter cell and no step back"
        );
    }
    use super::*;

    #[test]
    fn step_expert_activation_clamps_each_arm_by_the_official_contract() {
        let limit = Some(7.0);
        assert_eq!(step_expert_activation_host(20.0, 9.0, limit), 49.0);
        assert_eq!(step_expert_activation_host(20.0, -9.0, limit), -49.0);
        assert!(
            step_expert_activation_host(-20.0, 9.0, limit).abs()
                < step_expert_activation_host(-20.0, 9.0, None).abs()
        );
        assert!(validate_step_expert_activation_limit(Some(f32::NAN)).is_err());
        assert!(validate_step_expert_activation_limit(Some(0.0)).is_err());
        assert!(validate_step_expert_activation_limit(limit).is_ok());
    }

    #[test]
    fn moe_residual_host_preserves_official_add_order() {
        let output = moe_residual_host(&[1.0e20], &[-1.0e20], &[1.0]).unwrap();
        assert_eq!(output, [0.0]);
        assert_eq!(
            moe_residual_host(&[0.0], &[0.0, 1.0], &[0.0]).unwrap_err(),
            "MoE residual lengths residual=1 routed=2 shared=1"
        );
    }

    #[test]
    fn expert_owner_routes_preserve_global_pair_order_with_local_expert_ids() {
        let selected = [0, 36, 72, 108, 144, 180, 216, 252];
        let owners = partition_expert_owner_routes(288, 4, 1, 8, &selected).unwrap();
        assert_eq!(owners.len(), 4);
        for (rank, owner) in owners.iter().enumerate() {
            assert_eq!(owner.rank, rank);
            assert_eq!(owner.selected, vec![0, 36]);
            assert_eq!(owner.token_rows, vec![0, 0]);
            assert_eq!(owner.global_pairs, vec![rank * 2, rank * 2 + 1]);
        }
    }

    #[test]
    fn expert_owner_routes_validate_geometry_and_selected_experts() {
        assert!(partition_expert_owner_routes(288, 5, 1, 8, &[0; 8]).is_err());
        assert!(partition_expert_owner_routes(288, 4, 2, 8, &[0; 8]).is_err());
        let error = partition_expert_owner_routes(288, 4, 1, 8, &[288; 8]).unwrap_err();
        assert!(error.contains("outside 0..288"));
    }

    #[test]
    fn step_grouped_owner_routes_validate_dynamic_top8_shapes() {
        let selected = [
            1, 73, 80, 145, 152, 159, 217, 224, 12, 84, 91, 156, 163, 170, 228, 235,
        ];
        assert_eq!(
            validate_step_grouped_owner_routes(288, 2, &selected).unwrap(),
            16
        );
        let owners = partition_expert_owner_routes(288, 4, 2, 8, &selected).unwrap();
        assert_eq!(
            owners
                .iter()
                .map(|owner| owner.selected.len())
                .collect::<Vec<_>>(),
            vec![2, 4, 6, 4]
        );
        assert!(validate_step_grouped_owner_routes(288, 2, &selected[..8]).is_err());
        assert!(validate_step_grouped_owner_routes(288, 1, &[0; 8]).is_err());
        assert!(validate_step_grouped_owner_routes(287, 2, &selected).is_err());
    }

    #[test]
    fn weighted_route_combine_requires_a_canonical_pair_permutation() {
        let owner0 = [0usize, 3];
        let owner1 = [1usize, 2];
        let owners = [owner0.as_slice(), owner1.as_slice()];
        assert_eq!(
            validate_weighted_route_combine(4096, 4, 3, 1, &owners, &[0.1, 0.2, 0.3, 0.4],)
                .unwrap(),
            WeightedRouteCombineShape {
                pairs: 4,
                max_pairs: 12,
            }
        );
        let duplicate = [owner0.as_slice(), &[1usize, 1][..]];
        assert!(
            validate_weighted_route_combine(4096, 4, 3, 1, &duplicate, &[0.1, 0.2, 0.3, 0.4],)
                .is_err()
        );
        assert!(
            validate_weighted_route_combine(4096, 4, 3, 1, &owners, &[0.1, f32::NAN, 0.3, 0.4],)
                .is_err()
        );
        assert!(
            validate_weighted_route_combine(4096, 4, 1, 2, &owners, &[0.1, 0.2, 0.3, 0.4],)
                .is_err()
        );
    }

    #[test]
    fn native_p2p_door_is_strict_and_default_off() {
        assert!(!parse_step_tp_native_p2p(None).unwrap());
        assert!(!parse_step_tp_native_p2p(Some("")).unwrap());
        assert!(!parse_step_tp_native_p2p(Some("0")).unwrap());
        assert!(parse_step_tp_native_p2p(Some("1")).unwrap());
        assert!(parse_step_tp_native_p2p(Some("true")).is_err());
        assert!(parse_step_tp_native_p2p(Some("2")).is_err());
    }

    #[test]
    fn bulk_p2p_door_is_strict_and_default_off() {
        assert!(!parse_step_tp_bulk_p2p(None).unwrap());
        assert!(!parse_step_tp_bulk_p2p(Some("")).unwrap());
        assert!(!parse_step_tp_bulk_p2p(Some("0")).unwrap());
        assert!(parse_step_tp_bulk_p2p(Some("1")).unwrap());
        assert!(parse_step_tp_bulk_p2p(Some("true")).is_err());
        assert!(parse_step_tp_bulk_p2p(Some("2")).is_err());
    }

    #[test]
    fn ep_device_arithmetic_door_is_strict_and_default_off() {
        assert!(!parse_step_ep_device_arithmetic(None).unwrap());
        assert!(!parse_step_ep_device_arithmetic(Some("")).unwrap());
        assert!(!parse_step_ep_device_arithmetic(Some("0")).unwrap());
        assert!(parse_step_ep_device_arithmetic(Some("1")).unwrap());
        assert!(parse_step_ep_device_arithmetic(Some("true")).is_err());
        assert!(parse_step_ep_device_arithmetic(Some("2")).is_err());
    }

    #[test]
    fn f32_mirror_door_is_strict_and_default_off() {
        assert!(!parse_step_tp_f32_mirror(None).unwrap());
        assert!(!parse_step_tp_f32_mirror(Some("")).unwrap());
        assert!(!parse_step_tp_f32_mirror(Some("0")).unwrap());
        assert!(parse_step_tp_f32_mirror(Some("1")).unwrap());
        assert!(parse_step_tp_f32_mirror(Some("true")).is_err());
        assert!(parse_step_tp_f32_mirror(Some("2")).is_err());
    }

    fn matrix(out_features: usize, in_features: usize) -> (Vec<u8>, Vec<f32>) {
        let codes = (0..out_features * in_features)
            .map(|index| (index % 251) as u8)
            .collect();
        let scales = (0..out_features.div_ceil(FP8_BLOCK) * in_features.div_ceil(FP8_BLOCK))
            .map(|index| index as f32 + 1.0)
            .collect();
        (codes, scales)
    }

    fn bf16_matrix_bytes(out_features: usize, in_features: usize) -> Vec<u8> {
        (0..out_features * in_features)
            .flat_map(|value| (value as u16).to_le_bytes())
            .collect()
    }

    fn decode_u16(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect()
    }

    #[test]
    fn bf16_matrix_rejects_wrong_byte_count() {
        let bytes = vec![0u8; 4 * 4 * 2 - 1];
        let matrix = Bf16Matrix {
            bytes: &bytes,
            out_features: 4,
            in_features: 4,
        };
        assert!(matrix.validate().unwrap_err().contains("4x4x2"));
    }

    #[test]
    fn replicated_device_rows_require_exact_rank_local_shapes() {
        assert_eq!(
            replicated_device_row_values(3, 4096, 4, &[12_288; 4]).unwrap(),
            12_288
        );
        assert!(replicated_device_row_values(0, 4096, 4, &[0; 4]).is_err());
        assert!(replicated_device_row_values(3, 0, 4, &[0; 4]).is_err());
        assert!(replicated_device_row_values(3, 4096, 4, &[12_288; 3]).is_err());
        assert!(
            replicated_device_row_values(3, 4096, 4, &[12_288, 12_288, 12_287, 12_288]).is_err()
        );
        assert!(replicated_device_row_values(usize::MAX, 2, 1, &[0]).is_err());
    }

    #[test]
    fn replicated_device_row_refresh_requires_exact_root_source() {
        assert_eq!(
            replicated_device_row_source_values(1, 12_288, 12_288, 3, 3).unwrap(),
            12_288
        );
        assert!(replicated_device_row_source_values(0, 12_288, 0, 3, 3).is_err());
        assert!(replicated_device_row_source_values(1, 0, 0, 3, 3).is_err());
        assert!(replicated_device_row_source_values(1, 12_288, 12_287, 3, 3).is_err());
        assert!(replicated_device_row_source_values(1, 12_288, 12_288, 2, 3).is_err());
        assert!(replicated_device_row_source_values(usize::MAX, 2, 0, 3, 3).is_err());
    }

    #[test]
    fn step_bf16_canonical_rows_are_topology_invariant_through_tp8() {
        for tp in [1, 2, 4, 8] {
            assert_eq!(step_bf16_canonical_chunk_rows(8_192, tp).unwrap(), 1_024);
            assert_eq!(step_bf16_canonical_chunk_rows(12_288, tp).unwrap(), 1_536);
            assert_eq!(step_bf16_canonical_chunk_rows(1_024, tp).unwrap(), 128);
            assert_eq!(step_bf16_canonical_chunk_cols(8_192, tp).unwrap(), 1_024);
            assert_eq!(step_bf16_canonical_chunk_cols(12_288, tp).unwrap(), 1_536);
        }
        assert!(step_bf16_canonical_chunk_rows(12_288, 3).is_err());
        assert!(step_bf16_canonical_chunk_rows(1_001, 2).is_err());
        assert!(step_bf16_canonical_chunk_cols(12_288, 3).is_err());
        assert!(step_bf16_canonical_chunk_cols(1_001, 2).is_err());
    }

    #[test]
    fn cache_rows_split_by_token_then_rank() {
        let rows = (0u8..24).collect::<Vec<_>>();
        assert_eq!(
            cache_rank_rows(&rows, 3, 4, 2, 0).unwrap(),
            vec![0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]
        );
        assert_eq!(
            cache_rank_rows(&rows, 3, 4, 2, 1).unwrap(),
            vec![4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23]
        );
        assert!(cache_rank_rows(&rows[..23], 3, 4, 2, 0).is_err());
        assert!(cache_rank_rows(&rows, 3, 4, 2, 2).is_err());
    }

    #[test]
    fn bf16_column_shard_preserves_contiguous_output_rows() {
        let bytes = bf16_matrix_bytes(4, 4);
        let matrix = Bf16Matrix {
            bytes: &bytes,
            out_features: 4,
            in_features: 4,
        };
        let shard = bf16_column_shard(matrix, 2, 1).unwrap();
        assert_eq!(shard.out_features, 2);
        assert_eq!(shard.in_features, 4);
        assert_eq!(decode_u16(shard.bytes), (8..16).collect::<Vec<_>>());
    }

    #[test]
    fn bf16_row_shard_preserves_each_input_column_window() {
        let bytes = bf16_matrix_bytes(3, 4);
        let matrix = Bf16Matrix {
            bytes: &bytes,
            out_features: 3,
            in_features: 4,
        };
        let shard = bf16_row_shard(matrix, 2, 1).unwrap();
        assert_eq!(decode_u16(&shard), vec![2, 3, 6, 7, 10, 11]);
    }

    #[test]
    fn bf16_row_block_preserves_global_column_order() {
        let bytes = bf16_matrix_bytes(3, 8);
        let matrix = Bf16Matrix {
            bytes: &bytes,
            out_features: 3,
            in_features: 8,
        };
        let block = bf16_row_block(matrix, 2, 3).unwrap();
        assert_eq!(decode_u16(&block), vec![2, 3, 4, 10, 11, 12, 18, 19, 20]);
    }

    #[test]
    fn column_shard_preserves_contiguous_weight_and_scale_rows() {
        let (codes, scales) = matrix(1280, 4096);
        let matrix = E4m3BlockMatrix {
            codes: &codes,
            scales: &scales,
            out_features: 1280,
            in_features: 4096,
        };
        let shard = column_shard(matrix, 2, 1).unwrap();
        assert_eq!(shard.out_features, 640);
        assert_eq!(shard.codes, &codes[640 * 4096..]);
        assert_eq!(shard.scales, &scales[5 * 32..]);
    }

    #[test]
    fn row_shard_preserves_each_weight_and_scale_column_window() {
        let (codes, scales) = matrix(4096, 1280);
        let matrix = E4m3BlockMatrix {
            codes: &codes,
            scales: &scales,
            out_features: 4096,
            in_features: 1280,
        };
        let (shard_codes, shard_scales) = row_shard(matrix, 2, 1).unwrap();
        assert_eq!(shard_codes.len(), 4096 * 640);
        assert_eq!(&shard_codes[..640], &codes[640..1280]);
        assert_eq!(&shard_codes[640..1280], &codes[1280 + 640..2560]);
        assert_eq!(shard_scales.len(), 32 * 5);
        assert_eq!(&shard_scales[..5], &scales[5..10]);
        assert_eq!(&shard_scales[5..10], &scales[15..20]);
    }

    #[test]
    fn activation_shards_keep_token_rows_separate() {
        let activations: Vec<f32> = (0..2 * 8).map(|value| value as f32).collect();
        assert_eq!(
            activation_shard(&activations, 2, 8, 2, 1),
            vec![4.0, 5.0, 6.0, 7.0, 12.0, 13.0, 14.0, 15.0],
        );
    }

    #[test]
    fn expert_bank_selects_expert_major_code_and_scale_planes() {
        let expert_count = 2;
        let out_features = 128;
        let in_features = 128;
        let code_stride = out_features * in_features;
        let codes: Vec<u8> = (0..expert_count * code_stride)
            .map(|index| (index % 251) as u8)
            .collect();
        let scales = vec![1.0f32, 2.0];
        let bank = E4m3ExpertBank {
            codes: &codes,
            scales: &scales,
            expert_count,
            out_features,
            in_features,
        };
        bank.validate().unwrap();
        let expert = bank.expert(1).unwrap();
        assert_eq!(expert.codes, &codes[code_stride..]);
        assert_eq!(expert.scales, &[2.0]);
    }

    #[test]
    fn expert_bank_rejects_non_positive_scale() {
        let codes = vec![0u8; 128 * 128];
        let scales = vec![0.0f32];
        let bank = E4m3ExpertBank {
            codes: &codes,
            scales: &scales,
            expert_count: 1,
            out_features: 128,
            in_features: 128,
        };
        assert!(bank.validate().unwrap_err().contains("non-positive"));
    }

    #[test]
    fn tensor_parallel_column_bank_keeps_each_expert_scale_plane_separate() {
        let expert_count = 2;
        let out_features = 256;
        let in_features = 128;
        let code_stride = out_features * in_features;
        let scale_stride = 2;
        let codes = (0..expert_count * code_stride)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let scales = vec![10.0f32, 11.0, 20.0, 21.0];
        let bank = E4m3ExpertBank {
            codes: &codes,
            scales: &scales,
            expert_count,
            out_features,
            in_features,
        };

        let rank = pack_column_bank_rank(bank, 2, 1).unwrap();
        assert_eq!(rank.out_features, 128);
        assert_eq!(rank.in_features, 128);
        assert_eq!(rank.codes.len(), expert_count * 128 * 128);
        assert_eq!(rank.scales, vec![11.0, 21.0]);
        assert_eq!(&rank.codes[..128 * 128], &codes[128 * 128..256 * 128]);
        assert_eq!(
            &rank.codes[128 * 128..],
            &codes[code_stride + 128 * 128..2 * code_stride]
        );
        assert_eq!(scale_stride, scales.len() / expert_count);
    }

    #[test]
    fn tensor_parallel_row_bank_keeps_each_expert_scale_plane_separate() {
        let expert_count = 2;
        let out_features = 128;
        let in_features = 256;
        let code_stride = out_features * in_features;
        let codes = (0..expert_count * code_stride)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let scales = vec![10.0f32, 11.0, 20.0, 21.0];
        let bank = E4m3ExpertBank {
            codes: &codes,
            scales: &scales,
            expert_count,
            out_features,
            in_features,
        };

        let rank = pack_row_bank_rank(bank, 2, 1).unwrap();
        assert_eq!(rank.out_features, 128);
        assert_eq!(rank.in_features, 128);
        assert_eq!(rank.k_blocks, Some(1));
        assert_eq!(rank.codes.len(), expert_count * 128 * 128);
        assert_eq!(rank.scales, vec![11.0, 21.0]);
        assert_eq!(&rank.codes[..128], &codes[128..256]);
        assert_eq!(
            &rank.codes[128 * 128..128 * 128 + 128],
            &codes[code_stride + 128..code_stride + 256]
        );
    }

    #[test]
    fn tensor_parallel_row_bank_preserves_global_k_block_order() {
        let expert_count = 2;
        let out_features = 256;
        let in_features = 512;
        let code_stride = out_features * in_features;
        let mut codes = vec![0u8; expert_count * code_stride];
        for expert in 0..expert_count {
            for row in 0..out_features {
                for block in 0..4 {
                    let value = (expert * 80 + block * 16 + row % 16) as u8;
                    let start = expert * code_stride + row * in_features + block * FP8_BLOCK;
                    codes[start..start + FP8_BLOCK].fill(value);
                }
            }
        }
        let scales = vec![
            1.0f32, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0, 101.0, 102.0, 103.0, 104.0, 111.0,
            112.0, 113.0, 114.0,
        ];
        let bank = E4m3ExpertBank {
            codes: &codes,
            scales: &scales,
            expert_count,
            out_features,
            in_features,
        };

        let rank = pack_row_bank_rank(bank, 2, 1).unwrap();
        assert_eq!(rank.out_features, out_features);
        assert_eq!(rank.in_features, 256);
        assert_eq!(rank.k_blocks, Some(2));
        assert_eq!(rank.code_stride, out_features * 256);
        assert_eq!(rank.scale_stride, 4);
        assert_eq!(&rank.scales[..4], &[3.0, 13.0, 4.0, 14.0]);
        assert_eq!(&rank.scales[4..], &[103.0, 113.0, 104.0, 114.0]);

        let block_stride = out_features * FP8_BLOCK;
        assert!(rank.codes[..FP8_BLOCK].iter().all(|&code| code == 32));
        assert!(
            rank.codes[block_stride..block_stride + FP8_BLOCK]
                .iter()
                .all(|&code| code == 48)
        );
        assert!(
            rank.codes[rank.code_stride..rank.code_stride + FP8_BLOCK]
                .iter()
                .all(|&code| code == 112)
        );
        assert!(
            rank.codes
                [rank.code_stride + block_stride..rank.code_stride + block_stride + FP8_BLOCK]
                .iter()
                .all(|&code| code == 128)
        );
    }

    #[test]
    fn automatic_parallel_policy_needs_only_one_device_set_not_layer_recipes() {
        assert_eq!(parse_auto_parallel_devices(None, None).unwrap(), None);
        assert_eq!(
            parse_auto_parallel_devices(Some("auto"), Some("0,1,2,3")).unwrap(),
            Some(vec![0, 1, 2, 3])
        );
        assert!(parse_auto_parallel_devices(Some("auto"), None).is_err());
        assert!(parse_auto_parallel_devices(Some("auto"), Some("0,1,1")).is_err());
        assert!(parse_auto_parallel_devices(Some("auto"), Some("0,1,2,3,4")).is_err());
        assert!(parse_auto_parallel_devices(Some("ep"), Some("0,1")).is_err());
    }

    #[test]
    fn automatic_ep_device_router_flag_is_strict() {
        assert!(!parse_parallel_ep_device_router(None).unwrap());
        assert!(!parse_parallel_ep_device_router(Some("0")).unwrap());
        assert!(parse_parallel_ep_device_router(Some("1")).unwrap());
        assert!(parse_parallel_ep_device_router(Some("true")).is_err());
    }

    #[test]
    fn automatic_ep_pair_down_flag_is_strict_and_defaults_off() {
        assert!(!parse_parallel_ep_pair_down(None).unwrap());
        assert!(!parse_parallel_ep_pair_down(Some("0")).unwrap());
        assert!(parse_parallel_ep_pair_down(Some("1")).unwrap());
        assert!(parse_parallel_ep_pair_down(Some("true")).is_err());
    }

    #[test]
    fn automatic_ep_q8_activation_flag_is_strict() {
        assert!(!parse_parallel_ep_q8_act(None).unwrap());
        assert!(!parse_parallel_ep_q8_act(Some("0")).unwrap());
        assert!(parse_parallel_ep_q8_act(Some("1")).unwrap());
        assert!(parse_parallel_ep_q8_act(Some("true")).is_err());
    }

    #[test]
    fn automatic_ep_q8_scope_is_explicit_and_strict() {
        assert_eq!(parse_parallel_ep_q8_scope(None).unwrap(), None);
        assert_eq!(
            parse_parallel_ep_q8_scope(Some("all")).unwrap(),
            Some(ParallelEpQ8Scope::All)
        );
        assert_eq!(
            parse_parallel_ep_q8_scope(Some("gate-up")).unwrap(),
            Some(ParallelEpQ8Scope::GateUp)
        );
        assert_eq!(
            parse_parallel_ep_q8_scope(Some("down")).unwrap(),
            Some(ParallelEpQ8Scope::Down)
        );
        assert!(parse_parallel_ep_q8_scope(Some("input")).is_err());
    }

    #[test]
    fn w4a16_device_ep_accepts_a_capacity_backed_active_prefix() {
        let width = 4096;
        assert_eq!(
            nvfp4_ep_active_input_values(160 * width, 44, width).unwrap(),
            44 * width
        );
        assert_eq!(
            nvfp4_ep_active_input_values(44 * width, 44, width).unwrap(),
            44 * width
        );
        assert!(nvfp4_ep_active_input_values(43 * width, 44, width).is_err());
        assert!(
            nvfp4_ep_active_input_values(160 * width, NVFP4_EP_DEVICE_BATCH_CAP + 1, width)
                .is_err()
        );
    }

    #[test]
    fn step_ep_layer_specs_are_literal_and_fail_closed() {
        assert!(parse_step_ep_layer_specs(None).unwrap().is_empty());
        assert!(parse_step_ep_layer_specs(Some("0")).unwrap().is_empty());
        assert_eq!(
            parse_step_ep_layer_specs(Some("24@1,2")).unwrap(),
            vec![StepEpLayerSpec {
                layer: 24,
                devices: vec![1, 2],
            }]
        );
        assert_eq!(
            parse_step_ep_layer_specs(Some("24-25@1,2;31@0,2")).unwrap(),
            vec![
                StepEpLayerSpec {
                    layer: 24,
                    devices: vec![1, 2],
                },
                StepEpLayerSpec {
                    layer: 25,
                    devices: vec![1, 2],
                },
                StepEpLayerSpec {
                    layer: 31,
                    devices: vec![0, 2],
                },
            ]
        );
        assert!(parse_step_ep_layer_specs(Some("24@1")).is_err());
        assert!(parse_step_ep_layer_specs(Some("24@1,1")).is_err());
        assert!(parse_step_ep_layer_specs(Some("layer@1,2")).is_err());
        assert!(parse_step_ep_layer_specs(Some("25-24@1,2")).is_err());
        assert!(parse_step_ep_layer_specs(Some("0-128@1,2")).is_err());
        assert!(parse_step_ep_layer_specs(Some("24-25@1,2;25@0,2")).is_err());
        assert!(parse_step_ep_layer_specs(Some("all@0,1")).is_err());
    }

    #[test]
    fn step_tp_layer_specs_share_the_fail_closed_layer_contract() {
        assert!(parse_step_tp_layer_specs(None).unwrap().is_empty());
        assert!(parse_step_tp_layer_specs(Some("0")).unwrap().is_empty());
        assert_eq!(
            parse_step_tp_layer_specs(Some("24-25@1,2")).unwrap(),
            vec![
                StepTpLayerSpec {
                    layer: 24,
                    devices: vec![1, 2],
                },
                StepTpLayerSpec {
                    layer: 25,
                    devices: vec![1, 2],
                },
            ]
        );
        let error = parse_step_tp_layer_specs(Some("24@1")).unwrap_err();
        assert!(error.contains("MEMRA_STEP_TP"));
        assert!(parse_step_tp_layer_specs(Some("24@1,1")).is_err());
        assert!(parse_step_tp_layer_specs(Some("24-25@1,2;25@0,2")).is_err());

        let all = parse_step_tp_layer_specs(Some("all@0,1,2,3,4,5,6,7")).unwrap();
        assert_eq!(all.len(), STEP37_TRUNK_LAYERS);
        assert_eq!(all.first().unwrap().layer, 0);
        assert_eq!(all.last().unwrap().layer, STEP37_TRUNK_LAYERS - 1);
        let devices = (0..8).collect::<Vec<_>>();
        assert!(all.iter().all(|spec| spec.devices == devices));
        assert!(parse_step_tp_layer_specs(Some("all@0,1;44@0,1")).is_err());
    }
}

// ===== Whole-token graph builder (increment B) ==================================================
//
// The decode fns are already sectioned at every e/rank/root seam (the stage flow, sweeps_rank,
// finish splits, the dcw arm). `graph_section` is the one annotation those seams call: eager
// mode runs the closure verbatim; build mode wraps it in a stream capture on the section's
// device and records a child + its dependency edges. A token then assembles as ONE multi-device
// parent (children per section per layer), launched once per token — the launch-collapse the
// per-layer minis could not reach (routes-mini negative, 2026-08-21).

/// One captured section: the child graph plus which parent node it became, and the CUDA
/// context it was captured under (exec memset updates need it).
struct TokenGraphChild {
    #[allow(dead_code)]
    // allow: keep-alive: the child graph must outlive the exec instantiated from it
    graph: cudarc::driver::CudaGraph,
    node: cudarc::driver::sys::CUgraphNode,
    ctx: cudarc::driver::sys::CUcontext,
}

/// Exec-updatable fa geometry discovered in one attention rank child: the three partial-pool
/// memsets, the dcw fa kernel, and its combine — everything a bucket change touches. Node
/// handles address the parent's CLONED child graphs (the M1-probed update path).
struct TokenGraphFaSite {
    ctx: cudarc::driver::sys::CUcontext,
    memset_o: cudarc::driver::sys::CUgraphNode,
    memset_m: [cudarc::driver::sys::CUgraphNode; 2],
    fa: cudarc::driver::sys::CUgraphNode,
    combine: cudarc::driver::sys::CUgraphNode,
    window: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
}

pub struct TokenGraphBuilder {
    parent: cudarc::driver::sys::CUgraph,
    children: Vec<TokenGraphChild>,
    /// Nodes every NEXT section must depend on (the frontier): one node for serial flow,
    /// several while a parallel group is open.
    frontier: Vec<cudarc::driver::sys::CUgraphNode>,
    /// Detached sections: forked from the frontier at issue time, joined ONLY by the next
    /// non-group section (they never gate a parallel group merge — the SH1 shape).
    pending_detached: Vec<cudarc::driver::sys::CUgraphNode>,
    /// Open parallel group: sections issued under the same group id fork from the SAME
    /// predecessor set and merge into the frontier together when the group closes.
    group: Option<(
        u32,
        Vec<cudarc::driver::sys::CUgraphNode>,
        Vec<cudarc::driver::sys::CUgraphNode>,
    )>,
}

// SAFETY: single decode thread; graph handles are process handles.
unsafe impl Send for TokenGraphBuilder {}

impl TokenGraphBuilder {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        use cudarc::driver::sys;
        let mut parent: sys::CUgraph = std::ptr::null_mut();
        let r = unsafe { sys::cuGraphCreate(&mut parent, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("token graph create: {r:?}").into());
        }
        Ok(Self {
            parent,
            children: Vec::new(),
            frontier: Vec::new(),
            pending_detached: Vec::new(),
            group: None,
        })
    }

    fn push_child(
        &mut self,
        graph: cudarc::driver::CudaGraph,
        parallel_group: Option<u32>,
        detached: bool,
        absorb: bool,
        ctx: cudarc::driver::sys::CUcontext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use cudarc::driver::sys;
        // Resolve the dependency set: serial sections depend on the current frontier; a
        // parallel-group section depends on the frontier AS OF the group opening; a
        // DETACHED section forks like a group member but joins only the next serial section.
        let deps: Vec<sys::CUgraphNode> = match (&mut self.group, parallel_group) {
            (Some((open, base, _)), Some(group)) if *open == group => base.clone(),
            (state, Some(group)) => {
                // opening a new group (closing any previous one first)
                if let Some((_, _, members)) = state.take() {
                    self.frontier = members;
                }
                let base = self.frontier.clone();
                *state = Some((group, base.clone(), Vec::new()));
                base
            }
            (state, None) if detached => match state.as_ref() {
                Some((_, base, _)) => base.clone(),
                None => self.frontier.clone(),
            },
            (state, None) => {
                if let Some((_, _, members)) = state.take() {
                    self.frontier = members;
                }
                let mut deps = self.frontier.clone();
                if absorb {
                    deps.append(&mut self.pending_detached);
                }
                deps
            }
        };
        let mut node: sys::CUgraphNode = std::ptr::null_mut();
        let r = unsafe {
            sys::cuGraphAddChildGraphNode(
                &mut node,
                self.parent,
                if deps.is_empty() {
                    std::ptr::null()
                } else {
                    deps.as_ptr()
                },
                deps.len(),
                graph.cu_graph(),
            )
        };
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("token graph child: {r:?}").into());
        }
        match (&mut self.group, parallel_group, detached) {
            (_, None, true) => self.pending_detached.push(node),
            (Some((_, _, members)), Some(_), _) => members.push(node),
            _ => self.frontier = vec![node],
        }
        self.children.push(TokenGraphChild { graph, node, ctx });
        Ok(())
    }

    pub fn finish(mut self) -> Result<TokenGraph, Box<dyn std::error::Error>> {
        use cudarc::driver::sys;
        if let Some((_, _, members)) = self.group.take() {
            self.frontier = members;
        }
        // Discover the fa sites BEFORE instantiate: the parent's cloned child graphs hold
        // the node handles the exec update path (M1) addresses.
        let mut fa_sites = Vec::new();
        for child in &self.children {
            if let Some(site) = discover_fa_site(child.node, child.ctx)? {
                fa_sites.push(site);
            }
        }
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let r = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, self.parent, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("token graph instantiate: {r:?}").into());
        }
        Ok(TokenGraph {
            exec,
            parent: self.parent,
            _children: self.children,
            fa_sites,
        })
    }
}

/// Walk one child graph; if it carries the attention-section signature (exactly three MEMSET
/// nodes chained memset->memset->memset->fa_kernel->combine_kernel), return its update site.
fn discover_fa_site(
    child_node: cudarc::driver::sys::CUgraphNode,
    ctx: cudarc::driver::sys::CUcontext,
) -> Result<Option<TokenGraphFaSite>, Box<dyn std::error::Error>> {
    use cudarc::driver::sys;
    fn cu_try(r: sys::CUresult, what: &str) -> Result<(), Box<dyn std::error::Error>> {
        if r == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{what}: {r:?}").into())
        }
    }
    let mut graph: sys::CUgraph = std::ptr::null_mut();
    unsafe {
        cu_try(
            sys::cuGraphChildGraphNodeGetGraph(child_node, &mut graph),
            "fa-site child GetGraph",
        )?;
    }
    let mut count: usize = 0;
    unsafe {
        cu_try(
            sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut count),
            "fa-site GetNodes(count)",
        )?;
    }
    let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); count];
    unsafe {
        cu_try(
            sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count),
            "fa-site GetNodes",
        )?;
    }
    nodes.truncate(count);
    let node_type =
        |node: sys::CUgraphNode| -> Result<sys::CUgraphNodeType, Box<dyn std::error::Error>> {
            let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            unsafe {
                cu_try(
                    sys::cuGraphNodeGetType(node, &mut ty),
                    "fa-site NodeGetType",
                )?;
            }
            Ok(ty)
        };
    let memsets: Vec<sys::CUgraphNode> = {
        let mut v = Vec::new();
        for &node in &nodes {
            if node_type(node)? == sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET {
                v.push(node);
            }
        }
        v
    };
    if memsets.len() != 3 {
        return Ok(None);
    }
    // Single-stream capture makes the chain linear: follow dependent edges from each memset.
    let dependents =
        |node: sys::CUgraphNode| -> Result<Vec<sys::CUgraphNode>, Box<dyn std::error::Error>> {
            let mut n: usize = 0;
            unsafe {
                cu_try(
                    sys::cuGraphNodeGetDependentNodes_v2(
                        node,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        &mut n,
                    ),
                    "fa-site GetDependentNodes(count)",
                )?;
            }
            let mut v: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); n];
            unsafe {
                cu_try(
                    sys::cuGraphNodeGetDependentNodes_v2(
                        node,
                        v.as_mut_ptr(),
                        std::ptr::null_mut(),
                        &mut n,
                    ),
                    "fa-site GetDependentNodes",
                )?;
            }
            v.truncate(n);
            Ok(v)
        };
    // The LAST memset is the one whose direct dependent is a kernel (fa); the other two are
    // ordered among themselves but interchangeable for width updates.
    let mut fa: Option<sys::CUgraphNode> = None;
    let mut last_memset: Option<sys::CUgraphNode> = None;
    for &ms in &memsets {
        for dep in dependents(ms)? {
            if node_type(dep)? == sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
                fa = Some(dep);
                last_memset = Some(ms);
            }
        }
    }
    let (Some(fa), Some(_last)) = (fa, last_memset) else {
        return Ok(None);
    };
    let mut combine: Option<sys::CUgraphNode> = None;
    for dep in dependents(fa)? {
        if node_type(dep)? == sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            combine = Some(dep);
        }
    }
    let Some(combine) = combine else {
        return Ok(None);
    };
    // Read the fa launch geometry from its baked args (arg order pinned by fa_decode_dcw):
    // 6=hd 7=nh 8=nhkv 11=win 13=nsp 14=ski.
    let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
    unsafe {
        cu_try(
            sys::cuGraphKernelNodeGetParams_v2(fa, &mut params),
            "fa-site KernelNodeGetParams",
        )?;
    }
    let arg_i32 =
        |slot: usize| -> i32 { unsafe { *(*params.kernelParams.add(slot) as *const i32) } };
    let (hd, nh, nhkv, win) = (arg_i32(6), arg_i32(7), arg_i32(8), arg_i32(11));
    // Identify the o-partial memset (hd x wider than the m/l pair).
    let width_of = |node: sys::CUgraphNode| -> Result<usize, Box<dyn std::error::Error>> {
        let mut mp: sys::CUDA_MEMSET_NODE_PARAMS = unsafe { std::mem::zeroed() };
        unsafe {
            cu_try(
                sys::cuGraphMemsetNodeGetParams(node, &mut mp),
                "fa-site MemsetNodeGetParams",
            )?;
        }
        Ok(mp.width)
    };
    let mut widest = memsets[0];
    for &ms in &memsets[1..] {
        if width_of(ms)? > width_of(widest)? {
            widest = ms;
        }
    }
    let memset_m: Vec<sys::CUgraphNode> =
        memsets.iter().copied().filter(|&m| m != widest).collect();
    Ok(Some(TokenGraphFaSite {
        ctx,
        memset_o: widest,
        memset_m: [memset_m[0], memset_m[1]],
        fa,
        combine,
        window: win as usize,
        n_head: nh as usize,
        n_head_kv: nhkv as usize,
        head_dim: hd as usize,
    }))
}

pub struct TokenGraph {
    exec: cudarc::driver::sys::CUgraphExec,
    parent: cudarc::driver::sys::CUgraph,
    _children: Vec<TokenGraphChild>,
    fa_sites: Vec<TokenGraphFaSite>,
}

unsafe impl Send for TokenGraph {}

impl TokenGraph {
    /// Retarget every fa site to a new bucket via exec param updates (M1 path) — replaces the
    /// per-bucket whole-graph rebuild (~55ms) with ~450 node updates (~1ms). Per site the
    /// bucket caps at the layer window; nsp/ski/gridDimY and the partial-pool memset widths
    /// move together so the exec always matches what a fresh build at `bucket` would bake.
    pub fn retarget_bucket(&mut self, bucket: usize) -> Result<(), Box<dyn std::error::Error>> {
        use cudarc::driver::sys;
        fn cu_try(r: sys::CUresult, what: &str) -> Result<(), Box<dyn std::error::Error>> {
            if r == sys::CUresult::CUDA_SUCCESS {
                Ok(())
            } else {
                Err(format!("{what}: {r:?}").into())
            }
        }
        for site in &self.fa_sites {
            let layer_bucket = if site.window > 0 {
                bucket.min(site.window)
            } else {
                bucket
            };
            let sp = crate::fa_split_keys(layer_bucket, site.n_head_kv);
            let nsp = layer_bucket.div_ceil(sp).max(1);
            // fa kernel: nsp (slot 13), ski (slot 14), gridDimY = nsp.
            let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
            unsafe {
                cu_try(
                    sys::cuGraphKernelNodeGetParams_v2(site.fa, &mut params),
                    "retarget fa GetParams",
                )?;
                *(*params.kernelParams.add(13) as *mut i32) = nsp as i32;
                *(*params.kernelParams.add(14) as *mut i32) = sp as i32;
                params.gridDimY = nsp as u32;
                cu_try(
                    sys::cuGraphExecKernelNodeSetParams_v2(self.exec, site.fa, &params),
                    "retarget fa SetParams",
                )?;
            }
            // combine: nsp (slot 6).
            let mut cparams: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
            unsafe {
                cu_try(
                    sys::cuGraphKernelNodeGetParams_v2(site.combine, &mut cparams),
                    "retarget combine GetParams",
                )?;
                *(*cparams.kernelParams.add(6) as *mut i32) = nsp as i32;
                cu_try(
                    sys::cuGraphExecKernelNodeSetParams_v2(self.exec, site.combine, &cparams),
                    "retarget combine SetParams",
                )?;
            }
            // partial-pool memsets: o = nh*nsp*hd elements, m/l = nh*nsp.
            let set_width =
                |node: sys::CUgraphNode, width: usize| -> Result<(), Box<dyn std::error::Error>> {
                    let mut mp: sys::CUDA_MEMSET_NODE_PARAMS = unsafe { std::mem::zeroed() };
                    unsafe {
                        cu_try(
                            sys::cuGraphMemsetNodeGetParams(node, &mut mp),
                            "retarget memset GetParams",
                        )?;
                    }
                    mp.width = width;
                    unsafe {
                        cu_try(
                            sys::cuGraphExecMemsetNodeSetParams(self.exec, node, &mp, site.ctx),
                            "retarget memset SetParams",
                        )?;
                    }
                    Ok(())
                };
            set_width(site.memset_o, site.n_head * nsp * site.head_dim)?;
            set_width(site.memset_m[0], site.n_head * nsp)?;
            set_width(site.memset_m[1], site.n_head * nsp)?;
        }
        Ok(())
    }

    pub fn launch(&self, e: &Engine) -> Result<(), Box<dyn std::error::Error>> {
        use cudarc::driver::sys;
        let _main = e.gpu.enter_main()?;
        let r = unsafe { sys::cuGraphLaunch(self.exec, e.stream().cu_stream() as sys::CUstream) };
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("token graph launch: {r:?}").into());
        }
        Ok(())
    }
}

impl Drop for TokenGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::sys::cuGraphExecDestroy(self.exec);
            let _ = cudarc::driver::sys::cuGraphDestroy(self.parent);
        }
    }
}

std::thread_local! {
    static TOKEN_GRAPH_BUILDER: std::cell::RefCell<Option<TokenGraphBuilder>> =
        const { std::cell::RefCell::new(None) };
}

/// Arm the thread-local builder (build mode) — the next `graph_section` calls capture.
pub fn token_graph_build_begin() -> Result<(), Box<dyn std::error::Error>> {
    let builder = TokenGraphBuilder::new()?;
    TOKEN_GRAPH_BUILDER.with(|cell| *cell.borrow_mut() = Some(builder));
    Ok(())
}

/// Take the finished parent (ends build mode).
pub fn token_graph_build_finish() -> Result<TokenGraph, Box<dyn std::error::Error>> {
    let builder = TOKEN_GRAPH_BUILDER
        .with(|cell| cell.borrow_mut().take())
        .ok_or("token graph build was not begun")?;
    builder.finish()
}

/// True while the thread-local builder is armed.
pub fn token_graph_building() -> bool {
    TOKEN_GRAPH_BUILDER.with(|cell| cell.borrow().is_some())
}

/// The section annotation: eager mode runs the closure verbatim; build mode wraps it in a
/// stream capture on `engine`'s stream and records the child. Sections sharing a
/// `parallel_group` id fork from the same predecessor set and merge together. The closure
/// must be capture-safe (raw copies at cross-context seams, no host syncs, no events).
pub fn graph_section<F>(
    engine: &Engine,
    parallel_group: Option<u32>,
    f: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    graph_section_opts(engine, parallel_group, false, false, f)
}

/// Serial section that ALSO joins every pending detached section (the SH1 consumer shape).
pub fn graph_section_absorbing<F>(engine: &Engine, f: F) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    graph_section_opts(engine, None, false, true, f)
}

/// `graph_section` with the DETACHED shape: forks from the current frontier (or the open
/// group base) and is joined only by the next serial section — never gates a group merge.
pub fn graph_section_detached<F>(engine: &Engine, f: F) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    graph_section_opts(engine, None, true, false, f)
}

pub fn graph_section_opts<F>(
    engine: &Engine,
    parallel_group: Option<u32>,
    detached: bool,
    absorb: bool,
    f: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    let building = token_graph_building();
    if !building {
        let mut f = f;
        return f();
    }
    let (child, ctx) = {
        let _main = engine.gpu.enter_main()?;
        let mut ctx: cudarc::driver::sys::CUcontext = std::ptr::null_mut();
        let r = unsafe { cudarc::driver::sys::cuCtxGetCurrent(&mut ctx) };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("graph section ctx query: {r:?}").into());
        }
        let mut f = f;
        // NO WARMUP RUNS: section bodies carry device side effects (dcw appends, counter
        // incs) that a warmup would really execute — the len_d-drift crash of 2026-08-21.
        let (child, _retained) = engine.capture_graph_retained_nowarm(|_| f())?;
        (child, ctx)
    };
    TOKEN_GRAPH_BUILDER.with(|cell| {
        cell.borrow_mut()
            .as_mut()
            .expect("builder checked above")
            .push_child(child, parallel_group, detached, absorb, ctx)
    })
}
