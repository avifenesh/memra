//! glm5_next T=1 DECODE CUDA GRAPHS, per pipeline stage (door `MEMRA_GLM5_DECODE_GRAPH`,
//! default OFF — lane/b200-glm5-graph-20260902).
//!
//! WHY THIS EXISTS. The nsys census of a plain t=1 decode token on 2x B200 SXM
//! (GLM-5.3-Flash NVFP4, resident PP2) reads ~2,900 kernel launches per token and ~6 ms of
//! launch/gap at ~2.2 us each inside a ~24 ms token. That remainder is issue cost, not
//! dependency latency: the host spends its own time enqueueing 45 layers of small work. A
//! replayable graph is the only mechanism that removes issue cost without changing a single
//! kernel — and this family had none, because the walk was not capturable at all.
//!
//! WHAT MADE IT UNCAPTURABLE, and what this lane moved. A capture region admits no
//! `cuStreamSynchronize` and no pageable HtoD. The T=1 walk carried three:
//!
//!   1. the per-MoE-layer router readback (`Engine::moe_router_sigmoid_topk_host`, lib.rs) —
//!      a pinned sel/w DtoH pair plus a full device drain, 42 per token, existing ONLY so the
//!      host could compute `base + ex*stride`. Moved to the device by the T=1 arm of door D
//!      (`vrows_t1_dev` in `hybrid_forward.rs`, kernel `moe_vrows_tables_from_sel`);
//!   2. the shared expert's `e.htod(&vec![1.0f32; t])` per MoE layer — already solved by door
//!      H (`MEMRA_HTOD_DIET`, `add_scaled_rows_ones`); this door REQUIRES it;
//!   3. the MLA/DSA layers' host-derived launch geometry (`let slot = layer.len` at
//!      `HybridModel::mla_attn_cached_pre_wo`, and the kpool `n_pools`/`select_k`/`width`
//!      derived from it) plus the synchronizing 4-byte `len_d` mirror store. NOT moved in this
//!      lane — so MLA/DSA layers stay EAGER and are the reason the capture is per-run rather
//!      than whole-token. See `research/b200-glm5-graph-20260902/LANE.md` for the file:line
//!      inventory and what a device-counter (`_dc`) kpool arm would need.
//!
//! WHAT IS CAPTURED. Per pipeline stage, the maximal CONTIGUOUS runs of KDA-mixer layers
//! inside that stage's `[lo, hi)` range: hc pre/post glue, the KDA mixer (projections, conv
//! ring, delta-rule scan, gated norm, `wo`), the FFN-input norm, the routed MoE with device
//! tables, and the shared expert. Those layers hold NO position-derived launch parameter: the
//! conv ring rolls in-kernel and the recurrent state is read/written by pointer, so a run
//! graph is position-INDEPENDENT and never needs a re-capture per token. The stage's KV/latent
//! state stays where it is; the only thing carried across the eager/graph seam is the stream
//! state, through a stable `x_io` buffer.
//!
//! ONE CONSEQUENCE WORTH STATING RATHER THAN LEAVING IMPLICIT: a captured run always runs the
//! WORKSPACE form of the walk (`hyper_range_decode_ws_body` against this pool's private
//! `HyperDecodeWs`), whatever `MEMRA_HC_DECODE_WS` says, because a capture needs stable operand
//! addresses. So under this door the captured layers get the `MEMRA_HC_DECODE_WS` semantics for
//! free. That is not a numeric change — the two hc walks are kept call-for-call in step and
//! their byte identity is gated by `tests/hc_decode_ws_gpu.rs` — but it IS a dispatch the door
//! makes on the caller's behalf, so it is written here and in the FLAGS.md row rather than
//! discovered from a counter.
//!
//! THE ONE THING THAT IS NOT POSITION-INDEPENDENT: the KDA recurrent-state PING-PONG.
//! `kda::kda_cached` runs the scan `ssm_state -> ssm_state_alt` and then swaps the two owned
//! buffers on the HOST; its own doc says the arrangement is "NOT capture-safe: a captured
//! graph bakes capture-time pointers and never re-runs the host swap". Rather than add an
//! in-graph copy-back (which would be an extra launch per KDA layer per token, and a different
//! program), each run is captured TWICE — once in each ping-pong phase — and the replays
//! alternate. Phase p reads exactly the buffer phase p-1 wrote, which is what the eager step
//! does; the host fields are swapped after every replay so a fall-back to eager, a prime, or a
//! snapshot sees the same pointers it would have seen. Capture itself is side-effect free
//! (stream capture RECORDS, it does not execute — hence `capture_graph_retained_nowarm`, whose
//! warmup-free contract exists for exactly this class of body).
//!
//! POOL IDENTITY AND INVALIDATION. A run graph bakes this cache's own state pointers, so the
//! pool lives on the `Cache` (as `Any`, because memra-kv must not depend on the engine's
//! cudarc handles) and is keyed by `(device ordinal, lo, hi)`. It also records the `cache.pos`
//! it expects next: any seam that rewinds or re-seats a session (rollback, reuse-pool retire,
//! a prefix restore) moves `pos`, and the pool is dropped and re-captured rather than replayed
//! against state it no longer describes. Fail-closed, never silently stale.
//!
//! THE DEVICE MoE ARM IS THE DOOR'S PROGRAM, NOT A CAPTURE SIDE EFFECT. `vrows_t1_dev` engages
//! wherever `MEMRA_GLM5_DECODE_GRAPH` is set, including on a range whose capture this module
//! refused. Keeping the two halves independent is deliberate: a refusal then changes only
//! whether the launches are RECORDED, never what they compute, so the eager fall-through of a
//! doored process is one program rather than two.
//!
//! ELIGIBILITY IS PRE-DECIDED AND FAIL-CLOSED (`HybridModel::glm5_decode_graph_ready`, which
//! wraps `glm5_decode_graph_refusal` plus the launch-headroom guard). Any miss — a
//! non-KDA layer inside a run, a MoE layer whose device-table arm would not fire, an armed
//! observation mode, a TP shard, a sharded/absent expert slab, driver free below the graph
//! launch floor — takes the eager walk for the whole range, byte-identically, and says so once.

use crate::Engine;
use crate::hybrid::{HybridModel, Mixer};
use cudarc::driver::{CudaGraph, CudaSlice};
use memra_kv::Cache;
use std::sync::atomic::Ordering;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// One captured contiguous KDA-layer run, in both recurrent-state ping-pong phases.
///
/// FIELD ORDER IS LOAD-BEARING. Rust drops fields in declaration order, so `graphs` (which
/// destroys the instantiated execs) must be declared BEFORE `_keeper` (which frees the buffers
/// those execs baked). Freeing first and destroying second is a use-after-free inside the driver.
/// The same reasoning orders `StageGraphs`: `runs` before `x_io`/`ws`/`f16`.
struct RunGraph {
    lo: usize,
    hi: usize,
    /// `[phase 0, phase 1]` — phase p was captured with `recur[il].ssm_state` naming the buffer
    /// the eager walk would read at a step of that parity.
    graphs: [CudaGraph; 2],
    /// WHICH PING-PONG PHASE THIS RUN'S NEXT REPLAY MUST USE — per RUN, not per stage, and that
    /// distinction is the whole defect box takes 1 through 12 chased.
    ///
    /// `kda_cached` swaps `ssm_state`/`ssm_state_alt` once per KDA layer per token, so a layer's
    /// buffer assignment alternates with the TOKEN index. Every run of a stage therefore sits at
    /// the same parity as every other run, and each advances by one per token. Holding one
    /// `phase` on the stage and flipping it inside `glm5_replay_run` flipped it ONCE PER RUN
    /// instead: with `r` runs in a stage, run 0 replayed phase 0 (correct), run 1 replayed phase
    /// 1 while its layers were still at phase 0, run 2 replayed phase 0 again, and so on. A
    /// wrong-phase graph reads the buffer the eager walk WROTE this token and writes the one it
    /// READ, which is a plausible-looking wrong answer, not a crash.
    ///
    /// It is invisible with one run per stage, which is why the 2-layer rig fixture passed 16
    /// steps bit-identically and the 8-layer one (runs = [0,3) and [4,7)) failed on the capture
    /// step with run [0,3) byte-identical and run [4,7) diverging — the first trace that ever
    /// separated the two. On the box's `runs=6` stages every other run was wrong from token 0,
    /// which compounds to the constant `graph=0` tape.
    phase: usize,
    /// Everything the captured bodies allocated and would otherwise return to an engine pool.
    /// Held for the life of the graphs: a transient handed back to the pool and re-issued to
    /// eager work is a buffer every replay scribbles (the draft-graph root cause).
    /// memra#131: the first replay of this run is COMPARED bitwise against an eager walk of the
    /// same layers on the same input (recurrent state snapshotted and restored around the
    /// reference), and the stage is latched eager on any mismatch. Until that check has passed
    /// the door has not earned its `engaged` line: a health flag that is not derived from the
    /// output is the defect the issue names. Counts the replays verified so far: the run is
    /// trusted once it reaches `selfcheck_n()` (`MEMRA_GLM5_GRAPH_SELFCHECK_N`, default 1).
    /// Each verified replay compares the OUTPUT and the recurrent STATE the replay wrote
    /// (conv/ssm/alt of every captured layer) against the eager walk's, bitwise.
    checked: u32,
    _keeper: Vec<Box<dyn std::any::Any + Send>>,
}

// The raw CUgraph/CUgraphExec handles inside `CudaGraph` are CONTEXT-bound, not thread-bound —
// the same claim `graph_update::KernelNode` makes. A `Cache` moves between worker threads
// (reuse pool, stage split), and its captured graphs move with it; what must not happen is two
// threads replaying one exec concurrently, and the engine already serializes a session's decode
// work on one stream at a time.
unsafe impl Send for RunGraph {}
unsafe impl Send for StageGraphs {}

/// The captured graphs for ONE pipeline stage of one session.
struct StageGraphs {
    dev: usize,
    lo: usize,
    hi: usize,
    runs: Vec<RunGraph>,
    /// Stable stream-state buffer the run graphs read and write. The walk's hc ping-pong makes
    /// an EVEN number of buffer swaps per layer (post-attn and post-FFN), so a run of any length
    /// leaves its output in this same buffer — that is the replay contract.
    x_io: CudaSlice<f32>,
    /// DEDICATED OUTPUT. Box run 4 (2026-09-02) had the door running the whole walk cleanly and
    /// still producing token 0 at every step: zero logits, i.e. the captured range's result never
    /// reached the eager remainder. The old contract was an ARGUMENT — "the hc walk swaps `x`
    /// against `ws.xb` twice per layer, so an even number of swaps leaves the output back in
    /// `x_io`" — and an argument about which of two aliased buffers holds the answer is exactly
    /// the thing not to rely on inside a captured graph. The captured body now ENDS with a copy
    /// of its live state into this third buffer, recorded as a memcpy node, so the replay's
    /// output lands in one known place whatever the parity did.
    x_out: CudaSlice<f32>,
    /// Private hc-glue workspace, resident during capture and never returned to the engine pool.
    ws: crate::hyper::HyperDecodeWs,
    /// Private f16 GEMM scratch, swapped resident around capture AND around every replay: the
    /// graphs bake its cvt/Lt pointers, and an eager GEMM between replays would cross-contaminate
    /// them (the `PrimeGraph` precedent).
    f16: Option<crate::f16_ffi::F16Scratch>,
    /// The `cache.pos` this pool describes. A mismatch means the session moved under the pool.
    next_pos: usize,
    /// `(layer, conv_state, ssm_state, ssm_state_alt)` DEVICE POINTERS as of capture, for every
    /// captured layer. The graphs baked these addresses, so a seam that REPLACES a state buffer
    /// rather than overwriting it leaves the graphs pointing at freed memory — a silent
    /// wrong-answer or a segfault, whichever the allocator hands back. `pos` continuity catches
    /// a rewind; this catches a re-seat, which `pos` alone cannot see. Verified before every
    /// token's replays; any mismatch drops the stage and re-captures.
    state_sig: Vec<LayerSig>,
    /// Has an exec of this stage been LAUNCHED since the last time the stream was drained? The
    /// re-capture path destroys execs and frees the buffers they baked, and doing that with a
    /// launch still outstanding is a destroy-in-use; this is what the error line reports so a box
    /// run can say whether that was the state, rather than leaving it to inference.
    launched_since_sync: bool,
}

/// The `(conv, ssm, ssm_alt)` device pointers of one layer's recurrent state, or `None` when the
/// layer has no recurrent slot.
/// The buffers a stage's graphs bake, carried across a re-capture unchanged. Splitting them out
/// is what lets a rebuild destroy ONLY the graphs.
struct StageBufs {
    x_io: CudaSlice<f32>,
    x_out: CudaSlice<f32>,
    ws: crate::hyper::HyperDecodeWs,
    f16: Option<crate::f16_ffi::F16Scratch>,
}

/// Per-session pool: one [`StageGraphs`] per pipeline stage this cache has decoded through.
#[derive(Default)]
pub(crate) struct Glm5DecodeGraphs {
    stages: Vec<StageGraphs>,
    /// `(device ordinal, lo, hi)` keys that are latched to the eager walk for the rest of the
    /// session: a capture that FAILED once, or (since box run 3) a stage whose state signature
    /// moved under it. Both are permanent eager fall-throughs rather than per-token retries.
    failed: Vec<(usize, usize, usize)>,
}

/// Maximal contiguous runs of `[lo, hi)` over which `capturable` holds. Split out from
/// [`kda_runs`] with no model in it so the unit test drives THIS function rather than a
/// look-alike copy of it: the run geometry is the whole capture contract, and a wrong split
/// either captures a layer whose launch geometry depends on `cache.pos` or silently drops one.
fn capturable_runs(
    lo: usize,
    hi: usize,
    capturable: impl Fn(usize) -> bool,
) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start: Option<usize> = None;
    for il in lo..hi {
        match (capturable(il), start) {
            (true, None) => start = Some(il),
            (false, Some(a)) => {
                runs.push((a, il));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(a) = start {
        runs.push((a, hi));
    }
    runs
}

/// Maximal contiguous runs of KDA-mixer layers inside `[lo, hi)`. A TP-sharded KDA layer is not
/// on this path at all, so it breaks a run rather than joining one.
fn kda_runs(m: &HybridModel, lo: usize, hi: usize) -> Vec<(usize, usize)> {
    capturable_runs(
        lo,
        hi,
        |il| matches!(&m.layers[il].mixer, Mixer::Kda(la) if la.tp.is_none()),
    )
}

/// One captured layer's state-buffer identity.
///
/// PHASE-INVARIANT BY CONSTRUCTION: `kda_cached` swaps `ssm_state`/`ssm_state_alt` on the HOST
/// every step and the replay mirrors that swap, so the ping-pong pair is recorded UNORDERED
/// (`p_lo`, `p_hi`). A swap cannot change it; a genuine re-seat still does.
#[derive(Clone, Copy, PartialEq, Eq)]
struct LayerSig {
    il: usize,
    conv: u64,
    p_lo: u64,
    p_hi: u64,
}

fn recur_sig(e: &Engine, cache: &Cache, il: usize) -> Option<LayerSig> {
    use cudarc::driver::DevicePtr;
    let rl = cache.recur.get(il)?.as_ref()?;
    let st = e.stream();
    let (conv, _g0) = rl.conv_state.device_ptr(&st);
    let (s, _g1) = rl.ssm_state.device_ptr(&st);
    let (a, _g2) = rl.ssm_state_alt.device_ptr(&st);
    Some(LayerSig {
        il,
        conv,
        p_lo: s.min(a),
        p_hi: s.max(a),
    })
}

/// The signature of every layer inside `[lo, hi)`'s captured runs, in capture order.
fn stage_sig(m: &HybridModel, e: &Engine, cache: &Cache, lo: usize, hi: usize) -> Vec<LayerSig> {
    kda_runs(m, lo, hi)
        .iter()
        .flat_map(|(a, b)| *a..*b)
        .filter_map(|il| recur_sig(e, cache, il))
        .collect()
}

/// NAME what moved, rather than reporting a bare "moved". Box run 3 (2026-09-02) showed the
/// engine deciding to re-capture on step 2 of a real artifact where the ping-pong swap is the
/// only thing that should have changed — so the interesting output is WHICH element differs and
/// what the two pointers were, not that some element did.
fn sig_diff(old: &[LayerSig], new: &[LayerSig]) -> Option<String> {
    if old.len() != new.len() {
        return Some(format!(
            "layer-count {} -> {} (a captured layer lost or gained its recurrent slot)",
            old.len(),
            new.len()
        ));
    }
    for (o, n) in old.iter().zip(new.iter()) {
        if o.il != n.il {
            return Some(format!("layer order {} -> {}", o.il, n.il));
        }
        if o.conv != n.conv {
            return Some(format!(
                "layer {} conv_state 0x{:x} -> 0x{:x}",
                o.il, o.conv, n.conv
            ));
        }
        if o.p_lo != n.p_lo || o.p_hi != n.p_hi {
            return Some(format!(
                "layer {} ssm pair {{0x{:x}, 0x{:x}}} -> {{0x{:x}, 0x{:x}}}",
                o.il, o.p_lo, o.p_hi, n.p_lo, n.p_hi
            ));
        }
    }
    None
}

/// Announce the door's decision, with the reason. Once per DISTINCT message, not once per
/// process: a single `Once` would let an early refusal on one stage silence the engagement line
/// of another, and the gate reads these to explain a vacuous run.
/// `MEMRA_GLM5_GRAPH_SELFCHECK_N` (default 1): how many of a run's first replays are verified
/// against the eager walk (output and written state) before the run is trusted. 1 is the
/// fence; a larger N is the diagnostic that says at which replay, and in which ping-pong
/// phase, a replay first diverges.
fn selfcheck_n() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("MEMRA_GLM5_GRAPH_SELFCHECK_N")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(1)
    })
}

fn note(msg: &str) {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::BTreeSet<String>>> =
        std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    if seen.lock().unwrap().insert(msg.to_string()) {
        eprintln!("[glm5-decode-graph] {msg}");
    }
}

/// Where a capture is when something fails, for the one-line error report. The rig cannot
/// reproduce this path (no glm5 artifact, exactness-only card), so the box run is the only probe
/// and it has to come back with enough to name the failing driver call rather than a bare
/// `DriverError(CUDA_ERROR_INVALID_VALUE)`.
#[derive(Clone, Copy)]
struct CapCtx {
    dev: usize,
    lo: usize,
    hi: usize,
    run: usize,
    runs: usize,
    a: usize,
    b: usize,
    phase: usize,
    /// False for the pool's first build, true for every rebuild after an invalidation. This is
    /// THE axis that matters: the first build works on the box and the rebuild does not.
    recapture: bool,
}

impl std::fmt::Display for CapCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dev={} stage=[{}, {}) run={}/{} layers=[{}, {}) phase={} recapture={}",
            self.dev,
            self.lo,
            self.hi,
            self.run,
            self.runs,
            self.a,
            self.b,
            self.phase,
            self.recapture
        )
    }
}

/// `cuStreamIsCapturing` on this engine's stream. A capture that begins on a stream already
/// ACTIVE or INVALIDATED is one of the few things that returns `CUDA_ERROR_INVALID_VALUE` from
/// `cuStreamBeginCapture`, and it is invisible without asking.
fn capture_status(e: &Engine) -> &'static str {
    use cudarc::driver::sys;
    let mut st = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let rc = unsafe { sys::cuStreamIsCapturing(e.stream().cu_stream(), &mut st) };
    if rc != sys::CUresult::CUDA_SUCCESS {
        return "query-failed";
    }
    match st {
        sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE => "none",
        sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE => "ACTIVE",
        // The enum is exhaustive in cudarc 0.19, so no catch-all arm (clippy rejects an
        // unreachable one). A future CUDA status would be a compile error here, which is the
        // right way to find out.
        sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED => "INVALIDATED",
    }
}

fn free_mb(e: &Engine) -> String {
    match e.ctx().mem_get_info() {
        Ok((free, total)) => format!("{}/{}MB", free >> 20, total >> 20),
        Err(_) => "?".to_string(),
    }
}

/// The error line every failable step in this module routes through. Printed BEFORE the error is
/// returned, so a run that dies still says which call, where, and what the stream looked like.
fn capture_error(e: &Engine, ctx: &CapCtx, call: &str, err: &dyn std::fmt::Display) {
    eprintln!(
        "[glm5-decode-graph] capture-error: call={call} {ctx} stream_capture={} free={} \
         ledger={} err={err}",
        capture_status(e),
        free_mb(e),
        crate::glm5_sel_ledger::armed(),
    );
}

/// Run one failable driver step with context: report before propagating.
fn step<T>(
    e: &Engine,
    ctx: &CapCtx,
    call: &str,
    r: Result<T, Box<dyn std::error::Error>>,
) -> Res<T> {
    match r {
        Ok(v) => Ok(v),
        Err(err) => {
            capture_error(e, ctx, call, &err);
            Err(err)
        }
    }
}

/// ONE capture, every driver step named. This is a deliberate re-implementation of
/// `Engine::capture_graph_retained_nowarm` (same mode, same instantiate flag, same
/// event-tracking discipline) for one reason: that helper returns a bare `DriverError` and the
/// box run cannot tell `cuStreamBeginCapture` from `cuStreamEndCapture`+instantiate from
/// `cuGraphUpload`. Keep the two in step if either moves.
///
/// The stream's capture status is probed BEFORE the begin. A stream left ACTIVE (a previous
/// capture that never ended) or INVALIDATED (a capture killed by an illegal in-region operation)
/// makes `cuStreamBeginCapture` return `CUDA_ERROR_INVALID_VALUE`, and that is one of the very
/// few callers that produce exactly this error.
fn capture_one<F>(e: &Engine, ctx: &CapCtx, mut body: F) -> Res<CudaGraph>
where
    F: FnMut(&Engine) -> Res<()>,
{
    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
    let pre = capture_status(e);
    if pre != "none" {
        capture_error(
            e,
            ctx,
            "pre-begin-status",
            &format!("stream capture status is {pre}"),
        );
        return Err(format!("glm5 decode graph: stream capture status {pre} before begin").into());
    }
    step(
        e,
        ctx,
        "synchronize(before begin_capture)",
        e.stream().synchronize().map_err(Into::into),
    )?;
    // Capture is strictly single-stream; cudarc's per-buffer cross-stream event waits are not
    // permitted inside a capture region (the `capture_graph` note in lib.rs).
    let was_tracking = e.ctx().is_event_tracking();
    if was_tracking {
        unsafe { e.ctx().disable_event_tracking() };
    }
    let out = (|| -> Res<CudaGraph> {
        step(
            e,
            ctx,
            "cuStreamBeginCapture(RELAXED)",
            e.stream()
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(Into::into),
        )?;
        // The body's own error is reported by whichever `step` inside it failed; a body failure
        // still has to END the capture or the stream stays ACTIVE and every later begin fails.
        let body_res = body(e);
        let ended = e
            .stream()
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        if let Err(err) = body_res {
            capture_error(e, ctx, "capture body", &err);
            return Err(err);
        }
        let graph = step(
            e,
            ctx,
            "cuStreamEndCapture+cuGraphInstantiate",
            ended.map_err(Into::into),
        )?
        .ok_or_else(|| {
            capture_error(e, ctx, "cuStreamEndCapture", &"capture produced no graph");
            "glm5 decode graph: capture produced no graph (stream was not capturing)"
        })?;
        step(e, ctx, "cuGraphUpload", graph.upload().map_err(Into::into))?;
        Ok(graph)
    })();
    if was_tracking {
        unsafe { e.ctx().enable_event_tracking() };
    }
    out
}

/// Checksum + liveness of a stream-state buffer, for the gate trace. `nz` is the point: an
/// all-zero hidden and a wrong-but-live hidden look identical in a token stream and completely
/// different here.
fn x_sum(e: &Engine, x: &CudaSlice<f32>) -> String {
    match e.dtoh(x) {
        Ok(v) => {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            let mut nz = 0usize;
            let mut absmax = 0f32;
            for f in &v {
                h ^= f.to_bits() as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
                if *f != 0.0 {
                    nz += 1;
                }
                if f.abs() > absmax {
                    absmax = f.abs();
                }
            }
            format!("sum=0x{h:016x} nz={nz}/{} absmax={absmax:.6e}", v.len())
        }
        Err(err) => format!("sum=? ({err})"),
    }
}

#[allow(clippy::too_many_arguments)] // allow: a trace line's fields are its whole point
fn trace_seg(
    e: &Engine,
    dev: usize,
    lo: usize,
    hi: usize,
    a: usize,
    b: usize,
    arm: &str,
    pos: usize,
    x: &CudaSlice<f32>,
) {
    if !crate::glm5_graph_trace_on() {
        return;
    }
    eprintln!(
        "[glm5-graph-trace] pos={pos} dev={dev} stage=[{lo}, {hi}) seg=[{a}, {b}) arm={arm} {}",
        x_sum(e, x)
    );
}

impl HybridModel {
    /// Everything this door needs that is NOT a property of one layer. Returns the refusal
    /// reason so the once-note can name it instead of failing silently.
    fn glm5_decode_graph_refusal(
        &self,
        e: &Engine,
        cache: &Cache,
        lo: usize,
        hi: usize,
    ) -> Option<String> {
        if self.hyper.is_none() {
            return Some("model carries no HyperConnections topology".into());
        }
        if self.cfg.sigmoid_router().is_none() {
            return Some(
                "this trunk has no sigmoid router (the device-table MoE arm needs one)".into(),
            );
        }
        if crate::glm5_graph_no_capture() {
            return Some(
                "MEMRA_GLM5_GRAPH_NO_CAPTURE: capture disabled while the door's device-table MoE \
                 arm stays engaged (the half of the bisect MEMRA_GLM5_GRAPH_HOST_MOE could not \
                 supply, since that one turns off both enablers at once)"
                    .into(),
            );
        }
        if crate::glm5_graph_host_moe() {
            return Some(
                "MEMRA_GLM5_GRAPH_HOST_MOE forces the host-oracle MoE: its per-layer readback and \
                 stream drain cannot live inside a capture region (bisect arm)"
                    .into(),
            );
        }
        if !crate::htod_diet_on() {
            return Some(
                "MEMRA_HTOD_DIET is off: the shared expert still uploads a pageable constant \
                 per MoE layer, which a capture region refuses"
                    .into(),
            );
        }
        if !crate::hybrid_forward::sigmoid_router_enabled() {
            return Some(
                "MEMRA_SIG_ROUTER=0 selects the host oracle (no device selection to capture)"
                    .into(),
            );
        }
        if crate::moesd::capture_active() || memra_reference::hidden_trace::enabled() {
            return Some("a host-visible route/hidden observer is armed".into());
        }
        for env in [
            "MEMRA_MOE_STATS",
            "MEMRA_MOE_TRACE",
            "MEMRA_MOE_WEIGHT_TRACE",
            "MEMRA_MOE_INPUT_TRACE_DIR",
            "MEMRA_SIG_ROUTER_LOGIT_TRACE",
            // `MEMRA_MOE_SEL_DUMP` (main #113, met this lane in a rebase 2026-09-03) is the one
            // that reads the selection back on the DEVICE arm rather than the host arm, which is
            // precisely the arm a captured body runs. Its DtoH pair would be RECORDED and not
            // executed inside a capture region, so the dump would write stale rows and the door
            // would look fine. Refused by name like every other observer.
            "MEMRA_MOE_SEL_DUMP",
        ] {
            if std::env::var_os(env).is_some() {
                return Some(format!(
                    "{env} is armed (its consumer reads the selection on the host)"
                ));
            }
        }
        if crate::spill_pread::worker_enabled() && crate::spill_pread::copy_h2d_enabled() {
            return Some("the NVMe worker H2D promotion reads the host selection".into());
        }
        if e.ctx().is_event_tracking() {
            return Some(
                "cudarc event tracking is on (MEMRA_EVT); capture refuses cross-stream waits"
                    .into(),
            );
        }
        for il in lo..hi {
            if let Mixer::Kda(la) = &self.layers[il].mixer
                && la.tp.is_some()
            {
                return Some(format!("layer {il} is glm5-TP sharded"));
            }
            if cache.recur.get(il).is_none() {
                return Some(format!(
                    "layer {il} has no recurrent-state slot in this cache"
                ));
            }
        }
        let runs = kda_runs(self, lo, hi);
        if runs.is_empty() {
            return Some(format!("stage [{lo}, {hi}) holds no KDA layer to capture"));
        }
        // Per-layer MoE admission, checked BEFORE the capture opens: a layer that fell through to
        // the host readback would issue a `cuStreamSynchronize` inside the capture region and fail
        // the whole request instead of yielding to the eager walk. Only the layers that would
        // actually be captured are checked.
        for (a, b) in runs {
            for il in a..b {
                if !HybridModel::glm5_t1_dev_moe_ready(e, &self.layers[il], &self.cfg, il) {
                    return Some(format!(
                        "layer {il}: the T=1 device-table MoE arm would not fire (it needs \
                         slab-local uniform q8 experts, a PRE clamp, and n_used <= 8)"
                    ));
                }
            }
        }
        None
    }

    /// The eager walk, split at the SAME run boundaries the graph arm replays at, so the two
    /// arms' trace lines line up and the first differing segment names the seam. Splitting the
    /// loop changes nothing: `hyper_range_decode_eager` over `[lo, m)` then `[m, hi)` issues the
    /// identical kernel sequence it issues over `[lo, hi)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hyper_range_decode_eager_traced(
        &self,
        e: &Engine,
        topology: &crate::hyper::HyperTopology,
        mut x: CudaSlice<f32>,
        lo: usize,
        hi: usize,
        pos_d: &CudaSlice<i32>,
        pos: usize,
        cache: &mut Cache,
    ) -> Res<CudaSlice<f32>> {
        let dev = e.ctx().ordinal();
        let runs = kda_runs(self, lo, hi);
        let mut cursor = lo;
        for (a, b) in runs {
            if a > cursor {
                x = self.hyper_range_decode_eager(e, topology, x, cursor, a, pos_d, pos, cache)?;
                trace_seg(e, dev, lo, hi, cursor, a, "eager-gap", pos, &x);
            }
            x = self.hyper_range_decode_eager(e, topology, x, a, b, pos_d, pos, cache)?;
            trace_seg(e, dev, lo, hi, a, b, "eager-run", pos, &x);
            cursor = b;
        }
        if cursor < hi {
            x = self.hyper_range_decode_eager(e, topology, x, cursor, hi, pos_d, pos, cache)?;
            trace_seg(e, dev, lo, hi, cursor, hi, "eager-gap", pos, &x);
        }
        Ok(x)
    }

    /// Door admission, evaluated BEFORE the walk takes ownership of the stream state so a
    /// refusal costs nothing and the caller keeps `x`. Names its reason once per distinct reason.
    pub(crate) fn glm5_decode_graph_ready(
        &self,
        e: &Engine,
        cache: &Cache,
        lo: usize,
        hi: usize,
    ) -> bool {
        // A stage whose capture already failed once stays eager for the life of this session —
        // the reason was printed then, so this arm is silent by design.
        if cache
            .glm5_decode_graph
            .as_ref()
            .and_then(|b| b.downcast_ref::<Glm5DecodeGraphs>())
            .is_some_and(|p| {
                p.failed
                    .iter()
                    .any(|&(d, l, h)| d == e.ctx().ordinal() && l == lo && h == hi)
            })
        {
            return false;
        }
        if let Some(why) = self.glm5_decode_graph_refusal(e, cache, lo, hi) {
            note(&format!("eager: {why}"));
            return false;
        }
        // GRAPH-LAUNCH HEADROOM GUARD (spec::GRAPH_LAUNCH_MIN_FREE): below the driver-free floor
        // `cuGraphLaunch` segfaults inside libcuda. This route HAS a byte-identical eager twin,
        // so it yields the whole range rather than failing the request.
        if !crate::spec::graph_launch_headroom_ok(e) {
            static SUSPENDED: std::sync::Once = std::sync::Once::new();
            SUSPENDED.call_once(|| crate::spec::graph_replay_suspended_note("glm5-decode-graph"));
            return false;
        }
        true
    }

    /// The graphed twin of `hyper_range_decode` (door `MEMRA_GLM5_DECODE_GRAPH`). Replays the
    /// stage's captured KDA runs and walks every other layer eagerly, in the SAME layer order.
    /// Only called once [`Self::glm5_decode_graph_ready`] has admitted the range.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hyper_range_decode_graphed(
        &self,
        e: &Engine,
        topology: &crate::hyper::HyperTopology,
        mut x: CudaSlice<f32>,
        lo: usize,
        hi: usize,
        pos_d: &CudaSlice<i32>,
        pos: usize,
        cache: &mut Cache,
    ) -> Res<CudaSlice<f32>> {
        let dev = e.ctx().ordinal();
        let n_embd = self.cfg.n_embd as usize;
        let width = topology.streams * n_embd;

        // Read the live state signature BEFORE the pool takes its mutable borrow of the cache.
        let live_sig = stage_sig(self, e, cache, lo, hi);
        // Carried out of a torn-down stage by the re-capture path below, so a rebuild reuses the
        // buffers the old graphs baked instead of allocating fresh ones. A FIRST capture has
        // nothing to reuse and this stays None.
        let mut reuse: Option<StageBufs> = None;
        // CONSERVATIVE DECISION (box run 3, 2026-09-02). Two facts from that run drive this.
        // FIRST: the engine decided to re-capture on step 2 of a real artifact, where the
        // ping-pong swap is the only thing that should have moved and the signature is already
        // invariant under it — so something else moves on the real walk and this lane does not yet
        // know what. SECOND: the short `re-capture:` note printed and the long decision line
        // immediately after it did NOT, so the `CUDA_ERROR_INVALID_VALUE` comes from the TEARDOWN
        // between them (the synchronize, the `remove`, or the exec drop), not from `capture_one` —
        // no `capture-error:` line appeared either.
        //
        // So re-capture is DISABLED until a receipt says what moves and why the teardown fails.
        // An invalidated stage falls through to the byte-identical eager walk and latches, which
        // (a) removes the failing teardown from the token path entirely and (b) lets the box price
        // the FIRST-capture case, which is the actual product question. The diff is NAMED so the
        // next run says which element moved and what the two pointers were.
        let mut latch_eager = false;
        // `MEMRA_GLM5_GRAPH_RECAPTURE=1` (default OFF) chooses REBUILD over latch when the stage
        // is invalidated. See the flag's doc in lib.rs for why the default is what it is.
        let recapture_armed = crate::glm5_graph_recapture_on();
        let mut need_capture = {
            let pool = self.glm5_graph_pool(cache);
            match pool
                .stages
                .iter()
                .position(|s| s.dev == dev && s.lo == lo && s.hi == hi)
            {
                Some(i) => {
                    let st = &pool.stages[i];
                    let stale_pos = st.next_pos != pos;
                    let diff = sig_diff(&st.state_sig, &live_sig);
                    if stale_pos || diff.is_some() {
                        eprintln!(
                            "[glm5-decode-graph] {} dev={dev} stage=[{lo}, {hi}) pos={pos} \
                             expected_pos={} stale_pos={stale_pos} launched_since_sync={} \
                             stream_capture={} free={} sig_diff={}",
                            if recapture_armed {
                                "re-capture"
                            } else {
                                "eager-latch (MEMRA_GLM5_GRAPH_RECAPTURE is off; this stage runs \
                                 eager for the rest of the session and the walk stays \
                                 byte-identical)"
                            },
                            st.next_pos,
                            st.launched_since_sync,
                            capture_status(e),
                            free_mb(e),
                            diff.as_deref().unwrap_or("none"),
                        );
                        if recapture_armed {
                            // ORDER IS THE WHOLE FIX for box run 3's `CUDA_ERROR_INVALID_VALUE`.
                            // That run destroyed a stage's execs (and freed every buffer they
                            // baked) with a replay of those same execs still outstanding, which is
                            // a destroy-in-use. Drain FIRST, then drop, and only then capture.
                            true
                        } else {
                            pool.failed.push((dev, lo, hi));
                            latch_eager = true;
                            false
                        }
                    } else {
                        false
                    }
                }
                None => true,
            }
        };
        if recapture_armed && !latch_eager && need_capture {
            // A stage already in the pool has to be torn down before its key can be captured
            // again; a first capture has nothing to remove and the drain is a no-op there.
            let present = {
                let pool = self.glm5_graph_pool(cache);
                pool.stages
                    .iter()
                    .position(|s| s.dev == dev && s.lo == lo && s.hi == hi)
            };
            if let Some(i) = present {
                if let Err(err) = e.stream().synchronize() {
                    // A drain that fails leaves the outstanding replay unaccounted for, so the
                    // one thing not to do is destroy the exec anyway. Latch and stay eager.
                    note(&format!(
                        "eager from here on stage=[{lo}, {hi}) dev={dev}: the pre-teardown drain \
                         failed ({err}), so the stale execs are left alone"
                    ));
                    self.glm5_graph_pool(cache).failed.push((dev, lo, hi));
                    return self
                        .hyper_range_decode_eager(e, topology, x, lo, hi, pos_d, pos, cache);
                }
                let stale = self.glm5_graph_pool(cache).stages.remove(i);
                crate::GLM5_DECODE_GRAPH_RECAPTURES.fetch_add(1, Ordering::Relaxed);
                // BUFFER REUSE IS THE SAFETY ARGUMENT, so it has to actually happen (revuto
                // MEDIUM on #116: `reuse` used to be hard-bound `None`, so the rebuild allocated
                // fresh buffers while this comment and the FLAGS row both claimed it reused
                // them). Moving the buffers OUT does not free them and does not change their
                // addresses; it only takes them out of the value whose drop destroys the execs.
                // The execs therefore die while every address they baked is still live, which is
                // the ordering `StageGraphs`' field order documents, and the rebuild allocates
                // and frees nothing of its own against the stream-ordered mempool the graphs'
                // alloc nodes draw from.
                let StageGraphs {
                    runs,
                    x_io,
                    x_out,
                    ws,
                    f16,
                    ..
                } = stale;
                reuse = Some(StageBufs {
                    x_io,
                    x_out,
                    ws,
                    f16,
                });
                drop(runs);
                need_capture = true;
            }
        }
        if latch_eager {
            // The stale stage is deliberately LEFT IN THE POOL: destroying its execs here is the
            // very call sequence box run 3 died in, and `glm5_decode_graph_ready` will never
            // consult it again now that the key is latched. It is released with the cache.
            return self.hyper_range_decode_eager(e, topology, x, lo, hi, pos_d, pos, cache);
        }
        if need_capture {
            // memra#131 ROOT CAUSE FIX: warm every run this capture will record. Lazily built
            // per-weight caches (the MEMRA_GLM5_W8 q8_0 mirrors, built on first decode use) were
            // being built INSIDE the capture: kernels recorded, never executed, entries marked
            // built, and every later reader, the door's own eager walk included, read
            // uninitialised bytes (cells 8/9: finite input, finite state, all-NaN KDA mixer at
            // the first layer of the second captured run). One eager pass of each captured
            // range on a COPY of the input, with the recurrent state snapshotted and restored
            // around it, builds those caches with executed kernels before the capture opens.
            // Cost: one eager stage walk per capture per session.
            if let Err(err) = self
                .glm5_warm_runs_before_capture(e, topology, &x, lo, hi, width, pos_d, pos, cache)
            {
                note(&format!(
                    "eager from here on stage=[{lo}, {hi}) dev={dev}: the pre-capture warm walk failed ({err})"
                ));
                self.glm5_graph_pool(cache).failed.push((dev, lo, hi));
                return self.hyper_range_decode_eager(e, topology, x, lo, hi, pos_d, pos, cache);
            }
        }
        if need_capture
            && let Err(err) =
                self.glm5_capture_stage(e, topology, lo, hi, cache, dev, width, pos, reuse)
        {
            // A CAPTURE FAILURE IS NEVER THE REQUEST'S PROBLEM. This route has a byte-identical
            // eager twin, so a failed capture degrades to it instead of failing the token. The
            // stage is latched off for this session so the next token does not retry (and
            // re-thrash the stream) forever, and the reason is printed once.
            note(&format!(
                "eager from here on stage=[{lo}, {hi}) dev={dev}: capture failed ({err})"
            ));
            // The failed capture may have left transients queued; drain before eager work reuses
            // the stream, and record the refusal so `glm5_decode_graph_ready` yields immediately.
            let _ = e.stream().synchronize();
            self.glm5_graph_pool(cache).failed.push((dev, lo, hi));
            return self.hyper_range_decode_eager(e, topology, x, lo, hi, pos_d, pos, cache);
        }

        let runs: Vec<(usize, usize)> = kda_runs(self, lo, hi);
        let mut cursor = lo;
        for (a, b) in runs {
            if a > cursor {
                x = self.hyper_range_decode_eager(e, topology, x, cursor, a, pos_d, pos, cache)?;
                trace_seg(e, dev, lo, hi, cursor, a, "graph-gap", pos, &x);
            }
            // memra#131 SELF-CHECK, once per run per session, BEFORE the door may claim health.
            let unchecked = self
                .glm5_graph_pool(cache)
                .stages
                .iter()
                .find(|s| s.dev == dev && s.lo == lo && s.hi == hi)
                .and_then(|st| st.runs.iter().find(|r| r.lo == a && r.hi == b))
                .is_some_and(|r| r.checked < selfcheck_n());
            if unchecked {
                let (x_out, ok) = self.glm5_selfcheck_run(
                    e, topology, dev, lo, hi, a, b, x, width, pos_d, pos, cache,
                )?;
                x = x_out;
                if !ok {
                    // The run's state was restored and the eager walk re-run for real; the
                    // stage is latched eager for the session. Finish this token eager from
                    // here, byte-identically, and never replay this stage again.
                    x = self.hyper_range_decode_eager(e, topology, x, b, hi, pos_d, pos, cache)?;
                    return Ok(x);
                }
            } else {
                x = self.glm5_replay_run(e, dev, lo, hi, a, b, x, width, cache)?;
            }
            trace_seg(e, dev, lo, hi, a, b, "graph-run", pos, &x);
            cursor = b;
        }
        if cursor < hi {
            x = self.hyper_range_decode_eager(e, topology, x, cursor, hi, pos_d, pos, cache)?;
            trace_seg(e, dev, lo, hi, cursor, hi, "graph-gap", pos, &x);
        }
        {
            let pool = self.glm5_graph_pool(cache);
            if let Some(st) = pool
                .stages
                .iter_mut()
                .find(|s| s.dev == dev && s.lo == lo && s.hi == hi)
            {
                st.next_pos = pos + 1;
            }
        }
        Ok(x)
    }

    fn glm5_graph_pool<'a>(&self, cache: &'a mut Cache) -> &'a mut Glm5DecodeGraphs {
        if cache
            .glm5_decode_graph
            .as_ref()
            .and_then(|b| b.downcast_ref::<Glm5DecodeGraphs>())
            .is_none()
        {
            cache.glm5_decode_graph = Some(Box::new(Glm5DecodeGraphs::default()));
        }
        cache
            .glm5_decode_graph
            .as_mut()
            .and_then(|b| b.downcast_mut::<Glm5DecodeGraphs>())
            .expect("just installed")
    }

    /// Capture every KDA run of `[lo, hi)` in both ping-pong phases.
    ///
    /// Capture RECORDS, it does not execute, so the two passes cost no device state — but the
    /// HOST-side swap inside `kda_cached` runs on both passes, which is exactly how phase 1 gets
    /// the opposite pointer assignment. Two passes leave the host fields back where they started,
    /// so the pool starts at phase 0.
    #[allow(clippy::too_many_arguments)] // allow: the capture's inputs are the stage's identity plus the reusable buffer set
    fn glm5_capture_stage(
        &self,
        e: &Engine,
        topology: &crate::hyper::HyperTopology,
        lo: usize,
        hi: usize,
        cache: &mut Cache,
        dev: usize,
        width: usize,
        pos: usize,
        reuse: Option<StageBufs>,
    ) -> Res<()> {
        let n_embd = self.cfg.n_embd as usize;
        let runs = kda_runs(self, lo, hi);
        let recapture = reuse.is_some();
        // BUFFER REUSE ON RE-CAPTURE, and this is a correctness argument, not a saving. On a
        // rebuild the ONLY thing that has to change is the graphs; `x_io`, the private hc
        // workspace and the private f16 scratch hold no state across tokens (every one is fully
        // overwritten before any read). Freeing them and allocating new ones churns the same
        // stream-ordered mempool the captured graphs' own alloc nodes draw from, immediately
        // before instantiating new graphs against it — which is a plausible source of
        // `CUDA_ERROR_INVALID_VALUE` from `cuGraphInstantiate` and buys nothing. Reused buffers
        // also mean the rebuilt graphs bake the SAME addresses the old ones did.
        let alloc_ctx = CapCtx {
            dev,
            lo,
            hi,
            run: 0,
            runs: runs.len(),
            a: lo,
            b: hi,
            phase: 0,
            recapture,
        };
        let (x_io, x_out, ws, f16) = match reuse {
            Some(b) => (b.x_io, b.x_out, b.ws, b.f16),
            None => (
                step(e, &alloc_ctx, "alloc(x_io)", e.zeros(width))?,
                step(e, &alloc_ctx, "alloc(x_out)", e.zeros(width))?,
                step(
                    e,
                    &alloc_ctx,
                    "alloc(HyperDecodeWs)",
                    crate::hyper::HyperDecodeWs::new(e, topology, n_embd),
                )?,
                None,
            ),
        };
        // Pre-sized GENEROUSLY on purpose: this is the m=1 activation staging buffer, and a
        // lazy GROW inside the capture region would be a mem node whose address the next launch
        // recycles. 4 MiB covers any single decode row this trunk can present (up to 2M f16
        // elements) with room over the widest KDA/MoE feature width, and the `ws` half is a fixed
        // 64 MiB by construction (`F16_WS_BYTES`). A few MiB of headroom is the right trade
        // against a capture-time grow. Reused verbatim on a re-capture (see above).
        let f16 = match f16 {
            Some(f) => f,
            None => step(
                e,
                &alloc_ctx,
                "alloc(F16Scratch)",
                crate::f16_ffi::F16Scratch::with_capacity(e, (4 << 20).max(n_embd * 16)),
            )?,
        };

        let mut stage = StageGraphs {
            dev,
            lo,
            hi,
            runs: Vec::with_capacity(runs.len()),
            x_io,
            x_out,
            ws,
            f16: None,
            next_pos: pos,
            state_sig: runs
                .iter()
                .flat_map(|(a, b)| *a..*b)
                .filter_map(|il| recur_sig(e, cache, il))
                .collect(),
            launched_since_sync: false,
        };
        // Pre-arm the gate ledger's per-layer slots BEFORE the capture opens: an allocation
        // inside a capture region becomes a graph mem node whose address the next launch
        // recycles, and the ledger would then read freed memory.
        if crate::glm5_sel_ledger::armed()
            && let Some(moe) = self.cfg.moe.as_ref()
        {
            for (a, b) in &runs {
                for il in *a..*b {
                    crate::glm5_sel_ledger::prearm(e, il as u16, moe.expert_used_count as usize)?;
                }
            }
        }
        let prev_f16 = e.f16_scratch_swap(Some(f16));
        let captured = (|| -> Res<()> {
            // The captured bodies take verify-workspace transients; while this flag is set the
            // pool RETAINS instead of recycling, so nothing the graph baked is ever re-issued.
            crate::GLM5_GRAPH_CAPTURE_OPEN.store(true, Ordering::Relaxed);
            let mut ctx = CapCtx {
                dev,
                lo,
                hi,
                run: 0,
                runs: runs.len(),
                a: 0,
                b: 0,
                phase: 0,
                recapture,
            };
            let pos_d = step(
                e,
                &ctx,
                "htod_i32(pos_d)",
                e.htod_i32(&[pos as i32]).map(Some),
            )?
            .expect("htod returned a buffer");
            for (ri, (a, b)) in runs.iter().enumerate() {
                let (a, b) = (*a, *b);
                ctx.run = ri;
                ctx.a = a;
                ctx.b = b;
                let mut phase_graphs: Vec<CudaGraph> = Vec::with_capacity(2);
                for phase in 0..2 {
                    ctx.phase = phase;
                    let x_cell = std::cell::RefCell::new(&mut stage.x_io);
                    let out_cell = std::cell::RefCell::new(&mut stage.x_out);
                    let ws_cell = std::cell::RefCell::new(&mut stage.ws);
                    let cache_cell = std::cell::RefCell::new(&mut *cache);
                    let g = capture_one(e, &ctx, |e| {
                        self.hyper_range_decode_ws_body(
                            e,
                            topology,
                            &mut x_cell.borrow_mut(),
                            a,
                            b,
                            &pos_d,
                            pos,
                            &mut cache_cell.borrow_mut(),
                            &mut ws_cell.borrow_mut(),
                        )?;
                        // THE REPLAY CONTRACT, recorded rather than argued. The walk ping-pongs
                        // the stream state between `x_io` and `ws.xb`, so which physical buffer
                        // holds the answer at the end is a parity property of the layer count and
                        // the site count — and box run 4 produced zero logits at every step, which
                        // is what "the eager remainder read the buffer the graph did not write"
                        // looks like. One memcpy node into a THIRD buffer removes the question:
                        // `x_out` holds this run's output on every replay, whatever the parity.
                        let live = x_cell.borrow();
                        e.copy_into(&mut out_cell.borrow_mut(), 0, &live, width)
                    })?;
                    phase_graphs.push(g);
                }
                let mut it = phase_graphs.into_iter();
                let g0 = it.next().expect("phase 0 captured");
                let g1 = it.next().expect("phase 1 captured");
                // Everything the two captured bodies took from the verify workspace, held for the
                // life of the graphs that baked it (see `Engine::glm5_graph_keep`).
                let keeper = std::mem::take(&mut *e.glm5_graph_keep().lock().unwrap());
                stage.runs.push(RunGraph {
                    lo: a,
                    hi: b,
                    graphs: [g0, g1],
                    // Capture's two passes leave the host ping-pong fields exactly where they
                    // started, so every run's first replay is phase 0.
                    phase: 0,
                    checked: 0,
                    _keeper: keeper,
                });
                crate::GLM5_DECODE_GRAPH_LAYERS.fetch_add((b - a) as u64, Ordering::Relaxed);
            }
            Ok(())
        })();
        crate::GLM5_GRAPH_CAPTURE_OPEN.store(false, Ordering::Relaxed);
        stage.f16 = e.f16_scratch_swap(prev_f16);
        captured?;
        crate::GLM5_DECODE_GRAPH_CAPTURES.fetch_add(1, Ordering::Relaxed);
        // Not `note`: a rebuild must print EVERY time, or a box log cannot tell one capture from
        // six. Only the first-build line is deduplicated.
        let line = format!(
            "engaged dev={dev} stage=[{lo}, {hi}) runs={} captured_layers={} recapture={recapture} \
             free={} (2 ping-pong phases each; MLA/DSA layers stay eager)",
            stage.runs.len(),
            stage.runs.iter().map(|r| r.hi - r.lo).sum::<usize>(),
            free_mb(e),
        );
        if recapture {
            eprintln!("[glm5-decode-graph] {line}");
        } else {
            note(&line);
        }
        self.glm5_graph_pool(cache).stages.push(stage);
        Ok(())
    }

    /// memra#131: the first replay of run `[a, b)` checks ITSELF against the eager walk.
    ///
    /// Captured runs hold only KDA layers by construction, so their complete mutable state is
    /// the recurrent trio (`conv_state`, `ssm_state`, `ssm_state_alt`) per layer. Snapshot
    /// those; run the eager walk on a copy of `x` as the reference; put the state back (bytes
    /// AND the host ping-pong role, which `kda_cached` swaps); run the replay for real; compare
    /// the two outputs bitwise.
    ///
    /// memra#131: run every capturable KDA range of the stage once, eagerly, on a copy of the
    /// input, with each range's recurrent state (and its ssm role) snapshotted and restored, so
    /// that every lazily built per-weight cache the walk touches exists BEFORE the capture opens.
    #[allow(clippy::too_many_arguments)]
    fn glm5_warm_runs_before_capture(
        &self,
        e: &Engine,
        topology: &crate::hyper::HyperTopology,
        x: &CudaSlice<f32>,
        lo: usize,
        hi: usize,
        width: usize,
        pos_d: &CudaSlice<i32>,
        pos: usize,
        cache: &mut Cache,
    ) -> Res<()> {
        use cudarc::driver::DevicePtr;
        let runs = kda_runs(self, lo, hi);
        for (a, b) in runs {
            struct Snap {
                il: usize,
                conv: CudaSlice<f32>,
                ssm: CudaSlice<f32>,
                alt: CudaSlice<f32>,
                ssm_ptr: u64,
            }
            let mut snaps: Vec<Snap> = Vec::with_capacity(b - a);
            for il in a..b {
                let rl = cache.recur[il]
                    .as_ref()
                    .ok_or_else(|| format!("warm: layer {il} carries no recurrent state"))?;
                let ssm_ptr = {
                    let st = e.stream();
                    let (p, _g) = rl.ssm_state.device_ptr(&st);
                    p
                };
                let mut conv = e.uninit(rl.conv_state.len())?;
                let mut ssm = e.uninit(rl.ssm_state.len())?;
                let mut alt = e.uninit(rl.ssm_state_alt.len())?;
                e.copy_into(&mut conv, 0, &rl.conv_state, rl.conv_state.len())?;
                e.copy_into(&mut ssm, 0, &rl.ssm_state, rl.ssm_state.len())?;
                e.copy_into(&mut alt, 0, &rl.ssm_state_alt, rl.ssm_state_alt.len())?;
                snaps.push(Snap {
                    il,
                    conv,
                    ssm,
                    alt,
                    ssm_ptr,
                });
            }
            let mut xc = e.uninit(width)?;
            e.copy_into(&mut xc, 0, x, width)?;
            let _ = self.hyper_range_decode_eager(e, topology, xc, a, b, pos_d, pos, cache)?;
            for sn in &snaps {
                let rl = cache.recur[sn.il].as_mut().expect("snapshotted above");
                let p = {
                    let st = e.stream();
                    let (p, _g) = rl.ssm_state.device_ptr(&st);
                    p
                };
                if p != sn.ssm_ptr {
                    std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
                }
                e.copy_into(&mut rl.conv_state, 0, &sn.conv, sn.conv.len())?;
                e.copy_into(&mut rl.ssm_state, 0, &sn.ssm, sn.ssm.len())?;
                e.copy_into(&mut rl.ssm_state_alt, 0, &sn.alt, sn.alt.len())?;
            }
            note(&format!(
                "warmed run [{a}, {b}) before capture (one eager pass on a copy of the input, \
                 state restored): lazily built per-weight caches now exist"
            ));
        }
        Ok(())
    }

    /// `(x, true)`: the replay matched, `x` is its output, the run is marked checked — one
    /// eager run per run per session. `(x, false)`: it did not match; the state is restored a
    /// second time, the eager walk is run for real from a kept copy of the input so the token
    /// is correct, the stage is latched eager (`pool.failed`), and a loud line names the run,
    /// the first differing element and both values — a verdict derived from the OUTPUT rather
    /// than from the door's opinion of itself.
    #[allow(clippy::too_many_arguments)]
    fn glm5_selfcheck_run(
        &self,
        e: &Engine,
        topology: &crate::hyper::HyperTopology,
        dev: usize,
        lo: usize,
        hi: usize,
        a: usize,
        b: usize,
        x: CudaSlice<f32>,
        width: usize,
        pos_d: &CudaSlice<i32>,
        pos: usize,
        cache: &mut Cache,
    ) -> Res<(CudaSlice<f32>, bool)> {
        struct Snap {
            il: usize,
            conv: CudaSlice<f32>,
            ssm: CudaSlice<f32>,
            alt: CudaSlice<f32>,
            ssm_ptr: u64,
        }
        fn ptr_of(e: &Engine, s: &CudaSlice<f32>) -> u64 {
            use cudarc::driver::DevicePtr;
            let st = e.stream();
            let (p, _g) = s.device_ptr(&st);
            p
        }
        let mut snaps: Vec<Snap> = Vec::with_capacity(b - a);
        for il in a..b {
            let rl = cache.recur[il]
                .as_ref()
                .ok_or_else(|| format!("self-check: layer {il} carries no recurrent state"))?;
            let ssm_ptr = ptr_of(e, &rl.ssm_state);
            let mut conv = e.uninit(rl.conv_state.len())?;
            let mut ssm = e.uninit(rl.ssm_state.len())?;
            let mut alt = e.uninit(rl.ssm_state_alt.len())?;
            e.copy_into(&mut conv, 0, &rl.conv_state, rl.conv_state.len())?;
            e.copy_into(&mut ssm, 0, &rl.ssm_state, rl.ssm_state.len())?;
            e.copy_into(&mut alt, 0, &rl.ssm_state_alt, rl.ssm_state_alt.len())?;
            snaps.push(Snap {
                il,
                conv,
                ssm,
                alt,
                ssm_ptr,
            });
        }
        fn restore(e: &Engine, cache: &mut Cache, snaps: &[Snap]) -> Res<()> {
            for sn in snaps {
                let rl = cache.recur[sn.il].as_mut().expect("snapshotted above");
                if ptr_of(e, &rl.ssm_state) != sn.ssm_ptr {
                    // the eager walk's host ping-pong moved the role; move it back first
                    std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
                }
                e.copy_into(&mut rl.conv_state, 0, &sn.conv, sn.conv.len())?;
                e.copy_into(&mut rl.ssm_state, 0, &sn.ssm, sn.ssm.len())?;
                e.copy_into(&mut rl.ssm_state_alt, 0, &sn.alt, sn.alt.len())?;
            }
            Ok(())
        }
        // memra#131 cell 5: BEFORE anything runs, count non-finite elements in the recurrent
        // state this run will read and in the stage's pool buffers (the captured bodies' private
        // workspace and the x_io/x_out cells). Cell 4 showed a FINITE input whose eager reference
        // came back all-NaN, so the poison sits in one of these two places; this names which.
        let nonfinite = |v: &[f32]| v.iter().filter(|x| !x.is_finite()).count();
        let mut nf_state = 0usize;
        let mut nf_state_first = String::new();
        for sn in &snaps {
            for (name, buf) in [
                ("conv_state", &sn.conv),
                ("ssm_state", &sn.ssm),
                ("ssm_state_alt", &sn.alt),
            ] {
                let n = nonfinite(&e.dtoh(buf)?);
                if n > 0 && nf_state_first.is_empty() {
                    nf_state_first = format!("layer {} {name} {n}/{}", sn.il, buf.len());
                }
                nf_state += n;
            }
        }
        let (nf_pool, nf_pool_first) = {
            let pool = self.glm5_graph_pool(cache);
            let mut total = 0usize;
            let mut first = String::new();
            if let Some(st) = pool
                .stages
                .iter()
                .find(|s| s.dev == dev && s.lo == lo && s.hi == hi)
            {
                let bufs: [(&str, &CudaSlice<f32>); 10] = [
                    ("x_io", &st.x_io),
                    ("x_out", &st.x_out),
                    ("ws.h", &st.ws.h),
                    ("ws.y", &st.ws.y),
                    ("ws.z", &st.ws.z),
                    ("ws.xb", &st.ws.xb),
                    ("ws.mixes", &st.ws.mixes),
                    ("ws.pre", &st.ws.pre),
                    ("ws.post", &st.ws.post),
                    ("ws.comb", &st.ws.comb),
                ];
                for (name, buf) in bufs {
                    let n = nonfinite(&e.dtoh(buf)?);
                    if n > 0 && first.is_empty() {
                        first = format!("{name} {n}/{}", buf.len());
                    }
                    total += n;
                }
            }
            (total, first)
        };
        // Keep the input: the replay consumes `x`, and a mismatch needs it back.
        let mut x_keep = e.uninit(width)?;
        e.copy_into(&mut x_keep, 0, &x, width)?;
        let mut x_copy = e.uninit(width)?;
        e.copy_into(&mut x_copy, 0, &x, width)?;
        // Reference: eager on the copy. Keep the STATE it wrote too (host copies, by role: the
        // eager walk swaps the ssm roles once per layer, so `ssm` here is whatever buffer holds
        // the `ssm_state` role after the step), then put the pre-step state back.
        let x_ref = self.hyper_range_decode_eager(e, topology, x_copy, a, b, pos_d, pos, cache)?;
        let h_ref = e.dtoh(&x_ref)?;
        struct Post {
            il: usize,
            conv: Vec<f32>,
            ssm: Vec<f32>,
            alt: Vec<f32>,
            ssm_ptr: u64,
        }
        let mut post: Vec<Post> = Vec::with_capacity(b - a);
        for il in a..b {
            let rl = cache.recur[il].as_ref().expect("snapshotted above");
            post.push(Post {
                il,
                conv: e.dtoh(&rl.conv_state)?,
                ssm: e.dtoh(&rl.ssm_state)?,
                alt: e.dtoh(&rl.ssm_state_alt)?,
                ssm_ptr: ptr_of(e, &rl.ssm_state),
            });
        }
        restore(e, cache, &snaps)?;
        // memra#131 cell 6: the reference above took the persistent-workspace walk. Count the
        // non-finite elements in the ENGINE's pooled workspace (the buffers that walk reads and
        // writes; the stage pool checked earlier is the capture's private copy), then run a
        // SECOND reference through the allocating (workspace-free) walk on another copy of the
        // input and the same restored state, and compare the two bitwise. If the plain walk is
        // finite while the workspace walk is NaN, the poison is the workspace walk's, not the
        // layers'.
        let (nf_ews, nf_ews_first) = match e.hyper_ws_take() {
            Some(ws) => {
                let mut total = 0usize;
                let mut first = String::new();
                for (name, buf) in [
                    ("ws.h", &ws.h),
                    ("ws.y", &ws.y),
                    ("ws.z", &ws.z),
                    ("ws.xb", &ws.xb),
                    ("ws.mixes", &ws.mixes),
                    ("ws.pre", &ws.pre),
                    ("ws.post", &ws.post),
                    ("ws.comb", &ws.comb),
                ] {
                    let n = nonfinite(&e.dtoh(buf)?);
                    if n > 0 && first.is_empty() {
                        first = format!("{name} {n}/{}", buf.len());
                    }
                    total += n;
                }
                e.hyper_ws_put(ws);
                (total, first)
            }
            None => (0, "no pooled ws".to_string()),
        };
        let mut x_copy2 = e.uninit(width)?;
        e.copy_into(&mut x_copy2, 0, &x_keep, width)?;
        crate::hybrid_forward::HC_WS_FORCE_PLAIN.store(true, std::sync::atomic::Ordering::Relaxed);
        let plain = self.hyper_range_decode_eager(e, topology, x_copy2, a, b, pos_d, pos, cache);
        crate::hybrid_forward::HC_WS_FORCE_PLAIN.store(false, std::sync::atomic::Ordering::Relaxed);
        let x_plain = plain?;
        let h_plain = e.dtoh(&x_plain)?;
        restore(e, cache, &snaps)?;
        let nf_plain = nonfinite(&h_plain);
        let plain_vs_ws = match h_ref
            .iter()
            .zip(h_plain.iter())
            .position(|(r, p)| r.to_bits() != p.to_bits())
        {
            None => "bit-identical to the workspace eager".to_string(),
            Some(i) => format!(
                "differs from the workspace eager at element {i} (ws={:e} plain={:e})",
                h_ref[i], h_plain[i]
            ),
        };
        // Which verified replay this is for the run, and which ping-pong phase it will use.
        let (k, phase) = {
            let pool = self.glm5_graph_pool(cache);
            pool.stages
                .iter()
                .find(|s| s.dev == dev && s.lo == lo && s.hi == hi)
                .and_then(|st| st.runs.iter().find(|r| r.lo == a && r.hi == b))
                .map(|r| (r.checked + 1, r.phase))
                .unwrap_or((1, 0))
        };
        // The real step.
        let x_rep = self.glm5_replay_run(e, dev, lo, hi, a, b, x, width, cache)?;
        let h_rep = e.dtoh(&x_rep)?;
        // A NaN replay compared to a NaN eager walk is bit-identical, so an output-and-state
        // compare on poisoned data is a vacuous PASS. Box trace 2026-09-03 (memra#131): every
        // segment of dev 1's stage carried nz=16384/16384 absmax=0, the all-NaN signature, and
        // every run there "passed". Non-finite data is a failure of this stage's INPUT or of
        // the replay, and the line says which, with counts, so the poison can be walked back.
        let nonfinite = |v: &[f32]| v.iter().filter(|x| !x.is_finite()).count();
        let (nf_in, nf_ref, nf_rep) = {
            let h_in = e.dtoh(&x_keep)?;
            (nonfinite(&h_in), nonfinite(&h_ref), nonfinite(&h_rep))
        };
        // memra#131 cell 8: when the reference is poisoned from a finite input, walk the run ONE
        // LAYER AT A TIME on the restored state and name the first layer whose output is
        // non-finite (and whether the layer before it was finite), so the defect is placed at a
        // layer instead of a run.
        let layer_bisect = if nf_in == 0 && nf_ref > 0 {
            let mut report = String::new();
            let mut xl = e.uninit(width)?;
            e.copy_into(&mut xl, 0, &x_keep, width)?;
            let mut first_bad: Option<(usize, CudaSlice<f32>)> = None;
            for il in a..b {
                let mut xin = e.uninit(width)?;
                e.copy_into(&mut xin, 0, &xl, width)?;
                let out =
                    self.hyper_range_decode_eager(e, topology, xl, il, il + 1, pos_d, pos, cache)?;
                let n = nonfinite(&e.dtoh(&out)?);
                report.push_str(&format!(" L{il}:{n}"));
                if n > 0 {
                    first_bad = Some((il, xin));
                    break;
                }
                xl = out;
            }
            restore(e, cache, &snaps)?;
            // memra#131 cell 9: inside the first poisoned layer, the sub-steps of the eager body
            // on the restored state, each counted: hc pre (attn), norm, KDA mixer, hc post, hc pre
            // (mlp), norm, MoE FFN, hc post. The first non-finite sub-step is the site.
            let sub = match first_bad {
                Some((il, xin)) => {
                    let n_embd = self.cfg.n_embd as usize;
                    let eps = self.cfg.rms_eps;
                    let layer = &self.layers[il];
                    let mut steps = String::new();
                    if let Some(hyper) = layer.hyper.as_ref() {
                        let (y, mix) =
                            crate::hyper::pre(e, topology, &hyper.attn, &xin, 1, n_embd)?;
                        steps.push_str(&format!(" pre_attn.y:{}", nonfinite(&e.dtoh(&y)?)));
                        let mut h = e.uninit(n_embd)?;
                        e.rms_norm(&y, layer.attn_norm.float_data(), &mut h, n_embd, 1, eps)?;
                        steps.push_str(&format!(" attn_norm.h:{}", nonfinite(&e.dtoh(&h)?)));
                        let mixed = match &layer.mixer {
                            crate::hybrid::Mixer::Kda(la) if la.tp.is_none() => {
                                Some(crate::kda::kda_decode_cached(e, la, &h, eps, cache, il)?)
                            }
                            _ => None,
                        };
                        if let Some(mixed) = mixed.as_ref() {
                            steps.push_str(&format!(" kda_mixer:{}", nonfinite(&e.dtoh(mixed)?)));
                            let x1 = crate::hyper::post(e, topology, mixed, &xin, &mix, 1, n_embd)?;
                            steps.push_str(&format!(" post_attn.x:{}", nonfinite(&e.dtoh(&x1)?)));
                            let (y2, mix2) =
                                crate::hyper::pre(e, topology, &hyper.mlp, &x1, 1, n_embd)?;
                            steps.push_str(&format!(" pre_mlp.y:{}", nonfinite(&e.dtoh(&y2)?)));
                            let mut z = e.uninit(n_embd)?;
                            e.rms_norm(
                                &y2,
                                layer.post_attn_norm.float_data(),
                                &mut z,
                                n_embd,
                                1,
                                eps,
                            )?;
                            steps.push_str(&format!(" mlp_norm.z:{}", nonfinite(&e.dtoh(&z)?)));
                            let ffn_out =
                                self.hyper_ffn_branch(e, layer, &z, 1, il, false, None)?;
                            steps.push_str(&format!(" ffn_out:{}", nonfinite(&e.dtoh(&ffn_out)?)));
                            let x2 =
                                crate::hyper::post(e, topology, &ffn_out, &x1, &mix2, 1, n_embd)?;
                            steps.push_str(&format!(" post_mlp.x:{}", nonfinite(&e.dtoh(&x2)?)));
                        } else {
                            steps.push_str(" (mixer is not a plain KDA layer; sub-steps skipped)");
                        }
                    }
                    restore(e, cache, &snaps)?;
                    format!("; inside L{il} sub-steps non-finite:{steps}")
                }
                None => String::new(),
            };
            format!("; per-layer non-finite after each layer:{report}{sub}")
        } else {
            String::new()
        };
        if nf_in + nf_ref + nf_rep > 0 {
            eprintln!(
                "[glm5-decode-graph] SELF-CHECK FAILED dev={dev} stage=[{lo}, {hi}) \
                 run=[{a}, {b}) pos={pos} replay={k} phase={phase}: NON-FINITE data: input \
                 {nf_in}/{width}, eager output {nf_ref}/{width}, replay output {nf_rep}/{width}; \
                 BEFORE the walk: recurrent state non-finite {nf_state} (first: {}), stage pool \
                 non-finite {nf_pool} (first: {}), engine ws non-finite {nf_ews} (first: {}); PLAIN \
                 eager twin: non-finite {nf_plain}/{width}, {plain_vs_ws}{layer_bisect} \
                 (a NaN-identical compare proves nothing); the stage is latched EAGER for this \
                 session and this token is recomputed eager. The door did NOT engage.{}",
                if nf_state_first.is_empty() {
                    "none"
                } else {
                    &nf_state_first
                },
                if nf_pool_first.is_empty() {
                    "none"
                } else {
                    &nf_pool_first
                },
                if nf_ews_first.is_empty() {
                    "none"
                } else {
                    &nf_ews_first
                },
                if nf_in > 0 {
                    " The INPUT was already poisoned: the defect is upstream of this stage."
                } else if nf_state > 0 {
                    " The recurrent STATE was already poisoned before this run's walk."
                } else if nf_pool > 0 {
                    " The stage POOL (workspace / x cells) was already poisoned before this run's walk."
                } else {
                    " Input, state and pool were finite: the eager walk itself produced the poison."
                }
            );
            restore(e, cache, &snaps)?;
            let x_real =
                self.hyper_range_decode_eager(e, topology, x_keep, a, b, pos_d, pos, cache)?;
            self.glm5_graph_pool(cache).failed.push((dev, lo, hi));
            return Ok((x_real, false));
        }
        let first_diff = h_ref
            .iter()
            .zip(h_rep.iter())
            .position(|(r, p)| r.to_bits() != p.to_bits())
            .map(|i| {
                format!(
                    "OUTPUT element {i}/{width} (eager={:e} replay={:e})",
                    h_ref[i], h_rep[i]
                )
            });
        // Output identical: now the state the replay left behind, buffer by buffer, against
        // the eager walk's. A replay that answers right and writes the wrong state passes an
        // output-only check and poisons the NEXT token, which is what an output-only check
        // cannot see.
        let state_diff = if first_diff.is_some() {
            None
        } else {
            let mut found = None;
            for p in &post {
                let rl = cache.recur[p.il].as_ref().expect("snapshotted above");
                if ptr_of(e, &rl.ssm_state) != p.ssm_ptr {
                    found = Some(format!(
                        "layer {} ssm ROLE: the replay left `ssm_state` at a different parity \
                         than the eager walk",
                        p.il
                    ));
                    break;
                }
                let bufs: [(&str, Vec<f32>, &Vec<f32>); 3] = [
                    ("conv_state", e.dtoh(&rl.conv_state)?, &p.conv),
                    ("ssm_state", e.dtoh(&rl.ssm_state)?, &p.ssm),
                    ("ssm_state_alt", e.dtoh(&rl.ssm_state_alt)?, &p.alt),
                ];
                for (name, got, want) in &bufs {
                    if let Some(i) = got
                        .iter()
                        .zip(want.iter())
                        .position(|(g, w)| g.to_bits() != w.to_bits())
                    {
                        found = Some(format!(
                            "layer {} {name} element {i}/{} (eager={:e} replay={:e})",
                            p.il,
                            want.len(),
                            want[i],
                            got[i]
                        ));
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            found
        };
        match first_diff.or(state_diff) {
            None => {
                let n = selfcheck_n();
                let pool = self.glm5_graph_pool(cache);
                if let Some(st) = pool
                    .stages
                    .iter_mut()
                    .find(|s| s.dev == dev && s.lo == lo && s.hi == hi)
                    && let Some(r) = st.runs.iter_mut().find(|r| r.lo == a && r.hi == b)
                {
                    r.checked += 1;
                }
                note(&format!(
                    "self-check PASS dev={dev} stage=[{lo}, {hi}) run=[{a}, {b}) pos={pos} \
                     replay={k}/{n} phase={phase}: output ({width} elements) and the written \
                     recurrent state of {} layers are bit-identical to the eager walk on the \
                     same state{}",
                    b - a,
                    if k >= n {
                        "; this run is trusted for the session"
                    } else {
                        ""
                    }
                ));
                Ok((x_rep, true))
            }
            Some(what) => {
                eprintln!(
                    "[glm5-decode-graph] SELF-CHECK FAILED dev={dev} stage=[{lo}, {hi}) \
                     run=[{a}, {b}) pos={pos} replay={k} phase={phase}: replay differs from the \
                     eager walk at {what}; the stage is latched EAGER for this session and this \
                     token is recomputed eager. The door did NOT engage."
                );
                restore(e, cache, &snaps)?;
                let x_real =
                    self.hyper_range_decode_eager(e, topology, x_keep, a, b, pos_d, pos, cache)?;
                self.glm5_graph_pool(cache).failed.push((dev, lo, hi));
                Ok((x_real, false))
            }
        }
    }

    /// Replay one captured run: stream state in, launch, stream state out, mirror the host-side
    /// ping-pong the eager step would have done, advance the phase.
    #[allow(clippy::too_many_arguments)]
    fn glm5_replay_run(
        &self,
        e: &Engine,
        dev: usize,
        lo: usize,
        hi: usize,
        a: usize,
        b: usize,
        x: CudaSlice<f32>,
        width: usize,
        cache: &mut Cache,
    ) -> Res<CudaSlice<f32>> {
        let phase;
        {
            let pool = self.glm5_graph_pool(cache);
            let st = pool
                .stages
                .iter_mut()
                .find(|s| s.dev == dev && s.lo == lo && s.hi == hi)
                .ok_or("glm5 decode graph: stage pool vanished between capture and replay")?;
            let ri = st
                .runs
                .iter()
                .position(|r| r.lo == a && r.hi == b)
                .ok_or_else(|| format!("glm5 decode graph: no captured run [{a}, {b})"))?;
            phase = st.runs[ri].phase;
            let ctx = CapCtx {
                dev,
                lo,
                hi,
                run: ri,
                runs: st.runs.len(),
                a,
                b,
                phase,
                recapture: false,
            };
            step(
                e,
                &ctx,
                "memcpy_dtod(x -> x_io)",
                e.copy_into(&mut st.x_io, 0, &x, width),
            )?;
            let prev = e.f16_scratch_swap(st.f16.take());
            let launched = st.runs[ri].graphs[phase].launch();
            st.f16 = e.f16_scratch_swap(prev);
            // Set BEFORE propagating: a launch that errored may still have been submitted, and
            // the re-capture path has to assume the exec is live when it decides to destroy it.
            st.launched_since_sync = true;
            step(e, &ctx, "cuGraphLaunch", launched.map_err(Into::into))?;
            // THIS RUN's parity advances, and only this run's. See `RunGraph::phase`.
            st.runs[ri].phase ^= 1;
        }
        // The eager walk swaps `ssm_state`/`ssm_state_alt` once per KDA layer per step
        // (`kda::kda_cached`). The replay wrote the alternate buffer exactly as the eager step
        // would have; mirror the host swap so the NEXT phase's graph, a fallback to eager, a
        // prime, and any snapshot all see the pointers they expect.
        for il in a..b {
            if let Some(rl) = cache.recur[il].as_mut() {
                std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
            }
        }
        let out_ctx = CapCtx {
            dev,
            lo,
            hi,
            run: 0,
            runs: 0,
            a,
            b,
            phase: 0,
            recapture: false,
        };
        let mut out = step(e, &out_ctx, "alloc(out)", e.uninit(width))?;
        {
            let pool = self.glm5_graph_pool(cache);
            let st = pool
                .stages
                .iter()
                .find(|s| s.dev == dev && s.lo == lo && s.hi == hi)
                .expect("checked above");
            step(
                e,
                &out_ctx,
                "memcpy_dtod(x_out -> out)",
                e.copy_into(&mut out, 0, &st.x_out, width),
            )?;
        }
        crate::GLM5_DECODE_GRAPH_REPLAYS.fetch_add(1, Ordering::Relaxed);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::capturable_runs;

    // The run splitter IS the capture geometry: a wrong split either captures a layer whose
    // launch geometry depends on `cache.pos` (wrong answers on replay) or drops a run.
    #[test]
    fn runs_split_on_the_uncapturable_layers() {
        // 3 capturable + 1 not, repeating — the glm5_next trunk shape (KDA runs broken by the
        // MLA/DSA layer whose kpool geometry is derived on the host).
        let cap = |il: usize| il % 4 != 3;
        assert_eq!(
            capturable_runs(0, 12, cap),
            vec![(0, 3), (4, 7), (8, 11)],
            "whole-trunk split"
        );
        // A pipeline stage cut that starts mid-run keeps only its own layers.
        assert_eq!(capturable_runs(5, 12, cap), vec![(5, 7), (8, 11)]);
        // A run reaching the stage end is CLOSED at `hi`, not dropped.
        assert_eq!(capturable_runs(0, 3, cap), vec![(0, 3)]);
        // Nothing capturable in range -> no runs, and the caller takes the eager walk.
        assert!(capturable_runs(0, 4, |il| il == 4).is_empty());
        // An empty range is not a run.
        assert!(capturable_runs(7, 7, |_| true).is_empty());
    }

    // Every run holds an EVEN number of hc buffer swaps (two per layer: post-attention and
    // post-FFN), which is what makes "the output is back in `x_io`" the replay contract rather
    // than a coincidence of run length.
    #[test]
    fn every_run_length_makes_an_even_number_of_swaps() {
        for (a, b) in capturable_runs(0, 12, |il| il % 4 != 3) {
            assert_eq!(2 * (b - a) % 2, 0, "run [{a}, {b}) swaps must be even");
        }
    }
}
