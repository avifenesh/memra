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
struct RunGraph {
    lo: usize,
    hi: usize,
    /// `[phase 0, phase 1]` — phase p was captured with `recur[il].ssm_state` naming the buffer
    /// the eager walk would read at a step of that parity.
    graphs: [CudaGraph; 2],
    /// Everything the captured bodies allocated and would otherwise return to an engine pool.
    /// Held for the life of the graphs: a transient handed back to the pool and re-issued to
    /// eager work is a buffer every replay scribbles (the draft-graph root cause).
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
    /// Private hc-glue workspace, resident during capture and never returned to the engine pool.
    ws: crate::hyper::HyperDecodeWs,
    /// Private f16 GEMM scratch, swapped resident around capture AND around every replay: the
    /// graphs bake its cvt/Lt pointers, and an eager GEMM between replays would cross-contaminate
    /// them (the `PrimeGraph` precedent).
    f16: Option<crate::f16_ffi::F16Scratch>,
    /// Which ping-pong phase the NEXT replay must use.
    phase: usize,
    /// The `cache.pos` this pool describes. A mismatch means the session moved under the pool.
    next_pos: usize,
    /// `(layer, conv_state, ssm_state, ssm_state_alt)` DEVICE POINTERS as of capture, for every
    /// captured layer. The graphs baked these addresses, so a seam that REPLACES a state buffer
    /// rather than overwriting it leaves the graphs pointing at freed memory — a silent
    /// wrong-answer or a segfault, whichever the allocator hands back. `pos` continuity catches
    /// a rewind; this catches a re-seat, which `pos` alone cannot see. Verified before every
    /// token's replays; any mismatch drops the stage and re-captures.
    state_sig: Vec<(usize, u64, u64, u64)>,
}

/// The `(conv, ssm, ssm_alt)` device pointers of one layer's recurrent state, or `None` when the
/// layer has no recurrent slot.
fn recur_ptrs(e: &Engine, cache: &Cache, il: usize) -> Option<(u64, u64, u64)> {
    use cudarc::driver::DevicePtr;
    let rl = cache.recur.get(il)?.as_ref()?;
    let st = e.stream();
    let (c, _g0) = rl.conv_state.device_ptr(&st);
    let (s, _g1) = rl.ssm_state.device_ptr(&st);
    let (a, _g2) = rl.ssm_state_alt.device_ptr(&st);
    Some((c, s, a))
}

/// Per-session pool: one [`StageGraphs`] per pipeline stage this cache has decoded through.
#[derive(Default)]
pub(crate) struct Glm5DecodeGraphs {
    stages: Vec<StageGraphs>,
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

/// Announce the door's decision, with the reason. Once per DISTINCT message, not once per
/// process: a single `Once` would let an early refusal on one stage silence the engagement line
/// of another, and the gate reads these to explain a vacuous run.
fn note(msg: &str) {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::BTreeSet<String>>> =
        std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    if seen.lock().unwrap().insert(msg.to_string()) {
        eprintln!("[glm5-decode-graph] {msg}");
    }
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

    /// Door admission, evaluated BEFORE the walk takes ownership of the stream state so a
    /// refusal costs nothing and the caller keeps `x`. Names its reason once per distinct reason.
    pub(crate) fn glm5_decode_graph_ready(
        &self,
        e: &Engine,
        cache: &Cache,
        lo: usize,
        hi: usize,
    ) -> bool {
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

        // Read the live state pointers BEFORE the pool takes its mutable borrow of the cache.
        let live_sig: Vec<(usize, u64, u64, u64)> = kda_runs(self, lo, hi)
            .iter()
            .flat_map(|(a, b)| *a..*b)
            .filter_map(|il| recur_ptrs(e, cache, il).map(|(c, s, a)| (il, c, s, a)))
            .collect();
        let need_capture = {
            let pool = self.glm5_graph_pool(cache);
            match pool
                .stages
                .iter()
                .position(|s| s.dev == dev && s.lo == lo && s.hi == hi)
            {
                Some(i) => {
                    // Two independent invalidation checks, because they see different failures.
                    // `pos` continuity catches a REWIND (rollback, reuse retire, prefix restore):
                    // the state contents and the ping-pong phase are no longer the ones the
                    // capture described. The pointer signature catches a RE-SEAT: a seam that
                    // REPLACED a state buffer rather than overwriting it, which leaves the baked
                    // addresses dangling and which `pos` alone cannot see.
                    let stale_pos = pool.stages[i].next_pos != pos;
                    let stale_ptr = pool.stages[i].state_sig != live_sig;
                    if stale_pos || stale_ptr {
                        if stale_ptr {
                            note("re-capture: a captured layer's recurrent-state buffer moved");
                        }
                        pool.stages.remove(i);
                        true
                    } else {
                        false
                    }
                }
                None => true,
            }
        };
        if need_capture {
            self.glm5_capture_stage(e, topology, lo, hi, cache, dev, width, pos)?;
        }

        let runs: Vec<(usize, usize)> = kda_runs(self, lo, hi);
        let mut cursor = lo;
        for (a, b) in runs {
            if a > cursor {
                x = self.hyper_range_decode_eager(e, topology, x, cursor, a, pos_d, pos, cache)?;
            }
            x = self.glm5_replay_run(e, dev, lo, hi, a, b, x, width, cache)?;
            cursor = b;
        }
        if cursor < hi {
            x = self.hyper_range_decode_eager(e, topology, x, cursor, hi, pos_d, pos, cache)?;
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
    #[allow(clippy::too_many_arguments)]
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
    ) -> Res<()> {
        let n_embd = self.cfg.n_embd as usize;
        let runs = kda_runs(self, lo, hi);
        let x_io = e.zeros(width)?;
        let ws = crate::hyper::HyperDecodeWs::new(e, topology, n_embd)?;
        // Pre-sized GENEROUSLY on purpose: this is the m=1 activation staging buffer, and a
        // lazy GROW inside the capture region would be a mem node whose address the next launch
        // recycles. 4 MiB covers any single decode row this trunk can present (up to 2M f16
        // elements) with room over the widest KDA/MoE feature width, and the `ws` half is a fixed
        // 64 MiB by construction (`F16_WS_BYTES`). A few MiB of headroom is the right trade
        // against a capture-time grow.
        let f16 = crate::f16_ffi::F16Scratch::with_capacity(e, (4 << 20).max(n_embd * 16))?;

        let mut stage = StageGraphs {
            dev,
            lo,
            hi,
            runs: Vec::with_capacity(runs.len()),
            x_io,
            ws,
            f16: None,
            phase: 0,
            next_pos: pos,
            state_sig: runs
                .iter()
                .flat_map(|(a, b)| *a..*b)
                .filter_map(|il| recur_ptrs(e, cache, il).map(|(c, s, a)| (il, c, s, a)))
                .collect(),
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
            let pos_d = e.htod_i32(&[pos as i32])?;
            for (a, b) in &runs {
                let (a, b) = (*a, *b);
                let mut phase_graphs: Vec<(CudaGraph, Vec<Box<dyn std::any::Any + Send>>)> =
                    Vec::with_capacity(2);
                for _phase in 0..2 {
                    let x_cell = std::cell::RefCell::new(&mut stage.x_io);
                    let ws_cell = std::cell::RefCell::new(&mut stage.ws);
                    let cache_cell = std::cell::RefCell::new(&mut *cache);
                    let g = e.capture_graph_retained_nowarm(|e| {
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
                        )
                    })?;
                    phase_graphs.push(g);
                }
                let mut it = phase_graphs.into_iter();
                let (g0, mut k0) = it.next().expect("phase 0 captured");
                let (g1, k1) = it.next().expect("phase 1 captured");
                k0.extend(k1);
                k0.extend(std::mem::take(&mut *e.glm5_graph_keep().lock().unwrap()));
                stage.runs.push(RunGraph {
                    lo: a,
                    hi: b,
                    graphs: [g0, g1],
                    _keeper: k0,
                });
                crate::GLM5_DECODE_GRAPH_LAYERS.fetch_add((b - a) as u64, Ordering::Relaxed);
            }
            Ok(())
        })();
        crate::GLM5_GRAPH_CAPTURE_OPEN.store(false, Ordering::Relaxed);
        stage.f16 = e.f16_scratch_swap(prev_f16);
        captured?;
        crate::GLM5_DECODE_GRAPH_CAPTURES.fetch_add(1, Ordering::Relaxed);
        note(&format!(
            "engaged dev={dev} stage=[{lo}, {hi}) runs={} captured_layers={} (2 ping-pong phases \
             each; MLA/DSA layers stay eager)",
            stage.runs.len(),
            stage.runs.iter().map(|r| r.hi - r.lo).sum::<usize>(),
        ));
        self.glm5_graph_pool(cache).stages.push(stage);
        Ok(())
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
            let run = st
                .runs
                .iter()
                .find(|r| r.lo == a && r.hi == b)
                .ok_or_else(|| format!("glm5 decode graph: no captured run [{a}, {b})"))?;
            phase = st.phase;
            e.copy_into(&mut st.x_io, 0, &x, width)?;
            let prev = e.f16_scratch_swap(st.f16.take());
            let launched = run.graphs[phase].launch();
            st.f16 = e.f16_scratch_swap(prev);
            launched?;
            st.phase ^= 1;
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
        let mut out = e.uninit(width)?;
        {
            let pool = self.glm5_graph_pool(cache);
            let st = pool
                .stages
                .iter()
                .find(|s| s.dev == dev && s.lo == lo && s.hi == hi)
                .expect("checked above");
            e.copy_into(&mut out, 0, &st.x_io, width)?;
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
