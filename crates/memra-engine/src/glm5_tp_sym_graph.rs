//! Per-rank CUDA graphs for the SYMMETRIC TP walk (lane/glm5-tp-sym-graph-20260907), with the
//! MLA layers split into graphed halves around an eager middle. The default on sm_100a builds
//! whenever the walk engages, since 2026-11-04, with no switch ([`on`]): on the 2x B200 pair the
//! eager walk read 53.30 / 53.38 tok/s, the KDA runs graphed 71.11 / 78.76, and with the MLA
//! halves 82.58 / 82.18, every arm's 160-token tape equal to the eager walk's (tpwalk3 v5, two
//! reps; darklanes `research/glm5-b200-mint-20260904/LANE.md`, ledger
//! `VERDICT:glm5-tp-sym-graph-byte-identical-1.65x`). Other builds run the walk eager: no receipt
//! there. The eager arm stays reachable for a tape check through
//! `MEMRA_GLM5_GRAPH_NO_CAPTURE=1`, which every glm5 graph door declines on
//! (`glm5_graph_process_refusal`).
//!
//! WHY. tpwalk2 on the 2x B200 pair (2026-09-06, main with the split, the one-shot and the
//! device-routed vrows chain): pp2 59.8 tok/s, tp2s 34.8, tp2s1 (one-shot) 34.6. The split
//! halved every rank's device work and the reduce cost did not move the number: the wall is the
//! HOST, one thread issuing both ranks' launches across two contexts, eagerly, while the PP-2
//! control replays its KDA runs as CUDA graphs (`glm5_decode_graph`, ON since 2026-09-04, which
//! refuses TP-sharded layers by construction). This module gives the symmetric walk the same
//! mechanism: every contiguous run of KDA layers is recorded ONCE PER RANK as a graph on that
//! rank's own stream, and a token replays two graphs per run instead of issuing ~40 launches per
//! layer per rank.
//!
//! WHAT MAKES THE SYMMETRIC WALK CAPTURABLE where the root-orchestrated one is not: its
//! crossings are the one-shot all-reduce (`memra_tp_ar_1stage_kernel`), a kernel whose flags
//! live in device memory and whose sequence word is read ON DEVICE (`self_sg->seq[b] + 1`), so a
//! replay keeps counting; no events cross the ranks inside a layer, so two simultaneous
//! captures (one per stream, RELAXED) record no cross-capture dependency. The per-token fan-out
//! of the residual and the positions uses the event-published transport and stays OUTSIDE the
//! captured runs, landing in this state's STABLE peer buffers.
//!
//! FIXED BUFFERS. A graph bakes every operand address. The residual stream is carried in
//! `x_io` (root) / `x_peer_io` (peer), the positions in `pos_peer_io`, the hc workspaces are
//! this state's own pair (never the per-token pool), and the caller's `x` is copied in before
//! the first run and out after the last layer. Each layer swaps the residual with `ws.xb` twice
//! per rank, so a run of whole layers leaves it in the buffer it entered in
//! (`glm5_decode_graph::every_run_length_makes_an_even_number_of_swaps`).
//!
//! THE KDA PING-PONG. `kda_cached` swaps the recurrent state buffers on the HOST after every
//! KDA layer; a replay runs no host code, so the device alternates between the two pointer
//! assignments token by token. Like the plain graph door, every run is captured TWICE back to back
//! (capture records, it does not execute, but the host swaps run on both passes and leave the
//! fields where they started). After every replay the host swap is MIRRORED for each graphed
//! KDA layer on every rank, as the plain graph door does, so `glm5_tp_recur` always names the buffer
//! the device last wrote: an eager reader between graphed tokens (a t>1 prime on a reuse-pool
//! continuation, the suffix prime, a prefix snapshot, the eager fallback) reads the current
//! state. The phase a token replays is not a parity counter: it is read from the planes, by
//! comparing each graphed layer's live `(conv, ssm, alt)` pointers with the ones the phase-0
//! capture baked. An eager walk between replays advances the roles by however many tokens it
//! ran and the next replay follows; a plane that matches neither phase (re-seated buffers) or
//! layers that disagree latch the state EAGER before anything of the token has run.
//!
//! PRECONDITIONS are the plain graph door's process-level ones (`glm5_graph_process_refusal`): with
//! `MEMRA_HTOD_DIET` off the shared expert uploads a pageable constant per MoE layer, which a
//! capture records as a host pointer and replays as stale bytes; the bare arm of tpwalk4 v2
//! (2026-09-07 00:56Z) decoded garbage from token 2 that way, tape silently different while the
//! posture arm (diet on) was byte-identical. The graph declines by name instead.
//!
//! ALLOCATIONS INSIDE THE RECORDED RUN become graph memory nodes (the per-layer temporaries drop
//! before the run ends); the one-shot's stage buffers are pre-sized before the first capture so
//! nothing allocated inside outlives it. If a capture or instantiate refuses, the state latches
//! EAGER for the session and says so once; the walk stays byte-identical because the KDA planes'
//! ping-pong roles are put back first (a refusal inside phase 0, or between the phases, leaves
//! the layers it passed swapped an odd number of times).
use crate::Engine;
use crate::glm5_decode_graph::{CapCtx, capture_error, step};
use crate::hybrid::{HybridModel, Mixer};
use crate::hyper::{HyperDecodeWs, HyperTopology};
use cudarc::driver::{CudaGraph, CudaSlice};
use memra_kv::Cache;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Whether the symmetric walk records its per-rank graphs: on sm_100a builds, always (the
/// receipt's card; module doc). Keyed on `BUILT_CUDA_ARCH` like the other per-device defaults.
pub fn on() -> bool {
    crate::BUILT_CUDA_ARCH == "100a"
}

/// Replays of a whole graphed token (both ranks): the engagement receipt.
pub static GLM5_TP_SYM_GRAPH_TOKENS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// One recordable step of the walk: a whole KDA layer, or a half of a split MLA layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Seg {
    Kda(usize),
    MlaPre(usize),
    MlaFfn(usize),
}

/// One piece of the token: a graph over a run of segments (`[phase][rank]`), the eager middle
/// of a split MLA layer, or a whole layer the plan keeps eager (dense FFN, or MLA when the
/// half-capture is off).
enum Piece {
    Graph {
        segs: Vec<Seg>,
        graphs: [[CudaGraph; 2]; 2],
    },
    MlaMid(usize),
    Eager(usize),
}

/// The plan before capture: what each piece will be.
#[derive(Debug, PartialEq, Eq)]
enum Item {
    Seg(Seg),
    MlaMid(usize),
    Eager(usize),
}

// SAFETY (the same reading as `glm5_decode_graph`'s pool): the raw CUgraph/CUgraphExec handles
// inside `CudaGraph` are CONTEXT-bound, not thread-bound; every launch here enters the owning
// context first (`enter_main`), and the session cache moves between worker threads only between
// tokens, never while a token is mid-replay.
unsafe impl Send for SymGraphState {}

pub(crate) struct SymGraphState {
    lo: usize,
    hi: usize,
    x_io: CudaSlice<f32>,
    x_peer_io: CudaSlice<f32>,
    pos_peer_io: CudaSlice<i32>,
    ws_root: HyperDecodeWs,
    ws_peer: HyperDecodeWs,
    /// Pre-`wo` attention output of a split MLA layer, per rank: the eager middle writes it,
    /// the FFN piece bakes it.
    mla_a_root: Option<CudaSlice<f32>>,
    mla_a_peer: Option<CudaSlice<f32>>,
    pieces: Vec<Piece>,
    /// The first token of the state ran eager (segment workspaces allocated outside capture).
    warm: bool,
    /// Every graphed KDA layer's per-rank `(conv, ssm, alt)` pointers as the phase-0 capture
    /// baked them; `replay_phase` reads each token's phase from the live planes against it.
    sig: Vec<(usize, Vec<[u64; 3]>)>,
    failed: bool,
}

/// The plan over `[lo, hi)`: TP-sharded KDA+MoE layers are whole segments; TP-sharded MLA+MoE
/// layers split into PRE / eager middle / FFN when the segment workspace seam is on (the sm_100a
/// default; its `MEMRA_MLA_SEG_WS=0` rollback keeps them whole); everything else (the dense
/// layers without their peer replica, MLA without the seam) stays a whole eager layer. Keeping
/// every MLA layer whole with the seam on read 71.11 / 78.76 tok/s against the halves' 82.58 /
/// 82.18 (tpwalk3 v5), so that arm has no switch.
fn plan_items(m: &HybridModel, lo: usize, hi: usize) -> Vec<Item> {
    let mla_split = Engine::mla_seg_ws_on();
    let mut out = Vec::with_capacity(hi.saturating_sub(lo) * 3);
    for il in lo..hi {
        let moe = matches!(&m.layers[il].ffn, crate::hybrid::Ffn::Moe(_));
        // a dense-FFN layer is capturable once the peer holds the FFN replica (no fan-out)
        let dense_local = m.layers[il]
            .tp_glue
            .first()
            .is_some_and(|g| g.dense.is_some());
        match &m.layers[il].mixer {
            Mixer::Kda(la) if la.tp.is_some() && (moe || dense_local) => {
                out.push(Item::Seg(Seg::Kda(il)))
            }
            Mixer::Mla(mla) if mla.tp.is_some() && moe && mla_split => {
                out.push(Item::Seg(Seg::MlaPre(il)));
                out.push(Item::MlaMid(il));
                out.push(Item::Seg(Seg::MlaFfn(il)));
            }
            _ => out.push(Item::Eager(il)),
        }
    }
    out
}

/// Consecutive segments become one graph piece; the plan's other items stay their own pieces.
fn group_runs(items: &[Item]) -> Vec<Vec<Seg>> {
    let mut runs = Vec::new();
    let mut cur: Vec<Seg> = Vec::new();
    for it in items {
        match it {
            Item::Seg(sg) => cur.push(*sg),
            _ => {
                if !cur.is_empty() {
                    runs.push(std::mem::take(&mut cur));
                }
            }
        }
    }
    if !cur.is_empty() {
        runs.push(cur);
    }
    runs
}

fn seg_layer(sg: Seg) -> usize {
    match sg {
        Seg::Kda(il) | Seg::MlaPre(il) | Seg::MlaFfn(il) => il,
    }
}

/// The graphed KDA layers of a plan: every layer whose host ping-pong a replay must mirror.
fn graphed_kda_layers(items: &[Item]) -> Vec<usize> {
    items
        .iter()
        .filter_map(|it| match it {
            Item::Seg(Seg::Kda(il)) => Some(*il),
            _ => None,
        })
        .collect()
}

/// Layer `il`'s per-rank `(conv_state, ssm_state, ssm_state_alt)` device pointers, or `None`
/// when the layer carries no TP planes.
fn plane_ptrs(
    e: &Engine,
    rt: &crate::glm5_tp::Glm5TpRt,
    cache: &Cache,
    il: usize,
) -> Res<Option<Vec<[u64; 3]>>> {
    use cudarc::driver::DevicePtr;
    let Some(planes) = cache.glm5_tp_recur.get(il).and_then(Option::as_ref) else {
        return Ok(None);
    };
    let mut out = Vec::with_capacity(planes.len());
    for (r, pl) in planes.iter().enumerate() {
        let dev = crate::glm5_tp::rank_engine(e, rt, r);
        let _m = dev.gpu.enter_main()?;
        let st = dev.stream();
        let (c, _gc) = pl.conv_state.device_ptr(&st);
        let (s, _gs) = pl.ssm_state.device_ptr(&st);
        let (a, _ga) = pl.ssm_state_alt.device_ptr(&st);
        out.push([c, s, a]);
    }
    Ok(Some(out))
}

/// The phase-0 signature of `layers`: what the capture is about to bake.
fn kda_sig(
    e: &Engine,
    rt: &crate::glm5_tp::Glm5TpRt,
    cache: &Cache,
    layers: &[usize],
) -> Res<Vec<(usize, Vec<[u64; 3]>)>> {
    let mut out = Vec::with_capacity(layers.len());
    for &il in layers {
        let ptrs = plane_ptrs(e, rt, cache, il)?
            .ok_or_else(|| format!("layer {il}: graphed KDA layer carries no TP planes"))?;
        out.push((il, ptrs));
    }
    Ok(out)
}

/// Which captured phase matches the live planes: 0 when every graphed layer on every rank holds
/// the phase-0 `(conv, ssm, alt)` the capture baked, 1 when every one holds it with `ssm` and
/// `alt` exchanged. Anything else (a plane gone or re-seated, layers or ranks that disagree) is
/// an error naming the first plane that does not fit: no captured graph reads those buffers.
fn phase_of(
    sig: &[(usize, Vec<[u64; 3]>)],
    live: &[(usize, Option<Vec<[u64; 3]>>)],
) -> Result<usize, String> {
    if sig.len() != live.len() {
        return Err(format!(
            "{} graphed KDA layer(s) baked, {} read live",
            sig.len(),
            live.len()
        ));
    }
    let mut phase: Option<usize> = None;
    for ((il, want), (il_live, got)) in sig.iter().zip(live) {
        if il != il_live {
            return Err(format!("layer order moved: baked {il}, live {il_live}"));
        }
        let got = got
            .as_ref()
            .ok_or_else(|| format!("layer {il}: TP planes are gone"))?;
        if got.len() != want.len() {
            return Err(format!(
                "layer {il}: {} rank plane(s) baked, {} live",
                want.len(),
                got.len()
            ));
        }
        for (r, (w, g)) in want.iter().zip(got).enumerate() {
            let p = if g == w {
                0
            } else if *g == [w[0], w[2], w[1]] {
                1
            } else {
                return Err(format!(
                    "layer {il} rank {r}: live (conv, ssm, alt) {g:#x?} matches neither baked \
                     phase of {w:#x?}"
                ));
            };
            match phase {
                None => phase = Some(p),
                Some(q) if q == p => {}
                Some(q) => {
                    return Err(format!(
                        "layer {il} rank {r} sits at phase {p}, an earlier plane at {q}"
                    ));
                }
            }
        }
    }
    Ok(phase.unwrap_or(0))
}

/// The live phase of the graphed layers (see [`phase_of`]).
fn replay_phase(
    e: &Engine,
    rt: &crate::glm5_tp::Glm5TpRt,
    cache: &Cache,
    sig: &[(usize, Vec<[u64; 3]>)],
) -> Res<Result<usize, String>> {
    let mut live = Vec::with_capacity(sig.len());
    for (il, _) in sig {
        live.push((*il, plane_ptrs(e, rt, cache, *il)?));
    }
    Ok(phase_of(sig, &live))
}

/// The host ping-pong a replay of `layers` stands for: swap every rank plane's `ssm_state` and
/// `ssm_state_alt`, as the eager `kda_tp_partials_sym` does after its kernels, so the host
/// roles name the buffer the replay wrote.
fn mirror_swaps(cache: &mut Cache, layers: impl IntoIterator<Item = usize>) -> Res<()> {
    for il in layers {
        let planes = cache
            .glm5_tp_recur
            .get_mut(il)
            .and_then(Option::as_mut)
            .ok_or_else(|| format!("layer {il}: TP recur planes vanished under a replay"))?;
        for pl in planes.iter_mut() {
            std::mem::swap(&mut pl.ssm_state, &mut pl.ssm_state_alt);
        }
    }
    Ok(())
}

/// Swap back every plane whose `ssm_state` no longer holds the pointer the phase-0 signature
/// saw. A capture body runs the HOST swap of `kda_tp_partials_sym` and executes nothing, so a
/// capture refused partway (or at end/instantiate after phase 0) leaves the layers it passed
/// swapped an odd number of times. The bytes need no restore: a capture records and never
/// executes.
fn restore_ssm_roles(
    e: &Engine,
    rt: &crate::glm5_tp::Glm5TpRt,
    cache: &mut Cache,
    sig: &[(usize, Vec<[u64; 3]>)],
) -> Res<()> {
    for (il, want) in sig {
        let live = plane_ptrs(e, rt, cache, *il)?
            .ok_or_else(|| format!("layer {il}: TP recur planes vanished during capture"))?;
        let planes = cache.glm5_tp_recur[*il]
            .as_mut()
            .ok_or_else(|| format!("layer {il}: TP recur planes vanished during capture"))?;
        for ((pl, w), g) in planes.iter_mut().zip(want).zip(&live) {
            if g[1] != w[1] {
                std::mem::swap(&mut pl.ssm_state, &mut pl.ssm_state_alt);
            }
        }
    }
    Ok(())
}

/// Two simultaneous stream captures, one per rank, of one body that issues to both.
fn capture_pair<F>(e: &Engine, peer: &Engine, ctx: &CapCtx, mut body: F) -> Res<[CudaGraph; 2]>
where
    F: FnMut() -> Res<()>,
{
    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
    for eng in [e, peer] {
        let _m = eng.gpu.enter_main()?;
        step(
            eng,
            ctx,
            "synchronize(before begin_capture)",
            eng.stream().synchronize().map_err(Into::into),
        )?;
    }
    let tracking: Vec<bool> = [e, peer]
        .iter()
        .map(|eng| eng.ctx().is_event_tracking())
        .collect();
    for (i, eng) in [e, peer].iter().enumerate() {
        if tracking[i] {
            unsafe { eng.ctx().disable_event_tracking() };
        }
    }
    let out = (|| -> Res<[CudaGraph; 2]> {
        for eng in [e, peer] {
            let _m = eng.gpu.enter_main()?;
            step(
                eng,
                ctx,
                "cuStreamBeginCapture(RELAXED)",
                eng.stream()
                    .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                    .map_err(Into::into),
            )?;
        }
        let body_res = body();
        let mut graphs = Vec::with_capacity(2);
        let mut end_err: Option<Box<dyn std::error::Error>> = None;
        for eng in [e, peer] {
            let _m = eng.gpu.enter_main()?;
            match eng.stream().end_capture(
                CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            ) {
                Ok(Some(g)) => graphs.push(g),
                Ok(None) => {
                    end_err.get_or_insert_with(|| "capture produced no graph".into());
                }
                Err(err) => {
                    end_err.get_or_insert_with(|| Box::new(err) as Box<dyn std::error::Error>);
                }
            }
        }
        if let Err(err) = body_res {
            capture_error(e, ctx, "capture body", &err);
            return Err(err);
        }
        if let Some(err) = end_err {
            capture_error(e, ctx, "cuStreamEndCapture+cuGraphInstantiate", &err);
            return Err(err);
        }
        let mut it = graphs.into_iter();
        let g0 = it.next().ok_or("no root graph")?;
        let g1 = it.next().ok_or("no peer graph")?;
        {
            let _m = e.gpu.enter_main()?;
            step(
                e,
                ctx,
                "cuGraphUpload(root)",
                g0.upload().map_err(Into::into),
            )?;
        }
        {
            let _m = peer.gpu.enter_main()?;
            step(
                peer,
                ctx,
                "cuGraphUpload(peer)",
                g1.upload().map_err(Into::into),
            )?;
        }
        Ok([g0, g1])
    })();
    for (i, eng) in [e, peer].iter().enumerate() {
        if tracking[i] {
            unsafe { eng.ctx().enable_event_tracking() };
        }
    }
    out
}

fn take_state(
    m: &HybridModel,
    e: &Engine,
    peer: &Engine,
    topology: &HyperTopology,
    cache: &mut Cache,
    lo: usize,
    hi: usize,
) -> Res<Box<SymGraphState>> {
    let n_embd = m.cfg.n_embd as usize;
    // the residual carries every hyper-connection stream: `streams * n_embd` per token
    let width = topology.streams * n_embd;
    if let Some(b) = cache.glm5_tp_sym_graph.take() {
        match b.downcast::<SymGraphState>() {
            Ok(st) if st.lo == lo && st.hi == hi => return Ok(st),
            Ok(_) | Err(_) => {}
        }
    }
    // the pre-`wo` handoff width of a TP-sharded MLA layer (all MLA layers share one geometry)
    let mla_a = (lo..hi).find_map(|il| match &m.layers[il].mixer {
        Mixer::Mla(mla) if mla.tp.is_some() => Some(mla.wo.in_features()),
        _ => None,
    });
    Ok(Box::new(SymGraphState {
        lo,
        hi,
        x_io: e.zeros(width)?,
        x_peer_io: peer.zeros(width)?,
        pos_peer_io: {
            let _m = peer.gpu.enter_main()?;
            peer.htod_i32(&[0])?
        },
        ws_root: HyperDecodeWs::new(e, topology, n_embd)?,
        ws_peer: {
            let _m = peer.gpu.enter_main()?;
            HyperDecodeWs::new(peer, topology, n_embd)?
        },
        mla_a_root: mla_a.map(|n| e.zeros(n)).transpose()?,
        mla_a_peer: mla_a
            .map(|n| {
                let _m = peer.gpu.enter_main()?;
                peer.zeros(n)
            })
            .transpose()?,
        pieces: Vec::new(),
        warm: false,
        sig: Vec::new(),
        failed: false,
    }))
}

/// The graphed token. Returns Ok(true) when this token ran through the state (graphs, plus the
/// eager MLA layers), Ok(false) when the state is latched eager and the caller's loop must run.
#[allow(clippy::too_many_arguments)] // allow: the walk's inputs for both ranks
pub(crate) fn walk_graphed(
    m: &HybridModel,
    e: &Engine,
    peer: &Engine,
    rt: &std::sync::Arc<crate::glm5_tp::Glm5TpRt>,
    topology: &HyperTopology,
    x: &mut CudaSlice<f32>,
    x_peer: &mut CudaSlice<f32>,
    _ws: &mut HyperDecodeWs,
    _ws_peer: &mut HyperDecodeWs,
    pos_d: &CudaSlice<i32>,
    pos_peer: &CudaSlice<i32>,
    lo: usize,
    hi: usize,
    cache: &mut Cache,
) -> Res<bool> {
    let _numeric = m.enter_numeric();
    let n_embd = m.cfg.n_embd as usize;
    let width = topology.streams * n_embd;
    if x.len() != width || x_peer.len() != width || pos_d.len() != 1 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            eprintln!(
                "[glm5-tp-sym-graph] declined: t=1 walks only (x.len()={} pos.len()={})",
                x.len(),
                pos_d.len()
            )
        });
        return Ok(false);
    }
    // the plain door's process-level preconditions hold here too (the bare arm of tpwalk4 v2
    // captured a pageable shared-expert upload and decoded garbage from token 2)
    if let Some(reason) = crate::glm5_decode_graph::glm5_graph_process_refusal(e) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            eprintln!(
                "[glm5-tp-sym-graph] declined: {reason}; this session's symmetric walk stays EAGER"
            )
        });
        return Ok(false);
    }
    let items = plan_items(m, lo, hi);
    if !items.iter().any(|it| matches!(it, Item::Seg(_))) {
        return Ok(false);
    }
    // the one-shot's stage buffers, sized before anything is recorded
    rt.ar_prepare(e, n_embd)?;
    let dev = e.ctx().ordinal();
    // The state lives in the session cache and the layer step needs the cache too: take it out
    // for the token and put it back on every path.
    let mut st = take_state(m, e, peer, topology, cache, lo, hi)?;
    let r = walk_token(
        m, e, peer, rt, topology, x, x_peer, pos_d, pos_peer, lo, hi, cache, dev, &items, &mut st,
    );
    cache.glm5_tp_sym_graph = Some(st);
    r
}

#[allow(clippy::too_many_arguments)] // allow: the token walk over both ranks and its state
fn walk_token(
    m: &HybridModel,
    e: &Engine,
    peer: &Engine,
    rt: &crate::glm5_tp::Glm5TpRt,
    topology: &HyperTopology,
    x: &mut CudaSlice<f32>,
    x_peer: &mut CudaSlice<f32>,
    pos_d: &CudaSlice<i32>,
    pos_peer: &CudaSlice<i32>,
    lo: usize,
    hi: usize,
    cache: &mut Cache,
    dev: usize,
    items: &[Item],
    st: &mut SymGraphState,
) -> Res<bool> {
    let width = topology.streams * m.cfg.n_embd as usize;
    if st.failed {
        return Ok(false);
    }
    // stable entry buffers: residual in (all hc streams), peer residual in, peer position in
    {
        let src = x.slice(0..width);
        let mut dst = st.x_io.slice_mut(0..width);
        e.stream().memcpy_dtod(&src, &mut dst)?;
    }
    {
        let _m = peer.gpu.enter_main()?;
        let src = x_peer.slice(0..width);
        let mut dst = st.x_peer_io.slice_mut(0..width);
        peer.stream().memcpy_dtod(&src, &mut dst)?;
        let ps = pos_peer.slice(0..1);
        let mut pd = st.pos_peer_io.slice_mut(0..1);
        peer.stream().memcpy_dtod(&ps, &mut pd)?;
    }
    // the first token of the state runs EAGER through the stable buffers: the MLA segment
    // workspaces and their q8 pairs are allocated on it, outside any capture region
    if !st.warm {
        let SymGraphState {
            x_io,
            x_peer_io,
            pos_peer_io,
            ws_root,
            ws_peer,
            ..
        } = &mut *st;
        for il in lo..hi {
            m.sym_layer_step(
                e,
                peer,
                rt,
                topology,
                il,
                x_io,
                x_peer_io,
                ws_root,
                ws_peer,
                pos_d,
                pos_peer_io,
                cache,
            )?;
        }
        st.warm = true;
        let src = st.x_io.slice(0..width);
        let mut dst = x.slice_mut(0..width);
        e.stream().memcpy_dtod(&src, &mut dst)?;
        return Ok(true);
    }
    // the pieces are recorded on first use, both phases back to back
    if st.pieces.is_empty() {
        let roles = kda_sig(e, rt, cache, &graphed_kda_layers(items))?;
        let runs = group_runs(items);
        let n = runs.len();
        let mut pieces: Vec<Piece> = Vec::with_capacity(items.len());
        let mut ri = 0usize;
        let mut i = 0usize;
        while i < items.len() {
            match &items[i] {
                Item::MlaMid(il) => {
                    pieces.push(Piece::MlaMid(*il));
                    i += 1;
                }
                Item::Eager(il) => {
                    pieces.push(Piece::Eager(*il));
                    i += 1;
                }
                Item::Seg(_) => {
                    let segs = runs[ri].clone();
                    let a = seg_layer(segs[0]);
                    let b = seg_layer(*segs.last().expect("a run has segments")) + 1;
                    let mut phases: Vec<[CudaGraph; 2]> = Vec::with_capacity(2);
                    for phase in 0..2 {
                        let ctx = CapCtx {
                            dev,
                            lo,
                            hi,
                            run: ri,
                            runs: n,
                            a,
                            b,
                            phase,
                            recapture: false,
                        };
                        let SymGraphState {
                            x_io,
                            x_peer_io,
                            pos_peer_io,
                            ws_root,
                            ws_peer,
                            mla_a_root,
                            mla_a_peer,
                            ..
                        } = &mut *st;
                        let cap = capture_pair(e, peer, &ctx, || {
                            for sg in &segs {
                                match *sg {
                                    Seg::Kda(il) => m.sym_layer_step(
                                        e,
                                        peer,
                                        rt,
                                        topology,
                                        il,
                                        x_io,
                                        x_peer_io,
                                        ws_root,
                                        ws_peer,
                                        pos_d,
                                        pos_peer_io,
                                        cache,
                                    )?,
                                    Seg::MlaPre(il) => m.sym_mla_pre_piece(
                                        e,
                                        peer,
                                        topology,
                                        il,
                                        x_io,
                                        x_peer_io,
                                        ws_root,
                                        ws_peer,
                                        pos_d,
                                        pos_peer_io,
                                    )?,
                                    Seg::MlaFfn(il) => {
                                        let (ar, ap) = match (&*mla_a_root, &*mla_a_peer) {
                                            (Some(ar), Some(ap)) => (ar, ap),
                                            _ => {
                                                return Err(format!(
                                                    "layer {il}: sym MLA FFN piece without handoff buffers"
                                                )
                                                .into());
                                            }
                                        };
                                        m.sym_mla_ffn_piece(
                                            e, peer, rt, topology, il, ar, ap, x_io, x_peer_io,
                                            ws_root, ws_peer,
                                        )?
                                    }
                                }
                            }
                            Ok(())
                        });
                        match cap {
                            Ok(g) => phases.push(g),
                            Err(err) => {
                                // the eager walk reads the planes next: undo the odd swaps first
                                restore_ssm_roles(e, rt, cache, &roles)?;
                                eprintln!(
                                    "[glm5-tp-sym-graph] capture refused for run [{a}, {b}) phase {phase}: {err}; \
                                     this session's symmetric walk stays EAGER (byte-identical)"
                                );
                                st.failed = true;
                                st.pieces.clear();
                                return Ok(false);
                            }
                        }
                    }
                    let mut it = phases.into_iter();
                    let p0 = it.next().ok_or("phase 0 missing")?;
                    let p1 = it.next().ok_or("phase 1 missing")?;
                    pieces.push(Piece::Graph {
                        graphs: [p0, p1],
                        segs,
                    });
                    i += runs[ri].len();
                    ri += 1;
                }
            }
        }
        let n_graph = pieces
            .iter()
            .filter(|p| matches!(p, Piece::Graph { .. }))
            .count();
        let n_mid = pieces
            .iter()
            .filter(|p| matches!(p, Piece::MlaMid(_)))
            .count();
        let n_eager = pieces
            .iter()
            .filter(|p| matches!(p, Piece::Eager(_)))
            .count();
        let n_segs: usize = pieces
            .iter()
            .map(|p| match p {
                Piece::Graph { segs, .. } => segs.len(),
                _ => 0,
            })
            .sum();
        st.pieces = pieces;
        st.sig = roles;
        eprintln!(
            "[glm5-tp-sym-graph] engaged: {n_graph} graph piece(s) over {n_segs} segment(s) recorded \
             per rank in both ping-pong phases over layers [{lo}, {hi}); {n_mid} MLA middle(s) eager \
             between pieces, {n_eager} whole layer(s) eager"
        );
    }
    // The phase this token replays, read from the planes: the host roles name the buffer the
    // device last wrote (every replay mirrors its swaps, every eager walk swaps for itself), so
    // tokens that ran eager in between (a continuation prime, the eager fallback) move it too.
    // A plane that fits neither baked phase latches EAGER here, before any of the token ran.
    let phase = match replay_phase(e, rt, cache, &st.sig)? {
        Ok(p) => p,
        Err(why) => {
            eprintln!(
                "[glm5-tp-sym-graph] the KDA planes no longer fit the recorded graphs ({why}); \
                 this session's symmetric walk stays EAGER (byte-identical)"
            );
            // the execs stay in the state and drop with the cache: a replay of them may still be
            // in flight on the streams
            st.failed = true;
            return Ok(false);
        }
    };
    // the token: graph pieces replay, the middles and the whole eager layers run in order
    let SymGraphState {
        x_io,
        x_peer_io,
        pos_peer_io,
        ws_root,
        ws_peer,
        mla_a_root,
        mla_a_peer,
        pieces,
        ..
    } = &mut *st;
    for piece in pieces.iter() {
        match piece {
            Piece::Graph { graphs, segs } => {
                {
                    let _m = e.gpu.enter_main()?;
                    graphs[phase][0].launch()?;
                }
                {
                    let _m = peer.gpu.enter_main()?;
                    graphs[phase][1].launch()?;
                }
                // the replay wrote each KDA layer's alternate buffer, as the eager step would
                // have: mirror the host swap so every later reader sees the pointers it expects
                mirror_swaps(
                    cache,
                    segs.iter().filter_map(|sg| match sg {
                        Seg::Kda(il) => Some(*il),
                        _ => None,
                    }),
                )?;
            }
            Piece::MlaMid(il) => {
                let (ar, ap) = match (mla_a_root.as_mut(), mla_a_peer.as_mut()) {
                    (Some(ar), Some(ap)) => (ar, ap),
                    _ => {
                        return Err(
                            format!("layer {il}: sym MLA middle without handoff buffers").into(),
                        );
                    }
                };
                m.sym_mla_mid_eager(
                    e,
                    peer,
                    *il,
                    ws_root,
                    ws_peer,
                    pos_d,
                    pos_peer_io,
                    cache,
                    ar,
                    ap,
                )?;
            }
            Piece::Eager(il) => {
                m.sym_layer_step(
                    e,
                    peer,
                    rt,
                    topology,
                    *il,
                    x_io,
                    x_peer_io,
                    ws_root,
                    ws_peer,
                    pos_d,
                    pos_peer_io,
                    cache,
                )?;
            }
        }
    }
    GLM5_TP_SYM_GRAPH_TOKENS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // residual out
    {
        let src = st.x_io.slice(0..width);
        let mut dst = x.slice_mut(0..width);
        e.stream().memcpy_dtod(&src, &mut dst)?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::{Item, Seg, graphed_kda_layers, phase_of};

    /// Two graphed KDA layers on two ranks, phase-0 pointers `(conv, ssm, alt)`.
    fn sig() -> Vec<(usize, Vec<[u64; 3]>)> {
        vec![
            (3, vec![[0x100, 0x200, 0x300], [0x1100, 0x1200, 0x1300]]),
            (4, vec![[0x400, 0x500, 0x600], [0x1400, 0x1500, 0x1600]]),
        ]
    }

    fn live(planes: &[(usize, Vec<[u64; 3]>)]) -> Vec<(usize, Option<Vec<[u64; 3]>>)> {
        planes
            .iter()
            .map(|(il, p)| (*il, Some(p.clone())))
            .collect()
    }

    fn swapped(planes: &[(usize, Vec<[u64; 3]>)]) -> Vec<(usize, Vec<[u64; 3]>)> {
        planes
            .iter()
            .map(|(il, p)| (*il, p.iter().map(|w| [w[0], w[2], w[1]]).collect()))
            .collect()
    }

    #[test]
    fn sym_graph_phase_is_read_from_the_planes() {
        let s = sig();
        // as captured: phase 0
        assert_eq!(phase_of(&s, &live(&s)), Ok(0));
        // one replay mirrored, or an odd number of eager tokens in between: phase 1
        assert_eq!(phase_of(&s, &live(&swapped(&s))), Ok(1));
        // an even number of eager tokens after an odd number of replays lands on phase 1 too:
        // the planes say so, no token count is involved
        assert_eq!(phase_of(&s, &live(&swapped(&swapped(&swapped(&s))))), Ok(1));
        // nothing graphed: one phase
        assert_eq!(phase_of(&[], &[]), Ok(0));
    }

    #[test]
    fn sym_graph_phase_refuses_planes_no_graph_baked() {
        let s = sig();
        // layers that disagree
        let mut mixed = live(&s);
        mixed[1] = live(&swapped(&s))[1].clone();
        assert!(phase_of(&s, &mixed).unwrap_err().contains("layer 4 rank 0"));
        // ranks that disagree
        let mut ranks = live(&s);
        ranks[0].1.as_mut().unwrap()[1] = [0x1100, 0x1300, 0x1200];
        assert!(phase_of(&s, &ranks).unwrap_err().contains("layer 3 rank 1"));
        // a re-seated ssm buffer
        let mut reseat = live(&s);
        reseat[0].1.as_mut().unwrap()[0][1] = 0x9900;
        assert!(
            phase_of(&s, &reseat)
                .unwrap_err()
                .contains("matches neither")
        );
        // a re-seated conv ring
        let mut conv = live(&s);
        conv[1].1.as_mut().unwrap()[1][0] = 0x9900;
        assert!(phase_of(&s, &conv).unwrap_err().contains("layer 4 rank 1"));
        // planes gone
        let mut gone = live(&s);
        gone[0].1 = None;
        assert!(phase_of(&s, &gone).unwrap_err().contains("gone"));
        // a layer count that moved
        assert!(phase_of(&s, &live(&s[..1])).is_err());
    }

    #[test]
    fn sym_graph_mirrors_only_graphed_kda_layers() {
        let items = [
            Item::Seg(Seg::Kda(0)),
            Item::Seg(Seg::MlaPre(1)),
            Item::MlaMid(1),
            Item::Seg(Seg::MlaFfn(1)),
            Item::Eager(2),
            Item::Seg(Seg::Kda(3)),
        ];
        assert_eq!(graphed_kda_layers(&items), vec![0, 3]);
    }
}
