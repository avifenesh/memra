//! Per-rank CUDA graphs for the SYMMETRIC TP walk (door `MEMRA_GLM5_TP_SYM_GRAPH`, default OFF,
//! lane/glm5-tp-sym-graph-20260907).
//!
//! WHY. tpwalk2 on the 2x B200 pair (2026-09-06, main with the split, the one-shot and the
//! device-routed vrows chain): pp2 59.8 tok/s, tp2s 34.8, tp2s1 (one-shot) 34.6. The split
//! halved every rank's device work and the reduce cost did not move the number: the wall is the
//! HOST, one thread issuing both ranks' launches across two contexts, eagerly, while the PP-2
//! control replays its KDA runs as CUDA graphs (`glm5_decode_graph`, ON since 2026-09-04, which
//! refuses TP-sharded layers by construction). This door gives the symmetric walk the same
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
//! assignments token by token. Like the plain door, every run is captured TWICE back to back
//! (capture records, it does not execute, but the host swaps run on both passes and leave the
//! fields where they started), and replays alternate phase per token. Every KDA layer in the
//! range is inside a run (MLA layers stay eager and own no host swap), so the phase is the token
//! parity of this state.
//!
//! ALLOCATIONS INSIDE THE RECORDED RUN become graph memory nodes (the per-layer temporaries drop
//! before the run ends); the one-shot's stage buffers are pre-sized before the first capture so
//! nothing allocated inside outlives it. If a capture or instantiate refuses, the state latches
//! EAGER for the session and says so once; the walk stays byte-identical.
use crate::Engine;
use crate::glm5_decode_graph::{CapCtx, capture_error, step};
use crate::hybrid::{HybridModel, Mixer};
use crate::hyper::{HyperDecodeWs, HyperTopology};
use cudarc::driver::{CudaGraph, CudaSlice};
use memra_kv::Cache;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub fn on() -> bool {
    std::env::var("MEMRA_GLM5_TP_SYM_GRAPH").as_deref() == Ok("1")
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

pub fn mla_pieces_on() -> bool {
    std::env::var("MEMRA_GLM5_TP_SYM_GRAPH_MLA").as_deref() != Ok("0")
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
    phase: usize,
    failed: bool,
}

/// The plan over `[lo, hi)`: TP-sharded KDA+MoE layers are whole segments; TP-sharded MLA+MoE
/// layers split into PRE / eager middle / FFN when the segment workspace seam is on; everything
/// else (the dense layers, MLA without the seam) stays a whole eager layer.
fn plan_items(m: &HybridModel, lo: usize, hi: usize) -> Vec<Item> {
    let mla_split = mla_pieces_on() && Engine::mla_seg_ws_on();
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
        phase: 0,
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
        eprintln!(
            "[glm5-tp-sym-graph] engaged: {n_graph} graph piece(s) over {n_segs} segment(s) recorded \
             per rank in both ping-pong phases over layers [{lo}, {hi}); {n_mid} MLA middle(s) eager \
             between pieces, {n_eager} whole layer(s) eager"
        );
    }
    // the token: graph pieces replay, the middles and the whole eager layers run in order
    let phase = st.phase;
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
            Piece::Graph { graphs, .. } => {
                {
                    let _m = e.gpu.enter_main()?;
                    graphs[phase][0].launch()?;
                }
                {
                    let _m = peer.gpu.enter_main()?;
                    graphs[phase][1].launch()?;
                }
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
    st.phase ^= 1;
    GLM5_TP_SYM_GRAPH_TOKENS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // residual out
    {
        let src = st.x_io.slice(0..width);
        let mut dst = x.slice_mut(0..width);
        e.stream().memcpy_dtod(&src, &mut dst)?;
    }
    Ok(true)
}
