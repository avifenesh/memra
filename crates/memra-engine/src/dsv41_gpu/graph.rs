//! A span of a decode step as one CUDA graph a rank (`MEMRA_DSV41_DECODE_GRAPH=1`).
//!
//! WHY. A one-row layer on four ranks is 52 launches a rank, most of them a few blocks wide and a
//! few microseconds long, and the thread that issues them spends 204 us a layer doing it. A graph
//! replays the same kernels with one call, so the host stops being the floor of a layer, and a
//! branch on the second stream costs the host nothing: forks become free.
//!
//! WHAT A CAPTURE IS. [`Group::capture`] runs a closure with every rank's stream recording: the
//! launches the closure issues become nodes of that rank's graph and nothing executes. The
//! closure is the code the step runs anyway, so a replay is the same kernels with the same
//! arguments in the same order on each stream; the second stream's work is a branch of the graph
//! (its fork and join events are recorded with it). Three things make a recorded launch valid
//! later:
//!
//! - its buffers stay where they were: under a capture every temporary comes from a free list
//!   of the capture's own ([`TmpPool`] with `capture` set), allocated outside the stream's
//!   order, and the capture keeps the list, so no other launch is ever handed that memory;
//! - its arguments hold nothing the host derived for that one step except the addresses of
//!   the closure's inputs, which are named to the capture: it finds every node argument that
//!   holds an address inside an input buffer, and a replay whose inputs sit elsewhere writes
//!   the new addresses into those arguments first (`cuGraphExecKernelNodeSetParams`, measured
//!   at 0.08 us a node). An input no node reads refuses the capture. The small gathers number
//!   themselves from a count on the card (the device-push gather), so a recorded gather is
//!   the gather of whatever step replays it;
//! - nothing in it waits for the host: a capture that recorded a copy, an allocation or an
//!   event from outside is refused and the span stays launch by launch.
//!
//! SELF-CHECK. The first [`SELF_CHECKS`] replays of a span run it both ways, launch by launch on
//! the step's inputs and as the graph on copies of them (so every input's address is one the
//! capture never saw), and compare what each wrote, byte for byte. A difference fails the step.
//!
//! WHAT IS CAPTURED TODAY: the half of a layer that holds no position (the post-mix of the
//! attention output, the FFN's mixes, norm, router, plan, expert tiles, shared expert, the MoE's
//! gathers, combine and post-mix), for calls of at most 8 rows. The attention half takes the
//! position as launch arguments and stays launch by launch until its entry reads them from the
//! card.

use super::tp::{CommStats, Group};
use super::{Buf, Ctx, Scratch, Side, TmpPool};
use crate::graph_update::KernelNode;
use cudarc::driver::{CudaGraph, sys};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Replays of a span that are compared with the launch-by-launch run before it is trusted.
pub const SELF_CHECKS: u32 = 2;

/// Whether `MEMRA_DSV41_DECODE_GRAPH` asks for the graph (`1`; unset or `0`: off).
pub fn graph_env() -> Result<bool, String> {
    match std::env::var("MEMRA_DSV41_DECODE_GRAPH").ok().as_deref() {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(v) => Err(format!("MEMRA_DSV41_DECODE_GRAPH={v}: 0 or 1")),
    }
}

impl TmpPool {
    /// A capture's free list: see the field.
    fn capture() -> Arc<Self> {
        Arc::new(Self {
            capture: true,
            ..Self::default()
        })
    }
}

/// What a capture swapped out of a [`Ctx`]: the free lists and scratch of its two streams.
pub struct Saved {
    pool: Arc<TmpPool>,
    scr: Scratch,
    side_pool: Arc<TmpPool>,
    side_scr: Scratch,
}

/// What a rank's graph reads and writes besides its inputs: every buffer its launches took.
pub struct Keep {
    _pools: [Arc<TmpPool>; 2],
    _scr: [Scratch; 2],
}

/// A rank's instantiated graph. It is launched only from the rank's issue thread, on the stream
/// it was captured from.
struct RankGraph(CudaGraph);
// SAFETY: the handles are bound to the rank's context, not to a thread; one thread at a time
// launches a rank's graph ([`Group::replay`] runs inside a section of the rank's issue thread)
unsafe impl Send for RankGraph {}
unsafe impl Sync for RankGraph {}

impl Ctx {
    /// Start recording: what this rank issues on its two streams from here becomes a graph.
    /// Everything issued before has run (the capture does not wait behind it), and the streams'
    /// temporaries come from the capture's own lists until [`Self::capture_end`].
    fn capture_begin(&mut self) -> Result<Saved, String> {
        use cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;
        let st = self.ops.st.clone();
        if self.side.is_none() {
            self.side = Some(Side::new(&st)?);
        }
        let side = self.side.as_mut().unwrap();
        side.ops.sync()?;
        self.ops.sync()?;
        let saved = Saved {
            pool: std::mem::replace(&mut self.ops.pool, TmpPool::capture()),
            scr: std::mem::take(&mut self.scr),
            side_pool: std::mem::replace(&mut side.ops.pool, TmpPool::capture()),
            side_scr: std::mem::take(&mut side.scr),
        };
        // relaxed: an allocation outside the stream's order is allowed while it records
        if let Err(e) = st.begin_capture(CU_STREAM_CAPTURE_MODE_RELAXED) {
            self.capture_restore(saved);
            return Err(format!("begin capture: {e:?}"));
        }
        Ok(saved)
    }

    /// Put the streams' own lists and scratch back; what the capture used comes out.
    fn capture_restore(&mut self, saved: Saved) -> Keep {
        let side = self
            .side
            .as_mut()
            .expect("the capture made the side stream");
        Keep {
            _pools: [
                std::mem::replace(&mut self.ops.pool, saved.pool),
                std::mem::replace(&mut side.ops.pool, saved.side_pool),
            ],
            _scr: [
                std::mem::replace(&mut self.scr, saved.scr),
                std::mem::replace(&mut side.scr, saved.side_scr),
            ],
        }
    }

    /// End the recording and instantiate the graph. `Ok(None)`: the recording is not a graph
    /// this module replays (it holds a node that is not a kernel or a zero fill); the caller
    /// stays launch by launch.
    fn capture_end(&mut self, saved: Saved) -> Result<Option<(RankGraph, Keep, usize)>, String> {
        use cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        let ended = self
            .ops
            .st
            .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        let keep = self.capture_restore(saved);
        let graph = ended
            .map_err(|e| format!("end capture: {e:?}"))?
            .ok_or("end capture: the stream was not recording")?;
        let census =
            crate::graph_update::node_census(&graph).map_err(|e| format!("graph nodes: {e}"))?;
        let mut kernels = 0;
        for (ty, n) in &census {
            match ty.as_str() {
                "CU_GRAPH_NODE_TYPE_KERNEL" => kernels = *n,
                "CU_GRAPH_NODE_TYPE_MEMSET" | "CU_GRAPH_NODE_TYPE_EMPTY" => {}
                _ => {
                    eprintln!("[dsv41-graph] capture refused: {n} nodes of {ty}");
                    return Ok(None);
                }
            }
        }
        graph.upload().map_err(|e| format!("graph upload: {e:?}"))?;
        Ok(Some((RankGraph(graph), keep, kernels)))
    }
}

/// More arguments than any kernel here takes: where the walk of a node's arguments stops.
const MAX_ARGS: usize = 64;

/// Where a rank's graph holds its inputs' addresses.
struct Binds {
    /// the kernel nodes with at least one such argument
    nodes: Vec<KernelNode>,
    /// per input: the address the instantiated graph holds now
    cur: Vec<u64>,
    /// per input: (node, the pointer-sized word of that node's argument staging, the offset
    /// into the input it points at)
    words: Vec<Vec<(usize, *mut u64, u64)>>,
}
// SAFETY: the node handles and their argument staging are the driver's, valid for the graph's
// life and bound to the rank's context, not to a thread; a rank's `Binds` sits in a mutex
unsafe impl Send for Binds {}

impl Binds {
    /// Every argument word of `graph`'s kernel nodes that holds an address inside one of
    /// `ins` (address, bytes). `Err`: an input no node reads, so the capture cannot follow it.
    fn find(graph: &CudaGraph, ins: &[(u64, usize)]) -> Result<Self, String> {
        let all = crate::graph_update::kernel_nodes(graph).map_err(|e| format!("nodes: {e}"))?;
        let mut nodes = Vec::new();
        let mut words: Vec<Vec<(usize, *mut u64, u64)>> = ins.iter().map(|_| Vec::new()).collect();
        for nd in all {
            if nd.params.kernelParams.is_null() {
                return Err(format!(
                    "{}: its arguments are not listed one by one",
                    nd.name
                ));
            }
            let mut hit = false;
            // SAFETY: `func` is the node's function; the staging holds one pointer an
            // argument, each to at least that argument's size. The driver has no count of a
            // function's arguments here, so they are walked until it names no more.
            unsafe {
                for a in 0..MAX_ARGS {
                    let (mut off, mut size) = (0usize, 0usize);
                    if sys::cuFuncGetParamInfo(nd.params.func, a, &mut off, &mut size)
                        != sys::CUresult::CUDA_SUCCESS
                    {
                        break;
                    }
                    let base = *nd.params.kernelParams.add(a) as *mut u8;
                    for k in 0..size / 8 {
                        let w = base.add(k * 8) as *mut u64;
                        let v = w.read_unaligned();
                        for (i, &(addr, len)) in ins.iter().enumerate() {
                            if v >= addr && v < addr + len.max(1) as u64 {
                                words[i].push((nodes.len(), w, v - addr));
                                hit = true;
                            }
                        }
                    }
                }
            }
            if hit {
                nodes.push(nd);
            }
        }
        if let Some(i) = words.iter().position(|w| w.is_empty()) {
            return Err(format!("input {i} is read by no node"));
        }
        Ok(Self {
            nodes,
            cur: ins.iter().map(|x| x.0).collect(),
            words,
        })
    }

    /// Point the graph at `ins` (one address an input); returns whether anything moved.
    fn bind(&mut self, graph: &CudaGraph, ins: &[u64]) -> Result<bool, String> {
        if ins.len() != self.cur.len() {
            return Err("dsv41 graph: a replay names another number of inputs".into());
        }
        let mut dirty = vec![false; self.nodes.len()];
        for (i, &a) in ins.iter().enumerate() {
            if a == self.cur[i] {
                continue;
            }
            for &(nd, w, off) in &self.words[i] {
                // SAFETY: `w` is a word of the node's own argument staging (see `find`)
                unsafe { w.write_unaligned(a + off) };
                dirty[nd] = true;
            }
            self.cur[i] = a;
        }
        let mut moved = false;
        for (nd, d) in self.nodes.iter().zip(dirty) {
            if d {
                moved = true;
                // SAFETY: the node and its parameters were read from this graph
                unsafe { crate::graph_update::set_exec_params(graph, nd.node, &nd.params) }
                    .map_err(|e| format!("node update: {e}"))?;
            }
        }
        Ok(moved)
    }
}

/// A captured span on every rank, with what its closure returned.
pub struct Capture<T> {
    graphs: Vec<RankGraph>,
    binds: Vec<Mutex<Binds>>,
    _keep: Vec<Keep>,
    /// the closure's result: buffers of the capture's own lists, rewritten by every replay
    pub out: T,
    /// kernel nodes of rank 0's graph
    pub kernels: usize,
    /// what the span's gathers add to the group's counters
    stats: CommStats,
}

impl Group {
    /// Whether this group's launches can be recorded at all: no host read sits inside a span
    /// (trace, timed collectives) and the gathers are the in-kernel ones.
    /// Whether a span of this group's launches can be recorded: no trace and no timing, the
    /// gathers inside kernels, and nothing in the half-layer that waits for the host: an expert
    /// cache resolves its misses on the host from the plan it reads back, and a route log reads
    /// each block's routing back. A group that cannot record keeps every span launch by launch.
    pub fn can_capture(&self) -> bool {
        !self.trace
            && !self.timed
            && self.route_log.is_none()
            && self.cx.iter().all(|c| c.experts.is_none())
            && (self.n() == 1 || self.transport == crate::tp_transport::TpTransport::DevicePush)
    }

    /// Record what `body` issues on every rank as one graph a rank; nothing executes.
    /// `ins[r]`: rank r's input buffers (address, bytes), the ones a later replay may name at
    /// other addresses. `Ok(None)`: a rank's recording was refused ([`Ctx::capture_end`], or an
    /// input none of its launches reads).
    pub fn capture<T>(
        &mut self,
        ins: &[Vec<(u64, usize)>],
        body: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<Option<Capture<T>>, String> {
        if ins.len() != self.n() {
            return Err("dsv41 graph: one input list per rank".into());
        }
        if !self.can_capture() {
            return Err("dsv41 graph: this group's launches cannot be recorded".into());
        }
        let n = self.n();
        let mut saved: Vec<Option<Saved>> = (0..n).map(|_| None).collect();
        let began = self.each_mut(&mut saved, |_, c, s| {
            *s = Some(c.capture_begin()?);
            Ok(())
        });
        let s0 = self.stats;
        self.capturing = true;
        let out = match began {
            Ok(_) => body(self),
            Err(e) => Err(e),
        };
        self.capturing = false;
        // every rank that began ends, whatever happened, or its stream stays recording
        let ended = self.each_mut(&mut saved, |_, c, s| match s.take() {
            Some(s) => c.capture_end(s),
            None => Ok(None),
        });
        let s1 = std::mem::replace(&mut self.stats, s0);
        let out = out?;
        let mut graphs = Vec::with_capacity(n);
        let mut binds = Vec::with_capacity(n);
        let mut keep = Vec::with_capacity(n);
        let mut kernels = 0;
        for (r, e) in ended?.into_iter().enumerate() {
            let Some((g, k, nk)) = e else {
                return Ok(None);
            };
            if r == 0 {
                kernels = nk;
            }
            let _ = self.cx[r].ops.sv();
            match Binds::find(&g.0, &ins[r]) {
                Ok(b) => binds.push(Mutex::new(b)),
                Err(e) => {
                    eprintln!("[dsv41-graph] capture refused: rank {r}: {e}");
                    return Ok(None);
                }
            }
            graphs.push(g);
            keep.push(k);
        }
        Ok(Some(Capture {
            graphs,
            binds,
            _keep: keep,
            out,
            kernels,
            stats: CommStats {
                collectives: s1.collectives - s0.collectives,
                copies: s1.copies - s0.copies,
                bytes: s1.bytes - s0.bytes,
                secs: 0.0,
            },
        }))
    }

    /// Launch every rank's graph of `c` on its stream, on the inputs `ins[r]` (addresses, in
    /// the capture's order). Returns whether a rank's inputs had moved since its last launch.
    pub fn replay<T>(&mut self, c: &Capture<T>, ins: &[Vec<u64>]) -> Result<bool, String> {
        if ins.len() != self.n() {
            return Err("dsv41 graph: one input list per rank".into());
        }
        let (graphs, binds) = (&c.graphs, &c.binds);
        let moved = self.each(|r, _| {
            let g = &graphs[r].0;
            let moved = binds[r]
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .bind(g, &ins[r])?;
            g.launch().map_err(|e| format!("graph launch: {e:?}"))?;
            Ok(moved)
        })?;
        self.stats.collectives += c.stats.collectives;
        self.stats.copies += c.stats.copies;
        self.stats.bytes += c.stats.bytes;
        Ok(moved.contains(&true))
    }
}

/// A span's captured form.
pub struct Piece {
    pub cap: Capture<Vec<Buf>>,
    /// replays compared with the launch-by-launch run so far
    pub checked: u32,
}

/// A span's state in [`Graphs`].
pub enum Slot {
    /// run once launch by launch (its lazy state exists); the next call records it
    Seen,
    Ready(Box<Piece>),
    /// its recording was refused: launch by launch from here on
    Refused,
}

/// What the door did (printed by the benches).
#[derive(Debug, Default, Clone, Copy)]
pub struct GraphStats {
    pub captures: u64,
    pub replays: u64,
    /// replays whose inputs sat at other addresses than the launch before: retuned
    pub moved: u64,
    pub refused: u64,
    pub checks: u64,
}

/// The most spans a group keeps: every resident layer of a trunk and a draft at each of the 8
/// row counts is 400. Past it the map is emptied (a layer that is loaded for every call names a
/// new span each time and would grow it without end).
pub const MAX_SPANS: usize = 1024;

/// The group's captured spans, by (the layer's load on rank 0, rows).
#[derive(Default)]
pub struct Graphs {
    pub on: bool,
    pub slots: HashMap<(u64, usize), Slot>,
    pub stats: GraphStats,
}

impl Graphs {
    pub fn from_env() -> Result<Self, String> {
        let mut g = Self::default();
        g.on = graph_env()?;
        Ok(g)
    }
}

impl Drop for Graphs {
    fn drop(&mut self) {
        if self.on {
            let s = self.stats;
            eprintln!(
                "[dsv41-graph] spans={} captures={} replays={} checked={} moved={} refused={}",
                self.slots.len(),
                s.captures,
                s.replays,
                s.checks,
                s.moved,
                s.refused
            );
        }
    }
}

impl<T> Capture<T> {
    /// Rank `r`'s graph, for node inspection and parameter updates ([`crate::graph_update`]).
    pub fn graph(&self, r: usize) -> &CudaGraph {
        &self.graphs[r].0
    }
}

impl Group {
    /// What a launch costs the host and the card three ways, on every rank at once: `k` RMSNorm
    /// launches of one 5,120-wide row issued one by one, the same `k` as one graph launch, and
    /// `k` parameter updates of the graph's nodes (`cuGraphExecKernelNodeSetParams`, the way a
    /// captured launch takes a new argument or grid). Medians over `reps`, microseconds.
    pub fn graph_probe(&mut self, k: usize, reps: usize) -> Result<Vec<String>, String> {
        use std::time::Instant;
        let d = self.cx[0].g.dim;
        let eps = self.cx[0].g.norm_eps;
        let bufs = self.each(|_, c| {
            let x = Buf::from_bytes(&c.ops.st, &vec![0x3fu8; d * 2])?;
            let w = Buf::from_bytes(&c.ops.st, &vec![0x3fu8; d * 2])?;
            Ok((x, w, c.ops.buf(d * 2)?))
        })?;
        let issue = |g: &mut Self| {
            g.each(|r, c| {
                let st = c.ops.st.clone();
                let (x, w, y) = &bufs[r];
                for _ in 0..k {
                    c.ops.rmsnorm(x.p(&st), d, w, y.p(&st), d, 1, d, eps)?;
                }
                Ok(())
            })
            .map(|_| ())
        };
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };
        let none: Vec<Vec<(u64, usize)>> = (0..self.n()).map(|_| Vec::new()).collect();
        let cap = self
            .capture(&none, issue)?
            .ok_or("graph probe: the capture was refused")?;
        let same: Vec<Vec<u64>> = (0..self.n()).map(|_| Vec::new()).collect();
        let (mut ei, mut ew, mut gi, mut gw, mut up) = (vec![], vec![], vec![], vec![], vec![]);
        for _ in 0..reps.max(1) {
            self.sync_all()?;
            let t = Instant::now();
            issue(self)?;
            ei.push(t.elapsed().as_secs_f64() * 1e6);
            self.sync_all()?;
            ew.push(t.elapsed().as_secs_f64() * 1e6);
            let t = Instant::now();
            self.replay(&cap, &same)?;
            gi.push(t.elapsed().as_secs_f64() * 1e6);
            self.sync_all()?;
            gw.push(t.elapsed().as_secs_f64() * 1e6);
            // rank 0's nodes, each given its own parameters again
            let nodes = crate::graph_update::kernel_nodes(cap.graph(0))
                .map_err(|e| format!("graph nodes: {e}"))?;
            let _ = self.cx[0].ops.sv();
            let t = Instant::now();
            for nd in &nodes {
                // SAFETY: the node and its parameters were just read from this graph
                unsafe { crate::graph_update::set_exec_params(cap.graph(0), nd.node, &nd.params) }
                    .map_err(|e| format!("node update: {e}"))?;
            }
            up.push(t.elapsed().as_secs_f64() * 1e6 / nodes.len().max(1) as f64);
        }
        let n = self.n();
        Ok(vec![
            format!(
                "graph probe: {n} ranks, {k} launches a rank, {} kernel nodes in rank 0's graph",
                cap.kernels
            ),
            format!(
                "one by one: issue {:.1} us ({:.2} a launch a rank), done {:.1} us ({:.2} a launch)",
                med(&mut ei),
                med(&mut ei) / k as f64,
                med(&mut ew),
                med(&mut ew) / k as f64
            ),
            format!(
                "one graph: launch call {:.1} us ({:.3} a node a rank), done {:.1} us ({:.2} a node)",
                med(&mut gi),
                med(&mut gi) / k as f64,
                med(&mut gw),
                med(&mut gw) / k as f64
            ),
            format!("node parameter update: {:.3} us a node", med(&mut up)),
        ])
    }
}
