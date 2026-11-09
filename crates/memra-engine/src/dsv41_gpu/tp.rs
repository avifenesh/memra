//! DeepSeek-V4.1 on N ranks: tensor-parallel attention, expert-parallel MoE and a vocab-parallel
//! head over the single-card kernels, generic in N.
//!
//! The partition ([`Part`]): each rank holds `heads / n` attention heads (`wq_b` rows, sinks), the
//! `o_groups / n` output groups those heads feed (`wo_a` rows), `dim / n` rows of `wo_b`,
//! `n_experts / n` routed experts and `vocab / n` head rows. Everything else is replicated: the mHC
//! mixes on the full stream, `wq_a`, the single KV head (`wkv`), the compressor, the indexer and its
//! keys, the router, the shared expert and Engram (host-resident table, gathered per rank). Every
//! rank therefore holds the full residual stream and the full CSA2 state. At prefill the indexer's
//! query rows split across the ranks (all heads, replicated weights) and the selection rows are
//! gathered; the candidate pool is not (each rank's later index layers select only its own rows,
//! so it keeps only those, [`super::Shared::cand_row0`]); a decode row is scored on every rank.
//!
//! The collectives are gathers only, so every value is computed by exactly the arithmetic of the
//! single-card program, in its order:
//!   - attention: each rank writes its groups' `wo_a` output columns, all-gather those columns, each
//!     rank computes its `wo_b` output columns over the full row (the K sum of every output element
//!     is the single-card sum), all-gather the output columns;
//!   - MoE: every rank runs the router on its own replicated copy and builds the same host
//!     dispatch plan from its read-back (pairs sorted by expert, tokens ascending inside an expert,
//!     as the per-expert loop orders them; a rank whose routing differs refuses); each rank runs its
//!     experts' pairs through one grouped GEMM per projection, all-gather the bf16 expert output
//!     rows, and every rank adds each token's rows in ascending expert id into an fp32 zero, then
//!     the shared expert, then rounds (the per-expert loop's scatter-add order);
//!   - indexer at prefill: each rank selects its query rows, all-gather the selection rows;
//!   - head: each rank's vocab rows, gathered on the host.
//!
//! No sum is split across ranks, so the N-rank program writes the single-card bytes: the gates
//! compare bytes, not a band. A head-split indexer would split the index score's head sum; it is
//! a measurement arm here ([`Group::index_split_bench`]), not the program.
//!
//! Transport: the fleet flag `MEMRA_TP_TRANSPORT` ([`crate::tp_transport`]), with its laws: the
//! peer-pull arm copies on the CONSUMER's stream after a publication event recorded on the
//! producer's stream, and the producer waits on a release event recorded after the consumer's copy
//! before it touches or frees the buffer again; no atomics, no peer flag polling; a byte-integrity
//! ladder through the exact gather primitive in every ordered pair before any layer runs. The
//! host-canonical arm (the default on this build, as for every TP walk without a receipt for the
//! peer arm on its card) bounces each block through host memory.
//!
//! Issue: between collectives each rank's work runs on its own host thread ([`Group::each`],
//! `MEMRA_DSV41_TP_ISSUE`), and nothing waits for a device except the routing read-back (the host
//! builds the dispatch plan), Engram's staging refill and the head's logits. Buffers go back to
//! the stream-ordered pool on their own stream; a buffer a peer copied out of is freed only after
//! the producer's stream waited on that copy's release event. cudarc's per-buffer event tracking
//! is off on the ranks' contexts for the same reason (`Group::new`).

// Rank-indexed loops walk several per-rank vectors (contexts, buffers, shares) in step; an index is
// the clearer shape there, as in `tp_transport.rs`.
#![allow(clippy::needless_range_loop)]

use super::{
    BW, Buf, Ctx, Dp, Geo, LayerState, LayerW, Part, Shared, Trace, bytes_f32, bytes_i32, ck, ffi,
    load_bf16_rows, load_vec_bf16,
};
use crate::Engine;
use crate::tp_transport::{self, TpTransport};
use cudarc::driver::CudaEvent;
use memra_gguf::safetensors::StModel;
use rayon::prelude::*;
use std::time::Instant;

/// A byte region of a rank buffer: `rows` rows of `width` bytes, `pitch` apart, from `off`. A
/// gather copies a producer's region into the same region of every other rank's buffer.
#[derive(Debug, Clone, Copy)]
pub struct Blk {
    pub off: usize,
    pub width: usize,
    pub pitch: usize,
    pub rows: usize,
}

impl Blk {
    pub fn flat(off: usize, len: usize) -> Self {
        Self {
            off,
            width: len,
            pitch: len.max(1),
            rows: 1,
        }
    }
    pub fn bytes(&self) -> usize {
        self.width * self.rows
    }
    fn end(&self) -> usize {
        if self.rows == 0 {
            self.off
        } else {
            self.off + (self.rows - 1) * self.pitch + self.width
        }
    }
}

/// Host nanoseconds spent blocked on a device read-back (the routing plan, the head's logits),
/// summed over the ranks: the share of a step the host waited for the cards rather than issuing.
pub static READBACK_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `b.host()` with its blocked time counted in [`READBACK_WAIT_NS`].
fn host_timed(b: &Buf, st: &std::sync::Arc<cudarc::driver::CudaStream>) -> Result<Vec<u8>, String> {
    let t = Instant::now();
    let v = b.host(st);
    READBACK_WAIT_NS.fetch_add(
        t.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    v
}

/// What the collectives moved (per process, all ranks).
#[derive(Debug, Default, Clone, Copy)]
pub struct CommStats {
    pub collectives: u64,
    pub copies: u64,
    pub bytes: u64,
    /// wall time inside timed collectives (`Group::timed`)
    pub secs: f64,
}

/// N ranks, one engine and one [`Ctx`] each (rank 0 first). Ranks may share a device (the code
/// path over N streams on one card: proves the partition, not the fabric).
pub struct Group {
    pub devices: Vec<usize>,
    pub engines: Vec<Engine>,
    pub cx: Vec<Ctx>,
    pub transport: TpTransport,
    pub transport_flag: &'static str,
    pub_ev: Vec<CudaEvent>,
    rel_ev: Vec<CudaEvent>,
    pub stats: CommStats,
    /// synchronize and time every collective (measurement runs)
    pub timed: bool,
    /// keep `attn_out`, `moe_out` and the routing in each rank's [`Trace`]
    pub trace: bool,
    /// one host thread per rank issues that rank's work between collectives (None: this thread
    /// issues the ranks in turn, `MEMRA_DSV41_TP_ISSUE=serial`)
    pool: Option<rayon::ThreadPool>,
}

/// How the host issues the ranks' work: one thread per rank (the default at N > 1) or this thread
/// for every rank in turn (`MEMRA_DSV41_TP_ISSUE=serial`, the measurement and rollback seam). The
/// streams, the kernels and the collectives are the same in both, so are the bytes.
pub const ISSUE_ENV: &str = "MEMRA_DSV41_TP_ISSUE";

/// [`Group::each`] over a pool and the rank contexts, for callers that also borrow other fields.
fn each_in<T, F>(pool: Option<&rayon::ThreadPool>, cx: &mut [Ctx], f: F) -> Result<Vec<T>, String>
where
    T: Send,
    F: Fn(usize, &mut Ctx) -> Result<T, String> + Sync + Send,
{
    match pool {
        Some(p) => p.install(|| {
            cx.par_iter_mut()
                .enumerate()
                .map(|(r, c)| f(r, c))
                .collect()
        }),
        None => cx.iter_mut().enumerate().map(|(r, c)| f(r, c)).collect(),
    }
}

fn issue_parallel() -> Result<bool, String> {
    match std::env::var(ISSUE_ENV).ok().as_deref() {
        None | Some("") | Some("parallel") => Ok(true),
        Some("serial") => Ok(false),
        Some(x) => Err(format!("{ISSUE_ENV}={x:?}: want parallel or serial")),
    }
}

fn de(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}

/// Device-to-device copy of `b` from `src` into the same region of `dst`, on `ops`'s stream.
fn copy_blk(ops: &super::Ops, src: &Buf, dst: &Buf, b: Blk) -> Result<(), String> {
    copy_addr(ops, src.p(&ops.st).addr(), dst.p(&ops.st).addr(), b)
}

/// [`copy_blk`] between two device base addresses.
fn copy_addr(ops: &super::Ops, src: u64, dst: u64, b: Blk) -> Result<(), String> {
    use cudarc::driver::sys::*;
    if b.bytes() == 0 {
        return Ok(());
    }
    let s0 = src + b.off as u64;
    let d0 = dst + b.off as u64;
    let st = ops.sv() as CUstream;
    let r = if b.rows == 1 {
        unsafe { cuMemcpyDtoDAsync_v2(d0, s0, b.width, st) }
    } else {
        let p = CUDA_MEMCPY2D {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcHost: std::ptr::null(),
            srcDevice: s0,
            srcArray: std::ptr::null_mut(),
            srcPitch: b.pitch,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstHost: std::ptr::null_mut(),
            dstDevice: d0,
            dstArray: std::ptr::null_mut(),
            dstPitch: b.pitch,
            WidthInBytes: b.width,
            Height: b.rows,
        };
        unsafe { cuMemcpy2DAsync_v2(&p, st) }
    };
    if r != CUresult::CUDA_SUCCESS {
        return Err(format!("dsv41 tp copy: {r:?}"));
    }
    Ok(())
}

/// Region `b` of `src` into packed host bytes (`width` per row); drains the stream.
fn dtoh_blk(ops: &super::Ops, src: &Buf, b: Blk) -> Result<Vec<u8>, String> {
    use cudarc::driver::sys::*;
    let mut host = vec![0u8; b.bytes()];
    let p = CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: src.p(&ops.st).addr() + b.off as u64,
        srcArray: std::ptr::null_mut(),
        srcPitch: b.pitch,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
        dstHost: host.as_mut_ptr() as *mut std::ffi::c_void,
        dstDevice: 0,
        dstArray: std::ptr::null_mut(),
        dstPitch: b.width,
        WidthInBytes: b.width,
        Height: b.rows,
    };
    let r = unsafe { cuMemcpy2DAsync_v2(&p, ops.sv() as CUstream) };
    if r != CUresult::CUDA_SUCCESS {
        return Err(format!("dsv41 tp dtoh: {r:?}"));
    }
    ops.sync()?;
    Ok(host)
}

/// Packed host bytes into region `b` of `dst`, queued on `ops`'s stream (the caller keeps `host`
/// alive until that stream drains).
fn htod_blk(ops: &super::Ops, host: &[u8], dst: &Buf, b: Blk) -> Result<(), String> {
    use cudarc::driver::sys::*;
    let p = CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
        srcHost: host.as_ptr() as *const std::ffi::c_void,
        srcDevice: 0,
        srcArray: std::ptr::null_mut(),
        srcPitch: b.width,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: dst.p(&ops.st).addr() + b.off as u64,
        dstArray: std::ptr::null_mut(),
        dstPitch: b.pitch,
        WidthInBytes: b.width,
        Height: b.rows,
    };
    let r = unsafe { cuMemcpy2DAsync_v2(&p, ops.sv() as CUstream) };
    if r != CUresult::CUDA_SUCCESS {
        return Err(format!("dsv41 tp htod: {r:?}"));
    }
    Ok(())
}

impl Group {
    /// Ranks on `devices` (rank r on `devices[r]`), each with a [`Ctx`] covering `max_pos`.
    pub fn new(devices: &[usize], g: &Geo, max_pos: usize) -> Result<Self, String> {
        let n = devices.len();
        Part::check(g, n)?;
        let (transport, flag) = tp_transport::transport_env()?;
        if transport == TpTransport::DevicePush {
            return Err(format!(
                "{flag}=device-push: the dsv41 route moves its blocks with host-canonical or peer-pull"
            ));
        }
        let parallel = issue_parallel()?;
        let mut engines = Vec::with_capacity(n);
        for &d in devices {
            let e = Engine::new(d).map_err(|e| format!("rank on device {d}: {e}"))?;
            // Every rank buffer lives on its rank's one stream, and the only cross-stream uses
            // are the gathers, which order themselves with the publication and release events
            // (the transport laws above). cudarc's per-buffer event tracking would add an event
            // record and a stream wait to every pointer fetch on the hot path, so the buffers
            // allocated from here on go untracked.
            unsafe { e.ctx().disable_event_tracking() };
            engines.push(e);
        }
        let distinct = |a: usize, b: usize| devices[a] != devices[b];
        if transport == TpTransport::PeerPull {
            for a in 0..n {
                for b in 0..n {
                    if a != b && distinct(a, b) {
                        crate::tp::grant_peer_access(&engines[a], &engines[b], "dsv41 tp")
                            .map_err(|e| e.to_string())?;
                    }
                }
            }
        }
        let mut cx = Vec::with_capacity(n);
        let mut pub_ev = Vec::with_capacity(n);
        let mut rel_ev = Vec::with_capacity(n);
        for e in &engines {
            let st = e.stream();
            pub_ev.push(st.context().new_event(None).map_err(de)?);
            rel_ev.push(st.context().new_event(None).map_err(de)?);
            cx.push(Ctx::new(st, g.clone(), max_pos)?);
        }
        let pool = if n > 1 && parallel {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .thread_name(|i| format!("dsv41-rank{i}"))
                    .build()
                    .map_err(|e| format!("dsv41 tp issue threads: {e}"))?,
            )
        } else {
            None
        };
        let mut gr = Self {
            devices: devices.to_vec(),
            engines,
            cx,
            transport,
            transport_flag: flag,
            pub_ev,
            rel_ev,
            stats: CommStats::default(),
            timed: false,
            trace: false,
            pool,
        };
        gr.probe()?;
        gr.stats = CommStats::default();
        eprintln!(
            "[dsv41-tp] ranks={n} devices={devices:?} transport={} ({flag}) issue={} gathers-only=true",
            transport.name(),
            gr.issue_name()
        );
        Ok(gr)
    }

    pub fn n(&self) -> usize {
        self.cx.len()
    }

    /// `parallel` (one host thread per rank) or `serial`.
    pub fn issue_name(&self) -> &'static str {
        if self.pool.is_some() {
            "parallel"
        } else {
            "serial"
        }
    }

    pub fn sync_all(&self) -> Result<(), String> {
        for c in &self.cx {
            c.ops.sync()?;
        }
        Ok(())
    }

    /// `f(r, rank r's Ctx)` for every rank, each rank on its own host thread (or in turn under
    /// `MEMRA_DSV41_TP_ISSUE=serial`). Results in rank order. A rank's work only queues on its own
    /// stream, so the order the host issues the ranks in never changes a byte.
    pub fn each<T, F>(&mut self, f: F) -> Result<Vec<T>, String>
    where
        T: Send,
        F: Fn(usize, &mut Ctx) -> Result<T, String> + Sync + Send,
    {
        let Self { pool, cx, .. } = self;
        each_in(pool.as_ref(), cx, f)
    }

    /// [`Self::each`] with one per-rank mutable item `us[r]` handed to rank r.
    pub fn each_mut<U, T, F>(&mut self, us: &mut [U], f: F) -> Result<Vec<T>, String>
    where
        U: Send,
        T: Send,
        F: Fn(usize, &mut Ctx, &mut U) -> Result<T, String> + Sync + Send,
    {
        if us.len() != self.cx.len() {
            return Err(format!(
                "dsv41 tp: {} per-rank items for {} ranks",
                us.len(),
                self.cx.len()
            ));
        }
        let Self { pool, cx, .. } = self;
        match pool {
            Some(p) => p.install(|| {
                cx.par_iter_mut()
                    .zip(us.par_iter_mut())
                    .enumerate()
                    .map(|(r, (c, u))| f(r, c, u))
                    .collect()
            }),
            None => cx
                .iter_mut()
                .zip(us.iter_mut())
                .enumerate()
                .map(|(r, (c, u))| f(r, c, u))
                .collect(),
        }
    }

    /// The byte-integrity ladder: every rank's block, flat and strided, at three sizes, through
    /// [`Self::allgather`] into every other rank. One differing byte refuses the group.
    fn probe(&mut self) -> Result<(), String> {
        let n = self.n();
        if n == 1 {
            return Ok(());
        }
        for &(width, rows) in &[(4096usize, 1usize), (2 * 1024 * 1024, 1), (5120, 33)] {
            let pitch = width * n + 64;
            let len = pitch * rows;
            let pat = |r: usize, i: usize| -> u8 {
                ((i as u32).wrapping_mul(0x9e37_79b9) >> 11) as u8 ^ (r as u8).wrapping_mul(37)
            };
            let blk = |r: usize| Blk {
                off: r * width,
                width,
                pitch,
                rows,
            };
            let mut bufs = Vec::with_capacity(n);
            for r in 0..n {
                let mut host = vec![0xA5u8; len];
                for row in 0..rows {
                    for i in 0..width {
                        host[row * pitch + r * width + i] = pat(r, row * width + i);
                    }
                }
                bufs.push(Buf::from_bytes(&self.cx[r].ops.st, &host)?);
            }
            let refs: Vec<&Buf> = bufs.iter().collect();
            let blks: Vec<Blk> = (0..n).map(blk).collect();
            self.allgather(&refs, &blks)?;
            self.sync_all()?;
            for c in 0..n {
                let got = bufs[c].host(&self.cx[c].ops.st)?;
                for p in 0..n {
                    for row in 0..rows {
                        for i in 0..width {
                            if got[row * pitch + p * width + i] != pat(p, row * width + i) {
                                return Err(format!(
                                    "dsv41 tp {} probe: rank {p} -> rank {c} differs at {width} x {rows}",
                                    self.transport.name()
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// All-gather in place: rank p produced region `blks[p]` of `bufs[p]`; afterwards every rank's
    /// buffer holds every region. Pure movement.
    pub fn allgather(&mut self, bufs: &[&Buf], blks: &[Blk]) -> Result<(), String> {
        let n = self.n();
        if n == 1 {
            return Ok(());
        }
        if bufs.len() != n || blks.len() != n {
            return Err("allgather: one buffer and one region per rank".into());
        }
        for b in blks {
            if bufs.iter().any(|x| b.end() > x.len) || (b.rows > 1 && b.width > b.pitch) {
                return Err(format!("allgather: region {b:?} outside a rank buffer"));
            }
        }
        let t0 = if self.timed {
            self.sync_all()?;
            Some(Instant::now())
        } else {
            None
        };
        match self.transport {
            TpTransport::PeerPull if self.pool.is_some() => {
                // the same three steps with each rank's calls on its own thread: every producer
                // publishes, then every consumer pulls on its own stream and releases, then every
                // producer waits for the releases (each step completes on the host before the next,
                // so a wait always sees this gather's record)
                let addr: Vec<u64> = (0..n)
                    .map(|r| bufs[r].p(&self.cx[r].ops.st).addr())
                    .collect();
                let Self {
                    pool,
                    cx,
                    pub_ev,
                    rel_ev,
                    ..
                } = self;
                let (pool, pub_ev, rel_ev) = (pool.as_ref(), &*pub_ev, &*rel_ev);
                each_in(pool, cx, |r, c| pub_ev[r].record(&c.ops.st).map_err(de))?;
                let copies = each_in(pool, cx, |c, x| {
                    let mut k = 0u64;
                    for p in 0..n {
                        if p == c || blks[p].bytes() == 0 {
                            continue;
                        }
                        x.ops.st.wait(&pub_ev[p]).map_err(de)?;
                        copy_addr(&x.ops, addr[p], addr[c], blks[p])?;
                        k += 1;
                    }
                    rel_ev[c].record(&x.ops.st).map_err(de)?;
                    Ok(k)
                })?;
                each_in(pool, cx, |p, x| {
                    for c in 0..n {
                        if c != p {
                            x.ops.st.wait(&rel_ev[c]).map_err(de)?;
                        }
                    }
                    Ok(())
                })?;
                self.stats.copies += copies.iter().sum::<u64>();
            }
            TpTransport::PeerPull => {
                for p in 0..n {
                    self.pub_ev[p].record(&self.cx[p].ops.st).map_err(de)?;
                }
                for c in 0..n {
                    for p in 0..n {
                        if p == c || blks[p].bytes() == 0 {
                            continue;
                        }
                        self.cx[c].ops.st.wait(&self.pub_ev[p]).map_err(de)?;
                        copy_blk(&self.cx[c].ops, bufs[p], bufs[c], blks[p])?;
                        self.stats.copies += 1;
                    }
                }
                for c in 0..n {
                    self.rel_ev[c].record(&self.cx[c].ops.st).map_err(de)?;
                }
                for p in 0..n {
                    for c in 0..n {
                        if c != p {
                            self.cx[p].ops.st.wait(&self.rel_ev[c]).map_err(de)?;
                        }
                    }
                }
            }
            TpTransport::HostCanonical => {
                for p in 0..n {
                    if blks[p].bytes() == 0 {
                        continue;
                    }
                    let host = dtoh_blk(&self.cx[p].ops, bufs[p], blks[p])?;
                    for c in 0..n {
                        if c != p {
                            htod_blk(&self.cx[c].ops, &host, bufs[c], blks[p])?;
                            self.stats.copies += 2;
                        }
                    }
                    for c in 0..n {
                        if c != p {
                            self.cx[c].ops.sync()?;
                        }
                    }
                }
            }
            TpTransport::DevicePush => return Err("allgather: device-push is not armed".into()),
        }
        self.stats.collectives += 1;
        self.stats.bytes += blks
            .iter()
            .map(|b| (b.bytes() * (n - 1)) as u64)
            .sum::<u64>();
        if let Some(t) = t0 {
            self.sync_all()?;
            self.stats.secs += t.elapsed().as_secs_f64();
        }
        Ok(())
    }

    /// Every rank's share of layer `id`, the ranks loading in parallel.
    pub fn load_layer(
        &self,
        stm: &StModel,
        g: &Geo,
        id: usize,
        mint: bool,
    ) -> Result<Vec<LayerW>, String> {
        self.load_layer_with(stm, g, id, mint, true)
    }

    /// [`Self::load_layer`]; with `routed` false the shares leave their routed experts to the
    /// ranks' expert caches ([`LayerW::load_part_with`]).
    pub fn load_layer_with(
        &self,
        stm: &StModel,
        g: &Geo,
        id: usize,
        mint: bool,
        routed: bool,
    ) -> Result<Vec<LayerW>, String> {
        let n = self.n();
        let ops: Vec<&super::Ops> = self.cx.iter().map(|c| &c.ops).collect();
        std::thread::scope(|sc| {
            let hs: Vec<_> = ops
                .iter()
                .enumerate()
                .map(|(r, o)| {
                    let o: &super::Ops = o;
                    sc.spawn(move || {
                        LayerW::load_part_with(o, stm, g, id, mint, Part { rank: r, n }, routed)
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| {
                    h.join()
                        .map_err(|_| "layer load thread panicked".to_string())?
                })
                .collect()
        })
    }

    /// One trunk block on every rank (the N-rank [`Ctx::block_forward`]) over borrowed streams:
    /// a copy of each `h[r]` through [`Self::block_forward_owned`], then a drain (the gates time
    /// and read it).
    #[allow(clippy::too_many_arguments)]
    pub fn block_forward(
        &mut self,
        lw: &[LayerW],
        h: &[Buf],
        pre: &[Buf],
        sp: usize,
        s: usize,
        sh: &mut [Shared],
        ls: &mut [LayerState],
        tr: &mut [Trace],
    ) -> Result<(Vec<Buf>, Vec<Buf>), String> {
        if h.len() != self.n() {
            return Err("tp block_forward: one share per rank".into());
        }
        let hs = self.each(|r, c| super::clone_buf(&c.ops, &h[r]))?;
        let out = self.block_forward_owned(lw, hs, pre, sp, s, sh, ls, tr)?;
        self.sync_all()?;
        Ok(out)
    }

    /// One trunk block on every rank: `h[r]`, `pre[r]`, `sh[r]`, `ls[r]` are rank r's replicated
    /// copies. The stream buffers are taken and returned holding the block's output (both mHC
    /// posts write in place, [`Ctx::hc_post_in_place`]), with each rank's new pre-mix. Nothing
    /// here waits for the device except the routing plan's read-back.
    #[allow(clippy::too_many_arguments)]
    pub fn block_forward_owned(
        &mut self,
        lw: &[LayerW],
        h: Vec<Buf>,
        pre: &[Buf],
        sp: usize,
        s: usize,
        sh: &mut [Shared],
        ls: &mut [LayerState],
        tr: &mut [Trace],
    ) -> Result<(Vec<Buf>, Vec<Buf>), String> {
        let n = self.n();
        if lw.len() != n || h.len() != n || pre.len() != n || sh.len() != n || ls.len() != n {
            return Err("tp block_forward: one share per rank".into());
        }
        for (r, w) in lw.iter().enumerate() {
            if w.part != (Part { rank: r, n }) {
                return Err(format!("tp block_forward: rank {r} holds {:?}", w.part));
            }
        }
        let d = self.cx[0].g.dim;
        let mix = self.each(|r, c| {
            let w = &lw[r];
            let (_m, p, po, co) =
                c.hc_mix(&w.hc_attn_fn, &w.hc_attn_scale, &w.hc_attn_base, &h[r], s)?;
            let x = c.hc_pre_norm(&h[r], &pre[r], &w.attn_norm, s)?;
            Ok((p, po, co, x))
        })?;
        let (mut a_pre, mut a_post, mut a_comb, mut xs) = (vec![], vec![], vec![], vec![]);
        for (p, po, co, x) in mix {
            a_pre.push(p);
            a_post.push(po);
            a_comb.push(co);
            xs.push(x);
        }
        let a = self.attention(lw, &xs, sp, s, sh, ls, tr)?;
        drop(xs);
        let ffn = self.each(|r, c| {
            let w = &lw[r];
            // h[r] becomes the stream after attention
            c.hc_post_in_place(&a[r], &h[r], &a_post[r], &a_comb[r], s)?;
            let (_m, p, po, co) =
                c.hc_mix(&w.hc_ffn_fn, &w.hc_ffn_scale, &w.hc_ffn_base, &h[r], s)?;
            let x2 = c.hc_pre_norm(&h[r], &a_pre[r], &w.ffn_norm, s)?;
            let m = c.ops.tmp(s * d * 2)?;
            Ok((p, po, co, x2, m))
        })?;
        drop((a_pre, a_post, a_comb));
        let a = if self.trace { Some(a) } else { None };
        let (mut f_pre, mut f_post, mut f_comb, mut x2, mut m) =
            (vec![], vec![], vec![], vec![], vec![]);
        for (p, po, co, x, mm) in ffn {
            f_pre.push(p);
            f_post.push(po);
            f_comb.push(co);
            x2.push(x);
            m.push(mm);
        }
        let cm = self.cx[0].chunk.min(s).max(1);
        for r0 in (0..s).step_by(cm) {
            let rows = cm.min(s - r0);
            self.moe(lw, &x2, r0, rows, &m, tr)?;
        }
        drop(x2);
        self.each(|r, c| c.hc_post_in_place(&m[r], &h[r], &f_post[r], &f_comb[r], s))?;
        if let Some(a) = a {
            for (r, (aa, mm)) in a.into_iter().zip(m).enumerate() {
                tr[r].attn_out = Some(aa);
                tr[r].moe_out = Some(mm);
            }
        }
        Ok((h, f_pre))
    }

    /// Attention on every rank: the keys side replicated, then per query slice each rank's heads
    /// and output groups, the `wo_a` columns gathered, each rank's `wo_b` columns, the output
    /// columns gathered. Returns each rank's full `[s][dim]` output.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        lw: &[LayerW],
        xs: &[Buf],
        sp: usize,
        s: usize,
        sh: &mut [Shared],
        ls: &mut [LayerState],
        tr: &mut [Trace],
    ) -> Result<Vec<Buf>, String> {
        let n = self.n();
        let g = self.cx[0].g.clone();
        let (d, hd) = (g.dim, g.head_dim);
        // the indexer's query rows split across the ranks at prefill (measured: `dsv41-gate idx`,
        // research/.../tp/README.md); one decode row is scored on every rank, with nothing to move
        let split = n > 1 && s >= 2 * n && lw[0].index_source && lw[0].ratio > 0;
        let per = s.div_ceil(n);
        let span = |r: usize| (r * per).min(s)..((r + 1) * per).min(s);
        let keys = {
            let mut io: Vec<(&mut Shared, &mut LayerState, &mut Trace)> = sh
                .iter_mut()
                .zip(ls.iter_mut())
                .zip(tr.iter_mut())
                .map(|((a, b), c)| (a, b, c))
                .collect();
            self.each_mut(&mut io, |r, c, (shr, lsr, trr)| {
                c.index_rows = split.then(|| span(r));
                let k = c.attn_keys(&lw[r], &xs[r], sp, s, shr, lsr, trr);
                c.index_rows = None;
                k
            })?
        };
        // The selection rows are gathered: every rank's heads attend every query row. The
        // candidate pool is not: each rank's later index layers select only its own rows and so
        // read only its own candidate rows (`Shared::cand_row0`).
        if split && sh[0].topk.is_some() {
            let tw = sh[0].topk_w;
            let refs: Vec<&Buf> = sh.iter().map(|x| x.topk.as_ref().unwrap()).collect();
            let blks: Vec<Blk> = (0..n)
                .map(|r| Blk::flat(span(r).start * tw * 4, span(r).len() * tw * 4))
                .collect();
            self.allgather(&refs, &blks)?;
        }
        let c = self.cx[0].chunk.min(s).max(1);
        let oa_ld = g.o_groups * g.o_lora;
        let bufs = self.each(|r, cx| {
            let ops = &cx.ops;
            Ok((
                ops.tmp(c * lw[r].heads_l * hd * 2)?,
                ops.tmp(c * lw[r].heads_l * hd * 2)?,
                ops.tmp(c * oa_ld * 2)?,
                ops.tmp(s * d * 2)?,
            ))
        })?;
        let (mut q, mut o, mut oa, mut out) = (vec![], vec![], vec![], vec![]);
        for (a, b, cc, dd) in bufs {
            q.push(a);
            o.push(b);
            oa.push(cc);
            out.push(dd);
        }
        let shr: &[Shared] = sh;
        for r0 in (0..s).step_by(c) {
            let rows = c.min(s - r0);
            {
                let mut trs: Vec<&mut Trace> = tr.iter_mut().collect();
                self.each_mut(&mut trs, |r, cx, trr| {
                    let oap = oa[r].p(&cx.ops.st);
                    cx.attn_heads(
                        &lw[r], &keys[r], &shr[r], sp, r0, rows, &q[r], &o[r], oap, oa_ld, false,
                        trr,
                    )
                })?;
            }
            let blks: Vec<Blk> = lw
                .iter()
                .map(|w| Blk {
                    off: w.group0 * g.o_lora * 2,
                    width: w.groups_l * g.o_lora * 2,
                    pitch: oa_ld * 2,
                    rows,
                })
                .collect();
            let refs: Vec<&Buf> = oa.iter().collect();
            self.allgather(&refs, &blks)?;
            self.each(|r, cx| {
                let st = cx.ops.st.clone();
                cx.attn_wo_b(
                    &lw[r],
                    oa[r].p(&st),
                    oa_ld,
                    rows,
                    out[r].at(&st, r0 * d * 2),
                    d,
                )
            })?;
            let blks: Vec<Blk> = lw
                .iter()
                .map(|w| Blk {
                    off: r0 * d * 2 + w.dim0 * 2,
                    width: w.wo_b.n * 2,
                    pitch: d * 2,
                    rows,
                })
                .collect();
            let refs: Vec<&Buf> = out.iter().collect();
            self.allgather(&refs, &blks)?;
        }
        Ok(out)
    }

    /// MoE for rows `[r0, r0 + rows)` of `x2` (every rank's copy), written into the same rows of
    /// `m`. One rank runs the one-card MoE ([`Ctx::moe`]: device plan, no host round trip). N ranks:
    /// every rank routes its replicated copy (the same bytes, so the same selection) and builds the
    /// device plan of its experts on its own thread; rank 0's expert offsets size the gathers (the
    /// one host read); each rank's experts run through the tile-list GEMMs, the expert rows are
    /// gathered, and every rank adds each token's rows in ascending expert id with the replicated
    /// shared expert.
    fn moe(
        &mut self,
        lw: &[LayerW],
        x2: &[Buf],
        r0: usize,
        rows: usize,
        m: &[Buf],
        tr: &mut [Trace],
    ) -> Result<(), String> {
        let n = self.n();
        let g = self.cx[0].g.clone();
        let (ne, topk, d) = (g.n_experts, g.topk, g.dim);
        if n == 1 {
            let c = &mut self.cx[0];
            let st = c.ops.st.clone();
            return c.moe(
                &lw[0],
                x2[0].at(&st, r0 * d * 2),
                rows,
                m[0].at(&st, r0 * d * 2),
                &mut tr[0],
            );
        }
        let trace = self.trace;
        let plans = {
            let mut trs: Vec<&mut Trace> = tr.iter_mut().collect();
            self.each_mut(&mut trs, |r, c, trr| {
                let st = c.ops.st.clone();
                let x = x2[r].at(&st, r0 * d * 2);
                let (sel, selw) = c.moe_route(&lw[r], x, rows)?;
                if trace {
                    trr.route_sel
                        .extend_from_slice(&bytes_i32(&host_timed(&sel, &st)?));
                    trr.route_w
                        .extend_from_slice(&bytes_f32(&host_timed(&selw, &st)?));
                }
                let ex = lw[r].expert0..lw[r].expert0 + lw[r].experts.len();
                c.moe_plan(&sel, &selw, rows, ex)
            })?
        };
        let off: Vec<usize> = bytes_i32(&host_timed(&plans[0].off, &self.cx[0].ops.st)?)
            .into_iter()
            .map(|x| x as usize)
            .collect();
        if off[ne] != rows * topk {
            return Err(format!(
                "route: {} of {} pairs placed",
                off[ne],
                rows * topk
            ));
        }
        let np = rows * topk;
        let parts = self.each(|r, c| {
            let er = Part { rank: r, n }.range(ne);
            let (p0, p1) = (off[er.start], off[er.end]);
            let st = c.ops.st.clone();
            let x = x2[r].at(&st, r0 * d * 2);
            let sl = c.ops.tmp(np * d * 2)?;
            let so = if lw[r].experts.is_empty() && p1 > p0 {
                // experts off the card: queue the misses' copies first, so they overlap the
                // shared expert (an independent output; launch order never changes a value)
                let need: Vec<usize> = er.clone().filter(|&e| off[e + 1] > off[e]).collect();
                let cache = c
                    .experts
                    .as_mut()
                    .ok_or("the layer holds no routed experts and the rank has no expert cache")?;
                let bases = cache.resolve(&c.ops, lw[r].id, &need)?;
                let tab = c.cached_expert_table(er.clone(), &need, &bases)?;
                let so = c.shared_expert(&lw[r], x, rows)?;
                c.experts_tiles(&lw[r], x, &plans[r], p0, p1, &sl, Some(&tab))?;
                so
            } else {
                c.experts_tiles(&lw[r], x, &plans[r], p0, p1, &sl, None)?;
                c.shared_expert(&lw[r], x, rows)?
            };
            Ok((sl, so, Blk::flat(p0 * d * 2, (p1 - p0) * d * 2)))
        })?;
        let (mut slots, mut shared_out, mut blks) = (vec![], vec![], vec![]);
        for (sl, so, b) in parts {
            slots.push(sl);
            shared_out.push(so);
            blks.push(b);
        }
        let refs: Vec<&Buf> = slots.iter().collect();
        self.allgather(&refs, &blks)?;
        self.each(|r, c| {
            let st = c.ops.st.clone();
            ck("moe_combine", unsafe {
                ffi::memra_v41_moe_combine(
                    slots[r].p(&st).c(),
                    d as i64,
                    plans[r].order.p(&st).i(),
                    topk as i32,
                    shared_out[r].p(&st).c(),
                    d as i64,
                    m[r].at(&st, r0 * d * 2).m(),
                    d as i64,
                    rows as i32,
                    d as i32,
                    c.ops.sv(),
                )
            })
        })?;
        if trace {
            for r in 0..n {
                let st = self.cx[r].ops.st.clone();
                let err = bytes_i32(&plans[r].err.host(&st)?)[0];
                if err != 0 {
                    return Err(format!("rank {r} route: {err} expert ids out of range"));
                }
            }
        }
        // The slot, plan and shared-expert buffers are freed next on their own streams, after
        // the combine; a peer's copy out of a slot is ordered before the free by the gather's
        // release event, which the producer's stream waits on.
        Ok(())
    }

    /// Logits of rows `[r0, r0 + rows)`: the collapse with the carried pre-mix and the final norm
    /// on every rank, each rank's vocab rows of the fp32 head, gathered on the host `[rows][vocab]`.
    pub fn head_logits(
        &mut self,
        h: &[Buf],
        pre: &[Buf],
        r0: usize,
        rows: usize,
        nw: &[Buf],
        hw: &[BW],
    ) -> Result<Vec<f32>, String> {
        let n = self.n();
        let g = self.cx[0].g.clone();
        let (hc, d) = (g.hc, g.dim);
        let parts = self.each(|r, c| {
            let st = c.ops.st.clone();
            let x = c.ops.buf(rows * d * 2)?;
            ck("hc_pre", unsafe {
                ffi::memra_v41_hc_pre(
                    h[r].at(&st, r0 * hc * d * 2).m(),
                    pre[r].at(&st, r0 * hc * 4).f(),
                    rows as i32,
                    hc as i32,
                    d as i32,
                    x.p(&st).m(),
                    d as i64,
                    c.ops.sv(),
                )
            })?;
            c.ops
                .rmsnorm(x.p(&st), d, &nw[r], x.p(&st), d, rows, d, g.norm_eps)?;
            let lg = c.ops.buf(rows * hw[r].n * 4)?;
            c.ops
                .linear_bf16(x.p(&st), d, rows, &hw[r], lg.p(&st), hw[r].n, true)?;
            Ok(bytes_f32(&host_timed(&lg, &st)?))
        })?;
        let mut out = vec![0f32; rows * g.vocab];
        let mut col = 0;
        for r in 0..n {
            let v = &parts[r];
            let w = hw[r].n;
            for i in 0..rows {
                out[i * g.vocab + col..i * g.vocab + col + w]
                    .copy_from_slice(&v[i * w..(i + 1) * w]);
            }
            col += w;
        }
        Ok(out)
    }

    /// Each rank's final norm and vocab rows of the head.
    pub fn load_head(&self, stm: &StModel, g: &Geo) -> Result<(Vec<Buf>, Vec<BW>), String> {
        let n = self.n();
        let (mut nw, mut hw) = (vec![], vec![]);
        for r in 0..n {
            let ops = &self.cx[r].ops;
            nw.push(load_vec_bf16(ops, stm, "norm.weight", g.dim)?);
            hw.push(load_bf16_rows(
                ops,
                stm,
                "head.weight",
                g.vocab,
                g.dim,
                Part { rank: r, n }.range(g.vocab),
            )?);
        }
        Ok((nw, hw))
    }

    /// The head-split indexer arm, measured against the model's replicated indexer on the same
    /// inputs: rank r scores `index_heads / n` heads into fp32 partials, the partials are gathered
    /// on every rank, added in rank order, rounded and masked, then the top-k. Query rows run in
    /// slices of `slice`, from row `first` (the rows before it are left unselected: `first = rows - 1`
    /// is one decode-shaped row over every key). Returns `(selection [rows][topk_w] as rank 0 holds
    /// it, seconds of the q projection + score + gather + finish + top-k)`. `x` `[rows][dim]` and `qr` `[rows][q_lora]`
    /// are the layer's attention input and query latent of a prefill; `index_k` holds `n_c` keys,
    /// replicated per rank. Index layers without a candidate pool only.
    #[allow(clippy::too_many_arguments)]
    pub fn index_split_bench(
        &mut self,
        lw: &[LayerW],
        x: &[Buf],
        qr: &[Buf],
        index_k: &[Buf],
        n_c: usize,
        rows: usize,
        ratio: usize,
        topk_w: usize,
        slice: usize,
        first: usize,
    ) -> Result<(Vec<i32>, f64), String> {
        let n = self.n();
        let g = self.cx[0].g.clone();
        let ih = g.index_heads;
        if !ih.is_multiple_of(n) || !matches!(ih / n, 16 | 32) {
            return Err(format!(
                "index heads {ih} do not split into 16/32-head tiles at {n}"
            ));
        }
        let hl = ih / n;
        let qd = hl * g.index_dim;
        let wscale = ((g.index_dim as f64).powf(-0.5) * (ih as f64).powf(-0.5)) as f32;
        // this rank's head rows of the indexer's wq_b and weights_proj (outside the timing)
        let mut wqs = Vec::with_capacity(n);
        let mut wps = Vec::with_capacity(n);
        for r in 0..n {
            let c = &self.cx[r];
            let iw = lw[r].idx.as_ref().ok_or("not an index source")?;
            let kb = iw.wq_b.k / 32;
            wqs.push(super::QW {
                w: sub_buf(&c.ops, &iw.wq_b.w, r * qd * iw.wq_b.k, qd * iw.wq_b.k)?,
                s: sub_buf(&c.ops, &iw.wq_b.s, r * qd / 32 * kb, qd / 32 * kb)?,
                n: qd,
                k: iw.wq_b.k,
                fmt: iw.wq_b.fmt,
            });
            wps.push(BW {
                w: sub_buf(&c.ops, &iw.wproj.w, r * hl * g.dim * 2, hl * g.dim * 2)?,
                n: hl,
                k: g.dim,
            });
        }
        let sl = slice.min(rows - first.min(rows)).max(1);
        let mut parts = Vec::with_capacity(n);
        let mut outs = Vec::with_capacity(n);
        for r in 0..n {
            parts.push(self.cx[r].ops.buf(n * sl * n_c * 4)?);
            outs.push(self.cx[r].ops.buf(rows * topk_w * 4)?);
        }
        self.sync_all()?;
        let t0 = Instant::now();
        for r0 in (first..rows).step_by(sl) {
            let m = sl.min(rows - r0);
            for r in 0..n {
                let c = &mut self.cx[r];
                let st = c.ops.st.clone();
                let q = c.ops.buf(m * qd * 2)?;
                c.ops.linear_q(
                    qr[r].at(&st, r0 * g.q_lora * 2),
                    g.q_lora,
                    m,
                    &wqs[r],
                    q.p(&st),
                    qd,
                    false,
                    &mut c.scr,
                )?;
                c.ops
                    .rope(q.p(&st), qd, m, hl, g.index_dim, &c.rope_comp, r0, 1, false)?;
                c.ops.fp4_qdq(q.p(&st), qd, m, qd, 32)?;
                let w = c.ops.buf(m * hl * 2)?;
                c.ops.linear_bf16(
                    x[r].at(&st, r0 * g.dim * 2),
                    g.dim,
                    m,
                    &wps[r],
                    w.p(&st),
                    hl,
                    false,
                )?;
                ck("index_score_part", unsafe {
                    ffi::memra_v41_index_score_part(
                        q.p(&st).c(),
                        qd as i64,
                        index_k[r].p(&st).c(),
                        n_c as i32,
                        w.p(&st).c(),
                        hl as i64,
                        wscale,
                        hl as i32,
                        parts[r].at(&st, r * sl * n_c * 4).fm(),
                        n_c as i64,
                        m as i32,
                        c.ops.sv(),
                    )
                })?;
            }
            let refs: Vec<&Buf> = parts.iter().collect();
            let blks: Vec<Blk> = (0..n)
                .map(|r| Blk::flat(r * sl * n_c * 4, m * n_c * 4))
                .collect();
            self.allgather(&refs, &blks)?;
            for r in 0..n {
                let c = &mut self.cx[r];
                let st = c.ops.st.clone();
                let ptrs: Vec<u8> = (0..n)
                    .flat_map(|k| parts[r].at(&st, k * sl * n_c * 4).addr().to_le_bytes())
                    .collect();
                let pd = Buf::from_bytes(&st, &ptrs)?;
                let score = c.ops.buf(m * n_c * 2)?;
                ck("index_score_finish", unsafe {
                    ffi::memra_v41_index_score_finish(
                        pd.p(&st).c(),
                        n as i32,
                        n_c as i64,
                        n_c as i32,
                        r0 as i32,
                        ratio as i32,
                        std::ptr::null(),
                        0,
                        1,
                        score.p(&st).m(),
                        n_c as i64,
                        m as i32,
                        c.ops.sv(),
                    )
                })?;
                ck("topk", unsafe {
                    ffi::memra_v41_topk(
                        score.p(&st).c(),
                        n_c as i64,
                        n_c as i32,
                        topk_w as i32,
                        topk_w as i32,
                        0,
                        outs[r].at(&st, r0 * topk_w * 4).im(),
                        topk_w as i64,
                        m as i32,
                        c.ops.sv(),
                    )
                })?;
            }
            // the next slice reuses the partial buffers every rank just read
            self.sync_all()?;
        }
        let secs = t0.elapsed().as_secs_f64();
        let sel0 = bytes_i32(&outs[0].host(&self.cx[0].ops.st)?);
        Ok((sel0, secs))
    }
}

impl Group {
    /// The row-split arm: the model's indexer (all heads, replicated weights) with the query rows
    /// split into `n` contiguous ranges, one per rank, then the selection rows gathered. The same
    /// per-row program as the replicated indexer, so the selection is the replicated one by
    /// construction (checked by the caller). Returns `(rank 0's gathered selection, seconds)`.
    #[allow(clippy::too_many_arguments)]
    pub fn index_rows_bench(
        &mut self,
        lw: &[LayerW],
        x: &[Buf],
        qr: &[Buf],
        index_k: &[Buf],
        n_c: usize,
        rows: usize,
        slice: usize,
        first: usize,
    ) -> Result<(Vec<i32>, f64), String> {
        let n = self.n();
        let mut shs = Vec::with_capacity(n);
        for r in 0..n {
            shs.push(self.cx[r].index_bench_state(&lw[r], &index_k[r], n_c, rows)?);
        }
        let tw = shs[0].topk_w;
        let first = first.min(rows);
        let per = (rows - first).div_ceil(n);
        let span = |r: usize| {
            (
                (first + r * per).min(rows),
                (first + (r + 1) * per).min(rows),
            )
        };
        self.sync_all()?;
        let t0 = Instant::now();
        for r in 0..n {
            let (a, b) = span(r);
            for r0 in (a..b).step_by(slice.max(1)) {
                let r1 = (r0 + slice.max(1)).min(b);
                self.cx[r].index_select(&lw[r], &x[r], &qr[r], r0, r1, 0, rows, 0, &mut shs[r])?;
            }
        }
        let refs: Vec<&Buf> = shs.iter().map(|x| x.topk.as_ref().unwrap()).collect();
        let blks: Vec<Blk> = (0..n)
            .map(|r| {
                let (a, b) = span(r);
                Blk::flat(a * tw * 4, (b - a) * tw * 4)
            })
            .collect();
        self.allgather(&refs, &blks)?;
        self.sync_all()?;
        let secs = t0.elapsed().as_secs_f64();
        let sel = bytes_i32(&shs[0].topk.as_ref().unwrap().host(&self.cx[0].ops.st)?);
        Ok((sel, secs))
    }
}

impl Ctx {
    /// A fresh CSA2 state holding a copy of `index_k` (`n_c` keys) and room for the selection of
    /// `rows` prefill rows (and the candidate pool on the candidate source).
    fn index_bench_state(
        &mut self,
        lw: &LayerW,
        index_k: &Buf,
        n_c: usize,
        rows: usize,
    ) -> Result<Shared, String> {
        let mut sh = Shared {
            index_k: Some(super::clone_buf(&self.ops, index_k)?),
            index_k_cap: n_c,
            ..Default::default()
        };
        let tw = self.g.index_topk.min(n_c);
        sh.topk = Some(self.ops.buf(rows * tw * 4)?);
        sh.topk_w = tw;
        if lw.cand_source {
            let nb = n_c.div_ceil(self.g.cand_block);
            sh.cand = Some(self.ops.buf(rows * nb)?);
            sh.cand_ld = nb;
        }
        Ok(sh)
    }

    /// The keys side of attention alone for a prefill of `s` rows (the CSA2 state a layer leaves:
    /// compressed KV, index keys, selection), for benches that need a source layer's own keys.
    pub fn keys_only(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        s: usize,
        sh: &mut Shared,
        ls: &mut LayerState,
        tr: &mut Trace,
    ) -> Result<(), String> {
        self.attn_keys(lw, x, 0, s, sh, ls, tr).map(|_| ())
    }

    /// The model's indexer (all heads) over the `rows` rows of a prefill on this card, in query
    /// slices, timed: `(selection [rows][topk_w], seconds)`.
    #[allow(clippy::too_many_arguments)]
    pub fn index_replicated_bench(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        qr: &Buf,
        index_k: &Buf,
        n_c: usize,
        rows: usize,
        slice: usize,
        first: usize,
    ) -> Result<(Vec<i32>, f64), String> {
        let mut sh = self.index_bench_state(lw, index_k, n_c, rows)?;
        self.ops.sync()?;
        let t0 = Instant::now();
        for r0 in (first.min(rows)..rows).step_by(slice.max(1)) {
            let r1 = (r0 + slice.max(1)).min(rows);
            self.index_select(lw, x, qr, r0, r1, 0, rows, 0, &mut sh)?;
        }
        self.ops.sync()?;
        let secs = t0.elapsed().as_secs_f64();
        Ok((
            bytes_i32(&sh.topk.as_ref().unwrap().host(&self.ops.st)?),
            secs,
        ))
    }
}

/// A copy of `len` bytes of `b` from `off` (measurement arms only).
fn sub_buf(ops: &super::Ops, b: &Buf, off: usize, len: usize) -> Result<Buf, String> {
    let o = ops.buf(len)?;
    super::copy_d2d(ops, b, off, &o, 0, len)?;
    Ok(o)
}

/// The MoE dispatch plan of one slice: pairs sorted by expert, tokens ascending inside an expert.
pub struct MoePlan {
    /// pair offsets per expert, `ne + 1`
    pub off: Vec<usize>,
    pub cnt: Vec<usize>,
    /// token (row within the slice) and routing weight of each pair
    pub ptok: Vec<i32>,
    pub pw: Vec<f32>,
    /// `[rows][topk]`: each token's pair indices in ascending expert id
    pub order: Vec<i32>,
}

impl MoePlan {
    pub fn new(
        sel: &[i32],
        selw: &[f32],
        rows: usize,
        topk: usize,
        ne: usize,
    ) -> Result<Self, String> {
        if sel.len() != rows * topk || selw.len() != rows * topk {
            return Err("moe plan: routing size".into());
        }
        let mut cnt = vec![0usize; ne];
        for &e in sel {
            if e < 0 || e as usize >= ne {
                return Err(format!("route: expert {e}"));
            }
            cnt[e as usize] += 1;
        }
        let mut off = vec![0usize; ne + 1];
        for e in 0..ne {
            off[e + 1] = off[e] + cnt[e];
        }
        let np = off[ne];
        let mut fill = off.clone();
        let mut ptok = vec![0i32; np];
        let mut pw = vec![0f32; np];
        let mut order = vec![-1i32; rows * topk];
        let mut tok: Vec<(i32, i32)> = Vec::with_capacity(topk);
        for t in 0..rows {
            tok.clear();
            for j in 0..topk {
                let e = sel[t * topk + j] as usize;
                let p = fill[e];
                fill[e] += 1;
                ptok[p] = t as i32;
                pw[p] = selw[t * topk + j];
                tok.push((e as i32, p as i32));
            }
            tok.sort_unstable();
            for (j, &(_, p)) in tok.iter().enumerate() {
                order[t * topk + j] = p;
            }
        }
        Ok(Self {
            off,
            cnt,
            ptok,
            pw,
            order,
        })
    }
}

impl Ctx {
    /// The shared expert on `rows` rows of `x`: bf16 `[rows][dim]`.
    pub fn shared_expert(&mut self, lw: &LayerW, x: Dp, rows: usize) -> Result<Buf, String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let (d, inter) = (g.dim, g.moe_inter);
        let [s1, s2, s3] = &lw.shared;
        let gb = self.ops.tmp(rows * inter * 2)?;
        let ub = self.ops.tmp(rows * inter * 2)?;
        let hb = self.ops.tmp(rows * inter * 2)?;
        let out = self.ops.tmp(rows * d * 2)?;
        self.ops
            .linear_q(x, d, rows, s1, gb.p(&st), inter, false, &mut self.scr)?;
        self.ops
            .linear_q(x, d, rows, s3, ub.p(&st), inter, false, &mut self.scr)?;
        ck("swiglu", unsafe {
            ffi::memra_v41_swiglu(
                gb.p(&st).m(),
                ub.p(&st).m(),
                inter as i64,
                std::ptr::null(),
                g.swiglu_limit,
                hb.p(&st).m(),
                inter as i64,
                rows as i32,
                inter as i32,
                self.ops.sv(),
            )
        })?;
        self.ops.linear_q(
            hb.p(&st),
            inter,
            rows,
            s2,
            out.p(&st),
            d,
            false,
            &mut self.scr,
        )?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_plan_orders_pairs_by_expert_then_token_and_tokens_by_expert() {
        // 3 tokens, top-2 of 4 experts; the router lists experts by score, not by id
        let sel = [2, 0, 3, 2, 0, 1];
        let w = [0.5, 0.25, 0.75, 0.125, 1.0, 2.0];
        let p = MoePlan::new(&sel, &w, 3, 2, 4).unwrap();
        assert_eq!(p.cnt, vec![2, 1, 2, 1]);
        assert_eq!(p.off, vec![0, 2, 3, 5, 6]);
        // expert 0: tokens 0, 2; expert 1: token 2; expert 2: tokens 0, 1; expert 3: token 1
        assert_eq!(p.ptok, vec![0, 2, 2, 0, 1, 1]);
        assert_eq!(p.pw, vec![0.25, 1.0, 2.0, 0.5, 0.125, 0.75]);
        // token 0: experts 0 (pair 0), 2 (pair 3); token 1: 2 (pair 4), 3 (pair 5); token 2: 0, 1
        assert_eq!(p.order, vec![0, 3, 4, 5, 1, 2]);
        assert!(MoePlan::new(&[4, 0], &[1.0, 1.0], 1, 2, 4).is_err());
    }

    #[test]
    fn blk_end_covers_the_last_row_only_to_its_width() {
        let b = Blk {
            off: 8,
            width: 4,
            pitch: 16,
            rows: 3,
        };
        assert_eq!(b.end(), 8 + 2 * 16 + 4);
        assert_eq!(b.bytes(), 12);
        assert_eq!(Blk::flat(10, 0).bytes(), 0);
    }
}
