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
//!   - MoE: every rank runs the router on its own replicated copy and builds the same dispatch
//!     plan (pairs sorted by expert, tokens ascending inside an expert, as the per-expert loop
//!     orders them). The copies are the same bytes through the same kernels, so the routings
//!     agree by construction; nothing compares them at run time (the N-rank gates compare the
//!     ranks' bytes with the single-card program's). Each rank runs its
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
//! the producer's stream waited on that copy's release event, and a rank's second stream orders
//! its buffers with its fork and join events (`Group::new` says how). cudarc's per-buffer event
//! tracking is off on the ranks' contexts.

// Rank-indexed loops walk several per-rank vectors (contexts, buffers, shares) in step; an index is
// the clearer shape there, as in `tp_transport.rs`.
#![allow(clippy::needless_range_loop)]

use super::{
    BW, BatchRow, BatchSel, Buf, Ctx, Dp, Geo, LayerState, LayerW, Part, Shared, StepLayer,
    StepRef, Trace, bytes_f32, bytes_i32, ck, ffi, load_bf16_rows, load_vec_bf16,
};
use crate::Engine;
use crate::tp_transport::{self, TpTransport};
use cudarc::driver::CudaEvent;
use memra_gguf::safetensors::StModel;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

/// Which keys side a block's attention runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    /// a trunk layer: its window ring, compressor, indexer
    Trunk,
    /// a DSpark draft stage over its block: every row reads the first `filled` slots of the
    /// stage's ring of the trunk's rows and then the whole block
    Draft { filled: usize },
}

/// What a block's attention reads on its keys side: one call's layer state on every rank, or a
/// batch of sequences' rows.
enum Side<'a, 'b> {
    Layer {
        sh: &'a mut [Shared],
        ls: &'a mut [LayerState],
        kind: AttnKind,
    },
    Batch {
        pos: &'a [Dp],
        rows: &'a mut [Vec<BatchRow<'b>>],
        sel: &'a mut [BatchSel],
        /// each rank's step block and the layer's place in it (a served step); None: the
        /// copy form of the keys side (a traced call, the gates' reference)
        step: Option<(&'a [Dp], &'a StepLayer)>,
    },
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
    /// device-push: each rank's inbox, two slots a producer ([`Group::gather_push`])
    inbox: Vec<Buf>,
    /// device-push: each rank's count of small gathers, in its device memory. A gather's number
    /// is the count plus one, read by the kernels themselves: every rank runs the same gathers
    /// in the same order, so the counts agree and no launch carries a number from the host
    push_ctr: Vec<Buf>,
    /// each rank's inbox of the large gathers and its packed block, kept across gathers and
    /// grown to the largest one seen ([`Group::gather_packed`])
    big_in: Vec<Option<Buf>>,
    pk_keep: Vec<Option<Buf>>,
    /// the gate's control case: the producer stores the word before the data
    pub push_word_first: bool,
    pub stats: CommStats,
    /// synchronize and time every collective (measurement runs)
    pub timed: bool,
    /// keep `attn_out`, `moe_out` and the routing in each rank's [`Trace`]
    pub trace: bool,
    /// one host thread per rank issues that rank's work between collectives (None: this thread
    /// issues the ranks in turn, `MEMRA_DSV41_TP_ISSUE=serial`)
    pool: Option<Crew>,
    /// spans of a decode step captured as CUDA graphs ([`super::graph`])
    pub graphs: super::graph::Graphs,
    /// the ranks' streams are recording ([`Group::capture`]): only launches may be issued
    pub(super) capturing: bool,
    /// with [`Group::log_routes`]: every block's routing and one-row MoE output, in call order
    pub route_log: Option<RouteLog>,
    /// why the group can no longer run a forward: a gather whose peer never arrived
    /// ([`Self::route_verdict`]); every later call refuses ([`Self::live`])
    pub lost: Option<String>,
}

/// What the router chose and what the MoE returned, call by call in the order the blocks ran
/// (rank 0's replicated copy): the measurement behind a gate that judges a change of a sum's
/// order, where the bytes move and the question is which experts and how far.
#[derive(Default)]
pub struct RouteLog {
    /// (layer id, rows) of each block
    pub calls: Vec<(usize, usize)>,
    /// i32 `[rows][top-k]` a block, appended: the routed expert ids
    pub ids: Vec<i32>,
    /// f32 `[rows][top-k]` a block: their weights
    pub w: Vec<f32>,
    /// bf16 `[dim]` of every one-row block's MoE output, appended (a decode step's; a prefill's
    /// rows are counted by their routing alone)
    pub moe: Vec<u8>,
}

/// What the second half of a block reads, a buffer a rank: the residual streams `h` (written in
/// place), the attention output, and the attention site's pre-mix, post-mix and combine weights.
#[derive(Clone, Copy)]
struct FfnIn<'a> {
    h: &'a [Buf],
    a: &'a [Buf],
    a_pre: &'a [Buf],
    a_post: &'a [Buf],
    a_comb: &'a [Buf],
}

/// Bytes ahead of a slot's payload in a device-push inbox: the sequence word, padded so the
/// payload starts on its own cache lines.
const PUSH_HDR: usize = 256;
/// One producer's slot in a rank's inbox.
const PUSH_SLOT: usize = PUSH_HDR + ffi::PUSH_MAX;
/// The most bytes a rank stores into its peers in one gather inside kernels
/// ([`Group::gather_push`]: its region times its peers); a gather that would store more goes
/// through the kept blocks and the copy engine ([`Group::gather_packed`]). A push is one block
/// of a kernel storing the region into every peer, so its time grows with the bytes (4.4 us at
/// 1 KiB, 16.7 at 128 KiB, 27.4 at 256 KiB on two cards); a copy-engine gather pays launches,
/// events and copy calls first and then moves bytes at the link's rate (14.5, 23.2, 25.5 us).
/// The two cross at 227 KiB on two cards (`dsv41-bench p2p`'s `gather` lines, three runs within
/// 0.5 us). With more peers the kernel stores the region once a peer, so the bound is on the
/// stores, not on the region.
const PUSH_LIMIT: usize = 224 << 10;

/// How long an issue thread spins for its next section before it parks. A one-token forward is
/// some twenty sections a layer, microseconds apart (measured: all but one gap a layer under
/// 10 us, the one being the routing read a one-token MoE no longer makes), so 200 us keeps the
/// threads hot through a step; a parked thread costs a wake of 10 to 100 us, paid once a step
/// while the logits are read. A longer spin only burns the cores other work wants.
const CREW_SPIN: Duration = Duration::from_micros(200);

type Section<'a> = dyn Fn(usize) + Sync + 'a;

struct CrewShared {
    /// bumped once a section, after `job` and `left` are set
    section: AtomicU64,
    /// the section's body; valid from the bump until `left` is zero
    job: UnsafeCell<*const Section<'static>>,
    /// workers that have not finished the section
    left: AtomicUsize,
    parked: Vec<AtomicBool>,
    caller: Mutex<Option<std::thread::Thread>>,
    caller_parked: AtomicBool,
    panicked: AtomicBool,
    stop: AtomicBool,
}

// SAFETY: `job` is written by the one caller of `Crew::run` before the `section` bump and read by
// the workers after they see the bump; `run` takes `&mut Crew`, so there is one caller at a time
unsafe impl Sync for CrewShared {}
unsafe impl Send for CrewShared {}

/// The issue threads of ranks 1..N; the thread that calls [`Crew::run`] issues rank 0. A section
/// is handed over through an atomic the threads spin on ([`CREW_SPIN`]), so consecutive sections
/// of a decode step cost no thread wake.
struct Crew {
    sh: Arc<CrewShared>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Crew {
    fn new(n: usize) -> Result<Self, String> {
        let sh = Arc::new(CrewShared {
            section: AtomicU64::new(0),
            job: UnsafeCell::new(std::ptr::null::<fn(usize)>() as *const Section<'static>),
            left: AtomicUsize::new(0),
            parked: (1..n).map(|_| AtomicBool::new(false)).collect(),
            caller: Mutex::new(None),
            caller_parked: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            stop: AtomicBool::new(false),
        });
        let mut threads = Vec::with_capacity(n - 1);
        for w in 0..n - 1 {
            let sh = sh.clone();
            threads.push(
                std::thread::Builder::new()
                    .name(format!("dsv41-rank{}", w + 1))
                    .spawn(move || Self::work(&sh, w))
                    .map_err(|e| format!("dsv41 tp issue threads: {e}"))?,
            );
        }
        Ok(Self { sh, threads })
    }

    fn work(sh: &CrewShared, w: usize) {
        let mut seen = 0u64;
        loop {
            let t = Instant::now();
            let mut spins = 0u32;
            loop {
                let s = sh.section.load(Ordering::SeqCst);
                if s != seen {
                    seen = s;
                    break;
                }
                if sh.stop.load(Ordering::SeqCst) {
                    return;
                }
                spins = spins.wrapping_add(1);
                if spins & 1023 != 0 || t.elapsed() < CREW_SPIN {
                    std::hint::spin_loop();
                    continue;
                }
                // park; the caller unparks a thread it sees parked after its bump, and a bump
                // between the store and the check below is seen by that check
                sh.parked[w].store(true, Ordering::SeqCst);
                if sh.section.load(Ordering::SeqCst) == seen && !sh.stop.load(Ordering::SeqCst) {
                    std::thread::park();
                }
                sh.parked[w].store(false, Ordering::SeqCst);
            }
            // SAFETY: the caller set `job` before the bump this thread just saw and keeps the
            // closure alive until `left` is zero, which this thread's decrement below precedes
            let f: &Section<'_> = unsafe { &**sh.job.get() };
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(w + 1))).is_err() {
                sh.panicked.store(true, Ordering::SeqCst);
            }
            if sh.left.fetch_sub(1, Ordering::SeqCst) == 1
                && sh.caller_parked.load(Ordering::SeqCst)
                && let Some(c) = sh.caller.lock().unwrap_or_else(|e| e.into_inner()).as_ref()
            {
                c.unpark();
            }
        }
    }

    /// `f(r)` for every rank: rank 0 on this thread, rank r on worker r - 1. Returns when all
    /// have finished.
    fn run(&mut self, f: &Section<'_>) -> Result<(), String> {
        struct Wait<'a>(&'a CrewShared);
        impl Drop for Wait<'_> {
            // also on a panic in rank 0's body: no worker may outlive the closure it borrows
            fn drop(&mut self) {
                let sh = self.0;
                let t = Instant::now();
                let mut spins = 0u32;
                while sh.left.load(Ordering::SeqCst) != 0 {
                    spins = spins.wrapping_add(1);
                    if spins & 1023 != 0 || t.elapsed() < CREW_SPIN {
                        std::hint::spin_loop();
                        continue;
                    }
                    sh.caller_parked.store(true, Ordering::SeqCst);
                    if sh.left.load(Ordering::SeqCst) != 0 {
                        std::thread::park();
                    }
                    sh.caller_parked.store(false, Ordering::SeqCst);
                }
            }
        }
        let sh = &*self.sh;
        *sh.caller.lock().unwrap_or_else(|e| e.into_inner()) = Some(std::thread::current());
        // SAFETY: the lifetime is erased for the workers; `Wait` keeps this frame, and so `f`,
        // alive until every worker has finished the section
        unsafe {
            *sh.job.get() =
                std::mem::transmute::<*const Section<'_>, *const Section<'static>>(f as *const _);
        }
        sh.left.store(self.threads.len(), Ordering::SeqCst);
        sh.section.fetch_add(1, Ordering::SeqCst);
        for (w, t) in self.threads.iter().enumerate() {
            if sh.parked[w].load(Ordering::SeqCst) {
                t.thread().unpark();
            }
        }
        {
            let _wait = Wait(sh);
            f(0);
        }
        if sh.panicked.swap(false, Ordering::SeqCst) {
            return Err("dsv41 tp: a rank's issue thread panicked".into());
        }
        Ok(())
    }
}

impl Drop for Crew {
    fn drop(&mut self) {
        self.sh.stop.store(true, Ordering::SeqCst);
        for t in self.threads.drain(..) {
            t.thread().unpark();
            let _ = t.join();
        }
    }
}

/// A pointer the ranks index disjointly inside one section.
struct RankPtr<T>(*mut T);
// SAFETY: every rank's thread touches only its own element
unsafe impl<T: Send> Sync for RankPtr<T> {}
impl<T> RankPtr<T> {
    /// # Safety
    /// `r` is inside the slice the pointer came from and no other thread holds element `r`.
    #[allow(clippy::mut_from_ref)]
    unsafe fn rank(&self, r: usize) -> &mut T {
        unsafe { &mut *self.0.add(r) }
    }
}

/// The results of a section, one slot a rank.
fn rank_slots<T>(n: usize) -> Vec<Mutex<Option<Result<T, String>>>> {
    (0..n).map(|_| Mutex::new(None)).collect()
}

fn rank_results<T>(out: Vec<Mutex<Option<Result<T, String>>>>) -> Result<Vec<T>, String> {
    out.into_iter()
        .map(|m| {
            m.into_inner()
                .unwrap_or_else(|e| e.into_inner())
                .unwrap_or_else(|| Err("dsv41 tp: a rank's section did not run".into()))
        })
        .collect()
}

/// How the host issues the ranks' work: one thread per rank (the default at N > 1) or this thread
/// for every rank in turn (`MEMRA_DSV41_TP_ISSUE=serial`, the measurement and rollback seam). The
/// streams, the kernels and the collectives are the same in both, so are the bytes.
pub const ISSUE_ENV: &str = "MEMRA_DSV41_TP_ISSUE";

/// [`Group::each`] over a pool and the rank contexts, for callers that also borrow other fields.
fn each_in<T, F>(pool: Option<&mut Crew>, cx: &mut [Ctx], f: F) -> Result<Vec<T>, String>
where
    T: Send,
    F: Fn(usize, &mut Ctx) -> Result<T, String> + Sync + Send,
{
    match pool {
        Some(p) => {
            let n = cx.len();
            assert_eq!(n, p.threads.len() + 1, "dsv41 tp: one issue thread a rank");
            let (base, out) = (RankPtr(cx.as_mut_ptr()), rank_slots(n));
            p.run(&|r| {
                // SAFETY: r < n and each rank runs once, on its own thread
                let c = unsafe { base.rank(r) };
                *out[r].lock().unwrap_or_else(|e| e.into_inner()) = Some(f(r, c));
            })?;
            rank_results(out)
        }
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

/// A copy between two ranks' cards: region `b` of the rank buffer at `src` (on the card of
/// context `src_ctx`) into the same region at `dst`, on `ops`'s stream (the destination's),
/// through the peer copy call that names both contexts. A contiguous region (one row, or rows
/// with no gap) is one call; a strided one never reaches here (the packed gather moves it
/// through contiguous blocks), and one that did is refused rather than copied another way. WHY: on four cards a copy by plain device
/// address out of a peer's stream-ordered pool block, issued on the consumer's stream, runs at
/// 2.5-2.9 GB/s a rank where the peer call between the same blocks runs at about 50; cudaMalloc
/// blocks or a copy issued by the producer run at 50 too (p2p-probe2 in the perf receipts).
/// Same stream and the same events, so the transport's laws hold unchanged.
fn copy_peer(ops: &super::Ops, dst: u64, src_ctx: usize, src: u64, b: Blk) -> Result<(), String> {
    use cudarc::driver::sys::*;
    if b.bytes() == 0 {
        return Ok(());
    }
    if b.rows > 1 && b.width < b.pitch {
        return Err(format!(
            "dsv41 tp copy_peer: a strided region ({} rows of {} bytes, pitch {}) has no peer path; \
             the packed gather moves strided regions through contiguous blocks",
            b.rows, b.width, b.pitch
        ));
    }
    let r = unsafe {
        cuMemcpyPeerAsync(
            dst + b.off as u64,
            ops.st.context().cu_ctx(),
            src + b.off as u64,
            src_ctx as CUcontext,
            b.bytes(),
            ops.sv() as CUstream,
        )
    };
    if r != CUresult::CUDA_SUCCESS {
        return Err(format!("dsv41 tp peer copy: {r:?}"));
    }
    Ok(())
}

/// Each rank's context handle, for [`copy_peer`] (as an integer, so the rank threads can share it).
fn cu_ctxs(cx: &[Ctx]) -> Vec<usize> {
    cx.iter()
        .map(|c| c.ops.st.context().cu_ctx() as usize)
        .collect()
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
        // no transport named: the measured winner on two cards at every width of a decode call
        // and in prefill, the gathers inside kernels up to the push limit and through kept
        // blocks above it (one rank moves nothing; more ranks than an inbox holds keep the
        // build's default)
        let named = !tp_transport::set_transport_names().is_empty();
        let unset = if n > 1 && n <= ffi::PUSH_PEERS + 1 {
            TpTransport::DevicePush
        } else {
            tp_transport::default_transport()
        };
        let (mut transport, flag) = tp_transport::transport_env_or(unset)?;
        if transport == TpTransport::DevicePush && n > ffi::PUSH_PEERS + 1 {
            return Err(format!(
                "{flag}=device-push: at most {} ranks",
                ffi::PUSH_PEERS + 1
            ));
        }
        let parallel = issue_parallel()?;
        let mut engines = Vec::with_capacity(n);
        for &d in devices {
            let e = Engine::new(d).map_err(|e| format!("rank on device {d}: {e}"))?;
            // A rank issues on two streams: its main one and the second stream of a few-row
            // branch ([`super::Side`]), and its buffers cross between them. The branches order
            // that themselves: a branch starts behind a fork event of the main stream and the
            // main stream joins it before it reads what the branch wrote; what a branch appends
            // to grows on the main stream before the fork ([`super::Ctx::attn_keys`]); a
            // branch's temporaries go back to the branch's free list, which only a later branch
            // takes from (behind its own fork); and a branch buffer that leaves the free list is
            // freed behind an event of the main stream ([`super::Fence`]), because ranks that
            // share a card share its pool. Between ranks the only uses are the gathers, ordered
            // by their publication and release events (the transport laws above). cudarc's
            // per-buffer event tracking would not order any of this (`Buf::p` lets its guard go
            // before the launch it serves) and would add an event record and a stream wait to
            // every pointer fetch on the hot path, so the buffers allocated from here on go
            // untracked.
            unsafe { e.ctx().disable_event_tracking() };
            engines.push(e);
        }
        let distinct = |a: usize, b: usize| devices[a] != devices[b];
        if transport != TpTransport::HostCanonical {
            'grant: for a in 0..n {
                for b in 0..n {
                    if a != b && distinct(a, b) {
                        match crate::tp::grant_peer_access(&engines[a], &engines[b], "dsv41 tp") {
                            Ok(()) => {}
                            // a named transport that cannot run refuses the load; the default
                            // falls back to the one that needs no access between the cards
                            Err(e) if named => return Err(e.to_string()),
                            Err(e) => {
                                eprintln!(
                                    "[dsv41-tp] no peer access between devices {} and {} ({e}): \
                                     the gathers go through the host",
                                    devices[a], devices[b]
                                );
                                transport = TpTransport::HostCanonical;
                                break 'grant;
                            }
                        }
                    }
                }
            }
        }
        let mut cx = Vec::with_capacity(n);
        let mut pub_ev = Vec::with_capacity(n);
        let mut rel_ev = Vec::with_capacity(n);
        let mut inbox = Vec::new();
        let mut push_ctr = Vec::new();
        for e in &engines {
            let st = e.stream();
            pub_ev.push(st.context().new_event(None).map_err(de)?);
            rel_ev.push(st.context().new_event(None).map_err(de)?);
            let c = Ctx::new(st, g.clone(), max_pos)?;
            if transport == TpTransport::DevicePush {
                // zeroed: no word holds a sequence number before the first gather
                inbox.push(c.ops.buf(2 * n * PUSH_SLOT)?);
                push_ctr.push(c.ops.buf(4)?);
            }
            cx.push(c);
        }
        let pool = if n > 1 && parallel {
            Some(Crew::new(n)?)
        } else {
            None
        };
        // calls of 9 rows and more: how far the small-row bodies go at this rank count
        super::set_wide_rows_default(n);
        let mut gr = Self {
            devices: devices.to_vec(),
            engines,
            cx,
            transport,
            transport_flag: flag,
            pub_ev,
            rel_ev,
            inbox,
            push_ctr,
            big_in: (0..n).map(|_| None).collect(),
            pk_keep: (0..n).map(|_| None).collect(),
            push_word_first: false,
            stats: CommStats::default(),
            timed: false,
            trace: false,
            pool,
            graphs: super::graph::Graphs::from_env()?,
            capturing: false,
            route_log: None,
            lost: None,
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

    /// Refuse a forward once the group is lost ([`Self::lost`]).
    pub fn live(&self) -> Result<(), String> {
        match &self.lost {
            Some(why) => Err(format!("the dsv41 group is lost: {why}")),
            None => Ok(()),
        }
    }

    /// What the ranks' route flags (`(call, transport)` of [`Ctx::route_flags`], rank order) say
    /// about the calls they were read with: a transport flag loses the group for good (a peer's
    /// slice of a gather never arrived, so the gathers' sequence words can no longer be trusted);
    /// a call flag fails these calls only.
    pub fn route_verdict(&mut self, flags: &[(i32, i32)]) -> Result<(), String> {
        if let Some(r) = flags.iter().position(|f| f.1 & 0x10000 != 0) {
            let why = format!("rank {r}: a peer's slice of a gather never arrived");
            self.lost = Some(why.clone());
            return Err(why);
        }
        if let Some(r) = flags.iter().position(|f| f.0 != 0) {
            return Err(format!(
                "rank {r} route: a router row selected an expert outside the model's (a non-finite \
                 router row)"
            ));
        }
        Ok(())
    }

    /// `parallel` (one host thread per rank) or `serial`.
    pub fn issue_name(&self) -> &'static str {
        if self.pool.is_some() {
            "parallel"
        } else {
            "serial"
        }
    }

    /// Start the route log ([`RouteLog`]): every block from here reads its routing back to the
    /// host (a measurement run; it costs a host read a block).
    pub fn log_routes(&mut self) {
        for c in &mut self.cx {
            c.keep_routes = true;
        }
        self.route_log = Some(RouteLog::default());
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
        each_in(pool.as_mut(), cx, f)
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
            Some(p) => {
                let n = cx.len();
                assert_eq!(n, p.threads.len() + 1, "dsv41 tp: one issue thread a rank");
                let (base, ubase, out) = (
                    RankPtr(cx.as_mut_ptr()),
                    RankPtr(us.as_mut_ptr()),
                    rank_slots(n),
                );
                p.run(&|r| {
                    // SAFETY: r < n == us.len() and each rank runs once, on its own thread
                    let (c, u) = unsafe { (base.rank(r), ubase.rank(r)) };
                    *out[r].lock().unwrap_or_else(|e| e.into_inner()) = Some(f(r, c, u));
                })?;
                rank_results(out)
            }
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
        // a small flat block, a large one, a small strided one, and a strided one of megabytes
        // (the packed gather)
        for &(width, rows) in &[
            (4096usize, 1usize),
            (2 * 1024 * 1024, 1),
            (5120, 33),
            (5120, 257),
        ] {
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
        let push = self.transport == TpTransport::DevicePush
            && blks.iter().all(|b| b.bytes() * (n - 1) <= PUSH_LIMIT);
        if self.capturing && !push {
            // the other gathers wait on events and issue copy calls from the host
            return Err("allgather: only a gather inside kernels can be recorded".into());
        }
        // a strided region goes through a contiguous block on both sides (a small one under
        // device-push is pushed instead)
        let packed = self.transport != TpTransport::HostCanonical
            && blks.iter().any(|b| b.rows > 1 && b.width < b.pitch);
        match self.transport {
            TpTransport::DevicePush if push => self.gather_push(bufs, blks)?,
            // every large gather under device-push, and every strided one under peer-pull:
            // a peer copy between blocks the ranks keep
            TpTransport::DevicePush => self.gather_packed(bufs, blks)?,
            TpTransport::PeerPull if packed => self.gather_packed(bufs, blks)?,
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
                let (pub_ev, rel_ev) = (&*pub_ev, &*rel_ev);
                let ctx = cu_ctxs(cx);
                each_in(pool.as_mut(), cx, |r, c| {
                    pub_ev[r].record(&c.ops.st).map_err(de)
                })?;
                let copies = each_in(pool.as_mut(), cx, |c, x| {
                    let mut k = 0u64;
                    // ring order, as in the packed gather
                    for p in (1..n).map(|j| (c + j) % n) {
                        if blks[p].bytes() == 0 {
                            continue;
                        }
                        x.ops.st.wait(&pub_ev[p]).map_err(de)?;
                        copy_peer(&x.ops, addr[c], ctx[p], addr[p], blks[p])?;
                        k += 1;
                    }
                    rel_ev[c].record(&x.ops.st).map_err(de)?;
                    Ok(k)
                })?;
                each_in(pool.as_mut(), cx, |p, x| {
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
                let ctx = cu_ctxs(&self.cx);
                for c in 0..n {
                    for p in 0..n {
                        if p == c || blks[p].bytes() == 0 {
                            continue;
                        }
                        self.cx[c].ops.st.wait(&self.pub_ev[p]).map_err(de)?;
                        let (st_p, st_c) = (&self.cx[p].ops.st, &self.cx[c].ops.st);
                        copy_peer(
                            &self.cx[c].ops,
                            bufs[c].p(st_c).addr(),
                            ctx[p],
                            bufs[p].p(st_p).addr(),
                            blks[p],
                        )?;
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

    /// A gather through contiguous blocks that live across gathers: each producer packs its
    /// region into its kept block with a kernel on its own card and publishes; every consumer
    /// pulls that block with one peer copy a peer, in ring order, into its kept inbox, unpacks
    /// it into its buffer with a kernel on its card and releases; a producer waits for the
    /// releases, so its block is never repacked under a reader. The blocks live across gathers
    /// and grow to the largest gather seen. Every pull is [`copy_peer`]: the same pull by plain
    /// device addresses out of a peer's pool block ran at 2.5-2.9 GB/s a rank on four cards
    /// (it was the copy path of a four-card prefill), the peer call at about 50.
    fn gather_packed(&mut self, bufs: &[&Buf], blks: &[Blk]) -> Result<(), String> {
        let n = self.n();
        let size: Vec<usize> = blks.iter().map(|b| (b.bytes() + 255) & !255).collect();
        // rank c's inbox holds the peers' slots in rank order
        let slot =
            |c: usize, p: usize| -> usize { (0..p).filter(|&q| q != c).map(|q| size[q]).sum() };
        for r in 0..n {
            let need = slot(r, n).max(256);
            if self.big_in[r].as_ref().is_none_or(|b| b.len < need) {
                self.big_in[r] = None;
                self.big_in[r] = Some(self.cx[r].ops.tmp(need)?);
            }
            let need = size[r].max(256);
            if self.pk_keep[r].as_ref().is_none_or(|b| b.len < need) {
                self.pk_keep[r] = None;
                self.pk_keep[r] = Some(self.cx[r].ops.tmp(need)?);
            }
        }
        let addr: Vec<u64> = (0..n)
            .map(|r| bufs[r].p(&self.cx[r].ops.st).addr())
            .collect();
        let kept = |v: &[Option<Buf>], cx: &[Ctx]| -> Vec<u64> {
            (0..n)
                .map(|r| v[r].as_ref().unwrap().p(&cx[r].ops.st).addr())
                .collect()
        };
        let (inb, pk) = (kept(&self.big_in, &self.cx), kept(&self.pk_keep, &self.cx));
        let ctx = cu_ctxs(&self.cx);
        let Self {
            pool,
            cx,
            pub_ev,
            rel_ev,
            ..
        } = self;
        let (pub_ev, rel_ev) = (&*pub_ev, &*rel_ev);
        let move_blk = |ops: &super::Ops, base: u64, b: Blk, packed: u64, unpack: i32| {
            ck("blk_pack", unsafe {
                ffi::memra_v41_blk_pack(
                    (base + b.off as u64) as *mut std::ffi::c_void,
                    b.pitch as i64,
                    b.width as i64,
                    b.rows as i64,
                    packed as *mut std::ffi::c_void,
                    unpack,
                    ops.sv(),
                )
            })
        };
        each_in(pool.as_mut(), cx, |r, c| {
            move_blk(&c.ops, addr[r], blks[r], pk[r], 0)?;
            pub_ev[r].record(&c.ops.st).map_err(de)
        })?;
        let pull = |c: usize, x: &mut Ctx| -> Result<u64, String> {
            let mut k = 0u64;
            // ring order: rank c pulls from c + 1, c + 2, ..., so at any moment a card is the
            // source of one copy and the destination of one
            for p in (1..n).map(|j| (c + j) % n) {
                let bytes = blks[p].bytes();
                if bytes == 0 {
                    continue;
                }
                let to = inb[c] + slot(c, p) as u64;
                x.ops.st.wait(&pub_ev[p]).map_err(de)?;
                copy_peer(&x.ops, to, ctx[p], pk[p], Blk::flat(0, bytes))?;
                move_blk(&x.ops, addr[c], blks[p], to, 1)?;
                k += 1;
            }
            rel_ev[c].record(&x.ops.st).map_err(de)?;
            Ok(k)
        };
        let copies = each_in(pool.as_mut(), cx, pull)?;
        each_in(pool.as_mut(), cx, |p, x| {
            for c in 0..n {
                if c != p {
                    x.ops.st.wait(&rel_ev[c]).map_err(de)?;
                }
            }
            Ok(())
        })?;
        self.stats.copies += copies.iter().sum::<u64>();
        Ok(())
    }

    /// A gather whose ranks each store at most [`PUSH_LIMIT`] bytes, inside kernels (device-push): each
    /// rank's push kernel packs its region into its slot of every peer's inbox and then stores
    /// this gather's sequence number in the slot's word; each rank's pull kernel, queued right
    /// behind it, spins on the words of its own inbox and unpacks the slots into its buffer. Two
    /// launches a rank and nothing else: no event, no copy call, no host wait. The number is the
    /// rank's own count of gathers plus one, kept on its card and stored by its pull, so a
    /// launch holds nothing the host counted and a captured gather replays as it is. Slots
    /// alternate with the number: a rank cannot push gather g + 2 before its pull of g + 1 saw
    /// every peer's word, and a peer pushes g + 1 after its own pull of g, so a slot is never
    /// rewritten under its reader. Larger regions keep the copy-engine gather (peer-pull).
    fn gather_push(&mut self, bufs: &[&Buf], blks: &[Blk]) -> Result<(), String> {
        let n = self.n();
        let word_first = self.push_word_first as i32;
        // a producer's slot of an even gather; an odd gather's is `alt` bytes on
        let slot = |producer: usize| producer * PUSH_SLOT;
        let alt = (n * PUSH_SLOT) as i64;
        let addr: Vec<u64> = (0..n)
            .map(|r| bufs[r].p(&self.cx[r].ops.st).addr())
            .collect();
        let inbox: Vec<u64> = (0..n)
            .map(|r| self.inbox[r].p(&self.cx[r].ops.st).addr())
            .collect();
        let ctr: Vec<u64> = (0..n)
            .map(|r| self.push_ctr[r].p(&self.cx[r].ops.st).addr())
            .collect();
        let Self { pool, cx, .. } = self;
        each_in(pool.as_mut(), cx, |r, c| {
            let st = c.ops.st.clone();
            let (mut out, mut inn) = (ffi::V41Push::default(), ffi::V41Pull::default());
            let mut k = 0;
            for p in (0..n).filter(|&p| p != r) {
                out.w[k] = inbox[p] + slot(r) as u64;
                out.d[k] = out.w[k] + PUSH_HDR as u64;
                inn.w[k] = inbox[r] + slot(p) as u64;
                inn.d[k] = inn.w[k] + PUSH_HDR as u64;
                inn.off[k] = blks[p].off as i64;
                inn.pitch[k] = blks[p].pitch as i64;
                inn.width[k] = blks[p].width as i32;
                inn.rows[k] = blks[p].rows as i32;
                k += 1;
            }
            let b = blks[r];
            ck("gather", unsafe {
                ffi::memra_v41_gather(
                    (addr[r] + b.off as u64) as *const std::ffi::c_void,
                    b.pitch as i64,
                    b.width as i32,
                    b.rows as i32,
                    &out,
                    &inn,
                    k as i32,
                    ctr[r] as *mut u32,
                    alt,
                    addr[r] as *mut std::ffi::c_void,
                    c.route_bad.at(&st, 4).im(),
                    word_first,
                    c.ops.sv(),
                )
            })
        })?;
        self.stats.copies += (n * (n - 1)) as u64;
        Ok(())
    }

    /// The gate's stress of the small gathers: `gathers` gathers over five region shapes with no
    /// host wait between them, every rank setting its region to a pattern of (gather, rank) right
    /// before a gather and counting right after it the words of every region that differ from
    /// the producer's pattern. Returns `(words that differed, ranks whose sticky flag is set)`:
    /// `(0, 0)` when every consumer read what was pushed. `word_first`: the control, the producer
    /// stores the word before the data, which this must catch.
    pub fn gather_stress(
        &mut self,
        gathers: usize,
        word_first: bool,
    ) -> Result<(u64, u64), String> {
        let n = self.n();
        // (words a row, rows) of a rank's region: one word, a one-token hidden gather, an output
        // row, a strided block, the largest push
        let shapes = [
            (1usize, 1usize),
            (288, 7),
            (640, 1),
            (2048, 3),
            (PUSH_LIMIT / (n - 1).max(1) / 4, 1),
        ];
        let was = std::mem::replace(&mut self.push_word_first, word_first);
        let cnt = self.each(|_, c| c.ops.buf(4))?;
        for (si, &(w, rows)) in shapes.iter().enumerate() {
            let pitch_w = n * w + 16;
            let bufs = self.each(|_, c| c.ops.buf(pitch_w * 4 * rows))?;
            let refs: Vec<&Buf> = bufs.iter().collect();
            let blks: Vec<Blk> = (0..n)
                .map(|r| Blk {
                    off: r * w * 4,
                    width: w * 4,
                    pitch: pitch_w * 4,
                    rows,
                })
                .collect();
            let pat = |c: &Ctx, b: &Buf, p: usize, seed: u32, diff: Dp| {
                let st = c.ops.st.clone();
                ck("pat", unsafe {
                    ffi::memra_v41_pat(
                        b.at(&st, p * w * 4).m(),
                        pitch_w as i64,
                        w as i32,
                        rows as i32,
                        seed,
                        diff.im(),
                        c.ops.sv(),
                    )
                })
            };
            for g in 0..gathers.div_ceil(shapes.len()) {
                let seed = |p: usize| ((si * 1_000_003 + g) * 8 + p) as u32;
                self.each(|r, c| pat(c, &bufs[r], r, seed(r), Dp::NULL))?;
                self.allgather(&refs, &blks)?;
                self.each(|r, c| {
                    let st = c.ops.st.clone();
                    (0..n).try_for_each(|p| pat(c, &bufs[r], p, seed(p), cnt[r].p(&st)))
                })?;
                if g % 512 == 511 {
                    self.sync_all()?;
                }
            }
            self.sync_all()?;
        }
        self.push_word_first = was;
        let (mut diff, mut flagged) = (0u64, 0u64);
        for r in 0..n {
            let st = self.cx[r].ops.st.clone();
            diff += bytes_i32(&cnt[r].host(&st)?)[0] as u64;
            let (call, transport) = self.cx[r].route_flags()?;
            flagged += (call != 0 || transport != 0) as u64;
        }
        Ok((diff, flagged))
    }

    /// How fast the cards move bytes to each other (`dsv41-bench p2p`): blocks of 8, 64 and 256
    /// MiB by the copy engine between blocks kept across the reps (`copy`), by the copy engine
    /// between blocks allocated for each rep (`fresh`), both through the gathers' peer call
    /// ([`copy_peer`]), and by a kernel storing into the peer's
    /// memory (`store`, issued on the source's stream), one flow alone and the
    /// flow sets a gather makes: both directions of a pair, one source read by several cards,
    /// one card written by several, the ring, and every pair at once. A line a case: GB/s a
    /// flow, best and median of `reps` (host time from the first issue to every stream drained).
    pub fn p2p_probe(&mut self, reps: usize) -> Result<Vec<String>, String> {
        let n = self.n();
        if n < 2 {
            return Err("p2p: two ranks or more".into());
        }
        let mut sets: Vec<(&str, Vec<(usize, usize)>)> = vec![
            ("one flow 0>1", vec![(0, 1)]),
            ("both directions of a pair", vec![(0, 1), (1, 0)]),
        ];
        if n >= 3 {
            sets.push(("one source, two readers", vec![(0, 1), (0, 2)]));
            sets.push(("one card written by two", vec![(1, 0), (2, 0)]));
        }
        if n >= 4 {
            sets.push(("two disjoint pairs", vec![(0, 1), (2, 3)]));
        }
        if n >= 3 {
            sets.push(("ring", (0..n).map(|r| (r, (r + 1) % n)).collect()));
            let all: Vec<(usize, usize)> = (0..n)
                .flat_map(|a| (0..n).filter(move |&b| b != a).map(move |b| (a, b)))
                .collect();
            sets.push(("every pair", all));
        }
        let mut out = Vec::new();
        let ctx = cu_ctxs(&self.cx);
        for mib in [8usize, 64, 256] {
            let size = mib << 20;
            let src = self.each(|_, c| c.ops.buf(size))?;
            let dst = self.each(|_, c| c.ops.buf(n * size))?;
            let sa: Vec<u64> = (0..n)
                .map(|r| src[r].p(&self.cx[r].ops.st).addr())
                .collect();
            let da: Vec<u64> = (0..n)
                .map(|r| dst[r].p(&self.cx[r].ops.st).addr())
                .collect();
            for (name, flows) in &sets {
                for mech in ["copy", "fresh", "store"] {
                    let mut gbs = Vec::with_capacity(reps);
                    // "fresh": a copy between two blocks allocated for it and never exchanged
                    // before (kept until the case ends, so the pool cannot hand one back)
                    let mut held: Vec<Buf> = Vec::new();
                    for _ in 0..reps {
                        let mut ends = Vec::new();
                        if mech == "fresh" {
                            for &(s, d) in flows {
                                let a = self.cx[s].ops.buf(size)?;
                                let b = self.cx[d].ops.buf(size)?;
                                ends.push((
                                    a.p(&self.cx[s].ops.st).addr(),
                                    b.p(&self.cx[d].ops.st).addr(),
                                ));
                                held.push(a);
                                held.push(b);
                            }
                        }
                        self.sync_all()?;
                        let t0 = Instant::now();
                        for (fi, &(s, d)) in flows.iter().enumerate() {
                            let to = da[d] + (s * size) as u64;
                            if mech == "copy" {
                                copy_peer(&self.cx[d].ops, to, ctx[s], sa[s], Blk::flat(0, size))?;
                            } else if mech == "fresh" {
                                let (from, to) = ends[fi];
                                copy_peer(&self.cx[d].ops, to, ctx[s], from, Blk::flat(0, size))?;
                            } else {
                                ck("blk_pack", unsafe {
                                    ffi::memra_v41_blk_pack(
                                        sa[s] as *mut std::ffi::c_void,
                                        size as i64,
                                        size as i64,
                                        1,
                                        to as *mut std::ffi::c_void,
                                        2,
                                        self.cx[s].ops.sv(),
                                    )
                                })?;
                            }
                        }
                        self.sync_all()?;
                        gbs.push(size as f64 / 1e9 / t0.elapsed().as_secs_f64());
                    }
                    self.sync_all()?;
                    drop(held);
                    let first = gbs[0];
                    gbs.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    out.push(format!(
                        "p2p {mech:5} {mib:3} MiB {name}: {:.1} GB/s a flow best, {:.1} median, {first:.1} first, {} flows, {:.1} GB/s together",
                        gbs[0],
                        gbs[gbs.len() / 2],
                        flows.len(),
                        gbs[gbs.len() / 2] * flows.len() as f64,
                    ));
                }
            }
        }
        Ok(out)
    }

    /// What one gather costs by the size of a rank's region, inside kernels against through the
    /// kept blocks and the copy engine (device-push only): `gathers` gathers back to back with
    /// one wait at the end, microseconds a gather, the median of `reps`. One row of the given
    /// bytes a rank, and the same bytes as 8 strided rows. This is what sets [`PUSH_LIMIT`].
    pub fn gather_probe(&mut self, gathers: usize, reps: usize) -> Result<Vec<String>, String> {
        let n = self.n();
        if n < 2 || self.transport != TpTransport::DevicePush {
            return Err("gather probe: two ranks or more on device-push".into());
        }
        let mut out = Vec::new();
        let mut kib = 1usize;
        while kib << 10 <= ffi::PUSH_MAX {
            for rows in [1usize, 8] {
                let width = (kib << 10) / rows;
                // a rank's region: its columns of `rows` rows, the row a word longer than the
                // ranks' columns so the region is strided
                let pitch = n * width + 16;
                let bufs = self.each(|_, c| c.ops.buf(pitch * rows))?;
                let refs: Vec<&Buf> = bufs.iter().collect();
                let blks: Vec<Blk> = (0..n)
                    .map(|r| Blk {
                        off: r * width,
                        width,
                        pitch,
                        rows,
                    })
                    .collect();
                let mut us = [Vec::new(), Vec::new()];
                for _ in 0..reps.max(1) {
                    for (m, v) in us.iter_mut().enumerate() {
                        self.sync_all()?;
                        let t0 = Instant::now();
                        for _ in 0..gathers.max(1) {
                            if m == 0 {
                                self.gather_push(&refs, &blks)?;
                            } else {
                                self.gather_packed(&refs, &blks)?;
                            }
                        }
                        self.sync_all()?;
                        v.push(t0.elapsed().as_secs_f64() * 1e6 / gathers.max(1) as f64);
                    }
                }
                let med = |v: &mut Vec<f64>| {
                    v.sort_by(|a, b| a.total_cmp(b));
                    v[v.len() / 2]
                };
                let [a, b] = &mut us;
                out.push(format!(
                    "gather {kib:5} KiB a rank in {rows} rows, {n} ranks: in kernels {:.1} us, kept blocks {:.1} us",
                    med(a),
                    med(b)
                ));
            }
            kib *= 2;
        }
        Ok(out)
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
        self.load_layer_named(stm, g, &format!("layers.{id}"), id, mint, routed)
    }

    /// [`Self::load_layer_with`] for the block stored under `prefix` ([`LayerW::load_part_named`]).
    pub fn load_layer_named(
        &self,
        stm: &StModel,
        g: &Geo,
        prefix: &str,
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
                        LayerW::load_part_named(
                            o,
                            stm,
                            g,
                            prefix,
                            id,
                            mint,
                            Part { rank: r, n },
                            routed,
                        )
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
        self.block_forward_kind(lw, h, pre, sp, s, sh, ls, tr, AttnKind::Trunk)
    }

    /// [`Self::block_forward_owned`] with the attention's keys side chosen by `kind`: a trunk
    /// layer's, or a draft stage's over its block ([`AttnKind::Draft`]).
    #[allow(clippy::too_many_arguments)]
    pub fn block_forward_kind(
        &mut self,
        lw: &[LayerW],
        h: Vec<Buf>,
        pre: &[Buf],
        sp: usize,
        s: usize,
        sh: &mut [Shared],
        ls: &mut [LayerState],
        tr: &mut [Trace],
        kind: AttnKind,
    ) -> Result<(Vec<Buf>, Vec<Buf>), String> {
        if sh.len() != self.n() || ls.len() != self.n() {
            return Err("tp block_forward: one share per rank".into());
        }
        self.block_forward_side(lw, h, pre, sp, s, tr, Side::Layer { sh, ls, kind })
    }

    /// One trunk block for a decode call whose rows belong to different sequences: `rows[r]`
    /// is rank r's view of the call's rows (row i is one token of its sequence at its own
    /// position, [`BatchRow`]), `pos[r]` the rows' positions as a device i32 array, `sel[r]` the
    /// batch's selection carried between layers. `h[r]` / `pre[r]` hold the rows' streams side
    /// by side. Everything but the keys side of attention is [`Self::block_forward_kind`]'s; the
    /// keys side is [`Ctx::attn_keys_step`] over each rank's step block (`step`: the blocks and
    /// the layer's place in them), or without one [`Ctx::attn_keys_batch`]. A row's bytes are
    /// those of its sequence's one-token call (`dsv41-spec batch-rows`).
    #[allow(clippy::too_many_arguments)]
    pub fn block_forward_batch(
        &mut self,
        lw: &[LayerW],
        h: Vec<Buf>,
        pre: &[Buf],
        pos: &[Dp],
        rows: &mut [Vec<BatchRow<'_>>],
        sel: &mut [BatchSel],
        step: Option<(&[Dp], &StepLayer)>,
        tr: &mut [Trace],
    ) -> Result<(Vec<Buf>, Vec<Buf>), String> {
        let n = self.n();
        if pos.len() != n || rows.len() != n || sel.len() != n {
            return Err("tp block_forward_batch: one share per rank".into());
        }
        let s = rows[0].len();
        if s == 0 || rows.iter().any(|x| x.len() != s) {
            return Err("tp block_forward_batch: the ranks hold the same rows".into());
        }
        self.block_forward_side(
            lw,
            h,
            pre,
            0,
            s,
            tr,
            Side::Batch {
                pos,
                rows,
                sel,
                step,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn block_forward_side(
        &mut self,
        lw: &[LayerW],
        h: Vec<Buf>,
        pre: &[Buf],
        sp: usize,
        s: usize,
        tr: &mut [Trace],
        side: Side<'_, '_>,
    ) -> Result<(Vec<Buf>, Vec<Buf>), String> {
        let n = self.n();
        if lw.len() != n || h.len() != n || pre.len() != n {
            return Err("tp block_forward: one share per rank".into());
        }
        for (r, w) in lw.iter().enumerate() {
            if w.part != (Part { rank: r, n }) {
                return Err(format!("tp block_forward: rank {r} holds {:?}", w.part));
            }
        }
        // many rows on N ranks: each rank computes its share of the mHC mixes' partial sums
        // (up to the small-row limit each rank runs the one-pass kernel over the whole call)
        let wide = super::WIDE_MIX.load(Ordering::Relaxed).max(super::SMALL_M);
        let split_mix = n > 1 && 8 % n == 0 && s > wide;
        let parts = if split_mix {
            Some(self.hc_parts(lw, &h, s, false)?)
        } else {
            None
        };
        let mix = self.each(|r, c| {
            let w = &lw[r];
            let (_m, p, po, co) = match &parts {
                Some(ps) => c.hc_mix_finish(&ps[r], &w.hc_attn_scale, &w.hc_attn_base, s)?,
                // a few rows: under the attention, on the second stream
                None => {
                    c.hc_mix_under(&w.hc_attn_fn, &w.hc_attn_scale, &w.hc_attn_base, &h[r], s)?
                }
            };
            let x = c.hc_pre_norm(&h[r], &pre[r], &w.attn_norm, s)?;
            Ok((p, po, co, x))
        })?;
        drop(parts);
        let (mut a_pre, mut a_post, mut a_comb, mut xs) = (vec![], vec![], vec![], vec![]);
        for (p, po, co, x) in mix {
            a_pre.push(p);
            a_post.push(po);
            a_comb.push(co);
            xs.push(x);
        }
        let a = match side {
            Side::Layer { sh, ls, kind } => self.attention(lw, &xs, sp, s, sh, ls, tr, kind)?,
            Side::Batch {
                pos,
                rows,
                sel,
                step,
            } => {
                let keys = {
                    let mut io: Vec<(&mut Vec<BatchRow<'_>>, &mut BatchSel)> =
                        rows.iter_mut().zip(sel.iter_mut()).collect();
                    self.each_mut(&mut io, |r, c, (rr, se)| match step {
                        Some((blk, sl)) => c.attn_keys_step(
                            &lw[r],
                            &xs[r],
                            pos[r],
                            StepRef { blk: blk[r], sl },
                            rr,
                            se,
                        ),
                        None => c.attn_keys_batch(&lw[r], &xs[r], pos[r], rr, se),
                    })?
                };
                // the rows' selections are in the keys' own index rows
                let none: Vec<Shared> = (0..n).map(|_| Shared::default()).collect();
                self.attention_run(lw, &keys, &none, 0, s, tr)?
            }
        };
        drop(xs);
        let parts = if split_mix {
            // h[r] becomes the stream after attention
            self.each(|r, c| c.hc_post_in_place(&a[r], &h[r], &a_post[r], &a_comb[r], s))?;
            Some(self.hc_parts(lw, &h, s, true)?)
        } else {
            None
        };
        let ins = FfnIn {
            h: &h,
            a: &a,
            a_pre: &a_pre,
            a_post: &a_post,
            a_comb: &a_comb,
        };
        let few = s <= super::SMALL_M && parts.is_none();
        let (f_pre, m) = if self.graphs.on && few && self.can_capture() {
            (self.ffn_half_graph(lw, &ins, s, tr)?, None)
        } else {
            let (p, m) = self.ffn_half(lw, &ins, s, parts, true, tr)?;
            (p, Some(m))
        };
        drop((a_pre, a_post, a_comb));
        if self.trace {
            for (r, (aa, mm)) in a.into_iter().zip(m.into_iter().flatten()).enumerate() {
                tr[r].attn_out = Some(aa);
                tr[r].moe_out = Some(mm);
            }
        }
        Ok((h, f_pre))
    }

    /// The second half of a block on every rank, from the attention output `a` to the block's
    /// output in `h` (both mHC posts write `h` in place): the post-mix of `a`, the FFN's mixes
    /// (`parts`: their gathered partial sums on many rows; else under the MoE on the second
    /// stream), the collapse and norm, the MoE, its post-mix. Nothing in it depends on a
    /// position. `join`: the attention's mixes are still on the second stream and are joined
    /// here (a recorded span joins them before it starts instead). Returns each rank's new
    /// pre-mix and MoE output.
    fn ffn_half(
        &mut self,
        lw: &[LayerW],
        i: &FfnIn<'_>,
        s: usize,
        parts: Option<Vec<Buf>>,
        join: bool,
        tr: &mut [Trace],
    ) -> Result<(Vec<Buf>, Vec<Buf>), String> {
        let d = self.cx[0].g.dim;
        let FfnIn {
            h,
            a,
            a_pre,
            a_post,
            a_comb,
        } = *i;
        let ffn = self.each(|r, c| {
            let w = &lw[r];
            let (_m, p, po, co) = match &parts {
                Some(ps) => c.hc_mix_finish(&ps[r], &w.hc_ffn_scale, &w.hc_ffn_base, s)?,
                None => {
                    // the attention's mixes ran under it; h[r] becomes the stream after attention
                    if join {
                        c.side_join()?;
                    }
                    c.hc_post_in_place(&a[r], &h[r], &a_post[r], &a_comb[r], s)?;
                    // and the FFN's mixes run under the MoE
                    c.hc_mix_under(&w.hc_ffn_fn, &w.hc_ffn_scale, &w.hc_ffn_base, &h[r], s)?
                }
            };
            let x2 = c.hc_pre_norm(&h[r], &a_pre[r], &w.ffn_norm, s)?;
            let m = c.ops.tmp(s * d * 2)?;
            Ok((p, po, co, x2, m))
        })?;
        drop(parts);
        let (mut f_pre, mut f_post, mut f_comb, mut x2, mut m) =
            (vec![], vec![], vec![], vec![], vec![]);
        for (p, po, co, x, mm) in ffn {
            f_pre.push(p);
            f_post.push(po);
            f_comb.push(co);
            x2.push(x);
            m.push(mm);
        }
        // split experts: the MoE slice of one card (four query slices), so an expert's rows fill
        // whole tiles; the other N-rank paths keep the query slice their gathers were sized for
        let cm = if lw.iter().all(|w| w.split) {
            super::decode::moe_slice(self.cx[0].chunk)
        } else {
            self.cx[0].chunk
        };
        let cm = cm.min(s).max(1);
        for r0 in (0..s).step_by(cm) {
            let rows = cm.min(s - r0);
            self.moe(lw, &x2, r0, rows, &m, tr)?;
        }
        drop(x2);
        if let Some(log) = self.route_log.as_mut() {
            // rank 0's copy of the block's routing, and a one-row block's MoE output
            let out = if s == 1 {
                m[0].host(&self.cx[0].ops.st)?
            } else {
                Vec::new()
            };
            let (ids, w) = if self.trace {
                (tr[0].route_sel.clone(), tr[0].route_w.clone())
            } else {
                for t in tr.iter_mut().skip(1) {
                    t.route_sel.clear();
                    t.route_w.clear();
                }
                (
                    std::mem::take(&mut tr[0].route_sel),
                    std::mem::take(&mut tr[0].route_w),
                )
            };
            log.calls.push((lw[0].id, s));
            log.ids.extend(ids);
            log.w.extend(w);
            log.moe.extend(out);
        }
        self.each(|r, c| {
            c.side_join()?;
            c.hc_post_in_place(&m[r], &h[r], &f_post[r], &f_comb[r], s)
        })?;
        Ok((f_pre, m))
    }

    /// [`Self::ffn_half`] of a few rows through the layer's captured graph
    /// ([`super::graph`]): the first call of a (layer, rows) runs launch by launch, the second
    /// records it, and from then on a call is one graph launch a rank, with the graph pointed at
    /// the call's input buffers first when they sit elsewhere than at the launch before. The
    /// first replays are compared with the launch-by-launch run, byte for byte, in the stream
    /// and in the new pre-mix. The returned pre-mix is a copy of the capture's buffer, which
    /// the span's next replay rewrites.
    fn ffn_half_graph(
        &mut self,
        lw: &[LayerW],
        i: &FfnIn<'_>,
        s: usize,
        tr: &mut [Trace],
    ) -> Result<Vec<Buf>, String> {
        use super::graph::{Piece, SELF_CHECKS, Slot};
        // the attention's mixes: joined on the host, so the span starts with nothing of the
        // second stream outside it
        self.each(|_, c| c.side_join())?;
        // rank 0's load of the layer names the span: the capture holds that load's weights
        let key = (lw[0].uid, s);
        if self.graphs.slots.len() > super::graph::MAX_SPANS {
            // loads that are gone (layers streamed in and dropped) leave their spans behind
            self.graphs.slots.clear();
        }
        // 0: new, 1: seen once, 2: refused, 3: recorded
        let found = match self.graphs.slots.get(&key) {
            None => 0,
            Some(Slot::Seen) => 1,
            Some(Slot::Refused) => 2,
            Some(Slot::Ready(_)) => 3,
        };
        match found {
            0 => {
                self.graphs.slots.insert(key, Slot::Seen);
                return Ok(self.ffn_half(lw, i, s, None, false, tr)?.0);
            }
            2 => return Ok(self.ffn_half(lw, i, s, None, false, tr)?.0),
            1 => {
                let spans = self.ffn_spans(i);
                let cap = self.capture(&spans, |g| Ok(g.ffn_half(lw, i, s, None, false, tr)?.0))?;
                let Some(cap) = cap else {
                    self.graphs.stats.refused += 1;
                    self.graphs.slots.insert(key, Slot::Refused);
                    return Ok(self.ffn_half(lw, i, s, None, false, tr)?.0);
                };
                self.graphs.stats.captures += 1;
                let piece = Piece { cap, checked: 0 };
                self.graphs.slots.insert(key, Slot::Ready(Box::new(piece)));
            }
            _ => {}
        }
        let Some(Slot::Ready(mut p)) = self.graphs.slots.remove(&key) else {
            unreachable!("the span was just found or recorded");
        };
        let res = if p.checked < SELF_CHECKS {
            self.ffn_half_check(lw, i, s, &p, tr).map(|()| true)
        } else {
            let at: Vec<Vec<u64>> = self
                .ffn_spans(i)
                .into_iter()
                .map(|v| v.into_iter().map(|x| x.0).collect())
                .collect();
            self.replay(&p.cap, &at)
        };
        // The call gets its own copy of the new pre-mix. The capture's buffer is rewritten by
        // the span's next replay, and in a forward of several sequences that is the next
        // sequence's call of this layer, issued before this call's next layer reads it.
        let out = match &res {
            Ok(_) => self.each(|r, c| super::clone_buf(&c.ops, &p.cap.out[r])),
            Err(_) => Ok(Vec::new()),
        };
        if let Ok(moved) = res {
            if p.checked < SELF_CHECKS {
                p.checked += 1;
                self.graphs.stats.checks += 1;
            }
            self.graphs.stats.replays += 1;
            self.graphs.stats.moved += moved as u64;
        }
        self.graphs.slots.insert(key, Slot::Ready(p));
        res.and(out)
    }

    /// Each rank's inputs of the second half, (address, bytes), in [`FfnIn`]'s order.
    fn ffn_spans(&self, i: &FfnIn<'_>) -> Vec<Vec<(u64, usize)>> {
        (0..self.n())
            .map(|r| {
                let st = &self.cx[r].ops.st;
                [i.h, i.a, i.a_pre, i.a_post, i.a_comb]
                    .iter()
                    .map(|b| (b[r].p(st).addr(), b[r].len))
                    .collect()
            })
            .collect()
    }

    /// One self-checked replay: the span launch by launch on the call's inputs, and the graph
    /// on copies of all five taken before (so the graph is pointed at addresses it was not
    /// captured on), and the two results (stream and new pre-mix) compared on the host.
    fn ffn_half_check(
        &mut self,
        lw: &[LayerW],
        i: &FfnIn<'_>,
        s: usize,
        p: &super::graph::Piece,
        tr: &mut [Trace],
    ) -> Result<(), String> {
        let copy = |g: &mut Self, b: &[Buf]| g.each(|r, c| super::clone_buf(&c.ops, &b[r]));
        let (h2, a2) = (copy(self, i.h)?, copy(self, i.a)?);
        let (pre2, post2, comb2) = (
            copy(self, i.a_pre)?,
            copy(self, i.a_post)?,
            copy(self, i.a_comb)?,
        );
        let twin = FfnIn {
            h: &h2,
            a: &a2,
            a_pre: &pre2,
            a_post: &post2,
            a_comb: &comb2,
        };
        let (pre_e, _m) = self.ffn_half(lw, i, s, None, false, tr)?;
        let at: Vec<Vec<u64>> = self
            .ffn_spans(&twin)
            .into_iter()
            .map(|v| v.into_iter().map(|x| x.0).collect())
            .collect();
        self.replay(&p.cap, &at)?;
        self.sync_all()?;
        for r in 0..self.n() {
            let st = self.cx[r].ops.st.clone();
            for (what, e, g) in [
                ("stream", &i.h[r], &h2[r]),
                ("pre-mix", &pre_e[r], &p.cap.out[r]),
            ] {
                let (e, g) = (e.host(&st)?, g.host(&st)?);
                if e.len() != g.len() {
                    return Err(format!(
                        "dsv41 graph: layer {} rank {r}: {what} sizes",
                        lw[r].id
                    ));
                }
                if let Some(at) = e.iter().zip(&g).position(|(x, y)| x != y) {
                    return Err(format!(
                        "dsv41 graph: layer {} rank {r}: the replay's {what} differs from the \
                         launch-by-launch run at byte {at} of {}",
                        lw[r].id,
                        e.len()
                    ));
                }
            }
        }
        Ok(())
    }

    /// The mHC mixes' partial sums of `s` rows of the streams `h` with every rank computing its
    /// share: rank r runs the virtual warps `[r * 8 / N, (r + 1) * 8 / N)` of the partial pass over
    /// its whole stream copy (the 8 sums are independent and each reads an eighth of the mix
    /// weights), and the sums are gathered. Each rank then holds all 8, the ones the finish adds.
    fn hc_parts(
        &mut self,
        lw: &[LayerW],
        h: &[Buf],
        s: usize,
        ffn: bool,
    ) -> Result<Vec<Buf>, String> {
        let n = self.n();
        let nb = 8 / n;
        let slot = {
            let g = &self.cx[0].g;
            ((2 + g.hc) * g.hc + 1) * 4
        };
        let parts = self.each(|r, c| {
            let f = if ffn {
                &lw[r].hc_ffn_fn
            } else {
                &lw[r].hc_attn_fn
            };
            c.hc_mix_part(f, &h[r], s, r * nb, nb)
        })?;
        let refs: Vec<&Buf> = parts.iter().collect();
        let blks: Vec<Blk> = (0..n)
            .map(|r| Blk {
                off: r * nb * slot,
                width: nb * slot,
                pitch: 8 * slot,
                rows: s,
            })
            .collect();
        self.allgather(&refs, &blks)?;
        Ok(parts)
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
        kind: AttnKind,
    ) -> Result<Vec<Buf>, String> {
        let n = self.n();
        // the indexer's query rows split across the ranks at prefill (measured: `dsv41-gate idx`,
        // research/.../tp/README.md); one decode row is scored on every rank, with nothing to move
        let split = n > 1 && sp == 0 && s >= 2 * n && lw[0].index_source && lw[0].ratio > 0;
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
                let k = match kind {
                    AttnKind::Trunk => c.attn_keys(&lw[r], &xs[r], sp, s, shr, lsr, trr),
                    AttnKind::Draft { filled } => c.draft_keys(&lw[r], &xs[r], sp, s, filled, lsr),
                };
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
        self.attention_run(lw, &keys, sh, sp, s, tr)
    }

    /// The query side of attention on every rank from the ranks' keys: per query slice each
    /// rank's heads and output groups, the `wo_a` columns gathered, each rank's `wo_b` columns,
    /// the output columns gathered.
    fn attention_run(
        &mut self,
        lw: &[LayerW],
        keys: &[super::AttnKeys],
        shr: &[Shared],
        sp: usize,
        s: usize,
        tr: &mut [Trace],
    ) -> Result<Vec<Buf>, String> {
        let g = self.cx[0].g.clone();
        let (d, hd) = (g.dim, g.head_dim);
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
        // one rank with its experts on the card: the one-card MoE, no host round trip. One rank
        // with the experts behind its cache takes the path below (the cache resolves on the host).
        if n == 1 && !lw[0].experts.is_empty() {
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
        if lw.iter().all(|w| w.split) {
            return self.moe_split(lw, x2, r0, rows, m, tr);
        }
        let trace = self.trace || self.route_log.is_some();
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
                // the rank's range, on the card or behind its cache (an off-card share holds
                // no expert, so its own list would be empty)
                c.moe_plan(&sel, &selw, rows, Part { rank: r, n }.range(ne))
            })?
        };
        let np = rows * topk;
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
                c.experts_tiles(&lw[r], x, &plans[r], p0, p1, sl.p(&st), Some(&tab))?;
                so
            } else {
                c.experts_tiles(&lw[r], x, &plans[r], p0, p1, sl.p(&st), None)?;
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

    /// [`Self::moe`] with every expert split by rows across the ranks ([`LayerW::split`]). Every
    /// rank routes and plans all the experts, computes its hidden columns of every routed pair and
    /// of the shared expert, the hidden columns are gathered, every rank computes its output
    /// columns from the full hidden rows and combines them itself (each token's expert rows in
    /// ascending expert id, then the shared expert: the unsplit order, per column), and the output
    /// columns are gathered. Each step is row-wise, so the bytes are the one-card MoE's. A rank
    /// reads 1/N of every routed expert whatever the routing, nothing is read back to the host,
    /// and the gathers are `[pairs + rows][inter]` and `[rows][dim]` instead of `[pairs][dim]`.
    fn moe_split(
        &mut self,
        lw: &[LayerW],
        x2: &[Buf],
        r0: usize,
        rows: usize,
        m: &[Buf],
        tr: &mut [Trace],
    ) -> Result<(), String> {
        let g = self.cx[0].g.clone();
        let (ne, topk, d, inter) = (g.n_experts, g.topk, g.dim, g.moe_inter);
        let np = rows * topk;
        let trace = self.trace || self.route_log.is_some();
        // a few rows: one bf16 gather of the hidden columns (a gather's latency, not its bytes)
        let quant = rows > super::SMALL_M;
        let parts = {
            let mut trs: Vec<&mut Trace> = tr.iter_mut().collect();
            self.each_mut(&mut trs, |r, c, trr| {
                let st = c.ops.st.clone();
                let x = x2[r].at(&st, r0 * d * 2);
                let hb = c.ops.tmp((np + rows) * inter * 2)?;
                let forked = c.split_shared_fork(&lw[r], x, rows, np, &hb)?;
                let (sel, selw) = c.moe_route(&lw[r], x, rows)?;
                if trace {
                    trr.route_sel
                        .extend_from_slice(&bytes_i32(&host_timed(&sel, &st)?));
                    trr.route_w
                        .extend_from_slice(&bytes_f32(&host_timed(&selw, &st)?));
                }
                let plan = c.moe_plan(&sel, &selw, rows, 0..ne)?;
                c.split_hidden(&lw[r], x, &plan, &hb)?;
                if forked {
                    c.split_shared_join()?;
                }
                // many rows: each rank quantizes its own hidden columns and the codes are
                // gathered, half the bytes of the bf16 rows (the 32-wide quantization blocks lie
                // inside a rank's columns, so the codes are those of the whole row)
                let q = if quant {
                    Some(c.split_quant_own(&lw[r], &hb, np + rows)?)
                } else {
                    None
                };
                Ok((plan, hb, q))
            })?
        };
        let (mut plans, mut hbs, mut qs) = (vec![], vec![], vec![]);
        for (plan, hb, q) in parts {
            plans.push(plan);
            hbs.push(hb);
            qs.push(q);
        }
        if quant {
            let codes: Vec<&Buf> = qs.iter().map(|q| &q.as_ref().unwrap().0).collect();
            let blks: Vec<Blk> = lw
                .iter()
                .map(|w| Blk {
                    off: w.hid0,
                    width: w.hid_l,
                    pitch: inter,
                    rows: np + rows,
                })
                .collect();
            self.allgather(&codes, &blks)?;
            let scales: Vec<&Buf> = qs.iter().map(|q| &q.as_ref().unwrap().1).collect();
            let blks: Vec<Blk> = lw
                .iter()
                .map(|w| Blk {
                    off: w.hid0 / 32,
                    width: w.hid_l / 32,
                    pitch: inter / 32,
                    rows: np + rows,
                })
                .collect();
            self.allgather(&scales, &blks)?;
        } else {
            let refs: Vec<&Buf> = hbs.iter().collect();
            let blks: Vec<Blk> = lw
                .iter()
                .map(|w| Blk {
                    off: w.hid0 * 2,
                    width: w.hid_l * 2,
                    pitch: inter * 2,
                    rows: np + rows,
                })
                .collect();
            self.allgather(&refs, &blks)?;
        }
        self.each(|r, c| {
            let st = c.ops.st.clone();
            let out = m[r].at(&st, r0 * d * 2);
            match &qs[r] {
                Some((q, sc)) => c.split_out_q(&lw[r], &plans[r], q, sc, rows, out),
                None => c.split_out(&lw[r], &plans[r], &hbs[r], rows, out),
            }
        })?;
        let refs: Vec<&Buf> = m.iter().collect();
        let blks: Vec<Blk> = lw
            .iter()
            .map(|w| Blk {
                off: r0 * d * 2 + w.dim0 * 2,
                width: w.shared[1].n * 2,
                pitch: d * 2,
                rows,
            })
            .collect();
        self.allgather(&refs, &blks)?;
        if trace {
            for r in 0..self.n() {
                let st = self.cx[r].ops.st.clone();
                let err = bytes_i32(&plans[r].err.host(&st)?)[0];
                if err != 0 {
                    return Err(format!("rank {r} route: {err} expert ids out of range"));
                }
            }
        }
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
            let lg = bytes_f32(&host_timed(&lg, &st)?);
            // the stream is drained: the flags of the calls since the last read
            Ok((lg, c.route_flags()?))
        })?;
        let (parts, flags): (Vec<Vec<f32>>, Vec<(i32, i32)>) = parts.into_iter().unzip();
        self.route_verdict(&flags)?;
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
    /// The split MoE's first half on this rank for the routed pairs: its hidden columns
    /// `[hid0, hid0 + hid_l)` of every pair, rows `[0, np)` of `hb` (bf16 `[np + rows][inter]`).
    /// w1 and w3 of the rank's hidden rows, then the SwiGLU, which is per element. The shared
    /// expert's rows are [`split_hidden_shared`].
    pub fn split_hidden(
        &mut self,
        lw: &LayerW,
        x: Dp,
        plan: &super::MoePlanDev,
        hb: &Buf,
    ) -> Result<(), String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let (d, inter, np, hn, h0) = (g.dim, g.moe_inter, plan.np, lw.hid_l, lw.hid0);
        if plan.bm > super::SMALL_M {
            // many rows: the input quantized once a token, each tile reading its pairs' tokens
            // through the plan's pair-to-token list (a sixth of the quantization of the pairs'
            // gathered copies, and the same codes)
            let rows = np / g.topk;
            let aq = self.ops.tmp(rows * d)?;
            let as_ = self.ops.tmp(rows * d / 32)?;
            self.ops
                .act_quant(x, d, Dp::NULL, rows, d, aq.p(&st), as_.p(&st))?;
            let gb = self.ops.tmp(np * hn * 2)?;
            let ub = self.ops.tmp(np * hn * 2)?;
            let (wfmt, kdiv) = split_fmt(lw)?;
            for (proj, out) in [(0, &gb), (2, &ub)] {
                ck("gemm_f8_tiles_rows", unsafe {
                    ffi::memra_v41_gemm_f8_tiles_rows(
                        aq.p(&st).c(),
                        d as i64,
                        as_.p(&st).c(),
                        (d / 32) as i64,
                        plan.ptok.p(&st).i(),
                        plan.tiles.p(&st).c(),
                        plan.ntiles.p(&st).i(),
                        plan.max_tiles as i32,
                        lw.exp_tab.p(&st).c(),
                        proj,
                        (d / 2) as i64,
                        (d / kdiv) as i64,
                        wfmt,
                        out.p(&st).m(),
                        hn as i64,
                        0,
                        hn as i32,
                        d as i32,
                        plan.bm as i32,
                        self.ops.sv(),
                    )
                })?;
            }
            return ck("swiglu", unsafe {
                ffi::memra_v41_swiglu(
                    gb.p(&st).m(),
                    ub.p(&st).m(),
                    hn as i64,
                    plan.pw.p(&st).f(),
                    g.swiglu_limit,
                    hb.at(&st, h0 * 2).m(),
                    inter as i64,
                    np as i32,
                    hn as i32,
                    self.ops.sv(),
                )
            });
        }
        // a few rows: w1, w3 and the SwiGLU in one launch over the pairs' gathered rows
        let aq = self.ops.tmp(np * d)?;
        let as_ = self.ops.tmp(np * d / 32)?;
        self.ops
            .act_quant(x, d, plan.ptok.p(&st), np, d, aq.p(&st), as_.p(&st))?;
        let (wfmt, kdiv) = split_fmt(lw)?;
        ck("swiglu_tiles", unsafe {
            ffi::memra_v41_swiglu_tiles(
                aq.p(&st).c(),
                d as i64,
                as_.p(&st).c(),
                (d / 32) as i64,
                plan.tiles.p(&st).c(),
                plan.ntiles.p(&st).i(),
                plan.max_tiles as i32,
                lw.exp_tab.p(&st).c(),
                (d / 2) as i64,
                (d / kdiv) as i64,
                wfmt,
                plan.pw.p(&st).f(),
                g.swiglu_limit,
                hb.at(&st, h0 * 2).m(),
                inter as i64,
                hn as i32,
                d as i32,
                plan.bm as i32,
                self.ops.sv(),
            )
        })
    }

    /// The split MoE's second half on this rank: its output columns `[dim0, dim0 + n)` of every
    /// pair and of the shared expert from the gathered hidden rows `hb`, combined into the same
    /// columns of `out` (`[rows][dim]`).
    pub fn split_out(
        &mut self,
        lw: &LayerW,
        plan: &super::MoePlanDev,
        hb: &Buf,
        rows: usize,
        out: Dp,
    ) -> Result<(), String> {
        let st = self.ops.st.clone();
        let (inter, n) = (self.g.moe_inter, plan.np + rows);
        let q = self.ops.tmp(n * inter)?;
        let sc = self.ops.tmp(n * inter / 32)?;
        self.ops
            .act_quant(hb.p(&st), inter, Dp::NULL, n, inter, q.p(&st), sc.p(&st))?;
        self.split_out_q(lw, plan, &q, &sc, rows, out)
    }

    /// The e4m3 codes and ue8m0 scales of this rank's hidden columns of the first `n` rows of
    /// `hb`, in their columns of `[n][inter]` codes and `[n][inter / 32]` scales (the other
    /// columns are the peers').
    pub fn split_quant_own(
        &mut self,
        lw: &LayerW,
        hb: &Buf,
        n: usize,
    ) -> Result<(Buf, Buf), String> {
        let st = self.ops.st.clone();
        let (inter, hn, h0) = (self.g.moe_inter, lw.hid_l, lw.hid0);
        let q = self.ops.tmp(n * inter)?;
        let sc = self.ops.tmp(n * inter / 32)?;
        ck("act_quant", unsafe {
            ffi::memra_v41_act_quant(
                hb.at(&st, h0 * 2).c(),
                inter as i64,
                std::ptr::null(),
                q.at(&st, h0).m(),
                inter as i64,
                sc.at(&st, h0 / 32).m(),
                (inter / 32) as i64,
                std::ptr::null_mut(),
                0,
                n as i32,
                hn as i32,
                self.ops.sv(),
            )
        })?;
        Ok((q, sc))
    }

    /// [`Self::split_out`] from the hidden rows' codes `q` `[np + rows][inter]` and scales `sc`
    /// `[np + rows][inter / 32]`: the pairs' rows first, then the shared expert's.
    pub fn split_out_q(
        &mut self,
        lw: &LayerW,
        plan: &super::MoePlanDev,
        q: &Buf,
        sc: &Buf,
        rows: usize,
        out: Dp,
    ) -> Result<(), String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let (d, inter, np, dn) = (g.dim, g.moe_inter, plan.np, lw.shared[1].n);
        let o = self.ops.tmp((np + rows) * dn * 2)?;
        self.split_tiles(lw, plan, q, sc, inter, 1, o.p(&st), dn)?;
        self.ops.gemm_q(
            q.at(&st, np * inter),
            sc.at(&st, np * inter / 32),
            rows,
            &lw.shared[1],
            o.at(&st, np * dn * 2),
            dn,
            false,
        )?;
        ck("moe_combine", unsafe {
            ffi::memra_v41_moe_combine(
                o.p(&st).c(),
                dn as i64,
                plan.order.p(&st).i(),
                g.topk as i32,
                o.at(&st, np * dn * 2).c(),
                dn as i64,
                Dp(out.addr() + (lw.dim0 * 2) as u64).m(),
                d as i64,
                rows as i32,
                dn as i32,
                self.ops.sv(),
            )
        })
    }

    /// One projection of the split experts over the plan's tile list: `nout` rows of every routed
    /// expert's weight (this rank's slice) against the quantized rows `a`, into `c` `[np][nout]`.
    #[allow(clippy::too_many_arguments)]
    fn split_tiles(
        &self,
        lw: &LayerW,
        plan: &super::MoePlanDev,
        a: &Buf,
        a_s: &Buf,
        k: usize,
        proj: i32,
        c: Dp,
        nout: usize,
    ) -> Result<(), String> {
        let st = self.ops.st.clone();
        let (wfmt, kdiv) = split_fmt(lw)?;
        ck("gemm_f8_tiles", unsafe {
            ffi::memra_v41_gemm_f8_tiles(
                a.p(&st).c(),
                k as i64,
                a_s.p(&st).c(),
                (k / 32) as i64,
                plan.tiles.p(&st).c(),
                plan.ntiles.p(&st).i(),
                plan.max_tiles as i32,
                lw.exp_tab.p(&st).c(),
                proj,
                (k / 2) as i64,
                (k / kdiv) as i64,
                wfmt,
                c.m(),
                nout as i64,
                0,
                nout as i32,
                k as i32,
                plan.bm as i32,
                self.ops.sv(),
            )
        })
    }

    /// The split MoE's shared expert on the side stream when the forward is a few rows (it reads
    /// only `x`, so it runs under the router's chain and the plan), else on the main stream.
    /// Returns whether the caller owes a [`Self::split_shared_join`] before `hb` is read.
    pub fn split_shared_fork(
        &mut self,
        lw: &LayerW,
        x: Dp,
        rows: usize,
        np: usize,
        hb: &Buf,
    ) -> Result<bool, String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        if rows > super::SIDE_LIMITS[1].load(Ordering::Relaxed) {
            split_hidden_shared(&self.ops, &mut self.scr, &g, lw, x, rows, np, hb)?;
            return Ok(false);
        }
        if self.side.is_none() {
            self.side = Some(super::Side::new(&st)?);
        }
        let side = self.side.as_mut().unwrap();
        side.fork(&st)?;
        split_hidden_shared(&side.ops, &mut side.scr, &g, lw, x, rows, np, hb)?;
        Ok(true)
    }

    /// The main stream waits for the side stream's shared expert.
    pub fn split_shared_join(&self) -> Result<(), String> {
        match &self.side {
            Some(s) => s.join(&self.ops.st),
            None => Err("split MoE: no side stream to join".into()),
        }
    }

    /// The shared FP8 expert of `rows` rows of `x`: bf16 `[rows][dim]`.
    pub fn shared_expert(&mut self, lw: &LayerW, x: Dp, rows: usize) -> Result<Buf, String> {
        let g = self.g.clone();
        shared_expert_on(&self.ops, &mut self.scr, &g, lw, x, rows)
    }
}

/// [`Ctx::shared_expert`] on `ops`'s stream with `scr` as its activation scratch. `w1` and `w3`
/// read the same rows, so the rows are quantized once for both.
pub fn shared_expert_on(
    ops: &super::Ops,
    scr: &mut super::Scratch,
    g: &Geo,
    lw: &LayerW,
    x: Dp,
    rows: usize,
) -> Result<Buf, String> {
    let st = ops.st.clone();
    let (d, inter) = (g.dim, g.moe_inter);
    let [s1, s2, s3] = &lw.shared;
    let gb = ops.tmp(rows * inter * 2)?;
    let ub = ops.tmp(rows * inter * 2)?;
    let hb = ops.tmp(rows * inter * 2)?;
    let out = ops.tmp(rows * d * 2)?;
    let (q, qs) = scr.actq(ops, rows, d)?;
    ops.act_quant(x, d, Dp::NULL, rows, d, q, qs)?;
    ops.gemm_q(q, qs, rows, s1, gb.p(&st), inter, false)?;
    ops.gemm_q(q, qs, rows, s3, ub.p(&st), inter, false)?;
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
            ops.sv(),
        )
    })?;
    ops.linear_q(hb.p(&st), inter, rows, s2, out.p(&st), d, false, scr)?;
    Ok(out)
}

/// The tile kernels' weight format code and scale divisor of a rank's FP4 experts.
fn split_fmt(lw: &LayerW) -> Result<(i32, usize), String> {
    match lw.experts.first().map(|w| w[0].fmt) {
        Some(super::QFmt::Mxfp4) => Ok((1, 32)),
        Some(super::QFmt::Nvfp4Minted { .. }) => Ok((2, 16)),
        _ => Err("split experts: no FP4 experts on the rank".into()),
    }
}

/// The split MoE's first half on a rank for the shared expert, on `ops`' stream: its hidden columns
/// `[hid0, hid0 + hid_l)` of every token, rows `[np, np + rows)` of `hb` (bf16 `[np + rows][inter]`).
#[allow(clippy::too_many_arguments)]
pub fn split_hidden_shared(
    ops: &super::Ops,
    scr: &mut super::Scratch,
    g: &Geo,
    lw: &LayerW,
    x: Dp,
    rows: usize,
    np: usize,
    hb: &Buf,
) -> Result<(), String> {
    let st = ops.st.clone();
    let (d, inter, hn, h0) = (g.dim, g.moe_inter, lw.hid_l, lw.hid0);
    let (q, qs) = scr.actq(ops, rows, d)?;
    ops.act_quant(x, d, Dp::NULL, rows, d, q, qs)?;
    let gs = ops.tmp(rows * hn * 2)?;
    let us = ops.tmp(rows * hn * 2)?;
    ops.gemm_q(q, qs, rows, &lw.shared[0], gs.p(&st), hn, false)?;
    ops.gemm_q(q, qs, rows, &lw.shared[2], us.p(&st), hn, false)?;
    ck("swiglu", unsafe {
        ffi::memra_v41_swiglu(
            gs.p(&st).m(),
            us.p(&st).m(),
            hn as i64,
            std::ptr::null(),
            g.swiglu_limit,
            hb.at(&st, np * inter * 2 + h0 * 2).m(),
            inter as i64,
            rows as i32,
            hn as i32,
            ops.sv(),
        )
    })
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
