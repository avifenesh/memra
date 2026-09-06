//! TP-N TRANSPORT — `MEMRA_TP_TRANSPORT` (born lane/glm5-tp-transport 2026-09-01 as
//! `MEMRA_GLM5_TP_TRANSPORT`, rank-widened by lane/glm5-composition 2026-09-01, generalized
//! lane/glm5-extract2; the family alias stays honored).
//!
//! ENGINE-GENERIC BY CONSTRUCTION, and this is a claim about content rather than a hope: not
//! one function below knows a layer type, a mixer, an expert or a model family. The whole
//! module is rank indices, publication events, and named point-to-point hop shapes over dense
//! blocks. The glm5 TP-N shard walk is today's ONLY CONSUMER (it owns its rank geometry, its
//! owner-first law and its shard maps); an hy3/step TP walk will arm the same transport once
//! it exists, and `pp.rs`'s `BoundarySlot` is where the shape came from.
//!
//! WHAT THIS IS. The data-movement layer under a TP execution program (`MEMRA_GLM5_TP` is the
//! first). The TP
//! seam's arithmetic is column-parallel-over-gather: every cross-rank hop is PURE MOVEMENT
//! (the one arithmetic site is the MoE slot-ordered combine, and it stays on root). So the
//! transport is swappable WITHOUT changing a single bit, and this module is the seam where
//! the swap happens — one named function per hop shape, two arms behind each:
//!
//!   * `host-canonical` (v1, the DEFAULT): every hop is `dtoh` -> host buffer -> `htod`.
//!     `Engine::dtoh` ends in `stream().synchronize()`, so each leg is a full stream drain
//!     plus a PCIe host round trip. This is what every banked glm5 TP number was measured
//!     on (`research/glm53-flash-bringup-20260827/tp2-battery-20260831/RESULTS.md` cell 2:
//!     "v1 transport = host-canonical (every boot announces `transport=host-canonical` on
//!     all four seams)").
//!   * `peer-pull`: the hop is a DEVICE copy issued on the CONSUMER's stream, reading the
//!     producer's buffer, ordered by a publication event. Zero host legs, zero stream
//!     drains. PULL and not push by design (see the transport laws below).
//!
//! RANK WIDENING (lane/glm5-composition). The 2026-09-01 original of this module hardcoded
//! two ranks (`Rank::Root`/`Rank::Peer`). Ranks are now plain indices (`0` = root, the
//! owner-first law unchanged) and every hop shape takes per-rank parts; at two ranks each
//! arm reproduces the v1 walk HOP FOR HOP (same primitive count, same order, same census),
//! which is what keeps the banked TP-2 numbers comparable across this widening. The
//! publication link carries one pub event and one release event PER RANK — `cuEventRecord`
//! overwrites and `cuStreamWaitEvent` captures state at call time, so the record-then-wait
//! pairs a single host thread issues in program order stay correct at any rank count.
//!
//! WHY PULL. On the RTX PRO 6000 PCIe fabric a peer transport's direction is not symmetric
//! and the consumer-issued shape wins twice:
//!
//!   1. `research/pro6000-multicard-research-20260901/RESEARCH.md` §2.2b measures
//!      "approximately 52 GB/s for peer reads and 2.6 GB/s for peer writes, so all three
//!      communication phases use pull transfers" (b12x#139, 16x PRO 6000) — 110 us push vs
//!      26 us pull for the same collective. That asymmetry is scoped to SM-ISSUED
//!      (kernel load/store) traffic; a copy-engine `cuMemcpyDtoDAsync` measures ~54-56 GB/s
//!      in BOTH directions (§2.2, §3.2 `read_ce`/`write_ce`). We use the copy engine, so the
//!      20x is NOT our headline reason — it is the reason a future fused collective must be
//!      pull-shaped, and the reason we never build the push twin.
//!   2. The ORDERING reason, which IS our headline reason and is specific to our tax. When
//!      the consumer issues the copy, the consuming kernel on that same stream is ordered
//!      after it for free. A push (producer-issued, the `pp.rs` `BoundaryTransport::Peer`
//!      shape) needs a second event + wait to publish into the consumer's stream. Our
//!      measured tax is a ROUND-TRIP and LAUNCH count tax, not a byte tax
//!      (RESEARCH.md §2.6, §6.3c; three upstream maintainers agree, §1.5g), so removing
//!      launches per hop is the lever, and pull removes one ordering primitive per hop.
//!
//! NO ATOMICS, ANYWHERE. Every SM120 PCIe pair reports `NativeAtomicSupported=0`
//! (RESEARCH.md §2.1): "CAS on peer memory silently loses barrier tokens under PCIe load"
//! and it fails LOAD-DEPENDENTLY, so microbenchmarks pass. This transport uses only
//! copy-engine copies and CUDA events — no peer flag polling, no compare-and-swap, no
//! payload-value sentinels (RESEARCH.md §6.13: a payload-sentinel prototype caused a
//! launch-day production stall). If a fused device-resident collective is ever built here,
//! §6.13's protocol is pre-specified and its bug list is pre-registered.
//!
//! WHAT THIS IS NOT. This is not a collective. There is no reduce anywhere in the glm5 TP
//! program (that is what makes decode BYTE identity the gate bar instead of a tolerance
//! band), so there is no all-reduce to fuse, no ring, no barrier. Everything here is a
//! point-to-point copy of a dense block plus a byte-preserving strided placement.
//!
//! FAIL-CLOSED. Arming `peer-pull` runs a byte-integrity pull ladder through the EXACT
//! primitive the walk uses, in EVERY ordered rank pair, before any layer is sharded; a
//! single differing byte refuses the load by name. That is a TRANSFER-tier check by
//! RESEARCH.md §2.4's ladder and it is honest about it: it proves our copy path moves the
//! right bytes, and it does NOT prove a kernel peer-dereference works (only a
//! `simpleP2P`-class KERNEL peer read does — banked separately as the lane's
//! `peer-read-probe.cu`, run by `HEALTH.sh` at every box window).
//!
//! Engagement markers: the TAG IS THE CALLER'S (the phase-1 `spec_phase` pattern) — glm5
//! passes `"glm5-tp-transport"`, so every banked receipt line from lane/glm5-tp-transport and
//! lane/glm5-composition keeps its exact bytes while a second family gets its own marker.
//! Counters below are the per-token instrument and are the TRANSPORT's, not the family's.

// lane/clippy-zero-restore-20260901: this transport's exact code shape is receipt-bound
// (lane/glm5-composition keeps its exact bytes, header above); index loops stay as gated.
#![allow(clippy::needless_range_loop)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use cudarc::driver::{CudaEvent, CudaSlice};

use crate::Engine;

/// Rank index. `0` is root (the model's own engine — the owner-first rank law); `1..ranks`
/// are the peer engines in `MEMRA_GLM5_TP` device order.
pub const ROOT: usize = 0;

// ------------------------------------------------------------------------------------------
// Flag
// ------------------------------------------------------------------------------------------

/// Which transport the armed `MEMRA_GLM5_TP` seam moves its bytes with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TpTransport {
    /// v1: `dtoh` -> host -> `htod` per hop. Every banked glm5 TP number.
    HostCanonical,
    /// Consumer-issued device peer copy per hop, ordered by a publication event.
    PeerPull,
    /// PRODUCER-issued device peer store per hop, ordered by a publication event.
    ///
    /// The difference from `peer-pull` is direction, and it is the whole point. A pull is a
    /// consumer-side read, so the reading rank cannot start until the producing rank has
    /// finished: at ~90 joins per token that keeps the two cards single-file, which is why
    /// peer-pull measured WORSE than the host bounce rather than better. A push is issued on
    /// the producer's own stream, so each rank hands its bytes over and carries on, and the
    /// consumer's stream is ordered after by the same publication event. Movement only: every
    /// hop this transport serves is a fan-out, a gather or a concat, so the arm is
    /// byte-identical to `host-canonical` by construction, exactly as `peer-pull` is.
    DevicePush,
}

impl TpTransport {
    /// The receipt spelling that goes in every announce line and every gate log.
    pub fn name(self) -> &'static str {
        match self {
            Self::HostCanonical => "host-canonical",
            Self::PeerPull => "peer-pull",
            Self::DevicePush => "device-push",
        }
    }
}

/// The general fleet flag.
pub const TRANSPORT_ENV: &str = "MEMRA_TP_TRANSPORT";
/// The family alias lane/glm5-tp-transport shipped. Still honored — its gate arms, its box
/// scripts and every banked transport receipt set it. Never silently dead.
pub const TRANSPORT_ENV_GLM5: &str = "MEMRA_GLM5_TP_TRANSPORT";

/// Parse a transport value. Pure so the law is unit-testable without touching the process
/// environment. `flag` is the name the operator actually set, so a refusal names it.
///
/// DEFAULT IS `host-canonical` (OFF), by decision, not by accident: on the day this flag
/// landed the peer-pull arm had zero receipts on real peer hardware (the rig is a single
/// card — `LAW:rig-exactness-only` — so its gate arms run the code path over N contexts on
/// ONE device and can only prove bit-preservation, never fabric engagement). Unmeasured
/// behaviour does not default ON. The default flips in the same commit as the box window's
/// interleaved re-price receipt, and the FLAGS.md row carries both arms plus the rollback
/// seam (`=0`).
pub fn parse_transport(flag: &str, value: Option<&str>) -> Result<TpTransport, String> {
    match value {
        None | Some("") | Some("0") | Some("host-canonical") => Ok(TpTransport::HostCanonical),
        Some("1") | Some("peer-pull") => Ok(TpTransport::PeerPull),
        Some("2") | Some("device-push") => Ok(TpTransport::DevicePush),
        Some(other) => Err(format!(
            "{flag}={other:?} is not a known transport \
             (host-canonical | 0 = the v1 host-staged arm, the default; peer-pull | 1 = the \
             consumer-issued device peer copy arm; device-push | 2 = the producer-issued \
             device peer store arm)"
        )),
    }
}

/// Resolve the general name and the family alias to ONE armed `(name, value)`, then parse.
///
/// This is a VALUED flag resolved ONCE at arm time, not a per-call boolean door, so a
/// disagreeing pair REFUSES the load naming both names — the `ep_map::resolve_ep_map_env`
/// stance. (Falling closed, which `lib.rs::alias_door_from` does for per-call doors, is only
/// correct where an abort would kill live sessions; arming happens before any session exists,
/// and a load that silently picked a transport the operator did not choose is worse than a
/// refused load.)
pub fn resolve_transport(
    general: Option<&str>,
    alias: Option<&str>,
) -> Result<(TpTransport, &'static str), String> {
    let (armed, value) = match (general, alias) {
        (Some(g), Some(a)) if g != a => {
            return Err(format!(
                "{TRANSPORT_ENV}={g:?} and {TRANSPORT_ENV_GLM5}={a:?} disagree — the alias and \
                 the general flag name ONE transport (unset one); refused rather than silently \
                 picking a precedence winner"
            ));
        }
        (Some(g), _) => (TRANSPORT_ENV, Some(g)),
        (None, Some(a)) => (TRANSPORT_ENV_GLM5, Some(a)),
        (None, None) => (TRANSPORT_ENV, None),
    };
    Ok((parse_transport(armed, value)?, armed))
}

/// Every transport name currently SET, for ROLLBACK ADVICE only — never for policy.
///
/// Why advice needs this and the armed name is not enough: a disagreeing pair is already
/// refused at resolve time, so by the time anything can fail we are in one of two states —
/// exactly one name set, or BOTH set to the same value. In the second state the armed name is
/// the general one, so telling the operator to zero only that leaves the alias still asking
/// for `peer-pull`, and THAT disagreement refuses the load. Advice that creates a second
/// failure is worse than no advice.
pub fn set_transport_names() -> Vec<&'static str> {
    [TRANSPORT_ENV, TRANSPORT_ENV_GLM5]
        .into_iter()
        .filter(|k| std::env::var_os(k).is_some_and(|v| !v.is_empty()))
        .collect()
}

/// Render [`set_transport_names`] as `NAME=0 [NAME=0]` — what to actually type.
fn rollback_advice() -> String {
    let names = set_transport_names();
    if names.is_empty() {
        return format!("{TRANSPORT_ENV}=0");
    }
    names
        .iter()
        .map(|n| format!("{n}=0"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Live read of [`resolve_transport`]. Returns the transport and the ARMED FLAG NAME, so
/// downstream refusals and peer-access grants cite the flag the operator typed.
pub fn transport_env() -> Result<(TpTransport, &'static str), String> {
    resolve_transport(
        std::env::var(TRANSPORT_ENV).ok().as_deref(),
        std::env::var(TRANSPORT_ENV_GLM5).ok().as_deref(),
    )
}

// ------------------------------------------------------------------------------------------
// Instrument — the per-token movement census (lane stage 1b)
// ------------------------------------------------------------------------------------------
//
// RESEARCH.md's shortlist item 2 asks for "our per-token collective COUNT and total BYTES on
// the TP arm" and warns that bytes alone cannot explain the tax. These counters answer both
// halves from the live walk instead of from arithmetic, in BOTH arms, so an A/B reads the
// movement change directly rather than inferring it from tok/s.

/// Host legs: one per `dtoh` or `htod` a TP hop performed. Pinned 0 on the peer-pull arm.
pub static TP_HOST_LEGS: AtomicU64 = AtomicU64::new(0);

/// Host legs that ended in a full stream drain (`Engine::dtoh` synchronizes). This is the
/// count the 13-18 ms/token v1 join tax reconstructs from. Pinned 0 on the peer-pull arm.
pub static TP_HOST_SYNCS: AtomicU64 = AtomicU64::new(0);

/// Consumer-issued cross-rank device copies. Pinned 0 on the host-canonical arm.
pub static TP_PEER_PULLS: AtomicU64 = AtomicU64::new(0);

/// Producer-issued cross-rank device stores. Pinned 0 on both other arms.
pub static TP_DEVICE_PUSHES: AtomicU64 = AtomicU64::new(0);

/// Publication events recorded/awaited to order a cross-rank copy. Pinned 0 on the
/// host-canonical arm (whose ordering is the host sync itself).
pub static TP_PUB_EVENTS: AtomicU64 = AtomicU64::new(0);

/// Same-rank device copies a hop issued (own-part placements and dense block moves). Present
/// in both arms; the host-canonical arm reaches its own part through the host instead.
pub static TP_LOCAL_COPIES: AtomicU64 = AtomicU64::new(0);

/// Total bytes a TP hop moved across a rank boundary, both arms, counted once per crossing
/// (a host-canonical hop crosses PCIe twice and is charged twice — that IS its cost).
pub static TP_XFER_BYTES: AtomicU64 = AtomicU64::new(0);

macro_rules! snapshot_fns {
    ($($name:ident => $counter:ident),* $(,)?) => {
        $(
            /// Snapshot for a gate's before/after delta.
            pub fn $name() -> u64 { $counter.load(Ordering::Relaxed) }
        )*
    };
}

snapshot_fns! {
    tp_host_legs => TP_HOST_LEGS,
    tp_host_syncs => TP_HOST_SYNCS,
    tp_peer_pulls => TP_PEER_PULLS,
    tp_device_pushes => TP_DEVICE_PUSHES,
    tp_pub_events => TP_PUB_EVENTS,
    tp_local_copies => TP_LOCAL_COPIES,
    tp_xfer_bytes => TP_XFER_BYTES,
}

/// One line carrying every movement counter, for a gate log or a box-window receipt.
pub fn transport_census_line(tag: &str, transport: TpTransport) -> String {
    format!(
        "[{tag}] census transport={} host_legs={} host_syncs={} peer_pulls={} \
         device_pushes={} pub_events={} local_copies={} xfer_bytes={}",
        transport.name(),
        tp_host_legs(),
        tp_host_syncs(),
        tp_peer_pulls(),
        tp_device_pushes(),
        tp_pub_events(),
        tp_local_copies(),
        tp_xfer_bytes(),
    )
}

fn charge_host_leg(bytes: usize, sync: bool) {
    TP_HOST_LEGS.fetch_add(1, Ordering::Relaxed);
    if sync {
        TP_HOST_SYNCS.fetch_add(1, Ordering::Relaxed);
    }
    TP_XFER_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

fn charge_peer_pull(bytes: usize) {
    TP_PEER_PULLS.fetch_add(1, Ordering::Relaxed);
    TP_XFER_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

fn charge_device_push(bytes: usize) {
    TP_DEVICE_PUSHES.fetch_add(1, Ordering::Relaxed);
    TP_XFER_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

fn charge_local_copy() {
    TP_LOCAL_COPIES.fetch_add(1, Ordering::Relaxed);
}

// ------------------------------------------------------------------------------------------
// The publication link
// ------------------------------------------------------------------------------------------

/// The peer-pull ordering primitives, built once per TP runtime and REUSED per hop.
///
/// One publication event and one release event PER RANK, not per hop or per pair:
/// `cuEventRecord` overwrites, and `cuStreamWaitEvent` captures the event's state at call
/// time, so a single host thread issuing record-then-wait in program order gets correct
/// ordering with `2 * ranks` events total. Creating an event per hop would put a driver
/// allocation on the per-token path — the exact class of cost this lane exists to remove.
///
/// The RELEASE (write-after-read) half is NOT optional. Building this transport with only
/// the publication half is exactly the defect the rig gate caught on its FIRST run: arm X1
/// (peer-pull, sequential EP walk) failed DECODE SELF-CONSISTENCY at step 17 — two
/// repetitions of the same greedy walk diverged — because every per-slot expert row is a
/// fresh stream-ordered allocation on the peer whose async free was enqueued on the PEER
/// stream while the ROOT stream's pull was still reading it. The step seam banks the same
/// hazard as a keepalive comment (`tp.rs:7831-7834`). And note the more valuable half of
/// that gate run: arm X2 (peer-pull composed with the EP dispatch diet) PASSED EVERY BAR on
/// the same broken build — a VACUOUS GREEN; the undieted arm is the one that can see it,
/// which is why both arms are gated and why the sequential walk is not retired.
pub struct PeerPullLink {
    /// `pub_ev[r]` is recorded on rank r's stream; awaited by a consumer before it reads
    /// rank r's memory.
    pub_ev: Vec<CudaEvent>,
    /// `rel_ev[r]` is recorded on rank r's stream AFTER it has finished READING another
    /// rank's buffer; the PRODUCING rank waits on it before proceeding, which is what stops
    /// its stream-ordered allocator from recycling the buffer the reader's copy engine is
    /// still reading.
    rel_ev: Vec<CudaEvent>,
}

impl PeerPullLink {
    pub fn new(engines: &[&Engine]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut pub_ev = Vec::with_capacity(engines.len());
        let mut rel_ev = Vec::with_capacity(engines.len());
        for e in engines {
            pub_ev.push(e.ctx().new_event(None)?);
            rel_ev.push(e.ctx().new_event(None)?);
        }
        Ok(Self { pub_ev, rel_ev })
    }
}

/// Everything a hop needs: the rank engines (index = rank, `[0]` = root), the transport arm,
/// and (on the peer-pull arm) the publication link. Built by the caller per hop from the TP
/// runtime, so this module never has to own the runtime type and the borrow shapes at the
/// call sites stay exactly as they are today.
pub struct Hop<'a> {
    pub engines: Vec<&'a Engine>,
    pub transport: TpTransport,
    pub link: Option<&'a PeerPullLink>,
}

impl<'a> Hop<'a> {
    pub fn ranks(&self) -> usize {
        self.engines.len()
    }

    pub fn engine(&self, r: usize) -> &'a Engine {
        self.engines[r]
    }

    fn link(&self) -> Result<&'a PeerPullLink, Box<dyn std::error::Error>> {
        self.link.ok_or_else(|| {
            "glm5-tp transport: the peer-pull arm is armed without a publication link \
             (runtime construction bug — refused rather than issuing an unordered peer read)"
                .into()
        })
    }

    /// Order `consumer`'s stream after everything `producer` has enqueued. One event record
    /// plus one cross-context wait; NO host boundary.
    ///
    /// THE DESIGN DELTA vs the step seam, stated because it is the whole point of this lane.
    /// `tp.rs`'s native-P2P pull sites fence the producer with a host
    /// `engine.stream().synchronize()` — see the "PRODUCER FENCE (2026-08-20 flake fix)"
    /// comments at `tp.rs:3543` and `tp.rs:2636`. That is correct and it is also a full
    /// stream drain, i.e. it keeps the exact cost class this lane exists to delete: a
    /// native-P2P arm built that way still pays one host sync per hop and would read as
    /// "native P2P did not help". This link instead uses the `pp.rs` `BoundarySlot` ordering
    /// contract (`ev_tx.record(s_tx)` then `s_rx.wait(&ev_tx)`, `pp.rs:2647`/`pp.rs:2678`),
    /// which the hy3 PP-4 qualification closed at 50/50 fresh processes and 200/200 runtime
    /// probes with zero byte mismatch on this exact card class — and which the tp2 lane
    /// already cherry-picked onto this line (`tp2-20260831/LANE.md` stage 1). So: PULL
    /// direction from `tp.rs`, EVENT ordering from `pp.rs`, host syncs from neither.
    fn publish(&self, producer: usize, consumer: usize) -> Result<(), Box<dyn std::error::Error>> {
        let link = self.link()?;
        let ev = &link.pub_ev[producer];
        {
            let prod = self.engine(producer);
            let _main = prod.gpu.enter_main()?;
            ev.record(&prod.stream())?;
        }
        {
            let cons = self.engine(consumer);
            let _main = cons.gpu.enter_main()?;
            cons.stream().wait(ev)?;
        }
        TP_PUB_EVENTS.fetch_add(2, Ordering::Relaxed);
        Ok(())
    }

    /// The write-after-read half: order `producer`'s stream after `consumer` has finished
    /// reading producer-owned memory, so the producer cannot recycle it under the reader.
    /// Costs one event record plus one cross-context wait; no host boundary.
    fn release(&self, producer: usize, consumer: usize) -> Result<(), Box<dyn std::error::Error>> {
        let link = self.link()?;
        let ev = &link.rel_ev[consumer];
        {
            let cons = self.engine(consumer);
            let _main = cons.gpu.enter_main()?;
            ev.record(&cons.stream())?;
        }
        {
            let prod = self.engine(producer);
            let _main = prod.gpu.enter_main()?;
            prod.stream().wait(ev)?;
        }
        TP_PUB_EVENTS.fetch_add(2, Ordering::Relaxed);
        Ok(())
    }

    /// PRODUCER-side strided store: `from`'s stream writes `rows x row_len` floats from `src`
    /// into `to`'s `dst`, at the given element strides and offsets. The destination address is
    /// resolved under the CONSUMER'S OWN context before the producer's context is entered:
    /// reading a peer's `device_ptr` while another context is current resolved to the wrong
    /// address once and cost a debugging round (`tp_ar`'s note carries the receipt).
    ///
    /// Ordering is the caller's job: `publish(from, to)` after the push, before `to` reads.
    #[allow(clippy::too_many_arguments)] // allow: (producer, src, src_off, src_stride) x (consumer, dst, dst_off, dst_stride) x (rows, row_len) IS the shape of a strided cross-rank store, and a struct would hide which side owns which pointer
    fn push_2d(
        &self,
        from: usize,
        src: &CudaSlice<f32>,
        src_off: usize,
        src_stride: usize,
        to: usize,
        dst: &mut CudaSlice<f32>,
        dst_off: usize,
        dst_stride: usize,
        rows: usize,
        row_len: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        if rows == 0 || row_len == 0 {
            return Err("glm5-tp push: zero geometry".into());
        }
        if src_off + (rows - 1) * src_stride + row_len > src.len() {
            return Err("glm5-tp push: source range escapes the buffer".into());
        }
        if dst_off + (rows - 1) * dst_stride + row_len > dst.len() {
            return Err("glm5-tp push: destination range escapes the buffer".into());
        }
        let dst_ptr = {
            let cons = self.engine(to);
            let _main = cons.gpu.enter_main()?;
            let s = cons.stream();
            dst.device_ptr_mut(&s).0 as *mut f32
        };
        let prod = self.engine(from);
        let _main = prod.gpu.enter_main()?;
        let s = prod.stream();
        let src_ptr = src.device_ptr(&s).0 as *const f32;
        // SAFETY: peer access is granted between the ranks, both ranges are bounds-checked
        // above against their own buffers, and the destination address was resolved under its
        // owning context.
        let rc = unsafe {
            crate::tp_ar::memra_tp_ar_push_2d(
                src_ptr.add(src_off),
                dst_ptr.add(dst_off),
                rows as i64,
                row_len as i64,
                src_stride as i64,
                dst_stride as i64,
                s.cu_stream() as *mut std::os::raw::c_void,
            )
        };
        if rc != 0 {
            return Err(format!("memra_tp_ar_push_2d rc {rc}").into());
        }
        charge_device_push(rows * row_len * std::mem::size_of::<f32>());
        Ok(())
    }

    /// The i32 twin of [`Self::push_2d`], contiguous. A 4-byte word copy carries bits through
    /// unchanged, so routing it through the f32 kernel is exact movement, not a conversion.
    fn push_2d_i32(
        &self,
        from: usize,
        src: &CudaSlice<i32>,
        to: usize,
        dst: &mut CudaSlice<i32>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        if n == 0 || n > src.len() || n > dst.len() {
            return Err("glm5-tp push (i32): range escapes a buffer".into());
        }
        let dst_ptr = {
            let cons = self.engine(to);
            let _main = cons.gpu.enter_main()?;
            let s = cons.stream();
            dst.device_ptr_mut(&s).0 as *mut f32
        };
        let prod = self.engine(from);
        let _main = prod.gpu.enter_main()?;
        let s = prod.stream();
        let src_ptr = src.device_ptr(&s).0 as *const f32;
        // SAFETY: as `push_2d`, with both ranges bounds-checked above; the cast reinterprets
        // 4-byte words and the kernel only copies them.
        let rc = unsafe {
            crate::tp_ar::memra_tp_ar_push_2d(
                src_ptr,
                dst_ptr,
                1,
                n as i64,
                n as i64,
                n as i64,
                s.cu_stream() as *mut std::os::raw::c_void,
            )
        };
        if rc != 0 {
            return Err(format!("memra_tp_ar_push_2d (i32) rc {rc}").into());
        }
        charge_device_push(n * std::mem::size_of::<i32>());
        Ok(())
    }

    /// The one cross-rank primitive: `consumer`'s stream copies `n` f32 from the producer's
    /// `src[src_off..]` into its own `dst[dst_off..]`. A PEER READ by construction — the
    /// copy is enqueued on the reading side.
    #[allow(clippy::too_many_arguments)] // allow: (producer, src, src_off) x (consumer, dst, dst_off) x n IS the shape of a cross-rank ranged copy; collapsing it into a struct would hide which side owns which pointer, and that is the one thing a peer copy must never be vague about
    fn pull_f32(
        &self,
        producer: usize,
        src: &CudaSlice<f32>,
        src_off: usize,
        consumer: usize,
        dst: &mut CudaSlice<f32>,
        dst_off: usize,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if src.len() < src_off + n || dst.len() < dst_off + n {
            return Err(format!(
                "glm5-tp peer pull geometry: src {}..{} of {} -> dst {}..{} of {}",
                src_off,
                src_off + n,
                src.len(),
                dst_off,
                dst_off + n,
                dst.len(),
            )
            .into());
        }
        {
            let consumer_engine = self.engine(consumer);
            // Rank-local CUDA scope: the consuming rank's context must be current for the copy
            // to be enqueued on its stream and for the peer source pointer to resolve through
            // that context's UVA mapping (the `Gpu::enter_main` contract in memra-runtime, and
            // the discipline every `tp.rs` cross-context copy site follows). The scope CLOSES
            // before the release fence below, which needs the producer's context current.
            let _main = consumer_engine.gpu.enter_main()?;
            let mut view = dst.slice_mut(dst_off..dst_off + n);
            consumer_engine
                .stream()
                .memcpy_dtod(&src.slice(src_off..src_off + n), &mut view)?;
        }
        // WRITE-AFTER-READ FENCE. The read above runs on the CONSUMER's stream against memory
        // the PRODUCER owns. Without this, the producer's stream is free to enqueue that
        // buffer's async free (or any reuse of the recycled block) while the copy engine is
        // still reading it — a silent cross-rank corruption that presents as run-to-run
        // NON-DETERMINISM, which is how the rig gate found it (arm X1, decode
        // self-consistency, step 17). See [`PeerPullLink`].
        self.release(producer, consumer)?;
        charge_peer_pull(n * std::mem::size_of::<f32>());
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------
// Hop shapes
// ------------------------------------------------------------------------------------------
//
// Four shapes cover every cross-rank movement in the glm5 TP program. The call sites name
// the shape; the arm lives here. Both arms move the SAME bytes by the same layout rules, so
// swapping the arm cannot change a bit — which is why the gate bar stays decode BYTE
// identity across the swap rather than a band.

/// FAN-OUT: replicate root's `src` into a fresh buffer on EVERY peer rank. Returns the peer
/// buffers indexed `[rank - 1]` (the `rt.peers` convention).
///
/// The mixer input `x`/`h` and the MoE activation `z` take this shape. Host-canonical pays
/// ONE draining `dtoh` plus one `htod` per peer (identical to v1 at two ranks); peer-pull
/// pays one publication chain and one peer read per peer.
pub fn fanout_f32(
    hop: &Hop<'_>,
    src: &CudaSlice<f32>,
    n: usize,
) -> Result<Vec<CudaSlice<f32>>, Box<dyn std::error::Error>> {
    let bytes = n * std::mem::size_of::<f32>();
    let mut out = Vec::with_capacity(hop.ranks() - 1);
    match hop.transport {
        TpTransport::HostCanonical => {
            let host = hop.engine(ROOT).dtoh_view(&src.slice(0..n))?;
            charge_host_leg(bytes, true);
            for to in 1..hop.ranks() {
                out.push(hop.engine(to).htod(&host)?);
                charge_host_leg(bytes, false);
            }
        }
        TpTransport::PeerPull => {
            for to in 1..hop.ranks() {
                let mut buf = hop.engine(to).uninit(n)?;
                hop.publish(ROOT, to)?;
                hop.pull_f32(ROOT, src, 0, to, &mut buf, 0, n)?;
                out.push(buf);
            }
        }
        TpTransport::DevicePush => {
            for to in 1..hop.ranks() {
                let mut buf = hop.engine(to).uninit(n)?;
                hop.push_2d(ROOT, src, 0, n, to, &mut buf, 0, n, 1, n)?;
                // Order AFTER the store: the event is recorded on root's stream once the push is
                // enqueued there, and the consumer waits on it.
                hop.publish(ROOT, to)?;
                out.push(buf);
            }
        }
    }
    Ok(out)
}

/// FAN-OUT to ONE rank: replicate root's `src` into a fresh buffer on rank `to`. The EP
/// diet's shape (a rank whose routing owns zero pairs receives zero activation bytes — the
/// placement-map multiplier); at two ranks it is exactly the whole-group fan-out.
pub fn fanout_f32_to(
    hop: &Hop<'_>,
    to: usize,
    src: &CudaSlice<f32>,
    n: usize,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let bytes = n * std::mem::size_of::<f32>();
    match hop.transport {
        TpTransport::HostCanonical => {
            let host = hop.engine(ROOT).dtoh_view(&src.slice(0..n))?;
            charge_host_leg(bytes, true);
            let out = hop.engine(to).htod(&host)?;
            charge_host_leg(bytes, false);
            Ok(out)
        }
        TpTransport::PeerPull => {
            let mut buf = hop.engine(to).uninit(n)?;
            hop.publish(ROOT, to)?;
            hop.pull_f32(ROOT, src, 0, to, &mut buf, 0, n)?;
            Ok(buf)
        }
        TpTransport::DevicePush => {
            let mut buf = hop.engine(to).uninit(n)?;
            hop.push_2d(ROOT, src, 0, n, to, &mut buf, 0, n, 1, n)?;
            hop.publish(ROOT, to)?;
            Ok(buf)
        }
    }
}

/// FAN-OUT, i32 twin (the MLA position vector). Same shape, same accounting.
pub fn fanout_i32(
    hop: &Hop<'_>,
    src: &CudaSlice<i32>,
    n: usize,
) -> Result<Vec<CudaSlice<i32>>, Box<dyn std::error::Error>> {
    let bytes = n * std::mem::size_of::<i32>();
    let mut out = Vec::with_capacity(hop.ranks() - 1);
    match hop.transport {
        TpTransport::HostCanonical => {
            let host = hop.engine(ROOT).dtoh_i32(src)?;
            charge_host_leg(bytes, true);
            for to in 1..hop.ranks() {
                out.push(hop.engine(to).htod_i32(&host)?);
                charge_host_leg(bytes, false);
            }
        }
        TpTransport::PeerPull => {
            for to in 1..hop.ranks() {
                let mut buf = hop.engine(to).uninit_i32(n)?;
                hop.publish(ROOT, to)?;
                {
                    let consumer = hop.engine(to);
                    let _main = consumer.gpu.enter_main()?;
                    let mut view = buf.slice_mut(0..n);
                    consumer.stream().memcpy_dtod(&src.slice(0..n), &mut view)?;
                }
                // The same write-after-read fence `Hop::pull_f32` applies; the i32 twin is
                // not exempt, and an exemption here would be a hazard that only shows up
                // under load.
                hop.release(ROOT, to)?;
                charge_peer_pull(bytes);
                out.push(buf);
            }
        }
        TpTransport::DevicePush => {
            for to in 1..hop.ranks() {
                let mut buf = hop.engine(to).uninit_i32(n)?;
                hop.push_2d_i32(ROOT, src, to, &mut buf, n)?;
                hop.publish(ROOT, to)?;
                out.push(buf);
            }
        }
    }
    Ok(out)
}

/// GATHER: reconstruct the FULL token-major `[t, ranks * part]` tensor on EVERY rank from
/// each rank's dense `[t, part]` shard (`parts[r]` resident on rank r, laid at columns
/// `r*part..(r+1)*part`). Returns the full tensors indexed by rank.
///
/// This is the column-parallel-over-gather join: after it, each rank's `wo` column slice is
/// a full-K matvec over identical bytes, which is what makes the join pure movement.
///
/// The peer-pull arm moves ONE dense block per (producer, consumer) direction and
/// reconstructs the interleave with `place_rows_strided` — the kernel whose own doc says it
/// "exists so multi-GPU collectives can move one dense shard per rank and reconstruct the
/// canonical token-major matrix without issuing one peer copy per token". At `t == 1` the
/// placement degenerates to a contiguous range and is skipped entirely: the pull lands
/// straight at its column offset.
pub fn gather_parts(
    hop: &Hop<'_>,
    parts: &[&CudaSlice<f32>],
    t: usize,
    part: usize,
) -> Result<Vec<CudaSlice<f32>>, Box<dyn std::error::Error>> {
    let ranks = hop.ranks();
    if t == 0 || part == 0 {
        return Err("glm5-tp gather: zero geometry".into());
    }
    if parts.len() != ranks {
        return Err(format!("glm5-tp gather: {} parts for {ranks} ranks", parts.len()).into());
    }
    let full = ranks * part;
    let span = t * part;
    for (r, p) in parts.iter().enumerate() {
        if p.len() < span {
            return Err(format!(
                "glm5-tp gather geometry: part[{r}] {} for t={t} part={part}",
                p.len()
            )
            .into());
        }
    }
    let bytes = span * std::mem::size_of::<f32>();
    match hop.transport {
        TpTransport::HostCanonical => {
            // Drain every rank's part (peers first, root last — v1's order at two ranks),
            // assemble the token-major full matrix once on host, upload to every rank.
            let mut hosts: Vec<Option<Vec<f32>>> = (0..ranks).map(|_| None).collect();
            for r in (0..ranks).rev() {
                hosts[r] = Some(hop.engine(r).dtoh_view(&parts[r].slice(0..span))?);
                charge_host_leg(bytes, true);
            }
            let mut full_host = vec![0f32; t * full];
            for (r, h) in hosts.iter().enumerate() {
                let h = h.as_ref().expect("drained above");
                for tok in 0..t {
                    full_host[tok * full + r * part..tok * full + (r + 1) * part]
                        .copy_from_slice(&h[tok * part..(tok + 1) * part]);
                }
            }
            let mut out = Vec::with_capacity(ranks);
            for r in 0..ranks {
                out.push(hop.engine(r).htod(&full_host)?);
                charge_host_leg(t * full * std::mem::size_of::<f32>(), false);
            }
            Ok(out)
        }
        TpTransport::DevicePush => {
            // Every producer writes its own column band into EVERY rank's full matrix, its own
            // included (a same-rank push is a local strided copy on that rank's own stream), and
            // then publishes once so each consumer's stream is ordered after it. Same bytes and
            // same token-major layout as the other two arms; the difference is that a rank hands
            // its band over and carries on instead of waiting to be read.
            let mut out = Vec::with_capacity(ranks);
            for r in 0..ranks {
                out.push(hop.engine(r).uninit(t * full)?);
            }
            for sfrom in 0..ranks {
                for (d, dst) in out.iter_mut().enumerate() {
                    hop.push_2d(
                        sfrom,
                        parts[sfrom],
                        0,
                        part,
                        d,
                        dst,
                        sfrom * part,
                        full,
                        t,
                        part,
                    )?;
                    if sfrom != d {
                        hop.publish(sfrom, d)?;
                    }
                }
            }
            Ok(out)
        }
        TpTransport::PeerPull => {
            // Every producer publishes to every consumer (record-then-wait per ordered
            // pair — the same primitive count as v1's two publish calls at two ranks),
            // then every consumer pulls each foreign part.
            for s in 0..ranks {
                for d in 0..ranks {
                    if s != d {
                        hop.publish(s, d)?;
                    }
                }
            }
            let mut out = Vec::with_capacity(ranks);
            for d in 0..ranks {
                let mut full_d = hop.engine(d).uninit(t * full)?;
                if t == 1 {
                    // Decode: the strided placement IS a contiguous range, so land every
                    // part directly at its column offset and skip the placement kernels.
                    for s in 0..ranks {
                        if s == d {
                            hop.engine(d).copy_range_into(
                                &mut full_d,
                                s * part,
                                parts[s],
                                0,
                                part,
                            )?;
                            charge_local_copy();
                        } else {
                            hop.pull_f32(s, parts[s], 0, d, &mut full_d, s * part, part)?;
                        }
                    }
                } else {
                    // Prime: one dense block per foreign direction, then byte-preserving
                    // placements.
                    for s in 0..ranks {
                        if s == d {
                            hop.engine(d).place_rows_strided(
                                parts[s],
                                &mut full_d,
                                part,
                                t,
                                full,
                                s * part,
                            )?;
                            charge_local_copy();
                        } else {
                            let mut foreign = hop.engine(d).uninit(span)?;
                            hop.pull_f32(s, parts[s], 0, d, &mut foreign, 0, span)?;
                            hop.engine(d).place_rows_strided(
                                &foreign,
                                &mut full_d,
                                part,
                                t,
                                full,
                                s * part,
                            )?;
                            charge_local_copy();
                        }
                    }
                }
                out.push(full_d);
            }
            Ok(out)
        }
    }
}

/// CONCAT-ON-ROOT: assemble the mixer output `[t, ranks * part]` on ROOT from the per-rank
/// column-`wo` slices, rank r's rows at columns `r*part..(r+1)*part`.
///
/// Root is the only consumer (the residual stream and mHC are root-owned), so only the
/// foreign directions cross. Same `t == 1` fast path as [`gather_parts`].
pub fn concat_parts_on_root(
    hop: &Hop<'_>,
    parts: &[&CudaSlice<f32>],
    t: usize,
    part: usize,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let ranks = hop.ranks();
    if t == 0 || part == 0 {
        return Err("glm5-tp concat: zero geometry".into());
    }
    if parts.len() != ranks {
        return Err(format!("glm5-tp concat: {} parts for {ranks} ranks", parts.len()).into());
    }
    let full = ranks * part;
    let span = t * part;
    for (r, p) in parts.iter().enumerate() {
        if p.len() < span {
            return Err(format!(
                "glm5-tp concat geometry: part[{r}] {} for t={t} part={part}",
                p.len()
            )
            .into());
        }
    }
    let bytes = span * std::mem::size_of::<f32>();
    match hop.transport {
        TpTransport::HostCanonical => {
            let mut hosts: Vec<Option<Vec<f32>>> = (0..ranks).map(|_| None).collect();
            for r in (0..ranks).rev() {
                hosts[r] = Some(hop.engine(r).dtoh_view(&parts[r].slice(0..span))?);
                charge_host_leg(bytes, true);
            }
            let mut out_host = vec![0f32; t * full];
            for (r, h) in hosts.iter().enumerate() {
                let h = h.as_ref().expect("drained above");
                for tok in 0..t {
                    out_host[tok * full + r * part..tok * full + (r + 1) * part]
                        .copy_from_slice(&h[tok * part..(tok + 1) * part]);
                }
            }
            let out = hop.engine(ROOT).htod(&out_host)?;
            charge_host_leg(t * full * std::mem::size_of::<f32>(), false);
            Ok(out)
        }
        TpTransport::DevicePush => {
            let mut out = hop.engine(ROOT).uninit(t * full)?;
            for sfrom in 0..ranks {
                hop.push_2d(
                    sfrom,
                    parts[sfrom],
                    0,
                    part,
                    ROOT,
                    &mut out,
                    sfrom * part,
                    full,
                    t,
                    part,
                )?;
                if sfrom != ROOT {
                    hop.publish(sfrom, ROOT)?;
                }
            }
            Ok(out)
        }
        TpTransport::PeerPull => {
            let mut out = hop.engine(ROOT).uninit(t * full)?;
            for s in 1..ranks {
                hop.publish(s, ROOT)?;
            }
            if t == 1 {
                for s in 0..ranks {
                    if s == ROOT {
                        hop.engine(ROOT)
                            .copy_range_into(&mut out, s * part, parts[s], 0, part)?;
                        charge_local_copy();
                    } else {
                        hop.pull_f32(s, parts[s], 0, ROOT, &mut out, s * part, part)?;
                    }
                }
            } else {
                for s in 0..ranks {
                    if s == ROOT {
                        hop.engine(ROOT).place_rows_strided(
                            parts[s],
                            &mut out,
                            part,
                            t,
                            full,
                            s * part,
                        )?;
                        charge_local_copy();
                    } else {
                        let mut foreign = hop.engine(ROOT).uninit(span)?;
                        hop.pull_f32(s, parts[s], 0, ROOT, &mut foreign, 0, span)?;
                        hop.engine(ROOT).place_rows_strided(
                            &foreign,
                            &mut out,
                            part,
                            t,
                            full,
                            s * part,
                        )?;
                        charge_local_copy();
                    }
                }
            }
            Ok(out)
        }
    }
}

/// v1-SHAPE FAN-OUT, part 1: stage the producer's whole block to HOST in one draining leg.
///
/// Exists only so the `host-canonical` arm of the SEQUENTIAL EP walk keeps its v1 hop pattern
/// exactly — one `dtoh` of `z` per layer-call, then one row `htod` per token per consuming
/// rank. That pattern is what the banked 22.65 tok/s engine-twin number was measured on, and
/// `MEMRA_GLM5_TP_TRANSPORT=0` has to reproduce it hop-for-hop, not just byte-for-byte, or the
/// rollback arm silently becomes a different (faster) program and the A/B measures two changes.
pub fn host_stage_block(
    hop: &Hop<'_>,
    from: usize,
    src: &CudaSlice<f32>,
    n: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let host = hop.engine(from).dtoh_view(&src.slice(0..n))?;
    charge_host_leg(n * std::mem::size_of::<f32>(), true);
    Ok(host)
}

/// v1-SHAPE FAN-OUT, part 2: upload one staged host row to rank `to` (one non-draining leg).
pub fn host_row_to(
    hop: &Hop<'_>,
    to: usize,
    row: &[f32],
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let out = hop.engine(to).htod(row)?;
    charge_host_leg(std::mem::size_of_val(row), false);
    Ok(out)
}

/// BLOCK-RETURN: land rank `from`'s dense `[rows, width]` block into `dst[dst_off..]` on
/// root. The EP walk's shape: the dieted arm returns one compact block per (layer-call,
/// rank); the v1 sequential arm returns one expert row per foreign-owned slot. `n` is
/// `rows * width`.
pub fn return_block_to_root(
    hop: &Hop<'_>,
    from: usize,
    blk: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    dst_off: usize,
    n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = n * std::mem::size_of::<f32>();
    match hop.transport {
        TpTransport::HostCanonical => {
            let host = hop.engine(from).dtoh_view(&blk.slice(0..n))?;
            charge_host_leg(bytes, true);
            hop.engine(ROOT).htod_f32_into_at(&host, dst, dst_off)?;
            charge_host_leg(bytes, false);
            Ok(())
        }
        TpTransport::PeerPull => {
            hop.publish(from, ROOT)?;
            hop.pull_f32(from, blk, 0, ROOT, dst, dst_off, n)
        }
        TpTransport::DevicePush => {
            hop.push_2d(from, blk, 0, n, ROOT, dst, dst_off, n, 1, n)?;
            hop.publish(from, ROOT)
        }
    }
}

/// BLOCK-RETURN into a FRESH root buffer (the v1 per-slot EP shape, which hands the returned
/// row straight to `axpy_into`).
pub fn return_row_to_root(
    hop: &Hop<'_>,
    from: usize,
    y: &CudaSlice<f32>,
    n: usize,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    match hop.transport {
        TpTransport::HostCanonical => {
            let host = hop.engine(from).dtoh_view(&y.slice(0..n))?;
            charge_host_leg(n * std::mem::size_of::<f32>(), true);
            let out = hop.engine(ROOT).htod(&host)?;
            charge_host_leg(n * std::mem::size_of::<f32>(), false);
            Ok(out)
        }
        TpTransport::PeerPull => {
            let mut out = hop.engine(ROOT).uninit(n)?;
            hop.publish(from, ROOT)?;
            hop.pull_f32(from, y, 0, ROOT, &mut out, 0, n)?;
            Ok(out)
        }
        TpTransport::DevicePush => {
            let mut out = hop.engine(ROOT).uninit(n)?;
            hop.push_2d(from, y, 0, n, ROOT, &mut out, 0, n, 1, n)?;
            hop.publish(from, ROOT)?;
            Ok(out)
        }
    }
}

// ------------------------------------------------------------------------------------------
// Arming: the byte-integrity pull ladder
// ------------------------------------------------------------------------------------------

/// Payload sizes the arm-time ladder validates, mirroring the step seam's
/// `NATIVE_P2P_PROBE_WORDS` (16 KiB .. 64 MiB) so a glm5 refusal and a step refusal name the
/// same fabric with the same numbers. The bottom rung is deliberately BELOW our decode hop
/// sizes (a decode gather part is <= 16 KiB at hidden 4096 f32) and the top rung above any
/// prime chunk block, because RESEARCH.md §2.4 records that LL and Simple protocol paths
/// "can fail differently" by size and §2.10 records phantom readings that only a byte check
/// catches.
const PULL_PROBE_WORDS: &[usize] = &[4_096, 16_384, 262_144, 16_777_216];

static TRANSPORT_MARKED: AtomicBool = AtomicBool::new(false);

/// Arm the transport for a TP load. On the peer-pull arm this validates the EXACT primitive
/// the walk uses, in EVERY ordered rank pair, at every rung of [`PULL_PROBE_WORDS`], against
/// a poisoned destination — and refuses the load on a single differing word.
///
/// `same_device_gate` is the one-card rig emulation (N contexts on one device): the pull
/// primitive is exercised unchanged, the ladder still runs, and the announce says so, but no
/// claim about a real fabric is made or implied.
///
/// What this DOES prove: our copy path moves the right bytes in every direction at four
/// sizes. What it does NOT prove (RESEARCH.md §2.4's ladder, stated so nobody reads it
/// wider): that a KERNEL dereference of a peer pointer works. On direct-attach (`NODE`)
/// hosts the driver stages SM-issued peer access through system memory by default (§2.3b)
/// and "`nvidia-smi topo -p2p r` returns OK while `cudaMemcpy` looks healthy, so neither
/// detects it". Only a `simpleP2P`-class kernel peer read does; that lives in the lane's
/// `peer-read-probe.cu` and is a HEALTH.sh item, not a serving-path dependency for the
/// host-canonical and peer-pull arms, which never dereference a peer pointer from a kernel.
///
/// `device-push` DOES, so that warning is live for it: its ladder below runs the PUSH primitive
/// rather than the pull, because proving a pull says nothing about an SM-issued store. On the
/// served pair the link is NV18 (eighteen NVLink links, 53.125 GB/s each), where SM peer access
/// is native; on a direct-attach host the ladder is what catches the staged path, and it catches
/// it as a byte mismatch rather than as a quiet slowdown.
pub fn arm_transport(
    transport: TpTransport,
    armed_flag: &str,
    tag: &str,
    engines: &[&Engine],
    same_device_gate: bool,
) -> Result<Option<PeerPullLink>, Box<dyn std::error::Error>> {
    if transport == TpTransport::HostCanonical {
        announce(
            tag,
            transport,
            same_device_gate,
            "host staging, no peer mapping",
        );
        return Ok(None);
    }
    if same_device_gate {
        // N contexts on ONE device: there is no peer to grant (`cuDeviceCanAccessPeer`
        // of a device with itself is not a peer relation) and no fabric to cross. The pull
        // primitive still runs, unchanged, over the contexts' UVA mappings — which is
        // what makes the rig gate a real test of the CODE and an explicit non-test of the
        // FABRIC. Skipping the grant here is why the gate arm cannot silently pass on a box
        // where the grant would have failed: on a real group the grants run and refuse.
        eprintln!(
            "[{tag}] same-device gate: peer-access grant SKIPPED (one device, {} \
             contexts); the ladder below proves bit-preservation only, never fabric engagement",
            engines.len(),
        );
    } else {
        for (i, a) in engines.iter().enumerate() {
            for (j, b) in engines.iter().enumerate() {
                if i != j {
                    crate::tp::grant_peer_access(
                        a,
                        b,
                        &format!("{armed_flag}={}", transport.name()),
                    )?;
                }
            }
        }
    }
    let link = PeerPullLink::new(engines)?;
    let hop = Hop {
        engines: engines.to_vec(),
        transport,
        link: Some(&link),
    };
    // MEASUREMENT HYGIENE. The ladder below drives the SAME instrumented primitive the walk
    // does, so it charges the movement census — and it charges it HARD: 4 rungs x every
    // ordered pair is >100 MiB of arm-time traffic against ~15 MiB per DECODE TOKEN of real
    // walk traffic. The first gate run made this visible and unmissable: `xfer_bytes` on the
    // peer-pull arm read 25x the host-canonical arm's, which is the opposite of the truth
    // (peer-pull crosses PCIe HALF as much, because a host bounce crosses it twice). A box
    // window reading `xfer_bytes` deltas would have derived a bytes/token figure that was
    // ~97% arming noise. So: snapshot before, restore after. Arm-time validation traffic is
    // not walk traffic and must not appear in a per-token census. The ladder's own receipt
    // is its PASS line.
    let census_before = (
        TP_PEER_PULLS.load(Ordering::Relaxed),
        TP_DEVICE_PUSHES.load(Ordering::Relaxed),
        TP_PUB_EVENTS.load(Ordering::Relaxed),
        TP_XFER_BYTES.load(Ordering::Relaxed),
        TP_HOST_LEGS.load(Ordering::Relaxed),
        TP_HOST_SYNCS.load(Ordering::Relaxed),
        TP_LOCAL_COPIES.load(Ordering::Relaxed),
    );
    let ranks = engines.len();
    for &words in PULL_PROBE_WORDS {
        for producer in 0..ranks {
            for consumer in 0..ranks {
                if consumer == producer {
                    continue;
                }
                let expected: Vec<f32> = (0..words)
                    .map(|i| {
                        f32::from_bits(
                            (i as u32)
                                .wrapping_mul(0x9e37_79b9)
                                .wrapping_add(((producer as u32) + 1) << 16)
                                // Keep every probe word a finite, non-NaN pattern so a
                                // comparison failure is a TRANSPORT fact and never a float
                                // identity artifact.
                                & 0x7f7f_ffff,
                        )
                    })
                    .collect();
                let poison: Vec<f32> = expected.iter().map(|v| -*v - 1.0).collect();
                let src = hop.engine(producer).htod(&expected)?;
                let mut dst = hop.engine(consumer).htod(&poison)?;
                // Each arm validates ITS OWN primitive. A pull ladder says nothing about a push:
                // the doc above records that on direct-attach hosts the driver stages SM-issued
                // peer access through system memory while `topo -p2p r` and `cudaMemcpy` both
                // look healthy, and `device-push` is the first transport here that dereferences
                // a peer pointer FROM A KERNEL. Proving the wrong primitive would be exactly the
                // silent pass that warning describes.
                match transport {
                    TpTransport::DevicePush => {
                        hop.push_2d(
                            producer, &src, 0, words, consumer, &mut dst, 0, words, 1, words,
                        )?;
                        hop.publish(producer, consumer)?;
                    }
                    _ => {
                        hop.publish(producer, consumer)?;
                        hop.pull_f32(producer, &src, 0, consumer, &mut dst, 0, words)?;
                    }
                }
                let actual = hop.engine(consumer).dtoh(&dst)?;
                let mismatches = actual
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                if mismatches != 0 {
                    return Err(format!(
                        "{armed_flag}={} byte-integrity ladder FAILED: \
                         rank{producer}->rank{consumer} at {} bytes, {mismatches}/{words} \
                         words differ (refused before any layer was sharded; roll back \
                         with: {})",
                        transport.name(),
                        words * std::mem::size_of::<f32>(),
                        rollback_advice(),
                    )
                    .into());
                }
            }
        }
    }
    // Restore the census to its pre-ladder state (see the hygiene note above).
    let restore = |counter: &AtomicU64, before: u64| {
        let now = counter.load(Ordering::Relaxed);
        counter.fetch_sub(now.saturating_sub(before), Ordering::Relaxed);
    };
    // Paired with the snapshot by NAME, not by tuple position: adding `TP_DEVICE_PUSHES` to the
    // snapshot shifted every index by one and the compiler could not see it (they are all u64),
    // so each delta would have been restored onto the wrong counter.
    for (counter, before) in [
        (&TP_PEER_PULLS, census_before.0),
        (&TP_DEVICE_PUSHES, census_before.1),
        (&TP_PUB_EVENTS, census_before.2),
        (&TP_XFER_BYTES, census_before.3),
        (&TP_HOST_LEGS, census_before.4),
        (&TP_HOST_SYNCS, census_before.5),
        (&TP_LOCAL_COPIES, census_before.6),
    ] {
        restore(counter, before);
    }
    eprintln!(
        "[{tag}] {} byte-integrity ladder PASS: directions={} \
         byte_ladder={:?} mismatches=0 same_device_gate={same_device_gate} \
         census_excluded=arm-time-ladder-traffic",
        transport.name(),
        ranks * (ranks - 1),
        PULL_PROBE_WORDS
            .iter()
            .map(|w| w * std::mem::size_of::<f32>())
            .collect::<Vec<_>>(),
    );
    announce(
        tag,
        transport,
        same_device_gate,
        "consumer-issued cuMemcpyDtoDAsync, event-published, atomics-free",
    );
    Ok(Some(link))
}

fn announce(tag: &str, transport: TpTransport, same_device_gate: bool, how: &str) {
    if TRANSPORT_MARKED.swap(true, Ordering::Relaxed) {
        return;
    }
    eprintln!(
        "[{tag}] armed transport={} shape={how} same_device_gate={same_device_gate} \
         performance_claim=false",
        transport.name(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_parse_is_literal_and_fail_closed() {
        for off in [None, Some(""), Some("0"), Some("host-canonical")] {
            assert_eq!(
                parse_transport(TRANSPORT_ENV, off).unwrap(),
                TpTransport::HostCanonical
            );
        }
        for on in [Some("1"), Some("peer-pull")] {
            assert_eq!(
                parse_transport(TRANSPORT_ENV, on).unwrap(),
                TpTransport::PeerPull
            );
        }
        for on in [Some("2"), Some("device-push")] {
            assert_eq!(
                parse_transport(TRANSPORT_ENV, on).unwrap(),
                TpTransport::DevicePush
            );
        }
        // Every other spelling refuses BY NAME rather than silently picking an arm — a
        // typo'd transport must never serve the other one. Asserted under BOTH names, so a
        // banked script setting the alias reads its own name back.
        for bad in [
            "peer_pull",
            "peerpull",
            "p2p",
            "native",
            "device_push",
            "devicepush",
            "push",
            "3",
            "on",
            "true",
        ] {
            for flag in [TRANSPORT_ENV, TRANSPORT_ENV_GLM5] {
                let err =
                    parse_transport(flag, Some(bad)).expect_err("unknown transport must refuse");
                assert!(err.contains(flag), "{err}");
                assert!(err.contains(bad), "{err}");
                assert!(err.contains("host-canonical"), "{err}");
                assert!(err.contains("peer-pull"), "{err}");
                assert!(err.contains("device-push"), "{err}");
            }
        }
    }

    #[test]
    fn alias_resolution_honors_both_names_and_refuses_a_disagreeing_pair() {
        // unset/unset -> the default, cited under the general name
        assert_eq!(
            resolve_transport(None, None).unwrap(),
            (TpTransport::HostCanonical, TRANSPORT_ENV)
        );
        // either name alone arms, and the ARMED NAME comes back with it
        assert_eq!(
            resolve_transport(Some("peer-pull"), None).unwrap(),
            (TpTransport::PeerPull, TRANSPORT_ENV)
        );
        assert_eq!(
            resolve_transport(None, Some("peer-pull")).unwrap(),
            (TpTransport::PeerPull, TRANSPORT_ENV_GLM5)
        );
        // an agreeing pair resolves to the general name
        assert_eq!(
            resolve_transport(Some("1"), Some("1")).unwrap(),
            (TpTransport::PeerPull, TRANSPORT_ENV)
        );
        // a DISAGREEING pair refuses, naming both — arming happens before any session exists,
        // so refusing is safe and a silently-chosen transport is not
        for (g, a) in [("peer-pull", "0"), ("0", "1"), ("1", "host-canonical")] {
            let err = resolve_transport(Some(g), Some(a))
                .expect_err("a disagreeing pair must refuse the load");
            assert!(err.contains(TRANSPORT_ENV), "{err}");
            assert!(err.contains(TRANSPORT_ENV_GLM5), "{err}");
            assert!(err.contains("disagree"), "{err}");
        }
        // NOT a disagreement by meaning, but IS one by string: two spellings of the same arm.
        // Deliberately literal rather than normalize-then-compare, so an operator who typed
        // two different things is told instead of having one silently win.
        assert!(resolve_transport(Some("0"), Some("host-canonical")).is_err());
    }

    #[test]
    fn default_is_the_v1_arm() {
        // The written default (docs/FLAGS.md): unmeasured behaviour does not default ON.
        assert_eq!(
            parse_transport(TRANSPORT_ENV, None).unwrap(),
            TpTransport::HostCanonical
        );
        assert_eq!(TpTransport::HostCanonical.name(), "host-canonical");
        assert_eq!(TpTransport::PeerPull.name(), "peer-pull");
    }

    /// The census line is a receipt: every counter must be named in it, because a box window
    /// greps this line and a missing field reads as a zero.
    #[test]
    fn census_line_names_every_counter() {
        let line = transport_census_line("glm5-tp-transport", TpTransport::PeerPull);
        for field in [
            "transport=peer-pull",
            "host_legs=",
            "host_syncs=",
            "peer_pulls=",
            "pub_events=",
            "local_copies=",
            "xfer_bytes=",
        ] {
            assert!(line.contains(field), "{line} is missing {field}");
        }
    }
}
