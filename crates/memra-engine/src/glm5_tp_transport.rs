//! glm5_next TP-N TRANSPORT — `MEMRA_GLM5_TP_TRANSPORT` (lane/glm5-tp-transport, 2026-09-01;
//! rank-widened by lane/glm5-composition, 2026-09-01).
//!
//! WHAT THIS IS. The data-movement layer under the `MEMRA_GLM5_TP` execution program. The TP
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
//! Engagement markers: `[glm5-tp-transport]`. Counters below are the per-token instrument.

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
pub enum Glm5TpTransport {
    /// v1: `dtoh` -> host -> `htod` per hop. Every banked glm5 TP number.
    HostCanonical,
    /// Consumer-issued device peer copy per hop, ordered by a publication event.
    PeerPull,
}

impl Glm5TpTransport {
    /// The receipt spelling that goes in every announce line and every gate log.
    pub fn name(self) -> &'static str {
        match self {
            Self::HostCanonical => "host-canonical",
            Self::PeerPull => "peer-pull",
        }
    }
}

/// Parse `MEMRA_GLM5_TP_TRANSPORT`. Pure over the value so the law is unit-testable without
/// touching the process environment.
///
/// DEFAULT IS `host-canonical` (OFF), by decision, not by accident: on the day this flag
/// landed the peer-pull arm had zero receipts on real peer hardware (the rig is a single
/// card — `LAW:rig-exactness-only` — so its gate arms run the code path over two contexts on
/// ONE device and can only prove bit-preservation, never fabric engagement). Unmeasured
/// behaviour does not default ON. The default flips in the same commit as the box window's
/// interleaved re-price receipt, and the FLAGS.md row carries both arms plus the rollback
/// seam (`MEMRA_GLM5_TP_TRANSPORT=0`).
pub fn parse_transport(value: Option<&str>) -> Result<Glm5TpTransport, String> {
    match value {
        None | Some("") | Some("0") | Some("host-canonical") => Ok(Glm5TpTransport::HostCanonical),
        Some("1") | Some("peer-pull") => Ok(Glm5TpTransport::PeerPull),
        Some(other) => Err(format!(
            "MEMRA_GLM5_TP_TRANSPORT={other:?} is not a known transport \
             (host-canonical | 0 = the v1 host-staged arm, the default; peer-pull | 1 = the \
             consumer-issued device peer copy arm)"
        )),
    }
}

/// Live read of [`parse_transport`].
pub fn transport_env() -> Result<Glm5TpTransport, String> {
    parse_transport(std::env::var("MEMRA_GLM5_TP_TRANSPORT").ok().as_deref())
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
pub static GLM5_TP_HOST_LEGS: AtomicU64 = AtomicU64::new(0);

/// Host legs that ended in a full stream drain (`Engine::dtoh` synchronizes). This is the
/// count the 13-18 ms/token v1 join tax reconstructs from. Pinned 0 on the peer-pull arm.
pub static GLM5_TP_HOST_SYNCS: AtomicU64 = AtomicU64::new(0);

/// Consumer-issued cross-rank device copies. Pinned 0 on the host-canonical arm.
pub static GLM5_TP_PEER_PULLS: AtomicU64 = AtomicU64::new(0);

/// Publication events recorded/awaited to order a cross-rank copy. Pinned 0 on the
/// host-canonical arm (whose ordering is the host sync itself).
pub static GLM5_TP_PUB_EVENTS: AtomicU64 = AtomicU64::new(0);

/// Same-rank device copies a hop issued (own-part placements and dense block moves). Present
/// in both arms; the host-canonical arm reaches its own part through the host instead.
pub static GLM5_TP_LOCAL_COPIES: AtomicU64 = AtomicU64::new(0);

/// Total bytes a TP hop moved across a rank boundary, both arms, counted once per crossing
/// (a host-canonical hop crosses PCIe twice and is charged twice — that IS its cost).
pub static GLM5_TP_XFER_BYTES: AtomicU64 = AtomicU64::new(0);

macro_rules! snapshot_fns {
    ($($name:ident => $counter:ident),* $(,)?) => {
        $(
            /// Snapshot for a gate's before/after delta.
            pub fn $name() -> u64 { $counter.load(Ordering::Relaxed) }
        )*
    };
}

snapshot_fns! {
    glm5_tp_host_legs => GLM5_TP_HOST_LEGS,
    glm5_tp_host_syncs => GLM5_TP_HOST_SYNCS,
    glm5_tp_peer_pulls => GLM5_TP_PEER_PULLS,
    glm5_tp_pub_events => GLM5_TP_PUB_EVENTS,
    glm5_tp_local_copies => GLM5_TP_LOCAL_COPIES,
    glm5_tp_xfer_bytes => GLM5_TP_XFER_BYTES,
}

/// One line carrying every movement counter, for a gate log or a box-window receipt.
pub fn transport_census_line(transport: Glm5TpTransport) -> String {
    format!(
        "[glm5-tp-transport] census transport={} host_legs={} host_syncs={} peer_pulls={} \
         pub_events={} local_copies={} xfer_bytes={}",
        transport.name(),
        glm5_tp_host_legs(),
        glm5_tp_host_syncs(),
        glm5_tp_peer_pulls(),
        glm5_tp_pub_events(),
        glm5_tp_local_copies(),
        glm5_tp_xfer_bytes(),
    )
}

fn charge_host_leg(bytes: usize, sync: bool) {
    GLM5_TP_HOST_LEGS.fetch_add(1, Ordering::Relaxed);
    if sync {
        GLM5_TP_HOST_SYNCS.fetch_add(1, Ordering::Relaxed);
    }
    GLM5_TP_XFER_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

fn charge_peer_pull(bytes: usize) {
    GLM5_TP_PEER_PULLS.fetch_add(1, Ordering::Relaxed);
    GLM5_TP_XFER_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

fn charge_local_copy() {
    GLM5_TP_LOCAL_COPIES.fetch_add(1, Ordering::Relaxed);
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
    pub transport: Glm5TpTransport,
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
        GLM5_TP_PUB_EVENTS.fetch_add(2, Ordering::Relaxed);
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
        GLM5_TP_PUB_EVENTS.fetch_add(2, Ordering::Relaxed);
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
        Glm5TpTransport::HostCanonical => {
            let host = hop.engine(ROOT).dtoh_view(&src.slice(0..n))?;
            charge_host_leg(bytes, true);
            for to in 1..hop.ranks() {
                out.push(hop.engine(to).htod(&host)?);
                charge_host_leg(bytes, false);
            }
        }
        Glm5TpTransport::PeerPull => {
            for to in 1..hop.ranks() {
                let mut buf = hop.engine(to).uninit(n)?;
                hop.publish(ROOT, to)?;
                hop.pull_f32(ROOT, src, 0, to, &mut buf, 0, n)?;
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
        Glm5TpTransport::HostCanonical => {
            let host = hop.engine(ROOT).dtoh_view(&src.slice(0..n))?;
            charge_host_leg(bytes, true);
            let out = hop.engine(to).htod(&host)?;
            charge_host_leg(bytes, false);
            Ok(out)
        }
        Glm5TpTransport::PeerPull => {
            let mut buf = hop.engine(to).uninit(n)?;
            hop.publish(ROOT, to)?;
            hop.pull_f32(ROOT, src, 0, to, &mut buf, 0, n)?;
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
        Glm5TpTransport::HostCanonical => {
            let host = hop.engine(ROOT).dtoh_i32(src)?;
            charge_host_leg(bytes, true);
            for to in 1..hop.ranks() {
                out.push(hop.engine(to).htod_i32(&host)?);
                charge_host_leg(bytes, false);
            }
        }
        Glm5TpTransport::PeerPull => {
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
        Glm5TpTransport::HostCanonical => {
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
        Glm5TpTransport::PeerPull => {
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
        Glm5TpTransport::HostCanonical => {
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
        Glm5TpTransport::PeerPull => {
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
        Glm5TpTransport::HostCanonical => {
            let host = hop.engine(from).dtoh_view(&blk.slice(0..n))?;
            charge_host_leg(bytes, true);
            hop.engine(ROOT).htod_f32_into_at(&host, dst, dst_off)?;
            charge_host_leg(bytes, false);
            Ok(())
        }
        Glm5TpTransport::PeerPull => {
            hop.publish(from, ROOT)?;
            hop.pull_f32(from, blk, 0, ROOT, dst, dst_off, n)
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
        Glm5TpTransport::HostCanonical => {
            let host = hop.engine(from).dtoh_view(&y.slice(0..n))?;
            charge_host_leg(n * std::mem::size_of::<f32>(), true);
            let out = hop.engine(ROOT).htod(&host)?;
            charge_host_leg(n * std::mem::size_of::<f32>(), false);
            Ok(out)
        }
        Glm5TpTransport::PeerPull => {
            let mut out = hop.engine(ROOT).uninit(n)?;
            hop.publish(from, ROOT)?;
            hop.pull_f32(from, y, 0, ROOT, &mut out, 0, n)?;
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
/// `peer-read-probe.cu` and is a HEALTH.sh item, not a serving-path dependency — this
/// transport never dereferences a peer pointer from a kernel.
pub fn arm_transport(
    transport: Glm5TpTransport,
    engines: &[&Engine],
    same_device_gate: bool,
) -> Result<Option<PeerPullLink>, Box<dyn std::error::Error>> {
    if transport == Glm5TpTransport::HostCanonical {
        announce(transport, same_device_gate, "host staging, no peer mapping");
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
            "[glm5-tp-transport] same-device gate: peer-access grant SKIPPED (one device, {} \
             contexts); the ladder below proves bit-preservation only, never fabric engagement",
            engines.len(),
        );
    } else {
        for (i, a) in engines.iter().enumerate() {
            for (j, b) in engines.iter().enumerate() {
                if i != j {
                    crate::tp::grant_peer_access(a, b, "MEMRA_GLM5_TP_TRANSPORT=peer-pull")?;
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
        GLM5_TP_PEER_PULLS.load(Ordering::Relaxed),
        GLM5_TP_PUB_EVENTS.load(Ordering::Relaxed),
        GLM5_TP_XFER_BYTES.load(Ordering::Relaxed),
        GLM5_TP_HOST_LEGS.load(Ordering::Relaxed),
        GLM5_TP_HOST_SYNCS.load(Ordering::Relaxed),
        GLM5_TP_LOCAL_COPIES.load(Ordering::Relaxed),
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
                hop.publish(producer, consumer)?;
                hop.pull_f32(producer, &src, 0, consumer, &mut dst, 0, words)?;
                let actual = hop.engine(consumer).dtoh(&dst)?;
                let mismatches = actual
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                if mismatches != 0 {
                    return Err(format!(
                        "MEMRA_GLM5_TP_TRANSPORT=peer-pull byte-integrity ladder FAILED: \
                         rank{producer}->rank{consumer} at {} bytes, {mismatches}/{words} \
                         words differ (refused before any layer was sharded; roll back with \
                         MEMRA_GLM5_TP_TRANSPORT=0)",
                        words * std::mem::size_of::<f32>(),
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
    restore(&GLM5_TP_PEER_PULLS, census_before.0);
    restore(&GLM5_TP_PUB_EVENTS, census_before.1);
    restore(&GLM5_TP_XFER_BYTES, census_before.2);
    restore(&GLM5_TP_HOST_LEGS, census_before.3);
    restore(&GLM5_TP_HOST_SYNCS, census_before.4);
    restore(&GLM5_TP_LOCAL_COPIES, census_before.5);
    eprintln!(
        "[glm5-tp-transport] peer-pull byte-integrity ladder PASS: directions={} \
         byte_ladder={:?} mismatches=0 same_device_gate={same_device_gate} \
         census_excluded=arm-time-ladder-traffic",
        ranks * (ranks - 1),
        PULL_PROBE_WORDS
            .iter()
            .map(|w| w * std::mem::size_of::<f32>())
            .collect::<Vec<_>>(),
    );
    announce(
        transport,
        same_device_gate,
        "consumer-issued cuMemcpyDtoDAsync, event-published, atomics-free",
    );
    Ok(Some(link))
}

fn announce(transport: Glm5TpTransport, same_device_gate: bool, how: &str) {
    if TRANSPORT_MARKED.swap(true, Ordering::Relaxed) {
        return;
    }
    eprintln!(
        "[glm5-tp-transport] armed transport={} shape={how} same_device_gate={same_device_gate} \
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
                parse_transport(off).unwrap(),
                Glm5TpTransport::HostCanonical
            );
        }
        for on in [Some("1"), Some("peer-pull")] {
            assert_eq!(parse_transport(on).unwrap(), Glm5TpTransport::PeerPull);
        }
        // Every other spelling refuses BY NAME rather than silently picking an arm — a
        // typo'd transport must never serve the other one.
        for bad in ["peer_pull", "peerpull", "p2p", "native", "2", "on", "true"] {
            let err = parse_transport(Some(bad)).expect_err("unknown transport must refuse");
            assert!(err.contains("MEMRA_GLM5_TP_TRANSPORT"), "{err}");
            assert!(err.contains(bad), "{err}");
            assert!(err.contains("host-canonical"), "{err}");
            assert!(err.contains("peer-pull"), "{err}");
        }
    }

    #[test]
    fn default_is_the_v1_arm() {
        // The written default (docs/FLAGS.md): unmeasured behaviour does not default ON.
        assert_eq!(
            parse_transport(None).unwrap(),
            Glm5TpTransport::HostCanonical
        );
        assert_eq!(Glm5TpTransport::HostCanonical.name(), "host-canonical");
        assert_eq!(Glm5TpTransport::PeerPull.name(), "peer-pull");
    }

    /// The census line is a receipt: every counter must be named in it, because a box window
    /// greps this line and a missing field reads as a zero.
    #[test]
    fn census_line_names_every_counter() {
        let line = transport_census_line(Glm5TpTransport::PeerPull);
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
