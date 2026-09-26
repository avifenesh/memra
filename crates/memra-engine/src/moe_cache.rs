//! EDGE-1 §B: SLRU GPU expert-residency cache (MOE-SLRU-PLAN §B).
//!
//! Stage-1 `moe_ffn` re-stages EVERY routed expert EVERY token over PCIe into one scratch slot.
//! The same ~15-20% of experts recur (the "hot expert" mass), so an SLRU residency cache makes
//! the steady-state re-stage count -> ~0. The cache holds N fixed-address GPU slots (never
//! re-allocated, never fragmented), a `BlockId -> slot` residency table, an SLRU eviction policy
//! (probation + protected segments; the second-miss "ghost" admission filter was measured a net
//! loss in both regimes and removed 2026-07-08 — first-miss admit is the policy) so a one-off cold
//! expert can never evict a genuinely hot one.
//!
//! THE bit-identity property (MOE-SLRU-PLAN §B.3): a cache HIT and a MISS feed `qmatvec_view` the
//! *same* block bytes — the only difference is whether the `memcpy_htod` ran. So the cache-hit
//! weight path is byte-for-byte identical to stage-every-token. TWO gates pin this, and they
//! cover different classes: `src/bin/kernel_check.rs` `d2-cache-bit-identity` pins ONE block of a
//! real GGUF checkpoint (dtypes `IQ3_S | IQ4_XS | Q6_K | Q8_0` — everything else, NVFP4 included,
//! takes its `cells.skip` arm), and `tests/glm5_moe_residency_gpu.rs` pins it END TO END on a
//! glm5_next fixture for the safetensors NVFP4 macro-carrying class, in CI, without a checkpoint.
//!
//! QUANT-FORMAT AGNOSTIC, and that is load-bearing for the safetensors NVFP4 class (glm5_next /
//! GLM-5.3-Flash, Step-3.7-Flash-NVFP4, the unsloth 35B-A3B ST class). A slot is `max_block_bytes`
//! of opaque bytes keyed by `BlockId`; nothing here reads a qtype, a block stride, or a scale.
//! The loader has already repacked modelopt NVFP4 (`weight` + per-16 `weight_scale`) into ONE
//! contiguous per-expert block in memra's internal `block_nvfp4` layout
//! (`nvfp4_repack::repack_modelopt_to_gguf`, `row_bytes = in_f / 64 * 36`), so an NVFP4 block is
//! staged and hit exactly like a k-quant GGUF block. The per-expert `weight_scale_2` MACRO scale
//! is NOT in the block — it rides `HostExps::macros` and is folded post-matmul by the MoE forward
//! — so residency can never move it, and hit/miss stay bit-identical for macro-carrying banks too.
//!
//! Gated behind `MEMRA_MOE_CACHE`, **default ON since 2026-07-08** (`docs/FLAGS.md`: the row is
//! spelled `MEMRA_MOE_CACHE=0` = stage-every-token, i.e. `=0` is the ROLLBACK, not the default).
//! `Engine::moe_cache_enabled()` is `var("MEMRA_MOE_CACHE") != Ok("0")`. This line previously read
//! "default off => current stage-every-token behavior", which was the pre-2026-07-08 state and had
//! been stale for seven weeks; it was corrected in the glm53-flash bring-up lane (2026-08-28) after
//! it was quoted as the live default in a placement plan.

use crate::Engine;
use crate::banked_residency::SLOT_TAIL_PAD_BYTES;
use crate::model::{ExpertKeepalive, ExpertSource};
use crate::spill_pread::{PreadPool, PreadStats, ReadTicket, SpillIoMode};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, HostSlice, SyncOnDrop};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Which projection of an expert (gate/up/down are three distinct GGUF blocks per expert).
pub const PROJ_GATE: u8 = 0;
pub const PROJ_UP: u8 = 1;
pub const PROJ_DOWN: u8 = 2;

/// Residency key: expert `ex` of layer `layer` projection `proj` is a distinct block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockId {
    pub layer: u16,
    pub proj: u8,
    pub ex: u16,
}

/// DAY58 (I8f): the `(id, bytes)` pairs the bank proxy accepted, a dense table indexed by layer,
/// projection and expert (`0` = not validated), grown on the first insertion that needs it and
/// never shrunk while the cache lives. Replaces a SipHash map whose lookup was most of the
/// `validate` stage on every admission.
#[derive(Default)]
pub(crate) struct ValidatedMemo(Vec<Vec<Vec<usize>>>);
impl ValidatedMemo {
    /// True iff the proxy accepted exactly this pair; a zero byte count is never held.
    #[inline]
    pub(crate) fn holds(&self, id: BlockId, bytes: usize) -> bool {
        bytes != 0
            && self
                .0
                .get(usize::from(id.layer))
                .and_then(|p| p.get(usize::from(id.proj)))
                .and_then(|e| e.get(usize::from(id.ex)))
                == Some(&bytes)
    }
    /// Record a pair the proxy accepted (the caller's guard: only after `bank.validate`).
    pub(crate) fn insert(&mut self, id: BlockId, bytes: usize) {
        let (l, p, x) = (
            usize::from(id.layer),
            usize::from(id.proj),
            usize::from(id.ex),
        );
        if self.0.len() <= l {
            self.0.resize_with(l + 1, Vec::new);
        }
        let projs = &mut self.0[l];
        if projs.len() <= p {
            projs.resize_with(p + 1, Vec::new);
        }
        let experts = &mut projs[p];
        if experts.len() <= x {
            experts.resize(x + 1, 0);
        }
        experts[x] = bytes;
    }
}
impl BlockId {
    #[inline]
    pub fn new(layer: u16, proj: u8, ex: u16) -> Self {
        BlockId { layer, proj, ex }
    }
}

/// Where a dispatched block landed (always a retained resident slot since the first-miss-admit
/// policy, 2026-07-08 — the transient staging tier went with the ghost filter).
#[derive(Clone, Copy, Debug)]
pub enum DispatchSlot {
    Resident(usize),
}

/// Intrusive-list constants: `NIL` terminates a list; `seg` tags which segment holds a slot.
const NIL: u32 = u32::MAX;
const SEG_NONE: u8 = 0;
const SEG_PROBATION: u8 = 1;
const SEG_PROTECTED: u8 = 2;

/// Per-slot intrusive doubly-linked node (slot indices are the arena — one node per GPU slot,
/// shared by every class's two segments; a slot is in at most one segment at a time).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SlotLink {
    prev: u32,
    next: u32,
    seg: u8,
}
impl SlotLink {
    const fn none() -> Self {
        SlotLink {
            prev: NIL,
            next: NIL,
            seg: SEG_NONE,
        }
    }
}

/// One SLRU segment as an intrusive doubly-linked list (front = LRU, back = MRU — the exact
/// order contract the previous VecDeque carried). Every operation the hit path needs is O(1):
/// `push_back` (MRU insert), `pop_front` (LRU evict), `unlink` (promotion removal by slot id).
/// This is the Q5 audit fix (research/sweep-audits-20260805/AUDIT.md item 2): the VecDeque
/// `position()+remove()` promotion was O(n_slots) PER HIT once the cache filled — ~46k slots x
/// ~850 hits/token = ~40M host ops/token in the spill regime (measured 48.5 -> 46.0 tok/s on
/// the 35B rtx6000 decode). Same eviction decisions for the same access pattern; only the cost of
/// locating/removing a slot changed.
#[derive(Debug)]
struct SlruList {
    head: u32,
    tail: u32,
    len: usize,
}
impl SlruList {
    const fn new() -> Self {
        SlruList {
            head: NIL,
            tail: NIL,
            len: 0,
        }
    }

    /// MRU insert. The slot must not currently be in any segment.
    fn push_back(&mut self, slot: usize, seg: u8, links: &mut [SlotLink]) {
        debug_assert_eq!(
            links[slot].seg, SEG_NONE,
            "slot {slot} already in a segment"
        );
        let s = slot as u32;
        links[slot] = SlotLink {
            prev: self.tail,
            next: NIL,
            seg,
        };
        if self.tail != NIL {
            links[self.tail as usize].next = s;
        } else {
            self.head = s;
        }
        self.tail = s;
        self.len += 1;
    }

    /// LRU removal.
    fn pop_front(&mut self, links: &mut [SlotLink]) -> Option<usize> {
        if self.head == NIL {
            return None;
        }
        let s = self.head as usize;
        self.unlink(s, links);
        Some(s)
    }

    /// O(1) removal by slot id (the promotion path). The slot must be a member of THIS list.
    fn unlink(&mut self, slot: usize, links: &mut [SlotLink]) {
        let l = links[slot];
        debug_assert_ne!(l.seg, SEG_NONE, "unlink of slot {slot} not in a segment");
        if l.prev != NIL {
            links[l.prev as usize].next = l.next;
        } else {
            debug_assert_eq!(self.head, slot as u32);
            self.head = l.next;
        }
        if l.next != NIL {
            links[l.next as usize].prev = l.prev;
        } else {
            debug_assert_eq!(self.tail, slot as u32);
            self.tail = l.prev;
        }
        links[slot] = SlotLink::none();
        self.len -= 1;
    }

    /// Front-to-back (LRU-to-MRU) iteration — the victim-scan order of the old VecDeque.
    fn iter<'a>(&self, links: &'a [SlotLink]) -> SlruIter<'a> {
        SlruIter {
            links,
            cur: self.head,
        }
    }
}

struct SlruIter<'a> {
    links: &'a [SlotLink],
    cur: u32,
}
impl Iterator for SlruIter<'_> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.cur == NIL {
            return None;
        }
        let s = self.cur as usize;
        self.cur = self.links[s].next;
        Some(s)
    }
}

/// One fixed-address size class with an independent SLRU. Separating queues by capacity prevents a
/// small mixed-layout block from consuming the scarce slots that can hold a larger block.
struct SlotClass {
    capacity: usize,
    probation: SlruList,
    protected: SlruList,
    free: Vec<usize>,
    protected_cap: usize,
}

impl SlotClass {
    /// HIT promotion under a FULL class (SLRU rules 4-6): probation hit -> protected MRU
    /// (demoting protected LRU back to probation MRU while over cap); protected hit -> bump to
    /// MRU; a slot in neither segment (defensive) inserts at protected MRU. Every arm is O(1).
    fn on_hit_full(&mut self, slot: usize, links: &mut [SlotLink]) {
        match links[slot].seg {
            SEG_PROBATION => {
                self.probation.unlink(slot, links);
                self.push_protected(slot, links);
            }
            SEG_PROTECTED => {
                self.protected.unlink(slot, links);
                self.protected.push_back(slot, SEG_PROTECTED, links); // MRU
            }
            // not in either segment (shouldn't happen for a resident slot) — treat as protected MRU
            _ => self.push_protected(slot, links),
        }
    }

    /// Push a slot to protected MRU; if protected exceeds its cap, demote its LRU front to probation.
    fn push_protected(&mut self, slot: usize, links: &mut [SlotLink]) {
        self.protected.push_back(slot, SEG_PROTECTED, links);
        while self.protected.len > self.protected_cap {
            if let Some(demoted) = self.protected.pop_front(links) {
                self.probation.push_back(demoted, SEG_PROBATION, links);
            } else {
                break;
            }
        }
    }

    /// LRU victim (rule 7 per-class): probation front first, else protected front. O(1).
    fn pop_lru(&mut self, links: &mut [SlotLink]) -> Option<usize> {
        self.probation
            .pop_front(links)
            .or_else(|| self.protected.pop_front(links))
    }

    /// Remove `slot` from whichever segment holds it (O(1) via the seg tag). No-op if in neither.
    fn unlink_from_segment(&mut self, slot: usize, links: &mut [SlotLink]) {
        match links[slot].seg {
            SEG_PROBATION => self.probation.unlink(slot, links),
            SEG_PROTECTED => self.protected.unlink(slot, links),
            _ => {}
        }
    }
}

/// SLRU GPU expert-residency cache. Slots remain fixed-address for the cache lifetime. Uniform
/// models use one class; mixed-layout models may preallocate several exact-capacity classes.
pub struct MoeSlotCache {
    // Typed qualification door, default OFF. Owner-only service is never stored here.
    banked: Option<memra_tier::bank::ExpertBankProxy>,
    banked_pending: Option<BankedLease>,
    /// DAY46: leases whose H2D is enqueued, each with the event recorded after its copy (DAY64:
    /// a prefetched group's after its last copy), oldest first. A lease is finished only once its
    /// event has completed (or after a whole-stream drain at teardown); at most `BANKED_INFLIGHT`
    /// leases are open, counted in `banked_inflight_leases` and `banked_prefetched`.
    banked_inflight: VecDeque<(BankedLease, Arc<CudaEvent>)>,
    banked_inflight_leases: usize,
    /// DAY50/DAY64: leases held by prefetched groups not yet in flight (a group counts every
    /// record it leased until its last staged member is consumed).
    banked_prefetched: usize,
    /// DAY64 (I15): each prefetched expert's group while a member is pending, by group number.
    banked_groups: std::collections::HashMap<u64, LeaseGroup>,
    next_group: u64,
    /// `--expert-bank-stages` only: timing events around each banked H2D, read once landed.
    bank_copy_timings: VecDeque<(CudaEvent, CudaEvent)>,
    /// DAY48: `(id, bytes)` pairs the bank's `validate` accepted. The catalog is immutable for
    /// the door's life, so a pair that validated once always does; any other pair (a new id, or
    /// a known id with another byte count) still goes to the proxy, and a refusal is never kept.
    banked_validated: ValidatedMemo,
    /// DAY60: `--moe-dispatch-clock` only.
    dispatch_clock: Option<DispatchClock>,
    /// `--expert-bank-stages` only (DAY40): the CUDA-thread half of the door's log-only stage
    /// clock. `None` without the door and without the flag, so no legacy statement reads it.
    bank_clock: Option<BankAdmitClock>,
    slots: Vec<CudaSlice<u8>>, // fixed GPU buffers; capacities live in `classes`
    slot_class: Vec<usize>,    // slot index -> size-class index
    classes: Vec<SlotClass>,
    /// Per-slot intrusive SLRU node (prev/next/segment). One arena for all classes: a slot
    /// belongs to exactly one class, and to at most one of that class's two segments.
    links: Vec<SlotLink>,
    occupant: Vec<Option<BlockId>>, // slots[s] currently holds occupant[s]  (the residency bitmask)
    table: HashMap<BlockId, usize>, // BlockId -> slot index (O(1) residency lookup)
    /// Exponentially aged online access scores for the optional mixed-layout LFU victim policy.
    /// Scores survive eviction and perf-counter resets; an opt-in decode-epoch decay prevents a
    /// batched prompt from permanently outweighing recent token-to-token reuse.
    frequencies: HashMap<BlockId, f32>,
    /// Copy-stream prefetches that have reserved a slot but are not visible in `table` until the
    /// consumer inserts an explicit compute-stream wait for `ready`. Pending slots are absent from
    /// both SLRU queues, so neither synchronous admission nor another prefetch can evict them.
    pending: HashMap<BlockId, PendingBlock>,
    /// Source owners whose copy completed submission but not yet DMA completion. They are reaped
    /// only after the recorded copy-stream event reports complete.
    inflight_sources: Vec<(Arc<CudaEvent>, ExpertKeepalive)>,
    /// Owners for copies whose completion could not be proved. Kept until a whole-stream drain;
    /// leaked with the GPU slots if teardown cannot establish safety.
    quarantined_sources: Vec<ExpertKeepalive>,
    /// Unique owners used by demand/fallback H2D on the compute stream. `stage_expert` receives a
    /// raw byte slice, so cudarc cannot attach its own source-lifetime event. Retain each backing
    /// allocation once until cache teardown instead of paying one CUDA event per miss.
    compute_sources: HashMap<KeepaliveKey, ExpertKeepalive>,
    /// Opt-in positioned-read backends. Pinned buffers remain owned here until their explicit
    /// compute-stream completion events fire.
    pread: Option<PreadPool>,
    /// Known-next reads submitted to disk workers but not yet consumed by dispatch. They own pinned
    /// buffers, not GPU slots; all CUDA submission remains on the caller thread.
    worker_reads: HashMap<BlockId, WorkerRead>,
    pread_requested: bool,
    pread_fallbacks: u64,
    /// Retained so an event-creation failure after copy submission can be drained again during
    /// teardown. A slot touched by an unprovable copy is quarantined outside every cache queue.
    copy_stream: Arc<CudaStream>,
    copy_stream_unknown: bool,
    compute_stream: Arc<CudaStream>,
    compute_stream_unknown: bool,

    n: usize,
    max_block_bytes: usize,
    size_aware: bool,
    frequency_evict: bool,
    frequency_decay: Option<f32>,
    /// Relative LFU value of a NextN/MTP access. The MTP block is keyed at `u16::MAX` and is
    /// latency-critical during speculative decode, but contributes only one layer of observations
    /// versus the full trunk. Keep the neutral default; local fixed-residency profiling may raise
    /// it after an exact throughput sweep.
    mtp_frequency_weight: f32,
    last_forward_layer: Option<u16>,
    last_forward_t: usize,
    /// Stable-residency mode for heterogeneous CPU/GPU expert execution. Once frozen, callers may
    /// still read resident slots, but must stage cache misses through transient scratch instead of
    /// changing which experts execute on each backend.
    frozen: bool,

    // --- LAUNCH-STRUCTURE STAGE 3 (2026-07-05): device-side expert-pointer indirection ---
    /// Resident-block count per LAYER (all 3 projections summed). When a layer reaches
    /// 3*n_expert every routed block of that layer is cache-resident at a fixed address, so the
    /// whole layer can dispatch via the DEVICE pointer table with ZERO host routing (no router
    /// DtoH, no per-layer stream sync — the round-trip stall the decode profile measured at
    /// ~36us x 40 layers/token). Maintained by admit/evict.
    per_layer: HashMap<u16, u32>,
    /// Per-layer device pointer row [3, n_expert] of slot base addresses (u64), uploaded lazily
    /// when the layer first reads as fully resident. Slots are fixed-address for the cache's
    /// lifetime, so a row stays valid until an eviction touches that layer (which drops the row
    /// -> re-upload on next full residency).
    dev_rows: HashMap<u16, CudaSlice<u64>>,
    /// Layers whose one-shot prewarm was already attempted (success or not) — spill rigs whose
    /// free slots can't hold a full layer must not re-scan 3*n_expert blocks every token.
    prewarm_tried: HashSet<u16>,

    // --- §D.4 instrumentation ---
    // Lifetime counter, only for the explicit bank qualification installer.
    banked_evictions: u64,
    pub hits: u64,
    pub misses: u64,
    pub staged_bytes: u64, // total H2D bytes the cache caused (admit + first-miss transient)
}

/// DAY60 (`--moe-dispatch-clock`, log only): the slot cache's dispatch and prefetch entry points
/// bracketed for BOTH programs (the legacy cache and the door), with each dispatch's outcome, and
/// the prefetch path's parts. The door's own stage clock (`BankAdmitClock`) stays as it is.
#[derive(Default)]
pub(crate) struct DispatchClock {
    dispatch_calls: u64,
    dispatch_ns: u64,
    dispatch_hits: u64,
    dispatch_pending: u64,
    dispatch_sync: u64,
    prefetch_calls: u64,
    prefetch_ns: u64,
    prefetch_issued: u64,
    pf_reserve_ns: u64,
    pf_stage_ns: u64,
    pf_retire_ns: u64,
    pf_resident_ns: u64,
    pf_demand_ns: u64,
}
impl DispatchClock {
    fn line(&self) -> String {
        format!(
            "dispatch_calls={} dispatch_ns={} dispatch_hits={} dispatch_pending={} dispatch_sync={} prefetch_calls={} prefetch_ns={} prefetch_issued={} pf_reserve_ns={} pf_stage_ns={} pf_retire_ns={} pf_resident_ns={} pf_demand_ns={}",
            self.dispatch_calls,
            self.dispatch_ns,
            self.dispatch_hits,
            self.dispatch_pending,
            self.dispatch_sync,
            self.prefetch_calls,
            self.prefetch_ns,
            self.prefetch_issued,
            self.pf_reserve_ns,
            self.pf_stage_ns,
            self.pf_retire_ns,
            self.pf_resident_ns,
            self.pf_demand_ns
        )
    }
}

/// CUDA-thread half of the door's stage clock (`research/spill-c-20260919/DAY40.md` section
/// 2): host wall nanoseconds per bracket of `admit_banked` and the banked branch of
/// `admit_native`, plus the GPU time of each H2D read from two timing events on the compute
/// stream. Log-only: it reads `Instant` and events and changes no decision.
#[derive(Clone, Copy, Debug, Default)]
struct BankAdmitClock {
    admits: u64,
    gpu_hits: u64,
    gpu_misses: u64,
    validate_ns: u64,
    demand_ns: u64,
    reserve_ns: u64,
    enqueue_ns: u64,
    copy_gpu_ns: u64,
    copy_events: u64,
    event_errors: u64,
    drain_ns: u64,
    sync2_ns: u64,
    finish_ns: u64,
    miss_total_ns: u64,
    /// DAY46: front-of-queue retirement (event checks and `finish`), and the full-queue wait.
    retire_ns: u64,
    wait_ns: u64,
    /// DAY50: leases a prefetch took, and misses served from a pending prefetch.
    prefetches: u64,
    prefetch_hits: u64,
}
impl BankAdmitClock {
    fn line(&self) -> String {
        format!(
            "admits={} gpu_hits={} gpu_misses={} validate_ns={} demand_ns={} reserve_ns={} enqueue_ns={} copy_gpu_ns={} copy_events={} event_errors={} drain_ns={} sync2_ns={} finish_ns={} miss_total_ns={} retire_ns={} wait_ns={} prefetches={} prefetch_hits={}",
            self.admits,
            self.gpu_hits,
            self.gpu_misses,
            self.validate_ns,
            self.demand_ns,
            self.reserve_ns,
            self.enqueue_ns,
            self.copy_gpu_ns,
            self.copy_events,
            self.event_errors,
            self.drain_ns,
            self.sync2_ns,
            self.finish_ns,
            self.miss_total_ns,
            self.retire_ns,
            self.wait_ns,
            self.prefetches,
            self.prefetch_hits
        )
    }
}

/// DAY46: leases the door keeps in flight on the compute stream before it waits on the oldest.
pub(crate) const BANKED_INFLIGHT: usize = 32;

fn clock_ns(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

struct PendingBlock {
    slot: usize,
    ready: Arc<CudaEvent>,
    keepalive: Option<ExpertKeepalive>,
    /// DAY50/DAY64: the door prefetch's group this block belongs to; its lease is finished with
    /// the group's, only after every staged member is consumed and the last copy completes.
    group: Option<u64>,
}

/// DAY64 (I15, `research/spill-c-20260919/DAY64.md`): a lease the door holds, one record's (the
/// miss path) or one prefetched expert's group of up to three records under one ticket.
enum BankedLease {
    One(memra_tier::bank::ExpertLeaseToken),
    Group(memra_tier::bank::ExpertGroupToken),
}
impl BankedLease {
    fn leases(&self) -> usize {
        match self {
            BankedLease::One(_) => 1,
            BankedLease::Group(token) => token.records().len(),
        }
    }
    fn finish(
        &self,
        bank: &memra_tier::bank::ExpertBankProxy,
    ) -> memra_tier::contracts::Result<()> {
        match self {
            BankedLease::One(token) => bank.finish(token),
            BankedLease::Group(token) => bank.finish_group(token),
        }
    }
}

/// DAY64 (I15): a prefetched expert's group while any staged member is still pending. The copies
/// run on one stream in block order, so the last staged member's event completing proves every
/// earlier member's copy landed.
struct LeaseGroup {
    token: memra_tier::bank::ExpertGroupToken,
    unconsumed: usize,
    last_ready: Arc<CudaEvent>,
}

#[derive(Clone, Copy)]
struct WorkerRead {
    ticket: ReadTicket,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum KeepaliveKey {
    Pinned(usize),
    Buffer(usize),
    Mmap(usize),
}

impl KeepaliveKey {
    fn from_owner(owner: &ExpertKeepalive) -> Self {
        match owner {
            ExpertKeepalive::Pinned(value) => Self::Pinned(Arc::as_ptr(value) as usize),
            ExpertKeepalive::Buffer(value) => Self::Buffer(Arc::as_ptr(value) as usize),
            ExpertKeepalive::Mmap(value) => Self::Mmap(Arc::as_ptr(value) as usize),
        }
    }
}

/// Exact-length view over one CUDA-pinned pool allocation. cudarc's raw `&[u8]` HostSlice waits
/// for the whole stream before returning, while passing `PinnedHostSlice` would copy its full
/// capacity. This wrapper submits exactly the expert prefix; the caller records and retains the
/// completion event before the backing allocation can be reused.
struct ExactPinnedPrefix<'a>(&'a [u8]);

impl HostSlice<u8> for ExactPinnedPrefix<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [u8], SyncOnDrop<'a>) {
        // SAFETY: the pread staging helpers record an explicit event immediately after the async
        // memcpy and PreadPool retains both allocation and event until it completes.
        (self.0, SyncOnDrop::Record(None))
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [u8], SyncOnDrop<'a>) {
        panic!("ExactPinnedPrefix is a source-only HostSlice")
    }
}

fn stage_on_copy_stream(
    e: &Engine,
    host_bytes: &[u8],
    slot: &mut CudaSlice<u8>,
) -> Result<Arc<CudaEvent>, (Box<dyn std::error::Error>, bool)> {
    // Protect all earlier compute-stream users of a reused slot before the copy stream overwrites it.
    let prior = match e.stream().record_event(None) {
        Ok(prior) => prior,
        Err(err) => return Err((err.into(), true)),
    };
    if let Err(err) = e.copy_stream.wait(&prior) {
        return Err((err.into(), true));
    }
    match e.stage_expert_async(host_bytes, slot, 0) {
        Ok(ready) => Ok(Arc::new(ready)),
        Err(err) => {
            // The H2D may have been submitted before event creation failed. Never release either the
            // destination slot or pinned source until the copy stream has drained.
            match e.copy_stream.synchronize() {
                Ok(()) => Err((err, true)),
                Err(sync_err) => Err((std::io::Error::other(format!(
                    "copy-stream H2D setup failed ({err}); stream drain also failed ({sync_err})"
                )).into(), false)),
            }
        }
    }
}

fn stage_pread_on_compute_stream(
    e: &Engine,
    host_bytes: &[u8],
    slot: &mut CudaSlice<u8>,
) -> Result<Arc<CudaEvent>, Box<dyn std::error::Error>> {
    let ready = Arc::new(e.ctx().new_event(None)?);
    let source = ExactPinnedPrefix(host_bytes);
    let mut dst = slot.slice_mut(0..host_bytes.len());
    e.stream().memcpy_htod(&source, &mut dst)?;
    ready.record(&e.stream())?;
    Ok(ready)
}

/// Allocate the same fraction of every exact block-size class under one byte budget. This avoids
/// biasing residency toward either low-bit or high-bit tiers while eliminating max-slot padding.
fn size_class_plan(block_bytes: &[usize], budget_bytes: usize) -> Vec<(usize, usize)> {
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for &bytes in block_bytes.iter().filter(|&&bytes| bytes > 0) {
        *counts.entry(bytes).or_insert(0) += 1;
    }
    if counts.is_empty() || budget_bytes == 0 {
        return Vec::new();
    }
    let total_bytes: u128 = counts
        .iter()
        .map(|(&bytes, &count)| (bytes as u128 + SLOT_TAIL_PAD_BYTES as u128) * count as u128)
        .sum();
    let budget = budget_bytes as u128;
    let mut plan: Vec<(usize, usize, u128)> = counts
        .iter()
        .map(|(&bytes, &count)| {
            let scaled = count as u128 * budget;
            (
                bytes,
                (scaled / total_bytes).min(count as u128) as usize,
                scaled % total_bytes,
            )
        })
        .collect();
    let mut used: u128 = plan
        .iter()
        .map(|(bytes, count, _)| (*bytes as u128 + SLOT_TAIL_PAD_BYTES as u128) * *count as u128)
        .sum();

    // Hamilton-style remainder pass keeps class proportions close after flooring. There are only
    // a handful of layout classes, so one additional slot per class covers all rounding loss.
    let mut order: Vec<usize> = (0..plan.len()).collect();
    order.sort_by(|&a, &b| plan[b].2.cmp(&plan[a].2).then(a.cmp(&b)));
    for index in order {
        let (bytes, count, _) = plan[index];
        let available = counts[&bytes];
        let required = bytes as u128 + SLOT_TAIL_PAD_BYTES as u128;
        if count < available && used + required <= budget {
            plan[index].1 += 1;
            used += required;
        }
    }
    plan.into_iter()
        .filter_map(|(bytes, count, _)| (count > 0).then_some((bytes, count)))
        .collect()
}

impl MoeSlotCache {
    /// Build the cache sizing N from free VRAM (MOE-SLRU-PLAN §B.4): probe free VRAM AFTER residents
    /// are loaded; N is shared across ALL layers so it must hold the WHOLE-MODEL hot set, not one
    /// layer's. The 35B-A3B keeps its 256 experts HOST-resident, so the GPU has ~20+ GB free at
    /// decode — empirically a 256-slot cache thrashes (~2-7% hit) while a few-thousand-slot cache
    /// reaches ~85%+ steady-state. So the DEFAULT auto-sizes N to fill `MEMRA_MOE_VRAM_FRAC` (default
    /// 0.85) of free VRAM, clamped to [256, ~hot-set]. `MEMRA_MOE_SLOTS` forces an exact N.
    pub fn new(e: &Engine, max_block_bytes: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Self::build(e, max_block_bytes, None)
    }

    /// Exact uniform slot count for the `--experts-via-tier --expert-bank-gpu-bytes=N`
    /// door. The installer already refused a budget below eight slots or above the hard
    /// ceiling, so this path reads no `MEMRA_MOE_SLOTS`, never clamps and never raises
    /// the count; below eight it refuses instead of allocating the floor.
    pub(crate) fn with_exact_slots(
        e: &Engine,
        max_block_bytes: usize,
        slots: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::build(e, max_block_bytes, Some(slots))
    }

    fn build(
        e: &Engine,
        max_block_bytes: usize,
        exact_slots: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (free, _total) = e.ctx().mem_get_info()?;
        let hard_bytes = hard_slot_bytes(free, max_block_bytes);
        let forced_slots = match exact_slots {
            Some(slots) => Some(slots),
            None => std::env::var("MEMRA_MOE_SLOTS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok()),
        };
        let requested_bytes = if let Some(n) = forced_slots {
            n.saturating_mul(max_block_bytes + SLOT_TAIL_PAD_BYTES)
        } else {
            // auto: fill MEMRA_MOE_VRAM_FRAC of free VRAM with slots (default 85%).
            // DEFAULT 0.85 (2026-07-06 local sweep: 0.40=25.0, 0.60=28.0, 0.85=28.5 tok/s on the
            // spill-regime 35B — hit-rate 87.8% -> 99.2%, PCIe 55 -> 3.8 MB/tok; the 0.80
            // hard-headroom cap below still bounds the true allocation, so 0.85 requests the max).
            // Rigs co-running other GPU work should set MEMRA_MOE_VRAM_FRAC lower.
            let frac = std::env::var("MEMRA_MOE_VRAM_FRAC")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.85);
            (free as f64 * frac) as usize
        };
        let budget_bytes = requested_bytes.min(hard_bytes);
        let layout = e.moe_cache_layout().unwrap_or_default();
        let size_aware = forced_slots.is_none()
            && std::env::var("MEMRA_MOE_SIZE_AWARE").as_deref() == Ok("1")
            && !layout.is_empty();
        let frequency_evict = std::env::var("MEMRA_MOE_LFU").as_deref() == Ok("1");
        let frequency_decay = if frequency_evict {
            cache_lfu_decay()
        } else {
            None
        };
        let mtp_frequency_weight = cache_lfu_mtp_weight();
        let mut class_plan = if size_aware {
            size_class_plan(&layout, budget_bytes)
        } else {
            Vec::new()
        };
        if let Some(n) = exact_slots {
            if n < 8 {
                return Err(format!(
                    "experts-via-tier GPU bank slot count {n} is below the eight-slot minimum"
                )
                .into());
            }
            class_plan = vec![(max_block_bytes, n)];
        } else if class_plan.iter().map(|(_, count)| count).sum::<usize>() < 8 {
            let n = (budget_bytes / (max_block_bytes + SLOT_TAIL_PAD_BYTES)).max(8);
            class_plan = vec![(max_block_bytes, n)];
        }
        let n: usize = class_plan.iter().map(|(_, count)| count).sum();

        let mut slots = Vec::with_capacity(n);
        let mut slot_class = Vec::with_capacity(n);
        let mut classes = Vec::with_capacity(class_plan.len());
        let mut occupant = Vec::with_capacity(n);
        for (class_index, &(capacity, count)) in class_plan.iter().enumerate() {
            let start = slots.len();
            for _ in 0..count {
                // Tail pad: wide expert dots may issue an aligned read past the final block.
                slots.push(e.alloc_u8(capacity + SLOT_TAIL_PAD_BYTES)?);
                slot_class.push(class_index);
                occupant.push(None);
            }
            let free_slots = (start..start + count).rev().collect();
            classes.push(SlotClass {
                capacity,
                probation: SlruList::new(),
                protected: SlruList::new(),
                free: free_slots,
                protected_cap: ((count as f64 * 0.8) as usize).max(1),
            });
        }
        let links = vec![SlotLink::none(); n];
        if size_aware {
            let allocated: usize = class_plan
                .iter()
                .map(|(bytes, count)| (bytes + SLOT_TAIL_PAD_BYTES) * count)
                .sum();
            eprintln!(
                "[moe-cache] size-aware fixed slots: {n} slots in {} classes, {:.2} GB / {:.2} GB budget",
                class_plan.len(),
                allocated as f64 / 1e9,
                budget_bytes as f64 / 1e9
            );
        }
        let pread_mode = crate::spill_pread::configured_mode();
        let pread_requested = pread_mode != SpillIoMode::Mmap;
        let pread = if pread_requested {
            match PreadPool::try_new(e, max_block_bytes, pread_mode) {
                Ok(pool) => Some(pool),
                Err(err) => {
                    eprintln!(
                        "[spill-pread] pinned-buffer initialization failed ({err}); using mmap"
                    );
                    None
                }
            }
        } else {
            None
        };

        Ok(MoeSlotCache {
            banked: None,
            banked_pending: None,
            banked_inflight: VecDeque::new(),
            banked_inflight_leases: 0,
            banked_prefetched: 0,
            banked_groups: std::collections::HashMap::new(),
            next_group: 0,
            bank_copy_timings: VecDeque::new(),
            banked_validated: ValidatedMemo::default(),
            dispatch_clock: e.moe_dispatch_clock().then(DispatchClock::default),
            bank_clock: None,
            slots,
            slot_class,
            classes,
            links,
            occupant,
            table: HashMap::with_capacity(n * 2),
            frequencies: HashMap::with_capacity(layout.len().max(n * 2)),
            pending: HashMap::new(),
            inflight_sources: Vec::new(),
            quarantined_sources: Vec::new(),
            compute_sources: HashMap::new(),
            pread,
            worker_reads: HashMap::new(),
            pread_requested,
            pread_fallbacks: 0,
            copy_stream: e.copy_stream.clone(),
            copy_stream_unknown: false,
            compute_stream: e.stream().clone(),
            compute_stream_unknown: false,
            n,
            max_block_bytes,
            size_aware,
            frequency_evict,
            frequency_decay,
            mtp_frequency_weight,
            last_forward_layer: None,
            last_forward_t: 0,
            frozen: false,
            per_layer: HashMap::new(),
            dev_rows: HashMap::new(),
            prewarm_tried: HashSet::new(),
            banked_evictions: 0,
            hits: 0,
            misses: 0,
            staged_bytes: 0,
        })
    }

    #[inline]
    pub fn n_slots(&self) -> usize {
        self.n
    }
    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }
    pub fn freeze(&mut self) {
        if !self.frozen {
            self.frozen = true;
            let (_, complete, one_projection, two_projections, stranded_blocks) =
                self.expert_residency_shape();
            eprintln!(
                "[moe-cache] residency frozen: {} slots, {} resident blocks; \
                 {complete} complete experts, {one_projection} one-projection fragments, \
                 {two_projections} two-projection fragments ({stranded_blocks} stranded blocks)",
                self.n,
                self.table.len()
            );
            let mut mtp_masks = HashMap::<u16, u8>::new();
            for id in self.table.keys().filter(|id| id.layer == u16::MAX) {
                *mtp_masks.entry(id.ex).or_insert(0) |= 1u8 << id.proj;
            }
            if !mtp_masks.is_empty() {
                let complete = mtp_masks.values().filter(|&&mask| mask == 0b111).count();
                eprintln!(
                    "[moe-cache] frozen MTP residency: {} blocks, {complete} complete experts",
                    mtp_masks
                        .values()
                        .map(|mask| mask.count_ones() as usize)
                        .sum::<usize>()
                );
            }
        }
    }

    pub(crate) fn expert_residency_shape(&self) -> (usize, usize, usize, usize, usize) {
        let mut masks = HashMap::<(u16, u16), u8>::new();
        for id in self.table.keys() {
            *masks.entry((id.layer, id.ex)).or_insert(0) |= 1u8 << id.proj;
        }
        let complete = masks.values().filter(|&&mask| mask == 0b111).count();
        let one_projection = masks
            .values()
            .filter(|&&mask| mask.count_ones() == 1)
            .count();
        let two_projections = masks
            .values()
            .filter(|&&mask| mask.count_ones() == 2)
            .count();
        let stranded_blocks = one_projection + 2 * two_projections;
        (
            masks.len(),
            complete,
            one_projection,
            two_projections,
            stranded_blocks,
        )
    }

    #[inline]
    pub fn max_block_bytes(&self) -> usize {
        self.max_block_bytes
    }

    /// O(1) residency check (the ktransformers `generate_gpu_experts_masks` analog).
    #[inline]
    pub fn resident(&self, id: BlockId) -> Option<usize> {
        self.table.get(&id).copied()
    }

    #[inline]
    fn frequency_increment(&self, id: BlockId) -> f32 {
        if id.layer == u16::MAX {
            self.mtp_frequency_weight
        } else {
            1.0
        }
    }

    /// Record a routed block that a fused all-hit path consumed without going through dispatch.
    /// Warmup-only callers use this to make the LFU profile reflect actual grouped GPU traffic;
    /// frozen serving skips it because residency can no longer change.
    pub(crate) fn note_profile_hit(&mut self, id: BlockId) {
        if self.frozen || !self.table.contains_key(&id) {
            return;
        }
        let increment = self.frequency_increment(id);
        *self.frequencies.entry(id).or_insert(0.0) += increment;
    }

    /// HIT promotion (SLRU): on a probation hit promote to protected; on a protected hit bump to MRU.
    ///
    /// O(1) EARLY-OUT (STAGING-ELISION stage, 2026-07-04): while FREE slots remain, `admit` pops
    /// `free` and `evict_one` is unreachable — recency order is dead state until the cache fills.
    /// Kept even though promotion is now O(1) either way (audit-fix Q5, 2026-08-06): skipping it
    /// preserves the not-yet-full ordering behavior BYTE-FOR-BYTE with the pre-fix policy (slots
    /// stay in admission order until the class fills), and on 96GB rigs (slots >= whole-model
    /// block count) every HIT stays a pure table lookup forever.
    ///
    /// FULL-CLASS promotion was the Q5 audit item (research/sweep-audits-20260805/AUDIT.md
    /// item 2): the old VecDeque `position()+remove()` was O(n_slots) PER HIT — at ~46k slots x
    /// ~850 hits/token ~40M host ops/token, measured as the fast-admit A/B regression
    /// 48.5 -> 46.0 tok/s on the 35B rtx6000 decode (the 2026-07-04 fix only DEFERRED the scan to
    /// the spill regime, where the cache is permanently full). The intrusive-list rewrite makes
    /// every arm O(1) with IDENTICAL eviction decisions for the same access pattern (list order
    /// == the old VecDeque order at every step; unit-pinned by `slru_intrusive_tests`).
    /// Bookkeeping-only: the dispatched bytes are identical either way (the D.2 gate pins it).
    fn on_hit(&mut self, slot: usize) {
        let class_index = self.slot_class[slot];
        let class = &mut self.classes[class_index];
        if !class.free.is_empty() {
            return;
        }
        class.on_hit_full(slot, &mut self.links);
    }

    fn remove_occupant(&mut self, slot: usize) {
        if let Some(old) = self.occupant[slot].take() {
            if self.banked.is_some() {
                self.banked_evictions += 1;
            }
            self.table.remove(&old);
            self.on_block_evicted(old.layer);
        }
    }

    /// Lowest cumulative-frequency resident in one class; ties keep ordinary LRU order. A cold
    /// admission therefore becomes the sacrificial slot on the next miss instead of displacing a
    /// prompt-proven hot expert. `keep` protects the expert whose kernels are currently queued.
    /// (Deliberately still O(n_slots) PER EVICTION — the opt-in LFU policy is a full-scan argmin
    /// by definition; the Q5 fix targeted the per-HIT scan. Eviction order over the linked lists
    /// == the old probation-then-protected VecDeque order.)
    fn frequency_victim_in_class(&mut self, class_index: usize, keep: &[BlockId]) -> Option<usize> {
        let class = &self.classes[class_index];
        let candidate = class
            .probation
            .iter(&self.links)
            .chain(class.protected.iter(&self.links))
            .enumerate()
            .filter_map(|(position, slot)| {
                let id = self.occupant[slot]?;
                (!keep.contains(&id)).then_some((
                    self.frequencies.get(&id).copied().unwrap_or(0.0),
                    position,
                    slot,
                ))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let (_, _, slot) = candidate?;
        self.classes[class_index].unlink_from_segment(slot, &mut self.links);
        Some(slot)
    }

    /// Pick the LRU victim from the smallest class that can hold `required` bytes.
    fn evict_one(&mut self, required: usize) -> Option<usize> {
        for class_index in 0..self.classes.len() {
            if self.classes[class_index].capacity < required {
                continue;
            }
            let slot = if self.frequency_evict {
                self.frequency_victim_in_class(class_index, &[])
            } else {
                self.classes[class_index].pop_lru(&mut self.links)
            };
            if let Some(slot) = slot {
                self.remove_occupant(slot);
                return Some(slot);
            }
        }
        None
    }

    /// Pick a resident victim that is not needed by the expert currently being computed. Pending
    /// slots never enter the SLRU queues, so they are excluded automatically. Returns `None` rather
    /// than evicting a protected block; the caller then leaves this block to the synchronous path.
    /// (LRU-to-MRU scan skipping `keep` members — same visit order as the old VecDeque scan;
    /// O(n) worst per PREFETCH eviction only, unchanged from the pre-fix shape.)
    fn evict_one_excluding(&mut self, required: usize, keep: &[BlockId]) -> Option<usize> {
        fn take(
            q: &mut SlruList,
            links: &mut [SlotLink],
            occupant: &[Option<BlockId>],
            keep: &[BlockId],
        ) -> Option<usize> {
            let slot = q
                .iter(links)
                .find(|&s| occupant[s].is_some_and(|id| !keep.contains(&id)))?;
            q.unlink(slot, links);
            Some(slot)
        }
        for class_index in 0..self.classes.len() {
            if self.classes[class_index].capacity < required {
                continue;
            }
            let slot = if self.frequency_evict {
                self.frequency_victim_in_class(class_index, keep)
            } else {
                let class = &mut self.classes[class_index];
                take(&mut class.probation, &mut self.links, &self.occupant, keep)
                    .or_else(|| take(&mut class.protected, &mut self.links, &self.occupant, keep))
            };
            if let Some(slot) = slot {
                self.remove_occupant(slot);
                return Some(slot);
            }
        }
        None
    }

    /// STAGE 3 bookkeeping: a resident block of `layer` was evicted — the layer is no longer fully
    /// resident, so its device pointer row (if uploaded) must be invalidated. NOTE: the row's device
    /// buffer is dropped here, which is safe because the fully-resident fast path is only taken when
    /// `dev_rows` contains the layer at DISPATCH time and all launches consuming the row were
    /// enqueued BEFORE this eviction's staging memcpy on the same stream (single-stream ordering).
    fn on_block_evicted(&mut self, layer: u16) {
        if let Some(c) = self.per_layer.get_mut(&layer) {
            *c -= 1;
        }
        self.dev_rows.remove(&layer);
    }

    fn reserve_slot(&mut self, required: usize) -> Option<usize> {
        for class in &mut self.classes {
            if class.capacity >= required
                && let Some(slot) = class.free.pop()
            {
                return Some(slot);
            }
        }
        self.evict_one(required)
    }

    fn release_reserved_slot(&mut self, slot: usize) {
        debug_assert!(self.occupant[slot].is_none());
        self.classes[self.slot_class[slot]].free.push(slot);
    }

    fn publish(&mut self, id: BlockId, slot: usize) {
        self.occupant[slot] = Some(id);
        self.table.insert(id, slot);
        self.classes[self.slot_class[slot]].probation.push_back(
            slot,
            SEG_PROBATION,
            &mut self.links,
        );
        *self.per_layer.entry(id.layer).or_insert(0) += 1;
    }

    fn reap_copy_sources(&mut self) {
        self.inflight_sources
            .retain(|(ready, _)| !ready.is_complete());
    }

    fn retain_compute_source(&mut self, owner: Option<ExpertKeepalive>) {
        if let Some(owner) = owner {
            let key = KeepaliveKey::from_owner(&owner);
            self.compute_sources.entry(key).or_insert(owner);
        }
    }

    /// Admit a block: evict a victim, stage `host_bytes` into its slot, register residency, place in
    /// probation (new admissions enter probation — they earn promotion on a later hit).
    /// All synchronous admission routes (demand, force, restage and prewarm) meet
    /// here, so installing the typed door cannot retain a legacy mmap bypass.
    fn admit(
        &mut self,
        id: BlockId,
        host_bytes: &[u8],
        e: &Engine,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if self.banked.is_some() {
            return self.admit_banked(id, host_bytes.len(), e);
        }
        self.admit_native(id, host_bytes, e)
    }

    pub(crate) fn bank_pressure(&self) -> (usize, usize, u64) {
        (
            self.slots.len(),
            self.slots.iter().map(|s| s.len()).sum(),
            self.banked_evictions,
        )
    }

    pub(crate) fn install_banked(
        &mut self,
        bank: memra_tier::bank::ExpertBankProxy,
        stage_clock: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.banked.is_some()
            || !self.table.is_empty()
            || !self.pending.is_empty()
            || !self.worker_reads.is_empty()
            || self.frequency_evict
            || self.frozen
            || self.pread_requested
            || self.staged_bytes != 0
        {
            return Err("banked experts require an untouched default-SLRU cache".into());
        }
        self.banked = Some(bank);
        self.bank_clock = stage_clock.then(BankAdmitClock::default);
        Ok(())
    }

    /// The door's stage line (DAY40): this cache's clock, then the owner's report through
    /// the proxy (owner thread only, like every bank call). `Ok(None)` without the door or
    /// without `--expert-bank-stages`.
    pub(crate) fn bank_stage_line(&mut self) -> Result<Option<String>, Box<dyn std::error::Error>> {
        self.settle_copy_timings();
        let (Some(clock), Some(bank)) = (&self.bank_clock, &self.banked) else {
            return Ok(None);
        };
        let owner = bank
            .stage_report()?
            .unwrap_or_else(|| "| owner absent".to_owned());
        Ok(Some(format!("{} {owner}", clock.line())))
    }

    fn admit_banked(
        &mut self,
        id: BlockId,
        bytes: usize,
        e: &Engine,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if self.banked_pending.is_some() || self.frozen {
            return Err("banked expert has unretired H2D or unsupported frozen dispatch".into());
        }
        let bank = self.banked.as_ref().ok_or("bank proxy absent")?.clone();
        let local = (id.layer, id.proj, id.ex);
        let clocked = self.bank_clock.is_some();
        let entered = clocked.then(std::time::Instant::now);
        if !self.banked_validated.holds(id, bytes) {
            bank.validate(local, bytes)?;
            self.banked_validated.insert(id, bytes);
        }
        if let (Some(started), Some(clock)) = (entered, self.bank_clock.as_mut()) {
            clock.admits += 1;
            clock.validate_ns += clock_ns(started);
        }
        if let Some(slot) = self.table.get(&id).copied() {
            self.hits += 1;
            if let Some(clock) = self.bank_clock.as_mut() {
                clock.gpu_hits += 1;
            }
            self.on_hit(slot);
            return Ok(slot);
        }
        self.misses += 1;
        // DAY50: a block a door prefetch staged is consumed only after the compute stream waits
        // on its copy event; its lease then retires on that event like any in-flight lease.
        if let Some(pending) = self.pending.remove(&id) {
            if let Err(err) = e.compute_wait(pending.ready.as_ref()) {
                self.pending.insert(id, pending);
                return Err(err);
            }
            if let Some(group) = pending.group {
                self.consume_group_member(group);
            }
            if let Some(clock) = self.bank_clock.as_mut() {
                clock.gpu_misses += 1;
                clock.prefetch_hits += 1;
            }
            self.publish(id, pending.slot);
            return Ok(pending.slot);
        }
        // DAY61 (I12): finished leases retire where a lease is taken (here and in the prefetch),
        // not on every admission; a GPU hit and a prefetched block's consumption take none, and
        // the in-flight bound is waited on here, before this demand, as before.
        self.retire_banked(&bank, 1)?;
        let demanded = clocked.then(std::time::Instant::now);
        let token = bank.demand(local, bytes)?;
        if let (Some(started), Some(clock)) = (demanded, self.bank_clock.as_mut()) {
            clock.gpu_misses += 1;
            clock.demand_ns += clock_ns(started);
        }
        if token.record() != local {
            // The proxy refuses a mismatched lease before minting a token; this is the
            // consumer's own assertion at the seam. Retire the lease if it ever fires so no
            // host use leaks, and copy nothing.
            bank.finish(&token)?;
            return Err("banked lease names another expert than the one demanded".into());
        }
        // DAY46: borrow the payload on the owner and enqueue its H2D with a completion event;
        // no stream drain. The lease is finished only once that event completes.
        let staged = bank.with_bytes(&token, |payload| self.stage_banked(id, payload, e));
        let finishing = clocked.then(std::time::Instant::now);
        let outcome = match staged {
            Ok(Ok((slot, Some(done)))) => {
                self.banked_inflight
                    .push_back((BankedLease::One(token), Arc::new(done)));
                self.banked_inflight_leases += 1;
                Ok(slot)
            }
            // The copy's completion was proven by a stream drain (its event could not be
            // recorded): finish now.
            Ok(Ok((slot, None))) => match bank.finish(&token) {
                Ok(()) => Ok(slot),
                Err(err) => {
                    self.banked_pending = Some(BankedLease::One(token));
                    Err(err.into())
                }
            },
            // The enqueue failed. A proven drain released the slot and nothing reads the
            // lease; an unknown stream keeps the lease open (never finished) with the slot
            // outside every table.
            Ok(Err(err)) => {
                if self.compute_stream_unknown {
                    self.banked_pending = Some(BankedLease::One(token));
                } else if let Err(finish_err) = bank.finish(&token) {
                    self.banked_pending = Some(BankedLease::One(token));
                    return Err(finish_err.into());
                }
                Err(err)
            }
            // The registry refused the borrow: the lease stays open and further admissions
            // refuse (fail closed).
            Err(err) => {
                self.banked_pending = Some(BankedLease::One(token));
                Err(err.into())
            }
        };
        if let (Some(started), Some(clock)) = (finishing, self.bank_clock.as_mut()) {
            clock.finish_ns += clock_ns(started);
            if let Some(entered) = entered {
                clock.miss_total_ns += clock_ns(entered);
            }
        }
        outcome
    }

    /// DAY46: retire, oldest first, every in-flight lease whose copy event has completed; when the
    /// open leases plus the `need` about to be taken would pass `BANKED_INFLIGHT`, wait on the
    /// oldest event (never the stream) first. A `finish` the registry refuses keeps its lease open
    /// and fails closed. DAY64 (I15): an entry is one record's lease or one prefetched group's.
    fn retire_banked(
        &mut self,
        bank: &memra_tier::bank::ExpertBankProxy,
        need: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let started = self.bank_clock.is_some().then(std::time::Instant::now);
        while let Some((_, done)) = self.banked_inflight.front() {
            if !done.is_complete() {
                if self.banked_inflight_leases + self.banked_prefetched + need <= BANKED_INFLIGHT {
                    break;
                }
                let waiting = self.bank_clock.is_some().then(std::time::Instant::now);
                done.synchronize()?;
                if let (Some(waited), Some(clock)) = (waiting, self.bank_clock.as_mut()) {
                    clock.wait_ns += clock_ns(waited);
                }
            }
            let (lease, _done) = self.banked_inflight.pop_front().expect("front checked");
            self.banked_inflight_leases -= lease.leases();
            if let Err(err) = lease.finish(bank) {
                self.banked_pending = Some(lease);
                return Err(err.into());
            }
        }
        self.settle_copy_timings();
        if let (Some(started), Some(clock)) = (started, self.bank_clock.as_mut()) {
            clock.retire_ns += clock_ns(started);
        }
        Ok(())
    }

    /// DAY64 (I15): a prefetched block of `group` was consumed; once every staged member is, the
    /// group's ticket goes in flight on its last copy's event. A group a failure path took out of
    /// the table (its lease held by `banked_pending`) has nothing left to do here.
    fn consume_group_member(&mut self, group: u64) {
        let Some(entry) = self.banked_groups.get_mut(&group) else {
            return;
        };
        entry.unconsumed -= 1;
        if entry.unconsumed > 0 {
            return;
        }
        let entry = self.banked_groups.remove(&group).expect("present above");
        let leases = entry.token.records().len();
        self.banked_prefetched -= leases;
        self.banked_inflight_leases += leases;
        self.banked_inflight
            .push_back((BankedLease::Group(entry.token), entry.last_ready));
    }

    /// DAY46: every in-flight lease finished after one stream drain (teardown and the gate's
    /// close). A failed drain keeps every lease open. DAY64: every prefetched group finished once,
    /// its unconsumed members' slots returned first.
    pub(crate) fn retire_all_banked(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.banked_inflight.is_empty() && self.banked_groups.is_empty() {
            return Ok(());
        }
        let bank = self.banked.as_ref().ok_or("bank proxy absent")?.clone();
        self.compute_stream.synchronize()?;
        if !self.banked_groups.is_empty() {
            // DAY50: unconsumed prefetches' copies ran on the copy stream.
            self.copy_stream.synchronize()?;
        }
        while let Some((lease, _done)) = self.banked_inflight.pop_front() {
            self.banked_inflight_leases -= lease.leases();
            if let Err(err) = lease.finish(&bank) {
                self.banked_pending = Some(lease);
                return Err(err.into());
            }
        }
        let members: Vec<BlockId> = self
            .pending
            .iter()
            .filter(|(_, p)| p.group.is_some())
            .map(|(id, _)| *id)
            .collect();
        for id in members {
            let pending = self.pending.remove(&id).expect("listed above");
            // The slot's copy landed (both streams drained) but it was never published: return
            // it to its class's free list.
            self.occupant[pending.slot] = None;
            self.release_reserved_slot(pending.slot);
        }
        let groups: Vec<u64> = self.banked_groups.keys().copied().collect();
        for group in groups {
            let entry = self.banked_groups.remove(&group).expect("listed above");
            self.banked_prefetched -= entry.token.records().len();
            let lease = BankedLease::Group(entry.token);
            if let Err(err) = lease.finish(&bank) {
                self.banked_pending = Some(lease);
                return Err(err.into());
            }
        }
        self.settle_copy_timings();
        Ok(())
    }

    /// DAY50/DAY64 (I15, `research/spill-c-20260919/DAY64.md`): the door's prefetch of one
    /// expert's blocks through the owner, under one ticket. Per block, in the order given, the
    /// table and pending skips, the validate memo, the host residency check and a GPU slot outside
    /// `keep` (the same reservations, in the same order, as one call per block made); then one
    /// owner call leases every block that passed; then each lease's bytes go to its slot on the
    /// copy stream after an event that orders the slot's earlier compute-stream readers, in block
    /// order. Returns how many blocks were staged; a block not taken stays with the demand path.
    fn prefetch_banked_group(
        &mut self,
        blocks: &[(BlockId, usize)],
        keep: &[BlockId],
        e: &Engine,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if self.banked_pending.is_some() || self.frozen {
            return Ok(0);
        }
        let mut wanted = [(BlockId::new(0, 0, 0), 0usize); memra_tier::bank::MAX_GROUP];
        let mut count = 0;
        for &(id, bytes) in blocks {
            if count == wanted.len()
                || self.table.contains_key(&id)
                || self.pending.contains_key(&id)
            {
                continue;
            }
            wanted[count] = (id, bytes);
            count += 1;
        }
        if count == 0 {
            return Ok(0);
        }
        let bank = self.banked.as_ref().ok_or("bank proxy absent")?.clone();
        let retiring = self.dispatch_clock.is_some().then(std::time::Instant::now);
        self.retire_banked(&bank, count)?;
        self.dispatch_clock_mark(retiring, |c, ns| c.pf_retire_ns += ns);
        // The prefix that fits the in-flight bound (one call per block took blocks while it did).
        let room =
            BANKED_INFLIGHT.saturating_sub(self.banked_inflight_leases + self.banked_prefetched);
        let wanted = &wanted[..count.min(room)];
        let mut chosen = [(BlockId::new(0, 0, 0), 0usize, 0usize); memra_tier::bank::MAX_GROUP];
        let mut locals = [((0u16, 0u8, 0u16), 0usize); memra_tier::bank::MAX_GROUP];
        let mut taken = 0;
        let mut asked = [(0u16, 0u8, 0u16); memra_tier::bank::MAX_GROUP];
        for (n, &(id, bytes)) in wanted.iter().enumerate() {
            let local = (id.layer, id.proj, id.ex);
            if !self.banked_validated.holds(id, bytes) {
                bank.validate(local, bytes)?;
                self.banked_validated.insert(id, bytes);
            }
            asked[n] = local;
        }
        // DAY77 (I17): the group's host residency in one registry entry; reserving a GPU slot
        // touches no host state, so the answers are the ones one query per member gave.
        let asking = self.dispatch_clock.is_some().then(std::time::Instant::now);
        let resident = bank.host_resident_many(&asked[..wanted.len()]);
        let mut reserving = self.dispatch_clock_mark(asking, |c, ns| c.pf_resident_ns += ns);
        let resident = resident?;
        for (n, &(id, bytes)) in wanted.iter().enumerate() {
            let local = asked[n];
            if !resident[n] {
                continue;
            }
            let reserved = self.reserve_prefetch_slot(bytes, keep);
            reserving = self
                .dispatch_clock_mark(reserving, |c, ns| c.pf_reserve_ns += ns)
                .or(reserving);
            let Some(slot) = reserved else {
                continue;
            };
            chosen[taken] = (id, bytes, slot);
            locals[taken] = (local, bytes);
            taken += 1;
        }
        if taken == 0 {
            return Ok(0);
        }
        let chosen = &chosen[..taken];
        let demanding = self.dispatch_clock.is_some().then(std::time::Instant::now);
        let demanded = bank.demand_many(&locals[..taken]);
        let staging = self.dispatch_clock_mark(demanding, |c, ns| c.pf_demand_ns += ns);
        let token = match demanded {
            Ok(token) => token,
            Err(err) => {
                self.release_chosen(chosen);
                return Err(err.into());
            }
        };
        if token.records().len() != taken
            || token
                .records()
                .iter()
                .zip(&locals[..taken])
                .any(|(got, (want, _))| got != want)
        {
            self.release_chosen(chosen);
            bank.finish_group(&token)?;
            return Err("banked group names other experts than the ones demanded".into());
        }
        let group = self.next_group;
        self.next_group += 1;
        let mut last_ready: Option<Arc<CudaEvent>> = None;
        let mut staged = 0;
        // DAY77 (I17): every member staged in one registry entry, in order; each member's copy and
        // record exactly as one `with_bytes_at` per member did.
        let mut entered = 0;
        let walked = bank.with_bytes_each(&token, |index, payload| {
            entered = index + 1;
            let (id, bytes, slot) = chosen[index];
            match stage_on_copy_stream(e, payload, &mut self.slots[slot]) {
                Ok(ready) => {
                    self.occupant[slot] = Some(id);
                    self.pending.insert(
                        id,
                        PendingBlock {
                            slot,
                            ready: ready.clone(),
                            keepalive: None,
                            group: Some(group),
                        },
                    );
                    last_ready = Some(ready);
                    staged += 1;
                    self.staged_bytes += bytes as u64;
                    if let Some(clock) = self.bank_clock.as_mut() {
                        clock.prefetches += 1;
                    }
                    Ok(())
                }
                Err(failed) => Err((index, failed)),
            }
        });
        match walked {
            Ok(Ok(())) => {}
            Ok(Err((index, (err, reusable)))) => {
                let slot = chosen[index].2;
                // The members after this one were never staged: their slots go back.
                self.release_chosen(&chosen[index + 1..]);
                if reusable {
                    // The copy stream drained: this slot is free and every staged member's
                    // copy landed.
                    self.release_reserved_slot(slot);
                    match last_ready {
                        Some(last_ready) => self.keep_group(group, token, staged, last_ready),
                        None => bank.finish_group(&token)?,
                    }
                } else {
                    // Unknown copy completion: this slot stays outside every table and the
                    // whole group's lease stays open; the copy stream is marked for Drop's
                    // drain.
                    self.copy_stream_unknown = true;
                    self.banked_pending = Some(BankedLease::Group(token));
                }
                self.dispatch_clock_mark(staging, |c, ns| c.pf_stage_ns += ns);
                return Err(err);
            }
            Err(err) => {
                // The registry refused the borrow (before member `entered` was lent): the lease
                // stays open and further admissions refuse (fail closed).
                self.release_chosen(&chosen[entered..]);
                self.banked_pending = Some(BankedLease::Group(token));
                self.dispatch_clock_mark(staging, |c, ns| c.pf_stage_ns += ns);
                return Err(err.into());
            }
        }
        self.dispatch_clock_mark(staging, |c, ns| c.pf_stage_ns += ns);
        self.keep_group(
            group,
            token,
            staged,
            last_ready.expect("every chosen member staged"),
        );
        Ok(staged)
    }

    /// DAY64 (I15): the reserved GPU slots of blocks that were chosen but never staged.
    fn release_chosen(&mut self, chosen: &[(BlockId, usize, usize)]) {
        for &(_, _, slot) in chosen {
            self.release_reserved_slot(slot);
        }
    }

    /// DAY64 (I15): record a group whose `staged` members are pending on the copy stream.
    fn keep_group(
        &mut self,
        group: u64,
        token: memra_tier::bank::ExpertGroupToken,
        staged: usize,
        last_ready: Arc<CudaEvent>,
    ) {
        self.banked_prefetched += token.records().len();
        self.banked_groups.insert(
            group,
            LeaseGroup {
                token,
                unconsumed: staged,
                last_ready,
            },
        );
    }

    /// DAY60: add the time since `started` to one dispatch-clock counter and return a fresh start
    /// for the next part; `None` in and out without the clock.
    fn dispatch_clock_mark(
        &mut self,
        started: Option<std::time::Instant>,
        add: impl FnOnce(&mut DispatchClock, u64),
    ) -> Option<std::time::Instant> {
        let (Some(started), Some(clock)) = (started, self.dispatch_clock.as_mut()) else {
            return None;
        };
        add(clock, clock_ns(started));
        Some(std::time::Instant::now())
    }

    /// The stage clock's copy timing events whose copy has landed (`--expert-bank-stages` only).
    fn settle_copy_timings(&mut self) {
        let Some(clock) = self.bank_clock.as_mut() else {
            return;
        };
        while let Some((_, after)) = self.bank_copy_timings.front() {
            if !after.is_complete() {
                break;
            }
            let (before, after) = self.bank_copy_timings.pop_front().expect("front checked");
            match before.elapsed_ms(&after) {
                Ok(ms) => {
                    clock.copy_gpu_ns += (f64::from(ms) * 1.0e6) as u64;
                    clock.copy_events += 1;
                }
                Err(_) => clock.event_errors += 1,
            }
        }
    }

    /// DAY46: the door's copy of one lease into a GPU slot. Reserve the slot, enqueue the H2D on
    /// the compute stream, record the completion event after it, publish. No drain: every
    /// consumer is on the same stream. A failed enqueue drains to decide the slot (released on a
    /// proven drain, left outside every table on an unknown one); a failed event record drains
    /// and, proven, publishes with `None` (the copy is complete).
    fn stage_banked(
        &mut self,
        id: BlockId,
        host_bytes: &[u8],
        e: &Engine,
    ) -> Result<(usize, Option<CudaEvent>), Box<dyn std::error::Error>> {
        let reserving = self.bank_clock.is_some().then(std::time::Instant::now);
        let slot = self.reserve_slot(host_bytes.len()).ok_or_else(|| {
            std::io::Error::other(format!(
                "no MoE cache slot can hold {} bytes (max class {})",
                host_bytes.len(),
                self.classes.last().map(|class| class.capacity).unwrap_or(0)
            ))
        })?;
        if let (Some(started), Some(clock)) = (reserving, self.bank_clock.as_mut()) {
            clock.reserve_ns += clock_ns(started);
        }
        let timed = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let copy_before = self
            .bank_clock
            .as_ref()
            .map(|_| e.stream().record_event(timed));
        let enqueuing = self.bank_clock.is_some().then(std::time::Instant::now);
        let staged = e.stage_expert(host_bytes, &mut self.slots[slot], 0);
        let enqueue_ns = enqueuing.map(clock_ns);
        let copy_after = copy_before.as_ref().map(|_| e.stream().record_event(timed));
        if let Err(err) = staged {
            return match e.stream().synchronize() {
                Ok(()) => {
                    self.release_reserved_slot(slot);
                    Err(err)
                }
                Err(sync_err) => {
                    // Keep the slot outside free/table/SLRU. Drop retries the stream drain and
                    // leaks every slot if CUDA never provides a completion proof.
                    self.compute_stream_unknown = true;
                    Err(std::io::Error::other(format!(
                        "compute-stream H2D setup failed ({err}); stream drain also failed ({sync_err})"
                    )).into())
                }
            };
        }
        if let Some(clock) = self.bank_clock.as_mut() {
            clock.enqueue_ns += enqueue_ns.unwrap_or(0);
            match (copy_before, copy_after) {
                (Some(Ok(before)), Some(Ok(after))) => {
                    self.bank_copy_timings.push_back((before, after));
                }
                (None, None) => {}
                _ => clock.event_errors += 1,
            }
        }
        let done = match e.stream().record_event(None) {
            Ok(done) => Some(done),
            Err(_) => {
                // No completion event: prove the copy with a drain before publishing.
                if let Err(err) = e.stream().synchronize() {
                    self.compute_stream_unknown = true;
                    return Err(err.into());
                }
                None
            }
        };
        self.staged_bytes += host_bytes.len() as u64;
        self.publish(id, slot);
        Ok((slot, done))
    }

    fn admit_native(
        &mut self,
        id: BlockId,
        host_bytes: &[u8],
        e: &Engine,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let slot = self.reserve_slot(host_bytes.len()).ok_or_else(|| {
            std::io::Error::other(format!(
                "no MoE cache slot can hold {} bytes (max class {})",
                host_bytes.len(),
                self.classes.last().map(|class| class.capacity).unwrap_or(0)
            ))
        })?;
        // Pending copy-stream admissions are not in either SLRU queue, so `evict_one` cannot return
        // an in-flight slot. This synchronous copy and its consumer remain ordered on gpu.stream.
        if let Err(err) = e.stage_expert(host_bytes, &mut self.slots[slot], 0) {
            return match e.stream().synchronize() {
                Ok(()) => {
                    self.release_reserved_slot(slot);
                    Err(err)
                }
                Err(sync_err) => {
                    // Keep the slot outside free/table/SLRU. Drop retries the stream drain and
                    // leaks every slot if CUDA never provides a completion proof.
                    self.compute_stream_unknown = true;
                    Err(std::io::Error::other(format!(
                        "compute-stream H2D setup failed ({err}); stream drain also failed ({sync_err})"
                    )).into())
                }
            };
        }
        self.staged_bytes += host_bytes.len() as u64;
        self.publish(id, slot);
        Ok(slot)
    }

    fn note_pread_fallback(&mut self, reason: &dyn std::fmt::Display) {
        self.pread_fallbacks += 1;
        if let Some(pool) = self.pread.as_mut() {
            pool.note_fallback();
        }
        if self.pread_fallbacks <= 3 {
            eprintln!("[spill-pread] falling back to mmap: {reason}");
        }
    }

    /// Start one MoE forward's worker-I/O scope. Any ticket left by an earlier error/early return is
    /// no longer a valid lookahead target; cancel it before this scope submits its own known-next
    /// reads. In-flight CPU reads keep their buffers until completion restores them safely.
    pub(crate) fn begin_worker_scope(&mut self) {
        if self.worker_reads.is_empty() {
            return;
        }
        let tickets: Vec<_> = self
            .worker_reads
            .drain()
            .map(|(_, read)| read.ticket)
            .collect();
        if let Some(pool) = self.pread.as_mut().filter(|pool| pool.is_worker()) {
            for ticket in tickets {
                let _ = pool.cancel_worker(ticket);
            }
        }
    }

    /// Age cumulative LFU at decode-token boundaries. A batched prompt may touch one block many
    /// times before decode begins; treating those touches as permanent future-use votes poisons a
    /// spill cache. The first T=1 sweep starts a fresh frequency epoch while preserving populated
    /// GPU slots. Later decode sweeps exponentially age history so recent cross-token reuse can
    /// displace stale prompt-specific experts.
    ///
    /// MoE layers are visited in ascending order and the cache is model-global, so
    /// `layer <= previous_layer` marks a new model forward. This changes victim selection only;
    /// every hit and miss still feeds identical expert bytes to the same GPU kernel.
    pub(crate) fn begin_forward_epoch(&mut self, layer: u16, t: usize) {
        let Some(decay) = self.frequency_decay else {
            self.last_forward_layer = Some(layer);
            self.last_forward_t = t;
            return;
        };
        let new_sweep = self
            .last_forward_layer
            .is_some_and(|previous| layer <= previous);
        if new_sweep && t == 1 {
            if self.last_forward_t != 1 {
                self.frequencies.clear();
            } else {
                self.frequencies.retain(|_, score| {
                    *score *= decay;
                    *score >= 1.0e-3
                });
            }
        }
        self.last_forward_layer = Some(layer);
        self.last_forward_t = t;
    }

    fn dispatch_disk(
        &mut self,
        id: BlockId,
        file: &Arc<std::fs::File>,
        offset: u64,
        len: usize,
        fallback: &[u8],
        e: &Engine,
    ) -> Result<DispatchSlot, Box<dyn std::error::Error>> {
        if self.pread.is_none() {
            if self.pread_requested {
                self.note_pread_fallback(&"pinned-buffer backend unavailable");
            }
            return Ok(DispatchSlot::Resident(self.admit(id, fallback, e)?));
        }

        let pending = self.worker_reads.remove(&id);
        let pool = self.pread.as_mut().unwrap();
        let read = if pool.is_worker() {
            let ticket = match pending {
                Some(read) => Ok(Some(read.ticket)),
                None => pool.submit_worker(file.clone(), offset, len),
            };
            match ticket {
                Ok(Some(ticket)) => match pool.wait_worker(ticket) {
                    Ok(index) => Ok(index),
                    Err(err) => {
                        // Read errors normally release in wait_worker. A worker/channel failure may
                        // return earlier; cancel defensively so the next scope cannot lose the slot.
                        let _ = pool.cancel_worker(ticket);
                        Err(err)
                    }
                },
                Ok(None) => Err(std::io::Error::other("worker read ring is busy").into()),
                Err(err) => Err(err),
            }
        } else {
            debug_assert!(pending.is_none());
            pool.read(file.as_ref(), offset, len)
        };
        let index = match read {
            Ok(index) => index,
            Err(err) => {
                self.note_pread_fallback(err.as_ref());
                return Ok(DispatchSlot::Resident(self.admit(id, fallback, e)?));
            }
        };

        // The blocking read happens before eviction, so an I/O failure leaves cache residency
        // untouched and can safely use the mmap oracle.
        let slot = self.reserve_slot(len).ok_or_else(|| {
            std::io::Error::other(format!(
                "no MoE cache slot can hold {len} bytes (max class {})",
                self.classes.last().map(|class| class.capacity).unwrap_or(0)
            ))
        })?;
        let ready = {
            let bytes = match self.pread.as_ref().unwrap().bytes(index, len) {
                Ok(bytes) => bytes,
                Err(err) => {
                    self.pread.as_mut().unwrap().abort_read(index);
                    self.release_reserved_slot(slot);
                    self.note_pread_fallback(err.as_ref());
                    return Ok(DispatchSlot::Resident(self.admit(id, fallback, e)?));
                }
            };
            stage_pread_on_compute_stream(e, bytes, &mut self.slots[slot])
        };
        let ready = match ready {
            Ok(ready) => ready,
            Err(err) => {
                // A memcpy or event-record failure can occur after submission. Synchronize the
                // retained compute stream before either source or destination is reused. If CUDA
                // cannot prove completion, quarantine both and fail instead of risking UAF.
                match e.stream().synchronize() {
                    Ok(()) => {
                        self.pread.as_mut().unwrap().abort_read(index);
                        self.release_reserved_slot(slot);
                        self.note_pread_fallback(err.as_ref());
                        return Ok(DispatchSlot::Resident(self.admit(id, fallback, e)?));
                    }
                    Err(sync_err) => {
                        self.pread.as_mut().unwrap().mark_unknown_h2d(index);
                        return Err(std::io::Error::other(format!(
                            "pread H2D setup failed ({err}); CUDA stream drain also failed ({sync_err})"
                        )).into());
                    }
                }
            }
        };
        self.pread.as_mut().unwrap().mark_h2d(index, ready);
        // Copy and dependent GEMM share the compute stream, so stream order is the consumer fence.
        // Publish only after both memcpy submission and explicit completion-event recording.
        self.staged_bytes += len as u64;
        self.publish(id, slot);
        Ok(DispatchSlot::Resident(slot))
    }

    /// The dispatch decision for one (BlockId, host_bytes). Returns where the block landed; resolve
    /// the device buffer with `buf()`. On the bit-identity-critical path the buffer holds EXACTLY
    /// `host_bytes` either way (a HIT skipped the copy; the prior stage wrote the same bytes).
    ///
    /// Policy (MOE-SLRU-PLAN §B.2, first-miss admit since 2026-07-06):
    /// - HIT  (table[id] = s): promote, return s. ZERO PCIe.
    /// - MISS: admit (stage into a retained slot, evicting an SLRU victim when full).
    pub fn dispatch(
        &mut self,
        id: BlockId,
        host_bytes: &[u8],
        e: &Engine,
    ) -> Result<DispatchSlot, Box<dyn std::error::Error>> {
        self.dispatch_source(
            id,
            ExpertSource::Memory {
                bytes: host_bytes,
                keepalive: None,
            },
            e,
        )
    }

    /// DAY60: the dispatch clock's cumulative line, `None` without the clock.
    pub(crate) fn dispatch_clock_line(&self) -> Option<String> {
        self.dispatch_clock.as_ref().map(DispatchClock::line)
    }

    pub(crate) fn dispatch_source(
        &mut self,
        id: BlockId,
        source: ExpertSource<'_>,
        e: &Engine,
    ) -> Result<DispatchSlot, Box<dyn std::error::Error>> {
        if self.dispatch_clock.is_none() {
            return self.dispatch_source_unclocked(id, source, e);
        }
        // DAY60: the whole dispatch, both programs, and its outcome.
        let (started, hits, was_pending) = (
            std::time::Instant::now(),
            self.hits,
            self.pending.contains_key(&id),
        );
        let result = self.dispatch_source_unclocked(id, source, e);
        let hit = self.hits > hits;
        if let Some(clock) = self.dispatch_clock.as_mut() {
            clock.dispatch_calls += 1;
            clock.dispatch_ns += clock_ns(started);
            if result.is_ok() {
                if hit {
                    clock.dispatch_hits += 1;
                } else if was_pending {
                    clock.dispatch_pending += 1;
                } else {
                    clock.dispatch_sync += 1;
                }
            }
        }
        result
    }

    fn dispatch_source_unclocked(
        &mut self,
        id: BlockId,
        source: ExpertSource<'_>,
        e: &Engine,
    ) -> Result<DispatchSlot, Box<dyn std::error::Error>> {
        if self.banked.is_some() {
            let bytes = match &source {
                ExpertSource::Memory { bytes, .. } => bytes.len(),
                ExpertSource::Disk { len, .. } => *len,
            };
            return self.admit_banked(id, bytes, e).map(DispatchSlot::Resident);
        }
        self.reap_copy_sources();
        let increment = self.frequency_increment(id);
        *self.frequencies.entry(id).or_insert(0.0) += increment;
        if let Some(s) = self.table.get(&id).copied() {
            self.hits += 1;
            self.on_hit(s);
            return Ok(DispatchSlot::Resident(s));
        }
        if let Some(pending) = self.pending.remove(&id) {
            if let Err(err) = e.compute_wait(pending.ready.as_ref()) {
                self.pending.insert(id, pending);
                return Err(err);
            }
            self.misses += 1;
            let slot = pending.slot;
            if let Some(keepalive) = pending.keepalive {
                self.inflight_sources.push((pending.ready, keepalive));
            }
            self.publish(id, slot);
            return Ok(DispatchSlot::Resident(slot));
        }
        self.misses += 1;
        // FIRST-MISS ADMIT (the only policy since 2026-07-08; the second-miss "ghost" filter and
        // its seams MEMRA_MOE_GHOST / MEMRA_MOE_FAST_ADMIT are gone). Measured record: while FREE
        // slots remain, admission evicts nothing — filtering only delayed residency (96GB: 83.7%
        // steady hit-rate instead of ~100%, 74 MB/token avoidable PCIe; 2026-07-04). In the SPILL
        // regime (cache permanently full, local 35B) the filter made every cold block pay TWO H2D
        // copies — ~6% of token PCIe, measured ABOVE its eviction-protection benefit (24.2 -> 25.0
        // tok/s with it off, 2026-07-06). First-miss admit evicts an SLRU victim when full; the
        // SLRU probation segment still protects the protected set. Bit-identity unchanged: the
        // slot holds byte-for-byte the same GGUF block (D.2 gate).
        match source {
            ExpertSource::Memory { bytes, keepalive } => {
                // Retain before H2D submission so even the setup-error path cannot release a
                // pinned/mapped source while CUDA may still be reading it.
                self.retain_compute_source(keepalive);
                let slot = self.admit(id, bytes, e)?;
                Ok(DispatchSlot::Resident(slot))
            }
            ExpertSource::Disk {
                file,
                offset,
                len,
                fallback,
                keepalive,
            } => {
                // The owner is only needed when dispatch_disk falls back to mmap, but retaining
                // the usually shared mmap Arc once keeps every fallback branch simple and safe.
                self.retain_compute_source(Some(keepalive));
                self.dispatch_disk(id, file, offset, len, fallback, e)
            }
        }
    }

    /// Deterministically stage a known-future block on the copy stream. The slot is reserved but is
    /// not considered resident until `dispatch` inserts a compute-stream wait for the returned copy
    /// event. Before overwriting a reused slot, the copy stream waits for all compute work already
    /// queued at this call site; the caller issues prefetch before the current expert's kernels, so
    /// the transfer can overlap those kernels without racing any earlier consumer of the victim.
    ///
    /// `keep` is the current expert's gate/up/down ids. If no safe victim exists, return `false` and
    /// let the normal synchronous miss path handle the block.
    pub fn prefetch(
        &mut self,
        id: BlockId,
        host_bytes: &[u8],
        keep: &[BlockId],
        e: &Engine,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        self.prefetch_source(
            id,
            ExpertSource::Memory {
                bytes: host_bytes,
                keepalive: None,
            },
            keep,
            e,
        )
    }

    fn reserve_prefetch_slot(&mut self, required: usize, keep: &[BlockId]) -> Option<usize> {
        for class in &mut self.classes {
            if class.capacity >= required
                && let Some(slot) = class.free.pop()
            {
                return Some(slot);
            }
        }
        self.evict_one_excluding(required, keep)
    }

    fn prefetch_bytes(
        &mut self,
        id: BlockId,
        host_bytes: &[u8],
        keepalive: Option<ExpertKeepalive>,
        keep: &[BlockId],
        e: &Engine,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let reserving = self.dispatch_clock.is_some().then(std::time::Instant::now);
        let reserved = self.reserve_prefetch_slot(host_bytes.len(), keep);
        let staging = self.dispatch_clock_mark(reserving, |c, ns| c.pf_reserve_ns += ns);
        let Some(slot) = reserved else {
            return Ok(false);
        };
        let staged = stage_on_copy_stream(e, host_bytes, &mut self.slots[slot]);
        self.dispatch_clock_mark(staging, |c, ns| c.pf_stage_ns += ns);
        let ready = match staged {
            Ok(ready) => ready,
            Err((err, reusable)) => {
                if reusable {
                    self.release_reserved_slot(slot);
                } else {
                    // The slot is absent from free/table/SLRU and cannot be reused. Drop retries a
                    // whole copy-stream drain and leaks all slots if CUDA still cannot prove safety.
                    self.copy_stream_unknown = true;
                    if let Some(keepalive) = keepalive {
                        self.quarantined_sources.push(keepalive);
                    }
                    eprintln!(
                        "[moe-cache] quarantining slot {slot} after unprovable copy completion"
                    );
                }
                return Err(err);
            }
        };
        self.occupant[slot] = Some(id);
        self.pending.insert(
            id,
            PendingBlock {
                slot,
                ready,
                keepalive,
                group: None,
            },
        );
        self.staged_bytes += host_bytes.len() as u64;
        Ok(true)
    }

    pub(crate) fn prefetch_source(
        &mut self,
        id: BlockId,
        source: ExpertSource<'_>,
        keep: &[BlockId],
        e: &Engine,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        if self.dispatch_clock.is_none() {
            return self.prefetch_source_unclocked(id, source, keep, e);
        }
        // DAY60: the whole prefetch, both programs.
        let started = std::time::Instant::now();
        let result = self.prefetch_source_unclocked(id, source, keep, e);
        if let Some(clock) = self.dispatch_clock.as_mut() {
            clock.prefetch_calls += 1;
            clock.prefetch_ns += clock_ns(started);
            if matches!(result, Ok(true)) {
                clock.prefetch_issued += 1;
            }
        }
        result
    }

    /// DAY64 (I15, `research/spill-c-20260919/DAY64.md`): one routed expert's three blocks. The
    /// legacy cache (REF) takes them one `prefetch_source` call per block, its program unchanged;
    /// under the door they are leased together (`prefetch_banked_group`), one ticket per expert.
    pub(crate) fn prefetch_expert(
        &mut self,
        blocks: [(BlockId, ExpertSource<'_>); 3],
        keep: &[BlockId],
        e: &Engine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.banked.is_none() {
            for (id, source) in blocks {
                let _ = self.prefetch_source(id, source, keep, e)?;
            }
            return Ok(());
        }
        let sized = blocks.map(|(id, source)| {
            let bytes = match &source {
                ExpertSource::Memory { bytes, .. } => bytes.len(),
                ExpertSource::Disk { len, .. } => *len,
            };
            (id, bytes)
        });
        if self.dispatch_clock.is_none() {
            return self.prefetch_banked_group(&sized, keep, e).map(|_| ());
        }
        // DAY60's clock, the door's grouped form: one prefetch call per block considered.
        let started = std::time::Instant::now();
        let result = self.prefetch_banked_group(&sized, keep, e);
        if let Some(clock) = self.dispatch_clock.as_mut() {
            clock.prefetch_calls += sized.len() as u64;
            clock.prefetch_ns += clock_ns(started);
            if let Ok(staged) = result {
                clock.prefetch_issued += staged as u64;
            }
        }
        result.map(|_| ())
    }

    fn prefetch_source_unclocked(
        &mut self,
        id: BlockId,
        source: ExpertSource<'_>,
        keep: &[BlockId],
        e: &Engine,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        if self.banked.is_some() {
            // No detached legacy prefetch may bypass the owner service: under the door a
            // prefetch takes its lease through the owner (DAY50).
            let bytes = match &source {
                ExpertSource::Memory { bytes, .. } => bytes.len(),
                ExpertSource::Disk { len, .. } => *len,
            };
            return self
                .prefetch_banked_group(&[(id, bytes)], keep, e)
                .map(|staged| staged > 0);
        }
        self.reap_copy_sources();
        if self.table.contains_key(&id)
            || self.pending.contains_key(&id)
            || self.worker_reads.contains_key(&id)
        {
            return Ok(false);
        }
        match source {
            ExpertSource::Memory { bytes, keepalive } => {
                self.prefetch_bytes(id, bytes, keepalive, keep, e)
            }
            ExpertSource::Disk {
                file,
                offset,
                len,
                fallback,
                keepalive,
            } => {
                if self.pread.as_ref().is_some_and(PreadPool::is_worker) {
                    match self.pread.as_mut().unwrap().submit_worker_speculative(
                        file.clone(),
                        offset,
                        len,
                    ) {
                        Ok(Some(ticket)) => {
                            self.worker_reads.insert(id, WorkerRead { ticket });
                            Ok(true)
                        }
                        Ok(None) => Ok(false),
                        Err(err) => {
                            self.note_pread_fallback(err.as_ref());
                            Ok(false)
                        }
                    }
                } else if self.pread.is_some() {
                    // Blocking `pread` remains demand-only so it cannot delay current compute.
                    Ok(false)
                } else {
                    self.prefetch_bytes(id, fallback, Some(keepalive), keep, e)
                }
            }
        }
    }

    /// Pre-warm: force-admit a block (used by the §D.2 bit-identity gate to make all blocks resident).
    pub fn force_admit(
        &mut self,
        id: BlockId,
        host_bytes: &[u8],
        e: &Engine,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if self.banked.is_some() {
            return self.admit_banked(id, host_bytes.len(), e);
        }
        if let Some(s) = self.table.get(&id).copied() {
            return Ok(s);
        }
        self.admit(id, host_bytes, e)
    }

    /// STAGE 3 one-shot PREWARM: force-admit every block of `layer` while FREE slots can hold it
    /// (never evicts — a spill rig whose cache can't fit the layer just skips; organic residency
    /// still applies). Runs at most once per layer (success or not). The H2D copies are the SAME
    /// stage_expert bytes the miss path would issue — bit-identity unchanged; this only front-loads
    /// them so the device-dispatch fast path fires from token 0 instead of after the SLRU fill.
    /// Frozen residency as (layer, proj, ex) triples in slot order, for the freeze-profile
    /// sidecar. Slot order keeps the restage admit sequence close to the original placement.
    pub fn export_residency(&self) -> Vec<(u16, u8, u16)> {
        self.occupant
            .iter()
            .flatten()
            .map(|id| (id.layer, id.proj, id.ex))
            .collect()
    }

    /// Admit one specific block from a saved freeze profile, reading through the layer's
    /// established expert source (the same recipe as `prewarm_layer`, but id-targeted so a
    /// persisted residency set restages without a profiling warmup). Returns false for ids
    /// that no longer resolve (changed plan, pruned expert) — the caller counts and reports.
    pub fn restage_block(
        &mut self,
        id: BlockId,
        m: &crate::hybrid::MoeWeights,
        e: &Engine,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        if self.banked.is_some() {
            return Err("banked expert freeze-profile restage is not qualified".into());
        }
        if self.table.contains_key(&id) {
            return Ok(true);
        }
        let exps = match id.proj {
            PROJ_GATE => &m.gate_exps,
            PROJ_UP => &m.up_exps,
            PROJ_DOWN => &m.down_exps,
            _ => return Ok(false),
        };
        if id.ex as usize >= exps.n_expert {
            return Ok(false);
        }
        if m.active_experts
            .as_ref()
            .is_some_and(|active| !active[id.ex as usize])
        {
            return Ok(false);
        }
        if exps.expert_layout(id.ex as usize).len == 0 {
            return Ok(false);
        }
        match exps.expert_source(id.ex as usize) {
            ExpertSource::Memory { bytes, keepalive } => {
                self.retain_compute_source(keepalive);
                self.admit(id, bytes, e)?;
            }
            ExpertSource::Disk {
                fallback,
                keepalive,
                ..
            } => {
                self.retain_compute_source(Some(keepalive));
                self.admit(id, fallback, e)?;
            }
        }
        Ok(true)
    }

    pub fn prewarm_layer(
        &mut self,
        layer: u16,
        m: &crate::hybrid::MoeWeights,
        e: &Engine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.prewarm_tried.insert(layer) {
            return Ok(());
        }
        let n_expert = m.gate_exps.n_expert;
        if self.pread.is_some()
            && (0..n_expert).any(|ex| {
                matches!(m.gate_exps.expert_source(ex), ExpertSource::Disk { .. })
                    || matches!(m.up_exps.expert_source(ex), ExpertSource::Disk { .. })
                    || matches!(m.down_exps.expert_source(ex), ExpertSource::Disk { .. })
            })
        {
            // Prewarm is a whole-layer scan. It must not silently turn explicit demand I/O back
            // into an mmap walk; organic misses will populate the cache through dispatch_source.
            return Ok(());
        }
        let resident = self.per_layer.get(&layer).copied().unwrap_or(0) as usize;
        let missing = 3 * n_expert - resident;
        if self.size_aware {
            return Ok(());
        } // heterogeneous prewarm needs a per-class fit proof
        if self
            .classes
            .iter()
            .map(|class| class.free.len())
            .sum::<usize>()
            < missing
        {
            return Ok(()); // won't evict for a prewarm
        }
        for ex in 0..n_expert {
            for (proj, exps) in [
                (PROJ_GATE, &m.gate_exps),
                (PROJ_UP, &m.up_exps),
                (PROJ_DOWN, &m.down_exps),
            ] {
                let id = BlockId::new(layer, proj, ex as u16);
                if self.table.contains_key(&id) {
                    continue;
                }
                match exps.expert_source(ex) {
                    ExpertSource::Memory { bytes, keepalive } => {
                        self.retain_compute_source(keepalive);
                        self.admit(id, bytes, e)?;
                    }
                    ExpertSource::Disk {
                        fallback,
                        keepalive,
                        ..
                    } => {
                        self.retain_compute_source(Some(keepalive));
                        self.admit(id, fallback, e)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// STAGE 3: device pointer row for a FULLY-RESIDENT layer. Returns the [3, n_expert] u64 slot
    /// base-address table (proj-major: gate row, up row, down row) if EVERY block of `layer` is
    /// cache-resident, else None (caller falls back to host routing). The row is built+uploaded on
    /// first full residency and reused until an eviction touches the layer. `n_expert` is the
    /// layer's expert count (the full-residency threshold is 3*n_expert blocks).
    pub fn layer_dev_row(
        &mut self,
        layer: u16,
        n_expert: usize,
        e: &Engine,
    ) -> Result<Option<&CudaSlice<u64>>, Box<dyn std::error::Error>> {
        if self.per_layer.get(&layer).copied().unwrap_or(0) as usize != 3 * n_expert {
            return Ok(None);
        }
        if !self.dev_rows.contains_key(&layer) {
            use cudarc::driver::DevicePtr;
            let mut host = vec![0u64; 3 * n_expert];
            for proj in 0..3u8 {
                for ex in 0..n_expert {
                    let Some(&s) = self.table.get(&BlockId::new(layer, proj, ex as u16)) else {
                        // count said fully resident but a block is missing — inconsistent; bail safe.
                        return Ok(None);
                    };
                    let __s_ev = e.stream();
                    let (p, _ev) = self.slots[s].device_ptr(&__s_ev);
                    host[proj as usize * n_expert + ex] = p;
                }
            }
            let row = e.stream().clone_htod(&host)?;
            self.dev_rows.insert(layer, row);
        }
        Ok(self.dev_rows.get(&layer))
    }

    /// Resolve a `DispatchSlot` to the device buffer to feed `qmatvec_view`.
    #[inline]
    pub fn buf(&self, d: DispatchSlot) -> &CudaSlice<u8> {
        match d {
            DispatchSlot::Resident(s) => &self.slots[s],
        }
    }

    /// Read-only access to a slot's device buffer (the `qmatvec_view` source on a HIT).
    #[inline]
    pub fn slot(&self, s: usize) -> &CudaSlice<u8> {
        &self.slots[s]
    }

    /// Hit rate over this cache's lifetime (for the §D.4 print).
    pub fn hit_rate(&self) -> f64 {
        let tot = self.hits + self.misses;
        if tot == 0 {
            0.0
        } else {
            self.hits as f64 / tot as f64
        }
    }

    /// Reset the per-window perf counters (lets the run print steady-state vs warmup separately).
    pub fn reset_counters(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.staged_bytes = 0;
    }

    pub(crate) fn pread_stats(&self) -> Option<PreadStats> {
        if !self.pread_requested {
            return None;
        }
        let mut stats = self
            .pread
            .as_ref()
            .map(PreadPool::stats)
            .unwrap_or_default();
        stats.fallbacks = self.pread_fallbacks;
        Some(stats)
    }
}

fn cache_lfu_decay() -> Option<f32> {
    let raw = std::env::var("MEMRA_MOE_LFU_DECAY").ok()?;
    match parse_cache_lfu_decay(Some(&raw)) {
        Ok(value) => value,
        Err(reason) => {
            eprintln!(
                "[moe-cache] invalid MEMRA_MOE_LFU_DECAY={raw:?} ({reason}); disabling LFU decay"
            );
            None
        }
    }
}

fn cache_lfu_mtp_weight() -> f32 {
    const DEFAULT: f32 = 1.0;
    let raw = std::env::var("MEMRA_MOE_LFU_MTP_WEIGHT").ok();
    match parse_cache_lfu_mtp_weight(raw.as_deref()) {
        Ok(value) => value,
        Err(reason) => {
            eprintln!(
                "[moe-cache] invalid MEMRA_MOE_LFU_MTP_WEIGHT={:?} ({reason}); using {DEFAULT}",
                raw.as_deref().unwrap_or("")
            );
            DEFAULT
        }
    }
}

fn parse_cache_lfu_mtp_weight(raw: Option<&str>) -> Result<f32, &'static str> {
    let value = raw
        .unwrap_or("1")
        .parse::<f32>()
        .map_err(|_| "expected a number")?;
    if value.is_finite() && (0.25..=64.0).contains(&value) {
        Ok(value)
    } else {
        Err("expected a finite multiplier from 0.25 through 64")
    }
}

fn parse_cache_lfu_decay(raw: Option<&str>) -> Result<Option<f32>, &'static str> {
    let Some(raw) = raw else { return Ok(None) };
    let value = raw.parse::<f32>().map_err(|_| "expected a number")?;
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(Some(value))
    } else {
        Err("expected a finite fraction greater than 0 and at most 1")
    }
}

/// Machine hard ceiling for the slot bank: `MEMRA_MOE_HARD_VRAM_FRAC` of free VRAM with
/// two slots of slack. One implementation for the native constructor and for the
/// experts-via-tier installer that refuses a budget above it before any allocation.
pub(crate) fn hard_slot_bytes(free: usize, max_block_bytes: usize) -> usize {
    // Keep two blocks of slack after the machine-specific hard ceiling. The default remains
    // 80%; tightly provisioned spill rigs may raise it only after an OOM-gated local sweep.
    let hard_frac = cache_hard_vram_frac();
    ((free as f64 * hard_frac) as usize).saturating_sub(2 * (max_block_bytes + SLOT_TAIL_PAD_BYTES))
}

fn cache_hard_vram_frac() -> f64 {
    const DEFAULT: f64 = 0.80;
    let raw = std::env::var("MEMRA_MOE_HARD_VRAM_FRAC").ok();
    match parse_cache_hard_vram_frac(raw.as_deref()) {
        Ok(value) => value,
        Err(reason) => {
            eprintln!(
                "[moe-cache] invalid MEMRA_MOE_HARD_VRAM_FRAC={:?} ({reason}); using {DEFAULT}",
                raw.as_deref().unwrap_or("")
            );
            DEFAULT
        }
    }
}

fn parse_cache_hard_vram_frac(raw: Option<&str>) -> Result<f64, &'static str> {
    let value = raw
        .unwrap_or("0.80")
        .parse::<f64>()
        .map_err(|_| "expected a number")?;
    if value.is_finite() && (0.10..=0.95).contains(&value) {
        Ok(value)
    } else {
        Err("expected a finite fraction from 0.10 through 0.95")
    }
}

#[cfg(test)]
mod slru_intrusive_tests {
    //! Q5 policy-equivalence proof (research/audit-fixes2-20260805): the intrusive-list SLRU
    //! must make the SAME eviction decisions for the same access pattern as the pre-fix
    //! VecDeque SLRU. `OldSlru` below is the pre-fix implementation transcribed verbatim
    //! (position()+remove() promotion, probation-then-protected pop_front eviction,
    //! protected_cap demotion loop); both are driven with identical randomized op sequences
    //! and their full segment orders compared after EVERY op.
    use super::{SEG_PROBATION, SEG_PROTECTED, SlotClass, SlotLink, SlruList};
    use std::collections::VecDeque;

    /// The pre-fix policy, verbatim (moe_cache.rs @ 61953206 lines 540-571).
    struct OldSlru {
        probation: VecDeque<usize>,
        protected: VecDeque<usize>,
        protected_cap: usize,
    }
    impl OldSlru {
        fn on_hit_full(&mut self, slot: usize) {
            if let Some(pos) = self.probation.iter().position(|&x| x == slot) {
                self.probation.remove(pos);
                self.push_protected(slot);
            } else if let Some(pos) = self.protected.iter().position(|&x| x == slot) {
                self.protected.remove(pos);
                self.protected.push_back(slot); // MRU
            } else {
                self.push_protected(slot);
            }
        }
        fn push_protected(&mut self, slot: usize) {
            self.protected.push_back(slot);
            while self.protected.len() > self.protected_cap {
                if let Some(demoted) = self.protected.pop_front() {
                    self.probation.push_back(demoted);
                } else {
                    break;
                }
            }
        }
        fn pop_lru(&mut self) -> Option<usize> {
            self.probation
                .pop_front()
                .or_else(|| self.protected.pop_front())
        }
        fn take_excluding(&mut self, banned: &[usize]) -> Option<usize> {
            let take = |q: &mut VecDeque<usize>| {
                q.iter()
                    .position(|&s| !banned.contains(&s))
                    .and_then(|pos| q.remove(pos))
            };
            take(&mut self.probation).or_else(|| take(&mut self.protected))
        }
    }

    fn new_pair(n: usize, protected_cap: usize) -> (SlotClass, Vec<SlotLink>, OldSlru) {
        let class = SlotClass {
            capacity: 1,
            probation: SlruList::new(),
            protected: SlruList::new(),
            free: Vec::new(),
            protected_cap,
        };
        let links = vec![SlotLink::none(); n];
        let old = OldSlru {
            probation: VecDeque::new(),
            protected: VecDeque::new(),
            protected_cap,
        };
        (class, links, old)
    }

    fn orders_match(class: &SlotClass, links: &[SlotLink], old: &OldSlru) -> bool {
        let np: Vec<usize> = class.probation.iter(links).collect();
        let nt: Vec<usize> = class.protected.iter(links).collect();
        let op: Vec<usize> = old.probation.iter().copied().collect();
        let ot: Vec<usize> = old.protected.iter().copied().collect();
        np == op && nt == ot
    }

    /// Deterministic PRNG (SplitMix64) — no dev-dependencies.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    #[test]
    fn same_eviction_decisions_randomized_soak() {
        // Sweep several (n_slots, protected_cap) shapes including cap=1 edge and the 0.8 default.
        for &(n, cap) in &[(8usize, 1usize), (16, 12), (64, 51), (128, 102)] {
            let (mut class, mut links, mut old) = new_pair(n, cap);
            let mut rng = Rng(0xC0FFEE ^ (n as u64) << 8 ^ cap as u64);
            let mut resident: Vec<usize> = Vec::new();
            let mut free: Vec<usize> = (0..n).rev().collect();
            for step in 0..200_000 {
                let op = rng.below(100);
                if op < 55 && !resident.is_empty() {
                    // HIT on a random resident slot (full-class path — free handled below).
                    let slot = resident[rng.below(resident.len())];
                    class.on_hit_full(slot, &mut links);
                    old.on_hit_full(slot);
                } else if op < 80 {
                    // ADMIT: free slot first (publish -> probation MRU), else evict LRU + reuse.
                    let slot = if let Some(s) = free.pop() {
                        s
                    } else {
                        let v_new = class.pop_lru(&mut links);
                        let v_old = old.pop_lru();
                        assert_eq!(
                            v_new, v_old,
                            "victim diverged at step {step} (n={n} cap={cap})"
                        );
                        let v = v_new.unwrap();
                        resident.retain(|&s| s != v);
                        v
                    };
                    class.probation.push_back(slot, SEG_PROBATION, &mut links);
                    old.probation.push_back(slot);
                    resident.push(slot);
                } else if op < 92 && resident.len() > 2 {
                    // PREFETCH eviction: skip up to 3 "keep" slots (evict_one_excluding shape).
                    let banned: Vec<usize> = (0..3.min(resident.len()))
                        .map(|_| resident[rng.below(resident.len())])
                        .collect();
                    let take_new = {
                        let q = &mut class.probation;
                        let found = q.iter(&links).find(|s| !banned.contains(s));
                        match found {
                            Some(s) => {
                                q.unlink(s, &mut links);
                                Some(s)
                            }
                            None => {
                                let q = &mut class.protected;
                                q.iter(&links).find(|s| !banned.contains(s)).inspect(|&s| {
                                    q.unlink(s, &mut links);
                                })
                            }
                        }
                    };
                    let take_old = old.take_excluding(&banned);
                    assert_eq!(
                        take_new, take_old,
                        "excluding-victim diverged at step {step}"
                    );
                    if let Some(v) = take_new {
                        resident.retain(|&s| s != v);
                        free.push(v);
                    }
                } else if !resident.is_empty() {
                    // Defensive arm: hit on a slot in NEITHER segment (unlink first, then hit).
                    let slot = resident[rng.below(resident.len())];
                    match links[slot].seg {
                        SEG_PROBATION => class.probation.unlink(slot, &mut links),
                        SEG_PROTECTED => class.protected.unlink(slot, &mut links),
                        _ => {}
                    }
                    if let Some(pos) = old.probation.iter().position(|&x| x == slot) {
                        old.probation.remove(pos);
                    } else if let Some(pos) = old.protected.iter().position(|&x| x == slot) {
                        old.protected.remove(pos);
                    }
                    class.on_hit_full(slot, &mut links);
                    old.on_hit_full(slot);
                }
                assert!(
                    orders_match(&class, &links, &old),
                    "segment order diverged at step {step} (n={n} cap={cap})"
                );
            }
        }
    }

    #[test]
    fn hit_promotion_is_o1_not_on() {
        // Op-count proof: time-per-hit must not grow with n_slots. 46k slots x 850 hits — the
        // audit's spill shape — as a wall-clock microbench: the old structure walked ~n/2 per
        // hit (~20M steps); the intrusive list does constant work. Assert the per-hit cost at
        // 46k slots stays within 8x of the 1k-slot cost (an O(n) scan would be ~46x+).
        fn bench(n: usize, hits: usize) -> std::time::Duration {
            let (mut class, mut links, _) = new_pair(n, (n as f64 * 0.8) as usize);
            for s in 0..n {
                class.probation.push_back(s, SEG_PROBATION, &mut links);
            }
            let mut rng = Rng(0xBEEF);
            let t0 = std::time::Instant::now();
            for _ in 0..hits {
                class.on_hit_full(rng.below(n), &mut links);
            }
            t0.elapsed()
        }
        // Warm both shapes once (alloc noise), then measure.
        bench(1_000, 10_000);
        bench(46_000, 10_000);
        let small = bench(1_000, 850_000).as_secs_f64() / 850_000.0;
        let large = bench(46_000, 850_000).as_secs_f64() / 850_000.0;
        assert!(
            large < small * 8.0,
            "per-hit cost scaled with n_slots: {:.1}ns @1k vs {:.1}ns @46k",
            small * 1e9,
            large * 1e9
        );
    }

    #[test]
    fn slru_list_basic_invariants() {
        let mut links = vec![SlotLink::none(); 4];
        let mut l = SlruList::new();
        assert_eq!(l.pop_front(&mut links), None);
        l.push_back(2, SEG_PROBATION, &mut links);
        l.push_back(0, SEG_PROBATION, &mut links);
        l.push_back(3, SEG_PROBATION, &mut links);
        assert_eq!(l.iter(&links).collect::<Vec<_>>(), vec![2, 0, 3]);
        assert_eq!(l.len, 3);
        l.unlink(0, &mut links); // middle
        assert_eq!(l.iter(&links).collect::<Vec<_>>(), vec![2, 3]);
        l.unlink(3, &mut links); // tail
        assert_eq!(l.iter(&links).collect::<Vec<_>>(), vec![2]);
        assert_eq!(l.pop_front(&mut links), Some(2)); // head
        assert_eq!(l.len, 0);
        assert_eq!(l.head, super::NIL);
        assert_eq!(l.tail, super::NIL);
        assert!(links.iter().all(|k| k.seg == super::SEG_NONE));
    }
}

#[cfg(test)]
mod vram_fraction_tests {
    use super::{
        SLOT_TAIL_PAD_BYTES, parse_cache_hard_vram_frac, parse_cache_lfu_decay,
        parse_cache_lfu_mtp_weight, size_class_plan,
    };

    #[test]
    fn hard_vram_fraction_defaults_and_rejects_unsafe_values() {
        assert_eq!(parse_cache_hard_vram_frac(None), Ok(0.80));
        assert_eq!(parse_cache_hard_vram_frac(Some("0.82")), Ok(0.82));
        assert!(parse_cache_hard_vram_frac(Some("NaN")).is_err());
        assert_eq!(parse_cache_hard_vram_frac(Some("0.95")), Ok(0.95));
        assert!(parse_cache_hard_vram_frac(Some("0.96")).is_err());
        assert!(parse_cache_hard_vram_frac(Some("1.0")).is_err());
        assert!(parse_cache_hard_vram_frac(Some("bad")).is_err());
    }

    #[test]
    fn lfu_decay_is_opt_in_and_bounded() {
        assert_eq!(parse_cache_lfu_decay(None), Ok(None));
        assert_eq!(parse_cache_lfu_decay(Some("0.8")), Ok(Some(0.8)));
        assert_eq!(parse_cache_lfu_decay(Some("1")), Ok(Some(1.0)));
        for value in ["0", "-0.1", "1.1", "NaN", "bad"] {
            assert!(
                parse_cache_lfu_decay(Some(value)).is_err(),
                "accepted {value}"
            );
        }
    }

    #[test]
    fn lfu_mtp_weight_defaults_and_is_bounded() {
        assert_eq!(parse_cache_lfu_mtp_weight(None), Ok(1.0));
        assert_eq!(parse_cache_lfu_mtp_weight(Some("4")), Ok(4.0));
        for value in ["0", "0.1", "65", "NaN", "bad"] {
            assert!(
                parse_cache_lfu_mtp_weight(Some(value)).is_err(),
                "accepted {value}"
            );
        }
    }

    #[test]
    fn size_class_plan_preserves_classes_and_never_exceeds_budget() {
        let blocks = [100usize, 100, 100, 200, 200, 400];
        let budget = (108 * 2) + 208 + 408;
        let plan = size_class_plan(&blocks, budget);
        assert!(plan.iter().all(|(_, count)| *count > 0));
        assert!(
            plan.iter()
                .map(|(bytes, count)| (bytes + SLOT_TAIL_PAD_BYTES) * count)
                .sum::<usize>()
                <= budget
        );
        assert!(plan.iter().all(|(bytes, count)| {
            *count <= blocks.iter().filter(|block| **block == *bytes).count()
        }));
    }

    #[test]
    fn size_class_plan_returns_full_inventory_when_it_fits() {
        let blocks = [100usize, 100, 200, 400];
        let budget: usize = blocks.iter().map(|bytes| bytes + SLOT_TAIL_PAD_BYTES).sum();
        assert_eq!(
            size_class_plan(&blocks, budget),
            vec![(100, 2), (200, 1), (400, 1)]
        );
    }

    #[test]
    fn size_class_plan_does_not_overflow_on_pathological_sizes() {
        let plan = size_class_plan(&[usize::MAX, usize::MAX], usize::MAX);
        assert!(plan.is_empty());
    }
}

impl Drop for MoeSlotCache {
    fn drop(&mut self) {
        // Event tracking is intentionally disabled in Engine. Drain explicit copy-stream handoffs
        // before either the destination slots or pinned read buffers begin field destruction.
        let mut safe_to_drop_slots = true;
        if !self.banked_inflight.is_empty() || !self.banked_groups.is_empty() {
            // DAY46/DAY50: one drain of each stream proves every in-flight and prefetched copy; a
            // failed drain keeps them all open.
            if self.compute_stream.synchronize().is_err() || self.copy_stream.synchronize().is_err()
            {
                safe_to_drop_slots = false;
            } else if let Some(bank) = &self.banked {
                while let Some((lease, _done)) = self.banked_inflight.pop_front() {
                    // Wrong-thread teardown refuses; the owner registry retains backing.
                    let _ = lease.finish(bank);
                }
                self.banked_inflight_leases = 0;
                // DAY64: every prefetched group once, whatever its members' state.
                for (_, entry) in self.banked_groups.drain() {
                    let _ = BankedLease::Group(entry.token).finish(bank);
                }
                for pending in self.pending.values_mut() {
                    pending.group = None;
                }
                self.banked_prefetched = 0;
            }
        }
        if let Some(lease) = &self.banked_pending {
            if self.compute_stream.synchronize().is_err() {
                safe_to_drop_slots = false;
            } else if let Some(bank) = &self.banked {
                // Wrong-thread teardown refuses; owner registry retains backing.
                if lease.finish(bank).is_ok() {
                    self.banked_pending = None;
                }
            }
        }
        if self.compute_stream_unknown || !self.compute_sources.is_empty() {
            if let Err(err) = self.compute_stream.synchronize() {
                safe_to_drop_slots = false;
                eprintln!(
                    "[moe-cache] unknown compute-stream drain failed ({err}); leaking GPU slots for safety"
                );
                for (_, keepalive) in self.compute_sources.drain() {
                    std::mem::forget(keepalive);
                }
            } else {
                self.compute_stream_unknown = false;
                self.compute_sources.clear();
            }
        }
        let need_copy_drain = self.copy_stream_unknown
            || !self.pending.is_empty()
            || !self.inflight_sources.is_empty()
            || !self.quarantined_sources.is_empty();
        if need_copy_drain {
            if let Err(err) = self.copy_stream.synchronize() {
                safe_to_drop_slots = false;
                eprintln!(
                    "[moe-cache] unknown copy-stream drain failed ({err}); leaking GPU slots for safety"
                );
                for (_, keepalive) in self.inflight_sources.drain(..) {
                    std::mem::forget(keepalive);
                }
                for keepalive in self.quarantined_sources.drain(..) {
                    std::mem::forget(keepalive);
                }
                for (_, pending) in self.pending.drain() {
                    if let Some(keepalive) = pending.keepalive {
                        std::mem::forget(keepalive);
                    }
                }
            } else {
                self.copy_stream_unknown = false;
                self.inflight_sources.clear();
                self.quarantined_sources.clear();
                self.pending.clear();
            }
        }
        if let Some(pool) = self.pread.as_mut() {
            safe_to_drop_slots &= pool.drain();
        } else if self.pread_requested && self.pread_fallbacks != 0 {
            eprintln!(
                "[spill-pread] backend unavailable; mmap_fallbacks={}",
                self.pread_fallbacks
            );
        }
        if !safe_to_drop_slots {
            for slot in self.slots.drain(..) {
                std::mem::forget(slot);
            }
        }
    }
}

// Compile in ordinary library builds, not only tests: PP workers share Engine.
// Putting an owner-only bank/lease in this graph must fail at this boundary,
// before the scoped worker spawns produce a cascade of Send/Sync diagnostics.
const _: fn() = || {
    fn send_sync<T: Send + Sync>() {}
    fn send<T: Send>() {}
    send_sync::<Engine>();
    // Engine protects the cache with a Mutex; its pread receiver is !Sync.
    send::<MoeSlotCache>();
    send_sync::<memra_tier::bank::ExpertBankProxy>();
    send_sync::<memra_tier::bank::ExpertLeaseToken>();
};

#[path = "banked_residency/native.rs"]
mod banked_native;
