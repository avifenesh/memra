//! Native owner-stream H2D/D2H implementation of the frozen tier contract.
//! No worker may submit CUDA work. Unknown completion retains backing and quota.
use crate::PinnedHostBuf;
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
    DevicePtr, DeviceRepr, DriverError, HostSlice, LaunchConfig, PushKernelArg, SyncOnDrop, result,
    sys,
};
use cudarc::nvrtc::Ptx;
use memra_kv::KvPlane;
/// One item's receipt term (the source digest and the witnessed destination digest), re-exported
/// so the worker names the receipt through the engine, its only tier surface.
pub use memra_tier::conformance::ReceiptTerm;
/// WP-A day 42 (`DAY42.md` design S2): the four-lane program's CPU oracle, re-exported for the
/// worker's span receipt cells (the engine is its only tier surface).
pub use memra_tier::conformance::receipt_digest;
use memra_tier::conformance::receipt_digest_from_lanes;
use memra_tier::{bank::SharedBudget, contracts::*};

/// The D2D receipt kernels (WP-A day 22, memra#536 Move 2 slice 3; `cu/tier_receipt.cu`): the
/// copy-stream digest whose CPU oracle is `memra_tier::conformance::receipt_digest`, and the
/// `d2d-delay` fault's spin. Loaded by `new_with_copy_stream` only; `new` has no D2D class.
const TIER_RECEIPT_FATBIN: &[u8] = include_bytes!(env!("MEMRA_TIER_RECEIPT_FATBIN"));
/// The `d2d-delay` fault's early-reader delay ahead of the copy (`inject_d2d_early_reader`).
pub const D2D_DELAY_FAULT_NS: u64 = 200_000_000;
/// WP-A day 38: the `d2h-delay` fault's hold of a demote's landing (`inject_d2h_delay`; a host-side
/// hold since design G''', DAY38 section 14), long enough for the next request to arrive in the
/// copy phase.
pub const D2H_DELAY_FAULT_NS: u64 = 3_000_000_000;
use std::{
    cell::{Cell, Ref, RefCell},
    collections::HashMap,
    rc::Rc,
    sync::Arc,
};

/// The `cuMemHostAlloc` flags of a contract pinned host allocation (WP-A day 13,
/// `research/spill-a-20260919/PINNED-FLAGS.md`). Write-combined is what cudarc's `alloc_pinned`
/// passed before this seam existed: the attribute that suits CPU stores and DMA in both directions
/// but makes every CPU READ of the bytes uncached. The door's byte attestation reads every demoted
/// plane on the CPU (the completion hash, the bind hash), so the arm is a measured question: it
/// moves only on the balanced same-window A/B the owner law asks for, on the card class it applies
/// to, and the production default is per device (`for_device`, WP-A day 14, lead ruling 22,
/// `docs/decisions/PINNED-DESTINATIONS.md`). The enum's `Default` is the arm for a card class with
/// no receipt. No environment variable selects it; the transfer gate passes the arm it measures.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PinnedKind {
    /// `CU_MEMHOSTALLOC_WRITECOMBINED`: cudarc `alloc_pinned`'s flag bits; the arm for every card
    /// class without a receipt.
    #[default]
    WriteCombined,
    /// `cuMemHostAlloc(.., 0)`: cached, page-locked, the OFF tier's `PinnedHostBuf` attribute.
    Cached,
}
impl PinnedKind {
    /// The production default for a device, by card class (WP-A day 14, lead ruling 22): `Cached`
    /// where the class carries a receipt under the pre-registered rule of
    /// `research/spill-a-20260919/DAY13.md`, `WriteCombined` everywhere else. Keyed on the device
    /// name exactly as `parallel::HardwareTarget` keys the product shape (`"RTX PRO 6000"` with
    /// `"Blackwell"`; `"RTX 5090"`); no environment read. Receipts: the RTX PRO 6000 Blackwell
    /// class, `DAY13.md` `pro-single-day13/pinned-ab-160m-s2`, `cached_arm=wins-on-this-card`
    /// (bind hash 77.6 against 1711 ms at 10/10 pairs, D2H 3.00 against 3.01 ms, byte exact 22/22);
    /// the RTX 5090 class, `DAY14.md` `rtx5090-day14/pinned-ab-160m`, `cached_arm=inconclusive`
    /// (bind hash 37.5 against 1449 ms at 10/10 pairs, but D2H cached not above write-combined at
    /// 5/10 pairs and above in both orders' medians, 7.34 against 7.27 ms), so that class keeps
    /// write-combined until a cell on it meets the rule or the lead rules on the D2H clause.
    pub fn for_device(name: &str) -> PinnedKind {
        use crate::parallel::HardwareTarget;
        match HardwareTarget::from_device_name(name) {
            Ok(HardwareTarget::RtxPro6000Blackwell) => PinnedKind::Cached,
            Ok(HardwareTarget::Rtx5090) | Err(_) => PinnedKind::WriteCombined,
        }
    }
    /// The flag bits handed to `cuMemHostAlloc`. `WriteCombined` is the constant cudarc passes.
    pub const fn host_alloc_flags(self) -> u32 {
        match self {
            PinnedKind::WriteCombined => sys::CU_MEMHOSTALLOC_WRITECOMBINED,
            PinnedKind::Cached => 0,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            PinnedKind::WriteCombined => "write-combined",
            PinnedKind::Cached => "cached",
        }
    }
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "write-combined" => Some(PinnedKind::WriteCombined),
            "cached" => Some(PinnedKind::Cached),
            _ => None,
        }
    }
}

/// The engine-owned page-locked backing of a `CudaPinnedLease`: one `cuMemHostAlloc` with the
/// arm's flag bits through the existing `result::malloc_host` FFI, freed with `result::free_host`
/// after its tracking event is synchronized. It presents exactly the `HostSlice` contract cudarc's
/// `PinnedHostSlice` presents (the owner stream waits on the tracking event before a copy and
/// records it after), so the two arms differ in the flag bits and in nothing else: the same
/// `memcpy_htod` / `memcpy_dtoh` driver calls, in the same order, on the same owner stream. Host
/// access (`as_ptr`, `as_slice` and their mutable twins) synchronizes the tracking event first, as
/// `PinnedHostSlice` does, so no CPU read observes bytes a queued copy still owns.
pub struct PinnedBacking {
    ptr: *mut u8,
    len: usize,
    /// WP-A day 63 (`DAY63.md` design L1.1): the allocation's size (a pooled lease's size class).
    /// Nothing reads past `len`: every slice, view and copy spans `len`.
    capacity: usize,
    kind: PinnedKind,
    event: CudaEvent,
}
impl PinnedBacking {
    /// # Safety
    /// The bytes are unset after this call; the caller initializes every byte before any safe
    /// slice exists (`alloc_host_kind` zero-fills through `as_mut_ptr`).
    unsafe fn alloc(
        context: &Arc<CudaContext>,
        bytes: usize,
        kind: PinnedKind,
    ) -> std::result::Result<Self, DriverError> {
        context.bind_to_thread()?;
        let ptr = unsafe { result::malloc_host(bytes, kind.host_alloc_flags())? }.cast::<u8>();
        assert!(!ptr.is_null());
        let event = match context.new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC)) {
            Ok(event) => event,
            Err(e) => {
                // SAFETY: `ptr` came from `malloc_host` above and nothing else owns it.
                let _ = unsafe { result::free_host(ptr.cast()) };
                return Err(e);
            }
        };
        Ok(Self {
            ptr,
            len: bytes,
            capacity: bytes,
            kind,
            event,
        })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// WP-A day 63 (L1.1): the allocation's size; a lease exposes only `len` of it.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// WP-A day 63 (L1.3): a pooled backing handed to a new lease of `len` bytes (at most its
    /// capacity; every byte up to the capacity was initialized by the backing's first lease).
    fn set_len(&mut self, len: usize) {
        assert!(len <= self.capacity, "a lease never exceeds its backing");
        self.len = len;
    }
    pub fn kind(&self) -> PinnedKind {
        self.kind
    }
    /// The context this was created in.
    pub fn context(&self) -> &Arc<CudaContext> {
        self.event.context()
    }
    /// Waits for any scheduled work on the backing to complete, then the host pointer.
    pub fn as_ptr(&self) -> std::result::Result<*const u8, DriverError> {
        self.event.synchronize()?;
        Ok(self.ptr)
    }
    pub fn as_mut_ptr(&mut self) -> std::result::Result<*mut u8, DriverError> {
        self.event.synchronize()?;
        Ok(self.ptr)
    }
    /// WP-A day 34: the start and length WITHOUT the tracking event's host wait, for a read-only view
    /// of an H2D source while its copy may still read it (two readers, no writer: the source's last
    /// writer, the demote's D2H, was observed complete when its receipt was taken). Nothing is read here.
    fn raw_view(&self) -> (*const u8, usize) {
        (self.ptr.cast_const(), self.len)
    }
    pub fn as_slice(&self) -> std::result::Result<&[u8], DriverError> {
        self.event.synchronize()?;
        // SAFETY: `ptr` is a live `malloc_host` allocation of `len` bytes owned by `self`, every
        // byte was initialized before this backing was handed out, and the event synchronize
        // above retired every queued copy that could still write it.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr, self.len) })
    }
    pub fn as_mut_slice(&mut self) -> std::result::Result<&mut [u8], DriverError> {
        self.event.synchronize()?;
        // SAFETY: as `as_slice`, with `&mut self` guaranteeing the single mutable view.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) })
    }
}
impl Drop for PinnedBacking {
    fn drop(&mut self) {
        let ctx = self.event.context();
        ctx.record_err(self.event.synchronize());
        // SAFETY: `ptr` was returned by `malloc_host` and is freed exactly once, here.
        ctx.record_err(unsafe { result::free_host(self.ptr.cast()) });
    }
}
impl HostSlice<u8> for PinnedBacking {
    fn len(&self) -> usize {
        self.len
    }
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [u8], SyncOnDrop<'a>) {
        stream.context().record_err(stream.wait(&self.event));
        (
            // SAFETY: the caller's contract (`HostSlice`): the slice is used only with `stream`,
            // which now waits on every prior use recorded on the tracking event.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) },
            SyncOnDrop::Record(Some((&self.event, stream))),
        )
    }
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [u8], SyncOnDrop<'a>) {
        stream.context().record_err(stream.wait(&self.event));
        (
            // SAFETY: as `stream_synced_slice`, with `&mut self` for the single mutable view.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) },
            SyncOnDrop::Record(Some((&self.event, stream))),
        )
    }
}

fn cuda<T>(r: std::result::Result<T, cudarc::driver::DriverError>) -> Result<T> {
    r.map_err(|e| {
        eprintln!("tier-transfer CUDA error: {e}");
        Error::Quarantined
    })
}
fn event_done(e: &CudaEvent) -> Result<bool> {
    cuda(e.context().bind_to_thread())?;
    // Unlike is_complete(), distinguish NOT_READY from a lost context/event.
    match unsafe { result::event::query(e.cu_event()) } {
        Ok(()) => Ok(true),
        Err(e) if e.0 == sys::cudaError_enum::CUDA_ERROR_NOT_READY => Ok(false),
        Err(e) => cuda(Err(e)),
    }
}

/// A real CUDA-pinned, exclusive and initialized host allocation. Not Send.
/// All bytes, including padding, are initialized before any safe slice exists.
pub struct CudaPinnedLease {
    allocation: Rc<PinnedAllocation>,
}
struct PinnedAllocation {
    backing: Option<PinnedBacking>,
    charge: Option<ChargedLease>,
    pin: Option<LeasePin>,
    governor: SharedBudget,
    /// WP-A day 63 (L1.4): where the backing goes when the lease drops.
    pool: Option<Rc<LeasePool>>,
}
impl std::fmt::Debug for CudaPinnedLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaPinnedLease")
            .field("bytes", &self.storage_bytes())
            .field("kind", &self.pinned_kind())
            .finish()
    }
}
impl PinnedLease for CudaPinnedLease {
    fn storage_bytes(&self) -> u64 {
        self.allocation
            .backing
            .as_ref()
            .map_or(0, |b| b.len() as u64)
    }
    fn valid_bytes(&self) -> u64 {
        self.storage_bytes()
    }
    fn alignment(&self) -> u32 {
        1
    }
    fn numa_node(&self) -> Option<u32> {
        None
    }
    fn bytes(&self) -> Result<&[u8]> {
        cuda(
            self.allocation
                .backing
                .as_ref()
                .ok_or(Error::AlreadyReleased)?
                .as_slice(),
        )
    }
}
impl CudaPinnedLease {
    /// The arm this lease was allocated under; `None` once the backing is released.
    pub fn pinned_kind(&self) -> Option<PinnedKind> {
        self.allocation.backing.as_ref().map(PinnedBacking::kind)
    }
    pub fn write(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() as u64 != self.valid_bytes() {
            return Err(Error::InvalidLayout);
        }
        cuda(
            Rc::get_mut(&mut self.allocation)
                .ok_or(Error::Busy)?
                .backing
                .as_mut()
                .ok_or(Error::AlreadyReleased)?
                .as_mut_slice(),
        )?
        .copy_from_slice(bytes);
        Ok(())
    }
    /// WP-A day 35 (`DAY35.md` design M'): a read-only view of this lease's bytes for a hashing
    /// thread off the owner thread (the lease itself is not `Send`: an `Rc` and a CUDA event). The
    /// pinned slice's tracking event is synchronized first, exactly as `bytes()` does, so the view
    /// covers settled bytes.
    ///
    /// # Safety
    ///
    /// The caller keeps this lease (and every clone of it) alive and unwritten until the view is
    /// dropped; a caller that cannot know the reader is done leaks the lease instead of freeing it.
    pub unsafe fn read_view(&self) -> Result<PinnedLeaseView> {
        let b = self.bytes()?;
        Ok(PinnedLeaseView {
            ptr: b.as_ptr(),
            len: b.len(),
        })
    }
}
/// WP-A day 35 (`DAY35.md` design M'): a read-only view of a taken pinned lease's bytes
/// (`CudaPinnedLease::read_view`), for the bind's re-hash on the caller's hash helper.
pub struct PinnedLeaseView {
    ptr: *const u8,
    len: usize,
}
// SAFETY: read-only; the constructor's contract keeps the bytes alive and unwritten while the view
// exists, so moving it to another thread moves only that read access.
unsafe impl Send for PinnedLeaseView {}
impl PinnedLeaseView {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The same `checksum` program the bind runs, over the same bytes.
    pub fn digest(&self) -> Digest {
        // SAFETY: `ptr` spans `len` settled bytes the constructor's caller keeps alive and unwritten.
        checksum(unsafe { std::slice::from_raw_parts(self.ptr, self.len) })
    }
}
impl std::fmt::Debug for PinnedLeaseView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedLeaseView")
            .field("len", &self.len)
            .finish()
    }
}
impl Drop for PinnedAllocation {
    fn drop(&mut self) {
        // Live/unknown DMA leases never reach here: Entry's shutdown path leaks them.
        // Synchronize the pinned allocation's own cudarc tracking event before free.
        if self.backing.as_ref().is_some_and(|b| b.as_ptr().is_err()) {
            std::mem::forget(self.backing.take());
            std::mem::forget(self.pin.take());
            std::mem::forget(self.charge.take());
            return;
        }
        let backing = self.backing.take();
        drop(self.pin.take());
        if let Some(c) = self.charge.take()
            && self.governor.borrow_mut().release(&c).is_err()
        {
            std::mem::forget(c);
        }
        // WP-A day 63 (`DAY63.md` design L1.4): the backing parks in the pool (charged to the pool
        // tenant) instead of `cuMemFreeHost`, unless the pool is closed or full; then it frees.
        match (backing, &self.pool) {
            (Some(b), Some(pool)) => pool.put(b),
            (b, _) => drop(b),
        }
    }
}

/// WP-A day 63 (`DAY63.md` design L1): the size class of a pooled lease backing. At or below 1 MiB
/// the next power of two, above it the next 1 MiB multiple.
pub fn lease_class(bytes: usize) -> usize {
    const MIB: usize = 1 << 20;
    if bytes <= MIB {
        bytes.max(1).next_power_of_two()
    } else {
        bytes.div_ceil(MIB) * MIB
    }
}

/// WP-A day 63 (`DAY63.md` design L1): the transfer engine's pool of pinned lease backings. A
/// dropped lease's backing parks here (its tracking event synchronized) and the next allocation of
/// its class and kind takes it, so no `cuMemFreeHost` (a context-wide wait) and no fresh
/// `cuMemHostAlloc` plus zero fill reach the serving path in the steady state. Idle bytes are
/// charged to the pinned ledger under the pool tenant, capped at `cap`. Closed at the tier's latch
/// and when the engine drops: every idle backing frees, every pool charge releases.
pub struct LeasePool {
    idle: RefCell<Vec<(PinnedBacking, ChargedLease)>>,
    idle_bytes: Cell<u64>,
    cap: Cell<u64>,
    open: Cell<bool>,
    governor: SharedBudget,
    dimensions: usize,
    /// (taken from the pool, allocated fresh) since the engine was built (log only).
    counts: Cell<(u64, u64)>,
}
impl LeasePool {
    fn new(governor: SharedBudget) -> Self {
        let dimensions = governor.borrow().used().device.len();
        Self {
            idle: RefCell::new(Vec::new()),
            idle_bytes: Cell::new(0),
            cap: Cell::new(0),
            open: Cell::new(true),
            governor,
            dimensions,
            counts: Cell::new((0, 0)),
        }
    }
    /// The pool tenant's digest (a domain disjoint from every tenant salt).
    pub fn tenant() -> [u8; 32] {
        digest("host-tier-lease-pool", b"pinned lease backings")
    }
    /// An idle backing of exactly this class and kind, its pool charge released.
    fn take(&self, class: usize, kind: PinnedKind) -> Option<PinnedBacking> {
        let (backing, charge) = {
            let mut idle = self.idle.borrow_mut();
            let i = idle
                .iter()
                .position(|(b, _)| b.capacity == class && b.kind == kind)?;
            idle.swap_remove(i)
        };
        self.idle_bytes.set(self.idle_bytes.get() - class as u64);
        if self.governor.borrow_mut().release(&charge).is_err() {
            std::mem::forget(charge);
        }
        Some(backing)
    }
    /// A dropped lease's backing: parked under a pool charge when the pool is open and has room,
    /// freed otherwise.
    fn put(&self, backing: PinnedBacking) {
        let class = backing.capacity as u64;
        if !self.open.get() || self.idle_bytes.get() + class > self.cap.get() {
            drop(backing);
            return;
        }
        let mut request = BudgetRequest {
            bytes: TierBudget::zero(self.dimensions),
            priority: Priority::Backup,
            deadline: Deadline(u64::MAX),
            tenant: Self::tenant(),
        };
        request.bytes.pinned = class;
        let charge = self.governor.borrow_mut().reserve(&request);
        match charge {
            Ok(charge) => {
                self.idle_bytes.set(self.idle_bytes.get() + class);
                self.idle.borrow_mut().push((backing, charge));
            }
            Err(_) => drop(backing),
        }
    }
    /// Close: every idle backing frees and every pool charge releases; later drops free.
    /// Returns (backings freed, bytes). Idempotent.
    pub fn close(&self) -> (usize, u64) {
        self.open.set(false);
        let idle = std::mem::take(&mut *self.idle.borrow_mut());
        let (n, bytes) = (idle.len(), self.idle_bytes.replace(0));
        for (backing, charge) in idle {
            drop(backing);
            if self.governor.borrow_mut().release(&charge).is_err() {
                std::mem::forget(charge);
            }
        }
        (n, bytes)
    }
}
struct Item {
    host: Option<CudaPinnedLease>,
    device: Option<DeviceLease>,
    bytes: u64,
    direction: CopyDirection,
    event: Option<CudaEvent>,
    taken: bool,
    source_retired: bool,
}
#[derive(Default)]
struct Retention {
    graph_pins: Rc<()>,
    consumer_event: Option<(FenceId, CudaEvent)>,
}
impl Retention {
    fn idle(&self) -> Result<bool> {
        Ok(Rc::strong_count(&self.graph_pins) == 1
            && match &self.consumer_event {
                Some((_, event)) => event_done(event)?,
                None => true,
            })
    }
}
struct Entry {
    items: Vec<Option<Item>>,
    completion: Completion,
    expected: Vec<Vec<SegmentExpectation>>,
    charge: Option<ChargedLease>,
    cancelled: bool,
    published: bool,
    retired: bool,
    unknown: bool,
    source: Retention,
    destination: Retention,
    /// WP-A day 22 (Move 2 slice 3): the D2D receipt's lanes (`ReceiptScratch`); `None` for an
    /// H2D or D2H batch, whose receipt is the host-side `checksum` in `progress`.
    receipt: Option<ReceiptScratch>,
    /// WP-A day 30 (Move 2 owed item 1, the D2H half): the f32 spans attached to a D2H batch
    /// (`submit_d2h_spans`), owned here from attach until `take_d2h_spans`.
    spans: Option<SpanBatch>,
    /// The batch's spans were taken back (a second take is `AlreadyReleased`).
    spans_taken: bool,
    /// WP-A day 32 (Move 2 owed item 1, the H2D half): the f32 spans attached to an H2D batch
    /// (`submit_h2d_spans`), owned here from attach until `take_h2d_spans`.
    h2d_spans: Option<H2dSpanBatch>,
    /// The batch's H2D spans were taken back (a second take is `AlreadyReleased`).
    h2d_spans_taken: bool,
    /// WP-A day 34 (`DAY34.md` design K): the H2D items' completion checksums, deferred to the
    /// caller's hash helper (`defer_h2d_checksums`); `None` keeps them in `progress`.
    deferred: Option<DeferredSums>,
    /// WP-A day 38 (`DAY38.md` design G, `memra_tier::conformance::d2h_device_receipt`): a D2H
    /// batch's receipt taken on the copy stream (design G4) over each item's DEVICE source (the program
    /// `checksum`, byte for byte), sealed by one receipt event; `None` keeps the owner-thread
    /// checksum of the landed bytes in `progress` (a batch on the owner stream).
    d2h_receipt: Option<D2hDeviceReceipt>,
}
/// WP-A day 38: one D2H batch's device receipt: the digests (32 bytes per item, the batch's item
/// order) on the device and their pinned twin, sealed by `scratch.event`; `timing` brackets the
/// digest kernels on the copy stream (a log-only reading, never waited on). `hold_until`: the
/// `d2h-delay` fault's host-side hold (designs G''' and G4, DAY38 sections 14 and 17): no item lands before it,
/// and `synchronize` returns only after it. `None` in production.
struct D2hDeviceReceipt {
    scratch: ReceiptScratch,
    timing: Option<(CudaEvent, CudaEvent)>,
    hold_until: Option<std::time::Instant>,
}
/// The by-value argument of `d2h_receipt_sha256` (`cu/tier_receipt.cu` `ReceiptItems`): up to
/// `RECEIPT_ITEMS` items per launch, their device pointers and byte lengths.
const RECEIPT_ITEMS: usize = 64;
#[repr(C)]
#[derive(Clone, Copy)]
struct ReceiptItems {
    n: u64,
    ptr: [u64; RECEIPT_ITEMS],
    len: [u64; RECEIPT_ITEMS],
}
// SAFETY: plain `#[repr(C)]` integers, laid out exactly as the kernel's `struct ReceiptItems`
// (`u64 n; u64 ptr[64]; u64 len[64]`), passed by value.
unsafe impl DeviceRepr for ReceiptItems {}
/// WP-A day 34: the deferred checksums of one H2D batch: the views still out and each item's supplied
/// digest (the item lands only with it, `memra_tier::conformance::h2d_deferred_checksum_lands_with_its_digests`).
struct DeferredSums {
    out: usize,
    supplied: Vec<Option<Digest>>,
}
/// WP-A day 34 (`DAY34.md` design K): a read-only view of one accepted H2D item's host source, for a
/// hashing thread off the owner thread. The engine keeps the source until the view comes back through
/// `supply_h2d_checksums` (the ticket cannot land, retire its sources or retire until then), and nothing
/// writes the source meanwhile (the entry's own handle cannot write while the ticket's twin lives).
pub struct H2dSourceView {
    item: u32,
    ptr: *const u8,
    len: usize,
}
// SAFETY: the view is read-only and the engine keeps its bytes alive and unwritten until it returns
// (see the type's doc); moving it to another thread moves only that read access.
unsafe impl Send for H2dSourceView {}
impl H2dSourceView {
    /// The item this view reads.
    pub fn item(&self) -> u32 {
        self.item
    }
    /// The bytes it covers (the item's whole host source, as `progress` hashes it).
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The same `checksum` program `progress` runs, over the same bytes.
    pub fn digest(&self) -> Digest {
        // SAFETY: `ptr` spans `len` initialized bytes of a pinned host allocation the engine keeps
        // alive and unwritten while this view is out (the type's contract).
        checksum(unsafe { std::slice::from_raw_parts(self.ptr, self.len) })
    }
}
impl std::fmt::Debug for H2dSourceView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H2dSourceView")
            .field("item", &self.item)
            .field("len", &self.len)
            .finish()
    }
}
impl Drop for Entry {
    fn drop(&mut self) {
        if !self.retired {
            // Drop is not a DMA/consumer/graph fence. Keep every owned input and
            // its accounting alive, including after a submission error.
            std::mem::forget(std::mem::take(&mut self.items));
            std::mem::forget(self.charge.take());
            std::mem::forget(self.receipt.take());
            // A span's copy may still read its source and write its destination: a leak, never
            // a free, exactly as the items.
            std::mem::forget(self.spans.take());
            // WP-A day 32: the same for an H2D span (its destination may be written, its staging
            // source read).
            std::mem::forget(self.h2d_spans.take());
            // WP-A day 38 (design G'): the D2H device receipt's lanes may still be written by the
            // receipt kernel and its pinned twin by the lanes' D2H (on the copy stream since G4); a
            // free here (the lanes on the owner stream, unordered with that stream) could hand the
            // memory out under that write. A leak, never a free.
            std::mem::forget(self.d2h_receipt.take());
        }
    }
}
/// One typed f32 span of a D2H demote batch (WP-A day 30, memra#536 Move 2 owed item 1, the D2H
/// half; `memra_tier::conformance::d2h_span_batch`): the whole of an OWNED device source (an
/// evicted prefix entry's recurrent plane) copied into an OWNED cached pinned destination of
/// exactly its byte length. Attached to a live D2H ticket by `CudaTransfers::submit_d2h_spans`
/// and handed back, source and landed destination, by `take_d2h_spans`.
pub struct D2hSpan {
    pub source: CudaSlice<f32>,
    pub destination: PinnedHostBuf,
}
impl std::fmt::Debug for D2hSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("D2hSpan")
            .field("source_len", &self.source.len())
            .field("destination_bytes", &self.destination.len())
            .finish()
    }
}
/// A batch's spans in attach order, each with the completion event recorded after its copy on
/// the copy stream (`None` when the enqueue or the record failed: the batch is quarantined).
struct SpanBatch {
    slots: Vec<(D2hSpan, Option<CudaEvent>)>,
    landed: bool,
    /// WP-A day 42 (`DAY42.md` design S2): the spans' receipt lanes, 64 bytes per span (the four
    /// SOURCE lanes, written by one launch per 64 spans ahead of the copies, then the four LANDED
    /// lanes, written at the seal). The batch lands with its copies; the lanes leave with the spans
    /// at the take, as the batch's `SpanReceipt`. `None` on an engine without the receipt kernels.
    receipt: Option<ReceiptScratch>,
}
/// WP-A day 42 (`DAY42.md` design S2): the id of one D2H span batch's receipt, handed out by
/// `CudaTransfers::take_d2h_spans` with the landed spans; `seal_d2h_span_receipt` enqueues its
/// landed digests, `d2h_span_receipt` reads its (source, landed) pairs once observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanReceiptId(u64);
/// WP-A day 42 (design S2): the landed spans of a D2H batch in attach order and their receipt's
/// id (`None` on an engine without the receipt kernels).
pub struct TakenD2hSpans {
    pub spans: Vec<D2hSpan>,
    pub receipt: Option<SpanReceiptId>,
}
/// WP-A day 42 (design S2): one D2H span batch's receipt after its take. Unsealed, its lanes hold
/// the source digests (complete: the batch landed after them in stream order) and nothing else is
/// enqueued on them. Sealed, the copy stream may still write its lanes (the landed digests) and
/// its twin (their D2H) until `scratch.event` is observed; `observed` records that observation.
/// Dropped sealed and unobserved (the engine's drop), it LEAKS its lanes and twin, the `Entry`
/// rule: a leak, never a free.
struct SpanReceipt {
    id: u64,
    scratch: Option<ReceiptScratch>,
    /// Each span's landed byte count, in attach order (the seal checks the staging against it).
    lens: Vec<u64>,
    sealed: bool,
    observed: bool,
}
impl Drop for SpanReceipt {
    fn drop(&mut self) {
        if self.sealed && !self.observed {
            std::mem::forget(self.scratch.take());
        }
    }
}
/// WP-A day 40 (`DAY40.md` design S): one landed H2D span with its device receipt: the four-lane
/// digest of its DESTINATION (the device f32 plane, after the copy; one launch per 64 spans since
/// design S2). The caller compares it with the source digest kept at the demote. `None` on an
/// engine without the receipt kernels.
pub struct LandedH2dSpan {
    pub span: H2dSpan,
    pub destination_digest: Option<Digest>,
}
/// One typed f32 span of an H2D promote batch (WP-A day 32, memra#536 Move 2 owed item 1, the H2D
/// half; `memra_tier::conformance::h2d_span_batch`): an OWNED, fully written cached pinned source
/// (a staging buffer the hash helper filled from the resident image's recurrent plane) copied into
/// the whole of an OWNED fresh device destination of exactly its byte length, allocated on the
/// owner stream. Attached to a live H2D ticket by `CudaTransfers::submit_h2d_spans` and handed back,
/// staging and landed destination, by `take_h2d_spans`, only behind the owner stream's wait on the
/// span's completion event.
pub struct H2dSpan {
    pub source: PinnedHostBuf,
    pub destination: CudaSlice<f32>,
}
impl std::fmt::Debug for H2dSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H2dSpan")
            .field("source_bytes", &self.source.len())
            .field("destination_len", &self.destination.len())
            .finish()
    }
}
/// An H2D batch's spans in attach order, each with the completion event recorded after its copy
/// on the copy stream (`None` when the enqueue or the record failed: the batch is quarantined);
/// `fenced` once the owner stream waits on every span event (`install_consumer_wait`).
struct H2dSpanBatch {
    slots: Vec<(H2dSpan, Option<CudaEvent>)>,
    landed: bool,
    fenced: bool,
    /// WP-A day 40 (design S): the spans' destination digests, 32 bytes per span, sealed by
    /// `receipt.event` after them; the batch lands only with it.
    receipt: Option<ReceiptScratch>,
    /// WP-A day 64 (`DAY64.md` section 4 step 1, log only): four timing events on the copy stream
    /// (the span work's start, after the fill, after the last span copy, after the digests and the
    /// lanes' D2H), read by `h2d_span_timing` once complete and never waited on.
    timing: Vec<CudaEvent>,
}
/// WP-A day 42 (`DAY42.md` design S2): the by-value argument of `span_receipt_digests`
/// (`cu/tier_receipt.cu` `SpanItems`): up to `SPAN_ITEMS` spans per launch, their device
/// addresses, byte lengths, and the device address of each span's four u64 lanes.
const SPAN_ITEMS: usize = 64;
#[repr(C)]
#[derive(Clone, Copy)]
struct SpanItems {
    n: u64,
    ptr: [u64; SPAN_ITEMS],
    len: [u64; SPAN_ITEMS],
    out: [u64; SPAN_ITEMS],
}
// SAFETY: plain `#[repr(C)]` integers, laid out exactly as the kernel's `struct SpanItems`
// (`u64 n; u64 ptr[64]; u64 len[64]; u64 out[64]`), passed by value.
unsafe impl DeviceRepr for SpanItems {}
/// WP-A day 46 (`DAY46.md` design S3): where a span digest launch reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpanMemory {
    /// Device planes (the D2H sources, the H2D destinations).
    Device,
    /// Pinned host staging through its device address (the D2H landed bytes): PCIe-bound.
    PinnedHost,
}
/// WP-A day 46 (`DAY46.md` design S3, DAY42 section 3): the x-blocks per span of one
/// `span_receipt_digests` launch over `spans` spans (the y dimension), so the launch holds at most
/// one 256-thread block per SM (when `spans <= sms`) and every SM keeps room for the owner's
/// blocks. S2 ran up to 192 x 64 blocks; during its landed launches the owner stream ran no kernel.
/// Pinned host reads take one block per span (outstanding loads, not SMs, bound them); device reads
/// take `sms / spans`, at least one, and never more than the largest span needs.
fn span_blocks(sms: u64, spans: u64, longest: u64, memory: SpanMemory) -> u32 {
    let need = longest.div_ceil(8).div_ceil(2048).max(1);
    let per_span = match memory {
        SpanMemory::PinnedHost => 1,
        SpanMemory::Device => (sms / spans.max(1)).max(1),
    };
    per_span.min(need).min(2048) as u32
}
/// WP-A day 40 (design S): the four little-endian u64 lanes at byte offset `o` of a receipt's lanes.
fn lanes_at(lanes: &[u8], o: usize) -> [u64; 4] {
    let mut l = [0u64; 4];
    for (k, lane) in l.iter_mut().enumerate() {
        let at = o + 8 * k;
        *lane = u64::from_le_bytes(lanes[at..at + 8].try_into().unwrap());
    }
    l
}
/// WP-A day 33 (design F): the fill of one filled H2D span batch, run by the copy stream's host
/// function: each resident plane (an owned `Arc`) and its span's staging target (a raw pointer
/// and a byte length, checked equal at the attach). WP-A day 39 (`DAY39.md` design T): the fill
/// runs on `threads` host threads at most (the engine's `fill_threads`).
struct SpanFillTask {
    items: Vec<(Arc<Vec<f32>>, FillTarget)>,
    threads: usize,
}
/// A staging buffer's start and byte length (`PinnedHostBuf::fill_target`).
type FillTarget = (*mut u8, usize);
// SAFETY: the task moves once to the driver's callback thread; each target is written by that
// thread and the fill threads it spawns and joins (`SpanFillTask::run`, disjoint byte ranges),
// while the engine owns the buffer and nobody else touches it (`attach_h2d_spans`).
unsafe impl Send for SpanFillTask {}
/// WP-A day 39 (design T): the most threads a promote's staging fill runs on, on any host.
pub(crate) const FILL_THREADS_MAX: usize = 12;
/// WP-A day 39 (design T): the least bytes a fill thread is given; a fill under twice this runs
/// on one thread.
pub(crate) const FILL_SHARE_FLOOR: usize = 4 << 20;
/// WP-A day 39 (design T): the fill threads of this host, `min(12, max(1, cpus / 2))`, fixed when
/// the engine is built (`DAY39.md` section 5: at or below the physical cores of an SMT host).
pub(crate) fn fill_threads_for_host(cpus: usize) -> usize {
    (cpus / 2).clamp(1, FILL_THREADS_MAX)
}
/// WP-A day 39 (design T): the threads one fill of `bytes` runs on, `min(host, max(1, bytes / 4
/// MiB))`.
pub(crate) fn fill_threads_for_bytes(host: usize, bytes: usize) -> usize {
    host.min((bytes / FILL_SHARE_FLOOR).max(1)).max(1)
}
/// WP-A day 39 (design T): a fill's byte list (`lens`, in attach order) cut into `t` contiguous
/// shares of `ceil(total / t)` bytes (the last one shorter), each `(item, byte offset, bytes)`; a
/// cut that falls inside an item splits it across two shares. Every byte is in exactly one share,
/// in order. The day-39 survey probe's `shares`, verbatim in its arithmetic.
pub(crate) fn fill_shares(lens: &[usize], t: usize) -> Vec<Vec<(usize, usize, usize)>> {
    let t = t.max(1);
    let total: usize = lens.iter().sum();
    let per = total.div_ceil(t).max(1);
    let mut out = vec![Vec::new(); t];
    let (mut k, mut used) = (0usize, 0usize);
    for (i, &n) in lens.iter().enumerate() {
        let mut off = 0;
        while off < n {
            let take = (n - off).min(per - used);
            out[k].push((i, off, take));
            off += take;
            used += take;
            if used == per && k + 1 < t {
                k += 1;
                used = 0;
            }
        }
    }
    out
}
/// One fill copy of a share: `n` bytes from `src` to `dst`.
struct FillCopy {
    src: *const u8,
    dst: *mut u8,
    n: usize,
}
// SAFETY: a share's copies are run by exactly one thread (a fill thread, or the callback thread
// when the host gives no thread); every source is a plane the task's `Arc` keeps alive and nobody
// writes, every destination range is written by that thread only (the shares are disjoint), and
// every fill thread is joined before the host function returns.
unsafe impl Send for FillCopy {}
unsafe impl Sync for FillCopy {}
fn run_fill_copies(copies: &[FillCopy]) {
    for c in copies {
        // SAFETY: see `FillCopy`; `src` holds `n` initialized bytes, `dst` has room for `n`, and a
        // heap `Vec` never overlaps a pinned host allocation.
        unsafe { std::ptr::copy_nonoverlapping(c.src, c.dst, c.n) };
    }
}
impl SpanFillTask {
    fn run(&self) {
        let lens: Vec<usize> = self
            .items
            .iter()
            .map(|(plane, (_, len))| (*len).min(plane.len() * 4))
            .collect();
        let t = fill_threads_for_bytes(self.threads, lens.iter().sum());
        let shares: Vec<Vec<FillCopy>> = fill_shares(&lens, t)
            .into_iter()
            .map(|share| {
                share
                    .into_iter()
                    .map(|(i, off, n)| {
                        let (plane, (dst, _)) = &self.items[i];
                        // SAFETY: `off + n <= lens[i]`, within both the plane's initialized bytes
                        // and the staging buffer (`attach_h2d_spans`).
                        unsafe {
                            FillCopy {
                                src: plane.as_ptr().cast::<u8>().add(off),
                                dst: dst.add(off),
                                n,
                            }
                        }
                    })
                    .collect()
            })
            .collect();
        if shares.len() == 1 {
            run_fill_copies(&shares[0]);
            return;
        }
        std::thread::scope(|s| {
            let mut here = vec![0usize];
            for (k, share) in shares.iter().enumerate().skip(1) {
                // A thread the host will not give leaves its share to this thread: every byte is
                // written either way, before the host function returns.
                let spawned = std::thread::Builder::new()
                    .name("memra-fill".into())
                    .spawn_scoped(s, move || run_fill_copies(share));
                if spawned.is_err() {
                    here.push(k);
                }
            }
            for k in here {
                run_fill_copies(&shares[k]);
            }
        });
    }
}
/// The host function (`cuLaunchHostFunc`) of a filled span batch: takes its task back, runs the
/// fill, drops the task (the plane `Arc`s) on this thread. Never unwinds across the FFI boundary.
unsafe extern "C" fn span_fill_on_copy_stream(arg: *mut std::ffi::c_void) {
    // SAFETY: `arg` is the `Box<SpanFillTask>` `attach_h2d_spans` leaked for exactly this call.
    let task = unsafe { Box::from_raw(arg.cast::<SpanFillTask>()) };
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| task.run()));
}
/// The receipt kernels of `cu/tier_receipt.cu`, loaded once per `CudaTransfers` with a copy stream.
struct ReceiptKernels {
    _module: Arc<CudaModule>,
    digest: CudaFunction,
    delay: CudaFunction,
    /// WP-A day 38 (design G): the framed SHA-256 of a D2H batch's device sources, and the
    /// `d2h-source-flip` fault's one-byte flip.
    sha256: CudaFunction,
    flip: CudaFunction,
    /// WP-A day 42 (`DAY42.md` design S2): the batched four-lane digest of the f32 spans.
    spans: CudaFunction,
}
/// One D2D batch's receipt lanes (WP-A day 22, `memra_tier::conformance::d2d_receipt_witnessed`):
/// per item, 64 bytes on the device (four u64 lanes of the SOURCE digest, taken on the copy stream
/// behind the producer fence, then four of the DESTINATION digest, taken after the copy), copied
/// once per batch into `pinned` on the copy stream and sealed by `event`, recorded after that copy.
/// `progress` reads the lanes only after `event` is observed complete, so an item lands with its
/// receipt or not at all.
struct ReceiptScratch {
    lanes: CudaSlice<u8>,
    pinned: PinnedBacking,
    /// The lanes' zero-fill, recorded on the OWNER stream at allocation; every writer of the lanes
    /// is ordered behind it explicitly (the copy stream waits on it before its first digest; the
    /// fault's owner-stream reader follows it in stream order). Nothing here relies on cudarc's
    /// implicit event tracking: the engine's context disables it (`Engine::new`).
    zeroed: CudaEvent,
    event: Option<CudaEvent>,
}
/// The engine's receipt of a landed D2D batch (`CudaTransfers::d2d_receipt`): per item the source
/// digest (the item's expectation) and the destination digest (the item's checksum), and the
/// host-contract gate's verdict over them (`Completion::require`, `device = true`).
pub struct D2dReceipt {
    pub bytes: u64,
    pub items: Vec<ReceiptTerm>,
    pub verdict: Result<()>,
}
/// One same-device copy of the capture class (WP-A day 20, memra#536 Move 2 slice 1): `bytes`
/// from the head of a BORROWED live source (a session cache's plane the caller keeps) into an
/// OWNED registered destination lease, behind `producer_fence`. See
/// `CudaTransfers::submit_d2d_capture`.
pub struct D2dCapture<'a> {
    pub source: &'a CudaSlice<u8>,
    pub destination: DeviceLease,
    pub bytes: u64,
    pub producer_fence: FenceId,
}
/// One same-device copy of the restore class (WP-A day 21, memra#536 Move 2 slice 2): `bytes`
/// from the head of a BORROWED published source (a device prefix entry's plane, held by the
/// device LRU's pin from submit through acknowledge) into a BORROWED destination view (the
/// admitted request's fresh session cache plane, exactly `bytes` long), behind `producer_fence`.
/// Nothing on either side is the registry's. See `CudaTransfers::submit_d2d_restore`.
pub struct D2dRestore<'a> {
    pub source: &'a CudaSlice<u8>,
    pub destination: CudaViewMut<'a, u8>,
    pub bytes: u64,
    pub producer_fence: FenceId,
}
/// Opaque graph-use retention. Drop only after graph execution/destruction retires.
pub struct GraphPin {
    _pins: Vec<Rc<()>>,
}

pub struct CudaTransfers {
    stream: Arc<CudaStream>,
    /// WP-A day 17 (memra#536, Move 1 first slice): the COPY stream. `None` keeps every copy on
    /// the owner stream (the day-16 program). `Some`: a copy of either direction is issued on this
    /// stream behind the producer fence's event (`copy.wait(producer)`) and its completion event
    /// is recorded here. A D2H installs NO owner-stream wait at submit (its destination's consumer
    /// is the host, whose wait is `event_done` in `progress`). An H2D on the copy stream installs
    /// its owner-stream wait at the SETTLE (day 19, rule 3 of the tier crate's conformance,
    /// `install_consumer_wait`): until then the item is not `consumer_fenced`, so it is not
    /// publishable; day 18 installed that wait at submit, which queued every kernel the tenant
    /// issued after the submit behind the copy's landing. Created from the owner context on the
    /// owner thread; `check_thread` still pins every call.
    copy: Option<Arc<CudaStream>>,
    /// WP-A day 38 (`DAY38.md` design G'', section 8): the receipt twins (the pinned host side of a
    /// batch's receipt lanes), reused by exact byte length. A twin returns here when its batch is
    /// acknowledged (every write to it observed) instead of being freed: `cuMemFreeHost` waits for
    /// every stream's queued work in the context (DAY37's probe), so a per-batch free on the owner
    /// thread would hold it behind any long copy-stream or receipt-stream work. Freed only when the
    /// engine drops (the latch or shutdown).
    twin_pool: RefCell<Vec<PinnedBacking>>,
    /// WP-A day 63 (`DAY63.md` design L1): the lease backing pool (closed at the latch and at drop).
    lease_pool: Rc<LeasePool>,
    /// WP-A day 39 (`DAY39.md` design T): the host threads a filled promote's staging fill runs
    /// on at most (`fill_threads_for_host` of the host's available parallelism, read once here);
    /// each fill takes `fill_threads_for_bytes` of it.
    fill_threads: usize,
    governor: SharedBudget,
    /// The pinned destination arm `alloc_host` takes on this device (`PinnedKind::for_device` of
    /// the owner context's device name, resolved once at construction).
    pinned_default: PinnedKind,
    owner: DeviceOwner,
    thread: std::thread::ThreadId,
    device: u32,
    sequence: u64,
    fence_sequence: u64,
    producers: HashMap<u64, (FenceId, CudaEvent)>,
    allocations: HashMap<u64, ChargedLease>,
    entries: HashMap<TransferTicket, Entry>,
    /// WP-A day 22: the D2D receipt kernels; `Some` exactly when `copy` is.
    receipt: Option<ReceiptKernels>,
    /// The one-shot `d2d-delay` fault (`inject_d2d_early_reader`): the next D2D submit of either
    /// class delays its copy by this many nanoseconds and takes its destination digest from an
    /// unordered early reader on the owner stream. `None` in production.
    early_reader: Option<u64>,
    /// WP-A day 38: the one-shot `d2h-source-flip` fault (`inject_d2h_source_flip`): the next
    /// device-receipt D2H batch flips one byte of its first item's source after the digest and
    /// before the copy. `false` in production.
    source_flip: bool,
    /// WP-A day 38: the one-shot `d2h-delay` fault (`inject_d2h_delay`): the next device-receipt
    /// D2H batch lands no earlier than this many nanoseconds after its submission (a host-side
    /// hold since design G'''; no stream runs a spin). `None` in production.
    d2h_delay: Option<u64>,
    /// WP-A day 40 (design S): the one-shot `span-flip-landed` fault (`inject_span_flip_landed`): the
    /// next D2H span batch flips one byte of its first span's pinned staging on the copy stream after
    /// its copy (and, design S2, before its event, so the landing covers it). `false` in production.
    span_flip_landed: bool,
    /// WP-A day 42 (`DAY42.md` design S2): the one live D2H span receipt (taken, sealed or not, not
    /// yet read), and the sealed receipts a caller abandoned or a later take displaced, freed only
    /// once their events are observed (`reap_span_receipts`); `span_receipt_seq` numbers them.
    span_receipt: Option<SpanReceipt>,
    span_reap: Vec<SpanReceipt>,
    span_receipt_seq: u64,
    /// WP-A day 46 (`DAY46.md` design S3): the card's SM count (`span_blocks`), read once.
    sm_count: u64,
    /// Test only (the native `d2h_span` and day-32 `h2d_span` cells): the next span batch's second
    /// enqueue fails, whichever direction it is.
    #[cfg(test)]
    span_enqueue_fault: bool,
}
/// WP-A day 63 (L1.6): the lease pool closes with the engine (idle backings free, pool charges
/// release); leases still held elsewhere free at their own drop.
impl Drop for CudaTransfers {
    fn drop(&mut self) {
        self.lease_pool.close();
    }
}
impl CudaTransfers {
    /// Construct on the designated CUDA owner thread with its existing stream.
    pub fn new(owner: Arc<CudaStream>, governor: SharedBudget) -> Result<Self> {
        let device = u32::try_from(owner.context().ordinal()).map_err(|_| Error::Overflow)?;
        if device as usize >= governor.borrow().used().device.len() {
            return Err(Error::WrongOwner);
        }
        cuda(owner.context().bind_to_thread())?;
        let pinned_default = PinnedKind::for_device(&owner.context().name().unwrap_or_default());
        // WP-A day 46 (design S3): the SM count bounds every span digest launch (`span_blocks`).
        let sm_count = cuda(
            owner
                .context()
                .attribute(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        )?
        .max(1) as u64;
        Ok(Self {
            stream: owner,
            copy: None,
            twin_pool: RefCell::new(Vec::new()),
            lease_pool: Rc::new(LeasePool::new(governor.clone())),
            fill_threads: fill_threads_for_host(
                std::thread::available_parallelism().map_or(1, |n| n.get()),
            ),
            governor,
            pinned_default,
            owner: DeviceOwner::new(device),
            thread: std::thread::current().id(),
            device,
            sequence: 0,
            fence_sequence: 0,
            producers: HashMap::new(),
            allocations: HashMap::new(),
            entries: HashMap::new(),
            receipt: None,
            early_reader: None,
            source_flip: false,
            d2h_delay: None,
            span_flip_landed: false,
            span_receipt: None,
            span_reap: Vec::new(),
            span_receipt_seq: 0,
            sm_count,
            #[cfg(test)]
            span_enqueue_fault: false,
        })
    }
    /// `new`, plus a second stream of the same context for the copies (WP-A day 17 for the D2H,
    /// day 18 for the H2D; memra#536 Move 1). The stream is created on the owner thread; a failure
    /// to create it is a construction refusal, never a silent fall back to the owner stream.
    pub fn new_with_copy_stream(owner: Arc<CudaStream>, governor: SharedBudget) -> Result<Self> {
        let mut t = Self::new(owner, governor)?;
        t.copy = Some(cuda(t.stream.context().new_stream())?);
        // WP-A day 22: the D2D classes carry a receipt, so the copy stream comes with its kernels;
        // a module that will not load is a construction refusal, never a receipt-less class.
        let module = cuda(
            t.stream
                .context()
                .load_module(Ptx::from_binary(TIER_RECEIPT_FATBIN.to_vec())),
        )?;
        let digest = cuda(module.load_function("d2d_receipt_digest"))?;
        let delay = cuda(module.load_function("tier_delay_spin"))?;
        let sha256 = cuda(module.load_function("d2h_receipt_sha256"))?;
        let flip = cuda(module.load_function("tier_flip_byte"))?;
        let spans = cuda(module.load_function("span_receipt_digests"))?;
        t.receipt = Some(ReceiptKernels {
            _module: module,
            digest,
            delay,
            sha256,
            flip,
            spans,
        });
        Ok(t)
    }
    /// WP-A day 38 (`DAY38.md` design G): whether this engine takes a D2H batch's receipt on the
    /// copy stream over the device sources (the copy stream with the receipt kernels; designs G'
    /// to G''' ran it on a receipt stream, G4 puts every piece of side work on the copy stream),
    /// rather than on the owner thread over the landed bytes.
    pub fn d2h_receipts_on_device(&self) -> bool {
        self.copy.is_some() && self.receipt.is_some()
    }
    /// The `MEMRA_KV_HOST_FAULT=d2h-source-flip` fault (WP-A day 38, the device receipt's red arm):
    /// the NEXT device-receipt D2H batch flips one byte of its first accepted item's source on the
    /// copy stream after the digest and before the batch's copies (stream order, design G4), so the
    /// landed bytes differ from the receipt and the caller's witness must refuse the image.
    /// One-shot; diagnostics only.
    pub fn inject_d2h_source_flip(&mut self) {
        self.source_flip = true;
    }
    /// The `MEMRA_KV_HOST_FAULT=d2h-delay` fault (WP-A day 38, the copy-phase park's red arm): the
    /// NEXT device-receipt D2H batch is held unlanded for `delay_ns` after its submission, on the
    /// host (designs G''' and G4, DAY38 sections 14 and 17): `progress` lands none of its items
    /// before the hold ends and `synchronize` returns only after it, so its copy phase lasts at
    /// least that long for the owner while the copy stream runs free (a spin on the one side stream
    /// would hold every later copy of every class, G's plain-arm failure). One-shot.
    pub fn inject_d2h_delay(&mut self, delay_ns: u64) {
        self.d2h_delay = Some(delay_ns);
    }
    /// The `MEMRA_KV_HOST_FAULT=span-flip-landed` fault (WP-A day 40, design S's D2H red arm): the
    /// NEXT D2H span batch flips one byte of its first span's landed staging on the copy stream
    /// after that span's copy and before that span's event (design S2, `DAY42.md` section 1a: the
    /// landing covers the flip), so the span's landed digest (the seal's) differs from its source
    /// digest and the caller must refuse the image. One-shot; diagnostics only.
    pub fn inject_span_flip_landed(&mut self) {
        self.span_flip_landed = true;
    }
    /// Test only (DAY38 design G''): the receipt twins waiting in the pool.
    #[cfg(test)]
    fn twin_pool_len(&self) -> usize {
        self.twin_pool.borrow().len()
    }
    /// WP-A day 38 (log only): the device receipt's digest kernels' receipt-stream time for `ticket`,
    /// read only if the end event already completed (never a host wait); `None` for a batch
    /// without a device receipt, while the kernels have not finished, or while the `d2h-delay`
    /// hold is on (the batch has not landed for the owner).
    pub fn d2h_receipt_gpu_ms(&self, ticket: &TransferTicket) -> Option<f32> {
        let r = self.entries.get(ticket)?.d2h_receipt.as_ref()?;
        if r.hold_until.is_some_and(|h| std::time::Instant::now() < h) {
            return None;
        }
        let (start, end) = r.timing.as_ref()?;
        if !event_done(end).ok()? {
            return None;
        }
        start.elapsed_ms(end).ok()
    }
    /// The `MEMRA_KV_HOST_FAULT=d2d-delay-*` fault (WP-A day 22, the red arm of the receipt): the
    /// NEXT D2D submit of either class runs `tier_delay_spin(delay_ns)` ONCE on the copy stream,
    /// between the first item's producer wait and its copy (the stream is serial, so every item's
    /// copy waits behind it), and takes each item's DESTINATION digest from an unordered early
    /// reader (the owner stream at submit, after the producer fence, with no wait on the copy's
    /// event: the read a publication or a first prime chunk issued before the completion event
    /// would make). The source digest stays on the copy stream behind the producer fence, so the
    /// receipt must differ and the settle must refuse it. One-shot; diagnostics only.
    pub fn inject_d2d_early_reader(&mut self, delay_ns: u64) {
        self.early_reader = Some(delay_ns);
    }
    /// One batch's receipt scratch: zeroed device lanes (64 bytes per item) and a zeroed pinned
    /// twin of the same size. Allocated before the batch is charged, so a refusal here submits
    /// nothing.
    fn receipt_scratch(&self, items: usize) -> Result<ReceiptScratch> {
        self.receipt_scratch_bytes(items.checked_mul(64).ok_or(Error::Overflow)?)
    }
    /// `receipt_scratch` for `bytes` of lanes (a D2H device receipt takes 32 bytes per item).
    fn receipt_scratch_bytes(&self, bytes: usize) -> Result<ReceiptScratch> {
        // Zeroed on the OWNER stream and fenced by `zeroed`: the copy stream waits on it before
        // its first digest, and the fault's early reader (owner stream) follows it in order.
        let lanes = cuda(self.stream.alloc_zeros::<u8>(bytes))?;
        let zeroed = cuda(self.stream.record_event(None))?;
        // WP-A day 38 (design G''): a pooled twin of exactly this length first, else a fresh one.
        let pooled = {
            let mut pool = self.twin_pool.borrow_mut();
            pool.iter()
                .position(|b| b.len == bytes)
                .map(|i| pool.swap_remove(i))
        };
        let mut pinned = match pooled {
            Some(twin) => twin,
            // SAFETY: every byte is zero-filled through `as_mut_slice` before the backing is used.
            None => cuda(unsafe {
                PinnedBacking::alloc(self.stream.context(), bytes, PinnedKind::Cached)
            })?,
        };
        cuda(pinned.as_mut_slice())?.fill(0);
        Ok(ReceiptScratch {
            lanes,
            pinned,
            zeroed,
            event: None,
        })
    }
    /// `d2d_receipt_digest` over `span` into `lanes` (the item's 32-byte lane block) on `stream`.
    fn digest_on(
        &self,
        stream: &Arc<CudaStream>,
        span: &CudaView<'_, u8>,
        lanes: &mut CudaViewMut<'_, u8>,
    ) -> Result<()> {
        let k = self.receipt.as_ref().ok_or(Error::Unsupported)?;
        let n = span.len() as u64;
        let blocks = n.div_ceil(8).div_ceil(2048).clamp(1, 2048) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&k.digest);
        b.arg(span).arg(&n).arg(lanes);
        // SAFETY: documented FFI of `cu/tier_receipt.cu`: `d2d_receipt_digest(const u8* p, u64 n,
        // u64* out)` reads exactly `n` bytes of `span` (its length) and adds into the four u64
        // lanes of `lanes` (32 bytes, zeroed at allocation); argument order and types match.
        cuda(unsafe { b.launch(cfg) }).map(|_| ())
    }
    /// WP-A day 42 (`DAY42.md` design S2): `span_receipt_digests` over `spans` (each a device
    /// address and a byte length: a span's device plane, or its pinned staging through its device
    /// address) on `stream`, ONE launch per `SPAN_ITEMS` spans; span `k` adds into the four u64
    /// lanes at device address `lanes + offset + stride * k` (zeroed at allocation, fenced by the
    /// caller's wait on the zero-fill). Each span's value is `d2d_receipt_digest`'s. WP-A day 46
    /// (`DAY46.md` design S3): the grid is bounded, `span_blocks`, so a launch never fills the card.
    fn span_digests_on(
        &self,
        stream: &Arc<CudaStream>,
        spans: &[(u64, u64)],
        lanes: u64,
        offset: u64,
        stride: u64,
        memory: SpanMemory,
    ) -> Result<()> {
        let k = self.receipt.as_ref().ok_or(Error::Unsupported)?;
        for (c, chunk) in spans.chunks(SPAN_ITEMS).enumerate() {
            let mut items = SpanItems {
                n: chunk.len() as u64,
                ptr: [0; SPAN_ITEMS],
                len: [0; SPAN_ITEMS],
                out: [0; SPAN_ITEMS],
            };
            let mut longest = 0u64;
            for (i, &(ptr, n)) in chunk.iter().enumerate() {
                let at = (c * SPAN_ITEMS + i) as u64;
                items.ptr[i] = ptr;
                items.len[i] = n;
                items.out[i] = lanes + offset + stride * at;
                longest = longest.max(n);
            }
            let blocks = span_blocks(self.sm_count, chunk.len() as u64, longest, memory);
            let cfg = LaunchConfig {
                grid_dim: (blocks, chunk.len() as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = stream.launch_builder(&k.spans);
            b.arg(&items);
            // SAFETY: documented FFI of `cu/tier_receipt.cu`: `span_receipt_digests(SpanItems
            // items)` reads exactly `len[k]` bytes at `ptr[k]` (a live span buffer the engine owns
            // or the caller keeps alive until an event after this launch is observed, ordered on
            // `stream` by the caller) and adds into the four u64 lanes at `out[k]`, inside the
            // caller's zeroed lanes; `items` is passed by value, laid out as the kernel's struct.
            cuda(unsafe { b.launch(cfg) })?;
        }
        Ok(())
    }
    /// WP-A day 42 (design S2): the `span-flip-landed` fault's flip, one byte of `staging` (a
    /// span's pinned destination through its device address) on `stream`.
    fn flip_staging_on(&self, stream: &Arc<CudaStream>, staging: &PinnedHostBuf) -> Result<()> {
        let k = self.receipt.as_ref().ok_or(Error::Unsupported)?;
        let n = staging.len() as u64;
        let at = staging.device_address().map_err(|_| Error::Quarantined)? + (n.max(1) - 1).min(5);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&k.flip);
        b.arg(&at);
        // SAFETY: documented FFI of `cu/tier_receipt.cu`: `tier_flip_byte(u8* p)` XORs the one byte
        // at `p`, inside span 0's pinned staging (the engine owns it; the copy before this launch on
        // `stream` writes it, and the span's event after it covers the flip).
        cuda(unsafe { b.launch(cfg) }).map(|_| ())
    }
    /// `tier_delay_spin(ns)` on `stream` (the fault's delay).
    fn delay_on(&self, stream: &Arc<CudaStream>, ns: u64) -> Result<()> {
        let k = self.receipt.as_ref().ok_or(Error::Unsupported)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&k.delay);
        b.arg(&ns);
        // SAFETY: documented FFI of `cu/tier_receipt.cu`: `tier_delay_spin(u64 ns)` takes one
        // scalar and touches no memory.
        cuda(unsafe { b.launch(cfg) }).map(|_| ())
    }
    /// WP-A day 38 (`DAY38.md` design G4, section 17; `memra_tier::conformance::d2h_device_receipt`):
    /// take a D2H batch's receipt on the COPY stream, ahead of the item copies the caller issues on
    /// the same stream after this returns (G's order; designs G' to G''' ran it on a receipt stream
    /// beside the copies, and a second side stream running kernels moved every later owner kernel
    /// boundary, sections 13 to 16). The copy stream waits on every accepted D2H item's producer
    /// event and on the lanes' zero-fill, `d2h_receipt_sha256` digests each accepted D2H item's
    /// device source (the program `checksum`, byte for byte; any other item hashes nothing and its
    /// lanes are never read), one D2H of the digests goes into the pinned twin, and the receipt event
    /// seals it. The digest only reads the sources, the ticket's registered device leases,
    /// untouched by any writer until the ticket retires its sources. The flip fault rides here,
    /// one-shot: one byte of the first accepted source flips after the digest, so the copies after
    /// it in stream order land the flipped byte. The `d2h-delay` fault is a host-side hold of the
    /// landing (`hold_until`), no spin.
    fn seal_d2h_device_receipt(
        &mut self,
        ops: &[TransferOp<CudaPinnedLease>],
        errors: &[Option<Error>],
        scratch: ReceiptScratch,
    ) -> Result<D2hDeviceReceipt> {
        // Move-then-match: from the first enqueue on, the copy stream may write the lanes and the
        // twin, so on ANY error the scratch leaks (never a free under a pending write); only the
        // success exit hands it out.
        let mut scratch = std::mem::ManuallyDrop::new(scratch);
        // Designs G''' and G4: the `d2h-delay` fault is spent by this batch and held on the host.
        let hold_until = self
            .d2h_delay
            .take()
            .map(|ns| std::time::Instant::now() + std::time::Duration::from_nanos(ns));
        let sealed = self.seal_d2h_device_receipt_into(ops, errors, &mut scratch);
        sealed.map(|timing| D2hDeviceReceipt {
            scratch: std::mem::ManuallyDrop::into_inner(scratch),
            timing: Some(timing),
            hold_until,
        })
    }
    /// The enqueues of `seal_d2h_device_receipt`, into the caller's scratch.
    fn seal_d2h_device_receipt_into(
        &mut self,
        ops: &[TransferOp<CudaPinnedLease>],
        errors: &[Option<Error>],
        scratch: &mut ReceiptScratch,
    ) -> Result<(CudaEvent, CudaEvent)> {
        let copy = self.copy.clone().ok_or(Error::Unsupported)?;
        let mut sources: Vec<(u64, u64)> = Vec::with_capacity(ops.len());
        let mut waited: Vec<u64> = Vec::new();
        for (op, error) in ops.iter().zip(errors) {
            match (op, error) {
                (TransferOp::D2h(o), None) => {
                    if let Some(f) = o.producer_fence
                        && !waited.contains(&f.sequence)
                    {
                        cuda(copy.wait(&self.producers[&f.sequence].1))?;
                        waited.push(f.sequence);
                    }
                    let backing = self.owner.resolve::<Rc<RefCell<KvPlane>>>(&o.device)?;
                    let plane = backing.borrow();
                    let view = plane.slice(..o.bytes as usize);
                    let (ptr, _record) = view.device_ptr(&copy);
                    sources.push((ptr, o.bytes));
                }
                _ => sources.push((0, 0)),
            }
        }
        cuda(copy.wait(&scratch.zeroed))?;
        let k = self.receipt.as_ref().ok_or(Error::Unsupported)?;
        // The two bracket events carry timing (the log-only kernel reading); every other event of
        // the engine keeps cudarc's timing-disabled default.
        let timed = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let start = cuda(copy.record_event(timed))?;
        for (chunk, part) in sources.chunks(RECEIPT_ITEMS).enumerate() {
            let mut items = ReceiptItems {
                n: part.len() as u64,
                ptr: [0; RECEIPT_ITEMS],
                len: [0; RECEIPT_ITEMS],
            };
            for (j, &(ptr, len)) in part.iter().enumerate() {
                items.ptr[j] = ptr;
                items.len[j] = len;
            }
            let at = chunk * RECEIPT_ITEMS * 32;
            let mut out = scratch.lanes.slice_mut(at..at + part.len() * 32);
            let cfg = LaunchConfig {
                grid_dim: (part.len().div_ceil(32) as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = copy.launch_builder(&k.sha256);
            b.arg(&items).arg(&mut out);
            // SAFETY: documented FFI of `cu/tier_receipt.cu`: `d2h_receipt_sha256(ReceiptItems items,
            // u8* out)` reads `items.len[j]` bytes at each non-zero `items.ptr[j]` (live device
            // sources of this batch, ordered behind their producer fences above) and writes exactly
            // `32 * items.n` bytes of `out` (this chunk's slice of the lanes).
            cuda(unsafe { b.launch(cfg) })?;
        }
        let end = cuda(copy.record_event(timed))?;
        if std::mem::take(&mut self.source_flip)
            && let Some(&(ptr, len)) = sources.iter().find(|(p, n)| *p != 0 && *n > 0)
        {
            let at = ptr + (len - 1).min(5);
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = copy.launch_builder(&k.flip);
            b.arg(&at);
            // SAFETY: documented FFI of `cu/tier_receipt.cu`: `tier_flip_byte(u8* p)` XORs the one
            // byte at `p`, inside the first accepted item's live source (offset below its length).
            cuda(unsafe { b.launch(cfg) })?;
        }
        cuda(copy.memcpy_dtoh(&scratch.lanes, &mut scratch.pinned))?;
        scratch.event = Some(cuda(copy.record_event(None))?);
        Ok((start, end))
    }
    /// Seal a batch's receipt: one D2H of the lanes into the pinned twin on the copy stream, then
    /// the receipt event; `progress` reads the lanes only after that event. A failure quarantines
    /// the batch (its items may be in flight).
    fn seal_receipt(copy: &Arc<CudaStream>, entry: &mut Entry, mut scratch: ReceiptScratch) {
        let sealed = (|| -> Result<()> {
            cuda(copy.memcpy_dtoh(&scratch.lanes, &mut scratch.pinned))?;
            scratch.event = Some(cuda(copy.record_event(None))?);
            Ok(())
        })();
        if sealed.is_err() {
            entry.unknown = true;
        }
        entry.receipt = Some(scratch);
    }
    /// The receipt of a landed D2D batch (WP-A day 22, `d2d_receipt_witnessed` rules 2 to 4):
    /// `NotReady` before the landing; after it, per item the source digest (taken behind the
    /// producer fence) and the destination digest (taken after the copy), and the host-contract
    /// gate's verdict over them: `Ok` when every destination witnesses its source, `Corrupt` on a
    /// receipt-less or mismatching item (for the restore class, read after `install_consumer_wait`,
    /// since the gate also requires the fence). A ticket that is not a same-device batch is
    /// `Unsupported`. The receipt reads bytes and changes none.
    pub fn d2d_receipt(&mut self, ticket: &TransferTicket) -> Result<D2dReceipt> {
        self.progress(ticket)?;
        let e = &self.entries[ticket];
        if e.items
            .iter()
            .flatten()
            .any(|i| i.direction != CopyDirection::DeviceToDevice)
        {
            return Err(Error::Unsupported);
        }
        if !e.completion.producer_done {
            return Err(Error::NotReady);
        }
        let mut items = Vec::with_capacity(e.items.len());
        let mut bytes = 0u64;
        for (i, item) in e.items.iter().enumerate() {
            let Some(item) = item else {
                continue;
            };
            bytes = bytes.saturating_add(item.bytes);
            items.push(ReceiptTerm {
                source: e.expected[i][0].checksum,
                destination: e.completion.items[i].segments[0].checksum,
            });
        }
        let verdict = e.completion.require(ticket, &e.expected, true);
        Ok(D2dReceipt {
            bytes,
            items,
            verdict,
        })
    }
    /// The copy stream when one exists (`new_with_copy_stream`); `None` under `new`.
    pub fn copy_stream(&self) -> Option<&Arc<CudaStream>> {
        self.copy.as_ref()
    }
    fn check_thread(&self) -> Result<()> {
        if std::thread::current().id() != self.thread {
            return Err(Error::WrongOwner);
        }
        Ok(())
    }
    pub fn used(&self) -> TierBudget {
        self.governor.borrow().used()
    }
    fn request(&self) -> BudgetRequest {
        BudgetRequest {
            bytes: TierBudget::zero(self.used().device.len()),
            priority: Priority::Demand,
            deadline: Deadline(u64::MAX),
            tenant: [0; 32],
        }
    }
    /// The pinned destination arm this device's `alloc_host` allocates under.
    pub fn pinned_default(&self) -> PinnedKind {
        self.pinned_default
    }
    /// WP-A day 63 (L1.5): the lease pool's idle cap in bytes (0, the default, pools nothing).
    pub fn set_lease_pool_cap(&self, bytes: u64) {
        self.lease_pool.cap.set(bytes);
    }
    /// WP-A day 63 (L1.6): the lease pool's handle, so the tier's latch can close it without
    /// borrowing the engine.
    pub fn lease_pool(&self) -> Rc<LeasePool> {
        self.lease_pool.clone()
    }
    /// WP-A day 63 (L1.7, log only): (taken from the pool, allocated fresh) since the engine was
    /// built, and the pool's idle bytes.
    pub fn lease_pool_counts(&self) -> (u64, u64, u64) {
        let (taken, fresh) = self.lease_pool.counts.get();
        (taken, fresh, self.lease_pool.idle_bytes.get())
    }
    /// Admission policy belongs to the caller; only the physical dimension is set here.
    /// The allocation carries the device's default flag bits (`pinned_default`: cached on a card
    /// class with a receipt, write-combined elsewhere).
    pub fn alloc_host(&mut self, bytes: usize, request: BudgetRequest) -> Result<CudaPinnedLease> {
        self.alloc_host_kind(bytes, request, self.pinned_default)
    }
    /// `alloc_host` with the `cuMemHostAlloc` flag bits chosen by the caller (WP-A day 13). The
    /// transfer gate measures the arms with it; production callers take `alloc_host`.
    pub fn alloc_host_kind(
        &mut self,
        bytes: usize,
        mut request: BudgetRequest,
        kind: PinnedKind,
    ) -> Result<CudaPinnedLease> {
        self.check_thread()?;
        if bytes == 0 || bytes > isize::MAX as usize {
            return Err(Error::InvalidLayout);
        }
        // WP-A day 63 (`DAY63.md` design L1.2, L1.3): the lease is charged its size class (the
        // pinned memory it holds), then takes an idle backing of that class and kind, or a fresh
        // one zero-filled over the whole class. A pooled backing is not refilled: every byte of it
        // was initialized by its first lease, and nothing reads past the new lease's length.
        let class = lease_class(bytes);
        request.bytes = TierBudget::zero(self.used().device.len());
        request.bytes.pinned = class as u64;
        let charge = self.governor.borrow_mut().reserve(&request)?;
        let pooled = self.lease_pool.take(class, kind);
        let (taken, fresh) = self.lease_pool.counts.get();
        self.lease_pool.counts.set(if pooled.is_some() {
            (taken + 1, fresh)
        } else {
            (taken, fresh + 1)
        });
        // Initialize through a raw pointer: constructing an uninitialized slice
        // would itself be invalid, even if immediately followed by fill().
        let allocation = (|| {
            let mut backing = match pooled {
                Some(b) => b,
                None => {
                    let mut b =
                        cuda(unsafe { PinnedBacking::alloc(self.stream.context(), class, kind) })?;
                    unsafe {
                        cuda(b.as_mut_ptr())?.write_bytes(0, class);
                    }
                    b
                }
            };
            backing.set_len(bytes);
            Ok(backing)
        })();
        match allocation {
            Ok(backing) => Ok(CudaPinnedLease {
                allocation: Rc::new(PinnedAllocation {
                    backing: Some(backing),
                    pin: Some(charge.pin()?),
                    charge: Some(charge),
                    governor: self.governor.clone(),
                    pool: Some(self.lease_pool.clone()),
                }),
            }),
            Err(e) => {
                self.governor.borrow_mut().release(&charge)?;
                Err(e)
            }
        }
    }
    pub fn alloc_device(
        &mut self,
        bytes: usize,
        generation: u64,
        request: BudgetRequest,
    ) -> Result<DeviceLease> {
        self.check_thread()?;
        if bytes == 0 || bytes > isize::MAX as usize {
            return Err(Error::InvalidLayout);
        }
        let charge = self.device_charge(bytes, request)?;
        match cuda(self.stream.alloc_zeros::<u8>(bytes)) {
            Ok(backing) => self.register_charged(backing.into(), generation, charge),
            Err(e) => {
                self.governor.borrow_mut().release(&charge)?;
                Err(e)
            }
        }
    }
    fn device_charge(&self, bytes: usize, mut request: BudgetRequest) -> Result<ChargedLease> {
        request.bytes = TierBudget::zero(self.used().device.len());
        request.bytes.device[self.device as usize] = bytes as u64;
        self.governor.borrow_mut().reserve(&request)
    }
    /// Move an existing buffer (never an unowned raw pointer) into the sealed registry.
    /// Its CUDA stream must be the designated owner stream. Do not double-charge a
    /// buffer already accounted elsewhere; callers must transfer its admission first.
    pub fn register_device(
        &mut self,
        backing: impl Into<KvPlane>,
        generation: u64,
        request: BudgetRequest,
    ) -> Result<DeviceLease> {
        self.check_thread()?;
        let backing = backing.into();
        if !Arc::ptr_eq(backing.stream(), &self.stream) || backing.is_empty() {
            return Err(Error::WrongOwner);
        }
        let charge = self.device_charge(backing.physical_bytes(), request)?;
        self.register_charged(backing, generation, charge)
    }
    fn register_charged(
        &mut self,
        backing: KvPlane,
        generation: u64,
        charge: ChargedLease,
    ) -> Result<DeviceLease> {
        let lease = self.owner.register(
            generation,
            backing.len() as u64,
            Box::new(Rc::new(RefCell::new(backing))),
            &charge,
        )?;
        self.allocations.insert(lease.allocation_id(), charge);
        Ok(lease)
    }
    pub fn retain_device(&self, lease: &DeviceLease) -> Result<DeviceLease> {
        self.owner.retain(lease)
    }
    /// A second owned handle on one pinned allocation: the host mirror of `retain_device`
    /// (lane/spill-c-20260919 day 16, Option C). Both handles own the allocation and its governor
    /// charge; the last one dropped releases both (`PinnedAllocation::drop`). What it is for: an
    /// H2D whose caller keeps the host copy resident after the copy. The contract's H2D consumes
    /// the source lease it is handed (`retire_source`), so a caller that must keep its copy hands
    /// the op a twin and keeps its own handle. The engine already accepts a shared H2D source (a
    /// taken D2H destination may be an H2D source before acknowledgement; `validate` counts owners
    /// for a D2H destination only). While a twin lives, `write` on either handle refuses `Busy`
    /// and a D2H into either refuses `Busy`: nothing mutates bytes another owner may be reading.
    /// Owner thread and owner context only; a released allocation refuses `AlreadyReleased`.
    pub fn retain_host(&self, lease: &CudaPinnedLease) -> Result<CudaPinnedLease> {
        self.check_thread()?;
        let backing = lease
            .allocation
            .backing
            .as_ref()
            .ok_or(Error::AlreadyReleased)?;
        if !Arc::ptr_eq(backing.context(), self.stream.context()) {
            return Err(Error::WrongOwner);
        }
        Ok(CudaPinnedLease {
            allocation: lease.allocation.clone(),
        })
    }
    /// Refuse live ticket ownership without waiting on the CUDA stream. A
    /// synchronous wait here would consume the producer-pending refusal state
    /// (and deadlock callers whose producer needs an explicit owner advance).
    fn require_unbound(&self, lease: &DeviceLease) -> Result<()> {
        self.owner.resolve::<Rc<RefCell<KvPlane>>>(lease)?;
        if self.entries.values().any(|e| {
            !e.retired
                && e.items.iter().flatten().any(|i| {
                    i.device
                        .as_ref()
                        .is_some_and(|d| d.allocation_id() == lease.allocation_id())
                })
        }) {
            return Err(Error::Busy);
        }
        Ok(())
    }
    /// WP-A day 48 (`DAY48.md` design S4): the owner stream is drained, the copy stream is not.
    /// Every copy-stream operation on a registered lease is an item (or a receipt, digest or fault
    /// over the items) of a ticket that names it; `require_unbound` refuses while any unretired
    /// ticket names the lease, and a ticket retires only after its landing observed every item and
    /// receipt event. So nothing on the copy stream can still read or write this lease here, and a
    /// whole-stream drain would wait on other memory's work only (DAY46 section 3: the capture
    /// settle held 3.2 ms behind a demote's span receipt).
    pub fn release_device(&mut self, lease: &DeviceLease) -> Result<()> {
        self.check_thread()?;
        self.require_unbound(lease)?;
        cuda(self.stream.synchronize())?;
        self.owner.release(lease)?;
        let charge = self
            .allocations
            .get(&lease.allocation_id())
            .ok_or(Error::ForeignLease)?;
        self.governor.borrow_mut().release(charge)?;
        self.allocations.remove(&lease.allocation_id());
        Ok(())
    }
    /// Diagnostic release: observe the private backing Rc without retaining it.
    /// A zero post-release count proves the RefCell<CudaSlice> destructor ran;
    /// it does NOT prove the async pool returned physical memory to the driver.
    pub fn release_device_observed(&mut self, lease: &DeviceLease) -> Result<(usize, usize)> {
        let (before, weak) = {
            let backing = self.owner.resolve::<Rc<RefCell<KvPlane>>>(lease)?;
            (Rc::strong_count(&backing), Rc::downgrade(&backing))
        };
        self.release_device(lease)?;
        Ok((before, weak.strong_count()))
    }
    /// Registry occupancy, not physical residency.
    pub fn device_registry_len(&self) -> usize {
        self.allocations.len()
    }
    /// Transfer native backing out without a copy. The caller assumes accounting
    /// after this returns; live leases/bindings fail Busy without losing ownership.
    pub fn take_device(&mut self, lease: &DeviceLease) -> Result<CudaSlice<u8>> {
        if self
            .owner
            .resolve::<Rc<RefCell<KvPlane>>>(lease)?
            .borrow()
            .is_vmm()
        {
            return Err(Error::Unsupported);
        }
        self.take_plane(lease)?
            .into_pooled()
            .map_err(|_| Error::Unsupported)
    }
    /// Transfer the complete typed owner, never a raw VMM CudaSlice.
    /// WP-A day 48 (design S4): the owner stream is drained, the copy stream is not (see
    /// `release_device`).
    pub fn take_plane(&mut self, lease: &DeviceLease) -> Result<KvPlane> {
        self.check_thread()?;
        self.require_unbound(lease)?;
        cuda(self.stream.synchronize())?;
        let backing = self.owner.resolve::<Rc<RefCell<KvPlane>>>(lease)?.clone();
        self.owner.release(lease)?;
        let charge = self
            .allocations
            .get(&lease.allocation_id())
            .ok_or(Error::ForeignLease)?;
        self.governor.borrow_mut().release(charge)?;
        self.allocations.remove(&lease.allocation_id());
        // No public API can clone this private Rc. Registry release removed the
        // sole other owner after checking all sealed lease and binding retention.
        Ok(Rc::try_unwrap(backing)
            .map_err(|_| Error::Busy)?
            .into_inner())
    }
    /// Retire only source ownership after observed DMA completion. This never
    /// marks the whole ticket retired or releases destination consumer bindings.
    /// D2H callers can then release_device their source registry lease; H2D host
    /// source backing and its pinned-budget charge are dropped here.
    pub fn retire_source(&mut self, ticket: &TransferTicket) -> Result<()> {
        self.progress(ticket)?;
        let e = self.entries.get(ticket).ok_or(Error::UnknownTicket)?;
        if e.retired {
            return Ok(());
        }
        if Self::holds_cancelled_source(e) {
            return Err(Error::Busy);
        }
        if !e.completion.producer_done || !e.source.idle()? {
            return Err(Error::Busy);
        }
        // Another live H2D ticket may still bind this D2H source as a consumer
        // destination. Do not detach the source while that binding remains live.
        for source in e
            .items
            .iter()
            .flatten()
            .filter(|i| i.direction == CopyDirection::DeviceToHost)
        {
            if let Some(device) = &source.device
                && self.entries.values().any(|other| {
                    !other.retired
                        && other.items.iter().flatten().any(|i| {
                            i.direction == CopyDirection::HostToDevice
                                && i.device
                                    .as_ref()
                                    .is_some_and(|d| d.allocation_id() == device.allocation_id())
                        })
                })
            {
                return Err(Error::Busy);
            }
        }
        let e = self.entries.get_mut(ticket).unwrap();
        for item in e.items.iter_mut().flatten() {
            if item.source_retired {
                continue;
            }
            match item.direction {
                CopyDirection::HostToDevice => {
                    item.host.take();
                }
                CopyDirection::DeviceToHost => {
                    item.device.take();
                }
                // A capture's source is borrowed (the caller's live plane): nothing to retire.
                CopyDirection::DeviceToDevice => {}
            }
            item.source_retired = true;
        }
        Ok(())
    }
    /// Day-11 rule 1 (lead ruling 9): a cancelled restore still holding an H2D source the
    /// caller has not recovered. Such a ticket is never drained by `retire` or `retire_source`.
    fn holds_cancelled_source(e: &Entry) -> bool {
        e.cancelled
            && e.items
                .iter()
                .flatten()
                .any(|i| i.direction == CopyDirection::HostToDevice && !i.source_retired)
    }
    /// Day-11 rule 1: hand a cancelled restore's untouched host source back to the caller,
    /// exactly once, after its DMA has been observed complete and every source consumer and
    /// source graph pin has retired. The lease keeps its own pinned charge; the caller owns it
    /// again as it did before submission. See `TransferEngine::recover_source`.
    pub fn recover_source(
        &mut self,
        ticket: &TransferTicket,
        item: u32,
    ) -> Result<CudaPinnedLease> {
        self.progress(ticket)?;
        let e = self.entries.get(ticket).ok_or(Error::UnknownTicket)?;
        if e.retired || e.published {
            return Err(Error::AlreadyReleased);
        }
        if !e.cancelled {
            return Err(Error::NotReady);
        }
        // WP-A day 34 (`h2d_deferred_checksum` rule 3): a source a view still reads stays here.
        if e.deferred.as_ref().is_some_and(|d| d.out > 0) {
            return Err(Error::Busy);
        }
        let i = e
            .items
            .get(item as usize)
            .ok_or(Error::InvalidLayout)?
            .as_ref()
            .ok_or(Error::Rejected)?;
        if i.direction != CopyDirection::HostToDevice {
            return Err(Error::Unsupported);
        }
        if i.source_retired {
            return Err(Error::AlreadyReleased);
        }
        // The DMA or a source consumer may still read the source: no hand-back under a live read.
        if !e.completion.items[item as usize].segments[0].producer_done || !e.source.idle()? {
            return Err(Error::Busy);
        }
        let e = self.entries.get_mut(ticket).unwrap();
        let i = e.items[item as usize].as_mut().unwrap();
        let host = i.host.take().ok_or(Error::AlreadyReleased)?;
        i.source_retired = true;
        Ok(host)
    }
    /// Resolve only a sealed view issued by this backend, while its exact ticket
    /// is still live. Consumers must launch on owner_stream(), then record_consumer.
    pub fn resolve_ready(&self, ready: &ReadyView<'_>) -> Result<Ref<'_, RefCell<KvPlane>>> {
        self.check_thread()?;
        let e = self
            .entries
            .get(&ready.ticket())
            .ok_or(Error::UnknownTicket)?;
        if e.cancelled || e.retired {
            return Err(Error::NotReady);
        }
        self.owner
            .resolve::<Rc<RefCell<KvPlane>>>(ready.destination())
            .map(|r| Ref::map(r, |b| b.as_ref()))
    }
    /// Launch/read the exact ready destination without escaping its lease. The
    /// callback must submit all uses on the supplied owner stream, not a peer.
    pub fn with_destination<R>(
        &mut self,
        ticket: &TransferTicket,
        item: u32,
        current: Epochs,
        use_device: impl FnOnce(&CudaSlice<u8>, &Arc<CudaStream>) -> Result<R>,
    ) -> Result<R> {
        self.publishable(ticket, current)?;
        let e = self.entries.get_mut(ticket).unwrap();
        let i = e
            .items
            .get(item as usize)
            .ok_or(Error::InvalidLayout)?
            .as_ref()
            .ok_or(Error::Rejected)?;
        if i.direction != CopyDirection::HostToDevice {
            return Err(Error::Unsupported);
        }
        self.owner.ready_view(
            i.device.as_ref().ok_or(Error::AlreadyReleased)?,
            &e.completion,
            &e.expected,
            current,
        )?;
        e.published = true;
        let backing = self
            .owner
            .resolve::<Rc<RefCell<KvPlane>>>(i.device.as_ref().ok_or(Error::AlreadyReleased)?)?;
        use_device(&backing.borrow(), &self.stream)
    }
    pub fn owner_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
    fn next_fence(&mut self, generation: u64) -> Result<FenceId> {
        self.fence_sequence = self.fence_sequence.checked_add(1).ok_or(Error::Overflow)?;
        Ok(FenceId {
            issuer: self.owner.issuer(),
            owner: self.device,
            generation,
            sequence: self.fence_sequence,
        })
    }
    /// Record after source production on the owner stream. Numeric FenceId fields
    /// alone are never authority: submissions also require this retained CUDA event.
    pub fn record_producer(&mut self, generation: u64) -> Result<FenceId> {
        self.check_thread()?;
        let f = self.next_fence(generation)?;
        let event = cuda(self.stream.record_event(None))?;
        self.producers.insert(f.sequence, (f, event));
        Ok(f)
    }
    pub fn release_producer(&mut self, fence: FenceId) -> Result<()> {
        self.check_thread()?;
        let (f, event) = self
            .producers
            .get(&fence.sequence)
            .ok_or(Error::WrongOwner)?;
        if f != &fence {
            return Err(Error::WrongOwner);
        }
        if !event_done(event)? {
            return Err(Error::Busy);
        }
        self.producers.remove(&fence.sequence);
        Ok(())
    }
    /// Record strictly after consumer submission. Each event is bound to exactly
    /// one ticket; a pre-copy producer fence cannot be reused as consumer completion.
    pub fn record_consumer(&mut self, ticket: &TransferTicket) -> Result<FenceId> {
        self.check_thread()?;
        let e = self.entries.get(ticket).ok_or(Error::UnknownTicket)?;
        if !e.published || e.retired {
            return Err(Error::NotReady);
        }
        let f = self.next_fence(ticket.epochs.dst_gen)?;
        let event = cuda(self.stream.record_event(None))?;
        self.entries
            .get_mut(ticket)
            .unwrap()
            .destination
            .consumer_event = Some((f, event));
        Ok(f)
    }
    /// Rule 3 (WP-A day 19, `memra_tier::conformance::h2d_reader_fence` under
    /// `ReaderWaitInstall::AtSettle`; day 21, `d2d_restore_ready`): install the OWNER stream's
    /// wait on every item's completion event that is not yet fenced, then fence it. The
    /// destination's consumer is the owner stream (the D2D restore and every kernel after it for
    /// an H2D; the request's first prime chunk for a D2D restore); until this runs an off-owner
    /// H2D is not `consumer_fenced` and `ready_view` refuses it `NotReady`, and a D2D restore is
    /// not `ready`. Idempotent: an item fenced at submit (the owner-stream program, every D2H,
    /// every D2D capture) is left alone; a D2H is never fenced here (its consumer is the host).
    /// Recorded strictly before `ready_view` and before `record_consumer`, whose event then orders
    /// behind the copy. A CUDA error here leaves the ticket unpublished with its destinations
    /// bound; the caller unwinds through `cancel` as for any pre-publication refusal.
    pub fn install_consumer_wait(&mut self, ticket: &TransferTicket) -> Result<()> {
        self.check_thread()?;
        let e = self.entries.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        if e.unknown {
            return Err(Error::Quarantined);
        }
        if e.cancelled || e.retired {
            return Err(Error::NotReady);
        }
        let dst_gen = ticket.epochs.dst_gen;
        let mut fences = 0u64;
        for (i, item) in e.items.iter().enumerate() {
            let Some(item) = item else {
                continue;
            };
            let s = &e.completion.items[i].segments[0];
            if s.consumer_fenced || item.direction == CopyDirection::DeviceToHost {
                continue;
            }
            if s.status == ItemStatus::Quarantined {
                return Err(Error::Quarantined);
            }
            fences += 1;
        }
        if self.fence_sequence.checked_add(fences).is_none() {
            return Err(Error::Overflow);
        }
        let stream = self.stream.clone();
        for i in 0..e.items.len() {
            let Some(item) = e.items[i].as_ref() else {
                continue;
            };
            let s = &e.completion.items[i].segments[0];
            if s.consumer_fenced || item.direction == CopyDirection::DeviceToHost {
                continue;
            }
            let event = item.event.as_ref().ok_or(Error::Quarantined)?;
            if let Err(err) = cuda(stream.wait(event)) {
                e.unknown = true;
                return Err(err);
            }
            self.fence_sequence += 1;
            let fence = FenceId {
                issuer: self.owner.issuer(),
                owner: self.device,
                generation: dst_gen,
                sequence: self.fence_sequence,
            };
            let s = &mut e.completion.items[i].segments[0];
            s.consumer_fence = Some(fence);
            s.consumer_fenced = true;
        }
        // WP-A day 32 (rule 3 of `h2d_span_batch`, rule 3 of `h2d_reader_fence` extended over the
        // spans): the owner stream waits on every H2D span's completion event; only then does
        // `take_h2d_spans` hand a destination out. A span without an event is quarantined.
        if let Some(b) = &mut e.h2d_spans
            && !b.fenced
        {
            for (_, event) in &b.slots {
                let Some(event) = event.as_ref() else {
                    e.unknown = true;
                    return Err(Error::Quarantined);
                };
                if let Err(err) = cuda(stream.wait(event)) {
                    e.unknown = true;
                    return Err(err);
                }
            }
            b.fenced = true;
        }
        e.completion.consumer_fenced = e
            .completion
            .items
            .iter()
            .filter(|i| i.accepted)
            .all(|i| i.segments.iter().all(|s| s.consumer_fenced));
        Ok(())
    }
    /// The capture class (WP-A day 20, memra#536 Move 2 slice 1;
    /// `memra_tier::conformance::d2d_capture_publish`): ONE batch of same-device copies issued on
    /// the COPY stream behind each op's producer fence (an owner-stream event recorded after the
    /// boundary chunk), from the head of a BORROWED live source into an OWNED registered
    /// destination lease. Not a `TransferOp`: `ContiguousCopy` takes two owned leases and the
    /// registry admits only moved buffers, and a capture's source is a session cache's plane the
    /// decoding session keeps (its rows `0..bytes` are append-only per position, the capture law;
    /// the recurrent state, which the next step overwrites, is not in this class and stays on the
    /// owner stream). All-or-nothing: a refused op refuses the whole batch with nothing submitted
    /// (the destinations drop here; the caller keeps its retained twins and takes the planes
    /// back through them). No owner-stream wait is installed at any point: the destination's
    /// consumer is the caller's publication into the device prefix index, host-ordered after
    /// `capture_landed`, so each item is fenced at submit as a D2H is. No checksum term (slice
    /// 3): `Completion::require`, `ready_view` and `take_destination` refuse a capture item, so
    /// the host-contract publication gate cannot publish it. Requires the copy stream
    /// (`new_with_copy_stream`); under `new` the class does not exist (`Unsupported`): the
    /// on-tick program is the caller's own `prefix_snapshot`.
    pub fn submit_d2d_capture(
        &mut self,
        ops: Vec<D2dCapture<'_>>,
        epochs: Epochs,
    ) -> Result<TransferTicket> {
        self.check_thread()?;
        // The one-shot arm is spent by THIS submit whether or not admission refuses it (revuto
        // round 2 on integ38 #639): taken ahead of every fallible step, so a refused submit cannot
        // leave the arm live for the next batch of either class.
        let fault = self.early_reader.take();
        let Some(copy) = self.copy.clone() else {
            return Err(Error::Unsupported);
        };
        if ops.is_empty() {
            return Err(Error::EmptyBatch);
        }
        if ops.len() > u32::MAX as usize
            || self.sequence == u64::MAX
            || self.fence_sequence.checked_add(ops.len() as u64).is_none()
        {
            return Err(Error::Overflow);
        }
        for op in &ops {
            if op.bytes == 0
                || op.bytes > op.destination.bytes()
                || op.bytes > op.source.len() as u64
            {
                return Err(Error::InvalidLayout);
            }
            if op.destination.generation() != epochs.dst_gen {
                return Err(Error::StaleEpoch);
            }
            let backing = self
                .owner
                .resolve::<Rc<RefCell<KvPlane>>>(&op.destination)?;
            if !Arc::ptr_eq(backing.borrow().stream(), &self.stream)
                || !Arc::ptr_eq(op.source.stream().context(), self.stream.context())
            {
                return Err(Error::WrongOwner);
            }
            if self
                .producers
                .get(&op.producer_fence.sequence)
                .is_none_or(|(actual, _)| actual != &op.producer_fence)
            {
                return Err(Error::WrongOwner);
            }
        }
        // WP-A day 22 (slice 3): the receipt's lanes and the one-shot fault, before the charge.
        let mut scratch = self.receipt_scratch(ops.len())?;
        let mut request = self.request();
        request.bytes.inflight = ops.len() as u64;
        let charge = self.governor.borrow_mut().reserve(&request)?;
        self.sequence += 1;
        let ticket = TransferTicket {
            issuer: self.owner.issuer(),
            sequence: self.sequence,
            epochs,
        };
        let mut entry = Entry {
            items: vec![],
            completion: Completion {
                ticket,
                items: vec![],
                producer_done: false,
                consumer_fenced: false,
            },
            expected: vec![],
            charge: Some(charge),
            cancelled: false,
            published: false,
            retired: false,
            unknown: false,
            source: Retention::default(),
            destination: Retention::default(),
            receipt: None,
            spans: None,
            spans_taken: false,
            h2d_spans: None,
            h2d_spans_taken: false,
            deferred: None,
            d2h_receipt: None,
        };
        for (i, op) in ops.into_iter().enumerate() {
            let mut s = SegmentCompletion {
                segment: 0,
                status: ItemStatus::Pending,
                valid_bytes: 0,
                io_bytes: 0,
                checksum: None,
                epochs,
                producer_done: false,
                consumer_fenced: false,
                consumer_fence: None,
                error: None,
            };
            entry.expected.push(vec![SegmentExpectation {
                valid_bytes: op.bytes,
                io_bytes: op.bytes,
                checksum: [0; 32],
            }]);
            let mut item = Item {
                host: None,
                device: Some(op.destination),
                bytes: op.bytes,
                direction: CopyDirection::DeviceToDevice,
                event: None,
                taken: false,
                source_retired: false,
            };
            // From this point any CUDA error may mean work was submitted: accept + quarantine.
            let submit = (|| {
                cuda(copy.wait(&self.producers[&op.producer_fence.sequence].1))?;
                let backing = self.owner.resolve::<Rc<RefCell<KvPlane>>>(
                    item.device.as_ref().ok_or(Error::AlreadyReleased)?,
                )?;
                let n = op.bytes as usize;
                // WP-A day 22 (slice 3, `d2d_receipt_witnessed`): the SOURCE digest behind the
                // producer fence, the copy, then the DESTINATION digest after it, all on the copy
                // stream; the fault's arm delays the copy and reads the destination early.
                cuda(copy.wait(&scratch.zeroed))?;
                // Once per BATCH (revuto on integ38 #639): the copy stream is serial, so one spin
                // ahead of the first item delays every item's copy; a spin per item multiplied
                // the documented 200 ms by the item count (6.4 s on a 32-plane entry) and any
                // Block settle of the ticket held the owner thread for that whole window.
                if let Some(ns) = fault
                    && i == 0
                {
                    self.delay_on(&copy, ns)?;
                }
                self.digest_on(
                    &copy,
                    &op.source.slice(0..n),
                    &mut scratch.lanes.slice_mut(64 * i..64 * i + 32),
                )?;
                cuda(copy.memcpy_dtod(
                    &op.source.slice(0..n),
                    &mut backing.borrow_mut().slice_mut(..n),
                ))?;
                if fault.is_none() {
                    self.digest_on(
                        &copy,
                        &backing.borrow().slice(..n),
                        &mut scratch.lanes.slice_mut(64 * i + 32..64 * i + 64),
                    )?;
                } else {
                    // The early reader: the owner stream reads the destination now, unordered with
                    // the delayed copy; the copy stream waits on that read before its event so the
                    // lanes the settle reads are the early reader's.
                    self.digest_on(
                        &self.stream,
                        &backing.borrow().slice(..n),
                        &mut scratch.lanes.slice_mut(64 * i + 32..64 * i + 64),
                    )?;
                    let read = cuda(self.stream.record_event(None))?;
                    cuda(copy.wait(&read))?;
                }
                drop(backing);
                item.event = Some(cuda(copy.record_event(None))?);
                // Fenced at submit, as an off-owner D2H: the consumer is the caller's publication
                // after the event is observed complete; no owner-stream wait exists or is owed.
                s.consumer_fence = Some(self.next_fence(epochs.dst_gen)?);
                s.consumer_fenced = true;
                s.io_bytes = item.bytes;
                Ok(())
            })();
            if let Err(error) = submit {
                s.status = ItemStatus::Quarantined;
                s.error = Some(error);
                entry.unknown = true;
            }
            entry.items.push(Some(item));
            entry.completion.items.push(ItemOutcome {
                item: i as u32,
                accepted: true,
                segments: vec![s],
            });
        }
        Self::seal_receipt(&copy, &mut entry, scratch);
        self.entries.insert(ticket, entry);
        Ok(ticket)
    }
    /// The capture's publication predicate (`d2d_capture_publish` rule 1 and 2): every item's
    /// completion event observed complete. The caller publishes into the device prefix index
    /// only when this answers `true`, then `retire(ticket, None)`, `acknowledge`, and takes the
    /// planes back through its twins. A ticket that is not a capture is refused `Unsupported`; a
    /// quarantined observation is `Quarantined`, never `true`.
    pub fn capture_landed(&mut self, ticket: &TransferTicket) -> Result<bool> {
        self.progress(ticket)?;
        let e = &self.entries[ticket];
        if e.items
            .iter()
            .flatten()
            .any(|i| i.direction != CopyDirection::DeviceToDevice)
        {
            return Err(Error::Unsupported);
        }
        Ok(e.completion.producer_done)
    }
    /// The restore class (WP-A day 21, memra#536 Move 2 slice 2;
    /// `memra_tier::conformance::d2d_restore_ready`): ONE batch of same-device copies issued on
    /// the COPY stream behind each op's producer fence (an owner-stream event recorded at submit,
    /// after the recurrent-state copies), from the head of a BORROWED published source (a device
    /// prefix entry's plane, pinned by the device LRU from submit through acknowledge: the
    /// producer-side guarantee for a borrowed source) into a BORROWED destination view (the parked
    /// request's fresh session cache plane). Nothing on either side is the registry's, so no
    /// destination lease exists and `take_destination`, `ready_view` and `Completion::require`
    /// refuse the items (no checksum term until slice 3). All-or-nothing: a refused op refuses the
    /// whole batch with nothing submitted. The items are NOT fenced at submit (rule 3: the
    /// destination's consumer is the owner stream, whose wait on each completion event is
    /// installed by `install_consumer_wait` at the settle, before the request is re-admitted);
    /// `restore_landed` answers the landing, and the caller's `ready` is the landing plus the
    /// installed wait. Requires the copy stream; under `new` the class does not exist
    /// (`Unsupported`): the on-tick program is the caller's own `prefix_restore_at`.
    pub fn submit_d2d_restore(
        &mut self,
        ops: Vec<D2dRestore<'_>>,
        epochs: Epochs,
    ) -> Result<TransferTicket> {
        self.check_thread()?;
        // The one-shot arm is spent by THIS submit whether or not admission refuses it (revuto
        // round 2 on integ38 #639): taken ahead of every fallible step, so a refused submit cannot
        // leave the arm live for the next batch of either class.
        let fault = self.early_reader.take();
        let Some(copy) = self.copy.clone() else {
            return Err(Error::Unsupported);
        };
        if ops.is_empty() {
            return Err(Error::EmptyBatch);
        }
        if ops.len() > u32::MAX as usize || self.sequence == u64::MAX {
            return Err(Error::Overflow);
        }
        for op in &ops {
            if op.bytes == 0
                || op.bytes != op.destination.len() as u64
                || op.bytes > op.source.len() as u64
            {
                return Err(Error::InvalidLayout);
            }
            if !Arc::ptr_eq(op.source.stream().context(), self.stream.context()) {
                return Err(Error::WrongOwner);
            }
            if self
                .producers
                .get(&op.producer_fence.sequence)
                .is_none_or(|(actual, _)| actual != &op.producer_fence)
            {
                return Err(Error::WrongOwner);
            }
        }
        // WP-A day 22 (slice 3): the receipt's lanes and the one-shot fault, before the charge.
        let mut scratch = self.receipt_scratch(ops.len())?;
        let mut request = self.request();
        request.bytes.inflight = ops.len() as u64;
        let charge = self.governor.borrow_mut().reserve(&request)?;
        self.sequence += 1;
        let ticket = TransferTicket {
            issuer: self.owner.issuer(),
            sequence: self.sequence,
            epochs,
        };
        let mut entry = Entry {
            items: vec![],
            completion: Completion {
                ticket,
                items: vec![],
                producer_done: false,
                consumer_fenced: false,
            },
            expected: vec![],
            charge: Some(charge),
            cancelled: false,
            published: false,
            retired: false,
            unknown: false,
            source: Retention::default(),
            destination: Retention::default(),
            receipt: None,
            spans: None,
            spans_taken: false,
            h2d_spans: None,
            h2d_spans_taken: false,
            deferred: None,
            d2h_receipt: None,
        };
        for (i, mut op) in ops.into_iter().enumerate() {
            let mut s = SegmentCompletion {
                segment: 0,
                status: ItemStatus::Pending,
                valid_bytes: 0,
                io_bytes: 0,
                checksum: None,
                epochs,
                producer_done: false,
                // Unfenced at submit (rule 3): the owner stream's wait is installed at the settle.
                consumer_fenced: false,
                consumer_fence: None,
                error: None,
            };
            entry.expected.push(vec![SegmentExpectation {
                valid_bytes: op.bytes,
                io_bytes: op.bytes,
                checksum: [0; 32],
            }]);
            let mut item = Item {
                host: None,
                device: None,
                bytes: op.bytes,
                direction: CopyDirection::DeviceToDevice,
                event: None,
                taken: false,
                source_retired: false,
            };
            // From this point any CUDA error may mean work was submitted: accept + quarantine.
            let submit = (|| {
                cuda(copy.wait(&self.producers[&op.producer_fence.sequence].1))?;
                let n = op.bytes as usize;
                // WP-A day 22 (slice 3): source digest behind the fence, the copy, the destination
                // digest after it, on the copy stream; the fault's arm as in the capture class.
                cuda(copy.wait(&scratch.zeroed))?;
                // Once per BATCH (revuto on integ38 #639): the copy stream is serial, so one spin
                // ahead of the first item delays every item's copy; a spin per item multiplied
                // the documented 200 ms by the item count (6.4 s on a 32-plane entry) and any
                // Block settle of the ticket held the owner thread for that whole window.
                if let Some(ns) = fault
                    && i == 0
                {
                    self.delay_on(&copy, ns)?;
                }
                self.digest_on(
                    &copy,
                    &op.source.slice(0..n),
                    &mut scratch.lanes.slice_mut(64 * i..64 * i + 32),
                )?;
                cuda(copy.memcpy_dtod(&op.source.slice(0..n), &mut op.destination))?;
                if fault.is_none() {
                    self.digest_on(
                        &copy,
                        &op.destination.as_view(),
                        &mut scratch.lanes.slice_mut(64 * i + 32..64 * i + 64),
                    )?;
                } else {
                    self.digest_on(
                        &self.stream,
                        &op.destination.as_view(),
                        &mut scratch.lanes.slice_mut(64 * i + 32..64 * i + 64),
                    )?;
                    let read = cuda(self.stream.record_event(None))?;
                    cuda(copy.wait(&read))?;
                }
                item.event = Some(cuda(copy.record_event(None))?);
                s.io_bytes = item.bytes;
                Ok(())
            })();
            if let Err(error) = submit {
                s.status = ItemStatus::Quarantined;
                s.error = Some(error);
                entry.unknown = true;
            }
            entry.items.push(Some(item));
            entry.completion.items.push(ItemOutcome {
                item: i as u32,
                accepted: true,
                segments: vec![s],
            });
        }
        Self::seal_receipt(&copy, &mut entry, scratch);
        self.entries.insert(ticket, entry);
        Ok(ticket)
    }
    /// The restore's landing predicate (`d2d_restore_ready` rules 1 and 2): every item's
    /// completion event observed complete. Landing is NOT readiness: the caller's `ready` is this
    /// plus every item fenced by `install_consumer_wait`, read from `poll`'s `consumer_fenced`.
    /// A ticket that is not a same-device batch is refused `Unsupported`; a quarantined
    /// observation is `Quarantined`, never `true`.
    pub fn restore_landed(&mut self, ticket: &TransferTicket) -> Result<bool> {
        self.capture_landed(ticket)
    }
    /// Retain both sides for legacy whole-transfer graph users.
    pub fn pin_graph(&mut self, ticket: &TransferTicket) -> Result<GraphPin> {
        let source = self.pin_source_graph(ticket)?;
        let destination = self.pin_destination_graph(ticket)?;
        Ok(GraphPin {
            _pins: source._pins.into_iter().chain(destination._pins).collect(),
        })
    }
    /// Source and destination graph lifetimes are independent. A new source
    /// graph cannot be attached after source ownership has been retired.
    pub fn pin_source_graph(&mut self, ticket: &TransferTicket) -> Result<GraphPin> {
        self.check_thread()?;
        let e = self.entries.get(ticket).ok_or(Error::UnknownTicket)?;
        if e.retired || e.cancelled || e.items.iter().flatten().any(|i| i.source_retired) {
            return Err(Error::NotReady);
        }
        Ok(GraphPin {
            _pins: vec![e.source.graph_pins.clone()],
        })
    }
    pub fn pin_destination_graph(&mut self, ticket: &TransferTicket) -> Result<GraphPin> {
        self.check_thread()?;
        let e = self.entries.get(ticket).ok_or(Error::UnknownTicket)?;
        if e.retired || e.cancelled {
            return Err(Error::NotReady);
        }
        Ok(GraphPin {
            _pins: vec![e.destination.graph_pins.clone()],
        })
    }
    /// Record after source consumer work submitted on the owner stream.
    pub fn record_source_consumer(&mut self, ticket: &TransferTicket) -> Result<FenceId> {
        self.check_thread()?;
        let e = self.entries.get(ticket).ok_or(Error::UnknownTicket)?;
        if e.retired || e.items.iter().flatten().any(|i| i.source_retired) {
            return Err(Error::NotReady);
        }
        let f = self.next_fence(ticket.epochs.src_gen)?;
        let event = cuda(self.stream.record_event(None))?;
        self.entries.get_mut(ticket).unwrap().source.consumer_event = Some((f, event));
        Ok(f)
    }
    /// A lost observation is not completion. Explicit recovery re-observes real
    /// recorded CUDA events; it never synthesizes an event or a successful copy.
    pub fn quarantine_observation(&mut self, ticket: &TransferTicket) -> Result<()> {
        self.check_thread()?;
        let e = self.entries.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        if e.retired {
            return Err(Error::AlreadyReleased);
        }
        e.unknown = true;
        Ok(())
    }
    pub fn synchronize(&mut self, ticket: &TransferTicket) -> Result<()> {
        self.check_thread()?;
        let e = self.entries.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        for item in e.items.iter().flatten() {
            cuda(item.event.as_ref().ok_or(Error::Quarantined)?.synchronize())?;
        }
        // Integ38 (lead review of WP-A day 22): a D2D batch lands only with its receipt lanes, and
        // the receipt's D2H is recorded on the copy stream AFTER the last item's event. A host
        // wait on the items alone leaves a window in which `progress` sees no lanes, the items
        // stay unlanded and the settle's Block arm latches the tier for a copy that had landed.
        // The receipt event is part of the batch's landing, so the wait covers it too.
        if let Some(r) = &e.receipt {
            cuda(r.event.as_ref().ok_or(Error::Quarantined)?.synchronize())?;
        }
        // WP-A day 38 (design G'): a device-receipt D2H batch lands only with its receipt; G' to
        // G''' ran it beside the copies, so the items could complete first, and the host wait
        // covers the receipt event (the integ38 shape). Under G4 the receipt precedes the copies on
        // the copy stream, and the wait is kept: the landing's definition, not the stream order.
        if let Some(r) = &e.d2h_receipt {
            cuda(
                r.scratch
                    .event
                    .as_ref()
                    .ok_or(Error::Quarantined)?
                    .synchronize(),
            )?;
            // Design G''': a held batch lands at the hold's end, so the host wait covers it too.
            if let Some(h) = r.hold_until {
                let now = std::time::Instant::now();
                if now < h {
                    std::thread::sleep(h - now);
                }
            }
        }
        // WP-A day 30: a batch with spans lands with them (rule 2 of `d2h_span_batch`), so the
        // host wait covers every span's event; a span without an event is quarantined.
        if let Some(b) = &e.spans {
            for (_, event) in &b.slots {
                cuda(event.as_ref().ok_or(Error::Quarantined)?.synchronize())?;
            }
        }
        // WP-A day 32: the same for the spans of an H2D batch (rule 2 of `h2d_span_batch`).
        if let Some(b) = &e.h2d_spans {
            for (_, event) in &b.slots {
                cuda(event.as_ref().ok_or(Error::Quarantined)?.synchronize())?;
            }
            // WP-A day 40 (design S): the H2D spans' receipt is part of their landing.
            if let Some(r) = &b.receipt {
                cuda(r.event.as_ref().ok_or(Error::Quarantined)?.synchronize())?;
            }
        }
        e.unknown = false;
        self.progress(ticket)
    }
    /// WP-A day 30 (memra#536 Move 2 owed item 1, the D2H half; `memra_tier::conformance::
    /// d2h_span_batch`): attach typed f32 spans to a live D2H batch right after its submission.
    /// The copy stream waits on a fresh owner-stream event (every writer of every source is on
    /// the owner stream, so it is ordered before the copy), then per span one full-length
    /// `cuMemcpyDtoHAsync` with no host wait and one completion event. From here the engine OWNS
    /// each source and destination until `take_d2h_spans`: the engine context runs without
    /// cudarc's event tracking, so a caller-side free of a source would not be ordered behind the
    /// copy. Rule 1, refusal before enqueue is whole: no copy stream (`Unsupported`), an unknown,
    /// retired or quarantined ticket, a ticket that is not a D2H batch (`Unsupported`), one that
    /// already carries spans (`Busy`), an empty batch, a zero-length span or a destination whose
    /// length is not the source's (`InvalidLayout`), a source of another stream (`WrongOwner`), or
    /// a failed owner-stream record or copy-stream wait: every span comes back, nothing enqueued,
    /// nothing quarantined. Rule 5: an enqueue or event error from the first enqueue on keeps
    /// every span and quarantines the ticket (`Ok`: the batch was accepted and cannot land).
    #[allow(clippy::result_large_err)]
    pub fn submit_d2h_spans(
        &mut self,
        ticket: &TransferTicket,
        spans: Vec<D2hSpan>,
    ) -> std::result::Result<(), (Error, Vec<D2hSpan>)> {
        let admitted = (|| {
            self.check_thread()?;
            let copy = self.copy.clone().ok_or(Error::Unsupported)?;
            let e = self.entries.get(ticket).ok_or(Error::UnknownTicket)?;
            if e.retired {
                return Err(Error::AlreadyReleased);
            }
            if e.unknown {
                return Err(Error::Quarantined);
            }
            if e.spans.is_some() || e.spans_taken {
                return Err(Error::Busy);
            }
            if e.receipt.is_some()
                || e.items
                    .iter()
                    .flatten()
                    .any(|i| i.direction != CopyDirection::DeviceToHost)
            {
                return Err(Error::Unsupported);
            }
            if spans.is_empty() {
                return Err(Error::EmptyBatch);
            }
            for s in &spans {
                let bytes = s.source.len().checked_mul(4).ok_or(Error::Overflow)?;
                if bytes == 0 || s.destination.len() != bytes {
                    return Err(Error::InvalidLayout);
                }
                if !Arc::ptr_eq(s.source.stream(), &self.stream) {
                    return Err(Error::WrongOwner);
                }
            }
            // WP-A day 40 (design S): the spans' receipt scratch, allocated before anything is
            // enqueued, so a refusal here submits nothing.
            let scratch = match self.receipt {
                Some(_) => Some(self.receipt_scratch(spans.len())?),
                None => None,
            };
            let fence = cuda(self.stream.record_event(None))?;
            cuda(copy.wait(&fence))?;
            Ok((copy, scratch))
        })();
        let (copy, mut scratch) = match admitted {
            Ok(admitted) => admitted,
            Err(error) => return Err((error, spans)),
        };
        #[cfg(test)]
        let mut fault = std::mem::take(&mut self.span_enqueue_fault);
        let mut failed = false;
        // WP-A day 42 (`DAY42.md` design S2): every span's SOURCE digest behind the producer fence,
        // before any copy, in ONE launch per 64 spans (the four-lane program over the device planes
        // the copies read). Any error from the first enqueue on quarantines the batch (rule 5).
        if let Some(scratch) = scratch.as_mut() {
            let digested = (|| -> Result<()> {
                cuda(copy.wait(&scratch.zeroed))?;
                let sources: Vec<(u64, u64)> = spans
                    .iter()
                    .map(|s| (s.source.device_ptr(&copy).0, (s.source.len() * 4) as u64))
                    .collect();
                let lanes = scratch.lanes.device_ptr(&copy).0;
                self.span_digests_on(&copy, &sources, lanes, 0, 64, SpanMemory::Device)
            })();
            failed = digested.is_err();
        }
        // WP-A day 42 (design S2, `DAY42.md` section 1a item 2): the `span-flip-landed` fault flips
        // one byte of span 0's staging after its copy and before its event, so the landing covers it.
        let mut flip = std::mem::take(&mut self.span_flip_landed) && scratch.is_some();
        let mut slots = Vec::with_capacity(spans.len());
        for mut span in spans {
            let event = if failed {
                None
            } else {
                // SAFETY: the engine owns `span` (source and destination) from here until
                // `take_d2h_spans`, which hands it back only after this span's event is observed
                // complete (`progress`); every writer of the source is on the owner stream,
                // ordered before the copy by the copy stream's wait above; `mark_landed` runs only
                // in `take_d2h_spans`, after that observation.
                #[allow(unused_mut)]
                let mut enqueued = unsafe {
                    span.destination
                        .enqueue_from_device_f32(&span.source, &copy)
                };
                #[cfg(test)]
                if fault && !slots.is_empty() {
                    fault = false;
                    enqueued = Err("injected span enqueue failure (test)".into());
                }
                let flipped = match enqueued {
                    Ok(()) => {
                        !std::mem::take(&mut flip)
                            || self.flip_staging_on(&copy, &span.destination).is_ok()
                    }
                    Err(_) => false,
                };
                flipped.then(|| copy.record_event(None).ok()).flatten()
            };
            failed |= event.is_none();
            slots.push((span, event));
        }
        let e = self.entries.get_mut(ticket).unwrap();
        e.spans = Some(SpanBatch {
            slots,
            landed: false,
            receipt: scratch,
        });
        if failed {
            e.unknown = true;
        }
        Ok(())
    }
    /// WP-A day 30: the landed spans of a D2H batch, in attach order, each destination now
    /// readable. `NotReady` until every item's AND every span's event is observed complete (the
    /// KV items' landing alone is not the batch's), `Quarantined` after a span error,
    /// `AlreadyReleased` on a second take, `Unsupported` for a batch that carries no spans. The
    /// batch cannot retire until its spans are taken. WP-A day 42 (`DAY42.md` design S2): with the
    /// batch's span receipt id; its lanes hold the source digests, complete (the batch landed after
    /// them in stream order), and nothing more is enqueued here. At most one span receipt is live:
    /// an earlier one is displaced (`abandon_span_receipt`).
    pub fn take_d2h_spans(&mut self, ticket: &TransferTicket) -> Result<TakenD2hSpans> {
        self.progress(ticket)?;
        let e = self.entries.get_mut(ticket).unwrap();
        if e.retired || e.spans_taken {
            return Err(Error::AlreadyReleased);
        }
        let Some(b) = &e.spans else {
            return Err(Error::Unsupported);
        };
        if !b.landed || !e.completion.producer_done {
            return Err(Error::NotReady);
        }
        let b = e.spans.take().unwrap();
        e.spans_taken = true;
        let lens: Vec<u64> = b
            .slots
            .iter()
            .map(|(span, _)| span.destination.len() as u64)
            .collect();
        let spans = b
            .slots
            .into_iter()
            .map(|(mut span, _)| {
                // SAFETY: `progress` observed this span's event complete (`b.landed`), recorded
                // on the copy stream after its enqueue: the bytes landed.
                unsafe { span.destination.mark_landed() };
                span
            })
            .collect();
        let receipt = match b.receipt {
            Some(scratch) => {
                self.span_receipt_seq += 1;
                let id = self.span_receipt_seq;
                let displaced = self.span_receipt.replace(SpanReceipt {
                    id,
                    scratch: Some(scratch),
                    lens,
                    sealed: false,
                    observed: false,
                });
                if let Some(r) = displaced {
                    self.abandon_span_receipt(r);
                }
                Some(SpanReceiptId(id))
            }
            None => None,
        };
        self.reap_span_receipts();
        Ok(TakenD2hSpans { spans, receipt })
    }
    /// WP-A day 42 (`DAY42.md` design S2): enqueue the LANDED digests of the live span receipt
    /// `id` on the copy stream: ONE launch per 64 spans over each span's pinned staging through its
    /// device address (`staging`, the take's destinations in attach order, each its span's length),
    /// the lanes' D2H into the twin, and the receipt event. The caller keeps every staging buffer
    /// alive and unreused until `d2h_span_receipt` (or its wait) has observed the receipt; no other
    /// writer of a staging buffer may run meanwhile. Refusals before any enqueue: an unknown or
    /// displaced id (`UnknownTicket`), a second seal (`Busy`), a count or length that is not the
    /// take's (`InvalidLayout`), a staging buffer without a device address (`Quarantined`). From
    /// the first enqueue on, an error leaves the receipt sealed and never observable
    /// (`Quarantined` at its read; a leak at the engine's drop).
    pub fn seal_d2h_span_receipt(
        &mut self,
        id: SpanReceiptId,
        staging: &[&PinnedHostBuf],
    ) -> Result<()> {
        self.check_thread()?;
        let copy = self.copy.clone().ok_or(Error::Unsupported)?;
        let r = match self.span_receipt.as_ref() {
            Some(r) if r.id == id.0 => r,
            _ => return Err(Error::UnknownTicket),
        };
        if r.sealed {
            return Err(Error::Busy);
        }
        if staging.len() != r.lens.len()
            || staging
                .iter()
                .zip(&r.lens)
                .any(|(b, &n)| b.len() as u64 != n)
        {
            return Err(Error::InvalidLayout);
        }
        let landed = staging
            .iter()
            .map(|b| {
                b.device_address()
                    .map(|p| (p, b.len() as u64))
                    .map_err(|_| Error::Quarantined)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut r = self.span_receipt.take().unwrap();
        r.sealed = true;
        let scratch = r.scratch.as_mut().unwrap();
        let sealed = (|| -> Result<()> {
            let lanes = scratch.lanes.device_ptr(&copy).0;
            self.span_digests_on(&copy, &landed, lanes, 32, 64, SpanMemory::PinnedHost)?;
            cuda(copy.memcpy_dtoh(&scratch.lanes, &mut scratch.pinned))?;
            scratch.event = Some(cuda(copy.record_event(None))?);
            Ok(())
        })();
        self.span_receipt = Some(r);
        sealed.map_err(|_| Error::Quarantined)
    }
    /// WP-A day 42 (design S2): the live span receipt `id`'s pairs, per span in attach order the
    /// (source, landed) four-lane digests, once its event is observed (`Ok(None)` while pending,
    /// never a host wait). The receipt then leaves the engine (its twin back to the pool). `Busy`
    /// before its seal, `UnknownTicket` for an unknown or displaced id, `Quarantined` for a seal
    /// that failed or an event error.
    pub fn d2h_span_receipt(&mut self, id: SpanReceiptId) -> Result<Option<Vec<(Digest, Digest)>>> {
        self.check_thread()?;
        self.reap_span_receipts();
        let r = match self.span_receipt.as_ref() {
            Some(r) if r.id == id.0 => r,
            _ => return Err(Error::UnknownTicket),
        };
        if !r.sealed {
            return Err(Error::Busy);
        }
        let event = r
            .scratch
            .as_ref()
            .and_then(|s| s.event.as_ref())
            .ok_or(Error::Quarantined)?;
        if !event_done(event)? {
            return Ok(None);
        }
        let mut r = self.span_receipt.take().unwrap();
        r.observed = true;
        let scratch = r.scratch.take().unwrap();
        let lanes = cuda(scratch.pinned.as_slice())?.to_vec();
        // Every write to the twin was observed (the receipt event); back to the pool (G'').
        self.twin_pool.borrow_mut().push(scratch.pinned);
        Ok(Some(
            r.lens
                .iter()
                .enumerate()
                .map(|(k, &n)| {
                    (
                        receipt_digest_from_lanes(lanes_at(&lanes, 64 * k), n),
                        receipt_digest_from_lanes(lanes_at(&lanes, 64 * k + 32), n),
                    )
                })
                .collect(),
        ))
    }
    /// WP-A day 42 (design S2): `d2h_span_receipt` after a host wait on the receipt's event (a
    /// `Block` settle).
    pub fn d2h_span_receipt_wait(&mut self, id: SpanReceiptId) -> Result<Vec<(Digest, Digest)>> {
        self.check_thread()?;
        match self.span_receipt.as_ref() {
            Some(r) if r.id == id.0 && r.sealed => {
                let event = r
                    .scratch
                    .as_ref()
                    .and_then(|s| s.event.as_ref())
                    .ok_or(Error::Quarantined)?;
                cuda(event.synchronize())?;
            }
            Some(r) if r.id == id.0 => return Err(Error::Busy),
            _ => return Err(Error::UnknownTicket),
        }
        self.d2h_span_receipt(id)?.ok_or(Error::Quarantined)
    }
    /// WP-A day 42 (design S2): the caller gives up the live span receipt `id` (a demote that ends
    /// without publishing). Unknown ids are a no-op (the receipt was read or displaced already).
    pub fn d2h_span_receipt_abandon(&mut self, id: SpanReceiptId) {
        if self.check_thread().is_err() {
            return;
        }
        if self.span_receipt.as_ref().is_some_and(|r| r.id == id.0) {
            let r = self.span_receipt.take().unwrap();
            self.abandon_span_receipt(r);
        }
        self.reap_span_receipts();
    }
    /// An unsealed receipt drops at once (its lanes are complete and nothing else was enqueued on
    /// them; its twin was never written by the device and goes back to the pool); a sealed one
    /// waits in the reap list until its event is observed.
    fn abandon_span_receipt(&mut self, mut r: SpanReceipt) {
        if r.sealed {
            self.span_reap.push(r);
        } else if let Some(scratch) = r.scratch.take() {
            self.twin_pool.borrow_mut().push(scratch.pinned);
        }
    }
    /// Free every reaped receipt whose event is observed complete (its lanes and twin written for
    /// the last time); one whose event is pending or failed stays (a leak at the engine's drop).
    fn reap_span_receipts(&mut self) {
        let mut kept = Vec::with_capacity(self.span_reap.len());
        for mut r in std::mem::take(&mut self.span_reap) {
            let done = r
                .scratch
                .as_ref()
                .and_then(|s| s.event.as_ref())
                .map(|e| event_done(e).unwrap_or(false))
                .unwrap_or(false);
            if done {
                r.observed = true;
                if let Some(scratch) = r.scratch.take() {
                    self.twin_pool.borrow_mut().push(scratch.pinned);
                }
            } else {
                kept.push(r);
            }
        }
        self.span_reap = kept;
    }
    /// WP-A day 32 (memra#536 Move 2 owed item 1, the H2D half; `memra_tier::conformance::
    /// h2d_span_batch`): attach typed f32 spans to a live H2D batch right after its submission.
    /// The copy stream waits on a fresh owner-stream event (every destination was allocated on the
    /// owner stream, stream-ordered, so the allocation is ordered before the copy), then per span
    /// one full-length `cuMemcpyHtoDAsync` from the fully written pinned source with no host wait
    /// and one completion event. From here the engine OWNS each source and destination until
    /// `take_h2d_spans`: the staging must not be rewritten or freed while the copy reads it, and the
    /// destination must not be read or freed while the copy writes it (the engine context runs
    /// without cudarc's event tracking). Rule 1, refusal before enqueue is whole: no copy stream
    /// (`Unsupported`), an unknown, retired, cancelled or quarantined ticket, a ticket that is not
    /// an H2D batch (`Unsupported`), one that already carries spans of either class (`Busy`), an
    /// empty batch, a zero-length span, a source whose length is not the destination's or a source
    /// not fully written (`InvalidLayout`), a destination of another stream (`WrongOwner`), or a
    /// failed owner-stream record or copy-stream wait: every span comes back, nothing enqueued,
    /// nothing quarantined. Rule 5: an enqueue or event error from the first enqueue on keeps every
    /// span and quarantines the ticket (`Ok`: the batch was accepted and cannot land).
    #[allow(clippy::result_large_err)]
    pub fn submit_h2d_spans(
        &mut self,
        ticket: &TransferTicket,
        spans: Vec<H2dSpan>,
    ) -> std::result::Result<(), (Error, Vec<H2dSpan>)> {
        self.attach_h2d_spans(ticket, spans, None)
            .map_err(|(e, spans, _)| (e, spans))
    }
    /// WP-A day 33 (`memra_tier::conformance::h2d_span_fill_ordered_before_its_copy`, `DAY33.md`
    /// design F): `submit_h2d_spans` for spans whose staging sources are FILLED on the copy stream.
    /// `fills[k]` is the resident plane of span `k`, exactly its source's length. After the same
    /// rule-1 admission (except that a source need not be written yet) and the copy stream's wait
    /// on a fresh owner-stream event, ONE host function (`cuLaunchHostFunc`) on the copy stream
    /// copies every plane into its span's staging buffer, then each span's copy and event follow in
    /// stream order, so no copy runs before the fill and a span's event observed complete implies
    /// the fill ran. The owner thread waits for nothing. A launch error is a refusal before any copy
    /// (every span and fill back); an enqueue or event error after the first copy quarantines, as
    /// rule 5. `take_h2d_spans` marks each source written.
    #[allow(clippy::result_large_err, clippy::type_complexity)]
    pub fn submit_h2d_spans_filled(
        &mut self,
        ticket: &TransferTicket,
        spans: Vec<H2dSpan>,
        fills: Vec<Arc<Vec<f32>>>,
    ) -> std::result::Result<(), (Error, Vec<H2dSpan>, Vec<Arc<Vec<f32>>>)> {
        self.attach_h2d_spans(ticket, spans, Some(fills))
            .map_err(|(e, spans, fills)| (e, spans, fills.unwrap_or_default()))
    }
    #[allow(clippy::result_large_err, clippy::type_complexity)]
    fn attach_h2d_spans(
        &mut self,
        ticket: &TransferTicket,
        mut spans: Vec<H2dSpan>,
        fills: Option<Vec<Arc<Vec<f32>>>>,
    ) -> std::result::Result<(), (Error, Vec<H2dSpan>, Option<Vec<Arc<Vec<f32>>>>)> {
        let filled = fills.is_some();
        let admitted = (|| {
            self.check_thread()?;
            let copy = self.copy.clone().ok_or(Error::Unsupported)?;
            let e = self.entries.get(ticket).ok_or(Error::UnknownTicket)?;
            if e.retired {
                return Err(Error::AlreadyReleased);
            }
            if e.cancelled {
                return Err(Error::Cancelled);
            }
            if e.unknown {
                return Err(Error::Quarantined);
            }
            if e.h2d_spans.is_some() || e.h2d_spans_taken || e.spans.is_some() || e.spans_taken {
                return Err(Error::Busy);
            }
            if e.receipt.is_some()
                || e.items
                    .iter()
                    .flatten()
                    .any(|i| i.direction != CopyDirection::HostToDevice)
            {
                return Err(Error::Unsupported);
            }
            if spans.is_empty() {
                return Err(Error::EmptyBatch);
            }
            for (k, s) in spans.iter().enumerate() {
                let bytes = s.destination.len().checked_mul(4).ok_or(Error::Overflow)?;
                if bytes == 0 || s.source.len() != bytes || !(filled || s.source.is_written()) {
                    return Err(Error::InvalidLayout);
                }
                // Day 33: each fill is exactly its source's length (the host function's copy).
                if let Some(fills) = &fills
                    && (fills.len() != spans.len() || fills[k].len().checked_mul(4) != Some(bytes))
                {
                    return Err(Error::InvalidLayout);
                }
                if !Arc::ptr_eq(s.destination.stream(), &self.stream) {
                    return Err(Error::WrongOwner);
                }
            }
            // WP-A day 40 (design S): the destination digests' scratch, allocated before anything
            // is enqueued.
            let scratch =
                match self.receipt {
                    Some(_) => Some(self.receipt_scratch_bytes(
                        spans.len().checked_mul(32).ok_or(Error::Overflow)?,
                    )?),
                    None => None,
                };
            let fence = cuda(self.stream.record_event(None))?;
            cuda(copy.wait(&fence))?;
            Ok((copy, scratch))
        })();
        let (copy, mut scratch) = match admitted {
            Ok(admitted) => admitted,
            Err(error) => return Err((error, spans, fills)),
        };
        // WP-A day 64 (`DAY64.md` section 4 step 1, log only): a timing event at each phase
        // boundary of the span work (the start, after the fill, after the copies, after the seal).
        let timed = |copy: &Arc<CudaStream>| {
            copy.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .ok()
        };
        let mut timing: Vec<CudaEvent> = timed(&copy).into_iter().collect();
        // Day 33 (design F): the fill, ONE host function on the copy stream ahead of every copy;
        // day 39 (design T): split inside it across the engine's fill threads.
        if let Some(fills) = fills {
            let task = Box::new(SpanFillTask {
                items: fills
                    .into_iter()
                    .zip(spans.iter_mut())
                    .map(|(plane, span)| {
                        let dst = span.source.fill_target();
                        (plane, dst)
                    })
                    .collect(),
                threads: self.fill_threads,
            });
            let raw = Box::into_raw(task);
            // SAFETY: `span_fill_on_copy_stream` takes the box back exactly once, when the copy
            // stream reaches it; each target is a staging buffer the engine owns from here until
            // `take_h2d_spans`, which runs only after that span's event, which stream order puts
            // after this host function; nobody else reads or writes a target in between, and each
            // plane is an owned `Arc` the task holds. On a launch error the driver did not take the
            // box, so it is taken back here.
            let launched = unsafe {
                result::stream::launch_host_function(
                    copy.cu_stream(),
                    span_fill_on_copy_stream,
                    raw.cast(),
                )
            };
            if let Err(e) = launched {
                // SAFETY: the launch failed, so the driver holds no reference to `raw`.
                let task = unsafe { Box::from_raw(raw) };
                let fills = task.items.into_iter().map(|(plane, _)| plane).collect();
                return Err((cuda::<()>(Err(e)).unwrap_err(), spans, Some(fills)));
            }
        }
        timing.extend(timed(&copy));
        #[cfg(test)]
        let mut fault = std::mem::take(&mut self.span_enqueue_fault);
        let mut failed = false;
        let mut slots = Vec::with_capacity(spans.len());
        for mut span in spans {
            let event = if failed {
                None
            } else {
                // SAFETY: the engine owns `span` (source and destination) from here until
                // `take_h2d_spans`, which hands it back only after this span's event is observed
                // complete (`progress`) and the owner stream's wait on it is installed
                // (`install_consumer_wait`); the source was fully written before the attach (checked
                // above) and nobody writes it while the engine owns it; the destination's allocation
                // is on the owner stream, ordered before the copy by the copy stream's wait above.
                #[allow(unused_mut)]
                let mut enqueued = unsafe {
                    if filled {
                        // Day 33: the source is written by the host function ahead of this copy
                        // in stream order, not yet on the host's side.
                        span.source
                            .enqueue_to_device_f32_after_fill(&mut span.destination, &copy)
                    } else {
                        span.source
                            .enqueue_to_device_f32(&mut span.destination, &copy)
                    }
                };
                #[cfg(test)]
                if fault && !slots.is_empty() {
                    fault = false;
                    enqueued = Err("injected span enqueue failure (test)".into());
                }
                enqueued.ok().and_then(|()| copy.record_event(None).ok())
            };
            failed |= event.is_none();
            slots.push((span, event));
        }
        timing.extend(timed(&copy));
        // WP-A day 40 (design S): after every copy, each span's DESTINATION digest over its device
        // plane (design S2: ONE launch per 64 spans), then one D2H of the lanes into the twin and
        // the span receipt's event: the batch lands only with it.
        if !failed && let Some(scratch) = scratch.as_mut() {
            let sealed = (|| -> Result<()> {
                cuda(copy.wait(&scratch.zeroed))?;
                let destinations: Vec<(u64, u64)> = slots
                    .iter()
                    .map(|(span, _)| {
                        (
                            span.destination.device_ptr(&copy).0,
                            (span.destination.len() * 4) as u64,
                        )
                    })
                    .collect();
                let lanes = scratch.lanes.device_ptr(&copy).0;
                self.span_digests_on(&copy, &destinations, lanes, 0, 32, SpanMemory::Device)?;
                cuda(copy.memcpy_dtoh(&scratch.lanes, &mut scratch.pinned))?;
                scratch.event = Some(cuda(copy.record_event(None))?);
                Ok(())
            })();
            failed |= sealed.is_err();
        }
        timing.extend(timed(&copy));
        let e = self.entries.get_mut(ticket).unwrap();
        e.h2d_spans = Some(H2dSpanBatch {
            slots,
            landed: false,
            fenced: false,
            receipt: scratch,
            timing,
        });
        if failed {
            e.unknown = true;
        }
        Ok(())
    }
    /// WP-A day 34 (`memra_tier::conformance::h2d_deferred_checksum_lands_with_its_digests`,
    /// `DAY34.md` design K): defer this H2D batch's completion checksums to the caller, right after
    /// the batch's submission. One read-only view per accepted item (its index and its whole host
    /// source, the bytes `progress` would hash); from here an item lands only with the checksum
    /// `supply_h2d_checksums` hands back, and the sources stay owned while a view is out.
    /// Refused: an unknown, retired, cancelled or quarantined ticket, a ticket that is not an H2D
    /// batch or carries a D2D receipt (`Unsupported`), one already deferred or already hashed by
    /// `progress` (`Busy`).
    pub fn defer_h2d_checksums(&mut self, ticket: &TransferTicket) -> Result<Vec<H2dSourceView>> {
        self.check_thread()?;
        let e = self.entries.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        if e.retired {
            return Err(Error::AlreadyReleased);
        }
        if e.cancelled {
            return Err(Error::Cancelled);
        }
        if e.unknown {
            return Err(Error::Quarantined);
        }
        if e.receipt.is_some()
            || e.items
                .iter()
                .flatten()
                .any(|i| i.direction != CopyDirection::HostToDevice)
        {
            return Err(Error::Unsupported);
        }
        if e.deferred.is_some()
            || e.completion
                .items
                .iter()
                .any(|i| i.segments.iter().any(|s| s.checksum.is_some()))
        {
            return Err(Error::Busy);
        }
        let mut views = Vec::new();
        for (i, item) in e.items.iter().enumerate() {
            let Some(item) = item else {
                continue;
            };
            // No host wait: `bytes()` would wait for this item's copy (its tracking event is recorded
            // after the copy); the view only reads, as the copy does.
            let (ptr, len) = item
                .host
                .as_ref()
                .and_then(|h| h.allocation.backing.as_ref())
                .ok_or(Error::AlreadyReleased)?
                .raw_view();
            views.push(H2dSourceView {
                item: i as u32,
                ptr,
                len,
            });
        }
        e.deferred = Some(DeferredSums {
            out: views.len(),
            supplied: vec![None; e.items.len()],
        });
        Ok(views)
    }
    /// WP-A day 34: the views back with their digests; each digest becomes its item's completion
    /// checksum and expectation, the assignment `progress` makes for an undeferred item, so the
    /// caller's `Completion::require` against the demote-time receipts is unchanged. A view of
    /// another ticket or item (`WrongOwner`), a second return (`AlreadyReleased`) or a ticket with
    /// nothing deferred (`Unsupported`) is refused before any digest is taken.
    pub fn supply_h2d_checksums(
        &mut self,
        ticket: &TransferTicket,
        digests: Vec<(H2dSourceView, Digest)>,
    ) -> Result<()> {
        self.check_thread()?;
        let e = self.entries.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        let d = e.deferred.as_mut().ok_or(Error::Unsupported)?;
        for (view, _) in &digests {
            let i = view.item as usize;
            let item = e
                .items
                .get(i)
                .and_then(Option::as_ref)
                .ok_or(Error::WrongOwner)?;
            let (ptr, len) = item
                .host
                .as_ref()
                .and_then(|h| h.allocation.backing.as_ref())
                .ok_or(Error::AlreadyReleased)?
                .raw_view();
            if ptr != view.ptr || len != view.len {
                return Err(Error::WrongOwner);
            }
            if d.supplied[i].is_some() {
                return Err(Error::AlreadyReleased);
            }
        }
        for (view, sum) in digests {
            d.supplied[view.item as usize] = Some(sum);
            d.out -= 1;
        }
        Ok(())
    }
    /// WP-A day 32: the landed spans of an H2D batch, in attach order, each destination readable
    /// on the owner stream. `NotReady` until every item's AND every span's event is observed
    /// complete AND the owner stream's wait on every span event is installed
    /// (`install_consumer_wait`; rule 3 of `h2d_span_batch`: landing is not a fence),
    /// `Quarantined` after a span error, `AlreadyReleased` on a second take, `Unsupported` for a
    /// batch that carries no H2D spans. The batch cannot retire until its spans are taken.
    /// WP-A day 64 (`DAY64.md` step 1, log only): which parts of an H2D batch's landing have been
    /// observed, read from their events without changing any state: (the items' and spans' copies,
    /// the spans' destination-digest receipt; `true` where there is none). No decision reads it.
    pub fn h2d_landing_parts(&self, ticket: &TransferTicket) -> Result<(bool, bool)> {
        self.check_thread()?;
        let e = self.entries.get(ticket).ok_or(Error::UnknownTicket)?;
        let done = |ev: Option<&CudaEvent>| ev.ok_or(Error::Quarantined).and_then(event_done);
        let mut copies = true;
        for item in e.items.iter().flatten() {
            copies &= done(item.event.as_ref())?;
        }
        let mut receipt = true;
        if let Some(b) = &e.h2d_spans
            && !b.landed
        {
            for (_, event) in &b.slots {
                copies &= done(event.as_ref())?;
            }
            if let Some(r) = &b.receipt {
                receipt = done(r.event.as_ref())?;
            }
        }
        Ok((copies, receipt))
    }
    /// WP-A day 64 (`DAY64.md` section 4 step 1, log only): the span work's (fill, copies, digests
    /// and the lanes' D2H) elapsed milliseconds from its four timing events; `None` unless every
    /// event is complete (never a wait) or when the batch carries no timing.
    pub fn h2d_span_timing(&self, ticket: &TransferTicket) -> Option<(f32, f32, f32)> {
        let b = self.entries.get(ticket)?.h2d_spans.as_ref()?;
        let [t0, fill, copies, sealed] = b.timing.as_slice() else {
            return None;
        };
        for e in [t0, fill, copies, sealed] {
            if !event_done(e).ok()? {
                return None;
            }
        }
        Some((
            t0.elapsed_ms(fill).ok()?,
            fill.elapsed_ms(copies).ok()?,
            copies.elapsed_ms(sealed).ok()?,
        ))
    }
    pub fn take_h2d_spans(&mut self, ticket: &TransferTicket) -> Result<Vec<LandedH2dSpan>> {
        self.progress(ticket)?;
        let e = self.entries.get_mut(ticket).unwrap();
        if e.retired || e.h2d_spans_taken {
            return Err(Error::AlreadyReleased);
        }
        let Some(b) = &e.h2d_spans else {
            return Err(Error::Unsupported);
        };
        if !b.landed || !e.completion.producer_done || !b.fenced {
            return Err(Error::NotReady);
        }
        // WP-A day 40 (design S): the destination digests, readable since `progress` observed the
        // receipt event.
        let lanes = match &b.receipt {
            Some(r) => Some(cuda(r.pinned.as_slice())?.to_vec()),
            None => None,
        };
        let b = e.h2d_spans.take().unwrap();
        e.h2d_spans_taken = true;
        if let Some(r) = b.receipt {
            self.twin_pool.borrow_mut().push(r.pinned);
        }
        Ok(b.slots
            .into_iter()
            .enumerate()
            .map(|(k, (mut span, _))| {
                // SAFETY: `progress` observed this span's event complete, recorded after its copy,
                // which stream order puts after its fill (day 33) when it had one: the source's bytes
                // are written (and, without a fill, were already).
                unsafe { span.source.mark_landed() };
                let n = (span.destination.len() * 4) as u64;
                LandedH2dSpan {
                    destination_digest: lanes
                        .as_ref()
                        .map(|l| receipt_digest_from_lanes(lanes_at(l, 32 * k), n)),
                    span,
                }
            })
            .collect())
    }
    fn validate(&self, op: &TransferOp<CudaPinnedLease>, epochs: Epochs) -> Result<()> {
        let (o, direction) = match op {
            TransferOp::H2d(o) => (o, CopyDirection::HostToDevice),
            TransferOp::D2h(o) => (o, CopyDirection::DeviceToHost),
            _ => return Err(Error::Unsupported),
        };
        o.validate(direction, epochs)?;
        // The frozen host contract has no subrange mutation: this implementation
        // accepts exactly the whole initialized logical host range, not a short prefix.
        if direction == CopyDirection::DeviceToHost && Rc::strong_count(&o.host.allocation) != 1 {
            return Err(Error::Busy);
        }
        if o.bytes != o.host.valid_bytes() {
            return Err(Error::InvalidLayout);
        }
        let backing = self.owner.resolve::<Rc<RefCell<KvPlane>>>(&o.device)?;
        if !Arc::ptr_eq(backing.borrow().stream(), &self.stream)
            || !Arc::ptr_eq(
                o.host
                    .allocation
                    .backing
                    .as_ref()
                    .ok_or(Error::AlreadyReleased)?
                    .context(),
                self.stream.context(),
            )
        {
            return Err(Error::WrongOwner);
        }
        if let Some(f) = o.producer_fence
            && self
                .producers
                .get(&f.sequence)
                .is_none_or(|(actual, _)| actual != &f)
        {
            return Err(Error::WrongOwner);
        }
        Ok(())
    }
    fn progress(&mut self, ticket: &TransferTicket) -> Result<()> {
        self.check_thread()?;
        let e = self.entries.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        if e.unknown {
            return Err(Error::Quarantined);
        }
        // WP-A day 22 (Move 2 slice 3): a D2D batch's receipt lanes, readable once the receipt
        // event (recorded after the lanes' D2H, after every item's event) is observed complete.
        let receipt_lanes: Option<Vec<u8>> = match &e.receipt {
            None => None,
            Some(r) => {
                let sealed = r
                    .event
                    .as_ref()
                    .ok_or(Error::Quarantined)
                    .and_then(event_done);
                match sealed {
                    Ok(true) => match r.pinned.as_slice() {
                        Ok(lanes) => Some(lanes.to_vec()),
                        Err(_) => {
                            e.unknown = true;
                            return Err(Error::Quarantined);
                        }
                    },
                    Ok(false) => None,
                    Err(err) => {
                        e.unknown = true;
                        return Err(err);
                    }
                }
            }
        };
        // WP-A day 38 (design G): a D2H batch's device receipt, readable once its receipt event
        // (recorded after the digests' D2H, before any copy) is observed complete.
        let d2h_lanes: Option<Vec<u8>> = match &e.d2h_receipt {
            None => None,
            Some(r) => {
                let sealed = r
                    .scratch
                    .event
                    .as_ref()
                    .ok_or(Error::Quarantined)
                    .and_then(event_done);
                // Design G''': the `d2h-delay` fault's host-side hold keeps the batch unlanded.
                let held = r.hold_until.is_some_and(|h| std::time::Instant::now() < h);
                match sealed {
                    Ok(true) if held => None,
                    Ok(true) => match r.scratch.pinned.as_slice() {
                        Ok(lanes) => Some(lanes.to_vec()),
                        Err(_) => {
                            e.unknown = true;
                            return Err(Error::Quarantined);
                        }
                    },
                    Ok(false) => None,
                    Err(err) => {
                        e.unknown = true;
                        return Err(err);
                    }
                }
            }
        };
        for (i, item) in e.items.iter_mut().enumerate() {
            let Some(item) = item else {
                continue;
            };
            let s = &mut e.completion.items[i].segments[0];
            if s.producer_done {
                continue;
            }
            let done = item
                .event
                .as_ref()
                .ok_or(Error::Quarantined)
                .and_then(event_done);
            match done {
                Ok(false) => continue,
                Err(err) => {
                    e.unknown = true;
                    return Err(err);
                }
                Ok(true) => (),
            }
            if s.status == ItemStatus::Quarantined {
                e.unknown = true;
                return Err(Error::Quarantined);
            }
            if item.direction == CopyDirection::DeviceToDevice {
                // WP-A day 22 (Move 2 slice 3, `d2d_receipt_witnessed`): a D2D item lands WITH its
                // receipt or not yet. The lanes are read after the receipt event; the destination
                // digest becomes the item's checksum and the source digest its expectation, so
                // the gate clause `s.checksum != Some(e.checksum) => Corrupt` is the comparison.
                // A batch without lanes (no scratch) stays receipt-less and refused by the same
                // clause. Publication is still the caller's, after `capture_landed`.
                let Some(lanes) = &receipt_lanes else {
                    continue;
                };
                let off = 64 * i;
                if lanes.len() >= off + 64 {
                    let take = |o: usize| -> [u64; 4] {
                        let mut l = [0u64; 4];
                        for (k, lane) in l.iter_mut().enumerate() {
                            let at = o + 8 * k;
                            *lane = u64::from_le_bytes(lanes[at..at + 8].try_into().unwrap());
                        }
                        l
                    };
                    e.expected[i][0].checksum = receipt_digest_from_lanes(take(off), item.bytes);
                    s.checksum = Some(receipt_digest_from_lanes(take(off + 32), item.bytes));
                } else {
                    s.checksum = None;
                }
                s.producer_done = true;
                s.status = ItemStatus::Complete;
                s.valid_bytes = item.bytes;
                continue;
            }
            // WP-A day 38 (`d2h_device_receipt` rules 1 and 2): a device-receipt D2H item's copy
            // landed; it lands with its receipt observed or not yet, and its checksum and
            // expectation are its SOURCE's digest (the program `checksum` on the copy stream). The
            // owner thread hashes nothing; the caller's re-hash of the landed bytes before any
            // publication is the witness (rule 3, the bind).
            if e.d2h_receipt.is_some() && item.direction == CopyDirection::DeviceToHost {
                let Some(lanes) = &d2h_lanes else {
                    continue;
                };
                let off = 32 * i;
                let Some(sum) = lanes.get(off..off + 32) else {
                    e.unknown = true;
                    return Err(Error::Quarantined);
                };
                let mut digest: Digest = [0; 32];
                digest.copy_from_slice(sum);
                s.producer_done = true;
                s.status = ItemStatus::Complete;
                s.valid_bytes = item.bytes;
                s.checksum = Some(digest);
                e.expected[i][0].checksum = digest;
                continue;
            }
            // WP-A day 34 (`h2d_deferred_checksum` rule 1): a deferred H2D item's copy landed; the
            // item lands with its supplied checksum (the caller's hash helper, same program, same
            // bytes) or not yet. Nothing is hashed here.
            if let Some(d) = &e.deferred
                && item.direction == CopyDirection::HostToDevice
            {
                s.status = ItemStatus::Complete;
                s.valid_bytes = item.bytes;
                if let Some(sum) = d.supplied[i] {
                    s.checksum = Some(sum);
                    e.expected[i][0].checksum = sum;
                    s.producer_done = true;
                }
                continue;
            }
            s.producer_done = true;
            s.status = ItemStatus::Complete;
            s.valid_bytes = item.bytes;
            s.checksum = Some(checksum(
                item.host.as_ref().ok_or(Error::AlreadyReleased)?.bytes()?,
            ));
            e.expected[i][0].checksum = s.checksum.unwrap();
        }
        // WP-A day 30 (`d2h_span_batch` rule 2): a batch with spans is producer-done only when
        // every span's event is observed complete too; an event error quarantines the batch.
        let spans_landed = match &mut e.spans {
            None => true,
            Some(b) if b.landed => true,
            Some(b) => {
                let mut all = true;
                for (_, event) in &b.slots {
                    match event
                        .as_ref()
                        .ok_or(Error::Quarantined)
                        .and_then(event_done)
                    {
                        Ok(true) => (),
                        Ok(false) => all = false,
                        Err(err) => {
                            e.unknown = true;
                            return Err(err);
                        }
                    }
                }
                b.landed = all;
                all
            }
        };
        // WP-A day 32 (`h2d_span_batch` rule 2): the same fold over an H2D batch's spans.
        let h2d_spans_landed = match &mut e.h2d_spans {
            None => true,
            Some(b) if b.landed => true,
            Some(b) => {
                let mut all = true;
                for (_, event) in &b.slots {
                    match event
                        .as_ref()
                        .ok_or(Error::Quarantined)
                        .and_then(event_done)
                    {
                        Ok(true) => (),
                        Ok(false) => all = false,
                        Err(err) => {
                            e.unknown = true;
                            return Err(err);
                        }
                    }
                }
                // WP-A day 40 (design S): the H2D spans' receipt event is part of their landing
                // (a D2H span batch lands with its copies since design S2; its receipt is read
                // after the take, `d2h_span_receipt`).
                if all && let Some(r) = &b.receipt {
                    match r
                        .event
                        .as_ref()
                        .ok_or(Error::Quarantined)
                        .and_then(event_done)
                    {
                        Ok(true) => (),
                        Ok(false) => all = false,
                        Err(err) => {
                            e.unknown = true;
                            return Err(err);
                        }
                    }
                }
                b.landed = all;
                all
            }
        };
        e.completion.producer_done = spans_landed
            && h2d_spans_landed
            && e.completion
                .items
                .iter()
                .filter(|i| i.accepted)
                .all(|i| i.segments.iter().all(|s| s.producer_done));
        e.completion.consumer_fenced = e
            .completion
            .items
            .iter()
            .filter(|i| i.accepted)
            .all(|i| i.segments.iter().all(|s| s.consumer_fenced));
        Ok(())
    }
    fn publishable(&mut self, ticket: &TransferTicket, current: Epochs) -> Result<()> {
        ticket.epochs.require(current)?;
        let e = self.entries.get(ticket).ok_or(Error::UnknownTicket)?;
        if e.cancelled {
            return Err(Error::Cancelled);
        }
        if e.retired {
            return Err(Error::AlreadyReleased);
        }
        self.progress(ticket)?;
        let e = &self.entries[ticket];
        e.completion.require(ticket, &e.expected, true)
    }
}
fn epochs(op: &TransferOp<CudaPinnedLease>) -> Epochs {
    match op {
        TransferOp::H2d(o) | TransferOp::D2h(o) => o.epochs,
        TransferOp::P2p(o) => o.epochs,
        TransferOp::NvmeRead(o) => o.epochs,
    }
}
#[allow(clippy::result_large_err)]
impl TransferEngine for CudaTransfers {
    type Host = CudaPinnedLease;
    fn h2d(
        &mut self,
        op: CopyOp<Self::Host>,
    ) -> std::result::Result<TransferTicket, Rejected<CopyOp<Self::Host>>> {
        match self.submit_batch(vec![TransferOp::H2d(op)]) {
            Ok(b) => Ok(b.ticket),
            Err(r) => {
                let TransferOp::H2d(op) = r.op.into_iter().next().unwrap() else {
                    unreachable!()
                };
                Err(Rejected { op, error: r.error })
            }
        }
    }
    fn d2h(
        &mut self,
        op: CopyOp<Self::Host>,
    ) -> std::result::Result<TransferTicket, Rejected<CopyOp<Self::Host>>> {
        match self.submit_batch(vec![TransferOp::D2h(op)]) {
            Ok(b) => Ok(b.ticket),
            Err(r) => {
                let TransferOp::D2h(op) = r.op.into_iter().next().unwrap() else {
                    unreachable!()
                };
                Err(Rejected { op, error: r.error })
            }
        }
    }
    fn p2p(
        &mut self,
        op: ContiguousCopy,
    ) -> std::result::Result<TransferTicket, Rejected<ContiguousCopy>> {
        Err(Rejected {
            op,
            error: Error::Unsupported,
        })
    }
    fn nvme_read(
        &mut self,
        op: ReadPlan<Self::Host>,
    ) -> std::result::Result<TransferTicket, Rejected<ReadPlan<Self::Host>>> {
        Err(Rejected {
            op,
            error: Error::Unsupported,
        })
    }
    fn submit_batch(
        &mut self,
        ops: Vec<TransferOp<Self::Host>>,
    ) -> Submission<TransferOp<Self::Host>> {
        let admission = (|| {
            self.check_thread()?;
            let first = epochs(ops.first().ok_or(Error::EmptyBatch)?);
            if ops.len() > u32::MAX as usize {
                return Err(Error::Overflow);
            }
            if ops.iter().any(|op| epochs(op) != first) {
                return Err(Error::StaleEpoch);
            }
            if self.sequence == u64::MAX
                || self.fence_sequence.checked_add(ops.len() as u64).is_none()
            {
                return Err(Error::Overflow);
            }
            let errors: Vec<_> = ops
                .iter()
                .map(|op| self.validate(op, first).err())
                .collect();
            if errors.iter().all(Option::is_some) {
                return Err(errors[0].clone().unwrap());
            }
            // WP-A day 38 (`DAY38.md` design G'): a D2H batch on the copy stream takes its receipt
            // on the device. Its scratch is allocated here, before the batch is charged, so a
            // refusal submits nothing and hands every op back.
            let device_receipt = self.d2h_receipts_on_device()
                && ops
                    .iter()
                    .zip(&errors)
                    .any(|(op, e)| e.is_none() && matches!(op, TransferOp::D2h(_)));
            let scratch = if device_receipt {
                Some(self.receipt_scratch_bytes(ops.len().checked_mul(32).ok_or(Error::Overflow)?)?)
            } else {
                None
            };
            let mut request = self.request();
            request.bytes.inflight = ops.len() as u64;
            let charge = self.governor.borrow_mut().reserve(&request)?;
            Ok((first, errors, charge, scratch))
        })();
        let (epochs, errors, charge, scratch) = match admission {
            Ok(v) => v,
            Err(error) => return Err(Rejected { op: ops, error }),
        };
        self.sequence += 1;
        let ticket = TransferTicket {
            issuer: self.owner.issuer(),
            sequence: self.sequence,
            epochs,
        };
        let mut entry = Entry {
            items: vec![],
            completion: Completion {
                ticket,
                items: vec![],
                producer_done: false,
                consumer_fenced: false,
            },
            expected: vec![],
            charge: Some(charge),
            cancelled: false,
            published: false,
            retired: false,
            unknown: false,
            source: Retention::default(),
            destination: Retention::default(),
            receipt: None,
            spans: None,
            spans_taken: false,
            h2d_spans: None,
            h2d_spans_taken: false,
            deferred: None,
            d2h_receipt: None,
        };
        // WP-A day 38 (design G4): the receipt digests every accepted D2H item's DEVICE source on
        // the copy stream behind the producer fences, ahead of the copies below on the same stream.
        // Any error from the first enqueue on may leave work in flight: the batch is accepted and
        // quarantined, never handed back.
        if let Some(scratch) = scratch {
            match self.seal_d2h_device_receipt(&ops, &errors, scratch) {
                Ok(receipt) => entry.d2h_receipt = Some(receipt),
                Err(_) => entry.unknown = true,
            }
        }
        let mut acceptances = vec![];
        for (i, (op, error)) in ops.into_iter().zip(errors).enumerate() {
            let mut s = SegmentCompletion {
                segment: 0,
                status: ItemStatus::Pending,
                valid_bytes: 0,
                io_bytes: 0,
                checksum: None,
                epochs,
                producer_done: false,
                consumer_fenced: false,
                consumer_fence: None,
                error: error.clone(),
            };
            let accepted = error.is_none();
            if let Some(error) = error {
                s.status = ItemStatus::Rejected;
                entry.items.push(None);
                entry.expected.push(vec![SegmentExpectation {
                    valid_bytes: 0,
                    io_bytes: 0,
                    checksum: [0; 32],
                }]);
                acceptances.push(ItemAcceptance::Rejected {
                    item: i as u32,
                    op,
                    error,
                });
            } else {
                let (op, direction) = match op {
                    TransferOp::H2d(o) => (o, CopyDirection::HostToDevice),
                    TransferOp::D2h(o) => (o, CopyDirection::DeviceToHost),
                    _ => unreachable!(),
                };
                entry.expected.push(vec![SegmentExpectation {
                    valid_bytes: op.bytes,
                    io_bytes: op.bytes,
                    checksum: [0; 32],
                }]);
                let producer_fence = op.producer_fence;
                let mut item = Item {
                    host: Some(op.host),
                    device: Some(op.device),
                    bytes: op.bytes,
                    direction,
                    event: None,
                    taken: false,
                    source_retired: false,
                };
                // From this point any CUDA error may mean work was submitted:
                // accept + quarantine, NEVER return potentially-live owned inputs.
                // WP-A day 17 and 18: the stream this item's copy and completion event live on.
                // Both directions take the copy stream when one exists (the D2H since day 17, the
                // H2D since day 18); every copy under `new` takes the owner stream (the day-16
                // program, statement for statement).
                let issue: Arc<CudaStream> = match &self.copy {
                    Some(copy) => copy.clone(),
                    None => self.stream.clone(),
                };
                let off_owner = !Arc::ptr_eq(&issue, &self.stream);
                let submit = (|| {
                    if let Some(f) = producer_fence {
                        cuda(issue.wait(&self.producers[&f.sequence].1))?;
                    }
                    if direction == CopyDirection::HostToDevice {
                        self.owner.bind_destination(
                            ticket,
                            item.device.as_ref().ok_or(Error::AlreadyReleased)?,
                        )?;
                    }
                    let backing = self.owner.resolve::<Rc<RefCell<KvPlane>>>(
                        item.device.as_ref().ok_or(Error::AlreadyReleased)?,
                    )?;
                    let host = item.host.as_mut().ok_or(Error::AlreadyReleased)?;
                    match direction {
                        CopyDirection::DeviceToDevice => unreachable!("validated above"),
                        // A taken host destination may be used as an immutable
                        // H2D source before the earlier ticket is acknowledged.
                        CopyDirection::HostToDevice => cuda(
                            issue.memcpy_htod(
                                host.allocation
                                    .backing
                                    .as_ref()
                                    .ok_or(Error::AlreadyReleased)?,
                                &mut backing.borrow_mut().slice_mut(..item.bytes as usize),
                            ),
                        )?,
                        CopyDirection::DeviceToHost => {
                            let allocation =
                                Rc::get_mut(&mut host.allocation).ok_or(Error::Busy)?;
                            cuda(issue.memcpy_dtoh(
                                &backing.borrow().slice(..item.bytes as usize),
                                allocation.backing.as_mut().ok_or(Error::AlreadyReleased)?,
                            ))?;
                        }
                    }
                    drop(backing);
                    item.event = Some(cuda(issue.record_event(None))?);
                    // Install the wait separately from observing producer completion. On the
                    // owner stream (`new`, the day-16 program) the wait is a same-stream no-op and
                    // the item is fenced at submit, statement for statement. Off the owner stream:
                    // a D2H installs NO owner wait and is fenced at submit (its destination's
                    // consumer is the host, which waits on `event_done` in `progress` before the
                    // checksum; an owner wait would queue the tick's kernels behind the copy, the
                    // serialization Move 1 removed on day 17); an H2D is NOT fenced at submit
                    // (rule 3, day 19: `consumer_fenced` is the installed reader wait, never the
                    // copy's landing), its owner-stream wait is installed by
                    // `install_consumer_wait` at the settle, after the completion is observed and
                    // before `ready_view`, so the kernels the tenant issues between submit and
                    // settle do not queue behind the copy (day 18 installed the wait here and
                    // they did). The engine keeps the destination bound and unpublished until then.
                    if !off_owner {
                        cuda(self.stream.wait(item.event.as_ref().unwrap()))?;
                    }
                    if !off_owner || direction == CopyDirection::DeviceToHost {
                        s.consumer_fence = Some(self.next_fence(epochs.dst_gen)?);
                        s.consumer_fenced = true;
                    }
                    s.io_bytes = item.bytes;
                    Ok(())
                })();
                if let Err(error) = submit {
                    s.status = ItemStatus::Quarantined;
                    s.error = Some(error);
                    entry.unknown = true;
                }
                entry.items.push(Some(item));
                acceptances.push(ItemAcceptance::Accepted { item: i as u32 });
            }
            entry.completion.items.push(ItemOutcome {
                item: i as u32,
                accepted,
                segments: vec![s],
            });
        }
        self.entries.insert(ticket, entry);
        Ok(BatchSubmission {
            ticket,
            items: acceptances,
        })
    }
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion> {
        self.progress(ticket)?;
        Ok(self.entries[ticket].completion.clone())
    }
    fn cancel(&mut self, ticket: &TransferTicket) -> Result<CancelState> {
        self.check_thread()?;
        let e = self.entries.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        if e.published {
            return Ok(CancelState::AlreadyPublished);
        }
        // Day-11 rule 1: a source that left the ticket (per-side retirement, whole retirement
        // or recovery) cannot be followed by a revocation over it.
        if e.retired || e.items.iter().flatten().any(|i| i.source_retired) {
            return Err(Error::AlreadyReleased);
        }
        e.cancelled = true;
        Ok(CancelState::PublicationRevoked)
    }
    fn ready_view(
        &mut self,
        ticket: &TransferTicket,
        item: u32,
        current: Epochs,
    ) -> Result<ReadyView<'_>> {
        self.publishable(ticket, current)?;
        let e = self.entries.get_mut(ticket).unwrap();
        let i = e
            .items
            .get(item as usize)
            .ok_or(Error::InvalidLayout)?
            .as_ref()
            .ok_or(Error::Rejected)?;
        if i.direction != CopyDirection::HostToDevice {
            return Err(Error::Unsupported);
        }
        if i.taken {
            return Err(Error::AlreadyReleased);
        }
        let ready = self.owner.ready_view(
            i.device.as_ref().ok_or(Error::AlreadyReleased)?,
            &e.completion,
            &e.expected,
            current,
        )?;
        // Exposing a consumer-ready view IS publication, not just take().
        e.published = true;
        Ok(ready)
    }
    fn take_destination(
        &mut self,
        ticket: &TransferTicket,
        item: u32,
        current: Epochs,
    ) -> Result<Destination<Self::Host>> {
        self.publishable(ticket, current)?;
        let e = self.entries.get_mut(ticket).unwrap();
        let i = e
            .items
            .get_mut(item as usize)
            .ok_or(Error::InvalidLayout)?
            .as_mut()
            .ok_or(Error::Rejected)?;
        if i.taken {
            return Err(Error::AlreadyReleased);
        }
        if i.direction == CopyDirection::DeviceToDevice {
            // A capture's destination leaves through the caller's retained twin (`take_plane`)
            // after `retire` and `acknowledge`, never through the host-contract publication.
            return Err(Error::Unsupported);
        }
        let destination = if i.direction == CopyDirection::HostToDevice {
            Destination::Device(
                self.owner
                    .retain(i.device.as_ref().ok_or(Error::AlreadyReleased)?)?,
            )
        } else {
            // Destination residency is independent of source-side retirement.
            // Keep a sealed backing owner until acknowledgement: graph uses
            // must survive even if the caller drops its returned lease early.
            // Both owners share ONE physical allocation and governor charge.
            Destination::Host(CudaPinnedLease {
                allocation: i
                    .host
                    .as_ref()
                    .ok_or(Error::AlreadyReleased)?
                    .allocation
                    .clone(),
            })
        };
        i.taken = true;
        e.published = true;
        Ok(destination)
    }
    fn retire(&mut self, ticket: &TransferTicket, consumer_done: Option<FenceId>) -> Result<()> {
        self.progress(ticket)?;
        let e = self.entries.get_mut(ticket).unwrap();
        if e.retired {
            return Ok(());
        }
        if !e.completion.producer_done || !e.source.idle()? || !e.destination.idle()? {
            return Err(Error::Busy);
        }
        // WP-A day 30 (`d2h_span_batch` rule 4): landed spans not yet taken back keep the batch.
        if e.spans.is_some() {
            return Err(Error::Busy);
        }
        // WP-A day 32 (`h2d_span_batch` rule 4): the same for an H2D batch's spans.
        if e.h2d_spans.is_some() {
            return Err(Error::Busy);
        }
        if let Some(f) = consumer_done {
            let (actual, event) = e
                .destination
                .consumer_event
                .as_ref()
                .ok_or(Error::WrongOwner)?;
            if actual != &f {
                return Err(Error::WrongOwner);
            }
            if !event_done(event)? {
                return Err(Error::Busy);
            }
        } else if e.published {
            return Err(Error::Busy);
        }
        if Self::holds_cancelled_source(e) {
            // Day-11 rule 1: the caller recovers a cancelled restore's source; retirement waits.
            return Err(Error::Busy);
        }
        if e.items
            .iter()
            .flatten()
            .any(|i| i.direction == CopyDirection::HostToDevice)
        {
            self.owner.retire_binding(ticket)?;
        }
        // Detach the taken host destination only at acknowledgement. Source
        // ownership and untaken destinations can retire now that all uses ended.
        for slot in &mut e.items {
            if let Some(item) = slot
                && item.direction == CopyDirection::DeviceToHost
                && item.taken
            {
                item.device.take();
                item.event.take();
                item.source_retired = true;
            } else {
                slot.take();
            }
        }
        let charge = e.charge.as_ref().ok_or(Error::AlreadyReleased)?;
        self.governor.borrow_mut().release(charge)?;
        e.charge.take();
        e.retired = true;
        Ok(())
    }
    fn retire_source(&mut self, ticket: &TransferTicket) -> Result<()> {
        CudaTransfers::retire_source(self, ticket)
    }
    fn recover_source(&mut self, ticket: &TransferTicket, item: u32) -> Result<Self::Host> {
        CudaTransfers::recover_source(self, ticket, item)
    }
    fn retired(&mut self, ticket: &TransferTicket) -> Result<bool> {
        self.check_thread()?;
        Ok(self
            .entries
            .get(ticket)
            .ok_or(Error::UnknownTicket)?
            .retired)
    }
    fn acknowledge(&mut self, ticket: &TransferTicket) -> Result<()> {
        if !self.retired(ticket)? {
            return Err(Error::Busy);
        }
        // WP-A day 38 (design G''): a retired batch's receipt twins go back to the pool (every write
        // to them was observed before the batch landed); the device lanes drop, a stream-ordered
        // free. Nothing pinned is freed here.
        if let Some(mut e) = self.entries.remove(ticket) {
            let mut pool = self.twin_pool.borrow_mut();
            if let Some(r) = e.receipt.take() {
                pool.push(r.pinned);
            }
            if let Some(r) = e.d2h_receipt.take() {
                pool.push(r.scratch.pinned);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WP-A day 13: the enum's `Default` (the arm for a card class with no receipt) is exactly
    /// the flag bits cudarc's `alloc_pinned` passed before the seam existed, and the other arm is
    /// the cached attribute the OFF tier allocates with. The bits per kind do not move.
    #[test]
    fn pinned_kind_default_is_todays_write_combined_flag_bits() {
        assert_eq!(PinnedKind::default(), PinnedKind::WriteCombined);
        assert_eq!(
            PinnedKind::default().host_alloc_flags(),
            sys::CU_MEMHOSTALLOC_WRITECOMBINED
        );
        assert_eq!(PinnedKind::WriteCombined.host_alloc_flags(), 4);
        assert_eq!(PinnedKind::Cached.host_alloc_flags(), 0);
        for kind in [PinnedKind::WriteCombined, PinnedKind::Cached] {
            let flags = kind.host_alloc_flags();
            assert_eq!(flags & sys::CU_MEMHOSTALLOC_PORTABLE, 0, "{kind:?}");
            assert_eq!(flags & sys::CU_MEMHOSTALLOC_DEVICEMAP, 0, "{kind:?}");
            assert_eq!(PinnedKind::parse(kind.name()), Some(kind));
        }
        assert_eq!(PinnedKind::parse("wc"), None);
        assert_eq!(PinnedKind::parse("Cached"), None);
        assert_eq!(PinnedKind::parse(""), None);
    }

    /// WP-A day 14, lead ruling 22: the per-device default resolves by card class. `Cached` for
    /// the receipted RTX PRO 6000 Blackwell class (every edition name the fleet has shown),
    /// `WriteCombined` for the RTX 5090 class (its cell was inconclusive on the D2H clause) and
    /// for every class without a receipt (an unknown card, an empty name); the same names
    /// `parallel::HardwareTarget` keys on, so the two tables cannot drift.
    #[test]
    fn pinned_kind_per_device_default_resolves_by_card_class() {
        for name in [
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        ] {
            assert_eq!(PinnedKind::for_device(name), PinnedKind::Cached, "{name}");
            assert_eq!(PinnedKind::for_device(name).host_alloc_flags(), 0, "{name}");
        }
        for name in [
            "NVIDIA GeForce RTX 5090 Laptop GPU",
            "NVIDIA GeForce RTX 5090",
            "NVIDIA H100 80GB HBM3",
            "NVIDIA B200",
            "NVIDIA RTX PRO 6000 Ada Generation",
            "",
        ] {
            assert_eq!(
                PinnedKind::for_device(name),
                PinnedKind::WriteCombined,
                "{name}"
            );
            assert_eq!(
                PinnedKind::for_device(name).host_alloc_flags(),
                sys::CU_MEMHOSTALLOC_WRITECOMBINED,
                "{name}"
            );
        }
        // The elsewhere arm IS the enum's default: a class with no receipt gets today's bits.
        assert_eq!(PinnedKind::for_device("unknown"), PinnedKind::default());
    }

    /// The production allocator takes the device's resolved default and nothing in this file
    /// allocates pinned memory any other way; the default is resolved from the device name once,
    /// in the constructor, and no environment variable is read anywhere in the file.
    #[test]
    fn alloc_host_delegates_with_the_default_kind_and_no_other_pinned_allocation_remains() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        let start = body.find("pub fn alloc_host(").unwrap();
        let end = body[start..].find("pub fn alloc_host_kind(").unwrap() + start;
        assert!(
            body[start..end].contains("self.alloc_host_kind(bytes, request, self.pinned_default)"),
            "alloc_host must delegate with the device's resolved default"
        );
        assert_eq!(
            body.matches("PinnedKind::for_device(").count(),
            1,
            "the default is resolved exactly once, in CudaTransfers::new"
        );
        assert!(body.contains(
            "let pinned_default = PinnedKind::for_device(&owner.context().name().unwrap_or_default());"
        ));
        assert_eq!(
            body.matches("std::env::").count(),
            0,
            "no environment read selects the arm"
        );
        assert_eq!(
            body.matches("env::var").count(),
            0,
            "no environment read selects the arm"
        );
        assert_eq!(
            body.matches("result::malloc_host(").count(),
            1,
            "exactly one cuMemHostAlloc site, inside PinnedBacking::alloc"
        );
        // Call syntax only: the doc comments name cudarc's `alloc_pinned` as the history.
        assert_eq!(body.matches("alloc_pinned::<").count(), 0);
        assert_eq!(body.matches(".alloc_pinned(").count(), 0);
        assert!(body.contains("result::malloc_host(bytes, kind.host_alloc_flags())"));
    }

    /// WP-A day 17, 18 and 19 (memra#536 Move 1): under a copy stream every copy is issued there
    /// behind the producer fence's event; the submit-time owner wait exists on the owner stream
    /// only; an off-owner D2H is fenced at submit with no owner wait (its consumer is the host);
    /// an off-owner H2D is fenced by `install_consumer_wait` at the settle (rule 3), never at
    /// submit. Source census, CPU.
    #[test]
    fn copy_stream_issue_and_owner_wait_rules_are_as_stated() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        let submit = body.find("fn submit_batch(").unwrap();
        let submit_body = &body[submit..body[submit..].find("\n    fn poll(").unwrap() + submit];
        let issue = submit_body
            .find("let issue: Arc<CudaStream> = match &self.copy {")
            .expect("one issue-stream selection, on the copy stream's presence alone");
        assert!(submit_body[issue..issue + 200].contains("Some(copy) => copy.clone(),"));
        assert!(submit_body[issue..issue + 200].contains("None => self.stream.clone(),"));
        let producer_wait = submit_body
            .find("cuda(issue.wait(&self.producers[&f.sequence].1))?;")
            .unwrap();
        let copy_at = submit_body.find("issue.memcpy_htod(").unwrap();
        let owner_wait = submit_body
            .find("if !off_owner {")
            .expect("the submit-time owner wait exists on the owner stream only");
        assert!(producer_wait < copy_at && copy_at < owner_wait);
        assert!(
            submit_body[owner_wait..owner_wait + 120]
                .contains("self.stream.wait(item.event.as_ref().unwrap())")
        );
        let fenced_at_submit = submit_body
            .find("if !off_owner || direction == CopyDirection::DeviceToHost {")
            .expect("fenced at submit: the owner-stream program and an off-owner D2H only");
        assert!(owner_wait < fenced_at_submit);
        assert!(
            submit_body[fenced_at_submit..fenced_at_submit + 200]
                .contains("s.consumer_fenced = true;")
        );
        assert_eq!(submit_body.matches("s.consumer_fenced = true;").count(), 1);
        assert_eq!(submit_body.matches(".memcpy_htod(").count(), 1);
        assert_eq!(submit_body.matches(".memcpy_dtoh(").count(), 1);
        // Rule 3's install: the owner stream waits on each unfenced H2D item's event, then fences
        // it; nothing else in the engine fences an item after submit.
        let install = body.find("pub fn install_consumer_wait(").unwrap();
        let install_body = &body[install..body[install..].find("\n    pub fn ").unwrap() + install];
        let wait = install_body.find("cuda(stream.wait(event))").unwrap();
        let fence = install_body.find("s.consumer_fenced = true;").unwrap();
        assert!(wait < fence);
        // Day 21: the install skips a D2H only (its consumer is the host); an unfenced H2D and an
        // unfenced D2D restore are fenced here, a D2D capture (fenced at submit) is left alone.
        assert!(install_body.contains("item.direction == CopyDirection::DeviceToHost"));
        assert!(!install_body.contains("!= CopyDirection::HostToDevice"));
        // Day 20: the capture class fences at submit too (its consumer is the caller's
        // publication); three fencing statements in the body, none elsewhere (day 21: the restore
        // class adds none, its items are fenced by the install).
        assert_eq!(body.matches("s.consumer_fenced = true;").count(), 3);
    }

    /// WP-A day 21 (memra#536 Move 2 slice 2, `memra_tier::conformance::d2d_restore_ready`): the
    /// restore class exists only with the copy stream, issues behind the producer event on that
    /// stream into a borrowed destination view of exactly the item's bytes, registers nothing,
    /// records its completion event there, installs NO wait and NO fence at submit (rule 3: the
    /// install at the settle fences it), carries no checksum term, and `restore_landed` is the
    /// same-device landing predicate. Source census, CPU.
    #[test]
    fn d2d_restore_rules_are_as_stated() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        let submit = body.find("pub fn submit_d2d_restore(").unwrap();
        let submit_body = &body[submit..body[submit..].find("\n    pub fn ").unwrap() + submit];
        let needs_copy = submit_body
            .find("let Some(copy) = self.copy.clone() else {")
            .expect("the restore class requires the copy stream");
        assert!(
            submit_body[needs_copy..needs_copy + 120].contains("return Err(Error::Unsupported);")
        );
        assert!(submit_body.contains("op.bytes != op.destination.len() as u64"));
        let producer_wait = submit_body
            .find("cuda(copy.wait(&self.producers[&op.producer_fence.sequence].1))?;")
            .unwrap();
        let copy_at = submit_body
            .find("cuda(copy.memcpy_dtod(&op.source.slice(0..n), &mut op.destination))?;")
            .unwrap();
        let event_at = submit_body
            .find("item.event = Some(cuda(copy.record_event(None))?);")
            .unwrap();
        assert!(producer_wait < copy_at && copy_at < event_at);
        assert!(
            !submit_body.contains("self.stream.wait("),
            "no owner-stream wait exists at submit in the restore class"
        );
        // Day 22: the fault's early reader RECORDS an owner-stream event (the copy stream waits on
        // it); it never makes the owner stream wait on anything.
        assert_eq!(
            submit_body
                .matches("self.stream.record_event(None)")
                .count(),
            1
        );
        assert!(
            !submit_body.contains("s.consumer_fenced = true;")
                && submit_body.contains("consumer_fenced: false,"),
            "a restore item is unfenced at submit (rule 3)"
        );
        assert!(
            !submit_body.contains("register_device") && submit_body.contains("device: None,"),
            "nothing on either side is the registry's"
        );
        assert!(submit_body.contains("direction: CopyDirection::DeviceToDevice,"));
        assert!(submit_body.contains("checksum: None,"));
        // Two same-device copy statements in the body: the capture's and the restore's.
        assert_eq!(body.matches(".memcpy_dtod(").count(), 2);
        let landed = body.find("pub fn restore_landed(").unwrap();
        let landed_body = &body[landed..body[landed..].find("\n    pub fn ").unwrap() + landed];
        assert!(landed_body.contains("self.capture_landed(ticket)"));
    }

    /// WP-A day 20 (memra#536 Move 2 slice 1, `memra_tier::conformance::d2d_capture_publish`):
    /// the capture class exists only with the copy stream, issues behind the producer event on
    /// that stream, installs NO owner-stream wait, records its completion event there, carries no
    /// checksum term (so the host-contract gate refuses it), and `capture_landed` answers the
    /// engine's `producer_done` for capture tickets only.
    #[test]
    fn d2d_capture_rules_are_as_stated() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        let submit = body.find("pub fn submit_d2d_capture(").unwrap();
        let submit_body = &body[submit..body[submit..].find("\n    pub fn ").unwrap() + submit];
        let needs_copy = submit_body
            .find("let Some(copy) = self.copy.clone() else {")
            .expect("the capture class requires the copy stream");
        assert!(
            submit_body[needs_copy..needs_copy + 120].contains("return Err(Error::Unsupported);")
        );
        let producer_wait = submit_body
            .find("cuda(copy.wait(&self.producers[&op.producer_fence.sequence].1))?;")
            .unwrap();
        let copy_at = submit_body.find("cuda(copy.memcpy_dtod(").unwrap();
        let event_at = submit_body
            .find("item.event = Some(cuda(copy.record_event(None))?);")
            .unwrap();
        let fenced_at = submit_body.find("s.consumer_fenced = true;").unwrap();
        assert!(producer_wait < copy_at && copy_at < event_at && event_at < fenced_at);
        assert!(
            !submit_body.contains("self.stream.wait("),
            "no owner-stream wait exists in the capture class"
        );
        assert!(submit_body.contains("direction: CopyDirection::DeviceToDevice,"));
        assert!(submit_body.contains("checksum: None,"));
        // Day 21: the restore class adds the second same-device copy statement.
        assert_eq!(body.matches(".memcpy_dtod(").count(), 2);
        let progress = body.find("fn progress(").unwrap();
        let progress_body = &body[progress..body[progress..].find("\n    fn ").unwrap() + progress];
        let d2d_arm = progress_body
            .find("if item.direction == CopyDirection::DeviceToDevice {")
            .expect("progress has a capture arm");
        // Day 22 (slice 3): the arm fills the receipt from the lanes (the destination digest is
        // the checksum, the source digest the expectation) and lands the item only with them.
        let arm = &progress_body[d2d_arm..d2d_arm + 1600];
        assert!(arm.contains("let Some(lanes) = &receipt_lanes else {"));
        assert!(arm.contains(
            "e.expected[i][0].checksum = receipt_digest_from_lanes(take(off), item.bytes);"
        ));
        assert!(
            arm.contains(
                "s.checksum = Some(receipt_digest_from_lanes(take(off + 32), item.bytes));"
            )
        );
        assert!(arm.contains("continue;"));
        // Both classes digest the source behind the producer fence, copy, then digest the
        // destination; the fault's arm reads the destination on the owner stream instead.
        for class in ["pub fn submit_d2d_capture(", "pub fn submit_d2d_restore("] {
            let at = body.find(class).unwrap();
            let class_body = &body[at..body[at..].find("\n    pub fn ").unwrap() + at];
            let wait = class_body
                .find("cuda(copy.wait(&self.producers[&op.producer_fence.sequence].1))?;")
                .unwrap();
            let zeroed = class_body
                .find("cuda(copy.wait(&scratch.zeroed))?;")
                .unwrap();
            let delay = class_body.find("self.delay_on(&copy, ns)?;").unwrap();
            assert!(
                wait < zeroed && zeroed < delay,
                "{class}: the lanes' zero-fill fence"
            );
            // Revuto on integ38 (#639): the spin is issued once per batch, at the first item; a
            // spin per item multiplied the documented delay by the item count.
            let take = class_body
                .find("let fault = self.early_reader.take();")
                .expect("the arm is taken");
            let first_refusal = class_body.find("if ops.is_empty() {").unwrap();
            assert!(
                take < first_refusal,
                "{class}: the arm is spent before any admission refusal (revuto round 2, #639)"
            );
            let once = class_body
                .find("if let Some(ns) = fault\n                    && i == 0\n")
                .expect("the delay is gated on the first item");
            assert!(
                once < delay && delay - once < 120,
                "{class}: the gate wraps the spin"
            );
            assert_eq!(
                class_body.matches("self.delay_on(&copy, ns)?;").count(),
                1,
                "{class}: one spin site"
            );
            let src_digest = class_body.find("self.digest_on(\n                    &copy,\n                    &op.source.slice(0..n),").unwrap();
            let copy_at = class_body.find(".memcpy_dtod(").unwrap();
            let dst_digest = class_body.find("if fault.is_none() {").unwrap();
            let early = class_body.find("&self.stream,").unwrap();
            let event_at = class_body
                .find("item.event = Some(cuda(copy.record_event(None))?);")
                .unwrap();
            assert!(wait < delay && delay < src_digest && src_digest < copy_at);
            assert!(
                copy_at < dst_digest && dst_digest < early && early < event_at,
                "{class}"
            );
            assert!(class_body.contains("Self::seal_receipt(&copy, &mut entry, scratch);"));
            let scratch = class_body
                .find("self.receipt_scratch(ops.len())?;")
                .unwrap();
            let charge = class_body.find(".reserve(&request)?;").unwrap();
            assert!(
                scratch < charge,
                "{class}: the scratch is allocated before the charge"
            );
        }
        let landed = body.find("pub fn capture_landed(").unwrap();
        let landed_body = &body[landed..body[landed..].find("\n    pub fn ").unwrap() + landed];
        assert!(landed_body.contains("self.progress(ticket)?;"));
        assert!(landed_body.contains("i.direction != CopyDirection::DeviceToDevice"));
        assert!(landed_body.contains("Ok(e.completion.producer_done)"));
        let take = body.find("fn take_destination(").unwrap();
        let take_body = &body[take..body[take..].find("\n    fn ").unwrap() + take];
        assert!(take_body.contains("if i.direction == CopyDirection::DeviceToDevice {"));
    }

    /// WP-A day 37 (`DAY37.md` section 8, finding 5): the number of native cells in this module,
    /// one context each in the pool below (the census pins the count; day 42 added
    /// `span_receipt_digests_are_the_program_per_span`, day 48
    /// `day48_a_take_back_waits_for_its_own_lease_only`).
    const NATIVE_CELLS: usize = 17;
    /// The module's native cells' contexts: `NATIVE_CELLS` non-primary contexts created in ONE step,
    /// at the first `cell_context()` call (before that cell's body runs; every other cell waits
    /// here), and held for the whole test process by this static, so no context is created or
    /// destroyed while any cell runs.
    static CELL_CONTEXTS: std::sync::OnceLock<Vec<Arc<CudaContext>>> = std::sync::OnceLock::new();
    static NEXT_CELL_CONTEXT: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    /// A native cell OWNS its context: the next one of the pool, created by no one else and
    /// driven only by the calling cell's thread, the shape the engine's contract states (one CUDA
    /// owner thread per context; `check_thread`). On the device's shared primary context,
    /// parallel cells were several owner threads on one context, and there a pinned free, a
    /// synchronous device free or a module load on any of them held every other owner's driver
    /// calls until the context's device work drained, another cell's 300 ms hold included; across
    /// contexts only a context's creation or destruction holds anything (`day37-hold-probe`), and
    /// the pool does both outside every cell's body.
    fn cell_context() -> Arc<CudaContext> {
        let pool = CELL_CONTEXTS.get_or_init(|| {
            (0..NATIVE_CELLS)
                .map(|_| CudaContext::new_non_primary(0, 0).unwrap())
                .collect()
        });
        let k = NEXT_CELL_CONTEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        pool.get(k)
            .unwrap_or_else(|| {
                panic!(
                    "the cell context pool holds {NATIVE_CELLS} contexts, one per native cell of \
                     this module; a new native cell raises NATIVE_CELLS"
                )
            })
            .clone()
    }
    fn native_fixture() -> (CudaTransfers, Arc<CudaStream>, SharedBudget) {
        native_fixture_on(&cell_context())
    }
    /// The day-20 fixture on a context the cell already owns (a second `CudaTransfers` on the
    /// cell's own thread and context, as before when every cell shared the primary one).
    fn native_fixture_on(ctx: &Arc<CudaContext>) -> (CudaTransfers, Arc<CudaStream>, SharedBudget) {
        use memra_tier::tier::governor::Governor;
        let stream = ctx.new_stream().unwrap();
        let cap = TierBudget {
            version: 1,
            device: vec![1 << 30],
            peer: vec![0],
            replicas: vec![0],
            pinned: 1 << 30,
            pageable: 1 << 30,
            staging: 1 << 30,
            loaders: 64,
            inflight: 64,
            nvme: 0,
        };
        let gov: SharedBudget = Rc::new(RefCell::new(
            Governor::new(cap, TierBudget::zero(1), 64, 0, Arc::new(|| 0)).unwrap(),
        ));
        (
            CudaTransfers::new(stream.clone(), gov.clone()).unwrap(),
            stream,
            gov,
        )
    }
    fn request() -> BudgetRequest {
        BudgetRequest {
            bytes: TierBudget::zero(1),
            priority: Priority::Demand,
            deadline: Deadline(u64::MAX),
            tenant: [0; 32],
        }
    }
    /// WP-A day 38 (`DAY38.md` design G): the device receipt's order, by text. The scratch is
    /// allocated in the admission, before the batch is charged; the receipt is sealed before the
    /// item loop issues any copy; the digest precedes the flip, the flip precedes the lanes' D2H,
    /// the seal precedes the delay; `progress`'s device-receipt branch comes before the
    /// owner-thread checksum and hashes nothing.
    #[test]
    fn d2h_device_receipt_rules_are_as_stated() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        let fn_body = |name: &str| -> &str {
            let at = body.find(name).unwrap();
            &body[at..at + body[at..].find("\n    }\n").unwrap()]
        };
        let submit = fn_body("    fn submit_batch(");
        let scratch = submit.find("self.receipt_scratch_bytes(").unwrap();
        let reserve = submit
            .find("self.governor.borrow_mut().reserve(&request)?")
            .unwrap();
        assert!(
            scratch < reserve,
            "the scratch is allocated before the charge"
        );
        let seal = submit
            .find("self.seal_d2h_device_receipt(&ops, &errors, scratch)")
            .unwrap();
        let items = submit
            .find("for (i, (op, error)) in ops.into_iter().zip(errors).enumerate()")
            .unwrap();
        assert!(
            seal < items,
            "the receipt is sealed before any copy is issued"
        );
        // Move-then-match: the wrapper leaks the scratch on any error of the enqueues.
        let wrapper = fn_body("    fn seal_d2h_device_receipt(");
        assert!(wrapper.contains("let mut scratch = std::mem::ManuallyDrop::new(scratch);"));
        let into_inner = wrapper
            .find("std::mem::ManuallyDrop::into_inner(scratch)")
            .unwrap();
        assert!(
            wrapper.find("sealed.map(").unwrap() < into_inner,
            "handed out on success only"
        );
        let sealer = fn_body("    fn seal_d2h_device_receipt_into(");
        // Design G4 (section 17): the whole receipt on the copy stream, the one side stream, ahead
        // of the batch's copies on it.
        assert!(sealer.contains("let copy = self.copy.clone().ok_or(Error::Unsupported)?;"));
        let digest = sealer.find("copy.launch_builder(&k.sha256)").unwrap();
        let flip = sealer.find("copy.launch_builder(&k.flip)").unwrap();
        let lanes = sealer
            .find("copy.memcpy_dtoh(&scratch.lanes, &mut scratch.pinned)")
            .unwrap();
        let event = sealer.find("scratch.event = Some(").unwrap();
        // Design G''' (DAY38 section 14): the `d2h-delay` fault is a host-side hold; no stream
        // runs a spin for it. The wrapper spends the fault into the receipt's `hold_until`,
        // `progress` lands nothing while it is on, and `synchronize` returns only after it.
        assert!(
            !sealer.contains("delay_on("),
            "no spin on the receipt stream"
        );
        assert!(!sealer.contains("self.d2h_delay"));
        assert!(
            wrapper.contains("let hold_until = self\n            .d2h_delay\n            .take()")
        );
        assert!(wrapper.find("let hold_until").unwrap() < wrapper.find("sealed.map(").unwrap());
        assert!(!sealer.contains("d2h_delay"));
        assert!(digest < flip && flip < lanes && lanes < event);
        assert!(sealer.find("copy.wait(&self.producers").unwrap() < digest);
        assert!(sealer.find("copy.wait(&scratch.zeroed)").unwrap() < digest);
        // G4: one stream, so the flip needs no wait: it precedes the copies in stream order, and the
        // seal precedes the item loop (asserted above); submit_batch waits on nothing of its own.
        assert!(!submit.contains("flipped"));
        assert_eq!(submit.matches("copy.wait(").count(), 0);
        // G'': a retired batch's twins return to the pool at acknowledge; the scratch takes from it.
        let ack = fn_body("    fn acknowledge(&mut self, ticket: &TransferTicket)");
        assert!(
            ack.contains("pool.push(r.pinned);") && ack.contains("pool.push(r.scratch.pinned);")
        );
        let take = fn_body("    fn receipt_scratch_bytes(");
        let pooled = take.find(".position(|b| b.len == bytes)").unwrap();
        let fresh = take.find("PinnedBacking::alloc(").unwrap();
        assert!(pooled < fresh, "a pooled twin first");
        // G': an unretired entry's drop leaks the receipt scratch, as every other in-flight input.
        let drop_at = body.find("impl Drop for Entry {").unwrap();
        let drop_body = &body[drop_at..drop_at + body[drop_at..].find("\n}\n").unwrap()];
        assert!(drop_body.contains("std::mem::forget(self.d2h_receipt.take());"));
        // G': the host wait covers the receipt event (the receipt runs beside the copies).
        let sync = fn_body("    pub fn synchronize(&mut self, ticket: &TransferTicket)");
        let receipt_wait = sync.find("if let Some(r) = &e.d2h_receipt {").unwrap();
        let block =
            &sync[receipt_wait..receipt_wait + sync[receipt_wait..].find("\n        }").unwrap()];
        assert!(block.contains("r.scratch") && block.contains(".synchronize()"));
        let sync_at = sync.find("if let Some(r) = &e.d2h_receipt {").unwrap();
        let sync_rest = &sync[sync_at..];
        assert!(
            sync_rest.find(".synchronize(),").unwrap()
                < sync_rest.find("if let Some(h) = r.hold_until {").unwrap()
                && sync_rest.contains("std::thread::sleep(h - now);"),
            "the host wait covers the hold after the receipt event"
        );
        let progress = fn_body("    fn progress(");
        let held = progress
            .find("let held = r.hold_until.is_some_and(|h| std::time::Instant::now() < h);")
            .unwrap();
        let held_arm = progress.find("Ok(true) if held => None,").unwrap();
        let read = progress
            .find("Ok(true) => match r.scratch.pinned.as_slice() {")
            .unwrap();
        assert!(
            held < held_arm && held_arm < read,
            "a held batch reads no lanes"
        );
        let branch = progress
            .find("if e.d2h_receipt.is_some() && item.direction == CopyDirection::DeviceToHost {")
            .unwrap();
        let host_sum = progress.rfind("s.checksum = Some(checksum(").unwrap();
        assert!(branch < host_sum);
        let arm = &progress
            [branch..branch + progress[branch..].find("continue;\n            }").unwrap()];
        assert!(
            !arm.contains("checksum("),
            "the device-receipt branch hashes nothing"
        );
    }
    /// WP-A day 42 (`DAY42.md` design S2): the span receipts' order, by source. The D2H attach
    /// allocates the scratch in its admission (before the fence, so a refusal submits nothing),
    /// digests every span's SOURCE after the zero-fill and before the first copy in one batched
    /// call, and flips span 0's staging (under its fault) after its copy and before its event; it
    /// enqueues nothing after the copies. `progress` lands a D2H span batch on its copies (the H2D
    /// fold alone waits on a receipt event). The take moves the lanes into the one live span
    /// receipt, displacing an earlier one, and enqueues nothing. The seal checks the take's count
    /// and lengths before its first enqueue, then the landed digests in one batched call, the
    /// lanes' D2H and the event; the read observes the event before it reads the twin; an
    /// abandoned or displaced sealed receipt frees only once its event is observed. The H2D attach
    /// digests every DESTINATION after the copies in one batched call. The batched helper launches
    /// once per 64 spans.
    #[test]
    fn span_receipt_rules_are_as_stated() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        let fn_body = |name: &str| -> &str {
            let at = body.find(name).unwrap();
            &body[at..at + body[at..].find("\n    }\n").unwrap()]
        };
        let at = |b: &str, needle: &str| b.find(needle).unwrap_or_else(|| panic!("{needle}"));
        let d2h = fn_body("    pub fn submit_d2h_spans(");
        let scratch = at(d2h, "Some(_) => Some(self.receipt_scratch(spans.len())?),");
        let fence = at(d2h, "let fence = cuda(self.stream.record_event(None))?;");
        let zeroed = at(d2h, "cuda(copy.wait(&scratch.zeroed))?;");
        let source = at(
            d2h,
            "self.span_digests_on(&copy, &sources, lanes, 0, 64, SpanMemory::Device)",
        );
        let copies = at(d2h, ".enqueue_from_device_f32(&span.source, &copy)");
        let flip = at(d2h, "self.flip_staging_on(&copy, &span.destination)");
        let event = at(d2h, "copy.record_event(None).ok()");
        let batch = at(d2h, "receipt: scratch,");
        assert!(scratch < fence && fence < zeroed && zeroed < source && source < copies);
        assert!(copies < flip && flip < event && event < batch);
        assert_eq!(
            d2h.matches("self.span_digests_on(").count(),
            1,
            "the attach digests the sources only"
        );
        assert!(!d2h.contains("memcpy_dtoh"), "no lanes D2H at the attach");
        let progress = fn_body("    fn progress(");
        assert_eq!(
            progress
                .matches("if all && let Some(r) = &b.receipt {")
                .count(),
            1,
            "only the H2D fold waits on a receipt event"
        );
        let sync = fn_body("    pub fn synchronize(");
        assert_eq!(sync.matches("if let Some(r) = &b.receipt {").count(), 1);
        let take = fn_body("    pub fn take_d2h_spans(");
        assert!(!take.contains("launch") && !take.contains("span_digests_on"));
        assert!(
            at(
                take,
                "let displaced = self.span_receipt.replace(SpanReceipt {"
            ) < at(take, "self.abandon_span_receipt(r);")
        );
        let seal = fn_body("    pub fn seal_d2h_span_receipt(");
        let layout = at(seal, "return Err(Error::InvalidLayout);");
        let address = at(seal, ".device_address()");
        let sealed = at(seal, "r.sealed = true;");
        let landed = at(
            seal,
            "self.span_digests_on(&copy, &landed, lanes, 32, 64, SpanMemory::PinnedHost)?;",
        );
        let lanes = at(
            seal,
            "cuda(copy.memcpy_dtoh(&scratch.lanes, &mut scratch.pinned))?;",
        );
        let event = at(
            seal,
            "scratch.event = Some(cuda(copy.record_event(None))?);",
        );
        assert!(layout < address && address < sealed && sealed < landed);
        assert!(landed < lanes && lanes < event);
        let read = fn_body("    pub fn d2h_span_receipt(");
        assert!(
            at(read, "if !event_done(event)? {") < at(read, "r.observed = true;")
                && at(read, "r.observed = true;") < at(read, "cuda(scratch.pinned.as_slice())?")
        );
        assert!(
            read.contains("receipt_digest_from_lanes(lanes_at(&lanes, 64 * k), n)")
                && read.contains("receipt_digest_from_lanes(lanes_at(&lanes, 64 * k + 32), n)")
        );
        let abandon = fn_body("    fn abandon_span_receipt(");
        assert!(at(abandon, "if r.sealed {") < at(abandon, "self.span_reap.push(r);"));
        let reap = fn_body("    fn reap_span_receipts(");
        assert!(at(reap, "if done {") < at(reap, "r.observed = true;"));
        let drop = fn_body("impl Drop for SpanReceipt {");
        assert!(drop.contains("if self.sealed && !self.observed {"));
        let h2d = fn_body("    fn attach_h2d_spans(");
        let scratch = at(h2d, "spans.len().checked_mul(32).ok_or(Error::Overflow)?,");
        let fence = at(h2d, "let fence = cuda(self.stream.record_event(None))?;");
        let copies = at(h2d, ".enqueue_to_device_f32(&mut span.destination, &copy)");
        let zeroed = at(h2d, "cuda(copy.wait(&scratch.zeroed))?;");
        let dest = at(
            h2d,
            "self.span_digests_on(&copy, &destinations, lanes, 0, 32, SpanMemory::Device)?;",
        );
        let lanes = at(
            h2d,
            "cuda(copy.memcpy_dtoh(&scratch.lanes, &mut scratch.pinned))?;",
        );
        let event = at(h2d, "scratch.event = Some(cuda(copy.record_event(None))?);");
        assert!(scratch < fence && fence < copies && copies < zeroed && zeroed < dest);
        assert!(dest < lanes && lanes < event);
        let take = fn_body("    pub fn take_h2d_spans(");
        assert!(take.contains("self.twin_pool.borrow_mut().push(r.pinned);"));
        assert!(take.contains("receipt_digest_from_lanes(lanes_at(l, 32 * k), n)"));
        let helper = fn_body("    fn span_digests_on(");
        assert!(
            at(
                helper,
                "for (c, chunk) in spans.chunks(SPAN_ITEMS).enumerate() {"
            ) < at(helper, "cuda(unsafe { b.launch(cfg) })?;")
        );
        assert_eq!(
            helper.matches("b.launch(cfg)").count(),
            1,
            "one launch per chunk"
        );
        assert!(helper.contains("grid_dim: (blocks, chunk.len() as u32, 1),"));
    }
    /// WP-A day 48 (`DAY48.md` design S4): both release paths refuse a lease an unretired ticket
    /// names (`require_unbound` first), drain the owner stream, and never drain the copy stream;
    /// a ticket retires only after its landing (`producer_done`). Source census, CPU.
    #[test]
    fn day48_release_paths_drain_the_owner_stream_only() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        let fn_body = |name: &str| -> &str {
            let at = body.find(name).unwrap();
            &body[at..at + body[at..].find("\n    }\n").unwrap()]
        };
        let at = |b: &str, needle: &str| b.find(needle).unwrap_or_else(|| panic!("{needle}"));
        for name in [
            "    pub fn release_device(&mut self, lease: &DeviceLease)",
            "    pub fn take_plane(&mut self, lease: &DeviceLease)",
        ] {
            let f = fn_body(name);
            assert!(
                at(f, "self.require_unbound(lease)?;") < at(f, "cuda(self.stream.synchronize())?;")
            );
            assert!(!f.contains("copy"), "{name} touches no copy stream");
        }
        let unbound = fn_body("    fn require_unbound(");
        assert!(unbound.contains("!e.retired"));
        assert!(unbound.contains("d.allocation_id() == lease.allocation_id()"));
        let retire = fn_body("fn retire(&mut self, ticket: &TransferTicket, consumer_done");
        assert!(
            retire.contains("if !e.completion.producer_done"),
            "a ticket retires only after its landing"
        );
    }
    /// WP-A day 48 (`DAY48.md` design S4, clause (a), on a card): a take-back waits for its own
    /// lease's work only. A lease a live D2H ticket names is refused `Busy` and stays registered;
    /// once the ticket retired, the lease releases within 50 ms while a 300 ms spin holds the copy
    /// stream (the whole-stream drain would have waited for it).
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn day48_a_take_back_waits_for_its_own_lease_only() {
        use memra_tier::tier::governor::Governor;
        let ctx = cell_context();
        let stream = ctx.new_stream().unwrap();
        let cap = TierBudget {
            version: 1,
            device: vec![1 << 30],
            peer: vec![0],
            replicas: vec![0],
            pinned: 1 << 30,
            pageable: 1 << 30,
            staging: 1 << 30,
            loaders: 64,
            inflight: 64,
            nvme: 0,
        };
        let gov: SharedBudget = Rc::new(RefCell::new(
            Governor::new(cap, TierBudget::zero(1), 64, 0, Arc::new(|| 0)).unwrap(),
        ));
        let mut t = CudaTransfers::new_with_copy_stream(stream.clone(), gov).unwrap();
        let copy = t.copy_stream().unwrap().clone();
        let epochs = Epochs {
            state: 0,
            src_gen: 1,
            dst_gen: 1,
        };
        let bytes = 1usize << 20;
        let pattern: Vec<u8> = (0..bytes).map(|i| (i * 7 % 253) as u8).collect();
        let mut plane = stream.alloc_zeros::<u8>(bytes).unwrap();
        stream.memcpy_htod(&pattern, &mut plane).unwrap();
        let keep = t.register_device(plane, 1, request()).unwrap();
        let device = t.retain_device(&keep).unwrap();
        let host = t.alloc_host(bytes, request()).unwrap();
        let producer = t.record_producer(1).unwrap();
        let ticket = t
            .d2h(CopyOp {
                host,
                device,
                bytes: bytes as u64,
                epochs,
                producer_fence: Some(producer),
            })
            .unwrap();
        assert_eq!(
            t.release_device(&keep).err(),
            Some(Error::Busy),
            "a live ticket names it"
        );
        assert_eq!(
            t.device_registry_len(),
            1,
            "the refused lease stays registered"
        );
        t.synchronize(&ticket).unwrap();
        t.retire_source(&ticket).unwrap();
        let Destination::Host(landed) = t.take_destination(&ticket, 0, epochs).unwrap() else {
            panic!("D2H destination is not host")
        };
        assert_eq!(landed.bytes().unwrap(), pattern.as_slice());
        let consumer = t.record_consumer(&ticket).unwrap();
        stream.synchronize().unwrap();
        t.retire(&ticket, Some(consumer)).unwrap();
        t.acknowledge(&ticket).unwrap();
        t.release_producer(producer).unwrap();
        // Unrelated copy-stream work: a 300 ms spin.
        t.delay_on(&copy, 300_000_000).unwrap();
        let t0 = std::time::Instant::now();
        t.release_device(&keep).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        assert!(
            ms < 50.0,
            "the release waited {ms:.1} ms for the copy stream's unrelated spin"
        );
        assert_eq!(t.device_registry_len(), 0);
        copy.synchronize().unwrap();
    }
    /// WP-A day 46 (`DAY46.md` design S3): the grid rule. A launch over `k` spans holds at most one
    /// block per SM when `k <= sms`; pinned host reads take one block per span; no span takes more
    /// blocks than its words need; S2's shapes (192 x 64, 192 x 32) are not reachable.
    #[test]
    fn day46_span_digest_grids_leave_room_for_the_owner() {
        let mib3 = 3u64 << 20;
        for sms in [82u64, 170, 188] {
            for k in [1u64, 2, 32, 48, 64] {
                let b = span_blocks(sms, k, mib3, SpanMemory::Device) as u64;
                assert!(b >= 1);
                if k <= sms {
                    assert!(b * k <= sms, "sms={sms} k={k} b={b}");
                }
                assert_eq!(span_blocks(sms, k, mib3, SpanMemory::PinnedHost), 1);
            }
        }
        assert_eq!(span_blocks(188, 64, mib3, SpanMemory::Device), 2);
        assert_eq!(span_blocks(188, 32, mib3, SpanMemory::Device), 5);
        assert_eq!(span_blocks(82, 64, mib3, SpanMemory::Device), 1);
        // A span shorter than one block's stride takes one block whatever the card.
        assert_eq!(span_blocks(188, 1, 4096, SpanMemory::Device), 1);
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        assert!(body.contains(
            "let blocks = span_blocks(self.sm_count, chunk.len() as u64, longest, memory);"
        ));
        assert!(!body.contains(".div_ceil(2048).clamp(1, 2048) as u32;\n            let cfg = LaunchConfig {\n                grid_dim: (blocks, chunk.len()"));
        assert_eq!(
            body.matches("SpanMemory::PinnedHost)").count(),
            1,
            "the landed launch alone"
        );
        assert_eq!(
            body.matches("SpanMemory::Device)").count(),
            2,
            "the sources and the H2D destinations"
        );
    }
    /// WP-A day 42 (`DAY42.md` design S2, clause (a), on a card): the batched kernel is the CPU
    /// oracle per span, bitwise. 70 spans (two launches: 64 and 6) of lengths 1 B to 3 MiB + 3 at
    /// byte offsets 0 to 7, over device memory and over pinned host memory through its device
    /// address; each span's lanes against `memra_tier::conformance::receipt_digest` of its bytes.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn span_receipt_digests_are_the_program_per_span() {
        use memra_tier::tier::governor::Governor;
        let ctx = cell_context();
        let stream = ctx.new_stream().unwrap();
        let cap = TierBudget {
            version: 1,
            device: vec![1 << 30],
            peer: vec![0],
            replicas: vec![0],
            pinned: 1 << 30,
            pageable: 1 << 30,
            staging: 1 << 30,
            loaders: 64,
            inflight: 64,
            nvme: 0,
        };
        let gov: SharedBudget = Rc::new(RefCell::new(
            Governor::new(cap, TierBudget::zero(1), 64, 0, Arc::new(|| 0)).unwrap(),
        ));
        let t = CudaTransfers::new_with_copy_stream(stream.clone(), gov).unwrap();
        let copy = t.copy_stream().unwrap().clone();
        let total = (3usize << 20) + 3 + 8;
        let bytes: Vec<u8> = (0..total)
            .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9) >> 7) as u8)
            .collect();
        let sizes = [
            1usize,
            2,
            7,
            8,
            9,
            15,
            16,
            17,
            63,
            64,
            65,
            255,
            4095,
            4096,
            4097,
            61_445,
            1 << 20,
            (1 << 20) + 3,
            (3 << 20) + 3,
        ];
        let spans: Vec<(usize, usize)> = (0..70).map(|i| (i % 8, sizes[i % sizes.len()])).collect();
        let device = stream.clone_htod(&bytes).unwrap();
        let mut host = PinnedHostBuf::new(total).unwrap();
        host.copy_from_slice(&bytes).unwrap();
        stream.synchronize().unwrap();
        // Day 46 (design S3): each base at the bounded grid its launches take, and device memory at
        // the pinned host rule's one block per span too (the program is grid-independent).
        let bases = [
            ("device", device.device_ptr(&copy).0, SpanMemory::Device),
            (
                "device at one block per span",
                device.device_ptr(&copy).0,
                SpanMemory::PinnedHost,
            ),
            (
                "pinned host",
                host.device_address().unwrap(),
                SpanMemory::PinnedHost,
            ),
        ];
        for (what, base, memory) in bases {
            let lanes = stream.alloc_zeros::<u8>(32 * spans.len()).unwrap();
            stream.synchronize().unwrap();
            let items: Vec<(u64, u64)> = spans
                .iter()
                .map(|&(off, n)| (base + off as u64, n as u64))
                .collect();
            let at = lanes.device_ptr(&copy).0;
            t.span_digests_on(&copy, &items, at, 0, 32, memory).unwrap();
            copy.synchronize().unwrap();
            let got = stream.clone_dtoh(&lanes).unwrap();
            for (k, &(off, n)) in spans.iter().enumerate() {
                assert_eq!(
                    receipt_digest_from_lanes(lanes_at(&got, 32 * k), n as u64),
                    memra_tier::conformance::receipt_digest(&bytes[off..off + n]),
                    "{what} span {k}: {n} bytes at offset {off}"
                );
            }
        }
    }
    /// WP-A day 38 (`DAY38.md` design G4, section 17): the engine runs its side work on ONE stream
    /// beside the owner's, the copy stream, and has no receipt stream. BOX7's bisection (sections
    /// 13e to 13k) and the 5090's G''' cell (section 16) placed the tenant's per-demote decode hump
    /// on a second side stream: kernels there moved every later owner kernel boundary on both cards,
    /// and on the 5090 even with the copy stream kernel-free. Source census, CPU.
    #[test]
    fn one_side_stream_beside_the_owner() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        assert!(!body.contains("receipt_stream"), "no second side stream");
        // Streams the engine creates: the copy stream only (the owner's is the caller's).
        assert_eq!(body.matches(".new_stream()").count(), 1);
        // Direct launches: the four helpers' own (`stream.`, their parameter: the view digest,
        // day 42's batched span digests and span flip, the spin) and the copy stream's (the D2H
        // receipt's SHA-256 and flip).
        let launches: Vec<&str> = body
            .match_indices("launch_builder(")
            .map(|(at, _)| {
                let line_start = body[..at].rfind('\n').unwrap() + 1;
                body[line_start..at].trim()
            })
            .collect();
        assert_eq!(
            launches,
            [
                "let mut b = stream.",
                "let mut b = stream.",
                "let mut b = stream.",
                "let mut b = stream.",
                "let mut b = copy.",
                "let mut b = copy."
            ],
            "every direct launch is a helper's or the copy stream's"
        );
        // The helpers' call sites pass the copy stream, or the owner stream for the `d2d-delay`
        // early reader's destination digest (one per D2D class).
        let mut on_copy = 0;
        let mut on_owner = 0;
        for helper in [
            "self.digest_on(",
            "self.delay_on(",
            "self.span_digests_on(",
            "self.flip_staging_on(",
        ] {
            for (at, _) in body.match_indices(helper) {
                let arg = body[at + helper.len()..].trim_start();
                if arg.starts_with("&copy,") {
                    on_copy += 1;
                } else if arg.starts_with("&self.stream,") {
                    on_owner += 1;
                } else {
                    panic!(
                        "a kernel helper called on another stream: {}",
                        &arg[..40.min(arg.len())]
                    );
                }
            }
        }
        assert_eq!(
            (on_copy, on_owner),
            (10, 2),
            "four digests, two spins and (day 42) three batched span digests and the span flip on the \
             copy stream, two early readers"
        );
        // The release paths drain the owner stream and the copy stream.
        // WP-A day 48 (design S4): the release paths drain the owner stream only.
        assert_eq!(body.matches("synchronize_copy_stream").count(), 0);
    }
    /// WP-A day 38 (`DAY38.md` design G, `memra_tier::conformance::d2h_device_receipt`, on a card).
    /// A three-item D2H batch (lengths 1 MiB, 1 MiB + 3 and 61,445 bytes) under a 300 ms
    /// `d2h-delay` hold (host-side since design G'''): not landed while the hold is on, even after
    /// the copies completed;
    /// landed after the host wait with every item's
    /// checksum the receipt program over its SOURCE bytes, bitwise, and equal to the program over
    /// the landed host bytes (the witness); the kernel's receipt-stream time reads back once landed.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn d2h_device_receipt_lands_with_the_source_digest() {
        d2h_device_receipt_cell(false);
    }
    /// WP-A day 38: the `d2h-source-flip` red arm on a card. The same batch with the one-shot flip:
    /// item 0's receipt is still the program over its ORIGINAL source bytes, and the program over
    /// its landed bytes differs from it (the witness refuses); the other items agree.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn d2h_source_flip_is_witnessed_by_the_landed_bytes() {
        d2h_device_receipt_cell(true);
    }
    fn d2h_device_receipt_cell(flip: bool) {
        use memra_tier::tier::governor::Governor;
        let ctx = cell_context();
        let stream = ctx.new_stream().unwrap();
        let cap = TierBudget {
            version: 1,
            device: vec![1 << 30],
            peer: vec![0],
            replicas: vec![0],
            pinned: 1 << 30,
            pageable: 1 << 30,
            staging: 1 << 30,
            loaders: 64,
            inflight: 64,
            nvme: 0,
        };
        let gov: SharedBudget = Rc::new(RefCell::new(
            Governor::new(cap, TierBudget::zero(1), 64, 0, Arc::new(|| 0)).unwrap(),
        ));
        let mut t = CudaTransfers::new_with_copy_stream(stream.clone(), gov).unwrap();
        assert!(t.d2h_receipts_on_device());
        let copy = t.copy_stream().unwrap().clone();
        let epochs = Epochs {
            state: 0,
            src_gen: 1,
            dst_gen: 1,
        };
        let lens = [1usize << 20, (1 << 20) + 3, 61_445];
        let patterns: Vec<Vec<u8>> = lens
            .iter()
            .enumerate()
            .map(|(k, &n)| {
                (0..n)
                    .map(|i| ((i * 29 + k * 11 + (i >> 7)) % 251) as u8)
                    .collect()
            })
            .collect();
        let producer = t.record_producer(1).unwrap();
        let mut keeps = Vec::new();
        let mut ops = Vec::new();
        for p in &patterns {
            let mut plane = stream.alloc_zeros::<u8>(p.len()).unwrap();
            stream.memcpy_htod(p, &mut plane).unwrap();
            let keep = t.register_device(plane, 1, request()).unwrap();
            let device = t.retain_device(&keep).unwrap();
            let host = t.alloc_host(p.len(), request()).unwrap();
            keeps.push(keep);
            ops.push(TransferOp::D2h(CopyOp {
                host,
                device,
                bytes: p.len() as u64,
                epochs,
                producer_fence: Some(producer),
            }));
        }
        // The producer fence covers the planes' uploads (recorded before them above, so record a
        // fresh one after them and use it).
        let producer = {
            t.release_producer(producer).unwrap();
            let fresh = t.record_producer(1).unwrap();
            for op in &mut ops {
                if let TransferOp::D2h(o) = op {
                    o.producer_fence = Some(fresh);
                }
            }
            fresh
        };
        // Design G': the receipt runs on the receipt stream; the fault's 300 ms hold (host-side since
        // G''') keeps the batch unlanded after the copies have completed: a copy alone is not a
        // landing (the tier rule's rule 1, on a card).
        t.inject_d2h_delay(300_000_000);
        if flip {
            t.inject_d2h_source_flip();
        }
        let ticket = t.submit_batch(ops).map_err(|r| r.error).unwrap().ticket;
        if !flip {
            copy.synchronize().unwrap();
        }
        let c = t.poll(&ticket).unwrap();
        assert!(!c.producer_done, "not landed while the hold is on");
        assert!(
            t.d2h_receipt_gpu_ms(&ticket).is_none(),
            "no reading before the batch lands"
        );
        // The `Block` settle's shape: ONE host wait on the ticket, then the completion. The hold is
        // still on here, so the wait must cover it (designs G' and G''').
        t.synchronize(&ticket).unwrap();
        let c = t.poll(&ticket).unwrap();
        assert!(
            c.producer_done,
            "copies and receipt observed after one host wait on the ticket: landed"
        );
        let gpu_ms = t
            .d2h_receipt_gpu_ms(&ticket)
            .expect("the kernels' time reads back once landed");
        eprintln!(
            "D2H DEVICE RECEIPT cell flip={flip} items={} gpu_ms={gpu_ms:.3}",
            lens.len()
        );
        t.retire_source(&ticket).unwrap();
        for keep in &keeps {
            t.release_device(keep).unwrap();
        }
        for (i, p) in patterns.iter().enumerate() {
            let receipt = c.items[i].segments[0]
                .checksum
                .expect("a landed item has its receipt");
            assert_eq!(
                receipt,
                checksum(p),
                "item {i}: the receipt is the program over its source"
            );
            let Destination::Host(host) = t.take_destination(&ticket, i as u32, epochs).unwrap()
            else {
                panic!("D2H destination is not host")
            };
            let landed = checksum(host.bytes().unwrap());
            if flip && i == 0 {
                assert_ne!(
                    landed, receipt,
                    "the witness over the landed bytes sees the flip"
                );
                let mut want = p.clone();
                want[(p.len() - 1).min(5)] ^= 0x40;
                assert_eq!(
                    host.bytes().unwrap(),
                    want.as_slice(),
                    "exactly the one byte flipped"
                );
            } else {
                assert_eq!(
                    landed, receipt,
                    "item {i}: the witness agrees with the receipt"
                );
            }
        }
        let consumer = t.record_consumer(&ticket).unwrap();
        stream.synchronize().unwrap();
        t.retire(&ticket, Some(consumer)).unwrap();
        t.acknowledge(&ticket).unwrap();
        t.release_producer(producer).unwrap();
        // Design G'': the batch's twin went back to the pool; a second batch of the same shape
        // takes it (the pool does not grow) and its receipt is still the program over ITS sources.
        assert_eq!(
            t.twin_pool_len(),
            1,
            "the acknowledged batch's twin is pooled"
        );
        let mut keeps = Vec::new();
        let mut parts = Vec::new();
        let second: Vec<Vec<u8>> = patterns
            .iter()
            .map(|p| p.iter().map(|b| b.wrapping_add(17)).collect())
            .collect();
        for p in &second {
            let mut plane = stream.alloc_zeros::<u8>(p.len()).unwrap();
            stream.memcpy_htod(p, &mut plane).unwrap();
            let keep = t.register_device(plane, 1, request()).unwrap();
            let device = t.retain_device(&keep).unwrap();
            let host = t.alloc_host(p.len(), request()).unwrap();
            keeps.push(keep);
            parts.push((host, device, p.len() as u64));
        }
        // The producer fence after the uploads (a D2H without one is refused `NotReady` by the
        // contract's own validation).
        let producer = t.record_producer(1).unwrap();
        let ops = parts
            .into_iter()
            .map(|(host, device, bytes)| {
                TransferOp::D2h(CopyOp {
                    host,
                    device,
                    bytes,
                    epochs,
                    producer_fence: Some(producer),
                })
            })
            .collect();
        let ticket = t.submit_batch(ops).map_err(|r| r.error).unwrap().ticket;
        assert_eq!(
            t.twin_pool_len(),
            0,
            "the second batch took the pooled twin"
        );
        t.synchronize(&ticket).unwrap();
        let c = t.poll(&ticket).unwrap();
        assert!(c.producer_done);
        for (i, p) in second.iter().enumerate() {
            assert_eq!(c.items[i].segments[0].checksum, Some(checksum(p)));
        }
        t.retire_source(&ticket).unwrap();
        for keep in &keeps {
            t.release_device(keep).unwrap();
        }
        for i in 0..second.len() {
            let _ = t.take_destination(&ticket, i as u32, epochs).unwrap();
        }
        let consumer = t.record_consumer(&ticket).unwrap();
        stream.synchronize().unwrap();
        t.retire(&ticket, Some(consumer)).unwrap();
        t.acknowledge(&ticket).unwrap();
        t.release_producer(producer).unwrap();
        assert_eq!(
            t.twin_pool_len(),
            1,
            "the twin came back once more; the pool did not grow"
        );
    }
    /// WP-A day 39 (`DAY39.md` design T): the fill's thread counts and shares. `fill_threads_for_host`
    /// is `min(12, max(1, cpus / 2))`; `fill_threads_for_bytes` keeps a fill under 8 MiB on one
    /// thread; `fill_shares` covers every byte of the list exactly once, each share contiguous and
    /// in order, every share `ceil(total / t)` bytes but the last.
    #[test]
    fn day39_fill_shares_cover_every_byte_once() {
        use super::{FILL_SHARE_FLOOR, fill_shares, fill_threads_for_bytes, fill_threads_for_host};
        for (cpus, t) in [
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 2),
            (8, 4),
            (24, 12),
            (92, 12),
            (192, 12),
        ] {
            assert_eq!(fill_threads_for_host(cpus), t, "cpus={cpus}");
        }
        for host in [1usize, 2, 12] {
            for (bytes, t) in [
                (0usize, 1usize),
                (1, 1),
                (FILL_SHARE_FLOOR - 1, 1),
                (2 * FILL_SHARE_FLOOR - 1, 1),
                (2 * FILL_SHARE_FLOOR, 2),
                (52_690_944, 12),
                (156_893_184, 12),
            ] {
                assert_eq!(
                    fill_threads_for_bytes(host, bytes),
                    host.min(t),
                    "host={host} bytes={bytes}"
                );
            }
        }
        let b27 = [vec![3usize << 20; 48], vec![120 << 10; 48]].concat();
        let b9 = [vec![2usize << 20; 24], vec![96 << 10; 24]].concat();
        let odd = vec![1usize, 7, 0, 4 << 20, 3, (5 << 20) + 1, 13];
        for lens in [&b27, &b9, &odd, &vec![1usize], &vec![0usize]] {
            let total: usize = lens.iter().sum();
            for t in [1usize, 2, 3, 5, 12] {
                let shares = fill_shares(lens, t);
                assert_eq!(shares.len(), t);
                let per = total.div_ceil(t).max(1);
                // Concatenated in order, the shares walk the list byte for byte.
                let (mut item, mut off) = (0usize, 0usize);
                for (k, share) in shares.iter().enumerate() {
                    let bytes: usize = share.iter().map(|&(_, _, n)| n).sum();
                    if k + 1 < t && total >= per * (k + 1) {
                        assert_eq!(bytes, per, "share {k} of {t} over {total} B");
                    }
                    for &(i, o, n) in share {
                        while item < lens.len() && off == lens[item] {
                            item += 1;
                            off = 0;
                        }
                        assert_eq!((i, o), (item, off), "contiguous and in order");
                        assert!(n > 0 && o + n <= lens[i]);
                        off += n;
                    }
                }
                let covered: usize = shares.iter().flatten().map(|&(_, _, n)| n).sum();
                assert_eq!(covered, total, "every byte exactly once");
            }
        }
        // The 27B's list at 12 threads cuts inside a 3 MiB plane: a share starts mid-plane.
        assert!(fill_shares(&b27, 12).iter().any(|s| s[0].1 != 0));
    }
    /// WP-A day 39 (`DAY39.md` design T): the fill through `SpanFillTask::run` writes every
    /// destination bitwise equal to its plane at 1, 2, 5 and 12 threads, over a list whose cuts
    /// fall inside planes (heap destinations; CPU only).
    #[test]
    fn day39_threaded_fill_is_bitwise_the_planes() {
        let lens = [(3usize << 20) + 12, 7 << 20, 20, (5 << 20) + 4, 1 << 20];
        let planes: Vec<Arc<Vec<f32>>> = lens
            .iter()
            .enumerate()
            .map(|(k, &n)| {
                Arc::new(
                    (0..n / 4)
                        .map(|i| (i as f32) * 0.375 - 3.0 * k as f32)
                        .collect(),
                )
            })
            .collect();
        for threads in [1usize, 2, 5, 12] {
            let mut dsts: Vec<Vec<u8>> = lens.iter().map(|&n| vec![0xa5u8; n]).collect();
            let task = super::SpanFillTask {
                items: planes
                    .iter()
                    .zip(dsts.iter_mut())
                    .map(|(p, d)| (p.clone(), (d.as_mut_ptr(), d.len())))
                    .collect(),
                threads,
            };
            task.run();
            drop(task);
            for (p, d) in planes.iter().zip(&dsts) {
                assert_eq!(d.as_slice(), f32_bytes(p), "threads={threads}");
            }
        }
    }
    /// WP-A day 37 (`DAY37.md` section 8): every native cell of this module owns a context of the
    /// pool (`cell_context()`), the pool is the module's only context constructor, and its size is
    /// the native cell count.
    /// WP-A day 63 (`DAY63.md` design L1.1): the size class table.
    #[test]
    fn day63_the_lease_class_table() {
        const MIB: usize = 1 << 20;
        for (bytes, class) in [
            (1, 1),
            (3, 4),
            (4096, 4096),
            (4096 + 3, 8192),
            (MIB, MIB),
            (MIB + 1, 2 * MIB),
            (5 * MIB + 3, 6 * MIB),
            (4_718_592, 5 * MIB),
        ] {
            assert_eq!(lease_class(bytes), class, "{bytes}");
        }
    }

    /// WP-A day 63 (design L1; CPU census): nothing reads past a lease's length (every slice and
    /// view of a backing spans `len`; `capacity` appears only in its field, its accessor, the
    /// pool's class match and cap arithmetic, and `set_len`'s bound); the lease is charged its
    /// class; the pool closes when the engine drops; `cuMemFreeHost` is reached only through the
    /// backing's drop and the allocation's own failure path.
    #[test]
    fn day63_nothing_reads_past_a_lease_length_and_frees_stay_in_one_place() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        let backing = &body[body.find("impl PinnedBacking {").unwrap()
            ..body.find("impl HostSlice<u8> for PinnedBacking {").unwrap()];
        assert!(!backing.contains("from_raw_parts(self.ptr, self.capacity)"));
        assert!(!backing.contains("from_raw_parts_mut(self.ptr, self.capacity)"));
        assert_eq!(
            backing.matches("self.capacity").count(),
            2,
            "the accessor and set_len's bound"
        );
        let slice = &body[body.find("impl HostSlice<u8> for PinnedBacking {").unwrap()..];
        let slice = &slice[..slice.find("\n}\n").unwrap()];
        assert!(!slice.contains("capacity"), "the copies span the length");
        assert_eq!(
            body.matches(".capacity").count() - body.matches("fn capacity").count(),
            3,
            "take, put, and the view-free class read"
        );
        let alloc = &body[body.find("pub fn alloc_host_kind(").unwrap()..];
        let alloc = &alloc[..alloc.find("\n    }\n").unwrap()];
        assert!(alloc.contains("request.bytes.pinned = class as u64;"));
        assert!(alloc.contains("backing.set_len(bytes);"));
        let drop_engine = &body[body.find("impl Drop for CudaTransfers {").unwrap()..];
        assert!(drop_engine[..200].contains("self.lease_pool.close();"));
        assert_eq!(
            body.matches("result::free_host(").count(),
            2,
            "the backing's drop and its failed alloc"
        );
    }

    /// WP-A day 63 (design L1, clause (a), on a card): a dropped lease's backing is the next
    /// same-class lease's (the same host pointer) with the new length; its bytes and every copy of
    /// it span that length; the pinned ledger holds the pool's idle bytes after the drop and
    /// returns to zero after the pool closes.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn day63_a_dropped_lease_backs_the_next_same_class_lease() {
        let (mut t, stream, gov) = native_fixture();
        t.set_lease_pool_cap(1 << 20);
        let kind = t.pinned_default();
        let mut a = t.alloc_host_kind(4096 + 3, request(), kind).unwrap();
        a.write(&vec![0xa5u8; 4096 + 3]).unwrap();
        assert_eq!(
            gov.borrow().used().pinned,
            8192,
            "the lease is charged its class"
        );
        let ptr = a.bytes().unwrap().as_ptr();
        drop(a);
        assert_eq!(
            gov.borrow().used().pinned,
            8192,
            "the pool holds the idle class"
        );
        assert_eq!(t.lease_pool_counts(), (0, 1, 8192));
        let mut b = t.alloc_host_kind(5000, request(), kind).unwrap();
        assert_eq!(t.lease_pool_counts(), (1, 1, 0), "the pool served it");
        assert_eq!(b.bytes().unwrap().as_ptr(), ptr, "the same backing");
        assert_eq!(b.valid_bytes(), 5000);
        assert_eq!(b.bytes().unwrap().len(), 5000);
        // SAFETY: `b` is not written while the view is out.
        assert_eq!(unsafe { b.read_view() }.unwrap().len(), 5000);
        let pattern: Vec<u8> = (0..5000).map(|i| (i * 13 % 251) as u8).collect();
        b.write(&pattern).unwrap();
        let backing = b.allocation.backing.as_ref().unwrap();
        let dev = stream.clone_htod(backing).unwrap();
        assert_eq!(dev.len(), 5000, "the H2D spans the length");
        let back = stream.clone_dtoh(&dev).unwrap();
        assert_eq!(back, pattern);
        assert_eq!(gov.borrow().used().pinned, 8192);
        drop(b);
        let (n, bytes) = t.lease_pool().close();
        assert_eq!((n, bytes), (1, 8192));
        assert_eq!(gov.borrow().used().pinned, 0, "the ledger returns to zero");
        let c = t.alloc_host_kind(100, request(), kind).unwrap();
        drop(c);
        assert_eq!(gov.borrow().used().pinned, 0, "a closed pool frees");
    }

    /// WP-A day 63 (design L1.4, on a card): a drop past the pool's cap frees; the engine's drop
    /// closes the pool.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn day63_a_drop_past_the_cap_frees_and_the_engine_closes_the_pool() {
        let (mut t, _stream, gov) = native_fixture();
        t.set_lease_pool_cap(12288);
        let kind = t.pinned_default();
        let a = t.alloc_host_kind(8192, request(), kind).unwrap();
        let b = t.alloc_host_kind(8192, request(), kind).unwrap();
        assert_eq!(gov.borrow().used().pinned, 16384);
        drop(a);
        drop(b);
        assert_eq!(
            gov.borrow().used().pinned,
            8192,
            "one parked, one past the cap freed"
        );
        assert_eq!(t.lease_pool_counts().2, 8192);
        drop(t);
        assert_eq!(
            gov.borrow().used().pinned,
            0,
            "the engine's drop closed the pool"
        );
    }

    #[test]
    fn native_cells_own_their_context() {
        let src = include_str!("tier_transfer.rs");
        let tests = &src[src.find("#[cfg(test)]\nmod tests").unwrap()..];
        assert_eq!(tests.matches(concat!("CudaContext::", "new(")).count(), 0);
        assert_eq!(
            tests
                .matches(concat!("CudaContext::", "new_non_primary("))
                .count(),
            1
        );
        let marker = "#[ignore = \"native CUDA required";
        let helpers = [
            "native_fixture",
            "receipt_fixture",
            "d2h_device_receipt_cell",
        ];
        let mut cells = 0;
        for (at, _) in tests.match_indices(marker) {
            let rest = &tests[at..];
            let end = rest[1..]
                .find("\n    #[test]")
                .map(|e| e + 1)
                .unwrap_or(rest.len());
            let body = &rest[..end];
            let name = body.lines().nth(1).unwrap_or_default().trim();
            assert!(
                body.contains("cell_context()")
                    || helpers.iter().any(|h| body.contains(&format!("{h}("))),
                "{name} does not own its context"
            );
            cells += 1;
        }
        assert_eq!(cells, NATIVE_CELLS, "one pool context per native cell");
        // Every fixture a native cell may take its context through builds it with the pool.
        for h in helpers {
            let at = tests.find(&format!("fn {h}(")).unwrap();
            let body = &tests[at..at + tests[at..].find("\n    }\n").unwrap()];
            assert!(
                body.contains("cell_context()")
                    || body.contains("native_fixture_on(&cell_context())"),
                "{h} does not take its context from the pool"
            );
        }
    }
    /// WP-A day 37 (`DAY37.md` section 1, finding 5's instrument): poll a timed-hold cell's
    /// ticket until its first KV item is observed landed, the same loop the cells ran inline
    /// (its 5 s bound from the loop's start), and time it from the hold's enqueue: the instant
    /// of the observing poll, the poll count, the longest single `poll` call and the longest gap
    /// between two polls. Prints one `HOLD READING` line and returns it with the completion, so
    /// the rule-2 assertion's message carries the same fields. Reads only.
    fn poll_until_first_item_landed(
        t: &mut CudaTransfers,
        ticket: &TransferTicket,
        hold_at: std::time::Instant,
        cell: &str,
    ) -> (Completion, String) {
        let t0 = std::time::Instant::now();
        let (mut polls, mut longest_poll, mut longest_gap) = (0u32, 0f64, 0f64);
        let mut last_end: Option<std::time::Instant> = None;
        let (c, seen) = loop {
            let start = std::time::Instant::now();
            if let Some(end) = last_end {
                longest_gap = longest_gap.max(start.duration_since(end).as_secs_f64() * 1e3);
            }
            let c = t.poll(ticket).unwrap();
            let end = std::time::Instant::now();
            polls += 1;
            longest_poll = longest_poll.max(end.duration_since(start).as_secs_f64() * 1e3);
            last_end = Some(end);
            if c.items[0].segments[0].producer_done {
                break (c, end);
            }
            assert!(t0.elapsed().as_secs() < 5, "the KV item never landed");
        };
        let line = format!(
            "HOLD READING cell={cell} first_item_seen_ms={:.2} polls={polls} \
             longest_poll_ms={longest_poll:.2} longest_gap_ms={longest_gap:.2} \
             batch_landed_at_first_sight={}",
            seen.duration_since(hold_at).as_secs_f64() * 1e3,
            c.producer_done
        );
        eprintln!("{line}");
        (c, line)
    }
    /// WP-A day 37 (`DAY37.md` section 2a): a step clock over a timed-hold cell's CUDA-touching
    /// steps, each with its start offset from the hold's enqueue (negative before it) and its
    /// duration; `line` prints them as one `HOLD STEPS` line. Interior mutability so the cells'
    /// span-building closures can time their own calls; the calls and their order are unchanged.
    struct StepClock {
        hold_at: std::cell::Cell<Option<std::time::Instant>>,
        steps: RefCell<Vec<(String, std::time::Instant, f64)>>,
    }
    impl StepClock {
        fn new() -> Self {
            Self {
                hold_at: std::cell::Cell::new(None),
                steps: RefCell::new(Vec::new()),
            }
        }
        fn hold(&self) -> std::time::Instant {
            let at = std::time::Instant::now();
            self.hold_at.set(Some(at));
            at
        }
        fn time<T>(&self, label: &str, f: impl FnOnce() -> T) -> T {
            let start = std::time::Instant::now();
            let out = f();
            let ms = start.elapsed().as_secs_f64() * 1e3;
            self.steps.borrow_mut().push((label.to_string(), start, ms));
            out
        }
        fn line(&self, cell: &str) -> String {
            let hold = self.hold_at.get().expect("the hold was enqueued");
            let offset = |at: std::time::Instant| -> f64 {
                if at >= hold {
                    at.duration_since(hold).as_secs_f64() * 1e3
                } else {
                    -(hold.duration_since(at).as_secs_f64() * 1e3)
                }
            };
            let steps: Vec<String> = self
                .steps
                .borrow()
                .iter()
                .map(|(label, at, ms)| format!("{label}@{:.2}+{ms:.2}", offset(*at)))
                .collect();
            let line = format!("HOLD STEPS cell={cell} {}", steps.join(" "));
            eprintln!("{line}");
            line
        }
    }

    /// WP-A day 20 (memra#536 Move 2 slice 1): the capture class on a card. Two owned planes are
    /// registered as destinations with retained twins; a borrowed source holds a pattern; one
    /// capture batch on the copy stream behind an owner-stream producer fence; the host-contract
    /// gate refuses the items (no checksum term); `capture_landed` turns true; `retire(None)`,
    /// `acknowledge`, the planes come back through the twins and hold the pattern byte for byte.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn d2d_capture_lands_on_the_copy_stream_and_publishes_only_after_its_event() {
        use memra_tier::tier::governor::Governor;
        let ctx = cell_context();
        let stream = ctx.new_stream().unwrap();
        let cap = TierBudget {
            version: 1,
            device: vec![1 << 30],
            peer: vec![0],
            replicas: vec![0],
            pinned: 1 << 30,
            pageable: 1 << 30,
            staging: 1 << 30,
            loaders: 64,
            inflight: 64,
            nvme: 0,
        };
        let gov: SharedBudget = Rc::new(RefCell::new(
            Governor::new(cap, TierBudget::zero(1), 64, 0, Arc::new(|| 0)).unwrap(),
        ));
        let mut t = CudaTransfers::new_with_copy_stream(stream.clone(), gov).unwrap();
        let bytes = 8usize << 20;
        let pattern: Vec<u8> = (0..bytes).map(|i| (i * 13 % 251) as u8).collect();
        let mut source = stream.alloc_zeros::<u8>(bytes).unwrap();
        stream.memcpy_htod(&pattern, &mut source).unwrap();
        let generation = 1u64;
        let epochs = Epochs {
            state: 0,
            src_gen: 1,
            dst_gen: generation,
        };
        let mut twins = Vec::new();
        let mut destinations = Vec::new();
        for _ in 0..2 {
            let fresh = stream.alloc_zeros::<u8>(bytes).unwrap();
            let lease = t.register_device(fresh, generation, request()).unwrap();
            twins.push(t.retain_device(&lease).unwrap());
            destinations.push(lease);
        }
        // The class does not exist without the copy stream.
        let (mut on_owner, _s, _g) = native_fixture_on(&ctx);
        assert!(matches!(
            on_owner.submit_d2d_capture(Vec::new(), epochs),
            Err(Error::Unsupported)
        ));
        let producer = t.record_producer(generation).unwrap();
        let ops: Vec<D2dCapture<'_>> = destinations
            .into_iter()
            .map(|destination| D2dCapture {
                source: &source,
                destination,
                bytes: bytes as u64,
                producer_fence: producer,
            })
            .collect();
        let ticket = t.submit_d2d_capture(ops, epochs).unwrap();
        // The host-contract gate never publishes a capture, landed or not.
        assert!(t.ready_view(&ticket, 0, epochs).is_err());
        assert!(t.take_destination(&ticket, 0, epochs).is_err());
        let mut polls = 0u32;
        while !t.capture_landed(&ticket).unwrap() {
            polls += 1;
            assert!(
                polls < 1_000_000,
                "the copy stream never completed the capture"
            );
        }
        let c = t.poll(&ticket).unwrap();
        assert!(c.producer_done && c.consumer_fenced);
        // Day 22 (slice 3): every item lands WITH its receipt: the destination digest is the
        // checksum and equals the CPU oracle over the pattern; the gate opens on it.
        let oracle = memra_tier::conformance::receipt_digest(&pattern);
        for item in &c.items {
            assert_eq!(item.segments[0].valid_bytes, bytes as u64);
            assert_eq!(
                item.segments[0].checksum,
                Some(oracle),
                "the destination digest is the oracle's"
            );
        }
        let receipt = t.d2d_receipt(&ticket).unwrap();
        assert_eq!(receipt.items.len(), 2);
        assert_eq!(receipt.bytes, 2 * bytes as u64);
        for term in &receipt.items {
            assert_eq!(term.source, oracle, "the source digest, behind the fence");
            assert_eq!(term.destination, Some(oracle));
        }
        assert_eq!(receipt.verdict, Ok(()), "a matching receipt opens the gate");
        let expected: Vec<Vec<SegmentExpectation>> = (0..2)
            .map(|_| {
                vec![SegmentExpectation {
                    valid_bytes: bytes as u64,
                    io_bytes: bytes as u64,
                    checksum: oracle,
                }]
            })
            .collect();
        assert_eq!(c.require(&ticket, &expected, true), Ok(()));
        // The engine's publication stays refused by DIRECTION: a capture publishes through the
        // caller's index insert, never through `ready_view`.
        assert!(matches!(
            t.ready_view(&ticket, 0, epochs),
            Err(Error::Unsupported)
        ));
        t.release_producer(producer).unwrap();
        t.retire(&ticket, None).unwrap();
        assert!(t.retired(&ticket).unwrap());
        t.acknowledge(&ticket).unwrap();
        for twin in &twins {
            let plane = t.take_plane(twin).unwrap().into_pooled().unwrap();
            let back = stream.clone_dtoh(&plane).unwrap();
            assert_eq!(back, pattern, "the capture holds the source bytes");
        }
        assert_eq!(t.device_registry_len(), 0);
    }

    /// WP-A day 21 (memra#536 Move 2 slice 2, `memra_tier::conformance::d2d_restore_ready`): the
    /// restore class on a card. Two borrowed destination planes (the fixture's own, as a session
    /// cache's would be) and a borrowed patterned source; one restore batch on the copy stream
    /// behind an owner-stream producer fence; nothing registered; the items are unfenced at
    /// submit and the host-contract gate refuses them; `restore_landed` turns true; the install
    /// fences every item (the owner stream now waits on the events); `retire(None)`,
    /// `acknowledge`; an owner-stream readback after the install holds the pattern byte for byte.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn d2d_restore_lands_on_the_copy_stream_and_is_ready_only_after_the_installed_wait() {
        use memra_tier::tier::governor::Governor;
        let ctx = cell_context();
        let stream = ctx.new_stream().unwrap();
        let cap = TierBudget {
            version: 1,
            device: vec![1 << 30],
            peer: vec![0],
            replicas: vec![0],
            pinned: 1 << 30,
            pageable: 1 << 30,
            staging: 1 << 30,
            loaders: 64,
            inflight: 64,
            nvme: 0,
        };
        let gov: SharedBudget = Rc::new(RefCell::new(
            Governor::new(cap, TierBudget::zero(1), 64, 0, Arc::new(|| 0)).unwrap(),
        ));
        let mut t = CudaTransfers::new_with_copy_stream(stream.clone(), gov).unwrap();
        let bytes = 8usize << 20;
        let pattern: Vec<u8> = (0..bytes).map(|i| (i * 17 % 253) as u8).collect();
        let mut source = stream.alloc_zeros::<u8>(bytes).unwrap();
        stream.memcpy_htod(&pattern, &mut source).unwrap();
        let generation = 1u64;
        let epochs = Epochs {
            state: 0,
            src_gen: 1,
            dst_gen: generation,
        };
        let mut destinations: Vec<CudaSlice<u8>> = (0..2)
            .map(|_| stream.alloc_zeros::<u8>(bytes).unwrap())
            .collect();
        // The class does not exist without the copy stream.
        let (mut on_owner, _s, _g) = native_fixture_on(&ctx);
        assert!(matches!(
            on_owner.submit_d2d_restore(Vec::new(), epochs),
            Err(Error::Unsupported)
        ));
        let producer = t.record_producer(generation).unwrap();
        let ticket = {
            let ops: Vec<D2dRestore<'_>> = destinations
                .iter_mut()
                .map(|destination| D2dRestore {
                    source: &source,
                    destination: destination.slice_mut(..bytes),
                    bytes: bytes as u64,
                    producer_fence: producer,
                })
                .collect();
            t.submit_d2d_restore(ops, epochs).unwrap()
        };
        assert_eq!(
            t.device_registry_len(),
            0,
            "nothing registered on either side"
        );
        // The host-contract gate never publishes a restore, landed or not.
        assert!(t.ready_view(&ticket, 0, epochs).is_err());
        assert!(t.take_destination(&ticket, 0, epochs).is_err());
        let mut polls = 0u32;
        while !t.restore_landed(&ticket).unwrap() {
            polls += 1;
            assert!(
                polls < 1_000_000,
                "the copy stream never completed the restore"
            );
        }
        let c = t.poll(&ticket).unwrap();
        assert!(c.producer_done);
        assert!(
            !c.consumer_fenced,
            "landed and still unfenced: no wait was installed at submit"
        );
        let oracle = memra_tier::conformance::receipt_digest(&pattern);
        for item in &c.items {
            assert_eq!(item.segments[0].valid_bytes, bytes as u64);
            assert_eq!(
                item.segments[0].checksum,
                Some(oracle),
                "day 22: landed with its receipt"
            );
            assert!(!item.segments[0].consumer_fenced);
        }
        // Day 22: before the install the gate still says NotReady (the fence is part of it).
        let early = t.d2d_receipt(&ticket).unwrap();
        assert_eq!(early.verdict, Err(Error::NotReady));
        // Rule 3: the install fences every item with a real owner-stream wait.
        t.install_consumer_wait(&ticket).unwrap();
        let c = t.poll(&ticket).unwrap();
        assert!(
            c.producer_done && c.consumer_fenced,
            "landed and fenced: ready"
        );
        for item in &c.items {
            assert!(item.segments[0].consumer_fenced && item.segments[0].consumer_fence.is_some());
        }
        let receipt = t.d2d_receipt(&ticket).unwrap();
        assert_eq!(
            receipt.verdict,
            Ok(()),
            "landed, fenced, matching: the gate opens"
        );
        for term in &receipt.items {
            assert_eq!(term.source, oracle);
            assert_eq!(term.destination, Some(oracle));
        }
        assert!(matches!(
            t.ready_view(&ticket, 0, epochs),
            Err(Error::Unsupported)
        ));
        t.release_producer(producer).unwrap();
        t.retire(&ticket, None).unwrap();
        assert!(t.retired(&ticket).unwrap());
        t.acknowledge(&ticket).unwrap();
        // The reader (owner) stream reads after the installed wait: the pattern, byte for byte.
        for destination in &destinations {
            let back = stream.clone_dtoh(destination).unwrap();
            assert_eq!(back, pattern, "the restore holds the source bytes");
        }
    }

    /// The receipt cells' fixture. The engine's context runs with cudarc's event tracking DISABLED
    /// (`Engine::new`, `gpu.ctx.disable_event_tracking()`), so no slice carries implicit
    /// cross-stream waits and the receipt's ordering is the explicit fences alone; the fixture
    /// matches (day 22, first sitting: with tracking on, cudarc made the fault's owner-stream
    /// reader wait on the copy-stream memcpy's write event, so the "early" reader read the landed
    /// copy and the cell could not show the fault the server shows).
    fn receipt_fixture() -> (CudaTransfers, Arc<CudaStream>) {
        use memra_tier::tier::governor::Governor;
        let ctx = cell_context();
        // SAFETY: no slice of this context exists yet; every slice below is created untracked,
        // exactly as the engine's are.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().unwrap();
        let cap = TierBudget {
            version: 1,
            device: vec![1 << 30],
            peer: vec![0],
            replicas: vec![0],
            pinned: 1 << 30,
            pageable: 1 << 30,
            staging: 1 << 30,
            loaders: 64,
            inflight: 64,
            nvme: 0,
        };
        let gov: SharedBudget = Rc::new(RefCell::new(
            Governor::new(cap, TierBudget::zero(1), 64, 0, Arc::new(|| 0)).unwrap(),
        ));
        (
            CudaTransfers::new_with_copy_stream(stream.clone(), gov).unwrap(),
            stream,
        )
    }

    /// WP-A day 22 (memra#536 Move 2 slice 3): the device digest IS the CPU oracle. Three spans (a
    /// word multiple, a tail of 5 bytes, and a one-byte span) are digested on the copy stream into
    /// zeroed lanes; the host folds the byte count; the result equals
    /// `memra_tier::conformance::receipt_digest` over the same bytes read back, and a one-byte
    /// flip on the device moves it.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn d2d_receipt_digest_matches_the_cpu_oracle() {
        let (t, stream) = receipt_fixture();
        let copy = t.copy_stream().unwrap().clone();
        for &n in &[8usize << 20, (8usize << 20) + 5, 1usize] {
            let pattern: Vec<u8> = (0..n).map(|i| (i * 29 % 241) as u8).collect();
            let mut span = stream.alloc_zeros::<u8>(n).unwrap();
            stream.memcpy_htod(&pattern, &mut span).unwrap();
            // Integ38 (the local 5090 sitting): the upload runs on the owner stream and the digest
            // on the copy stream; with cudarc's tracking off (as the engine runs) nothing orders
            // them, and the 8 MiB span lost the race on the 5090 (`n=8388608: the device digest is
            // the oracle's` failed, three of three spans green on the target card). The engine's
            // route waits on the producer fence; the test states its producer the same way.
            stream.synchronize().unwrap();
            let mut lanes = copy.alloc_zeros::<u8>(32).unwrap();
            t.digest_on(&copy, &span.slice(..n), &mut lanes.slice_mut(..32))
                .unwrap();
            let back = copy.clone_dtoh(&lanes).unwrap();
            let mut l = [0u64; 4];
            for (k, lane) in l.iter_mut().enumerate() {
                *lane = u64::from_le_bytes(back[8 * k..8 * k + 8].try_into().unwrap());
            }
            let device = receipt_digest_from_lanes(l, n as u64);
            let readback = stream.clone_dtoh(&span).unwrap();
            assert_eq!(readback, pattern);
            assert_eq!(
                device,
                memra_tier::conformance::receipt_digest(&readback),
                "n={n}: the device digest is the oracle's"
            );
            // A one-byte flip moves the device digest (fresh lanes).
            let mut flipped = pattern.clone();
            flipped[n / 2] ^= 0x80;
            stream.memcpy_htod(&flipped, &mut span).unwrap();
            stream.synchronize().unwrap();
            let mut lanes2 = copy.alloc_zeros::<u8>(32).unwrap();
            t.digest_on(&copy, &span.slice(..n), &mut lanes2.slice_mut(..32))
                .unwrap();
            let back2 = copy.clone_dtoh(&lanes2).unwrap();
            assert_ne!(back2, back, "n={n}: a one-byte flip moves the lanes");
        }
    }

    /// WP-A day 22, the red arm on a card: `inject_d2d_early_reader` makes the next capture read
    /// its destination on the owner stream before the delayed copy; the receipt's destination
    /// digest is the fresh plane's (zeros), differs from the source's, and the gate answers
    /// `Corrupt`; the ticket still retires and acknowledges and the planes come back (the copy
    /// did land, late). A second batch without the fault matches again (one-shot).
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn d2d_early_reader_fault_is_refused_by_the_receipt() {
        let (mut t, stream) = receipt_fixture();
        let bytes = 4usize << 20;
        let pattern: Vec<u8> = (0..bytes).map(|i| (i * 11 % 239) as u8).collect();
        let mut source = stream.alloc_zeros::<u8>(bytes).unwrap();
        stream.memcpy_htod(&pattern, &mut source).unwrap();
        let generation = 1u64;
        let epochs = Epochs {
            state: 0,
            src_gen: 1,
            dst_gen: generation,
        };
        let oracle = memra_tier::conformance::receipt_digest(&pattern);
        let stale = memra_tier::conformance::receipt_digest(&vec![0u8; bytes]);
        for (round, fault) in [(0, true), (1, false)] {
            let fresh = stream.alloc_zeros::<u8>(bytes).unwrap();
            let lease = t.register_device(fresh, generation, request()).unwrap();
            let twin = t.retain_device(&lease).unwrap();
            if fault {
                t.inject_d2d_early_reader(D2D_DELAY_FAULT_NS);
            }
            let producer = t.record_producer(generation).unwrap();
            let ticket = t
                .submit_d2d_capture(
                    vec![D2dCapture {
                        source: &source,
                        destination: lease,
                        bytes: bytes as u64,
                        producer_fence: producer,
                    }],
                    epochs,
                )
                .unwrap();
            assert!(matches!(
                t.d2d_receipt(&ticket),
                Err(Error::NotReady) | Ok(_)
            ));
            while !t.capture_landed(&ticket).unwrap() {}
            let receipt = t.d2d_receipt(&ticket).unwrap();
            assert_eq!(receipt.items.len(), 1);
            assert_eq!(receipt.items[0].source, oracle, "round {round}");
            if fault {
                assert_eq!(
                    receipt.items[0].destination,
                    Some(stale),
                    "the early reader saw the fresh plane"
                );
                assert_eq!(receipt.verdict, Err(Error::Corrupt), "refused");
            } else {
                assert_eq!(receipt.items[0].destination, Some(oracle));
                assert_eq!(receipt.verdict, Ok(()), "the fault was one-shot");
            }
            t.release_producer(producer).unwrap();
            t.retire(&ticket, None).unwrap();
            t.acknowledge(&ticket).unwrap();
            let plane = t.take_plane(&twin).unwrap().into_pooled().unwrap();
            let back = stream.clone_dtoh(&plane).unwrap();
            assert_eq!(
                back, pattern,
                "the copy did land (late); nothing was published on it"
            );
        }
        assert_eq!(t.device_registry_len(), 0);
    }

    /// WP-A day 22, cell (v): the digest's price against the copy's own time on this card,
    /// event-timed on the copy stream over one 158 MB span (the 27B plain entry's size class),
    /// N=5 per order, both orders. A reading, printed verbatim; the pre-registered rule of day 19
    /// (`OWNER-THREAD-OFFLOAD.md`) is applied by the day's write-up, never here.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn d2d_receipt_digest_price_against_the_copy() {
        let (t, stream) = receipt_fixture();
        let copy = t.copy_stream().unwrap().clone();
        let n = 158usize << 20;
        let pattern: Vec<u8> = (0..n).map(|i| (i * 7 % 251) as u8).collect();
        let mut src = stream.alloc_zeros::<u8>(n).unwrap();
        stream.memcpy_htod(&pattern, &mut src).unwrap();
        let mut dst = stream.alloc_zeros::<u8>(n).unwrap();
        let mut lanes = copy.alloc_zeros::<u8>(32).unwrap();
        stream.synchronize().unwrap();
        // cudarc's default event flags disable timing; the cell needs `CU_EVENT_DEFAULT`.
        let timing = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let timed = |f: &mut dyn FnMut()| -> f32 {
            let a = copy.record_event(timing).unwrap();
            f();
            let b = copy.record_event(timing).unwrap();
            b.synchronize().unwrap();
            a.elapsed_ms(&b).unwrap()
        };
        let median = |v: &mut Vec<f32>| -> f32 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        // Warm both once.
        timed(&mut || copy.memcpy_dtod(&src, &mut dst).unwrap());
        timed(&mut || {
            t.digest_on(&copy, &src.slice(..n), &mut lanes.slice_mut(..32))
                .unwrap()
        });
        for order in ["copy-first", "digest-first"] {
            let mut copies = Vec::new();
            let mut digests = Vec::new();
            for _ in 0..5 {
                if order == "copy-first" {
                    copies.push(timed(&mut || copy.memcpy_dtod(&src, &mut dst).unwrap()));
                    digests.push(timed(&mut || {
                        t.digest_on(&copy, &src.slice(..n), &mut lanes.slice_mut(..32))
                            .unwrap()
                    }));
                } else {
                    digests.push(timed(&mut || {
                        t.digest_on(&copy, &src.slice(..n), &mut lanes.slice_mut(..32))
                            .unwrap()
                    }));
                    copies.push(timed(&mut || copy.memcpy_dtod(&src, &mut dst).unwrap()));
                }
            }
            let cm = median(&mut copies.clone());
            let dm = median(&mut digests.clone());
            println!(
                "D2D-RECEIPT PRICE order={order} bytes={n} n_per_arm=5 copy_ms={copies:?} copy_median={cm:.3} \
                 digest_ms={digests:?} digest_median={dm:.3} pair_median={:.3} pair_over_copy={:.2}",
                2.0 * dm,
                2.0 * dm / cm
            );
        }
        let back = stream.clone_dtoh(&dst).unwrap();
        assert_eq!(back, pattern);
    }

    /// The arm is honoured by the driver, not just recorded: `cuMemHostGetFlags` on the lease's
    /// pointer returns the arm's flag bits, the default lease reads back this device's resolved
    /// default (the card class's arm, and the driver's record of it), and a host write followed
    /// by a host read is exact under both arms.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn pinned_kind_arm_is_honoured_by_the_driver() {
        let (mut t, stream, gov) = native_fixture();
        let bytes = 1 << 20;
        let pattern: Vec<u8> = (0..bytes).map(|i| (i * 7 % 251) as u8).collect();
        let name = stream.context().name().unwrap();
        let resolved = PinnedKind::for_device(&name);
        assert_eq!(t.pinned_default(), resolved, "{name}");
        let default = t.alloc_host(bytes, request()).unwrap();
        assert_eq!(default.pinned_kind(), Some(resolved), "{name}");
        let mut default_flags: std::ffi::c_uint = u32::MAX;
        // SAFETY: the pointer is a live cuMemHostAlloc allocation owned by `default` for the
        // duration of the call; the query writes only `default_flags`.
        unsafe {
            sys::cuMemHostGetFlags(
                &mut default_flags,
                default.bytes().unwrap().as_ptr() as *mut _,
            )
            .result()
            .unwrap()
        };
        assert_eq!(
            default_flags & sys::CU_MEMHOSTALLOC_WRITECOMBINED,
            resolved.host_alloc_flags(),
            "{name}: default lease driver flags {default_flags}"
        );
        drop(default);
        for kind in [PinnedKind::WriteCombined, PinnedKind::Cached] {
            let mut lease = t.alloc_host_kind(bytes, request(), kind).unwrap();
            assert_eq!(lease.pinned_kind(), Some(kind));
            assert_eq!(lease.bytes().unwrap(), vec![0u8; bytes].as_slice());
            let mut flags: std::ffi::c_uint = u32::MAX;
            // SAFETY: the pointer is a live cuMemHostAlloc allocation owned by `lease` for the
            // duration of the call; the query writes only `flags`.
            unsafe {
                sys::cuMemHostGetFlags(&mut flags, lease.bytes().unwrap().as_ptr() as *mut _)
                    .result()
                    .unwrap()
            };
            // The driver's record carries the arm's bit; on a UVA platform it also reports
            // `CU_MEMHOSTALLOC_DEVICEMAP` (every pinned allocation is device-mapped), so the
            // write-combined arm reads back 6 and the cached arm 2 (day 13, first sitting).
            assert_eq!(
                flags & sys::CU_MEMHOSTALLOC_WRITECOMBINED,
                kind.host_alloc_flags(),
                "{kind:?}: driver flags {flags}"
            );
            assert_eq!(flags & sys::CU_MEMHOSTALLOC_PORTABLE, 0, "{kind:?}");
            lease.write(&pattern).unwrap();
            assert_eq!(lease.bytes().unwrap(), pattern.as_slice());
            assert_eq!(checksum(lease.bytes().unwrap()), checksum(&pattern));
            assert_eq!(gov.borrow().used().pinned, bytes as u64);
            drop(lease);
            assert_eq!(gov.borrow().used().pinned, 0);
        }
    }
    /// Integ38 (lead review of WP-A day 22): the Block arm's host wait covers the receipt event,
    /// not only the items' events. A D2D item lands only with its lanes (`progress`), and the
    /// receipt's D2H is recorded on the copy stream after the last item's event; a `synchronize`
    /// that returned on the items alone could hand `capture_landed` a landed copy with unread
    /// lanes, and the settle would latch the tier for a batch that had landed. The receipt wait
    /// sits between the item waits and the `unknown` reset, and both run through one wait path.
    #[test]
    fn integ38_synchronize_waits_on_the_receipt_event_after_the_items() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        let at = body.find("pub fn synchronize(").unwrap();
        let sync = &body[at..at + body[at..].find("\n    }\n").unwrap()];
        let items = sync
            .find("cuda(item.event.as_ref().ok_or(Error::Quarantined)?.synchronize())?;")
            .expect("the items' host wait");
        let receipt = sync
            .find("cuda(r.event.as_ref().ok_or(Error::Quarantined)?.synchronize())?;")
            .expect("the receipt's host wait");
        let reset = sync.find("e.unknown = false;").unwrap();
        let progress = sync.find("self.progress(ticket)").unwrap();
        assert!(items < receipt && receipt < reset && reset < progress);
        assert!(sync.contains("if let Some(r) = &e.receipt {"));
        // `seal_receipt` records that event after the lanes' D2H; a seal that failed leaves it
        // `None` and marks the entry unknown, which the wait reads as `Quarantined`.
        let seal = body.find("fn seal_receipt(").unwrap();
        let seal_body = &body[seal..seal + body[seal..].find("\n    }\n").unwrap()];
        let dtoh = seal_body
            .find("copy.memcpy_dtoh(&scratch.lanes, &mut scratch.pinned)")
            .unwrap();
        let ev = seal_body
            .find("scratch.event = Some(cuda(copy.record_event(None))?);")
            .unwrap();
        assert!(dtoh < ev);
        assert!(seal_body.contains("entry.unknown = true;"));
        // One host-wait path over recorded events: the items' and the receipt's, nothing else.
        assert_eq!(
            body.matches("event.as_ref().ok_or(Error::Quarantined)?.synchronize())?;")
                .count(),
            5,
            "items, receipt, (day 30) D2H spans and (day 32) H2D spans, (day 40) the H2D span receipt"
        );
    }

    /// WP-A day 30 (memra#536 Move 2 owed item 1, the D2H half): the span class's order, by
    /// source. The copy stream waits on a fresh owner-stream event before the first enqueue; each
    /// span's event is recorded after its enqueue; `progress` folds the span events into
    /// `producer_done`; `retire` is `Busy` while a span is untaken; a span becomes readable only
    /// in `take_d2h_spans`, after the landing; an unretired entry's drop forgets its spans.
    #[test]
    fn d2h_span_rules_are_as_stated() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        let fn_body = |name: &str| {
            let at = body.find(name).unwrap_or_else(|| panic!("{name} missing"));
            &body[at..at + body[at..].find("\n    }\n").unwrap()]
        };
        let submit = fn_body("pub fn submit_d2h_spans(");
        let at =
            |b: &str, needle: &str| b.find(needle).unwrap_or_else(|| panic!("{needle} missing"));
        assert!(
            at(submit, "let fence = cuda(self.stream.record_event(None))?;")
                < at(submit, "cuda(copy.wait(&fence))?;")
        );
        assert!(
            at(submit, "cuda(copy.wait(&fence))?;")
                < at(submit, "enqueue_from_device_f32(&span.source, &copy)")
        );
        assert!(
            at(submit, "enqueue_from_device_f32(&span.source, &copy)")
                < at(submit, "copy.record_event(None).ok()")
        );
        assert!(
            at(submit, "Err(error) => return Err((error, spans)),")
                < at(submit, "for mut span in spans {")
        );
        assert!(submit.contains("e.unknown = true;"));
        assert!(!submit.contains("synchronize("), "no host wait at attach");
        let take = fn_body("pub fn take_d2h_spans(");
        assert!(
            at(take, "if !b.landed || !e.completion.producer_done {")
                < at(take, "span.destination.mark_landed()")
        );
        assert_eq!(
            body.matches(".mark_landed()").count(),
            2,
            "one readable-making site per span direction (day 33: the H2D take)"
        );
        let progress = fn_body("fn progress(");
        assert!(progress.contains("e.completion.producer_done = spans_landed"));
        let retire = fn_body("fn retire(&mut self, ticket: &TransferTicket, consumer_done");
        assert!(at(retire, "if e.spans.is_some() {") < at(retire, "e.retired = true;"));
        let drop_at = body.find("impl Drop for Entry {").unwrap();
        let drop_body = &body[drop_at..drop_at + body[drop_at..].find("\n}\n").unwrap()];
        assert!(drop_body.contains("std::mem::forget(self.spans.take());"));
    }

    /// WP-A day 30 (memra#536 Move 2 owed item 1, the D2H half; `memra_tier::conformance::
    /// d2h_span_batch`, `d2h_span_taken_on_the_items_landing_fails` and
    /// `d2h_span_enqueue_failure_quarantines`, on a card). One KV item D2H batch; three owned f32
    /// sources on the owner stream with patterns and three unwritten cached pinned destinations.
    /// A refused attach hands every span back and quarantines nothing; a 300 ms spin on the copy
    /// stream between the item and the spans holds the spans, so the item lands while they run:
    /// the batch is not landed, the take is `NotReady`, the retire `Busy`. After the host wait the
    /// batch lands, the retire waits for the take, the spans come back once with the patterns bit
    /// for bit, and the ticket retires and is acknowledged. A second batch with an injected second
    /// enqueue failure is quarantined and hands nothing back.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn d2h_span_batch_lands_with_its_ticket_on_the_copy_stream() {
        use memra_tier::tier::governor::Governor;
        let ctx = cell_context();
        let stream = ctx.new_stream().unwrap();
        let cap = TierBudget {
            version: 1,
            device: vec![1 << 30],
            peer: vec![0],
            replicas: vec![0],
            pinned: 1 << 30,
            pageable: 1 << 30,
            staging: 1 << 30,
            loaders: 64,
            inflight: 64,
            nvme: 0,
        };
        let gov: SharedBudget = Rc::new(RefCell::new(
            Governor::new(cap, TierBudget::zero(1), 64, 0, Arc::new(|| 0)).unwrap(),
        ));
        let mut t = CudaTransfers::new_with_copy_stream(stream.clone(), gov).unwrap();
        let copy = t.copy_stream().unwrap().clone();
        let epochs = Epochs {
            state: 0,
            src_gen: 1,
            dst_gen: 1,
        };
        let kv_bytes = 1usize << 20;
        let kv_pattern: Vec<u8> = (0..kv_bytes).map(|i| (i * 11 % 251) as u8).collect();
        let submit_kv = |t: &mut CudaTransfers| {
            let mut plane = stream.alloc_zeros::<u8>(kv_bytes).unwrap();
            stream.memcpy_htod(&kv_pattern, &mut plane).unwrap();
            let keep = t.register_device(plane, 1, request()).unwrap();
            let device = t.retain_device(&keep).unwrap();
            let host = t.alloc_host(kv_bytes, request()).unwrap();
            let producer = t.record_producer(1).unwrap();
            let ticket = t
                .d2h(CopyOp {
                    host,
                    device,
                    bytes: kv_bytes as u64,
                    epochs,
                    producer_fence: Some(producer),
                })
                .unwrap();
            (ticket, producer, keep)
        };
        let lens = [3usize << 18, 5 << 18, 1 << 20];
        let patterns: Vec<Vec<f32>> = lens
            .iter()
            .enumerate()
            .map(|(k, &n)| (0..n).map(|i| (i as f32) * 0.5 + k as f32).collect())
            .collect();
        let clock = StepClock::new();
        let spans = |bad: bool| -> Vec<D2hSpan> {
            patterns
                .iter()
                .enumerate()
                .map(|(k, p)| {
                    let source = clock.time("htod-pageable", || stream.clone_htod(p).unwrap());
                    let len = if bad && k == 1 {
                        p.len() * 4 - 4
                    } else {
                        p.len() * 4
                    };
                    D2hSpan {
                        source,
                        destination: clock.time("pinned-alloc", || {
                            PinnedHostBuf::new_unwritten(len).unwrap()
                        }),
                    }
                })
                .collect()
        };
        let (ticket, producer, keep) = submit_kv(&mut t);
        // Rule 1: a refused attach hands every span back and quarantines nothing.
        let (error, back) = t.submit_d2h_spans(&ticket, spans(true)).unwrap_err();
        assert_eq!(error, Error::InvalidLayout);
        assert_eq!(back.len(), 3);
        assert!(t.poll(&ticket).is_ok());
        // Hold the spans behind a 300 ms spin on the copy stream (after the KV item's copy).
        let hold_at = clock.hold();
        clock.time("delay", || t.delay_on(&copy, 300_000_000).unwrap());
        let set = spans(false);
        clock.time("submit", || t.submit_d2h_spans(&ticket, set).unwrap());
        let set = spans(false);
        let (error, back) = clock.time("submit-busy", || {
            t.submit_d2h_spans(&ticket, set).unwrap_err()
        });
        assert_eq!(
            (error, back.len()),
            (Error::Busy, 3),
            "one span batch per ticket"
        );
        // Rule 2: the KV item lands while the spans run; the batch has not landed.
        let (c, reading) = poll_until_first_item_landed(&mut t, &ticket, hold_at, "d2h_span_batch");
        let steps = clock.line("d2h_span_batch");
        assert!(
            !c.producer_done,
            "a batch with a running span has not landed ({reading}) ({steps})"
        );
        assert_eq!(
            t.take_d2h_spans(&ticket).err(),
            Some(Error::NotReady),
            "the red arm"
        );
        assert_eq!(t.retire(&ticket, None), Err(Error::Busy));
        // Every event observed: landed; retire waits for the take (rule 4).
        t.synchronize(&ticket).unwrap();
        assert!(t.poll(&ticket).unwrap().producer_done);
        assert_eq!(t.retire(&ticket, None), Err(Error::Busy));
        let taken = t.take_d2h_spans(&ticket).unwrap();
        let landed = &taken.spans;
        assert_eq!(landed.len(), 3);
        for (span, p) in landed.iter().zip(&patterns) {
            assert_eq!(span.destination.len(), p.len() * 4);
            assert_eq!(
                span.destination.as_slice(),
                f32_bytes(p),
                "the span landed the source bit for bit"
            );
            assert_eq!(span.source.len(), p.len(), "the source comes back whole");
        }
        assert_eq!(
            t.take_d2h_spans(&ticket).err(),
            Some(Error::AlreadyReleased)
        );
        // Day 42 (`DAY42.md` design S2): the take hands out the batch's span receipt, unsealed: a
        // read is `Busy`; a seal with the wrong count or lengths refuses before any enqueue; the
        // seal enqueues the landed digests; a second seal is `Busy`; the read after the wait gives
        // each span's (source, landed) pair, both the four-lane program over the pattern, bitwise;
        // the receipt then leaves the engine.
        let id = taken
            .receipt
            .expect("a span receipt on an engine with the receipt kernels");
        assert_eq!(t.d2h_span_receipt(id).err(), Some(Error::Busy));
        let staging: Vec<&PinnedHostBuf> = landed.iter().map(|s| &s.destination).collect();
        assert_eq!(
            t.seal_d2h_span_receipt(id, &staging[..2]).err(),
            Some(Error::InvalidLayout)
        );
        let reversed: Vec<&PinnedHostBuf> = staging.iter().rev().copied().collect();
        assert_eq!(
            t.seal_d2h_span_receipt(id, &reversed).err(),
            Some(Error::InvalidLayout),
            "every length is the take's, in attach order"
        );
        t.seal_d2h_span_receipt(id, &staging).unwrap();
        assert_eq!(
            t.seal_d2h_span_receipt(id, &staging).err(),
            Some(Error::Busy)
        );
        let pairs = t.d2h_span_receipt_wait(id).unwrap();
        assert_eq!(pairs.len(), 3);
        for (k, ((source, landed), p)) in pairs.iter().zip(&patterns).enumerate() {
            let oracle = memra_tier::conformance::receipt_digest(f32_bytes(p));
            assert_eq!(
                *source, oracle,
                "span {k}: the source digest is the program"
            );
            assert_eq!(
                *landed, oracle,
                "span {k}: the landed digest is the program"
            );
        }
        assert_eq!(t.d2h_span_receipt(id).err(), Some(Error::UnknownTicket));
        drop(taken);
        t.retire_source(&ticket).unwrap();
        t.release_device(&keep).unwrap();
        let Destination::Host(host) = t.take_destination(&ticket, 0, epochs).unwrap() else {
            panic!("D2H destination is not host")
        };
        assert_eq!(host.bytes().unwrap(), kv_pattern.as_slice());
        let consumer = t.record_consumer(&ticket).unwrap();
        stream.synchronize().unwrap();
        t.retire(&ticket, Some(consumer)).unwrap();
        t.acknowledge(&ticket).unwrap();
        t.release_producer(producer).unwrap();
        // Day 42 (design S2), the red arm on a card: `span-flip-landed` flips one byte of span 0's
        // staging after its copy and before its event: the landed bytes carry it, span 0's landed
        // digest is the program over them, its source digest over the unflipped source; every
        // other span's pair agrees. The receipt is sealed here and abandoned before its read: it
        // waits in the reap list and frees once its event is observed.
        let settle = |t: &mut CudaTransfers, ticket: &TransferTicket, producer, keep| {
            t.retire_source(ticket).unwrap();
            t.release_device(&keep).unwrap();
            let _ = t.take_destination(ticket, 0, epochs).unwrap();
            let consumer = t.record_consumer(ticket).unwrap();
            stream.synchronize().unwrap();
            t.retire(ticket, Some(consumer)).unwrap();
            t.acknowledge(ticket).unwrap();
            t.release_producer(producer).unwrap();
        };
        let (ticket, producer, keep) = submit_kv(&mut t);
        t.inject_span_flip_landed();
        t.submit_d2h_spans(&ticket, spans(false)).unwrap();
        t.synchronize(&ticket).unwrap();
        let taken = t.take_d2h_spans(&ticket).unwrap();
        let id = taken.receipt.unwrap();
        let staging: Vec<&PinnedHostBuf> = taken.spans.iter().map(|s| &s.destination).collect();
        t.seal_d2h_span_receipt(id, &staging).unwrap();
        let pairs = t.d2h_span_receipt_wait(id).unwrap();
        for (k, ((source, landed), (span, p))) in pairs
            .iter()
            .zip(taken.spans.iter().zip(&patterns))
            .enumerate()
        {
            let oracle = memra_tier::conformance::receipt_digest(f32_bytes(p));
            assert_eq!(*source, oracle, "span {k}: the source digest");
            let landed_now = memra_tier::conformance::receipt_digest(span.destination.as_slice());
            assert_eq!(
                *landed, landed_now,
                "span {k}: the landed digest is the landed bytes'"
            );
            if k == 0 {
                assert_ne!(landed, source, "the flip is witnessed by span 0 alone");
            } else {
                assert_eq!(landed, source, "span {k} agrees");
            }
        }
        drop(taken);
        settle(&mut t, &ticket, producer, keep);
        // An abandoned sealed receipt waits for its event; a displaced unsealed one drops at once.
        let (ticket, producer, keep) = submit_kv(&mut t);
        t.submit_d2h_spans(&ticket, spans(false)).unwrap();
        t.synchronize(&ticket).unwrap();
        let taken = t.take_d2h_spans(&ticket).unwrap();
        let sealed = taken.receipt.unwrap();
        let staging: Vec<&PinnedHostBuf> = taken.spans.iter().map(|s| &s.destination).collect();
        t.seal_d2h_span_receipt(sealed, &staging).unwrap();
        t.d2h_span_receipt_abandon(sealed);
        assert_eq!(t.d2h_span_receipt(sealed).err(), Some(Error::UnknownTicket));
        copy.synchronize().unwrap();
        t.d2h_span_receipt_abandon(sealed);
        assert!(t.span_reap.is_empty(), "reaped once its event is observed");
        drop(taken);
        settle(&mut t, &ticket, producer, keep);
        let (ticket, producer, keep) = submit_kv(&mut t);
        t.submit_d2h_spans(&ticket, spans(false)).unwrap();
        t.synchronize(&ticket).unwrap();
        let first = t.take_d2h_spans(&ticket).unwrap();
        settle(&mut t, &ticket, producer, keep);
        let (ticket, producer, keep) = submit_kv(&mut t);
        t.submit_d2h_spans(&ticket, spans(false)).unwrap();
        t.synchronize(&ticket).unwrap();
        let second = t.take_d2h_spans(&ticket).unwrap();
        assert_eq!(
            t.d2h_span_receipt(first.receipt.unwrap()).err(),
            Some(Error::UnknownTicket),
            "a take displaces the earlier receipt"
        );
        assert!(t.span_reap.is_empty(), "an unsealed receipt drops at once");
        drop((first, second));
        settle(&mut t, &ticket, producer, keep);
        // Rule 5: an injected second-enqueue failure quarantines the ticket; nothing comes back.
        let (ticket, _producer, _keep) = submit_kv(&mut t);
        t.span_enqueue_fault = true;
        t.submit_d2h_spans(&ticket, spans(false)).unwrap();
        copy.synchronize().unwrap();
        assert_eq!(t.poll(&ticket).err(), Some(Error::Quarantined));
        assert_eq!(t.take_d2h_spans(&ticket).err(), Some(Error::Quarantined));
        assert!(t.retire(&ticket, None).is_err());
        stream.synchronize().unwrap();
    }
    /// WP-A day 32 (memra#536 Move 2 owed item 1, the H2D half): the H2D span class's order, by
    /// source. The copy stream waits on a fresh owner-stream event before the first enqueue; each
    /// span's event is recorded after its enqueue; `progress` folds the H2D span events into
    /// `producer_done`; `install_consumer_wait` installs the owner stream's wait on every span
    /// event; `take_h2d_spans` hands a destination out only once the batch landed AND that wait is
    /// installed; `retire` is `Busy` while an H2D span is untaken; an unretired entry's drop forgets
    /// its H2D spans. No host wait at attach.
    #[test]
    fn h2d_span_rules_are_as_stated() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        let fn_body = |name: &str| {
            let at = body.find(name).unwrap_or_else(|| panic!("{name} missing"));
            &body[at..at + body[at..].find("\n    }\n").unwrap()]
        };
        let at =
            |b: &str, needle: &str| b.find(needle).unwrap_or_else(|| panic!("{needle} missing"));
        // Day 33: both public attaches are the one body, `attach_h2d_spans`.
        assert!(
            fn_body("pub fn submit_h2d_spans(")
                .contains("self.attach_h2d_spans(ticket, spans, None)")
        );
        assert!(
            fn_body("pub fn submit_h2d_spans_filled(")
                .contains("self.attach_h2d_spans(ticket, spans, Some(fills))")
        );
        let submit = fn_body("fn attach_h2d_spans(");
        assert!(
            at(submit, "let fence = cuda(self.stream.record_event(None))?;")
                < at(submit, "cuda(copy.wait(&fence))?;")
        );
        // Day 33, rule 6: the fill is launched on the copy stream after its wait and before any copy.
        assert!(
            at(submit, "cuda(copy.wait(&fence))?;")
                < at(submit, "result::stream::launch_host_function(")
        );
        assert!(
            at(submit, "result::stream::launch_host_function(")
                < at(
                    submit,
                    "enqueue_to_device_f32_after_fill(&mut span.destination, &copy)"
                )
        );
        assert!(
            at(submit, "cuda(copy.wait(&fence))?;")
                < at(
                    submit,
                    "enqueue_to_device_f32(&mut span.destination, &copy)"
                )
        );
        assert!(
            at(
                submit,
                "enqueue_to_device_f32(&mut span.destination, &copy)"
            ) < at(submit, "copy.record_event(None).ok()")
        );
        assert!(
            at(submit, "Err(error) => return Err((error, spans, fills)),")
                < at(submit, "for mut span in spans {")
        );
        assert!(submit.contains("!(filled || s.source.is_written())"));
        assert!(submit.contains("fills[k].len().checked_mul(4) != Some(bytes)"));
        assert!(submit.contains("e.unknown = true;"));
        assert!(!submit.contains("synchronize("), "no host wait at attach");
        assert_eq!(
            body.matches("launch_host_function(").count(),
            1,
            "one fill launch site"
        );
        let task = fn_body("unsafe extern \"C\" fn span_fill_on_copy_stream(");
        assert!(
            task.contains("std::panic::catch_unwind("),
            "no unwind across the FFI"
        );
        let take = fn_body("pub fn take_h2d_spans(");
        assert!(
            at(
                take,
                "if !b.landed || !e.completion.producer_done || !b.fenced {"
            ) < at(take, "span.source.mark_landed()")
        );
        let install = fn_body("pub fn install_consumer_wait(");
        assert!(at(install, "if let Some(b) = &mut e.h2d_spans") < at(install, "b.fenced = true;"));
        assert_eq!(
            body.matches("b.fenced = true;").count(),
            1,
            "one fencing site"
        );
        let progress = fn_body("fn progress(");
        assert!(progress.contains("&& h2d_spans_landed"));
        let retire = fn_body("fn retire(&mut self, ticket: &TransferTicket, consumer_done");
        assert!(at(retire, "if e.h2d_spans.is_some() {") < at(retire, "e.retired = true;"));
        let drop_at = body.find("impl Drop for Entry {").unwrap();
        let drop_body = &body[drop_at..drop_at + body[drop_at..].find("\n}\n").unwrap()];
        assert!(drop_body.contains("std::mem::forget(self.h2d_spans.take());"));
    }

    /// WP-A day 32 (memra#536 Move 2 owed item 1, the H2D half; `memra_tier::conformance::
    /// h2d_span_batch`, `h2d_span_read_before_its_wait_is_unordered` and
    /// `h2d_span_enqueue_failure_quarantines`, on a card). One KV item H2D batch; three fully
    /// written cached pinned sources with patterns and three fresh owner-stream destinations. A
    /// refused attach (a length mismatch, then an unwritten source) hands every span back and
    /// quarantines nothing; a 300 ms spin on the copy stream between the item and the spans holds
    /// the spans, so the item lands while they run: the batch is not landed, the take is
    /// `NotReady`, the retire `Busy`. After the host wait the batch has landed and the take is
    /// STILL `NotReady` (no reader wait yet); after `install_consumer_wait` the spans come back
    /// once, the owner stream reads each destination bit for bit, and the ticket publishes,
    /// retires and is acknowledged with its KV item landed. A second batch with an injected
    /// second enqueue failure is quarantined and hands nothing back.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn h2d_span_batch_lands_with_its_ticket_on_the_copy_stream() {
        use memra_tier::tier::governor::Governor;
        let ctx = cell_context();
        let stream = ctx.new_stream().unwrap();
        let cap = TierBudget {
            version: 1,
            device: vec![1 << 30],
            peer: vec![0],
            replicas: vec![0],
            pinned: 1 << 30,
            pageable: 1 << 30,
            staging: 1 << 30,
            loaders: 64,
            inflight: 64,
            nvme: 0,
        };
        let gov: SharedBudget = Rc::new(RefCell::new(
            Governor::new(cap, TierBudget::zero(1), 64, 0, Arc::new(|| 0)).unwrap(),
        ));
        let mut t = CudaTransfers::new_with_copy_stream(stream.clone(), gov).unwrap();
        let copy = t.copy_stream().unwrap().clone();
        let epochs = Epochs {
            state: 0,
            src_gen: 1,
            dst_gen: 1,
        };
        let kv_bytes = 1usize << 20;
        let kv_pattern: Vec<u8> = (0..kv_bytes).map(|i| (i * 13 % 251) as u8).collect();
        let submit_kv = |t: &mut CudaTransfers| {
            let plane = stream.alloc_zeros::<u8>(kv_bytes).unwrap();
            let keep = t.register_device(plane, 1, request()).unwrap();
            let device = t.retain_device(&keep).unwrap();
            let mut host = t.alloc_host(kv_bytes, request()).unwrap();
            host.write(&kv_pattern).unwrap();
            let producer = t.record_producer(1).unwrap();
            let ticket = t
                .h2d(CopyOp {
                    host,
                    device,
                    bytes: kv_bytes as u64,
                    epochs,
                    producer_fence: Some(producer),
                })
                .unwrap();
            (ticket, producer, keep)
        };
        let lens = [3usize << 18, 5 << 18, 1 << 20];
        let patterns: Vec<Vec<f32>> = lens
            .iter()
            .enumerate()
            .map(|(k, &n)| (0..n).map(|i| (i as f32) * 0.25 - k as f32).collect())
            .collect();
        // `bad`: 1 makes span 1's destination one f32 short; 2 leaves span 1's source unwritten.
        let clock = StepClock::new();
        let spans = |bad: u8| -> Vec<H2dSpan> {
            patterns
                .iter()
                .enumerate()
                .map(|(k, p)| {
                    let mut source = clock.time("pinned-alloc", || {
                        PinnedHostBuf::new_unwritten(p.len() * 4).unwrap()
                    });
                    if !(bad == 2 && k == 1) {
                        clock.time("pinned-write", || {
                            source.copy_from_slice(f32_bytes(p)).unwrap()
                        });
                    }
                    let n = if bad == 1 && k == 1 {
                        p.len() - 1
                    } else {
                        p.len()
                    };
                    H2dSpan {
                        source,
                        destination: clock
                            .time("device-alloc", || stream.alloc_zeros::<f32>(n).unwrap()),
                    }
                })
                .collect()
        };
        let (ticket, producer, keep) = submit_kv(&mut t);
        // Rule 1: a refused attach hands every span back and quarantines nothing.
        for bad in [1, 2] {
            let (error, back) = t.submit_h2d_spans(&ticket, spans(bad)).unwrap_err();
            assert_eq!((error, back.len()), (Error::InvalidLayout, 3), "bad={bad}");
            assert!(t.poll(&ticket).is_ok());
        }
        // Hold the spans behind a 300 ms spin on the copy stream (after the KV item's copy).
        let hold_at = clock.hold();
        clock.time("delay", || t.delay_on(&copy, 300_000_000).unwrap());
        let set = spans(0);
        clock.time("submit", || t.submit_h2d_spans(&ticket, set).unwrap());
        let set = spans(0);
        let (error, back) = clock.time("submit-busy", || {
            t.submit_h2d_spans(&ticket, set).unwrap_err()
        });
        assert_eq!(
            (error, back.len()),
            (Error::Busy, 3),
            "one span batch per ticket"
        );
        // Rule 2: the KV item lands while the spans run; the batch has not landed.
        let (c, reading) = poll_until_first_item_landed(&mut t, &ticket, hold_at, "h2d_span_batch");
        let steps = clock.line("h2d_span_batch");
        assert!(
            !c.producer_done,
            "a batch with a running span has not landed ({reading}) ({steps})"
        );
        assert_eq!(t.take_h2d_spans(&ticket).err(), Some(Error::NotReady));
        assert_eq!(t.retire(&ticket, None), Err(Error::Busy));
        // Rule 3: landed is not readable; the take waits for the reader wait.
        t.synchronize(&ticket).unwrap();
        assert!(t.poll(&ticket).unwrap().producer_done);
        assert_eq!(
            t.take_h2d_spans(&ticket).err(),
            Some(Error::NotReady),
            "the red arm: no destination before the owner stream's wait"
        );
        assert_eq!(t.retire(&ticket, None), Err(Error::Busy));
        t.install_consumer_wait(&ticket).unwrap();
        let landed = t.take_h2d_spans(&ticket).unwrap();
        assert_eq!(landed.len(), 3);
        for (l, p) in landed.iter().zip(&patterns) {
            let span = &l.span;
            assert_eq!(
                span.source.as_slice(),
                f32_bytes(p),
                "the staging comes back whole"
            );
            assert_eq!(
                stream.clone_dtoh(&span.destination).unwrap(),
                *p,
                "the owner stream reads the span's destination bit for bit"
            );
            // Day 40 (design S): the destination digest is the program over the plane, bitwise.
            assert_eq!(
                l.destination_digest,
                Some(memra_tier::conformance::receipt_digest(f32_bytes(p)))
            );
        }
        assert_eq!(
            t.take_h2d_spans(&ticket).err(),
            Some(Error::AlreadyReleased)
        );
        // Rule 4: the KV item publishes and the ticket retires and is acknowledged as before.
        t.ready_view(&ticket, 0, epochs).unwrap();
        let consumer = t.record_consumer(&ticket).unwrap();
        stream.synchronize().unwrap();
        t.retire_source(&ticket).unwrap();
        t.release_producer(producer).unwrap();
        t.retire(&ticket, Some(consumer)).unwrap();
        t.acknowledge(&ticket).unwrap();
        let plane = t.take_plane(&keep).unwrap().into_pooled().unwrap();
        assert_eq!(stream.clone_dtoh(&plane).unwrap(), kv_pattern);
        // Rule 5: an injected second-enqueue failure quarantines the ticket; nothing comes back.
        let (ticket, _producer, _keep) = submit_kv(&mut t);
        t.span_enqueue_fault = true;
        t.submit_h2d_spans(&ticket, spans(0)).unwrap();
        copy.synchronize().unwrap();
        assert_eq!(t.poll(&ticket).err(), Some(Error::Quarantined));
        assert_eq!(
            t.install_consumer_wait(&ticket).err(),
            Some(Error::Quarantined)
        );
        assert_eq!(t.take_h2d_spans(&ticket).err(), Some(Error::Quarantined));
        assert!(t.retire(&ticket, None).is_err());
        stream.synchronize().unwrap();
    }
    /// WP-A day 33 (`memra_tier::conformance::h2d_span_fill_ordered_before_its_copy`, rule 6, on a
    /// card; `DAY33.md` design F). One KV item H2D batch; three FILLED spans: unwritten cached
    /// pinned staging sources, their resident planes as `Arc`s, fresh owner-stream destinations. A
    /// fill of the wrong length is refused whole (every span and fill back). A 300 ms copy-stream
    /// hold ahead of the attach keeps the fill and the copies queued while the KV item (submitted
    /// before the hold) lands: the batch has not landed and the take is `NotReady`. After the host
    /// wait and the reader wait the spans come back once, each staging source now readable with its
    /// plane's bytes and each destination bit for bit the plane (the fill ran before the copy); the
    /// ticket publishes, retires and is acknowledged. The owner thread never waited at the attach.
    /// A second filled batch with an injected second-enqueue fault is quarantined, nothing back.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn h2d_span_filled_batch_fills_on_the_copy_stream_before_its_copies() {
        use memra_tier::tier::governor::Governor;
        let ctx = cell_context();
        let stream = ctx.new_stream().unwrap();
        let cap = TierBudget {
            version: 1,
            device: vec![1 << 30],
            peer: vec![0],
            replicas: vec![0],
            pinned: 1 << 30,
            pageable: 1 << 30,
            staging: 1 << 30,
            loaders: 64,
            inflight: 64,
            nvme: 0,
        };
        let gov: SharedBudget = Rc::new(RefCell::new(
            Governor::new(cap, TierBudget::zero(1), 64, 0, Arc::new(|| 0)).unwrap(),
        ));
        let mut t = CudaTransfers::new_with_copy_stream(stream.clone(), gov).unwrap();
        let copy = t.copy_stream().unwrap().clone();
        let epochs = Epochs {
            state: 0,
            src_gen: 1,
            dst_gen: 1,
        };
        let kv_bytes = 1usize << 20;
        let kv_pattern: Vec<u8> = (0..kv_bytes).map(|i| (i * 17 % 251) as u8).collect();
        let submit_kv = |t: &mut CudaTransfers| {
            let plane = stream.alloc_zeros::<u8>(kv_bytes).unwrap();
            let keep = t.register_device(plane, 1, request()).unwrap();
            let device = t.retain_device(&keep).unwrap();
            let mut host = t.alloc_host(kv_bytes, request()).unwrap();
            host.write(&kv_pattern).unwrap();
            let producer = t.record_producer(1).unwrap();
            let ticket = t
                .h2d(CopyOp {
                    host,
                    device,
                    bytes: kv_bytes as u64,
                    epochs,
                    producer_fence: Some(producer),
                })
                .unwrap();
            (ticket, producer, keep)
        };
        let lens = [3usize << 18, 5 << 18, 1 << 20];
        // Day 39 (design T): the fill of these 12 MiB runs on three threads, and its second share
        // starts inside the second plane.
        t.fill_threads = 3;
        let fill_bytes: Vec<usize> = lens.iter().map(|n| n * 4).collect();
        let cut = super::fill_shares(
            &fill_bytes,
            super::fill_threads_for_bytes(t.fill_threads, fill_bytes.iter().sum()),
        );
        assert_eq!(cut.len(), 3);
        assert_eq!(cut[1][0], (1, 1 << 20, 4 << 20));
        let planes: Vec<Arc<Vec<f32>>> = lens
            .iter()
            .enumerate()
            .map(|(k, &n)| {
                Arc::new(
                    (0..n)
                        .map(|i| (i as f32) * 0.125 + 7.0 * k as f32)
                        .collect(),
                )
            })
            .collect();
        let clock = StepClock::new();
        let spans = || -> Vec<H2dSpan> {
            planes
                .iter()
                .map(|p| H2dSpan {
                    source: clock.time("pinned-alloc", || {
                        PinnedHostBuf::new_unwritten(p.len() * 4).unwrap()
                    }),
                    destination: clock.time("device-alloc", || {
                        stream.alloc_zeros::<f32>(p.len()).unwrap()
                    }),
                })
                .collect()
        };
        let (ticket, producer, keep) = submit_kv(&mut t);
        // Rule 1 for the filled attach: a fill one f32 short is refused whole.
        let mut bad = planes.clone();
        bad[1] = Arc::new(bad[1][1..].to_vec());
        let (error, back, fills) = t
            .submit_h2d_spans_filled(&ticket, spans(), bad)
            .unwrap_err();
        assert_eq!(
            (error, back.len(), fills.len()),
            (Error::InvalidLayout, 3, 3)
        );
        assert!(
            t.poll(&ticket).is_ok(),
            "a refusal before enqueue quarantines nothing"
        );
        // Hold the copy stream 300 ms (after the KV item's copy), then attach: the fill and every
        // span copy queue behind the hold.
        let hold_at = clock.hold();
        clock.time("delay", || t.delay_on(&copy, 300_000_000).unwrap());
        let attached_at = std::time::Instant::now();
        let set = spans();
        clock.time("submit", || {
            t.submit_h2d_spans_filled(&ticket, set, planes.clone())
                .map_err(|(e, _, _)| e)
                .unwrap()
        });
        let attach_ms = attached_at.elapsed().as_secs_f64() * 1e3;
        assert!(
            attach_ms < 100.0,
            "the owner thread does not wait for the fill at the attach ({attach_ms:.2} ms; {})",
            clock.line("h2d_span_filled_batch")
        );
        let (c, reading) =
            poll_until_first_item_landed(&mut t, &ticket, hold_at, "h2d_span_filled_batch");
        let steps = clock.line("h2d_span_filled_batch");
        assert!(
            !c.producer_done,
            "the fill and the copies are still queued ({reading}) ({steps})"
        );
        assert_eq!(t.take_h2d_spans(&ticket).err(), Some(Error::NotReady));
        t.synchronize(&ticket).unwrap();
        assert!(t.poll(&ticket).unwrap().producer_done);
        assert_eq!(
            t.take_h2d_spans(&ticket).err(),
            Some(Error::NotReady),
            "no destination before the owner stream's wait"
        );
        t.install_consumer_wait(&ticket).unwrap();
        let landed = t.take_h2d_spans(&ticket).unwrap();
        assert_eq!(landed.len(), 3);
        for (l, p) in landed.iter().zip(&planes) {
            let span = &l.span;
            assert_eq!(
                span.source.as_slice(),
                f32_bytes(p),
                "the fill wrote the staging with the plane's bytes"
            );
            assert_eq!(
                stream.clone_dtoh(&span.destination).unwrap(),
                **p,
                "the copy ran after the fill: the destination is the plane, bit for bit"
            );
            assert_eq!(
                l.destination_digest,
                Some(memra_tier::conformance::receipt_digest(f32_bytes(p)))
            );
        }
        assert_eq!(
            t.take_h2d_spans(&ticket).err(),
            Some(Error::AlreadyReleased)
        );
        t.ready_view(&ticket, 0, epochs).unwrap();
        let consumer = t.record_consumer(&ticket).unwrap();
        stream.synchronize().unwrap();
        t.retire_source(&ticket).unwrap();
        t.release_producer(producer).unwrap();
        t.retire(&ticket, Some(consumer)).unwrap();
        t.acknowledge(&ticket).unwrap();
        let plane = t.take_plane(&keep).unwrap().into_pooled().unwrap();
        assert_eq!(stream.clone_dtoh(&plane).unwrap(), kv_pattern);
        // Rule 5 for the filled attach: an injected second-enqueue failure quarantines.
        let (ticket, _producer, _keep) = submit_kv(&mut t);
        t.span_enqueue_fault = true;
        t.submit_h2d_spans_filled(&ticket, spans(), planes.clone())
            .map_err(|(e, _, _)| e)
            .unwrap();
        copy.synchronize().unwrap();
        assert_eq!(t.poll(&ticket).err(), Some(Error::Quarantined));
        assert_eq!(t.take_h2d_spans(&ticket).err(), Some(Error::Quarantined));
        assert!(t.retire(&ticket, None).is_err());
        stream.synchronize().unwrap();
    }
    /// WP-A day 33, acceptance (d) on a card (`DAY33.md` design F item 3: the owner thread waits for
    /// nothing): a host function occupying the COPY stream must not hold the owner thread's CUDA
    /// work. A test host function sleeps 200 ms on the copy stream; meanwhile the owner thread, in
    /// the same context, allocates on its stream, copies to and from the device through pageable
    /// memory, records an event and synchronizes its stream, all timed. They must finish while the
    /// host function still sleeps (under 50 ms in all, the copy stream still busy), and device work
    /// queued on the copy stream behind the host function must wait for it (the order rule 6 relies
    /// on).
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn h2d_fill_host_function_does_not_hold_the_owner_thread() {
        unsafe extern "C" fn sleep_200ms(_: *mut std::ffi::c_void) {
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        let ctx = cell_context();
        let owner = ctx.new_stream().unwrap();
        let copy = ctx.new_stream().unwrap();
        let host_bytes: Vec<u8> = (0..(4usize << 20)).map(|i| (i % 253) as u8).collect();
        let mut behind = owner.alloc_zeros::<u8>(host_bytes.len()).unwrap();
        owner.synchronize().unwrap();
        let started = std::time::Instant::now();
        // SAFETY: a no-argument test host function; the driver calls it once.
        unsafe {
            result::stream::launch_host_function(
                copy.cu_stream(),
                sleep_200ms,
                std::ptr::null_mut(),
            )
        }
        .unwrap();
        // Device work on the copy stream queued behind the host function (a memset: no host
        // memory involved, so enqueueing it cannot itself wait).
        let landed = {
            copy.memset_zeros(&mut behind).unwrap();
            copy.record_event(None).unwrap()
        };
        let owner_t = std::time::Instant::now();
        let mut d = owner.alloc_zeros::<u8>(host_bytes.len()).unwrap();
        owner.memcpy_htod(&host_bytes, &mut d).unwrap();
        let back = owner.clone_dtoh(&d).unwrap();
        let ev = owner.record_event(None).unwrap();
        owner.synchronize().unwrap();
        ev.synchronize().unwrap();
        let owner_ms = owner_t.elapsed().as_secs_f64() * 1e3;
        let copy_busy = !event_done(&landed).unwrap();
        eprintln!(
            "owner-thread work while the copy stream's host function sleeps: {owner_ms:.2} ms; the \
             copy behind the host function still pending: {copy_busy}"
        );
        assert_eq!(back, host_bytes);
        assert!(
            owner_ms < 50.0,
            "the owner thread was held {owner_ms:.2} ms"
        );
        assert!(
            copy_busy,
            "the copy stream's work waits behind its host function"
        );
        landed.synchronize().unwrap();
        assert!(
            started.elapsed() >= std::time::Duration::from_millis(200),
            "the copy landed only after the host function returned"
        );
    }
    /// WP-A day 34 (`DAY34.md` design K): the deferred H2D checksum's order, by source. `progress`
    /// takes the deferred branch before its own checksum of a host source, and lands a deferred item
    /// only with its supplied digest; `defer_h2d_checksums` takes its views without the tracking
    /// event's host wait (no `bytes()`); `supply_h2d_checksums` checks every view before it takes
    /// any digest; `recover_source` refuses while a view is out.
    #[test]
    fn h2d_deferred_checksum_rules_are_as_stated() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        let fn_body = |name: &str| {
            let at = body.find(name).unwrap_or_else(|| panic!("{name} missing"));
            &body[at..at + body[at..].find("\n    }\n").unwrap()]
        };
        let at =
            |b: &str, needle: &str| b.find(needle).unwrap_or_else(|| panic!("{needle} missing"));
        let progress = fn_body("fn progress(");
        assert!(
            at(progress, "if let Some(d) = &e.deferred")
                < at(
                    progress,
                    "item.host.as_ref().ok_or(Error::AlreadyReleased)?.bytes()?"
                )
        );
        assert!(progress.contains("if let Some(sum) = d.supplied[i] {"));
        let defer = fn_body("pub fn defer_h2d_checksums(");
        assert!(defer.contains(".raw_view();"));
        assert!(!defer.contains(".bytes()"), "no host wait at the defer");
        let supply = fn_body("pub fn supply_h2d_checksums(");
        assert!(
            at(supply, "return Err(Error::AlreadyReleased);")
                < at(supply, "d.supplied[view.item as usize] = Some(sum);")
        );
        let recover = fn_body("pub fn recover_source(");
        assert!(recover.contains("if e.deferred.as_ref().is_some_and(|d| d.out > 0) {"));
    }

    /// WP-A day 34 (`memra_tier::conformance::h2d_deferred_checksum_lands_with_its_digests` and
    /// `h2d_deferred_checksum_mismatch_is_corrupt`, on a card). One KV item H2D batch on the copy
    /// stream, its checksum deferred: the copy lands and the item does not (no digest), the retire and
    /// the source's retire are `Busy`; the view's digest is taken on ANOTHER thread and supplied; the
    /// item lands, and the gate against the demote-time checksum (the pattern's `checksum`) opens; a
    /// second supply is refused. A second batch supplied a wrong digest lands and the gate reads
    /// `Corrupt`.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn h2d_deferred_checksum_lands_with_the_supplied_digests() {
        use memra_tier::tier::governor::Governor;
        let ctx = cell_context();
        let stream = ctx.new_stream().unwrap();
        let cap = TierBudget {
            version: 1,
            device: vec![1 << 30],
            peer: vec![0],
            replicas: vec![0],
            pinned: 1 << 30,
            pageable: 1 << 30,
            staging: 1 << 30,
            loaders: 64,
            inflight: 64,
            nvme: 0,
        };
        let gov: SharedBudget = Rc::new(RefCell::new(
            Governor::new(cap, TierBudget::zero(1), 64, 0, Arc::new(|| 0)).unwrap(),
        ));
        let mut t = CudaTransfers::new_with_copy_stream(stream.clone(), gov).unwrap();
        let epochs = Epochs {
            state: 0,
            src_gen: 1,
            dst_gen: 1,
        };
        let kv_bytes = 1usize << 20;
        let pattern: Vec<u8> = (0..kv_bytes).map(|i| (i * 29 % 251) as u8).collect();
        let submit = |t: &mut CudaTransfers| {
            let plane = stream.alloc_zeros::<u8>(kv_bytes).unwrap();
            let keep = t.register_device(plane, 1, request()).unwrap();
            let device = t.retain_device(&keep).unwrap();
            let mut host = t.alloc_host(kv_bytes, request()).unwrap();
            host.write(&pattern).unwrap();
            let producer = t.record_producer(1).unwrap();
            let ticket = t
                .h2d(CopyOp {
                    host,
                    device,
                    bytes: kv_bytes as u64,
                    epochs,
                    producer_fence: Some(producer),
                })
                .unwrap();
            (ticket, producer, keep)
        };
        let receipt = vec![vec![SegmentExpectation {
            valid_bytes: kv_bytes as u64,
            io_bytes: kv_bytes as u64,
            checksum: checksum(&pattern),
        }]];
        let (ticket, producer, keep) = submit(&mut t);
        let views = t.defer_h2d_checksums(&ticket).unwrap();
        assert_eq!(views.len(), 1);
        assert_eq!(t.defer_h2d_checksums(&ticket).err(), Some(Error::Busy));
        t.synchronize(&ticket).unwrap();
        let c = t.poll(&ticket).unwrap();
        assert_eq!(
            c.items[0].segments[0].status,
            ItemStatus::Complete,
            "the copy landed"
        );
        assert!(!c.producer_done, "the item lands only with its digest");
        assert_eq!(t.retire_source(&ticket).err(), Some(Error::Busy));
        assert_eq!(t.retire(&ticket, None).err(), Some(Error::Busy));
        // The digest on another thread, as the hash helper takes it.
        let hashed: Vec<(H2dSourceView, Digest)> = std::thread::spawn(move || {
            views
                .into_iter()
                .map(|v| {
                    let d = v.digest();
                    (v, d)
                })
                .collect()
        })
        .join()
        .unwrap();
        t.supply_h2d_checksums(&ticket, hashed).unwrap();
        let c = t.poll(&ticket).unwrap();
        assert!(c.producer_done);
        assert!(
            c.require(&ticket, &receipt, false).is_ok(),
            "the supplied digest is the demote-time checksum"
        );
        t.install_consumer_wait(&ticket).unwrap();
        t.ready_view(&ticket, 0, epochs).unwrap();
        let consumer = t.record_consumer(&ticket).unwrap();
        stream.synchronize().unwrap();
        t.retire_source(&ticket).unwrap();
        t.release_producer(producer).unwrap();
        t.retire(&ticket, Some(consumer)).unwrap();
        t.acknowledge(&ticket).unwrap();
        let plane = t.take_plane(&keep).unwrap().into_pooled().unwrap();
        assert_eq!(stream.clone_dtoh(&plane).unwrap(), pattern);
        // A wrong supplied digest: landed, and the gate refuses it.
        let (ticket, _producer, _keep) = submit(&mut t);
        let views = t.defer_h2d_checksums(&ticket).unwrap();
        t.synchronize(&ticket).unwrap();
        let wrong: Vec<(H2dSourceView, Digest)> = views
            .into_iter()
            .map(|v| {
                let mut d = v.digest();
                d[0] ^= 0xff;
                (v, d)
            })
            .collect();
        t.supply_h2d_checksums(&ticket, wrong).unwrap();
        let c = t.poll(&ticket).unwrap();
        assert!(c.producer_done);
        assert_eq!(c.require(&ticket, &receipt, false), Err(Error::Corrupt));
        stream.synchronize().unwrap();
    }
    fn f32_bytes(p: &[f32]) -> &[u8] {
        // SAFETY: an f32 slice is plain bytes of four times its length.
        unsafe { std::slice::from_raw_parts(p.as_ptr().cast(), p.len() * 4) }
    }
}
