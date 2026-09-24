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
use memra_tier::conformance::receipt_digest_from_lanes;
use memra_tier::{bank::SharedBudget, contracts::*};

/// The D2D receipt kernels (WP-A day 22, memra#536 Move 2 slice 3; `cu/tier_receipt.cu`): the
/// copy-stream digest whose CPU oracle is `memra_tier::conformance::receipt_digest`, and the
/// `d2d-delay` fault's spin. Loaded by `new_with_copy_stream` only; `new` has no D2D class.
const TIER_RECEIPT_FATBIN: &[u8] = include_bytes!(env!("MEMRA_TIER_RECEIPT_FATBIN"));
/// The `d2d-delay` fault's early-reader delay ahead of the copy (`inject_d2d_early_reader`).
pub const D2D_DELAY_FAULT_NS: u64 = 200_000_000;
/// WP-A day 38: the `d2h-delay` fault's receipt-stream spin ahead of a demote's receipt digest
/// (`inject_d2h_delay`, design G'), long enough for the next request to arrive in the copy phase.
pub const D2H_DELAY_FAULT_NS: u64 = 3_000_000_000;
use std::{
    cell::{Ref, RefCell},
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
        drop(self.backing.take());
        drop(self.pin.take());
        if let Some(c) = self.charge.take()
            && self.governor.borrow_mut().release(&c).is_err()
        {
            std::mem::forget(c);
        }
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
    /// batch's receipt taken on the receipt stream over each item's DEVICE source (the program
    /// `checksum`, byte for byte), sealed by one receipt event; `None` keeps the owner-thread
    /// checksum of the landed bytes in `progress` (a batch on the owner stream).
    d2h_receipt: Option<D2hDeviceReceipt>,
}
/// WP-A day 38: one D2H batch's device receipt: the digests (32 bytes per item, the batch's item
/// order) on the device and their pinned twin, sealed by `scratch.event`; `timing` brackets the
/// digest kernels on the receipt stream (a log-only reading, never waited on).
struct D2hDeviceReceipt {
    scratch: ReceiptScratch,
    timing: Option<(CudaEvent, CudaEvent)>,
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
            // receipt stream's kernel and its pinned twin by the lanes' D2H; a free here (the lanes
            // on the owner stream, unordered with the receipt stream) could hand the memory out
            // under that write. A leak, never a free.
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
}
/// WP-A day 33 (design F): the fill of one filled H2D span batch, run by the copy stream's host
/// function: each resident plane (an owned `Arc`) and its span's staging target (a raw pointer
/// and a byte length, checked equal at the attach).
struct SpanFillTask {
    items: Vec<(Arc<Vec<f32>>, FillTarget)>,
}
/// A staging buffer's start and byte length (`PinnedHostBuf::fill_target`).
type FillTarget = (*mut u8, usize);
// SAFETY: the task moves once to the driver's callback thread; each target is written by that
// thread only, while the engine owns the buffer and nobody else touches it (`attach_h2d_spans`).
unsafe impl Send for SpanFillTask {}
impl SpanFillTask {
    fn run(&self) {
        for (plane, (dst, len)) in &self.items {
            let n = (*len).min(plane.len() * 4);
            // SAFETY: `dst` is the start of an exclusive, live staging buffer of `len` bytes
            // (see `attach_h2d_spans`); `plane` holds `plane.len() * 4 >= n` initialized bytes;
            // the two do not overlap (a heap `Vec` and a pinned host allocation).
            unsafe { std::ptr::copy_nonoverlapping(plane.as_ptr().cast::<u8>(), *dst, n) };
        }
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
    /// WP-A day 38 (`DAY38.md` design G', section 5): the RECEIPT stream, a third stream of the same
    /// context created with the copy stream. A D2H batch's device receipt (the digest kernels, the
    /// lanes' D2H, the receipt event) runs here, concurrently with the item copies on the copy
    /// stream, so no other copy-stream consumer queues behind the kernel. `Some` exactly when `copy`
    /// is.
    receipt_stream: Option<Arc<CudaStream>>,
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
    /// D2H batch's receipt waits this many nanoseconds behind a receipt-stream spin. `None` in
    /// production.
    d2h_delay: Option<u64>,
    /// Test only (the native `d2h_span` and day-32 `h2d_span` cells): the next span batch's second
    /// enqueue fails, whichever direction it is.
    #[cfg(test)]
    span_enqueue_fault: bool,
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
        Ok(Self {
            stream: owner,
            copy: None,
            receipt_stream: None,
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
        t.receipt_stream = Some(cuda(t.stream.context().new_stream())?);
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
        t.receipt = Some(ReceiptKernels {
            _module: module,
            digest,
            delay,
            sha256,
            flip,
        });
        Ok(t)
    }
    /// WP-A day 38 (`DAY38.md` design G): whether this engine takes a D2H batch's receipt on the
    /// receipt stream over the device sources (the copy and receipt streams with the receipt
    /// kernels, design G'), rather than
    /// on the owner thread over the landed bytes.
    pub fn d2h_receipts_on_device(&self) -> bool {
        self.copy.is_some() && self.receipt_stream.is_some() && self.receipt.is_some()
    }
    /// The `MEMRA_KV_HOST_FAULT=d2h-source-flip` fault (WP-A day 38, the device receipt's red arm):
    /// the NEXT device-receipt D2H batch flips one byte of its first accepted item's source on the
    /// receipt stream after the digest, and the batch's copies wait on it, so the landed bytes
    /// differ from the
    /// receipt and the caller's witness must refuse the image. One-shot; diagnostics only.
    pub fn inject_d2h_source_flip(&mut self) {
        self.source_flip = true;
    }
    /// The `MEMRA_KV_HOST_FAULT=d2h-delay` fault (WP-A day 38, the copy-phase park's red arm): the
    /// NEXT device-receipt D2H batch's receipt waits `delay_ns` behind a spin on the RECEIPT stream
    /// (design G'), so the batch lands, and its copy phase lasts, at least that long, while the copy
    /// stream's other work is not delayed. One-shot.
    pub fn inject_d2h_delay(&mut self, delay_ns: u64) {
        self.d2h_delay = Some(delay_ns);
    }
    /// WP-A day 38 (log only): the device receipt's digest kernels' receipt-stream time for `ticket`,
    /// read only if the end event already completed (never a host wait); `None` for a batch
    /// without a device receipt or while the kernels have not finished.
    pub fn d2h_receipt_gpu_ms(&self, ticket: &TransferTicket) -> Option<f32> {
        let (start, end) = self
            .entries
            .get(ticket)?
            .d2h_receipt
            .as_ref()?
            .timing
            .as_ref()?;
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
        // SAFETY: every byte is zero-filled through `as_mut_slice` before the backing is used.
        let mut pinned = cuda(unsafe {
            PinnedBacking::alloc(self.stream.context(), bytes, PinnedKind::Cached)
        })?;
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
    /// WP-A day 38 (`DAY38.md` design G', section 5; `memra_tier::conformance::d2h_device_receipt`):
    /// take a D2H batch's receipt on the RECEIPT stream, concurrently with the item copies the
    /// caller issues on the copy stream after this returns. The receipt stream waits on every
    /// accepted D2H item's producer event and on the lanes' zero-fill, `d2h_receipt_sha256` digests
    /// each accepted D2H item's device source (the program `checksum`, byte for byte; any other item
    /// hashes nothing and its lanes are never read), one D2H of the digests goes into the pinned
    /// twin, and the receipt event seals it. The digest and the copies both only read the sources,
    /// the ticket's registered device leases, untouched by any writer until the ticket retires its
    /// sources. The two faults ride here, one-shot: the delay spin AHEAD of the digest (the receipt,
    /// and so the landing, waits; the copy stream does not), and the one-byte flip after the digest
    /// with an event after it, returned for the caller's copy stream to wait on (the copy after the
    /// flip).
    fn seal_d2h_device_receipt(
        &mut self,
        ops: &[TransferOp<CudaPinnedLease>],
        errors: &[Option<Error>],
        scratch: ReceiptScratch,
    ) -> Result<(D2hDeviceReceipt, Option<CudaEvent>)> {
        // Move-then-match: from the first enqueue on, the receipt stream may write the lanes and
        // the twin, so on ANY error the scratch leaks (never a free under a pending write); only
        // the success exit hands it out.
        let mut scratch = std::mem::ManuallyDrop::new(scratch);
        let sealed = self.seal_d2h_device_receipt_into(ops, errors, &mut scratch);
        sealed.map(|(timing, flipped)| {
            (
                D2hDeviceReceipt {
                    scratch: std::mem::ManuallyDrop::into_inner(scratch),
                    timing: Some(timing),
                },
                flipped,
            )
        })
    }
    /// The enqueues of `seal_d2h_device_receipt`, into the caller's scratch.
    fn seal_d2h_device_receipt_into(
        &mut self,
        ops: &[TransferOp<CudaPinnedLease>],
        errors: &[Option<Error>],
        scratch: &mut ReceiptScratch,
    ) -> Result<((CudaEvent, CudaEvent), Option<CudaEvent>)> {
        let rs = self.receipt_stream.clone().ok_or(Error::Unsupported)?;
        let mut sources: Vec<(u64, u64)> = Vec::with_capacity(ops.len());
        let mut waited: Vec<u64> = Vec::new();
        for (op, error) in ops.iter().zip(errors) {
            match (op, error) {
                (TransferOp::D2h(o), None) => {
                    if let Some(f) = o.producer_fence
                        && !waited.contains(&f.sequence)
                    {
                        cuda(rs.wait(&self.producers[&f.sequence].1))?;
                        waited.push(f.sequence);
                    }
                    let backing = self.owner.resolve::<Rc<RefCell<KvPlane>>>(&o.device)?;
                    let plane = backing.borrow();
                    let view = plane.slice(..o.bytes as usize);
                    let (ptr, _record) = view.device_ptr(&rs);
                    sources.push((ptr, o.bytes));
                }
                _ => sources.push((0, 0)),
            }
        }
        cuda(rs.wait(&scratch.zeroed))?;
        if let Some(ns) = self.d2h_delay.take() {
            self.delay_on(&rs, ns)?;
        }
        let k = self.receipt.as_ref().ok_or(Error::Unsupported)?;
        // The two bracket events carry timing (the log-only kernel reading); every other event of
        // the engine keeps cudarc's timing-disabled default.
        let timed = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let start = cuda(rs.record_event(timed))?;
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
            let mut b = rs.launch_builder(&k.sha256);
            b.arg(&items).arg(&mut out);
            // SAFETY: documented FFI of `cu/tier_receipt.cu`: `d2h_receipt_sha256(ReceiptItems items,
            // u8* out)` reads `items.len[j]` bytes at each non-zero `items.ptr[j]` (live device
            // sources of this batch, ordered behind their producer fences above) and writes exactly
            // `32 * items.n` bytes of `out` (this chunk's slice of the lanes).
            cuda(unsafe { b.launch(cfg) })?;
        }
        let end = cuda(rs.record_event(timed))?;
        let mut flipped = None;
        if std::mem::take(&mut self.source_flip)
            && let Some(&(ptr, len)) = sources.iter().find(|(p, n)| *p != 0 && *n > 0)
        {
            let at = ptr + (len - 1).min(5);
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = rs.launch_builder(&k.flip);
            b.arg(&at);
            // SAFETY: documented FFI of `cu/tier_receipt.cu`: `tier_flip_byte(u8* p)` XORs the one
            // byte at `p`, inside the first accepted item's live source (offset below its length).
            cuda(unsafe { b.launch(cfg) })?;
            flipped = Some(cuda(rs.record_event(None))?);
        }
        cuda(rs.memcpy_dtoh(&scratch.lanes, &mut scratch.pinned))?;
        scratch.event = Some(cuda(rs.record_event(None))?);
        Ok(((start, end), flipped))
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
    /// Drain the copy stream (a no-op without one). Every path that hands device storage back
    /// to the pool (`release_device`, `take_plane`) drains both streams, so a source plane
    /// is never returned under a D2H that is still reading it.
    fn synchronize_copy_stream(&self) -> Result<()> {
        if let Some(copy) = &self.copy {
            cuda(copy.synchronize())?;
        }
        Ok(())
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
        request.bytes = TierBudget::zero(self.used().device.len());
        request.bytes.pinned = bytes as u64;
        let charge = self.governor.borrow_mut().reserve(&request)?;
        // Initialize through a raw pointer: constructing an uninitialized slice
        // would itself be invalid, even if immediately followed by fill().
        let allocation = (|| {
            let mut backing =
                cuda(unsafe { PinnedBacking::alloc(self.stream.context(), bytes, kind) })?;
            unsafe {
                cuda(backing.as_mut_ptr())?.write_bytes(0, bytes);
            }
            Ok(backing)
        })();
        match allocation {
            Ok(backing) => Ok(CudaPinnedLease {
                allocation: Rc::new(PinnedAllocation {
                    backing: Some(backing),
                    pin: Some(charge.pin()?),
                    charge: Some(charge),
                    governor: self.governor.clone(),
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
    pub fn release_device(&mut self, lease: &DeviceLease) -> Result<()> {
        self.check_thread()?;
        self.require_unbound(lease)?;
        cuda(self.stream.synchronize())?;
        self.synchronize_copy_stream()?;
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
    pub fn take_plane(&mut self, lease: &DeviceLease) -> Result<KvPlane> {
        self.check_thread()?;
        self.require_unbound(lease)?;
        cuda(self.stream.synchronize())?;
        self.synchronize_copy_stream()?;
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
        // WP-A day 38 (design G'): a device-receipt D2H batch lands only with its receipt, and the
        // receipt runs on the receipt stream CONCURRENTLY with the item copies, so the items can
        // complete first; the host wait covers the receipt event too, or a `Block` settle would read
        // the batch unlanded after its wait (the integ38 shape, for this class).
        if let Some(r) = &e.d2h_receipt {
            cuda(
                r.scratch
                    .event
                    .as_ref()
                    .ok_or(Error::Quarantined)?
                    .synchronize(),
            )?;
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
            let fence = cuda(self.stream.record_event(None))?;
            cuda(copy.wait(&fence))?;
            Ok(copy)
        })();
        let copy = match admitted {
            Ok(copy) => copy,
            Err(error) => return Err((error, spans)),
        };
        #[cfg(test)]
        let mut fault = std::mem::take(&mut self.span_enqueue_fault);
        let mut failed = false;
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
                enqueued.ok().and_then(|()| copy.record_event(None).ok())
            };
            failed |= event.is_none();
            slots.push((span, event));
        }
        let e = self.entries.get_mut(ticket).unwrap();
        e.spans = Some(SpanBatch {
            slots,
            landed: false,
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
    /// batch cannot retire until its spans are taken.
    pub fn take_d2h_spans(&mut self, ticket: &TransferTicket) -> Result<Vec<D2hSpan>> {
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
        Ok(b.slots
            .into_iter()
            .map(|(mut span, _)| {
                // SAFETY: `progress` observed this span's event complete (`b.landed`), recorded
                // on the copy stream after its enqueue: the bytes landed.
                unsafe { span.destination.mark_landed() };
                span
            })
            .collect())
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
            let fence = cuda(self.stream.record_event(None))?;
            cuda(copy.wait(&fence))?;
            Ok(copy)
        })();
        let copy = match admitted {
            Ok(copy) => copy,
            Err(error) => return Err((error, spans, fills)),
        };
        // Day 33 (design F): the fill, ONE host function on the copy stream ahead of every copy.
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
        let e = self.entries.get_mut(ticket).unwrap();
        e.h2d_spans = Some(H2dSpanBatch {
            slots,
            landed: false,
            fenced: false,
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
    pub fn take_h2d_spans(&mut self, ticket: &TransferTicket) -> Result<Vec<H2dSpan>> {
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
        let b = e.h2d_spans.take().unwrap();
        e.h2d_spans_taken = true;
        Ok(b.slots
            .into_iter()
            .map(|(mut span, _)| {
                // SAFETY: `progress` observed this span's event complete, recorded after its copy,
                // which stream order puts after its fill (day 33) when it had one: the source's bytes
                // are written (and, without a fill, were already).
                unsafe { span.source.mark_landed() };
                span
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
                match sealed {
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
            // expectation are its SOURCE's digest (the program `checksum` on the receipt stream). The
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
        // WP-A day 38 (design G'): the receipt digests every accepted D2H item's DEVICE source on
        // the receipt stream behind the producer fences, concurrently with the copies below, which
        // are unchanged. Any error from the first enqueue on may leave work in flight: the batch is
        // accepted and quarantined, never handed back.
        if let Some(scratch) = scratch {
            match self.seal_d2h_device_receipt(&ops, &errors, scratch) {
                Ok((receipt, flipped)) => {
                    entry.d2h_receipt = Some(receipt);
                    // Design G': under the `d2h-source-flip` fault only, the copies wait on the
                    // flip (the fault's definition, the copy after the flip); otherwise the copy
                    // stream waits on nothing of the receipt stream.
                    if let (Some(flipped), Some(copy)) = (flipped, &self.copy)
                        && cuda(copy.wait(&flipped)).is_err()
                    {
                        entry.unknown = true;
                    }
                }
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
        self.entries.remove(ticket);
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
    /// one context each in the pool below (the census pins the count).
    const NATIVE_CELLS: usize = 13;
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
        // Design G' (section 5): the whole receipt on the receipt stream, never the copy stream.
        assert!(
            sealer.contains("let rs = self.receipt_stream.clone().ok_or(Error::Unsupported)?;")
        );
        assert!(!sealer.contains("self.copy"));
        let digest = sealer.find("rs.launch_builder(&k.sha256)").unwrap();
        let flip = sealer.find("rs.launch_builder(&k.flip)").unwrap();
        let flipped = sealer
            .find("flipped = Some(cuda(rs.record_event(None))?);")
            .unwrap();
        let lanes = sealer
            .find("rs.memcpy_dtoh(&scratch.lanes, &mut scratch.pinned)")
            .unwrap();
        let event = sealer.find("scratch.event = Some(").unwrap();
        let delay = sealer.find("self.delay_on(&rs, ns)").unwrap();
        assert!(
            delay < digest,
            "the delay holds the receipt, ahead of the digest"
        );
        assert!(digest < flip && flip < flipped && flipped < lanes && lanes < event);
        assert!(sealer.find("rs.wait(&self.producers").unwrap() < digest);
        assert!(sealer.find("rs.wait(&scratch.zeroed)").unwrap() < digest);
        // The copy stream waits on the receipt stream only under the flip fault, before any copy.
        let flip_wait = submit
            .find("&& cuda(copy.wait(&flipped)).is_err()")
            .unwrap();
        assert!(seal < flip_wait && flip_wait < items);
        assert_eq!(submit.matches("copy.wait(").count(), 1);
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
        let progress = fn_body("    fn progress(");
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
    /// WP-A day 38 (`DAY38.md` design G, `memra_tier::conformance::d2h_device_receipt`, on a card).
    /// A three-item D2H batch (lengths 1 MiB, 1 MiB + 3 and 61,445 bytes) whose receipt waits behind a
    /// 300 ms receipt-stream hold: not landed while the digest waits, even after the copies completed;
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
        // Design G': the receipt runs on the receipt stream; the fault's 300 ms hold ahead of its
        // digest keeps the batch unlanded after the copies (not behind it, unless the flip is
        // armed) have completed: a copy alone is not a landing (the tier rule's rule 1, on a card).
        t.inject_d2h_delay(300_000_000);
        if flip {
            t.inject_d2h_source_flip();
        }
        let ticket = t.submit_batch(ops).map_err(|r| r.error).unwrap().ticket;
        if !flip {
            copy.synchronize().unwrap();
        }
        let c = t.poll(&ticket).unwrap();
        assert!(
            !c.producer_done,
            "not landed while the digest waits behind the hold"
        );
        assert!(
            t.d2h_receipt_gpu_ms(&ticket).is_none(),
            "no reading before the kernels ran"
        );
        // The `Block` settle's shape: ONE host wait on the ticket, then the completion. The receipt
        // still waits behind its 300 ms hold here, so the wait must cover it (design G').
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
    }
    /// WP-A day 37 (`DAY37.md` section 8): every native cell of this module owns a context of the
    /// pool (`cell_context()`), the pool is the module's only context constructor, and its size is
    /// the native cell count.
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
            4,
            "items, receipt, (day 30) D2H spans and (day 32) H2D spans"
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
        let landed = t.take_d2h_spans(&ticket).unwrap();
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
        for (span, p) in landed.iter().zip(&patterns) {
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
        for (span, p) in landed.iter().zip(&planes) {
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
