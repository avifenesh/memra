//! Native owner-stream H2D/D2H implementation of the frozen tier contract.
//! No worker may submit CUDA work. Unknown completion retains backing and quota.
use cudarc::driver::{
    CudaContext, CudaEvent, CudaSlice, CudaStream, DriverError, HostSlice, SyncOnDrop, result, sys,
};
use memra_kv::KvPlane;
use memra_tier::{bank::SharedBudget, contracts::*};
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

/// Where on-demand KV grows run on a device (WP-B day 37), by card class, keyed exactly as
/// [`PinnedKind::for_device`]: the stage-0 rule of `research/spill-b-20260919/DAY37.md` 1.5 per
/// class that has its receipt. The RTX 5090 class reads `helper` (DAY37 2.1: `GROW-PLACEMENT
/// extent=1 busy_p95_sum_us=11330.8 ... -> helper`). A class with no stage-0 receipt takes the
/// helper arm, which never holds the owner thread across a driver call's tail. Returns the
/// placement and the name of its source. The planes themselves are the serving default only
/// where the deciding cell read PROMOTE-ELIGIBLE, the full-power RTX PRO 6000 Blackwell
/// Workstation Edition (`docs/decisions/KV-VMM-SERVING-DEFAULT.md`); there the placement is
/// `inline`, the arm that cell qualified. The 5090's `helper` row is kept for its own cell.
pub fn kv_vmm_placement_for_device(name: &str) -> (memra_kv::VmmGrowPlacement, &'static str) {
    use crate::parallel::HardwareTarget;
    // DAY37 2.5: stage 0 on one RTX PRO 6000 Blackwell Workstation Edition read
    // `busy_p95_sum_us=55.3 ... -> inline`. The receipt covers the full-power Workstation card
    // only; the class's other variants (the Max-Q card, the Server Edition) keep the no-receipt
    // default until their own stage 0.
    if HardwareTarget::is_pro6000_workstation(name) {
        return (memra_kv::VmmGrowPlacement::Inline, "pro6000-ws-receipt");
    }
    match HardwareTarget::from_device_name(name) {
        Ok(HardwareTarget::Rtx5090) => (memra_kv::VmmGrowPlacement::Helper, "rtx5090-receipt"),
        Ok(HardwareTarget::RtxPro6000Blackwell) | Err(_) => {
            (memra_kv::VmmGrowPlacement::Helper, "no-receipt-default")
        }
    }
}

#[cfg(test)]
mod kv_vmm_placement_tests {
    use super::kv_vmm_placement_for_device as p;
    use memra_kv::VmmGrowPlacement::{Helper, Inline};

    /// Each class's stage-0 receipt selects its placement; a card without one takes the default.
    #[test]
    fn placement_follows_each_class_receipt() {
        assert_eq!(
            p("NVIDIA GeForce RTX 5090 Laptop GPU"),
            (Helper, "rtx5090-receipt")
        );
        assert_eq!(
            p("NVIDIA RTX PRO 6000 Blackwell Workstation Edition"),
            (Inline, "pro6000-ws-receipt")
        );
        assert_eq!(
            p("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"),
            (Helper, "no-receipt-default")
        );
        assert_eq!(
            p("NVIDIA RTX PRO 6000 Blackwell Server Edition"),
            (Helper, "no-receipt-default")
        );
        assert_eq!(p("NVIDIA H100 80GB HBM3"), (Helper, "no-receipt-default"));
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
    /// WP-A day 69 (`DAY69.md` design P.2): the pool's epoch when this lease was allocated; the
    /// backing parks at the drop only if no drain has run since.
    epoch: u64,
    /// WP-A day 70 (`DAY70.md` design Q.2): the pool's charge for this lease's tail (its backing's
    /// capacity minus its length), released at the drop before the backing parks.
    slack: Option<PoolSlack>,
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
            std::mem::forget(self.slack.take());
            return;
        }
        let backing = self.backing.take();
        drop(self.pin.take());
        if let Some(c) = self.charge.take()
            && self.governor.borrow_mut().release(&c).is_err()
        {
            std::mem::forget(c);
        }
        // WP-A day 70 (`DAY70.md` design Q.2): the tail's charge goes back to the pool first, so the
        // park below sees the pool's charge without it.
        if let Some(slack) = self.slack.take() {
            match &self.pool {
                Some(pool) => pool.release_slack(slack),
                None => std::mem::forget(slack),
            }
        }
        // WP-A day 63 (`DAY63.md` design L1.4): the backing parks in the pool (charged to the pool
        // tenant) instead of `cuMemFreeHost`, unless the pool is closed or full, or (day 69, P.2) a
        // drain ran since this lease was allocated; then it frees.
        match (backing, &self.pool) {
            (Some(b), Some(pool)) => pool.put(b, self.epoch),
            (b, _) => drop(b),
        }
    }
}

/// WP-A day 69 (`DAY69.md` design P.3): what the lease pool reads of a backing, its size class and
/// its kind, so the pool's program runs in a CPU test with a stand-in backing. Production pools
/// `PinnedBacking`.
pub trait PoolBacking {
    /// The allocation's size (the size class it was allocated at).
    fn class(&self) -> usize;
    /// The arm it was allocated under.
    fn pool_kind(&self) -> PinnedKind;
}
impl PoolBacking for PinnedBacking {
    fn class(&self) -> usize {
        self.capacity()
    }
    fn pool_kind(&self) -> PinnedKind {
        self.kind
    }
}

/// WP-A day 70 (`DAY70.md` design Q.1): a lease's request charge: its length (main's charge and
/// the host LRU's unit). What its backing holds beyond the length is the pool's (`PoolSlack`).
pub fn lease_charge(bytes: usize) -> u64 {
    bytes as u64
}

/// WP-A day 70 (`DAY70.md` design Q.2): a live lease's tail (its backing's capacity minus its
/// length), charged to the pool tenant within the pool's cap.
struct PoolSlack {
    charge: ChargedLease,
    bytes: u64,
}

/// WP-A day 70 (`DAY70.md` design Q.2): what the pool hands a new lease: an idle backing of its
/// class (the idle charge turned into the lease's smaller tail charge), a fresh backing at its class
/// (its tail charged within the pool's cap), or a fresh backing at its length (main's allocation).
enum PoolLease<B> {
    Pooled(B, Option<PoolSlack>),
    FreshClass(Option<PoolSlack>),
    FreshLength,
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
/// and when the engine drops: every idle backing frees, every pool charge releases. Day 69
/// (`DAY69.md` design P): drained at every tenant purge, and a backing leased before a drain never
/// parks again, so no backing that held bytes before a purge is handed to a later lease.
pub struct LeasePool<B: PoolBacking = PinnedBacking> {
    idle: RefCell<Vec<(B, ChargedLease)>>,
    idle_bytes: Cell<u64>,
    /// Day 70 (Q.2): the live leases' tails the pool is charged for; with `idle_bytes`, at or
    /// under `cap`.
    slack_bytes: Cell<u64>,
    cap: Cell<u64>,
    open: Cell<bool>,
    /// Day 69 (P.2): advanced by every drain; a lease records it at allocation.
    epoch: Cell<u64>,
    governor: SharedBudget,
    dimensions: usize,
    /// (taken from the pool, allocated fresh) since the engine was built (log only).
    counts: Cell<(u64, u64)>,
}
impl<B: PoolBacking> LeasePool<B> {
    fn new(governor: SharedBudget) -> Self {
        let dimensions = governor.borrow().used().device.len();
        Self {
            idle: RefCell::new(Vec::new()),
            idle_bytes: Cell::new(0),
            slack_bytes: Cell::new(0),
            cap: Cell::new(0),
            open: Cell::new(true),
            epoch: Cell::new(0),
            governor,
            dimensions,
            counts: Cell::new((0, 0)),
        }
    }
    /// Day 69 (P.2): the epoch a lease allocated now records.
    pub fn epoch(&self) -> u64 {
        self.epoch.get()
    }
    /// The pool tenant's digest (a domain disjoint from every tenant salt).
    pub fn tenant() -> [u8; 32] {
        digest("host-tier-lease-pool", b"pinned lease backings")
    }
    /// Day 70 (Q.2): a charge of `bytes` on the pinned dimension under the pool tenant.
    fn pool_request(&self, bytes: u64) -> BudgetRequest {
        let mut request = BudgetRequest {
            bytes: TierBudget::zero(self.dimensions),
            priority: Priority::Backup,
            deadline: Deadline(u64::MAX),
            tenant: Self::tenant(),
        };
        request.bytes.pinned = bytes;
        request
    }
    /// Day 70 (Q.2): what the pool is charged for now, idle backings plus live tails.
    pub fn held(&self) -> u64 {
        self.idle_bytes.get() + self.slack_bytes.get()
    }
    /// Day 70 (Q.2, Q.4): a tail of `bytes` charged to the pool when it is open and its cap has
    /// room (a zero tail needs no charge); `Err` otherwise, and never a request's refusal: the lease
    /// takes its length instead.
    fn reserve_slack(&self, bytes: u64) -> std::result::Result<Option<PoolSlack>, ()> {
        if bytes == 0 {
            return Ok(None);
        }
        if !self.open.get() || self.held() + bytes > self.cap.get() {
            return Err(());
        }
        let request = self.pool_request(bytes);
        let charge = self
            .governor
            .borrow_mut()
            .reserve(&request)
            .map_err(|_| ())?;
        self.slack_bytes.set(self.slack_bytes.get() + bytes);
        Ok(Some(PoolSlack { charge, bytes }))
    }
    /// Day 70 (Q.2): a dropped lease's tail charge back.
    fn release_slack(&self, slack: PoolSlack) {
        self.slack_bytes.set(self.slack_bytes.get() - slack.bytes);
        if self.governor.borrow_mut().release(&slack.charge).is_err() {
            std::mem::forget(slack.charge);
        }
    }
    /// Day 70 (Q.2, Q.4): the backing a new lease of `len` bytes (class `class`) gets: an idle one of
    /// its class and kind, a fresh one at its class while the pool can carry the tail, else a fresh
    /// one at its length.
    fn lease(&self, class: usize, len: usize, kind: PinnedKind) -> PoolLease<B> {
        let tail = (class - len) as u64;
        if let Some(backing) = self.take(class, kind) {
            match self.reserve_slack(tail) {
                Ok(slack) => return PoolLease::Pooled(backing, slack),
                // Not expected (the idle charge just released covers the tail): the backing frees.
                Err(()) => drop(backing),
            }
        }
        match self.reserve_slack(tail) {
            Ok(slack) => PoolLease::FreshClass(slack),
            Err(()) => PoolLease::FreshLength,
        }
    }
    /// An idle backing of exactly this class and kind, its pool charge released.
    fn take(&self, class: usize, kind: PinnedKind) -> Option<B> {
        let (backing, charge) = {
            let mut idle = self.idle.borrow_mut();
            let i = idle
                .iter()
                .position(|(b, _)| b.class() == class && b.pool_kind() == kind)?;
            idle.swap_remove(i)
        };
        self.idle_bytes.set(self.idle_bytes.get() - class as u64);
        if self.governor.borrow_mut().release(&charge).is_err() {
            std::mem::forget(charge);
        }
        Some(backing)
    }
    /// A dropped lease's backing: parked under a pool charge when the pool is open, has room, and
    /// the lease was allocated in the current epoch (day 69, P.2: no drain since), and (day 70, Q.2)
    /// the backing is class-sized (a lease allocated at its length never parks); freed otherwise.
    fn put(&self, backing: B, epoch: u64) {
        let class = backing.class() as u64;
        if !self.open.get()
            || epoch != self.epoch.get()
            || lease_class(class as usize) as u64 != class
            || self.held() + class > self.cap.get()
        {
            drop(backing);
            return;
        }
        let request = self.pool_request(class);
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
        self.drain()
    }
    /// WP-A day 69 (`DAY69.md` design P.1): the epoch advances first, so every lease allocated
    /// before this call frees at its drop instead of parking; then every idle backing frees (its
    /// drop, `cuMemFreeHost`, as before L') and its pool charge releases. The pool stays open.
    /// Returns (backings freed, bytes).
    pub fn drain(&self) -> (usize, u64) {
        self.epoch.set(self.epoch.get() + 1);
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
}
impl Drop for Entry {
    fn drop(&mut self) {
        if !self.retired {
            // Drop is not a DMA/consumer/graph fence. Keep every owned input and
            // its accounting alive, including after a submission error.
            std::mem::forget(std::mem::take(&mut self.items));
            std::mem::forget(self.charge.take());
        }
    }
}
/// Opaque graph-use retention. Drop only after graph execution/destruction retires.
pub struct GraphPin {
    _pins: Vec<Rc<()>>,
}

pub struct CudaTransfers {
    stream: Arc<CudaStream>,
    /// WP-A day 63 (`DAY63.md` design L1): the lease backing pool (closed at the latch and at drop).
    lease_pool: Rc<LeasePool>,
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
        Ok(Self {
            stream: owner,
            lease_pool: Rc::new(LeasePool::new(governor.clone())),
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
        })
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
        // WP-A day 70 (`DAY70.md` design Q, over day 63's L1.2 and L1.3): the lease is charged its
        // length (main's charge, the host LRU's unit); the pool hands it an idle backing of its
        // class and kind, or a fresh one zero-filled at its class, and is charged the backing's tail
        // within its cap; past the cap the lease is a fresh backing at its length. A pooled backing
        // is not refilled: every byte of it was initialized by its first lease, and nothing reads
        // past the new lease's length.
        let class = lease_class(bytes);
        request.bytes = TierBudget::zero(self.used().device.len());
        request.bytes.pinned = lease_charge(bytes);
        let charge = self.governor.borrow_mut().reserve(&request)?;
        let (pooled, slack, size) = match self.lease_pool.lease(class, bytes, kind) {
            PoolLease::Pooled(backing, slack) => (Some(backing), slack, class),
            PoolLease::FreshClass(slack) => (None, slack, class),
            PoolLease::FreshLength => (None, None, bytes),
        };
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
                        cuda(unsafe { PinnedBacking::alloc(self.stream.context(), size, kind) })?;
                    unsafe {
                        cuda(b.as_mut_ptr())?.write_bytes(0, size);
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
                    epoch: self.lease_pool.epoch(),
                    slack,
                }),
            }),
            Err(e) => {
                if let Some(slack) = slack {
                    self.lease_pool.release_slack(slack);
                }
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
    /// WP-A day 48 (`DAY48.md` design S4): `require_unbound` refuses while any unretired ticket
    /// names the lease, then the owner stream, which carries every copy, is drained.
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
    /// Transfer the complete typed owner, never a raw VMM CudaSlice. The release order of
    /// `release_device`.
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
                // No class of this engine submits a same-device item.
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
        e.unknown = false;
        self.progress(ticket)
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
            s.producer_done = true;
            s.status = ItemStatus::Complete;
            s.valid_bytes = item.bytes;
            s.checksum = Some(checksum(
                item.host.as_ref().ok_or(Error::AlreadyReleased)?.bytes()?,
            ));
            e.expected[i][0].checksum = s.checksum.unwrap();
        }
        e.completion.producer_done = e
            .completion
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
            let mut request = self.request();
            request.bytes.inflight = ops.len() as u64;
            let charge = self.governor.borrow_mut().reserve(&request)?;
            Ok((first, errors, charge))
        })();
        let (epochs, errors, charge) = match admission {
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
        };
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
                // Every copy and its completion event live on the owner stream (the day-16
                // program).
                let issue: Arc<CudaStream> = self.stream.clone();
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
                    // owner stream the wait is a same-stream no-op and the item is fenced at
                    // submit.
                    cuda(self.stream.wait(item.event.as_ref().unwrap()))?;
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

    /// WP-A day 37 (`DAY37.md` section 8, finding 5): the number of native cells in this module,
    /// one context each in the pool below (the census pins the count; the copy-stream cells left
    /// with the contracts door on 2026-11-04).
    const NATIVE_CELLS: usize = 5;
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
    /// WP-A day 48 (`DAY48.md` design S4): both release paths refuse a lease an unretired ticket
    /// names (`require_unbound` first), then drain the owner stream; a ticket retires only after
    /// its landing (`producer_done`). Source census, CPU.
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
            2,
            "set_len's bound and the pool's class read (`PoolBacking::class`, day 69)"
        );
        let alloc = &body[body.find("pub fn alloc_host_kind(").unwrap()..];
        let alloc = &alloc[..alloc.find("\n    }\n").unwrap()];
        // Day 70 (Q.1): the lease is charged its length; the pool its tail.
        assert!(alloc.contains("request.bytes.pinned = lease_charge(bytes);"));
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

    /// WP-A day 69 (`DAY69.md` design P, cell (a)): a stand-in backing that carries a tenant's
    /// bytes and counts its frees, so the pool's program runs without a card.
    struct StandIn {
        class: usize,
        bytes: Vec<u8>,
        freed: Rc<Cell<usize>>,
    }
    impl PoolBacking for StandIn {
        fn class(&self) -> usize {
            self.class
        }
        fn pool_kind(&self) -> PinnedKind {
            PinnedKind::Cached
        }
    }
    impl Drop for StandIn {
        fn drop(&mut self) {
            self.freed.set(self.freed.get() + 1);
        }
    }

    /// WP-A day 69 (`DAY69.md` design P, cell (a), CPU): after a drain no idle backing is left and
    /// every idle one is freed with its pool charge released; a backing leased before the drain and
    /// dropped after it frees instead of parking; one leased after the drain parks and is taken. So
    /// no backing that carried the purged tenant's bytes is ever handed to a later lease.
    #[test]
    fn day69_a_drain_frees_every_idle_backing_and_no_earlier_backing_parks_again() {
        use memra_tier::tier::governor::Governor;
        let cap = TierBudget {
            version: 1,
            device: vec![0],
            peer: vec![0],
            replicas: vec![0],
            pinned: 1 << 20,
            pageable: 0,
            staging: 0,
            loaders: 4,
            inflight: 4,
            nvme: 0,
        };
        let gov: SharedBudget = Rc::new(RefCell::new(
            Governor::new(cap, TierBudget::zero(1), 4, 0, Arc::new(|| 0)).unwrap(),
        ));
        let pool: LeasePool<StandIn> = LeasePool::new(gov.clone());
        pool.cap.set(1 << 16);
        let freed = Rc::new(Cell::new(0));
        let tenant = |class| StandIn {
            class,
            bytes: vec![0xa5; class],
            freed: freed.clone(),
        };
        // Two of the purged tenant's backings park (idle), a third is still leased at the purge.
        let before = pool.epoch();
        pool.put(tenant(4096), before);
        pool.put(tenant(8192), before);
        assert_eq!(pool.idle_bytes.get(), 12288);
        assert_eq!(gov.borrow().used().pinned, 12288, "the pool charge");
        let live = tenant(4096);
        // The purge's drain.
        assert_eq!(pool.drain(), (2, 12288));
        assert!(pool.idle.borrow().is_empty(), "no idle backing is left");
        assert_eq!(freed.get(), 2, "both idle backings freed");
        assert_eq!(gov.borrow().used().pinned, 0, "the pool charge released");
        assert!(pool.take(4096, PinnedKind::Cached).is_none());
        // The late drop: leased before the drain, dropped after it. It frees; it never parks.
        pool.put(live, before);
        assert_eq!(freed.get(), 3, "the earlier backing freed at its drop");
        assert!(pool.idle.borrow().is_empty());
        assert!(
            pool.take(4096, PinnedKind::Cached).is_none(),
            "nothing to hand out"
        );
        // A lease of the new epoch parks and is taken as before.
        let after = pool.epoch();
        assert_eq!(after, before + 1);
        pool.put(
            StandIn {
                class: 4096,
                bytes: vec![0x3c; 4096],
                freed: freed.clone(),
            },
            after,
        );
        let got = pool
            .take(4096, PinnedKind::Cached)
            .expect("a new-epoch backing parks");
        assert!(
            got.bytes.iter().all(|&b| b == 0x3c),
            "never the purged tenant's bytes"
        );
        assert_eq!(freed.get(), 3);
        drop(got);
        // close is the drain with the pool closed: the idle backing frees, a later put frees.
        pool.put(tenant(4096), after);
        assert_eq!(pool.idle_bytes.get(), 4096, "a current-epoch put parks");
        assert_eq!(pool.close(), (1, 4096));
        assert!(pool.epoch() > after);
        pool.put(tenant(4096), pool.epoch());
        assert_eq!(pool.idle_bytes.get(), 0, "a closed pool parks nothing");
        assert_eq!(
            freed.get(),
            6,
            "the taken one, the closed one and the late put freed too"
        );
        assert_eq!(gov.borrow().used().pinned, 0);
    }

    /// WP-A day 69 (`DAY69.md` design P, cell (b), CPU census): the allocation records the pool's
    /// epoch and the drop hands it to `put`; `put` parks only in the current epoch; `drain`
    /// advances the epoch before it frees and `close` is the drain with the pool closed.
    #[test]
    fn day69_the_pool_parks_only_in_its_epoch_and_the_drain_advances_it_first() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        let alloc = &body[body.find("pub fn alloc_host_kind(").unwrap()..];
        let alloc = &alloc[..alloc.find("\n    }\n").unwrap()];
        assert!(alloc.contains("epoch: self.lease_pool.epoch(),"));
        let drop_alloc = &body[body.find("impl Drop for PinnedAllocation {").unwrap()..];
        let drop_alloc = &drop_alloc[..drop_alloc.find("\n}\n").unwrap()];
        assert!(drop_alloc.contains("pool.put(b, self.epoch)"));
        let pool = &body[body.find("impl<B: PoolBacking> LeasePool<B> {").unwrap()..];
        let pool = &pool[..pool.find("\n}\n").unwrap() + 3];
        let put = &pool[pool
            .find("    fn put(&self, backing: B, epoch: u64) {")
            .unwrap()..];
        let put = &put[..put.find("\n    }\n").unwrap()];
        assert!(put.contains("|| epoch != self.epoch.get()"));
        let drain = &pool[pool
            .find("    pub fn drain(&self) -> (usize, u64) {")
            .unwrap()..];
        let drain = &drain[..drain.find("\n    }\n").unwrap()];
        let advance = drain.find("self.epoch.set(self.epoch.get() + 1);").unwrap();
        assert!(
            advance < drain.find("std::mem::take(").unwrap(),
            "the epoch first"
        );
        let close = &pool[pool
            .find("    pub fn close(&self) -> (usize, u64) {")
            .unwrap()..];
        let close = &close[..close.find("\n    }\n").unwrap()];
        assert!(close.contains("self.open.set(false);\n        self.drain()"));
    }

    /// WP-A day 70 (`DAY70.md` design Q): a pinned ledger of `pinned` bytes, and one lease through
    /// the pool the way `alloc_host_kind` takes it (census (c) pins the statements): the request
    /// charged `lease_charge`, then `LeasePool::lease`, with a stand-in at the size it names.
    fn q_ledger(pinned: usize) -> SharedBudget {
        use memra_tier::tier::governor::Governor;
        let cap = TierBudget {
            version: 1,
            device: vec![0],
            peer: vec![0],
            replicas: vec![0],
            pinned: pinned as u64,
            pageable: 0,
            staging: 0,
            loaders: 4,
            inflight: 4,
            nvme: 0,
        };
        Rc::new(RefCell::new(
            Governor::new(cap, TierBudget::zero(1), 4, 0, Arc::new(|| 0)).unwrap(),
        ))
    }
    struct QLease {
        charge: ChargedLease,
        slack: Option<PoolSlack>,
        backing: StandIn,
    }
    fn q_lease(
        gov: &SharedBudget,
        pool: &LeasePool<StandIn>,
        len: usize,
        freed: &Rc<Cell<usize>>,
    ) -> Result<QLease> {
        let mut r = request();
        r.bytes.pinned = lease_charge(len);
        let charge = gov.borrow_mut().reserve(&r)?;
        let class = lease_class(len);
        let fresh = |size| StandIn {
            class: size,
            bytes: vec![0; 0],
            freed: freed.clone(),
        };
        let (backing, slack) = match pool.lease(class, len, PinnedKind::Cached) {
            PoolLease::Pooled(b, slack) => (b, slack),
            PoolLease::FreshClass(slack) => (fresh(class), slack),
            PoolLease::FreshLength => (fresh(len), None),
        };
        Ok(QLease {
            charge,
            slack,
            backing,
        })
    }
    /// The allocation's drop, in its order: the length back, the tail back, then the park.
    fn q_drop(gov: &SharedBudget, pool: &LeasePool<StandIn>, lease: QLease, epoch: u64) {
        gov.borrow_mut().release(&lease.charge).unwrap();
        if let Some(slack) = lease.slack {
            pool.release_slack(slack);
        }
        pool.put(lease.backing, epoch);
    }

    /// WP-A day 70 (`DAY70.md` design Q, cell (a), CPU): DAY70 section 1's placement sequence
    /// through the real pool. Short-prefix residents at one budget of length, the pool full of
    /// other-class idle backings, the staging set's charge, then a short demote: every lease is
    /// admitted, the requests hold exactly main's charges, and the pool stays within its cap.
    #[test]
    fn day70_short_prefix_residents_and_a_full_pool_admit_every_lease_main_admits() {
        const MIB: usize = 1 << 20;
        let budget = 64 * MIB;
        let plane = 600 * 1024;
        let residents = budget / plane;
        let staging = 8 * MIB;
        let demote = 16;
        let freed = Rc::new(Cell::new(0));
        let gov = q_ledger(3 * budget);
        let pool: LeasePool<StandIn> = LeasePool::new(gov.clone());
        pool.cap.set(budget as u64);
        for _ in 0..16 {
            pool.put(
                StandIn {
                    class: 4 * MIB,
                    bytes: Vec::new(),
                    freed: freed.clone(),
                },
                pool.epoch(),
            );
        }
        assert_eq!(pool.held(), budget as u64, "the pool full of another class");
        let held: Vec<_> = (0..residents)
            .map(|_| q_lease(&gov, &pool, plane, &freed).unwrap())
            .collect();
        let mut r = request();
        r.bytes.pinned = staging as u64;
        let _staging = gov.borrow_mut().reserve(&r).unwrap();
        let demoted: Vec<_> = (0..demote)
            .map(|_| q_lease(&gov, &pool, plane, &freed))
            .collect();
        assert!(
            demoted.iter().all(|d| d.is_ok()),
            "every demote lease admitted"
        );
        assert!(pool.held() <= budget as u64, "the pool within its cap");
        let requests = gov.borrow().used().pinned - pool.held();
        // Main: the same sequence charged its length on the two-budget ledger, no pool.
        let main = q_ledger(2 * budget);
        let mut main_held = Vec::new();
        for _ in 0..residents + demote {
            let mut r = request();
            r.bytes.pinned = plane as u64;
            main_held.push(main.borrow_mut().reserve(&r).unwrap());
        }
        let mut r = request();
        r.bytes.pinned = staging as u64;
        main_held.push(main.borrow_mut().reserve(&r).unwrap());
        assert_eq!(
            requests,
            main.borrow().used().pinned,
            "the requests hold main's charges"
        );
        eprintln!(
            "DAY70 Q: {} demote leases admitted; requests {requests} B (main {} B); pool {} of {} B",
            demoted.len(),
            main.borrow().used().pinned,
            pool.held(),
            budget
        );
        drop((held, demoted));
    }

    /// WP-A day 70 (`DAY70.md` design Q, cell (b), CPU): the pool carries the tails within its cap.
    /// A fresh lease takes its class while its tail fits and its length past it; a pooled take
    /// turns the idle charge into the smaller tail charge; a length-sized backing never parks; the
    /// drop releases the tail before the park; a drain leaves live tails charged.
    #[test]
    fn day70_the_pool_carries_the_tail_within_its_cap() {
        let freed = Rc::new(Cell::new(0));
        let gov = q_ledger(1 << 20);
        let pool: LeasePool<StandIn> = LeasePool::new(gov.clone());
        pool.cap.set(16384);
        let pinned = || gov.borrow().used().pinned;
        // A fresh lease at its class, its tail on the pool.
        let a = q_lease(&gov, &pool, 3000, &freed).unwrap();
        assert_eq!(a.backing.class, 4096);
        assert_eq!((pool.held(), pinned()), (1096, 4096), "3000 + a 1096 tail");
        let e = pool.epoch();
        q_drop(&gov, &pool, a, e);
        assert_eq!(
            (pool.held(), pinned()),
            (4096, 4096),
            "parked: the idle class alone"
        );
        // A pooled take: the idle 4096 becomes a 1596 tail.
        let b = q_lease(&gov, &pool, 2500, &freed).unwrap();
        assert_eq!(b.backing.class, 4096);
        assert_eq!((pool.held(), pinned()), (1596, 4096));
        // Fresh class leases while their tails fit the cap; the next is at its length.
        let c = q_lease(&gov, &pool, 9000, &freed).unwrap();
        assert_eq!((c.backing.class, pool.held()), (16384, 1596 + 7384));
        let d = q_lease(&gov, &pool, 5000, &freed).unwrap();
        assert_eq!((d.backing.class, pool.held()), (8192, 8980 + 3192));
        let f = q_lease(&gov, &pool, 3000, &freed).unwrap();
        assert_eq!((f.backing.class, pool.held()), (4096, 12172 + 1096));
        let g = q_lease(&gov, &pool, 5000, &freed).unwrap();
        assert_eq!(
            g.backing.class, 5000,
            "the tail would pass the cap: the lease is its length"
        );
        assert_eq!(pool.held(), 13268, "no tail charged for it");
        assert_eq!(pinned(), 2500 + 9000 + 5000 + 3000 + 5000 + 13268);
        // A length-sized backing never parks.
        let before = freed.get();
        q_drop(&gov, &pool, g, e);
        assert_eq!(freed.get(), before + 1);
        assert_eq!(pool.idle_bytes.get(), 0);
        // A drain frees the idle and leaves the live tails.
        let fd = q_lease(&gov, &pool, 1024, &freed).unwrap();
        q_drop(&gov, &pool, fd, e);
        assert_eq!(pool.idle_bytes.get(), 1024);
        assert_eq!(pool.drain(), (1, 1024));
        assert_eq!(pool.slack_bytes.get(), 13268, "live tails stay charged");
        // Their drops after the drain: tails released, backings freed (the earlier epoch).
        for l in [b, c, d, f] {
            q_drop(&gov, &pool, l, e);
        }
        assert_eq!((pool.held(), pinned()), (0, 0));
        assert!(pool.idle.borrow().is_empty());
    }

    /// WP-A day 70 (`DAY70.md` design Q, cell (c), CPU census): the lease is charged its length
    /// through `lease_charge` and takes its backing through `LeasePool::lease`; the tail's charge
    /// names the pool tenant; `put` parks only a class-sized backing within the pool's held bytes;
    /// the allocation's drop releases the length, then the tail, then parks.
    #[test]
    fn day70_the_lease_is_its_length_and_the_pool_its_tail() {
        let src = include_str!("tier_transfer.rs");
        let body = &src[..src.find("#[cfg(test)]\nmod tests").unwrap()];
        let span = |sig: &str, end: &str| {
            let at = body.find(sig).unwrap_or_else(|| panic!("{sig}"));
            &body[at..at + body[at..].find(end).unwrap()]
        };
        assert!(span("pub fn lease_charge(", "\n}\n").contains("    bytes as u64"));
        let alloc = span("pub fn alloc_host_kind(", "\n    }\n");
        let charge = alloc
            .find("request.bytes.pinned = lease_charge(bytes);")
            .unwrap();
        let lease = alloc
            .find("self.lease_pool.lease(class, bytes, kind)")
            .unwrap();
        assert!(charge < lease, "the length is charged first");
        assert!(alloc.contains("PoolLease::FreshLength => (None, None, bytes),"));
        let pool = span("impl<B: PoolBacking> LeasePool<B> {", "\n}\n");
        let request = &pool[pool.find("    fn pool_request(").unwrap()..];
        assert!(request[..request.find("\n    }\n").unwrap()].contains("tenant: Self::tenant(),"));
        let reserve = &pool[pool.find("    fn reserve_slack(").unwrap()..];
        let reserve = &reserve[..reserve.find("\n    }\n").unwrap()];
        assert!(reserve.contains("self.held() + bytes > self.cap.get()"));
        assert!(reserve.contains("self.pool_request(bytes)"));
        let put = &pool[pool
            .find("    fn put(&self, backing: B, epoch: u64) {")
            .unwrap()..];
        let put = &put[..put.find("\n    }\n").unwrap()];
        assert!(put.contains("|| lease_class(class as usize) as u64 != class"));
        assert!(put.contains("|| self.held() + class > self.cap.get()"));
        let drop_alloc = span("impl Drop for PinnedAllocation {", "\n}\n");
        let len = drop_alloc.find("self.charge.take()").unwrap();
        let tail = drop_alloc.find("pool.release_slack(slack)").unwrap();
        let park = drop_alloc.find("pool.put(b, self.epoch)").unwrap();
        assert!(len < tail && tail < park, "length, tail, park");
    }

    /// WP-A day 70 (`DAY70.md` design Q, cell (d), on a card): a lease whose tail would pass the
    /// pool's cap is a backing at its length, charged its length alone, and frees at its drop.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn day70_a_lease_past_the_pool_cap_is_its_length_and_never_parks() {
        let (mut t, _stream, gov) = native_fixture();
        t.set_lease_pool_cap(8192);
        let kind = t.pinned_default();
        let full = t.alloc_host_kind(8192, request(), kind).unwrap();
        drop(full);
        assert_eq!(t.lease_pool_counts(), (0, 1, 8192), "the pool at its cap");
        let mut a = t.alloc_host_kind(3000, request(), kind).unwrap();
        assert_eq!(
            a.allocation.backing.as_ref().unwrap().capacity(),
            3000,
            "at its length"
        );
        assert_eq!(gov.borrow().used().pinned, 8192 + 3000, "its length alone");
        a.write(&vec![0x5au8; 3000]).unwrap();
        assert_eq!(a.bytes().unwrap().len(), 3000);
        drop(a);
        assert_eq!(gov.borrow().used().pinned, 8192, "freed at its drop");
        assert_eq!(t.lease_pool_counts(), (0, 2, 8192), "it never parked");
        let b = t.alloc_host_kind(8000, request(), kind).unwrap();
        assert_eq!(
            t.lease_pool_counts(),
            (1, 2, 0),
            "a class lease takes the idle backing"
        );
        assert_eq!(gov.borrow().used().pinned, 8192, "8000 and a 192 tail");
        drop(b);
    }

    /// WP-A day 69 (`DAY69.md` design P, cell (d), on a card): a pooled lease that carried a
    /// tenant's bytes is never the next same-class lease once the pool drains; the next one is
    /// fresh and reads zeros, and the ledger holds the leases alone.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn day69_a_drained_backing_is_never_the_next_lease() {
        let (mut t, _stream, gov) = native_fixture();
        t.set_lease_pool_cap(1 << 20);
        let kind = t.pinned_default();
        let mut a = t.alloc_host_kind(8192, request(), kind).unwrap();
        a.write(&vec![0xa5u8; 8192]).unwrap();
        let mut held = t.alloc_host_kind(8192, request(), kind).unwrap();
        held.write(&vec![0xa5u8; 8192]).unwrap();
        drop(a);
        assert_eq!(t.lease_pool_counts(), (0, 2, 8192), "one parked");
        assert_eq!(t.lease_pool().drain(), (1, 8192));
        assert_eq!(gov.borrow().used().pinned, 8192, "the held lease alone");
        drop(held);
        assert_eq!(
            t.lease_pool_counts().2,
            0,
            "the earlier lease freed at its drop"
        );
        assert_eq!(gov.borrow().used().pinned, 0);
        let b = t.alloc_host_kind(8192, request(), kind).unwrap();
        assert_eq!(
            t.lease_pool_counts(),
            (0, 3, 0),
            "fresh, never a drained backing"
        );
        assert!(
            b.bytes().unwrap().iter().all(|&x| x == 0),
            "a fresh lease reads zeros"
        );
        drop(b);
        assert_eq!(t.lease_pool_counts().2, 8192, "the new epoch parks again");
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
        let helpers = ["native_fixture"];
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
}
