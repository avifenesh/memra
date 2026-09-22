//! Native owner-stream H2D/D2H implementation of the frozen tier contract.
//! No worker may submit CUDA work. Unknown completion retains backing and quota.
use cudarc::driver::{
    CudaContext, CudaEvent, CudaSlice, CudaStream, CudaViewMut, DriverError, HostSlice, SyncOnDrop,
    result, sys,
};
use memra_kv::KvPlane;
use memra_tier::{bank::SharedBudget, contracts::*};
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
    /// `new`, plus a second stream of the same context for the copies (WP-A day 17 for the D2H,
    /// day 18 for the H2D; memra#536 Move 1). The stream is created on the owner thread; a failure
    /// to create it is a construction refusal, never a silent fall back to the owner stream.
    pub fn new_with_copy_stream(owner: Arc<CudaStream>, governor: SharedBudget) -> Result<Self> {
        let mut t = Self::new(owner, governor)?;
        t.copy = Some(cuda(t.stream.context().new_stream())?);
        Ok(t)
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
                cuda(copy.memcpy_dtod(
                    &op.source.slice(0..n),
                    &mut backing.borrow_mut().slice_mut(..n),
                ))?;
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
                cuda(copy.memcpy_dtod(&op.source.slice(0..n), &mut op.destination))?;
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
            if item.direction == CopyDirection::DeviceToDevice {
                // WP-A day 20 (Move 2 slice 1): a capture has no host bytes, so no checksum
                // term yet (slice 3 decides the device digest or the `Unwitnessed` arm). The
                // expectation keeps its zero digest, so `Completion::require` refuses the item
                // `Corrupt` by construction: no path publishes a capture through the
                // host-contract gate before its receipt exists. Publication is the caller's,
                // after `capture_landed`.
                s.checksum = None;
                continue;
            }
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
        let body = &src[..src.find("#[cfg(test)]").unwrap()];
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
        let body = &src[..src.find("#[cfg(test)]").unwrap()];
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
        let body = &src[..src.find("#[cfg(test)]").unwrap()];
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
        let body = &src[..src.find("#[cfg(test)]").unwrap()];
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
        assert!(progress_body[d2d_arm..d2d_arm + 700].contains("s.checksum = None;"));
        assert!(progress_body[d2d_arm..d2d_arm + 700].contains("continue;"));
        let landed = body.find("pub fn capture_landed(").unwrap();
        let landed_body = &body[landed..body[landed..].find("\n    pub fn ").unwrap() + landed];
        assert!(landed_body.contains("self.progress(ticket)?;"));
        assert!(landed_body.contains("i.direction != CopyDirection::DeviceToDevice"));
        assert!(landed_body.contains("Ok(e.completion.producer_done)"));
        let take = body.find("fn take_destination(").unwrap();
        let take_body = &body[take..body[take..].find("\n    fn ").unwrap() + take];
        assert!(take_body.contains("if i.direction == CopyDirection::DeviceToDevice {"));
    }

    fn native_fixture() -> (CudaTransfers, Arc<CudaStream>, SharedBudget) {
        use memra_tier::tier::governor::Governor;
        let ctx = CudaContext::new(0).unwrap();
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

    /// WP-A day 20 (memra#536 Move 2 slice 1): the capture class on a card. Two owned planes are
    /// registered as destinations with retained twins; a borrowed source holds a pattern; one
    /// capture batch on the copy stream behind an owner-stream producer fence; the host-contract
    /// gate refuses the items (no checksum term); `capture_landed` turns true; `retire(None)`,
    /// `acknowledge`, the planes come back through the twins and hold the pattern byte for byte.
    #[test]
    #[ignore = "native CUDA required; run under the provided one-card exclusive lock"]
    fn d2d_capture_lands_on_the_copy_stream_and_publishes_only_after_its_event() {
        use memra_tier::tier::governor::Governor;
        let ctx = CudaContext::new(0).unwrap();
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
        let (mut on_owner, _s, _g) = native_fixture();
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
        for item in &c.items {
            assert_eq!(item.segments[0].valid_bytes, bytes as u64);
            assert!(
                item.segments[0].checksum.is_none(),
                "no checksum term in slice 1"
            );
        }
        let expected: Vec<Vec<SegmentExpectation>> = (0..2)
            .map(|_| {
                vec![SegmentExpectation {
                    valid_bytes: bytes as u64,
                    io_bytes: bytes as u64,
                    checksum: [0; 32],
                }]
            })
            .collect();
        assert_eq!(c.require(&ticket, &expected, true), Err(Error::Corrupt));
        assert!(matches!(
            t.ready_view(&ticket, 0, epochs),
            Err(Error::Corrupt)
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
        let ctx = CudaContext::new(0).unwrap();
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
        let (mut on_owner, _s, _g) = native_fixture();
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
        for item in &c.items {
            assert_eq!(item.segments[0].valid_bytes, bytes as u64);
            assert!(item.segments[0].checksum.is_none(), "no checksum term");
            assert!(!item.segments[0].consumer_fenced);
        }
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
        assert!(matches!(
            t.ready_view(&ticket, 0, epochs),
            Err(Error::Corrupt)
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
