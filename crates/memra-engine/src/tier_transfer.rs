//! Native owner-stream H2D/D2H implementation of the frozen tier contract.
//! No worker may submit CUDA work. Unknown completion retains backing and quota.
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, PinnedHostSlice, result, sys};
use memra_kv::KvPlane;
use memra_tier::{bank::SharedBudget, contracts::*};
use std::{
    cell::{Ref, RefCell},
    collections::HashMap,
    rc::Rc,
    sync::Arc,
};

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
    backing: Option<PinnedHostSlice<u8>>,
    charge: Option<ChargedLease>,
    pin: Option<LeasePin>,
    governor: SharedBudget,
}
impl std::fmt::Debug for CudaPinnedLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaPinnedLease")
            .field("bytes", &self.storage_bytes())
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
/// Opaque graph-use retention. Drop only after graph execution/destruction retires.
pub struct GraphPin {
    _pins: Vec<Rc<()>>,
}

pub struct CudaTransfers {
    stream: Arc<CudaStream>,
    governor: SharedBudget,
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
        Ok(Self {
            stream: owner,
            governor,
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
    /// Admission policy belongs to the caller; only the physical dimension is set here.
    pub fn alloc_host(
        &mut self,
        bytes: usize,
        mut request: BudgetRequest,
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
            let mut backing = cuda(unsafe { self.stream.context().alloc_pinned::<u8>(bytes) })?;
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
    pub fn release_device(&mut self, lease: &DeviceLease) -> Result<()> {
        self.check_thread()?;
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
    pub fn take_plane(&mut self, lease: &DeviceLease) -> Result<KvPlane> {
        self.check_thread()?;
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
            }
            item.source_retired = true;
        }
        Ok(())
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
                let submit = (|| {
                    if let Some(f) = producer_fence {
                        cuda(self.stream.wait(&self.producers[&f.sequence].1))?;
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
                        // A taken host destination may be used as an immutable
                        // H2D source before the earlier ticket is acknowledged.
                        CopyDirection::HostToDevice => cuda(
                            self.stream.memcpy_htod(
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
                            cuda(self.stream.memcpy_dtoh(
                                &backing.borrow().slice(..item.bytes as usize),
                                allocation.backing.as_mut().ok_or(Error::AlreadyReleased)?,
                            ))?;
                        }
                    }
                    drop(backing);
                    item.event = Some(cuda(self.stream.record_event(None))?);
                    // Install the wait separately from observing producer completion.
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
