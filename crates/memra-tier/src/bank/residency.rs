use super::*;
use crate::contracts::*;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

/// Log-only stage clock of the host bank lifecycle: the `--expert-bank-stages` diagnostic
/// of the MoE slot cache door (`research/spill-c-20260919/DAY40.md`). Host wall nanoseconds
/// summed per bracket and call counts; it reads `Instant` only, changes no decision, and is
/// absent unless `with_stage_clock` installed it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BankStageTimes {
    /// `stage()` calls and their whole wall.
    pub stages: u64,
    pub stage_ns: u64,
    /// `ReadWork::new` inside `stage()`: the output allocations and their zero-fill.
    pub alloc_ns: u64,
    /// `ReadWork::step` calls from `progress()`: slot allocation and zero-fill, the reader
    /// call, the assembly copy.
    pub steps: u64,
    pub step_ns: u64,
    /// Records verified in `progress()` and the per-segment checksum and zero-tail wall.
    pub verified: u64,
    pub verify_ns: u64,
    /// `publish()` whole.
    pub publish_ns: u64,
    /// `finish_host_use` + `retire` + `acknowledge`.
    pub retire_ns: u64,
    /// `collect_evicted`.
    pub collect_ns: u64,
    /// Day 63 (`research/spill-c-20260919/DAY63.md`), inside `stage`: the per-id catalog loop, the unique set and
    /// host cache pass, the governor reservations.
    pub stage_lookup_ns: u64,
    pub stage_cache_ns: u64,
    pub stage_charge_ns: u64,
    /// Day 63, inside `publish`: the output leases, the hotness and the SLRU.
    pub publish_output_ns: u64,
    pub publish_policy_ns: u64,
    /// Day 63, the retire side one call each (their sum is `retire_ns`), and the governor release inside
    /// `acknowledge`.
    pub host_use_ns: u64,
    pub retire_only_ns: u64,
    pub ack_ns: u64,
    pub ack_release_ns: u64,
}
impl BankStageTimes {
    /// `key=value` tokens in a fixed order, the form the day-40 reader parses.
    pub fn line(&self) -> String {
        format!(
            "stages={} stage_ns={} alloc_ns={} steps={} step_ns={} verified={} verify_ns={} publish_ns={} retire_ns={} collect_ns={} \
             stage_lookup_ns={} stage_cache_ns={} stage_charge_ns={} publish_output_ns={} publish_policy_ns={} \
             host_use_ns={} retire_only_ns={} ack_ns={} ack_release_ns={}",
            self.stages,
            self.stage_ns,
            self.alloc_ns,
            self.steps,
            self.step_ns,
            self.verified,
            self.verify_ns,
            self.publish_ns,
            self.retire_ns,
            self.collect_ns,
            self.stage_lookup_ns,
            self.stage_cache_ns,
            self.stage_charge_ns,
            self.publish_output_ns,
            self.publish_policy_ns,
            self.host_use_ns,
            self.retire_only_ns,
            self.ack_ns,
            self.ack_release_ns
        )
    }
}

fn clock_start(clock: &Option<BankStageTimes>) -> Option<Instant> {
    clock.is_some().then(Instant::now)
}

fn clock_add(
    clock: &mut Option<BankStageTimes>,
    start: Option<Instant>,
    field: impl FnOnce(&mut BankStageTimes, u64),
) {
    if let (Some(start), Some(clock)) = (start, clock.as_mut()) {
        field(
            clock,
            u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
        );
    }
}

static NEXT_SERVICE: AtomicU64 = AtomicU64::new(1);
struct Pending {
    ids: Vec<BankId>,
    work: Option<ReadWork>,
    missing: Vec<(BankId, CatalogRecord)>,
    charges: Vec<ChargedLease>,
    records: BTreeMap<BankId, BankLease>,
    unpublished: Vec<BankPublication>,
    completion: Completion,
    expected: Vec<Vec<SegmentExpectation>>,
    error: Option<Error>,
    plan: RowReadPlan,
    queue: Option<ChargedLease>,
    request: BudgetRequest,
    demand: bool,
    cancelled: bool,
    published: bool,
    host_use_done: bool,
    retired: bool,
}

/// What the host fill's offer of one record came to (day 45).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillOutcome {
    /// Published into a free slot of the host tier.
    Admitted,
    /// Already resident or pending, no free slot of a fitting class, or no charge left.
    Dropped,
    /// No free slot in any class: the tier is full and the fill has nothing left to do.
    Full,
    /// The bytes did not verify against the catalog's checksum (or a non-zero storage tail).
    Refused,
}

/// The host cache's index (day 43, `research/spill-c-20260919/DAY43.md`): the resident leases by
/// id, their storage bytes kept on every insert and removal, and the leases that left the index
/// (evicted, trimmed, replaced, or published without a slot) while still owned: the only
/// candidates `collect_evicted` can release. This replaces a scan of every owned lease against
/// every cached one, which was quadratic in the tier's size.
struct CacheIndex {
    map: BTreeMap<BankId, BankLease>,
    bytes: u64,
    candidates: Vec<BankLease>,
}
impl CacheIndex {
    fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            bytes: 0,
            candidates: Vec::new(),
        }
    }
    fn lease_bytes(lease: &BankLease) -> u64 {
        lease.layout().storage_bytes().expect("validated layout")
    }
    fn get(&self, id: &BankId) -> Option<&BankLease> {
        self.map.get(id)
    }
    fn contains_key(&self, id: &BankId) -> bool {
        self.map.contains_key(id)
    }
    /// Insert, and hand a replaced lease of another charge to the release candidates.
    fn insert(&mut self, id: BankId, lease: BankLease) {
        let charge = lease.charge().id();
        self.bytes += Self::lease_bytes(&lease);
        if let Some(old) = self.map.insert(id, lease) {
            self.bytes -= Self::lease_bytes(&old);
            if old.charge().id() != charge {
                self.candidates.push(old);
            }
        }
    }
    /// Remove an id that leaves the cache while its lease may still be owned.
    fn remove_evicted(&mut self, id: &BankId) -> bool {
        match self.map.remove(id) {
            Some(old) => {
                self.bytes -= Self::lease_bytes(&old);
                self.candidates.push(old);
                true
            }
            None => false,
        }
    }
    /// Remove the entry of a lease that is being released now (not a candidate).
    fn remove_released(&mut self, lease: &BankLease) -> bool {
        if self
            .map
            .get(lease.id())
            .is_some_and(|r| r.charge().id() == lease.charge().id())
        {
            let old = self.map.remove(lease.id()).expect("checked present");
            self.bytes -= Self::lease_bytes(&old);
            return true;
        }
        false
    }
}

/// Portable HOST-only implementation of the frozen bank lifecycle. Reads currently
/// run only in the explicit bounded `progress` pump; publication, cancellation and
/// last-use retirement are separate. This is not A's asynchronous DMA engine and
/// must not publish device pointers. A native GPU adapter must use TransferEngine
/// ReadyView plus actual owner/consumer fences instead of this host boundary.
/// Rc governor and shared BankLease make this service owner-thread-only.
pub struct BankService<D: BankDomain, H: Hotness<D>, R: ExactReader> {
    catalog: Catalog,
    budget: SharedBudget,
    heat: H,
    reader: R,
    policy: CoalescingPolicy,
    limits: BankLimits,
    cache: CacheIndex,
    owned: BTreeMap<(u64, u64), BankLease>,
    slru: Option<(SlruPolicy, LeasePin)>,
    pending: HashMap<TransferTicket, Pending>,
    issuer: u64,
    sequence: u64,
    clock: Option<BankStageTimes>,
    /// Day 47: where record buffers come from; `None` is a heap `Vec` per record.
    buffers: Option<Box<dyn HostBufferSource>>,
    _domain: PhantomData<D>,
}
impl<D: BankDomain, H: Hotness<D>, R: ExactReader> BankService<D, H, R> {
    pub fn new(
        catalog: Catalog,
        budget: SharedBudget,
        heat: H,
        reader: R,
        policy: CoalescingPolicy,
        limits: BankLimits,
    ) -> Result<Self> {
        policy.validate()?;
        if limits.items == 0 || limits.tickets == 0 {
            return Err(Error::Capacity);
        }
        let issuer = NEXT_SERVICE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::Overflow)?;
        Ok(Self {
            catalog,
            budget,
            heat,
            reader,
            policy,
            limits,
            cache: CacheIndex::new(),
            owned: BTreeMap::new(),
            slru: None,
            pending: HashMap::new(),
            issuer,
            sequence: 0,
            clock: None,
            buffers: None,
            _domain: PhantomData,
        })
    }
    /// Install a source of record buffers (day 47): every read then lands in a buffer from it,
    /// and the lease owns that buffer. Refused once any record is resident or pending.
    pub fn with_host_buffers(mut self, source: Box<dyn HostBufferSource>) -> Result<Self> {
        if !self.pending.is_empty() || !self.owned.is_empty() {
            return Err(Error::Busy);
        }
        self.buffers = Some(source);
        Ok(self)
    }
    /// Install the log-only stage clock (`BankStageTimes`). Diagnostic only: every bracket
    /// reads `Instant` and nothing else, so the lifecycle's decisions are unchanged.
    pub fn with_stage_clock(mut self) -> Self {
        self.clock = Some(BankStageTimes::default());
        self
    }
    /// The stage clock's totals, `None` unless `with_stage_clock` installed it.
    pub fn stage_times(&self) -> Option<&BankStageTimes> {
        self.clock.as_ref()
    }
    /// Install the CPU SLRU policy before any request. The charge is metadata
    /// only; exact output/backing remains separately charged by stage until final
    /// consumer retirement. No fixed CUDA slot pool is allocated by this adapter.
    pub fn with_slru(mut self, policy: SlruPolicy, metadata: &ChargedLease) -> Result<Self> {
        if !self.pending.is_empty()
            || !self.owned.is_empty()
            || self.slru.is_some()
            || !policy.is_empty()
        {
            return Err(Error::Busy);
        }
        let required = self.slru_metadata_bytes(policy.slots())?;
        if policy.capacity_bytes()? > self.limits.cache_bytes
            || metadata.bytes().pageable < required
        {
            return Err(Error::Capacity);
        }
        self.slru = Some((policy, metadata.pin()?));
        Ok(self)
    }
    pub fn slru_metadata_bytes(&self, slots: usize) -> Result<u64> {
        let max_id = self.catalog.entries.keys().try_fold(0u64, |max_id, id| {
            id.encode().map(|b| max_id.max(b.len() as u64))
        })?;
        // Two full-key maps + occupant key + queues, conservative node allowance.
        (slots as u64)
            .checked_mul(
                max_id
                    .checked_mul(3)
                    .and_then(|n| n.checked_add(1024))
                    .ok_or(Error::Overflow)?,
            )
            .and_then(|n| n.checked_add(4096))
            .ok_or(Error::Overflow)
    }
    pub fn slru_policy(&self) -> Option<&SlruPolicy> {
        self.slru.as_ref().map(|(p, _)| p)
    }
    /// Execute at most one bounded host read. CPU work must be pumped off a
    /// serving scheduler thread. Returning true means producer terminal, NOT GPU ready.
    pub fn progress(&mut self, ticket: &TransferTicket) -> Result<bool> {
        let p = self.pending.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        if p.completion.producer_done {
            return Ok(true);
        }
        let read = if p.cancelled {
            Err(Error::Cancelled)
        } else {
            self.reader.begin_request(&p.request, ticket.epochs);
            let work = p.work.as_mut().ok_or(Error::NotReady)?;
            let started = clock_start(&self.clock);
            let stepped = work.step(&p.missing, &p.plan, &mut self.reader, self.policy);
            clock_add(&mut self.clock, started, |c, ns| {
                c.steps += 1;
                c.step_ns += ns;
            });
            match stepped {
                Ok(false) => return Ok(false),
                Ok(true) => Ok(std::mem::take(&mut work.outputs)),
                Err(e) => Err(e),
            }
        };
        p.work = None;
        let missing = std::mem::take(&mut p.missing);
        let charges = std::mem::take(&mut p.charges);

        let mut error = read.as_ref().err().cloned();
        let mut unpublished = Vec::new();
        let mut items = Vec::new();
        if let Ok(outputs) = read {
            for (index, (((id, r), charge), bytes)) in
                missing.into_iter().zip(charges).zip(outputs).enumerate()
            {
                let mut base = 0usize;
                let mut segments = Vec::new();
                let started = clock_start(&self.clock);
                for (j, s) in r.layout.segments.iter().enumerate() {
                    let valid_end = base + s.valid_bytes as usize;
                    let end = base + s.storage_bytes as usize;
                    let view = bytes.bytes();
                    let hash = checksum(&view[base..valid_end]);
                    let valid =
                        hash == r.checksums[j] && view[valid_end..end].iter().all(|&v| v == 0);
                    if !valid {
                        error = Some(Error::Corrupt);
                    }
                    segments.push(SegmentCompletion {
                        segment: j as u32,
                        status: if valid {
                            ItemStatus::Complete
                        } else {
                            ItemStatus::Failed
                        },
                        valid_bytes: s.valid_bytes,
                        // Host slot -> record materialization bytes. The underlying
                        // ObjectReader transfer completion reports framed storage I/O;
                        // coalesced reads cannot be charged once per logical record.
                        io_bytes: s.storage_bytes,
                        checksum: Some(hash),
                        epochs: ticket.epochs,
                        producer_done: true,
                        consumer_fenced: false,
                        consumer_fence: None,
                        error: (!valid).then_some(Error::Corrupt),
                    });
                    base = end;
                }
                clock_add(&mut self.clock, started, |c, ns| {
                    c.verified += 1;
                    c.verify_ns += ns;
                });
                items.push(ItemOutcome {
                    item: index as u32,
                    accepted: true,
                    segments,
                });
                unpublished.push(BankPublication {
                    id,
                    layout: r.layout,
                    class: self.catalog.class,
                    charge,
                    backing: bytes.into_backing(),
                });
            }
        } else {
            // Enumerate EVERY accepted record/segment even when a coalesced
            // read fails. No short status vector or successful sibling survives.
            // Zero lengths here mean no verified logical bytes were delivered;
            // the reader's exact I/O error is retained independently.
            items = missing
                .iter()
                .enumerate()
                .map(|(i, (_, r))| ItemOutcome {
                    item: i as u32,
                    accepted: true,
                    segments: r
                        .layout
                        .segments
                        .iter()
                        .enumerate()
                        .map(|(j, _)| SegmentCompletion {
                            segment: j as u32,
                            status: if error == Some(Error::Cancelled) {
                                ItemStatus::Cancelled
                            } else {
                                ItemStatus::Failed
                            },
                            valid_bytes: 0,
                            io_bytes: 0,
                            checksum: None,
                            epochs: ticket.epochs,
                            producer_done: true,
                            consumer_fenced: false,
                            consumer_fence: None,
                            error: error.clone(),
                        })
                        .collect(),
                })
                .collect();
            // Synchronous read failed: no DMA exists and all temporary outputs
            // are gone; charges may be explicitly returned immediately.
            for charge in &charges {
                self.budget.borrow_mut().release(charge)?;
            }
        }
        p.error = error;
        p.unpublished = unpublished;
        p.completion.items = items;
        p.completion.producer_done = true;
        Ok(true)
    }
    /// Bounded lifecycle inventory for orderly host shutdown/receipt collection.
    pub fn tickets(&self) -> Vec<TransferTicket> {
        self.pending.keys().copied().collect()
    }
    pub fn reader(&self) -> &R {
        &self.reader
    }
    pub fn plan(&self, ticket: &TransferTicket) -> Result<&RowReadPlan> {
        Ok(&self.pending.get(ticket).ok_or(Error::UnknownTicket)?.plan)
    }
    pub fn completion(&self, ticket: &TransferTicket) -> Result<&Completion> {
        Ok(&self
            .pending
            .get(ticket)
            .ok_or(Error::UnknownTicket)?
            .completion)
    }
    pub fn cache_bytes(&self) -> u64 {
        self.cache.bytes
    }
    /// Admit a record the host fill read off the owner thread (day 45,
    /// `research/spill-c-20260919/DAY45.md`): the digest the fill computed over the bytes must
    /// equal the catalog's checksum (the check `progress` applies to a read), then a charge, a
    /// lease and publication into a FREE slot of the SLRU. Never evicts, never replaces: a
    /// resident or pending record, a full tier or a refused charge drops the bytes. Only
    /// single-segment payload records with no storage tail (the door's) are accepted.
    pub fn admit_filled(
        &mut self,
        id: &BankId,
        bytes: HostBytes,
        digest: Digest,
        request: &BudgetRequest,
    ) -> Result<FillOutcome> {
        let entry = self.catalog.entry(id)?;
        let resident_charge = entry.resident_charge_bytes();
        let record = entry.record.clone();
        let layout = &record.layout;
        if layout.segments.len() != 1 || layout.segments[0].role != Role::Payload {
            return Err(Error::Unsupported);
        }
        let segment = &layout.segments[0];
        if segment.valid_bytes != segment.storage_bytes {
            return Err(Error::Unsupported);
        }
        // The fill computed `digest` with the contract `checksum` over exactly these bytes (moved
        // here, not copied): the same function over the same bytes `progress` would verify.
        if bytes.len() as u64 != segment.storage_bytes || digest != record.checksums[0] {
            return Ok(FillOutcome::Refused);
        }
        if self.cache.contains_key(id) {
            return Ok(FillOutcome::Dropped);
        }
        let Some((policy, _)) = &mut self.slru else {
            return Err(Error::Unsupported);
        };
        if policy.resident(id).is_some() || policy.pending(id) {
            return Ok(FillOutcome::Dropped);
        }
        let Some(_slot) = policy.reserve_free(id, layout.storage_bytes()?)? else {
            return Ok(if policy.free_slots() == 0 {
                FillOutcome::Full
            } else {
                FillOutcome::Dropped
            });
        };
        let mut charge_request = request.clone();
        charge_request.bytes = TierBudget::zero(charge_request.bytes.device.len());
        charge_request.bytes.pageable = resident_charge?;
        let charge = match self.budget.borrow_mut().reserve(&charge_request) {
            Ok(charge) => charge,
            Err(_) => {
                policy.abort_retired(id)?;
                return Ok(FillOutcome::Dropped);
            }
        };
        match BankLease::from_backend(
            id.clone(),
            record.layout.clone(),
            self.catalog.class,
            charge,
            bytes.into_backing(),
        ) {
            Ok(lease) => {
                policy.publish(id)?;
                self.owned.insert(lease.charge().id(), lease.clone());
                self.cache.insert(id.clone(), lease);
                Ok(FillOutcome::Admitted)
            }
            Err(rejected) => {
                policy.abort_retired(id)?;
                self.budget.borrow_mut().release(&rejected.op.charge)?;
                Err(rejected.error)
            }
        }
    }
    /// Records the host cache holds (day 43; read-only).
    pub fn cached_records(&self) -> usize {
        self.cache.map.len()
    }
    /// Leases the service owns, cached or held by an open ticket (day 43; read-only).
    pub fn owned_leases(&self) -> usize {
        self.owned.len()
    }
    pub fn evict_cached(&mut self, id: &BankId) -> Result<bool> {
        self.catalog.record(id)?;
        if let Some((policy, _)) = &mut self.slru {
            policy.remove(id);
        }
        Ok(self.cache.remove_evicted(id))
    }
    /// Host consumer adapter calls after its last use (including speculative
    /// rollback). No CUDA/graph use is accepted by this backend. Does not release
    /// resources: retire/release still run, and borrowed views refuse Busy.
    pub fn finish_host_use(&mut self, ticket: &TransferTicket) -> Result<()> {
        let started = clock_start(&self.clock);
        let result = self.finish_host_use_unclocked(ticket);
        clock_add(&mut self.clock, started, |c, ns| {
            c.retire_ns += ns;
            c.host_use_ns += ns;
        });
        result
    }
    fn finish_host_use_unclocked(&mut self, ticket: &TransferTicket) -> Result<()> {
        let p = self.pending.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        if !p.published && !p.cancelled && p.error.is_none() {
            return Err(Error::Busy);
        }
        p.host_use_done = true;
        Ok(())
    }
    pub fn acknowledge(&mut self, ticket: &TransferTicket) -> Result<()> {
        let started = clock_start(&self.clock);
        let result = self.acknowledge_unclocked(ticket);
        clock_add(&mut self.clock, started, |c, ns| {
            c.retire_ns += ns;
            c.ack_ns += ns;
        });
        result
    }
    fn acknowledge_unclocked(&mut self, ticket: &TransferTicket) -> Result<()> {
        if !self
            .pending
            .get(ticket)
            .ok_or(Error::UnknownTicket)?
            .retired
        {
            return Err(Error::Busy);
        }
        // Keep tombstone/descriptor quota until actual acknowledgement removes it.
        let p = self.pending.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        if let Some(queue) = &p.queue {
            let releasing = clock_start(&self.clock);
            self.budget.borrow_mut().release(queue)?;
            clock_add(&mut self.clock, releasing, |c, ns| c.ack_release_ns += ns);
        }
        p.queue = None;
        self.pending.remove(ticket);
        Ok(())
    }
    pub(crate) fn can_release(&self, lease: &BankLease) -> Result<()> {
        if !self.owned.contains_key(&lease.charge().id()) {
            return Err(if lease.charge().state()? == ChargeState::Released {
                Error::AlreadyReleased
            } else {
                Error::ForeignLease
            });
        }
        if self.pending.values().any(|p| {
            !p.retired
                && p.records
                    .values()
                    .any(|r| r.charge().id() == lease.charge().id())
        }) {
            return Err(Error::Busy);
        }
        // Shared API deliberately doesn't expose a mutable-borrow probe. The
        // actual retire_backing below is the final borrowed-view guard.
        Ok(())
    }
    fn trim(&mut self) {
        while self.cache_bytes() > self.limits.cache_bytes {
            let id = self
                .cache
                .map
                .keys()
                .min_by_key(|id| self.heat.score(id))
                .cloned()
                .expect("nonempty cache");
            self.cache.remove_evicted(&id);
            if let Some((policy, _)) = &mut self.slru {
                policy.remove(&id);
            }
        }
    }
    /// Release inactive evicted allocations, never outstanding consumer tickets.
    pub fn collect_evicted(&mut self) -> Result<()> {
        let started = clock_start(&self.clock);
        let result = self.collect_evicted_unclocked();
        clock_add(&mut self.clock, started, |c, ns| c.collect_ns += ns);
        result
    }
    fn collect_evicted_unclocked(&mut self) -> Result<()> {
        // Every owned lease outside the cache entered `candidates` when it left the index or
        // was published without a slot, so these are exactly the leases the old scan found.
        let candidates = std::mem::take(&mut self.cache.candidates);
        let mut kept = Vec::new();
        let mut failure = None;
        for lease in candidates {
            let charge = lease.charge().id();
            if !self.owned.contains_key(&charge)
                || self
                    .cache
                    .get(lease.id())
                    .is_some_and(|c| c.charge().id() == charge)
            {
                continue;
            }
            if failure.is_some() {
                kept.push(lease);
                continue;
            }
            match self.release(&lease) {
                Ok(()) => {}
                Err(Error::Busy) => kept.push(lease),
                Err(e) => {
                    kept.push(lease);
                    failure = Some(e);
                }
            }
        }
        self.cache.candidates.extend(kept);
        match failure {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
impl<D: BankDomain, H: Hotness<D>, R: ExactReader> BankedResidency for BankService<D, H, R> {
    fn layout(&self, id: &BankId) -> Result<&RecordLayout> {
        if !D::accepts(&id.record) {
            return Err(Error::InvalidLayout);
        }
        Ok(&self.catalog.record(id)?.layout)
    }
    fn resident(&mut self, id: &BankId) -> Result<Option<BankLease>> {
        self.layout(id)?;
        // Advisory only: stage a ticket even for hits to register last-use lifetime.
        Ok(self.cache.get(id).cloned())
    }
    fn stage(&mut self, batch: BankBatch) -> Result<TransferTicket> {
        let started = clock_start(&self.clock);
        let result = self.stage_unclocked(batch);
        clock_add(&mut self.clock, started, |c, ns| {
            c.stages += 1;
            c.stage_ns += ns;
        });
        result
    }
    fn publish(&mut self, ticket: &TransferTicket, current: Epochs) -> Result<Vec<BankLease>> {
        let started = clock_start(&self.clock);
        let result = self.publish_unclocked(ticket, current);
        clock_add(&mut self.clock, started, |c, ns| c.publish_ns += ns);
        result
    }
    fn cancel(&mut self, ticket: &TransferTicket) -> Result<CancelState> {
        let p = self.pending.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        if p.published {
            Ok(CancelState::AlreadyPublished)
        } else {
            p.cancelled = true;
            Ok(CancelState::PublicationRevoked)
        }
    }
    fn retire(&mut self, ticket: &TransferTicket) -> Result<bool> {
        let started = clock_start(&self.clock);
        let result = self.retire_unclocked(ticket);
        clock_add(&mut self.clock, started, |c, ns| {
            c.retire_ns += ns;
            c.retire_only_ns += ns;
        });
        result
    }
    fn release(&mut self, lease: &BankLease) -> Result<()> {
        self.can_release(lease)?;
        lease.retire_backing()?;
        self.budget.borrow_mut().release(lease.charge())?;
        if self.cache.remove_released(lease)
            && let Some((policy, _)) = &mut self.slru
        {
            policy.remove(lease.id());
        }
        self.owned.remove(&lease.charge().id());
        Ok(())
    }
}
impl<D: BankDomain, H: Hotness<D>, R: ExactReader> BankService<D, H, R> {
    fn stage_unclocked(&mut self, batch: BankBatch) -> Result<TransferTicket> {
        if batch.ids.is_empty() {
            return Err(Error::EmptyBatch);
        }
        if batch.ids.len() > self.limits.items || self.pending.len() >= self.limits.tickets {
            return Err(Error::Capacity);
        }
        batch.request.validate()?;
        // One ticket is reserved for mandatory/demand work, even when hints stall.
        if batch.request.priority == Priority::OptionalPrefetch
            && self.pending.len() >= self.limits.tickets.saturating_sub(1)
        {
            return Err(Error::Capacity);
        }
        // This backend can promise host bytes only. Never accept a device request
        // and quietly substitute host readiness.
        if batch
            .request
            .bytes
            .device
            .iter()
            .chain(&batch.request.bytes.peer)
            .chain(&batch.request.bytes.replicas)
            .any(|&n| n != 0)
            || batch.request.bytes.pinned != 0
        {
            return Err(Error::Unsupported);
        }
        // Day 61 (I11 change 2): each id's catalog entry is read once, for its logical bytes and
        // its metadata allowance; the allowance's sum refuses at the point it always did.
        let looking = clock_start(&self.clock);
        let mut logical = 0u64;
        let mut metadata = Some(0u64);
        for id in &batch.ids {
            if !D::accepts(&id.record) {
                return Err(Error::InvalidLayout);
            }
            let entry = self.catalog.entry(id)?;
            logical = logical
                .checked_add(entry.record.layout.storage_bytes()?)
                .ok_or(Error::Overflow)?;
            metadata = metadata.and_then(|n| n.checked_add(entry.metadata));
        }
        clock_add(&mut self.clock, looking, |c, ns| c.stage_lookup_ns += ns);
        if logical > self.limits.batch_bytes {
            return Err(Error::Capacity);
        }
        let caching = clock_start(&self.clock);
        let unique: BTreeSet<_> = batch.ids.iter().cloned().collect();
        // Day 61 (I11 change 2): the host cache is read once per unique id, for a cached lease
        // or a missing record. Nothing below changes the cache before the ticket is recorded.
        let mut missing = Vec::new();
        let mut records = BTreeMap::new();
        for id in unique {
            match self.cache.get(&id) {
                Some(lease) => {
                    let lease = lease.clone();
                    records.insert(id, lease);
                }
                None => {
                    let record = self.catalog.record(&id)?.clone();
                    missing.push((id, record));
                }
            }
        }
        clock_add(&mut self.clock, caching, |c, ns| c.stage_cache_ns += ns);
        let plan = plan_reads(&missing, logical, &self.reader, self.policy)?;
        let sequence = self.sequence.checked_add(1).ok_or(Error::Overflow)?;
        let ticket = TransferTicket {
            issuer: self.issuer,
            sequence,
            epochs: batch.epochs,
        };
        // Charge output, slot and bounded metadata through one injected governor.
        // Canonical metadata + conservative per-node allowance is an estimate;
        // allocator/RSS calibration remains a native integration gate.
        let output_bytes = missing.iter().try_fold(0u64, |n, (id, _)| {
            n.checked_add(self.catalog.entry(id)?.resident_charge_bytes()?)
                .ok_or(Error::Overflow)
        })?;
        let metadata = metadata.ok_or(Error::Overflow)?;
        let slot = if missing.is_empty() {
            0
        } else {
            self.policy.slot_bytes
        };
        let required = output_bytes
            .checked_add(metadata)
            .and_then(|n| n.checked_add(slot))
            .ok_or(Error::Overflow)?;
        let mut queue_request = batch.request.clone();
        queue_request.bytes.pageable = queue_request.bytes.pageable.max(required) - output_bytes;
        queue_request.bytes.staging = queue_request.bytes.staging.max(slot);
        queue_request.bytes.inflight = queue_request.bytes.inflight.max(1);
        let charging = clock_start(&self.clock);
        let queue = self.budget.borrow_mut().reserve(&queue_request)?;
        let mut charges = Vec::new();
        for (id, _) in &missing {
            let mut request = batch.request.clone();
            request.bytes = TierBudget::zero(request.bytes.device.len());
            request.bytes.pageable = self.catalog.entry(id)?.resident_charge_bytes()?;
            let result = self.budget.borrow_mut().reserve(&request);
            match result {
                Ok(charge) => charges.push(charge),
                Err(e) => {
                    for charge in &charges {
                        self.budget.borrow_mut().release(charge)?;
                    }
                    self.budget.borrow_mut().release(&queue)?;
                    return Err(e);
                }
            }
        }
        clock_add(&mut self.clock, charging, |c, ns| c.stage_charge_ns += ns);
        let expected: Vec<Vec<SegmentExpectation>> = missing
            .iter()
            .map(|(_, r)| {
                r.layout
                    .segments
                    .iter()
                    .zip(&r.checksums)
                    .map(|(s, h)| SegmentExpectation {
                        valid_bytes: s.valid_bytes,
                        io_bytes: s.valid_bytes,
                        checksum: *h,
                    })
                    .collect()
            })
            .collect();
        let work = if missing.is_empty() {
            None
        } else {
            let started = clock_start(&self.clock);
            let work = ReadWork::new(&missing, self.buffers.as_deref_mut());
            clock_add(&mut self.clock, started, |c, ns| c.alloc_ns += ns);
            match work {
                Ok(work) => Some(work),
                Err(error) => {
                    for charge in &charges {
                        self.budget.borrow_mut().release(charge)?;
                    }
                    self.budget.borrow_mut().release(&queue)?;
                    return Err(error);
                }
            }
        };
        let items = missing
            .iter()
            .enumerate()
            .map(|(i, (_, r))| ItemOutcome {
                item: i as u32,
                accepted: true,
                segments: r
                    .layout
                    .segments
                    .iter()
                    .enumerate()
                    .map(|(j, _)| SegmentCompletion {
                        segment: j as u32,
                        status: ItemStatus::Pending,
                        valid_bytes: 0,
                        io_bytes: 0,
                        checksum: None,
                        epochs: ticket.epochs,
                        producer_done: false,
                        consumer_fenced: false,
                        consumer_fence: None,
                        error: None,
                    })
                    .collect(),
            })
            .collect();
        let producer_done = missing.is_empty();
        self.pending.insert(
            ticket,
            Pending {
                ids: batch.ids,
                work,
                missing,
                charges,
                records,
                unpublished: Vec::new(),
                completion: Completion {
                    ticket,
                    items,
                    producer_done,
                    consumer_fenced: false,
                },
                expected,
                error: None,
                plan,
                queue: Some(queue),
                request: batch.request.clone(),
                demand: batch.request.priority != Priority::OptionalPrefetch,
                cancelled: false,
                published: false,
                host_use_done: false,
                retired: false,
            },
        );
        self.sequence = sequence;
        Ok(ticket)
    }
    fn publish_unclocked(
        &mut self,
        ticket: &TransferTicket,
        current: Epochs,
    ) -> Result<Vec<BankLease>> {
        let p = self.pending.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        ticket.epochs.require(current)?;
        if p.cancelled {
            return Err(Error::Cancelled);
        }
        if p.published || p.retired {
            return Err(Error::Busy);
        }
        if let Some(e) = &p.error {
            return Err(e.clone());
        }
        if !p.completion.producer_done {
            return Err(Error::NotReady);
        }
        if !p.expected.is_empty() {
            p.completion.require(ticket, &p.expected, false)?;
        }
        while let Some(publication) = p.unpublished.pop() {
            match BankLease::from_backend(
                publication.id,
                publication.layout,
                publication.class,
                publication.charge,
                publication.backing,
            ) {
                Ok(lease) => {
                    self.owned.insert(lease.charge().id(), lease.clone());
                    p.records.insert(lease.id().clone(), lease);
                }
                Err(rejected) => {
                    // Preserve every resource and retry/release handle on refusal.
                    p.error = Some(rejected.error.clone());
                    p.unpublished.push(rejected.op);
                    return Err(rejected.error);
                }
            }
        }
        let outputting = clock_start(&self.clock);
        let output: Vec<_> = p.ids.iter().map(|id| p.records[id].clone()).collect();
        clock_add(&mut self.clock, outputting, |c, ns| {
            c.publish_output_ns += ns
        });
        let policing = clock_start(&self.clock);
        if p.demand {
            for id in &p.ids {
                self.heat.demand(id);
            }
        }
        p.published = true;
        if let Some((policy, _)) = &mut self.slru {
            // Host-only adapter serializes policy decisions at successful publication.
            // A prefetch hit is a no-op, not demand heat. Duplicate IDs retain order.
            for id in &p.ids {
                // Day 61 (I11 change 3): one SLRU lookup per id; `hit` answers residency and
                // promotes as it did after the separate check.
                let resident = if p.demand {
                    policy.hit(id)
                } else {
                    policy.resident(id).is_some()
                };
                if resident {
                    continue;
                }
                let lease = &p.records[id];
                if let Some(decision) = policy.reserve(id, lease.layout().storage_bytes()?, &[])? {
                    if let Some(old) = decision.evicted {
                        self.cache.remove_evicted(&old);
                    }
                    policy.publish(id)?;
                    self.cache.insert(id.clone(), lease.clone());
                } else {
                    // Published to this ticket with no host slot: owned, never cached.
                    self.cache.candidates.push(lease.clone());
                }
            }
        } else {
            for (id, lease) in &p.records {
                self.cache.insert(id.clone(), lease.clone());
            }
            self.trim();
        }
        clock_add(&mut self.clock, policing, |c, ns| c.publish_policy_ns += ns);
        Ok(output)
    }
    fn retire_unclocked(&mut self, ticket: &TransferTicket) -> Result<bool> {
        let p = self.pending.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        if !p.host_use_done || !p.completion.producer_done {
            return Ok(false);
        }
        if p.retired {
            return Ok(true);
        }
        // Only unpublished inputs are freed here; live published aliases retire
        // via release, after ALL tickets referencing that allocation have retired.
        for publication in &p.unpublished {
            self.budget.borrow_mut().release(&publication.charge)?;
        }
        p.unpublished.clear();
        p.retired = true;
        Ok(true)
    }
}
