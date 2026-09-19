use super::*;
use crate::contracts::*;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_SERVICE: AtomicU64 = AtomicU64::new(1);
struct Pending {
    ids: Vec<BankId>,
    records: BTreeMap<BankId, BankLease>,
    unpublished: Vec<BankPublication>,
    completion: Completion,
    expected: Vec<Vec<SegmentExpectation>>,
    error: Option<Error>,
    plan: RowReadPlan,
    queue: Option<ChargedLease>,
    demand: bool,
    cancelled: bool,
    published: bool,
    host_use_done: bool,
    retired: bool,
}

/// Portable HOST-only implementation of the frozen bank lifecycle. Reads currently
/// finish synchronously in `stage`, but tickets, publication, cancellation and
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
    cache: BTreeMap<BankId, BankLease>,
    owned: BTreeMap<(u64, u64), BankLease>,
    pending: HashMap<TransferTicket, Pending>,
    issuer: u64,
    sequence: u64,
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
            cache: BTreeMap::new(),
            owned: BTreeMap::new(),
            pending: HashMap::new(),
            issuer,
            sequence: 0,
            _domain: PhantomData,
        })
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
        self.cache
            .values()
            .map(|r| r.layout().storage_bytes().expect("validated layout"))
            .sum()
    }
    pub fn evict_cached(&mut self, id: &BankId) -> Result<bool> {
        self.catalog.record(id)?;
        Ok(self.cache.remove(id).is_some())
    }
    /// Host consumer adapter calls after its last use (including speculative
    /// rollback). No CUDA/graph use is accepted by this backend. Does not release
    /// resources: retire/release still run, and borrowed views refuse Busy.
    pub fn finish_host_use(&mut self, ticket: &TransferTicket) -> Result<()> {
        let p = self.pending.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        if !p.published && !p.cancelled && p.error.is_none() {
            return Err(Error::Busy);
        }
        p.host_use_done = true;
        Ok(())
    }
    pub fn acknowledge(&mut self, ticket: &TransferTicket) -> Result<()> {
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
            self.budget.borrow_mut().release(queue)?;
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
                .keys()
                .min_by_key(|id| self.heat.score(id))
                .cloned()
                .expect("nonempty cache");
            self.cache.remove(&id);
        }
    }
    /// Release inactive evicted allocations, never outstanding consumer tickets.
    pub fn collect_evicted(&mut self) -> Result<()> {
        let leases: Vec<_> = self
            .owned
            .values()
            .filter(|l| {
                !self
                    .cache
                    .values()
                    .any(|c| c.charge().id() == l.charge().id())
            })
            .cloned()
            .collect();
        for lease in leases {
            match self.release(&lease) {
                Ok(()) | Err(Error::Busy) => (),
                Err(e) => return Err(e),
            }
        }
        Ok(())
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
        if batch.ids.is_empty() {
            return Err(Error::EmptyBatch);
        }
        if batch.ids.len() > self.limits.items || self.pending.len() >= self.limits.tickets {
            return Err(Error::Capacity);
        }
        batch.request.validate()?;
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
        let mut logical = 0u64;
        for id in &batch.ids {
            logical = logical
                .checked_add(self.layout(id)?.storage_bytes()?)
                .ok_or(Error::Overflow)?;
        }
        if logical > self.limits.batch_bytes {
            return Err(Error::Capacity);
        }
        let unique: BTreeSet<_> = batch.ids.iter().cloned().collect();
        let missing: Vec<_> = unique
            .iter()
            .filter(|id| !self.cache.contains_key(*id))
            .map(|id| Ok((id.clone(), self.catalog.record(id)?.clone())))
            .collect::<Result<_>>()?;
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
        let output_bytes = missing.iter().try_fold(0u64, |n, (id, r)| {
            n.checked_add(r.resident_charge_bytes(id)?)
                .ok_or(Error::Overflow)
        })?;
        let metadata = batch.ids.iter().try_fold(0u64, |n, id| {
            n.checked_add(
                (id.encode()?.len() + self.catalog.record(id)?.layout.encode()?.len() + 1024)
                    as u64,
            )
            .ok_or(Error::Overflow)
        })?;
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
        let queue = self.budget.borrow_mut().reserve(&queue_request)?;
        let mut charges = Vec::new();
        for (id, r) in &missing {
            let mut request = batch.request.clone();
            request.bytes = TierBudget::zero(request.bytes.device.len());
            request.bytes.pageable = r.resident_charge_bytes(id)?;
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
        let expected: Vec<Vec<SegmentExpectation>> = missing
            .iter()
            .map(|(_, r)| {
                r.layout
                    .segments
                    .iter()
                    .zip(&r.checksums)
                    .map(|(s, h)| SegmentExpectation {
                        valid_bytes: s.valid_bytes,
                        io_bytes: s.storage_bytes,
                        checksum: *h,
                    })
                    .collect()
            })
            .collect();
        let read = read_records(&missing, &plan, &mut self.reader, self.policy);
        let mut error = read.as_ref().err().cloned();
        let mut unpublished = Vec::new();
        let mut items = Vec::new();
        if let Ok(outputs) = read {
            for (index, (((id, r), charge), bytes)) in
                missing.into_iter().zip(charges).zip(outputs).enumerate()
            {
                let mut base = 0usize;
                let mut segments = Vec::new();
                for (j, s) in r.layout.segments.iter().enumerate() {
                    let valid_end = base + s.valid_bytes as usize;
                    let end = base + s.storage_bytes as usize;
                    let hash = checksum(&bytes[base..valid_end]);
                    let valid =
                        hash == r.checksums[j] && bytes[valid_end..end].iter().all(|&v| v == 0);
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
                    backing: Box::new(bytes),
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
                            status: ItemStatus::Failed,
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
        let records = unique
            .into_iter()
            .filter_map(|id| self.cache.get(&id).map(|l| (id, l.clone())))
            .collect();
        self.pending.insert(
            ticket,
            Pending {
                ids: batch.ids,
                records,
                unpublished,
                completion: Completion {
                    ticket,
                    items,
                    producer_done: true,
                    consumer_fenced: false,
                },
                expected,
                error,
                plan,
                queue: Some(queue),
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
    fn publish(&mut self, ticket: &TransferTicket, current: Epochs) -> Result<Vec<BankLease>> {
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
        let output: Vec<_> = p.ids.iter().map(|id| p.records[id].clone()).collect();
        if p.demand {
            for id in &p.ids {
                self.heat.demand(id);
            }
        }
        p.published = true;
        self.cache
            .extend(p.records.iter().map(|(id, l)| (id.clone(), l.clone())));
        self.trim();
        Ok(output)
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
        let p = self.pending.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        if !p.host_use_done {
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
    fn release(&mut self, lease: &BankLease) -> Result<()> {
        self.can_release(lease)?;
        lease.retire_backing()?;
        self.budget.borrow_mut().release(lease.charge())?;
        self.cache
            .retain(|_, r| r.charge().id() != lease.charge().id());
        self.owned.remove(&lease.charge().id());
        Ok(())
    }
}
