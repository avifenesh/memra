use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{self, ThreadId};

static NEXT_SERVICE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankTicket {
    service: u64,
    sequence: u64,
}

/// A real engine adapter must check the designated CUDA owner, bind the exact transfer
/// ticket, and insert/observe the consumer-stream fence. Disk completion is insufficient.
/// The day-1 test implementation supplies a deterministic fake fence, NOT CUDA evidence.
pub trait ConsumerFence {
    fn ready_on_owner(&mut self, ticket: BankTicket) -> Result<bool>;
}

struct Record {
    id: BankId,
    layout: RecordLayout,
    class: LayoutClass,
    bytes: Vec<u8>,
    _permit: Box<dyn BudgetPermit>,
}
#[derive(Clone)]
pub struct BankLease(Arc<Record>);
impl BankLease {
    pub fn id(&self) -> &BankId {
        &self.0.id
    }
    pub fn layout(&self) -> &RecordLayout {
        &self.0.layout
    }
    pub fn bytes(&self) -> &[u8] {
        &self.0.bytes
    }
}
/// A uniform-only adapter consumes this lease, not a detached catalog proof.
/// It owns exactly the validated bytes/layouts and preserves their budget lifetimes.
/// ```compile_fail
/// use memra_bank_prototype::{BankLease, UniformLease};
/// fn uniform_kernel(_: &UniformLease) {}
/// fn mixed_dispatch(lease: &BankLease) { uniform_kernel(lease); }
/// ```
pub struct UniformLease(Vec<BankLease>);
impl UniformLease {
    pub fn try_new(leases: Vec<BankLease>) -> Result<Self> {
        let first = leases.first().ok_or(BankError::EmptyBatch)?;
        for lease in &leases {
            if lease.0.class != LayoutClass::Uniform || !first.layout().same_program(lease.layout())
            {
                return Err(BankError::MixedLayout);
            }
        }
        Ok(Self(leases))
    }
    pub fn records(&self) -> &[BankLease] {
        &self.0
    }
}

struct PendingRecord {
    id: BankId,
    layout: RecordLayout,
    permit: Box<dyn BudgetPermit>,
}
struct PendingBatch {
    ids: Vec<BankId>,
    demand: bool,
    missing: Vec<PendingRecord>,
    records: BTreeMap<BankId, BankLease>,
    host_ready: bool,
    _queue: Box<dyn BudgetPermit>,
}

pub trait BankedResidency {
    fn layout(&self, id: &BankId) -> Result<&RecordLayout>;
    fn resident(&mut self, id: &BankId) -> Result<Option<BankLease>>;
    fn stage(&mut self, batch: BankBatch) -> Result<BankTicket>;
    fn publish(
        &mut self,
        ticket: BankTicket,
        fence: &mut dyn ConsumerFence,
    ) -> Result<Vec<BankLease>>;
}

/// CPU state-machine skeleton. Cache references and demand leases alias the same charged
/// allocation. Eviction cannot free a borrowed lease; cancellation releases only CPU-owned
/// bytes. Real DMA cancellation/retirement must remain in A's transfer/pinned-lease layer.
/// !Send/!Sync keeps stage/read/publish on the constructing owner thread.
pub struct BankService<D, H> {
    catalog: Catalog,
    budget: SharedBudget,
    hotness: H,
    cache: BTreeMap<BankId, BankLease>,
    pending: BTreeMap<u64, PendingBatch>,
    cache_limit: u64,
    batch_limit: u64,
    item_limit: usize,
    queue_limit: usize,
    owner: ThreadId,
    service: u64,
    sequence: u64,
    _domain: PhantomData<D>,
    _owner_only: PhantomData<Rc<()>>,
}
impl<D: BankDomain, H: Hotness<D>> BankService<D, H> {
    pub fn new(
        catalog: Catalog,
        budget: SharedBudget,
        hotness: H,
        cache_limit: u64,
        batch_limit: u64,
        item_limit: usize,
        queue_limit: usize,
    ) -> Self {
        Self {
            catalog,
            budget,
            hotness,
            cache: BTreeMap::new(),
            pending: BTreeMap::new(),
            cache_limit,
            batch_limit,
            item_limit,
            queue_limit,
            owner: thread::current().id(),
            service: NEXT_SERVICE.fetch_add(1, Ordering::Relaxed),
            sequence: 0,
            _domain: PhantomData,
            _owner_only: PhantomData,
        }
    }
    fn owner(&self) -> Result<()> {
        if thread::current().id() != self.owner {
            return Err(BankError::WrongOwner);
        }
        Ok(())
    }
    fn ticket(&self, ticket: BankTicket) -> Result<u64> {
        self.owner()?;
        if ticket.service != self.service || !self.pending.contains_key(&ticket.sequence) {
            return Err(BankError::UnknownTicket);
        }
        Ok(ticket.sequence)
    }
    /// Reads are explicit, bounded and exact. No cache entry becomes visible here.
    /// A failure retires this CPU-only batch atomically, including successfully read siblings.
    pub fn read(&mut self, ticket: BankTicket, reader: &mut dyn ExactReader) -> Result<()> {
        let seq = self.ticket(ticket)?;
        let mut pending = self.pending.remove(&seq).ok_or(BankError::UnknownTicket)?;
        if pending.host_ready {
            self.pending.insert(seq, pending);
            return Ok(());
        }
        for record in pending.missing.drain(..) {
            let len = usize::try_from(record.layout.bytes()?).map_err(|_| BankError::TooLarge)?;
            let mut bytes = vec![0; len];
            let mut at = 0;
            for seg in &record.layout.segments {
                let end = at + usize::try_from(seg.len).map_err(|_| BankError::TooLarge)?;
                reader.read_exact(&seg.tensor, seg.offset, &mut bytes[at..end])?;
                at = end;
            }
            let id = record.id.clone();
            pending.records.insert(
                id,
                BankLease(Arc::new(Record {
                    id: record.id,
                    layout: record.layout,
                    class: self.catalog.class(),
                    bytes,
                    _permit: record.permit,
                })),
            );
        }
        pending.host_ready = true;
        self.pending.insert(seq, pending);
        Ok(())
    }
    pub fn cancel(&mut self, ticket: BankTicket) -> Result<()> {
        let seq = self.ticket(ticket)?;
        self.pending.remove(&seq);
        Ok(())
    }
    /// Governor-pressure hook: drop cache references only; active leases remain charged.
    pub fn evict_cached(&mut self, id: &BankId) -> Result<bool> {
        self.owner()?;
        self.catalog.layout(id)?;
        Ok(self.cache.remove(id).is_some())
    }
    pub fn cache_bytes(&self) -> u64 {
        self.cache.values().map(|v| v.bytes().len() as u64).sum()
    }
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
    fn trim_cache(&mut self) {
        while self.cache_bytes() > self.cache_limit {
            let victim = self
                .cache
                .keys()
                .min_by_key(|id| self.hotness.score(id))
                .cloned();
            if let Some(victim) = victim {
                self.cache.remove(&victim);
            } else {
                break;
            }
        }
    }
}
impl<D: BankDomain, H: Hotness<D>> BankedResidency for BankService<D, H> {
    fn layout(&self, id: &BankId) -> Result<&RecordLayout> {
        self.catalog.layout(id)
    }
    fn resident(&mut self, id: &BankId) -> Result<Option<BankLease>> {
        self.owner()?;
        self.catalog.layout(id)?;
        if !D::accepts(&id.record) {
            return Err(BankError::InvalidLayout);
        }
        self.hotness.demand(id);
        Ok(self.cache.get(id).cloned())
    }
    fn stage(&mut self, batch: BankBatch) -> Result<BankTicket> {
        self.owner()?;
        if batch.ids.len() > self.item_limit {
            return Err(BankError::TooLarge);
        }
        if self.pending.len() >= self.queue_limit {
            return Err(BankError::Backpressure);
        }
        // Validate against THIS service, not whichever catalog constructed the batch.
        let mut bytes = 0u64;
        let unique: BTreeSet<_> = batch.ids.iter().cloned().collect();
        for id in &unique {
            if !D::accepts(&id.record) {
                return Err(BankError::InvalidLayout);
            }
            bytes = bytes
                .checked_add(self.catalog.layout(id)?.bytes()?)
                .ok_or(BankError::Overflow)?;
        }
        if bytes > self.batch_limit {
            return Err(BankError::TooLarge);
        }
        let sequence = self.sequence.checked_add(1).ok_or(BankError::Overflow)?;
        // All-or-none acceptance: RAII unwinds already-reserved permits on any refusal.
        let queue = self.budget.reserve(Charge {
            inflight: 1,
            ..Charge::default()
        })?;
        let mut missing = Vec::new();
        let mut records = BTreeMap::new();
        for id in unique {
            if let Some(lease) = self.cache.get(&id) {
                records.insert(id, lease.clone());
            } else {
                let layout = self.catalog.layout(&id)?.clone();
                let permit = self.budget.reserve(Charge {
                    host_bytes: layout.bytes()?,
                    ..Charge::default()
                })?;
                missing.push(PendingRecord { id, layout, permit });
            }
        }
        let host_ready = missing.is_empty();
        self.sequence = sequence;
        self.pending.insert(
            sequence,
            PendingBatch {
                ids: batch.ids,
                demand: batch.demand,
                missing,
                records,
                host_ready,
                _queue: queue,
            },
        );
        Ok(BankTicket {
            service: self.service,
            sequence,
        })
    }
    fn publish(
        &mut self,
        ticket: BankTicket,
        fence: &mut dyn ConsumerFence,
    ) -> Result<Vec<BankLease>> {
        let seq = self.ticket(ticket)?;
        if !self.pending[&seq].host_ready || !fence.ready_on_owner(ticket)? {
            return Err(BankError::Pending);
        }
        let pending = self.pending.remove(&seq).ok_or(BankError::UnknownTicket)?;
        let mut output = Vec::with_capacity(pending.ids.len());
        for id in &pending.ids {
            let lease = pending.records.get(id).ok_or(BankError::UnknownId)?.clone();
            if pending.demand {
                self.hotness.demand(id);
            }
            output.push(lease);
        }
        self.cache.extend(pending.records);
        self.trim_cache();
        Ok(output)
    }
}
