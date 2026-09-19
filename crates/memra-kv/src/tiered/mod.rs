//! CPU control-plane prototype for extending HostPrefixCache, not a second payload cache.
//! No serving callsites or GPU implementation are enabled by this module.
pub mod integration;
pub mod policy;
#[cfg(test)]
mod tests;

use crate::record::{Digest, KvBlockId, ProgramIdentity, RecordLayout, StateBundle, TierError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    LocalGpu(u32),
    PeerGpu(u32),
    PinnedHost,
    Nvme,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lookup {
    pub tier: Tier,
    pub storage_bytes: u64,
}
/// A backend lease must keep the immutable epoch and ALL its objects alive until release.
/// Handles are opaque IDs here; A's frozen implementation will own actual pinned storage.
#[derive(Debug)]
pub struct BlockLease {
    pub handle: u64,
    pub bundle: StateBundle,
    pub tier: Tier,
}
pub trait ObjectStore {
    fn lookup(&self, id: &KvBlockId) -> Vec<Lookup>;
    fn lease(&mut self, id: &KvBlockId, tier: Tier) -> Result<BlockLease, TierError>;
    fn release(&mut self, lease: BlockLease);
    /// Refuse dirty, leased or sole mandatory backing; actual cache owns this decision.
    fn evict(&mut self, id: &KvBlockId) -> Result<(), TierError>;
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransferTicket(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemStatus {
    Pending,
    Complete,
    Rejected,
    Failed,
}
#[derive(Clone, Debug)]
pub struct ItemCompletion {
    pub segment: usize,
    pub epoch: u64,
    pub status: ItemStatus,
    pub bytes: u64,
    pub checksum: Digest,
}
#[derive(Clone, Debug)]
pub struct Completion {
    pub items: Vec<ItemCompletion>,
    /// Includes disk + DMA completion, NOT just a metadata acknowledgement.
    pub producer_done: bool,
    /// Set only by the CUDA owner after the destination stream wait/event is installed.
    pub consumer_fenced: bool,
}
/// Submission error guarantees no work accepted; partially accepted batches MUST return
/// a ticket listing every segment (including rejects). cancel never authorizes freeing.
pub trait TransferEngine {
    fn prefetch(&mut self, lease: &BlockLease) -> Result<TransferTicket, TierError>;
    fn load(&mut self, lease: &BlockLease, device: u32) -> Result<TransferTicket, TierError>;
    fn poll(&mut self, ticket: TransferTicket) -> Completion;
    fn cancel(&mut self, ticket: TransferTicket);
    /// True only after all disk/DMA AND downstream consumer uses have retired.
    /// Unknown/error fence state MUST return false (quarantine).
    fn retired(&mut self, ticket: TransferTicket) -> bool;
}

/// Explicit physical charges; totals across B/C are to be owned by ONE governor.
/// Vector positions are physical CUDA device ordinals, never TP-divided logical bytes.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct TierBudget {
    pub devices: Vec<u64>,
    pub pinned: u64,
    pub staging: u64,
    pub nvme: u64,
    pub inflight: u64,
}
impl TierBudget {
    fn values(&self) -> impl Iterator<Item = u64> + '_ {
        self.devices
            .iter()
            .copied()
            .chain([self.pinned, self.staging, self.nvme, self.inflight])
    }
    fn fits(&self, used: &Self, cap: &Self) -> bool {
        self.devices.len() == cap.devices.len()
            && used.devices.len() == cap.devices.len()
            && self
                .values()
                .zip(used.values())
                .zip(cap.values())
                .all(|((n, u), c)| n <= c.saturating_sub(u))
    }
    fn add(&mut self, other: &Self) {
        for (v, n) in self.devices.iter_mut().zip(&other.devices) {
            *v += n;
        }
        self.pinned += other.pinned;
        self.staging += other.staging;
        self.nvme += other.nvme;
        self.inflight += other.inflight;
    }
    fn sub(&mut self, other: &Self) {
        for (v, n) in self.devices.iter_mut().zip(&other.devices) {
            *v -= n;
        }
        self.pinned -= other.pinned;
        self.staging -= other.staging;
        self.nvme -= other.nvme;
        self.inflight -= other.inflight;
    }
}
#[derive(Clone, Debug)]
pub struct TierAdmission {
    pub id: KvBlockId,
    pub program: ProgramIdentity,
    /// From the admitted plan/adapter, never inferred from an untrusted stored object.
    pub expected_layout: RecordLayout,
    pub source: Tier,
    pub target_device: u32,
    pub charge: TierBudget,
    pub mandatory: bool,
    /// Present for active state: the owner updates this SAME guard on fork/rollback.
    pub active: Option<Arc<Mutex<ActiveEpoch>>>,
}
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TierReservation(u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Reserved,
    Prefetching,
    HostReady,
    Loading,
    Ready,
    Cancelled,
    Failed,
}
struct Entry {
    lease: BlockLease,
    admission: TierAdmission,
    phase: Phase,
    tickets: Vec<TransferTicket>,
}
/// Scheduler plan holds the reservation, not lookup results. No raw device pointer escapes.
pub struct TierAdmissionPlan {
    pub reservation: TierReservation,
    pub policy: policy::PrefetchPolicy,
}

/// Materializer implementation belongs on the CUDA owner, at the existing attention boundary.
/// Associated operands contain native contiguous planes + stable len/base addresses, NOT
/// page-wise attention reductions. Holding operands keeps all ready leases alive.
pub trait KvMaterializer {
    type Operands;
    fn materialize(
        &mut self,
        bundle: &StateBundle,
        program: &ProgramIdentity,
        ready: &ReadyBlock,
        stable_target: u64,
    ) -> Result<Self::Operands, TierError>;
}
/// Cannot be constructed outside this module: disk completion cannot mint this capability.
#[derive(Debug)]
pub struct ReadyBlock {
    reservation: u64,
    id: KvBlockId,
    device: u32,
}
impl ReadyBlock {
    pub fn id(&self) -> &KvBlockId {
        &self.id
    }
    pub fn device(&self) -> u32 {
        self.device
    }
    pub fn reservation_id(&self) -> u64 {
        self.reservation
    }
}

pub trait TierStore {
    fn lookup(&self, id: &KvBlockId, eligible_peers: &[u32], local: u32) -> Option<Lookup>;
    fn admit(&mut self, admission: TierAdmission) -> Result<TierReservation, TierError>;
    fn prefetch(&mut self, reservation: &TierReservation) -> Result<(), TierError>;
    fn load(&mut self, reservation: &TierReservation) -> Result<(), TierError>;
    fn advance(&mut self, reservation: &TierReservation) -> Result<Phase, TierError>;
    fn cancel(&mut self, reservation: &TierReservation) -> Result<(), TierError>;
    fn retire(&mut self, reservation: &TierReservation) -> Result<bool, TierError>;
    fn evict(&mut self, id: &KvBlockId) -> Result<(), TierError>;
}
/// Metadata coordinator only. Object bytes remain in the existing host tier / A's store.
/// The governor is local to this prototype; integration must inject the shared B/C governor.
pub struct Hierarchy<S: ObjectStore, T: TransferEngine> {
    store: Option<S>,
    transfer: Option<T>,
    capacity: TierBudget,
    optional_capacity: TierBudget,
    used: TierBudget,
    entries: HashMap<u64, Entry>,
    next: u64,
}
impl<S: ObjectStore, T: TransferEngine> Hierarchy<S, T> {
    pub fn new(
        store: S,
        transfer: T,
        capacity: TierBudget,
        mandatory_headroom: TierBudget,
    ) -> Result<Self, TierError> {
        let zero = TierBudget {
            devices: vec![0; capacity.devices.len()],
            ..TierBudget::default()
        };
        if !mandatory_headroom.fits(&zero, &capacity) {
            return Err(TierError::Capacity);
        }
        let mut optional_capacity = capacity.clone();
        optional_capacity.sub(&mandatory_headroom);
        Ok(Self {
            store: Some(store),
            transfer: Some(transfer),
            capacity,
            optional_capacity,
            used: zero,
            entries: HashMap::new(),
            next: 0,
        })
    }
    fn check_active(&mut self, r: &TierReservation) -> Result<(), TierError> {
        let e = self.entries.get(&r.0).ok_or(TierError::Missing)?;
        let valid = match &e.admission.active {
            Some(active) => active
                .lock()
                .map_err(|_| TierError::Backend)?
                .accepts(&e.lease.bundle),
            None => true,
        };
        if !valid {
            self.cancel(r)?;
            return Err(TierError::StaleEpoch);
        }
        Ok(())
    }
    pub fn used(&self) -> &TierBudget {
        &self.used
    }
    pub fn phase(&self, r: &TierReservation) -> Result<Phase, TierError> {
        Ok(self.entries.get(&r.0).ok_or(TierError::Missing)?.phase)
    }
    /// READY is borrowed only for a synchronous owner callback; no clonable ready permit can
    /// survive cancel/retire and be presented later. Callback must register consumer fences
    /// with TransferEngine before returning if it launches async compute.
    pub fn with_ready<R>(
        &mut self,
        r: &TierReservation,
        program: &ProgramIdentity,
        consume: impl FnOnce(&StateBundle, &ReadyBlock, &mut T) -> R,
    ) -> Result<R, TierError> {
        self.check_active(r)?;
        let e = self.entries.get(&r.0).ok_or(TierError::Missing)?;
        if e.phase != Phase::Ready {
            return Err(TierError::WrongState);
        }
        if program != &e.lease.bundle.program {
            return Err(TierError::ProgramMismatch);
        }
        let ready = ReadyBlock {
            reservation: r.0,
            id: e.lease.bundle.id.clone(),
            device: e.admission.target_device,
        };
        Ok(consume(
            &e.lease.bundle,
            &ready,
            self.transfer.as_mut().expect("backend present until drop"),
        ))
    }
}
impl<S: ObjectStore, T: TransferEngine> TierStore for Hierarchy<S, T> {
    fn lookup(&self, id: &KvBlockId, eligible_peers: &[u32], local: u32) -> Option<Lookup> {
        self.store
            .as_ref()
            .expect("backend present until drop")
            .lookup(id)
            .into_iter()
            .filter_map(|hit| {
                let rank = match hit.tier {
                    Tier::LocalGpu(d) if d == local => 0,
                    Tier::PeerGpu(d) if eligible_peers.contains(&d) => 1,
                    Tier::PinnedHost => 2,
                    Tier::Nvme => 3,
                    _ => return None,
                };
                Some((rank, hit))
            })
            .min_by_key(|(rank, _)| *rank)
            .map(|(_, hit)| hit)
    }
    fn admit(&mut self, a: TierAdmission) -> Result<TierReservation, TierError> {
        if a.id.namespace != a.program.namespace() {
            return Err(TierError::ProgramMismatch);
        }
        let cap = if a.mandatory {
            &self.capacity
        } else {
            &self.optional_capacity
        };
        if !a.charge.fits(&self.used, cap) {
            return Err(TierError::Capacity);
        }
        let next = self.next.checked_add(1).ok_or(TierError::Overflow)?;
        // Re-check advisory hit under an actual backend lease before reserving bytes.
        let lease = self
            .store
            .as_mut()
            .expect("backend present until drop")
            .lease(&a.id, a.source)?;
        let validation = (|| {
            lease.bundle.validate()?;
            if lease.bundle.layout != a.expected_layout {
                return Err(TierError::InvalidLayout);
            }
            if let Some(active) = &a.active {
                if !active
                    .lock()
                    .map_err(|_| TierError::Backend)?
                    .accepts(&lease.bundle)
                {
                    return Err(TierError::StaleEpoch);
                }
            }
            if lease.bundle.id != a.id {
                return Err(TierError::StaleEpoch);
            }
            if lease.bundle.program != a.program {
                return Err(TierError::ProgramMismatch);
            }
            if lease.tier != a.source {
                return Err(TierError::Missing);
            }
            let bytes = lease.bundle.layout.storage_bytes()?;
            let device = a
                .charge
                .devices
                .get(a.target_device as usize)
                .ok_or(TierError::Capacity)?;
            // Entire exact attention operand must fit. Staging may stream in bounded slots.
            if *device < bytes || a.charge.inflight == 0 || a.charge.staging == 0 {
                return Err(TierError::Capacity);
            }
            Ok(())
        })();
        if let Err(err) = validation {
            self.store
                .as_mut()
                .expect("backend present until drop")
                .release(lease);
            return Err(err);
        }
        self.used.add(&a.charge);
        self.next = next;
        self.entries.insert(
            next,
            Entry {
                lease,
                admission: a,
                phase: Phase::Reserved,
                tickets: vec![],
            },
        );
        Ok(TierReservation(next))
    }
    fn prefetch(&mut self, r: &TierReservation) -> Result<(), TierError> {
        self.check_active(r)?;
        let e = self.entries.get_mut(&r.0).ok_or(TierError::Missing)?;
        if e.phase != Phase::Reserved {
            return Err(TierError::WrongState);
        }
        let ticket = self
            .transfer
            .as_mut()
            .expect("backend present until drop")
            .prefetch(&e.lease)?;
        e.tickets.push(ticket);
        e.phase = Phase::Prefetching;
        Ok(())
    }
    fn load(&mut self, r: &TierReservation) -> Result<(), TierError> {
        self.check_active(r)?;
        let e = self.entries.get_mut(&r.0).ok_or(TierError::Missing)?;
        if e.phase != Phase::HostReady {
            return Err(TierError::WrongState);
        }
        let ticket = self
            .transfer
            .as_mut()
            .expect("backend present until drop")
            .load(&e.lease, e.admission.target_device)?;
        e.tickets.push(ticket);
        e.phase = Phase::Loading;
        Ok(())
    }
    fn advance(&mut self, r: &TierReservation) -> Result<Phase, TierError> {
        self.check_active(r)?;
        let e = self.entries.get_mut(&r.0).ok_or(TierError::Missing)?;
        if !matches!(e.phase, Phase::Prefetching | Phase::Loading) {
            return Ok(e.phase);
        }
        let c = self
            .transfer
            .as_mut()
            .expect("backend present until drop")
            .poll(*e.tickets.last().ok_or(TierError::WrongState)?);
        let result = complete(&e.lease.bundle, &c);
        match result {
            Err(err) => {
                e.phase = Phase::Failed;
                for t in &e.tickets {
                    self.transfer
                        .as_mut()
                        .expect("backend present until drop")
                        .cancel(*t);
                }
                Err(err)
            }
            Ok(false) => Ok(e.phase),
            Ok(true) => {
                if e.phase == Phase::Prefetching {
                    e.phase = Phase::HostReady;
                } else if c.consumer_fenced {
                    e.phase = Phase::Ready;
                }
                Ok(e.phase)
            }
        }
    }
    fn cancel(&mut self, r: &TierReservation) -> Result<(), TierError> {
        let e = self.entries.get_mut(&r.0).ok_or(TierError::Missing)?;
        e.phase = Phase::Cancelled;
        for t in &e.tickets {
            self.transfer
                .as_mut()
                .expect("backend present until drop")
                .cancel(*t);
        }
        Ok(())
    }
    fn evict(&mut self, id: &KvBlockId) -> Result<(), TierError> {
        if self.entries.values().any(|e| &e.lease.bundle.id == id) {
            return Err(TierError::Busy);
        }
        self.store
            .as_mut()
            .expect("backend present until drop")
            .evict(id)
    }
    fn retire(&mut self, r: &TierReservation) -> Result<bool, TierError> {
        let e = self.entries.get(&r.0).ok_or(TierError::Missing)?;
        if !matches!(e.phase, Phase::Cancelled | Phase::Failed) {
            return Err(TierError::Busy);
        }
        if !e.tickets.iter().all(|t| {
            self.transfer
                .as_mut()
                .expect("backend present until drop")
                .retired(*t)
        }) {
            return Ok(false);
        }
        let e = self.entries.remove(&r.0).ok_or(TierError::Missing)?;
        self.used.sub(&e.admission.charge);
        self.store
            .as_mut()
            .expect("backend present until drop")
            .release(e.lease);
        Ok(true)
    }
}
fn complete(bundle: &StateBundle, c: &Completion) -> Result<bool, TierError> {
    if c.items.len() != bundle.layout.segments.len() {
        return Err(TierError::Incomplete);
    }
    let mut seen = vec![false; c.items.len()];
    let mut all = c.producer_done;
    for i in &c.items {
        let s = bundle
            .layout
            .segments
            .get(i.segment)
            .ok_or(TierError::Incomplete)?;
        if seen[i.segment] {
            return Err(TierError::Incomplete);
        }
        seen[i.segment] = true;
        if i.epoch != bundle.id.epoch {
            return Err(TierError::StaleEpoch);
        }
        match i.status {
            ItemStatus::Rejected => return Err(TierError::Rejected),
            ItemStatus::Failed => return Err(TierError::Backend),
            ItemStatus::Pending => all = false,
            ItemStatus::Complete => {
                if i.bytes != s.storage_bytes {
                    return Err(TierError::ShortRead);
                }
                if i.checksum != bundle.checksums[i.segment] {
                    return Err(TierError::Corrupt);
                }
            }
        }
    }
    Ok(all)
}

/// Epoch-only active lineage. Appends/rollback fork instead of mutating a leased generation.
/// Bytes and recurrent replay are still the existing Cache owner's responsibility.
#[derive(Clone, Debug)]
pub struct ActiveEpoch {
    epoch: u64,
    committed: u64,
    staged: u64,
}
impl ActiveEpoch {
    pub fn new(committed: u64) -> Self {
        Self {
            epoch: 0,
            committed,
            staged: committed,
        }
    }
    pub fn fork(&mut self, staged: u64) -> Result<u64, TierError> {
        if staged < self.committed {
            return Err(TierError::Incomplete);
        }
        self.epoch = self.epoch.checked_add(1).ok_or(TierError::Overflow)?;
        self.staged = staged;
        Ok(self.epoch)
    }
    pub fn commit(&mut self, epoch: u64, high_water: u64) -> Result<(), TierError> {
        if epoch != self.epoch {
            return Err(TierError::StaleEpoch);
        }
        if high_water < self.committed || high_water > self.staged {
            return Err(TierError::Incomplete);
        }
        self.committed = high_water;
        Ok(())
    }
    pub fn rollback(&mut self, high_water: u64) -> Result<(), TierError> {
        if high_water > self.committed {
            return Err(TierError::Incomplete);
        }
        self.epoch = self.epoch.checked_add(1).ok_or(TierError::Overflow)?;
        self.committed = high_water;
        self.staged = high_water;
        Ok(())
    }
    pub fn accepts(&self, bundle: &StateBundle) -> bool {
        bundle.id.epoch == self.epoch
            && bundle.id.end <= self.committed
            && bundle.committed_high_water <= self.committed
    }
}

/// Shutdown must obey the same retirement rule as cancel. Unknown completion deliberately
/// quarantines backend owners and leases instead of dropping potentially DMA-live memory.
impl<S: ObjectStore, T: TransferEngine> Drop for Hierarchy<S, T> {
    fn drop(&mut self) {
        let transfer = self.transfer.as_mut().expect("backend present until drop");
        for e in self.entries.values() {
            for t in &e.tickets {
                transfer.cancel(*t);
            }
        }
        let safe = self
            .entries
            .values()
            .all(|e| e.tickets.iter().all(|t| transfer.retired(*t)));
        if !safe {
            std::mem::forget(self.store.take());
            std::mem::forget(self.transfer.take());
            std::mem::forget(std::mem::take(&mut self.entries));
            return;
        }
        let store = self.store.as_mut().expect("backend present until drop");
        for (_, e) in self.entries.drain() {
            store.release(e.lease);
        }
    }
}
