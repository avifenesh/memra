//! Exact-byte KV coordination, extending the existing native payload owner via KvBacking.
//! CPU-tested only; no server path is enabled here.
pub mod hostprefix;
pub mod integration;
pub mod materializer;
pub mod policy;
#[cfg(test)]
mod tests;

use integration::KvBacking;
pub use memra_tier::contracts::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

struct Entry {
    plan: TierAdmission,
    block: BlockLease,
    phase: Phase,
    tickets: Vec<TransferTicket>,
    pin: Option<LeasePin>,
    published: bool,
}
/// The caller injects the SAME governor used by storage, banks, pools and peers.
/// No independent capacity ledger exists in the hierarchy.
pub struct Hierarchy<B: KvBacking<T>, T: TransferEngine, G: BudgetGovernor> {
    backing: Option<B>,
    transfer: Option<T>,
    governor: Arc<Mutex<G>>,
    entries: HashMap<(u64, u64), Entry>,
}
impl<B: KvBacking<T>, T: TransferEngine, G: BudgetGovernor> Hierarchy<B, T, G> {
    pub fn new(backing: B, transfer: T, governor: Arc<Mutex<G>>) -> Self {
        Self {
            backing: Some(backing),
            transfer: Some(transfer),
            governor,
            entries: HashMap::new(),
        }
    }
    pub fn phase(&self, r: &TierReservation) -> Result<Phase> {
        Ok(self
            .entries
            .get(&r.charge.id())
            .ok_or(Error::ForeignLease)?
            .phase)
    }
    fn epochs(&mut self, r: &TierReservation, current: Epochs) -> Result<()> {
        let e = self
            .entries
            .get(&r.charge.id())
            .ok_or(Error::ForeignLease)?;
        if e.plan.epochs != current {
            self.cancel(r)?;
            return Err(Error::StaleEpoch);
        }
        Ok(())
    }
    fn fail(&mut self, r: &TierReservation, error: Error) -> Error {
        let e = self
            .entries
            .get_mut(&r.charge.id())
            .expect("validated reservation");
        e.phase = if matches!(error, Error::Quarantined | Error::UnknownTicket) {
            Phase::Quarantined
        } else {
            Phase::Failed
        };
        for t in &e.tickets {
            let _ = self.transfer.as_mut().unwrap().cancel(t);
        }
        error
    }
    fn submit(
        &mut self,
        r: &TierReservation,
        ops: Vec<TransferOp<T::Host>>,
        phase: Phase,
    ) -> Result<TransferTicket> {
        let e = self
            .entries
            .get_mut(&r.charge.id())
            .ok_or(Error::ForeignLease)?;
        let validation = (|| {
            if ops.len() != e.block.bundle.layout.segments.len() {
                return Err(Error::Incomplete);
            }
            for op in &ops {
                let epoch = match op {
                    TransferOp::H2d(o) | TransferOp::D2h(o) => o.epochs,
                    TransferOp::P2p(o) => o.epochs,
                    TransferOp::NvmeRead(o) => o.epochs,
                };
                epoch.require(e.plan.epochs)?;
            }
            Ok(())
        })();
        if let Err(err) = validation {
            self.backing
                .as_mut()
                .unwrap()
                .reclaim_unsubmitted(ops, self.transfer.as_mut().unwrap());
            return Err(err);
        }
        let count = ops.len();
        let submitted = match self.transfer.as_mut().unwrap().submit_batch(ops) {
            Ok(s) => s,
            Err(rejected) => {
                self.backing
                    .as_mut()
                    .unwrap()
                    .reclaim_unsubmitted(rejected.op, self.transfer.as_mut().unwrap());
                return Err(rejected.error);
            }
        };
        // Keep the ticket even on malformed/partial acceptance: accepted resources cannot free.
        e.tickets.push(submitted.ticket);
        e.phase = phase;
        let validate = submitted
            .validate(count)
            .and_then(|()| submitted.ticket.epochs.require(e.plan.epochs));
        let rejected: Vec<_> = submitted
            .items
            .into_iter()
            .filter_map(|i| match i {
                ItemAcceptance::Rejected { op, .. } => Some(op),
                _ => None,
            })
            .collect();
        let partial = !rejected.is_empty();
        self.backing
            .as_mut()
            .unwrap()
            .reclaim_unsubmitted(rejected, self.transfer.as_mut().unwrap());
        if let Err(err) = validate {
            return Err(self.fail(r, err));
        }
        if partial {
            return Err(self.fail(r, Error::Rejected));
        }
        Ok(submitted.ticket)
    }
}
fn expected(bundle: &StateBundle) -> Vec<Vec<SegmentExpectation>> {
    bundle
        .layout
        .segments
        .iter()
        .zip(&bundle.checksums)
        .map(|(s, h)| {
            vec![SegmentExpectation {
                valid_bytes: s.valid_bytes,
                io_bytes: s.storage_bytes,
                checksum: *h,
            }]
        })
        .collect()
}
impl<B: KvBacking<T>, T: TransferEngine, G: BudgetGovernor> TierStore for Hierarchy<B, T, G> {
    fn lookup(&self, id: &KvBlockId, peers: &[u32], local: u32) -> Result<Option<Lookup>> {
        id.validate()?;
        Ok(self
            .backing
            .as_ref()
            .unwrap()
            .lookup(id)?
            .into_iter()
            .filter_map(|hit| {
                let rank = match hit.tier {
                    Tier::LocalGpu(d) if d == local => 0,
                    Tier::PeerGpu(d) if peers.contains(&d) => 1,
                    Tier::PinnedHost => 2,
                    Tier::Nvme => 3,
                    _ => return None,
                };
                Some((rank, hit))
            })
            .min_by_key(|(rank, _)| *rank)
            .map(|(_, hit)| hit))
    }
    fn admit(&mut self, p: TierAdmission) -> Result<TierReservation> {
        p.id.validate()?;
        p.program.validate()?;
        p.expected_layout.validate()?;
        p.request.validate()?;
        if p.id.namespace != p.program.namespace()? {
            return Err(Error::ProgramMismatch);
        }
        if p.id.epoch != p.epochs.state || p.id.end > p.committed_high_water {
            return Err(Error::StaleEpoch);
        }
        if p.request.tenant != p.program.tenant_salt {
            return Err(Error::ProgramMismatch);
        }
        if p.request
            .bytes
            .device
            .get(p.target_device as usize)
            .copied()
            .unwrap_or(0)
            < p.expected_layout.storage_bytes()?
            || p.request.bytes.inflight == 0
        {
            return Err(Error::Capacity);
        }
        let charge = self
            .governor
            .lock()
            .map_err(|_| Error::Quarantined)?
            .reserve(&p.request)?;
        let block = match self.backing.as_mut().unwrap().acquire(&p) {
            Ok(b) => b,
            Err(err) => {
                self.governor
                    .lock()
                    .map_err(|_| Error::Quarantined)?
                    .release(&charge)?;
                return Err(err);
            }
        };
        let validation = block
            .bundle
            .require(
                &p.program,
                &p.expected_layout,
                p.epochs.state,
                p.committed_high_water,
            )
            .and_then(|()| {
                if block.bundle.id != p.id || block.tier != p.source {
                    Err(Error::Conflict)
                } else {
                    Ok(())
                }
            });
        if let Err(err) = validation {
            self.backing.as_mut().unwrap().release(&block)?;
            self.governor
                .lock()
                .map_err(|_| Error::Quarantined)?
                .release(&charge)?;
            return Err(err);
        }
        let pin = charge.pin()?;
        self.entries.insert(
            charge.id(),
            Entry {
                plan: p,
                block,
                phase: Phase::Reserved,
                tickets: vec![],
                pin: Some(pin),
                published: false,
            },
        );
        Ok(TierReservation { charge })
    }
    fn prefetch(&mut self, r: &TierReservation) -> Result<TransferTicket> {
        let e = self
            .entries
            .get(&r.charge.id())
            .ok_or(Error::ForeignLease)?;
        if e.phase != Phase::Reserved {
            return Err(Error::NotReady);
        }
        let ops = self
            .backing
            .as_mut()
            .unwrap()
            .prepare_prefetch(&e.block, &e.plan, r)?;
        self.submit(r, ops, Phase::Prefetching)
    }
    fn load(&mut self, r: &TierReservation) -> Result<TransferTicket> {
        let e = self
            .entries
            .get(&r.charge.id())
            .ok_or(Error::ForeignLease)?;
        if e.phase != Phase::HostReady {
            return Err(Error::NotReady);
        }
        let ops = self.backing.as_mut().unwrap().prepare_load(
            &e.block,
            &e.plan,
            r,
            e.tickets.last().ok_or(Error::NotReady)?,
            self.transfer.as_mut().unwrap(),
        )?;
        self.submit(r, ops, Phase::Loading)
    }
    fn advance(&mut self, r: &TierReservation, current: Epochs) -> Result<Phase> {
        self.epochs(r, current)?;
        let e = self
            .entries
            .get(&r.charge.id())
            .ok_or(Error::ForeignLease)?;
        if !matches!(e.phase, Phase::Prefetching | Phase::Loading) {
            return Ok(e.phase);
        }
        let ticket = *e.tickets.last().ok_or(Error::NotReady)?;
        let device = e.phase == Phase::Loading;
        let check = self
            .transfer
            .as_mut()
            .unwrap()
            .poll(&ticket)
            .and_then(|c| c.require(&ticket, &expected(&e.block.bundle), device));
        match check {
            Err(Error::NotReady) => Ok(e.phase),
            Err(err) => Err(self.fail(r, err)),
            Ok(()) => {
                let e = self.entries.get_mut(&r.charge.id()).unwrap();
                e.phase = if device {
                    Phase::Ready
                } else {
                    Phase::HostReady
                };
                Ok(e.phase)
            }
        }
    }
    fn ready(
        &mut self,
        r: &TierReservation,
        program: &ProgramIdentity,
        current: Epochs,
    ) -> Result<&BlockLease> {
        self.epochs(r, current)?;
        let e = self
            .entries
            .get_mut(&r.charge.id())
            .ok_or(Error::ForeignLease)?;
        if e.phase != Phase::Ready {
            return Err(Error::NotReady);
        }
        e.block.bundle.require(
            program,
            &e.plan.expected_layout,
            current.state,
            e.plan.committed_high_water,
        )?;
        let ticket = e.tickets.last().ok_or(Error::NotReady)?;
        // Validate sealed owner-bound views, not the aggregate fence booleans alone.
        // No operands escape before ALL items validate. A partial backend publication
        // failure is quarantined (no retry/reuse); backends retain every accepted input.
        for item in 0..e.block.bundle.layout.segments.len() {
            let view = self
                .transfer
                .as_mut()
                .unwrap()
                .ready_view(ticket, item as u32, current);
            match view {
                Ok(v) if v.destination().device() == e.plan.target_device => (),
                result => {
                    e.phase = Phase::Quarantined;
                    return Err(match result {
                        Err(err) => err,
                        _ => Error::WrongOwner,
                    });
                }
            }
        }
        e.published = true;
        // BlockLease retains source provenance/charge; consumer device authority is
        // the transfer-owned ReadyView, never a relabelled host lease.
        Ok(&e.block)
    }
    fn cancel(&mut self, r: &TierReservation) -> Result<CancelState> {
        let e = self
            .entries
            .get_mut(&r.charge.id())
            .ok_or(Error::ForeignLease)?;
        if e.published {
            return Ok(CancelState::AlreadyPublished);
        }
        e.phase = Phase::Cancelled;
        let mut published = false;
        for t in &e.tickets {
            match self.transfer.as_mut().unwrap().cancel(t) {
                Ok(CancelState::AlreadyPublished) => published = true,
                Ok(CancelState::PublicationRevoked) => (),
                Err(err) => {
                    e.phase = Phase::Quarantined;
                    return Err(err);
                }
            }
        }
        if published {
            e.phase = Phase::Quarantined;
            return Ok(CancelState::AlreadyPublished);
        }
        Ok(CancelState::PublicationRevoked)
    }
    fn retire(&mut self, r: &TierReservation) -> Result<bool> {
        let e = self
            .entries
            .get_mut(&r.charge.id())
            .ok_or(Error::ForeignLease)?;
        if e.phase == Phase::Retired {
            return Ok(true);
        }
        if !e.published
            && !matches!(
                e.phase,
                Phase::Cancelled | Phase::Failed | Phase::Quarantined
            )
        {
            return Err(Error::Busy);
        }
        for t in &e.tickets {
            // Unknown remains retained. Errors are not evidence of retirement.
            if !self.transfer.as_mut().unwrap().retired(t)? {
                return Ok(false);
            }
        }
        e.phase = Phase::Retired;
        Ok(true)
    }
    fn release(&mut self, r: &TierReservation) -> Result<()> {
        if !self.retire(r)? {
            return Err(Error::Busy);
        }
        let e = self
            .entries
            .get_mut(&r.charge.id())
            .ok_or(Error::ForeignLease)?;
        // Acknowledge only on explicit release: all consumers must have handed back
        // their handles first; acknowledge may refuse Busy and preserves retry state.
        while let Some(t) = e.tickets.first() {
            self.transfer.as_mut().unwrap().acknowledge(t)?;
            e.tickets.remove(0);
        }
        if e.block.charge.state()? != ChargeState::Released {
            self.backing.as_mut().unwrap().release(&e.block)?;
        }
        e.pin.take();
        self.governor
            .lock()
            .map_err(|_| Error::Quarantined)?
            .release(&r.charge)?;
        self.entries.remove(&r.charge.id());
        Ok(())
    }
    fn evict(&mut self, id: &KvBlockId) -> Result<()> {
        if self.entries.values().any(|e| &e.block.bundle.id == id) {
            return Err(Error::Busy);
        }
        self.backing.as_mut().unwrap().evict(id)
    }
}
impl<B: KvBacking<T>, T: TransferEngine, G: BudgetGovernor> Drop for Hierarchy<B, T, G> {
    fn drop(&mut self) {
        // Explicit release is the only credit path. Unknown shutdown retains backend
        // owners AND source/reservation pins; do not rely on a fake's destructor.
        if !self.entries.is_empty() {
            for e in self.entries.values() {
                for t in &e.tickets {
                    let _ = self.transfer.as_mut().unwrap().cancel(t);
                }
            }
            std::mem::forget(self.backing.take());
            std::mem::forget(self.transfer.take());
            std::mem::forget(std::mem::take(&mut self.entries));
            std::mem::forget(self.governor.clone());
        }
    }
}

/// Native owner metadata guard. Every COW/rollback invalidates outstanding state tickets;
/// allocation generations remain independent. Bytes/recurrent replay are native work.
#[derive(Clone, Debug)]
pub struct ActiveEpoch {
    epochs: Epochs,
    committed: u64,
    staged: u64,
}
impl ActiveEpoch {
    pub fn new(epochs: Epochs, committed: u64) -> Self {
        Self {
            epochs,
            committed,
            staged: committed,
        }
    }
    pub fn current(&self) -> Epochs {
        self.epochs
    }
    pub fn fork(&mut self, staged: u64) -> Result<Epochs> {
        if staged < self.committed {
            return Err(Error::Incomplete);
        }
        self.epochs.state = self.epochs.state.checked_add(1).ok_or(Error::Overflow)?;
        self.staged = staged;
        Ok(self.epochs)
    }
    pub fn commit(&mut self, epoch: Epochs, high_water: u64) -> Result<()> {
        self.epochs.require(epoch)?;
        if high_water < self.committed || high_water > self.staged {
            return Err(Error::Incomplete);
        }
        self.committed = high_water;
        Ok(())
    }
    pub fn rollback(&mut self, high_water: u64) -> Result<()> {
        if high_water > self.committed {
            return Err(Error::Incomplete);
        }
        self.epochs.state = self.epochs.state.checked_add(1).ok_or(Error::Overflow)?;
        self.committed = high_water;
        self.staged = high_water;
        Ok(())
    }
    pub fn reallocate(&mut self, src_gen: u64, dst_gen: u64) {
        self.epochs.src_gen = src_gen;
        self.epochs.dst_gen = dst_gen;
    }
    pub fn require(&self, bundle: &StateBundle) -> Result<()> {
        if bundle.kind != StateKind::Active {
            return Err(Error::InvalidLayout);
        }
        bundle.require(
            &bundle.program,
            &bundle.layout,
            self.epochs.state,
            self.committed,
        )
    }
}
