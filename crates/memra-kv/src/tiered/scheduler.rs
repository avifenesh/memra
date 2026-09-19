//! Nonblocking owner-thread admission driver over the frozen TierAdmissionPlan.
//! Queued work is uncharged. TierStore::admit is the ONLY reservation path.
//! This is CPU control-plane integration, not an installed server scheduler.
use super::*;
use std::collections::BTreeMap;

struct Pending {
    admission: TierAdmission,
    policy: PrefetchPolicy,
    plan: Option<TierAdmissionPlan>,
    cancelled: bool,
    published: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    Queued,
    Phase(Phase),
    Retired,
}
pub struct Scheduler<S: TierStore> {
    store: S,
    limit: usize,
    next: u64,
    sequence: u64,
    requests: BTreeMap<u64, Pending>,
    last_served: HashMap<Digest, u64>,
}
impl<S: TierStore> Scheduler<S> {
    pub fn new(store: S, limit: usize) -> Self {
        Self {
            store,
            limit,
            next: 0,
            sequence: 0,
            requests: BTreeMap::new(),
            last_served: HashMap::new(),
        }
    }
    pub fn enqueue(&mut self, admission: TierAdmission, policy: PrefetchPolicy) -> Result<u64> {
        admission.request.validate()?;
        if self.requests.len() >= self.limit {
            return Err(Error::Capacity);
        }
        self.next = self.next.checked_add(1).ok_or(Error::Overflow)?;
        self.requests.insert(
            self.next,
            Pending {
                admission,
                policy,
                plan: None,
                cancelled: false,
                published: false,
            },
        );
        Ok(self.next)
    }
    pub fn cancel(&mut self, id: u64) -> Result<CancelState> {
        let p = self.requests.get_mut(&id).ok_or(Error::NotFound)?;
        if p.published {
            return Ok(CancelState::AlreadyPublished);
        }
        if let Some(plan) = &p.plan {
            let result = self.store.cancel(&plan.reservation)?;
            p.cancelled = true;
            Ok(result)
        } else {
            self.requests.remove(&id);
            self.prune();
            Ok(CancelState::PublicationRevoked)
        }
    }
    fn prune(&mut self) {
        self.last_served.retain(|tenant, _| {
            self.requests
                .values()
                .any(|p| &p.admission.request.tenant == tenant)
        });
    }
    /// Admit at most one request per tick; advance every admitted request once.
    /// Priority > deadline > least recently served tenant > FIFO, with no bypass of a
    /// capacity-blocked head. Fairness is within equal priority/deadline, never priority inversion.
    /// current() reads the native owner's LIVE epochs, not the captured admission epochs.
    pub fn tick(
        &mut self,
        now: u64,
        peers: &[u32],
        current: impl Fn(u64) -> Epochs,
    ) -> Result<Vec<(u64, Result<Progress>)>> {
        let mut events = vec![];
        let expired: Vec<_> = self
            .requests
            .iter()
            .filter(|(_, p)| p.plan.is_none() && p.admission.request.deadline.0 <= now)
            .map(|(&id, _)| id)
            .collect();
        for id in expired {
            self.cancel(id)?;
            events.push((id, Err(Error::Deadline)));
        }
        let first = self
            .requests
            .iter()
            .filter(|(_, p)| p.plan.is_none())
            .min_by_key(|(id, p)| {
                let r = &p.admission.request;
                (
                    r.priority,
                    r.deadline,
                    self.last_served.get(&r.tenant).copied().unwrap_or(0),
                    **id,
                )
            })
            .map(|(&id, _)| id);
        if let Some(id) = first {
            let p = self.requests.get_mut(&id).unwrap();
            let attempt = (|| {
                p.admission.epochs.require(current(id))?;
                let hit = self
                    .store
                    .lookup(&p.admission.id, peers, p.admission.target_device)?
                    .ok_or(Error::NotFound)?;
                p.admission.source = hit.tier;
                // Lookup remains advisory: admit rechecks identity and physical source lease.
                self.store.admit(p.admission.clone())
            })();
            match attempt {
                Ok(reservation) => {
                    self.sequence = self.sequence.checked_add(1).ok_or(Error::Overflow)?;
                    self.last_served
                        .insert(p.admission.request.tenant, self.sequence);
                    p.plan = Some(TierAdmissionPlan {
                        reservation,
                        policy: p.policy,
                    });
                    events.push((id, Ok(Progress::Phase(Phase::Reserved))));
                }
                Err(Error::Capacity | Error::Busy) => events.push((id, Ok(Progress::Queued))),
                Err(e) => {
                    self.requests.remove(&id);
                    events.push((id, Err(e)));
                }
            }
        }
        for (&id, p) in &mut self.requests {
            let Some(plan) = &p.plan else { continue };
            if p.cancelled {
                continue;
            }
            let result = (|| {
                // Timeout revokes publication but NEVER turns admitted state into cold recompute.
                if !p.published
                    && (now >= p.admission.request.deadline.0
                        || matches!(plan.policy, PrefetchPolicy::Timeout(d) if now >= d.0))
                {
                    self.store.cancel(&plan.reservation)?;
                    return Err(Error::Deadline);
                }
                let phase = self.store.advance(&plan.reservation, current(id))?;
                match phase {
                    Phase::Reserved => {
                        self.store.prefetch(&plan.reservation)?;
                        Ok(Progress::Phase(
                            if matches!(p.admission.source, Tier::LocalGpu(_) | Tier::PeerGpu(_)) {
                                Phase::Loading
                            } else {
                                Phase::Prefetching
                            },
                        ))
                    }
                    Phase::HostReady => {
                        self.store.load(&plan.reservation)?;
                        Ok(Progress::Phase(Phase::Loading))
                    }
                    _ => Ok(Progress::Phase(phase)),
                }
            })();
            if result.is_err() {
                // Cancel does not credit any bytes. Explicit drain/release below owns that.
                let _ = self.store.cancel(&plan.reservation);
                p.cancelled = true;
            }
            events.push((id, result));
        }
        self.prune();
        Ok(events)
    }
    /// No ReadyBlock escapes on Phase alone. ready() checks the complete sealed consumer
    /// views and LIVE epochs immediately before the owner-thread consumption callback.
    pub fn consume<R>(
        &mut self,
        id: u64,
        current: Epochs,
        consume: impl FnOnce(&BlockLease) -> R,
    ) -> Result<R> {
        let p = self.requests.get_mut(&id).ok_or(Error::NotFound)?;
        if p.cancelled {
            return Err(Error::Cancelled);
        }
        let plan = p.plan.as_ref().ok_or(Error::NotReady)?;
        let block = self
            .store
            .ready(&plan.reservation, &p.admission.program, current)?;
        p.published = true;
        Ok(consume(block))
    }
    /// Caller must observe all native last-use fences. Busy retains the request and quota.
    pub fn release(&mut self, id: u64) -> Result<Progress> {
        let p = self.requests.get(&id).ok_or(Error::NotFound)?;
        let plan = p.plan.as_ref().ok_or(Error::NotReady)?;
        self.store.release(&plan.reservation)?;
        self.requests.remove(&id);
        self.prune();
        Ok(Progress::Retired)
    }
}
