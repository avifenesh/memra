//! Atomic multidimensional admission; fixed backing is charged once, slices hold pins.
use crate::contracts::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub enum QueueOutcome {
    Admitted(u64, ChargedLease),
    Expired(u64),
}

pub struct Governor {
    issuer: LeaseIssuer,
    used: TierBudget,
    capacity: TierBudget,
    optional_capacity: TierBudget,
    clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    queue_limit: usize,
    dirty_limit: u64,
    dirty: HashMap<Digest, u64>,
    queue: Vec<(u64, BudgetRequest)>,
    last_served: HashMap<Digest, u64>,
    service_sequence: u64,
    evicted_floor: u64,
    charged_tenants: HashMap<(u64, u64), Digest>,
    next: u64,
}
impl Governor {
    /// Clock returns monotonic ns in the same domain as every request deadline.
    /// Limits are supplied by the owner, not hardware defaults or synthetic tuning.
    pub fn new(
        capacity: TierBudget,
        mandatory_headroom: TierBudget,
        queue_limit: usize,
        dirty_limit: u64,
        clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> Result<Self> {
        capacity.validate()?;
        let optional_capacity = capacity.checked_sub(&mandatory_headroom)?;
        Ok(Self {
            issuer: LeaseIssuer::default(),
            used: TierBudget::zero(capacity.device.len()),
            capacity,
            optional_capacity,
            clock,
            queue_limit,
            dirty_limit,
            dirty: HashMap::new(),
            queue: vec![],
            last_served: HashMap::new(),
            service_sequence: 0,
            evicted_floor: 0,
            charged_tenants: HashMap::new(),
            next: 0,
        })
    }
    pub fn set_dirty(&mut self, tenant: Digest, bytes: u64) -> Result<()> {
        let total = self
            .dirty
            .iter()
            .filter(|(t, _)| **t != tenant)
            .try_fold(bytes, |n, (_, b)| n.checked_add(*b).ok_or(Error::Overflow))?;
        if total > self.dirty_limit {
            return Err(Error::Capacity);
        }
        if bytes == 0 {
            self.dirty.remove(&tenant);
        } else {
            self.dirty.insert(tenant, bytes);
        }
        Ok(())
    }
    pub fn enqueue(&mut self, request: BudgetRequest) -> Result<u64> {
        request.validate()?;
        if request.deadline.0 <= (self.clock)() {
            return Err(Error::Deadline);
        }
        if self.queue.len() >= self.queue_limit {
            return Err(Error::Capacity);
        }
        self.next = self.next.checked_add(1).ok_or(Error::Overflow)?;
        self.queue.push((self.next, request));
        Ok(self.next)
    }
    fn prune_fairness(&mut self) {
        let active: HashSet<_> = self
            .queue
            .iter()
            .map(|(_, r)| r.tenant)
            .chain(self.charged_tenants.values().copied())
            .collect();
        let mut idle: Vec<_> = self
            .last_served
            .iter()
            .filter(|(tenant, _)| !active.contains(*tenant))
            .map(|(&tenant, &stamp)| (stamp, tenant))
            .collect();
        idle.sort_unstable();
        let excess = idle.len().saturating_sub(self.queue_limit);
        for (stamp, tenant) in idle.into_iter().take(excess) {
            self.last_served.remove(&tenant);
            self.evicted_floor = self.evicted_floor.max(stamp);
        }
    }
    pub fn cancel_queued(&mut self, id: u64) -> Result<()> {
        let i = self
            .queue
            .iter()
            .position(|(n, _)| *n == id)
            .ok_or(Error::NotFound)?;
        self.queue.remove(i);
        self.prune_fairness();
        Ok(())
    }
    /// Priority, deadline, least-recently-served tenant, FIFO. A blocked head is not
    /// bypassed by lower-priority work; caller retries after release/backpressure.
    /// Unknown/forgotten history is treated as served no later than the oldest
    /// forgotten tenant. The eviction watermark prevents a forgotten tenant from
    /// jumping ahead of an older queued stamp; only idle history is capped.
    pub fn dispatch(&mut self) -> Result<Option<QueueOutcome>> {
        if let Some(i) = self
            .queue
            .iter()
            .position(|(_, r)| r.deadline.0 <= (self.clock)())
        {
            let id = self.queue.remove(i).0;
            self.prune_fairness();
            return Ok(Some(QueueOutcome::Expired(id)));
        }
        let Some((i, _)) = self.queue.iter().enumerate().min_by_key(|(_, (n, r))| {
            (
                r.priority,
                r.deadline,
                self.last_served
                    .get(&r.tenant)
                    .copied()
                    .unwrap_or(self.evicted_floor),
                *n,
            )
        }) else {
            return Ok(None);
        };
        let (id, request) = self.queue[i].clone();
        let next = self
            .service_sequence
            .checked_add(1)
            .ok_or(Error::Overflow)?;
        let charge = match self.reserve_now(&request) {
            Err(Error::Capacity) => return Ok(None),
            r => r?,
        };
        self.queue.remove(i);
        self.service_sequence = next;
        self.last_served.insert(request.tenant, next);
        self.prune_fairness();
        Ok(Some(QueueOutcome::Admitted(id, charge)))
    }
    fn reserve_now(&mut self, r: &BudgetRequest) -> Result<ChargedLease> {
        r.validate()?;
        if r.deadline.0 <= (self.clock)() {
            return Err(Error::Deadline);
        }
        let cap = if r.priority == Priority::MandatoryActive {
            &self.capacity
        } else {
            &self.optional_capacity
        };
        if !r.bytes.fits(&self.used, cap)? {
            return Err(Error::Capacity);
        }
        let next = self.used.checked_add(&r.bytes)?;
        let lease = self.issuer.issue(r.bytes.clone())?;
        self.used = next;
        self.charged_tenants.insert(lease.id(), r.tenant);
        Ok(lease)
    }
}
impl BudgetGovernor for Governor {
    fn reserve(&mut self, r: &BudgetRequest) -> Result<ChargedLease> {
        // New arrivals cannot bypass queued work of equal/higher priority.
        if self
            .queue
            .iter()
            .any(|(_, q)| q.deadline.0 > (self.clock)() && q.priority <= r.priority)
        {
            return Err(Error::Busy);
        }
        self.reserve_now(r)
    }
    fn used(&self) -> TierBudget {
        self.used.clone()
    }
    fn mark(&mut self, l: &ChargedLease, s: ChargeState) -> Result<()> {
        self.issuer.mark(l, s)
    }
    fn release(&mut self, l: &ChargedLease) -> Result<()> {
        // Capability validation precedes accounting, including foreign/double release.
        self.issuer.release(l)?;
        self.used = self.used.checked_sub(l.bytes())?;
        self.charged_tenants.remove(&l.id());
        self.prune_fairness();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fairness_idle_cap_keeps_inflight_and_queued_stamps_and_eviction_floor() {
        let mut cap = TierBudget::zero(1);
        cap.pageable = 100;
        let mut g = Governor::new(cap, TierBudget::zero(1), 2, 0, Arc::new(|| 0)).unwrap();
        let request = |tenant, priority| {
            let mut bytes = TierBudget::zero(1);
            bytes.pageable = 1;
            BudgetRequest {
                bytes,
                priority,
                deadline: Deadline(10),
                tenant: [tenant; 32],
            }
        };
        let admit = |g: &mut Governor, want| {
            let Some(QueueOutcome::Admitted(id, charge)) = g.dispatch().unwrap() else {
                panic!()
            };
            assert_eq!(id, want);
            charge
        };
        // B has the oldest stamp and a backlog; C has an outstanding charge only.
        let b1 = g.enqueue(request(1, Priority::Demand)).unwrap();
        let b = admit(&mut g, b1);
        g.release(&b).unwrap();
        let c1 = g.enqueue(request(2, Priority::Demand)).unwrap();
        let c = admit(&mut g, c1);
        let b2 = g.enqueue(request(1, Priority::Demand)).unwrap();
        for tenant in 3..12 {
            let id = g
                .enqueue(request(tenant, Priority::MandatoryActive))
                .unwrap();
            let charge = admit(&mut g, id);
            g.release(&charge).unwrap();
            assert!(g.last_served.len() <= 4); // 2 protected + 2 idle
            assert_eq!(g.last_served[&[1; 32]], 1);
            assert_eq!(g.last_served[&[2; 32]], 2);
        }
        assert_eq!(g.evicted_floor, 9);
        assert_eq!(g.last_served[&[10; 32]], 10);
        assert_eq!(g.last_served[&[11; 32]], 11);
        let again = g.enqueue(request(3, Priority::Demand)).unwrap();
        let b = admit(&mut g, b2); // evicted tenant cannot jump ahead of backlog
        g.release(&b).unwrap();
        let a = admit(&mut g, again);
        g.release(&a).unwrap();
        g.release(&c).unwrap();
        assert!(g.last_served.len() <= 2);
    }
}
