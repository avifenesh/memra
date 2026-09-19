//! Explicit CPU-only construction seam for cross-crate tests. Never a CUDA backend.
//! Uses the caller's ONE governor. Routes default unavailable until injected.
use super::{topology::LinkHealth, validate_peer_charge};
use crate::contracts::*;
use std::{cell::RefCell, collections::HashMap, rc::Rc};

// Test-owned live route state, keyed by owner/source -> consumer/destination.
// These injected observations are not driver grants or hardware qualification.
pub struct RouteState {
    pub context_granted: bool,
    pub pool_granted: bool,
    pub link_health: LinkHealth,
}
impl RouteState {
    fn available(&self) -> bool {
        self.context_granted && self.pool_granted && self.link_health == LinkHealth::AtMaximum
    }
}
/// CPU backing and directed observations for capacity/lifetime schedules only.
/// Public owner/governor handles are test hooks, not hardware grants.
pub struct FakePeerCapacity<G: BudgetGovernor> {
    pub gov: Rc<RefCell<G>>,
    pub owners: Vec<DeviceOwner>,
    pub routes: HashMap<(u32, u32), RouteState>,
    pub state: u64,
    charges: HashMap<(u64, u64), bool>,
}
impl<G: BudgetGovernor> FakePeerCapacity<G> {
    pub fn new(gov: Rc<RefCell<G>>, devices: u32, state: u64) -> Self {
        Self {
            gov,
            owners: (0..devices).map(DeviceOwner::new).collect(),
            routes: HashMap::new(),
            state,
            charges: HashMap::new(),
        }
    }
    pub fn set_route(
        &mut self,
        source: u32,
        destination: u32,
        context: bool,
        pool: bool,
        health: LinkHealth,
    ) -> Result<()> {
        if source == destination
            || source as usize >= self.owners.len()
            || destination as usize >= self.owners.len()
        {
            return Err(Error::WrongOwner);
        }
        self.routes.insert(
            (source, destination),
            RouteState {
                context_granted: context,
                pool_granted: pool,
                link_health: health,
            },
        );
        Ok(())
    }
    pub fn route_available(&self, source: u32, destination: u32) -> bool {
        (source as usize) < self.owners.len()
            && (destination as usize) < self.owners.len()
            && self
                .routes
                .get(&(source, destination))
                .is_some_and(RouteState::available)
    }
}
impl<G: BudgetGovernor> PeerCapacity for FakePeerCapacity<G> {
    fn reserve(&mut self, plan: PeerPlan) -> Result<PeerLease> {
        if !self.route_available(plan.owner_device, plan.consumer_device) {
            return Err(Error::Unsupported);
        }
        if plan.epochs.state != self.state {
            return Err(Error::StaleEpoch);
        }
        validate_peer_charge(&plan)?;
        let length = usize::try_from(plan.bytes).map_err(|_| Error::Capacity)?;
        let charge = self.gov.borrow_mut().reserve(&plan.request)?;
        let owner = plan.owner_device as usize;
        let result = self.owners[owner].register(
            plan.epochs.dst_gen,
            plan.bytes,
            Box::new(RefCell::new(vec![0u8; length])),
            &charge,
        );
        match result {
            Ok(device) => {
                self.charges.insert(charge.id(), false);
                Ok(PeerLease {
                    plan,
                    charge,
                    device,
                })
            }
            Err(error) => {
                self.gov.borrow_mut().release(&charge)?;
                Err(error)
            }
        }
    }
    fn release(&mut self, lease: &PeerLease) -> Result<()> {
        let released = self
            .charges
            .get_mut(&lease.charge.id())
            .ok_or(Error::ForeignLease)?;
        if !matches!(
            lease.charge.state()?,
            ChargeState::Reserved | ChargeState::Retired
        ) {
            return Err(Error::Busy);
        }
        if !*released {
            self.owners[lease.device.device() as usize].release(&lease.device)?;
            *released = true;
        }
        self.gov.borrow_mut().release(&lease.charge)?;
        self.charges.remove(&lease.charge.id());
        Ok(())
    }
}
