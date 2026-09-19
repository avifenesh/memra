//! Additional PROPOSED day-1 interface shapes for A/D; no implementations or CUDA pointers.
use super::{TierAdmission, TierBudget, TransferTicket};
use crate::record::TierError;

/// A-owned aligned/quota-charged pool capability. Owner outlives both producer and consumer.
/// Drop while an operation is outstanding must quarantine, never return the slot to the pool.
pub trait PinnedLease {
    fn allocation_id(&self) -> u64;
    fn storage_bytes(&self) -> u64;
    fn alignment(&self) -> u32;
    fn numa_node(&self) -> Option<u32>;
}
#[derive(Clone, Debug)]
pub struct PeerPlan {
    pub owner_device: u32,
    pub consumer_device: u32,
    pub bytes: u64,
    pub alignment: u32,
    pub epoch: u64,
}
#[derive(Debug)]
pub struct PeerLease {
    pub id: u64,
    pub plan: PeerPlan,
}
pub trait PeerCapacity {
    /// Must check grants/topology, actual free/reserved bytes, and global governor permit.
    fn reserve(&mut self, plan: PeerPlan) -> Result<PeerLease, TierError>;
    /// Refuse release before the consuming operation retires, including failed/cancelled work.
    fn release(&mut self, lease: &PeerLease, retired: TransferTicket) -> Result<(), TierError>;
}
/// ONE instance, shared across KV, banks, loader, scratch and transfer slots. B implementation
/// follows lead freeze; prototype Hierarchy owns only an isolated CPU test ledger for now.
pub trait AdmissionGovernor {
    type Permit;
    fn reserve(&mut self, admission: &TierAdmission) -> Result<Self::Permit, TierError>;
    fn used(&self) -> TierBudget;
    fn release(&mut self, permit: Self::Permit) -> Result<(), TierError>;
}
