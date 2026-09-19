//! Native mapping seam, not another ObjectStore/TransferEngine or payload cache.
use memra_tier::contracts::*;

pub use memra_tier::contracts::{BudgetGovernor, PeerCapacity, PinnedLease};

/// Implement on the existing HostPrefixCache directory (and A's ObjectStore adapter).
/// Acquire owns a real source/backing lease charged by the SAME governor. Preparation
/// retains source/destination resources in shared transfer descriptors, one item per
/// ordered segment. It must not perform CUDA work on an I/O worker. Returned rejected
/// descriptors return to reclaim_unsubmitted (zero acceptance); accepted descriptors belong to A.
/// Destination scratch is covered by reservation.charge, not charged a second time.
/// release/evict must refuse dirty or sole mandatory backing, and leased objects.
pub trait KvBacking<T: TransferEngine> {
    fn lookup(&self, id: &KvBlockId) -> Result<Vec<Lookup>>;
    fn acquire(&mut self, plan: &TierAdmission) -> Result<BlockLease>;
    fn prepare_prefetch(
        &mut self,
        block: &BlockLease,
        plan: &TierAdmission,
        reservation: &TierReservation,
    ) -> Result<Vec<TransferOp<T::Host>>>;
    fn prepare_load(
        &mut self,
        block: &BlockLease,
        plan: &TierAdmission,
        reservation: &TierReservation,
        host_ticket: &TransferTicket,
        transfer: &mut T,
    ) -> Result<Vec<TransferOp<T::Host>>>;
    /// Own and reclaim descriptors returned with ZERO acceptance, including owner-registry
    /// allocation handles. If cleanup is uncertain, retain/quarantine inside the adapter.
    /// Preparation failures must perform the same cleanup before returning an error.
    fn reclaim_unsubmitted(&mut self, ops: Vec<TransferOp<T::Host>>, transfer: &mut T);
    fn release(&mut self, block: &BlockLease) -> Result<()>;
    fn evict(&mut self, id: &KvBlockId) -> Result<()>;
}
