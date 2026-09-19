//! PCIe peer transport uses the frozen shared contracts, never a parallel schema.
//! Native CUDA submission is not implemented here. The owner-thread CPU fake lives in tests.
pub mod topology;
pub use crate::contracts::{
    BatchSubmission, CancelState, Completion, ContiguousCopy, ContiguousSpan, DeviceLease,
    DeviceOwner, Epochs, ItemAcceptance, ItemOutcome, PeerBackend, PeerCapacity, PeerError,
    PeerLease, PeerPlan, ReadyView, StateBundleAdapter, TransferTicket,
};
use crate::contracts::{BudgetRequest, Error, Result};

/// Validate the physical and overlapping peer quota dimensions before asking the ONE
/// injected governor to reserve. No available-VRAM counter or independent allocator.
/// Native allocation padding must already be included in plan.bytes/request.
pub fn validate_peer_charge(plan: &PeerPlan) -> Result<&BudgetRequest> {
    plan.request.validate()?;
    let owner = plan.owner_device as usize;
    let consumer = plan.consumer_device as usize;
    if owner == consumer || consumer >= plan.request.bytes.device.len() {
        return Err(Error::WrongOwner);
    }
    if plan.bytes == 0
        || !plan.alignment.is_power_of_two()
        || !plan.bytes.is_multiple_of(u64::from(plan.alignment))
    {
        return Err(Error::InvalidLayout);
    }
    if plan.request.bytes.device.get(owner).copied().unwrap_or(0) < plan.bytes
        || plan.request.bytes.peer.get(owner).copied().unwrap_or(0) < plan.bytes
    {
        return Err(Error::Capacity);
    }
    Ok(&plan.request)
}
