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
    /// Local/peer sources bypass disk/host staging. Implementations must produce
    /// owner-bound direct descriptors; unsupported routes refuse, never host-bounce.
    fn prepare_direct(
        &mut self,
        _block: &BlockLease,
        _plan: &TierAdmission,
        _reservation: &TierReservation,
        _transfer: &mut T,
    ) -> Result<Vec<TransferOp<T::Host>>> {
        Err(Error::Unsupported)
    }
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

/// B's peer-source admission side of the frozen capacity seam. Lookup is advisory:
/// PeerCapacity rechecks directed live grants before reserving the source allocation.
/// This reserves SOURCE capacity only; local operands require their own admission and
/// fenced materialization. It never grants permission to read remote attention rows.
pub fn reserve_peer_source<P: PeerCapacity>(
    capacity: &mut P,
    hit: &Lookup,
    admission: &TierAdmission,
) -> Result<PeerLease> {
    admission.request.validate()?;
    admission.id.validate()?;
    admission.program.validate()?;
    admission.expected_layout.validate()?;
    if admission.id.namespace != admission.program.namespace()?
        || admission.request.tenant != admission.program.tenant_salt
    {
        return Err(Error::ProgramMismatch);
    }
    if admission.id.epoch != admission.epochs.state
        || admission.id.end > admission.committed_high_water
    {
        return Err(Error::StaleEpoch);
    }
    let Tier::PeerGpu(owner) = hit.tier else {
        return Err(Error::Unsupported);
    };
    if hit.tier != admission.source || owner == admission.target_device {
        return Err(Error::WrongOwner);
    }
    let bytes = admission.expected_layout.storage_bytes()?;
    if hit.storage_bytes != bytes {
        return Err(Error::InvalidLayout);
    }
    let mut request = admission.request.clone();
    request.bytes = TierBudget::zero(request.bytes.device.len());
    *request
        .bytes
        .device
        .get_mut(owner as usize)
        .ok_or(Error::WrongOwner)? = bytes;
    request.bytes.peer[owner as usize] = bytes;
    let alignment = admission
        .expected_layout
        .segments
        .iter()
        .map(|s| s.alignment)
        .max()
        .ok_or(Error::Incomplete)?;
    capacity.reserve(PeerPlan {
        owner_device: owner,
        consumer_device: admission.target_device,
        bytes,
        alignment,
        // PeerCapacity allocates its owner-side backing using dst_gen. This source
        // later becomes src_gen in the actual source→local-consumer copy descriptor.
        epochs: Epochs {
            dst_gen: admission.epochs.src_gen,
            ..admission.epochs
        },
        request,
    })
}
