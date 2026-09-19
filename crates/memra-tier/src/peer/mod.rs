//! Day-1 peer contract proposal. Pure metadata; no CUDA or alternate numeric executor.
//! Peer storage is PCIe capacity, never an operand for a remote attention gather.
//! Lead freezes/re-exports shared signatures; implementations remain on the CUDA owner.

pub mod topology;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerError {
    InvalidRange,
    Capacity,
    StaleEpoch,
    Busy,
    Cancelled,
    Unavailable,
    UnknownCompletion,
    ForeignLease,
    AlreadyReleased,
    ShortCopy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerPlan {
    pub device: u32,
    pub bytes: u64,
    /// Namespace/block generation supplied by the common governor, not a local counter.
    pub epoch: u64,
}

/// No raw address is part of a capacity lease. Concrete lease ownership is backend-private.
pub trait PeerLease {
    fn plan(&self) -> PeerPlan;
}

/// A delegate of B's global governor, NOT an independent available-VRAM budget.
pub trait PeerCapacity {
    type Lease: PeerLease;
    fn reserve(&mut self, plan: PeerPlan) -> Result<Self::Lease, PeerError>;
    /// Refuses Busy until I/O, DMA, consumer and graph pins retire. Errors retain the handle.
    /// Cancelled/old epochs still release their accounting after retirement; never leak a charge.
    fn release(&mut self, lease: &Self::Lease) -> Result<(), PeerError>;
}

/// A bounded contiguous extent, never a row-index list or a strided view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContiguousSpan {
    offset: u64,
    bytes: u64,
}
impl ContiguousSpan {
    pub fn new(offset: u64, bytes: u64, registered_bytes: u64) -> Result<Self, PeerError> {
        if bytes == 0
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > registered_bytes)
        {
            return Err(PeerError::InvalidRange);
        }
        Ok(Self { offset, bytes })
    }
    pub fn offset(self) -> u64 {
        self.offset
    }
    pub fn bytes(self) -> u64 {
        self.bytes
    }
}

/// Backend registration retains the actual allocation until every accepted ticket retires.
/// A clone is an ownership pin, not an unchecked pointer copy. Engine implementations resolve
/// handles internally and validate allocation generation and span bounds again at submission.
pub trait PeerRegistration: Clone {
    fn device(&self) -> u32;
    fn bytes(&self) -> u64;
    fn epoch(&self) -> u64;
}

#[derive(Debug, Clone)]
pub struct ContiguousCopy<R: PeerRegistration> {
    pub source: R,
    pub destination: R,
    pub source_span: ContiguousSpan,
    pub destination_span: ContiguousSpan,
    /// Producer-ready event; the backend checks its context/epoch against source registration.
    pub producer_fence: u64,
}
impl<R: PeerRegistration> ContiguousCopy<R> {
    pub fn validate(&self, epoch: u64) -> Result<(), PeerError> {
        if self.source.epoch() != epoch || self.destination.epoch() != epoch {
            return Err(PeerError::StaleEpoch);
        }
        ContiguousSpan::new(
            self.source_span.offset,
            self.source_span.bytes,
            self.source.bytes(),
        )?;
        ContiguousSpan::new(
            self.destination_span.offset,
            self.destination_span.bytes,
            self.destination.bytes(),
        )?;
        if self.source_span.bytes != self.destination_span.bytes {
            return Err(PeerError::InvalidRange);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelState {
    PublicationRevoked,
    AlreadyPublished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerProgress {
    Pending,
    /// Transfer bytes complete; consumer must still wait on consumer_fence before use.
    Complete {
        bytes: u64,
        consumer_fence: u64,
    },
    /// Cancellation is not retirement. Poll until Retired; unknown is quarantined indefinitely.
    CancelPending,
    Retired,
    Quarantined,
}

/// Implemented by native CUDA owner, called by A. No per-row remote gather operation exists.
/// A batch is an array of contiguous memcpy descriptors (NIXL/Mooncake/Tutti design lineage),
/// not a scatter attention program. Each input gets exactly one acceptance result, in order.
/// Source/destination registrations are retained for accepted items even after cancel.
pub trait PeerBackend {
    type Registration: PeerRegistration;
    type Ticket;
    /// A non-forgeable engine-owned permit for LOCAL consumer operands only.
    type LocalReady;
    fn submit(
        &mut self,
        epoch: u64,
        copies: Vec<ContiguousCopy<Self::Registration>>,
    ) -> Vec<Result<Self::Ticket, PeerError>>;
    fn poll(&mut self, ticket: &Self::Ticket) -> Result<PeerProgress, PeerError>;
    fn cancel(&mut self, ticket: &Self::Ticket) -> Result<CancelState, PeerError>;
    /// Validates current epoch, full byte count, destination device == consumer and installs
    /// stream wait. Atomically publishes or refuses; cancel after publish cannot revoke use.
    fn materialize_local(
        &mut self,
        ticket: &Self::Ticket,
        current_epoch: u64,
        consumer: u32,
    ) -> Result<Self::LocalReady, PeerError>;
    /// Records the last local use fence; graph pins must retire before this succeeds.
    fn retire_consumer(&mut self, ready: &Self::LocalReady) -> Result<(), PeerError>;
}

/// B owns StateBundle's concrete shape. This adapter separates committed capture/restore
/// from PP activation transport: activations alone are not a complete continuation.
/// Capture must inventory every required plane and owner alias, including auxiliary history,
/// logits/hidden and transaction high-water marks. Restore refuses missing groups or identity.
pub trait StateBundleAdapter {
    type StateBundle;
    type Identity;
    type RestoreTicket;
    fn capture_committed(&mut self, epoch: u64) -> Result<Self::StateBundle, PeerError>;
    fn restore_committed(
        &mut self,
        bundle: &Self::StateBundle,
        identity: &Self::Identity,
        epoch: u64,
    ) -> Result<Self::RestoreTicket, PeerError>;
}
