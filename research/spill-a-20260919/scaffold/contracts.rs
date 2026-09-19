// LEAD-OWNED: replace with frozen contracts. PROPOSAL, not a runtime ABI.
use crate::pool::PinnedLease;
pub type Digest = [u8; 32];
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidLayout,
    Overflow,
    Capacity,
    NotFound,
    Conflict,
    Corrupt,
    UnsupportedVersion(u32),
    ShortIo {
        expected: u64,
        actual: u64,
    },
    Io {
        kind: std::io::ErrorKind,
        os_code: Option<i32>,
    },
    UnknownTicket,
    StaleEpoch,
    NotReady,
    Cancelled,
    Quarantined,
    Unsupported,
}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io {
            kind: e.kind(),
            os_code: e.raw_os_error(),
        }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectKey {
    pub artifact: Digest,
    pub semantic_id: Digest,
    pub layout: Digest,
    pub generation: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferTicket {
    pub id: u64,
    pub epoch: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStatus {
    Pending,
    HostReady,
    DeviceReady,
    Failed,
    Cancelled,
    Quarantined,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FenceId {
    pub owner: u32,
    pub sequence: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    pub producer_done: bool,
    pub consumer_fenced: bool,
    pub segments: Vec<SegmentCompletion>,
    pub ticket: TransferTicket,
    pub accepted: bool,
    pub status: TransferStatus,
    pub valid_bytes: u64,
    pub io_bytes: u64,
    pub consumer_fence: Option<FenceId>,
    pub error: Option<Error>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelState {
    Retiring,
    Retired,
    Unknown,
}
// IDs resolve only inside the designated CUDA owner; never raw pointers.
#[derive(Debug)]
pub struct DeviceLease {
    pub owner: u32,
    pub allocation: u64,
    pub epoch: u64,
    pub bytes: u64,
}
pub struct CopyOp {
    pub host: PinnedLease,
    pub device: DeviceLease,
    pub bytes: u64,
    pub producer_fence: Option<FenceId>,
}
pub struct PeerCopyOp {
    pub source: DeviceLease,
    pub destination: DeviceLease,
    pub bytes: u64,
    pub producer_fence: FenceId,
}
pub struct ReadPlan {
    pub object: ObjectKey,
    pub chunk: u32,
    pub destination: PinnedLease,
    pub epoch: u64,
}
// Rejected ops return ownership, not just an error that drops a live source.
pub struct Rejected<T> {
    pub op: T,
    pub error: Error,
}
pub trait TransferEngine {
    // Err means zero accepted operations and returns every input. If anything was
    // accepted (including an uncertain submit), return a ticket + EVERY item.
    fn submit_batch(
        &mut self,
        ops: Vec<TransferOp>,
    ) -> std::result::Result<BatchSubmission, Rejected<Vec<TransferOp>>>;
    // true only when disk + DMA + all consumer use is proven retired. Err/unknown
    // is NOT retirement permission. cancel/poll never imply retirement.
    fn retired(&mut self, ticket: TransferTicket) -> Result<bool>;
    fn h2d(&mut self, op: CopyOp) -> std::result::Result<TransferTicket, Rejected<CopyOp>>;
    fn d2h(&mut self, op: CopyOp) -> std::result::Result<TransferTicket, Rejected<CopyOp>>;
    fn p2p(&mut self, op: PeerCopyOp) -> std::result::Result<TransferTicket, Rejected<PeerCopyOp>>;
    fn nvme_read(
        &mut self,
        op: ReadPlan,
    ) -> std::result::Result<TransferTicket, Rejected<ReadPlan>>;
    fn poll(&mut self, ticket: TransferTicket) -> Result<Completion>;
    fn cancel(&mut self, ticket: TransferTicket) -> Result<CancelState>;
    // poll does not retire leases; destination remains protected through consumer use.
    fn retire(&mut self, ticket: TransferTicket, consumer_done: Option<FenceId>) -> Result<()>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentCompletion {
    pub item: u32,
    pub segment: u32,
    pub epoch: u64,
    pub status: TransferStatus,
    pub valid_bytes: u64,
    pub io_bytes: u64,
    pub checksum: Option<Digest>,
    pub producer_done: bool,
    pub consumer_fenced: bool,
    pub consumer_fence: Option<FenceId>,
    pub error: Option<Error>,
}
pub enum TransferOp {
    H2d(CopyOp),
    D2h(CopyOp),
    P2p(PeerCopyOp),
    NvmeRead(ReadPlan),
}
pub enum ItemAcceptance {
    Accepted {
        item: u32,
    },
    // Rejects preserve source/destination ownership and their input index.
    Rejected {
        item: u32,
        op: TransferOp,
        error: Error,
    },
}
pub struct BatchSubmission {
    pub ticket: TransferTicket,
    pub items: Vec<ItemAcceptance>,
}
