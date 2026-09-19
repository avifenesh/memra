//! tier-contract-v1. Metadata never authorizes a different numerical program.
//!
//! Persist only explicitly versioned metadata, never live tickets/leases/fences.
//! Wire v1 is compact UTF-8 serde JSON in declaration order (no maps/floats),
//! externally tagged enums, decimal fixed-width integers and 32-element digest arrays.
//! Decode must reject unknown fields/versions and noncanonical bytes. See `Wire`.
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::cell::{Ref, RefCell};
use std::collections::{BTreeSet, HashMap};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};

pub const WIRE_VERSION: u32 = 1;
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
    ProgramMismatch,
    WrongOwner,
    ForeignLease,
    AlreadyReleased,
    NotReady,
    Busy,
    Cancelled,
    Quarantined,
    Unsupported,
    Incomplete,
    EmptyBatch,
    Rejected,
    MaskedId,
    MixedLayout,
    Deadline,
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
pub type TierError = Error;
pub type PeerError = Error;
pub type BankError = Error;
pub type BudgetError = Error;

/// Frame domain and payload lengths before SHA-256; never hash ambiguous concatenations.
pub fn digest(domain: &str, bytes: &[u8]) -> Digest {
    let mut h = Sha256::new();
    h.update(b"memra-tier\0v1\0");
    h.update((domain.len() as u64).to_le_bytes());
    h.update(domain.as_bytes());
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(bytes);
    h.finalize().into()
}
pub fn checksum(valid_bytes: &[u8]) -> Digest {
    digest("valid-bytes", valid_bytes)
}
fn version(v: u32) -> Result<()> {
    if v == WIRE_VERSION {
        Ok(())
    } else {
        Err(Error::UnsupportedVersion(v))
    }
}
/// Validate every nested version and semantic field before persistence or use.
/// Reject noncanonical encodings on decode; bound input before allocating metadata.
pub trait Wire: Serialize + for<'de> Deserialize<'de> {
    const DOMAIN: &'static str;
    fn validate(&self) -> Result<()>;
    fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|_| Error::Corrupt)
    }
    fn decode(bytes: &[u8]) -> Result<Self>
    where
        Self: Sized,
    {
        if bytes.len() > 16 * 1024 * 1024 {
            return Err(Error::Capacity);
        }
        let value: Self = serde_json::from_slice(bytes).map_err(|_| Error::Corrupt)?;
        if value.encode()? != bytes {
            return Err(Error::Corrupt);
        }
        Ok(value)
    }
    fn identity(&self) -> Result<Digest> {
        Ok(digest(Self::DOMAIN, &self.encode()?))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramIdentity {
    pub version: u32,
    pub artifact: Digest,
    pub serialized_plan: Digest,
    pub numeric: Digest,
    pub stream: Digest,
    pub tokenizer: Digest,
    pub template: Digest,
    pub adapter: Digest,
    pub modality: Digest,
    pub position: Digest,
    pub tenant_salt: Digest,
}
impl Wire for ProgramIdentity {
    const DOMAIN: &'static str = "program";
    fn validate(&self) -> Result<()> {
        version(self.version)
    }
}
impl ProgramIdentity {
    pub fn namespace(&self) -> Result<Digest> {
        self.identity()
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectKey {
    pub version: u32,
    pub artifact: Digest,
    pub semantic_id: Digest,
    pub layout: Digest,
    pub generation: u64,
}
impl Wire for ObjectKey {
    const DOMAIN: &'static str = "object-key";
    fn validate(&self) -> Result<()> {
        version(self.version)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvBlockId {
    pub version: u32,
    pub namespace: Digest,
    pub parent: Digest,
    pub tokens: Digest,
    pub start: u64,
    pub end: u64,
    pub group: u32,
    pub owner: u32,
    pub epoch: u64,
}
impl Wire for KvBlockId {
    const DOMAIN: &'static str = "kv-block";
    fn validate(&self) -> Result<()> {
        version(self.version)?;
        if self.start >= self.end {
            return Err(Error::InvalidLayout);
        }
        Ok(())
    }
}
impl KvBlockId {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        program: &ProgramIdentity,
        parent: Digest,
        tokens: &[u32],
        start: u64,
        group: u32,
        owner: u32,
        epoch: u64,
    ) -> Result<Self> {
        if tokens.is_empty() {
            return Err(Error::EmptyBatch);
        }
        let mut bytes = (tokens.len() as u64).to_le_bytes().to_vec();
        for t in tokens {
            bytes.extend(t.to_le_bytes());
        }
        Ok(Self {
            version: WIRE_VERSION,
            namespace: program.namespace()?,
            parent,
            tokens: digest("tokens", &bytes),
            start,
            end: start
                .checked_add(tokens.len() as u64)
                .ok_or(Error::Overflow)?,
            group,
            owner,
            epoch,
        })
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Role {
    Key,
    Value,
    Payload,
    Scale,
    MacroScale,
    Window,
    Recurrent,
    History,
    Tail,
    Selection,
    Draft,
    Logits,
    Hidden,
    Transaction,
    Adapter(u32),
}
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncodingId {
    pub version: u32,
    pub program: Digest,
    pub row_bytes: u64,
}
impl Wire for EncodingId {
    const DOMAIN: &'static str = "encoding";
    fn validate(&self) -> Result<()> {
        version(self.version)?;
        if self.row_bytes == 0 {
            return Err(Error::InvalidLayout);
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ByteSegment {
    pub version: u32,
    pub group: u32,
    pub page: u64,
    pub owner: u32,
    pub role: Role,
    /// Original tensor identity; None for request-owned dynamic state.
    pub tensor: Option<TensorId>,
    pub offset: u64,
    pub valid_bytes: u64,
    pub storage_bytes: u64,
    pub alignment: u32,
    pub encoding: EncodingId,
}
impl Wire for ByteSegment {
    const DOMAIN: &'static str = "segment";
    fn validate(&self) -> Result<()> {
        version(self.version)?;
        self.encoding.validate()?;
        if let Some(t) = &self.tensor {
            t.validate()?;
        }
        if self.valid_bytes == 0
            || self.valid_bytes > self.storage_bytes
            || !self.alignment.is_power_of_two()
            || !self.storage_bytes.is_multiple_of(u64::from(self.alignment))
            || !self.offset.is_multiple_of(u64::from(self.alignment))
        {
            return Err(Error::InvalidLayout);
        }
        self.offset
            .checked_add(self.storage_bytes)
            .ok_or(Error::Overflow)?;
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PageRequirement {
    AllPages,
    TrailingPages(u64),
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupRequirement {
    pub version: u32,
    pub group: u32,
    pub owner: u32,
    pub role: Role,
    /// Per-group count, not inferred from a global token count or TP degree.
    pub page_count: u64,
    pub pages: PageRequirement,
}
impl Wire for GroupRequirement {
    const DOMAIN: &'static str = "group";
    fn validate(&self) -> Result<()> {
        version(self.version)?;
        if self.page_count == 0 || matches!(self.pages, PageRequirement::TrailingPages(0)) {
            return Err(Error::InvalidLayout);
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordLayout {
    pub version: u32,
    pub segments: Vec<ByteSegment>,
    pub requirements: Vec<GroupRequirement>,
}
impl Wire for RecordLayout {
    const DOMAIN: &'static str = "layout";
    fn validate(&self) -> Result<()> {
        version(self.version)?;
        if self.requirements.is_empty() {
            return Err(Error::Incomplete);
        }
        let mut keys = BTreeSet::new();
        let mut groups = BTreeSet::new();
        for s in &self.segments {
            s.validate()?;
            if !keys.insert((s.group, s.owner, s.role, s.page)) {
                return Err(Error::InvalidLayout);
            }
        }
        for r in &self.requirements {
            r.validate()?;
            if !groups.insert((r.group, r.owner, r.role)) {
                return Err(Error::InvalidLayout);
            }
            let first = match r.pages {
                PageRequirement::AllPages => 0,
                PageRequirement::TrailingPages(n) => r.page_count.saturating_sub(n),
            };
            let pages: Vec<_> = keys
                .iter()
                .filter(|&&(g, o, k, _)| (g, o, k) == (r.group, r.owner, r.role))
                .collect();
            if pages
                .iter()
                .any(|&&(_, _, _, p)| p < first || p >= r.page_count)
                || pages.len() as u64 != r.page_count - first
            {
                return Err(Error::Incomplete);
            }
        }
        if self
            .segments
            .iter()
            .any(|s| !groups.contains(&(s.group, s.owner, s.role)))
        {
            return Err(Error::InvalidLayout);
        }
        self.storage_bytes()?;
        Ok(())
    }
}
impl RecordLayout {
    pub fn storage_bytes(&self) -> Result<u64> {
        self.segments.iter().try_fold(0u64, |n, s| {
            n.checked_add(s.storage_bytes).ok_or(Error::Overflow)
        })
    }
    /// Compare compute geometry, not physical offsets or original tensor names.
    pub fn same_program(&self, other: &Self) -> bool {
        self.version == other.version
            && self.requirements == other.requirements
            && self.segments.len() == other.segments.len()
            && self.segments.iter().zip(&other.segments).all(|(a, b)| {
                (
                    a.group,
                    a.page,
                    a.owner,
                    a.role,
                    a.valid_bytes,
                    a.storage_bytes,
                    a.alignment,
                    &a.encoding,
                ) == (
                    b.group,
                    b.page,
                    b.owner,
                    b.role,
                    b.valid_bytes,
                    b.storage_bytes,
                    b.alignment,
                    &b.encoding,
                )
            })
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnerAlias {
    pub version: u32,
    pub group: u32,
    pub consumer: u32,
    pub owner: u32,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateKind {
    ImmutablePrefix,
    Active,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateBundle {
    pub version: u32,
    pub id: KvBlockId,
    pub program: ProgramIdentity,
    pub layout: RecordLayout,
    pub kind: StateKind,
    pub committed_high_water: u64,
    pub owner_aliases: Vec<OwnerAlias>,
    pub checksums: Vec<Digest>,
}
impl Wire for StateBundle {
    const DOMAIN: &'static str = "state-bundle";
    fn validate(&self) -> Result<()> {
        version(self.version)?;
        self.id.validate()?;
        self.program.validate()?;
        self.layout.validate()?;
        if self.id.namespace != self.program.namespace()? {
            return Err(Error::ProgramMismatch);
        }
        if self.id.end > self.committed_high_water
            || self.checksums.len() != self.layout.segments.len()
        {
            return Err(Error::Incomplete);
        }
        if self.kind == StateKind::ImmutablePrefix && self.id.epoch != 0 {
            return Err(Error::StaleEpoch);
        }
        let mut aliases = BTreeSet::new();
        for a in &self.owner_aliases {
            version(a.version)?;
            if !aliases.insert((a.group, a.consumer))
                || !self
                    .layout
                    .segments
                    .iter()
                    .any(|s| s.group == a.group && s.owner == a.owner)
            {
                return Err(Error::InvalidLayout);
            }
        }
        Ok(())
    }
}
impl StateBundle {
    pub fn verify(&self, payloads: &[Vec<u8>]) -> Result<()> {
        self.validate()?;
        if payloads.len() != self.layout.segments.len() {
            return Err(Error::Incomplete);
        }
        for ((s, h), b) in self
            .layout
            .segments
            .iter()
            .zip(&self.checksums)
            .zip(payloads)
        {
            if b.len() as u64 != s.storage_bytes {
                return Err(Error::ShortIo {
                    expected: s.storage_bytes,
                    actual: b.len() as u64,
                });
            }
            let n = usize::try_from(s.valid_bytes).map_err(|_| Error::Overflow)?;
            if checksum(&b[..n]) != *h || b[n..].iter().any(|&v| v != 0) {
                return Err(Error::Corrupt);
            }
        }
        Ok(())
    }
    /// Seal only committed bytes. The native owner must first perform byte COW;
    /// sealing metadata never turns an uncommitted allocation into immutable storage.
    pub fn seal(&self) -> Result<Self> {
        self.validate()?;
        let mut b = self.clone();
        b.kind = StateKind::ImmutablePrefix;
        b.id.epoch = 0;
        b.committed_high_water = b.id.end;
        b.validate()?;
        Ok(b)
    }
    pub fn require(
        &self,
        program: &ProgramIdentity,
        layout: &RecordLayout,
        state_epoch: u64,
        high_water: u64,
    ) -> Result<()> {
        self.validate()?;
        if &self.program != program {
            return Err(Error::ProgramMismatch);
        }
        if &self.layout != layout {
            return Err(Error::InvalidLayout);
        }
        if self.id.epoch != state_epoch || self.committed_high_water > high_water {
            return Err(Error::StaleEpoch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Epochs {
    pub state: u64,
    pub src_gen: u64,
    pub dst_gen: u64,
}
impl Epochs {
    pub fn require(self, current: Self) -> Result<()> {
        if self == current {
            Ok(())
        } else {
            Err(Error::StaleEpoch)
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferTicket {
    pub issuer: u64,
    pub sequence: u64,
    pub epochs: Epochs,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FenceId {
    pub issuer: u64,
    pub owner: u32,
    pub generation: u64,
    pub sequence: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemStatus {
    Pending,
    Complete,
    Rejected,
    Failed,
    Cancelled,
    Quarantined,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentCompletion {
    pub segment: u32,
    pub status: ItemStatus,
    pub valid_bytes: u64,
    pub io_bytes: u64,
    pub checksum: Option<Digest>,
    pub epochs: Epochs,
    pub producer_done: bool,
    /// A wait is installed; this does NOT mean consumer execution has finished.
    pub consumer_fenced: bool,
    pub consumer_fence: Option<FenceId>,
    pub error: Option<Error>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemOutcome {
    pub item: u32,
    pub accepted: bool,
    pub segments: Vec<SegmentCompletion>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    pub ticket: TransferTicket,
    pub items: Vec<ItemOutcome>,
    pub producer_done: bool,
    pub consumer_fenced: bool,
}
#[derive(Debug, Clone)]
pub struct SegmentExpectation {
    pub valid_bytes: u64,
    pub io_bytes: u64,
    pub checksum: Digest,
}
impl Completion {
    /// Reject missing/duplicate/reordered/rejected/short/corrupt entries, not just bad aggregates.
    pub fn require(
        &self,
        ticket: &TransferTicket,
        expected: &[Vec<SegmentExpectation>],
        device: bool,
    ) -> Result<()> {
        if &self.ticket != ticket {
            return Err(Error::StaleEpoch);
        }
        if expected.is_empty() || self.items.len() != expected.len() {
            return Err(Error::Incomplete);
        }
        for (i, (item, segments)) in self.items.iter().zip(expected).enumerate() {
            if item.item as usize != i
                || segments.is_empty()
                || item.segments.len() != segments.len()
            {
                return Err(Error::Incomplete);
            }
            if !item.accepted {
                return Err(Error::Rejected);
            }
            for (j, (s, e)) in item.segments.iter().zip(segments).enumerate() {
                if s.segment as usize != j {
                    return Err(Error::Incomplete);
                }
                s.epochs.require(ticket.epochs)?;
                match s.status {
                    ItemStatus::Complete => (),
                    ItemStatus::Pending => return Err(Error::NotReady),
                    ItemStatus::Rejected => return Err(Error::Rejected),
                    ItemStatus::Cancelled => return Err(Error::Cancelled),
                    ItemStatus::Quarantined => return Err(Error::Quarantined),
                    ItemStatus::Failed => return Err(s.error.clone().unwrap_or(Error::Corrupt)),
                }
                if s.error.is_some() {
                    return Err(Error::Corrupt);
                }
                if s.valid_bytes != e.valid_bytes || s.io_bytes != e.io_bytes {
                    return Err(Error::ShortIo {
                        expected: e.io_bytes,
                        actual: s.io_bytes,
                    });
                }
                if s.checksum != Some(e.checksum) {
                    return Err(Error::Corrupt);
                }
                if !s.producer_done
                    || (device && (!s.consumer_fenced || s.consumer_fence.is_none()))
                {
                    return Err(Error::NotReady);
                }
            }
        }
        if !self.producer_done || (device && !self.consumer_fenced) {
            return Err(Error::NotReady);
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelState {
    PublicationRevoked,
    AlreadyPublished,
}
#[derive(Debug)]
pub struct Rejected<T> {
    pub op: T,
    pub error: Error,
}
pub type Submission<T> = std::result::Result<BatchSubmission<T>, Rejected<Vec<T>>>;
#[derive(Debug)]
pub enum ItemAcceptance<T> {
    Accepted { item: u32 },
    Rejected { item: u32, op: T, error: Error },
}
#[derive(Debug)]
pub struct BatchSubmission<T> {
    pub ticket: TransferTicket,
    pub items: Vec<ItemAcceptance<T>>,
}
impl<T> BatchSubmission<T> {
    pub fn validate(&self, count: usize) -> Result<()> {
        if count == 0 || self.items.len() != count {
            return Err(Error::Incomplete);
        }
        let mut accepted = false;
        for (i, entry) in self.items.iter().enumerate() {
            let index = match entry {
                ItemAcceptance::Accepted { item } => {
                    accepted = true;
                    *item
                }
                ItemAcceptance::Rejected { item, .. } => *item,
            };
            if index as usize != i {
                return Err(Error::Incomplete);
            }
        }
        if !accepted {
            return Err(Error::Rejected);
        }
        Ok(())
    }
}

/// Every field is an independent quota dimension. `device` includes all physical
/// device allocations; `peer`, `replicas`, `loaders`, `staging` are overlapping
/// ceilings, NOT additional bytes to sum into physical memory a second time.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TierBudget {
    pub version: u32,
    pub device: Vec<u64>,
    pub peer: Vec<u64>,
    pub replicas: Vec<u64>,
    pub pinned: u64,
    pub pageable: u64,
    pub staging: u64,
    pub loaders: u64,
    pub nvme: u64,
    pub inflight: u64,
}
impl Wire for TierBudget {
    const DOMAIN: &'static str = "budget";
    fn validate(&self) -> Result<()> {
        version(self.version)?;
        if self.device.len() != self.peer.len() || self.device.len() != self.replicas.len() {
            return Err(Error::InvalidLayout);
        }
        Ok(())
    }
}
impl TierBudget {
    pub fn zero(devices: usize) -> Self {
        Self {
            version: WIRE_VERSION,
            device: vec![0; devices],
            peer: vec![0; devices],
            replicas: vec![0; devices],
            ..Self::default()
        }
    }
    fn values(&self) -> Vec<u64> {
        self.device
            .iter()
            .chain(&self.peer)
            .chain(&self.replicas)
            .copied()
            .chain([
                self.pinned,
                self.pageable,
                self.staging,
                self.loaders,
                self.nvme,
                self.inflight,
            ])
            .collect()
    }
    pub fn fits(&self, used: &Self, capacity: &Self) -> Result<bool> {
        self.validate()?;
        used.validate()?;
        capacity.validate()?;
        if self.device.len() != used.device.len() || self.device.len() != capacity.device.len() {
            return Err(Error::InvalidLayout);
        }
        Ok(self
            .values()
            .iter()
            .zip(used.values())
            .zip(capacity.values())
            .all(|((n, u), c)| u <= c && *n <= c - u))
    }
    pub fn checked_add(&self, rhs: &Self) -> Result<Self> {
        self.combine(rhs, true)
    }
    pub fn checked_sub(&self, rhs: &Self) -> Result<Self> {
        self.combine(rhs, false)
    }
    fn combine(&self, rhs: &Self, add: bool) -> Result<Self> {
        self.validate()?;
        rhs.validate()?;
        if self.device.len() != rhs.device.len() {
            return Err(Error::InvalidLayout);
        }
        let op = |a: u64, b: u64| {
            if add {
                a.checked_add(b)
            } else {
                a.checked_sub(b)
            }
            .ok_or(Error::Overflow)
        };
        let vector = |a: &Vec<u64>, b: &Vec<u64>| {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| op(x, y))
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            version: WIRE_VERSION,
            device: vector(&self.device, &rhs.device)?,
            peer: vector(&self.peer, &rhs.peer)?,
            replicas: vector(&self.replicas, &rhs.replicas)?,
            pinned: op(self.pinned, rhs.pinned)?,
            pageable: op(self.pageable, rhs.pageable)?,
            staging: op(self.staging, rhs.staging)?,
            loaders: op(self.loaders, rhs.loaders)?,
            nvme: op(self.nvme, rhs.nvme)?,
            inflight: op(self.inflight, rhs.inflight)?,
        })
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    MandatoryActive,
    AdmittedRestore,
    Demand,
    OptionalPrefetch,
    Backup,
}
/// Nanoseconds on the governor's monotonic clock, never wall-clock/serialized restart time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Deadline(pub u64);
#[derive(Debug, Clone)]
pub struct BudgetRequest {
    pub bytes: TierBudget,
    pub priority: Priority,
    pub deadline: Deadline,
    pub tenant: Digest,
}
impl BudgetRequest {
    pub fn validate(&self) -> Result<()> {
        self.bytes.validate()?;
        if self
            .bytes
            .peer
            .iter()
            .zip(&self.bytes.device)
            .any(|(p, d)| p > d)
            || self
                .bytes
                .replicas
                .iter()
                .zip(&self.bytes.device)
                .any(|(p, d)| p > d)
        {
            return Err(Error::InvalidLayout);
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChargeState {
    Reserved,
    InUse,
    Quarantined,
    Retired,
    Released,
}
#[derive(Debug)]
struct ChargeRecord {
    state: ChargeState,
    pins: u64,
}
/// Non-Clone, non-serializable governor capability. Dropping it does not credit quota.
/// ```compile_fail
/// use memra_tier::contracts::ChargedLease;
/// fn duplicate(x: ChargedLease) { let _ = x.clone(); }
/// ```
#[derive(Debug)]
pub struct ChargedLease {
    issuer: u64,
    id: u64,
    bytes: TierBudget,
    record: Arc<Mutex<ChargeRecord>>,
}
impl ChargedLease {
    pub fn bytes(&self) -> &TierBudget {
        &self.bytes
    }
    pub fn id(&self) -> (u64, u64) {
        (self.issuer, self.id)
    }
    pub fn state(&self) -> Result<ChargeState> {
        Ok(self.record.lock().map_err(|_| Error::Quarantined)?.state)
    }
    /// Retain a physical owner, not another charge. A dropped caller cannot revoke this pin.
    pub fn pin(&self) -> Result<LeasePin> {
        let mut r = self.record.lock().map_err(|_| Error::Quarantined)?;
        if matches!(r.state, ChargeState::Released | ChargeState::Quarantined) {
            return Err(Error::Busy);
        }
        r.pins = r.pins.checked_add(1).ok_or(Error::Overflow)?;
        Ok(LeasePin(self.record.clone()))
    }
}
#[derive(Debug)]
pub struct LeasePin(Arc<Mutex<ChargeRecord>>);
impl Drop for LeasePin {
    fn drop(&mut self) {
        if let Ok(mut r) = self.0.lock() {
            r.pins -= 1;
        }
    }
}
static NEXT_ISSUER: AtomicU64 = AtomicU64::new(1);
fn issuer() -> u64 {
    NEXT_ISSUER
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .expect("issuer space exhausted")
}
/// Capability issuance/accounting helper, NOT an allocator or a second admission policy.
/// The ONE governor owns this registry and calls `issue` only after atomic admission.
#[derive(Debug)]
pub struct LeaseIssuer {
    issuer: u64,
    next: u64,
    records: HashMap<u64, Arc<Mutex<ChargeRecord>>>,
}
impl Default for LeaseIssuer {
    fn default() -> Self {
        Self {
            issuer: issuer(),
            next: 0,
            records: HashMap::new(),
        }
    }
}
impl LeaseIssuer {
    pub fn issue(&mut self, bytes: TierBudget) -> Result<ChargedLease> {
        bytes.validate()?;
        self.next = self.next.checked_add(1).ok_or(Error::Overflow)?;
        let record = Arc::new(Mutex::new(ChargeRecord {
            state: ChargeState::Reserved,
            pins: 0,
        }));
        self.records.insert(self.next, record.clone());
        Ok(ChargedLease {
            issuer: self.issuer,
            id: self.next,
            bytes,
            record,
        })
    }
    fn record(&self, lease: &ChargedLease) -> Result<&Arc<Mutex<ChargeRecord>>> {
        if lease.issuer != self.issuer {
            return Err(Error::ForeignLease);
        }
        self.records.get(&lease.id).ok_or(Error::AlreadyReleased)
    }
    pub fn mark(&self, lease: &ChargedLease, state: ChargeState) -> Result<()> {
        let mut r = self.record(lease)?.lock().map_err(|_| Error::Quarantined)?;
        let allowed = matches!(
            (r.state, state),
            (
                ChargeState::Reserved,
                ChargeState::InUse | ChargeState::Quarantined | ChargeState::Retired
            ) | (
                ChargeState::InUse,
                ChargeState::Quarantined | ChargeState::Retired
            ) | (ChargeState::Quarantined, ChargeState::Retired)
        );
        if !allowed {
            return Err(Error::Busy);
        }
        r.state = state;
        Ok(())
    }
    /// Require owner proof before marking Retired; reject Busy without consuming retry ownership.
    pub fn release(&mut self, lease: &ChargedLease) -> Result<()> {
        {
            let mut r = self.record(lease)?.lock().map_err(|_| Error::Quarantined)?;
            if r.pins != 0 || !matches!(r.state, ChargeState::Reserved | ChargeState::Retired) {
                return Err(Error::Busy);
            }
            r.state = ChargeState::Released;
        }
        self.records.remove(&lease.id);
        Ok(())
    }
}
/// Charge all WP residency/staging/replicas/loaders/slots atomically in ONE instance.
/// Preserve mandatory headroom; order priority then deadline then fair tenant FIFO.
/// Bound queues and dirty backlog; reject optional admission rather than starve a load.
/// Keep unknown work charged; release explicitly only after every physical use retires.
pub trait BudgetGovernor {
    fn reserve(&mut self, request: &BudgetRequest) -> Result<ChargedLease>;
    fn used(&self) -> TierBudget;
    fn mark(&mut self, lease: &ChargedLease, state: ChargeState) -> Result<()>;
    fn release(&mut self, lease: &ChargedLease) -> Result<()>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Durability {
    Ephemeral,
    Persistent,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkRef {
    pub version: u32,
    pub encoded_digest: Digest,
    pub valid_bytes: u64,
    pub storage_bytes: u64,
    pub checksum: Digest,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectManifest {
    pub version: u32,
    pub key: ObjectKey,
    pub valid_bytes: u64,
    pub chunks: Vec<ChunkRef>,
    pub durability: Durability,
}
impl Wire for ObjectManifest {
    const DOMAIN: &'static str = "object-manifest";
    fn validate(&self) -> Result<()> {
        version(self.version)?;
        self.key.validate()?;
        let mut sum = 0u64;
        for c in &self.chunks {
            version(c.version)?;
            if c.valid_bytes == 0 || c.valid_bytes > c.storage_bytes {
                return Err(Error::InvalidLayout);
            }
            sum = sum.checked_add(c.valid_bytes).ok_or(Error::Overflow)?;
        }
        if sum != self.valid_bytes {
            return Err(Error::Incomplete);
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct ObjectLease {
    pub manifest: ObjectManifest,
    pub charge: ChargedLease,
}
/// Keep lookup advisory and immutable; recheck full keys, lengths and hashes on lease/read.
/// Write bounded chunks synchronously; publish the root atomically ONLY after all chunks verify.
/// Reject persistent mode unless file and directory durability are proven. Never overwrite a root.
/// Cancel before root publication; return AlreadyPublished afterward without deleting that root.
/// Retain shared orphan chunks; release charged leases explicitly, never solely through Drop.
pub trait ObjectStore {
    type Transaction;
    fn lookup(&self, key: &ObjectKey) -> Result<Option<ObjectManifest>>;
    fn begin(
        &mut self,
        key: ObjectKey,
        valid_bytes: u64,
        durability: Durability,
    ) -> Result<Self::Transaction>;
    fn put(&mut self, txn: &mut Self::Transaction, payload: &[u8]) -> Result<()>;
    fn commit(&mut self, txn: &mut Self::Transaction) -> Result<ObjectManifest>;
    fn cancel(&mut self, txn: &mut Self::Transaction) -> Result<CancelState>;
    fn lease(&mut self, manifest: &ObjectManifest, request: &BudgetRequest) -> Result<ObjectLease>;
    fn read(&mut self, lease: &ObjectLease, chunk: u32, destination: &mut [u8]) -> Result<u64>;
    fn release(&mut self, lease: &ObjectLease) -> Result<()>;
    fn evict(&mut self, key: &ObjectKey) -> Result<()>;
}

/// Retain actual initialized host backing, alignment, NUMA and governor pin until
/// disk, DMA and consumer retirement. Never expose an uninitialized byte slice.
pub trait PinnedLease: std::fmt::Debug {
    fn storage_bytes(&self) -> u64;
    fn valid_bytes(&self) -> u64;
    fn alignment(&self) -> u32;
    fn numa_node(&self) -> Option<u32>;
    fn bytes(&self) -> Result<&[u8]>;
}
/// Allocation metadata is sealed; construction requires the designated device owner.
/// The retained backend resource and governor pin cannot be forged by I/O workers.
pub struct DeviceLease {
    allocation: Rc<DeviceAllocation>,
}
struct DeviceAllocation {
    issuer: u64,
    id: u64,
    device: u32,
    generation: u64,
    bytes: u64,
    resource: RefCell<Option<(Box<dyn std::any::Any>, LeasePin)>>,
}
impl std::fmt::Debug for DeviceLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceLease")
            .field("device", &self.device())
            .field("generation", &self.generation())
            .field("bytes", &self.bytes())
            .finish()
    }
}
impl DeviceLease {
    pub fn device(&self) -> u32 {
        self.allocation.device
    }
    pub fn generation(&self) -> u64 {
        self.allocation.generation
    }
    pub fn bytes(&self) -> u64 {
        self.allocation.bytes
    }
    pub fn allocation_id(&self) -> u64 {
        self.allocation.id
    }
}
/// Sealed borrowed readiness, never a serializable or clonable permission.
pub struct ReadyView<'a> {
    destination: &'a DeviceLease,
    ticket: TransferTicket,
}
impl ReadyView<'_> {
    pub fn destination(&self) -> &DeviceLease {
        self.destination
    }
    pub fn ticket(&self) -> TransferTicket {
        self.ticket
    }
}
/// Owner-thread capability factory, not a CUDA implementation. Backends must create
/// it on the designated CUDA owner, bind exact destinations at submission, and retain
/// it through unknown completion. Neither this capability nor device handles are Send.
pub struct DeviceOwner {
    issuer: u64,
    device: u32,
    thread: ThreadId,
    next: u64,
    allocations: HashMap<u64, Rc<DeviceAllocation>>,
    destinations: HashMap<TransferTicket, Vec<u64>>,
    _not_send: PhantomData<Rc<()>>,
}
impl DeviceOwner {
    pub fn issuer(&self) -> u64 {
        self.issuer
    }
    pub fn new(device: u32) -> Self {
        Self {
            issuer: issuer(),
            device,
            thread: thread::current().id(),
            next: 0,
            allocations: HashMap::new(),
            destinations: HashMap::new(),
            _not_send: PhantomData,
        }
    }
    fn check(&self, lease: &DeviceLease) -> Result<()> {
        if thread::current().id() != self.thread || lease.allocation.issuer != self.issuer {
            return Err(Error::WrongOwner);
        }
        if !self.allocations.contains_key(&lease.allocation.id) {
            return Err(Error::ForeignLease);
        }
        Ok(())
    }
    pub fn register(
        &mut self,
        generation: u64,
        bytes: u64,
        backing: Box<dyn std::any::Any>,
        charge: &ChargedLease,
    ) -> Result<DeviceLease> {
        if thread::current().id() != self.thread {
            return Err(Error::WrongOwner);
        }
        if bytes == 0
            || charge
                .bytes()
                .device
                .get(self.device as usize)
                .copied()
                .unwrap_or(0)
                < bytes
        {
            return Err(Error::Capacity);
        }
        self.next = self.next.checked_add(1).ok_or(Error::Overflow)?;
        let allocation = Rc::new(DeviceAllocation {
            issuer: self.issuer,
            id: self.next,
            device: self.device,
            generation,
            bytes,
            resource: RefCell::new(Some((backing, charge.pin()?))),
        });
        self.allocations.insert(self.next, allocation.clone());
        Ok(DeviceLease { allocation })
    }
    pub fn retain(&self, lease: &DeviceLease) -> Result<DeviceLease> {
        self.check(lease)?;
        Ok(DeviceLease {
            allocation: lease.allocation.clone(),
        })
    }
    pub fn resolve<T: 'static>(&self, lease: &DeviceLease) -> Result<Ref<'_, T>> {
        self.check(lease)?;
        Ref::filter_map(
            self.allocations[&lease.allocation.id].resource.borrow(),
            |r| r.as_ref()?.0.downcast_ref(),
        )
        .map_err(|_| Error::InvalidLayout)
    }
    pub fn bind_destination(&mut self, ticket: TransferTicket, lease: &DeviceLease) -> Result<()> {
        self.check(lease)?;
        if ticket.epochs.dst_gen != lease.generation() {
            return Err(Error::StaleEpoch);
        }
        self.destinations
            .entry(ticket)
            .or_default()
            .push(lease.allocation.id);
        Ok(())
    }
    pub fn ready_view<'a>(
        &self,
        lease: &'a DeviceLease,
        completion: &Completion,
        expected: &[Vec<SegmentExpectation>],
        current: Epochs,
    ) -> Result<ReadyView<'a>> {
        self.check(lease)?;
        completion.ticket.epochs.require(current)?;
        if !self
            .destinations
            .get(&completion.ticket)
            .is_some_and(|ids| ids.contains(&lease.allocation.id))
        {
            return Err(Error::UnknownTicket);
        }
        completion.require(&completion.ticket, expected, true)?;
        for item in &completion.items {
            for s in &item.segments {
                let f = s.consumer_fence.ok_or(Error::NotReady)?;
                if f.issuer != self.issuer
                    || f.owner != self.device
                    || f.generation != lease.generation()
                {
                    return Err(Error::WrongOwner);
                }
            }
        }
        Ok(ReadyView {
            destination: lease,
            ticket: completion.ticket,
        })
    }
    /// Call only after backend `retired(ticket)` observes disk/DMA/consumer/graph completion.
    /// A borrowed retry handle survives Busy. Allocation itself remains pinned by live leases.
    pub fn retire_binding(&mut self, ticket: &TransferTicket) -> Result<()> {
        if thread::current().id() != self.thread {
            return Err(Error::WrongOwner);
        }
        self.destinations
            .remove(ticket)
            .ok_or(Error::UnknownTicket)?;
        Ok(())
    }
    pub fn release(&mut self, lease: &DeviceLease) -> Result<()> {
        self.check(lease)?;
        if self
            .destinations
            .values()
            .any(|ids| ids.contains(&lease.allocation.id))
            || Rc::strong_count(&lease.allocation) != 2
        {
            return Err(Error::Busy);
        }
        lease.allocation.resource.borrow_mut().take();
        self.allocations.remove(&lease.allocation.id);
        Ok(())
    }
}
impl Drop for DeviceOwner {
    fn drop(&mut self) {
        // Unknown shutdown is not completion. Leak retained backing/pins rather than free
        // potentially DMA-live allocations; a controlled owner drain releases normally.
        for (_, allocation) in self.allocations.drain() {
            std::mem::forget(allocation);
        }
    }
}
#[derive(Debug)]
pub struct CopyOp<H> {
    pub host: H,
    pub device: DeviceLease,
    pub bytes: u64,
    pub epochs: Epochs,
    pub producer_fence: Option<FenceId>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyDirection {
    HostToDevice,
    DeviceToHost,
}
impl<H: PinnedLease> CopyOp<H> {
    pub fn validate(&self, direction: CopyDirection, current: Epochs) -> Result<()> {
        self.epochs.require(current)?;
        if self.bytes == 0
            || self.bytes > self.host.storage_bytes()
            || self.bytes > self.device.bytes()
        {
            return Err(Error::InvalidLayout);
        }
        let generation = match direction {
            CopyDirection::HostToDevice => current.dst_gen,
            CopyDirection::DeviceToHost => current.src_gen,
        };
        if self.device.generation() != generation {
            return Err(Error::StaleEpoch);
        }
        match direction {
            CopyDirection::HostToDevice => {
                self.host.bytes()?;
            }
            CopyDirection::DeviceToHost => {
                let f = self.producer_fence.ok_or(Error::NotReady)?;
                if f.issuer != self.device.allocation.issuer
                    || f.owner != self.device.device()
                    || f.generation != generation
                {
                    return Err(Error::WrongOwner);
                }
            }
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct ReadPlan<H> {
    pub object: ObjectKey,
    pub chunk: u32,
    pub destination: H,
    pub epochs: Epochs,
}
/// No row-index/scatter shape can be supplied to a peer operation.
/// ```compile_fail
/// use memra_tier::contracts::{ContiguousCopy, DeviceLease};
/// fn peer(_: ContiguousCopy) {}
/// fn scatter(rows: Vec<(DeviceLease,u64)>) { peer(rows); }
/// ```
#[derive(Debug)]
pub struct ContiguousCopy {
    source: DeviceLease,
    destination: DeviceLease,
    source_span: ContiguousSpan,
    destination_span: ContiguousSpan,
    pub epochs: Epochs,
    pub producer_fence: FenceId,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContiguousSpan {
    offset: u64,
    bytes: u64,
}
impl ContiguousSpan {
    pub fn new(offset: u64, bytes: u64, capacity: u64) -> Result<Self> {
        if bytes == 0 || offset.checked_add(bytes).is_none_or(|n| n > capacity) {
            return Err(Error::InvalidLayout);
        }
        Ok(Self { offset, bytes })
    }
    pub fn offset(&self) -> u64 {
        self.offset
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}
impl ContiguousCopy {
    pub fn new(
        source: DeviceLease,
        destination: DeviceLease,
        source_span: ContiguousSpan,
        destination_span: ContiguousSpan,
        epochs: Epochs,
        producer_fence: FenceId,
    ) -> Result<Self> {
        let copy = Self {
            source,
            destination,
            source_span,
            destination_span,
            epochs,
            producer_fence,
        };
        copy.validate(epochs)?;
        Ok(copy)
    }
    pub fn source(&self) -> &DeviceLease {
        &self.source
    }
    pub fn destination(&self) -> &DeviceLease {
        &self.destination
    }
    pub fn source_span(&self) -> ContiguousSpan {
        self.source_span
    }
    pub fn destination_span(&self) -> ContiguousSpan {
        self.destination_span
    }
    pub fn validate(&self, current: Epochs) -> Result<()> {
        self.epochs.require(current)?;
        if self.source.generation() != current.src_gen
            || self.destination.generation() != current.dst_gen
        {
            return Err(Error::StaleEpoch);
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
            return Err(Error::InvalidLayout);
        }
        if self.producer_fence.issuer != self.source.allocation.issuer
            || self.producer_fence.owner != self.source.device()
            || self.producer_fence.generation != current.src_gen
        {
            return Err(Error::WrongOwner);
        }
        Ok(())
    }
}
#[derive(Debug)]
pub enum TransferOp<H> {
    H2d(CopyOp<H>),
    D2h(CopyOp<H>),
    P2p(ContiguousCopy),
    NvmeRead(ReadPlan<H>),
}
#[derive(Debug)]
pub enum Destination<H> {
    Host(H),
    Device(DeviceLease),
}
/// Submit on designated device owners only; retain owned source/destination leases.
/// Return Err with every input iff ZERO operations were accepted; uncertain submission
/// is accepted+quarantined. Enumerate every input/segment, including rejects, in order.
/// Reject empty/all-rejected batches. Keep producer completion separate from installed
/// consumer waits and from consumer completion. Never publish on poll/cancel alone.
/// Revoke only before publication; after publication return AlreadyPublished.
/// Quarantine unknown status; keep memory charged until retired covers disk, DMA,
/// consumers AND graph pins. Retain tombstones until explicit acknowledge, never reuse IDs.
#[allow(clippy::result_large_err)] // Return owned inputs without allocating on a rejected submit.
pub trait TransferEngine {
    type Host: PinnedLease;
    fn h2d(
        &mut self,
        op: CopyOp<Self::Host>,
    ) -> std::result::Result<TransferTicket, Rejected<CopyOp<Self::Host>>>;
    fn d2h(
        &mut self,
        op: CopyOp<Self::Host>,
    ) -> std::result::Result<TransferTicket, Rejected<CopyOp<Self::Host>>>;
    fn p2p(
        &mut self,
        op: ContiguousCopy,
    ) -> std::result::Result<TransferTicket, Rejected<ContiguousCopy>>;
    fn nvme_read(
        &mut self,
        op: ReadPlan<Self::Host>,
    ) -> std::result::Result<TransferTicket, Rejected<ReadPlan<Self::Host>>>;
    fn submit_batch(
        &mut self,
        ops: Vec<TransferOp<Self::Host>>,
    ) -> Submission<TransferOp<Self::Host>>;
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion>;
    fn cancel(&mut self, ticket: &TransferTicket) -> Result<CancelState>;
    fn ready_view(
        &mut self,
        ticket: &TransferTicket,
        item: u32,
        current: Epochs,
    ) -> Result<ReadyView<'_>>;
    /// Publish/take exactly once; retain backend pins through subsequent consumer retirement.
    fn take_destination(
        &mut self,
        ticket: &TransferTicket,
        item: u32,
        current: Epochs,
    ) -> Result<Destination<Self::Host>>;
    fn retire(&mut self, ticket: &TransferTicket, consumer_done: Option<FenceId>) -> Result<()>;
    fn retired(&mut self, ticket: &TransferTicket) -> Result<bool>;
    fn acknowledge(&mut self, ticket: &TransferTicket) -> Result<()>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    LocalGpu(u32),
    PeerGpu(u32),
    PinnedHost,
    Nvme,
}
#[derive(Debug, Clone)]
pub struct Lookup {
    pub tier: Tier,
    pub storage_bytes: u64,
}
#[derive(Debug, Clone)]
pub struct TierAdmission {
    pub id: KvBlockId,
    pub program: ProgramIdentity,
    pub expected_layout: RecordLayout,
    pub source: Tier,
    pub target_device: u32,
    pub epochs: Epochs,
    pub committed_high_water: u64,
    pub request: BudgetRequest,
}
#[derive(Debug)]
pub struct TierReservation {
    pub charge: ChargedLease,
}
#[derive(Debug)]
pub struct BlockLease {
    pub bundle: StateBundle,
    pub tier: Tier,
    pub charge: ChargedLease,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Reserved,
    Prefetching,
    HostReady,
    Loading,
    Ready,
    Cancelled,
    Failed,
    Quarantined,
    Retired,
}
#[derive(Debug, Clone, Copy)]
pub enum PrefetchPolicy {
    BestEffort,
    Wait,
    Timeout(Deadline),
}
pub struct TierAdmissionPlan {
    pub reservation: TierReservation,
    pub policy: PrefetchPolicy,
}
/// Extend the existing prefix hierarchy; reserve before moving bytes. Recheck complete
/// identity/layout/current active epochs at admission AND consumption. Keep prefix epoch 0
/// immutable; perform active COW on mutation and fence the committed high-water mark.
/// Refuse missing required planes/working sets; never substitute a numerical program.
/// Keep local/peer direct paths explicit; HostReady is not Ready. Borrow leases through
/// retirement; retain sole active backing and graph addresses. Keep cancellation linearized
/// before ready publication; do not reclaim until every transfer is physically retired.
pub trait TierStore {
    fn lookup(&self, id: &KvBlockId, eligible_peers: &[u32], local: u32) -> Result<Option<Lookup>>;
    fn admit(&mut self, plan: TierAdmission) -> Result<TierReservation>;
    fn prefetch(&mut self, reservation: &TierReservation) -> Result<TransferTicket>;
    fn load(&mut self, reservation: &TierReservation) -> Result<TransferTicket>;
    fn advance(&mut self, reservation: &TierReservation, current: Epochs) -> Result<Phase>;
    fn ready(
        &mut self,
        reservation: &TierReservation,
        program: &ProgramIdentity,
        current: Epochs,
    ) -> Result<&BlockLease>;
    fn cancel(&mut self, reservation: &TierReservation) -> Result<CancelState>;
    fn retire(&mut self, reservation: &TierReservation) -> Result<bool>;
    fn release(&mut self, reservation: &TierReservation) -> Result<()>;
    fn evict(&mut self, id: &KvBlockId) -> Result<()>;
}
/// Materialize complete exact-order native operands into pre-admitted stable local scratch.
/// Recheck program/epochs on the CUDA owner; retain operands until registered consumer fences.
/// Refuse unsupported working sets/crossings; never add chunk-wise attention or requantization.
pub trait KvMaterializer {
    type Operands;
    fn materialize(
        &mut self,
        bundle: &StateBundle,
        program: &ProgramIdentity,
        ready: &ReadyView<'_>,
        current: Epochs,
    ) -> Result<Self::Operands>;
    fn retire(&mut self, operands: &Self::Operands, consumer_done: FenceId) -> Result<()>;
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorId {
    pub version: u32,
    pub artifact: Digest,
    pub name: String,
}
impl Wire for TensorId {
    const DOMAIN: &'static str = "tensor";
    fn validate(&self) -> Result<()> {
        version(self.version)?;
        if self.name.is_empty() {
            return Err(Error::InvalidLayout);
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Projection {
    Gate,
    Up,
    Down,
    Other(String),
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RecordId {
    Expert {
        layer: u32,
        original_id: u32,
        projection: Projection,
    },
    Row(u64),
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BankId {
    pub version: u32,
    pub tensor: TensorId,
    pub record: RecordId,
    pub layout: Digest,
}
impl Wire for BankId {
    const DOMAIN: &'static str = "bank-record";
    fn validate(&self) -> Result<()> {
        version(self.version)?;
        self.tensor.validate()?;
        if matches!(&self.record,RecordId::Expert {projection:Projection::Other(s),..} if s.is_empty())
        {
            return Err(Error::InvalidLayout);
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutClass {
    Uniform,
    PerRecord,
}
#[derive(Debug, Clone)]
pub struct BankBatch {
    pub ids: Vec<BankId>,
    pub epochs: Epochs,
    pub request: BudgetRequest,
}
type RetainedBankBacking = Rc<RefCell<Option<(Box<dyn std::any::Any>, LeasePin)>>>;
#[derive(Clone)]
pub struct BankLease {
    id: BankId,
    layout: RecordLayout,
    class: LayoutClass,
    charge: Arc<ChargedLease>,
    backing: RetainedBankBacking,
}
impl std::fmt::Debug for BankLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BankLease")
            .field("id", &self.id)
            .field("class", &self.class)
            .finish_non_exhaustive()
    }
}
impl BankLease {
    /// Backend publication boundary: validate the actual catalog source class, NOT a
    /// homogeneous subset of mixed metadata. The service must first validate ready fences.
    pub fn from_backend(
        id: BankId,
        layout: RecordLayout,
        class: LayoutClass,
        charge: ChargedLease,
        backing: Box<dyn std::any::Any>,
    ) -> Result<Self> {
        id.validate()?;
        layout.validate()?;
        if id.layout != layout.identity()? {
            return Err(Error::InvalidLayout);
        }
        if layout.segments.iter().any(|s| {
            s.tensor
                .as_ref()
                .is_none_or(|t| t.artifact != id.tensor.artifact)
        }) {
            return Err(Error::InvalidLayout);
        }
        let pin = charge.pin()?;
        Ok(Self {
            id,
            layout,
            class,
            charge: Arc::new(charge),
            backing: Rc::new(RefCell::new(Some((backing, pin)))),
        })
    }
    pub fn id(&self) -> &BankId {
        &self.id
    }
    pub fn layout(&self) -> &RecordLayout {
        &self.layout
    }
    pub fn charge(&self) -> &ChargedLease {
        &self.charge
    }
    pub fn resource<T: 'static>(&self) -> Result<Ref<'_, T>> {
        Ref::filter_map(self.backing.try_borrow().map_err(|_| Error::Busy)?, |r| {
            r.as_ref()?.0.downcast_ref()
        })
        .map_err(|_| Error::AlreadyReleased)
    }
    /// Backend-only lifecycle boundary: call after ALL aliased consumers retire.
    /// Invalidate every clone before crediting quota; a borrowed view refuses Busy.
    pub fn retire_backing(&self) -> Result<()> {
        self.backing
            .try_borrow_mut()
            .map_err(|_| Error::Busy)?
            .take()
            .ok_or(Error::AlreadyReleased)?;
        Ok(())
    }
}
/// Only this checked, non-Clone proof may enter uniform-only kernels.
/// ```compile_fail
/// use memra_tier::contracts::{BankLease,UniformLease};
/// fn uniform(_: &UniformLease) {}
/// fn mixed(x:&BankLease) { uniform(x); }
/// ```
#[derive(Debug)]
pub struct UniformLease(Vec<BankLease>);
impl UniformLease {
    /// Return original ownership on failure so the caller can explicitly release it.
    pub fn try_new(leases: Vec<BankLease>) -> std::result::Result<Self, Rejected<Vec<BankLease>>> {
        let error = if leases.is_empty() {
            Some(Error::EmptyBatch)
        } else if leases
            .iter()
            .any(|l| l.class != LayoutClass::Uniform || !l.layout.same_program(&leases[0].layout))
        {
            Some(Error::MixedLayout)
        } else {
            None
        };
        if let Some(error) = error {
            Err(Rejected { op: leases, error })
        } else {
            Ok(Self(leases))
        }
    }
    pub fn records(&self) -> &[BankLease] {
        &self.0
    }
    pub fn into_records(self) -> Vec<BankLease> {
        self.0
    }
}
/// Validate original artifact/tensor/expert/projection IDs and masks BEFORE staging.
/// Return a ticket for partial movement but publish ALL logical records or none, in original
/// order including duplicates. Require payload and scale planes and transfer-owned ready
/// fences. Retain native-GEMM consumers through retirement. Never route PerRecord sources
/// through UniformLease; charge residency and scratch through the common governor.
pub trait BankedResidency {
    fn layout(&self, id: &BankId) -> Result<&RecordLayout>;
    fn resident(&mut self, id: &BankId) -> Result<Option<BankLease>>;
    fn stage(&mut self, batch: BankBatch) -> Result<TransferTicket>;
    fn publish(&mut self, ticket: &TransferTicket, current: Epochs) -> Result<Vec<BankLease>>;
    fn cancel(&mut self, ticket: &TransferTicket) -> Result<CancelState>;
    fn retire(&mut self, ticket: &TransferTicket) -> Result<bool>;
    fn release(&mut self, lease: &BankLease) -> Result<()>;
}
#[derive(Debug, Clone)]
pub struct RowBatch {
    pub ids: Vec<BankId>,
    pub epochs: Epochs,
    pub request: BudgetRequest,
}
#[derive(Debug)]
pub struct RowLease {
    pub records: Vec<BankLease>,
}
/// Preserve requested row order/duplicates while deduplicating physical reads. Fetch all
/// row payload/scale segments atomically, with bounded staging smaller than a batch.
/// Refuse masked/wrong-namespace rows and any incomplete logical output. Distinguish
/// host-ready from consumer-ready; retain through projection fences and rollback checks.
pub trait RowService {
    fn gather(&mut self, batch: RowBatch) -> Result<TransferTicket>;
    fn publish(&mut self, ticket: &TransferTicket, current: Epochs) -> Result<RowLease>;
    fn cancel(&mut self, ticket: &TransferTicket) -> Result<CancelState>;
    fn retire(&mut self, ticket: &TransferTicket) -> Result<bool>;
    fn release(&mut self, lease: &RowLease) -> Result<()>;
}
pub enum ExpertDomain {}
pub enum RowDomain {}
mod sealed {
    pub trait Domain {}
}
impl sealed::Domain for ExpertDomain {}
impl sealed::Domain for RowDomain {}
/// Separate row and expert policy types; reject other record domains.
pub trait BankDomain: sealed::Domain {
    fn accepts(id: &RecordId) -> bool;
}
impl BankDomain for ExpertDomain {
    fn accepts(id: &RecordId) -> bool {
        matches!(id, RecordId::Expert { .. })
    }
}
impl BankDomain for RowDomain {
    fn accepts(id: &RecordId) -> bool {
        matches!(id, RecordId::Row(_))
    }
}
/// Bound metadata; count only demand, never predicted prefetch, as heat.
pub trait Hotness<D: BankDomain> {
    fn demand(&mut self, id: &BankId);
    fn score(&self, id: &BankId) -> u64;
}
/// Cap hints and revalidate original IDs; never let failed prediction omit demand work.
pub trait PrefetchHook<D: BankDomain> {
    type Context;
    fn predict(&self, context: &Self::Context, limit: usize) -> Vec<BankId>;
}

#[derive(Debug, Clone)]
pub struct PeerPlan {
    pub owner_device: u32,
    pub consumer_device: u32,
    pub bytes: u64,
    pub alignment: u32,
    pub epochs: Epochs,
    pub request: BudgetRequest,
}
#[derive(Debug)]
pub struct PeerLease {
    pub plan: PeerPlan,
    pub charge: ChargedLease,
    pub device: DeviceLease,
}
/// Validate directed peer AND pool grants tied to live context/topology/binary.
/// Reserve actual physical capacity through the SAME governor, not an independent allocator.
/// Reject unsupported P2P; never silently bounce through host. Borrow release handles so
/// Busy/foreign/double-release errors preserve accounting and retry ownership.
pub trait PeerCapacity {
    fn reserve(&mut self, plan: PeerPlan) -> Result<PeerLease>;
    fn release(&mut self, lease: &PeerLease) -> Result<()>;
}
/// Submit only contiguous batched descriptors on the designated CUDA owner. Return every
/// indexed acceptance including rejects; use TransferEngine's zero-accept Err semantics.
/// Retain both allocation generations separately from state epoch. Publish only local
/// destinations after complete byte/fence checks; never authorize remote attention scatter.
/// Revoke before publication, return AlreadyPublished afterward; quarantine unknowns until
/// disk/DMA/consumer/graph retirement. HostBounce is a distinct explicit route, not P2P.
pub trait PeerBackend {
    fn submit(&mut self, copies: Vec<ContiguousCopy>) -> Submission<ContiguousCopy>;
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion>;
    fn cancel(&mut self, ticket: &TransferTicket) -> Result<CancelState>;
    fn materialize_local(
        &mut self,
        ticket: &TransferTicket,
        item: u32,
        current: Epochs,
        consumer: u32,
    ) -> Result<ReadyView<'_>>;
    fn retire_consumer(&mut self, ticket: &TransferTicket, consumer_done: FenceId) -> Result<()>;
    fn retired(&mut self, ticket: &TransferTicket) -> Result<bool>;
    fn acknowledge(&mut self, ticket: &TransferTicket) -> Result<()>;
}
/// Capture/restore B's complete committed envelope, never an activation-only shadow state.
/// Refuse wrong programs, missing owners/planes, stale high-water marks and epochs.
pub trait StateBundleAdapter {
    fn capture_committed(&mut self, state_epoch: u64) -> Result<StateBundle>;
    fn restore_committed(
        &mut self,
        bundle: &StateBundle,
        identity: &ProgramIdentity,
        epochs: Epochs,
    ) -> Result<TransferTicket>;
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Endpoint {
    Device(u32),
    PinnedHost,
    Nvme,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouteKind {
    Local,
    PcieP2p,
    HostBounce,
    HostDevice,
    Storage,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceBudget {
    pub version: u32,
    pub device: u32,
    pub weight_bytes: u64,
    pub global_kv_bytes: u64,
    pub fixed_state_bytes: u64,
    pub staging_bytes: u64,
    pub loader_bytes: u64,
    pub scratch_bytes: u64,
    pub reserve_bytes: u64,
    pub replica_bytes: u64,
    pub total_bytes: u64,
    pub remaining_bytes: i128,
    pub estimates: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteBudget {
    pub version: u32,
    pub from: Endpoint,
    pub to: Endpoint,
    pub kind: RouteKind,
    pub demand_bytes_per_second: u64,
    pub physical_bytes_per_second: u64,
    pub measured_bytes_per_second: Option<u64>,
    pub within_measured_envelope: Option<bool>,
    pub estimates: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementReport {
    pub version: u32,
    pub per_device: Vec<DeviceBudget>,
    pub routes: Vec<RouteBudget>,
    pub host_bytes: u64,
    pub estimates: Vec<String>,
}
impl Wire for PlacementReport {
    const DOMAIN: &'static str = "placement";
    fn validate(&self) -> Result<()> {
        version(self.version)?;
        let mut devices = BTreeSet::new();
        for d in &self.per_device {
            version(d.version)?;
            if !devices.insert(d.device) {
                return Err(Error::InvalidLayout);
            }
            // replica_bytes is a breakdown of global_kv_bytes, not an additional allocation.
            if d.replica_bytes > d.global_kv_bytes {
                return Err(Error::InvalidLayout);
            }
            let total = [
                d.weight_bytes,
                d.global_kv_bytes,
                d.fixed_state_bytes,
                d.staging_bytes,
                d.loader_bytes,
                d.scratch_bytes,
                d.reserve_bytes,
            ]
            .into_iter()
            .try_fold(0u64, |a, b| a.checked_add(b).ok_or(Error::Overflow))?;
            if total != d.total_bytes {
                return Err(Error::InvalidLayout);
            }
        }
        for r in &self.routes {
            version(r.version)?;
            for ep in [r.from, r.to] {
                if let Endpoint::Device(d) = ep
                    && !devices.contains(&d)
                {
                    return Err(Error::InvalidLayout);
                }
            }
            let valid = match (r.from, r.to, r.kind) {
                (Endpoint::Device(a), Endpoint::Device(b), RouteKind::Local) => a == b,
                (
                    Endpoint::Device(a),
                    Endpoint::Device(b),
                    RouteKind::PcieP2p | RouteKind::HostBounce,
                ) => a != b,
                (Endpoint::PinnedHost, Endpoint::Device(_), RouteKind::HostDevice)
                | (Endpoint::Device(_), Endpoint::PinnedHost, RouteKind::HostDevice)
                | (Endpoint::PinnedHost, Endpoint::Nvme, RouteKind::Storage)
                | (Endpoint::Nvme, Endpoint::PinnedHost, RouteKind::Storage) => true,
                _ => false,
            };
            if !valid || r.measured_bytes_per_second == Some(0) {
                return Err(Error::InvalidLayout);
            }
            let legs = if r.kind == RouteKind::HostBounce {
                2
            } else {
                1
            };
            if r.physical_bytes_per_second
                != r.demand_bytes_per_second
                    .checked_mul(legs)
                    .ok_or(Error::Overflow)?
            {
                return Err(Error::InvalidLayout);
            }
            if r.within_measured_envelope
                != r.measured_bytes_per_second
                    .map(|rate| u128::from(r.demand_bytes_per_second) * 10 <= u128::from(rate) * 7)
            {
                return Err(Error::InvalidLayout);
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageSample {
    pub version: u32,
    pub fixture: String,
    pub backend_requested: String,
    pub backend_actual: String,
    pub status: String,
    pub valid_bytes: u64,
    pub padded_bytes: u64,
    pub io_bytes: u64,
    pub physical_bytes: Option<u64>,
    pub queue_ns: Option<u64>,
    pub io_ns: Option<u64>,
    pub h2d_ns: Option<u64>,
    pub d2h_ns: Option<u64>,
    pub p2p_ns: Option<u64>,
    pub total_ns: u64,
    pub inflight: u64,
    pub pinned_bytes: u64,
    pub pageable_bytes: Option<u64>,
    pub fallbacks: u64,
    pub payload_checksum: Digest,
}
impl Wire for StorageSample {
    const DOMAIN: &'static str = "storage-sample";
    fn validate(&self) -> Result<()> {
        version(self.version)?;
        if self.fixture.is_empty()
            || self.backend_requested.is_empty()
            || self.backend_actual.is_empty()
            || self.valid_bytes > self.padded_bytes
        {
            return Err(Error::InvalidLayout);
        }
        Ok(())
    }
}
