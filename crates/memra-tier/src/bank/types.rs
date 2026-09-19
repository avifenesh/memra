use std::collections::BTreeMap;
use std::sync::Arc;

pub type Digest = [u8; 32];
pub type Result<T> = std::result::Result<T, BankError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BankError {
    UnknownId,
    MaskedId,
    InvalidLayout,
    MixedLayout,
    EmptyBatch,
    TooLarge,
    Overflow,
    Backpressure,
    Pending,
    UnknownTicket,
    ReadFailed,
    WrongLength,
    WrongOwner,
}

/// Semantic identity, never a KV prefix hash. Tensor names are complete checkpoint IDs.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TensorId {
    pub artifact: Digest,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Projection {
    Gate,
    Up,
    Down,
    Other(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecordId {
    Expert {
        layer: u32,
        original_id: u32,
        projection: Projection,
    },
    Row(u64),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BankId {
    pub tensor: TensorId,
    pub record: RecordId,
    pub layout: Digest,
}

/// Separate planes (including macro scales) are explicit; no tensor substitution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub tensor: TensorId,
    pub offset: u64,
    pub len: u64,
    pub role: String,
}

/// `encoding` and `row_bytes` are authoritative for THIS record, never the projection.
/// Data and scale segments are concatenated in this declared order by the CPU prototype.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordLayout {
    pub encoding: String,
    pub row_bytes: u64,
    pub segments: Vec<Segment>,
}
impl RecordLayout {
    pub fn bytes(&self) -> Result<u64> {
        if self.row_bytes == 0 || self.encoding.is_empty() || self.segments.is_empty() {
            return Err(BankError::InvalidLayout);
        }
        self.segments.iter().try_fold(0u64, |sum, seg| {
            if seg.len == 0 || seg.role.is_empty() {
                return Err(BankError::InvalidLayout);
            }
            seg.offset.checked_add(seg.len).ok_or(BankError::Overflow)?;
            sum.checked_add(seg.len).ok_or(BankError::Overflow)
        })
    }
    pub(crate) fn same_program(&self, other: &Self) -> bool {
        self.encoding == other.encoding
            && self.row_bytes == other.row_bytes
            && self.segments.len() == other.segments.len()
            && self
                .segments
                .iter()
                .zip(&other.segments)
                .all(|(a, b)| a.role == b.role && a.len == b.len)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutClass {
    /// HostExps.layouts == None; caller attests original source is uniform.
    Uniform,
    /// HostExps.layouts == Some, even if a subset happens to have equal encodings.
    PerRecord,
}

pub struct Catalog {
    class: LayoutClass,
    entries: BTreeMap<BankId, Option<RecordLayout>>,
}
impl Catalog {
    pub fn new(class: LayoutClass, entries: Vec<(BankId, Option<RecordLayout>)>) -> Result<Self> {
        let mut map = BTreeMap::new();
        let mut uniform: Option<RecordLayout> = None;
        for (id, layout) in entries {
            if id.tensor.name.is_empty() {
                return Err(BankError::InvalidLayout);
            }
            if let Some(layout) = &layout {
                layout.bytes()?;
                if layout
                    .segments
                    .iter()
                    .any(|s| s.tensor.artifact != id.tensor.artifact || s.tensor.name.is_empty())
                {
                    return Err(BankError::InvalidLayout);
                }
                if class == LayoutClass::Uniform {
                    if let Some(first) = &uniform {
                        if !first.same_program(layout) {
                            return Err(BankError::MixedLayout);
                        }
                    } else {
                        uniform = Some(layout.clone());
                    }
                }
            }
            if map.insert(id, layout).is_some() {
                return Err(BankError::InvalidLayout);
            }
        }
        Ok(Self {
            class,
            entries: map,
        })
    }
    pub(crate) fn class(&self) -> LayoutClass {
        self.class
    }
    pub fn layout(&self, id: &BankId) -> Result<&RecordLayout> {
        self.entries
            .get(id)
            .ok_or(BankError::UnknownId)?
            .as_ref()
            .ok_or(BankError::MaskedId)
    }
    pub fn batch(&self, ids: Vec<BankId>) -> Result<BankBatch> {
        if ids.is_empty() {
            return Err(BankError::EmptyBatch);
        }
        for id in &ids {
            self.layout(id)?;
        }
        Ok(BankBatch { ids, demand: true })
    }
    /// Uniform-only compute adapters must accept this type, never a `BankBatch`.
    /// PerRecord sources cannot obtain the proof even for a homogeneous subset.
    pub fn uniform(&self, batch: BankBatch) -> Result<UniformBatch> {
        if self.class != LayoutClass::Uniform {
            return Err(BankError::MixedLayout);
        }
        for id in &batch.ids {
            self.layout(id)?;
        }
        Ok(UniformBatch(batch))
    }
}

#[derive(Clone, Debug)]
pub struct BankBatch {
    pub(crate) ids: Vec<BankId>,
    pub(crate) demand: bool,
}
impl BankBatch {
    pub fn ids(&self) -> &[BankId] {
        &self.ids
    }
}

/// Private constructor prevents casting mixed batches into a uniform-only adapter.
/// ```compile_fail
/// use memra_bank_prototype::{BankBatch, UniformBatch};
/// fn uniform_kernel(_: &UniformBatch) {}
/// fn mixed_dispatch(batch: &BankBatch) { uniform_kernel(batch); }
/// ```
pub struct UniformBatch(BankBatch);
impl UniformBatch {
    pub fn batch(&self) -> &BankBatch {
        &self.0
    }
}

/// B owns the one global governor. This is an injected interface, not a second allocator.
/// Output, hot-cache, staging and queue permits all use that SAME governor instance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Charge {
    pub host_bytes: u64,
    pub staging_bytes: u64,
    pub inflight: u32,
}
pub trait BudgetPermit: Send + Sync {}
pub trait BudgetGovernor: Send + Sync {
    fn reserve(&self, charge: Charge) -> Result<Box<dyn BudgetPermit>>;
}
pub type SharedBudget = Arc<dyn BudgetGovernor>;

/// A's ObjectStore/PinnedLease adapter will implement this exact, bounded host read seam.
/// Returning success means ALL bytes initialized; short/error reads never authorize publish.
pub trait ExactReader {
    fn read_exact(&mut self, tensor: &TensorId, offset: u64, dst: &mut [u8]) -> Result<()>;
}

/// Distinct generic domains prevent row heat being passed to an expert policy.
pub enum ExpertDomain {}
pub enum RowDomain {}
mod sealed {
    pub trait Domain {}
}
impl sealed::Domain for ExpertDomain {}
impl sealed::Domain for RowDomain {}
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
/// Validates bounded optional hints without adding demand heat. Empty means no prefetch.
pub fn prefetch_batch<D: BankDomain, P: PrefetchHook<D>>(
    catalog: &Catalog,
    predictor: &P,
    context: &P::Context,
    limit: usize,
) -> Result<Option<BankBatch>> {
    let ids = predictor.predict(context, limit);
    if ids.len() > limit {
        return Err(BankError::TooLarge);
    }
    if ids.is_empty() {
        return Ok(None);
    }
    if ids.iter().any(|id| !D::accepts(&id.record)) {
        return Err(BankError::InvalidLayout);
    }
    let mut batch = catalog.batch(ids)?;
    batch.demand = false;
    Ok(Some(batch))
}
pub trait Hotness<D> {
    // Implementations must bound their metadata to the catalog or an admitted quota.
    fn demand(&mut self, id: &BankId);
    /// Lower score is evicted first; predictions must NOT count as demand hits.
    fn score(&self, id: &BankId) -> u64;
}
pub trait PrefetchHook<D> {
    type Context;
    /// Caller enforces the cap and revalidates each original id through the catalog.
    /// Router or accepted n-gram history is adapter-owned, never interpreted by storage.
    fn predict(&self, context: &Self::Context, limit: usize) -> Vec<BankId>;
}
