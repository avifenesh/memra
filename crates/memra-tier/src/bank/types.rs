use crate::contracts::*;
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

/// The SAME governor instance is injected into banks, rows, KV and transfers.
pub type SharedBudget = Rc<RefCell<dyn BudgetGovernor>>;

/// Immutable source metadata. Checksums come from the artifact manifest, never a
/// successful read echoed as its own expected checksum. All scale planes are atomic.
#[derive(Clone, Debug)]
pub struct CatalogRecord {
    pub layout: RecordLayout,
    pub checksums: Vec<Digest>,
}
/// A ticket's metadata allowance for one record: the canonical encodings of its id and layout
/// plus a conservative 1024-byte per-node allowance. A pure function of the immutable catalog
/// entry, so `Catalog::new` computes it once per record (day 61, I11 change 1,
/// `research/spill-c-20260919/DAY61.md`); every ticket reads that value.
pub(crate) fn ticket_metadata(id: &BankId, layout: &RecordLayout) -> Result<u64> {
    Ok((id.encode()?.len() + layout.encode()?.len() + 1024) as u64)
}
/// One retained catalog record with its ticket metadata allowance.
pub(crate) struct CatalogEntry {
    pub(crate) record: CatalogRecord,
    /// `ticket_metadata(id, &record.layout)`, computed at `Catalog::new`.
    pub(crate) metadata: u64,
}
impl CatalogEntry {
    /// The record's resident charge: its storage bytes plus its metadata allowance.
    pub(crate) fn resident_charge_bytes(&self) -> Result<u64> {
        self.record
            .layout
            .storage_bytes()?
            .checked_add(self.metadata)
            .ok_or(Error::Overflow)
    }
}
pub struct Catalog {
    pub(crate) class: LayoutClass,
    pub(crate) entries: BTreeMap<BankId, Option<CatalogEntry>>,
}
impl Catalog {
    pub fn new(class: LayoutClass, entries: Vec<(BankId, Option<CatalogRecord>)>) -> Result<Self> {
        let mut map = BTreeMap::new();
        let mut originals = BTreeSet::new();
        let mut uniform: Option<RecordLayout> = None;
        for (id, record) in entries {
            id.validate()?;
            // One immutable source assignment per original router/table slot.
            // A second layout digest must not resurrect a masked original ID.
            if !originals.insert((id.tensor.clone(), id.record.clone())) {
                return Err(Error::Conflict);
            }
            if let Some(record) = &record {
                let layout = &record.layout;
                layout.validate()?;
                if id.layout != layout.identity()?
                    || record.checksums.len() != layout.segments.len()
                    || layout.segments.iter().any(|s| {
                        s.tensor
                            .as_ref()
                            .is_none_or(|t| t.artifact != id.tensor.artifact)
                    })
                {
                    return Err(Error::InvalidLayout);
                }
                if class == LayoutClass::Uniform {
                    if uniform
                        .as_ref()
                        .is_some_and(|first| !first.same_program(layout))
                    {
                        return Err(Error::MixedLayout);
                    }
                    uniform = Some(layout.clone());
                }
            }
            let entry = match record {
                Some(record) => Some(CatalogEntry {
                    metadata: ticket_metadata(&id, &record.layout)?,
                    record,
                }),
                None => None,
            };
            if map.insert(id, entry).is_some() {
                return Err(Error::Conflict);
            }
        }
        Ok(Self {
            class,
            entries: map,
        })
    }
    pub fn record(&self, id: &BankId) -> Result<&CatalogRecord> {
        Ok(&self.entry(id)?.record)
    }
    /// The ticket metadata allowance `Catalog::new` computed for `id` (day 61); the same
    /// refusals as `record`.
    pub fn metadata_allowance(&self, id: &BankId) -> Result<u64> {
        Ok(self.entry(id)?.metadata)
    }
    /// The record with its ticket metadata allowance; the same refusals as `record`.
    pub(crate) fn entry(&self, id: &BankId) -> Result<&CatalogEntry> {
        id.validate()?;
        self.entries
            .get(id)
            .ok_or(Error::NotFound)?
            .as_ref()
            .ok_or(Error::MaskedId)
    }
}

pub fn prefetch_batch<D: BankDomain, P: PrefetchHook<D>>(
    catalog: &Catalog,
    predictor: &P,
    context: &P::Context,
    limit: usize,
    epochs: Epochs,
    mut request: BudgetRequest,
) -> Result<Option<BankBatch>> {
    let ids = predictor.predict(context, limit);
    if ids.len() > limit {
        return Err(Error::Capacity);
    }
    if ids.is_empty() {
        return Ok(None);
    }
    for id in &ids {
        if !D::accepts(&id.record) {
            return Err(Error::InvalidLayout);
        }
        catalog.record(id)?;
    }
    request.priority = Priority::OptionalPrefetch;
    Ok(Some(BankBatch {
        ids,
        epochs,
        request,
    }))
}

/// Resource caps, not environment doors or hardware/performance defaults.
#[derive(Clone, Copy, Debug)]
pub struct BankLimits {
    pub cache_bytes: u64,
    pub batch_bytes: u64,
    pub items: usize,
    /// Includes unacknowledged tombstones, so metadata cannot grow unbounded.
    pub tickets: usize,
}

/// Read granularity policy for the portable host baseline only. 512 B minimizes
/// sparse 264 B row amplification among the day-1 candidates (1.94x vs 15.52x
/// at 4KiB). Native backends MUST raise this to their actual alignment; this is
/// not a claim of 512 B SSD physical reads or a promoted O_DIRECT default.
#[derive(Clone, Copy, Debug)]
pub struct CoalescingPolicy {
    pub granularity: u64,
    pub slot_bytes: u64,
}
impl Default for CoalescingPolicy {
    fn default() -> Self {
        Self {
            granularity: 512,
            slot_bytes: 4096,
        }
    }
}
impl CoalescingPolicy {
    pub fn validate(self) -> Result<()> {
        if !self.granularity.is_power_of_two()
            || self.slot_bytes == 0
            || !self.slot_bytes.is_multiple_of(self.granularity)
            || self.slot_bytes > usize::MAX as u64
        {
            return Err(Error::InvalidLayout);
        }
        Ok(())
    }
}

/// Exact immutable positioned-read seam used only by the explicit progress pump.
/// ObjectReader implements it with A's ticketed CPU ObjectStore/TransferEngine;
/// no CUDA submission, pinning or GPU permission is exposed here.
pub trait ExactReader {
    fn begin_request(&mut self, _request: &BudgetRequest, _epochs: Epochs) {}
    fn storage_bytes(&self, tensor: &TensorId) -> Result<u64>;
    fn read_exact(&mut self, tensor: &TensorId, offset: u64, dst: &mut [u8]) -> Result<()>;
}
