use super::*;
use crate::contracts::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadExtent {
    pub tensor: TensorId,
    pub offset: u64,
    pub len: u64,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RowReadPlan {
    pub extents: Vec<ReadExtent>,
    pub logical_bytes: u64,
    pub unique_useful_bytes: u64,
    /// Submitted aligned bytes, NOT measured SSD physical traffic.
    pub io_bytes: u64,
    pub straddling_segments: u64,
}
impl RowReadPlan {
    pub fn amplification(&self) -> Option<f64> {
        (self.unique_useful_bytes != 0)
            .then(|| self.io_bytes as f64 / self.unique_useful_bytes as f64)
    }
}

/// Coalesce each original tensor independently, including discontiguous scale
/// tensors; duplicates alias outputs. Bounds include explicitly readable padding.
pub fn plan_reads(
    records: &[(BankId, CatalogRecord)],
    logical_bytes: u64,
    reader: &dyn ExactReader,
    policy: CoalescingPolicy,
) -> Result<RowReadPlan> {
    policy.validate()?;
    let mut ranges: BTreeMap<TensorId, Vec<(u64, u64)>> = BTreeMap::new();
    let mut plan = RowReadPlan {
        logical_bytes,
        ..RowReadPlan::default()
    };
    for (_, record) in records {
        for segment in &record.layout.segments {
            let tensor = segment.tensor.as_ref().ok_or(Error::InvalidLayout)?;
            let end = segment
                .offset
                .checked_add(segment.storage_bytes)
                .ok_or(Error::Overflow)?;
            let lo = segment.offset / policy.granularity * policy.granularity;
            let hi = end
                .checked_add(policy.granularity - 1)
                .ok_or(Error::Overflow)?
                / policy.granularity
                * policy.granularity;
            if hi > reader.storage_bytes(tensor)? {
                return Err(Error::InvalidLayout);
            }
            plan.unique_useful_bytes = plan
                .unique_useful_bytes
                .checked_add(segment.valid_bytes)
                .ok_or(Error::Overflow)?;
            plan.straddling_segments += u64::from(hi - lo > policy.granularity);
            ranges.entry(tensor.clone()).or_default().push((lo, hi));
        }
    }
    for (tensor, mut spans) in ranges {
        spans.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (lo, hi) in spans {
            if let Some(last) = merged.last_mut()
                && lo <= last.1
            {
                last.1 = last.1.max(hi);
            } else {
                merged.push((lo, hi));
            }
        }
        for (lo, hi) in merged {
            plan.io_bytes = plan.io_bytes.checked_add(hi - lo).ok_or(Error::Overflow)?;
            plan.extents.push(ReadExtent {
                tensor: tensor.clone(),
                offset: lo,
                len: hi - lo,
            });
        }
    }
    Ok(plan)
}

/// One chunk per explicit pump; output stays private until every chunk verifies.
pub(crate) struct ReadWork {
    pub outputs: Vec<Vec<u8>>,
    extent: usize,
    offset: u64,
}
impl ReadWork {
    pub fn new(records: &[(BankId, CatalogRecord)]) -> Result<Self> {
        let outputs = records
            .iter()
            .map(|(_, r)| {
                let len =
                    usize::try_from(r.layout.storage_bytes()?).map_err(|_| Error::Capacity)?;
                let mut bytes = Vec::new();
                bytes.try_reserve_exact(len).map_err(|_| Error::Capacity)?;
                bytes.resize(len, 0);
                Ok(bytes)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            outputs,
            extent: 0,
            offset: 0,
        })
    }
    pub fn step(
        &mut self,
        records: &[(BankId, CatalogRecord)],
        plan: &RowReadPlan,
        reader: &mut dyn ExactReader,
        policy: CoalescingPolicy,
    ) -> Result<bool> {
        let Some(extent) = plan.extents.get(self.extent) else {
            return Ok(true);
        };
        let position = extent.offset + self.offset;
        let n = (extent.len - self.offset).min(policy.slot_bytes);
        let mut slot = Vec::new();
        slot.try_reserve_exact(n as usize)
            .map_err(|_| Error::Capacity)?;
        slot.resize(n as usize, 0);
        reader.read_exact(&extent.tensor, position, &mut slot)?;
        for ((_, record), output) in records.iter().zip(&mut self.outputs) {
            let mut base = 0usize;
            for s in &record.layout.segments {
                if s.tensor.as_ref() == Some(&extent.tensor) {
                    let lo = position.max(s.offset);
                    let hi = (position + n).min(s.offset + s.storage_bytes);
                    if lo < hi {
                        let from = (lo - position) as usize;
                        let to = base + (lo - s.offset) as usize;
                        output[to..to + (hi - lo) as usize]
                            .copy_from_slice(&slot[from..from + (hi - lo) as usize]);
                    }
                }
                base += s.storage_bytes as usize;
            }
        }
        self.offset += n;
        if self.offset == extent.len {
            self.extent += 1;
            self.offset = 0;
        }
        Ok(self.extent == plan.extents.len())
    }
}

/// Row policy and expert heat are deliberately distinct types. This host backend
/// publishes HOST-ready resources only, never permission to submit a projection.
pub struct BoundedRowService<H: Hotness<RowDomain>, R: ExactReader>(
    pub BankService<RowDomain, H, R>,
);
impl<H: Hotness<RowDomain>, R: ExactReader> RowService for BoundedRowService<H, R> {
    fn gather(&mut self, batch: RowBatch) -> Result<TransferTicket> {
        self.0.stage(BankBatch {
            ids: batch.ids,
            epochs: batch.epochs,
            request: batch.request,
        })
    }
    fn publish(&mut self, ticket: &TransferTicket, current: Epochs) -> Result<RowLease> {
        Ok(RowLease {
            records: self.0.publish(ticket, current)?,
        })
    }
    fn cancel(&mut self, ticket: &TransferTicket) -> Result<CancelState> {
        self.0.cancel(ticket)
    }
    fn retire(&mut self, ticket: &TransferTicket) -> Result<bool> {
        self.0.retire(ticket)
    }
    fn release(&mut self, lease: &RowLease) -> Result<()> {
        // Preflight outstanding tickets. A later borrowed-view Busy may follow
        // a partial release; retry skips already-released members without losing
        // any remaining handles. An entirely released row lease refuses replay.
        let mut unique = BTreeMap::new();
        for record in &lease.records {
            if record.charge().state()? != ChargeState::Released {
                unique.insert(record.charge().id(), record);
            }
        }
        if unique.is_empty() {
            return Err(Error::AlreadyReleased);
        }
        for record in unique.values() {
            self.0.can_release(record)?;
        }
        for record in unique.values() {
            self.0.release(record)?;
        }
        Ok(())
    }
}
