use super::*;
use std::collections::BTreeSet;

/// A bounded row request, in logical consumer order (including duplicates).
#[derive(Clone, Debug)]
pub struct RowBatch {
    pub rows: Vec<u64>,
}

/// Immutable table descriptor. Row stride includes on-disk padding; width includes
/// all payload/scales belonging to the row. Separate scale planes need separate tables
/// in this prototype; final shared RecordLayout must describe them as one atomic row.
#[derive(Clone, Debug)]
pub struct RowTable {
    pub tensor: TensorId,
    pub base: u64,
    pub rows: u64,
    pub width: u64,
    pub stride: u64,
    /// Physical readable length including explicitly declared tail padding.
    pub storage_bytes: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadExtent {
    pub offset: u64,
    pub len: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowReadPlan {
    pub extents: Vec<ReadExtent>,
    pub logical_bytes: u64,
    pub unique_useful_bytes: u64,
    pub physical_bytes: u64,
    pub straddling_rows: u64,
}
impl RowReadPlan {
    /// Unique useful / aligned physical bytes; None for an empty request.
    pub fn efficiency(&self) -> Option<f64> {
        (self.physical_bytes != 0)
            .then(|| self.unique_useful_bytes as f64 / self.physical_bytes as f64)
    }
    /// Physical / unique useful bytes. Duplicates do not artificially improve this ratio.
    pub fn amplification(&self) -> Option<f64> {
        (self.unique_useful_bytes != 0)
            .then(|| self.physical_bytes as f64 / self.unique_useful_bytes as f64)
    }
}
impl RowTable {
    fn range(&self, row: u64) -> Result<(u64, u64)> {
        if self.width == 0 || self.stride < self.width {
            return Err(BankError::InvalidLayout);
        }
        if row >= self.rows {
            return Err(BankError::UnknownId);
        }
        let begin = row
            .checked_mul(self.stride)
            .and_then(|v| self.base.checked_add(v))
            .ok_or(BankError::Overflow)?;
        let end = begin.checked_add(self.width).ok_or(BankError::Overflow)?;
        if end > self.storage_bytes {
            return Err(BankError::InvalidLayout);
        }
        Ok((begin, end))
    }
    /// Pure planning: aligned ranges merged across adjacent/overlapping rows, never gaps.
    /// This models backend request bytes, NOT actual SSD device bytes through page cache.
    pub fn plan(&self, batch: &RowBatch, granularity: u64) -> Result<RowReadPlan> {
        if granularity == 0 || !granularity.is_power_of_two() {
            return Err(BankError::InvalidLayout);
        }
        if self.width == 0 || self.stride < self.width {
            return Err(BankError::InvalidLayout);
        }
        let logical_bytes = self
            .width
            .checked_mul(batch.rows.len() as u64)
            .ok_or(BankError::Overflow)?;
        let unique: BTreeSet<_> = batch.rows.iter().copied().collect();
        let unique_useful_bytes = self
            .width
            .checked_mul(unique.len() as u64)
            .ok_or(BankError::Overflow)?;
        let mut extents: Vec<ReadExtent> = Vec::new();
        let mut straddling_rows = 0;
        for row in unique {
            let (begin, end) = self.range(row)?;
            let aligned_begin = begin / granularity * granularity;
            let aligned_end = end
                .checked_add(granularity - 1)
                .ok_or(BankError::Overflow)?
                / granularity
                * granularity;
            if aligned_end > self.storage_bytes {
                return Err(BankError::InvalidLayout);
            }
            if aligned_end - aligned_begin > granularity {
                straddling_rows += 1;
            }
            if let Some(last) = extents.last_mut() {
                let last_end = last.offset + last.len;
                if aligned_begin <= last_end {
                    last.len = last_end.max(aligned_end) - last.offset;
                    continue;
                }
            }
            extents.push(ReadExtent {
                offset: aligned_begin,
                len: aligned_end - aligned_begin,
            });
        }
        let physical_bytes = extents.iter().try_fold(0u64, |sum, e| {
            sum.checked_add(e.len).ok_or(BankError::Overflow)
        })?;
        Ok(RowReadPlan {
            extents,
            logical_bytes,
            unique_useful_bytes,
            physical_bytes,
            straddling_rows,
        })
    }
}

pub struct RowLease {
    bytes: Vec<u8>,
    pub plan: RowReadPlan,
    _permit: Box<dyn BudgetPermit>,
}
impl RowLease {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// CPU host-first exact gather. A future asynchronous engine adapter uses A's tickets
/// and pinned leases; this prototype returns HOST-ready bytes, not permission to project.
pub trait RowService {
    fn gather(&mut self, batch: RowBatch, reader: &mut dyn ExactReader) -> Result<RowLease>;
}
pub struct BoundedRowService {
    table: RowTable,
    budget: SharedBudget,
    granularity: u64,
    slot_bytes: u64,
    max_rows: usize,
    max_output_bytes: u64,
}
impl BoundedRowService {
    pub fn new(
        table: RowTable,
        budget: SharedBudget,
        granularity: u64,
        slot_bytes: u64,
        max_rows: usize,
        max_output_bytes: u64,
    ) -> Result<Self> {
        if granularity == 0
            || !granularity.is_power_of_two()
            || slot_bytes == 0
            || !slot_bytes.is_multiple_of(granularity)
            || slot_bytes > usize::MAX as u64
        {
            return Err(BankError::InvalidLayout);
        }
        table.plan(&RowBatch { rows: vec![] }, granularity)?;
        Ok(Self {
            table,
            budget,
            granularity,
            slot_bytes,
            max_rows,
            max_output_bytes,
        })
    }
}
impl RowService for BoundedRowService {
    fn gather(&mut self, batch: RowBatch, reader: &mut dyn ExactReader) -> Result<RowLease> {
        if batch.rows.len() > self.max_rows {
            return Err(BankError::TooLarge);
        }
        let plan = self.table.plan(&batch, self.granularity)?;
        if plan.logical_bytes > self.max_output_bytes {
            return Err(BankError::TooLarge);
        }
        let output_len = usize::try_from(plan.logical_bytes).map_err(|_| BankError::TooLarge)?;
        // The single physical slot is independent of the merged extent length.
        // Large batches progress incrementally, provided exact output fits admission.
        let output_permit = self.budget.reserve(Charge {
            host_bytes: plan.logical_bytes,
            ..Charge::default()
        })?;
        let slot_bytes = if batch.rows.is_empty() {
            0
        } else {
            self.slot_bytes
        };
        let _slot = self.budget.reserve(Charge {
            staging_bytes: slot_bytes,
            inflight: 1,
            ..Charge::default()
        })?;
        let mut bytes = vec![0; output_len];
        let mut slot = vec![0; slot_bytes as usize];
        for extent in &plan.extents {
            let mut position = extent.offset;
            let end = extent.offset + extent.len;
            while position < end {
                let n = (end - position).min(self.slot_bytes);
                reader.read_exact(&self.table.tensor, position, &mut slot[..n as usize])?;
                for (logical, &row) in batch.rows.iter().enumerate() {
                    let (begin, row_end) = self.table.range(row)?;
                    let overlap_begin = position.max(begin);
                    let overlap_end = (position + n).min(row_end);
                    if overlap_begin < overlap_end {
                        let from = (overlap_begin - position) as usize;
                        let to =
                            logical * self.table.width as usize + (overlap_begin - begin) as usize;
                        let len = (overlap_end - overlap_begin) as usize;
                        bytes[to..to + len].copy_from_slice(&slot[from..from + len]);
                    }
                }
                position += n;
            }
        }
        Ok(RowLease {
            bytes,
            plan,
            _permit: output_permit,
        })
    }
}
