//! Bounded PLE qualification adapter. No implicit CUDA permission, persistent
//! cache, source artifact validation, or production governor is implied here.
//! The owner injects the same budget for every gather. The immutable borrowed
//! table is already loaded by the native loader; only requested rows are indexed.
use memra_tier::{bank::*, contracts::*};
use std::{cell::Cell, collections::BTreeSet};

const MAX_ROWS: usize = 4096;
const MAX_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(crate) enum HostTable<'a> {
    F32(&'a [f32]),
    Bf16(&'a [u8]),
}
impl HostTable<'_> {
    fn bytes(self) -> usize {
        match self {
            Self::F32(v) => v.len() * 4,
            Self::Bf16(v) => v.len(),
        }
    }
    fn encoding(self) -> PleEncoding {
        match self {
            Self::F32(_) => PleEncoding::F32,
            Self::Bf16(_) => PleEncoding::Bf16,
        }
    }
    fn read(self, offset: u64, dst: &mut [u8]) -> Result<()> {
        let start = usize::try_from(offset).map_err(|_| Error::Overflow)?;
        let end = start.checked_add(dst.len()).ok_or(Error::Overflow)?;
        if end > self.bytes() {
            return Err(Error::InvalidLayout);
        }
        match self {
            Self::Bf16(v) => dst.copy_from_slice(&v[start..end]),
            Self::F32(v) => {
                // Explicit LE bytes, including unaligned bounded read boundaries.
                for (i, b) in dst.iter_mut().enumerate() {
                    let at = start + i;
                    *b = v[at / 4].to_bits().to_le_bytes()[at % 4];
                }
            }
        }
        Ok(())
    }
}

struct Reader<'a> {
    source: HostTable<'a>,
    tensor: TensorId,
    reads: u64,
}
impl ExactReader for Reader<'_> {
    fn storage_bytes(&self, tensor: &TensorId) -> Result<u64> {
        if tensor != &self.tensor {
            return Err(Error::NotFound);
        }
        Ok(self.source.bytes() as u64)
    }
    fn read_exact(&mut self, tensor: &TensorId, offset: u64, dst: &mut [u8]) -> Result<()> {
        self.storage_bytes(tensor)?;
        self.source.read(offset, dst)?;
        self.reads += 1;
        Ok(())
    }
}
struct NoCacheHeat;
impl Hotness<RowDomain> for NoCacheHeat {
    fn demand(&mut self, _: &BankId) {}
    fn score(&self, _: &BankId) -> u64 {
        0
    }
}

/// Retains the bounded expansion/metadata reservation through the existing H2D
/// consumer. Legacy vectors have no tier charge. No GPU allocation is represented.
pub(crate) struct GatheredRows {
    values: Vec<f32>,
    reservation: Option<(SharedBudget, ChargedLease)>,
}
impl GatheredRows {
    /// Submit only an exact copy of the already expanded, coalesced row window.
    /// The backend retains owned pinned/device leases after acceptance. Keeping
    /// this wrapper alive is not a substitute for DMA or consumer retirement.
    #[allow(
        dead_code,
        reason = "native device adapter is installed by the explicit gate"
    )]
    #[allow(clippy::result_large_err)]
    pub(crate) fn submit_device<T: TransferEngine>(
        &self,
        transfers: &mut T,
        op: CopyOp<T::Host>,
    ) -> std::result::Result<RowUpload, Rejected<CopyOp<T::Host>>> {
        let valid = (|| {
            let bytes = op.host.bytes()?;
            if bytes.len() != self.values.len().checked_mul(4).ok_or(Error::Overflow)?
                || bytes
                    .chunks_exact(4)
                    .zip(&self.values)
                    .any(|(raw, value)| raw != value.to_bits().to_le_bytes())
            {
                return Err(Error::Corrupt);
            }
            Ok(())
        })();
        if let Err(error) = valid {
            return Err(Rejected { op, error });
        }
        RowUpload::submit(transfers, op)
    }
    pub(crate) fn legacy(values: Vec<f32>) -> Self {
        Self {
            values,
            reservation: None,
        }
    }
}
impl std::ops::Deref for GatheredRows {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.values
    }
}
impl Drop for GatheredRows {
    fn drop(&mut self) {
        if let Some((budget, charge)) = &self.reservation {
            // CPU backing only; no outstanding views/pins are exported by this wrapper.
            let released = budget.borrow_mut().release(charge);
            debug_assert!(released.is_ok(), "PLE host output charge did not retire");
        }
    }
}

/// This typed door is only installed by the explicit native gate. It is not an
/// env switch and cannot change an existing request's numerical program.
pub(crate) struct PleRowsTier {
    budget: SharedBudget,
    pub(crate) calls: Cell<u64>,
    pub(crate) reads: Cell<u64>,
}
impl PleRowsTier {
    pub(crate) fn new(budget: SharedBudget) -> Self {
        Self {
            budget,
            calls: Cell::new(0),
            reads: Cell::new(0),
        }
    }
    pub(crate) fn gather(
        &self,
        source: HostTable<'_>,
        layer: u32,
        head_dim: usize,
        ids: &[i64],
    ) -> Result<GatheredRows> {
        let width = head_dim
            .checked_mul(source.encoding().element_bytes() as usize)
            .ok_or(Error::Overflow)?;
        let output_bytes = ids
            .len()
            .checked_mul(head_dim)
            .and_then(|n| n.checked_mul(4))
            .ok_or(Error::Overflow)?;
        if ids.is_empty()
            || ids.len() > MAX_ROWS
            || width == 0
            || !source.bytes().is_multiple_of(width)
            || output_bytes as u64 > MAX_BYTES
        {
            return Err(Error::Capacity);
        }
        let table = PleTable {
            // Ephemeral host-source namespace, never a claimed checkpoint digest.
            // No cache outlives this borrow, so layers/encodings cannot alias across calls.
            tensor: TensorId {
                version: WIRE_VERSION,
                artifact: digest("ple-host-qualification-v1", &[]),
                name: format!("layer.{layer}.ple.host-table"),
            },
            rows: (source.bytes() / width) as u64,
            head_dim: head_dim as u64,
            encoding: source.encoding(),
        };
        // Charge bounded caller metadata + expansion before allocating the window.
        // The source's existing loader allocation remains owned by that loader.
        let devices = self.budget.borrow().used().device.len();
        let mut req = BudgetRequest {
            bytes: TierBudget::zero(devices),
            priority: Priority::Demand,
            deadline: Deadline(u64::MAX),
            tenant: digest("ple-host-gate-owner-v1", &[]),
        };
        req.bytes.pageable = (output_bytes as u64)
            .checked_add((ids.len() as u64) * 8192)
            .and_then(|n| n.checked_add(width as u64))
            .ok_or(Error::Overflow)?;
        let metadata = self.budget.borrow_mut().reserve(&req)?;
        let result = (|| {
            let requested = ids
                .iter()
                .map(|&r| table.id(u64::try_from(r).map_err(|_| Error::NotFound)?))
                .collect::<Result<Vec<_>>>()?;
            let unique: BTreeSet<_> = ids.iter().copied().collect();
            let mut entries = Vec::with_capacity(unique.len());
            let mut scratch = vec![0; width];
            for row in unique {
                let row = row as u64;
                let layout = table.layout(row)?;
                source.read(row * width as u64, &mut scratch)?;
                entries.push((
                    table.id(row)?,
                    Some(CatalogRecord {
                        layout,
                        checksums: vec![checksum(&scratch)],
                    }),
                ));
            }
            // Checksums validate transport against this immutable host borrow;
            // they do not authenticate a checkpoint or a previously persisted object.
            let catalog = Catalog::new(LayoutClass::PerRecord, entries)?;
            let reader = Reader {
                source,
                tensor: table.tensor.clone(),
                reads: 0,
            };
            let mut rows = BoundedRowService(BankService::new(
                catalog,
                self.budget.clone(),
                NoCacheHeat,
                reader,
                CoalescingPolicy {
                    granularity: 1,
                    slot_bytes: 4096,
                },
                BankLimits {
                    cache_bytes: 0,
                    batch_bytes: MAX_BYTES,
                    items: MAX_ROWS,
                    tickets: 1,
                },
            )?);
            req.bytes = TierBudget::zero(devices);
            let epochs = Epochs {
                state: 0,
                src_gen: 0,
                dst_gen: 0,
            };
            let ticket = rows.gather(RowBatch {
                ids: requested,
                epochs,
                request: req,
            })?;
            let mut acknowledged = false;
            let consume = (|| {
                while !rows.0.progress(&ticket)? {}
                let leases = rows.publish(&ticket, epochs)?;
                let expanded = (|| {
                    let mut out = Vec::with_capacity(output_bytes / 4);
                    for lease in &leases.records {
                        out.extend(table.expand_row(lease)?);
                    }
                    Ok(out)
                })();
                // CPU expansion COPIES every byte to `out`. GPU consumers later own
                // that separate Vec via the existing synchronous htod, not these leases.
                rows.0.finish_host_use(&ticket)?;
                if !rows.retire(&ticket)? {
                    return Err(Error::NotReady);
                }
                rows.release(&leases)?;
                rows.0.acknowledge(&ticket)?;
                acknowledged = true;
                expanded
            })();
            if consume.is_err() && !acknowledged {
                rows.cancel(&ticket)?;
                while !rows.0.progress(&ticket)? {}
                rows.0.finish_host_use(&ticket)?;
                if !rows.retire(&ticket)? {
                    return Err(Error::NotReady);
                }
                rows.0.acknowledge(&ticket)?;
            }
            self.calls.set(self.calls.get() + 1);
            self.reads.set(self.reads.get() + rows.0.reader().reads);
            consume
        })();
        match result {
            Ok(values) => Ok(GatheredRows {
                values,
                reservation: Some((self.budget.clone(), metadata)),
            }),
            Err(error) => {
                self.budget.borrow_mut().release(&metadata)?;
                Err(error)
            }
        }
    }
}
