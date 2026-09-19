//! ObjectStore → CpuTransfers adapter for the explicit bank progress pump.
//! No I/O in stage/gather. This CPU pump is not an async OS worker or CUDA owner.
use super::*;
use crate::{
    contracts::*,
    io::transfer::CpuTransfers,
    pool::{Admission, FakePinnedPool},
};
use std::collections::BTreeMap;

pub struct ObjectReader<S: ObjectStore> {
    transfers: CpuTransfers<S>,
    pool: FakePinnedPool,
    objects: BTreeMap<TensorId, ObjectManifest>,
    epochs: Epochs,
    priority: Priority,
    pub submitted: u64,
    pub retired: u64,
    pub io_bytes: u64,
    /// Bounded diagnostic, not an ever-growing history of tickets.
    pub last_completion: Option<Completion>,
}
impl<S: ObjectStore> ObjectReader<S> {
    /// Loader supplies immutable object keys. Full tensor identity is mandatory;
    /// chunks are consecutive byte ranges of that exact tensor (including padding).
    pub fn new(
        store: S,
        bindings: Vec<(TensorId, ObjectKey)>,
        pool: FakePinnedPool,
        request: BudgetRequest,
        queue_charge: &ChargedLease,
    ) -> Result<Self> {
        let mut objects = BTreeMap::new();
        for (tensor, key) in bindings {
            tensor.validate()?;
            if key.artifact != tensor.artifact || key.semantic_id != tensor.identity()? {
                return Err(Error::InvalidLayout);
            }
            let manifest = store.lookup(&key)?.ok_or(Error::NotFound)?;
            if objects.insert(tensor, manifest).is_some() {
                return Err(Error::Conflict);
            }
        }
        Ok(Self {
            transfers: CpuTransfers::new(store, request, 1, queue_charge)?,
            pool,
            objects,
            epochs: Epochs {
                state: 0,
                src_gen: 0,
                dst_gen: 0,
            },
            priority: Priority::OptionalPrefetch,
            submitted: 0,
            retired: 0,
            io_bytes: 0,
            last_completion: None,
        })
    }
    fn chunk(
        &mut self,
        object: ObjectKey,
        chunk: u32,
        bytes: usize,
        range: std::ops::Range<usize>,
        dst: &mut [u8],
    ) -> Result<()> {
        let destination = self.pool.acquire(
            bytes,
            if self.priority == Priority::OptionalPrefetch {
                Admission::Optional
            } else {
                Admission::Demand
            },
        )?;
        let op = ReadPlan {
            object,
            chunk,
            destination,
            epochs: self.epochs,
        };
        let ticket = self.transfers.nvme_read(op).map_err(|r| r.error)?;
        self.submitted += 1;
        // Every accepted ticket is driven, inspected per item, retired and acknowledged,
        // including failed/short/corrupt reads. No successful sibling is published early.
        self.transfers.drive(&ticket)?;
        let completion = self.transfers.poll(&ticket)?;
        self.io_bytes += completion
            .items
            .iter()
            .flat_map(|i| &i.segments)
            .map(|s| s.io_bytes)
            .sum::<u64>();
        self.last_completion = Some(completion);
        let result = match self.transfers.take_destination(&ticket, 0, self.epochs) {
            Ok(Destination::Host(host)) => {
                let result = host.bytes().and_then(|source| {
                    let source = source.get(range).ok_or(Error::InvalidLayout)?;
                    if source.len() != dst.len() {
                        return Err(Error::InvalidLayout);
                    }
                    dst.copy_from_slice(source);
                    Ok(())
                });
                drop(host); // consumer done before retirement, never while borrowed
                result
            }
            Ok(_) => Err(Error::Unsupported),
            Err(e) => {
                self.transfers.cancel(&ticket)?;
                Err(e)
            }
        };
        self.transfers.retire(&ticket, None)?;
        if !self.transfers.retired(&ticket)? {
            return Err(Error::Busy);
        }
        self.retired += 1;
        self.transfers.acknowledge(&ticket)?;
        result
    }
}
impl<S: ObjectStore> ExactReader for ObjectReader<S> {
    fn begin_request(&mut self, request: &BudgetRequest, epochs: Epochs) {
        self.priority = request.priority;
        self.epochs = epochs;
    }
    fn storage_bytes(&self, tensor: &TensorId) -> Result<u64> {
        Ok(self.objects.get(tensor).ok_or(Error::NotFound)?.valid_bytes)
    }
    fn read_exact(&mut self, tensor: &TensorId, offset: u64, dst: &mut [u8]) -> Result<()> {
        let manifest = self.objects.get(tensor).ok_or(Error::NotFound)?.clone();
        let end = offset
            .checked_add(dst.len() as u64)
            .ok_or(Error::Overflow)?;
        if end > manifest.valid_bytes {
            return Err(Error::InvalidLayout);
        }
        let mut base = 0u64;
        let mut copied = 0;
        for (index, chunk) in manifest.chunks.iter().enumerate() {
            let chunk_end = base.checked_add(chunk.valid_bytes).ok_or(Error::Overflow)?;
            let lo = offset.max(base);
            let hi = end.min(chunk_end);
            if lo < hi {
                let n = usize::try_from(hi - lo).map_err(|_| Error::Capacity)?;
                self.chunk(
                    manifest.key.clone(),
                    index as u32,
                    usize::try_from(chunk.valid_bytes).map_err(|_| Error::Capacity)?,
                    (lo - base) as usize..(hi - base) as usize,
                    &mut dst[copied..copied + n],
                )?;
                copied += n;
            }
            base = chunk_end;
        }
        if copied != dst.len() {
            return Err(Error::ShortIo {
                expected: dst.len() as u64,
                actual: copied as u64,
            });
        }
        Ok(())
    }
}
