//! Owner-side admission, worker-side verified bytes. No governor or device authority
//! crosses the worker boundary. Pending extents are not published ObjectLeases.
use super::*;
use crate::io::{
    ReadAt,
    direct::{AlignedBuffer, AlignedFile, ReadMode},
};
use std::sync::Arc;

pub struct PreparedExtent {
    pub(crate) lease: ExtentLease,
    pub(crate) source: Arc<dyn ReadAt>,
}
/// Additive CPU extension; does not weaken the frozen ObjectStore::lease semantics.
pub trait BackgroundStore: ObjectStore {
    fn prepare_extent(
        &mut self,
        key: &ObjectKey,
        chunk: u32,
        valid: u64,
        request: &BudgetRequest,
    ) -> Result<PreparedExtent>;
    fn release_prepared(&mut self, lease: &ExtentLease) -> Result<()>;
}
struct VerifiedFile {
    file: std::fs::File,
    direct: bool,
    reference: ChunkRef,
}
impl ReadAt for VerifiedFile {
    fn read_at(&self, _: &mut [u8], _: u64) -> std::io::Result<usize> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "use verified read_exact",
        ))
    }
    fn read_exact(&self, dst: &mut [u8], offset: u64) -> Result<u64> {
        if offset != 0 || dst.len() as u64 != self.reference.valid_bytes {
            return Err(Error::InvalidLayout);
        }
        let size = self.reference.storage_bytes as usize + ALIGNMENT;
        if self.file.metadata()?.len() != size as u64 {
            return Err(Error::Corrupt);
        }
        let mut buffer = AlignedBuffer::new(size)?;
        if self.direct {
            AlignedFile::from_configured_file(self.file.try_clone()?)
                .read_aligned(0, buffer.bytes_mut())?;
        } else {
            crate::io::read_exact_at(&self.file, buffer.bytes_mut(), 0)?;
        }
        let bytes = buffer.bytes();
        if contracts::digest("extent", bytes) != self.reference.encoded_digest {
            return Err(Error::Corrupt);
        }
        let payload = decode_extent(bytes)?;
        if payload.len() != dst.len() || digest(payload) != self.reference.checksum {
            return Err(Error::Corrupt);
        }
        dst.copy_from_slice(payload);
        Ok(dst.len() as u64)
    }
}
impl<G: BudgetGovernor> BackgroundStore for ExtentStore<FileBackend, G> {
    fn prepare_extent(
        &mut self,
        key: &ObjectKey,
        chunk: u32,
        valid: u64,
        request: &BudgetRequest,
    ) -> Result<PreparedExtent> {
        let manifest = self.lookup(key)?.ok_or(Error::NotFound)?;
        let reference = manifest
            .chunks
            .get(chunk as usize)
            .ok_or(Error::InvalidLayout)?;
        if valid != reference.valid_bytes {
            return Err(Error::InvalidLayout);
        }
        // One bounded aligned worker scratch allocation per accepted read. Pool
        // backing is charged separately once, never counted again in this request.
        let scratch = reference.storage_bytes + ALIGNMENT as u64 + ALIGNMENT as u64 - 1;
        if request.bytes.pageable < scratch {
            return Err(Error::Capacity);
        }
        let mut r = request.clone();
        r.bytes.pageable = scratch;
        let lease = self.reserve_extent(&manifest, chunk, &r)?;
        let path = self.backend.path(false, reference.encoded_digest);
        let direct = self.backend.read_mode == ReadMode::Uncached;
        let opened = if direct {
            match self.backend.uncached_opener {
                Some(open) => open(&path),
                None => AlignedFile::open(&path).map(AlignedFile::into_file),
            }
        } else {
            std::fs::File::open(path).map_err(Error::from)
        };
        let file = match opened {
            Ok(file) => file,
            Err(error) => {
                self.release_extent(&lease)?;
                return Err(error);
            }
        };
        Ok(PreparedExtent {
            source: Arc::new(VerifiedFile {
                file,
                direct,
                reference: reference.clone(),
            }),
            lease,
        })
    }
    fn release_prepared(&mut self, lease: &ExtentLease) -> Result<()> {
        self.release_extent(lease)
    }
}
