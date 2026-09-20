//! Owning KV operands. VMM storage never escapes as an owned CudaSlice.
use cudarc::driver::{CudaSlice, CudaStream, CudaViewMut, sys};
use std::{
    ops::{Deref, Range},
    sync::Arc,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Allocation policy for native K/V planes only; recurrent/latent state is unchanged.
/// Existing cache constructors always select `Pooled`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvAllocator {
    #[default]
    Pooled,
    Vmm,
}
impl KvAllocator {
    pub(crate) fn allocate<T>(
        self,
        pooled: impl FnOnce() -> Result<T>,
        vmm: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        match self {
            Self::Pooled => pooled(),
            Self::Vmm => vmm(),
        }
    }
}

/// Mutable access exposes a borrow only, never a replaceable owning slice.
pub trait KvWrite {
    fn kv_view_mut(&mut self) -> CudaViewMut<'_, u8>;
}
impl KvWrite for CudaSlice<u8> {
    fn kv_view_mut(&mut self) -> CudaViewMut<'_, u8> {
        self.as_view_mut()
    }
}
impl KvWrite for CudaViewMut<'_, u8> {
    fn kv_view_mut(&mut self) -> CudaViewMut<'_, u8> {
        self.slice_mut(..)
    }
}

/// All whole allocation chunks inside a byte interval; partial edges stay resident.
pub fn whole_chunks(range: Range<usize>, granularity: usize) -> Result<Range<usize>> {
    if granularity == 0 || !granularity.is_power_of_two() {
        return Err("REFUSED: invalid VMM granularity".into());
    }
    if range.start > range.end {
        return Err("REFUSED: inverted VMM range".into());
    }
    let first = range.start / granularity + usize::from(!range.start.is_multiple_of(granularity));
    let end = range.end / granularity;
    Ok(first.min(end)..end)
}
fn validate_capabilities(supported: i32, granularity: usize) -> Result<()> {
    if supported != 1 {
        return Err("REFUSED: device does not support CUDA VMM".into());
    }
    whole_chunks(0..0, granularity)?;
    Ok(())
}
fn rounded(bytes: usize, granularity: usize) -> Result<usize> {
    whole_chunks(0..bytes, granularity)?;
    if bytes == 0 {
        return Err("REFUSED: empty VMM allocation".into());
    }
    bytes
        .checked_add(granularity - 1)
        .map(|n| n / granularity * granularity)
        .ok_or_else(|| "REFUSED: VMM allocation size overflow".into())
}
fn properties(device: i32) -> sys::CUmemAllocationProp {
    sys::CUmemAllocationProp {
        type_: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: sys::CUmemAllocationHandleType(0),
        location: sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: device,
        },
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: sys::CUmemAllocationProp_st__bindgen_ty_1 {
            compressionType: 0,
            gpuDirectRDMACapable: 0,
            usage: 0,
            reserved: [0; 4],
        },
    }
}

struct Chunk {
    handle: Option<sys::CUmemGenericAllocationHandle>,
    mapped: bool,
}
struct Mapping {
    base: sys::CUdeviceptr,
    reserved: bool,
    bytes: usize,
    granularity: usize,
    device: i32,
    chunks: Vec<Chunk>,
    stream: Arc<CudaStream>,
}
impl Mapping {
    fn map_chunk(&mut self, i: usize) -> Result<()> {
        if !self.reserved {
            return Err("REFUSED: VMM reservation is absent".into());
        }
        let prop = properties(self.device);
        let chunk = &mut self.chunks[i];
        if chunk.handle.is_none() {
            let mut handle = 0;
            // SAFETY: initialized allocation properties and valid handle output; size is queried granularity.
            unsafe { sys::cuMemCreate(&mut handle, self.granularity, &prop, 0).result()? };
            chunk.handle = Some(handle);
        }
        let address = self.base + (i * self.granularity) as u64;
        if !chunk.mapped {
            // SAFETY: address is an unused whole chunk in our reserved VA; handle owns this size.
            unsafe {
                sys::cuMemMap(address, self.granularity, 0, chunk.handle.unwrap(), 0).result()?
            };
            chunk.mapped = true;
        }
        // Always (re)set access, including retry after a prior SetAccess failure.
        let access = sys::CUmemAccessDesc {
            location: prop.location,
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        // SAFETY: this chunk is mapped and access descriptor names its owning device.
        unsafe { sys::cuMemSetAccess(address, self.granularity, &access, 1).result()? };
        Ok(())
    }
    fn release_chunk(&mut self, i: usize) -> Result<()> {
        let chunk = &mut self.chunks[i];
        if chunk.mapped {
            // SAFETY: caller synchronized the owner stream; this exact chunk is mapped by us.
            unsafe {
                sys::cuMemUnmap(self.base + (i * self.granularity) as u64, self.granularity)
                    .result()?
            };
            chunk.mapped = false;
        }
        if let Some(handle) = chunk.handle {
            // SAFETY: the owned handle has no remaining mapping or consumer.
            unsafe { sys::cuMemRelease(handle).result()? };
            chunk.handle = None;
        }
        Ok(())
    }
}
impl Drop for Mapping {
    fn drop(&mut self) {
        // Unknown completion leaks mappings rather than recycling possibly live storage.
        if self.stream.context().bind_to_thread().is_err() || self.stream.synchronize().is_err() {
            eprintln!("VMM cleanup quarantined: owner synchronization failed");
            return;
        }
        for i in 0..self.chunks.len() {
            if let Err(e) = self.release_chunk(i) {
                eprintln!("VMM cleanup quarantined: {e}");
                return;
            }
        }
        if !self.reserved {
            return;
        }
        // SAFETY: every mapping/handle has been released; base and bytes are our reservation.
        if let Err(e) = unsafe { sys::cuMemAddressFree(self.base, self.bytes).result() } {
            eprintln!("VMM VA cleanup failed: {e}");
        }
    }
}

pub struct KvPlane {
    operand: Option<CudaSlice<u8>>,
    mapping: Option<Mapping>,
    suspended: Option<Range<usize>>,
}
impl From<CudaSlice<u8>> for KvPlane {
    fn from(operand: CudaSlice<u8>) -> Self {
        Self {
            operand: Some(operand),
            mapping: None,
            suspended: None,
        }
    }
}
impl Deref for KvPlane {
    type Target = CudaSlice<u8>;
    fn deref(&self) -> &Self::Target {
        assert!(
            self.suspended.is_none(),
            "suspended VMM operand cannot be published"
        );
        self.operand.as_ref().expect("owned operand")
    }
}
impl KvWrite for KvPlane {
    fn kv_view_mut(&mut self) -> CudaViewMut<'_, u8> {
        assert!(
            self.suspended.is_none(),
            "suspended VMM operand cannot be published"
        );
        self.operand.as_mut().expect("owned operand").as_view_mut()
    }
}
impl KvPlane {
    pub fn as_view_mut(&mut self) -> CudaViewMut<'_, u8> {
        self.kv_view_mut()
    }
    pub fn slice_mut(&mut self, bounds: impl std::ops::RangeBounds<usize>) -> CudaViewMut<'_, u8> {
        assert!(
            self.suspended.is_none(),
            "suspended VMM operand cannot be published"
        );
        self.operand
            .as_mut()
            .expect("owned operand")
            .slice_mut(bounds)
    }
    pub fn is_vmm(&self) -> bool {
        self.mapping.is_some()
    }
    pub fn capacity_bytes(&self) -> usize {
        self.operand.as_ref().expect("owned operand").len()
    }
    pub fn granularity(&self) -> Option<usize> {
        self.mapping.as_ref().map(|m| m.granularity)
    }
    pub fn virtual_address(&self) -> Option<u64> {
        self.mapping.as_ref().map(|m| m.base)
    }
    pub fn physical_bytes(&self) -> usize {
        self.mapping.as_ref().map_or(self.capacity_bytes(), |m| {
            m.chunks.iter().filter(|c| c.handle.is_some()).count() * m.granularity
        })
    }
    /// Only pooled storage may transfer its operand out of this owner.
    pub fn into_pooled(mut self) -> Result<CudaSlice<u8>> {
        if self.is_vmm() {
            return Err("REFUSED: VMM operand ownership cannot escape".into());
        }
        Ok(self.operand.take().expect("owned operand"))
    }
    pub fn vmm(stream: Arc<CudaStream>, bytes: usize) -> Result<Self> {
        stream.context().bind_to_thread()?;
        let device = cudarc::driver::result::device::get(stream.context().ordinal() as i32)?;
        let mut supported = 0;
        // SAFETY: valid device and output pointer; unsupported/query failure refuses explicitly.
        unsafe {
            sys::cuDeviceGetAttribute(
                &mut supported,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                device,
            )
            .result()
        }
        .map_err(|e| format!("REFUSED: VMM support query failed: {e}"))?;
        if supported != 1 {
            return Err("REFUSED: device does not support CUDA VMM".into());
        }
        let prop = properties(device);
        let mut granularity = 0;
        // SAFETY: complete allocation properties and valid size output pointer.
        unsafe {
            sys::cuMemGetAllocationGranularity(
                &mut granularity,
                &prop,
                sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
            .result()
        }
        .map_err(|e| format!("REFUSED: VMM granularity query failed: {e}"))?;
        validate_capabilities(supported, granularity)?;
        let size = rounded(bytes, granularity)?;
        let mut base = 0;
        // SAFETY: valid output pointer, aligned nonzero size; no requested fixed address.
        unsafe { sys::cuMemAddressReserve(&mut base, size, granularity, 0, 0).result()? };
        let mut mapping = Mapping {
            base,
            reserved: true,
            bytes: size,
            granularity,
            device,
            stream: stream.clone(),
            chunks: (0..size / granularity)
                .map(|_| Chunk {
                    handle: None,
                    mapped: false,
                })
                .collect(),
        };
        for i in 0..mapping.chunks.len() {
            mapping.map_chunk(i)?;
        }
        // SAFETY (lead-approved cudarc constructor exception): base is our fully mapped,
        // accessible allocation of at least bytes bytes. KvPlane never exports ownership;
        // Drop leaks this operand BEFORE Mapping's VMM cleanup (never cudaFreeAsync).
        let operand = unsafe { stream.upgrade_device_ptr::<u8>(base, bytes) };
        let mut plane = Self {
            operand: Some(operand),
            mapping: Some(mapping),
            suspended: None,
        };
        stream.memset_zeros(&mut plane.kv_view_mut())?;
        stream.synchronize()?;
        Ok(plane)
    }
    /// Exclusive owner call after D2H completion/checksum and all producer uses retire.
    /// The owner is not readable while suspended; restoration must overwrite the prefix.
    pub fn demote_prefix(&mut self, bytes: usize) -> Result<usize> {
        if self.suspended.is_some() || bytes > self.capacity_bytes() {
            return Err("REFUSED: invalid or already suspended VMM prefix".into());
        }
        let m = self
            .mapping
            .as_mut()
            .ok_or("REFUSED: demotion requires VMM storage")?;
        m.stream.context().bind_to_thread()?;
        m.stream.synchronize()?;
        let chunks = whole_chunks(0..bytes, m.granularity)?;
        self.suspended = Some(chunks.clone()); // Fail closed even on partial unmap failure.
        for i in chunks.clone() {
            m.release_chunk(i)?;
        }
        Ok(chunks.len() * m.granularity)
    }
    /// Diagnostic only, while suspended: unmap the retained edge/capacity chunks WITHOUT
    /// releasing their physical handles, then release/re-reserve the original VA and
    /// re-map those same handles. No data is lost or copied. The four observations isolate
    /// unmap accounting from VA-reservation accounting; no consumer may execute here.
    /// Any failure leaves the operand suspended and the owner safely droppable.
    pub fn probe_demoted_va_release(&mut self) -> Result<[usize; 4]> {
        if self.suspended.is_none() {
            return Err("REFUSED: VA probe requires a suspended plane".into());
        }
        let m = self
            .mapping
            .as_mut()
            .ok_or("REFUSED: VA probe requires VMM")?;
        m.stream.context().bind_to_thread()?;
        m.stream.synchronize()?;
        let before = m.stream.context().mem_get_info()?.0;
        let retained: Vec<usize> = m
            .chunks
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.mapped.then_some(i))
            .collect();
        for &i in &retained {
            // SAFETY: exclusive suspended owner, stream retired; preserve handle/content.
            unsafe {
                sys::cuMemUnmap(m.base + (i * m.granularity) as u64, m.granularity).result()?
            };
            m.chunks[i].mapped = false;
        }
        let unmapped = m.stream.context().mem_get_info()?.0;
        // SAFETY: all chunks are now unmapped; this owner holds the entire reservation.
        unsafe { sys::cuMemAddressFree(m.base, m.bytes).result()? };
        m.reserved = false;
        let freed = m.stream.context().mem_get_info()?.0;
        let mut address = 0;
        // SAFETY: request the original aligned VA; success address is checked before use.
        unsafe {
            sys::cuMemAddressReserve(&mut address, m.bytes, m.granularity, m.base, 0).result()?
        };
        if address != m.base {
            // SAFETY: the unexpected reservation is unexposed and has no mappings.
            unsafe { sys::cuMemAddressFree(address, m.bytes).result()? };
            return Err("REFUSED: diagnostic could not re-reserve original VMM address".into());
        }
        m.reserved = true;
        for i in retained {
            m.map_chunk(i)?;
        }
        let restored = m.stream.context().mem_get_info()?.0;
        Ok([before, unmapped, freed, restored])
    }

    /// Remap fixed VA; caller must restore the host image before any consumer publication.
    pub fn remap_prefix(&mut self) -> Result<usize> {
        let chunks = self
            .suspended
            .clone()
            .ok_or("REFUSED: VMM prefix is not suspended")?;
        let m = self
            .mapping
            .as_mut()
            .ok_or("REFUSED: remap requires VMM storage")?;
        m.stream.context().bind_to_thread()?;
        for i in chunks.clone() {
            m.map_chunk(i)?;
        }
        let bytes = chunks.len() * m.granularity;
        self.suspended = None;
        Ok(bytes)
    }
}
impl Drop for KvPlane {
    fn drop(&mut self) {
        if self.mapping.is_some() {
            if let Some(operand) = self.operand.take() {
                operand.leak();
            }
        }
        // Pooled operand drops normally; VMM Mapping drops only after operand is disarmed.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocator_dispatch_has_no_bootstrap_or_fallback() {
        assert_eq!(KvAllocator::default(), KvAllocator::Pooled);
        assert_eq!(
            KvAllocator::Vmm
                .allocate(|| panic!("pooled bootstrap"), || Ok(7))
                .unwrap(),
            7
        );
        assert_eq!(
            KvAllocator::Pooled
                .allocate(|| Ok(9), || panic!("changed default"))
                .unwrap(),
            9
        );
        let error = KvAllocator::Vmm
            .allocate::<()>(|| panic!("silent fallback"), || Err("VMM refused".into()))
            .unwrap_err();
        assert_eq!(error.to_string(), "VMM refused");
    }
    #[test]
    fn frozen_qwen_prefix_whole_chunk_counts() {
        for (tokens, count) in [(8192, 7), (8064, 6), (32768, 29), (32640, 27)] {
            let count_actual = whole_chunks(0..tokens * 1088, 2 << 20).unwrap().len()
                + whole_chunks(0..tokens * 768, 2 << 20).unwrap().len();
            assert_eq!(count_actual, count);
        }
    }
    #[test]
    fn capability_refusals_are_named() {
        for (supported, granularity) in [(0, 2 << 20), (2, 2 << 20), (1, 0), (1, 3)] {
            assert!(
                validate_capabilities(supported, granularity)
                    .unwrap_err()
                    .to_string()
                    .starts_with("REFUSED:")
            );
        }
        assert!(validate_capabilities(1, 2 << 20).is_ok());
    }
    #[test]
    fn edges_empty_and_refusals() {
        assert_eq!(whole_chunks(1..7, 4).unwrap(), 1..1);
        assert_eq!(whole_chunks(1..12, 4).unwrap(), 1..3);
        assert_eq!(whole_chunks(9..9, 4).unwrap(), 2..2);
        assert!(whole_chunks(Range { start: 4, end: 3 }, 4).is_err());
        for g in [0, 3, 6] {
            assert!(whole_chunks(0..8, g).is_err());
        }
        assert!(rounded(0, 4).is_err());
        assert!(rounded(usize::MAX, 4).is_err());
        assert_eq!(rounded(8, 4).unwrap(), 8);
        assert_eq!(rounded(9, 4).unwrap(), 12);
    }
}
