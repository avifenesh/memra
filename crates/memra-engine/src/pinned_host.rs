//! Fixed, portable pinned storage. Only startup/shutdown allocate/free CUDA host memory.
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

type Error = Box<dyn std::error::Error>;

#[derive(Clone, Debug)]
struct Extents {
    free: BTreeMap<usize, usize>,
    capacity: usize,
}
impl Extents {
    fn new(capacity: usize) -> Self {
        Self {
            free: [(0, capacity)].into(),
            capacity,
        }
    }
    fn reserve(&mut self, sizes: &[usize]) -> Option<Vec<(usize, usize)>> {
        let mut next = self.clone();
        let mut regions = Vec::with_capacity(sizes.len());
        for &len in sizes {
            let size = len.max(1).checked_add(3)? & !3;
            let (&offset, &available) = next
                .free
                .iter()
                .filter(|(_, n)| **n >= size)
                .min_by_key(|(offset, n)| (**n, **offset))?;
            next.free.remove(&offset);
            if available > size {
                next.free.insert(offset + size, available - size);
            }
            regions.push((offset, size));
        }
        *self = next;
        Some(regions)
    }
    fn release(&mut self, mut offset: usize, mut size: usize) {
        assert!(
            offset
                .checked_add(size)
                .is_some_and(|end| end <= self.capacity)
        );
        if let Some((&left, &n)) = self.free.range(..offset).next_back() {
            assert!(left + n <= offset, "overlapping pinned region return");
            if left + n == offset {
                self.free.remove(&left);
                offset = left;
                size += n;
            }
        }
        if let Some((&right, &n)) = self.free.range(offset..).next() {
            assert!(offset + size <= right, "overlapping pinned region return");
            if offset + size == right {
                self.free.remove(&right);
                size += n;
            }
        }
        assert!(self.free.insert(offset, size).is_none());
    }
    fn free_bytes(&self) -> usize {
        self.free.values().sum()
    }
}

struct ArenaInner {
    ptr: *mut u8,
    context: Arc<CudaContext>,
    extents: Mutex<Extents>,
}
// Backing never moves. Only exclusive, non-overlapping region owners expose data;
// the mutex controls extent ownership, and each buffer's copies fence before return.
unsafe impl Send for ArenaInner {}
unsafe impl Sync for ArenaInner {}
impl Drop for ArenaInner {
    fn drop(&mut self) {
        // The retained creator context outlives CUDA host free.
        let _ = self.context.bind_to_thread();
        let _ = unsafe { cudarc::driver::result::free_host(self.ptr.cast()) };
    }
}

#[derive(Clone)]
pub struct PinnedHostArena {
    inner: Arc<ArenaInner>,
}
impl PinnedHostArena {
    /// Reserve the entire configured physical budget before readiness. Portable is
    /// cacheable (not write-combined) and recognized by all CUDA owner contexts.
    pub fn reserve(context: Arc<CudaContext>, bytes: usize) -> Result<Self, Error> {
        if bytes == 0 || bytes > isize::MAX as usize || !bytes.is_multiple_of(4) {
            return Err("pinned arena size must be positive and four-byte aligned".into());
        }
        context.bind_to_thread()?;
        let ptr = unsafe {
            cudarc::driver::result::malloc_host(
                bytes,
                cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE,
            )?
        }
        .cast::<u8>();
        // Initialize backing once at startup, not on request-time region recycling.
        // Thus safe slice access never exposes uninitialized Rust values.
        unsafe {
            ptr.write_bytes(0, bytes);
        }
        Ok(Self {
            inner: Arc::new(ArenaInner {
                ptr,
                context,
                extents: Mutex::new(Extents::new(bytes)),
            }),
        })
    }
    /// All regions or none. No CUDA call, no growth, no caller-visible raw pointer.
    pub fn try_reserve_planes(&self, sizes: &[usize]) -> Result<Vec<PinnedHostBuf>, Error> {
        let mut extents = self
            .inner
            .extents
            .lock()
            .map_err(|_| "pinned arena lock poisoned")?;
        let regions = extents
            .reserve(sizes)
            .ok_or("pinned arena capacity/fragmentation refusal")?;
        Ok(regions
            .into_iter()
            .zip(sizes)
            .map(|((offset, reserved), &len)| PinnedHostBuf {
                ptr: unsafe { self.inner.ptr.add(offset) },
                len,
                region: Some(Region {
                    arena: self.inner.clone(),
                    offset,
                    reserved,
                }),
            })
            .collect())
    }
    /// Physical backing, leased extents (including alignment), reusable extents.
    pub fn bytes(&self) -> (usize, usize, usize) {
        let e = self
            .inner
            .extents
            .lock()
            .expect("pinned arena lock poisoned");
        let free = e.free_bytes();
        (e.capacity, e.capacity - free, free)
    }
}
struct Region {
    arena: Arc<ArenaInner>,
    offset: usize,
    reserved: usize,
}
impl Drop for Region {
    fn drop(&mut self) {
        self.arena
            .extents
            .lock()
            .expect("pinned arena lock poisoned")
            .release(self.offset, self.reserved);
    }
}

/// Exclusive logical range, either legacy owned backing or one arena lease. Not Clone:
/// a second mutable owner can never alias a published/leased range.
pub struct PinnedHostBuf {
    ptr: *mut u8,
    len: usize,
    region: Option<Region>,
}
unsafe impl Send for PinnedHostBuf {}
impl PinnedHostBuf {
    pub fn new(len: usize) -> Result<Self, Error> {
        if len > isize::MAX as usize {
            return Err("pinned buffer exceeds slice limit".into());
        }
        let ptr = unsafe { cudarc::driver::result::malloc_host(len.max(1), 0)? }.cast::<u8>();
        // Safe slice access requires initialized backing, including the legacy path.
        unsafe {
            ptr.write_bytes(0, len);
        }
        Ok(Self {
            ptr,
            len,
            region: None,
        })
    }
    pub fn from_device_f32(src: &CudaSlice<f32>) -> Result<Self, Error> {
        let n = src.len().checked_mul(4).ok_or("pinned f32 size overflow")?;
        src.context().bind_to_thread()?;
        let mut out = Self::new(n)?;
        out.copy_from_device_f32(src)?;
        Ok(out)
    }
    pub fn copy_from_device_f32(&mut self, src: &CudaSlice<f32>) -> Result<(), Error> {
        let n = src.len().checked_mul(4).ok_or("pinned f32 size overflow")?;
        if n != self.len {
            return Err("pinned f32 copy length mismatch".into());
        }
        let dst = unsafe { std::slice::from_raw_parts_mut(self.ptr.cast::<f32>(), src.len()) };
        let copy = src.stream().memcpy_dtoh(src, dst);
        let fence = src.stream().synchronize();
        copy?;
        fence?;
        Ok(())
    }
    /// Copy a complete logical byte plane on its owning stream, fencing even
    /// when enqueue returns an error before the region can be recycled.
    pub fn copy_from_device_u8(&mut self, src: &CudaSlice<u8>) -> Result<(), Error> {
        if self.len > src.len() {
            return Err("pinned byte copy length mismatch".into());
        }
        let len = self.len;
        let copy = src
            .stream()
            .memcpy_dtoh(&src.slice(..len), self.as_mut_slice());
        let fence = src.stream().synchronize();
        copy?;
        fence?;
        Ok(())
    }
    pub fn to_device_f32(&self, stream: &Arc<CudaStream>) -> Result<CudaSlice<f32>, Error> {
        if !self.len.is_multiple_of(4) {
            return Err("pinned f32 byte length is not divisible by four".into());
        }
        let out = stream.clone_htod(self.as_f32_slice());
        let fence = stream.synchronize();
        let out = out?;
        fence?;
        Ok(out)
    }
    fn as_f32_slice(&self) -> &[f32] {
        assert!(self.len.is_multiple_of(4));
        unsafe { std::slice::from_raw_parts(self.ptr.cast(), self.len / 4) }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl Drop for PinnedHostBuf {
    fn drop(&mut self) {
        if self.region.is_none() {
            let _ = unsafe { cudarc::driver::result::free_host(self.ptr.cast()) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_arena_exact_fit_and_coalesced_reuse() {
        let mut e = Extents::new(64);
        let ranges = e.reserve(&[1, 12, 48]).unwrap();
        assert_eq!(ranges, [(0, 4), (4, 12), (16, 48)]);
        assert_eq!(e.free_bytes(), 0);
        e.release(4, 12);
        e.release(0, 4);
        e.release(16, 48);
        assert_eq!(e.reserve(&[64]), Some(vec![(0, 64)]));
    }
    #[test]
    fn pinned_arena_fragmentation_and_failed_batch_are_atomic() {
        let mut e = Extents::new(64);
        let r = e.reserve(&[16; 4]).unwrap();
        e.release(r[0].0, r[0].1);
        e.release(r[2].0, r[2].1);
        let before = e.free.clone();
        assert!(e.reserve(&[20]).is_none());
        assert!(e.reserve(&[16, 20]).is_none());
        assert_eq!(e.free, before);
        assert!(e.reserve(&[usize::MAX]).is_none());
        assert_eq!(e.free, before);
    }
    #[test]
    fn pinned_arena_disjoint_leases_and_zero_length() {
        let mut e = Extents::new(256);
        let r = e.reserve(&[0, 7, 33, 61, 128]).unwrap();
        for pair in r.windows(2) {
            assert!(pair[0].0 + pair[0].1 <= pair[1].0);
        }
        for (offset, len) in r.into_iter().rev() {
            e.release(offset, len);
        }
        assert_eq!(e.free_bytes(), 256);
    }
    #[test]
    fn pinned_arena_concurrent_leases_never_overlap() {
        let ledger = Arc::new(Mutex::new(Extents::new(4096)));
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let ledger = ledger.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let regions = ledger.lock().unwrap().reserve(&[64; 8]).unwrap();
                    barrier.wait();
                    regions
                })
            })
            .collect();
        let mut regions: Vec<_> = threads
            .into_iter()
            .flat_map(|t| t.join().unwrap())
            .collect();
        assert_eq!(ledger.lock().unwrap().free_bytes(), 0);
        regions.sort_unstable();
        for pair in regions.windows(2) {
            assert!(pair[0].0 + pair[0].1 <= pair[1].0);
        }
        for (offset, len) in regions {
            ledger.lock().unwrap().release(offset, len);
        }
        assert_eq!(ledger.lock().unwrap().free_bytes(), 4096);
    }

    #[test]
    fn pinned_arena_mock_host_promotion_overwrites_recycled_image() {
        let mut ledger = Extents::new(64);
        let mut backing = [0u8; 64];
        let source = [vec![17u8; 32], vec![93u8; 32]];
        for _ in 0..3 {
            let regions = ledger.reserve(&[32, 32]).unwrap();
            for ((offset, len), plane) in regions.iter().zip(&source) {
                backing[*offset..offset + len].copy_from_slice(plane);
            }
            let promoted: Vec<_> = regions
                .iter()
                .map(|(offset, len)| backing[*offset..offset + len].to_vec())
                .collect();
            assert_eq!(promoted, source);
            backing.fill(255);
            for (offset, len) in regions {
                ledger.release(offset, len);
            }
            assert_eq!(ledger.free_bytes(), 64);
        }
    }
}
