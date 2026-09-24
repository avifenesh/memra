//! Owning KV operands. VMM storage never escapes as an owned CudaSlice.
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, CudaViewMut, sys};
use std::{
    cell::Cell,
    ops::{Deref, Range},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
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

fn vmm_granularity(device: i32) -> Result<usize> {
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
    Ok(granularity)
}

static GRANULARITIES: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// The VMM allocation granularity of the stream's device (the boot line's `granularity=`), or
/// the named refusal when the device has no VMM support. Queried once per device ordinal.
pub fn vmm_granularity_for(stream: &CudaStream) -> Result<usize> {
    let ordinal = stream.context().ordinal();
    if let Some(&(_, g)) = GRANULARITIES
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .find(|(o, _)| *o == ordinal)
    {
        return Ok(g);
    }
    stream.context().bind_to_thread()?;
    let device = cudarc::driver::result::device::get(ordinal as i32)?;
    let g = vmm_granularity(device)?;
    GRANULARITIES
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .push((ordinal, g));
    Ok(g)
}

/// DAY37 addendum C: a plane is worth backing on demand only when its pooled capacity leaves at
/// least one whole granule unbacked after the initial extent. Below that line an on-demand plane
/// can never back fewer bytes than the pooled one (it maps whole granules up to a reservation
/// rounded above the capacity), so the pooled plane is never worse there.
pub fn on_demand_pays(capacity: usize, initial: usize, granularity: usize) -> bool {
    granularity > 0
        && capacity
            >= initial
                .div_ceil(granularity)
                .saturating_mul(granularity)
                .saturating_add(granularity)
}

// ---------------- on-demand planes (WP-B day 37, `MEMRA_KV_ALLOCATOR=vmm`) ----------------
//
// A plane reserves its whole capacity as virtual range once and backs it in extents: one
// `cuMemCreate` handle per grow, mapped at the end of the backed prefix. The address never
// changes, so graphs and kernels that bake it stay valid. A release never runs while work
// that may touch the range is in flight: it records an event and the extent is unmapped
// only after the event completes (`reap`); a dropped plane goes to the graveyard, reaped the
// same way. Nothing on these paths synchronizes the owner stream.
//
// Grow placement (DAY37 1.5, addendum A 1.10): `Inline` maps at the ensure point on the owner
// thread; `Helper` keeps a mapper thread one lookahead ahead (`want = need + 2 granules`) and
// the owner maps inline only when the mapper is behind (`waited`). Every state change of a
// plane (its extents, `want`, `dead`) happens under the plane's one lock, which the mapper
// holds across its driver calls and its zero-fill enqueue, so the owner never launches work on
// rows before their fill, and every release event is recorded after the mapper's last enqueue.

/// Where on-demand grows run (DAY37 1.5's rule, per card class).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VmmGrowPlacement {
    /// A mapper thread pre-maps; the owner maps only when the mapper is behind.
    #[default]
    Helper,
    /// The owner maps at the ensure point.
    Inline,
}
impl VmmGrowPlacement {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Helper => "helper",
            Self::Inline => "inline",
        }
    }
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "helper" => Some(Self::Helper),
            "inline" => Some(Self::Inline),
            _ => None,
        }
    }
}

static PLACEMENT_INLINE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Set the process's grow placement (the server, once at worker start).
pub fn vmm_set_grow_placement(p: VmmGrowPlacement) {
    PLACEMENT_INLINE.store(p == VmmGrowPlacement::Inline, Ordering::Relaxed);
}

/// The process's grow placement.
pub fn vmm_grow_placement() -> VmmGrowPlacement {
    if PLACEMENT_INLINE.load(Ordering::Relaxed) {
        VmmGrowPlacement::Inline
    } else {
        VmmGrowPlacement::Helper
    }
}

/// One owner-side grow: the extent mapped at `offset..offset + bytes`, the backed prefix after
/// it, the reserved range, the host wall of the driver calls plus the zero-fill enqueue (and any
/// wait for the plane's lock), and whether it ran because the mapper was behind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowEvent {
    pub offset: usize,
    pub bytes: usize,
    pub mapped: usize,
    pub reserved: usize,
    pub owner_us: u64,
    pub waited: bool,
}

static GROWS_TOTAL: AtomicU64 = AtomicU64::new(0);
static MAPPER_GROWS_TOTAL: AtomicU64 = AtomicU64::new(0);
static GROW_WAITS_TOTAL: AtomicU64 = AtomicU64::new(0);
static GROW_FAILURES_TOTAL: AtomicU64 = AtomicU64::new(0);
static RELEASED_BYTES_TOTAL: AtomicU64 = AtomicU64::new(0);
static QUARANTINED_BYTES_TOTAL: AtomicU64 = AtomicU64::new(0);

/// DAY37 addendum E3: bytes a failed unmap, release or grave reap took out of the release path
/// (kept mapped by their plane, or leaked), one `[kv-vmm] quarantined` line each. A failure never
/// leaves bytes counted as pending, so the idle wait of addendum D always ends.
pub fn vmm_quarantined_bytes() -> u64 {
    QUARANTINED_BYTES_TOTAL.load(Ordering::Relaxed)
}

fn quarantine(bytes: usize, what: &str) {
    QUARANTINED_BYTES_TOTAL.fetch_add(bytes as u64, Ordering::Relaxed);
    eprintln!("[kv-vmm] quarantined bytes={bytes}: {what}");
}

/// The helper placement's lookahead past the need (DAY37 addendum E5): one granule hides one
/// mapper grow (stage 0's busy p95 is about 11 ms per granule, a plane consumes a granule over
/// about a thousand rows); a second granule would only be retained memory.
const LOOKAHEAD_GRANULES: usize = 1;

/// The fault door of the gate (`MEMRA_KV_VMM_FAULT`, read by the server; addendum A 1.10).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VmmFaults {
    /// 1-based construction grow numbers that fail once.
    pub build: Vec<u64>,
    /// 1-based owner ensure-point grow numbers that fail once.
    pub ensure: Vec<u64>,
    /// Every mapper grow fails.
    pub mapper_all: bool,
}
impl VmmFaults {
    /// `build:<n>`, `ensure:<n>`, `mapper:all`, comma separated. `None` on any other token.
    pub fn parse(spec: &str) -> Option<Self> {
        let mut f = Self::default();
        for tok in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            match tok.split_once(':')? {
                ("build", n) => f.build.push(n.parse().ok().filter(|&n| n > 0)?),
                ("ensure", n) => f.ensure.push(n.parse().ok().filter(|&n| n > 0)?),
                ("mapper", "all") => f.mapper_all = true,
                _ => return None,
            }
        }
        Some(f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GrowClass {
    Build,
    Ensure,
    Mapper,
}

static BUILD_SEQ: AtomicU64 = AtomicU64::new(0);
static ENSURE_SEQ: AtomicU64 = AtomicU64::new(0);
static FAULTS: Mutex<Option<VmmFaults>> = Mutex::new(None);

/// Arm (or, with `None`, disarm) the fault door.
pub fn vmm_set_faults(faults: Option<VmmFaults>) {
    *FAULTS.lock().unwrap_or_else(|p| p.into_inner()) = faults;
}

fn fault_fires(class: GrowClass) -> bool {
    let mut guard = FAULTS.lock().unwrap_or_else(|p| p.into_inner());
    let Some(f) = guard.as_mut() else {
        return false;
    };
    let (list, seq) = match class {
        GrowClass::Mapper => return f.mapper_all,
        GrowClass::Build => (&mut f.build, &BUILD_SEQ),
        GrowClass::Ensure => (&mut f.ensure, &ENSURE_SEQ),
    };
    let n = seq.fetch_add(1, Ordering::Relaxed) + 1;
    match list.iter().position(|&x| x == n) {
        Some(i) => {
            list.remove(i);
            true
        }
        None => false,
    }
}

/// Process counters for `/metrics`: owner grows, mapper grows, owner grows that waited on a
/// behind mapper, grow failures, bytes released by reaps.
pub fn vmm_counters() -> (u64, u64, u64, u64, u64) {
    (
        GROWS_TOTAL.load(Ordering::Relaxed),
        MAPPER_GROWS_TOTAL.load(Ordering::Relaxed),
        GROW_WAITS_TOTAL.load(Ordering::Relaxed),
        GROW_FAILURES_TOTAL.load(Ordering::Relaxed),
        RELEASED_BYTES_TOTAL.load(Ordering::Relaxed),
    )
}

/// Whole granules covering `[0, bytes)`, capped at the reserved range.
fn backed_need(bytes: usize, granularity: usize, reserved: usize) -> usize {
    bytes
        .div_ceil(granularity)
        .saturating_mul(granularity)
        .min(reserved)
}

/// A release fence: an event recorded on the owner stream after every use of a range.
pub trait ReleaseFence: Send + Sync {
    fn is_complete(&self) -> bool;
    /// Block until complete (the grow-failure path only).
    fn wait(&self) -> Result<()>;
}
impl ReleaseFence for CudaEvent {
    fn is_complete(&self) -> bool {
        CudaEvent::is_complete(self)
    }
    fn wait(&self) -> Result<()> {
        Ok(self.synchronize()?)
    }
}

/// The driver calls an on-demand plane makes, behind one seam so the extent state machine runs
/// against a fake in CPU tests. The CUDA implementation is `CudaVmmDriver`.
trait VmmDriver: Send + Sync {
    /// `cuMemCreate` of `size` bytes; returns the handle.
    fn create(&self, size: usize) -> Result<u64>;
    /// `cuMemMap` of the handle at `address`.
    fn map(&self, address: u64, size: usize, handle: u64) -> Result<()>;
    /// `cuMemSetAccess` read-write for the owning device.
    fn set_access(&self, address: u64, size: usize) -> Result<()>;
    /// The stream-ordered zero fill on the owner stream.
    fn zero(&self, address: u64, bytes: usize) -> Result<()>;
    /// `cuMemUnmap`.
    fn unmap(&self, address: u64, size: usize) -> Result<()>;
    /// `cuMemRelease`.
    fn release(&self, handle: u64) -> Result<()>;
    /// `cuMemAddressFree` of the whole reservation.
    fn address_free(&self, base: u64, size: usize) -> Result<()>;
    /// A fence recorded on the owner stream now (`None` when it cannot be recorded).
    fn fence(&self) -> Option<Arc<dyn ReleaseFence>>;
    /// Synchronize the owner stream (only when no fence could be recorded).
    fn synchronize(&self) -> Result<()>;
    /// The mapper this plane belongs to (the context ordinal).
    fn mapper_key(&self) -> usize;
}

struct CudaVmmDriver {
    device: i32,
    stream: Arc<CudaStream>,
}
impl VmmDriver for CudaVmmDriver {
    fn create(&self, size: usize) -> Result<u64> {
        self.stream.context().bind_to_thread()?;
        let prop = properties(self.device);
        let mut handle = 0;
        // SAFETY: initialized properties, a valid handle output; size is whole granules.
        unsafe { sys::cuMemCreate(&mut handle, size, &prop, 0).result()? };
        Ok(handle)
    }
    fn map(&self, address: u64, size: usize, handle: u64) -> Result<()> {
        // SAFETY: [address, address + size) is an unmapped whole-granule span of the caller's
        // reservation and the handle owns `size` bytes.
        unsafe { sys::cuMemMap(address, size, 0, handle, 0).result()? };
        Ok(())
    }
    fn set_access(&self, address: u64, size: usize) -> Result<()> {
        let access = sys::CUmemAccessDesc {
            location: properties(self.device).location,
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        // SAFETY: the span is mapped to the owning device.
        unsafe { sys::cuMemSetAccess(address, size, &access, 1).result()? };
        Ok(())
    }
    fn zero(&self, address: u64, bytes: usize) -> Result<()> {
        // SAFETY: the span is mapped and accessible; the fill is ordered on the owner stream.
        unsafe { sys::cuMemsetD8Async(address, 0, bytes, self.stream.cu_stream()).result()? };
        Ok(())
    }
    fn unmap(&self, address: u64, size: usize) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        // SAFETY: the caller's release fence completed: no recorded work touches the span.
        unsafe { sys::cuMemUnmap(address, size).result()? };
        Ok(())
    }
    fn release(&self, handle: u64) -> Result<()> {
        // SAFETY: the handle is the caller's and has no mapping.
        unsafe { sys::cuMemRelease(handle).result()? };
        Ok(())
    }
    fn address_free(&self, base: u64, size: usize) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        // SAFETY: every extent of the reservation is unmapped and released.
        unsafe { sys::cuMemAddressFree(base, size).result()? };
        Ok(())
    }
    fn fence(&self) -> Option<Arc<dyn ReleaseFence>> {
        self.stream.context().bind_to_thread().ok()?;
        let ev = self.stream.record_event(None).ok()?;
        Some(Arc::new(ev))
    }
    fn synchronize(&self) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        Ok(self.stream.synchronize()?)
    }
    fn mapper_key(&self) -> usize {
        self.stream.context().ordinal()
    }
}

struct Extent {
    handle: sys::CUmemGenericAllocationHandle,
    offset: usize,
    bytes: usize,
    /// A requested release: the extent stays mapped and its bytes intact until this fence
    /// completes and a reap unmaps it. An ensure cancels it when the rows are needed again.
    release_after: Option<Arc<dyn ReleaseFence>>,
}

/// A plane's mutable state, under the lock the mapper shares.
struct OdState {
    /// Contiguous from offset 0, in offset order; releases only ever pend on a tail.
    extents: Vec<Extent>,
    /// The mapper maps up to here and never past it.
    want: usize,
    /// The plane dropped: the mapper skips it.
    dead: bool,
    /// The mapper's last failure, cleared by the next successful grow.
    mapper_error: Option<String>,
    /// A mapper request is queued for this plane.
    queued: bool,
}
impl OdState {
    fn end(&self) -> usize {
        self.extents.last().map_or(0, |e| e.offset + e.bytes)
    }
    fn physical_bytes(&self) -> usize {
        self.extents.iter().map(|e| e.bytes).sum()
    }
    fn live_bytes(&self) -> usize {
        self.extents
            .iter()
            .take_while(|e| e.release_after.is_none())
            .last()
            .map_or(0, |e| e.offset + e.bytes)
    }
    fn pending_release_bytes(&self) -> usize {
        self.extents
            .iter()
            .filter(|e| e.release_after.is_some())
            .map(|e| e.bytes)
            .sum()
    }
}

/// What the mapper needs of a plane, and the plane's state.
struct OdShared {
    base: u64,
    reserved: usize,
    /// The operand's length: the bytes the pooled plane would allocate.
    capacity: usize,
    granularity: usize,
    driver: Box<dyn VmmDriver>,
    state: Mutex<OdState>,
}

impl OdShared {
    fn lock(&self) -> std::sync::MutexGuard<'_, OdState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// create, map, set access, then the zero fill of the rows the operand spans, enqueued on
    /// the owner stream (the caller holds the plane's lock, so no owner launch on these rows can
    /// precede it). Every failure undoes what it did.
    fn map_extent(&self, st: &mut OdState, size: usize) -> Result<()> {
        let offset = st.end();
        let d = &self.driver;
        let handle = d.create(size)?;
        let address = self.base + offset as u64;
        if let Err(e) = d.map(address, size, handle) {
            let _ = d.release(handle);
            return Err(e);
        }
        if let Err(e) = d.set_access(address, size) {
            let _ = d.unmap(address, size);
            let _ = d.release(handle);
            return Err(e);
        }
        let zero_end = (offset + size).min(self.capacity);
        if zero_end > offset {
            if let Err(e) = d.zero(address, zero_end - offset) {
                let _ = d.unmap(address, size);
                let _ = d.release(handle);
                return Err(e);
            }
        }
        st.extents.push(Extent {
            handle,
            offset,
            bytes: size,
            release_after: None,
        });
        Ok(())
    }

    /// The tail extents whose release fence completed: unmap, release, pop. On a failure nothing
    /// stays pending (DAY37 addendum E3): a failed unmap cancels every pending release of the plane
    /// (the extents stay mapped and live, the plane keeps them until it drops); a failed release
    /// after the unmap pops the extent and leaks its handle.
    fn reap(&self, st: &mut OdState) -> usize {
        let mut released = 0;
        while let Some(last) = st.extents.last() {
            let Some(fence) = last.release_after.as_ref() else {
                break;
            };
            if !fence.is_complete() {
                break;
            }
            let (address, bytes, handle) =
                (self.base + last.offset as u64, last.bytes, last.handle);
            if let Err(e) = self.driver.unmap(address, bytes) {
                let kept = st.pending_release_bytes();
                for x in st.extents.iter_mut() {
                    x.release_after = None;
                }
                quarantine(
                    kept,
                    &format!(
                        "unmap failed ({e}); the plane's pending releases are cancelled and stay mapped"
                    ),
                );
                break;
            }
            st.extents.pop();
            match self.driver.release(handle) {
                Ok(()) => released += bytes,
                Err(e) => quarantine(
                    bytes,
                    &format!("release failed after unmap ({e}); the handle is leaked"),
                ),
            }
        }
        RELEASED_BYTES_TOTAL.fetch_add(released as u64, Ordering::Relaxed);
        released
    }
}

/// The mapper of one context: a thread that maps each handed plane up to its `want`.
struct Mapper {
    tx: std::sync::mpsc::Sender<std::sync::Weak<OdShared>>,
}

static MAPPERS: Mutex<Vec<(usize, Mapper)>> = Mutex::new(Vec::new());

fn mapper_grow(p: &OdShared) {
    let mut st = p.lock();
    st.queued = false;
    if st.dead {
        return;
    }
    let end = st.end();
    if end >= st.want {
        return;
    }
    // Never map behind a pending release (DAY37 addendum E1): pending extents are a tail at or
    // past `want`, so this cannot hold while `end < want`; the guard keeps the reap's tail order.
    if st.extents.iter().any(|e| e.release_after.is_some()) {
        return;
    }
    if fault_fires(GrowClass::Mapper) {
        GROW_FAILURES_TOTAL.fetch_add(1, Ordering::Relaxed);
        st.mapper_error = Some("injected mapper grow failure (MEMRA_KV_VMM_FAULT)".into());
        return;
    }
    let size = st.want - end;
    match p.map_extent(&mut st, size) {
        Ok(()) => {
            st.mapper_error = None;
            MAPPER_GROWS_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
        Err(e) => {
            GROW_FAILURES_TOTAL.fetch_add(1, Ordering::Relaxed);
            st.mapper_error = Some(e.to_string());
        }
    }
}

/// Hand a plane to its context's mapper (started on first use).
fn post_to_mapper(p: &Arc<OdShared>) {
    let ordinal = p.driver.mapper_key();
    let mut mappers = MAPPERS.lock().unwrap_or_else(|e| e.into_inner());
    if !mappers.iter().any(|(o, _)| *o == ordinal) {
        let (tx, rx) = std::sync::mpsc::channel::<std::sync::Weak<OdShared>>();
        let spawned = std::thread::Builder::new()
            .name(format!("memra-kv-vmm-mapper-{ordinal}"))
            .spawn(move || {
                while let Ok(weak) = rx.recv() {
                    if let Some(plane) = weak.upgrade() {
                        mapper_grow(&plane);
                    }
                }
            });
        if spawned.is_err() {
            eprintln!("[kv-vmm] mapper thread did not start; grows stay on the owner");
            return;
        }
        mappers.push((ordinal, Mapper { tx }));
    }
    if let Some((_, m)) = mappers.iter().find(|(o, _)| *o == ordinal) {
        let _ = m.tx.send(Arc::downgrade(p));
    }
}

struct OnDemand {
    shared: Arc<OdShared>,
}

impl OnDemand {
    fn physical_bytes(&self) -> usize {
        self.shared.lock().physical_bytes()
    }
    fn live_bytes(&self) -> usize {
        self.shared.lock().live_bytes()
    }
    fn ensure(&mut self, bytes: usize, class: GrowClass) -> Result<Option<GrowEvent>> {
        let p = &self.shared;
        let t0 = Instant::now();
        let mut st = p.lock();
        let need = backed_need(bytes, p.granularity, p.reserved);
        let helper = class == GrowClass::Ensure && vmm_grow_placement() == VmmGrowPlacement::Helper;
        if helper {
            let ahead = backed_need(
                need + LOOKAHEAD_GRANULES * p.granularity,
                p.granularity,
                p.reserved,
            );
            st.want = st.want.max(ahead);
        }
        // A pending release below the need or the `want` just set is cancelled: the extent is
        // still mapped, and every pending extent must stay a tail at or past `want`, or the mapper
        // would map behind it and the reap (tail first) could never release it (addendum E1).
        let live_to = need.max(st.want);
        for e in st.extents.iter_mut().filter(|e| e.offset < live_to) {
            e.release_after = None;
        }
        let end = st.end();
        if end >= need {
            if helper && end < st.want && !st.queued {
                st.queued = true;
                drop(st);
                post_to_mapper(p);
            }
            return Ok(None);
        }
        if fault_fires(class) {
            GROW_FAILURES_TOTAL.fetch_add(1, Ordering::Relaxed);
            return Err(Box::new(cudarc::driver::DriverError(
                sys::cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY,
            )));
        }
        if let Err(e) = p.map_extent(&mut st, need - end) {
            GROW_FAILURES_TOTAL.fetch_add(1, Ordering::Relaxed);
            return Err(e);
        }
        st.want = st.want.max(need);
        GROWS_TOTAL.fetch_add(1, Ordering::Relaxed);
        if helper {
            GROW_WAITS_TOTAL.fetch_add(1, Ordering::Relaxed);
            if st.end() < st.want && !st.queued {
                st.queued = true;
                drop(st);
                post_to_mapper(p);
            }
        }
        Ok(Some(GrowEvent {
            offset: end,
            bytes: need - end,
            mapped: need,
            reserved: p.reserved,
            owner_us: t0.elapsed().as_micros() as u64,
            waited: helper,
        }))
    }
    /// Schedule the release of every extent wholly past `keep_bytes`. The fence is recorded
    /// HERE, under the plane's lock and after `want` is pulled back: a mapper fill enqueued before
    /// the lock was taken is then ordered before the fence, and no mapper enqueue can follow it
    /// (the mapper never maps past `want`). A caller-recorded event would miss a fill enqueued
    /// between its record and this lock. No fence, no schedule: the extents stay mapped.
    fn request_release_beyond(&mut self, keep_bytes: usize) -> usize {
        let p = &self.shared;
        let mut st = p.lock();
        let keep = backed_need(keep_bytes, p.granularity, p.reserved);
        st.want = st.want.min(keep);
        if !st.extents.iter().any(|e| e.offset >= keep) {
            return 0;
        }
        let Some(fence) = p.driver.fence() else {
            eprintln!("[kv-vmm] trim skipped: no release fence could be recorded");
            return 0;
        };
        let mut scheduled = 0;
        for e in st.extents.iter_mut().filter(|e| e.offset >= keep) {
            // A later fence completes after the earlier one: replacing it is never earlier.
            e.release_after = Some(fence.clone());
            scheduled += e.bytes;
        }
        scheduled
    }
    fn reap(&mut self) -> usize {
        let p = &self.shared;
        let mut st = p.lock();
        p.reap(&mut st)
    }
    /// The plane dropped: marked dead under the lock (the mapper skips it from here), then every
    /// extent and the range go to the graveyard behind a fence on the owner stream. If no fence
    /// can be recorded, the stream is synchronized first; if that fails too the range is
    /// quarantined (leaked), never recycled while possibly live.
    fn bury(self) {
        let p = self.shared;
        let mut st = p.lock();
        st.dead = true;
        st.want = 0;
        let fence = match p.driver.fence() {
            Some(f) => f,
            None => {
                // Dropping the handles' bookkeeping without a release leaks the physical
                // memory and the range: quarantine, never a recycle of possibly live storage.
                if p.driver.synchronize().is_err() {
                    eprintln!("[kv-vmm] graveyard quarantined: owner synchronization failed");
                    st.extents.clear();
                    return;
                }
                match p.driver.fence() {
                    Some(f) => f,
                    None => {
                        eprintln!("[kv-vmm] graveyard quarantined: no release fence");
                        st.extents.clear();
                        return;
                    }
                }
            }
        };
        for e in st.extents.iter_mut() {
            e.release_after = Some(fence.clone());
        }
        drop(st);
        GRAVEYARD.lock().unwrap_or_else(|e| e.into_inner()).push(p);
    }
}

static GRAVEYARD: Mutex<Vec<Arc<OdShared>>> = Mutex::new(Vec::new());

/// Release every dropped on-demand plane whose release event completed: its extents, then its
/// reserved range. Returns (bytes released, graves still pending, bytes still pending). A grave
/// whose releases a failed unmap cancelled can never be reaped: it is quarantined out of the
/// graveyard (leaked, never recycled; addendum E3), so it is never counted as pending.
pub fn vmm_reap_graveyard() -> (usize, usize, usize) {
    let mut graves = GRAVEYARD.lock().unwrap_or_else(|p| p.into_inner());
    let mut released = 0;
    let mut kept = Vec::with_capacity(graves.len());
    for g in graves.drain(..) {
        let mut st = g.lock();
        released += g.reap(&mut st);
        if st.extents.is_empty() {
            drop(st);
            if let Err(e) = g.driver.address_free(g.base, g.reserved) {
                eprintln!("[kv-vmm] graveyard VA free failed: {e}");
            }
            continue;
        }
        if st.pending_release_bytes() == 0 {
            // Only a failed unmap cancels a grave's releases, and that reap already counted the
            // bytes: the grave leaves with one line and no second count.
            let bytes = st.physical_bytes();
            st.extents.clear();
            drop(st);
            eprintln!(
                "[kv-vmm] quarantined grave bytes={bytes}: its release failed; it leaves the \
                 graveyard with its range (leaked, never recycled)"
            );
            continue;
        }
        drop(st);
        kept.push(g);
    }
    let pending_bytes = kept.iter().map(|g| g.lock().pending_release_bytes()).sum();
    let pending = kept.len();
    *graves = kept;
    (released, pending, pending_bytes)
}

/// The grow-failure path only (DAY37 1.4): wait on every grave's release event, then reap.
/// This waits on events recorded at each drop, never on the owner stream as a whole.
pub fn vmm_reap_graveyard_blocking() -> Result<(usize, usize, usize)> {
    let fences: Vec<Arc<dyn ReleaseFence>> = GRAVEYARD
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .flat_map(|g| {
            g.lock()
                .extents
                .iter()
                .filter_map(|e| e.release_after.clone())
                .collect::<Vec<_>>()
        })
        .collect();
    for fence in fences {
        fence.wait()?;
    }
    Ok(vmm_reap_graveyard())
}

/// Bytes held by dropped on-demand planes whose release has not been reaped yet.
pub fn vmm_graveyard_bytes() -> usize {
    GRAVEYARD
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .map(|g| g.lock().physical_bytes())
        .sum()
}

thread_local! {
    /// The scoped ambient allocator for cache construction (`with_on_demand_kv`): the initial
    /// rows on-demand planes back, and how many on-demand planes the scope allocated.
    static ON_DEMAND_SCOPE: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

/// Construct under the on-demand allocator: every flat full-attention K/V plane a default
/// cache constructor (and the MTP draft scratch) allocates inside `f` is an on-demand plane
/// backing `initial_rows` rows. Returns `f`'s result and the number of on-demand planes.
/// Scopes do not nest: an inner scope replaces the outer for its duration and restores it.
pub fn with_on_demand_kv<R>(initial_rows: usize, f: impl FnOnce() -> R) -> (R, usize) {
    let prev = ON_DEMAND_SCOPE.with(|s| s.replace(Some((initial_rows, 0))));
    let out = f();
    let count = ON_DEMAND_SCOPE
        .with(|s| s.replace(prev))
        .map_or(0, |(_, n)| n);
    (out, count)
}

/// The ambient scope's initial rows, when construction is inside `with_on_demand_kv`.
pub fn on_demand_initial_rows() -> Option<usize> {
    ON_DEMAND_SCOPE.with(|s| s.get().map(|(rows, _)| rows))
}

/// Count one on-demand plane against the ambient scope (allocation sites call it).
pub fn note_on_demand_plane() {
    ON_DEMAND_SCOPE.with(|s| {
        if let Some((rows, n)) = s.get() {
            s.set(Some((rows, n + 1)));
        }
    });
}

pub struct KvPlane {
    operand: Option<CudaSlice<u8>>,
    mapping: Option<Mapping>,
    /// On-demand VMM storage (WP-B day 37, `MEMRA_KV_ALLOCATOR=vmm`): the full capacity is a
    /// reserved virtual range and only `ensure_mapped` extents are backed. Exclusive with
    /// `mapping`; the gate-only demote/restore arms refuse it.
    on_demand: Option<OnDemand>,
    suspended: Option<Range<usize>>,
}
impl From<CudaSlice<u8>> for KvPlane {
    fn from(operand: CudaSlice<u8>) -> Self {
        Self {
            operand: Some(operand),
            mapping: None,
            on_demand: None,
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
        self.mapping.is_some() || self.on_demand.is_some()
    }
    /// On-demand VMM storage: the operand's upper rows are backed only once `ensure_mapped`
    /// has covered them. No consumer may touch bytes past `mapped_bytes()`.
    pub fn is_on_demand(&self) -> bool {
        self.on_demand.is_some()
    }
    pub fn capacity_bytes(&self) -> usize {
        self.operand.as_ref().expect("owned operand").len()
    }
    pub fn granularity(&self) -> Option<usize> {
        self.mapping
            .as_ref()
            .map(|m| m.granularity)
            .or(self.on_demand.as_ref().map(|o| o.shared.granularity))
    }
    pub fn virtual_address(&self) -> Option<u64> {
        self.mapping
            .as_ref()
            .map(|m| m.base)
            .or(self.on_demand.as_ref().map(|o| o.shared.base))
    }
    pub fn physical_bytes(&self) -> usize {
        if let Some(o) = self.on_demand.as_ref() {
            return o.physical_bytes();
        }
        self.mapping.as_ref().map_or(self.capacity_bytes(), |m| {
            m.chunks.iter().filter(|c| c.handle.is_some()).count() * m.granularity
        })
    }
    /// Bytes from offset 0 that are mapped and not pending release: what consumers may touch.
    /// A pooled or fully mapped plane reports its whole capacity.
    pub fn mapped_bytes(&self) -> usize {
        self.on_demand.as_ref().map_or(self.capacity_bytes(), |o| {
            o.live_bytes().min(o.shared.capacity)
        })
    }
    /// The on-demand plane's reserved range; a pooled or fully mapped plane reports its
    /// physical bytes.
    pub fn reserved_bytes(&self) -> usize {
        self.on_demand
            .as_ref()
            .map_or(self.physical_bytes(), |o| o.shared.reserved)
    }
    /// On-demand planes: back `[0, bytes)` (whole granules, capped at the reserved range),
    /// cancelling any pending release of an extent inside it first. Returns the grow event,
    /// or `None` when the range was already mapped. Pooled and fully mapped planes: `None`.
    pub fn ensure_mapped(&mut self, bytes: usize) -> Result<Option<GrowEvent>> {
        match self.on_demand.as_mut() {
            Some(o) => o.ensure(bytes, GrowClass::Ensure),
            None => Ok(None),
        }
    }
    /// On-demand planes: every extent wholly past `keep_bytes` (rounded up to a granule) is
    /// released at a later `reap`, once a fence the plane records now, under its lock, completes.
    /// Returns the bytes scheduled.
    pub fn release_beyond(&mut self, keep_bytes: usize) -> usize {
        self.on_demand
            .as_mut()
            .map_or(0, |o| o.request_release_beyond(keep_bytes))
    }
    /// On-demand planes: backed bytes scheduled for release and not yet reaped (DAY37 addendum
    /// D: the run loop keeps polling while any are, so they land at idle).
    pub fn pending_release_bytes(&self) -> usize {
        self.on_demand
            .as_ref()
            .map_or(0, |o| o.shared.lock().pending_release_bytes())
    }
    /// On-demand planes: unmap and release the tail extents whose release event completed. A
    /// failure is quarantined inside (addendum E3) and leaves nothing pending.
    pub fn reap(&mut self) -> usize {
        self.on_demand.as_mut().map_or(0, OnDemand::reap)
    }
    /// Reserve `capacity` bytes of virtual range and back only `[0, initial)` (whole granules).
    /// The operand spans the whole capacity; nothing past `mapped_bytes()` may be touched.
    pub fn vmm_on_demand(stream: Arc<CudaStream>, capacity: usize, initial: usize) -> Result<Self> {
        stream.context().bind_to_thread()?;
        let device = cudarc::driver::result::device::get(stream.context().ordinal() as i32)?;
        let granularity = vmm_granularity(device)?;
        let reserved = rounded(capacity, granularity)?;
        let mut base = 0;
        // SAFETY: valid output pointer, aligned nonzero size; no requested fixed address.
        unsafe { sys::cuMemAddressReserve(&mut base, reserved, granularity, 0, 0).result()? };
        let od = OnDemand {
            shared: Arc::new(OdShared {
                base,
                reserved,
                capacity,
                granularity,
                driver: Box::new(CudaVmmDriver {
                    device,
                    stream: stream.clone(),
                }),
                state: Mutex::new(OdState {
                    extents: Vec::new(),
                    want: 0,
                    dead: false,
                    mapper_error: None,
                    queued: false,
                }),
            }),
        };
        // SAFETY (the lead-approved constructor exception of `vmm`): base is our reserved range
        // of at least `capacity` bytes. KvPlane never exports ownership, never lets a consumer
        // past `mapped_bytes()` through `ensure_mapped`'s contract, and Drop leaks the operand
        // BEFORE the range is released (never cudaFreeAsync).
        let operand = unsafe { stream.upgrade_device_ptr::<u8>(base, capacity) };
        let mut plane = Self {
            operand: Some(operand),
            mapping: None,
            on_demand: Some(od),
            suspended: None,
        };
        if initial > 0 {
            if let Some(o) = plane.on_demand.as_mut() {
                o.ensure(initial, GrowClass::Build)?;
            }
        }
        Ok(plane)
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
            on_demand: None,
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
        if self.mapping.is_some() || self.on_demand.is_some() {
            if let Some(operand) = self.operand.take() {
                operand.leak();
            }
        }
        if let Some(od) = self.on_demand.take() {
            od.bury();
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
    // ---- on-demand planes against a fake driver (WP-B day 37, DAY37 1.6 CPU clauses) ----

    use std::sync::atomic::AtomicBool;

    /// Globals (placement, faults, the graveyard) are process-wide: tests that touch them take
    /// this lock so they do not interleave.
    static GLOBALS: Mutex<()> = Mutex::new(());

    struct FakeFence(AtomicBool);
    impl ReleaseFence for FakeFence {
        fn is_complete(&self) -> bool {
            self.0.load(Ordering::SeqCst)
        }
        fn wait(&self) -> Result<()> {
            self.0.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    #[derive(Default)]
    struct FakeDriver {
        calls: Mutex<Vec<String>>,
        next_handle: AtomicU64,
        fail_map: AtomicBool,
        fail_unmap: AtomicBool,
        fail_release: AtomicBool,
        fences: Mutex<Vec<Arc<FakeFence>>>,
        key: usize,
    }
    impl FakeDriver {
        fn log(&self, c: String) {
            self.calls.lock().unwrap().push(c);
        }
    }
    impl VmmDriver for Arc<FakeDriver> {
        fn create(&self, size: usize) -> Result<u64> {
            let h = self.next_handle.fetch_add(1, Ordering::SeqCst) + 1;
            self.log(format!("create {size} -> {h}"));
            Ok(h)
        }
        fn map(&self, address: u64, size: usize, handle: u64) -> Result<()> {
            if self.fail_map.load(Ordering::SeqCst) {
                self.log(format!("map {address} {size} {handle} FAIL"));
                return Err("fake map failure".into());
            }
            self.log(format!("map {address} {size} {handle}"));
            Ok(())
        }
        fn set_access(&self, address: u64, size: usize) -> Result<()> {
            self.log(format!("access {address} {size}"));
            Ok(())
        }
        fn zero(&self, address: u64, bytes: usize) -> Result<()> {
            self.log(format!("zero {address} {bytes}"));
            Ok(())
        }
        fn unmap(&self, address: u64, size: usize) -> Result<()> {
            if self.fail_unmap.load(Ordering::SeqCst) {
                self.log(format!("unmap {address} {size} FAIL"));
                return Err("fake unmap failure".into());
            }
            self.log(format!("unmap {address} {size}"));
            Ok(())
        }
        fn release(&self, handle: u64) -> Result<()> {
            if self.fail_release.load(Ordering::SeqCst) {
                self.log(format!("release {handle} FAIL"));
                return Err("fake release failure".into());
            }
            self.log(format!("release {handle}"));
            Ok(())
        }
        fn address_free(&self, base: u64, size: usize) -> Result<()> {
            self.log(format!("free {base} {size}"));
            Ok(())
        }
        fn fence(&self) -> Option<Arc<dyn ReleaseFence>> {
            let f = Arc::new(FakeFence(AtomicBool::new(false)));
            self.fences.lock().unwrap().push(f.clone());
            Some(f)
        }
        fn synchronize(&self) -> Result<()> {
            Ok(())
        }
        fn mapper_key(&self) -> usize {
            self.key
        }
    }

    const G: usize = 2 << 20;

    fn fake_plane(capacity: usize, key: usize) -> (OnDemand, Arc<FakeDriver>) {
        let drv = Arc::new(FakeDriver {
            key,
            ..FakeDriver::default()
        });
        let od = OnDemand {
            shared: Arc::new(OdShared {
                base: 1 << 40,
                reserved: rounded(capacity, G).unwrap(),
                capacity,
                granularity: G,
                driver: Box::new(drv.clone()),
                state: Mutex::new(OdState {
                    extents: Vec::new(),
                    want: 0,
                    dead: false,
                    mapper_error: None,
                    queued: false,
                }),
            }),
        };
        (od, drv)
    }

    #[test]
    fn on_demand_pays_only_with_a_whole_granule_left_unbacked() {
        // A bounded request whose capacity is within a granule of its initial rows: pooled.
        assert!(!on_demand_pays(3 * G - 1, G + 1, G));
        assert!(!on_demand_pays(2 * G - 1, G, G));
        // Exactly one whole granule past the initial extent: on demand.
        assert!(on_demand_pays(2 * G, G, G));
        assert!(on_demand_pays(3 * G, G + 1, G));
        // An open request at the served context: on demand.
        assert!(on_demand_pays(262_144 * 1088 + 8, 1_500 * 1088 + 8, G));
        // No granularity (no VMM): never.
        assert!(!on_demand_pays(1 << 40, 0, 0));
        // The day-37 gate request: 18 planes of a few hundred KB each stay pooled.
        assert!(!on_demand_pays(2_301_440 / 9 + 64 * 1088, 2_301_440 / 9, G));
    }

    #[test]
    fn backed_need_is_whole_granules_capped_at_the_reservation() {
        assert_eq!(backed_need(0, G, 10 * G), 0);
        assert_eq!(backed_need(1, G, 10 * G), G);
        assert_eq!(backed_need(G, G, 10 * G), G);
        assert_eq!(backed_need(G + 1, G, 10 * G), 2 * G);
        assert_eq!(backed_need(100 * G, G, 10 * G), 10 * G);
    }

    #[test]
    fn inline_grows_map_one_extent_per_call_at_the_end_and_zero_the_operand_rows() {
        let _g = GLOBALS.lock().unwrap_or_else(|p| p.into_inner());
        vmm_set_grow_placement(VmmGrowPlacement::Inline);
        vmm_set_faults(None);
        // Capacity is not a whole number of granules: the last extent's fill stops at it.
        let cap = 5 * G - 100;
        let (mut od, drv) = fake_plane(cap, 901);
        let first = od.ensure(G + 7, GrowClass::Build).unwrap().unwrap();
        assert_eq!((first.offset, first.bytes, first.mapped), (0, 2 * G, 2 * G));
        assert!(!first.waited);
        // Inside the backed prefix: nothing.
        assert!(od.ensure(2 * G, GrowClass::Ensure).unwrap().is_none());
        // Past it: one new extent at the end, to the cap.
        let second = od.ensure(100 * G, GrowClass::Ensure).unwrap().unwrap();
        assert_eq!(
            (second.offset, second.bytes, second.mapped),
            (2 * G, 3 * G, 5 * G)
        );
        assert_eq!(od.physical_bytes(), 5 * G);
        assert_eq!(od.live_bytes(), 5 * G);
        let calls = drv.calls.lock().unwrap().clone();
        let base = 1u64 << 40;
        assert_eq!(
            calls,
            vec![
                format!("create {} -> 1", 2 * G),
                format!("map {base} {} 1", 2 * G),
                format!("access {base} {}", 2 * G),
                format!("zero {base} {}", 2 * G),
                format!("create {} -> 2", 3 * G),
                format!("map {} {} 2", base + 2 * G as u64, 3 * G),
                format!("access {} {}", base + 2 * G as u64, 3 * G),
                format!("zero {} {}", base + 2 * G as u64, 3 * G - 100),
            ]
        );
    }

    #[test]
    fn a_release_waits_for_its_fence_and_a_resume_cancels_it() {
        let _g = GLOBALS.lock().unwrap_or_else(|p| p.into_inner());
        vmm_set_grow_placement(VmmGrowPlacement::Inline);
        vmm_set_faults(None);
        let (mut od, drv) = fake_plane(10 * G, 902);
        od.ensure(2 * G, GrowClass::Build).unwrap();
        od.ensure(4 * G, GrowClass::Ensure).unwrap();
        od.ensure(6 * G, GrowClass::Ensure).unwrap();
        // Keep 2 granules plus a byte: the extent at [2G, 4G) is partly needed and stays; the
        // one at [4G, 6G) is wholly past and is scheduled.
        assert_eq!(od.request_release_beyond(3 * G + 1), 2 * G);
        let f = drv.fences.lock().unwrap().last().unwrap().clone();
        assert_eq!(od.live_bytes(), 4 * G);
        assert_eq!(od.physical_bytes(), 6 * G);
        // Fence incomplete: the reap releases nothing, and the bytes read as pending.
        assert_eq!(od.reap(), 0);
        assert_eq!(od.shared.lock().pending_release_bytes(), 2 * G);
        f.0.store(true, Ordering::SeqCst);
        assert_eq!(od.reap(), 2 * G);
        assert_eq!(od.physical_bytes(), 4 * G);
        assert!(
            drv.calls
                .lock()
                .unwrap()
                .iter()
                .any(|c| c.starts_with("unmap"))
        );
        // Schedule again, then a resume needs the rows before the fence completes: cancelled,
        // and the completed fence then releases nothing.
        assert_eq!(od.request_release_beyond(G), 2 * G);
        let f2 = drv.fences.lock().unwrap().last().unwrap().clone();
        assert!(od.ensure(4 * G, GrowClass::Ensure).unwrap().is_none());
        assert_eq!(od.live_bytes(), 4 * G);
        f2.0.store(true, Ordering::SeqCst);
        assert_eq!(od.reap(), 0);
        assert_eq!(od.physical_bytes(), 4 * G);
    }

    /// DAY37 addendum E3: a failed unmap cancels the plane's pending releases (they stay mapped
    /// and live) and a failed release after the unmap pops the extent; neither leaves bytes
    /// pending, and both count as quarantined.
    #[test]
    fn a_failed_reap_is_quarantined_and_leaves_nothing_pending() {
        let _g = GLOBALS.lock().unwrap_or_else(|p| p.into_inner());
        vmm_set_grow_placement(VmmGrowPlacement::Inline);
        vmm_set_faults(None);
        let (mut od, drv) = fake_plane(10 * G, 909);
        od.ensure(2 * G, GrowClass::Build).unwrap();
        od.ensure(4 * G, GrowClass::Ensure).unwrap();
        od.ensure(6 * G, GrowClass::Ensure).unwrap();
        assert_eq!(od.request_release_beyond(2 * G), 4 * G);
        for f in drv.fences.lock().unwrap().iter() {
            f.0.store(true, Ordering::SeqCst);
        }
        let q0 = vmm_quarantined_bytes();
        drv.fail_unmap.store(true, Ordering::SeqCst);
        assert_eq!(od.reap(), 0);
        assert_eq!(od.shared.lock().pending_release_bytes(), 0);
        assert_eq!(od.physical_bytes(), 6 * G);
        assert_eq!(
            od.live_bytes(),
            6 * G,
            "the cancelled extents stay mapped and live"
        );
        assert_eq!(vmm_quarantined_bytes() - q0, (4 * G) as u64);
        drv.fail_unmap.store(false, Ordering::SeqCst);
        // A release that fails after its unmap: the extent leaves the plane, its handle leaks.
        assert_eq!(od.request_release_beyond(4 * G), 2 * G);
        for f in drv.fences.lock().unwrap().iter() {
            f.0.store(true, Ordering::SeqCst);
        }
        drv.fail_release.store(true, Ordering::SeqCst);
        let q1 = vmm_quarantined_bytes();
        assert_eq!(od.reap(), 0);
        assert_eq!(od.shared.lock().pending_release_bytes(), 0);
        assert_eq!(od.physical_bytes(), 4 * G);
        assert_eq!(vmm_quarantined_bytes() - q1, (2 * G) as u64);
        drv.fail_release.store(false, Ordering::SeqCst);
        // The plane still grows at its end.
        assert!(od.ensure(6 * G, GrowClass::Ensure).unwrap().is_some());
        assert_eq!(od.live_bytes(), 6 * G);
    }

    /// DAY37 addendum E3: a grave whose reap fails leaves the graveyard (quarantined), so it is
    /// never counted as pending.
    #[test]
    fn a_grave_whose_reap_fails_is_quarantined_out_of_the_graveyard() {
        let _g = GLOBALS.lock().unwrap_or_else(|p| p.into_inner());
        vmm_set_grow_placement(VmmGrowPlacement::Inline);
        vmm_set_faults(None);
        let _ = vmm_reap_graveyard_blocking();
        let (mut od, drv) = fake_plane(8 * G, 910);
        od.ensure(3 * G, GrowClass::Build).unwrap();
        let shared = od.shared.clone();
        od.bury();
        for f in drv.fences.lock().unwrap().iter() {
            f.0.store(true, Ordering::SeqCst);
        }
        drv.fail_unmap.store(true, Ordering::SeqCst);
        let q0 = vmm_quarantined_bytes();
        let (released, pending, pending_bytes) = vmm_reap_graveyard();
        assert_eq!((released, pending, pending_bytes), (0, 0, 0));
        assert_eq!(vmm_graveyard_bytes(), 0);
        assert!(shared.lock().extents.is_empty());
        // Counted once, by the reap whose unmap failed; the grave leaves without a second count.
        assert_eq!(vmm_quarantined_bytes() - q0, (3 * G) as u64);
        assert!(
            !drv.calls
                .lock()
                .unwrap()
                .iter()
                .any(|c| c.starts_with("free")),
            "a quarantined range is never freed"
        );
        drv.fail_unmap.store(false, Ordering::SeqCst);
    }

    #[test]
    fn a_failed_map_undoes_its_create_and_leaves_no_extent() {
        let _g = GLOBALS.lock().unwrap_or_else(|p| p.into_inner());
        vmm_set_grow_placement(VmmGrowPlacement::Inline);
        vmm_set_faults(None);
        let (mut od, drv) = fake_plane(4 * G, 903);
        drv.fail_map.store(true, Ordering::SeqCst);
        assert!(od.ensure(G, GrowClass::Ensure).is_err());
        assert_eq!(od.physical_bytes(), 0);
        let calls = drv.calls.lock().unwrap().clone();
        assert_eq!(calls.last().unwrap(), "release 1");
        drv.fail_map.store(false, Ordering::SeqCst);
        assert!(od.ensure(G, GrowClass::Ensure).unwrap().is_some());
        assert_eq!(od.physical_bytes(), G);
    }

    #[test]
    fn the_fault_door_fails_the_named_grow_once_before_any_driver_call() {
        let _g = GLOBALS.lock().unwrap_or_else(|p| p.into_inner());
        vmm_set_grow_placement(VmmGrowPlacement::Inline);
        assert_eq!(
            VmmFaults::parse("build:2, ensure:1,mapper:all"),
            Some(VmmFaults {
                build: vec![2],
                ensure: vec![1],
                mapper_all: true
            })
        );
        assert_eq!(VmmFaults::parse("grow:1"), None);
        assert_eq!(VmmFaults::parse("ensure:0"), None);
        assert_eq!(VmmFaults::parse("ensure:x"), None);
        BUILD_SEQ.store(0, Ordering::SeqCst);
        ENSURE_SEQ.store(0, Ordering::SeqCst);
        vmm_set_faults(Some(VmmFaults {
            build: vec![],
            ensure: vec![1],
            mapper_all: false,
        }));
        let (mut od, drv) = fake_plane(4 * G, 904);
        let err = od.ensure(G, GrowClass::Ensure).unwrap_err().to_string();
        assert!(err.contains("OUT_OF_MEMORY"), "{err}");
        assert!(drv.calls.lock().unwrap().is_empty());
        // Once only: the next ensure grows.
        assert!(od.ensure(G, GrowClass::Ensure).unwrap().is_some());
        vmm_set_faults(None);
    }

    #[test]
    fn helper_placement_premaps_one_granule_past_the_need_and_the_owner_grows_when_behind() {
        let _g = GLOBALS.lock().unwrap_or_else(|p| p.into_inner());
        vmm_set_grow_placement(VmmGrowPlacement::Helper);
        vmm_set_faults(None);
        let (mut od, _drv) = fake_plane(20 * G, 905);
        // Nothing backed: the owner grows inline (waited) to the need, and the mapper is
        // handed the rest up to need + 1 granule (addendum E5).
        let ev = od.ensure(G, GrowClass::Ensure).unwrap().unwrap();
        assert!(ev.waited);
        assert_eq!(ev.mapped, G);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while od.physical_bytes() < 2 * G {
            assert!(
                std::time::Instant::now() < deadline,
                "the mapper never caught up"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(od.physical_bytes(), 2 * G, "one granule ahead, not two");
        // Within the pre-mapped headroom: no owner grow.
        assert!(od.ensure(2 * G, GrowClass::Ensure).unwrap().is_none());
        // A park trim pulls `want` back so the mapper cannot map past it after the fence.
        od.request_release_beyond(G);
        assert_eq!(od.shared.lock().want, G);
        vmm_set_grow_placement(VmmGrowPlacement::Inline);
    }

    /// DAY37 addendum E1: a resume whose lookahead reaches past a pending tail must not map after
    /// it. The mapper would append a live extent behind the pending one, the reap (tail first)
    /// could never release it, and its bytes would read as pending for the plane's life. Red on
    /// r3 (lookahead two granules, cancellation below the need only): extents `(5, 1, pending),
    /// (6, 1, live)` in granules. The geometry reads `LOOKAHEAD_GRANULES`, so the invariant is
    /// checked at the shipped lookahead.
    #[test]
    fn a_lookahead_past_a_pending_tail_cancels_it_and_never_maps_behind_it() {
        let _g = GLOBALS.lock().unwrap_or_else(|p| p.into_inner());
        vmm_set_faults(None);
        vmm_set_grow_placement(VmmGrowPlacement::Inline);
        let (mut od, drv) = fake_plane(20 * G, 908);
        // One-granule extents [0, 6G), inline.
        od.ensure(G, GrowClass::Build).unwrap();
        for k in 2..=6 {
            od.ensure(k * G, GrowClass::Ensure).unwrap();
        }
        // The park trim: keep 5 granules; the extent at [5G, 6G) pends.
        assert_eq!(od.request_release_beyond(5 * G), G);
        let f = drv.fences.lock().unwrap().last().unwrap().clone();
        // The resume needs 5 granules; its lookahead reaches past the pending tail's start.
        vmm_set_grow_placement(VmmGrowPlacement::Helper);
        assert!(od.ensure(4 * G + 1, GrowClass::Ensure).unwrap().is_none());
        let settle_to = (5 + LOOKAHEAD_GRANULES).max(6) * G;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while od.physical_bytes() < settle_to {
            assert!(
                std::time::Instant::now() < deadline,
                "the mapper never caught up"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        f.0.store(true, Ordering::SeqCst);
        let _ = od.reap();
        {
            let st = od.shared.lock();
            if let Some(i) = st.extents.iter().position(|e| e.release_after.is_some()) {
                assert!(
                    st.extents[i..].iter().all(|e| e.release_after.is_some()),
                    "a live extent sits behind a pending one: {:?}",
                    st.extents
                        .iter()
                        .map(|e| (e.offset / G, e.bytes / G, e.release_after.is_some()))
                        .collect::<Vec<_>>()
                );
            }
            assert_eq!(
                st.pending_release_bytes(),
                0,
                "a release stuck behind a live extent"
            );
            assert!(st.extents.iter().all(|e| e.offset < st.want.max(5 * G)));
        }
        assert_eq!(od.live_bytes(), od.physical_bytes());
        vmm_set_grow_placement(VmmGrowPlacement::Inline);
    }

    #[test]
    fn a_mapper_fault_leaves_the_owner_behind_and_growing_inline() {
        let _g = GLOBALS.lock().unwrap_or_else(|p| p.into_inner());
        vmm_set_grow_placement(VmmGrowPlacement::Helper);
        vmm_set_faults(Some(VmmFaults {
            build: vec![],
            ensure: vec![],
            mapper_all: true,
        }));
        let (mut od, _drv) = fake_plane(20 * G, 906);
        let ev = od.ensure(G, GrowClass::Ensure).unwrap().unwrap();
        assert!(ev.waited);
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(od.physical_bytes(), G, "the faulted mapper mapped nothing");
        assert!(od.shared.lock().mapper_error.is_some());
        let ev = od.ensure(2 * G, GrowClass::Ensure).unwrap().unwrap();
        assert!(ev.waited);
        assert_eq!(ev.mapped, 2 * G);
        vmm_set_faults(None);
        vmm_set_grow_placement(VmmGrowPlacement::Inline);
    }

    #[test]
    fn a_dropped_plane_goes_to_the_graveyard_and_frees_its_range_after_its_fence() {
        let _g = GLOBALS.lock().unwrap_or_else(|p| p.into_inner());
        vmm_set_grow_placement(VmmGrowPlacement::Inline);
        vmm_set_faults(None);
        // Drain anything an earlier test left.
        let _ = vmm_reap_graveyard_blocking();
        let (mut od, drv) = fake_plane(8 * G, 907);
        od.ensure(3 * G, GrowClass::Build).unwrap();
        od.ensure(5 * G, GrowClass::Ensure).unwrap();
        let shared = od.shared.clone();
        od.bury();
        assert!(shared.lock().dead);
        assert_eq!(shared.lock().want, 0);
        assert!(vmm_graveyard_bytes() >= 5 * G);
        // Fence incomplete: nothing released, the range is not freed.
        let (released, pending, _) = vmm_reap_graveyard();
        assert_eq!(released, 0);
        assert!(pending >= 1);
        assert!(
            !drv.calls
                .lock()
                .unwrap()
                .iter()
                .any(|c| c.starts_with("free"))
        );
        for f in drv.fences.lock().unwrap().iter() {
            f.0.store(true, Ordering::SeqCst);
        }
        let (released, _, _) = vmm_reap_graveyard();
        assert_eq!(released, 5 * G);
        let calls = drv.calls.lock().unwrap().clone();
        let n = calls.len();
        // The tail extent first, then the first, then the range.
        assert_eq!(
            calls[n - 5],
            format!("unmap {} {}", (1u64 << 40) + 3 * G as u64, 2 * G)
        );
        assert_eq!(calls[n - 1], format!("free {} {}", 1u64 << 40, 8 * G));
    }

    #[test]
    fn the_ambient_scope_counts_its_planes_and_restores_the_outer_scope() {
        assert_eq!(on_demand_initial_rows(), None);
        let ((), n) = with_on_demand_kv(100, || {
            assert_eq!(on_demand_initial_rows(), Some(100));
            note_on_demand_plane();
            let ((), inner) = with_on_demand_kv(7, || {
                assert_eq!(on_demand_initial_rows(), Some(7));
                note_on_demand_plane();
                note_on_demand_plane();
            });
            assert_eq!(inner, 2);
            assert_eq!(on_demand_initial_rows(), Some(100));
            note_on_demand_plane();
        });
        assert_eq!(n, 2);
        assert_eq!(on_demand_initial_rows(), None);
        // Outside a scope a note counts nothing.
        note_on_demand_plane();
        assert_eq!(on_demand_initial_rows(), None);
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
