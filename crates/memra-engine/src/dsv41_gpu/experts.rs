//! The routed experts off the card: every trunk layer's routed experts in pinned host memory, a
//! device LRU cache of expert slots per rank, and asynchronous H2D copies of the experts the
//! router picks.
//!
//! A slot holds one expert as the GEMMs read it (the checkpoint's MXFP4 `w1`, `w3`, `w2` codes and
//! their ue8m0 scales repacked into the CUTLASS GEMMs' layout, [`super::sf_pack`], packed back to
//! back), and the grouped GEMMs read a slot through the same pointer table they read a streamed
//! layer's experts through. The residency moves where the bytes
//! live, never which bytes a kernel reads, so the cached route writes the streamed route's numbers.
//!
//! The host store is anonymous memory filled from the checkpoint mmap and then page-locked: a
//! read-only file mapping cannot be page-locked on the target driver
//! (`CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED` is 0 there), so pinning the page cache
//! in place is not available. One region per layer, `n_experts` slots each.
//!
//! The device cache is a set of slabs of [`SLAB_SLOTS`] slots. A slice's experts are resolved
//! before its MoE runs: hits stamp their slot, misses take the least recently used slots outside
//! the slice, and their copies queue on the cache's copy stream behind a fence on the compute
//! stream; the compute stream waits for the copies before the expert GEMMs. Slabs are released
//! (least recently used first, never below the floor of one slice's experts) when a request needs
//! the memory, and grown back when it is free again.

use super::{Buf, Geo, Ops};
use cudarc::driver::{CudaContext, CudaEvent, CudaStream};
use memra_gguf::safetensors::StModel;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Expert slots per device slab.
pub const SLAB_SLOTS: usize = 32;

/// Where the six pieces of one expert sit inside a slot (and inside its host copy): `w1`, its
/// scale, `w3`, its scale, `w2`, its scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub off: [usize; 6],
    pub len: [usize; 6],
    pub bytes: usize,
}

impl Layout {
    pub fn of(g: &Geo) -> Self {
        let (d, i) = (g.dim, g.moe_inter);
        // w1/w3: [inter][dim/2] codes, [inter][dim/32] scales; w2: [dim][inter/2], [dim][inter/32]
        let len = [
            i * d / 2,
            i * d / 32,
            i * d / 2,
            i * d / 32,
            d * i / 2,
            d * i / 32,
        ];
        let mut off = [0usize; 6];
        let mut at = 0usize;
        for k in 0..6 {
            off[k] = at;
            at += len[k].next_multiple_of(256);
        }
        Self {
            off,
            len,
            bytes: at,
        }
    }

    /// The rows and inputs of scale piece `k` (1, 3, 5): its codes are one per row and 32 inputs.
    fn scale_dims(g: &Geo, k: usize) -> (usize, usize) {
        let (d, i) = (g.dim, g.moe_inter);
        if k < 4 { (i, d) } else { (d, i) }
    }

    /// The checkpoint tensors of piece `k` of expert `e` in layer `l`, with their dtype and shape.
    fn piece(&self, g: &Geo, l: usize, e: usize, k: usize) -> (String, &'static str, [u64; 2]) {
        let (d, i) = (g.dim as u64, g.moe_inter as u64);
        let w = ["w1", "w1", "w3", "w3", "w2", "w2"][k];
        let base = format!("layers.{l}.ffn.experts.{e}.{w}");
        let (n, kk) = if k < 4 { (i, d) } else { (d, i) };
        if k.is_multiple_of(2) {
            (format!("{base}.weight"), "I8", [n, kk / 2])
        } else {
            (format!("{base}.scale"), "F8_E8M0", [n, kk / 32])
        }
    }
}

/// `MemAvailable` from `/proc/meminfo`, in bytes.
pub fn host_available_bytes() -> Option<u64> {
    let txt = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = txt.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Every trunk layer's routed experts, page-locked in host memory (portable to every context).
pub struct HostExperts {
    pub layout: Layout,
    pub n_layers: usize,
    pub n_experts: usize,
    regions: Vec<memmap2::MmapMut>,
    ctx: Arc<CudaContext>,
    /// page-locked bytes
    pub bytes: usize,
    /// wall seconds to fill and lock the store
    pub fill_s: f64,
}

// The regions are only read after construction (by the copy engine); the mappings live as long
// as the store.
unsafe impl Send for HostExperts {}
unsafe impl Sync for HostExperts {}

impl HostExperts {
    /// Copy the routed experts of trunk layers `0..n_layers` out of the checkpoint mmap into one
    /// anonymous region per layer on `threads` threads, and page-lock each region. Refuses when
    /// the host's available memory cannot hold the store.
    pub fn fill(
        stm: &StModel,
        g: &Geo,
        n_layers: usize,
        ctx: &Arc<CudaContext>,
        threads: usize,
    ) -> Result<Self, String> {
        let layout = Layout::of(g);
        // the scale pieces are repacked in place: the layout must not grow them (rows a multiple
        // of 128 and K blocks of 4, true of every V4.1 expert shape)
        for k in [1, 3, 5] {
            let (n, kk) = Layout::scale_dims(g, k);
            if super::sf_bytes(n, kk) != layout.len[k] {
                return Err(format!(
                    "expert scale piece {k} ({n} x {kk}) does not fit the GEMM scale layout in place"
                ));
            }
        }
        let ne = g.n_experts;
        let per_layer = ne * layout.bytes;
        let bytes = per_layer * n_layers;
        if let Some(avail) = host_available_bytes()
            && avail < bytes as u64
        {
            return Err(format!(
                "the pinned expert store needs {bytes} bytes of host memory and {avail} are available"
            ));
        }
        // shapes first: a checkpoint that does not match refuses before any memory is touched
        for l in 0..n_layers {
            for e in 0..ne {
                for k in 0..6 {
                    let (name, dtype, shape) = layout.piece(g, l, e, k);
                    let info = stm
                        .info(&name)
                        .ok_or_else(|| format!("missing tensor {name}"))?;
                    if info.dtype != dtype || info.shape != shape {
                        return Err(format!(
                            "{name}: {} {:?}, want {dtype} {shape:?}",
                            info.dtype, info.shape
                        ));
                    }
                }
            }
        }
        let t0 = Instant::now();
        let mut regions = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            regions.push(
                memmap2::MmapMut::map_anon(per_layer)
                    .map_err(|e| format!("expert store region of {per_layer} bytes: {e}"))?,
            );
        }
        let work = std::sync::Mutex::new(regions.iter_mut().enumerate());
        let threads = threads.clamp(1, n_layers.max(1));
        let res: Result<(), String> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|_| {
                    sc.spawn(|| -> Result<(), String> {
                        ctx.bind_to_thread().map_err(|e| format!("{e:?}"))?;
                        loop {
                            let next = work.lock().map_err(|_| "poisoned")?.next();
                            let Some((l, region)) = next else {
                                return Ok(());
                            };
                            for e in 0..ne {
                                let slot = &mut region[e * layout.bytes..(e + 1) * layout.bytes];
                                for k in 0..6 {
                                    let (name, _, _) = layout.piece(g, l, e, k);
                                    let (_, src) =
                                        stm.raw(&name).ok_or_else(|| format!("missing {name}"))?;
                                    if src.len() != layout.len[k] {
                                        return Err(format!("{name}: {} bytes", src.len()));
                                    }
                                    let dst = &mut slot[layout.off[k]..layout.off[k] + src.len()];
                                    if k % 2 == 1 {
                                        let (n, kk) = Layout::scale_dims(g, k);
                                        let kbn = kk / 32;
                                        for r in 0..n {
                                            for b in 0..kbn {
                                                dst[super::sf_off(r, b, kbn)] = src[r * kbn + b];
                                            }
                                        }
                                    } else {
                                        dst.copy_from_slice(src);
                                    }
                                }
                            }
                            let rc = unsafe {
                                cudarc::driver::sys::cuMemHostRegister_v2(
                                    region.as_mut_ptr() as *mut std::ffi::c_void,
                                    region.len(),
                                    cudarc::driver::sys::CU_MEMHOSTREGISTER_PORTABLE,
                                )
                            };
                            if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                                return Err(format!("page-lock layer {l} experts: {rc:?}"));
                            }
                        }
                    })
                })
                .collect();
            for h in hs {
                h.join().map_err(|_| "expert store thread panicked")??;
            }
            Ok(())
        });
        let store = Self {
            layout,
            n_layers,
            n_experts: ne,
            regions,
            ctx: ctx.clone(),
            bytes,
            fill_s: t0.elapsed().as_secs_f64(),
        };
        // a failure part-way leaves some regions locked; dropping the store unlocks what it can
        res?;
        Ok(store)
    }

    /// The host copy of expert `e` of layer `l`: one slot's bytes in [`Layout`] order.
    pub fn expert(&self, l: usize, e: usize) -> &[u8] {
        let b = self.layout.bytes;
        &self.regions[l][e * b..(e + 1) * b]
    }
}

impl Drop for HostExperts {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        for r in &mut self.regions {
            // regions a failed fill never locked answer "not registered"; nothing else to do
            unsafe {
                cudarc::driver::sys::cuMemHostUnregister(r.as_mut_ptr() as *mut std::ffi::c_void);
            }
        }
    }
}

/// What the cache did, summed over its life.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub h2d_bytes: u64,
    pub evictions: u64,
    pub slabs_released: u64,
    pub slabs_grown: u64,
}

impl CacheStats {
    pub fn add(&mut self, o: &CacheStats) {
        self.lookups += o.lookups;
        self.hits += o.hits;
        self.misses += o.misses;
        self.h2d_bytes += o.h2d_bytes;
        self.evictions += o.evictions;
        self.slabs_released += o.slabs_released;
        self.slabs_grown += o.slabs_grown;
    }
}

/// One rank's device cache of routed-expert slots.
pub struct ExpertCache {
    host: Arc<HostExperts>,
    slot_bytes: usize,
    slabs: Vec<Option<Buf>>,
    /// per slot: 0 empty, else `layer * n_experts + expert + 1`
    key: Vec<u32>,
    /// per slot: the resolve that last used it
    stamp: Vec<u64>,
    slot_of: HashMap<u32, usize>,
    clock: u64,
    /// the fewest slots the cache keeps: one slice's experts on this rank
    pub floor_slots: usize,
    copy: Arc<CudaStream>,
    fence: CudaEvent,
    ready: CudaEvent,
    pub stats: CacheStats,
}

impl ExpertCache {
    /// An empty cache on `ops`'s card whose slots hold `host`'s experts; `floor_slots` is the
    /// most experts one slice can route to on this rank.
    pub fn new(ops: &Ops, host: Arc<HostExperts>, floor_slots: usize) -> Result<Self, String> {
        let ctx = ops.st.context().clone();
        let de = |e: cudarc::driver::DriverError| format!("expert cache: {e:?}");
        let copy = ctx.new_stream().map_err(de)?;
        let fence = ctx.new_event(None).map_err(de)?;
        let ready = ctx.new_event(None).map_err(de)?;
        Ok(Self {
            slot_bytes: host.layout.bytes,
            host,
            slabs: Vec::new(),
            key: Vec::new(),
            stamp: Vec::new(),
            slot_of: HashMap::new(),
            clock: 0,
            floor_slots,
            copy,
            fence,
            ready,
            stats: CacheStats::default(),
        })
    }

    pub fn slab_bytes(&self) -> usize {
        SLAB_SLOTS * self.slot_bytes
    }

    /// Slots on the card now.
    pub fn slots(&self) -> usize {
        self.slabs.iter().filter(|s| s.is_some()).count() * SLAB_SLOTS
    }

    pub fn device_bytes(&self) -> usize {
        self.slots() * self.slot_bytes
    }

    /// Experts resident now.
    pub fn resident(&self) -> usize {
        self.slot_of.len()
    }

    /// The fewest slabs that hold the floor.
    pub fn floor_slabs(&self) -> usize {
        self.floor_slots.div_ceil(SLAB_SLOTS)
    }

    /// Add slabs while `budget` bytes allow, up to `max_slots` slots. Returns the bytes added.
    pub fn grow(&mut self, ops: &Ops, budget: usize, max_slots: usize) -> Result<usize, String> {
        let sb = self.slab_bytes();
        let mut added = 0usize;
        while added + sb <= budget && self.slots() + SLAB_SLOTS <= max_slots.max(self.floor_slots) {
            let buf = ops.buf(sb)?;
            let i = match self.slabs.iter().position(|s| s.is_none()) {
                Some(i) => {
                    self.slabs[i] = Some(buf);
                    i
                }
                None => {
                    self.slabs.push(Some(buf));
                    self.key.extend(std::iter::repeat_n(0, SLAB_SLOTS));
                    self.stamp.extend(std::iter::repeat_n(0, SLAB_SLOTS));
                    self.slabs.len() - 1
                }
            };
            for s in i * SLAB_SLOTS..(i + 1) * SLAB_SLOTS {
                self.key[s] = 0;
                self.stamp[s] = 0;
            }
            added += sb;
            self.stats.slabs_grown += 1;
        }
        ops.sync()?;
        Ok(added)
    }

    /// Release slabs, least recently used first, until `want` bytes are freed or only the floor
    /// is left. Returns the bytes released.
    pub fn release(&mut self, ops: &Ops, want: usize) -> Result<usize, String> {
        // nothing may still read or write a slab that goes
        ops.sync()?;
        self.copy
            .synchronize()
            .map_err(|e| format!("expert cache: {e:?}"))?;
        let sb = self.slab_bytes();
        let mut freed = 0usize;
        while freed < want && self.slots() > self.floor_slabs() * SLAB_SLOTS {
            let victim = (0..self.slabs.len())
                .filter(|&i| self.slabs[i].is_some())
                .min_by_key(|&i| {
                    self.stamp[i * SLAB_SLOTS..(i + 1) * SLAB_SLOTS]
                        .iter()
                        .max()
                        .copied()
                        .unwrap_or(0)
                })
                .ok_or("expert cache: no slab to release")?;
            for s in victim * SLAB_SLOTS..(victim + 1) * SLAB_SLOTS {
                if self.key[s] != 0 {
                    self.slot_of.remove(&self.key[s]);
                    self.key[s] = 0;
                }
                self.stamp[s] = 0;
            }
            self.slabs[victim] = None;
            freed += sb;
            self.stats.slabs_released += 1;
        }
        ops.sync()?;
        Ok(freed)
    }

    fn base(&self, ops: &Ops, s: usize) -> u64 {
        let slab = self.slabs[s / SLAB_SLOTS]
            .as_ref()
            .expect("a live slot sits in a live slab");
        slab.p(&ops.st).addr() + ((s % SLAB_SLOTS) * self.slot_bytes) as u64
    }

    /// Make experts `experts` of layer `layer` resident and return each one's slot address (in
    /// the given order). Misses copy on the cache's stream after everything already queued on
    /// `ops`'s stream; `ops`'s stream waits for the copies before anything queued after this.
    pub fn resolve(
        &mut self,
        ops: &Ops,
        layer: usize,
        experts: &[usize],
    ) -> Result<Vec<u64>, String> {
        let ne = self.host.n_experts;
        if layer >= self.host.n_layers {
            return Err(format!(
                "expert cache: layer {layer} of {}",
                self.host.n_layers
            ));
        }
        self.clock += 1;
        let clock = self.clock;
        let mut slot_at = vec![usize::MAX; experts.len()];
        let mut miss = Vec::new();
        for (i, &e) in experts.iter().enumerate() {
            if e >= ne {
                return Err(format!("expert cache: expert {e} of {ne}"));
            }
            let key = (layer * ne + e + 1) as u32;
            match self.slot_of.get(&key) {
                Some(&s) => {
                    self.stamp[s] = clock;
                    slot_at[i] = s;
                }
                None => miss.push(i),
            }
        }
        self.stats.lookups += experts.len() as u64;
        self.stats.hits += (experts.len() - miss.len()) as u64;
        if !miss.is_empty() {
            // victims: empty slots first, then the least recently used, never one this call uses
            let mut cand: Vec<(u64, usize)> = (0..self.key.len())
                .filter(|&s| self.slabs[s / SLAB_SLOTS].is_some() && self.stamp[s] < clock)
                .map(|s| {
                    (
                        if self.key[s] == 0 {
                            0
                        } else {
                            self.stamp[s] + 1
                        },
                        s,
                    )
                })
                .collect();
            if cand.len() < miss.len() {
                return Err(format!(
                    "expert cache: {} slots cannot hold the {} experts one slice routes to",
                    self.slots(),
                    experts.len()
                ));
            }
            if cand.len() > miss.len() {
                cand.select_nth_unstable(miss.len() - 1);
                cand.truncate(miss.len());
            }
            let de = |e: cudarc::driver::DriverError| format!("expert cache: {e:?}");
            self.fence.record(&ops.st).map_err(de)?;
            self.copy.wait(&self.fence).map_err(de)?;
            let cs = self.copy.cu_stream();
            let mut failed = None;
            for (&i, &(_, s)) in miss.iter().zip(&cand) {
                let src = self.host.expert(layer, experts[i]);
                let rc = unsafe {
                    cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                        self.base(ops, s),
                        src.as_ptr() as *const std::ffi::c_void,
                        src.len(),
                        cs,
                    )
                };
                if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    failed = Some(rc);
                    break;
                }
                // the copy is queued: from here the slot holds this expert (a failed copy
                // leaves the slot mapped to what it held)
                if self.key[s] != 0 {
                    self.slot_of.remove(&self.key[s]);
                    self.stats.evictions += 1;
                }
                let key = (layer * ne + experts[i] + 1) as u32;
                self.key[s] = key;
                self.stamp[s] = clock;
                self.slot_of.insert(key, s);
                slot_at[i] = s;
                self.stats.misses += 1;
                self.stats.h2d_bytes += src.len() as u64;
            }
            // the copies queued before a failure are mapped: the stream waits for them either way
            self.ready.record(&self.copy).map_err(de)?;
            ops.st.wait(&self.ready).map_err(de)?;
            if let Some(rc) = failed {
                return Err(format!("expert cache H2D: {rc:?}"));
            }
        }
        Ok(slot_at.iter().map(|&s| self.base(ops, s)).collect())
    }
}

/// Effective free bytes on `ops`'s card and its total: the driver's free plus what the stream-
/// ordered pool holds mapped and unused (the engine pins the pool's release threshold, so freed
/// blocks stay mapped where `cuMemGetInfo` cannot see them). Drains the stream first.
///
/// The free bytes are the raw `cuMemGetInfo` reading. On a unified-memory device that is the
/// host's `MemFree` and leaves out page cache; memra#1307 moved the readers it named off it and
/// left this route's alone (this one and `decode.rs` `device_mem`). The route loads on one device
/// and only its size keeps it off a GB10 today. If it loads there, a level read from this needs
/// `unified_memory::allocatable_mem_info` and a delta between two calls needs
/// `unified_memory::delta_free_bytes` (`docs/decisions/UNIFIED-MEMORY-READERS.md`).
pub fn device_memory(ops: &Ops) -> Result<(u64, u64), String> {
    use cudarc::driver::sys;
    ops.sync()?;
    let ctx = ops.st.context();
    let (free, total) = ctx
        .mem_get_info()
        .map_err(|e| format!("mem_get_info: {e:?}"))?;
    let (mut reserved, mut used) = (0u64, 0u64);
    unsafe {
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        if sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()) == sys::CUresult::CUDA_SUCCESS
        {
            sys::cuMemPoolGetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                &mut reserved as *mut u64 as *mut std::ffi::c_void,
            );
            sys::cuMemPoolGetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                &mut used as *mut u64 as *mut std::ffi::c_void,
            );
        }
    }
    Ok((
        (free as u64).saturating_add(reserved.saturating_sub(used)),
        total as u64,
    ))
}

/// The pool's high-water mark of bytes in use since the last [`reset_peak`], on `ops`'s card.
pub fn pool_peak(ops: &Ops) -> u64 {
    use cudarc::driver::sys;
    let mut peak = 0u64;
    unsafe {
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        if sys::cuDeviceGetDefaultMemPool(&mut pool, ops.st.context().cu_device())
            == sys::CUresult::CUDA_SUCCESS
        {
            sys::cuMemPoolGetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                &mut peak as *mut u64 as *mut std::ffi::c_void,
            );
        }
    }
    peak
}

/// Bytes of the stream-ordered pool in use now, on `ops`'s card.
pub fn pool_used(ops: &Ops) -> u64 {
    use cudarc::driver::sys;
    let mut used = 0u64;
    unsafe {
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        if sys::cuDeviceGetDefaultMemPool(&mut pool, ops.st.context().cu_device())
            == sys::CUresult::CUDA_SUCCESS
        {
            sys::cuMemPoolGetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                &mut used as *mut u64 as *mut std::ffi::c_void,
            );
        }
    }
    used
}

/// Restart the pool's high-water mark at the bytes in use now.
pub fn reset_peak(ops: &Ops) {
    use cudarc::driver::sys;
    unsafe {
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        if sys::cuDeviceGetDefaultMemPool(&mut pool, ops.st.context().cu_device())
            == sys::CUresult::CUDA_SUCCESS
        {
            let mut zero = 0u64;
            sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                &mut zero as *mut u64 as *mut std::ffi::c_void,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_slot_layout_packs_the_six_pieces_on_256_byte_boundaries() {
        let g = Geo {
            dim: 5120,
            moe_inter: 2304,
            ..Geo::default()
        };
        let l = Layout::of(&g);
        assert_eq!(
            l.len,
            [5_898_240, 368_640, 5_898_240, 368_640, 5_898_240, 368_640]
        );
        assert_eq!(l.bytes, 18_800_640);
        assert!(l.off.iter().all(|o| o % 256 == 0));
        for k in 1..6 {
            assert_eq!(l.off[k], l.off[k - 1] + l.len[k - 1]);
        }
        assert_eq!(
            l.piece(&g, 3, 7, 4),
            (
                "layers.3.ffn.experts.7.w2.weight".to_string(),
                "I8",
                [5120, 1152]
            )
        );
        assert_eq!(
            l.piece(&g, 3, 7, 1),
            (
                "layers.3.ffn.experts.7.w1.scale".to_string(),
                "F8_E8M0",
                [2304, 160]
            )
        );
    }
}
