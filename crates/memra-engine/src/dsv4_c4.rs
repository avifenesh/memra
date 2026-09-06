//! Experimental active C4 residency. Only storage moves; values and index order do not.
//! C128/indexer/SWA stay on device. All access is ordered on the owning stage stream.
use cudarc::driver::{CudaSlice, CudaStream, CudaView, DevicePtr, DevicePtrMut, sys};
use std::{
    mem::ManuallyDrop,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

type Res<T> = Result<T, String>;
const HD: usize = 512;
const WIN: usize = 128;
const RECENT_MAX: usize = 8192;
static RECENT_GATHERS: AtomicU64 = AtomicU64::new(0);
// Gate-only profile arm. It is safe only when the recent sidecar covers the
// complete compressed history capacity: rows not yet in the sidecar remain the
// canonical host image, and no sidecar slot can wrap before capacity is reached.
static HOST_COPY_ELIDE: AtomicBool = AtomicBool::new(false);

pub(crate) fn set_host_copy_elision_for_gate(enabled: bool) -> bool {
    HOST_COPY_ELIDE.swap(enabled, Ordering::SeqCst)
}

pub(crate) fn host_copy_elision_enabled() -> bool {
    HOST_COPY_ELIDE.load(Ordering::SeqCst)
}

pub(crate) fn recent_gather_calls() -> u64 {
    RECENT_GATHERS.load(std::sync::atomic::Ordering::Relaxed)
}

fn recent_capacity(history: usize, requested: usize) -> Res<usize> {
    if requested > RECENT_MAX || history > (i32::MAX as usize) - RECENT_MAX {
        return Err("C4 recent/history row count outside supported range".into());
    }
    Ok(history.min(requested))
}

fn bind_runtime(stream: &Arc<CudaStream>) -> Res<()> {
    stream
        .context()
        .bind_to_thread()
        .map_err(|e| e.to_string())?;
    let rc = unsafe { crate::mmq_ffi::memra_bind_device(stream.context().ordinal() as i32) };
    if rc != 0 {
        return Err(format!("C4 recent runtime bind rc={rc}"));
    }
    Ok(())
}

struct C4Recent {
    values: CudaSlice<f32>,
    tags: CudaSlice<i32>,
}

impl C4Recent {
    fn new(stream: &Arc<CudaStream>, rows: usize) -> Res<Self> {
        Ok(Self {
            values: stream.alloc_zeros(rows * HD).map_err(|e| e.to_string())?,
            tags: stream
                .clone_htod(&vec![-1; rows])
                .map_err(|e| e.to_string())?,
        })
    }
    fn bytes(&self) -> u64 {
        (self.values.len() * 4 + self.tags.len() * 4) as u64
    }
    fn write(
        &mut self,
        stream: &Arc<CudaStream>,
        src: *const f32,
        first: usize,
        rows: usize,
        reset: bool,
    ) -> Res<()> {
        bind_runtime(stream)?;
        let cache_rows = self.tags.len();
        let (out, _out_record) = self.values.device_ptr_mut(stream);
        let (tags, _tag_record) = self.tags.device_ptr_mut(stream);
        unsafe {
            crate::dsv4_ffi::ck(
                "C4 recent write",
                crate::dsv4_ffi::memra_dsv4_c4_recent_write(
                    src,
                    out as *mut f32,
                    tags as *mut i32,
                    first as i32,
                    rows as i32,
                    cache_rows as i32,
                    reset as i32,
                    stream.cu_stream().cast(),
                ),
            )
        }
    }
}

pub(crate) struct C4HostStore {
    backing: ManuallyDrop<crate::PinnedHostBuf>,
    ptr: *mut f32,
    stream: Arc<CudaStream>,
    pub(crate) rows: usize,
    recent: Option<C4Recent>,
}
// One owning state/stream; no concurrent CPU access or shared mutable views.
unsafe impl Send for C4HostStore {}

impl C4HostStore {
    pub(crate) fn new(stream: Arc<CudaStream>, rows: usize) -> Res<Self> {
        Self::with_recent(stream, rows, 0)
    }

    pub(crate) fn with_recent(
        stream: Arc<CudaStream>,
        rows: usize,
        recent_rows: usize,
    ) -> Res<Self> {
        let recent_rows = recent_capacity(rows, recent_rows)?;
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        if stream
            .context()
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)
            .map_err(|e| e.to_string())?
            == 0
        {
            return Err(
                "active C4 requires unified addressing for cacheable pinned host memory".into(),
            );
        }
        let bytes = rows.checked_mul(HD * 4).ok_or("C4 host size overflow")?;
        let mut backing = crate::PinnedHostBuf::new(bytes).map_err(|e| e.to_string())?;
        backing.as_mut_slice().fill(0);
        let ptr = backing.as_mut_slice().as_mut_ptr().cast::<f32>();
        let recent = if recent_rows > 0 && rows > 0 {
            Some(C4Recent::new(&stream, recent_rows.min(rows))?)
        } else {
            None
        };
        // CUDA's UVA contract gives cacheable cuMemHostAlloc(flags=0) allocations
        // the same host/device pointer. Never substitute write-combined storage.
        Ok(Self {
            backing: ManuallyDrop::new(backing),
            ptr,
            stream,
            rows,
            recent,
        })
    }

    pub(crate) fn write(&mut self, row: usize, src: CudaView<'_, f32>) -> Res<()> {
        let range = row_range(row, src.len(), self.rows)?;
        if range.is_empty() {
            return Ok(());
        }
        let (src, _record) = src.device_ptr(&self.stream);
        let full_recent = HOST_COPY_ELIDE.load(Ordering::Acquire)
            && self
                .recent
                .as_ref()
                .is_some_and(|recent| recent.tags.len() == self.rows);
        if full_recent {
            // The sidecar is a complete, absolute-row image for this bounded
            // state. Keep the new row GPU-resident; gather_recent will select it
            // by tag while all older rows still come from the canonical host
            // image. No slot can wrap before the state reaches capacity.
            self.recent.as_mut().expect("full recent sidecar").write(
                &self.stream,
                src as *const f32,
                row,
                range.len() / HD,
                false,
            )?;
        } else {
            // Unlike the generic HostSlice wrapper, do not synchronously drain
            // after every emitted block. This allocation lives until Drop drains
            // the stream; CPU reads below drain first, and gather uses this exact
            // same stream.
            unsafe {
                let dst = std::slice::from_raw_parts_mut(self.ptr.add(range.start), range.len());
                cudarc::driver::result::memcpy_dtoh_async(dst, src, self.stream.cu_stream())
                    .map_err(|e| format!("C4 emit D2H: {e}"))?;
            }
            if let Some(recent) = self.recent.as_mut() {
                recent.write(
                    &self.stream,
                    src as *const f32,
                    row,
                    range.len() / HD,
                    false,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn read(&self, rows: usize) -> Res<Vec<f32>> {
        let range = row_range(
            0,
            rows.checked_mul(HD).ok_or("C4 read overflow")?,
            self.rows,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| format!("C4 read sync: {e}"))?;
        Ok(unsafe { std::slice::from_raw_parts(self.ptr, range.len()) }.to_vec())
    }

    /// Copy a canonical host snapshot directly into active pinned history.
    /// The owning stream drains before CPU mutation, including on reused stores.
    pub(crate) fn copy_from_host(&mut self, values: &[f32]) -> Res<()> {
        let range = row_range(0, values.len(), self.rows)?;
        self.stream
            .synchronize()
            .map_err(|e| format!("C4 host restore sync: {e}"))?;
        unsafe { std::slice::from_raw_parts_mut(self.ptr, range.len()) }.copy_from_slice(values);
        if let Some(recent) = self.recent.as_mut() {
            // Reset all tags, including after restoring a shorter/empty prefix.
            // The active pinned history remains alive and ordered on this stream.
            recent.write(&self.stream, self.ptr, 0, values.len() / HD, true)?;
        }
        Ok(())
    }

    /// Avoid an intermediate full-history Vec when building a canonical snapshot.
    pub(crate) fn copy_to_host(&self, values: &mut [f32]) -> Res<()> {
        let range = row_range(0, values.len(), self.rows)?;
        self.stream
            .synchronize()
            .map_err(|e| format!("C4 host snapshot sync: {e}"))?;
        values.copy_from_slice(unsafe { std::slice::from_raw_parts(self.ptr, range.len()) });
        Ok(())
    }

    pub(crate) fn bytes(&self) -> usize {
        self.rows * HD * 4
    }

    pub(crate) fn device_bytes(&self) -> u64 {
        self.recent.as_ref().map_or(0, C4Recent::bytes)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gather(
        &self,
        device: &CudaSlice<f32>,
        indices: &CudaSlice<i32>,
        work: &mut Option<C4Gather>,
        nq: usize,
        slots: usize,
        stride: usize,
        live_rows: usize,
        logical_transient: usize,
    ) -> Res<(*const f32, *const i32)> {
        if nq == 0
            || nq > 512
            || slots == 0
            || slots > 640
            || stride < slots
            || indices.len() < nq * stride
            || live_rows > self.rows
            || logical_transient != WIN + self.rows
            || device.len() < WIN * HD
            || !device.len().is_multiple_of(HD)
        {
            return Err("invalid active C4 gather shape".into());
        }
        C4Gather::ensure(work, &self.stream, nq, stride)?;
        let w = work.as_mut().expect("C4 gather workspace");
        let (device_ptr, _device_record) = device.device_ptr(&self.stream);
        let (idx, _idx_record) = indices.device_ptr(&self.stream);
        let (out, _out_record) = w.values.device_ptr_mut(&self.stream);
        let (out_idx, _out_idx_record) = w.indices.device_ptr_mut(&self.stream);
        if let Some(recent) = self.recent.as_ref() {
            bind_runtime(&self.stream)?;
            let (values, _values_record) = recent.values.device_ptr(&self.stream);
            let (tags, _tags_record) = recent.tags.device_ptr(&self.stream);
            unsafe {
                crate::dsv4_ffi::ck(
                    "C4 cached host gather",
                    crate::dsv4_ffi::memra_dsv4_c4_gather_recent(
                        device_ptr as *const f32,
                        self.ptr,
                        idx as *const i32,
                        out as *mut f32,
                        out_idx as *mut i32,
                        nq as i32,
                        slots as i32,
                        stride as i32,
                        live_rows as i32,
                        logical_transient as i32,
                        (device.len() / HD - WIN) as i32,
                        values as *const f32,
                        tags as *const i32,
                        recent.tags.len() as i32,
                        self.stream.cu_stream().cast(),
                    ),
                )?;
            }
            RECENT_GATHERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            unsafe {
                crate::dsv4_ffi::ck(
                    "C4 host gather",
                    crate::dsv4_ffi::memra_dsv4_c4_gather(
                        device_ptr as *const f32,
                        self.ptr,
                        idx as *const i32,
                        out as *mut f32,
                        out_idx as *mut i32,
                        nq as i32,
                        slots as i32,
                        stride as i32,
                        live_rows as i32,
                        logical_transient as i32,
                        (device.len() / HD - WIN) as i32,
                        self.stream.cu_stream().cast(),
                    ),
                )?;
            }
        }
        Ok((out as *const f32, out_idx as *const i32))
    }
}

impl Drop for C4HostStore {
    fn drop(&mut self) {
        // Even unwinding must not free pinned bytes still referenced by queued DMA
        // or kernels. A broken context leaks this allocation rather than risking UAF.
        if self.stream.synchronize().is_ok() {
            unsafe { ManuallyDrop::drop(&mut self.backing) };
        }
    }
}

pub(crate) struct C4Gather {
    values: CudaSlice<f32>,
    indices: CudaSlice<i32>,
}

impl C4Gather {
    pub(crate) fn bytes(&self) -> u64 {
        (self.values.len() * 4 + self.indices.len() * 4) as u64
    }

    /// Reserve all 640 slots once, not again each time a new compressed row
    /// increases the live top-k. Returns the incremental device allocation.
    pub(crate) fn ensure(
        work: &mut Option<Self>,
        stream: &Arc<CudaStream>,
        nq: usize,
        stride: usize,
    ) -> Res<u64> {
        let values = nq.checked_mul(640 * HD).ok_or("C4 gather size overflow")?;
        let ids = nq
            .checked_mul(stride)
            .ok_or("C4 gather index size overflow")?;
        let before = work.as_ref().map_or(0, Self::bytes);
        if work
            .as_ref()
            .is_none_or(|w| w.values.len() < values || w.indices.len() < ids)
        {
            let values = values.max(work.as_ref().map_or(0, |w| w.values.len()));
            let ids = ids.max(work.as_ref().map_or(0, |w| w.indices.len()));
            *work = Some(Self {
                values: stream.alloc_zeros(values).map_err(|e| e.to_string())?,
                indices: stream.alloc_zeros(ids).map_err(|e| e.to_string())?,
            });
        }
        Ok(work.as_ref().expect("gather workspace").bytes() - before)
    }
}

fn row_range(row: usize, elements: usize, capacity: usize) -> Res<std::ops::Range<usize>> {
    if !elements.is_multiple_of(HD) {
        return Err("C4 writes must contain whole 512-value rows".into());
    }
    let end = row
        .checked_add(elements / HD)
        .ok_or("C4 row range overflow")?;
    if end > capacity {
        return Err("C4 row range exceeds capacity".into());
    }
    let start = row.checked_mul(HD).ok_or("C4 element range overflow")?;
    Ok(start
        ..start
            .checked_add(elements)
            .ok_or("C4 element range overflow")?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recent_capacity_is_bounded_and_optional() {
        assert_eq!(recent_capacity(262144, 0), Ok(0));
        assert_eq!(recent_capacity(262144, 512), Ok(512));
        assert_eq!(recent_capacity(7, 512), Ok(7));
        assert_eq!(recent_capacity(0, 512), Ok(0));
        assert!(recent_capacity(10, 8193).is_err());
        assert!(recent_capacity(usize::MAX, 512).is_err());
    }

    #[test]
    #[ignore = "requires an exclusively locked non-serving CUDA device"]
    fn cuda_recent_c4_preserves_hits_misses_wrap_and_rollback() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().unwrap();
        let capacity = 65;
        let local: Vec<f32> = (0..(WIN + 6) * HD).map(|i| -(i as f32) / 32.0).collect();
        let device = stream.clone_htod(&local).unwrap();
        for recent_rows in [1, 7, 32, 65] {
            let mut host = C4HostStore::with_recent(stream.clone(), capacity, recent_rows).unwrap();
            assert_eq!(host.device_bytes(), (recent_rows * (HD + 1) * 4) as u64);
            let mut canonical: Vec<f32> = (0..capacity * HD)
                .map(|i| {
                    if i % 17 == 0 {
                        -0.0
                    } else {
                        f32::from_bits(0x3f000000 | ((i as u32).wrapping_mul(997) & 0x7fffff))
                    }
                })
                .collect();
            let mut work = None;
            let slots = 8;
            let stride = 11;
            let mut selected = vec![-1i32; stride];
            selected[..slots].copy_from_slice(&[
                0,
                -1,
                128,
                128 + 12,
                128 + 13,
                128 + 18,
                128 + 19,
                (WIN + capacity) as i32,
            ]);
            let ids = stream.clone_htod(&selected).unwrap();
            let verify =
                |host: &C4HostStore, work: &mut Option<C4Gather>, canon: &[f32], live: usize| {
                    host.gather(&device, &ids, work, 1, slots, stride, live, WIN + capacity)
                        .unwrap();
                    stream.synchronize().unwrap();
                    let values = stream.clone_dtoh(&work.as_ref().unwrap().values).unwrap();
                    let indices = stream.clone_dtoh(&work.as_ref().unwrap().indices).unwrap();
                    for (slot, &id) in selected[..slots].iter().enumerate() {
                        assert_eq!(indices[slot], if id < 0 { -1 } else { slot as i32 });
                        for d in 0..HD {
                            let want = if id < 0 {
                                0.0
                            } else if id < 128 {
                                local[id as usize * HD + d]
                            } else if id < (128 + live) as i32 {
                                canon[(id as usize - 128) * HD + d]
                            } else {
                                local[WIN * HD + d]
                            };
                            assert_eq!(
                                values[slot * HD + d].to_bits(),
                                want.to_bits(),
                                "cache={recent_rows} slot={slot} dim={d}"
                            );
                        }
                    }
                };
            host.copy_from_host(&canonical[..20 * HD]).unwrap();
            verify(&host, &mut work, &canonical, 20);
            // Future writes may evict live rows from the ring. Rolled-back reads
            // must miss by absolute tag, then fetch the unmodified host history.
            let future: Vec<f32> = (0..13 * HD).map(|i| 100.0 + i as f32 / 32.0).collect();
            let future_gpu = stream.clone_htod(&future).unwrap();
            host.write(20, future_gpu.slice(..)).unwrap();
            verify(&host, &mut work, &canonical, 20);
            // Re-emission changes exactly the same logical rows and cached slots.
            for v in &mut canonical[12 * HD..20 * HD] {
                *v = -*v;
            }
            let rewritten = stream.clone_htod(&canonical[12 * HD..20 * HD]).unwrap();
            host.write(12, rewritten.slice(..)).unwrap();
            verify(&host, &mut work, &canonical, 20);
            // Prove a recent hit actually bypasses host memory. This is a test-only
            // corruption; restore canonical bytes immediately after checking it.
            stream.synchronize().unwrap();
            let saved = unsafe { *host.ptr.add(19 * HD) };
            unsafe {
                *host.ptr.add(19 * HD) = 12345.0;
            }
            verify(&host, &mut work, &canonical, 20);
            unsafe {
                *host.ptr.add(19 * HD) = saved;
            }
            // Fixed-live-shape graph replay, with changing host contents and tags.
            stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .unwrap();
            host.gather(
                &device,
                &ids,
                &mut work,
                1,
                slots,
                stride,
                20,
                WIN + capacity,
            )
            .unwrap();
            let graph = stream
                .end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .unwrap()
                .unwrap();
            for pass in 0..4 {
                for v in &mut canonical[..20 * HD] {
                    *v = -*v;
                }
                host.copy_from_host(&canonical[..20 * HD]).unwrap();
                if pass % 2 == 1 {
                    host.write(20, future_gpu.slice(..)).unwrap();
                }
                graph.launch().unwrap();
                stream.synchronize().unwrap();
                let graph_values = stream.clone_dtoh(&work.as_ref().unwrap().values).unwrap();
                verify(&host, &mut work, &canonical, 20);
                let eager = stream.clone_dtoh(&work.as_ref().unwrap().values).unwrap();
                assert_eq!(
                    graph_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    eager.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
            drop(graph);
            // Empty and short restore must invalidate tags outside the new image.
            host.copy_from_host(&[]).unwrap();
            stream.synchronize().unwrap();
            assert!(
                stream
                    .clone_dtoh(&host.recent.as_ref().unwrap().tags)
                    .unwrap()
                    .iter()
                    .all(|&id| id == -1)
            );
            host.copy_from_host(&canonical[..HD]).unwrap();
            stream.synchronize().unwrap();
            let tags = stream
                .clone_dtoh(&host.recent.as_ref().unwrap().tags)
                .unwrap();
            assert_eq!(tags[0], 0);
            assert!(tags[1..].iter().all(|&id| id == -1));
            println!(
                "PASS recent_rows={recent_rows} hit_red=true wrap=true rollback=true reemit=true graph_replays=4 reset=true"
            );
        }
    }

    #[test]
    #[ignore = "requires an exclusively locked non-serving CUDA device"]
    fn cuda_direct_host_copy_preserves_bits_and_live_prefix() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut host = C4HostStore::new(stream, 8).unwrap();
        for rows in [8, 1, 0, 5, 8] {
            let values: Vec<f32> = (0..rows * HD)
                .map(|i| {
                    if i % 17 == 0 {
                        -0.0
                    } else {
                        f32::from_bits(0x3f000000 | ((i as u32).wrapping_mul(997) & 0x7fffff))
                    }
                })
                .collect();
            host.copy_from_host(&values).unwrap();
            let mut actual = vec![f32::NAN; values.len()];
            host.copy_to_host(&mut actual).unwrap();
            assert_eq!(
                values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        assert!(host.copy_from_host(&[0.0; 1]).is_err());
        assert!(host.copy_to_host(&mut [0.0; 1]).is_err());
        assert!(host.copy_from_host(&vec![0.0; 9 * HD]).is_err());
        assert_eq!(host.bytes(), 8 * HD * 4);
        println!(
            "PASS direct host copy: exact bits, signed zero, zero/shrunk/grown live prefixes and bounds"
        );
    }

    #[test]
    fn rows_are_checked_before_async_dma() {
        assert_eq!(row_range(2, 1024, 4).unwrap(), 1024..2048);
        assert!(row_range(3, 1024, 4).is_err());
        assert!(row_range(0, 511, 4).is_err());
        assert!(row_range(usize::MAX, 512, usize::MAX).is_err());
        assert_eq!(row_range(4, 0, 4).unwrap(), 2048..2048);
    }

    #[test]
    #[ignore = "requires an exclusively locked non-serving CUDA device"]
    fn cuda_active_c4_gather_preserves_rows_and_reemissions() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("context");
        let stream = ctx.new_stream().expect("stream");
        for (nq, rows) in [(1, 0), (1, 1), (6, 40), (32, 1025), (1, 262144)] {
            let capacity = rows + 7; // unread dead tail before logical transient rows
            let stride = 647;
            let logical_transient = WIN + capacity;
            let mut host = C4HostStore::new(stream.clone(), capacity).expect("host");
            let mut full: Vec<f32> = (0..rows * HD)
                .map(|i| f32::from_bits(0x3f000000 | ((i as u32).wrapping_mul(7319) & 0x7fffff)))
                .collect();
            let mut local: Vec<f32> = (0..(WIN + nq) * HD).map(|i| -(i as f32) / 32.0).collect();
            local[0] = -0.0;
            let device = stream.clone_htod(&local).expect("local GPU");
            if rows > 0 {
                let source = stream.clone_htod(&full).expect("source GPU");
                host.write(0, source.slice(..)).expect("initial D2H");
                stream.synchronize().expect("source retirement");
            }
            let mut idx = vec![-1; nq * stride];
            for q in 0..nq {
                for slot in 0..640 {
                    idx[q * stride + slot] = match slot % 5 {
                        0 => -1,
                        1 => ((q * 13 + slot) % WIN) as i32,
                        2 => (logical_transient + q) as i32,
                        _ if rows > 0 => (WIN + (rows - 1 - (slot * 7 + q) % rows)) as i32,
                        _ => -1,
                    };
                }
                if rows > 0 {
                    idx[q * stride + 3] = WIN as i32;
                }
            }
            let indices = stream.clone_htod(&idx).expect("indices");
            let mut work = Some(C4Gather {
                values: stream
                    .clone_htod(&vec![f32::from_bits(0x7fc12345); nq * 640 * HD + 17])
                    .expect("values"),
                indices: stream
                    .clone_htod(&vec![-73i32; nq * stride + 17])
                    .expect("output indices"),
            });
            for emission in 0..2 {
                if emission == 1 && rows > 0 {
                    for x in &mut full[..HD] {
                        *x = -*x;
                    }
                    let source = stream.clone_htod(&full[..HD]).expect("re-emitted GPU row");
                    host.write(0, source.slice(..))
                        .expect("rollback re-emission");
                    // No host access or explicit sync between this async D2H and
                    // the gather. Source retirement is stream-ordered by cudarc.
                }
                host.gather(
                    &device,
                    &indices,
                    &mut work,
                    nq,
                    640,
                    stride,
                    rows,
                    logical_transient,
                )
                .expect("gather");
                let work_ref = work.as_ref().unwrap();
                let values = stream.clone_dtoh(&work_ref.values).expect("read values");
                let mapped = stream.clone_dtoh(&work_ref.indices).expect("read indices");
                for q in 0..nq {
                    for slot in 0..640 {
                        let index = idx[q * stride + slot];
                        let row = q * 640 + slot;
                        assert_eq!(
                            mapped[q * stride + slot],
                            if index < 0 { -1 } else { row as i32 }
                        );
                        for x in 0..HD {
                            let expected = if index < 0 {
                                0.0
                            } else if (index as usize) < WIN {
                                local[index as usize * HD + x]
                            } else if (index as usize) < logical_transient {
                                full[(index as usize - WIN) * HD + x]
                            } else {
                                local[(WIN + index as usize - logical_transient) * HD + x]
                            };
                            assert_eq!(
                                values[row * HD + x].to_bits(),
                                expected.to_bits(),
                                "nq={nq} rows={rows} q={q} slot={slot} x={x} emission={emission}"
                            );
                        }
                    }
                    assert!(
                        mapped[q * stride + 640..(q + 1) * stride]
                            .iter()
                            .all(|i| *i == -73)
                    );
                }
                assert!(
                    values[nq * 640 * HD..]
                        .iter()
                        .all(|v| v.to_bits() == 0x7fc12345)
                );
                assert!(mapped[nq * stride..].iter().all(|i| *i == -73));
            }
            println!(
                "PASS active C4 gather nq={nq} live_rows={rows} capacity={capacity} guards/pads/bit-identity/re-emission"
            );
        }
    }
}
