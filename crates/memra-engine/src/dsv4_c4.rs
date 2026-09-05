//! Experimental active C4 residency. Only storage moves; values and index order do not.
//! C128/indexer/SWA stay on device. All access is ordered on the owning stage stream.
use cudarc::driver::{CudaSlice, CudaStream, CudaView, DevicePtr, DevicePtrMut, sys};
use std::{mem::ManuallyDrop, sync::Arc};

type Res<T> = Result<T, String>;
const HD: usize = 512;
const WIN: usize = 128;

pub(crate) struct C4HostStore {
    backing: ManuallyDrop<crate::PinnedHostBuf>,
    ptr: *mut f32,
    stream: Arc<CudaStream>,
    pub(crate) rows: usize,
}
// One owning state/stream; no concurrent CPU access or shared mutable views.
unsafe impl Send for C4HostStore {}

impl C4HostStore {
    pub(crate) fn new(stream: Arc<CudaStream>, rows: usize) -> Res<Self> {
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
        // CUDA's UVA contract gives cacheable cuMemHostAlloc(flags=0) allocations
        // the same host/device pointer. Never substitute write-combined storage.
        Ok(Self {
            backing: ManuallyDrop::new(backing),
            ptr,
            stream,
            rows,
        })
    }

    pub(crate) fn write(&mut self, row: usize, src: CudaView<'_, f32>) -> Res<()> {
        let range = row_range(row, src.len(), self.rows)?;
        if range.is_empty() {
            return Ok(());
        }
        let (src, _record) = src.device_ptr(&self.stream);
        // Unlike the generic HostSlice wrapper, do not synchronously drain after
        // every emitted block. This allocation lives until Drop drains the stream;
        // CPU reads below drain first, and gather uses this exact same stream.
        unsafe {
            let dst = std::slice::from_raw_parts_mut(self.ptr.add(range.start), range.len());
            cudarc::driver::result::memcpy_dtoh_async(dst, src, self.stream.cu_stream())
                .map_err(|e| format!("C4 emit D2H: {e}"))
        }
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

    pub(crate) fn bytes(&self) -> usize {
        self.rows * HD * 4
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
