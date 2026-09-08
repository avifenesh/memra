//! Two-card whole-expert residency and exact FP8 dispatch/slot return.
//! Attention/shared experts stay on their existing owner; routed GEMVs use both cards.
use crate::dsv4_ffi as k;
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr};
use memra_runtime::Gpu;
use std::{
    ffi::c_void,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

type Res<T> = Result<T, String>;

/// Gate-only route-load receipt for the whole-expert EP experiment. The matrix
/// executor owns one contiguous expert half per rank; these counters record the
/// actual selected-slot split before any future row/column-split rewrite.
#[derive(Clone, Copy, Debug, Default)]
pub struct EpRouteStats {
    pub calls: u64,
    pub observed_calls: u64,
    pub unobserved_calls: u64,
    pub local_slots: u64,
    pub peer_slots: u64,
    pub busier_slots: u64,
    pub one_row_calls: u64,
    pub local_hist: [u64; 9],
    pub busier_hist: [u64; 9],
}

static ROUTE_STATS_ENABLED: OnceLock<bool> = OnceLock::new();
static ROUTE_STATS_CALLS: AtomicU64 = AtomicU64::new(0);
static ROUTE_STATS_OBSERVED_CALLS: AtomicU64 = AtomicU64::new(0);
static ROUTE_STATS_UNOBSERVED_CALLS: AtomicU64 = AtomicU64::new(0);
static ROUTE_STATS_LOCAL_SLOTS: AtomicU64 = AtomicU64::new(0);
static ROUTE_STATS_PEER_SLOTS: AtomicU64 = AtomicU64::new(0);
static ROUTE_STATS_BUSIER_SLOTS: AtomicU64 = AtomicU64::new(0);
static ROUTE_STATS_ONE_ROW_CALLS: AtomicU64 = AtomicU64::new(0);
static ROUTE_STATS_LOCAL_HIST: OnceLock<[AtomicU64; 9]> = OnceLock::new();
static ROUTE_STATS_BUSIER_HIST: OnceLock<[AtomicU64; 9]> = OnceLock::new();

fn route_stats_enabled() -> bool {
    *ROUTE_STATS_ENABLED.get_or_init(|| {
        matches!(
            std::env::var("MEMRA_DSV4_EP_ROUTE_STATS").as_deref(),
            Ok("1")
        )
    })
}

fn route_stats_hist() -> (&'static [AtomicU64; 9], &'static [AtomicU64; 9]) {
    (
        ROUTE_STATS_LOCAL_HIST.get_or_init(|| std::array::from_fn(|_| AtomicU64::new(0))),
        ROUTE_STATS_BUSIER_HIST.get_or_init(|| std::array::from_fn(|_| AtomicU64::new(0))),
    )
}

fn route_stats_call_delta(
    observed_split: Option<(usize, usize)>,
    rows: usize,
    topk: usize,
) -> EpRouteStats {
    let mut delta = EpRouteStats {
        calls: 1,
        ..EpRouteStats::default()
    };
    let Some((local_slots, peer_slots)) = observed_split else {
        delta.unobserved_calls = 1;
        return delta;
    };
    delta.observed_calls = 1;
    delta.local_slots = local_slots as u64;
    delta.peer_slots = peer_slots as u64;
    let busier = local_slots.max(peer_slots);
    delta.busier_slots = busier as u64;
    if rows == 1 && topk < 9 {
        delta.one_row_calls = 1;
        delta.local_hist[local_slots.min(8)] = 1;
        delta.busier_hist[busier.min(8)] = 1;
    }
    delta
}

fn record_route_stats(observed_split: Option<(usize, usize)>, rows: usize, topk: usize) {
    if !route_stats_enabled() {
        return;
    }
    let delta = route_stats_call_delta(observed_split, rows, topk);
    ROUTE_STATS_CALLS.fetch_add(delta.calls, Ordering::Relaxed);
    ROUTE_STATS_OBSERVED_CALLS.fetch_add(delta.observed_calls, Ordering::Relaxed);
    ROUTE_STATS_UNOBSERVED_CALLS.fetch_add(delta.unobserved_calls, Ordering::Relaxed);
    ROUTE_STATS_LOCAL_SLOTS.fetch_add(delta.local_slots, Ordering::Relaxed);
    ROUTE_STATS_PEER_SLOTS.fetch_add(delta.peer_slots, Ordering::Relaxed);
    ROUTE_STATS_BUSIER_SLOTS.fetch_add(delta.busier_slots, Ordering::Relaxed);
    ROUTE_STATS_ONE_ROW_CALLS.fetch_add(delta.one_row_calls, Ordering::Relaxed);
    let (local, busy) = route_stats_hist();
    for (i, count) in delta.local_hist.into_iter().enumerate() {
        local[i].fetch_add(count, Ordering::Relaxed);
    }
    for (i, count) in delta.busier_hist.into_iter().enumerate() {
        busy[i].fetch_add(count, Ordering::Relaxed);
    }
}

pub fn route_stats_snapshot() -> EpRouteStats {
    let (local, busy) = route_stats_hist();
    EpRouteStats {
        calls: ROUTE_STATS_CALLS.load(Ordering::Relaxed),
        observed_calls: ROUTE_STATS_OBSERVED_CALLS.load(Ordering::Relaxed),
        unobserved_calls: ROUTE_STATS_UNOBSERVED_CALLS.load(Ordering::Relaxed),
        local_slots: ROUTE_STATS_LOCAL_SLOTS.load(Ordering::Relaxed),
        peer_slots: ROUTE_STATS_PEER_SLOTS.load(Ordering::Relaxed),
        busier_slots: ROUTE_STATS_BUSIER_SLOTS.load(Ordering::Relaxed),
        one_row_calls: ROUTE_STATS_ONE_ROW_CALLS.load(Ordering::Relaxed),
        local_hist: std::array::from_fn(|i| local[i].load(Ordering::Relaxed)),
        busier_hist: std::array::from_fn(|i| busy[i].load(Ordering::Relaxed)),
    }
}

impl EpRouteStats {
    pub fn delta(self, before: Self) -> Self {
        Self {
            calls: self.calls.saturating_sub(before.calls),
            observed_calls: self.observed_calls.saturating_sub(before.observed_calls),
            unobserved_calls: self
                .unobserved_calls
                .saturating_sub(before.unobserved_calls),
            local_slots: self.local_slots.saturating_sub(before.local_slots),
            peer_slots: self.peer_slots.saturating_sub(before.peer_slots),
            busier_slots: self.busier_slots.saturating_sub(before.busier_slots),
            one_row_calls: self.one_row_calls.saturating_sub(before.one_row_calls),
            local_hist: std::array::from_fn(|i| {
                self.local_hist[i].saturating_sub(before.local_hist[i])
            }),
            busier_hist: std::array::from_fn(|i| {
                self.busier_hist[i].saturating_sub(before.busier_hist[i])
            }),
        }
    }
}

struct PeerFailureDrain {
    owner: Arc<CudaStream>,
    peer: Arc<CudaStream>,
    complete: bool,
}
impl Drop for PeerFailureDrain {
    fn drop(&mut self) {
        if !self.complete {
            // Drain before borrowed peer destinations can be released on error.
            let _ = self.owner.synchronize();
            let _ = self.peer.synchronize();
        }
    }
}

pub(crate) struct EpLayer {
    pub peer_stage: usize,
    pub local_first: usize,
    pub count: usize,
    pub peer_first: usize,
    pub peer_w: CudaSlice<u8>,
    pub peer_sc: CudaSlice<u8>,
    /// Tiny global-id metadata, replicated; expert code/scale banks are not replicated.
    pub peer_s2: CudaSlice<f32>,
    pub peer_table: Option<CudaSlice<u64>>,
    /// True for the all-layer TP/EP loader, where each rank owns its expert-ID
    /// half from the outset.  The peer byte planes are intentionally empty in
    /// that topology; the paired rank computes its own local partial and the
    /// join happens in `reduce_rank_order_f32`.
    pub local_only: bool,
}

/// Stable state for the PRO two-rank one-shot reduction.  The signal and
/// refusal words are device-resident and survive every layer/token launch;
/// there is no per-layer host event or staging buffer. The walk reads both
/// refusal words at the drained token boundary before committing either cache.
pub(crate) struct TpEpArState {
    signal: [CudaSlice<u8>; 2],
    error: [CudaSlice<i32>; 2],
    launches: u64,
}

impl TpEpArState {
    pub(crate) fn new(owner: &Gpu, peer: &Gpu) -> Res<Self> {
        if owner.ctx.ordinal() == peer.ctx.ordinal() {
            return Err("TP/EP one-shot reduction requires two distinct devices".into());
        }
        let bytes = unsafe { crate::tp_ar::memra_tp_ar_signal_bytes() };
        if bytes <= 0 {
            return Err(format!(
                "TP/EP one-shot reduction returned invalid signal size {bytes}"
            ));
        }
        let owner_stream = owner.stream();
        let peer_stream = peer.stream();
        let signal0 = owner_stream
            .alloc_zeros::<u8>(bytes as usize)
            .map_err(|e| format!("TP/EP AR rank-0 signal: {e}"))?;
        let signal1 = peer_stream
            .alloc_zeros::<u8>(bytes as usize)
            .map_err(|e| format!("TP/EP AR rank-1 signal: {e}"))?;
        let error0 = owner_stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| format!("TP/EP AR rank-0 refusal: {e}"))?;
        let error1 = peer_stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| format!("TP/EP AR rank-1 refusal: {e}"))?;
        owner_stream
            .synchronize()
            .map_err(|e| format!("TP/EP AR rank-0 state init: {e}"))?;
        peer_stream
            .synchronize()
            .map_err(|e| format!("TP/EP AR rank-1 state init: {e}"))?;
        Ok(Self {
            signal: [signal0, signal1],
            error: [error0, error1],
            launches: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn all_reduce_into(
        &mut self,
        owner: &Gpu,
        peer: &Gpu,
        owner_input: &CudaSlice<f32>,
        peer_input: &CudaSlice<f32>,
        owner_output: &mut CudaSlice<f32>,
        peer_output: &mut CudaSlice<f32>,
        n: usize,
        capture: bool,
        fault: Option<([*const u64; 2], i32)>,
    ) -> Res<()> {
        if n == 0
            || owner_input.len() < n
            || peer_input.len() < n
            || owner_output.len() < n
            || peer_output.len() < n
        {
            return Err("TP/EP one-shot reduction buffer shape mismatch".into());
        }
        let owner_input_ptr;
        let owner_output_ptr;
        let owner_signal_ptr;
        let owner_error_ptr;
        {
            owner.ctx.bind_to_thread().map_err(|e| e.to_string())?;
            let stream = owner.stream();
            owner_input_ptr = owner_input.device_ptr(&stream).0 as *const f32;
            owner_output_ptr = owner_output.device_ptr_mut(&stream).0 as *mut f32;
            owner_signal_ptr = self.signal[0].device_ptr_mut(&stream).0 as *mut c_void;
            owner_error_ptr = self.error[0].device_ptr_mut(&stream).0 as *mut i32;
        }
        let peer_input_ptr;
        let peer_output_ptr;
        let peer_signal_ptr;
        let peer_error_ptr;
        {
            peer.ctx.bind_to_thread().map_err(|e| e.to_string())?;
            let stream = peer.stream();
            peer_input_ptr = peer_input.device_ptr(&stream).0 as *const f32;
            peer_output_ptr = peer_output.device_ptr_mut(&stream).0 as *mut f32;
            peer_signal_ptr = self.signal[1].device_ptr_mut(&stream).0 as *mut c_void;
            peer_error_ptr = self.error[1].device_ptr_mut(&stream).0 as *mut i32;
        }
        if std::ptr::eq(owner_input_ptr, owner_output_ptr)
            || std::ptr::eq(peer_input_ptr, peer_output_ptr)
        {
            return Err("TP/EP one-shot reduction output aliases an input".into());
        }
        let blocks = crate::tp_ar::ar_blocks_for(n);
        for (rank, gpu, in0, in1, out, self_signal, peer_signal, error) in [
            (
                0,
                owner,
                owner_input_ptr,
                peer_input_ptr,
                owner_output_ptr,
                owner_signal_ptr,
                peer_signal_ptr,
                owner_error_ptr,
            ),
            (
                1,
                peer,
                owner_input_ptr,
                peer_input_ptr,
                peer_output_ptr,
                peer_signal_ptr,
                owner_signal_ptr,
                peer_error_ptr,
            ),
        ] {
            if capture {
                gpu.ctx.bind_to_thread().map_err(|e| e.to_string())?;
            }
            let stream = gpu.stream();
            let rc = unsafe {
                if let Some((inputs, site)) = fault {
                    crate::tp_ar::memra_tp_ar_1stage_replay(
                        in0,
                        in1,
                        out,
                        self_signal,
                        peer_signal,
                        rank,
                        n as i64,
                        error,
                        crate::tp_ar::AR_SPIN_LIMIT,
                        blocks,
                        stream.cu_stream().cast(),
                        inputs[rank as usize].cast(),
                        site,
                    )
                } else {
                    crate::tp_ar::memra_tp_ar_1stage(
                        in0,
                        in1,
                        out,
                        self_signal,
                        peer_signal,
                        rank,
                        n as i64,
                        error,
                        crate::tp_ar::AR_SPIN_LIMIT,
                        blocks,
                        stream.cu_stream().cast(),
                    )
                }
            };
            if rc != 0 {
                return Err(format!(
                    "TP/EP one-shot reduction launch rc {rc} rank {rank}"
                ));
            }
        }
        self.launches += 1;
        Ok(())
    }

    pub(crate) fn launches(&self) -> u64 {
        self.launches
    }
    pub(crate) fn epochs_for_gate(&self, owner: &Gpu, peer: &Gpu) -> Res<[Vec<u32>; 2]> {
        let offset = unsafe { crate::tp_ar::memra_tp_ar_seq_offset_bytes() } as usize;
        let mut out = [Vec::new(), Vec::new()];
        for (rank, gpu) in [owner, peer].into_iter().enumerate() {
            if offset + 72 * 4 > self.signal[rank].len() {
                return Err("AR epoch view outside signal allocation".into());
            }
            gpu.ctx.bind_to_thread().map_err(|e| e.to_string())?;
            let stream = gpu.stream();
            let mut bytes = vec![0u8; 72 * 4];
            stream
                .memcpy_dtoh(
                    &self.signal[rank].slice(offset..offset + 72 * 4),
                    &mut bytes,
                )
                .map_err(|e| e.to_string())?;
            stream.synchronize().map_err(|e| e.to_string())?;
            out[rank] = bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
        }
        Ok(out)
    }

    pub(crate) fn refusal_words(&self, owner: &Gpu, peer: &Gpu) -> Res<[i32; 2]> {
        let mut out = [0i32; 2];
        for (i, gpu) in [owner, peer].into_iter().enumerate() {
            gpu.ctx.bind_to_thread().map_err(|e| e.to_string())?;
            let stream = gpu.stream();
            stream
                .memcpy_dtoh(&self.error[i], std::slice::from_mut(&mut out[i]))
                .map_err(|e| format!("TP/EP AR refusal read rank {i}: {e}"))?;
            stream
                .synchronize()
                .map_err(|e| format!("TP/EP AR refusal sync rank {i}: {e}"))?;
        }
        Ok(out)
    }

    /// Fault-injection only. The caller holds the model walk lock; both streams
    /// are drained before replacing the sticky words, including when clearing a
    /// completed red arm so another independent gate state can run.
    pub(crate) fn set_refusal_words_for_gate(
        &mut self,
        owner: &Gpu,
        peer: &Gpu,
        words: [i32; 2],
    ) -> Res<()> {
        for gpu in [owner, peer] {
            gpu.ctx.bind_to_thread().map_err(|e| e.to_string())?;
            gpu.stream().synchronize().map_err(|e| e.to_string())?;
        }
        for (rank, gpu) in [owner, peer].into_iter().enumerate() {
            gpu.ctx.bind_to_thread().map_err(|e| e.to_string())?;
            let stream = gpu.stream();
            stream
                .memcpy_htod(&words[rank..rank + 1], &mut self.error[rank])
                .map_err(|e| format!("TP/EP AR inject refusal rank {rank}: {e}"))?;
            stream.synchronize().map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

pub(crate) struct EpScratch {
    pub xq: CudaSlice<u8>,
    pub xs: CudaSlice<f32>,
    pub ids: CudaSlice<i32>,
    pub weights: CudaSlice<f32>,
    pub g1: CudaSlice<f32>,
    pub g3: CudaSlice<f32>,
    pub h: CudaSlice<f32>,
    pub hq: CudaSlice<u8>,
    pub hs: CudaSlice<f32>,
    pub contribution: CudaSlice<f32>,
    pub returned: CudaSlice<f32>,
    pub grouped: Option<crate::dsv4_grouped::GroupedWork>,
    tx_done: CudaEvent,
    rx_done: CudaEvent,
    pub owner_bytes: u64,
    pub peer_bytes: u64,
}

pub(crate) struct EpCompute<'a> {
    pub xq: &'a CudaSlice<u8>,
    pub xs: &'a CudaSlice<f32>,
    pub ids: &'a CudaSlice<i32>,
    pub weights: &'a CudaSlice<f32>,
    pub g1: &'a mut CudaSlice<f32>,
    pub g3: &'a mut CudaSlice<f32>,
    pub h: &'a mut CudaSlice<f32>,
    pub hq: &'a mut CudaSlice<u8>,
    pub hs: &'a mut CudaSlice<f32>,
    pub contribution: &'a mut CudaSlice<f32>,
}

impl EpScratch {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        owner: &Gpu,
        peer: &Gpu,
        rows: usize,
        topk: usize,
        hidden: usize,
        inter: usize,
        matrix_partition: Option<(usize, usize, usize)>,
    ) -> Res<Self> {
        if rows == 0
            || rows > 512
            || topk == 0
            || !hidden.is_multiple_of(128)
            || !inter.is_multiple_of(128)
        {
            return Err("invalid EP workspace shape".into());
        }
        let slots = rows * topk;
        let ps = peer.stream();
        let grouped = match matrix_partition {
            Some((global, first, count)) => Some(crate::dsv4_grouped::GroupedWork::new_partition(
                &ps, global, first, count, slots, hidden, inter,
            )?),
            None => None,
        };
        let xq = ps.alloc_zeros(rows * hidden).map_err(|e| e.to_string())?;
        let xs = ps
            .alloc_zeros(rows * hidden / 128)
            .map_err(|e| e.to_string())?;
        let ids = ps.alloc_zeros(slots).map_err(|e| e.to_string())?;
        let weights = ps.alloc_zeros(slots).map_err(|e| e.to_string())?;
        let g1 = ps.alloc_zeros(slots * inter).map_err(|e| e.to_string())?;
        let g3 = ps.alloc_zeros(slots * inter).map_err(|e| e.to_string())?;
        let h = ps.alloc_zeros(slots * inter).map_err(|e| e.to_string())?;
        let hq = ps.alloc_zeros(slots * inter).map_err(|e| e.to_string())?;
        let hs = ps
            .alloc_zeros(slots * inter / 128)
            .map_err(|e| e.to_string())?;
        let contribution = ps.alloc_zeros(slots * hidden).map_err(|e| e.to_string())?;
        let rx_done = peer.ctx.new_event(None).map_err(|e| e.to_string())?;
        let os = owner.stream();
        let returned = os.alloc_zeros(slots * hidden).map_err(|e| e.to_string())?;
        let tx_done = owner.ctx.new_event(None).map_err(|e| e.to_string())?;
        // Explicit ordering is mandatory with per-buffer event tracking off.
        ps.synchronize().map_err(|e| e.to_string())?;
        os.synchronize().map_err(|e| e.to_string())?;
        let owner_bytes = (slots * hidden * 4) as u64;
        let peer_bytes = (rows * (hidden + hidden / 128 * 4)
            + slots * (8 + inter * 13 + inter / 128 * 4 + hidden * 4))
            as u64
            + grouped.as_ref().map_or(0, |work| work.bytes);
        Ok(Self {
            xq,
            xs,
            ids,
            weights,
            g1,
            g3,
            h,
            hq,
            hs,
            contribution,
            returned,
            grouped,
            tx_done,
            rx_done,
            owner_bytes,
            peer_bytes,
        })
    }
}

pub(crate) fn split_slab(
    owner: &Gpu,
    peer: &Gpu,
    source: &CudaSlice<u8>,
    local_first: usize,
    peer_first: usize,
    count: usize,
    expert_stride: usize,
) -> Res<(CudaSlice<u8>, CudaSlice<u8>)> {
    let bytes = count
        .checked_mul(expert_stride)
        .ok_or("EP slab size overflow")?;
    let local_start = local_first
        .checked_mul(expert_stride)
        .ok_or("EP slab offset overflow")?;
    let peer_start = peer_first
        .checked_mul(expert_stride)
        .ok_or("EP slab offset overflow")?;
    if local_start
        .checked_add(bytes)
        .is_none_or(|n| n > source.len())
        || peer_start
            .checked_add(bytes)
            .is_none_or(|n| n > source.len())
    {
        return Err("EP shard outside original bank".into());
    }
    let os = owner.stream();
    let ps = peer.stream();
    let mut local = os.alloc_zeros::<u8>(bytes).map_err(|e| e.to_string())?;
    let mut remote = ps.alloc_zeros::<u8>(bytes).map_err(|e| e.to_string())?;
    let mut drain = PeerFailureDrain {
        owner: os.clone(),
        peer: ps.clone(),
        complete: false,
    };
    ps.synchronize().map_err(|e| e.to_string())?;
    os.memcpy_dtod(&source.slice(local_start..local_start + bytes), &mut local)
        .map_err(|e| e.to_string())?;
    // Use the already-probed 64 MiB copy class rather than one untested GiB copy.
    for offset in (0..bytes).step_by(64 << 20) {
        let end = (offset + (64 << 20)).min(bytes);
        peer_copy(
            &os,
            &ps,
            &source.slice(peer_start + offset..peer_start + end),
            &mut remote.slice_mut(offset..end),
            end - offset,
        )?;
    }
    os.synchronize().map_err(|e| e.to_string())?;
    // Every retained code/scale byte is checked before the full bank is retired.
    for offset in (0..bytes).step_by(64 << 20) {
        let end = (offset + (64 << 20)).min(bytes);
        let expected_local = os
            .clone_dtoh(&source.slice(local_start + offset..local_start + end))
            .map_err(|e| e.to_string())?;
        let actual_local = os
            .clone_dtoh(&local.slice(offset..end))
            .map_err(|e| e.to_string())?;
        if expected_local != actual_local {
            return Err(format!("EP local shard mismatch at chunk {offset}"));
        }
        let expected_peer = os
            .clone_dtoh(&source.slice(peer_start + offset..peer_start + end))
            .map_err(|e| e.to_string())?;
        let actual_peer = ps
            .clone_dtoh(&remote.slice(offset..end))
            .map_err(|e| e.to_string())?;
        if expected_peer != actual_peer {
            return Err(format!("EP peer shard mismatch at chunk {offset}"));
        }
    }
    drain.complete = true;
    Ok((local, remote))
}

pub(crate) fn peer_copy<T: DeviceRepr, S: DevicePtr<T>, D: DevicePtrMut<T>>(
    source: &Arc<CudaStream>,
    destination: &Arc<CudaStream>,
    src: &S,
    dst: &mut D,
    n: usize,
) -> Res<()> {
    if n > src.len() || n > dst.len() {
        return Err("EP peer copy exceeds workspace".into());
    }
    source
        .context()
        .bind_to_thread()
        .map_err(|e| e.to_string())?;
    let (sp, _read) = src.device_ptr(source);
    let (dp, _write) = dst.device_ptr_mut(source);
    unsafe {
        cudarc::driver::result::memcpy_peer_async(
            destination.context().cu_ctx(),
            dp,
            source.context().cu_ctx(),
            sp,
            n * std::mem::size_of::<T>(),
            source.cu_stream(),
        )
        .map_err(|e| format!("EP peer copy: {e}"))
    }
}

#[allow(clippy::too_many_arguments)]
fn chain(
    gpu: &Gpu,
    w: &CudaSlice<u8>,
    sc: &CudaSlice<u8>,
    s2: &CudaSlice<f32>,
    first: usize,
    count: usize,
    ws: &mut EpCompute<'_>,
    rows: usize,
    topk: usize,
    hidden: usize,
    inter: usize,
    limit: f32,
    reduction: i32,
) -> Res<()> {
    gpu.ctx.bind_to_thread().map_err(|e| e.to_string())?;
    let s = gpu.stream();
    let sv = s.cu_stream().cast::<c_void>();
    let slots = rows * topk;
    let wstride = (inter * hidden / 2) as i64;
    let sstride = (inter * hidden / 16) as i64;
    let wp = w.device_ptr(&s).0 as *const c_void;
    let sp = sc.device_ptr(&s).0 as *const c_void;
    let s2p = s2.device_ptr(&s).0 as *const f32;
    let selected = ws.ids.device_ptr(&s).0 as *const i32;
    unsafe {
        for (projection, out) in [(0, &mut ws.g1), (2, &mut ws.g3)] {
            k::ck(
                "EP w1/w3",
                k::memra_dsv4_fp4_gemm_sel_ep(
                    ws.xq.device_ptr(&s).0 as *const c_void,
                    ws.xs.device_ptr(&s).0 as *const f32,
                    wp,
                    sp,
                    s2p,
                    selected,
                    projection,
                    0,
                    0,
                    out.device_ptr_mut(&s).0 as *mut f32,
                    slots as i32,
                    inter as i32,
                    hidden as i32,
                    wstride,
                    sstride,
                    topk as i32,
                    reduction,
                    first as i32,
                    count as i32,
                    sv,
                ),
            )?;
        }
        k::ck(
            "EP swiglu",
            k::memra_dsv4_swiglu(
                ws.g1.device_ptr(&s).0 as *const f32,
                ws.g3.device_ptr(&s).0 as *const f32,
                ws.h.device_ptr_mut(&s).0 as *mut f32,
                slots as i32,
                inter as i32,
                limit,
                ws.weights.device_ptr(&s).0 as *const f32,
                sv,
            ),
        )?;
        k::ck(
            "EP FP8 h",
            k::memra_dsv4_act_quant_fp8(
                ws.h.device_ptr(&s).0 as *const f32,
                ws.hq.device_ptr_mut(&s).0 as *mut c_void,
                ws.hs.device_ptr_mut(&s).0 as *mut f32,
                slots as i32,
                inter as i32,
                sv,
            ),
        )?;
        k::ck(
            "EP w2",
            k::memra_dsv4_fp4_gemm_sel_ep(
                ws.hq.device_ptr(&s).0 as *const c_void,
                ws.hs.device_ptr(&s).0 as *const f32,
                wp,
                sp,
                s2p,
                selected,
                1,
                1,
                0,
                ws.contribution.device_ptr_mut(&s).0 as *mut f32,
                slots as i32,
                hidden as i32,
                inter as i32,
                wstride,
                sstride,
                0,
                reduction,
                first as i32,
                count as i32,
                sv,
            ),
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute(
    owner: &Gpu,
    peer: &Gpu,
    bank: &EpLayer,
    w: &CudaSlice<u8>,
    sc: &CudaSlice<u8>,
    s2: &CudaSlice<f32>,
    local: &mut EpCompute<'_>,
    remote: &mut EpScratch,
    rows: usize,
    topk: usize,
    hidden: usize,
    inter: usize,
    limit: f32,
    reduction: i32,
    serial_control: bool,
) -> Res<(u64, u64)> {
    let os = owner.stream();
    let ps = peer.stream();
    let slots = rows * topk;
    let mut drain = PeerFailureDrain {
        owner: os.clone(),
        peer: ps.clone(),
        complete: false,
    };
    // Quantize once on the owner. The wire carries the exact FP8 codes/scales,
    // global router ids and routing weights, never a substitute activation program.
    peer_copy(&os, &ps, local.xq, &mut remote.xq, rows * hidden)?;
    peer_copy(&os, &ps, local.xs, &mut remote.xs, rows * hidden / 128)?;
    peer_copy(&os, &ps, local.ids, &mut remote.ids, slots)?;
    peer_copy(&os, &ps, local.weights, &mut remote.weights, slots)?;
    remote.tx_done.record(&os).map_err(|e| e.to_string())?;
    chain(
        owner,
        w,
        sc,
        s2,
        bank.local_first,
        bank.count,
        local,
        rows,
        topk,
        hidden,
        inter,
        limit,
        reduction,
    )?;
    if serial_control {
        os.synchronize().map_err(|e| e.to_string())?;
    }
    peer.ctx.bind_to_thread().map_err(|e| e.to_string())?;
    ps.wait(&remote.tx_done).map_err(|e| e.to_string())?;
    chain(
        peer,
        &bank.peer_w,
        &bank.peer_sc,
        &bank.peer_s2,
        bank.peer_first,
        bank.count,
        &mut EpCompute {
            xq: &remote.xq,
            xs: &remote.xs,
            ids: &remote.ids,
            weights: &remote.weights,
            g1: &mut remote.g1,
            g3: &mut remote.g3,
            h: &mut remote.h,
            hq: &mut remote.hq,
            hs: &mut remote.hs,
            contribution: &mut remote.contribution,
        },
        rows,
        topk,
        hidden,
        inter,
        limit,
        reduction,
    )?;
    peer_copy(
        &ps,
        &os,
        &remote.contribution,
        &mut remote.returned,
        slots * hidden,
    )?;
    remote.rx_done.record(&ps).map_err(|e| e.to_string())?;
    owner.ctx.bind_to_thread().map_err(|e| e.to_string())?;
    os.wait(&remote.rx_done).map_err(|e| e.to_string())?;
    // Overwrite peer-owned slots rather than adding partial vectors. This keeps
    // signed-zero bits and the existing ascending-expert combine order unchanged.
    unsafe {
        k::ck(
            "EP original-slot return",
            k::memra_dsv4_ep_merge_slots(
                local.contribution.device_ptr_mut(&os).0 as *mut f32,
                remote.returned.device_ptr(&os).0 as *const f32,
                local.ids.device_ptr(&os).0 as *const i32,
                slots as i32,
                hidden as i32,
                bank.peer_first as i32,
                bank.count as i32,
                os.cu_stream().cast(),
            ),
        )?;
    }
    drain.complete = true;
    Ok((
        (rows * (hidden + hidden / 128 * 4) + slots * 8) as u64,
        (slots * hidden * 4) as u64,
    ))
}

/// Same matrix expert program on both ranks, staged around explicit dependencies.
/// Both gate/up chains are queued before either intermediate mirror validation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_matrix(
    owner: &Gpu,
    peer: &Gpu,
    bank: &EpLayer,
    table: &CudaSlice<u64>,
    scale2: &CudaSlice<f32>,
    scale2_host: &[f32],
    local: &mut EpCompute<'_>,
    local_work: &mut crate::dsv4_grouped::GroupedWork,
    remote: &mut EpScratch,
    rows: usize,
    topk: usize,
    hidden: usize,
    limit: f32,
    allow_gu_fuse: bool,
    serial_control: bool,
) -> Res<u64> {
    if bank.local_only {
        return Err("TP/EP local-only bank cannot enter peer-dispatch EP".into());
    }
    let peer_table = bank
        .peer_table
        .as_ref()
        .ok_or("EP matrix peer table missing")?;
    let global = bank
        .count
        .checked_mul(2)
        .ok_or("EP matrix expert count overflow")?;
    if !((bank.local_first == 0 && bank.peer_first == bank.count)
        || (bank.peer_first == 0 && bank.local_first == bank.count))
        || !local_work
            .routes
            .matches_partition(global, bank.local_first, bank.count)
        || !remote.grouped.as_ref().is_some_and(|work| {
            work.routes
                .matches_partition(global, bank.peer_first, bank.count)
        })
    {
        return Err("EP matrix bank/workspace ownership mismatch".into());
    }
    let os = owner.stream();
    let ps = peer.stream();
    let slots = rows * topk;
    let mut drain = PeerFailureDrain {
        owner: os.clone(),
        peer: ps.clone(),
        complete: false,
    };
    peer_copy(&os, &ps, local.xq, &mut remote.xq, rows * hidden)?;
    peer_copy(&os, &ps, local.xs, &mut remote.xs, rows * hidden / 128)?;
    peer_copy(&os, &ps, local.ids, &mut remote.ids, slots)?;
    peer_copy(&os, &ps, local.weights, &mut remote.weights, slots)?;
    remote.tx_done.record(&os).map_err(|e| e.to_string())?;
    peer.ctx.bind_to_thread().map_err(|e| e.to_string())?;
    ps.wait(&remote.tx_done).map_err(|e| e.to_string())?;
    let peer_work = remote
        .grouped
        .as_mut()
        .ok_or("EP matrix peer workspace missing")?;
    let mut peer_compute = EpCompute {
        xq: &remote.xq,
        xs: &remote.xs,
        ids: &remote.ids,
        weights: &remote.weights,
        g1: &mut remote.g1,
        g3: &mut remote.g3,
        h: &mut remote.h,
        hq: &mut remote.hq,
        hs: &mut remote.hs,
        contribution: &mut remote.contribution,
    };
    let mut route_calls =
        u64::from(local_work.prepare(owner, local, scale2, scale2_host, rows, topk, true)?);
    route_calls += u64::from(peer_work.prepare(
        peer,
        &peer_compute,
        &bank.peer_s2,
        scale2_host,
        rows,
        topk,
        true,
    )?);
    local_work.set_gu_fuse_for_plain(allow_gu_fuse);
    peer_work.set_gu_fuse_for_plain(allow_gu_fuse);
    if crate::dsv4_grouped::route_validation_enabled()
        && local_work.routes.live_slots + peer_work.routes.live_slots != slots
    {
        return Err("EP matrix partitions did not cover every selected slot".into());
    }
    record_route_stats(
        if local_work.routes.live_slots_observed && peer_work.routes.live_slots_observed {
            Some((local_work.routes.live_slots, peer_work.routes.live_slots))
        } else {
            None
        },
        rows,
        topk,
    );
    local_work.gate_up(owner, table, local, limit)?;
    if serial_control {
        local_work.down(owner, table, local)?;
        os.synchronize().map_err(|e| e.to_string())?;
        peer_work.gate_up(peer, peer_table, &mut peer_compute, limit)?;
    } else {
        peer_work.gate_up(peer, peer_table, &mut peer_compute, limit)?;
        local_work.down(owner, table, local)?;
    }
    peer_work.down(peer, peer_table, &mut peer_compute)?;
    peer_copy(
        &ps,
        &os,
        &remote.contribution,
        &mut remote.returned,
        slots * hidden,
    )?;
    remote.rx_done.record(&ps).map_err(|e| e.to_string())?;
    // The peer matrix stage explicitly bound CUDART as well as the driver.
    // Restore both before the owner's raw merge and following shared-expert work.
    crate::dsv4_grouped::bind_matrix(owner)?;
    os.wait(&remote.rx_done).map_err(|e| e.to_string())?;
    unsafe {
        k::ck(
            "EP matrix original-slot return",
            k::memra_dsv4_ep_merge_slots(
                local.contribution.device_ptr_mut(&os).0 as *mut f32,
                remote.returned.device_ptr(&os).0 as *const f32,
                local.ids.device_ptr(&os).0 as *const i32,
                slots as i32,
                hidden as i32,
                bank.peer_first as i32,
                bank.count as i32,
                os.cu_stream().cast(),
            ),
        )?;
    }
    drain.complete = true;
    Ok(route_calls)
}

/// Numeric class for the first TP/EP vertical slice. Each rank computes its
/// owned expert partition, then the PRO one-shot primitive evaluates the same
/// global-rank-ordered f32 sum on both ranks. This is deliberately named
/// separately from the old slot-overwrite EP combine.
pub(crate) const TP_EP_RANK_ORDER_NUMERIC_CLASS: &str =
    "dsv4_expert_id_ep_slot_order_f32_rank_reduce";

/// Run only the local expert half for an all-layer TP/EP rank. The caller owns
/// the replicated attention/router state and supplies the rank-local grouped
/// partition. There is no peer dispatch and therefore no PP owner hidden in
/// this function.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_matrix_local(
    gpu: &Gpu,
    bank: &EpLayer,
    table: &CudaSlice<u64>,
    scale2: &CudaSlice<f32>,
    scale2_host: &[f32],
    local: &mut EpCompute<'_>,
    work: &mut crate::dsv4_grouped::GroupedWork,
    rows: usize,
    topk: usize,
    limit: f32,
    allow_gu_fuse: bool,
) -> Res<u64> {
    if rows == 0 || topk == 0 {
        return Err("TP/EP local expert execution requires nonzero rows/topk".into());
    }
    if !bank.local_only || bank.peer_table.is_none() {
        return Err("TP/EP local expert execution requires a complete local matrix table".into());
    }
    // A local-only partition scatters only the live slots it owns into the caller's
    // contribution plane.  Unlike peer-dispatch EP, no later slot-merge overwrites the
    // complementary half.  Clear the external plane first so a reused one-row workspace
    // cannot feed a previous token/layer's non-owned partial into the rank-order reduce.
    gpu.stream()
        .memset_zeros(local.contribution)
        .map_err(|e| format!("TP/EP local contribution clear: {e}"))?;
    let calls = u64::from(work.prepare(gpu, local, scale2, scale2_host, rows, topk, true)?);
    work.set_gu_fuse_for_plain(allow_gu_fuse);
    work.gate_up(gpu, table, local, limit)?;
    work.down(gpu, table, local)?;
    Ok(calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_stats_call_classification_keeps_unknown_counts_out_of_slot_totals() {
        let unknown = route_stats_call_delta(None, 1, 6);
        assert_eq!(unknown.calls, 1);
        assert_eq!(unknown.observed_calls, 0);
        assert_eq!(unknown.unobserved_calls, 1);
        assert_eq!(unknown.local_slots, 0);
        assert_eq!(unknown.peer_slots, 0);
        assert_eq!(unknown.busier_slots, 0);
        assert_eq!(unknown.local_hist, [0; 9]);
        assert_eq!(unknown.busier_hist, [0; 9]);

        let observed_one_row = route_stats_call_delta(Some((3, 5)), 1, 6);
        assert_eq!(observed_one_row.observed_calls, 1);
        assert_eq!(observed_one_row.unobserved_calls, 0);
        assert_eq!(observed_one_row.local_slots, 3);
        assert_eq!(observed_one_row.peer_slots, 5);
        assert_eq!(observed_one_row.busier_slots, 5);
        assert_eq!(observed_one_row.one_row_calls, 1);
        assert_eq!(observed_one_row.local_hist[3], 1);
        assert_eq!(observed_one_row.busier_hist[5], 1);

        let observed_multi_row = route_stats_call_delta(Some((12, 8)), 2, 6);
        assert_eq!(observed_multi_row.observed_calls, 1);
        assert_eq!(observed_multi_row.one_row_calls, 0);
        assert_eq!(observed_multi_row.local_hist, [0; 9]);

        let before = EpRouteStats {
            calls: 4,
            observed_calls: 2,
            unobserved_calls: 2,
            ..EpRouteStats::default()
        };
        let after = EpRouteStats {
            calls: 7,
            observed_calls: 3,
            unobserved_calls: 4,
            ..EpRouteStats::default()
        };
        let delta = after.delta(before);
        assert_eq!(delta.calls, 3);
        assert_eq!(delta.observed_calls, 1);
        assert_eq!(delta.unobserved_calls, 2);
        assert_eq!(delta.local_slots, 0);
        assert_eq!(delta.busier_slots, 0);
    }

    #[test]
    #[ignore = "requires an exclusively locked non-serving CUDA device"]
    fn cuda_ep_partition_matches_original_full_bank() {
        let gpu = Gpu::new(0).expect("gpu");
        unsafe { gpu.ctx.disable_event_tracking() };
        let stream = gpu.stream();
        let ne = 8usize;
        let topk = 6usize;
        for (hidden, inter) in [(128usize, 256usize), (4096, 2048)] {
            let wstride = hidden * inter / 2;
            let sstride = hidden * inter / 16;
            let weights: Vec<u8> = (0..ne * 3 * wstride)
                .map(|i| (i.wrapping_mul(73) + 31) as u8)
                .collect();
            let scales: Vec<u8> = (0..ne * 3 * sstride)
                .map(|i| 0x20 + (i % 64) as u8)
                .collect();
            let macro_scales: Vec<f32> = (0..ne * 3).map(|i| 2f32.powi(i as i32 % 7 - 3)).collect();
            let w = stream.clone_htod(&weights).unwrap();
            let sc = stream.clone_htod(&scales).unwrap();
            let s2 = stream.clone_htod(&macro_scales).unwrap();
            for rows in [1usize, 6, 32] {
                let slots = rows * topk;
                let ids: Vec<i32> = (0..slots)
                    .map(|i| ((i / topk * 7 + i % topk * 3) % ne) as i32)
                    .collect();
                let selected = stream.clone_htod(&ids).unwrap();
                for projection in 0..3i32 {
                    let (n, kdim, arows, a_group, per_slot) = if projection == 1 {
                        (hidden, inter, slots, 0, 1)
                    } else {
                        (inter, hidden, rows, topk as i32, 0)
                    };
                    let a: Vec<u8> = (0..arows * kdim)
                        .map(|i| {
                            let b = (i.wrapping_mul(11) + 37) as u8;
                            if b & 127 == 127 { 0 } else { b }
                        })
                        .collect();
                    let ascale: Vec<f32> = (0..arows * kdim / 128)
                        .map(|i| 2f32.powi(i as i32 % 9 - 5))
                        .collect();
                    let aq = stream.clone_htod(&a).unwrap();
                    let asc = stream.clone_htod(&ascale).unwrap();
                    for reduction in [0, 1] {
                        let sentinel = f32::from_bits(0x7fc54321);
                        let mut baseline =
                            stream.clone_htod(&vec![sentinel; slots * n + 17]).unwrap();
                        let mut left = stream.clone_htod(&vec![sentinel; slots * n + 17]).unwrap();
                        let mut right = stream.clone_htod(&vec![sentinel; slots * n + 17]).unwrap();
                        unsafe {
                            k::ck(
                                "full-bank anchor",
                                k::memra_dsv4_fp4_gemm_sel_g_arm(
                                    aq.device_ptr(&stream).0 as *const c_void,
                                    asc.device_ptr(&stream).0 as *const f32,
                                    w.device_ptr(&stream).0 as *const c_void,
                                    sc.device_ptr(&stream).0 as *const c_void,
                                    s2.device_ptr(&stream).0 as *const f32,
                                    selected.device_ptr(&stream).0 as *const i32,
                                    projection,
                                    per_slot,
                                    0,
                                    baseline.device_ptr_mut(&stream).0 as *mut f32,
                                    slots as i32,
                                    n as i32,
                                    kdim as i32,
                                    wstride as i64,
                                    sstride as i64,
                                    a_group,
                                    reduction,
                                    stream.cu_stream().cast(),
                                ),
                            )
                            .unwrap();
                            for (first, out) in [(0usize, &mut left), (ne / 2, &mut right)] {
                                k::ck(
                                    "partition",
                                    k::memra_dsv4_fp4_gemm_sel_ep(
                                        aq.device_ptr(&stream).0 as *const c_void,
                                        asc.device_ptr(&stream).0 as *const f32,
                                        (w.device_ptr(&stream).0 + (first * 3 * wstride) as u64)
                                            as *const c_void,
                                        (sc.device_ptr(&stream).0 + (first * 3 * sstride) as u64)
                                            as *const c_void,
                                        s2.device_ptr(&stream).0 as *const f32,
                                        selected.device_ptr(&stream).0 as *const i32,
                                        projection,
                                        per_slot,
                                        0,
                                        out.device_ptr_mut(&stream).0 as *mut f32,
                                        slots as i32,
                                        n as i32,
                                        kdim as i32,
                                        wstride as i64,
                                        sstride as i64,
                                        a_group,
                                        reduction,
                                        first as i32,
                                        (ne / 2) as i32,
                                        stream.cu_stream().cast(),
                                    ),
                                )
                                .unwrap();
                            }
                        }
                        let inactive = stream.clone_dtoh(&left).unwrap();
                        for slot in 0..slots {
                            if ids[slot] >= (ne / 2) as i32 {
                                assert!(
                                    inactive[slot * n..(slot + 1) * n]
                                        .iter()
                                        .all(|v| v.to_bits() == 0)
                                );
                            }
                        }
                        unsafe {
                            k::ck(
                                "ordered slot merge",
                                k::memra_dsv4_ep_merge_slots(
                                    left.device_ptr_mut(&stream).0 as *mut f32,
                                    right.device_ptr(&stream).0 as *const f32,
                                    selected.device_ptr(&stream).0 as *const i32,
                                    slots as i32,
                                    n as i32,
                                    (ne / 2) as i32,
                                    (ne / 2) as i32,
                                    stream.cu_stream().cast(),
                                ),
                            )
                            .unwrap();
                        }
                        let expected = stream.clone_dtoh(&baseline).unwrap();
                        let actual = stream.clone_dtoh(&left).unwrap();
                        assert!(
                            expected
                                .iter()
                                .zip(&actual)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "partition mismatch h={hidden} i={inter} rows={rows} p={projection} reduce={reduction}"
                        );
                        assert!(
                            actual[slots * n..]
                                .iter()
                                .all(|v| v.to_bits() == sentinel.to_bits())
                        );
                    }
                }
                println!(
                    "PASS EP projection partitions hidden={hidden} intermediate={inter} rows={rows} both reductions, global scales, masks, original-slot merge and guards"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires one locked CUDA GPU; TP/EP local contribution reuse regression"]
    fn cuda_tp_ep_local_only_clears_reused_contribution_and_rank_sums() {
        use crate::dsv4_grouped::{GroupedWork, modelopt_table};
        use cudarc::driver::{CudaSlice, CudaStream};
        use std::sync::Arc;

        assert!(crate::moe_f16g_mode() >= 2);
        assert!(crate::moe_f16g_direct_on(crate::QT_NVFP4_MODELOPT));

        const GLOBAL: usize = 8;
        const COUNT: usize = GLOBAL / 2;
        const HIDDEN: usize = 4096;
        const INTER: usize = 2048;
        const TOPK: usize = 6;
        const ROWS: usize = 1;
        const SLOTS: usize = ROWS * TOPK;
        let wbytes = INTER * HIDDEN / 2;
        let sbytes = INTER * HIDDEN / 16;

        fn mix(mut x: u64) -> u64 {
            x ^= x >> 30;
            x = x.wrapping_mul(0xbf58476d1ce4e5b9);
            x ^= x >> 27;
            x = x.wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        }
        fn make_bank_bytes(
            first: usize,
            count: usize,
            wbytes: usize,
            sbytes: usize,
        ) -> (Vec<u8>, Vec<u8>) {
            let mut weights = Vec::with_capacity(count * 3 * wbytes);
            let mut scales = Vec::with_capacity(count * 3 * sbytes);
            for local in 0..count {
                let expert = first + local;
                for projection in 0..3 {
                    for i in 0..wbytes {
                        weights.push(
                            (mix(expert as u64 * 0x1000_0001
                                + projection as u64 * 0x100_003
                                + i as u64)
                                % 0x7e) as u8
                                + 1,
                        );
                    }
                    for i in 0..sbytes {
                        scales.push(
                            (0x10
                                + (mix(expert as u64 * 0x10_001
                                    + projection as u64 * 0x100_003
                                    + i as u64)
                                    % 0x6f) as u8)
                                & 0x7f,
                        );
                    }
                }
            }
            (weights, scales)
        }
        fn view(scratch: &mut EpScratch) -> EpCompute<'_> {
            EpCompute {
                xq: &scratch.xq,
                xs: &scratch.xs,
                ids: &scratch.ids,
                weights: &scratch.weights,
                g1: &mut scratch.g1,
                g3: &mut scratch.g3,
                h: &mut scratch.h,
                hq: &mut scratch.hq,
                hs: &mut scratch.hs,
                contribution: &mut scratch.contribution,
            }
        }
        fn scratch(gpu: &Gpu) -> EpScratch {
            EpScratch::new(gpu, gpu, ROWS, TOPK, HIDDEN, INTER, None).unwrap()
        }
        fn work(stream: &Arc<CudaStream>, first: usize, count: usize) -> GroupedWork {
            GroupedWork::new_partition(stream, GLOBAL, first, count, SLOTS, HIDDEN, INTER).unwrap()
        }
        fn bank(
            stream: &Arc<CudaStream>,
            first: usize,
            count: usize,
            weights: &[u8],
            scales: &[u8],
            scale2: &[f32],
        ) -> (EpLayer, CudaSlice<u64>, CudaSlice<u8>, CudaSlice<u8>) {
            let w = stream.clone_htod(weights).unwrap();
            let sc = stream.clone_htod(scales).unwrap();
            let table = modelopt_table(stream, &w, &sc, count, HIDDEN, INTER).unwrap();
            let peer_table = modelopt_table(stream, &w, &sc, count, HIDDEN, INTER).unwrap();
            let bank = EpLayer {
                peer_stage: 0,
                local_first: first,
                count,
                peer_first: (GLOBAL - count) - first,
                peer_w: stream.alloc_zeros::<u8>(0).unwrap(),
                peer_sc: stream.alloc_zeros::<u8>(0).unwrap(),
                peer_s2: stream.alloc_zeros::<f32>(0).unwrap(),
                peer_table: Some(peer_table),
                local_only: true,
            };
            // Keep scale2 in the caller; this argument documents that the bank uses the
            // global expert scale plane even though its weight/table planes are local.
            assert_eq!(scale2.len(), GLOBAL * 3);
            (bank, table, w, sc)
        }
        fn load_inputs(
            stream: &Arc<CudaStream>,
            scratch: &mut EpScratch,
            xq: &[u8],
            xs: &[f32],
            ids: &[i32],
            weights: &[f32],
            sentinel: u32,
        ) {
            stream.memcpy_htod(xq, &mut scratch.xq).unwrap();
            stream.memcpy_htod(xs, &mut scratch.xs).unwrap();
            stream.memcpy_htod(ids, &mut scratch.ids).unwrap();
            stream.memcpy_htod(weights, &mut scratch.weights).unwrap();
            let poisoned = vec![f32::from_bits(sentinel); SLOTS * HIDDEN];
            stream
                .memcpy_htod(&poisoned, &mut scratch.contribution)
                .unwrap();
        }
        // Keep the independently poisoned buffers and ownership fixture explicit.
        #[allow(clippy::too_many_arguments)]
        fn run_local(
            gpu: &Gpu,
            stream: &Arc<CudaStream>,
            bank: &EpLayer,
            table: &CudaSlice<u64>,
            work: &mut GroupedWork,
            scratch: &mut EpScratch,
            xq: &[u8],
            xs: &[f32],
            ids: &[i32],
            weights: &[f32],
            scale2: &CudaSlice<f32>,
            scale2_host: &[f32],
            sentinel: u32,
        ) -> Vec<f32> {
            load_inputs(stream, scratch, xq, xs, ids, weights, sentinel);
            execute_matrix_local(
                gpu,
                bank,
                table,
                scale2,
                scale2_host,
                &mut view(scratch),
                work,
                ROWS,
                TOPK,
                6.0,
                false,
            )
            .unwrap();
            stream.synchronize().unwrap();
            stream
                .clone_dtoh(&scratch.contribution.slice(..SLOTS * HIDDEN))
                .unwrap()
        }
        fn assert_slots(values: &[f32], expected: &[f32], ids: &[i32], first: usize, count: usize) {
            assert_eq!(values.len(), expected.len());
            for slot in 0..SLOTS {
                let owned = (first as i32..(first + count) as i32).contains(&ids[slot]);
                for col in 0..HIDDEN {
                    let got = values[slot * HIDDEN + col];
                    let want = expected[slot * HIDDEN + col];
                    if owned {
                        assert_eq!(got.to_bits(), want.to_bits(), "owned slot={slot} col={col}");
                    } else {
                        assert_eq!(got.to_bits(), 0, "non-owned slot={slot} col={col}");
                    }
                }
            }
        }

        let previous_route_validation = crate::dsv4_grouped::set_route_validation_for_gate(true);
        let gpu = Gpu::new(0).unwrap();
        unsafe { gpu.ctx.disable_event_tracking() };
        let stream = gpu.stream();
        let (weights_a, scales_a) = make_bank_bytes(0, COUNT, wbytes, sbytes);
        let (weights_b, scales_b) = make_bank_bytes(COUNT, COUNT, wbytes, sbytes);
        let (weights_full, scales_full) = make_bank_bytes(0, GLOBAL, wbytes, sbytes);
        let scale2_host: Vec<f32> = (0..GLOBAL * 3)
            .map(|i| 2f32.powi((i % 9) as i32 - 4))
            .collect();
        let scale2 = stream.clone_htod(&scale2_host).unwrap();
        let xq: Vec<u8> = (0..HIDDEN)
            .map(|i| (mix(i as u64 * 0x10001 + 17) % 0x7e) as u8 + 1)
            .collect();
        let xs: Vec<f32> = (0..HIDDEN / 128)
            .map(|i| 2f32.powi((i % 7) as i32 - 3))
            .collect();
        let weights_route: Vec<f32> = [0.11, 0.17, 0.23, 0.31, 0.07, 0.19].to_vec();
        let ids_a = [0, 1, 2, 3, 0, 1];
        let ids_b = [4, 5, 6, 7, 4, 5];
        let ids_mixed = [0, 4, 1, 5, 2, 6];

        let (bank_a, table_a, _wa, _sa) =
            bank(&stream, 0, COUNT, &weights_a, &scales_a, &scale2_host);
        let (bank_b, table_b, _wb, _sb) =
            bank(&stream, COUNT, COUNT, &weights_b, &scales_b, &scale2_host);
        let (bank_full, table_full, _wf, _sf) = bank(
            &stream,
            0,
            GLOBAL,
            &weights_full,
            &scales_full,
            &scale2_host,
        );

        // Clean-vs-poison A/B for each ownership half. The second poison run reuses the
        // same external EpCompute contribution allocation after switching ownership.
        let mut work_a_clean = work(&stream, 0, COUNT);
        let mut scratch_a_clean = scratch(&gpu);
        let clean_a = run_local(
            &gpu,
            &stream,
            &bank_a,
            &table_a,
            &mut work_a_clean,
            &mut scratch_a_clean,
            &xq,
            &xs,
            &ids_a,
            &weights_route,
            &scale2,
            &scale2_host,
            0,
        );
        let mut work_a_poison = work(&stream, 0, COUNT);
        let mut scratch_reuse = scratch(&gpu);
        let poison_a = run_local(
            &gpu,
            &stream,
            &bank_a,
            &table_a,
            &mut work_a_poison,
            &mut scratch_reuse,
            &xq,
            &xs,
            &ids_a,
            &weights_route,
            &scale2,
            &scale2_host,
            0x7fc5_4321,
        );
        assert_slots(&poison_a, &clean_a, &ids_a, 0, COUNT);
        assert_eq!(
            poison_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            clean_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let mut work_b_clean = work(&stream, COUNT, COUNT);
        let mut scratch_b_clean = scratch(&gpu);
        let clean_b = run_local(
            &gpu,
            &stream,
            &bank_b,
            &table_b,
            &mut work_b_clean,
            &mut scratch_b_clean,
            &xq,
            &xs,
            &ids_b,
            &weights_route,
            &scale2,
            &scale2_host,
            0,
        );
        let mut work_b_poison = work(&stream, COUNT, COUNT);
        let poison_b = run_local(
            &gpu,
            &stream,
            &bank_b,
            &table_b,
            &mut work_b_poison,
            &mut scratch_reuse,
            &xq,
            &xs,
            &ids_b,
            &weights_route,
            &scale2,
            &scale2_host,
            0x7fc6_5432,
        );
        assert_slots(&poison_b, &clean_b, &ids_b, COUNT, COUNT);
        assert_eq!(
            poison_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            clean_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        // Mixed ownership rank-order output: the two local partitions must sum bitwise to
        // the complete-bank grouped result. This is the canonical F32 rank-reduce class,
        // not the sorted-expert CPU oracle's reassociated order.
        let mut work_a_mixed = work(&stream, 0, COUNT);
        let mut scratch_a_mixed = scratch(&gpu);
        let mixed_a = run_local(
            &gpu,
            &stream,
            &bank_a,
            &table_a,
            &mut work_a_mixed,
            &mut scratch_a_mixed,
            &xq,
            &xs,
            &ids_mixed,
            &weights_route,
            &scale2,
            &scale2_host,
            0x7fc7_6543,
        );
        let mut work_b_mixed = work(&stream, COUNT, COUNT);
        let mut scratch_b_mixed = scratch(&gpu);
        let mixed_b = run_local(
            &gpu,
            &stream,
            &bank_b,
            &table_b,
            &mut work_b_mixed,
            &mut scratch_b_mixed,
            &xq,
            &xs,
            &ids_mixed,
            &weights_route,
            &scale2,
            &scale2_host,
            0x7fc8_7654,
        );
        let mut work_full = GroupedWork::new(&stream, GLOBAL, SLOTS, HIDDEN, INTER).unwrap();
        let mut scratch_full = scratch(&gpu);
        let full = run_local(
            &gpu,
            &stream,
            &bank_full,
            &table_full,
            &mut work_full,
            &mut scratch_full,
            &xq,
            &xs,
            &ids_mixed,
            &weights_route,
            &scale2,
            &scale2_host,
            0,
        );
        let rank_sum: Vec<f32> = mixed_a.iter().zip(&mixed_b).map(|(a, b)| a + b).collect();
        assert_eq!(
            rank_sum.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            full.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "rank-order local sum must equal complete-bank grouped output"
        );
        println!(
            "PASS TP/EP local contribution clear+rank-order identity hidden={HIDDEN} inter={INTER} global={GLOBAL} rows={ROWS} topk={TOPK}"
        );
        crate::dsv4_grouped::set_route_validation_for_gate(previous_route_validation);
    }
}
